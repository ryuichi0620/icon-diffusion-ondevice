#!/usr/bin/env python3
"""Stage 5: 焼き込み済みパイプラインを Core AI 形式（3 モデル）に変換する。

    python scripts/05_export_coreai.py --check-only   # まず PyTorch のまま形だけ確認（Core AI 不要）
    python scripts/05_export_coreai.py

拡散モデルは中身が 3 つ。毎ステップ回るのは UNet だけなので、そこだけ最適化できるよう分けて出す。

  TextEncoder  input_ids [1, 77] int32          -> [1, 77, 768]
  UNet         [B,4,64,64] + t + [B,77,768]     -> [B,4,64,64]
  VAEDecoder   [1,4,64,64]                      -> [1,3,512,512]  値域 [-1, 1]

あわせて Swift 側が必要とする資産も書き出す:
  - vocab.json / merges.txt      CLIP トークナイザ（Swift 実装が読む）
  - pipeline_config.json         β スケジュール等。Swift のスケジューラをこれに従わせる

coreai_torch の API は 2026-08-22 に実物（0.4.1）で検証済み。export_module() 参照。
TODO(core-ai): coreai_opt（量子化）は未検証・未実装。_apply_quantization() 参照。
変換部分は export_module() に閉じてあるので、ずれてもそこだけ直せばよい。

なお **iOS に載せるのは後回しでよい**。モデルを完成させるだけなら Stage 1〜3（＋generate.py）
で完結する。このスクリプトが要るのは端末で動かす段になってから。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icon_diffusion import load_config, load_torch_stack  # noqa: E402

# Swift 側のテストと突き合わせる 4 ステップのタイムステップは、**必ず実際の
# スケジューラから導出する**。以前はここに [999, 759, 519, 279] をハードコードしていて、
# smoke_test はその定数が pipeline_config.json に書かれた同じ定数と等しいことを
# 確かめるだけの空虚な検査になっていた（だから通り続けた）。
# 実際の diffusers は等間隔インデックスで選ぶので [999, 759, 499, 259] になる。
LCM_VERIFICATION_STEPS = 4


def lcm_timesteps(pipe, steps: int = LCM_VERIFICATION_STEPS) -> list[int]:
    """実際のスケジューラに 4 ステップを設定して、そのタイムステップを取り出す。"""
    from diffusers import LCMScheduler  # noqa: PLC0415

    scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(steps)
    return [int(t) for t in scheduler.timesteps]


def export_module(module, sample: tuple, out_path: Path, quantize: str) -> None:
    """PyTorch モジュールを 1 つ Core AI 形式に変換して保存する。

    2026-08-22 に coreai-torch 0.4.1 / coreai-core 1.0.0b2 の実物に対して検証済み。
    企画メモの記述のうち `program.save()` だけが誤りで、正しくは `save_asset()` だった。
    それ以外（TorchConverter / get_decomp_table / add_exported_program / to_coreai /
    optimize）はメモどおりの名前で実在する。
    """
    import torch

    try:
        from coreai_torch import TorchConverter, get_decomp_table  # type: ignore
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise SystemExit(
            "coreai-torch が見つかりません。\n"
            "  pip install coreai-torch\n"
            "パッケージ名が違う場合は WWDC26 セッション 325 / apple/coreai-torch を確認し、\n"
            "この関数（export_module）の import を直してください。\n"
            f"元のエラー: {exc}"
        ) from exc

    # 量子化は torch.export の前にモジュールへ当てる。変換後の program に対しては当てられない
    # （Quantizer は nn.Module を受け取る）。ここを間違えると静かに素通しになる。
    if quantize and quantize.lower() not in {"none", "off", ""}:
        module = _quantize_module(module, sample, quantize)

    exported = torch.export.export(module, args=sample).run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(exported).to_coreai()

    program.optimize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # save_asset は .aimodel を「ディレクトリ」として書き出す（main.mlirb / main.hash /
    # metadata.json を含むバンドル）。単一ファイルではないので、後段で is_file() を見ないこと。
    # minimum_os は既定 OSVersion.v27。端末の下限を上げたいときはここで指定する。
    program.save_asset(out_path)


def _quantize_module(module, sample: tuple, quantize: str):
    """重みを int8 に量子化したモジュールを返す。

    **重みだけを量子化し、activation には触らない。** これは意図的な選択で、理由は校正データ。
    activation を量子化するには代表的な入力を流して値の範囲を観測する必要がある（PTQ の校正）。
    ここで手に入るのは形だけ合わせた example なので、乱数で代用すると活性値の範囲を
    誤って学習し、**絵の品質が黙って落ちる**。サイズの大半は重みなので、
    校正なしで安全に効く範囲に留めてある。

    重み用の既定 spec は int8 / 対称 / per-channel(axis=0) で、qparams を重みから直接
    計算する StaticQParamsCalculator を使う。だから校正データが要らない。

    TODO(core-ai): activation まで量子化して更に速くしたい場合は、実際の生成から
    中間テンソルを捕まえて校正データにすること。Quantizer.calibration_mode() を使う。
    """
    import torch  # noqa: PLC0415

    if quantize.lower() not in {"int8"}:
        raise SystemExit(
            f"未対応の量子化指定です: {quantize}\n"
            "  configs/icon_lora.toml の export.quantize は int8 / none のみ対応しています。"
        )

    try:
        from coreai_opt import ExportBackend  # type: ignore  # noqa: PLC0415
        from coreai_opt.quantization import (  # type: ignore  # noqa: PLC0415
            ModuleQuantizerConfig,
            Quantizer,
            QuantizerConfig,
        )
        from coreai_opt.quantization.config.quantization_config import (  # type: ignore  # noqa: PLC0415
            default_weight_quantization_spec,
        )
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise SystemExit(
            "coreai-opt が見つかりません。\n"
            "  pip install coreai-opt\n"
            "（企画メモの `coreai-optimization` はリポジトリ名で、PyPI 名は `coreai-opt`）\n"
            f"元のエラー: {exc}"
        ) from exc

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            # "*": None で activation の量子化を明示的に切る。既定のままだと int8 になり、
            # 校正していないぶん範囲が出鱈目になる。
            op_input_spec={"*": None},
            op_output_spec={"*": None},
        )
    )

    quantizer = Quantizer(module, config)
    prepared = quantizer.prepare(sample)
    with torch.no_grad():
        return quantizer.finalize(prepared, backend=ExportBackend.CoreAI)


def build_examples(pipe, batch: int) -> dict[str, tuple]:
    import torch

    unet_config = pipe.unet.config
    latent_channels = int(getattr(unet_config, "in_channels", 4))
    latent_size = int(getattr(unet_config, "sample_size", 64))
    seq_len = int(getattr(pipe.tokenizer, "model_max_length", 77))
    embed_dim = int(pipe.text_encoder.config.hidden_size)

    return {
        "TextEncoder": (torch.zeros(1, seq_len, dtype=torch.int32),),
        "UNet": (
            torch.randn(batch, latent_channels, latent_size, latent_size),
            torch.tensor([999.0]),
            torch.randn(batch, seq_len, embed_dim),
        ),
        "VAEDecoder": (torch.randn(1, latent_channels, latent_size, latent_size),),
    }


def export_tokenizer_files(tokenizer, resources_dir: Path, *, model_path: Path) -> str:
    """Swift の CLIPTokenizer が読む vocab.json / merges.txt を取り出す。

    transformers のバージョンによって出どころが違う:
      - 5.x の CLIPTokenizer は fast 実装で、save_pretrained は tokenizer.json しか書かない
      - backend の BPE モデルを直接 save すると vocab.json / merges.txt が出る
      - 旧来の slow 実装なら save_pretrained がそのまま両方書く
    どれで取れたかを返す。取れなければ SystemExit。
    """
    wanted = ("vocab.json", "merges.txt")

    def take(directory: Path) -> bool:
        if not all((directory / name).is_file() for name in wanted):
            return False
        for name in wanted:
            shutil.copy2(directory / name, resources_dir / name)
        return True

    # 1) fast トークナイザの backend から直接取り出す（transformers 5.x の既定経路）
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                backend.model.save(tmp)
            except Exception:  # noqa: BLE001 - 経路が使えないだけなので次を試す
                pass
            else:
                if take(Path(tmp)):
                    return "backend_tokenizer.model.save"

    # 2) slow トークナイザなら save_pretrained がそのまま書く
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tokenizer.save_pretrained(tmp)
        except Exception:  # noqa: BLE001
            pass
        else:
            if take(Path(tmp)):
                return "tokenizer.save_pretrained"

    # 3) 変換元のディレクトリに元ファイルが残っていることもある
    if take(model_path / "tokenizer"):
        return f"{model_path.name}/tokenizer"

    raise SystemExit(
        "トークナイザから vocab.json / merges.txt を取り出せませんでした。\n"
        "Swift 側の CLIPTokenizer はこの 2 ファイルを読むので、手動で用意して\n"
        f"  {resources_dir}\n"
        "に置いてください。ベースモデルの tokenizer/ ディレクトリにあります。"
    )


def write_swift_resources(cfg, pipe, resources_dir: Path, model_path: Path) -> dict:
    """Swift 側が読むトークナイザ資産と pipeline_config.json を書き出す。"""
    import torch

    resources_dir.mkdir(parents=True, exist_ok=True)

    route = export_tokenizer_files(pipe.tokenizer, resources_dir, model_path=model_path)
    print(f"[stage5] トークナイザ資産の取得元: {route}")

    scheduler_config = pipe.scheduler.config
    unet_config = pipe.unet.config

    # 潜在から画像への拡大率は VAE のブロック数で決まる。SD1.5 は 8 だが決め打ちにしない
    # （ブロック構成が違うモデルに差し替えたとき、config だけ静かにズレるのを防ぐ）
    vae_scale_factor = int(
        getattr(pipe, "vae_scale_factor", 2 ** (len(pipe.vae.config.block_out_channels) - 1))
    )
    latent_size = int(getattr(unet_config, "sample_size", 64))

    num_train_timesteps = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    beta_start = float(getattr(scheduler_config, "beta_start", 0.00085))
    beta_end = float(getattr(scheduler_config, "beta_end", 0.012))
    beta_schedule = str(getattr(scheduler_config, "beta_schedule", "scaled_linear"))

    # Swift 側が同じ式で ᾱ を組み立てられているか検証するための実測値
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    timesteps_4step = lcm_timesteps(pipe)

    config = {
        "base_model": cfg.model.base_model,
        "lcm_lora": cfg.model.lcm_lora,
        "trigger_word": cfg.style.trigger_word,
        "prompt_template": cfg.style.prompt_template,
        "num_train_timesteps": num_train_timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "beta_schedule": beta_schedule,
        "original_inference_steps": int(getattr(scheduler_config, "original_inference_steps", 50)),
        "timestep_scaling": float(getattr(scheduler_config, "timestep_scaling", 10.0)),
        "sigma_data": 0.5,  # diffusers の LCMScheduler では config ではなくハードコード
        "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
        "vae_scaling_applied_in_model": True,
        "latent_channels": int(getattr(unet_config, "in_channels", 4)),
        "latent_size": latent_size,
        "vae_scale_factor": vae_scale_factor,
        "image_size": latent_size * vae_scale_factor,
        "text_embed_dim": int(pipe.text_encoder.config.hidden_size),
        "max_sequence_length": int(getattr(pipe.tokenizer, "model_max_length", 77)),
        "bos_token_id": int(pipe.tokenizer.bos_token_id),
        "eos_token_id": int(pipe.tokenizer.eos_token_id),
        # Swift のテストが突き合わせる値。実スケジューラから導出する
        "verification": {
            "timesteps_4step": timesteps_4step,
            "alphas_cumprod_at": {str(t): float(alphas_cumprod[t]) for t in timesteps_4step},
        },
    }

    (resources_dir / "pipeline_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="変換元（既定は Stage 4 の出力）")
    parser.add_argument("--output", default=None, help=".aimodel の出力先")
    parser.add_argument("--resources", default=None, help="Swift Package の Resources ディレクトリ")
    parser.add_argument("--batch", type=int, default=1, help="UNet のバッチ。CFG を使うなら 2")
    parser.add_argument("--quantize", default=None, help="int8 / none")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Core AI を使わず、PyTorch のまま 3 つのラッパーを実行して形だけ確認する",
    )
    parser.add_argument("--skip-resources", action="store_true", help="Swift 資産の書き出しをしない")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    model_path = Path(args.model).expanduser() if args.model else cfg.fuse.output_dir
    output_dir = Path(args.output).expanduser() if args.output else cfg.export.output_dir
    resources_dir = Path(args.resources).expanduser() if args.resources else cfg.export.swift_resources_dir
    quantize = cfg.export.quantize if args.quantize is None else args.quantize

    if not Path(model_path).is_dir():
        print(f"[stage5] 変換元がありません: {model_path}", file=sys.stderr)
        print("[stage5] 先に Stage 4 を実行してください: python scripts/04_fuse_lora.py", file=sys.stderr)
        return 1

    # 重い依存はここまで遅らせる（--help と引数チェックは torch 無しで通る）
    torch, pipeline_module = load_torch_stack()
    from icon_diffusion.export_wrappers import (  # noqa: PLC0415
        TextEncoderWrapper,
        UNetWrapper,
        VAEDecoderWrapper,
    )

    # 変換は CPU / fp32 で行う。fp16 のまま export すると精度が落ちる。
    print(f"[stage5] 変換元: {model_path}")
    pipe = pipeline_module.load_base_pipeline(
        cfg, torch.device("cpu"), torch.float32, model_path=model_path
    )

    modules = {
        "TextEncoder": TextEncoderWrapper(pipe.text_encoder).eval(),
        "UNet": UNetWrapper(pipe.unet).eval(),
        "VAEDecoder": VAEDecoderWrapper(pipe.vae).eval(),
    }
    examples = build_examples(pipe, args.batch)

    if not args.skip_resources:
        config = write_swift_resources(cfg, pipe, resources_dir, Path(model_path))
        print(f"[stage5] Swift 資産: {resources_dir}")
        print("[stage5]   vocab.json / merges.txt / pipeline_config.json")
        print(
            f"[stage5]   latent={config['latent_channels']}x{config['latent_size']}^2 "
            f"embed={config['text_embed_dim']} seq={config['max_sequence_length']}"
        )

    if args.check_only:
        print("[stage5] --check-only: PyTorch のまま形を確認します（Core AI は使いません）")
        with torch.no_grad():
            for name, module in modules.items():
                out = module(*examples[name])
                shapes = ", ".join(str(tuple(t.shape)) for t in examples[name])
                print(f"[stage5]   {name:12s} ({shapes}) -> {tuple(out.shape)} {out.dtype}")
        print("[stage5] 形は通りました。Core AI 変換に進めます")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, module in modules.items():
        out_path = output_dir / f"Icon{name}.aimodel"
        print(f"[stage5] 変換中: {name} -> {out_path.name}")
        with torch.no_grad():
            export_module(module, examples[name], out_path, quantize)
        print(f"[stage5]   完了: {out_path}")

    print(f"[stage5] 3 モデル出力 -> {output_dir}")
    print("[stage5] 次: この .aimodel と Resources を Xcode プロジェクトのバンドルに入れて Stage 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
