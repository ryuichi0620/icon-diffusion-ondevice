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

TODO(core-ai): coreai_torch / coreai_opt の API 名は企画メモの記述をそのまま写している。
WWDC26 セッション 324 / 325 と apple/coreai-torch の実物に対して要検証。
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

# LCM の 4 ステップは既知の固定値。Swift 側のテストと突き合わせるために書き出す
LCM_4STEP_TIMESTEPS = [999, 759, 519, 279]


def export_module(module, sample: tuple, out_path: Path, quantize: str) -> None:
    """PyTorch モジュールを 1 つ Core AI 形式に変換して保存する。

    TODO(core-ai): ここが企画メモの記述をそのまま写した部分。API がずれたらこの関数だけ直す。
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

    exported = torch.export.export(module, args=sample).run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(exported).to_coreai()

    if quantize and quantize.lower() not in {"none", "off", ""}:
        _apply_quantization(program, quantize)

    program.optimize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out_path))


def _apply_quantization(program, quantize: str) -> None:
    """TODO(core-ai): coreai_opt.quantization の Quantizer で圧縮をかける。

    企画メモには「`coreai_opt.quantization` の Quantizer で INT8 量子化などをかける」とだけあり、
    具体的な呼び出し方は未確認。推測で書くと黙って壊れるので、現状は警告して素通しする。
    fp32 のままでも 3 モデル揃えば疎通確認はできる（メモリは docs/architecture.md 参照）。
    """
    print(
        f"[stage5] ⚠ 量子化 ({quantize}) は未実装のためスキップします。\n"
        "         coreai_opt.quantization の API を確認して _apply_quantization() を実装してください。\n"
        "         fp16 のまま 3 モデル載せると 2GB を超えるので、端末で動かす前に必須です。"
    )


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
        # Swift のテストが突き合わせる固定値
        "verification": {
            "timesteps_4step": LCM_4STEP_TIMESTEPS,
            "alphas_cumprod_at": {str(t): float(alphas_cumprod[t]) for t in LCM_4STEP_TIMESTEPS},
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
