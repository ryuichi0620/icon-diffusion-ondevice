#!/usr/bin/env python3
"""パイプライン全体の疎通確認。**ネットワークも実モデルも要らない。**

    python scripts/smoke_test.py
    python scripts/smoke_test.py --keep      # 一時ファイルを残して中身を見る

ランダムな重みの極小モデル（UNet 数十万パラメータ / 画像 64x64）をその場で組み立て、
Stage 3 → generate → Stage 4 → Stage 5(--check-only) を実際に走らせる。

出てくる絵は当然ただのノイズ。確かめているのは**配管が通っているか**:
  - LCM スケジューラが差さるか
  - LoRA 2 枚の重ね掛けと重み設定が効くか
  - 焼き込みと保存・読み直しができるか
  - 背景の透過とコンタクトシートとマニフェストが出るか
  - Core AI に渡す 3 モデルの分割が形として成立しているか

数 GB のベースモデルを落とす前、あるいは半日かけて学習する前に、
手元の環境でコードが動くことだけ先に確かめるために使う。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SCRIPTS = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPTS.parent


def build_tokenizer_assets(directory: Path) -> int:
    """CLIP と同じ構造の極小 vocab / merges を作る。戻り値は語彙数。"""
    base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    base_set = set(base)
    characters = [chr(value) for value in base]
    offset = 0
    for value in range(256):
        if value not in base_set:
            characters.append(chr(256 + offset))
            offset += 1

    merges = ["c o", "co f", "cof f", "coff e", "coffe e</w>", "c u", "cu p</w>"]
    tokens = characters + [c + "</w>" for c in characters]
    tokens += ["".join(m.split()) for m in merges]
    tokens += ["<|startoftext|>", "<|endoftext|>"]

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_text(
        json.dumps({token: index for index, token in enumerate(tokens)}, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")
    return len(tokens)


def build_tiny_pipeline(out_dir: Path) -> None:
    """ランダム初期化の極小 Stable Diffusion を組んで保存する。"""
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    assets = out_dir.parent / "tokenizer_assets"
    vocab_size = build_tokenizer_assets(assets)

    tokenizer = CLIPTokenizer(
        vocab_file=str(assets / "vocab.json"),
        merges_file=str(assets / "merges.txt"),
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=77,
    )

    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            bos_token_id=vocab_size - 2,
            eos_token_id=vocab_size - 1,
            pad_token_id=vocab_size - 1,
            hidden_size=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_hidden_layers=5,
            vocab_size=vocab_size,
            layer_norm_eps=1e-5,
        )
    )

    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        norm_num_groups=8,
    )

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        norm_num_groups=8,
    )

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        ),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.save_pretrained(str(out_dir))


def build_tiny_lora(base_dir: Path, out_dir: Path, seed: int) -> None:
    """train_dreambooth_lora.py が出すのと同じ形（pytorch_lora_weights.safetensors）で保存する。"""
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict

    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_pretrained(str(base_dir), subfolder="unet")
    unet.add_adapter(
        LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    # peft は lora_B をゼロ初期化する。そのままだと LoRA の寄与が恒等ゼロになり、
    # 「重みを変えても絵が変わらない」ので検証にならない。実際に効く値を入れておく。
    for name, parameter in unet.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.05)

    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    out_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(out_dir), unet_lora_layers=state_dict, safe_serialization=True
    )


def write_config(path: Path, root: Path) -> None:
    path.write_text(
        f"""
[model]
base_model = "{root / 'base'}"
revision = ""
lcm_lora = "{root / 'lcm-lora'}"

[style]
trigger_word = "ICONST"
prompt_template = "an icon in {{trigger}} style, {{subject}}, flat, minimal"
fallback_subject = "a simple object"
negative_prompt = ""

[dataset]
input_dir = "{root / 'raw'}"
output_dir = "{root / 'processed'}"
resolution = 64
background = "#FFFFFF"

[train]
output_dir = "{root / 'icon-lora'}"
rank = 4
learning_rate = 1e-4
max_train_steps = 1
train_batch_size = 1
seed = 0
mixed_precision = "no"

[preview]
output_dir = "{root / 'preview'}"
steps = 2
guidance_scale = 1.0
style_weight = 0.8
lcm_weight = 1.0
seed = 0
subjects = ["a coffee cup", "a cup"]

[fuse]
output_dir = "{root / 'fused'}"

[export]
output_dir = "{root / 'coreai'}"
swift_resources_dir = "{root / 'swift-resources'}"
quantize = "none"
""".lstrip(),
        encoding="utf-8",
    )


def run(label: str, argv: list[str]) -> None:
    print(f"\n\033[36m── {label}\033[0m")
    print(f"   $ {' '.join(str(a) for a in argv[1:])}")
    result = subprocess.run(
        [sys.executable, *argv], cwd=TRAINING_ROOT, capture_output=True, text=True
    )
    for line in result.stdout.splitlines()[-6:]:
        print(f"   {line}")
    if result.returncode != 0:
        print(f"\n\033[31m   失敗 (exit {result.returncode})\033[0m")
        print(result.stdout[-2000:])
        print(result.stderr[-3000:])
        raise SystemExit(1)


def expect(path: Path, label: str) -> None:
    if not path.exists():
        print(f"\033[31m   期待した出力がありません: {label} -> {path}\033[0m")
        raise SystemExit(1)
    print(f"   ✓ {label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="一時ディレクトリを残す")
    parser.add_argument("--dir", default=None, help="作業ディレクトリを指定する")
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser() if args.dir else Path(tempfile.mkdtemp(prefix="icon-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"作業ディレクトリ: {root}")

    try:
        print("\n\033[36m── 極小モデルを組み立てる（ネットワーク不要）\033[0m")
        build_tiny_pipeline(root / "base")
        print("   ✓ base pipeline")
        build_tiny_lora(root / "base", root / "icon-lora", seed=1)
        print("   ✓ style LoRA (自作ぶんの代役)")
        build_tiny_lora(root / "base", root / "lcm-lora", seed=2)
        print("   ✓ LCM LoRA (既製ぶんの代役)")

        config = root / "config.toml"
        write_config(config, root)

        run("Stage 3: LCM を重ねて少ステップ生成", ["scripts/03_preview_lcm.py", "--config", str(config)])
        expect(root / "preview" / "contact_sheet.png", "preview/contact_sheet.png")

        # 前の実行の残骸と比べないよう、スイープは専用ディレクトリに出す
        sweep_dir = root / "preview-sweep"
        run("Stage 3: スタイル重みのスイープ",
            ["scripts/03_preview_lcm.py", "--config", str(config),
             "--output", str(sweep_dir), "--sweep-style", "0.0", "1.0"])
        # 重み 0.0（スタイル LoRA が効かない）と 1.0 で絵が変わらないなら set_adapters が効いていない
        off = sweep_dir / "01_w0.00_a-coffee-cup.png"
        on = sweep_dir / "02_w1.00_a-coffee-cup.png"
        for path in (off, on):
            assert path.is_file(), f"スイープの出力がありません: {path.name} ({[p.name for p in sorted(sweep_dir.glob('*.png'))]})"
        assert off.read_bytes() != on.read_bytes(), \
            "スタイル重み 0.0 と 1.0 で出力が同じ。set_adapters の重みが効いていない"
        print(f"   ✓ {off.name} と {on.name} で出力が変わる")

        run("generate: 量産＋背景の透過",
            ["scripts/generate.py", "--config", str(config), "--count", "2", "--transparent"])
        expect(root / "generated" / "manifest.json", "generated/manifest.json")
        expect(root / "generated" / "contact_sheet.png", "generated/contact_sheet.png")

        manifest = json.loads((root / "generated" / "manifest.json").read_text(encoding="utf-8"))
        images = manifest["images"]
        assert len(images) == 4, f"2 被写体 x 2 枚 = 4 のはずが {len(images)}"
        assert images[0]["seed"] != images[1]["seed"], "同じ被写体の 2 枚が同じ seed になっている"
        assert all(record["transparent"] for record in images), "透過フラグが立っていない"
        print(f"   ✓ manifest: {len(images)} 枚, seed={[r['seed'] for r in images]}")

        run("Stage 4: LoRA の焼き込み＋前後比較",
            ["scripts/04_fuse_lora.py", "--config", str(config)])
        expect(root / "fused" / "model_index.json", "fused/model_index.json")
        expect(root / "fuse_compare" / "before_fuse.png", "fuse_compare/before_fuse.png")

        run("generate: 焼き込み済みモデルで生成",
            ["scripts/generate.py", "--config", str(config), "--model", str(root / "fused"),
             "--out", str(root / "generated-fused"), "--subjects", "a coffee cup"])
        expect(root / "generated-fused" / "manifest.json", "generated-fused/manifest.json")

        run("Stage 5: Core AI 抜きで 3 モデルの形を確認",
            ["scripts/05_export_coreai.py", "--config", str(config), "--check-only"])
        expect(root / "swift-resources" / "pipeline_config.json", "swift-resources/pipeline_config.json")
        expect(root / "swift-resources" / "vocab.json", "swift-resources/vocab.json")

        exported = json.loads((root / "swift-resources" / "pipeline_config.json").read_text(encoding="utf-8"))
        # 以前はここで [999, 759, 519, 279] という定数と比べていたが、書き出し側も同じ定数を
        # ハードコードしていたので「定数が定数自身と等しい」ことを確かめるだけになっていた。
        # diffusers がタイムステップの選び方を変えても気づけなかった原因。
        # いまは diffusers に実際に計算させた値と突き合わせる。
        from diffusers import LCMScheduler  # noqa: PLC0415

        reference = LCMScheduler(
            num_train_timesteps=exported["num_train_timesteps"],
            beta_start=exported["beta_start"],
            beta_end=exported["beta_end"],
            beta_schedule=exported["beta_schedule"],
            original_inference_steps=exported["original_inference_steps"],
            timestep_scaling=exported["timestep_scaling"],
        )
        reference.set_timesteps(4)
        expected_timesteps = [int(t) for t in reference.timesteps]
        assert exported["verification"]["timesteps_4step"] == expected_timesteps, (
            f"LCM のタイムステップが diffusers と違う: "
            f"{exported['verification']['timesteps_4step']} != {expected_timesteps}"
        )

        # config が言う画像サイズと、実際に生成された画像のサイズが一致すること。
        # ここがズレると Swift 側が形の合わないテンソルを受け取る。
        from PIL import Image as _Image

        generated = sorted((root / "generated").glob("*.png"))[0]
        with _Image.open(generated) as sample:
            actual = sample.size
        assert actual == (exported["image_size"], exported["image_size"]), (
            f"pipeline_config.json の image_size={exported['image_size']} と "
            f"実際の生成サイズ {actual} が食い違っています"
        )
        print(f"   ✓ pipeline_config.json: latent={exported['latent_size']} "
              f"x{exported['vae_scale_factor']} -> image={exported['image_size']} (実測 {actual[0]}) "
              f"timesteps={exported['verification']['timesteps_4step']}")

        print("\n\033[32m全部通りました。パイプラインの配管は生きています。\033[0m")
        print("（絵の中身はランダム重みなのでノイズです。品質は実モデルでの Stage 2〜3 で見ること）")
        return 0
    finally:
        if args.keep or args.dir:
            print(f"\n一時ディレクトリを残しました: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
