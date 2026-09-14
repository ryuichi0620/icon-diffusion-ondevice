#!/usr/bin/env python3
"""形を先に決めてから、スタイルを塗る 2 段生成。

    python3 scripts/reference_style.py --prompts prompts.example.txt --strength 0.6

**狙い**: 「被写体が現実の物に見えない」のは、テキストだけで構造を指定しているから。
切り分け（docs/experiments.md Trial 15 の予備調査）で、**LoRA を外した素の SD1.5 は
鍵もカメラもロケットも正しく描ける**ことが分かっている。知識はあるのに引き出せていない。

そこで 2 段に分ける。

    1 段目: スタイル LoRA を外した素のモデルで、被写体を描かせる（構造を決める）
    2 段目: その絵を種に img2img で、スタイル LoRA を掛けて塗り直す（見た目を決める）

**参照画像を外部から持ってこないのが肝。** 商用可のアイコン集を使う手もあるが、
追加の資産も権利の確認も要らないぶん、まずこちらで天井を測る。

`--strength` が肝心の取引。低いほど 1 段目の形が残り、高いほどスタイルが強く乗る。
少ステップでは **実効ステップ数 = int(steps × strength)** なので、0.25 未満だと 0 步になって
何も起きない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 1 段目のプロンプト。**学習データを作ったときの指示に寄せる**（SOURCE.md の原文）。
# そこが目標の分布なので、素のモデルにも同じ方向を向かせる
REFERENCE_PROMPT = (
    "a minimal flat vector icon of {subject}, centered, on a plain white background, "
    "bold simple geometric shapes, no text, front view"
)
# 素のモデルは「アイコン = 丸バッジ」と解釈して円形の地を描きたがる。
# それを 2 段目が主役として塗り直してしまうので、ここで止める（実測で確認）
REFERENCE_NEGATIVE = (
    "photo, 3d render, shadow, gradient, text, watermark, "
    "circular badge, round frame, colored background, border, sticker"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompts", required=True, help="被写体を 1 行 1 件で書いたファイル")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strength", type=float, default=0.6,
                        help="2 段目の塗り直しの強さ。実効ステップ数 = int(steps × strength)")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--reference-steps", type=int, default=25,
                        help="1 段目のステップ数。素のモデルなので多めに回してよい")
    parser.add_argument("--reference-guidance", type=float, default=7.5)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-reference", action="store_true", help="1 段目も保存する")
    parser.add_argument("--reference-dir", default=None,
                        help="1 段目を生成せず、ここの <被写体>-01.png を参照として使う。"
                             "**学習データを指すと上限の測定になる**（カンニングなので比較には使えない）")
    args = parser.parse_args()

    from icon_diffusion import load_config, load_torch_stack  # noqa: PLC0415
    from icon_diffusion import pipeline as pipeline_module  # noqa: PLC0415
    from icon_diffusion.postprocess import remove_flat_background  # noqa: PLC0415

    torch, _ = load_torch_stack()
    from diffusers import StableDiffusionImg2ImgPipeline  # noqa: PLC0415

    cfg = load_config(args.config)
    device = pipeline_module.select_device(args.device)
    dtype = pipeline_module.select_dtype(device)

    subjects = [line.strip() for line in Path(args.prompts).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = out_dir / "reference"
    if args.keep_reference:
        reference_dir.mkdir(exist_ok=True)

    print(f"[ref] 1 段目: 素のモデル（{args.reference_steps} 步 / CFG {args.reference_guidance}）")
    print(f"[ref] 2 段目: スタイル LoRA + img2img（強さ {args.strength} "
          f"= 実効 {int(args.steps * args.strength)} 步）")

    supplied: dict[str, "object"] = {}
    if args.reference_dir:
        from PIL import Image as _Image  # noqa: PLC0415
        source = Path(args.reference_dir)
        for subject in subjects:
            slug = _safe(subject)
            found = sorted(source.glob(f"{slug}-*.png")) or sorted(source.glob(f"{slug}_*.png"))
            if not found:
                raise SystemExit(f"参照画像が見つかりません: {slug} in {source}")
            supplied[subject] = [
                _Image.open(f).convert("RGB").resize((512, 512), _Image.LANCZOS) for f in found
            ]
        print(f"[ref] 1 段目を飛ばし、{source} の画像を参照に使います")

    base = pipeline_module.load_base_pipeline(cfg, device, dtype)
    # 1 段目はスタイルを外す。LCM も外して素の挙動にする（構造をいちばん引き出せる設定）
    reference_pipe = base
    records = []

    # 先に 1 段目をすべて作る。アダプタの付け外しを 1 回で済ませるため
    references: dict[tuple[str, int], "object"] = {}
    for subject in subjects:
        if subject in supplied:
            pool = supplied[subject]
            for variation in range(args.count):
                references[(subject, variation)] = pool[variation % len(pool)]
            continue
        for variation in range(args.count):
            seed = args.seed + variation
            image = reference_pipe(
                REFERENCE_PROMPT.format(subject=subject),
                negative_prompt=REFERENCE_NEGATIVE,
                num_inference_steps=args.reference_steps,
                guidance_scale=args.reference_guidance,
                generator=pipeline_module.make_generator(seed),
            ).images[0]
            references[(subject, variation)] = image
            if args.keep_reference:
                image.save(reference_dir / f"{_safe(subject)}_{variation + 1:02d}.png")
        print(f"[ref] 1 段目 {subject}", flush=True)

    # 2 段目: 同じ重みを使い回して img2img に組み替える
    pipeline_module.attach_loras(
        base, cfg,
        style_lora_dir=args.lora_dir,
        style_weight=cfg.preview.style_weight,
        lcm_weight=cfg.preview.lcm_weight,
    )
    styled = StableDiffusionImg2ImgPipeline(**base.components)
    styled.set_progress_bar_config(disable=None)

    template = cfg.style.prompt_template
    for subject in subjects:
        for variation in range(args.count):
            seed = args.seed + variation
            prompt = template.format(trigger=cfg.style.trigger_word, subject=subject)
            started = time.perf_counter()
            image = styled(
                prompt=prompt,
                image=references[(subject, variation)],
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=pipeline_module.make_generator(seed),
            ).images[0]
            image = remove_flat_background(image)
            name = f"{_safe(subject)}_{variation + 1:02d}.png"
            image.save(out_dir / name)
            records.append({"file": name, "subject": subject, "seed": seed,
                            "strength": args.strength,
                            "seconds": round(time.perf_counter() - started, 2)})
            print(f"[ref] 2 段目 {name}", flush=True)

    (out_dir / "manifest.json").write_text(
        json.dumps({"strength": args.strength, "steps": args.steps,
                    "reference_steps": args.reference_steps,
                    "reference_prompt": REFERENCE_PROMPT, "images": records},
                   ensure_ascii=False, indent=1))
    print(f"[ref] 完了 -> {out_dir}")
    return 0


def _safe(text: str) -> str:
    return text.replace(" ", "-")


if __name__ == "__main__":
    raise SystemExit(main())
