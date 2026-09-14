#!/usr/bin/env python3
"""輪郭を条件に与えて、スタイルはモデルに描かせる（ControlNet）。

    python3 scripts/controlnet_style.py --prompts prompts.example.txt \
        --reference-dir outputs/ref-glyphs --out outputs/cn

**なぜ img2img ではなく ControlNet なのか**（Trial 18 の結論）。
少ステップの img2img は、入力が綺麗なほど動かない。単色グリフを入れて強さ 0.6〜0.8 で
塗り直しても、**パレット純度 1.9%**＝スタイルがまったく乗らなかった。強さを上げれば
今度は構造が崩れる。

ControlNet は構造を**別の条件信号**として渡し、絵そのものはノイズから作る。
だからスタイルはモデル側から出てくる。狙いはここ。

条件画像は Canny ではなく**形態学的な輪郭**（膨張 − 収縮）にしてある。入力が既に二値の
グリフなので、Canny のしきい値調整もノイズも要らず、1〜2 画素の綺麗な境界が得られる。
**白線・黒地に反転する**のを忘れないこと（ControlNet 系はその向きを期待している）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONTROLNET = "lllyasviel/sd-controlnet-canny"


def outline(image, width: int = 2):
    """二値グリフの境界を白線・黒地で返す。"""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from icon_diffusion.shapes import _morph  # noqa: PLC0415

    ink = np.asarray(image.convert("L"), dtype=np.uint8) < 128
    edge = _morph(ink, width, dilate=True) & ~_morph(ink, width, dilate=False)
    return Image.fromarray((edge * 255).astype(np.uint8)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--reference-dir", required=True, help="単色グリフの置き場所")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--control-scale", type=float, default=0.75,
                        help="条件の効き。1.0 に近いほど輪郭に忠実だがスタイルが痩せる")
    parser.add_argument("--control-end", type=float, default=0.8,
                        help="途中で条件を切る割合。終盤を自由にするとスタイルが乗りやすい")
    parser.add_argument("--controlnet", default=CONTROLNET)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-control", action="store_true")
    args = parser.parse_args()

    from PIL import Image  # noqa: PLC0415
    from icon_diffusion import load_config, load_torch_stack  # noqa: PLC0415
    from icon_diffusion import pipeline as pipeline_module  # noqa: PLC0415
    from icon_diffusion.postprocess import remove_flat_background  # noqa: PLC0415

    torch, _ = load_torch_stack()
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline  # noqa: PLC0415

    cfg = load_config(args.config)
    device = pipeline_module.select_device(args.device)
    dtype = pipeline_module.select_dtype(device)

    subjects = [line.strip() for line in Path(args.prompts).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.reference_dir)
    controls: dict[str, list] = {}
    for subject in subjects:
        slug = subject.replace(" ", "-")
        found = sorted(source.glob(f"{slug}-*.png")) or sorted(source.glob(f"{slug}_*.png"))
        if not found:
            raise SystemExit(f"グリフが見つかりません: {slug} in {source}")
        controls[subject] = [outline(Image.open(f).resize((512, 512), Image.LANCZOS))
                             for f in found]
    if args.keep_control:
        control_dir = out_dir / "control"
        control_dir.mkdir(exist_ok=True)
        for subject, images in controls.items():
            for i, image in enumerate(images):
                image.save(control_dir / f"{subject.replace(' ', '-')}_{i + 1:02d}.png")

    print(f"[cn] ControlNet: {args.controlnet}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    base = pipeline_module.load_base_pipeline(cfg, device, dtype)
    pipe = StableDiffusionControlNetPipeline(**base.components, controlnet=controlnet)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    pipeline_module.attach_loras(
        pipe, cfg, style_lora_dir=args.lora_dir,
        style_weight=cfg.preview.style_weight, lcm_weight=cfg.preview.lcm_weight,
    )

    template = cfg.style.prompt_template
    records = []
    for subject in subjects:
        pool = controls[subject]
        for variation in range(args.count):
            seed = args.seed + variation
            started = time.perf_counter()
            image = pipe(
                prompt=template.format(trigger=cfg.style.trigger_word, subject=subject),
                image=pool[variation % len(pool)],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                controlnet_conditioning_scale=args.control_scale,
                control_guidance_end=args.control_end,
                generator=pipeline_module.make_generator(seed),
            ).images[0]
            image = remove_flat_background(image)
            name = f"{subject.replace(' ', '-')}_{variation + 1:02d}.png"
            image.save(out_dir / name)
            records.append({"file": name, "subject": subject, "seed": seed,
                            "seconds": round(time.perf_counter() - started, 2)})
            print(f"[cn] {name}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(
        {"controlnet": args.controlnet, "control_scale": args.control_scale,
         "control_end": args.control_end, "steps": args.steps, "images": records},
        ensure_ascii=False, indent=1))
    print(f"[cn] 完了 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
