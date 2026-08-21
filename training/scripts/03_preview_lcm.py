#!/usr/bin/env python3
"""Stage 3: 高速化 LoRA（既製）と スタイル LoRA（自作）を重ねて、少ステップ生成を確認する。

    python scripts/03_preview_lcm.py
    python scripts/03_preview_lcm.py --steps 4 --style-weight 0.8
    python scripts/03_preview_lcm.py --sweep-style 0.4 0.6 0.8 1.0

ここが最初の山。「4 ステップでアイコンらしく出るか」をここで決める。
iPhone に載せる前に必ず Mac で仕上がりを確認すること。

出力は 1 枚ずつの PNG に加えて、並べて見るためのコンタクトシートも作る。
スタイルの揺れは並べないと気づけない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icon_diffusion import load_config  # noqa: E402
from icon_diffusion.pipeline import (  # noqa: E402
    attach_loras,
    load_base_pipeline,
    make_contact_sheet,
    make_generator,
    select_device,
    select_dtype,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=None, help="推論ステップ数（既定は設定の 4）")
    parser.add_argument("--guidance", type=float, default=None, help="CFG スケール")
    parser.add_argument("--style-weight", type=float, default=None, help="スタイル LoRA の効き具合")
    parser.add_argument("--lcm-weight", type=float, default=None, help="高速化 LoRA の効き具合")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None, help="被写体（既定は設定のリスト）")
    parser.add_argument("--lora-dir", default=None, help="スタイル LoRA のディレクトリ")
    parser.add_argument("--output", default=None, help="出力ディレクトリ")
    parser.add_argument("--device", default=None, help="mps / cuda / cpu")
    parser.add_argument(
        "--sweep-style",
        nargs="*",
        type=float,
        default=None,
        help="指定した重みを順に試す。被写体は先頭 1 つに固定して比較しやすくする",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    steps = args.steps or cfg.preview.steps
    guidance = cfg.preview.guidance_scale if args.guidance is None else args.guidance
    style_weight = cfg.preview.style_weight if args.style_weight is None else args.style_weight
    lcm_weight = cfg.preview.lcm_weight if args.lcm_weight is None else args.lcm_weight
    seed = cfg.preview.seed if args.seed is None else args.seed
    subjects = args.subjects if args.subjects else cfg.preview.subjects
    output_dir = Path(args.output).expanduser() if args.output else cfg.preview.output_dir

    if not subjects:
        print("[stage3] 被写体が空です。--subjects か設定の preview.subjects を埋めてください", file=sys.stderr)
        return 1

    device = select_device(args.device)
    dtype = select_dtype(device)
    print(f"[stage3] device={device} dtype={dtype}")
    print(f"[stage3] steps={steps} guidance={guidance} style_weight={style_weight} seed={seed}")
    if guidance > 1.0:
        print(
            "[stage3] 注意: guidance>1.0 は CFG が有効になり UNet 実行回数が 2 倍になります。"
            " 端末側は 1.0 を既定にする想定です（docs/architecture.md）"
        )

    pipe = load_base_pipeline(cfg, device, dtype)

    adapters = attach_loras(
        pipe,
        cfg,
        style_lora_dir=args.lora_dir,
        style_weight=style_weight,
        lcm_weight=lcm_weight,
    )
    has_style = "icon" in adapters

    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    labels = []
    records = []

    if args.sweep_style is not None:
        if not has_style:
            print("[stage3] スタイル LoRA が無いので重みのスイープはできません", file=sys.stderr)
            return 1
        weights = args.sweep_style or [0.4, 0.6, 0.8, 1.0]
        subject = subjects[0]
        print(f"[stage3] スタイル重みのスイープ: {weights}（被写体は '{subject}' に固定）")
        jobs = [(subject, w) for w in weights]
    else:
        jobs = [(subject, style_weight) for subject in subjects]

    for index, (subject, weight) in enumerate(jobs, start=1):
        if has_style:
            # スイープで重みだけ差し替える。LoRA の再読み込みは不要
            pipe.set_adapters(adapters, adapter_weights=[lcm_weight, weight])
        prompt = cfg.style.prompt(subject)
        started = time.perf_counter()
        image = pipe(
            prompt,
            negative_prompt=cfg.style.negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=make_generator(seed),
        ).images[0]
        elapsed = time.perf_counter() - started

        name = f"{index:02d}_w{weight:.2f}_{subject.replace(' ', '-')}.png"
        image.save(output_dir / name)
        images.append(image)
        labels.append(f"w={weight:.2f}  {subject}  ({elapsed:.1f}s)")
        records.append(
            {"file": name, "subject": subject, "prompt": prompt, "style_weight": weight, "seconds": round(elapsed, 2)}
        )
        print(f"[stage3] {name}  {elapsed:.1f}s")

    sheet = make_contact_sheet(images, labels, columns=3)
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    meta = {
        "steps": steps,
        "guidance_scale": guidance,
        "lcm_weight": lcm_weight,
        "seed": seed,
        "base_model": cfg.model.base_model,
        "lcm_lora": cfg.model.lcm_lora,
        "device": str(device),
        "results": records,
    }
    (output_dir / "preview.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[stage3] コンタクトシート: {sheet_path}")
    print("[stage3] 納得したら Stage 4: python scripts/04_fuse_lora.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
