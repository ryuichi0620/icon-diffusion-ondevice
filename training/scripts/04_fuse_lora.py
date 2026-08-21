#!/usr/bin/env python3
"""Stage 4: LoRA を UNet に焼き込んで、追加パーツ不要の一体型にする。

    python scripts/04_fuse_lora.py
    python scripts/04_fuse_lora.py --skip-verify
    python scripts/04_fuse_lora.py --device cpu     # fp32 で焼く（差が出たときの逃げ道）

焼き込みは不可逆で、しかも fp16 だと丸め誤差が乗る。Stage 3 で確認した絵と
変わっていないことを確かめてから Stage 5 に渡したいので、既定で
「焼く前 / 焼いた後」を同じ seed で生成して差分を出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icon_diffusion import load_config  # noqa: E402
from icon_diffusion.pipeline import (  # noqa: E402
    attach_loras,
    load_base_pipeline,
    make_generator,
    select_device,
    select_dtype,
)

# 焼き込み前後の平均絶対差がこれを超えたら目視確認を促す（0-1 正規化、約 5/255）
MAE_WARN_THRESHOLD = 0.02


def generate(pipe, cfg, subject: str, steps: int, guidance: float, seed: int):
    return pipe(
        cfg.style.prompt(subject),
        negative_prompt=cfg.style.negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=make_generator(seed),
    ).images[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--lora-dir", default=None, help="スタイル LoRA のディレクトリ")
    parser.add_argument("--output", default=None, help="焼き込み後のパイプラインの保存先")
    parser.add_argument("--style-weight", type=float, default=None)
    parser.add_argument("--lcm-weight", type=float, default=None)
    parser.add_argument("--device", default=None, help="mps / cuda / cpu")
    parser.add_argument("--skip-verify", action="store_true", help="焼き込み前後の比較をしない")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    style_weight = cfg.preview.style_weight if args.style_weight is None else args.style_weight
    lcm_weight = cfg.preview.lcm_weight if args.lcm_weight is None else args.lcm_weight
    output_dir = Path(args.output).expanduser() if args.output else cfg.fuse.output_dir

    device = select_device(args.device)
    dtype = select_dtype(device)
    print(f"[stage4] device={device} dtype={dtype}")
    print(f"[stage4] style_weight={style_weight} lcm_weight={lcm_weight}")

    pipe = load_base_pipeline(cfg, device, dtype)
    adapters = attach_loras(
        pipe,
        cfg,
        style_lora_dir=args.lora_dir,
        style_weight=style_weight,
        lcm_weight=lcm_weight,
    )
    print(f"[stage4] 焼き込むアダプタ: {adapters}")

    subject = cfg.preview.subjects[0] if cfg.preview.subjects else cfg.style.fallback_subject
    before = None
    if not args.skip_verify:
        print(f"[stage4] 焼き込み前の参照画像を生成: '{subject}'")
        before = generate(pipe, cfg, subject, cfg.preview.steps, cfg.preview.guidance_scale, cfg.preview.seed)

    # diffusers のバージョンによって fuse_lora のシグネチャが違うので両対応にする
    try:
        pipe.fuse_lora(adapter_names=adapters)
    except TypeError:
        pipe.fuse_lora()
    # 焼き込んだ後にアダプタ層を外す。残したままだと二重に効いてしまう
    pipe.unload_lora_weights()
    print("[stage4] 焼き込み完了")

    report: dict = {
        "base_model": cfg.model.base_model,
        "lcm_lora": cfg.model.lcm_lora,
        "adapters": adapters,
        "style_weight": style_weight,
        "lcm_weight": lcm_weight,
        "device": str(device),
        "dtype": str(dtype),
        "output_dir": str(output_dir),
    }

    if before is not None:
        print("[stage4] 焼き込み後の同条件画像を生成して比較します")
        after = generate(pipe, cfg, subject, cfg.preview.steps, cfg.preview.guidance_scale, cfg.preview.seed)

        a = np.asarray(before, dtype=np.float32) / 255.0
        b = np.asarray(after, dtype=np.float32) / 255.0
        mae = float(np.abs(a - b).mean())
        max_diff = float(np.abs(a - b).max())
        report["verify"] = {"subject": subject, "mae": mae, "max_abs_diff": max_diff}

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        compare_dir = output_dir.parent / "fuse_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        before.save(compare_dir / "before_fuse.png")
        after.save(compare_dir / "after_fuse.png")

        print(f"[stage4] 差分 MAE={mae:.5f} max={max_diff:.5f}  ({compare_dir})")
        if mae > MAE_WARN_THRESHOLD:
            print(
                f"[stage4] ⚠ 焼き込み前後で絵が変わっています (MAE={mae:.5f} > {MAE_WARN_THRESHOLD})。\n"
                "         fp16 の丸め誤差が疑わしいので --device cpu で fp32 のまま焼き直すか、\n"
                "         style_weight を下げて再確認してください"
            )
        else:
            print("[stage4] 焼き込み前後の差は許容範囲です")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(output_dir))
    (output_dir / "fuse_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[stage4] 保存 -> {output_dir}")
    print("[stage4] 次: python scripts/05_export_coreai.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
