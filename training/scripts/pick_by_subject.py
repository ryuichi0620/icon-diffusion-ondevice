#!/usr/bin/env python3
"""候補のうち「狙った被写体に一番近い 1 枚」を CLIP で選ぶ。

    python3 scripts/pick_by_subject.py outputs/generated --out outputs/picked
    python3 scripts/pick_by_subject.py outputs/generated --report-only

**形の採点（shapes.shape_report）は意味を測れない。** 丸の中に模様があるだけの絵が
「虫めがね 1.00」になる。一方でこのモデルは、4 枚出せばそのうち 1 枚は正しい被写体に
なっていることが多い（実測で 46/48 被写体）。つまり**正しい絵は既に生成できていて、
選べていないだけ**だった。ここを CLIP の画像↔テキスト一致度で選び直す。

実測（b5・8 步・48 枚 × 4 条件）:

    形の点で選ぶ  35/48 被写体
    CLIP で選ぶ   45/48 被写体
    4 枚のどれかが正解 46/48（＝ほぼ上限まで拾えている）

CLIP がこの絵柄で使えることは 12 択の正答率で確かめた（79%。偶然なら 8.3%）。
**ただし「正答率 79%」は 12 択の話で、人が見て 79% が正しく見えるという意味ではない。**
設定どうしを比べる物差しとしてだけ使うこと。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "openai/clip-vit-base-patch32"
# 「アイコンである」ことを文に入れる。写真として評価されると全部外れる
TEMPLATE = "a simple flat icon of {subject}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="候補が入ったディレクトリ（<被写体>_<連番>.png）")
    parser.add_argument("--out", default=None, help="選んだ 1 枚の書き出し先")
    parser.add_argument("--report-only", action="store_true", help="選ぶだけで書き出さない")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
    from icon_diffusion import shapes  # noqa: PLC0415
    from icon_diffusion.postprocess import remove_flat_background  # noqa: PLC0415

    directory = Path(args.input)
    files = [f for f in sorted(directory.glob("*.png")) if not f.stem.startswith("contact")]
    if not files:
        raise SystemExit(f"png がありません: {directory}")

    groups: dict[str, list[Path]] = {}
    for file in files:
        groups.setdefault(re.sub(r"_\d+$", "", file.stem), []).append(file)

    model = CLIPModel.from_pretrained(args.model).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    def embed_text(subjects: list[str]) -> "torch.Tensor":
        prompts = [TEMPLATE.format(subject=s.replace("-", " ")) for s in subjects]
        with torch.no_grad():
            got = model.get_text_features(
                **processor(text=prompts, return_tensors="pt", padding=True))
        features = got if isinstance(got, torch.Tensor) else got.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    subjects = sorted(groups)
    text = embed_text(subjects)

    out_dir = Path(args.out) if args.out else None
    if out_dir and not args.report_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    for position, subject in enumerate(subjects):
        scored = []
        for file in groups[subject]:
            image = Image.open(file)
            if image.mode != "RGBA" or int(np.asarray(image)[..., 3].min()) > 0:
                image = remove_flat_background(image.convert("RGB"))
            # 採点と同じ整形を通してから測る。端の切れ端が残っていると一致度が下がる
            polished, report = shapes.polish(image, size=224, background="white")
            with torch.no_grad():
                got = model.get_image_features(
                    **processor(images=polished.convert("RGB"), return_tensors="pt"))
            features = got if isinstance(got, torch.Tensor) else got.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            similarity = float(features @ text[position])
            scored.append((similarity, report["score"], file))

        scored.sort(key=lambda item: -item[0])
        best = scored[0]
        by_shape = max(scored, key=lambda item: item[1])
        mark = "" if best[2] == by_shape[2] else f"（形の点なら {by_shape[2].stem}）"
        print(f"[pick] {subject:22s} -> {best[2].name}  一致度 {best[0]:.3f} {mark}")
        if out_dir and not args.report_only:
            shutil.copy2(best[2], out_dir / f"{subject}.png")

    if out_dir and not args.report_only:
        print(f"[pick] 選んだ {len(subjects)} 枚 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
