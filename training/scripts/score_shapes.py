#!/usr/bin/env python3
"""生成済みアイコンの「形」を採点し、整形版を書き出す。

    python3 scripts/score_shapes.py outputs/tp-b5-seed42 --polished outputs/polish-b5-seed42
    python3 scripts/score_shapes.py outputs/sweep-* --quiet     # 設定ごとの比較

透過成功率は「背景が抜けたか」しか見ておらず、**形が歪かどうかを測れない**。
このスクリプトは icon_diffusion.shapes の採点（塗りの量・破片・輪郭の粗さ・中間色）を
まとめて掛け、設定やモデルの比較に使える数字にする。

被写体ごとに 4 枚出す運用なので、平均点だけでなく **4 枚の最良点の平均**（best-of-4）を
出す。実際に人が見るのは 4 枚のうちの一番良い 1 枚なので、こちらが体感に近い。
ファイル名は `<被写体>_<連番>.png` を前提にして被写体をまとめる（generate.py の命名）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS_SCORE = 0.75   # これ以上を「アイコンとして出せる形」とみなす（48 枚を人が見て決めた線）


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="採点する画像ディレクトリ")
    parser.add_argument("--polished", help="整形版の書き出し先（入力が 1 つのときだけ）")
    parser.add_argument("--size", type=int, default=1024, help="整形版の解像度")
    parser.add_argument("--json", help="採点結果の書き出し先")
    parser.add_argument("--quiet", action="store_true", help="1 枚ごとの行を出さない")
    args = parser.parse_args()

    from PIL import Image  # noqa: PLC0415  torch 無しで --help を通すため
    import numpy as np  # noqa: PLC0415
    from icon_diffusion import shapes  # noqa: PLC0415
    from icon_diffusion.postprocess import remove_flat_background  # noqa: PLC0415

    if args.polished and len(args.inputs) > 1:
        raise SystemExit("--polished は入力ディレクトリが 1 つのときだけ使えます")

    everything: dict[str, list[dict]] = {}
    for source in args.inputs:
        directory = Path(source)
        files = sorted(f for f in directory.glob("*.png") if not f.stem.startswith("contact"))
        if not files:
            print(f"[score] {directory}: png がありません")
            continue
        out_dir = Path(args.polished) if args.polished else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for file in files:
            image = Image.open(file)
            # 透過済みならそのまま、不透明なら先に背景を抜く（どちらの出力でも受ける）
            if image.mode != "RGBA" or int(np.asarray(image)[..., 3].min()) > 0:
                image = remove_flat_background(image.convert("RGB"))
            polished, report = shapes.polish(image, size=args.size)
            if out_dir:
                polished.save(out_dir / file.name)
            report["name"] = file.stem
            report["subject"] = re.sub(r"_\d+$", "", file.stem)
            rows.append(report)

        everything[str(directory)] = rows
        summarise(directory.name, rows, quiet=args.quiet)

    if args.json:
        Path(args.json).write_text(json.dumps(everything, indent=1, ensure_ascii=False))
        print(f"[score] 採点結果 -> {args.json}")
    return 0


def summarise(name: str, rows: list[dict], *, quiet: bool) -> None:
    if not quiet:
        for row in sorted(rows, key=lambda r: -r["score"]):
            print(f'  {row["score"]:.3f}  塗り {row["coverage"]:.2f}  塊 {row["parts"]:2d}  '
                  f'散り {row["stray"]:.2f}  粗さ {row["rough"]:.2f}  中間色 {row["mush"]:.2f}  '
                  f'{row["name"]}')
    scores = [row["score"] for row in rows]
    best: dict[str, float] = {}
    for row in rows:
        best[row["subject"]] = max(best.get(row["subject"], 0.0), row["score"])
    passed = sum(1 for score in scores if score >= PASS_SCORE)
    print(f"[score] {name}: {len(rows)} 枚  平均 {sum(scores) / len(scores):.3f}  "
          f"合格({PASS_SCORE}以上) {passed}/{len(rows)} = {passed / len(rows) * 100:.1f}%  "
          f"被写体ごとの最良の平均 {sum(best.values()) / len(best):.3f}  "
          f"（被写体 {len(best)} 件）")


if __name__ == "__main__":
    raise SystemExit(main())
