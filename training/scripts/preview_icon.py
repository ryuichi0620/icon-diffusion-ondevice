#!/usr/bin/env python3
"""アイコン 1 枚を「SNS で実際にどう見えるか」で検証するシートを作る。

    python3 scripts/preview_icon.py outputs/generated/a-coffee-cup_01.png
    python3 scripts/preview_icon.py outputs/generated --limit 12 --out outputs/preview-sns

資料（SNS アイコンの作り方）の検証手順をそのまま並べる。**目で見て落とすための道具で、
点数を出す道具ではない。**

  1. 1024px … 書き出しの原寸
  2. 98px / 48px … YouTube のコメント欄と X のタイムラインの実寸。
     ここで潰れるなら要素を削るしかない
  3. グレースケール … 色を抜いて明暗だけで成立するか
  4. 円クロップ … SNS はほぼ例外なく円に切り抜く
  5. 明background / 暗background … どちらのタイムラインでも沈まないか

各段の下に、輪郭の色と背景色のコントラスト比（WCAG 1.4.11 は 3:1 以上）を出す。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CARD = 1024
SMALL_SIZES = (98, 48)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="画像 1 枚、またはディレクトリ")
    parser.add_argument("--out", default=None, help="書き出し先（既定は入力の隣）")
    parser.add_argument("--background", default="auto",
                        choices=["none", "auto", "white", "navy", "coral", "dark"])
    parser.add_argument("--limit", type=int, default=8, help="ディレクトリのときの枚数")
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415
    from icon_diffusion import icon_fit, shapes  # noqa: PLC0415
    from icon_diffusion.postprocess import remove_flat_background  # noqa: PLC0415

    source = Path(args.input)
    files = ([f for f in sorted(source.glob("*.png")) if not f.stem.startswith("contact")][:args.limit]
             if source.is_dir() else [source])
    out_dir = Path(args.out) if args.out else (source if source.is_dir() else source.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        image = Image.open(file)
        if image.mode != "RGBA" or int(np.asarray(image)[..., 3].min()) > 0:
            image = remove_flat_background(image.convert("RGB"))
        big, report = shapes.polish(image, size=CARD, background=args.background)

        cell = 260
        columns = 2 + len(SMALL_SIZES) + 2
        sheet = Image.new("RGB", (cell * columns, cell + 44), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)

        def place(index: int, tile: Image.Image, caption: str) -> None:
            sheet.paste(tile.resize((cell - 8, cell - 8), Image.LANCZOS), (index * cell + 4, 4))
            draw.text((index * cell + 6, cell + 8), caption, fill=(20, 20, 20))

        def flatten(image_: Image.Image, under) -> Image.Image:
            base = Image.new("RGB", image_.size, under)
            base.paste(image_.convert("RGBA"), (0, 0), image_.convert("RGBA"))
            return base

        place(0, flatten(big, "white"), "原寸 1024px")
        for offset, size in enumerate(SMALL_SIZES, start=1):
            small = flatten(big.resize((size, size), Image.LANCZOS), "white")
            place(offset, small.resize((cell - 8, cell - 8), Image.NEAREST), f"実寸 {size}px")

        grey = flatten(big, "white").convert("L").convert("RGB")
        place(1 + len(SMALL_SIZES), grey, "グレースケール")

        mask = Image.new("L", big.size, 0)
        ImageDraw.Draw(mask).ellipse([0, 0, big.width - 1, big.height - 1], fill=255)
        for offset, (under, name) in enumerate((((245, 245, 245), "明るい地"),
                                                ((18, 18, 18), "暗い地")), start=2 + len(SMALL_SIZES)):
            tile = Image.new("RGB", big.size, under)
            tile.paste(flatten(big, under), (0, 0), mask)
            place(offset, tile, f"円クロップ／{name}")

        ratio = report.get("background_contrast")
        draw.text((6, cell + 26),
                  f"{file.stem}  形 {report['score']:.2f}  背景 {report.get('background', 'none')}"
                  f"  コントラスト {ratio:.2f}:1（3:1 以上が基準）" if ratio else
                  f"{file.stem}  形 {report['score']:.2f}",
                  fill=(60, 60, 60))

        target = out_dir / f"{file.stem}_sns.png"
        sheet.save(target)
        print(f"[preview] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
