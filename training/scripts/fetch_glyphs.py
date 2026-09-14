#!/usr/bin/env python3
"""商用可のアイコン集から、条件付け用の単色グリフを取ってくる。

    python3 scripts/fetch_glyphs.py --out data/glyphs

**出典と条件**: Google Material Symbols（**Apache License 2.0**）。
jsDelivr 経由で `@material-symbols/svg-400` の塗りつぶし版 SVG を 1 件ずつ取得する。
`data/glyphs/SOURCE.md` に出所を残すこと（docs/dataset.md の「権利まわり」と同じ扱い）。

**なぜ塗りつぶし版なのか**: ControlNet に渡すのは輪郭だが、線画版だと輪郭が二重に出る
（線の内側と外側）。塗りつぶしから境界を取れば 1 本になる。

ラスタ化は macOS の `qlmanage`。**viewBox を見てくれないので、先に width/height を
書き換える**（48 のままだと 512 の画布の隅に小さく描かれる）。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CDN = "https://cdn.jsdelivr.net/npm/@material-symbols/svg-400@latest/outlined/{name}-fill.svg"

# 評価に使う 12 被写体 -> Material Symbols の名前
MAPPING = {
    "a coffee cup": "local_cafe",
    "a house key": "key",
    "a calendar": "calendar_month",
    "a paper airplane": "send",
    "a light bulb": "lightbulb",
    "a shopping bag": "shopping_bag",
    "a camera": "photo_camera",
    "an umbrella": "umbrella",
    "a rocket": "rocket_launch",
    "an envelope": "mail",
    "a shopping cart": "shopping_cart",
    "a magnifying glass": "search",
}


def rasterize(svg: str, size: int) -> "Image.Image":
    from PIL import Image  # noqa: PLC0415

    # qlmanage は viewBox を無視するので、宣言サイズ自体を書き換える
    svg = re.sub(r'width="\d+"', f'width="{size}"', svg, count=1)
    svg = re.sub(r'height="\d+"', f'height="{size}"', svg, count=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "glyph.svg"
        path.write_text(svg)
        subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", tmp, str(path)],
                       capture_output=True, check=False)
        produced = Path(tmp) / "glyph.svg.png"
        if not produced.is_file():
            raise RuntimeError("qlmanage がラスタ化に失敗しました")
        return Image.open(produced).convert("RGBA").copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data" / "glyphs"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--coverage", type=float, default=0.30,
                        help="塗りの占有率。学習データに合わせる")
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from icon_diffusion import icon_fit  # noqa: PLC0415

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject, name in MAPPING.items():
        url = CDN.format(name=name)
        with urllib.request.urlopen(url, timeout=60) as response:
            svg = response.read().decode()
        image = rasterize(svg, args.size)

        # 透過を白地に合成してから二値化する
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, (0, 0), image)
        ink = np.asarray(flat.convert("L"), dtype=np.uint8) < 128

        # 学習データと同じ「中央・占有率 30%」に揃える。そろえないと ControlNet が
        # 画面いっぱいの輪郭を描いてしまい、余白の指定と食い違う
        ys, xs = np.nonzero(ink)
        centre_x, centre_y = xs.mean(), ys.mean()
        reach = float(np.sqrt((xs - centre_x) ** 2 + (ys - centre_y) ** 2).max())
        scale = min((args.size / 2 * 0.86) / max(reach, 1e-6),
                    (args.coverage * ink.size / max(ink.sum(), 1)) ** 0.5)
        source = Image.fromarray((ink * 255).astype(np.uint8), "L")
        a = args.size / (args.size * scale)
        placed = source.transform(
            (args.size, args.size), Image.AFFINE,
            (a, 0, centre_x - a * args.size / 2, 0, a, centre_y - a * args.size / 2),
            resample=Image.BILINEAR,
        )
        glyph = np.asarray(placed) >= 128
        Image.fromarray(np.where(glyph[..., None], 0, 255).astype(np.uint8).repeat(3, axis=2)) \
            .save(out_dir / f"{subject.replace(' ', '-')}-01.png")
        print(f"[glyph] {subject:22s} <- material-symbols/{name}")

    (out_dir / "SOURCE.md").write_text(
        "# 条件付け用グリフの出所\n\n"
        "| 項目 | 内容 |\n| --- | --- |\n"
        "| 出典 | Google Material Symbols |\n"
        "| ライセンス | **Apache License 2.0**（商用利用可・改変可・再配布可）|\n"
        "| 取得元 | https://cdn.jsdelivr.net/npm/@material-symbols/svg-400 |\n"
        "| 取得日 | 2026-09-14 |\n"
        "| 加工 | 塗りつぶし版 SVG を 512px でラスタ化し、二値化して中央・占有率 30% に揃えた |\n\n"
        "**この画像は学習に使っていない。** ControlNet の条件（輪郭）として推論時にだけ使う。\n"
        "Apache-2.0 は帰属表示を求めるので、配布物には NOTICE を同梱すること。\n\n"
        "被写体との対応:\n\n"
        + "\n".join(f"- `{s}` → `{n}`" for s, n in MAPPING.items()) + "\n",
        encoding="utf-8")
    print(f"[glyph] 完了 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
