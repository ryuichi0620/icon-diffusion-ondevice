#!/usr/bin/env python3
"""条件付け用グリフのライブラリを組み立てる（取得 → ラスタ化 → 引き当て表）。

    python3 scripts/build_glyph_library.py
    python3 scripts/build_glyph_library.py --threshold 0.14 --probe "a lighthouse"

**出典と条件**: Google Material Symbols / **Apache License 2.0**（商用可・改変可・再配布可。
帰属表示が要る）。パッケージは版を固定して取得する。**学習には使わない。**
推論時に ControlNet へ渡す輪郭としてだけ使う。

## 引き当ては名前ではなく絵で行う

Material Symbols の名前は `local_cafe` のように被写体語と綴りが一致しないものが多く、
文字列一致では引けない。そこで全グリフを SigLIP の**画像側**で埋め込んでおき、
被写体の**文章**を同じ空間に写して最近傍を取る。

## 当たらなかったときは条件を付けない

Material Symbols は UI のアイコン集であって物体の辞書ではない。実測では、学習にも評価にも
無い 12 被写体のうち 8 件は妥当なグリフに当たったが、4 件（折り鶴・スノードーム・クラゲ・
灯台）は存在しなかった。**外れたときの類似度は明確に低い**ので（0.10〜0.13 対 0.155〜0.19）、
閾値で落として文字だけの生成に回す。**間違った形を条件に与えるのは、条件を付けないより悪い。**
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PACKAGE = "@material-symbols/svg-400"
VERSION = "0.47.2"
TARBALL = f"https://registry.npmjs.org/{PACKAGE}/-/svg-400-{VERSION}.tgz"
JUDGE = "google/siglip2-base-patch16-224"
# 実測で決めた線。これを下回る一致度なら条件を付けない
DEFAULT_THRESHOLD = 0.14


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data" / "glyphs"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--coverage", type=float, default=0.30)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--probe", default=None, help="引き当てだけ試す（構築済みが要る）")
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

    out_dir = Path(args.out)
    vectors_path = out_dir / "vectors.npy"
    index_path = out_dir / "index.json"

    model = AutoModel.from_pretrained(JUDGE).eval()
    processor = AutoProcessor.from_pretrained(JUDGE)

    def embed_text(items: list[str]) -> "np.ndarray":
        prompts = [f"a simple flat icon of {s}" for s in items]
        chunks = []
        for start in range(0, len(prompts), 64):
            with torch.no_grad():
                got = model.get_text_features(**processor(
                    text=prompts[start:start + 64], return_tensors="pt", padding="max_length"))
            v = got if isinstance(got, torch.Tensor) else got.pooler_output
            chunks.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
        return np.concatenate(chunks)

    if args.probe:
        if not vectors_path.is_file():
            raise SystemExit("先に構築してください（--probe は構築済みが前提）")
        matrix = np.load(vectors_path)
        names = json.loads(index_path.read_text())
        row = (embed_text([args.probe]) @ matrix.T)[0]
        order = np.argsort(-row)[:5]
        for i in order:
            mark = "採用" if row[i] >= args.threshold else "閾値未満"
            print(f"  {names[i]:28s} {row[i]:.3f}  {mark}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print(f"[glyph] 取得: {PACKAGE}@{VERSION}")
        archive = work / "package.tgz"
        with urllib.request.urlopen(TARBALL, timeout=300) as response:
            archive.write_bytes(response.read())
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter="data")
        source = work / "package" / "outlined"
        license_text = (work / "package" / "LICENSE").read_text()

        # qlmanage は viewBox を見ないので、宣言サイズを書き換えてから渡す
        staged = work / "svg"
        staged.mkdir()
        for svg_path in sorted(source.glob("*-fill.svg")):
            svg = svg_path.read_text()
            svg = re.sub(r'width="\d+"', f'width="{args.size}"', svg, count=1)
            svg = re.sub(r'height="\d+"', f'height="{args.size}"', svg, count=1)
            (staged / svg_path.name.replace("-fill", "")).write_text(svg)
        print(f"[glyph] ラスタ化: {len(list(staged.glob('*.svg')))} 件")

        raster = work / "png"
        raster.mkdir()
        files = sorted(staged.glob("*.svg"))
        for start in range(0, len(files), 200):   # まとめて渡すと桁違いに速い
            subprocess.run(["qlmanage", "-t", "-s", str(args.size), "-o", str(raster)]
                           + [str(f) for f in files[start:start + 200]],
                           capture_output=True, check=False)

        names, images = [], []
        for png in sorted(raster.glob("*.png")):
            image = Image.open(png).convert("RGBA")
            flat = Image.new("RGB", image.size, "white")
            flat.paste(image, (0, 0), image)
            ink = np.asarray(flat.convert("L"), dtype=np.uint8) < 128
            if not ink.any():
                continue
            # 学習データと同じ「中央・占有率 30%」に揃える
            ys, xs = np.nonzero(ink)
            cx, cy = xs.mean(), ys.mean()
            reach = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
            scale = min((args.size / 2 * 0.86) / max(reach, 1e-6),
                        (args.coverage * ink.size / max(ink.sum(), 1)) ** 0.5)
            a = 1.0 / scale
            placed = Image.fromarray((ink * 255).astype(np.uint8), "L").transform(
                (args.size, args.size), Image.AFFINE,
                (a, 0, cx - a * args.size / 2, 0, a, cy - a * args.size / 2),
                resample=Image.BILINEAR)
            glyph = np.asarray(placed) >= 128
            name = png.stem.replace(".svg", "")
            Image.fromarray(np.where(glyph[..., None], 0, 255)
                            .astype(np.uint8).repeat(3, axis=2)).save(out_dir / f"{name}.png")
            names.append(name)
            images.append(Image.fromarray(np.where(glyph, 0, 255).astype(np.uint8)).convert("RGB"))

        print(f"[glyph] 埋め込み: {len(images)} 件")
        chunks = []
        for start in range(0, len(images), 64):
            with torch.no_grad():
                got = model.get_image_features(**processor(
                    images=images[start:start + 64], return_tensors="pt"))
            v = got if isinstance(got, torch.Tensor) else got.pooler_output
            chunks.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
        matrix = np.concatenate(chunks)
        np.save(vectors_path, matrix)
        index_path.write_text(json.dumps(names, ensure_ascii=False))
        (out_dir / "LICENSE").write_text(license_text)

    (out_dir / "SOURCE.md").write_text(
        "# 条件付け用グリフの出所\n\n"
        f"| 項目 | 内容 |\n| --- | --- |\n"
        f"| 出典 | Google Material Symbols |\n"
        f"| パッケージ | `{PACKAGE}@{VERSION}`（版を固定）|\n"
        f"| ライセンス | **Apache License 2.0**。全文は同ディレクトリの `LICENSE` |\n"
        f"| 件数 | {len(names)} |\n"
        f"| 加工 | 塗りつぶし版 SVG を {args.size}px でラスタ化 → 二値化 → "
        f"中央・占有率 {args.coverage:.0%} に揃える |\n\n"
        "**学習には使っていない。** ControlNet の条件（輪郭）として推論時にだけ使う。\n\n"
        "## 配布するときの義務\n\n"
        "Apache-2.0 は帰属表示を求める。アプリに同梱する場合は\n"
        "**ライセンス全文と著作権表示を画面から読める場所に置くこと**。\n\n"
        "## 形が共通になることについて\n\n"
        "この方式では出力のシルエットが Material Symbols と共通になる。\n"
        "ライセンス上は問題ないが、企画としての判断は別途必要（2026-09-15 に「可」と判断）。\n",
        encoding="utf-8")
    print(f"[glyph] 完了 -> {out_dir}（{len(names)} 件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
