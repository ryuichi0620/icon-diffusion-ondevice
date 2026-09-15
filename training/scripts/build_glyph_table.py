#!/usr/bin/env python3
"""語 → グリフの対応表を先に計算しておく（端末に判定器を載せずに引けるようにする）。

    python3 scripts/build_glyph_table.py

**なぜ表にするのか。** 引き当ての質は SigLIP でないと出ない（CLIP では 1 位の一致が
40 件中 5 件しかなく、類似度も 0.31〜0.40 に固まって閾値が効かない）。しかし SigLIP2 の
テキスト側は **282M パラメータ**で、その 197M が語彙 25.6 万の埋め込み表、しかも
トークナイザが SentencePiece。端末に載せるには大きすぎ、移植も重い。

そこで **Mac 側で SigLIP を使って先に全部引いておく**。端末は文字列で引くだけになる。
引き当ての質は SigLIP のまま、辞書に無い語だけが条件なしに落ちる。

語彙は macOS の `/usr/share/dict/words`（20.7 万語）＋学習に使った被写体の語句。
一致度が閾値未満の語は**表に入れない**（引けなかったことにする）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

JUDGE = "google/siglip2-base-patch16-224"
WORDS = Path("/usr/share/dict/words")
# 関門は 2 つ。**絵の一致だけでは足りない。**
# ベルトが window_closed に 0.147 で当たるなど、見当違いのグリフが妥当な範囲に入る。
# グリフの「名前」と被写体の意味的な近さを second gate にすると分離する
# （実測: 妥当な対応 0.895〜0.994 / 見当違い 0.763〜0.892）
DEFAULT_IMAGE_THRESHOLD = 0.14
DEFAULT_NAME_THRESHOLD = 0.90


def vocabulary() -> list[str]:
    """引けるようにしておきたい語句。単語と、学習に使った語句の両方。"""
    items: list[str] = []
    if WORDS.is_file():
        items += [w.strip() for w in WORDS.read_text().splitlines()
                  if w.strip().isalpha() and 3 <= len(w.strip()) <= 16 and w.strip().islower()]
    data = ROOT / "data"
    for path in data.glob("subjects-*.txt"):
        items += [l.strip() for l in path.read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    items += [p.stem.rsplit("-", 1)[0].replace("-", " ") for p in (data / "raw").glob("*.png")]
    prompts = ROOT / "prompts.example.txt"
    if prompts.is_file():
        items += [l.strip() for l in prompts.read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    # 「a coffee cup」と「coffee cup」の両方で引けるようにする
    stripped = [i[2:] for i in items if i.startswith("a ")] + \
               [i[3:] for i in items if i.startswith("an ")]
    return sorted({i.lower() for i in items + stripped if i})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", default=str(ROOT / "data" / "glyphs"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_IMAGE_THRESHOLD,
                        help="グリフの絵との一致度の下限")
    parser.add_argument("--name-threshold", type=float, default=DEFAULT_NAME_THRESHOLD,
                        help="グリフの名前との意味的な近さの下限。**見当違いを弾くのはこちら**")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="試運転用")
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

    library = Path(args.library)
    matrix = np.load(library / "vectors.npy")
    names = json.loads((library / "index.json").read_text())

    model = AutoModel.from_pretrained(JUDGE).eval()
    processor = AutoProcessor.from_pretrained(JUDGE)

    def embed(texts: list[str]) -> "np.ndarray":
        chunks = []
        for start in range(0, len(texts), args.batch):
            with torch.no_grad():
                got = model.get_text_features(**processor(
                    text=texts[start:start + args.batch],
                    return_tensors="pt", padding="max_length"))
            v = got if isinstance(got, torch.Tensor) else got.pooler_output
            chunks.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
        return np.concatenate(chunks)

    print("[table] グリフ名を埋め込み中")
    name_vectors = embed([f"a simple flat icon of {n.replace('_', ' ')}" for n in names])

    items = vocabulary()
    if args.limit:
        items = items[:args.limit]
    print(f"[table] 語彙 {len(items)} 件を引きます")

    table: dict[str, str] = {}
    kept = 0
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        with torch.no_grad():
            got = model.get_text_features(**processor(
                text=[f"a simple flat icon of {s}" for s in chunk],
                return_tensors="pt", padding="max_length"))
        v = got if isinstance(got, torch.Tensor) else got.pooler_output
        v = (v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
        scores = v @ matrix.T
        best = scores.argmax(axis=1)
        for word, index, row, query in zip(chunk, best, scores, v):
            index = int(index)
            if float(row[index]) < args.threshold:
                continue
            # 第 2 の関門: グリフの名前も被写体に近いこと
            if float(query @ name_vectors[index]) < args.name_threshold:
                continue
            table[word] = names[index]
            kept += 1
        if start % (args.batch * 40) == 0:
            print(f"[table]   {start}/{len(items)}  採用 {kept}", flush=True)

    out = library / "table.tsv.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for word in sorted(table):
            f.write(f"{word}\t{table[word]}\n")
    raw = sum(len(w) + len(g) + 2 for w, g in table.items())
    print(f"[table] 採用 {len(table)}/{len(items)} 語  "
          f"-> {out}（生 {raw/1e6:.1f}MB / 圧縮後 {out.stat().st_size/1e6:.2f}MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
