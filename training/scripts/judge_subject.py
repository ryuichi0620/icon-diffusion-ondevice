#!/usr/bin/env python3
"""生成物が「狙った被写体に見えるか」を測る。**このプロジェクトの意味側の物差し。**

    python3 scripts/judge_subject.py outputs/generated
    python3 scripts/judge_subject.py outputs/a outputs/b --paired    # 設定どうしの比較
    python3 scripts/judge_subject.py --ceiling                       # 判定器の天井だけ出す

## なぜこの形なのか（3 つとも過去に踏んだ失敗への対処）

**1. 開集合で測る。** 評価被写体 12 件だけを選択肢にすると、コーヒーカップ→ティーカップ、
買い物袋→南京錠、封筒→蝶ネクタイのような**近傍誤りが全部「正解」になる**。それは
ユーザーが違和感として挙げているものそのもので、12 択はそれを構造的に見られない。
既定では学習した全被写体（373 件）+ 評価 12 件を選択肢にする。

**2. 判定器と選別器を別系統にする。** `pick_by_subject.py` は OpenAI CLIP で選ぶ。
同じモデルで採点すると「テスト指標で選んで、そのテスト指標で測る」ことになり、
数字は自明に高く出る。既定の判定器は **SigLIP 2**（学習データも目的関数も別）。

**3. 判定器の天井を併記する。** 学習データ（gpt-image-2 の生画像）自身をこの物差しに
掛けると 6 割前後しか当たらない。**絶対値は厳しすぎるので、必ず天井と並べて読むこと。**
設定どうしの比較には `--paired`（同じ被写体・同じ連番どうしの符号検定）を使う。
平均どうしを並べる比較は、被写体間のばらつき（標準偏差 20 ポイント超）に埋もれる。
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVAL_SUBJECTS = ["a coffee cup", "a house key", "a calendar", "a paper airplane",
                 "a light bulb", "a shopping bag", "a camera", "an umbrella",
                 "a rocket", "an envelope", "a shopping cart", "a magnifying glass"]
JUDGES = {
    "siglip2": "google/siglip2-base-patch16-224",
    "laion": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "openai": "openai/clip-vit-base-patch32",   # 選別に使っているので判定には使わない
}
TEMPLATE = "a simple flat icon of {subject}"


def label_space() -> list[str]:
    data = ROOT / "data"
    pool = set(EVAL_SUBJECTS)
    for path in data.glob("subjects-*.txt"):
        pool |= {line.strip() for line in path.read_text().splitlines()
                 if line.strip() and not line.startswith("#")}
    pool |= {p.stem.rsplit("-", 1)[0].replace("-", " ") for p in (data / "raw").glob("*.png")}
    return sorted(pool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="採点する画像ディレクトリ")
    parser.add_argument("--judge", default="siglip2", choices=sorted(JUDGES))
    parser.add_argument("--closed", action="store_true",
                        help="評価 12 被写体だけを選択肢にする（近傍誤りを見逃す。比較用）")
    parser.add_argument("--paired", action="store_true",
                        help="2 つ以上の入力を、同じ被写体・同じ連番どうしで比べる")
    parser.add_argument("--ceiling", action="store_true",
                        help="学習データ自身を測って判定器の天井を出す")
    parser.add_argument("--ceiling-sample", type=int, default=150)
    parser.add_argument("--no-polish", action="store_true", help="整形せずに測る")
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
    from icon_diffusion import shapes  # noqa: PLC0415
    from icon_diffusion.postprocess import remove_flat_background  # noqa: PLC0415

    labels = EVAL_SUBJECTS if args.closed else label_space()
    model = AutoModel.from_pretrained(JUDGES[args.judge]).eval()
    processor = AutoProcessor.from_pretrained(JUDGES[args.judge])

    def embed_text(items: list[str]) -> "torch.Tensor":
        chunks = []
        for start in range(0, len(items), 64):
            batch = [TEMPLATE.format(subject=s) for s in items[start:start + 64]]
            with torch.no_grad():
                got = model.get_text_features(
                    **processor(text=batch, return_tensors="pt", padding="max_length"))
            chunks.append(got if isinstance(got, torch.Tensor) else got.pooler_output)
        features = torch.cat(chunks)
        return features / features.norm(dim=-1, keepdim=True)

    text = embed_text(labels)
    position = {s: i for i, s in enumerate(labels)}

    def judge(path: Path) -> int:
        image = Image.open(path)
        if not args.no_polish:
            if image.mode != "RGBA" or int(np.asarray(image)[..., 3].min()) > 0:
                image = remove_flat_background(image.convert("RGB"))
            image, _ = shapes.polish(image, size=224, background="white")
        with torch.no_grad():
            got = model.get_image_features(
                **processor(images=image.convert("RGB"), return_tensors="pt"))
        features = got if isinstance(got, torch.Tensor) else got.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return int((features @ text.T)[0].argmax())

    if args.ceiling:
        files = sorted((ROOT / "data" / "raw").glob("*.png"))
        random.seed(0)
        sample = random.sample(files, min(args.ceiling_sample, len(files)))
        hits = sum(labels[judge(f)] == f.stem.rsplit("-", 1)[0].replace("-", " ")
                   for f in sample
                   if f.stem.rsplit("-", 1)[0].replace("-", " ") in position)
        print(f"[judge] 天井（学習データ {len(sample)} 枚・{len(labels)} 択・{args.judge}）: "
              f"{hits}/{len(sample)} = {hits / len(sample):.1%}")
        if not args.inputs:
            return 0

    results: dict[str, dict[str, bool]] = {}
    for source in args.inputs:
        directory = Path(source)
        marks: dict[str, bool] = {}
        for file in sorted(directory.glob("*.png")):
            if file.stem.startswith("contact"):
                continue
            subject = re.sub(r"_\d+$", "", file.stem).replace("-", " ")
            if subject not in position:
                continue
            marks[file.stem] = labels[judge(file)] == subject
        results[directory.name] = marks
        hits = sum(marks.values())
        print(f"[judge] {directory.name}: {hits}/{len(marks)} = "
              f"{hits / max(len(marks), 1):.1%}（{len(labels)} 択・{args.judge}）")

    if args.paired and len(results) >= 2:
        names = list(results)
        base = results[names[0]]
        for other in names[1:]:
            shared = sorted(set(base) & set(results[other]))
            better = sum(results[other][k] and not base[k] for k in shared)
            worse = sum(base[k] and not results[other][k] for k in shared)
            # 符号検定（両側・正規近似を使わず二項分布そのままで）
            from math import comb  # noqa: PLC0415
            n, k = better + worse, min(better, worse)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
            print(f"[judge] {names[0]} -> {other}: 対応のある {len(shared)} 枚で "
                  f"改善 {better} / 悪化 {worse}（符号検定 p={p:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
