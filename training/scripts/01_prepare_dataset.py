#!/usr/bin/env python3
"""Stage 1: クラウド生成したアイコン画像を学習用に整形する。

    python scripts/01_prepare_dataset.py
    python scripts/01_prepare_dataset.py --input ~/Downloads/icons_raw --captions captions.csv

やること:
  1. 透過を背景色に合成して RGB 化
  2. パディングで正方形にしてから resolution にリサイズ
  3. metadata.jsonl にキャプションを書き出す（HF imagefolder 形式）
  4. スタイルの揃い具合を report.json に出し、揃っていなければ警告する

被写体名は --captions の CSV があればそれを、無ければファイル名から推測する。
詳細は docs/dataset.md を参照。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icon_diffusion import load_config  # noqa: E402
from icon_diffusion.images import (  # noqa: E402
    iter_source_images,
    measure,
    parse_hex_color,
    prepare_image,
    subject_from_filename,
    summarize,
)


def load_caption_overrides(path: Path | None) -> dict[str, str]:
    """`file_name,subject` の 2 列 CSV を読む。ヘッダ行があってもなくてもよい。"""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"キャプション CSV が見つかりません: {path}")

    overrides: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            key, value = row[0].strip(), row[1].strip()
            if not key or key.lower() in {"file_name", "filename", "file"}:
                continue
            # 拡張子の有無どちらでも引けるようにしておく
            overrides[key] = value
            overrides[Path(key).stem] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="設定ファイルのパス")
    parser.add_argument("--input", default=None, help="入力ディレクトリ（設定を上書き）")
    parser.add_argument("--output", default=None, help="出力ディレクトリ（設定を上書き）")
    parser.add_argument("--captions", default=None, help="file_name,subject の 2 列 CSV")
    parser.add_argument("--resolution", type=int, default=None, help="出力解像度（設定を上書き）")
    parser.add_argument("--limit", type=int, default=None, help="先頭 N 枚だけ処理する（試運転用）")
    parser.add_argument("--clean", action="store_true", help="出力ディレクトリを消してから作り直す")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    input_dir = Path(args.input).expanduser() if args.input else cfg.dataset.input_dir
    output_dir = Path(args.output).expanduser() if args.output else cfg.dataset.output_dir
    resolution = args.resolution or cfg.dataset.resolution
    background = parse_hex_color(cfg.dataset.background)

    sources = iter_source_images(input_dir)
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"[stage1] {input_dir} に対応形式の画像がありません", file=sys.stderr)
        return 1

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    overrides = load_caption_overrides(Path(args.captions).expanduser() if args.captions else None)

    print(f"[stage1] 入力      : {input_dir}  ({len(sources)} 枚)")
    print(f"[stage1] 出力      : {output_dir}")
    print(f"[stage1] 解像度    : {resolution}x{resolution}")
    print(f"[stage1] トリガー  : {cfg.style.trigger_word}")

    records: list[dict[str, str]] = []
    stats = []
    skipped: list[str] = []
    without_subject = 0

    for index, source in enumerate(sources, start=1):
        try:
            image = prepare_image(source, resolution, background)
        except Exception as exc:  # 壊れたファイルで全体を止めない
            skipped.append(f"{source.name}: {exc}")
            continue

        file_name = f"{index:04d}.png"
        image.save(images_dir / file_name, format="PNG", optimize=True)

        subject = overrides.get(source.name) or overrides.get(source.stem) or subject_from_filename(source)
        if not subject:
            without_subject += 1
        records.append(
            {
                "file_name": f"images/{file_name}",
                "text": cfg.style.prompt(subject),
                "subject": subject,
                "source": str(source.relative_to(input_dir)),
            }
        )
        stats.append(measure(image, file_name))

    if not records:
        print("[stage1] 有効な画像が 1 枚もありませんでした", file=sys.stderr)
        for line in skipped:
            print(f"  - {line}", file=sys.stderr)
        return 1

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = summarize(stats)
    report["config"] = str(cfg.path)
    report["input_dir"] = str(input_dir)
    report["resolution"] = resolution
    report["trigger_word"] = cfg.style.trigger_word
    report["skipped"] = skipped
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[stage1] 完了      : {len(records)} 枚 -> {images_dir}")
    print(f"[stage1] 例        : {records[0]['text']}")
    if without_subject:
        print(
            f"[stage1] 注意      : {without_subject} 枚は被写体名を取れず共通プロンプトになりました。"
            " --captions で補うと学習が締まります"
        )
    for line in skipped:
        print(f"[stage1] スキップ  : {line}")
    for warning in report["warnings"]:
        print(f"[stage1] ⚠ {warning}")
    if not report["warnings"]:
        print("[stage1] スタイルのばらつきは許容範囲です")
    print(f"[stage1] レポート  : {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
