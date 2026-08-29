#!/usr/bin/env python3
"""Stage 0: 学習用アイコン画像を gpt-image-2 で生成する。

    python3 scripts/00_generate_raw.py --subjects data/subjects-batch2.txt
    python3 scripts/00_generate_raw.py --subjects data/subjects-batch2.txt --dry-run

初回の 154 枚はその場のスクリプトで生成してリポジトリに残らなかった（運用メモの反省点）。
このスクリプトはその正式版。プロンプトは training/data/SOURCE.md に記録した原文と
一字一句同じにしてある。**ここを変えると既存データと分布がずれる。**

- 出力は data/raw/<slug>-01.png。既にあるファイルはスキップする（再実行が安全）
- レート制限は 1 分 5 枚（実測で 429 を食らった値）。送信側で 13 秒空ける
- 失敗は 30 秒待って 1 回だけ再試行。それでも駄目なら記録して先へ進む
  （1 枚のために全体を止めない。足りない分は再実行すれば拾われる）
- 実行には環境変数 OPENAI_API_KEY と、openai-image-gen スキル
  （~/.claude/skills/openai-image-gen）が要る
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path.home() / ".claude/skills/openai-image-gen/scripts/generate_image.py"

# SOURCE.md に記録した原文。**変えないこと。**
PROMPT = """A minimal flat vector app icon of {subject}, centered in the frame,
occupying about 70% of the canvas with equal empty margin on all four sides,
on a pure solid white #FFFFFF background.
Use strictly flat solid fills in exactly two colors: deep navy #1F2A44 and coral #FF6B4A.
No outlines, no strokes, no gradients, no shadows, no highlights, no texture,
no 3D, no perspective, no text, no letters. Orthographic straight-on front view.
Bold simple geometric shapes, chunky and readable at small size.
Single object only, nothing else in frame."""

# 評価専用に学習から外している被写体。**ここに足してはいけない**
HELD_OUT = {"a rocket", "an umbrella", "a shopping cart", "a magnifying glass"}


def slug(subject: str) -> str:
    return subject.replace(" ", "-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", required=True, help="被写体リスト（1 行 1 被写体）")
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--interval", type=float, default=13.0,
                        help="送信間隔秒。レート制限 5 枚/分 に合わせた既定 13 秒")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subjects = [line.strip() for line in Path(args.subjects).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    overlap = HELD_OUT & set(subjects)
    if overlap:
        raise SystemExit(f"評価専用の被写体が混ざっています: {sorted(overlap)}")
    if len(set(subjects)) != len(subjects):
        seen: set[str] = set()
        dupes = [s for s in subjects if s in seen or seen.add(s)]
        raise SystemExit(f"リスト内に重複があります: {dupes[:5]}")
    if not GENERATOR.is_file() and not args.dry_run:
        raise SystemExit(f"生成スクリプトがありません: {GENERATOR}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in subjects if not (out_dir / f"{slug(s)}-01.png").is_file()]
    print(f"[stage0] {len(subjects)} 被写体のうち未生成 {len(todo)} 枚"
          f"（既存はスキップ、費用目安 ${len(todo) * 0.006:.2f}）")
    if args.dry_run:
        for s in todo[:10]:
            print(f"[stage0]   {s} -> {slug(s)}-01.png")
        return 0

    failed: list[str] = []
    for index, subject in enumerate(todo):
        target = out_dir / f"{slug(subject)}-01.png"
        ok = False
        for attempt in (1, 2):
            result = subprocess.run(
                [sys.executable, str(GENERATOR),
                 "--prompt", PROMPT.format(subject=subject),
                 "--size", "1024x1024", "--quality", "low",
                 "--output", str(target)],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and target.is_file():
                ok = True
                break
            print(f"[stage0]   失敗（{attempt} 回目）: {subject}: "
                  f"{result.stderr.strip()[-200:]}", flush=True)
            time.sleep(30)
        print(f"[stage0] {index + 1}/{len(todo)} {'✓' if ok else '✗'} {subject}", flush=True)
        if not ok:
            failed.append(subject)
        time.sleep(args.interval)

    if failed:
        print(f"[stage0] 生成できなかった被写体 {len(failed)} 件: {failed}")
        print("[stage0] 再実行すれば未生成分だけ拾い直します")
    print(f"[stage0] 完了 -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
