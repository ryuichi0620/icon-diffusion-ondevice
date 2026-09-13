#!/usr/bin/env python3
"""完成したモデルでアイコンを量産する。フォルダに入れて、フォルダから出す。

    # 被写体リストから 1 つずつ 4 枚ずつ、背景を透過にして書き出す
    python scripts/generate.py --prompts prompts.example.txt --count 4 --transparent

    # その場で指定する
    python scripts/generate.py --subjects "a coffee cup" "a house key" --count 2

    # 焼き込み済みモデル (Stage 4 の出力) を使う
    python scripts/generate.py --model outputs/fused --prompts prompts.example.txt

`03_preview_lcm.py` が「LoRA の効きを診断する」道具なのに対して、こちらは
「できたモデルを使う」道具。スタイル重みのスイープは無く、代わりに
被写体リスト・枚数・透過処理・再現用の manifest がある。

出力:
    <out>/a-coffee-cup_01.png ...   1 枚ずつ
    <out>/contact_sheet.png         並べて見るためのシート
    <out>/manifest.json             どの seed でどのプロンプトから出たかの記録
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from icon_diffusion import load_config, load_torch_stack, shapes  # noqa: E402
from icon_diffusion.postprocess import (  # noqa: E402
    palette_purity,
    remove_flat_background,
    snap_to_palette,
    transparency_ratio,
)

SHEET_BACKGROUND = (235, 235, 235)


def read_prompt_file(path: Path) -> list[str]:
    """1 行 1 件。空行と # 始まりは読み飛ばす。"""
    if not path.is_file():
        raise FileNotFoundError(f"プロンプトファイルがありません: {path}")
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    if not lines:
        raise ValueError(f"プロンプトファイルが空です: {path}")
    return lines


def safe_name(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "icon"


def flatten_for_sheet(image: Image.Image) -> Image.Image:
    """コンタクトシート用に、透過を薄いグレーに敷いて可視化する。"""
    if image.mode != "RGBA":
        return image.convert("RGB")
    canvas = Image.new("RGBA", image.size, SHEET_BACKGROUND + (255,))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)

    source = parser.add_argument_group("何を作るか")
    source.add_argument("--prompts", default=None, help="被写体を 1 行 1 件で書いたファイル")
    source.add_argument("--subjects", nargs="*", default=None, help="被写体を直接指定する")
    source.add_argument(
        "--raw-prompts",
        default=None,
        help="テンプレートを通さない完成済みプロンプトのファイル（上級者向け）",
    )
    source.add_argument("--count", type=int, default=1, help="1 件あたりの枚数（seed を変える）")

    model = parser.add_argument_group("どのモデルで作るか")
    model.add_argument("--model", default=None, help="焼き込み済みパイプライン（既定はベース＋LoRA）")
    model.add_argument("--lora-dir", default=None, help="スタイル LoRA のディレクトリ")
    model.add_argument("--style-weight", type=float, default=None)
    model.add_argument("--style-blocks", default=None,
                       help="スタイル LoRA をブロック別にスケールする（例 down=0.0,mid=0.5,up=1.0）。"
                            "**再学習が要らない。** スタイル LoRA は attn2（テキストと画像を"
                            "結ぶ層）にも入っており、down を弱めると被写体の構図を土台に"
                            "任せられる")
    model.add_argument("--lcm-weight", type=float, default=None)

    sampling = parser.add_argument_group("生成の設定")
    sampling.add_argument("--steps", type=int, default=None, help="推論ステップ数（既定 4）")
    sampling.add_argument("--guidance", type=float, default=None)
    sampling.add_argument("--seed", type=int, default=None, help="基準 seed。2 枚目以降は +1 していく")
    sampling.add_argument("--device", default=None, help="mps / cuda / cpu")

    output = parser.add_argument_group("出力")
    output.add_argument("--out", default=None, help="出力ディレクトリ（既定 outputs/generated）")
    output.add_argument("--transparent", action="store_true", help="背景を透過にして RGBA で保存する")
    output.add_argument("--negative", default=None,
                        help="ネガティブプロンプトの上書き（実験用）。既定は設定ファイルの値")
    output.add_argument("--snap-palette", action="store_true",
                        help="不透明画素をパレット（ネイビー/コーラル/白）へ量子化する。"
                             "--transparent の後に適用される。色のはみ出しを決定的に消す")
    output.add_argument("--polish", action="store_true",
                        help="形を整えて滑らかに描き直す（ゴミ取り・穴埋め・輪郭の平滑化 + "
                             "パレット量子化）。--transparent を含む。--snap-palette は不要")
    output.add_argument("--background", default="none",
                        choices=["none", "auto", "white", "navy", "coral", "dark"],
                        help="敷く背景。SNS のアイコンは円に切り抜かれるので、透過のままだと"
                             "タイムラインの地の色に左右される。auto は輪郭の色に対して"
                             "WCAG 1.4.11 の 3:1 を満たす色を選ぶ（--polish と併用）")
    output.add_argument("--no-fit", action="store_true",
                        help="円クロップへの整形（視覚重心へ寄せ、大きさを揃える）をしない")
    output.add_argument("--polish-size", type=int, default=None,
                        help="整形後の解像度（既定は元のまま。1024 にすると拡大して書き出す）")
    output.add_argument("--tolerance", type=float, default=0.06, help="背景とみなす色差")
    output.add_argument("--feather", type=float, default=0.12, help="輪郭を半透明にする幅")
    output.add_argument("--no-sheet", action="store_true", help="コンタクトシートを作らない")

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    steps = args.steps or cfg.preview.steps
    guidance = cfg.preview.guidance_scale if args.guidance is None else args.guidance
    style_weight = cfg.preview.style_weight if args.style_weight is None else args.style_weight
    lcm_weight = cfg.preview.lcm_weight if args.lcm_weight is None else args.lcm_weight
    base_seed = cfg.preview.seed if args.seed is None else args.seed
    out_dir = Path(args.out).expanduser() if args.out else (cfg.preview.output_dir.parent / "generated")

    # 何を作るか決める。--raw-prompts はテンプレートを通さない
    try:
        if args.raw_prompts:
            prompts = [(text, text) for text in read_prompt_file(Path(args.raw_prompts).expanduser())]
        elif args.prompts:
            subjects = read_prompt_file(Path(args.prompts).expanduser())
            prompts = [(subject, cfg.style.prompt(subject)) for subject in subjects]
        else:
            subjects = args.subjects if args.subjects else cfg.preview.subjects
            if not subjects:
                raise ValueError("被写体がありません。--prompts か --subjects を指定してください")
            prompts = [(subject, cfg.style.prompt(subject)) for subject in subjects]
    except (FileNotFoundError, ValueError) as exc:
        print(f"[gen] {exc}", file=sys.stderr)
        return 1

    if args.count < 1:
        print("[gen] --count は 1 以上にしてください", file=sys.stderr)
        return 1

    # 重い依存はここまで遅らせる（--help と引数チェックは torch 無しで通る）
    _, pipeline_module = load_torch_stack()

    device = pipeline_module.select_device(args.device)
    dtype = pipeline_module.select_dtype(device)
    model_path = Path(args.model).expanduser() if args.model else None

    print(f"[gen] device={device} dtype={dtype}")
    print(f"[gen] steps={steps} guidance={guidance} seed={base_seed} count={args.count}")
    print(f"[gen] 被写体 {len(prompts)} 件 -> 合計 {len(prompts) * args.count} 枚")

    if model_path is not None:
        if not model_path.is_dir():
            print(f"[gen] モデルがありません: {model_path}", file=sys.stderr)
            return 1
        print(f"[gen] モデル: {model_path}（焼き込み済み）")
        pipe = pipeline_module.load_base_pipeline(cfg, device, dtype, model_path=model_path)
    else:
        print(f"[gen] モデル: {cfg.model.base_model} ＋ LoRA（style={style_weight} lcm={lcm_weight}）")
        pipe = pipeline_module.load_base_pipeline(cfg, device, dtype)
        blocks = None
        if args.style_blocks:
            blocks = {}
            for item in args.style_blocks.split(","):
                key, _, value = item.partition("=")
                blocks[key.strip()] = float(value)
            print(f"[gen] スタイル LoRA のブロック別スケール: {blocks}")
        pipeline_module.attach_loras(
            pipe,
            cfg,
            style_lora_dir=args.lora_dir,
            style_weight=style_weight,
            lcm_weight=lcm_weight,
            style_blocks=blocks,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    sheet_images: list[Image.Image] = []
    sheet_labels: list[str] = []
    total_seconds = 0.0

    for subject, prompt in prompts:
        for variation in range(args.count):
            seed = base_seed + variation
            started = time.perf_counter()
            image = pipe(
                prompt,
                negative_prompt=(args.negative if args.negative is not None
                                 else cfg.style.negative_prompt) or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=pipeline_module.make_generator(seed),
            ).images[0]
            elapsed = time.perf_counter() - started
            total_seconds += elapsed

            alpha_ratio = None
            purity = None
            report = None
            if args.transparent or args.polish:
                image = remove_flat_background(image, tolerance=args.tolerance, feather=args.feather)
                alpha_ratio = transparency_ratio(image)
            if args.polish:
                # 整形は量子化を内側でやる。--snap-palette と併用しても壊れないが不要
                image, report = shapes.polish(image, size=args.polish_size,
                                              fit=not args.no_fit,
                                              background=args.background)
                purity = palette_purity(image)
            elif args.snap_palette:
                image = snap_to_palette(image)
                purity = palette_purity(image)

            name = f"{safe_name(subject)}_{variation + 1:02d}.png"
            image.save(out_dir / name)

            records.append(
                {
                    "file": name,
                    "subject": subject,
                    "prompt": prompt,
                    "seed": seed,
                    "steps": steps,
                    "guidance_scale": guidance,
                    "transparent": bool(args.transparent),
                    "transparency_ratio": alpha_ratio,
                    "shape": report,
                    "seconds": round(elapsed, 2),
                }
            )
            sheet_images.append(flatten_for_sheet(image))
            sheet_labels.append(f"{subject} #{variation + 1}")

            suffix = f" 透過={alpha_ratio:.0%}" if alpha_ratio is not None else ""
            if report is not None:
                suffix += f" 形={report['score']:.2f}"
                if report.get("background", "none") != "none":
                    suffix += f" 背景={report['background']}"
            print(f"[gen] {name}  {elapsed:.1f}s{suffix}")

    if not args.no_sheet:
        sheet = pipeline_module.make_contact_sheet(sheet_images, sheet_labels, columns=min(4, len(sheet_images)))
        sheet.save(out_dir / "contact_sheet.png")

    manifest = {
        "base_model": cfg.model.base_model,
        "lcm_lora": cfg.model.lcm_lora,
        "model_path": str(model_path) if model_path else None,
        "trigger_word": cfg.style.trigger_word,
        "prompt_template": cfg.style.prompt_template,
        "negative_prompt": cfg.style.negative_prompt,
        "style_weight": None if model_path else style_weight,
        "lcm_weight": None if model_path else lcm_weight,
        "steps": steps,
        "guidance_scale": guidance,
        "base_seed": base_seed,
        "count_per_subject": args.count,
        "device": str(device),
        "images": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    average = total_seconds / max(len(records), 1)
    print(f"[gen] {len(records)} 枚 -> {out_dir}  (1 枚あたり {average:.1f}s)")
    if not args.no_sheet:
        print(f"[gen] コンタクトシート: {out_dir / 'contact_sheet.png'}")
    print(f"[gen] 記録: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
