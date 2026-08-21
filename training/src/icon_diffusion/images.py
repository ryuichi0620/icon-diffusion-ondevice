"""Stage 1 の画像前処理と、スタイルの揃い具合を測る統計。

前処理の方針は docs/dataset.md 参照。要点だけ:
  - 透過はアルファ合成する（α を捨てると境界に黒縁が出る）
  - 正方形化はクロップではなくパディング（アイコンは端まで意味があることが多い）
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 統計のしきい値。厳密な根拠がある値ではなく「これを超えたら目視で確認せよ」の目安。
SATURATION_STD_WARN = 0.15
BACKGROUND_STD_WARN = 0.08
# 上位 8 色でこれだけ覆えていればフラット塗りとみなす。
# 色数そのものは輪郭のアンチエイリアスで形状ごとに何十倍も変わるため指標にならない。
FLATNESS_MIN_WARN = 0.80
FLATNESS_STD_WARN = 0.10


def parse_hex_color(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ValueError(f"背景色は #RRGGBB 形式で指定してください: {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def iter_source_images(root: Path) -> list[Path]:
    """入力ディレクトリを再帰的に走査して、対応拡張子のファイルを名前順に返す。"""
    if not root.is_dir():
        raise NotADirectoryError(f"入力ディレクトリが見つかりません: {root}")
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    ]
    return files


def flatten_onto_background(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """アルファを背景色に正しく合成して RGB にする。"""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, background + (255,))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def pad_to_square(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """長辺に合わせてパディングし、被写体を中央に置いた正方形にする。"""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), background)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def prepare_image(
    path: Path,
    resolution: int,
    background: tuple[int, int, int],
) -> Image.Image:
    with Image.open(path) as raw:
        raw.load()
        image = flatten_onto_background(raw, background)
    image = pad_to_square(image, background)
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.LANCZOS)
    return image


@dataclass(frozen=True)
class ImageStats:
    """1 枚あたりのスタイル指標。値そのものより、枚数間のばらつきを見るために使う。"""

    file_name: str
    mean_rgb: tuple[float, float, float]
    mean_saturation: float
    border_rgb: tuple[float, float, float]
    palette_size: int
    flatness: float
    ink_ratio: float


def measure(image: Image.Image, file_name: str) -> ImageStats:
    array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)

    channel_max = array.max(axis=2)
    channel_min = array.min(axis=2)
    saturation = np.where(channel_max > 1e-6, (channel_max - channel_min) / np.maximum(channel_max, 1e-6), 0.0)

    # 外周 4 ピクセルの平均を「背景色」とみなす
    border = np.concatenate(
        [
            array[:4, :, :].reshape(-1, 3),
            array[-4:, :, :].reshape(-1, 3),
            array[:, :4, :].reshape(-1, 3),
            array[:, -4:, :].reshape(-1, 3),
        ]
    )
    border_rgb = border.mean(axis=0)

    # 5bit に量子化して色の分布を見る
    quantized = (np.asarray(image, dtype=np.uint8) >> 3).astype(np.int32)
    codes = quantized[:, :, 0] * 1024 + quantized[:, :, 1] * 32 + quantized[:, :, 2]
    counts = np.bincount(codes.ravel())
    palette_size = int(np.count_nonzero(counts))
    # 上位 8 色が画素の何割を占めるか。フラット塗りなら 1 に近く、
    # グラデーションや写真では急激に下がる。色数と違って形状に引きずられない。
    flatness = float(np.sort(counts)[::-1][:8].sum() / codes.size)

    # 背景から離れたピクセルの割合（＝被写体の占有率）
    distance = np.abs(array - border_rgb).sum(axis=2)
    ink_ratio = float((distance > 0.15).mean())

    return ImageStats(
        file_name=file_name,
        mean_rgb=tuple(float(v) for v in array.reshape(-1, 3).mean(axis=0)),  # type: ignore[arg-type]
        mean_saturation=float(saturation.mean()),
        border_rgb=tuple(float(v) for v in border_rgb),  # type: ignore[arg-type]
        palette_size=palette_size,
        flatness=flatness,
        ink_ratio=ink_ratio,
    )


def summarize(stats: list[ImageStats]) -> dict:
    """ばらつきを集計し、目視確認を促す警告を返す。"""
    if not stats:
        return {"count": 0, "warnings": ["画像が 1 枚も処理されませんでした"]}

    saturations = np.array([s.mean_saturation for s in stats])
    borders = np.array([s.border_rgb for s in stats])
    palettes = np.array([s.palette_size for s in stats], dtype=np.float64)
    flatnesses = np.array([s.flatness for s in stats])
    inks = np.array([s.ink_ratio for s in stats])

    saturation_std = float(saturations.std())
    border_std = float(borders.std(axis=0).max())
    flatness_std = float(flatnesses.std())

    warnings: list[str] = []
    if saturation_std > SATURATION_STD_WARN:
        warnings.append(
            f"彩度のばらつきが大きい (std={saturation_std:.3f} > {SATURATION_STD_WARN}). "
            "配色パレットが揃っているか確認してください"
        )
    if border_std > BACKGROUND_STD_WARN:
        warnings.append(
            f"背景色が揃っていない (std={border_std:.3f} > {BACKGROUND_STD_WARN}). "
            "単色 or 透過のどちらかに統一してください"
        )
    rough = [s for s in stats if s.flatness < FLATNESS_MIN_WARN]
    if rough:
        names = ", ".join(s.file_name for s in rough[:5])
        more = f" ほか {len(rough) - 5} 枚" if len(rough) > 5 else ""
        warnings.append(
            f"フラット塗りでない画像が {len(rough)} 枚あります (flatness < {FLATNESS_MIN_WARN}): {names}{more}. "
            "グラデーション・影・写真が混ざっていないか確認してください"
        )
    elif flatness_std > FLATNESS_STD_WARN:
        warnings.append(
            f"塗りの均一さがばらついています (std={flatness_std:.3f} > {FLATNESS_STD_WARN}). "
            "ディテール量が揃っているか確認してください"
        )
    if len(stats) < 50:
        warnings.append(f"{len(stats)} 枚は少なめ。50〜200 枚が目安です (docs/dataset.md)")

    return {
        "count": len(stats),
        "mean_saturation": {"mean": float(saturations.mean()), "std": saturation_std},
        "border_rgb_std_max": border_std,
        "flatness": {
            "min": float(flatnesses.min()),
            "mean": float(flatnesses.mean()),
            "std": flatness_std,
        },
        "palette_size": {
            "min": int(palettes.min()),
            "max": int(palettes.max()),
            "median": float(np.median(palettes)),
        },
        "ink_ratio": {"mean": float(inks.mean()), "std": float(inks.std())},
        "warnings": warnings,
        "images": [asdict(s) for s in stats],
    }


_INDEX_TOKEN = re.compile(r"^\d+$")
_SPLIT = re.compile(r"[-_\s]+")


def subject_from_filename(path: Path) -> str:
    """`coffee-cup-01.png` -> `coffee cup`。通し番号らしきトークンは落とす。"""
    tokens = [t for t in _SPLIT.split(path.stem.strip()) if t]
    tokens = [t for t in tokens if not _INDEX_TOKEN.match(t)]
    return " ".join(tokens).lower().strip()
