"""SNS アイコンとして成立させるための整え方と検査。

`shapes.py` が「形の粗さ」を扱うのに対し、こちらは**使われ方**を扱う。
SNS のプロフィール画像はほぼ例外なく**円形に切り抜かれ**、タイムラインでは
48px 前後まで縮む。そこで効く基準は 3 つで、どれも機械で測れる。

1. **円クロップのセーフエリア** — 正方形の四隅は必ず切れる。実測では b5 の出力の
   21%（10/48）が内接円からはみ出していた（最大 28.6%）
2. **隣接色とのコントラスト** — WCAG 2.1 の 1.4.11 は図形に 3:1 を求める。
   **このパレットではコーラル×白が 2.82:1 で未達**（ネイビー×白 14.26、
   ネイビー×コーラル 5.06 は問題ない）。主役がコーラルの絵は白背景で沈む
3. **小サイズでの判別** — 48px に落として線が消えないか

1 と 2 は後処理で決定的に直せる。1 は視覚重心に寄せて円に収める変形、
2 は背景色を敷くこと。3 は直せないので採点して候補から落とす。
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .postprocess import PALETTE

# 内接円の内側にどれだけ余白を残すか。資料の「四隅は使わない・捨て代を設ける」に対応
SAFE_RATIO = 0.92
# 何割の塗りを安全領域に収めるか。**100% にしてはいけない。**
# 細い先端が 1 本はみ出しているだけで全体が 6 割に縮み、余白だらけになる（実測）。
# 資料の安全領域も「重要な要素を中央寄りに」であって、全画素を円内に入れる話ではない
# （Threads の例: 320px の画布に対し中央 280px。この正方形の四隅は内接円の外にある）
SAFE_QUANTILE = 0.98
# 塗りが画面を埋め尽くすと圧迫感が出る。この割合を超えたら縮める
MAX_COVERAGE = 0.50
# 拡大しすぎると 512px の情報量を超えて眠くなる。縮小しすぎると小さくなりすぎる
SCALE_LIMITS = (0.55, 1.60)


def relative_luminance(rgb) -> float:
    """WCAG 2.1 の相対輝度。"""
    channels = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(channels <= 0.04045, channels / 12.92,
                      ((channels + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def contrast_ratio(a, b) -> float:
    """WCAG 2.1 のコントラスト比。1.4.11 は図形に 3:1 以上を求める。"""
    la, lb = relative_luminance(a), relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def visual_center(ink: np.ndarray) -> tuple[float, float]:
    """塗りの重心。**外接矩形の中心ではない。**

    資料の「オプティカル調整」に対応する。矩形の中心に合わせると、片側に
    重い絵（取っ手のあるカップなど）が視覚的に寄って見える。面積の重心なら
    見た目の釣り合いが取れる。
    """
    ys, xs = np.nonzero(ink)
    return float(xs.mean()), float(ys.mean())


# 画面の端に触れている塊のうち、これ未満の大きさなら背景の切れ端とみなす
SCRAP_SHARE = 0.15


def border_scraps(ink: np.ndarray, max_share: float = SCRAP_SHARE,
                  min_inside: float = 0.5) -> np.ndarray:
    """背景の切れ端とみなせる塊。**画面の端に触れていることが前提。**

    学習データは「四辺に等しい余白」を指定して作っているので、**端に接する塊は
    被写体ではないことが多い**（モデルが描いた枠や帯が画面で切れた残り）。実際に出るのは
    隅の三角、斜めの帯、上辺の棒など。円クロップで切れるだけでなく、大きさを揃える計算も
    これに引きずられる（隅のゴミ 1 つで全体が 6 割に縮んだ）。

    端に接していても、次のどちらでもなければ残す。被写体そのものが端に接することも
    あるため（横長のカメラなど）、それを消しては本末転倒になる。

    - 小さい（塗り全体の `max_share` 未満）
    - **内接円の中にほとんど無い**（`min_inside` 未満）。どうせ円クロップで消えるので、
      残しても画面を圧迫するだけで見えない
    """
    from .shapes import label_components  # noqa: PLC0415  循環参照を避ける

    components, count = label_components(ink)
    if count <= 1:
        return np.zeros_like(ink)
    areas = np.bincount(components.ravel(), minlength=count + 1)
    total = float(ink.sum())

    height, width = ink.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    inside_circle = (np.sqrt((grid_y - (height - 1) / 2) ** 2
                             + (grid_x - (width - 1) / 2) ** 2) <= width / 2)
    inside_areas = np.bincount(components[inside_circle].ravel(), minlength=count + 1)

    touching = set(np.unique(np.concatenate([
        components[0], components[-1], components[:, 0], components[:, -1],
    ])).tolist()) - {0}

    scrap = np.zeros(count + 1, dtype=bool)
    for label in touching:
        small = areas[label] / total < max_share
        mostly_cropped = inside_areas[label] / max(areas[label], 1) < min_inside
        if small or mostly_cropped:
            scrap[label] = True
    return scrap[components]


def drop_scraps(labels: np.ndarray) -> np.ndarray:
    """背景の切れ端を透過に落とした地図を返す。

    **採点は落とす前に行うこと。** 切れ端があること自体が絵の欠点なので、
    消してから採点すると下手な絵が上手に見えてしまう。
    """
    scrap = border_scraps(labels >= 0)
    if not scrap.any():
        return labels
    cleaned = labels.copy()
    cleaned[scrap] = -1
    return cleaned


def main_body(ink: np.ndarray) -> np.ndarray:
    """大きさと位置を測る対象。背景の切れ端を除いた塗り。

    **面積最大の連結成分ではない。** ロケットの炎や封筒のフラップのように、主役が
    複数の塊に分かれている絵は珍しくなく、最大の塊だけで測ると全体が拡大されすぎる
    （実測で 1.6 倍に振り切れて画面から溢れた）。
    """
    body = ink & ~border_scraps(ink)
    return body if body.any() else ink


def plan_fit(
    labels: np.ndarray,
    *,
    safe_ratio: float = SAFE_RATIO,
    safe_quantile: float = SAFE_QUANTILE,
    max_coverage: float = MAX_COVERAGE,
    limits: tuple[float, float] = SCALE_LIMITS,
) -> dict:
    """円クロップの安全領域に収まるよう、視覚重心へ寄せて大きさを揃える変形を決める。

    基準は**塗りの 98% が入る半径**。全画素を内接円に入れようとすると、細い先端が
    1 本はみ出しているだけで全体が 6 割に縮んで余白だらけになる（実測で確認）。
    先端が僅かに切れるのは許容し、代わりに主役を大きく保つ。

    大きさを外接半径ではなくこの分位半径で揃えるのは、資料の「オプティカル調整」に
    対応する。絵によって突起の有無が違うので、外接半径で揃えると見た目の大きさが揃わない。
    """
    height, width = labels.shape
    ink = labels >= 0
    if not ink.any():
        return {"scale": 1.0, "center": ((width - 1) / 2, (height - 1) / 2),
                "outside_before": 0.0, "limited_by": "empty"}

    # **主役（面積が最大の連結成分）だけで測る。** 本体から離れた破片まで含めると、
    # 隅に残った小さなゴミ 1 つに引きずられて全体が 6 割に縮む（実測で確認）。
    # 資料の「主役は 1 つ」に合わせ、破片は円クロップで切れてよいものとして扱う
    body = main_body(ink)
    centre_x, centre_y = visual_center(body)
    ys, xs = np.nonzero(body)
    distances = np.sqrt((xs - centre_x) ** 2 + (ys - centre_y) ** 2)
    reach = float(np.quantile(distances, safe_quantile))
    safe_radius = (width / 2.0) * safe_ratio

    scale = safe_radius / max(reach, 1e-6)
    limited_by = "円の安全領域"

    area = float(body.sum())
    coverage_after = area * scale * scale / (width * height)
    if coverage_after > max_coverage:
        scale *= math.sqrt(max_coverage / coverage_after)
        limited_by = "塗りすぎの上限"

    clamped = min(max(scale, limits[0]), limits[1])
    if clamped != scale:
        limited_by += "（倍率の上限で頭打ち）"

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    from_centre = np.sqrt((grid_y - (height - 1) / 2) ** 2 + (grid_x - (width - 1) / 2) ** 2)
    outside = float((ink & (from_centre > width / 2)).sum()) / float(ink.sum())

    return {"scale": float(clamped), "center": (centre_x, centre_y),
            "outside_before": outside, "limited_by": limited_by}


def legibility_report(labels: np.ndarray, *, small: int = 48) -> dict:
    """小サイズでの成立性を測る。資料の「検証」チェックリストに対応。

    - `survive`  … 48px に落としたとき残る塗りの割合。細い線はここで消える
    - `thin`     … 512px で 5 画素削ると消える割合（48px の 0.5 画素に相当）
    - `on_white` / `on_dark` … 主役の色と背景のコントラスト比
    - `grayscale` … 色を抜いたときの主役と背景の明度差（資料のグレースケール検証）
    """
    ink = labels >= 0
    if not ink.any():
        return {"survive": 0.0, "thin": 1.0, "on_white": 1.0, "on_dark": 1.0,
                "grayscale": 0.0, "dominant": -1}

    from .shapes import _morph  # noqa: PLC0415  循環参照を避ける

    shrunk = np.asarray(
        Image.fromarray((ink * 255).astype(np.uint8), "L").resize((small, small), Image.BOX),
        dtype=np.float32,
    ) / 255.0
    survive = float((shrunk >= 0.5).sum()) / max(float(shrunk.sum()), 1e-6)

    eroded = _morph(ink, 5, dilate=False)
    thin = 1.0 - float(eroded.sum()) / float(ink.sum())

    areas = [int((labels == value).sum()) for value in range(len(PALETTE))]
    dominant = int(np.argmax(areas))
    colour = PALETTE[dominant]
    grayscale = abs(relative_luminance(colour) - 1.0)

    return {
        "survive": float(min(survive, 1.0)),
        "thin": float(thin),
        "on_white": contrast_ratio(colour, [255, 255, 255]),
        "on_dark": contrast_ratio(colour, [0x12, 0x12, 0x12]),
        "grayscale": float(grayscale),
        "dominant": dominant,
    }


# --------------------------------------------------------------------------- 背景

BACKGROUNDS = {
    "none": None,
    "white": np.array([255, 255, 255], dtype=np.float32),
    "navy": PALETTE[0],
    "coral": PALETTE[1],
    # 資料が避けろと言う純黒ではなく #121212。暗所でのハレーションを避けるため
    "dark": np.array([0x12, 0x12, 0x12], dtype=np.float32),
}


def silhouette_colours(labels: np.ndarray, min_share: float = 0.10) -> list[int]:
    """背景に接している色。**面積ではなく輪郭の長さで数える。**

    WCAG 1.4.11 が求めるのは**隣接色**とのコントラストなので、背景に触れていない
    内側の色（カップの中の白など）まで数える必要はない。面積で数えると、どの背景色でも
    3:1 を満たせない絵が大半になってしまう（2 色 + 白が全部載っている絵では、白地だと
    白が、暗地だとネイビーが、コーラル地だとコーラルが必ず沈む）。
    """
    background = labels < 0
    if not background.any():
        return [value for value in range(len(PALETTE)) if (labels == value).any()]

    neighbours = np.zeros_like(background)
    neighbours[1:, :] |= background[:-1, :]
    neighbours[:-1, :] |= background[1:, :]
    neighbours[:, 1:] |= background[:, :-1]
    neighbours[:, :-1] |= background[:, 1:]
    edge = neighbours & ~background

    total = float(edge.sum())
    if total == 0:
        return [value for value in range(len(PALETTE)) if (labels == value).any()]
    return [value for value in range(len(PALETTE))
            if float((edge & (labels == value)).sum()) / total >= min_share]


def background_contrast(labels: np.ndarray, style: str) -> float:
    """その背景を敷いたときの、輪郭に出ている色の最小コントラスト比。"""
    colour = BACKGROUNDS.get(style)
    if colour is None:
        return float("inf")
    colours = silhouette_colours(labels)
    if not colours:
        return float("inf")
    return min(contrast_ratio(PALETTE[value], colour) for value in colours)


def choose_background(labels: np.ndarray) -> str:
    """輪郭の色に対して 3:1 を満たす背景のうち、**パレット内で最初に見つかったもの**。

    **コーラルは白地で 2.82:1 しかなく、WCAG 1.4.11 の 3:1 を満たさない。**
    シルエットがコーラルの絵に白を敷くと、タイムラインで沈む。

    コントラスト最大で選ばない理由は、そうすると 48 枚中 28 枚が #121212 になったから。
    数字は良くても**パレット外の色**で、アイコンの一貫性（資料の「トンマナ統一」）が崩れる。
    ネイビーを敷けばコーラルに 5.06:1、白に 14.26:1 で基準を満たせる。
    #121212 はパレット内のどれでも足りないときの逃げ道に留める。
    """
    ink = labels >= 0
    if not ink.any():
        return "white"
    for name in ("white", "navy", "coral"):
        if background_contrast(labels, name) >= 3.0:
            return name
    scores = {name: background_contrast(labels, name)
              for name in ("white", "navy", "coral", "dark")}
    return max(scores, key=scores.get)


def compose_background(image: Image.Image, style: str) -> Image.Image:
    """単色背景を敷く。**透過のままにしないのは、タイムラインの地の色に左右されないため**
    （資料の「背景色は単色を敷くと安定」。Apple もアプリアイコンに透過を認めていない）。

    SNS 側が円に切り抜くので、背景は正方形いっぱいに塗る。円だけ塗ると
    切り抜き位置のずれで角に透明が残る。
    """
    if style == "none":
        return image
    colour = BACKGROUNDS[style]
    background = Image.new("RGBA", image.size,
                           tuple(int(v) for v in colour) + (255,))
    background.alpha_composite(image.convert("RGBA"))
    return background
