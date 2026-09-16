"""生成した画像をアイコンとして使える形に整える。

拡散モデルは必ず「塗りつぶした背景」を描く。アイコンとして使うなら背景は透過であってほしい。
フラット塗りのアイコンに限れば、背景色は外周から連続した一様な領域なので確実に抜ける。
"""

from __future__ import annotations

from collections import deque

from dataclasses import dataclass

import numpy as np
from PIL import Image


def estimate_background(array: np.ndarray) -> np.ndarray:
    """外周 2 ピクセルの中央値を背景色とみなす。平均だと角のノイズに引っ張られる。"""
    border = np.concatenate(
        [
            array[:2, :, :].reshape(-1, 3),
            array[-2:, :, :].reshape(-1, 3),
            array[:, :2, :].reshape(-1, 3),
            array[:, -2:, :].reshape(-1, 3),
        ]
    )
    return np.median(border, axis=0)


def remove_flat_background(
    image: Image.Image,
    *,
    tolerance: float = 0.06,
    feather: float = 0.12,
) -> Image.Image:
    """一様な背景を透過にする。

    「背景色に近い」だけで消すと、アイコンの内側にある背景と同じ色の領域（白抜きなど）まで
    消えてしまう。そこで**外周から連続してつながっている領域だけ**を背景と判定する。

    - `tolerance` … これ以下の色差は完全な背景（alpha 0）
    - `feather`   … tolerance からこの値までは半透明。輪郭のアンチエイリアスを拾う

    輪郭画素は `mix(背景色, 前景色, alpha)` になっているので、alpha で割り戻して
    前景色を復元する。これをしないと縁に背景色のふちが残る。
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = rgb.shape
    background = estimate_background(rgb)

    # 背景色からの距離（0〜1 程度に収まるよう √3 で正規化）
    distance = np.linalg.norm(rgb - background, axis=2) / np.sqrt(3.0)

    # 「背景かもしれない」画素。ここから外周続きの成分だけを拾う
    candidate = distance <= (tolerance + feather)

    reachable = _flood_from_border(candidate)
    reachable |= _enclosed_seams(candidate, reachable)

    # 背景領域内では距離に応じて alpha を立ち上げ、それ以外は不透明
    ramp = np.clip((distance - tolerance) / max(feather, 1e-6), 0.0, 1.0)
    alpha = np.where(reachable, ramp, 1.0).astype(np.float32)

    # 縁の色を復元する: px = (1-a) * bg + a * fg  ->  fg = (px - (1-a) * bg) / a
    safe_alpha = np.maximum(alpha, 1e-3)[:, :, None]
    foreground = (rgb - (1.0 - alpha)[:, :, None] * background) / safe_alpha
    foreground = np.clip(foreground, 0.0, 1.0)
    # 完全透過の画素は色を持たないので、にじみ防止に背景色を入れておく
    foreground = np.where((alpha > 1e-3)[:, :, None], foreground, background)

    rgba = np.dstack(
        [
            (foreground * 255.0).round().astype(np.uint8),
            (alpha * 255.0).round().astype(np.uint8),
        ]
    )
    return Image.fromarray(rgba, mode="RGBA")


def _enclosed_seams(candidate: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    """外周から届かない背景色の領域のうち、**細いスジだけ**を背景に含める。

    外周続きだけを抜く規則は、湯気やカレンダーの升目のような「意匠としての白抜き」を
    守るために要る。だがその規則は、図形どうしの境目に残った地の色も守ってしまう。
    実測では 64 枚中 63 枚に、高さ 6〜10px・幅 350〜670px という細長い白が残っており、
    単色背景を敷くと割れ目のように見えるうえ、白 x コーラルは 2.82:1 で
    WCAG の 3:1 を満たさない。

    意匠としての白抜きは面なので侵食しても残る。スジは残らない。この違いで分ける
    （実測: 升目 7.7% / 電球のハイライト 5.3% は残り、スジは全て消える）。
    """
    from .shapes import label_components, _morph

    enclosed = candidate & ~reachable
    if not enclosed.any():
        return np.zeros_like(candidate)
    height, width = candidate.shape
    # 半径は画素数ではなく画布に対して決める。512 と 1024 で判定が変わっては困る
    radius = max(1, round(5 * min(height, width) / 1024))
    min_area = max(50, (height * width) // 5000)

    labels, count = label_components(enclosed)
    seams = np.zeros_like(candidate)
    for index in range(1, count + 1):
        part = labels == index
        if part.sum() < min_area:
            continue
        if not _morph(part, radius, dilate=False).any():
            seams |= part
    return seams


def _flood_from_border(candidate: np.ndarray) -> np.ndarray:
    """外周の候補画素から 4 近傍で連結成分を広げる。

    これがあるおかげで、アイコンの内側にある「背景と同じ色の抜き」は残る。
    """
    height, width = candidate.shape
    reachable = np.zeros_like(candidate, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(y: int, x: int) -> None:
        if candidate[y, x] and not reachable[y, x]:
            reachable[y, x] = True
            queue.append((y, x))

    for x in range(width):
        seed(0, x)
        seed(height - 1, x)
    for y in range(height):
        seed(y, 0)
        seed(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            seed(y - 1, x)
        if y + 1 < height:
            seed(y + 1, x)
        if x > 0:
            seed(y, x - 1)
        if x + 1 < width:
            seed(y, x + 1)

    return reachable


def transparency_ratio(image: Image.Image) -> float:
    """完全透過の画素の割合。透過処理が効いたかの目安。"""
    if image.mode != "RGBA":
        return 0.0
    alpha = np.asarray(image.split()[-1], dtype=np.uint8)
    return float((alpha == 0).mean())


# 学習データのパレット（docs/dataset.md・SOURCE.md の生成プロンプトで固定している 2 色 + 白）
PALETTE = np.array([
    [0x1F, 0x2A, 0x44],   # deep navy
    [0xFF, 0x6B, 0x4A],   # coral
    [0xFF, 0xFF, 0xFF],   # white（ハイライトや抜き）
], dtype=np.float32)


@dataclass(frozen=True)
class Theme:
    """描くときの色。**ラベル番号と色を切り離して持つ。**

    整形も採点も円クロップも「どのパレット色か」の**番号**の地図の上で動いていて、
    色になるのは `render_smooth` の最後の一行だけ。だから色を差し替えても形は
    1 画素も変わらない。ここを型で分けておかないと、利用者が色を選べるようにしたとき
    採点の基準まで一緒に動いてしまう。

    量子化（`snap_to_palette`）は `PALETTE` のままにすること。学習データが
    ネイビー + コーラル + 白なので、「モデルがどの色を塗ろうとしたか」は元の 2 色で
    判定するのが正しい。差し替えるのは**描くとき**だけ。
    """

    ink: tuple[int, int, int]        # ラベル 0（学習データのネイビーに当たる）
    accent: tuple[int, int, int]     # ラベル 1（同コーラル）
    # ラベル 2（抜き）。**None なら描かず、地の色を透かせる。**
    # 抜きは本来「地が見えている部分」で、独立した色ではない（Material Symbols も同じ）。
    # 白で固定すると、白が輪郭に出た絵では白地が必ず 1.0:1 で失格になり、背景選びが
    # 逃げ道の #121212 に落ちる。単色では抜きが形の主役になるので、ここが効く
    counter: tuple[int, int, int] | None = None

    @property
    def colours(self) -> np.ndarray:
        """`PALETTE` と同じ形の配列。行の順番がラベル番号に対応する。

        `counter` が None のときも形だけは 3 行返す（描くかどうかは `draws` が決める）。
        """
        counter = self.counter if self.counter is not None else (255, 255, 255)
        return np.array([self.ink, self.accent, counter], dtype=np.float32)

    def draws(self, label: int) -> bool:
        """そのラベルを塗るか。抜きを地に任せる設定では 2 を塗らない。"""
        return not (label == COUNTER_LABEL and self.counter is None)

    @property
    def is_mono(self) -> bool:
        return self.ink == self.accent

    @classmethod
    def mono(cls, ink: tuple[int, int, int],
             counter: tuple[int, int, int] | None = None) -> "Theme":
        """単色。**2 色目を 1 色目と同じにするだけ**で、別の経路は要らない。

        2 色の境目は地の色の細い輪として残るので、レンズや鍵穴は読める。
        """
        return cls(ink=ink, accent=ink, counter=counter)

    def pair_contrast(self) -> float:
        """2 色どうしのコントラスト比。**2 色は必ず隣り合う**ので、ここが効く。"""
        from .icon_fit import contrast_ratio  # noqa: PLC0415  循環参照を避ける
        return contrast_ratio(np.array(self.ink, dtype=np.float32),
                              np.array(self.accent, dtype=np.float32))

    def validate(self) -> None:
        """使えない組み合わせを弾く。

        **地とのコントラストはここで見ない。** 「2 色の両方が 3:1 を満たす地」は
        現行のネイビー + コーラルでも存在せず、プロジェクトは既に
        「背景に接している色だけを輪郭の長さで数える」規則で解いている
        （docs/experiments.md Trial 14）。ここで見るのは**互いに隣り合う 2 色**だけ。
        """
        if self.is_mono:
            return
        ratio = self.pair_contrast()
        if ratio < MIN_PAIR_CONTRAST:
            raise ValueError(
                f"2 色のコントラストが {ratio:.2f}:1 で、WCAG 1.4.11 の "
                f"{MIN_PAIR_CONTRAST}:1 に足りません。2 色は必ず隣り合うので、"
                f"明るさの近い色どうしは見分けられません（例: 青 #2563EB と "
                f"水色 #3B82F6 は 1.41:1）。"
            )


# WCAG 2.1 の 1.4.11（非テキストコントラスト）が求める比
MIN_PAIR_CONTRAST = 3.0

COUNTER_LABEL = 2   # パレットの 3 行目（抜き）

# **抜きは既定で地に任せる。** 96 枚で測ると、白で塗る従来のやり方より
# WCAG 1.4.11 の 3:1 を満たす割合が 51% -> 82%（中央値 5.06 -> 14.26:1）に上がり、
# パレット外の逃げ道 #121212 に落ちる絵が 10 枚 -> 0 枚になった。形の点は 0.579 で不変
# （色はラベル番号と切り離してあるので、形には触れない）。
# 白で塗る従来の挙動が要るときは counter=(255, 255, 255) を渡す
DEFAULT_THEME = Theme(ink=(0x1F, 0x2A, 0x44), accent=(0xFF, 0x6B, 0x4A))


def snap_to_palette(image: "Image.Image", *, feather: float = 1.0) -> "Image.Image":
    """不透明画素をパレットの最近色に量子化する。

    **推論時の色のはみ出し（ピンク・黄・薄紫）を決定的に消すため。** 学習データは
    厳密に 2 色 + 白なので、中間色は「モデルの迷い」であって意図された色ではない。
    学習で直すのではなく後処理で直すのは、ここが決定的に解ける種類の問題だから
    （背景の透過と同じ判断。docs/experiments.md Trial 9 の教訓）。

    透過処理の**後**に呼ぶこと。背景が残ったまま量子化すると、背景の中間色が
    ネイビーと白のどちらへも割れてまだらになる。ここでは alpha > 0 の画素だけを触る。

    feather は量子化後の境界を 1px ぼかしてジャギーを抑える。0 で無効。
    """
    from PIL import ImageFilter  # noqa: PLC0415

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    # 各画素とパレット各色の距離 → 最近色
    distances = np.linalg.norm(rgb[..., None, :] - PALETTE[None, None, :, :], axis=-1)
    nearest = PALETTE[np.argmin(distances, axis=-1)]

    snapped = rgba.copy()
    mask = alpha > 0
    snapped[..., :3][mask] = nearest[mask]

    result = Image.fromarray(snapped.astype(np.uint8), "RGBA")
    if feather > 0:
        # 量子化で立ったエッジだけ僅かに緩める。全体をぼかすとフラットさが死ぬので
        # 半径は小さく固定
        blurred = result.filter(ImageFilter.GaussianBlur(feather * 0.6))
        result = Image.blend(result, blurred, 0.35)
    return result


def palette_purity(image: "Image.Image", tolerance: float = 24.0) -> float:
    """不透明画素のうちパレット近傍色の割合。色のはみ出しの計測用（0-1）。"""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    mask = rgba[..., 3] > 0
    if not mask.any():
        return 1.0
    rgb = rgba[..., :3][mask]
    distances = np.linalg.norm(rgb[:, None, :] - PALETTE[None, :, :], axis=-1).min(axis=1)
    return float((distances <= tolerance).mean())
