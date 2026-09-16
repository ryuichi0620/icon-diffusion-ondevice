"""形を整えて滑らかに描き直す後処理と、「アイコンらしさ」の機械採点。

拡散モデルは 512px のラスタに絵を描く。そのため出力には三つの粗さが必ず乗る。

1. 中途半端な塗り — 中間色・グレー・かすれ。学習データは 2 色 + 白のフラット塗りだけ
   なので、中間色はモデルの迷いであって意図した色ではない
2. ゴミ — 本体から離れた小さな破片、輪郭のヒゲ、塗り残しの穴
3. ガタついた輪郭 — 512px の階段。アイコンとして見たときに「歪」に見える主因

いずれも学習で追い込むより後処理で決定的に消せる（背景透過・パレット量子化と同じ判断。
docs/experiments.md Trial 9 の教訓）。一方で**構造の破綻**（持ち手が 2 つ、ロケットが
矢印になる）は後処理では直せない。直せない分は `shape_report()` で採点して候補から
機械が落とす。整形と採点を同じモジュールに置いているのはこのため。

依存は numpy と Pillow だけに抑えてある。この処理は Swift へ移植して端末でも同じ結果を
出す前提なので、依存を足すほど移植が重くなる。
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from .postprocess import DEFAULT_THEME, PALETTE, Theme

# 整形の既定値。面積は canvas 比、半径は 512px 換算の画素数
DEFAULT_MIN_AREA = 0.0012      # これ未満の連結成分は捨てる（512px で約 315 画素 = 18px 角）
DEFAULT_MIN_HOLE = 0.0006      # これ未満の穴は塗り残しとみなして埋める
DEFAULT_OPEN = 1               # ヒゲ・1px の橋を切る収縮半径
DEFAULT_BLUR = 1.2             # 輪郭を均すぼかし半径（2.6 以上は線が太り細部が潰れる）


# --------------------------------------------------------------------------- ラベル化

def to_labels(image: Image.Image) -> np.ndarray:
    """RGBA を「どのパレット色か」の地図にする。透過は -1。

    色ごとに別々の形として扱わないと、ネイビーの上に乗ったコーラルの点までが
    「本体と地続きの一つの塊」に見えてしまい、ゴミ取りが効かない。
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    distances = np.linalg.norm(rgb[..., None, :] - PALETTE[None, None, :, :], axis=-1)
    labels = np.argmin(distances, axis=-1).astype(np.int8)
    labels[alpha < 128] = -1
    return labels


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4 近傍の連結成分ラベリング（0 = 背景、前景は 1..n）。

    画素ごとに探索せず行ランごとに union-find する。1 行あたりのランは数本なので
    512x512 でも一瞬で終わる。
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = mask[y]
        if not row.any():
            previous = []
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        current: list[tuple[int, int, int]] = []
        for start, end in zip(edges[0::2], edges[1::2]):
            overlapping = [label for (ps, pe, label) in previous if ps < end and start < pe]
            if overlapping:
                label = min(find(value) for value in overlapping)
                for value in overlapping:
                    union(label, value)
            else:
                parent.append(len(parent))
                label = len(parent) - 1
            labels[y, start:end] = label
            current.append((int(start), int(end), label))
        previous = current

    if len(parent) == 1:
        return labels, 0
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    labels = roots[labels]
    unique = np.unique(labels)
    unique = unique[unique != 0]
    if len(unique) == 0:
        return np.zeros_like(labels), 0
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, len(unique) + 1, dtype=np.int32)
    return remap[labels], len(unique)


# --------------------------------------------------------------------------- 整形

def _morph(mask: np.ndarray, radius: int, *, dilate: bool) -> np.ndarray:
    """3x3 の最大／最小フィルタを radius 回。PIL の C 実装に任せる。"""
    if radius <= 0:
        return mask
    image = Image.fromarray((mask * 255).astype(np.uint8), "L")
    kernel = ImageFilter.MaxFilter(3) if dilate else ImageFilter.MinFilter(3)
    for _ in range(radius):
        image = image.filter(kernel)
    return np.asarray(image) > 127


def drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """面積 min_area 未満の連結成分を落とす。"""
    labels, count = label_components(mask)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def fill_holes(mask: np.ndarray, min_hole: int) -> np.ndarray:
    """外周とつながっていない穴のうち、min_hole 未満のものを埋める。

    外周連結を見るのは背景透過と同じ理由で、**意図的な抜き**（虫眼鏡のレンズ、
    カップの内側など）を潰さないため。小さい穴だけ埋めるのは、それが塗り残しだから。
    """
    labels, count = label_components(~mask)
    if count == 0:
        return mask
    touching = set(np.unique(np.concatenate([
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    ])).tolist()) - {0}
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    fill = np.zeros(count + 1, dtype=bool)
    for label in range(1, count + 1):
        if label not in touching and areas[label] < min_hole:
            fill[label] = True
    return mask | fill[labels]


def mush_ratio(image: Image.Image, *, tolerance: float = 48.0, erode: int = 2) -> float:
    """塗りの内側がパレットから外れている割合 = 「中途半端な塗り」の量。

    **輪郭の 1〜2 画素は必ず中間色になる**（アンチエイリアス）。そこを数えると生の出力は
    どれも 7 割が「外れ」になってしまい、グレーの塗りやかすれと区別できない。そこで
    塗りを 2 画素だけ内側へ削ってから測る。許容も緩め（RGB 距離 48）に取り、色の
    僅かなブレではなく**別の色で塗ってある**ものだけを拾う。
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    ink = rgba[..., 3] >= 128
    interior = _morph(ink, erode, dilate=False)
    if not interior.any():
        return 0.0
    distances = np.linalg.norm(
        rgba[..., :3][interior][:, None, :] - PALETTE[None, :, :], axis=-1
    ).min(axis=1)
    return float((distances > tolerance).mean())


def clean_labels(
    labels: np.ndarray,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    min_hole: float = DEFAULT_MIN_HOLE,
    open_radius: int = DEFAULT_OPEN,
) -> np.ndarray:
    """色ごとにヒゲ取り・ゴミ取り・穴埋めをして、地図を作り直す。

    面積の大きい形から描き戻すのは、小さい形（ネイビーの上のコーラルの点など）が
    大きい形に飲み込まれないようにするため。どの色にも属さなくなった画素は近くの色で
    埋めずに透過へ落とす（無理に埋めると輪郭に偽の縁が出る）。
    """
    canvas = labels.shape[0] * labels.shape[1]
    min_area_px = max(1, int(min_area * canvas))
    min_hole_px = max(1, int(min_hole * canvas))

    pieces: list[tuple[int, int, np.ndarray]] = []
    for value in range(len(PALETTE)):
        mask = labels == value
        if not mask.any():
            continue
        # 開いてから閉じる: ヒゲと 1px の橋を切ってから、切り口と塗り残しを埋め直す
        mask = _morph(_morph(mask, open_radius, dilate=False), open_radius, dilate=True)
        mask = _morph(_morph(mask, open_radius, dilate=True), open_radius, dilate=False)
        mask = drop_small(mask, min_area_px)
        mask = fill_holes(mask, min_hole_px)
        if mask.any():
            pieces.append((int(mask.sum()), value, mask))

    cleaned = np.full_like(labels, -1)
    for _, value, mask in sorted(pieces, key=lambda item: -item[0]):
        cleaned[mask] = value
    return cleaned


def render_smooth(
    labels: np.ndarray,
    *,
    size: int | None = None,
    blur: float = DEFAULT_BLUR,
    edge: float = 1.2,
    fit: dict | None = None,
    theme: Theme | None = None,
) -> Image.Image:
    """地図をぼかした濃度場に変えてから描き直す。**これが「滑らかな線」の本体。**

    512px の階段は振幅がちょうど 1px なので、色ごとのマスクを 1px 強でぼかして 0.5 で
    切り直すと、階段だけが消えて角は角のまま残る（ガウスぼかしが動かす量は曲率に
    比例するため、直線と角はほとんど動かない）。ベクタに起こして描き直すのと結果は
    ほぼ同じで、実装が短く端末へ移植しても軽い。

    拡大は**二値化する前の濃度場に対して**行う。滑らかな場を拡大してから切るので、
    階段を拡大してからぼかすのと違って出力解像度に依存せず、512px の情報量のまま
    1024px の綺麗な輪郭が得られる。境界は `edge`（出力画素数）分だけ不透明度を
    寝かせてアンチエイリアスにする。

    `edge` の傾きは blur から決まる: ぼかした場の勾配は概ね 1/(2.5*blur)（元解像度の
    1 画素あたり）なので、出力 1 画素あたりでは拡大率で割った値になる。
    """
    height, width = labels.shape
    size = size or width
    scale = float(fit["scale"]) if fit else 1.0
    ratio = size * scale / width
    # 縮小するときは、縮めた後のぼけ幅が同じになるよう元解像度で強めにぼかす。
    # そうしないと濃度場を間引くときに階段が戻る
    source_blur = blur / min(scale, 1.0)
    # 濃度場 0.5 の等高線をまたぐ幅が edge 出力画素になるような傾き。
    # **1 を下回らせてはいけない。** 出力が元より小さいと境界は 1 画素未満で切り替わるため
    # 式の値が 1 未満になり、そのまま使うと不透明度がどこも 0 にも 255 にもならず
    # 全面が半透明の灰色になる（170px へ縮めて実際に出た）。傾き 1 は濃度場をそのまま
    # 不透明度にすることに等しく、面積比のアンチエイリアスとして正しい
    slope = max((2.5 * max(source_blur, 0.35) * ratio) / max(edge, 0.5), 1.0)

    # **ラベル番号は PALETTE のまま、塗る色だけ theme から取る。** 番号と色を
    # 分けてあるので、色を替えても形・採点・円クロップは 1 画素も変わらない
    active = theme or DEFAULT_THEME
    colours = active.colours

    pieces: list[tuple[int, int, np.ndarray]] = []
    for value in range(len(PALETTE)):
        mask = labels == value
        if not mask.any() or not active.draws(value):
            continue
        field = Image.fromarray((mask * 255).astype(np.uint8), "L")
        if blur > 0:
            field = field.filter(ImageFilter.GaussianBlur(source_blur))
        if fit is None:
            field = field.resize((size, size), Image.BILINEAR)
        else:
            # 視覚重心を画面の中心へ持ってきつつ、倍率を掛ける。
            # 出力 (x, y) -> 元画像 (a*x + c, e*y + f) の対応で書く
            centre_x, centre_y = fit["center"]
            a = width / (size * scale)
            e = height / (size * scale)
            field = field.transform(
                (size, size), Image.AFFINE,
                (a, 0, centre_x - a * size / 2, 0, e, centre_y - e * size / 2),
                resample=Image.BILINEAR,
            )
        alpha = np.clip((np.asarray(field, dtype=np.float32) / 255.0 - 0.5) * slope + 0.5, 0.0, 1.0)
        pieces.append((int(mask.sum()), value, alpha))

    # 大きい形から塗る。小さい形（ネイビーの上のコーラルの点など）を上に残すため
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    accumulated = np.zeros((size, size), dtype=np.float32)
    for _, value, alpha in sorted(pieces, key=lambda item: -item[0]):
        blend = alpha[..., None]
        rgb = colours[value][None, None, :] * blend + rgb * (1.0 - blend)
        accumulated = alpha + accumulated * (1.0 - alpha)

    rgba = np.concatenate([rgb, accumulated[..., None] * 255.0], axis=2)
    return Image.fromarray(np.clip(rgba, 0, 255).round().astype(np.uint8), "RGBA")


# --------------------------------------------------------------------------- 採点

# 採点の重み。**この値は tp-b5-seed42 の 48 枚を人が見て並べ替えた順序に合うよう
# 調整したもので、理論値ではない。** 別のモデル・別の被写体では要再調整。
IDEAL_COVERAGE = (0.16, 0.46)   # 学習データの被写体占有率（約 70% の枠に収まる塗り）
MAX_PARTS = 3                   # 意味のある塊の数。アイコンはこれ以下であってほしい
MAX_ROUGH = 2.3                 # 輪郭の複雑さ（円 = 1.27 前後、ギザギザほど大きい）


def shape_report(labels: np.ndarray, *, mush: float = 0.0) -> dict:
    """整形後の地図から「アイコンらしさ」を採点する。

    後処理で直せない**構造の破綻**を機械が落とすための指標。人が見て「歪」と言う絵は、
    たいてい次のどれかに当たる。

    - `coverage`  … 塗りが少なすぎ（かすれ）／多すぎ（塗り潰れ）
    - `stray`     … 本体から離れたところに絵が散っている（謎の突起・破片）
    - `parts`     … 塊が多すぎる。単一の物体には見えない
    - `rough`     … 輪郭が長すぎる。ギザギザ・ヒゲだらけ
    - `mush`      … 量子化前にパレットから外れていた画素の割合（中途半端な塗り）
    - `margin`    … 画面の端に当たっている。アイコンは四辺に余白が要る

    返す `score` は 1 が満点。閾値ではなく**候補の並べ替え**に使う前提（絶対値の意味は薄い）。
    """
    height, width = labels.shape
    canvas = height * width
    ink = labels >= 0
    area = int(ink.sum())
    if area == 0:
        return {"score": 0.0, "coverage": 0.0, "parts": 0, "stray": 1.0,
                "rough": 0.0, "mush": float(mush), "margin": 0.0, "symmetry": 0.0}

    coverage = area / canvas
    components, count = label_components(ink)
    areas = np.bincount(components.ravel(), minlength=count + 1)[1:]
    main = float(areas.max()) / area
    stray = 1.0 - main
    parts = int((areas >= 0.004 * canvas).sum())

    perimeter = int((ink[:, :-1] != ink[:, 1:]).sum() + (ink[:-1] != ink[1:]).sum())
    rough = perimeter / (2.0 * math.sqrt(math.pi * area))

    rows = np.flatnonzero(ink.any(axis=1))
    columns = np.flatnonzero(ink.any(axis=0))
    margin = min(rows[0], height - 1 - rows[-1], columns[0], width - 1 - columns[-1]) / width

    mirrored = ink[:, ::-1]
    symmetry = float((ink & mirrored).sum()) / float((ink | mirrored).sum())

    penalty = 0.0
    if coverage < IDEAL_COVERAGE[0]:
        penalty += 2.5 * (IDEAL_COVERAGE[0] - coverage)
    elif coverage > IDEAL_COVERAGE[1]:
        penalty += 2.5 * (coverage - IDEAL_COVERAGE[1])
    penalty += 1.2 * max(0.0, stray - 0.03)
    penalty += 0.08 * max(0, parts - MAX_PARTS)
    penalty += 0.35 * max(0.0, rough - MAX_ROUGH)
    # 色は量子化で決定的に直るので、形の欠点より軽く見る。強いブレは「モデルが
    # 迷った絵」の目印として効くので、外れが 35% を超えた分だけ引く
    penalty += 0.35 * max(0.0, mush - 0.35)
    penalty += 0.5 * max(0.0, (0.02 - margin) / 0.02)
    penalty += 0.15 * max(0.0, 0.55 - symmetry)

    return {
        "score": float(max(0.0, min(1.0, 1.0 - penalty))),
        "coverage": float(coverage),
        "parts": parts,
        "stray": float(stray),
        "rough": float(rough),
        "mush": float(mush),
        "margin": float(margin),
        "symmetry": symmetry,
    }


def polish(
    image: Image.Image,
    *,
    size: int | None = None,
    min_area: float = DEFAULT_MIN_AREA,
    min_hole: float = DEFAULT_MIN_HOLE,
    open_radius: int = DEFAULT_OPEN,
    blur: float = DEFAULT_BLUR,
    edge: float = 1.2,
    fit: bool = True,
    background: str = "none",
    theme: Theme | None = None,
) -> tuple[Image.Image, dict]:
    """透過済みの画像を「整えて滑らかに描き直し」、同時に採点する。

    `remove_flat_background()` の**後**に呼ぶこと。量子化はこの中でやるので
    `snap_to_palette()` を別に呼ぶ必要はない（二重に掛けても壊れはしない）。

    `fit` は円クロップへの対応（視覚重心で中央に寄せ、光学的な大きさを揃える）。
    `background` は敷く色（`auto` で最もコントラストの立つ色を選ぶ）。どちらも
    icon_fit.py 参照。**採点は変形の前に行う** — 変形は「塗りが薄い」「中心から
    ずれている」といった欠点を見かけ上つぶしてしまうので、それでは絵の善し悪しを
    測れなくなる。
    """
    from . import icon_fit  # noqa: PLC0415  循環参照を避ける

    mush = mush_ratio(image)
    labels = to_labels(image)
    cleaned = clean_labels(labels, min_area=min_area, min_hole=min_hole,
                           open_radius=open_radius)
    report = shape_report(cleaned, mush=mush)
    report.update(icon_fit.legibility_report(cleaned, theme=theme))

    drawn = cleaned
    plan = None
    if fit:
        # 背景の切れ端を落としてから測って描く。採点は落とす前の地図で済ませてある
        drawn = icon_fit.drop_scraps(cleaned)
        plan = icon_fit.plan_fit(drawn)
        report["outside_before"] = plan["outside_before"]
        report["fit_scale"] = plan["scale"]
        report["scraps"] = float((cleaned >= 0).sum() - (drawn >= 0).sum()) / \
            max(float((cleaned >= 0).sum()), 1.0)
    rendered = render_smooth(drawn, size=size, blur=blur, edge=edge, fit=plan,
                             theme=theme)

    if background == "auto":
        background = icon_fit.choose_background(drawn, theme)
    report["background"] = background
    if background != "none":
        # 輪郭の色と背景色のコントラスト比。WCAG 1.4.11 は図形に 3:1 を求める
        report["background_contrast"] = icon_fit.background_contrast(drawn, background, theme)
    return icon_fit.compose_background(rendered, background, theme), report
