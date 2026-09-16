import Foundation

/// 形を整えて滑らかに描き直す後処理と、「アイコンらしさ」の採点。
/// **training/src/icon_diffusion/shapes.py の移植で、あちらが参照実装。**
///
/// 端末でも同じ整形と同じ採点を掛けることで、Mac で測った数字（形の合格率）がそのまま
/// 端末の候補選びに繋がる。とくに採点は、後処理では直せない**構造の破綻**
/// （持ち手が 2 つ、ロケットが矢印になる）を機械が落とすための唯一の手段なので、
/// 参照実装と同じ式であることが重要。
///
/// 依存は Foundation だけに抑えてある（Accelerate も CoreImage も使わない）。
/// 512x512 で数十ミリ秒なので、1 枚 1.2 秒の生成に対して無視できる。
public enum IconShapes {
    /// 学習データのパレット（SOURCE.md の生成プロンプトで固定している 2 色 + 白）。
    /// **これは「どのパレット色か」を決める物差しであって、描くときの色ではない。**
    /// 描く色は `Theme` が持つ。参照実装は postprocess.py の PALETTE。
    static let palette: [(r: Double, g: Double, b: Double)] = [
        (0x1F, 0x2A, 0x44),   // deep navy
        (0xFF, 0x6B, 0x4A),   // coral
        (0xFF, 0xFF, 0xFF),   // white
    ]

    /// 抜き（パレットの 3 行目）のラベル番号
    static let counterLabel = 2

    /// 描くときの色。**ラベル番号と色を切り離して持つ。**
    ///
    /// 整形も採点も円クロップも番号の地図の上で動いていて、色になるのは `render` の
    /// 最後の一行だけ。だから色を差し替えても形は 1 画素も変わらない。
    /// 参照実装は postprocess.py の `Theme`。
    public struct Theme: Sendable, Equatable {
        public typealias RGB = (r: Double, g: Double, b: Double)

        /// ラベル 0（学習データのネイビーに当たる）
        public let ink: RGB
        /// ラベル 1（同コーラル）。`ink` と同じにすると単色になる
        public let accent: RGB
        /// ラベル 2（抜き）。**nil なら描かず、地の色を透かせる。**
        ///
        /// 抜きは本来「地が見えている部分」で、独立した色ではない。白で固定すると
        /// 白が輪郭に出た絵では白地が必ず 1.0:1 で失格になり、背景選びが逃げ道の
        /// #121212 に落ちる。96 枚の実測で 3:1 を満たす割合が 51% -> 82% に上がった
        public let counter: RGB?

        public init(ink: RGB, accent: RGB, counter: RGB? = nil) {
            self.ink = ink
            self.accent = accent
            self.counter = counter
        }

        /// 単色。**2 色目を 1 色目と同じにするだけ**で、別の経路は要らない
        public static func mono(_ ink: RGB, counter: RGB? = nil) -> Theme {
            Theme(ink: ink, accent: ink, counter: counter)
        }

        public static let `default` = Theme(ink: (0x1F, 0x2A, 0x44), accent: (0xFF, 0x6B, 0x4A))

        public var isMono: Bool { ink == accent }

        /// ラベル番号に対応する色。`counter` が nil のときも形だけは 3 つ返す
        var colours: [RGB] { [ink, accent, counter ?? (0xFF, 0xFF, 0xFF)] }

        /// そのラベルを塗るか。抜きを地に任せる設定では 2 を塗らない
        func draws(_ label: Int) -> Bool {
            !(label == IconShapes.counterLabel && counter == nil)
        }

        /// 2 色どうしのコントラスト比。**2 色は必ず隣り合う**ので、ここが効く
        public var pairContrast: Double {
            IconFit.contrastRatio(ink, accent)
        }

        /// 2 色の明暗の開き。`(相対輝度 + 0.05)` の比で見る
        public var luminanceSpread: Double {
            let a = IconFit.relativeLuminance(ink) + 0.05
            let b = IconFit.relativeLuminance(accent) + 0.05
            return max(a, b) / min(a, b)
        }

        /// **2 色とも同じ地に対して 3:1 を満たせるか。**
        ///
        /// 暗い側に `(L_bg+.05)/(L_a+.05) >= 3`、明るい側に `(L_b+.05)/(L_bg+.05) >= 3` が
        /// 同時に要るので、2 色は `3 x 3 = 9` 倍以上離れていなければならない。
        /// **既定のネイビー + コーラルは 5.1 倍しかなく、両色が地に接する絵では
        /// どの地を敷いても 3:1 に届かない。** 観測され続けた 2.82:1 の正体はこれで、
        /// 不具合ではなくパレットの選び方の帰結。単色ならこの制約は消える。
        public var backgroundAchievable: Bool {
            isMono || luminanceSpread >= IconFit.minContrast * IconFit.minContrast
        }

        /// 使える組み合わせか。地とのコントラストはここで見ない
        /// （「2 色の両方が 3:1 を満たす地」は既定の 2 色でも存在せず、
        /// 「背景に接している色だけを輪郭の長さで数える」規則で既に解いてある）
        public var isUsable: Bool { isMono || pairContrast >= IconFit.minContrast }

        // タプルを持つ構造体は Equatable を自動合成できないので手で書く
        public static func == (a: Theme, b: Theme) -> Bool {
            a.ink == b.ink && a.accent == b.accent
                && a.counter?.r == b.counter?.r
                && a.counter?.g == b.counter?.g
                && a.counter?.b == b.counter?.b
        }
    }

    // 参照実装の既定値。面積は canvas 比、半径は 512px 換算の画素数
    public static let defaultMinArea = 0.0012
    public static let defaultMinHole = 0.0006
    public static let defaultOpen = 1
    public static let defaultBlur = 1.2

    public struct Report {
        public var score: Double
        public var coverage: Double
        public var parts: Int
        public var stray: Double
        public var rough: Double
        public var mush: Double
        public var margin: Double
        public var symmetry: Double
        /// 整形前に内接円からはみ出していた塗りの割合（円クロップで切れていた量）
        public var outsideBefore: Double = 0
        /// 敷いた背景の名前。"none" なら透過のまま
        public var background: String = "none"
        /// 輪郭の色と背景色の最小コントラスト比。WCAG 1.4.11 は 3:1 以上を求める
        public var backgroundContrast: Double = .infinity
    }

    public struct Polished {
        public var image: GeneratedImage
        public var report: Report
    }

    // MARK: - 入口

    /// 透過済みの画像を整えて描き直し、同時に採点する。
    /// `IconPostprocess.process()` の**後**に呼ぶこと（背景が残っていると全面が絵になる）。
    /// `fit` は円クロップへの整形（切れ端を落とし、視覚重心へ寄せ、大きさを揃える）。
    /// `background` は敷く色（`auto` で 3:1 を満たすパレット色を選ぶ）。
    /// **採点は整形の前に行う** — 整形は「塗りが薄い」「中心からずれている」といった欠点を
    /// 見かけ上つぶすので、後で測ると絵の善し悪しが分からなくなる。
    public static func polish(_ image: GeneratedImage,
                       size: Int? = nil,
                       blur: Double = defaultBlur,
                       edge: Double = 1.2,
                       fit: Bool = true,
                       background: String = "none",
                       theme: Theme = .default) -> Polished {
        let width = image.width, height = image.height
        let labels = labelMap(image)
        let mush = mushRatio(image)
        let cleaned = clean(labels, width: width, height: height)
        var report = self.report(cleaned, width: width, height: height, mush: mush)

        var drawn = cleaned
        var plan: IconFit.Plan?
        if fit {
            drawn = IconFit.dropScraps(cleaned, width: width, height: height)
            let made = IconFit.plan(drawn, width: width, height: height)
            report.outsideBefore = made.outsideBefore
            plan = made
        }

        var rendered = render(drawn, width: width, height: height,
                              size: size ?? width, blur: blur, edge: edge, plan: plan,
                              theme: theme)

        var chosen = background
        if background == "auto" {
            chosen = IconFit.chooseBackground(drawn, width: width, height: height, theme: theme)
        }
        report.background = chosen
        if chosen != "none" {
            report.backgroundContrast = IconFit.backgroundContrast(
                drawn, width: width, height: height, background: chosen, theme: theme)
            rendered = IconFit.compose(rendered, background: chosen, theme: theme)
        }
        return Polished(image: rendered, report: report)
    }

    // MARK: - ラベル化

    /// RGBA を「どのパレット色か」の地図にする。透過は -1。
    /// 色ごとに別の形として扱わないと、ネイビーの上のコーラルの点が本体と地続きに見えて
    /// ゴミ取りが効かない。
    static func labelMap(_ image: GeneratedImage) -> [Int8] {
        let rgba = image.rgba
        var labels = [Int8](repeating: -1, count: image.width * image.height)
        for index in 0..<(image.width * image.height) {
            let offset = index * 4
            if rgba[offset + 3] < 128 { continue }
            labels[index] = Int8(nearestPalette(r: Double(rgba[offset]),
                                                g: Double(rgba[offset + 1]),
                                                b: Double(rgba[offset + 2])).index)
        }
        return labels
    }

    private static func nearestPalette(r: Double, g: Double, b: Double)
        -> (index: Int, distance: Double) {
        var best = 0
        var bestDistance = Double.greatestFiniteMagnitude
        for (index, colour) in palette.enumerated() {
            let dr = r - colour.r, dg = g - colour.g, db = b - colour.b
            let distance = dr * dr + dg * dg + db * db
            if distance < bestDistance { bestDistance = distance; best = index }
        }
        return (best, bestDistance.squareRoot())
    }

    /// 塗りの内側がパレットから外れている割合 = 「中途半端な塗り」の量。
    /// **輪郭の 1〜2 画素は必ず中間色になる**ので、2 画素だけ内側へ削ってから測る。
    /// そうしないと生の出力はどれも 7 割が「外れ」になり、グレーの塗りと区別できない。
    static func mushRatio(_ image: GeneratedImage,
                          tolerance: Double = 48.0, erode: Int = 2) -> Double {
        let width = image.width, height = image.height
        let rgba = image.rgba
        var ink = [Bool](repeating: false, count: width * height)
        for index in 0..<(width * height) { ink[index] = rgba[index * 4 + 3] >= 128 }
        let interior = morph(ink, width: width, height: height, radius: erode, dilate: false)

        var total = 0, outside = 0
        for index in 0..<(width * height) where interior[index] {
            total += 1
            let offset = index * 4
            let distance = nearestPalette(r: Double(rgba[offset]),
                                          g: Double(rgba[offset + 1]),
                                          b: Double(rgba[offset + 2])).distance
            if distance > tolerance { outside += 1 }
        }
        return total == 0 ? 0 : Double(outside) / Double(total)
    }

    // MARK: - 整形

    /// 色ごとにヒゲ取り・ゴミ取り・穴埋めをして、地図を作り直す。
    /// 面積の大きい形から描き戻すのは、小さい形が大きい形に飲み込まれないため。
    static func clean(_ labels: [Int8], width: Int, height: Int,
                      minArea: Double = defaultMinArea,
                      minHole: Double = defaultMinHole,
                      openRadius: Int = defaultOpen) -> [Int8] {
        let canvas = width * height
        let minAreaPixels = max(1, Int(minArea * Double(canvas)))
        let minHolePixels = max(1, Int(minHole * Double(canvas)))

        var pieces: [(area: Int, value: Int8, mask: [Bool])] = []
        for value in 0..<palette.count {
            var mask = [Bool](repeating: false, count: canvas)
            var any = false
            for index in 0..<canvas where labels[index] == Int8(value) {
                mask[index] = true
                any = true
            }
            if !any { continue }
            // 開いてから閉じる: ヒゲと 1px の橋を切ってから、切り口と塗り残しを埋め直す
            mask = morph(morph(mask, width: width, height: height, radius: openRadius, dilate: false),
                         width: width, height: height, radius: openRadius, dilate: true)
            mask = morph(morph(mask, width: width, height: height, radius: openRadius, dilate: true),
                         width: width, height: height, radius: openRadius, dilate: false)
            mask = dropSmall(mask, width: width, height: height, minArea: minAreaPixels)
            mask = fillHoles(mask, width: width, height: height, minHole: minHolePixels)
            let area = mask.reduce(0) { $0 + ($1 ? 1 : 0) }
            if area > 0 { pieces.append((area, Int8(value), mask)) }
        }

        var cleaned = [Int8](repeating: -1, count: canvas)
        for piece in pieces.sorted(by: { $0.area > $1.area }) {
            for index in 0..<canvas where piece.mask[index] { cleaned[index] = piece.value }
        }
        return cleaned
    }

    /// 3x3 の最大／最小フィルタを radius 回。分離可能なので横→縦の 2 パスで済ませる。
    static func morph(_ mask: [Bool], width: Int, height: Int,
                      radius: Int, dilate: Bool) -> [Bool] {
        guard radius > 0 else { return mask }
        var current = mask
        for _ in 0..<radius {
            var horizontal = [Bool](repeating: !dilate, count: width * height)
            for y in 0..<height {
                let row = y * width
                for x in 0..<width {
                    var value = current[row + x]
                    if x > 0 { value = dilate ? (value || current[row + x - 1])
                                              : (value && current[row + x - 1]) }
                    if x < width - 1 { value = dilate ? (value || current[row + x + 1])
                                                      : (value && current[row + x + 1]) }
                    horizontal[row + x] = value
                }
            }
            var vertical = [Bool](repeating: !dilate, count: width * height)
            for y in 0..<height {
                let row = y * width
                for x in 0..<width {
                    var value = horizontal[row + x]
                    if y > 0 { value = dilate ? (value || horizontal[row - width + x])
                                              : (value && horizontal[row - width + x]) }
                    if y < height - 1 { value = dilate ? (value || horizontal[row + width + x])
                                                       : (value && horizontal[row + width + x]) }
                    vertical[row + x] = value
                }
            }
            current = vertical
        }
        return current
    }

    /// 4 近傍の連結成分ラベリング（0 = 背景、前景は 1..count）。
    /// 画素ごとに探索せず行ランごとに union-find する。
    static func labelComponents(_ mask: [Bool], width: Int, height: Int)
        -> (labels: [Int32], count: Int) {
        var labels = [Int32](repeating: 0, count: width * height)
        var parent: [Int32] = [0]

        func find(_ value: Int32) -> Int32 {
            var root = value
            while parent[Int(root)] != root { root = parent[Int(root)] }
            var walk = value
            while parent[Int(walk)] != root {
                let next = parent[Int(walk)]
                parent[Int(walk)] = root
                walk = next
            }
            return root
        }
        func union(_ a: Int32, _ b: Int32) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[Int(max(ra, rb))] = min(ra, rb) }
        }

        var previous: [(start: Int, end: Int, label: Int32)] = []
        for y in 0..<height {
            var current: [(start: Int, end: Int, label: Int32)] = []
            var x = 0
            let row = y * width
            while x < width {
                guard mask[row + x] else { x += 1; continue }
                let start = x
                while x < width && mask[row + x] { x += 1 }
                let end = x
                let overlapping = previous.filter { $0.start < end && start < $0.end }
                    .map { find($0.label) }
                var label: Int32
                if let smallest = overlapping.min() {
                    label = smallest
                    for other in overlapping { union(label, other) }
                } else {
                    parent.append(Int32(parent.count))
                    label = Int32(parent.count - 1)
                }
                for index in start..<end { labels[row + index] = label }
                current.append((start, end, label))
            }
            previous = current
        }

        guard parent.count > 1 else { return (labels, 0) }
        var remap = [Int32](repeating: 0, count: parent.count)
        var next: Int32 = 0
        for index in 1..<parent.count {
            let root = find(Int32(index))
            if remap[Int(root)] == 0 {
                next += 1
                remap[Int(root)] = next
            }
        }
        for index in 0..<labels.count where labels[index] != 0 {
            labels[index] = remap[Int(find(labels[index]))]
        }
        return (labels, Int(next))
    }

    static func dropSmall(_ mask: [Bool], width: Int, height: Int, minArea: Int) -> [Bool] {
        let (labels, count) = labelComponents(mask, width: width, height: height)
        guard count > 0 else { return mask }
        var areas = [Int](repeating: 0, count: count + 1)
        for label in labels { areas[Int(label)] += 1 }
        return labels.map { $0 != 0 && areas[Int($0)] >= minArea }
    }

    /// 外周とつながっていない穴のうち、minHole 未満のものを埋める。
    /// 外周連結を見るのは、意図的な抜き（虫眼鏡のレンズなど）を潰さないため。
    static func fillHoles(_ mask: [Bool], width: Int, height: Int, minHole: Int) -> [Bool] {
        let inverted = mask.map { !$0 }
        let (labels, count) = labelComponents(inverted, width: width, height: height)
        guard count > 0 else { return mask }
        var areas = [Int](repeating: 0, count: count + 1)
        for label in labels { areas[Int(label)] += 1 }
        var touching = Set<Int32>()
        for x in 0..<width {
            touching.insert(labels[x])
            touching.insert(labels[(height - 1) * width + x])
        }
        for y in 0..<height {
            touching.insert(labels[y * width])
            touching.insert(labels[y * width + width - 1])
        }
        var filled = mask
        for index in 0..<labels.count {
            let label = labels[index]
            if label != 0 && !touching.contains(label) && areas[Int(label)] < minHole {
                filled[index] = true
            }
        }
        return filled
    }

    // MARK: - 採点

    // 採点の重み。**Mac 側で 48 枚を人が見て並べ替えた順序に合わせた値で、理論値ではない。**
    // 参照実装 shapes.py と同じ数字であること。片方だけ動かすと数字が繋がらなくなる
    static let idealCoverage = (low: 0.16, high: 0.46)
    static let maxParts = 3
    static let maxRough = 2.3
    /// これ以上を「アイコンとして出せる形」とみなす。候補の合否表示に使う
    public static let passScore = 0.75

    /// 整形後の地図から「アイコンらしさ」を採点する。
    /// 後処理で直せない**構造の破綻**を機械が落とすための指標。人が「歪」と言う絵は
    /// たいてい「塗りが薄すぎる/多すぎる」「破片が散っている」「塊が多すぎる」
    /// 「輪郭が長すぎる」「端に当たっている」のどれかに当たる。
    static func report(_ labels: [Int8], width: Int, height: Int, mush: Double) -> Report {
        let canvas = width * height
        let ink = labels.map { $0 >= 0 }
        let area = ink.reduce(0) { $0 + ($1 ? 1 : 0) }
        guard area > 0 else {
            return Report(score: 0, coverage: 0, parts: 0, stray: 1,
                          rough: 0, mush: mush, margin: 0, symmetry: 0)
        }

        let coverage = Double(area) / Double(canvas)
        let (components, count) = labelComponents(ink, width: width, height: height)
        var areas = [Int](repeating: 0, count: count + 1)
        for label in components { areas[Int(label)] += 1 }
        let largest = areas.dropFirst().max() ?? 0
        let stray = 1.0 - Double(largest) / Double(area)
        let parts = areas.dropFirst().filter { Double($0) >= 0.004 * Double(canvas) }.count

        var perimeter = 0
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                if x + 1 < width && ink[row + x] != ink[row + x + 1] { perimeter += 1 }
                if y + 1 < height && ink[row + x] != ink[row + width + x] { perimeter += 1 }
            }
        }
        let rough = Double(perimeter) / (2.0 * (Double.pi * Double(area)).squareRoot())

        var minX = width, maxX = -1, minY = height, maxY = -1
        var intersection = 0, union = 0
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                let here = ink[row + x]
                let mirrored = ink[row + (width - 1 - x)]
                if here || mirrored { union += 1 }
                if here && mirrored { intersection += 1 }
                if here {
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }
        let margin = Double(min(minY, height - 1 - maxY, minX, width - 1 - maxX)) / Double(width)
        let symmetry = union == 0 ? 0 : Double(intersection) / Double(union)

        var penalty = 0.0
        if coverage < idealCoverage.low {
            penalty += 2.5 * (idealCoverage.low - coverage)
        } else if coverage > idealCoverage.high {
            penalty += 2.5 * (coverage - idealCoverage.high)
        }
        penalty += 1.2 * max(0.0, stray - 0.03)
        penalty += 0.08 * Double(max(0, parts - maxParts))
        penalty += 0.35 * max(0.0, rough - maxRough)
        // 色は量子化で決定的に直るので形の欠点より軽く見る
        penalty += 0.35 * max(0.0, mush - 0.35)
        penalty += 0.5 * max(0.0, (0.02 - margin) / 0.02)
        penalty += 0.15 * max(0.0, 0.55 - symmetry)

        return Report(score: max(0.0, min(1.0, 1.0 - penalty)),
                      coverage: coverage, parts: parts, stray: stray, rough: rough,
                      mush: mush, margin: margin, symmetry: symmetry)
    }

    // MARK: - 描き直し

    /// 地図をぼかした濃度場に変えてから描き直す。**これが「滑らかな線」の本体。**
    ///
    /// 512px の階段は振幅がちょうど 1px なので、色ごとのマスクを 1px 強でぼかして 0.5 で
    /// 切り直すと、階段だけが消えて角は角のまま残る。拡大は**二値化する前の濃度場に対して**
    /// 行うので、出力を 1024px にしても輪郭が階段状にならない。
    static func render(_ labels: [Int8], width: Int, height: Int,
                       size: Int, blur: Double, edge: Double,
                       plan: IconFit.Plan? = nil,
                       theme: Theme = .default) -> GeneratedImage {
        let scale = plan?.scale ?? 1.0
        let ratio = Double(size) * scale / Double(width)
        // 縮小するときは、縮めた後のぼけ幅が同じになるよう元解像度で強めにぼかす
        let sourceBlur = blur / min(scale, 1.0)
        // **傾きを 1 未満にしてはいけない。** 出力が元より小さいと境界は 1 画素未満で
        // 切り替わるため式の値が 1 を下回り、不透明度がどこも 0 にも 255 にもならず
        // 全面が半透明の灰色になる（Mac 側で 170px へ縮めて実際に出した）
        let slope = max((2.5 * max(sourceBlur, 0.35) * ratio) / max(edge, 0.5), 1.0)

        // **ラベル番号は palette のまま、塗る色だけ theme から取る**
        let colours = theme.colours

        var pieces: [(area: Int, value: Int, alpha: [Float])] = []
        for value in 0..<palette.count where theme.draws(value) {
            var field = [Float](repeating: 0, count: width * height)
            var area = 0
            for index in 0..<(width * height) where labels[index] == Int8(value) {
                field[index] = 1
                area += 1
            }
            if area == 0 { continue }
            if blur > 0 {
                field = gaussianBlur(field, width: width, height: height, sigma: sourceBlur)
            }
            let scaled = resample(field, width: width, height: height, size: size, plan: plan)
            let alpha = scaled.map { value -> Float in
                Float(min(max((Double(value) - 0.5) * slope + 0.5, 0.0), 1.0))
            }
            pieces.append((area, value, alpha))
        }

        // 大きい形から塗る。小さい形（ネイビーの上のコーラルの点など）を上に残すため
        var rgb = [Float](repeating: 0, count: size * size * 3)
        var accumulated = [Float](repeating: 0, count: size * size)
        for piece in pieces.sorted(by: { $0.area > $1.area }) {
            let colour = colours[piece.value]
            for index in 0..<(size * size) {
                let alpha = piece.alpha[index]
                if alpha <= 0 { continue }
                let inverse = 1 - alpha
                rgb[index * 3] = Float(colour.r) * alpha + rgb[index * 3] * inverse
                rgb[index * 3 + 1] = Float(colour.g) * alpha + rgb[index * 3 + 1] * inverse
                rgb[index * 3 + 2] = Float(colour.b) * alpha + rgb[index * 3 + 2] * inverse
                accumulated[index] = alpha + accumulated[index] * inverse
            }
        }

        var out = [UInt8](repeating: 0, count: size * size * 4)
        for index in 0..<(size * size) {
            out[index * 4] = UInt8(min(max(rgb[index * 3].rounded(), 0), 255))
            out[index * 4 + 1] = UInt8(min(max(rgb[index * 3 + 1].rounded(), 0), 255))
            out[index * 4 + 2] = UInt8(min(max(rgb[index * 3 + 2].rounded(), 0), 255))
            out[index * 4 + 3] = UInt8(min(max((accumulated[index] * 255).rounded(), 0), 255))
        }
        return GeneratedImage(width: size, height: size, rgba: out)
    }

    /// ガウスぼかしを箱ぼかし 3 回で近似する（Kovesi の式）。
    /// Pillow の GaussianBlur も同じ近似なので、参照実装と結果がほぼ一致する。
    static func gaussianBlur(_ field: [Float], width: Int, height: Int, sigma: Double) -> [Float] {
        var current = field
        for size in boxSizes(sigma: sigma) {
            let radius = (size - 1) / 2
            if radius <= 0 { continue }
            current = boxBlur(current, width: width, height: height, radius: radius)
        }
        return current
    }

    private static func boxSizes(sigma: Double, passes: Int = 3) -> [Int] {
        let variance = 12.0 * sigma * sigma
        var lower = Int((variance / Double(passes) + 1.0).squareRoot())
        if lower % 2 == 0 { lower -= 1 }
        lower = max(lower, 1)
        let ideal = (variance - Double(passes * lower * lower)
                     - Double(4 * passes * lower) - Double(3 * passes))
            / Double(-4 * lower - 4)
        let split = Int(ideal.rounded())
        return (0..<passes).map { $0 < split ? lower : lower + 2 }
    }

    private static func boxBlur(_ field: [Float], width: Int, height: Int,
                                radius: Int) -> [Float] {
        let window = Float(radius * 2 + 1)
        var horizontal = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = y * width
            var sum: Float = 0
            for x in -radius...radius { sum += field[row + min(max(x, 0), width - 1)] }
            for x in 0..<width {
                horizontal[row + x] = sum / window
                sum -= field[row + min(max(x - radius, 0), width - 1)]
                sum += field[row + min(max(x + radius + 1, 0), width - 1)]
            }
        }
        var vertical = [Float](repeating: 0, count: width * height)
        for x in 0..<width {
            var sum: Float = 0
            for y in -radius...radius { sum += horizontal[min(max(y, 0), height - 1) * width + x] }
            for y in 0..<height {
                vertical[y * width + x] = sum / window
                sum -= horizontal[min(max(y - radius, 0), height - 1) * width + x]
                sum += horizontal[min(max(y + radius + 1, 0), height - 1) * width + x]
            }
        }
        return vertical
    }

    /// 双一次補間で拡大・移動する。標本位置は Pillow と同じ「画素の中心」の流儀。
    /// `plan` があるときは、視覚重心を画面の中心へ持ってきつつ倍率を掛ける。
    private static func resample(_ field: [Float], width: Int, height: Int,
                                 size: Int, plan: IconFit.Plan?) -> [Float] {
        if plan == nil && size == width && size == height { return field }
        var out = [Float](repeating: 0, count: size * size)
        var scaleX = Double(width) / Double(size)
        var scaleY = Double(height) / Double(size)
        var shiftX = 0.0, shiftY = 0.0
        if let plan {
            scaleX = Double(width) / (Double(size) * plan.scale)
            scaleY = Double(height) / (Double(size) * plan.scale)
            shiftX = plan.centerX - scaleX * Double(size) / 2
            shiftY = plan.centerY - scaleY * Double(size) / 2
        }
        for y in 0..<size {
            // Pillow の Image.transform は画素の中心を標本する（source = a*(x+0.5) + c - 0.5）。
            // ここを整数座標のままにすると参照実装と半画素ずれる（実測で alpha 最大差 143）
            let sourceY = plan == nil ? (Double(y) + 0.5) * scaleY - 0.5
                                      : (Double(y) + 0.5) * scaleY + shiftY - 0.5
            let y0 = Int(floor(sourceY))
            let fy = Float(sourceY - Double(y0))
            let row0 = min(max(y0, 0), height - 1) * width
            let row1 = min(max(y0 + 1, 0), height - 1) * width
            for x in 0..<size {
                let sourceX = plan == nil ? (Double(x) + 0.5) * scaleX - 0.5
                                          : (Double(x) + 0.5) * scaleX + shiftX - 0.5
                let x0 = Int(floor(sourceX))
                let fx = Float(sourceX - Double(x0))
                if plan != nil && (sourceX < -1 || sourceX > Double(width)
                                   || sourceY < -1 || sourceY > Double(height)) {
                    out[y * size + x] = 0   // 変形で画面外に出た先は塗り無し
                    continue
                }
                let c0 = min(max(x0, 0), width - 1)
                let c1 = min(max(x0 + 1, 0), width - 1)
                let top = field[row0 + c0] * (1 - fx) + field[row0 + c1] * fx
                let bottom = field[row1 + c0] * (1 - fx) + field[row1 + c1] * fx
                out[y * size + x] = top * (1 - fy) + bottom * fy
            }
        }
        return out
    }
}

/// SNS アイコンとして成立させるための整えと検査。
/// **training/src/icon_diffusion/icon_fit.py の移植で、あちらが参照実装。**
///
/// SNS のプロフィール画像はほぼ例外なく**円形に切り抜かれ**、タイムラインでは 48px 前後まで
/// 縮む。効く基準は 3 つで、どれも機械で測れる。
///
/// 1. **円クロップの安全領域** — 正方形の四隅は必ず切れる。b5 の出力は 21%（10/48）が
///    内接円からはみ出していた（最大 28.6%）
/// 2. **隣接色とのコントラスト** — WCAG 2.1 の 1.4.11 は図形に 3:1 を求める。
///    **このパレットではコーラル×白が 2.82:1 で未達**。白地に置くとタイムラインで沈む
/// 3. **小サイズでの判別** — 48px で線が消えないか
public enum IconFit {
    /// 内接円の内側にどれだけ余白を残すか
    public static let safeRatio = 0.92
    /// 何割の塗りを安全領域に収めるか。**1.0 にしてはいけない** —
    /// 細い先端が 1 本はみ出しているだけで全体が 6 割に縮む（Mac 側で実測）
    public static let safeQuantile = 0.98
    public static let maxCoverage = 0.50
    public static let scaleLimits = (low: 0.55, high: 1.60)
    /// 端に触れている塊のうち、これ未満なら背景の切れ端とみなす
    public static let scrapShare = 0.15

    public struct Plan {
        public var scale: Double
        public var centerX: Double
        public var centerY: Double
        public var outsideBefore: Double
    }

    // MARK: - コントラスト

    public static func relativeLuminance(_ colour: (r: Double, g: Double, b: Double)) -> Double {
        func channel(_ value: Double) -> Double {
            let v = value / 255.0
            return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * channel(colour.r) + 0.7152 * channel(colour.g) + 0.0722 * channel(colour.b)
    }

    public static func contrastRatio(_ a: (r: Double, g: Double, b: Double),
                                     _ b: (r: Double, g: Double, b: Double)) -> Double {
        let la = relativeLuminance(a), lb = relativeLuminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    /// 敷ける背景。**#121212 はパレット外なので最後の逃げ道**（純黒を避けるのは
    /// 暗所でのハレーション対策。参照実装と同じ値）
    public static let backgrounds: [(name: String, colour: (r: Double, g: Double, b: Double))] =
        backgrounds(for: .default)

    /// WCAG 2.1 の 1.4.11（非テキストコントラスト）が求める比
    public static let minContrast = 3.0

    /// パレット内の色を優先して敷く順。`dark` は最後の逃げ道
    public static let backgroundOrder = ["white", "ink", "accent"]

    /// 敷ける背景。**利用者が色を選んだら、敷ける色もその 2 色になる。**
    /// 「トンマナ統一」はパレットが固定だから成り立つ話ではなく、
    /// 「絵に使っている色で地も作る」という規則なので theme に追随させる。
    public static func backgrounds(for theme: IconShapes.Theme)
        -> [(name: String, colour: (r: Double, g: Double, b: Double))] {
        [
            ("white", (255, 255, 255)),
            ("ink", theme.ink),
            ("accent", theme.accent),
            ("dark", (0x12, 0x12, 0x12)),
        ]
    }

    /// 既定のパレットでの名前。設定ファイルや保存済みの値が navy / coral で書ける
    static func resolveBackground(_ name: String, theme: IconShapes.Theme)
        -> (r: Double, g: Double, b: Double)? {
        switch name {
        case "navy": return IconShapes.Theme.default.ink
        case "coral": return IconShapes.Theme.default.accent
        default: return backgrounds(for: theme).first(where: { $0.name == name })?.colour
        }
    }

    /// 背景に接している色。**面積ではなく輪郭の長さで数える。**
    /// WCAG が求めるのは隣接色との比なので、内側の白（カップの中など）は数えない。
    static func silhouetteColours(_ labels: [Int8], width: Int, height: Int,
                                  minShare: Double = 0.10) -> [Int] {
        var edgeCount = [Int](repeating: 0, count: IconShapes.palette.count)
        var total = 0
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let value = labels[index]
                guard value >= 0 else { continue }
                var touches = false
                if x > 0 { touches = touches || labels[index - 1] < 0 }
                if x < width - 1 { touches = touches || labels[index + 1] < 0 }
                if y > 0 { touches = touches || labels[index - width] < 0 }
                if y < height - 1 { touches = touches || labels[index + width] < 0 }
                if touches { edgeCount[Int(value)] += 1; total += 1 }
            }
        }
        guard total > 0 else {
            return (0..<IconShapes.palette.count).filter { value in
                labels.contains(Int8(value))
            }
        }
        return (0..<IconShapes.palette.count)
            .filter { Double(edgeCount[$0]) / Double(total) >= minShare }
    }

    public static func backgroundContrast(_ labels: [Int8], width: Int, height: Int,
                                          background: String,
                                          theme: IconShapes.Theme = .default) -> Double {
        guard let colour = resolveBackground(background, theme: theme) else { return .infinity }
        // 塗らない抜きは地そのものなので、隣接色として数えない
        let values = silhouetteColours(labels, width: width, height: height)
            .filter { theme.draws($0) }
        guard !values.isEmpty else { return .infinity }
        let palette = theme.colours
        return values.map { contrastRatio(palette[$0], colour) }.min() ?? .infinity
    }

    /// 3:1 を満たす背景のうち、**パレット内で最初に見つかったもの**。
    /// コントラスト最大で選ぶと大半が #121212 になり、パレット外の色で一貫性が崩れる。
    public static func chooseBackground(_ labels: [Int8], width: Int, height: Int,
                                        theme: IconShapes.Theme = .default) -> String {
        guard labels.contains(where: { $0 >= 0 }) else { return "white" }
        for name in backgroundOrder where
            backgroundContrast(labels, width: width, height: height,
                               background: name, theme: theme) >= minContrast {
            return name
        }
        return (backgroundOrder + ["dark"])
            .map { ($0, backgroundContrast(labels, width: width, height: height,
                                           background: $0, theme: theme)) }
            .max(by: { $0.1 < $1.1 })?.0 ?? "white"
    }

    // MARK: - 切れ端の除去と大きさ揃え

    /// 背景の切れ端（端に触れていて、小さいか、円クロップでほぼ消える塊）を透過に落とす。
    /// **採点は落とす前に行うこと** — 切れ端があること自体が絵の欠点なので、
    /// 消してから採点すると下手な絵が上手に見えてしまう。
    public static func dropScraps(_ labels: [Int8], width: Int, height: Int) -> [Int8] {
        let ink = labels.map { $0 >= 0 }
        let (components, count) = IconShapes.labelComponents(ink, width: width, height: height)
        guard count > 1 else { return labels }

        var areas = [Int](repeating: 0, count: count + 1)
        var inside = [Int](repeating: 0, count: count + 1)
        let centre = (Double(width - 1) / 2, Double(height - 1) / 2)
        let radius = Double(width) / 2
        for y in 0..<height {
            for x in 0..<width {
                let label = Int(components[y * width + x])
                guard label != 0 else { continue }
                areas[label] += 1
                let dx = Double(x) - centre.0, dy = Double(y) - centre.1
                if (dx * dx + dy * dy).squareRoot() <= radius { inside[label] += 1 }
            }
        }
        let total = Double(areas.dropFirst().reduce(0, +))

        var touching = Set<Int32>()
        for x in 0..<width {
            touching.insert(components[x])
            touching.insert(components[(height - 1) * width + x])
        }
        for y in 0..<height {
            touching.insert(components[y * width])
            touching.insert(components[y * width + width - 1])
        }

        var scrap = [Bool](repeating: false, count: count + 1)
        for label in touching where label != 0 {
            let small = Double(areas[Int(label)]) / total < scrapShare
            let cropped = Double(inside[Int(label)]) / Double(max(areas[Int(label)], 1)) < 0.5
            if small || cropped { scrap[Int(label)] = true }
        }
        guard scrap.contains(true) else { return labels }

        var cleaned = labels
        var remaining = 0
        for index in 0..<labels.count where scrap[Int(components[index])] {
            cleaned[index] = -1
        }
        for value in cleaned where value >= 0 { remaining += 1; _ = value }
        return remaining > 0 ? cleaned : labels
    }

    /// 円クロップの安全領域に収まるよう、視覚重心へ寄せて大きさを揃える変形を決める。
    /// 基準は**塗りの 98% が入る半径**。全画素を円内に入れようとすると余白だらけになる。
    public static func plan(_ labels: [Int8], width: Int, height: Int) -> Plan {
        let ink = labels.map { $0 >= 0 }
        guard ink.contains(true) else {
            return Plan(scale: 1, centerX: Double(width - 1) / 2,
                        centerY: Double(height - 1) / 2, outsideBefore: 0)
        }
        // 切れ端は呼び出し側で落としてある前提。ここでは残った塗り全体を主役とみなす
        var sumX = 0.0, sumY = 0.0, area = 0.0
        for y in 0..<height {
            for x in 0..<width where ink[y * width + x] {
                sumX += Double(x); sumY += Double(y); area += 1
            }
        }
        let centreX = sumX / area, centreY = sumY / area

        var distances: [Double] = []
        distances.reserveCapacity(Int(area))
        var outside = 0.0
        let canvasCentre = (Double(width - 1) / 2, Double(height - 1) / 2)
        for y in 0..<height {
            for x in 0..<width where ink[y * width + x] {
                let dx = Double(x) - centreX, dy = Double(y) - centreY
                distances.append((dx * dx + dy * dy).squareRoot())
                let cx = Double(x) - canvasCentre.0, cy = Double(y) - canvasCentre.1
                if (cx * cx + cy * cy).squareRoot() > Double(width) / 2 { outside += 1 }
            }
        }
        distances.sort()
        let position = safeQuantile * Double(distances.count - 1)
        let lower = Int(position)
        let upper = min(lower + 1, distances.count - 1)
        let reach = distances[lower] + (position - Double(lower)) * (distances[upper] - distances[lower])

        let safeRadius = Double(width) / 2 * safeRatio
        var scale = safeRadius / max(reach, 1e-6)
        let coverage = area * scale * scale / Double(width * height)
        if coverage > maxCoverage { scale *= (maxCoverage / coverage).squareRoot() }
        scale = min(max(scale, scaleLimits.low), scaleLimits.high)

        return Plan(scale: scale, centerX: centreX, centerY: centreY,
                    outsideBefore: outside / area)
    }
}


extension IconFit {
    /// 単色背景を敷く。**透過のままにしないのは、タイムラインの地の色に左右されないため。**
    /// SNS 側が円に切り抜くので、背景は正方形いっぱいに塗る。
    public static func compose(_ image: GeneratedImage, background: String,
                               theme: IconShapes.Theme = .default) -> GeneratedImage {
        guard let colour = resolveBackground(background, theme: theme) else { return image }
        let base = (UInt8(colour.r), UInt8(colour.g), UInt8(colour.b))
        var rgba = image.rgba
        for index in 0..<(image.width * image.height) {
            let offset = index * 4
            let alpha = Double(rgba[offset + 3]) / 255.0
            if alpha >= 1.0 { continue }
            for channel in 0..<3 {
                let ink = Double(rgba[offset + channel]) * alpha
                let under = Double(channel == 0 ? base.0 : (channel == 1 ? base.1 : base.2))
                rgba[offset + channel] = UInt8(min(max((ink + under * (1 - alpha)).rounded(), 0), 255))
            }
            rgba[offset + 3] = 255
        }
        return GeneratedImage(width: image.width, height: image.height, rgba: rgba)
    }
}
