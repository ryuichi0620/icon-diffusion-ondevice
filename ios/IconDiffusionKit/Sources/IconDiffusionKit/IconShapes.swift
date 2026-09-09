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
    /// 学習データのパレット（SOURCE.md の生成プロンプトで固定している 2 色 + 白）
    static let palette: [(r: Double, g: Double, b: Double)] = [
        (0x1F, 0x2A, 0x44),   // deep navy
        (0xFF, 0x6B, 0x4A),   // coral
        (0xFF, 0xFF, 0xFF),   // white
    ]

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
    }

    public struct Polished {
        public var image: GeneratedImage
        public var report: Report
    }

    // MARK: - 入口

    /// 透過済みの画像を整えて描き直し、同時に採点する。
    /// `IconPostprocess.process()` の**後**に呼ぶこと（背景が残っていると全面が絵になる）。
    public static func polish(_ image: GeneratedImage,
                       size: Int? = nil,
                       blur: Double = defaultBlur,
                       edge: Double = 1.2) -> Polished {
        let width = image.width, height = image.height
        let labels = labelMap(image)
        let mush = mushRatio(image)
        let cleaned = clean(labels, width: width, height: height)
        let report = self.report(cleaned, width: width, height: height, mush: mush)
        let rendered = render(cleaned, width: width, height: height,
                              size: size ?? width, blur: blur, edge: edge)
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
                       size: Int, blur: Double, edge: Double) -> GeneratedImage {
        let ratio = Double(size) / Double(width)
        let slope = (2.5 * max(blur, 0.35) * ratio) / max(edge, 0.5)

        var pieces: [(area: Int, value: Int, alpha: [Float])] = []
        for value in 0..<palette.count {
            var field = [Float](repeating: 0, count: width * height)
            var area = 0
            for index in 0..<(width * height) where labels[index] == Int8(value) {
                field[index] = 1
                area += 1
            }
            if area == 0 { continue }
            if blur > 0 { field = gaussianBlur(field, width: width, height: height, sigma: blur) }
            let scaled = bilinear(field, width: width, height: height, size: size)
            let alpha = scaled.map { value -> Float in
                Float(min(max((Double(value) - 0.5) * slope + 0.5, 0.0), 1.0))
            }
            pieces.append((area, value, alpha))
        }

        // 大きい形から塗る。小さい形（ネイビーの上のコーラルの点など）を上に残すため
        var rgb = [Float](repeating: 0, count: size * size * 3)
        var accumulated = [Float](repeating: 0, count: size * size)
        for piece in pieces.sorted(by: { $0.area > $1.area }) {
            let colour = palette[piece.value]
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

    /// 双一次補間で拡大する。標本位置は Pillow と同じ「画素の中心」の流儀に合わせてある。
    private static func bilinear(_ field: [Float], width: Int, height: Int,
                                 size: Int) -> [Float] {
        if size == width && size == height { return field }
        var out = [Float](repeating: 0, count: size * size)
        let scaleX = Double(width) / Double(size)
        let scaleY = Double(height) / Double(size)
        for y in 0..<size {
            let sourceY = (Double(y) + 0.5) * scaleY - 0.5
            let y0 = Int(floor(sourceY))
            let fy = Float(sourceY - Double(y0))
            let row0 = min(max(y0, 0), height - 1) * width
            let row1 = min(max(y0 + 1, 0), height - 1) * width
            for x in 0..<size {
                let sourceX = (Double(x) + 0.5) * scaleX - 0.5
                let x0 = Int(floor(sourceX))
                let fx = Float(sourceX - Double(x0))
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
