import CoreGraphics
import Foundation
import IconDiffusionKit

/// 生成アイコンの背景を透過にする。**training/src/icon_diffusion/postprocess.py の移植で、
/// あちらが参照実装。** 同じ処理を載せることで、Mac で測った「透過成功率 74.5%」が
/// そのまま端末の判定に一致する（アバター側 PartScreen と同じ設計判断）。
///
/// 縁の中央値色を背景とみなし、**縁から連結した領域だけ**を抜く。被写体内部の同色
/// （白い湯気など）を巻き込まないため（参照実装の _flood_from_border と同じ理由）。
///
/// 色の量子化と形の整形はこの後段の `IconShapes.polish()` が持つ。ここで量子化しないのは、
/// 「中途半端な塗り」の量（IconShapes.mushRatio）を**量子化する前の色**で測るため。
enum IconPostprocess {
    struct Result {
        var image: GeneratedImage
        /// 透過にできた画素の割合。**これが機械判定の指標**（>= 0.5 で合格）。
        /// Mac 側の歩留まり計測（透過成功率）と同じ定義。
        var transparency: Double
    }

    static func process(_ generated: GeneratedImage,
                        tolerance: Int = 24) -> Result {
        let width = generated.width
        let height = generated.height
        var rgba = generated.rgba

        // --- 1. 縁の中央値色 = 背景色 ---
        var borderR: [UInt8] = [], borderG: [UInt8] = [], borderB: [UInt8] = []
        func sample(_ x: Int, _ y: Int) {
            let o = (y * width + x) * 4
            borderR.append(rgba[o]); borderG.append(rgba[o + 1]); borderB.append(rgba[o + 2])
        }
        for x in stride(from: 0, to: width, by: 2) { sample(x, 0); sample(x, height - 1) }
        for y in stride(from: 0, to: height, by: 2) { sample(0, y); sample(width - 1, y) }
        func median(_ values: [UInt8]) -> Int {
            Int(values.sorted()[values.count / 2])
        }
        let bg = (median(borderR), median(borderG), median(borderB))

        // --- 縁からの塗りつぶしで背景を透過に ---
        var isBackground = [Bool](repeating: false, count: width * height)
        var queue: [Int] = []
        func nearBackground(_ index: Int) -> Bool {
            let o = index * 4
            return abs(Int(rgba[o]) - bg.0) <= tolerance
                && abs(Int(rgba[o + 1]) - bg.1) <= tolerance
                && abs(Int(rgba[o + 2]) - bg.2) <= tolerance
        }
        for x in 0..<width {
            for y in [0, height - 1] {
                let index = y * width + x
                if !isBackground[index] && nearBackground(index) {
                    isBackground[index] = true; queue.append(index)
                }
            }
        }
        for y in 0..<height {
            for x in [0, width - 1] {
                let index = y * width + x
                if !isBackground[index] && nearBackground(index) {
                    isBackground[index] = true; queue.append(index)
                }
            }
        }
        while let index = queue.popLast() {
            let x = index % width, y = index / width
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                let n = ny * width + nx
                if !isBackground[n] && nearBackground(n) {
                    isBackground[n] = true; queue.append(n)
                }
            }
        }

        // --- 1.5 外周から届かない地の色のうち、細いスジも背景に含める ---
        addEnclosedSeams(&isBackground, width: width, height: height,
                         candidate: nearBackground)

        // --- 2. 背景はアルファ 0、前景は色をそのまま残す ---
        var transparentCount = 0
        for index in 0..<(width * height) {
            let o = index * 4
            if isBackground[index] {
                rgba[o] = 0; rgba[o + 1] = 0; rgba[o + 2] = 0; rgba[o + 3] = 0
                transparentCount += 1
            } else {
                rgba[o + 3] = 255
            }
        }

        return Result(
            image: GeneratedImage(width: width, height: height, rgba: rgba),
            transparency: Double(transparentCount) / Double(width * height)
        )
    }

    /// 外周から届かない地の色の領域のうち、**細いスジだけ**を背景に足す。
    /// 参照実装は postprocess.py の `_enclosed_seams`。
    ///
    /// 外周続きだけを抜く規則は、湯気やカレンダーの升目のような意匠の白抜きを守るために
    /// 要る。だがその規則は、図形どうしの境目に残った地の色も守ってしまう。実測では
    /// 64 枚中 63 枚に高さ 6〜10px・幅 350〜670px の白が残り、単色背景を敷くと割れ目に
    /// 見えるうえ、白 x コーラルは 2.82:1 で WCAG の 3:1 を満たさない。
    ///
    /// 意匠の白抜きは面なので侵食しても残る。スジは残らない。この違いで分ける。
    static func addEnclosedSeams(_ isBackground: inout [Bool],
                                 width: Int, height: Int,
                                 candidate: (Int) -> Bool) {
        // 半径は画素数ではなく画布に対して決める。512 と 1024 で判定が変わっては困る
        let radius = max(1, Int((5.0 * Double(min(width, height)) / 1024.0).rounded()))
        let minArea = max(50, (width * height) / 5000)

        var visited = [Bool](repeating: false, count: width * height)
        // 成分ごとに確保し直すと 1024x1024 で 1MB ずつ積む。1 枚だけ使い回し、
        // 見終わった成分の画素だけ戻す
        var member = [Bool](repeating: false, count: width * height)
        for start in 0..<(width * height) {
            guard !visited[start], !isBackground[start], candidate(start) else { continue }
            var part: [Int] = []
            var queue = [start]
            visited[start] = true
            var minX = width, maxX = 0, minY = height, maxY = 0
            while let index = queue.popLast() {
                part.append(index)
                let x = index % width, y = index / width
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                    guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                    let n = ny * width + nx
                    if !visited[n], !isBackground[n], candidate(n) {
                        visited[n] = true; queue.append(n)
                    }
                }
            }
            guard part.count >= minArea else { continue }
            for index in part { member[index] = true }
            let survives = survivesErosion(part, member: member, width: width, height: height,
                                           radius: radius, minX: minX, maxX: maxX,
                                           minY: minY, maxY: maxY)
            for index in part { member[index] = false }
            if !survives {
                for index in part { isBackground[index] = true }
            }
        }
    }

    /// 半径 `radius` の侵食後に画素が残るか。参照実装は 3x3 の最小フィルタを radius 回なので、
    /// 円ではなく **一辺 2r+1 の正方形**で見るのが等価。囲む箱の中だけ調べれば足りる
    private static func survivesErosion(_ part: [Int], member: [Bool],
                                        width: Int, height: Int, radius: Int,
                                        minX: Int, maxX: Int, minY: Int, maxY: Int) -> Bool {
        // 箱の短辺が直径に満たなければ、中身を見るまでもなく消える
        if min(maxX - minX + 1, maxY - minY + 1) <= 2 * radius { return false }
        for index in part {
            let x = index % width, y = index / width
            if x - radius < 0 || x + radius >= width || y - radius < 0 || y + radius >= height {
                continue
            }
            var solid = true
            outer: for dy in -radius...radius {
                for dx in -radius...radius {
                    if !member[(y + dy) * width + (x + dx)] { solid = false; break outer }
                }
            }
            if solid { return true }
        }
        return false
    }
}


extension GeneratedImage {
    /// アルファを活かした CGImage。Kit の makeCGImage は noneSkipLast（アルファ無視）なので、
    /// 透過アイコンの表示・保存にはこちらを使う。IconPostprocess.process の出力は
    /// 背景画素の RGB を 0 にしてあるため、premultipliedLast として正しい。
    func makeAlphaCGImage() -> CGImage? {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
