import CoreGraphics
import Foundation
import IconDiffusionKit

/// 生成アイコンの後処理と機械判定。**training/src/icon_diffusion/postprocess.py の移植で、
/// あちらが参照実装。** 同じ処理を載せることで、Mac で測った「透過成功率 74.5%」が
/// そのまま端末の判定に一致する（アバター側 PartScreen と同じ設計判断）。
///
/// 処理は 2 段:
///   1. 背景の透過 — 縁の中央値色を背景とみなし、縁から許容誤差内を塗りつぶしで抜く。
///      **縁から連結した領域だけを抜く**のは、被写体内部の同色（白い湯気など）を
///      巻き込まないため（参照実装の _flood_from_border と同じ理由）
///   2. パレット量子化 — 残った画素をネイビー/コーラル/白の最近色へ。推論時の
///      色のはみ出し（ピンク・薄紫）は「モデルの迷い」であって意図色ではない
enum IconPostprocess {
    /// 学習データのパレット（SOURCE.md の生成プロンプトで固定している 2 色 + 白）
    static let palette: [(UInt8, UInt8, UInt8)] = [
        (0x1F, 0x2A, 0x44),   // deep navy
        (0xFF, 0x6B, 0x4A),   // coral
        (0xFF, 0xFF, 0xFF),   // white
    ]

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

        // --- 2. 前景をパレットへ量子化、背景はアルファ 0 ---
        var transparentCount = 0
        for index in 0..<(width * height) {
            let o = index * 4
            if isBackground[index] {
                rgba[o] = 0; rgba[o + 1] = 0; rgba[o + 2] = 0; rgba[o + 3] = 0
                transparentCount += 1
            } else {
                var best = 0
                var bestDistance = Int.max
                for (i, colour) in palette.enumerated() {
                    let dr = Int(rgba[o]) - Int(colour.0)
                    let dg = Int(rgba[o + 1]) - Int(colour.1)
                    let db = Int(rgba[o + 2]) - Int(colour.2)
                    let distance = dr * dr + dg * dg + db * db
                    if distance < bestDistance { bestDistance = distance; best = i }
                }
                rgba[o] = palette[best].0
                rgba[o + 1] = palette[best].1
                rgba[o + 2] = palette[best].2
                rgba[o + 3] = 255
            }
        }

        return Result(
            image: GeneratedImage(width: width, height: height, rgba: rgba),
            transparency: Double(transparentCount) / Double(width * height)
        )
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
