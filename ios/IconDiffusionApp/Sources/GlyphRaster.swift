import CoreGraphics
import Foundation
import ImageIO

/// グリフの PNG を読んで、輪郭（白線・黒地）を作る。
///
/// **Canny ではなく形態学的に取る。** 入力が既に二値のグリフなので、Canny の
/// しきい値調整もノイズも要らず、膨張 − 収縮で 1〜2 画素の綺麗な境界が得られる。
/// 参照実装は training/scripts/controlnet_style.py の `outline()`。
enum GlyphRaster {
    /// PNG を読んで「塗りかどうか」の真偽値に落とす。グリフは黒地に白ではなく
    /// **白地に黒**（Mac 側の書き出しに合わせる）
    static func load(_ url: URL, size: Int) -> [Bool]? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        var pixels = [UInt8](repeating: 0, count: size * size)
        guard let context = CGContext(
            data: &pixels, width: size, height: size, bitsPerComponent: 8,
            bytesPerRow: size, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return pixels.map { $0 < 128 }
    }

    /// 膨張 − 収縮で境界を取る。3x3 の最大／最小を `width` 回。
    static func outline(_ ink: [Bool], width: Int, height: Int, lineWidth: Int) -> [Bool] {
        let grown = morph(ink, width: width, height: height, radius: lineWidth, dilate: true)
        let shrunk = morph(ink, width: width, height: height, radius: lineWidth, dilate: false)
        return zip(grown, shrunk).map { $0 && !$1 }
    }

    /// 3x3 の最大／最小フィルタを radius 回。分離可能なので横→縦の 2 パスで済ませる
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
                    if x > 0 {
                        value = dilate ? (value || current[row + x - 1])
                                       : (value && current[row + x - 1])
                    }
                    if x < width - 1 {
                        value = dilate ? (value || current[row + x + 1])
                                       : (value && current[row + x + 1])
                    }
                    horizontal[row + x] = value
                }
            }
            var vertical = [Bool](repeating: !dilate, count: width * height)
            for y in 0..<height {
                let row = y * width
                for x in 0..<width {
                    var value = horizontal[row + x]
                    if y > 0 {
                        value = dilate ? (value || horizontal[row - width + x])
                                       : (value && horizontal[row - width + x])
                    }
                    if y < height - 1 {
                        value = dilate ? (value || horizontal[row + width + x])
                                       : (value && horizontal[row + width + x])
                    }
                    vertical[row + x] = value
                }
            }
            current = vertical
        }
        return current
    }
}
