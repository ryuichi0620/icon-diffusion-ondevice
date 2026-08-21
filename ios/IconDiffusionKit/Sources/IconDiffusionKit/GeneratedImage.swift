import Foundation

/// 生成結果。プラットフォーム非依存の RGBA8 で持つ。
public struct GeneratedImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// 行優先の RGBA8。長さは `width * height * 4`。アルファは常に 255。
    public let rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) {
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    /// VAE デコーダの出力 `[1, 3, H, W]`（値域 `[-1, 1]`、チャネル分離）を RGBA8 に直す。
    public init(planarCHW tensor: Tensor) throws {
        guard tensor.shape.count == 4, tensor.shape[0] == 1, tensor.shape[1] == 3 else {
            throw IconDiffusionError.invalidConfiguration(
                "デコード結果は [1, 3, H, W] である必要があります: \(tensor.shape)"
            )
        }
        let height = tensor.shape[2]
        let width = tensor.shape[3]
        let pixelCount = width * height

        var rgba = [UInt8](repeating: 255, count: pixelCount * 4)
        for channel in 0..<3 {
            let planeOffset = channel * pixelCount
            for pixel in 0..<pixelCount {
                // [-1, 1] -> [0, 1] -> [0, 255]
                let normalized = min(max(tensor.values[planeOffset + pixel] * 0.5 + 0.5, 0), 1)
                rgba[pixel * 4 + channel] = UInt8((normalized * 255).rounded())
            }
        }
        self.init(width: width, height: height, rgba: rgba)
    }
}

#if canImport(CoreGraphics)
import CoreGraphics

extension GeneratedImage {
    public func makeCGImage() -> CGImage? {
        let bytesPerPixel = 4
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: bytesPerPixel * 8,
            bytesPerRow: width * bytesPerPixel,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
#endif
