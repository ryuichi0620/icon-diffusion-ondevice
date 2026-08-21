import Foundation

/// 再現できる乱数源。SplitMix64 ＋ Box-Muller。
///
/// 拡散の初期潜在と、LCM が中間ステップで注ぎ直すノイズに使う。
/// 同じ seed なら端末をまたいでも同じ絵になる（`Float` の演算順序が同じである限り）。
///
/// - Important: PyTorch の `torch.randn` とは**別の乱数列**なので、Stage 3 の Mac プレビューと
///   同じ seed を指定しても同じ絵にはならない。端末側での再現性のための seed であって、
///   Mac との一致のためではない。
public struct SeededRandomSource: RandomNumberGenerator, Sendable {
    private var state: UInt64
    private var spareGaussian: Float?

    public init(seed: UInt64) {
        // seed 0 でも縮退しないように黄金比定数を足しておく
        self.state = seed &+ 0x9E37_79B9_7F4A_7C15
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// `[0, 1)` の一様乱数。上位 53bit だけ使う。
    public mutating func nextUniform() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// 標準正規分布。Box-Muller は 1 回で 2 つ作れるので、片方は次回に取っておく。
    public mutating func nextGaussian() -> Float {
        if let spare = spareGaussian {
            spareGaussian = nil
            return spare
        }
        // log(0) を踏まないよう (0, 1] に寄せる
        let u1 = 1.0 - nextUniform()
        let u2 = nextUniform()
        let radius = (-2.0 * Foundation.log(u1)).squareRoot()
        let angle = 2.0 * Double.pi * u2
        spareGaussian = Float(radius * Foundation.sin(angle))
        return Float(radius * Foundation.cos(angle))
    }

    public mutating func gaussianTensor(shape: [Int]) -> Tensor {
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        for index in 0..<count {
            values[index] = nextGaussian()
        }
        return Tensor(shape: shape, values: values)
    }
}
