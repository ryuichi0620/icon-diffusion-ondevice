import Foundation

/// バックエンド非依存の軽量テンソル。
///
/// Core AI の NDArray との相互変換は `CoreAIBackend` が受け持ち、スケジューラとパイプラインは
/// この型しか見ない。Core AI が無い環境でも推論ループを組み立てて検証できるようにするため。
public struct Tensor: Sendable, Equatable {
    public let shape: [Int]
    public let values: [Float]

    /// 呼び出し側が形を知っている前提のイニシャライザ。食い違いはプログラムの誤りとして落とす。
    public init(shape: [Int], values: [Float]) {
        precondition(shape.allSatisfy { $0 > 0 }, "shape に 0 以下の次元があります: \(shape)")
        precondition(
            values.count == shape.reduce(1, *),
            "要素数が shape と一致しません: shape=\(shape) count=\(values.count)"
        )
        self.shape = shape
        self.values = values
    }

    /// バックエンドから返ってきた実データ用。形が違えば例外にする。
    public init(shape: [Int], values: [Float], context: String) throws {
        guard shape.allSatisfy({ $0 > 0 }) else {
            throw IconDiffusionError.invalidConfiguration("shape に 0 以下の次元があります (\(context)): \(shape)")
        }
        guard values.count == shape.reduce(1, *) else {
            throw IconDiffusionError.shapeMismatch(
                expected: shape, actual: [values.count], context: context
            )
        }
        self.shape = shape
        self.values = values
    }

    public var count: Int { values.count }

    public static func zeros(shape: [Int]) -> Tensor {
        Tensor(shape: shape, values: [Float](repeating: 0, count: shape.reduce(1, *)))
    }

    public func scaled(by factor: Float) -> Tensor {
        Tensor(shape: shape, values: values.map { $0 * factor })
    }

    /// 同じ形の 2 つを要素ごとに合成する。CFG の合成やスケジューラの更新に使う。
    public static func combine(
        _ lhs: Tensor,
        _ rhs: Tensor,
        context: String = "combine",
        _ body: (Float, Float) -> Float
    ) throws -> Tensor {
        guard lhs.shape == rhs.shape else {
            throw IconDiffusionError.shapeMismatch(expected: lhs.shape, actual: rhs.shape, context: context)
        }
        var out = [Float](repeating: 0, count: lhs.count)
        for index in 0..<lhs.count {
            out[index] = body(lhs.values[index], rhs.values[index])
        }
        return Tensor(shape: lhs.shape, values: out)
    }

    /// 形が期待どおりか確かめる。バックエンドの出力を受けた直後に呼ぶ。
    public func requireShape(_ expected: [Int], context: String) throws {
        guard shape == expected else {
            throw IconDiffusionError.shapeMismatch(expected: expected, actual: shape, context: context)
        }
    }
}
