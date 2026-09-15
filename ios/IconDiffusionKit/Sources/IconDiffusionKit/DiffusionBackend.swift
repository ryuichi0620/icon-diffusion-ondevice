import Foundation

/// 3 つのモデルを動かす口。**Core AI への依存はこのプロトコルの裏だけ**に閉じ込める。
///
/// こうしておくと、
///   - Core AI の API がこちらの想定とずれても直すのは `CoreAIBackend` だけで済む
///   - Core AI が無い環境（Linux CI、非対応端末）でも推論ループを組んで検証できる
///   - 非対応端末向けの逃げ道を別の実装として足せる
public protocol DiffusionBackend: Sendable {
    /// 生成ごとに 1 回。`[1, maxSequenceLength]` -> `[1, maxSequenceLength, textEmbedDim]`
    func encodeText(inputIDs: [Int32]) async throws -> Tensor

    /// 生成ごとにステップ数だけ。ここだけが実行回数 × なので最適化の対象。
    func predictNoise(sample: Tensor, timestep: Float, encoderHiddenStates: Tensor) async throws -> Tensor

    /// 生成ごとに 1 回。`[1, C, S, S]` -> `[1, 3, imageSize, imageSize]`（値域 `[-1, 1]`）
    func decodeLatent(_ latent: Tensor) async throws -> Tensor

    /// 輪郭を条件に与える経路が載っているか。載っていないビルドでも動くよう既定は false
    var supportsControl: Bool { get }

    /// 輪郭つきのノイズ予測。ControlNet で残差を出し、それを受け取る UNet に渡す。
    ///
    /// **これが「被写体が現実の物に見えない」への答え**（docs/experiments.md Trial 19〜21）。
    /// テキストだけで構造を指定するのをやめ、形は外から与えてスタイルだけモデルに任せる。
    /// `control` は `[1, 3, imageSize, imageSize]` の輪郭画像（白線・黒地）。
    func predictNoise(sample: Tensor, timestep: Float, encoderHiddenStates: Tensor,
                     control: Tensor) async throws -> Tensor
}

extension DiffusionBackend {
    /// 既定は「載っていない」。StubBackend と、ControlNet 抜きで書き出したモデルがここに入る
    public var supportsControl: Bool { false }

    public func predictNoise(sample: Tensor, timestep: Float, encoderHiddenStates: Tensor,
                             control: Tensor) async throws -> Tensor {
        throw IconDiffusionError.unsupportedDevice(
            "輪郭を条件に与える経路が載っていません（Stage 5 を --controlnet 付きで実行してください）"
        )
    }
}

/// モデルを使わず、形だけ正しい値を返すバックエンド。
///
/// パイプラインの配線（トークナイズ → ループ → デコード → 画素化）を、
/// 1.5GB のモデルも Core AI も無しで検証するために使う。
/// 出力は入力から決まる決定的な値なので、テストで固定値と突き合わせられる。
public struct StubBackend: DiffusionBackend {
    public let configuration: PipelineConfiguration
    /// `predictNoise` が返すノイズ予測の強さ。0 にすると x0 推定が sample そのものになる。
    public let noiseGain: Float

    public init(configuration: PipelineConfiguration = .stableDiffusion15, noiseGain: Float = 0.1) {
        self.configuration = configuration
        self.noiseGain = noiseGain
    }

    public func encodeText(inputIDs: [Int32]) async throws -> Tensor {
        let shape = [1, configuration.maxSequenceLength, configuration.textEmbedDim]
        guard inputIDs.count == configuration.maxSequenceLength else {
            throw IconDiffusionError.shapeMismatch(
                expected: [1, configuration.maxSequenceLength],
                actual: [1, inputIDs.count],
                context: "StubBackend.encodeText"
            )
        }
        // プロンプトが違えば埋め込みも違う、という最低限の性質だけ持たせる
        var values = [Float](repeating: 0, count: shape.reduce(1, *))
        for position in 0..<configuration.maxSequenceLength {
            let token = Float(inputIDs[position])
            for channel in 0..<configuration.textEmbedDim {
                values[position * configuration.textEmbedDim + channel] =
                    (token * 0.001) + Float(channel) * 0.0001
            }
        }
        return Tensor(shape: shape, values: values)
    }

    public func predictNoise(
        sample: Tensor,
        timestep: Float,
        encoderHiddenStates: Tensor
    ) async throws -> Tensor {
        try sample.requireShape(configuration.latentShape, context: "StubBackend.predictNoise(sample)")
        // 埋め込みの平均を混ぜて、条件つき／無条件で違う値が返るようにする（CFG の検証用）
        let conditioning = encoderHiddenStates.values.reduce(0, +) / Float(encoderHiddenStates.count)
        return Tensor(
            shape: sample.shape,
            values: sample.values.map { $0 * noiseGain + conditioning + timestep * 1e-5 }
        )
    }

    public func decodeLatent(_ latent: Tensor) async throws -> Tensor {
        try latent.requireShape(configuration.latentShape, context: "StubBackend.decodeLatent")
        let shape = configuration.decodedImageShape
        let pixels = configuration.imageSize * configuration.imageSize
        var values = [Float](repeating: 0, count: shape.reduce(1, *))
        for channel in 0..<3 {
            for pixel in 0..<pixels {
                // 潜在を折り返して敷き詰め、[-1, 1] に収める
                let source = latent.values[(pixel + channel) % latent.count]
                values[channel * pixels + pixel] = max(-1, min(1, source))
            }
        }
        return Tensor(shape: shape, values: values)
    }
}
