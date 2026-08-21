import Foundation

/// LCM（Latent Consistency Model）のスケジューラ。
///
/// Core AI が動かすのは UNet の 1 回分だけなので、「ノイズ予測 → 少し綺麗にする」の後半は
/// 自分で持つ必要がある。diffusers の `LCMScheduler` と同じ式を実装している。
///
/// ノイズを外から渡す設計にしてあるのは、乱数を含んだまま検証できないから。
/// パイプラインが `SeededRandomSource` から供給し、テストは固定値を渡す。
public struct LCMScheduler: Sendable {
    public struct Configuration: Sendable, Equatable {
        public var numTrainTimesteps: Int
        public var betaStart: Double
        public var betaEnd: Double
        public var betaSchedule: String
        public var originalInferenceSteps: Int
        public var timestepScaling: Double
        public var sigmaData: Double

        public init(
            numTrainTimesteps: Int = 1000,
            betaStart: Double = 0.00085,
            betaEnd: Double = 0.012,
            betaSchedule: String = "scaled_linear",
            originalInferenceSteps: Int = 50,
            timestepScaling: Double = 10.0,
            sigmaData: Double = 0.5
        ) {
            self.numTrainTimesteps = numTrainTimesteps
            self.betaStart = betaStart
            self.betaEnd = betaEnd
            self.betaSchedule = betaSchedule
            self.originalInferenceSteps = originalInferenceSteps
            self.timestepScaling = timestepScaling
            self.sigmaData = sigmaData
        }

        public init(pipeline: PipelineConfiguration) {
            self.init(
                numTrainTimesteps: pipeline.numTrainTimesteps,
                betaStart: pipeline.betaStart,
                betaEnd: pipeline.betaEnd,
                betaSchedule: pipeline.betaSchedule,
                originalInferenceSteps: pipeline.originalInferenceSteps,
                timestepScaling: pipeline.timestepScaling,
                sigmaData: pipeline.sigmaData
            )
        }
    }

    public struct StepResult: Sendable, Equatable {
        /// 次のステップに渡す潜在。最終ステップでは `denoised` と同じ。
        public let previousSample: Tensor
        /// このステップ時点での x0 推定。途中経過を見たいときに使う。
        public let denoised: Tensor
    }

    public let configuration: Configuration
    public let timesteps: [Int]
    private let alphasCumprod: [Double]

    /// LCM では初期潜在をスケールしない（`init_noise_sigma = 1.0`）。
    public var initialNoiseSigma: Float { 1.0 }

    public init(configuration: Configuration = Configuration(), inferenceSteps: Int) throws {
        guard inferenceSteps >= 1 else {
            throw IconDiffusionError.invalidConfiguration("ステップ数は 1 以上にしてください: \(inferenceSteps)")
        }
        guard inferenceSteps <= configuration.originalInferenceSteps else {
            throw IconDiffusionError.invalidConfiguration(
                "ステップ数が original_inference_steps (\(configuration.originalInferenceSteps)) を超えています: \(inferenceSteps)"
            )
        }
        guard configuration.numTrainTimesteps >= configuration.originalInferenceSteps,
              configuration.originalInferenceSteps >= 1 else {
            throw IconDiffusionError.invalidConfiguration(
                "num_train_timesteps と original_inference_steps の関係が不正です"
            )
        }

        self.configuration = configuration
        self.alphasCumprod = try Self.makeAlphasCumprod(configuration)
        self.timesteps = Self.makeTimesteps(configuration, inferenceSteps: inferenceSteps)
    }

    // MARK: - スケジュール

    private static func makeAlphasCumprod(_ configuration: Configuration) throws -> [Double] {
        let count = configuration.numTrainTimesteps
        let betas: [Double]

        switch configuration.betaSchedule {
        case "scaled_linear":
            // SD 系の既定。√β を等間隔に並べてから 2 乗する
            let start = configuration.betaStart.squareRoot()
            let end = configuration.betaEnd.squareRoot()
            betas = (0..<count).map { index in
                let t = Double(index) / Double(count - 1)
                let value = start + (end - start) * t
                return value * value
            }
        case "linear":
            betas = (0..<count).map { index in
                let t = Double(index) / Double(count - 1)
                return configuration.betaStart + (configuration.betaEnd - configuration.betaStart) * t
            }
        default:
            throw IconDiffusionError.invalidConfiguration(
                "未対応の beta_schedule です: \(configuration.betaSchedule)"
            )
        }

        var cumulative = 1.0
        return betas.map { beta in
            cumulative *= (1.0 - beta)
            return cumulative
        }
    }

    /// LCM の飛び飛びタイムステップ。4 ステップなら `[999, 759, 519, 279]`。
    ///
    /// diffusers の `lcm_origin_timesteps[::-skipping_step][:num_inference_steps]` と同じ並び。
    private static func makeTimesteps(_ configuration: Configuration, inferenceSteps: Int) -> [Int] {
        let k = configuration.numTrainTimesteps / configuration.originalInferenceSteps
        let origin = (1...configuration.originalInferenceSteps).map { $0 * k - 1 }
        let skipping = max(1, configuration.originalInferenceSteps / inferenceSteps)

        var result: [Int] = []
        var index = origin.count - 1
        while index >= 0 && result.count < inferenceSteps {
            result.append(origin[index])
            index -= skipping
        }
        return result
    }

    public func alphaCumprod(at timestep: Int) -> Double {
        alphasCumprod[min(max(timestep, 0), alphasCumprod.count - 1)]
    }

    /// 境界条件のスケーリング。t が大きいほど c_out が 1 に近づき、モデル出力を強く信じる。
    public func scalings(forTimestep timestep: Int) -> (cSkip: Double, cOut: Double) {
        let scaled = Double(timestep) * configuration.timestepScaling
        let sigmaSquared = configuration.sigmaData * configuration.sigmaData
        let denominator = scaled * scaled + sigmaSquared
        return (sigmaSquared / denominator, scaled / denominator.squareRoot())
    }

    // MARK: - 1 ステップ

    /// - Parameters:
    ///   - modelOutput: UNet が返したノイズ予測（ε パラメータ化）
    ///   - sample: 現在の潜在
    ///   - stepIndex: `timesteps` の何番目か
    ///   - noise: 中間ステップで注ぎ直すノイズ。最終ステップでは使われない。
    ///     **これを渡さない（＝ゼロにする）と 4 ステップでも眠い絵になる。LCM の肝。**
    public func step(
        modelOutput: Tensor,
        sample: Tensor,
        stepIndex: Int,
        noise: Tensor
    ) throws -> StepResult {
        guard timesteps.indices.contains(stepIndex) else {
            throw IconDiffusionError.invalidConfiguration(
                "stepIndex が範囲外です: \(stepIndex) (steps=\(timesteps.count))"
            )
        }
        try modelOutput.requireShape(sample.shape, context: "LCMScheduler.step(modelOutput)")
        try noise.requireShape(sample.shape, context: "LCMScheduler.step(noise)")

        let isFinalStep = stepIndex == timesteps.count - 1
        let timestep = timesteps[stepIndex]
        let previousTimestep = isFinalStep ? timestep : timesteps[stepIndex + 1]

        let alphaT = alphaCumprod(at: timestep)
        let alphaPrev = alphaCumprod(at: previousTimestep)
        let sqrtAlphaT = Float(alphaT.squareRoot())
        let sqrtBetaT = Float((1.0 - alphaT).squareRoot())
        let (cSkip, cOut) = scalings(forTimestep: timestep)

        // x0 の推定 -> 境界条件で混ぜる
        let denoised = try Tensor.combine(sample, modelOutput, context: "LCMScheduler.denoised") { x, epsilon in
            let predictedOriginal = (x - sqrtBetaT * epsilon) / sqrtAlphaT
            return Float(cOut) * predictedOriginal + Float(cSkip) * x
        }

        if isFinalStep {
            return StepResult(previousSample: denoised, denoised: denoised)
        }

        let sqrtAlphaPrev = Float(alphaPrev.squareRoot())
        let sqrtBetaPrev = Float((1.0 - alphaPrev).squareRoot())
        let previousSample = try Tensor.combine(denoised, noise, context: "LCMScheduler.renoise") { d, z in
            sqrtAlphaPrev * d + sqrtBetaPrev * z
        }
        return StepResult(previousSample: previousSample, denoised: denoised)
    }
}
