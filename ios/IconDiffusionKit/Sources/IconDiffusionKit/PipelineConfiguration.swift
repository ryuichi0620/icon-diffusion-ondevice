import Foundation

/// Stage 5 (`05_export_coreai.py`) が書き出す `pipeline_config.json` の中身。
///
/// β スケジュールや潜在の形を Swift 側にハードコードすると、変換元のモデルを差し替えたときに
/// 静かにズレる。実際に変換したモデルから書き出した値をそのまま読むことで、その一群のバグを消す。
public struct PipelineConfiguration: Codable, Sendable, Equatable {
    public var baseModel: String
    public var triggerWord: String
    public var promptTemplate: String

    public var numTrainTimesteps: Int
    public var betaStart: Double
    public var betaEnd: Double
    public var betaSchedule: String
    public var originalInferenceSteps: Int
    public var timestepScaling: Double
    public var sigmaData: Double

    public var vaeScalingFactor: Double
    /// true なら scaling_factor の割り戻しは VAEDecoder.aimodel の中で済んでいる。
    /// Swift 側で掛け直すと二重になるので、この値を必ず見ること。
    public var vaeScalingAppliedInModel: Bool

    public var latentChannels: Int
    public var latentSize: Int
    public var imageSize: Int
    public var textEmbedDim: Int
    public var maxSequenceLength: Int
    public var bosTokenID: Int
    public var eosTokenID: Int

    public init(
        baseModel: String,
        triggerWord: String,
        promptTemplate: String,
        numTrainTimesteps: Int,
        betaStart: Double,
        betaEnd: Double,
        betaSchedule: String,
        originalInferenceSteps: Int,
        timestepScaling: Double,
        sigmaData: Double,
        vaeScalingFactor: Double,
        vaeScalingAppliedInModel: Bool,
        latentChannels: Int,
        latentSize: Int,
        imageSize: Int,
        textEmbedDim: Int,
        maxSequenceLength: Int,
        bosTokenID: Int,
        eosTokenID: Int
    ) {
        self.baseModel = baseModel
        self.triggerWord = triggerWord
        self.promptTemplate = promptTemplate
        self.numTrainTimesteps = numTrainTimesteps
        self.betaStart = betaStart
        self.betaEnd = betaEnd
        self.betaSchedule = betaSchedule
        self.originalInferenceSteps = originalInferenceSteps
        self.timestepScaling = timestepScaling
        self.sigmaData = sigmaData
        self.vaeScalingFactor = vaeScalingFactor
        self.vaeScalingAppliedInModel = vaeScalingAppliedInModel
        self.latentChannels = latentChannels
        self.latentSize = latentSize
        self.imageSize = imageSize
        self.textEmbedDim = textEmbedDim
        self.maxSequenceLength = maxSequenceLength
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
    }

    enum CodingKeys: String, CodingKey {
        case baseModel = "base_model"
        case triggerWord = "trigger_word"
        case promptTemplate = "prompt_template"
        case numTrainTimesteps = "num_train_timesteps"
        case betaStart = "beta_start"
        case betaEnd = "beta_end"
        case betaSchedule = "beta_schedule"
        case originalInferenceSteps = "original_inference_steps"
        case timestepScaling = "timestep_scaling"
        case sigmaData = "sigma_data"
        case vaeScalingFactor = "vae_scaling_factor"
        case vaeScalingAppliedInModel = "vae_scaling_applied_in_model"
        case latentChannels = "latent_channels"
        case latentSize = "latent_size"
        case imageSize = "image_size"
        case textEmbedDim = "text_embed_dim"
        case maxSequenceLength = "max_sequence_length"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
    }

    /// SD1.5 の既定値。`pipeline_config.json` が無い場面（テストや疎通確認）で使う。
    public static let stableDiffusion15 = PipelineConfiguration(
        baseModel: "stable-diffusion-v1-5",
        triggerWord: "ICONST",
        promptTemplate: "an icon in {trigger} style, {subject}, flat, minimal",
        numTrainTimesteps: 1000,
        betaStart: 0.00085,
        betaEnd: 0.012,
        betaSchedule: "scaled_linear",
        originalInferenceSteps: 50,
        timestepScaling: 10.0,
        sigmaData: 0.5,
        vaeScalingFactor: 0.18215,
        vaeScalingAppliedInModel: true,
        latentChannels: 4,
        latentSize: 64,
        imageSize: 512,
        textEmbedDim: 768,
        maxSequenceLength: 77,
        bosTokenID: 49406,
        eosTokenID: 49407
    )

    public static func load(contentsOf url: URL) throws -> PipelineConfiguration {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw IconDiffusionError.missingResource(url)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PipelineConfiguration.self, from: data)
    }

    /// Stage 1〜3 と同じ形のプロンプトを組み立てる。学習時と生成時で文の形を揃えるため。
    public func prompt(subject: String) -> String {
        promptTemplate
            .replacingOccurrences(of: "{trigger}", with: triggerWord)
            .replacingOccurrences(of: "{subject}", with: subject)
    }

    public var latentShape: [Int] { [1, latentChannels, latentSize, latentSize] }
    public var textEmbeddingShape: [Int] { [1, maxSequenceLength, textEmbedDim] }
    public var decodedImageShape: [Int] { [1, 3, imageSize, imageSize] }
}
