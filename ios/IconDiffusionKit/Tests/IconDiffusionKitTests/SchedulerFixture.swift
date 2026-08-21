import Foundation

/// `Fixtures/expected_scheduler.json` の型。生成元は Python の参照実装。
struct SchedulerFixture: Decodable {
    struct Scalings: Decodable {
        let cSkip: Double
        let cOut: Double
        enum CodingKeys: String, CodingKey {
            case cSkip = "c_skip"
            case cOut = "c_out"
        }
    }

    struct Step: Decodable {
        let index: Int
        let previousSample: [Double]
        let denoised: [Double]
        enum CodingKeys: String, CodingKey {
            case index
            case previousSample = "prev_sample"
            case denoised
        }
    }

    struct StepCase: Decodable {
        let sample: [Double]
        let modelOutput: [Double]
        let noise: [Double]
        let timesteps: [Int]
        let steps: [Step]
        enum CodingKeys: String, CodingKey {
            case sample
            case modelOutput = "model_output"
            case noise
            case timesteps
            case steps
        }
    }

    let betaStart: Double
    let betaEnd: Double
    let numTrainTimesteps: Int
    let originalInferenceSteps: Int
    let sigmaData: Double
    let timestepScaling: Double
    let timesteps: [String: [Int]]
    let alphasCumprodAt: [String: Double]
    let scalingsAt: [String: Scalings]
    let stepCase: StepCase
    let fullLoopFinal: [Double]

    enum CodingKeys: String, CodingKey {
        case betaStart = "beta_start"
        case betaEnd = "beta_end"
        case numTrainTimesteps = "num_train_timesteps"
        case originalInferenceSteps = "original_inference_steps"
        case sigmaData = "sigma_data"
        case timestepScaling = "timestep_scaling"
        case timesteps
        case alphasCumprodAt = "alphas_cumprod_at"
        case scalingsAt = "scalings_at"
        case stepCase = "step_case"
        case fullLoopFinal = "full_loop_final"
    }

    var schedulerConfiguration: LCMSchedulerConfigurationValues {
        LCMSchedulerConfigurationValues(
            numTrainTimesteps: numTrainTimesteps,
            betaStart: betaStart,
            betaEnd: betaEnd,
            originalInferenceSteps: originalInferenceSteps,
            timestepScaling: timestepScaling,
            sigmaData: sigmaData
        )
    }
}

/// フィクスチャの値をそのまま持つだけの箱（テスト内で組み立てを 1 か所にまとめるため）。
struct LCMSchedulerConfigurationValues {
    let numTrainTimesteps: Int
    let betaStart: Double
    let betaEnd: Double
    let originalInferenceSteps: Int
    let timestepScaling: Double
    let sigmaData: Double
}
