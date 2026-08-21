import Foundation
import Testing

@testable import IconDiffusionKit

@Suite("補助的な型")
struct SupportTypesTests {

    // MARK: - GeneratedImage

    @Test("VAE 出力 [1,3,H,W] の [-1,1] を RGBA8 に直す")
    func planarToRGBA() throws {
        // 幅 2 / 高さ 1。チャネルごとに面が分かれている
        let tensor = Tensor(
            shape: [1, 3, 1, 2],
            values: [
                -1.0, 1.0,     // R
                 0.0, 0.5,     // G
                -0.5, 1.0,     // B
            ]
        )
        let image = try GeneratedImage(planarCHW: tensor)

        #expect(image.width == 2)
        #expect(image.height == 1)
        // (v/2 + 0.5) * 255 を四捨五入
        #expect(Array(image.rgba[0..<4]) == [0, 128, 64, 255])
        #expect(Array(image.rgba[4..<8]) == [255, 191, 255, 255])
    }

    @Test("値域を外れた出力は 0-255 に丸め込む")
    func planarClampsOutOfRange() throws {
        let tensor = Tensor(shape: [1, 3, 1, 1], values: [5.0, -5.0, 0.0])
        let image = try GeneratedImage(planarCHW: tensor)
        #expect(Array(image.rgba) == [255, 0, 128, 255])
    }

    @Test("[1,3,H,W] でない形は弾く")
    func planarRejectsWrongRank() {
        #expect(throws: IconDiffusionError.self) {
            _ = try GeneratedImage(planarCHW: Tensor.zeros(shape: [1, 4, 8, 8]))
        }
        #expect(throws: IconDiffusionError.self) {
            _ = try GeneratedImage(planarCHW: Tensor.zeros(shape: [3, 8, 8]))
        }
    }

    // MARK: - Tensor

    @Test("形の違うテンソル同士は合成できない")
    func combineRejectsShapeMismatch() {
        let lhs = Tensor.zeros(shape: [1, 4, 8, 8])
        let rhs = Tensor.zeros(shape: [1, 4, 4, 4])
        #expect(throws: IconDiffusionError.self) {
            _ = try Tensor.combine(lhs, rhs) { $0 + $1 }
        }
    }

    @Test("要素数が合わないデータは throw する init で弾ける")
    func throwingInitValidatesCount() {
        #expect(throws: IconDiffusionError.self) {
            _ = try Tensor(shape: [1, 4, 8, 8], values: [1, 2, 3], context: "test")
        }
    }

    @Test("requireShape は期待どおりの形なら通る")
    func requireShapePasses() throws {
        let tensor = Tensor.zeros(shape: [1, 4, 8, 8])
        try tensor.requireShape([1, 4, 8, 8], context: "test")
        #expect(throws: IconDiffusionError.self) {
            try tensor.requireShape([1, 4, 8, 9], context: "test")
        }
    }

    // MARK: - 乱数

    @Test("同じ seed なら同じ乱数列、違う seed なら違う列")
    func randomIsSeedDeterministic() {
        var a = SeededRandomSource(seed: 42)
        var b = SeededRandomSource(seed: 42)
        var c = SeededRandomSource(seed: 43)

        let first = (0..<8).map { _ in a.nextGaussian() }
        let same = (0..<8).map { _ in b.nextGaussian() }
        let other = (0..<8).map { _ in c.nextGaussian() }

        #expect(first == same)
        #expect(first != other)
    }

    @Test("正規分布として妥当（平均 0・標準偏差 1）", arguments: [UInt64(0), 42, 12345])
    func gaussianDistributionIsSane(seed: UInt64) {
        var random = SeededRandomSource(seed: seed)
        let count = 20_000
        let samples = (0..<count).map { _ in Double(random.nextGaussian()) }

        let mean = samples.reduce(0, +) / Double(count)
        let variance = samples.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count - 1)
        let deviation = variance.squareRoot()

        #expect(abs(mean) < 0.05, "平均が 0 から離れています: \(mean)")
        #expect(abs(deviation - 1.0) < 0.05, "標準偏差が 1 から離れています: \(deviation)")
    }

    @Test("一様乱数は [0, 1) に収まる")
    func uniformStaysInRange() {
        var random = SeededRandomSource(seed: 1)
        for _ in 0..<20_000 {
            let value = random.nextUniform()
            #expect(value >= 0.0 && value < 1.0)
        }
    }

    @Test("gaussianTensor は指定した形で返る")
    func gaussianTensorShape() {
        var random = SeededRandomSource(seed: 7)
        let tensor = random.gaussianTensor(shape: [1, 4, 8, 8])
        #expect(tensor.shape == [1, 4, 8, 8])
        #expect(tensor.count == 256)
        #expect(tensor.values.allSatisfy({ $0.isFinite }))
    }

    // MARK: - 設定

    @Test("pipeline_config.json と同じキーで復元できる")
    func configurationDecodesSnakeCaseKeys() throws {
        let json = """
        {
          "base_model": "sd15", "trigger_word": "ICONST",
          "prompt_template": "an icon in {trigger} style, {subject}, flat, minimal",
          "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
          "beta_schedule": "scaled_linear", "original_inference_steps": 50,
          "timestep_scaling": 10.0, "sigma_data": 0.5,
          "vae_scaling_factor": 0.18215, "vae_scaling_applied_in_model": true,
          "latent_channels": 4, "latent_size": 64, "image_size": 512,
          "text_embed_dim": 768, "max_sequence_length": 77,
          "bos_token_id": 49406, "eos_token_id": 49407,
          "verification": { "timesteps_4step": [999, 759, 519, 279] }
        }
        """
        let configuration = try JSONDecoder().decode(
            PipelineConfiguration.self, from: Data(json.utf8)
        )
        // 未知のキー（verification）が増えても壊れないこと
        #expect(configuration == .stableDiffusion15.withBaseModel("sd15"))
        #expect(configuration.latentShape == [1, 4, 64, 64])
        #expect(configuration.decodedImageShape == [1, 3, 512, 512])
        #expect(
            configuration.prompt(subject: "a coffee cup")
                == "an icon in ICONST style, a coffee cup, flat, minimal"
        )
    }
}

private extension PipelineConfiguration {
    func withBaseModel(_ name: String) -> PipelineConfiguration {
        var copy = self
        copy.baseModel = name
        return copy
    }
}
