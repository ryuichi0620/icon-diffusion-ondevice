import Foundation
import Testing

@testable import IconDiffusionKit

@Suite("パイプラインの配線")
struct PipelineTests {
    /// テスト用の小さい設定。モデルは使わないので、形さえ整合していればよい。
    /// 512x512 のままだと 1 テストで 78 万要素を回すことになるので落としてある。
    static let smallConfiguration = PipelineConfiguration(
        baseModel: "test",
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
        latentSize: 8,
        imageSize: 64,
        textEmbedDim: 16,
        maxSequenceLength: 8,
        bosTokenID: 542,
        eosTokenID: 543
    )

    private func makePipeline(
        configuration: PipelineConfiguration = PipelineTests.smallConfiguration,
        backend: (any DiffusionBackend)? = nil
    ) throws -> IconDiffusionPipeline {
        let tokenizer = try CLIPTokenizer(
            vocabularyURL: try Fixtures.url("vocab.json"),
            mergesURL: try Fixtures.url("merges.txt"),
            maxLength: configuration.maxSequenceLength
        )
        let resolvedBackend: any DiffusionBackend =
            backend ?? StubBackend(configuration: configuration)
        return IconDiffusionPipeline(
            backend: resolvedBackend,
            tokenizer: tokenizer,
            configuration: configuration
        )
    }

    @Test("4 ステップ通して所定のサイズの画像が出る")
    func generatesImageOfExpectedSize() async throws {
        let pipeline = try makePipeline()
        let image = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4))

        #expect(image.width == Self.smallConfiguration.imageSize)
        #expect(image.height == Self.smallConfiguration.imageSize)
        #expect(image.rgba.count == image.width * image.height * 4)
        #expect(image.rgba.enumerated().allSatisfy({ $0.offset % 4 != 3 || $0.element == 255 }), "アルファは 255")
    }

    @Test("同じ seed なら同じ結果、違う seed なら違う結果")
    func generationIsReproducibleBySeed() async throws {
        let pipeline = try makePipeline()
        let first = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4, seed: 42))
        let same = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4, seed: 42))
        let other = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4, seed: 43))

        #expect(first == same, "同じ seed で結果が変わっています")
        #expect(first != other, "seed を変えても結果が変わっていません")
    }

    @Test("プロンプトが違えば結果も違う（埋め込みが実際に効いている）")
    func promptAffectsOutput() async throws {
        let pipeline = try makePipeline()
        let options = IconDiffusionPipeline.Options(steps: 4, seed: 7)
        let cup = try await pipeline.generate(prompt: "a coffee cup", options: options)
        let key = try await pipeline.generate(prompt: "a house key", options: options)
        #expect(cup != key)
    }

    @Test("generateIcon はテンプレートに被写体を埋めて生成する")
    func generateIconUsesPromptTemplate() async throws {
        let configuration = Self.smallConfiguration
        #expect(
            configuration.prompt(subject: "a coffee cup")
                == "an icon in ICONST style, a coffee cup, flat, minimal"
        )

        let pipeline = try makePipeline()
        let options = IconDiffusionPipeline.Options(steps: 4, seed: 1)
        let viaTemplate = try await pipeline.generateIcon(subject: "a coffee cup", options: options)
        let viaRaw = try await pipeline.generate(
            prompt: configuration.prompt(subject: "a coffee cup"), options: options
        )
        #expect(viaTemplate == viaRaw)
    }

    @Test("CFG の有無で UNet の実行回数が変わる")
    func guidanceScaleControlsUNetCallCount() async throws {
        let counter = CallCounter()
        let backend = CountingBackend(
            wrapped: StubBackend(configuration: Self.smallConfiguration),
            counter: counter
        )
        let pipeline = try makePipeline(backend: backend)

        _ = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4, guidanceScale: 1.0))
        let withoutGuidance = await counter.snapshot()
        #expect(withoutGuidance.unet == 4, "CFG 無しなら UNet はステップ数と同じ回数")
        #expect(withoutGuidance.textEncoder == 1, "テキストエンコーダは 1 回だけ")

        await counter.reset()
        _ = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4, guidanceScale: 1.5))
        let withGuidance = await counter.snapshot()
        #expect(withGuidance.unet == 8, "CFG 有りなら UNet は 2 倍")
        #expect(withGuidance.textEncoder == 2, "条件つきと無条件で 2 回")
        #expect(withGuidance.vaeDecoder == 1)
    }

    @Test("ステップ数を変えると UNet の実行回数も変わる", arguments: [1, 2, 4, 8])
    func stepCountControlsLoopLength(steps: Int) async throws {
        let counter = CallCounter()
        let backend = CountingBackend(
            wrapped: StubBackend(configuration: Self.smallConfiguration),
            counter: counter
        )
        let pipeline = try makePipeline(backend: backend)

        _ = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: steps))
        #expect(await counter.snapshot().unet == steps)
    }

    @Test("進捗コールバックが全ステップぶん呼ばれる")
    func progressCallbackFires() async throws {
        let recorder = ProgressRecorder()
        let pipeline = try makePipeline()
        _ = try await pipeline.generate(
            prompt: "a coffee cup",
            options: .init(steps: 4, onProgress: { completed, total in
                recorder.record(completed: completed, total: total)
            })
        )
        #expect(recorder.completedSteps == [1, 2, 3, 4])
        #expect(recorder.totals == [4, 4, 4, 4])
    }

    @Test("バックエンドが形の違うテンソルを返したら弾く")
    func rejectsBackendShapeMismatch() async throws {
        let pipeline = try makePipeline(backend: BrokenBackend(configuration: Self.smallConfiguration))
        await #expect(throws: IconDiffusionError.self) {
            _ = try await pipeline.generate(prompt: "a coffee cup", options: .init(steps: 4))
        }
    }
}

// MARK: - テスト用のバックエンド

/// 呼び出し回数を数えるだけのラッパー。CFG やステップ数の効きを外から確認する。
private actor CallCounter {
    struct Counts: Sendable {
        var textEncoder = 0
        var unet = 0
        var vaeDecoder = 0
    }

    private var counts = Counts()

    func countTextEncoder() { counts.textEncoder += 1 }
    func countUNet() { counts.unet += 1 }
    func countVAEDecoder() { counts.vaeDecoder += 1 }
    func snapshot() -> Counts { counts }
    func reset() { counts = Counts() }
}

private struct CountingBackend: DiffusionBackend {
    let wrapped: StubBackend
    let counter: CallCounter

    func encodeText(inputIDs: [Int32]) async throws -> Tensor {
        await counter.countTextEncoder()
        return try await wrapped.encodeText(inputIDs: inputIDs)
    }

    func predictNoise(
        sample: Tensor, timestep: Float, encoderHiddenStates: Tensor
    ) async throws -> Tensor {
        await counter.countUNet()
        return try await wrapped.predictNoise(
            sample: sample, timestep: timestep, encoderHiddenStates: encoderHiddenStates
        )
    }

    func decodeLatent(_ latent: Tensor) async throws -> Tensor {
        await counter.countVAEDecoder()
        return try await wrapped.decodeLatent(latent)
    }
}

/// わざと形の違うテンソルを返す。設定とモデルが食い違ったときに黙って進まないことを確かめる。
private struct BrokenBackend: DiffusionBackend {
    let configuration: PipelineConfiguration

    func encodeText(inputIDs: [Int32]) async throws -> Tensor {
        Tensor.zeros(shape: [1, configuration.maxSequenceLength, configuration.textEmbedDim + 1])
    }

    func predictNoise(
        sample: Tensor, timestep: Float, encoderHiddenStates: Tensor
    ) async throws -> Tensor {
        sample
    }

    func decodeLatent(_ latent: Tensor) async throws -> Tensor {
        Tensor.zeros(shape: configuration.decodedImageShape)
    }
}

/// 進捗コールバックの記録。コールバックは同期的に呼ばれるので、直列アクセスで足りる。
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var completed: [Int] = []
    private var totalValues: [Int] = []

    func record(completed step: Int, total: Int) {
        lock.lock()
        defer { lock.unlock() }
        completed.append(step)
        totalValues.append(total)
    }

    var completedSteps: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return completed
    }

    var totals: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return totalValues
    }
}
