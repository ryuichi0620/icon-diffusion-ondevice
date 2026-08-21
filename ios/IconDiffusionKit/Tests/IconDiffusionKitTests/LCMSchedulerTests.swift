import Foundation
import Testing

@testable import IconDiffusionKit

@Suite("LCM スケジューラ")
struct LCMSchedulerTests {
    private func makeConfiguration(_ fixture: SchedulerFixture) -> LCMScheduler.Configuration {
        let values = fixture.schedulerConfiguration
        return LCMScheduler.Configuration(
            numTrainTimesteps: values.numTrainTimesteps,
            betaStart: values.betaStart,
            betaEnd: values.betaEnd,
            betaSchedule: "scaled_linear",
            originalInferenceSteps: values.originalInferenceSteps,
            timestepScaling: values.timestepScaling,
            sigmaData: values.sigmaData
        )
    }

    @Test("タイムステップの並びが diffusers と一致する（4 ステップなら 999/759/519/279）")
    func timestepsMatchReference() throws {
        let fixture = try Fixtures.decode(SchedulerFixture.self, from: "expected_scheduler.json")
        let configuration = makeConfiguration(fixture)

        for (key, expected) in fixture.timesteps {
            let steps = try #require(Int(key))
            let scheduler = try LCMScheduler(configuration: configuration, inferenceSteps: steps)
            #expect(scheduler.timesteps == expected, "\(steps) ステップの並びが違います")
        }
    }

    @Test("ᾱ が β スケジュールから正しく組み立てられている")
    func alphasCumprodMatchReference() throws {
        let fixture = try Fixtures.decode(SchedulerFixture.self, from: "expected_scheduler.json")
        let scheduler = try LCMScheduler(configuration: makeConfiguration(fixture), inferenceSteps: 4)

        for (key, expected) in fixture.alphasCumprodAt {
            let timestep = try #require(Int(key))
            // ここは Swift 側も Double で計算しているので、ほぼ完全一致するはず
            #expect(
                isClose(scheduler.alphaCumprod(at: timestep), expected, relative: 1e-12, absolute: 1e-15),
                "ᾱ[\(timestep)] = \(scheduler.alphaCumprod(at: timestep)), 期待 \(expected)"
            )
        }
    }

    @Test("境界条件のスケーリング c_skip / c_out が一致する")
    func scalingsMatchReference() throws {
        let fixture = try Fixtures.decode(SchedulerFixture.self, from: "expected_scheduler.json")
        let scheduler = try LCMScheduler(configuration: makeConfiguration(fixture), inferenceSteps: 4)

        for (key, expected) in fixture.scalingsAt {
            let timestep = try #require(Int(key))
            let (cSkip, cOut) = scheduler.scalings(forTimestep: timestep)
            #expect(isClose(cSkip, expected.cSkip, relative: 1e-12), "c_skip[\(timestep)]")
            #expect(isClose(cOut, expected.cOut, relative: 1e-12), "c_out[\(timestep)]")
        }
    }

    @Test("1 ステップの計算が参照実装と一致する（各ステップ独立）")
    func singleStepMatchesReference() throws {
        let fixture = try Fixtures.decode(SchedulerFixture.self, from: "expected_scheduler.json")
        let scheduler = try LCMScheduler(configuration: makeConfiguration(fixture), inferenceSteps: 4)
        let shape = [1, 1, 2, 2]

        let sample = Tensor(shape: shape, values: fixture.stepCase.sample.map(Float.init))
        let modelOutput = Tensor(shape: shape, values: fixture.stepCase.modelOutput.map(Float.init))
        let noise = Tensor(shape: shape, values: fixture.stepCase.noise.map(Float.init))

        for expected in fixture.stepCase.steps {
            let result = try scheduler.step(
                modelOutput: modelOutput,
                sample: sample,
                stepIndex: expected.index,
                noise: noise
            )
            expectClose(result.denoised.values, expected.denoised, "step \(expected.index) denoised")
            expectClose(result.previousSample.values, expected.previousSample, "step \(expected.index) prev")
        }
    }

    @Test("4 ステップ通した最終潜在が一致する（累積誤差込み）")
    func fullLoopMatchesReference() throws {
        let fixture = try Fixtures.decode(SchedulerFixture.self, from: "expected_scheduler.json")
        let scheduler = try LCMScheduler(configuration: makeConfiguration(fixture), inferenceSteps: 4)
        let shape = [1, 1, 2, 2]

        let modelOutput = Tensor(shape: shape, values: fixture.stepCase.modelOutput.map(Float.init))
        let noise = Tensor(shape: shape, values: fixture.stepCase.noise.map(Float.init))
        var latents = Tensor(shape: shape, values: fixture.stepCase.sample.map(Float.init))

        for index in 0..<scheduler.timesteps.count {
            latents = try scheduler.step(
                modelOutput: modelOutput, sample: latents, stepIndex: index, noise: noise
            ).previousSample
        }
        // 4 ステップぶんの Float 演算が積み上がるので、単発より許容を緩める
        expectClose(latents.values, fixture.fullLoopFinal, relative: 1e-3, "full loop")
    }

    @Test("最終ステップではノイズを注ぎ直さない")
    func finalStepDoesNotRenoise() throws {
        let scheduler = try LCMScheduler(inferenceSteps: 4)
        let shape = [1, 1, 2, 2]
        let sample = Tensor(shape: shape, values: [0.5, -0.25, 1.0, 0.0])
        let modelOutput = Tensor(shape: shape, values: [0.1, 0.2, -0.3, 0.05])

        // 最終ステップは noise を無視するはずなので、まったく違うノイズでも結果が変わらない
        let quiet = try scheduler.step(
            modelOutput: modelOutput, sample: sample, stepIndex: 3, noise: Tensor.zeros(shape: shape)
        )
        let loud = try scheduler.step(
            modelOutput: modelOutput,
            sample: sample,
            stepIndex: 3,
            noise: Tensor(shape: shape, values: [99, -99, 99, -99])
        )
        #expect(quiet.previousSample == loud.previousSample)
        #expect(quiet.previousSample == quiet.denoised, "最終ステップは denoised をそのまま返す")
    }

    @Test("中間ステップではノイズが効く（LCM の肝）")
    func intermediateStepUsesNoise() throws {
        let scheduler = try LCMScheduler(inferenceSteps: 4)
        let shape = [1, 1, 2, 2]
        let sample = Tensor(shape: shape, values: [0.5, -0.25, 1.0, 0.0])
        let modelOutput = Tensor(shape: shape, values: [0.1, 0.2, -0.3, 0.05])

        let quiet = try scheduler.step(
            modelOutput: modelOutput, sample: sample, stepIndex: 0, noise: Tensor.zeros(shape: shape)
        )
        let loud = try scheduler.step(
            modelOutput: modelOutput,
            sample: sample,
            stepIndex: 0,
            noise: Tensor(shape: shape, values: [1, 1, 1, 1])
        )
        #expect(quiet.previousSample != loud.previousSample, "中間ステップでノイズが無視されています")
        #expect(quiet.denoised == loud.denoised, "denoised はノイズに依存しないはず")
    }

    @Test("設定が不正なら初期化で弾く", arguments: [0, -1, 51])
    func rejectsInvalidStepCounts(steps: Int) {
        #expect(throws: IconDiffusionError.self) {
            _ = try LCMScheduler(inferenceSteps: steps)
        }
    }

    @Test("未対応の beta_schedule は弾く")
    func rejectsUnknownBetaSchedule() {
        let configuration = LCMScheduler.Configuration(betaSchedule: "cosine")
        #expect(throws: IconDiffusionError.self) {
            _ = try LCMScheduler(configuration: configuration, inferenceSteps: 4)
        }
    }

    @Test("形が合わないテンソルは弾く")
    func rejectsShapeMismatch() throws {
        let scheduler = try LCMScheduler(inferenceSteps: 4)
        let sample = Tensor.zeros(shape: [1, 4, 8, 8])
        let wrong = Tensor.zeros(shape: [1, 4, 4, 4])
        #expect(throws: IconDiffusionError.self) {
            _ = try scheduler.step(modelOutput: wrong, sample: sample, stepIndex: 0, noise: sample)
        }
        #expect(throws: IconDiffusionError.self) {
            _ = try scheduler.step(modelOutput: sample, sample: sample, stepIndex: 0, noise: wrong)
        }
    }
}
