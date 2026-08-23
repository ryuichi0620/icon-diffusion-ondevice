import IconDiffusionKit
import SwiftUI

/// 被写体を入れて 4 ステップで 1 枚生成するだけの最小画面。
///
/// **モデルの読み込みは重い**（コンパイルが走る）ので、生成のたびにやらず
/// 一度だけ作って使い回す。端末で毎回コンパイルさせないためには
/// `AIModel.specialize` のキャッシュを使う（docs/architecture.md 6.6 節）。
struct ContentView: View {
    @State private var subject = "a coffee cup"
    @State private var image: CGImage?
    @State private var status = ""
    @State private var isBusy = false
    @State private var pipeline: IconDiffusionPipeline?
    @State private var didPrewarm = false

    var body: some View {
        VStack(spacing: 16) {
            Group {
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .aspectRatio(1, contentMode: .fit)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.quaternary)
                        .aspectRatio(1, contentMode: .fit)
                        .overlay(Text("まだ生成していません").foregroundStyle(.secondary))
                }
            }
            .frame(maxWidth: 360)

            TextField("被写体", text: $subject)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()

            Button(isBusy ? "生成中…" : "生成") {
                Task { await generate() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isBusy || subject.isEmpty)

            Text(status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            Spacer()
        }
        .padding()
        // モデルのコンパイルは重い。生成ボタンを押してから待たせず、
        // 画面が出た時点で済ませておく。2 回目以降はキャッシュから読むだけ。
        .task {
            guard !didPrewarm else { return }
            didPrewarm = true
            await prewarm()
        }
    }

    private func prewarm() async {
        #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
        do {
            guard let root = Bundle.main.resourceURL?.appendingPathComponent("Models") else { return }
            status = "モデルを準備中…"
            let started = Date()
            try await CoreAIBackend.prewarm(
                resources: PipelineResources.directory(root),
                preferredComputeUnit: .neuralEngine,
                onProgress: { done, total in
                    Task { @MainActor in status = "モデルを準備中… \(done)/\(total)" }
                }
            )
            status = String(format: "準備完了 %.1f 秒", Date().timeIntervalSince(started))
        } catch {
            status = "準備に失敗: \(error.localizedDescription)"
        }
        #else
        status = "Core AI が使えません（シミュレータでは動きません）"
        #endif
    }

    private func generate() async {
        isBusy = true
        defer { isBusy = false }
        do {
            let pipeline = try await preparedPipeline()
            status = "生成中…"
            let options = IconDiffusionPipeline.Options(
                steps: 4,
                guidanceScale: 1.0,   // 端末では CFG なし。UNet の実行回数が半分になる
                seed: UInt64.random(in: 0..<UInt64.max),
                onProgress: { done, total in
                    Task { @MainActor in status = "ステップ \(done)/\(total)" }
                }
            )
            let started = Date()
            let generated = try await pipeline.generateIcon(subject: subject, options: options)
            image = generated.makeCGImage()
            status = String(format: "完了 %.1f 秒", Date().timeIntervalSince(started))
        } catch {
            status = "失敗: \(error.localizedDescription)"
        }
    }

    private func preparedPipeline() async throws -> IconDiffusionPipeline {
        if let pipeline { return pipeline }
        // Core AI はシミュレータの SDK に入っていない（実機専用）。
        // ここを #if で切らないと、シミュレータ向けビルドがそもそも通らない。
        #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
        status = "モデルを読み込み中…"
        // .aimodel はディレクトリなので、バンドルにはフォルダ参照で入れている。
        // そのため直下ではなく Resources/ の下にある。bundled(in:) は
        // Bundle.url(forResource:) を使うのでサブフォルダを見てくれない。
        guard let root = Bundle.main.resourceURL?.appendingPathComponent("Models") else {
            throw IconDiffusionError.unsupportedDevice("バンドルのリソースが見つかりません")
        }
        let resources = PipelineResources.directory(root)
        try resources.validateModels()
        let configuration = try PipelineConfiguration.load(contentsOf: resources.configurationURL)
        // ANE を優先する。使えなければ Core AI 側が落としてくれる
        let backend = try await CoreAIBackend(
            resources: resources,
            configuration: configuration,
            preferredComputeUnit: .neuralEngine
        )
        let built = try IconDiffusionPipeline(resources: resources, backend: backend)
        pipeline = built
        return built
        #else
        throw IconDiffusionError.unsupportedDevice(
            "Core AI が使えません。シミュレータでは動きません（実機専用）。"
        )
        #endif
    }
}
