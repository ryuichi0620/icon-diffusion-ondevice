#if canImport(CoreAI)
import CoreAI
#endif
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

            HStack(spacing: 12) {
                Button(isBusy ? "生成中…" : "生成") {
                    Task { await generate() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isBusy || subject.isEmpty)

                Button("モデルを準備") {
                    Task { await prewarm() }
                }
                .buttonStyle(.bordered)
                .disabled(isBusy || didPrewarm)
            }

            Text(status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            Spacer()
        }
        .padding()
        // **起動直後に prewarm しない。** 3 モデルを一気にコンパイルすると
        // ピークメモリで std::bad_alloc に落ちる（実機で踏んだ）。
        // ユーザーが明示的に押したときだけ走らせる。
        //
        // ただし環境変数があれば自動で走る。実機を手で操作せずに検証するため:
        //
        //   DEVICECTL_CHILD_ICON_AUTO=generate \
        //     xcrun devicectl device process launch --console --terminate-existing \
        //     --device <id> com.egoshi.icondiffusion
        //
        // 起動引数（--auto-prewarm）は devicectl 自身のオプションとして食われるので使えない。
        // devicectl は DEVICECTL_CHILD_ 接頭辞の環境変数をアプリに渡してくれる。
        .task {
            let mode = ProcessInfo.processInfo.environment["ICON_AUTO"] ?? ""
            log("起動しました（ICON_AUTO=\(mode.isEmpty ? "なし" : mode)）")
            guard !didPrewarm, mode == "prewarm" || mode == "generate" else { return }
            await prewarm()
            if mode == "generate" {
                await generate()
            }
            log("自動実行を終えました")
        }
    }

    private func prewarm() async {
        #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
        isBusy = true
        defer { isBusy = false }
        do {
            guard let root = Bundle.main.resourceURL?.appendingPathComponent("Models") else { return }
            log("モデルを準備中…")
            let started = Date()
            log("使える計算ユニット: \(CoreAIBackend.availableComputeUnits)")
            // 切り分け用。ICON_UNIT=default で ANE 優先を外す、ICON_UNIT=cpu で CPU のみ。
            // ICON_ONLY=UNet のように 1 モデルだけ測ることもできる。
            let env = ProcessInfo.processInfo.environment
            let unit: ComputeUnitKind? = switch env["ICON_UNIT"] {
            case "default": nil
            case "cpu": .cpu
            case "gpu": .gpu
            default: .neuralEngine
            }
            let only = env["ICON_ONLY"]
            log("計算ユニット=\(unit.map { String(describing: $0) } ?? "既定") 対象=\(only ?? "全部")")
            try await CoreAIBackend.prewarm(
                resources: PipelineResources.directory(root),
                preferredComputeUnit: unit,
                only: only,
                onProgress: { done, total in
                    Task { @MainActor in log("モデルを準備中… \(done)/\(total)") }
                }
            )
            didPrewarm = true
            log(String(format: "準備完了 %.1f 秒", Date().timeIntervalSince(started)))
        } catch {
            log("準備に失敗: \(error.localizedDescription)")
        }
        #else
        log("Core AI が使えません（シミュレータでは動きません）")
        #endif
    }

    /// 画面と標準出力の両方に出す。実機を手で見られないときは --console で拾う。
    private func log(_ message: String) {
        status = message
        print("[app] \(message)")
    }

    private func generate() async {
        isBusy = true
        defer { isBusy = false }
        do {
            let pipeline = try await preparedPipeline()
            log("生成中…")
            let options = IconDiffusionPipeline.Options(
                steps: 4,
                guidanceScale: 1.0,   // 端末では CFG なし。UNet の実行回数が半分になる
                seed: UInt64.random(in: 0..<UInt64.max),
                onProgress: { done, total in
                    Task { @MainActor in log("ステップ \(done)/\(total)") }
                }
            )
            let started = Date()
            let generated = try await pipeline.generateIcon(subject: subject, options: options)
            image = generated.makeCGImage()
            log(String(format: "完了 %.1f 秒", Date().timeIntervalSince(started)))
        } catch {
            log("失敗: \(error.localizedDescription)")
        }
    }

    private func preparedPipeline() async throws -> IconDiffusionPipeline {
        if let pipeline { return pipeline }
        // Core AI はシミュレータの SDK に入っていない（実機専用）。
        // ここを #if で切らないと、シミュレータ向けビルドがそもそも通らない。
        #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
        log("モデルを読み込み中…")
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
