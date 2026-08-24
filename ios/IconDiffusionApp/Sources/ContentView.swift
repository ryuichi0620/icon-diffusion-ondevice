#if canImport(CoreAI)
import CoreAI
#endif
import IconDiffusionKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

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
            #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
            if mode == "selftest" || mode == "generate" {
                do {
                    let result = try CoreAIBackend.selfTestMarshalling()
                    log("受け渡しの自己テスト: 最大相対誤差 \(result.maxError) / \(result.count) 件")
                } catch {
                    log("受け渡しの自己テストに失敗: \(error.localizedDescription)")
                }
            }
            if mode == "selftest" { log("自動実行を終えました"); return }
            #endif
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
            let plan = Self.computeUnitPlan()
            let only = env["ICON_ONLY"]
            log("計算ユニット: \(plan.describe(["TextEncoder", "UNet", "VAEDecoder"])) "
                + "対象=\(only ?? "全部")")
            try await CoreAIBackend.prewarm(
                resources: PipelineResources.directory(root),
                computeUnits: plan,
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
    /// ANE を既定にしつつ、ICON_UNIT で切り替えられるようにする。
    ///
    /// **prewarm と生成の両方で同じ値を使うこと。** 以前は生成側が .neuralEngine 決め打ちで、
    /// ICON_UNIT を変えても生成の計算ユニットは変わらなかった。切り分けの前提が崩れる。
    private static func unit(_ name: String?) -> ComputeUnitKind? {
        switch name {
        case "default": return nil
        case "cpu": return .cpu
        case "gpu": return .gpu
        default: return .neuralEngine
        }
    }

    /// 既定は ComputeUnitPlan.recommended（UNet=ANE / VAEDecoder=GPU）。
    /// ICON_UNIT で全体を、ICON_UNIT_VAEDecoder のようにモデル単位で上書きできる。
    ///
    /// **prewarm と生成で同じプランを使うこと。** 違う指定で温めるとキャッシュが当たらず、
    /// 生成時にもう一度コンパイルが走って数分待たされる。
    static func computeUnitPlan() -> CoreAIBackend.ComputeUnitPlan {
        let env = ProcessInfo.processInfo.environment
        var plan = CoreAIBackend.ComputeUnitPlan.recommended
        if let raw = env["ICON_UNIT"] {
            // 全体指定は「既定の上書きを捨てて全部そのユニット」の意味にする。
            // 切り分けのための指定なので、一部だけ既定が残っていると紛らわしい。
            plan = CoreAIBackend.ComputeUnitPlan(fallback: unit(raw))
        }
        for model in ["TextEncoder", "UNet", "VAEDecoder"] {
            if let raw = env["ICON_UNIT_\(model)"] {
                plan.overrides[model] = unit(raw)
            }
        }
        return plan
    }

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
            if ProcessInfo.processInfo.environment["ICON_DEBUG"] == "1" {
                let bytes = generated.rgba
                let low = bytes.min() ?? 0, high = bytes.max() ?? 0
                let mean = bytes.reduce(0) { $0 + Int($1) } / max(bytes.count, 1)
                log("最終画像 RGBA: min=\(low) max=\(high) mean=\(mean)")
            }
            image = generated.makeCGImage()
            log(String(format: "完了 %.1f 秒", Date().timeIntervalSince(started)))
            // 画面を見られない状況（実機を手元で確認できないとき）のために書き出す。
            //   xcrun devicectl device copy from --device <id> \
            //     --domain-type appDataContainer --domain-identifier com.egoshi.icondiffusion \
            //     --source Documents/latest.png --destination ./latest.png
            saveForInspection(generated)
        } catch {
            log("失敗: \(error.localizedDescription)")
        }
    }

    /// 生成結果を Documents に PNG で置く。devicectl で取り出して目視確認するため。
    private func saveForInspection(_ generated: GeneratedImage) {
        guard let cgImage = generated.makeCGImage(),
              let documents = FileManager.default.urls(for: .documentDirectory,
                                                       in: .userDomainMask).first else {
            log("書き出し先が取れませんでした")
            return
        }
        let url = documents.appendingPathComponent("latest.png")
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            log("PNG を作れませんでした")
            return
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        if CGImageDestinationFinalize(destination) {
            log("書き出しました: Documents/latest.png")
        } else {
            log("PNG の書き出しに失敗しました")
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

        let backend = try await CoreAIBackend(
            resources: resources,
            configuration: configuration,
            computeUnits: Self.computeUnitPlan()
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
