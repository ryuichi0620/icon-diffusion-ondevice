#if canImport(CoreAI)
import CoreAI
#endif
import IconDiffusionKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// 被写体から**候補を 4 枚**生成し、後処理（透過 + パレット量子化）と機械判定を掛けて
/// 選ばせる画面。アバター側で実証した型の移植。
///
/// 1 枚だけ生成して祈る作りにしないのは歩留まりのため（透過成功率 74.5%。
/// 4 候補なら当たりが入る確率は約 99.6%）。機械判定は Mac 側の歩留まり計測と
/// 同一の定義（IconPostprocess = postprocess.py の移植）なので、数字がそのまま繋がる。
struct ContentView: View {
    struct Candidate: Identifiable {
        let id: Int
        let raw: GeneratedImage
        let processed: GeneratedImage
        let cgImage: CGImage
        let transparency: Double
        let seed: UInt64
        /// 合格 = 背景が抜けて（>=50%）、かつ絵が残っている（<=98%）
        var ok: Bool { transparency >= 0.5 && transparency <= 0.98 }
    }

    @State private var subject = "a coffee cup"
    @State private var candidates: [Candidate] = []
    @State private var selectedID: Int?
    @State private var status = ""
    @State private var isBusy = false
    @State private var pipeline: IconDiffusionPipeline?
    @State private var didPrewarm = false

    private let candidateCount = 4

    var body: some View {
        VStack(spacing: 14) {
            selectedView
            candidateGrid

            TextField("被写体（英語）", text: $subject)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()

            HStack(spacing: 12) {
                Button(isBusy ? "生成中…" : "候補を \(candidateCount) 枚生成") {
                    Task { await generateCandidates() }
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

            Spacer(minLength: 0)
        }
        .padding()
        // 起動直後に prewarm しない（初回コンパイルのピークメモリ対策）。
        // DEVICECTL_CHILD_ICON_AUTO=generate で無人実行。
        .task {
            let mode = ProcessInfo.processInfo.environment["ICON_AUTO"] ?? ""
            log("起動しました（ICON_AUTO=\(mode.isEmpty ? "なし" : mode)）")
            #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
            if mode == "selftest" {
                do {
                    let result = try CoreAIBackend.selfTestMarshalling()
                    log("受け渡しの自己テスト: 最大相対誤差 \(result.maxError) / \(result.count) 件")
                } catch {
                    log("受け渡しの自己テストに失敗: \(error.localizedDescription)")
                }
                log("自動実行を終えました")
                return
            }
            #endif
            guard !didPrewarm, mode == "prewarm" || mode == "generate" else { return }
            if let override = ProcessInfo.processInfo.environment["ICON_SUBJECT"] {
                subject = override
            }
            await prewarm()
            if mode == "generate" {
                await generateCandidates()
            }
            log("自動実行を終えました")
        }
    }

    @ViewBuilder private var selectedView: some View {
        Group {
            if let selected = candidates.first(where: { $0.id == selectedID }) {
                // 透過が見えるように市松の上へ載せる（Finder のプレビューと同じ流儀）
                Image(decorative: selected.cgImage, scale: 1)
                    .resizable()
                    .aspectRatio(1, contentMode: .fit)
                    .background(CheckerboardBackground())
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(.quaternary)
                    .aspectRatio(1, contentMode: .fit)
                    .overlay(
                        Text(candidates.isEmpty ? "まだ生成していません" : "候補をタップして選ぶ")
                            .foregroundStyle(.secondary)
                    )
            }
        }
        .frame(maxWidth: 300)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder private var candidateGrid: some View {
        if !candidates.isEmpty {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 4),
                      spacing: 8) {
                ForEach(candidates) { candidate in
                    candidateCell(candidate)
                }
            }
        }
    }

    // 緑 = 機械判定を通過（背景が抜けた）、赤 = 落ちた。落ちたものも見せて選べる。
    // 判定は背景の一様性しか見ておらず、被写体の正しさは人の目のほうが偉い
    private func candidateCell(_ candidate: Candidate) -> some View {
        let colour: Color = candidate.ok ? .green : .red
        let width: CGFloat = selectedID == candidate.id ? 4 : 2
        return Image(decorative: candidate.cgImage, scale: 1)
            .resizable()
            .aspectRatio(1, contentMode: .fit)
            .background(Color.white)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(colour, lineWidth: width))
            .onTapGesture { select(candidate) }
    }

    private func select(_ candidate: Candidate) {
        selectedID = candidate.id
        log(String(format: "候補 %d を選択（透過 %.0f%%・seed %llu）",
                   candidate.id + 1, candidate.transparency * 100, candidate.seed))
        saveForInspection(candidate.processed, name: "latest.png")
    }

    private static func unit(_ name: String?) -> ComputeUnitKind? {
        switch name {
        case "default": return nil
        case "cpu": return .cpu
        case "gpu": return .gpu
        default: return .neuralEngine
        }
    }

    /// 既定は ComputeUnitPlan.recommended（UNet=ANE / VAEDecoder=GPU）。
    static func computeUnitPlan() -> CoreAIBackend.ComputeUnitPlan {
        let env = ProcessInfo.processInfo.environment
        var plan = CoreAIBackend.ComputeUnitPlan.recommended
        if let raw = env["ICON_UNIT"] {
            plan = CoreAIBackend.ComputeUnitPlan(fallback: unit(raw))
        }
        for model in ["TextEncoder", "UNet", "VAEDecoder"] {
            if let raw = env["ICON_UNIT_\(model)"] {
                plan.overrides[model] = unit(raw)
            }
        }
        return plan
    }

    private func prewarm() async {
        #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
        isBusy = true
        defer { isBusy = false }
        do {
            guard let root = Bundle.main.resourceURL?.appendingPathComponent("Models") else { return }
            log("モデルを準備中…")
            let started = Date()
            let plan = Self.computeUnitPlan()
            log("計算ユニット: \(plan.describe(["TextEncoder", "UNet", "VAEDecoder"]))")
            try await CoreAIBackend.prewarm(
                resources: PipelineResources.directory(root),
                computeUnits: plan,
                only: ProcessInfo.processInfo.environment["ICON_ONLY"],
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

    private func log(_ message: String) {
        status = message
        print("[app] \(message)")
    }

    private func generateCandidates() async {
        isBusy = true
        defer { isBusy = false }
        candidates = []
        selectedID = nil
        do {
            let pipeline = try await preparedPipeline()
            let env = ProcessInfo.processInfo.environment
            // 生成条件は Mac 側スイープで確定した値を既定にし、実験用に環境変数で上書き可
            let steps = Int(env["ICON_STEPS"] ?? "") ?? 4
            let guidance = Float(env["ICON_GUIDANCE"] ?? "") ?? 1.0
            log("生成中…（\(steps) ステップ / guidance \(guidance)）")

            let started = Date()
            for index in 0..<candidateCount {
                log("候補 \(index + 1)/\(candidateCount) を生成中…")
                let seed = UInt64.random(in: 0..<UInt64.max)
                let options = IconDiffusionPipeline.Options(
                    steps: steps,
                    guidanceScale: guidance,
                    seed: seed
                )
                let generated = try await pipeline.generateIcon(subject: subject, options: options)
                let result = IconPostprocess.process(generated)
                guard let cgImage = result.image.makeAlphaCGImage() else { continue }
                let candidate = Candidate(id: index, raw: generated, processed: result.image,
                                          cgImage: cgImage, transparency: result.transparency,
                                          seed: seed)
                candidates.append(candidate)
                log(String(format: "候補 %d: %@（透過 %.0f%%）", index + 1,
                           candidate.ok ? "合格" : "不合格", result.transparency * 100))
                saveForInspection(result.image, name: "candidate\(index + 1).png")
            }

            let passed = candidates.filter(\.ok)
            log(String(format: "完了 %.1f 秒。機械判定 %d/%d 通過",
                       Date().timeIntervalSince(started), passed.count, candidates.count))
            if ProcessInfo.processInfo.environment["ICON_AUTO"] == "generate" {
                // 無人実行では人が選べない。**透過率最大を選んではいけない**（ほぼ空の
                // 絵が勝ってしまう。実機で確認済み）。学習データの被写体占有率は約 70% =
                // 透過率 0.55〜0.75 相当なので、0.7 に最も近いものを採用する
                let target = 0.7
                let pool = passed.isEmpty ? candidates : passed
                let choice = pool.min(by: {
                    abs($0.transparency - target) < abs($1.transparency - target)
                })
                if let choice { select(choice) }
            }
        } catch {
            log("失敗: \(error.localizedDescription)")
        }
    }

    /// 後処理済み（透過 RGBA）を Documents に置く。devicectl で取り出して確認するため。
    private func saveForInspection(_ image: GeneratedImage, name: String) {
        guard let cgImage = image.makeAlphaCGImage(),
              let documents = FileManager.default.urls(for: .documentDirectory,
                                                       in: .userDomainMask).first else { return }
        let url = documents.appendingPathComponent(name)
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { return }
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
    }

    private func preparedPipeline() async throws -> IconDiffusionPipeline {
        if let pipeline { return pipeline }
        #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
        log("モデルを読み込み中…")
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

/// 透過確認用の市松背景
struct CheckerboardBackground: View {
    var body: some View {
        Canvas { context, size in
            let cell: CGFloat = 12
            for row in 0..<Int(size.height / cell) + 1 {
                for column in 0..<Int(size.width / cell) + 1 {
                    if (row + column) % 2 == 0 {
                        context.fill(
                            Path(CGRect(x: CGFloat(column) * cell, y: CGFloat(row) * cell,
                                        width: cell, height: cell)),
                            with: .color(Color(white: 0.9))
                        )
                    }
                }
            }
        }
        .background(Color.white)
    }
}
