#if canImport(CoreAI)
import CoreAI
#endif
import IconDiffusionKit
import ImageIO
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// 被写体から**候補を複数枚**生成し、後処理（透過 → 形の整形 → 滑らかに描き直し）と
/// 機械判定を掛けて選ばせる画面。アバター側で実証した型の移植。
///
/// 1 枚だけ生成して祈る作りにしないのは、**形の当たり外れが大きい**から。整形で消せるのは
/// 中途半端な塗りと輪郭のガタつきまでで、構造の破綻（持ち手が 2 つ、ロケットが矢印になる）は
/// 消せない。そこで候補を出し、`IconShapes.report()` の点で並べ替えて上位を見せる。
/// 採点は Mac 側と同一の式（IconShapes = shapes.py の移植）なので、数字がそのまま繋がる。
struct ContentView: View {
    struct Candidate: Identifiable {
        let id: Int
        let raw: GeneratedImage
        /// 背景を敷く前（透過のみ）。背景を切り替えるたびに拡散を回し直さずに済む
        let transparent: GeneratedImage
        var processed: GeneratedImage
        var cgImage: CGImage
        let transparency: Double
        var report: IconShapes.Report
        let seed: UInt64
        /// 合格 = 形の点が閾値以上。**背景が抜けたかではなく形で判定する**
        /// （透過率だけ見ていた頃は、ほぼ空の絵が「合格」になっていた）
        var ok: Bool { report.score >= IconShapes.passScore }
    }

    @State private var subject = "a coffee cup"
    /// 敷く背景。**SNS のアイコンは円に切り抜かれるので、透過のままだと
    /// タイムラインの地の色に左右される。** auto は輪郭の色に対して WCAG 1.4.11 の
    /// 3:1 を満たすパレット色を選ぶ
    @State private var background = "auto"
    // 描くときの色。**拡散は回さない** — 色はラベル番号から引くだけなので
    // 生成済みの候補に掛け直すだけで一瞬で変わる
    @State private var inkColor = Color(red: 31/255, green: 42/255, blue: 68/255)
    @State private var accentColor = Color(red: 255/255, green: 107/255, blue: 74/255)
    @State private var mono = false
    /// 輪郭を条件に与えるか。**これが「被写体が現実の物に見えない」への答え**
    /// （Mac 側の実測で 58.3% -> 87.5%。docs/experiments.md Trial 19〜21）
    @State private var useControl = true
    @State private var glyphs: GlyphLibrary?
    @State private var resolvedGlyph: String?
    @State private var candidates: [Candidate] = []
    @State private var selectedID: Int?
    @State private var status = ""
    @State private var isBusy = false
    @State private var pipeline: IconDiffusionPipeline?
    @State private var didPrewarm = false

    /// 出力解像度。整形は濃度場を拡大してから二値化するので、512 で作った絵でも
    /// 1024 で輪郭が階段にならない
    private let outputSize = 1024
    private let candidateCount = 4

    var body: some View {
        VStack(spacing: 14) {
            selectedView
            candidateGrid

            TextField("被写体（英語）", text: $subject)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .onChange(of: subject) { _, _ in refreshGlyph() }

            if let glyphs {
                VStack(alignment: .leading, spacing: 4) {
                    Toggle("形を下敷きにする", isOn: $useControl)
                        .font(.footnote)
                    // 表に無い語を打たせないための補完。引けない語は条件なしに落ちる
                    let candidates = glyphs.suggestions(for: subject)
                    if useControl, !candidates.isEmpty, resolvedGlyph == nil {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(candidates, id: \.self) { word in
                                    Button(word) { subject = word }
                                        .font(.caption)
                                        .buttonStyle(.bordered)
                                }
                            }
                        }
                    }
                    Text(controlStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Picker("背景", selection: $background) {
                Text("自動").tag("auto")
                Text("白").tag("white")
                Text("主色").tag("ink")
                Text("透過").tag("none")
            }
            .pickerStyle(.segmented)
            .onChange(of: background) { _, _ in restyleAll() }

            HStack(spacing: 14) {
                ColorPicker("主色", selection: $inkColor, supportsOpacity: false)
                    .labelsHidden()
                if !mono {
                    ColorPicker("副色", selection: $accentColor, supportsOpacity: false)
                        .labelsHidden()
                }
                Toggle("単色", isOn: $mono).fixedSize()
                Spacer()
                if !mono && !theme.isUsable {
                    // 2 色は必ず隣り合うので、明るさが近いと見分けられない
                    Text(String(format: "2 色が近すぎます %.1f:1", theme.pairContrast))
                        .font(.caption).foregroundStyle(.orange)
                } else if !theme.backgroundAchievable {
                    // 9 倍離れていないと、どの地を敷いても片方が 3:1 に届かない
                    Text(String(format: "明暗の開き %.1f倍（地との 3:1 には 9 倍要る）",
                                theme.luminanceSpread))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .onChange(of: inkColor) { _, _ in restyleAll() }
            .onChange(of: accentColor) { _, _ in restyleAll() }
            .onChange(of: mono) { _, _ in restyleAll() }
            .onChange(of: useControl) { _, _ in refreshGlyph() }

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
        // 無人実行は起動引数で指示する:
        //   xcrun devicectl device process launch --device <UDID> --console \
        //     --terminate-existing com.egoshi.icondiffusion \
        //     -ICON_AUTO generate -ICON_SUBJECT "a coffee cup"
        .task {
            if let root = Bundle.main.resourceURL?
                .appendingPathComponent("Models").appendingPathComponent("Glyphs") {
                glyphs = GlyphLibrary(directory: root)
                refreshGlyph()
                log(glyphs == nil ? "グリフの表を読めませんでした"
                                  : "グリフ \(glyphs?.words.count ?? 0) 語を読みました")
            }
            // DEVICECTL_CHILD_ 接頭辞の環境変数は、この端末・この devicectl では届かなかった。
            // 起動引数（-ICON_AUTO generate）なら UserDefaults 経由で確実に読める
            let mode = Self.setting("ICON_AUTO") ?? ""
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
            if let override = Self.setting("ICON_SUBJECT") {
                subject = override
                refreshGlyph()
            }
            await prewarm()
            if mode == "generate" {
                await generateCandidates()
                saveThemeVariants()
            }
            log("自動実行を終えました")
        }
    }

    /// いまの被写体で輪郭が引けるかを、生成前に見せる
    private var controlStatus: String {
        guard useControl else { return "形の下敷きなし（文字だけで作ります）" }
        guard let name = resolvedGlyph else {
            return "この語では形を引けません。文字だけで作ります"
        }
        return "形の下敷き: \(name)"
    }

    private func refreshGlyph() {
        resolvedGlyph = useControl ? glyphs?.resolve(subject) : nil
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
                // 点の高い順に並べる。人が最初に目をやる位置に一番まともな形が来る
                ForEach(candidates.sorted(by: { $0.report.score > $1.report.score })) { candidate in
                    candidateCell(candidate)
                }
            }
        }
    }

    // 緑 = 形の点が閾値以上、赤 = 下回った。落ちたものも見せて選べるようにしてある。
    // 採点は形の素直さ（塗りの量・破片・輪郭の粗さ）しか見ておらず、
    // **それが本当にコーヒーカップに見えるかは人の目のほうが偉い**
    private func candidateCell(_ candidate: Candidate) -> some View {
        let colour: Color = candidate.ok ? .green : .red
        let width: CGFloat = selectedID == candidate.id ? 4 : 2
        return VStack(spacing: 2) {
            Image(decorative: candidate.cgImage, scale: 1)
                .resizable()
                .aspectRatio(1, contentMode: .fit)
                .background(Color.white)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(colour, lineWidth: width))
            Text(String(format: "%.2f", candidate.report.score))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(candidate.ok ? .primary : .secondary)
        }
        .contentShape(Rectangle())
        .onTapGesture { select(candidate) }
    }

    /// いま描くときに使う色。**形と採点はこれを見ていない**ので、
    /// 変えても候補の並び順は動かない
    private var theme: IconShapes.Theme {
        let ink = Self.rgb(inkColor)
        return mono ? .mono(ink) : IconShapes.Theme(ink: ink, accent: Self.rgb(accentColor))
    }

    /// SwiftUI の Color を 0〜255 の RGB に落とす。sRGB 以外の色空間で
    /// 渡ってくることがあるので、変換してから読む
    private static func rgb(_ colour: Color) -> (r: Double, g: Double, b: Double) {
        let resolved = UIColor(colour).cgColor
        guard let converted = resolved.converted(to: CGColorSpace(name: CGColorSpace.sRGB)!,
                                                 intent: .defaultIntent, options: nil),
              let parts = converted.components, parts.count >= 3 else {
            return IconShapes.Theme.default.ink
        }
        return (Double(parts[0]) * 255, Double(parts[1]) * 255, Double(parts[2]) * 255)
    }

    /// 背景を変えたときに掛け直す。拡散は回さないので一瞬で終わる
    private func restyleAll() {
        candidates = candidates.map { candidate in
            var updated = candidate
            let polished = IconShapes.polish(candidate.transparent, size: outputSize,
                                             background: background, theme: theme)
            guard let cgImage = polished.image.makeAlphaCGImage() else { return candidate }
            updated.processed = polished.image
            updated.cgImage = cgImage
            updated.report = polished.report
            return updated
        }
        if let selected = candidates.first(where: { $0.id == selectedID }) {
            saveForInspection(selected.processed, name: "latest.png")
        }
    }

    private func select(_ candidate: Candidate) {
        selectedID = candidate.id
        let report = candidate.report
        let contrast = report.backgroundContrast.isFinite
            ? String(format: "・地との差 %.1f:1", report.backgroundContrast) : ""
        log(String(format: "候補 %d を選択（形 %.2f・塊 %d・粗さ %.2f%@・背景 %@・seed %llu）",
                   candidate.id + 1, report.score, report.parts, report.rough,
                   contrast, report.background, candidate.seed))
        saveForInspection(candidate.processed, name: "latest.png")
    }

    /// 環境変数と起動引数の両方から読む。
    /// 起動引数は `-ICON_AUTO generate` の形で渡すと UserDefaults が拾ってくれる
    static func setting(_ key: String) -> String? {
        if let value = ProcessInfo.processInfo.environment[key], !value.isEmpty { return value }
        if let value = UserDefaults.standard.string(forKey: key), !value.isEmpty { return value }
        return nil
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
        Self.appendToLogFile(message)
    }

    /// Documents にも残す。**端末がロックされると devicectl の --console 接続が切れる**ので、
    /// 標準出力だけだと長い生成の結果を取りこぼす。後から copy from で回収できるようにしておく
    private static let logURL: URL? = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask).first?
        .appendingPathComponent("run.log")

    private static func appendToLogFile(_ message: String) {
        guard let url = logURL else { return }
        let stamp = ISO8601DateFormatter().string(from: Date())
        guard let data = "\(stamp) \(message)\n".data(using: .utf8) else { return }
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: url)
        }
    }

    private func generateCandidates() async {
        isBusy = true
        // 生成は端末で 1 分以上かかる。途中で自動ロックされると計算が止まるうえ、
        // devicectl の console 接続も切れて結果が分からなくなる
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            isBusy = false
            UIApplication.shared.isIdleTimerDisabled = false
        }
        candidates = []
        selectedID = nil
        do {
            let pipeline = try await preparedPipeline()
            let env = ProcessInfo.processInfo.environment
            // 生成条件は Mac 側の比較で確定した値を既定にし、実験用に環境変数で上書き可。
            // 8 步にしたのは形の採点で 4 步に勝ったから（96 枚ずつ・seed 2 通りの両方で
            // 平均 0.399 -> 0.493）。1 枚あたりの時間は倍になるが、形の当たりが増える
            let steps = Int(Self.setting("ICON_STEPS") ?? "") ?? 8
            let guidance = Float(Self.setting("ICON_GUIDANCE") ?? "") ?? 1.0
            log("生成中…（\(steps) ステップ / guidance \(guidance)）")

            let started = Date()
            for index in 0..<candidateCount {
                log("候補 \(index + 1)/\(candidateCount) を生成中…")
                let seed = UInt64.random(in: 0..<UInt64.max)
                var options = IconDiffusionPipeline.Options(
                    steps: steps,
                    guidanceScale: guidance,
                    seed: seed
                )
                if useControl, let name = resolvedGlyph, let glyphs {
                    // 輪郭は生成解像度に合わせる。契約は pipeline_config.json 側が持つ
                    options.control = glyphs.controlTensor(
                        for: name, size: pipeline.configuration.imageSize)
                }
                let generated = try await pipeline.generateIcon(subject: subject, options: options)
                let transparent = IconPostprocess.process(generated)
                let polished = IconShapes.polish(transparent.image, size: outputSize,
                                                 background: background, theme: theme)
                guard let cgImage = polished.image.makeAlphaCGImage() else { continue }
                let candidate = Candidate(id: index, raw: generated,
                                          transparent: transparent.image,
                                          processed: polished.image,
                                          cgImage: cgImage, transparency: transparent.transparency,
                                          report: polished.report, seed: seed)
                candidates.append(candidate)
                // 内訳まで出す。点だけだと、目で見て良い絵が落ちたときに
                // どの項目が効いたのか実機では追えない
                let r = polished.report
                log(String(format:
                    "候補 %d: %@（形 %.2f｜被覆 %.2f・塊 %d・散り %.2f・粗さ %.2f・"
                    + "余白 %.3f・左右 %.2f・中間色 %.2f・透過 %.0f%%）",
                    index + 1, candidate.ok ? "合格" : "不合格", r.score,
                    r.coverage, r.parts, r.stray, r.rough, r.margin, r.symmetry,
                    r.mush, transparent.transparency * 100))
                saveForInspection(polished.image, name: "candidate\(index + 1).png")
                // 採点の入力そのもの。これが無いと、実機の点と Mac の参照実装を
                // 同じ入力で突き合わせられない
                saveForInspection(transparent.image, name: "input\(index + 1).png")
            }

            let passed = candidates.filter(\.ok)
            log(String(format: "完了 %.1f 秒。機械判定 %d/%d 通過",
                       Date().timeIntervalSince(started), passed.count, candidates.count))
            // 形の点が最も高い候補を先に選んでおく（人はタップで変えられる）。
            // 以前は透過率が 0.7 に近いものを選んでいたが、透過率は形を何も見て
            // いないので、断片が散った絵が勝つことがあった
            if let best = candidates.max(by: { $0.report.score < $1.report.score }) {
                select(best)
            }
        } catch {
            log("失敗: \(error.localizedDescription)")
        }
    }

    /// 選んだ候補を色違いで書き出す。**端末側の色の差し替えを実際に確かめるため。**
    /// 拡散は回らない（色はラベル番号から引くだけ）ので、生成 1 回ぶんで全部出る。
    private func saveThemeVariants() {
        guard let best = candidates.first(where: { $0.id == selectedID }) ?? candidates.first
        else { return }
        let navy = IconShapes.Theme.default.ink
        let variants: [(String, IconShapes.Theme)] = [
            ("default", .default),
            ("mono", .mono(navy)),
            ("green-gold", IconShapes.Theme(ink: (20, 83, 45), accent: (245, 183, 0))),
            ("purple-pink", IconShapes.Theme(ink: (76, 29, 149), accent: (244, 114, 182))),
        ]
        for (name, theme) in variants {
            let out = IconShapes.polish(best.transparent, size: outputSize,
                                        background: "auto", theme: theme)
            saveForInspection(out.image, name: "theme-\(name).png")
            log(String(format: "色 %@: 形 %.2f（不変であるべき）・背景 %@・地との差 %.1f:1",
                       name, out.report.score, out.report.background,
                       out.report.backgroundContrast))
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
