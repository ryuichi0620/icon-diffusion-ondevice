#if canImport(CoreAI)
import CoreAI
#endif
import IconDiffusionKit
import ImageIO
import Photos
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// アイコン生成の本体。画面（`ProfileEditView`）は状態を持たず、拡散・後処理・色の
/// 差し替えはすべてここが握る。
///
/// **既定値はすべて実測で決めたもの**で、その場の好みではない。根拠は各行の注記と
/// docs/experiments.md を参照。画面から変えられるのは被写体・色・背景だけにしてあり、
/// ステップ数のような「外すと露骨に悪くなる」値は出していない。
@MainActor
@Observable
final class IconStudio {
    struct Candidate: Identifiable {
        let id: Int
        /// 背景を敷く前（透過のみ）。色や背景を変えるたびに拡散を回し直さずに済む
        let transparent: GeneratedImage
        var processed: GeneratedImage
        var cgImage: CGImage
        let transparency: Double
        var report: IconShapes.Report
        let seed: UInt64
        /// 機械判定の合否。**画面には出さない** — 人の目と 10 枚中 3 枚で食い違い、
        /// ほぼ空の絵を 0.88 で「合格」にした実績がある。開発用の欄にだけ出す
        var ok: Bool { report.score >= IconShapes.passScore }
    }

    // MARK: 入力

    var subject = "a coffee cup"
    /// 敷く背景。**SNS のアイコンは円に切り抜かれて他人のタイムラインに載る**ので、
    /// 透過のままだと地の色に左右される。auto は輪郭の色に対して WCAG 1.4.11 の
    /// 3:1 を満たすパレット色を選ぶ
    var background = "auto" { didSet { restyleAll() } }
    var inkColor = Color(red: 31 / 255, green: 42 / 255, blue: 68 / 255) { didSet { restyleAll() } }
    var accentColor = Color(red: 255 / 255, green: 107 / 255, blue: 74 / 255) { didSet { restyleAll() } }
    var mono = false { didSet { restyleAll() } }
    /// 輪郭を下敷きにするか。**既定で入れる。** 実機の絵を人が見て判定した結果、
    /// 下敷きなしは 5 件中 5 件が「何か分からない」で、下敷きありは 5 件中 3 件が可だった
    /// （Mac 側の機械判定でも 58.3% -> 87.5%。docs/experiments.md Trial 19〜21）
    var useControl = true { didSet { refreshGlyph() } }

    // MARK: 出力

    private(set) var candidates: [Candidate] = []
    var selectedID: Int?
    private(set) var status = ""
    /// 生成が落ちた理由。**画面に出す。** 開発用の欄に畳んだままだと、
    /// 押しても何も起きないアプリに見える
    private(set) var failure: String?
    private(set) var isBusy = false
    /// 0〜1。1 枚 5 秒前後 × 4 枚かかるので、押しっぱなしに見せない
    private(set) var progress: Double = 0
    private(set) var didPrewarm = false
    private(set) var glyphs: GlyphLibrary?
    private(set) var resolvedGlyph: String?

    /// 出力解像度。整形は濃度場を拡大してから二値化するので、512 で作った絵でも
    /// 1024 で輪郭が階段にならない
    let outputSize = 1024
    /// **1 枚だけ作る。** 4 枚並べて選ばせるのはやめた（選ぶ手間より作り直すほうが速く、
    /// 並べると機械の点で順序を付けたくなるが、その点は人の判定と食い違う）。
    /// 実機で 1 枚 5.4 秒。
    /// 計測のときだけ起動引数 `-ICON_COUNT 2` で枚数を上書きできる
    /// （2 枚目が速いか、同じ引っかかりが再現するかを 1 セッション内で見るため）
    let candidateCount = Int(IconStudio.setting("ICON_COUNT") ?? "") ?? 1

    private var pipeline: IconDiffusionPipeline?

    var selected: Candidate? { candidates.first { $0.id == selectedID } }

    /// いま描くときに使う色。**形と採点はこれを見ていない**ので、
    /// 変えても候補の並び順は動かない
    var theme: IconShapes.Theme {
        let ink = Self.rgb(inkColor)
        return mono ? .mono(ink) : IconShapes.Theme(ink: ink, accent: Self.rgb(accentColor))
    }

    /// いまの被写体で輪郭が引けるかを、生成前に見せる
    var controlStatus: String {
        guard useControl else { return "下書きなし（言葉だけで描きます）" }
        guard let name = resolvedGlyph else { return "この言葉の下書きは無いので、言葉だけで描きます" }
        return "下書き: \(name)"
    }

    // MARK: 起動

    func start() async {
        if let root = Bundle.main.resourceURL?
            .appendingPathComponent("Models").appendingPathComponent("Glyphs") {
            glyphs = GlyphLibrary(directory: root)
            refreshGlyph()
            log(glyphs == nil ? "グリフの表を読めませんでした"
                              : "グリフ \(glyphs?.words.count ?? 0) 語を読みました")
        }
        await runLaunchAutomation()
    }

    /// 無人実行の入口。端末での検証はこれ頼みなので、画面を作り替えても残す:
    ///   xcrun devicectl device process launch --device <UDID> --console \
    ///     --terminate-existing com.egoshi.icondiffusion \
    ///     -ICON_AUTO generate -ICON_SUBJECT "a coffee cup"
    private func runLaunchAutomation() async {
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
            await generate()
            saveThemeVariants()
        }
        log("自動実行を終えました")
    }

    func refreshGlyph() {
        resolvedGlyph = useControl ? glyphs?.resolve(subject) : nil
    }

    // MARK: 生成

    func generate() async {
        isBusy = true
        progress = 0
        failure = nil
        // 生成は端末で 20 秒以上かかる。途中で自動ロックされると計算が止まるうえ、
        // devicectl の console 接続も切れて結果が分からなくなる
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            isBusy = false
            progress = 0
            UIApplication.shared.isIdleTimerDisabled = false
        }
        candidates = []
        selectedID = nil
        do {
            let pipeline = try await preparedPipeline()
            // 生成条件は Mac 側の比較で確定した値。実験用に環境変数で上書きできる。
            // 8 步にしたのは形の採点で 4 步に勝ったから（96 枚ずつ・seed 2 通りの両方で
            // 平均 0.399 -> 0.493）。1 枚あたりの時間は倍になるが、形の当たりが増える
            let steps = Int(Self.setting("ICON_STEPS") ?? "") ?? 8
            let guidance = Float(Self.setting("ICON_GUIDANCE") ?? "") ?? 1.0
            log("生成中…（\(steps) ステップ / guidance \(guidance)）")

            let started = Date()
            // ループは残してある。枚数を戻して比べたくなったときのため
            for index in 0..<candidateCount {
                let base = Double(index) / Double(candidateCount)
                let share = 1.0 / Double(candidateCount)
                // 既定は毎回ちがう絵。**A/B を比べるときだけ** 起動引数で固定する
                // （`-ICON_SEED 42`。2 枚目以降は +1 していく）
                let seed = (UInt64(Self.setting("ICON_SEED") ?? "") ?? UInt64.random(in: 0..<UInt64.max))
                    &+ UInt64(index)
                var options = IconDiffusionPipeline.Options(
                    steps: steps,
                    guidanceScale: guidance,
                    seed: seed,
                    onProgress: { [weak self] done, total in
                        Task { @MainActor in
                            self?.progress = base + share * Double(done) / Double(max(total, 1))
                        }
                    }
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
                candidates.append(Candidate(id: index,
                                            transparent: transparent.image,
                                            processed: polished.image,
                                            cgImage: cgImage,
                                            transparency: transparent.transparency,
                                            report: polished.report, seed: seed))
                progress = base + share
                // 内訳まで残す。点だけだと、目で見て良い絵が落ちたときに
                // どの項目が効いたのか実機では追えない
                let r = polished.report
                log(String(format:
                    "候補 %d: %@（形 %.2f｜被覆 %.2f・塊 %d・散り %.2f・粗さ %.2f・"
                    + "余白 %.3f・左右 %.2f・中間色 %.2f・透過 %.0f%%）",
                    index + 1, polished.report.score >= IconShapes.passScore ? "合格" : "不合格",
                    r.score, r.coverage, r.parts, r.stray, r.rough, r.margin, r.symmetry,
                    r.mush, transparent.transparency * 100))
                saveForInspection(polished.image, name: "candidate\(index + 1).png")
                // 採点の入力そのもの。これが無いと、実機の点と Mac の参照実装を
                // 同じ入力で突き合わせられない
                saveForInspection(transparent.image, name: "input\(index + 1).png")
            }

            let passed = candidates.filter(\.ok)
            log(String(format: "完了 %.1f 秒。機械判定 %d/%d 通過",
                       Date().timeIntervalSince(started), passed.count, candidates.count))
            // 点が最も高いものを採る。1 枚なら選択の余地は無いが、枚数を戻したときに
            // 先頭が決まらなくなるので式は残す。**この点は当てにならない**
            // （人の判定と 10 枚中 7 枚しか合わない）ので、良し悪しは人が見て決める
            if let best = candidates.max(by: { $0.report.score < $1.report.score }) {
                select(best.id)
            }
        } catch {
            failure = error.localizedDescription
            log("失敗: \(error.localizedDescription)")
        }
    }

    func select(_ id: Int) {
        selectedID = id
        guard let candidate = selected else { return }
        let report = candidate.report
        let contrast = report.backgroundContrast.isFinite
            ? String(format: "・地との差 %.1f:1", report.backgroundContrast) : ""
        log(String(format: "候補 %d を選択（形 %.2f・塊 %d・粗さ %.2f%@・背景 %@・seed %llu）",
                   candidate.id + 1, report.score, report.parts, report.rough,
                   contrast, report.background, candidate.seed))
        saveForInspection(candidate.processed, name: "latest.png")
    }

    /// 色や背景を変えたときに掛け直す。**拡散は回らない**（色はラベル番号から
    /// 引くだけ）ので、候補 4 枚ぶんでも一瞬で終わる
    private func restyleAll() {
        guard !candidates.isEmpty else { return }
        let theme = theme
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
        if let selected { saveForInspection(selected.processed, name: "latest.png") }
    }

    // MARK: 書き出し

    /// 写真アプリに書き出す。**実際に SNS のアイコンに設定するにはここを通る**ので、
    /// 作って終わりにしない。追加のみの許可で足りる
    func saveToPhotos() {
        guard let selected, let data = Self.pngData(selected.cgImage) else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] authorization in
            Task { @MainActor in
                guard authorization == .authorized || authorization == .limited else {
                    self?.log("写真へのアクセスが許可されていません（設定で変えられます）")
                    return
                }
                do {
                    try await PHPhotoLibrary.shared().performChanges {
                        let request = PHAssetCreationRequest.forAsset()
                        request.addResource(with: .photo, data: data, options: nil)
                    }
                    self?.log("写真に保存しました")
                } catch {
                    self?.log("写真に保存できませんでした: \(error.localizedDescription)")
                }
            }
        }
    }

    /// PNG に符号化する。写真への書き出しと、プロフィール画像の保存の両方で使う
    static func pngData(_ image: CGImage) -> Data? {
        let buffer = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            buffer as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return buffer as Data
    }

    /// 選んだ候補を色違いで書き出す。**端末側の色の差し替えを実際に確かめるため。**
    /// 拡散は回らないので、生成 1 回ぶんで全部出る
    func saveThemeVariants() {
        guard let best = selected ?? candidates.first else { return }
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

    /// 後処理済み（透過 RGBA）を Documents に置く。devicectl で取り出して確認するため
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

    // MARK: モデル

    func prewarm() async {
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
                    Task { @MainActor in self.log("モデルを準備中… \(done)/\(total)") }
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
        // 文言は IconDiffusionError 側が「この端末では Core AI を使えません:」を
        // 前置きするので、ここでは理由だけ書く
        throw IconDiffusionError.unsupportedDevice("シミュレータでは動きません（実機専用）。")
        #endif
    }

    // Core AI の型に触る部分。**シミュレータでは CoreAI が import できない**ので、
    // ここを #if で囲っていないと画面の見た目すら確かめられない
    #if canImport(CoreAI) && ICON_DIFFUSION_COREAI
    private static func unit(_ name: String?) -> ComputeUnitKind? {
        switch name {
        case "default": return nil
        case "cpu": return .cpu
        case "gpu": return .gpu
        default: return .neuralEngine
        }
    }

    /// 既定は ComputeUnitPlan.recommended（UNet=ANE / VAEDecoder=GPU）。
    /// **環境変数は devicectl では届かない**ので、起動引数からも読む
    /// （`-ICON_UNIT_TextEncoder gpu` の形）
    static func computeUnitPlan() -> CoreAIBackend.ComputeUnitPlan {
        var plan = CoreAIBackend.ComputeUnitPlan.recommended
        if let raw = setting("ICON_UNIT") {
            plan = CoreAIBackend.ComputeUnitPlan(fallback: unit(raw))
        }
        for model in ["TextEncoder", "ControlNet", "UNet", "UNetControlled", "VAEDecoder"] {
            if let raw = setting("ICON_UNIT_\(model)") {
                plan.overrides[model] = unit(raw)
            }
        }
        return plan
    }
    #endif

    // MARK: 雑用

    /// 環境変数と起動引数の両方から読む。
    /// 起動引数は `-ICON_AUTO generate` の形で渡すと UserDefaults が拾ってくれる
    static func setting(_ key: String) -> String? {
        if let value = ProcessInfo.processInfo.environment[key], !value.isEmpty { return value }
        if let value = UserDefaults.standard.string(forKey: key), !value.isEmpty { return value }
        return nil
    }

    /// SwiftUI の Color を 0〜255 の RGB に落とす。sRGB 以外の色空間で
    /// 渡ってくることがあるので、変換してから読む
    static func rgb(_ colour: Color) -> (r: Double, g: Double, b: Double) {
        let resolved = UIColor(colour).cgColor
        guard let converted = resolved.converted(to: CGColorSpace(name: CGColorSpace.sRGB)!,
                                                 intent: .defaultIntent, options: nil),
              let parts = converted.components, parts.count >= 3 else {
            return IconShapes.Theme.default.ink
        }
        return (Double(parts[0]) * 255, Double(parts[1]) * 255, Double(parts[2]) * 255)
    }

    func log(_ message: String) {
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
}
