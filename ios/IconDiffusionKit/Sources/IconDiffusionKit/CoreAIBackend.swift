//
//  CoreAIBackend.swift
//
//  ★ このファイルだけが Core AI に依存する。ほかはすべて純 Swift。
//
//  既定ではビルド対象から外してある。有効にするには Package.swift のターゲットに
//  定義を足す:
//
//      .target(
//          name: "IconDiffusionKit",
//          swiftSettings: [.define("ICON_DIFFUSION_COREAI")]
//      )
//
//  **Core AI は iOS 27 / macOS 27 以降。** iOS 26 SDK には CoreAI.framework が無い。
//  ビルドには Xcode 27 以降が要る。
//
//  ---- API の検証状況（2026-08-23）----
//
//  iOS 27.0 SDK の CoreAI モジュールに対して、下の呼び出しがすべて
//  型チェックを通ることを確認済み。**推測で書いた箇所はもう無い。**
//  ただし「コンパイルが通る」までで、**実機で動かしてはいない。**
//
//  検証に使った SDK:
//    Xcode 27.0 beta / iPhoneOS27.0.sdk / CoreAI（CoreAIRuntime + CoreAIDelegates の再エクスポート）
//
//  モデルの入出力名は `Resources/pipeline_config.json` の `io` から取る。
//  ここにハードコードしない。変換側（05_export_coreai.py）を変えたら自動で追随する。
//

#if canImport(CoreAI) && ICON_DIFFUSION_COREAI

import CoreAI
import Foundation

@available(iOS 27.0, macOS 27.0, *)
public actor CoreAIBackend: DiffusionBackend {
    /// 変換側と揃えた既定の入出力名。`pipeline_config.json` に `io` が無いときだけ使う。
    private enum DefaultIO {
        static let textEncoder = (inputs: ["input_ids"], output: "text_embeddings")
        static let unet = (inputs: ["sample", "timestep", "encoder_hidden_states"], output: "noise_pred")
        static let vaeDecoder = (inputs: ["latent"], output: "image")
    }

    /// 浮動小数の入出力を fp16 で作るか fp32 で作るか。
    ///
    /// **ANE は fp32 を扱えない**ので、変換側は既定で fp16 で書き出す。ここがモデルと
    /// 食い違うとランタイムが弾くか、黙って壊れた値を返す。`pipeline_config.json` の
    /// `io` に書かれた dtype から決める。
    /// モデルごとに計算ユニットを変えるための指定。
    ///
    /// **ANE で結果が化けるモデルが実在するため、モデル単位で逃がせる必要がある。**
    /// 全体を GPU に落とす手もあるが、4 ステップ回る UNet まで巻き添えになる。
    /// 既定を ANE にしたまま、壊れるモデルだけ上書きするのが狙い。
    public struct ComputeUnitPlan: Sendable {
        /// 上書きの無いモデルに使う。nil なら Core AI に任せる。
        public var fallback: ComputeUnitKind?
        /// モデル名（TextEncoder / UNet / VAEDecoder）ごとの上書き。
        public var overrides: [String: ComputeUnitKind?]

        public init(fallback: ComputeUnitKind? = nil,
                    overrides: [String: ComputeUnitKind?] = [:]) {
            self.fallback = fallback
            self.overrides = overrides
        }

        /// 上書きに「nil を入れる」と「キーが無い」は別物。前者は「このモデルだけ
        /// Core AI 任せ」を意味する。二重 Optional のまま ?? で畳むと意図どおりになる。
        public subscript(model: String) -> ComputeUnitKind? {
            overrides[model] ?? fallback
        }

        func options(for model: String) -> SpecializationOptions {
            self[model].map { SpecializationOptions(preferredComputeUnitKind: $0) } ?? .default
        }

        /// 実機で測って決めた既定。**UNet は ANE、VAEDecoder は GPU。**
        ///
        /// VAEDecoder を ANE で動かすと出力が壊れる。同じ .aimodel を Mac で走らせると
        /// fp32 の PyTorch に対して相対誤差 1.45% で正常なので、モデルではなく
        /// ANE 上での実行が原因。実測（"a coffee cup" / 4 ステップ）:
        ///
        ///     decodeLatent 出力   ANE: [-13.27, 26.59]   GPU: [-0.85, 1.13]
        ///     Python 参照                                     [-0.78,  1.09]
        ///
        /// 全体を GPU に落とせば直るが、4 ステップ回る UNet まで巻き添えになる。
        /// UNet は ANE で参照と一致しているので、VAEDecoder だけ逃がす。
        /// VAE は 1 回しか回らないので、GPU に置いても全体時間への影響は小さい。
        public static let recommended = ComputeUnitPlan(
            fallback: .neuralEngine,
            overrides: ["VAEDecoder": .gpu]
        )

        /// 各モデルがどのユニットになるかを 1 行で返す。ログに出して取り違えに気づくため。
        public func describe(_ models: [String]) -> String {
            models.map { "\($0)=\(self[$0].map { String(describing: $0) } ?? "既定")" }
                .joined(separator: " ")
        }
    }

    private enum FloatPrecision {
        case fp16, fp32

        init(dtype: String) {
            self = (dtype == "float16") ? .fp16 : .fp32
        }

        /// **型を明示して作る。** `NDArray(scalars:shape:)` は要素型の推論が
        /// 署名から読み取りにくく、取り違えるとビット列が別の型として解釈されて
        /// 静かに壊れる。`scalarType` を指定して型付きビューに書き写す。
        func makeNDArray(_ values: [Float], shape: [Int]) -> NDArray {
            switch self {
            case .fp16:
                var array = NDArray(shape: shape, scalarType: .float16)
                var view = array.mutableView(as: Float16.self)
                view.copyElements(fromContentsOf: values.map { Float16($0) })
                return array
            case .fp32:
                var array = NDArray(shape: shape, scalarType: .float32)
                var view = array.mutableView(as: Float.self)
                view.copyElements(fromContentsOf: values)
                return array
            }
        }

        var scalarType: NDArray.ScalarType {
            switch self {
            case .fp16: return .float16
            case .fp32: return .float32
            }
        }

        /// 連続でないビューを先頭から読むと壊れる。実測では連続だが、前提が変わったら
        /// 黙って壊れる種類の話なので明示的に弾く。
        ///
        /// 要素型も先に見る。`view(as:)` は型が合わないと throw ではなく fatalError で
        /// 落ちるので、`try?` では受け止められない。実際 input_ids（int32）をここへ
        /// 渡してプロセスごと落とした。呼ぶ側で型を選別させるより、ここで弾くほうが確実。
        func scalars(of array: NDArray) throws -> [Float] {
            let count = array.shape.reduce(1, *)
            guard array.scalarType == scalarType else {
                throw IconDiffusionError.backendFailure(
                    "要素型が違います。期待 \(scalarType) / 実際 \(array.scalarType)"
                )
            }
            switch self {
            case .fp16:
                let view = array.view(as: Float16.self)
                guard view.isContiguous else {
                    throw IconDiffusionError.backendFailure("出力が連続していません（fp16）")
                }
                return view.withUnsafePointer { pointer, _, _ in
                    UnsafeBufferPointer(start: pointer, count: count).map(Float.init)
                }
            case .fp32:
                let view = array.view(as: Float.self)
                guard view.isContiguous else {
                    throw IconDiffusionError.backendFailure("出力が連続していません（fp32）")
                }
                return view.withUnsafePointer { pointer, _, _ in
                    Array(UnsafeBufferPointer(start: pointer, count: count))
                }
            }
        }
    }

    private let configuration: PipelineConfiguration
    private let textEncoder: InferenceFunction
    private let unet: InferenceFunction
    private let vaeDecoder: InferenceFunction

    private let textEncoderIO: (inputs: [String], output: String)
    private let unetIO: (inputs: [String], output: String)
    private let vaeDecoderIO: (inputs: [String], output: String)
    private let precision: FloatPrecision

    // 輪郭を条件に与える経路。**無ければ nil で、生成は従来どおり動く**
    private let controlNet: InferenceFunction?
    private let controlledUNet: InferenceFunction?
    private let controlNetIO: (inputs: [String], outputs: [String])?
    private let controlledUNetIO: (inputs: [String], output: String)?
    /// actor の計算プロパティはプロトコル要件を満たせない（隔離を跨ぐ）ので、
    /// 初期化時に確定する不変値として持つ
    public nonisolated let supportsControl: Bool
    /// 素の経路で残差つき UNet を使うときに差し込むゼロ。**毎ステップ作り直さない**
    /// （13 本で約 660 万要素あり、作り直すと marshalling だけで効いてくる）
    private let zeroResiduals: [String: NDArray]?

    /// - Parameter preferredComputeUnit: 省略すると Core AI に任せる。
    ///   `.neuralEngine` を指定すると ANE を優先する（`ComputeUnitKind.availableKinds` で確認できる）。
    public init(
        resources: PipelineResources,
        configuration: PipelineConfiguration,
        preferredComputeUnit: ComputeUnitKind? = nil
    ) async throws {
        try await self.init(resources: resources, configuration: configuration,
                            computeUnits: ComputeUnitPlan(fallback: preferredComputeUnit))
    }

    /// - Parameter computeUnits: モデルごとの計算ユニット。既定は Core AI 任せ。
    public init(
        resources: PipelineResources,
        configuration: PipelineConfiguration,
        computeUnits: ComputeUnitPlan
    ) async throws {
        try resources.validateModels()
        self.configuration = configuration

        // ローカル関数にすると self を暗黙に捕まえたと見なされ、全プロパティの初期化前に
        // self を使ったことになってビルドが通らない。static に出す。
        self.textEncoder = try await Self.load(
            resources.textEncoderURL, "TextEncoder", computeUnits.options(for: "TextEncoder"))
        // 素の UNet が無い構成では、残差つき UNet を素の経路にも使う。
        // 残差にゼロを渡せば出力は完全に一致する（最大差 0.0 で確認済み）
        let plainUNetURL = resources.usesControlledUNetOnly
            ? (resources.controlledUNetURL ?? resources.unetURL) : resources.unetURL
        self.unet = try await Self.load(
            plainUNetURL, resources.usesControlledUNetOnly ? "UNetControlled" : "UNet",
            computeUnits.options(for: "UNet"))
        self.vaeDecoder = try await Self.load(
            resources.vaeDecoderURL, "VAEDecoder", computeUnits.options(for: "VAEDecoder"))

        // ControlNet は「あれば載せる」。ファイルが無い構成でも従来どおり動く必要がある
        var loadedControlNet: InferenceFunction?
        var loadedControlledUNet: InferenceFunction?
        if let controlURL = resources.controlNetURL,
           let controlledURL = resources.controlledUNetURL,
           FileManager.default.fileExists(atPath: controlURL.path),
           FileManager.default.fileExists(atPath: controlledURL.path),
           configuration.control?["ControlNet"] != nil {
            loadedControlNet = try await Self.load(
                controlURL, "ControlNet", computeUnits.options(for: "ControlNet"))
            loadedControlledUNet = try await Self.load(
                controlledURL, "UNetControlled", computeUnits.options(for: "UNetControlled"))
        }
        self.controlNet = loadedControlNet
        self.controlledUNet = loadedControlledUNet
        if let section = configuration.control?["ControlNet"],
           let controlled = configuration.control?["UNetControlled"], loadedControlNet != nil {
            self.controlNetIO = (section.inputs.map(\.name), section.outputs.map(\.name))
            self.controlledUNetIO = (controlled.inputs.map(\.name),
                                     controlled.outputs.first?.name ?? "noise_pred")
        } else {
            self.controlNetIO = nil
            self.controlledUNetIO = nil
        }

        self.supportsControl = loadedControlNet != nil && loadedControlledUNet != nil
        self.textEncoderIO = Self.names(configuration, "TextEncoder", DefaultIO.textEncoder)
        // 契約も使うモデルのほうを見る
        if resources.usesControlledUNetOnly, let controlled = configuration.control?["UNetControlled"] {
            self.unetIO = (controlled.inputs.map(\.name).prefix(3).map { $0 },
                           controlled.outputs.first?.name ?? "noise_pred")
            self.zeroResiduals = Self.makeZeroResiduals(controlled, precision: FloatPrecision(
                dtype: controlled.inputs.first(where: { $0.name == "sample" })?.dtype ?? "float16"))
        } else {
            self.unetIO = Self.names(configuration, "UNet", DefaultIO.unet)
            self.zeroResiduals = nil
        }
        self.vaeDecoderIO = Self.names(configuration, "VAEDecoder", DefaultIO.vaeDecoder)
        // UNet の sample の dtype を代表として見る。3 モデルとも同じ精度で書き出される
        self.precision = FloatPrecision(
            dtype: configuration.io?["UNet"]?.inputs.first(where: { $0.name == "sample" })?.dtype
                ?? "float16"
        )
    }

    /// 残差ぶんのゼロを 1 回だけ作る
    private static func makeZeroResiduals(_ io: PipelineConfiguration.ModelIO,
                                          precision: FloatPrecision) -> [String: NDArray] {
        var result: [String: NDArray] = [:]
        for input in io.inputs where input.name.hasPrefix("residual_") {
            let count = input.shape.reduce(1, *)
            result[input.name] = precision.makeNDArray([Float](repeating: 0, count: count),
                                                       shape: input.shape)
        }
        return result
    }

    private static func load(
        _ url: URL, _ label: String, _ options: SpecializationOptions
    ) async throws -> InferenceFunction {
        let model = try await loadModel(url, label, options)
        // 変換時の entrypoint 名は "main" 固定（05_export_coreai.py）
        guard let function = try model.loadFunction(named: "main") else {
            throw IconDiffusionError.backendFailure(
                "\(label) に main がありません（あるのは \(model.functionNames)）"
            )
        }
        return function
    }

    /// `.aimodel` は端末向けにコンパイルしてから動く。**そのコンパイルが重い。**
    /// 毎回やらせないよう、まずキャッシュを見て、無ければ specialize して保存する。
    ///
    /// 破棄条件は既定のまま（ストレージ逼迫時と、元の `.aimodel` が変わった / 消えたとき）。
    /// 後者があるので、モデルを作り直したときに古いコンパイル結果を掴む事故は起きない。
    private static func loadModel(
        _ url: URL, _ label: String, _ options: SpecializationOptions
    ) async throws -> AIModel {
        // キャッシュが壊れていても致命的ではない。コンパイルし直せば済む
        if let cached = try? AIModelCache.default.model(for: url, options: options) {
            return cached
        }
        do {
            return try await AIModel.specialize(
                contentsOf: url, options: options, cache: .default, cachePolicy: .default
            )
        } catch {
            throw IconDiffusionError.unsupportedDevice(
                "\(label) を読み込めませんでした: \(error.localizedDescription)"
            )
        }
    }

    /// 3 モデルをコンパイルしてキャッシュに入れておく。
    ///
    /// **生成ボタンを押してから待たせないために使う。** アプリの起動直後や
    /// オンボーディングで呼んでおけば、以後の `init` はキャッシュから読むだけになる。
    /// 既にキャッシュ済みなら何もしない。
    ///
    /// - Parameter onProgress: (完了したモデル数, 全体数)。UI の進捗表示用。
    /// - Parameter only: 指定するとそのモデルだけ準備する。1 つあたりの所要時間を測るとき用。
    public static func prewarm(
        resources: PipelineResources,
        preferredComputeUnit: ComputeUnitKind? = nil,
        only: String? = nil,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws {
        try await prewarm(resources: resources,
                          computeUnits: ComputeUnitPlan(fallback: preferredComputeUnit),
                          only: only, onProgress: onProgress)
    }

    /// - Parameter computeUnits: モデルごとの計算ユニット。**生成時と同じ指定を渡すこと。**
    ///   違う指定で温めるとキャッシュが当たらず、生成時にもう一度コンパイルが走る。
    public static func prewarm(
        resources: PipelineResources,
        computeUnits: ComputeUnitPlan,
        only: String? = nil,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws {
        try resources.validateModels()
        var targets = [
            (resources.textEncoderURL, "TextEncoder"),
            (resources.unetURL, "UNet"),
            (resources.vaeDecoderURL, "VAEDecoder"),
        ]
        if let only {
            targets = targets.filter { $0.1 == only }
        }
        for (index, target) in targets.enumerated() {
            _ = try await loadModel(target.0, target.1, computeUnits.options(for: target.1))
            onProgress?(index + 1, targets.count)
        }
    }

    /// NDArray の受け渡しだけを往復させて、値が壊れないか確かめる。
    ///
    /// **モデルを一切触らない**ので即座に終わる。実機で絵が壊れたとき、原因が
    /// 受け渡しなのかモデルなのかを切り分けるために要る。
    /// 戻り値は (最大誤差, 件数)。fp16 の丸め誤差ぶんは残るので 0 にはならない。
    public static func selfTestMarshalling() throws -> (maxError: Float, count: Int) {
        let values: [Float] = (0..<2048).map { index in
            // 正負・大小・小数を混ぜる。全部同じ値だと並び替えの誤りに気づけない
            let t = Float(index)
            return (t.truncatingRemainder(dividingBy: 7) - 3) * (1 + t / 512)
        }
        let shape = [1, 2, 32, 32]
        var maxError: Float = 0
        for precision in [FloatPrecision.fp16, FloatPrecision.fp32] {
            let array = precision.makeNDArray(values, shape: shape)
            let restored = try precision.scalars(of: array)
            guard restored.count == values.count else {
                throw IconDiffusionError.backendFailure(
                    "要素数が変わりました: \(values.count) -> \(restored.count)"
                )
            }
            for (a, b) in zip(values, restored) {
                maxError = max(maxError, abs(a - b) / max(abs(a), 1))
            }
        }
        return (maxError, values.count)
    }

    /// 実行できる計算ユニットの一覧。端末で何が使えるかの確認用。
    public static var availableComputeUnits: [String] {
        ComputeUnitKind.availableKinds.map { String(describing: $0) }.sorted()
    }

    private static func names(
        _ configuration: PipelineConfiguration,
        _ key: String,
        _ fallback: (inputs: [String], output: String)
    ) -> (inputs: [String], output: String) {
        guard let spec = configuration.io?[key], let first = spec.outputs.first else {
            return fallback
        }
        return (spec.inputs.map(\.name), first.name)
    }

    // MARK: - DiffusionBackend

    public func encodeText(inputIDs: [Int32]) async throws -> Tensor {
        // input_ids は int32。Float に丸めて渡すと語彙 ID がずれる
        // トークン ID は整数のまま渡す。float に落とすと語彙 ID がずれる
        let ids = NDArray(scalars: inputIDs, shape: [1, inputIDs.count])
        return try await run(
            textEncoder,
            inputs: [textEncoderIO.inputs[0]: ids],
            output: textEncoderIO.output,
            expecting: configuration.textEmbeddingShape,
            context: "encodeText"
        )
    }

    public func predictNoise(
        sample: Tensor,
        timestep: Float,
        encoderHiddenStates: Tensor
    ) async throws -> Tensor {
        // timestep は float32 の [1]。整数で渡すとランタイムが弾く
        var inputs = [
            unetIO.inputs[0]: precision.makeNDArray(sample.values, shape: sample.shape),
            unetIO.inputs[1]: precision.makeNDArray([timestep], shape: [1]),
            unetIO.inputs[2]: precision.makeNDArray(encoderHiddenStates.values,
                                                    shape: encoderHiddenStates.shape),
        ]
        if let zeroResiduals {
            inputs.merge(zeroResiduals) { current, _ in current }
        }
        return try await run(
            unet, inputs: inputs, output: unetIO.output,
            expecting: sample.shape, context: "predictNoise"
        )
    }



    /// 輪郭つきのノイズ予測。**毎ステップ 2 モデル回る**（ControlNet → 残差つき UNet）。
    ///
    /// ControlNet が返す 13 本の残差をそのまま UNet の入力に差す。名前は
    /// `pipeline_config.json` の `control` セクションが唯一の情報源で、Swift 側には書かない。
    public func predictNoise(
        sample: Tensor,
        timestep: Float,
        encoderHiddenStates: Tensor,
        control: Tensor
    ) async throws -> Tensor {
        guard let controlNet, let controlledUNet,
              let controlNetIO, let controlledUNetIO else {
            throw IconDiffusionError.unsupportedDevice(
                "輪郭を条件に与える経路が載っていません（Stage 5 を --controlnet 付きで実行してください）"
            )
        }
        let shared = [
            precision.makeNDArray(sample.values, shape: sample.shape),
            precision.makeNDArray([timestep], shape: [1]),
            precision.makeNDArray(encoderHiddenStates.values, shape: encoderHiddenStates.shape),
        ]
        var controlInputs: [String: NDArray] = [:]
        for (name, array) in zip(controlNetIO.inputs, shared) { controlInputs[name] = array }
        controlInputs[controlNetIO.inputs[3]] =
            precision.makeNDArray(control.values, shape: control.shape)

        var residuals: InferenceFunction.Outputs
        do {
            residuals = try await controlNet.run(inputs: controlInputs)
        } catch {
            throw IconDiffusionError.backendFailure(
                "predictNoise(control): \(error.localizedDescription)")
        }

        var unetInputs: [String: NDArray] = [:]
        for (name, array) in zip(controlledUNetIO.inputs, shared) { unetInputs[name] = array }
        for name in controlNetIO.outputs {
            guard let value = residuals.remove(name), let array = value.ndArray else {
                throw IconDiffusionError.backendFailure(
                    "predictNoise(control): 残差 \(name) がありません")
            }
            unetInputs[name] = array
        }
        return try await run(
            controlledUNet, inputs: unetInputs, output: controlledUNetIO.output,
            expecting: sample.shape, context: "predictNoise(control)"
        )
    }

    public func decodeLatent(_ latent: Tensor) async throws -> Tensor {
        return try await run(
            vaeDecoder,
            inputs: [vaeDecoderIO.inputs[0]: precision.makeNDArray(latent.values, shape: latent.shape)],
            output: vaeDecoderIO.output,
            expecting: configuration.decodedImageShape,
            context: "decodeLatent"
        )
    }

    // MARK: - Core AI との受け渡し

    private func run(
        _ function: InferenceFunction,
        inputs: [String: NDArray],
        output name: String,
        expecting shape: [Int],
        context: String
    ) async throws -> Tensor {
        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await function.run(inputs: inputs)
        } catch {
            throw IconDiffusionError.backendFailure("\(context): \(error.localizedDescription)")
        }

        guard let value = outputs.remove(name) else {
            throw IconDiffusionError.backendFailure(
                "\(context): 出力 \(name) がありません（あるのは \(Array(outputs.names))）"
            )
        }
        guard let array = value.ndArray else {
            throw IconDiffusionError.backendFailure("\(context): 出力 \(name) が NDArray ではありません")
        }
        // 形の検証はここで必ず行う。ズレたまま先に進むと、絵が壊れる形でしか気づけない
        guard array.shape == shape else {
            throw IconDiffusionError.backendFailure(
                "\(context): 出力の形が違います。期待 \(shape) / 実際 \(array.shape)"
            )
        }
        let values = try precision.scalars(of: array)
        if ProcessInfo.processInfo.environment["ICON_DEBUG"] == "1" {
            // 入力側も測る。出力が変なとき、入力が既に変なのかここで分かる
            for name in inputs.keys.sorted() {
                print("[stats]   \(context) 入力 \(name): \(Self.summarize(inputs[name]!, precision))")
            }
        }
        Self.debugLog(context, inputs: inputs, output: values)
        return try Tensor(shape: shape, values: values, context: "CoreAIBackend.\(context)")
    }

    /// ICON_DEBUG=1 のとき、各段のテンソルの統計を出す。
    ///
    /// **絵が壊れたときにどこで壊れたかを特定するため。** 形は requireShape が見ているので、
    /// ここで見たいのは値のほう。Python 側の同じ段の値と突き合わせる。
    private static func debugLog(_ context: String, inputs: [String: NDArray], output: [Float]) {
        guard ProcessInfo.processInfo.environment["ICON_DEBUG"] == "1" else { return }
        print("[stats] \(context) 出力 \(summary(output)) n=\(output.count)")
    }

    /// 入力は float とは限らない（input_ids は int32）。要素型で読み方を変える。
    private static func summarize(_ array: NDArray, _ precision: FloatPrecision) -> String {
        if array.scalarType == .int32 {
            let view = array.view(as: Int32.self)
            guard view.isContiguous else { return "int32・非連続" }
            let count = array.shape.reduce(1, *)
            let ids = view.withUnsafePointer { pointer, _, _ in
                Array(UnsafeBufferPointer(start: pointer, count: count))
            }
            return "int32 min=\(ids.min() ?? 0) max=\(ids.max() ?? 0) n=\(count)"
        }
        guard let values = try? precision.scalars(of: array) else {
            return "読めません（\(array.scalarType)）"
        }
        return summary(values)
    }

    static func summary(_ values: [Float]) -> String {
        guard let low = values.min(), let high = values.max() else { return "空" }
        let mean = values.reduce(0, +) / Float(values.count)
        let finite = values.allSatisfy { $0.isFinite }
        return String(format: "min=%.4f max=%.4f mean=%.4f", low, high, mean)
            + (finite ? "" : " ★NaN/Inf あり")
    }
}

#endif
