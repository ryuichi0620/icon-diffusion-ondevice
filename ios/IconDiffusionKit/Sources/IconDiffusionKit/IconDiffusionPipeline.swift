import Foundation

/// アイコン生成の本体。企画メモ Stage 6 の「拡散の数回まわす部分」。
///
///   1. テキストエンコーダを 1 回だけ実行（プロンプト -> 埋め込み）
///   2. UNet をステップ数だけループ（ノイズ予測 -> スケジューラで少し綺麗にする）
///   3. VAE デコーダで最後に画像へ変換
///
/// Core AI には直接触らない。`DiffusionBackend` 越しに 3 つのモデルを呼ぶだけなので、
/// `StubBackend` に差し替えればモデル無しで配線を検証できる。
public struct IconDiffusionPipeline: Sendable {
    public struct Options: Sendable {
        /// 推論ステップ数。LCM の想定は 2〜4。
        public var steps: Int
        /// 1.0 以下なら CFG を使わない（UNet の実行回数が半分になる）。
        /// 端末側は 1.0 が既定。理由は docs/architecture.md「CFG を使うか」。
        public var guidanceScale: Float
        public var seed: UInt64
        /// CFG を使うときの無条件側プロンプト。`guidanceScale <= 1.0` なら無視される。
        public var negativePrompt: String
        /// (完了ステップ数, 全ステップ数)。UI の進捗表示用。
        public var onProgress: (@Sendable (Int, Int) -> Void)?
        /// 輪郭を条件に与えるときの画像 `[1, 3, imageSize, imageSize]`（白線・黒地）。
        /// **形はここから、スタイルはモデルから。** nil なら従来どおり文字だけで作る
        public var control: Tensor?
        /// 条件の効き。1.0 に近いほど輪郭に忠実だがスタイルが痩せる。
        /// Mac 側のスイープでは 0.75 が最良で、0.95 まで上げると輪郭をなぞるだけになった
        public var controlScale: Float
        /// 途中で条件を切る割合。終盤を自由にするとスタイルが乗りやすい
        public var controlEnd: Float

        public init(
            steps: Int = 4,
            guidanceScale: Float = 1.0,
            seed: UInt64 = 0,
            negativePrompt: String = "",
            control: Tensor? = nil,
            controlScale: Float = 0.75,
            controlEnd: Float = 0.8,
            onProgress: (@Sendable (Int, Int) -> Void)? = nil
        ) {
            self.steps = steps
            self.guidanceScale = guidanceScale
            self.seed = seed
            self.negativePrompt = negativePrompt
            self.control = control
            self.controlScale = controlScale
            self.controlEnd = controlEnd
            self.onProgress = onProgress
        }
    }

    public let configuration: PipelineConfiguration
    private let backend: any DiffusionBackend
    private let tokenizer: CLIPTokenizer

    public init(
        backend: any DiffusionBackend,
        tokenizer: CLIPTokenizer,
        configuration: PipelineConfiguration
    ) {
        self.backend = backend
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    /// `pipeline_config.json` と トークナイザ資産を読み込んで組み立てる。
    /// モデルの読み込みは backend 側の責任なので、ここでは触らない。
    public init(resources: PipelineResources, backend: any DiffusionBackend) throws {
        try resources.validateTokenizer()
        let configuration = try PipelineConfiguration.load(contentsOf: resources.configurationURL)
        let tokenizer = try CLIPTokenizer(
            vocabularyURL: resources.vocabularyURL,
            mergesURL: resources.mergesURL,
            maxLength: configuration.maxSequenceLength
        )
        self.init(backend: backend, tokenizer: tokenizer, configuration: configuration)
    }

    /// 学習時と同じ形のプロンプト（`an icon in ICONST style, <被写体>, flat, minimal`）に整えて生成する。
    /// アプリからは基本こちらを使う。トリガーワードの付け忘れが起きない。
    public func generateIcon(subject: String, options: Options = Options()) async throws -> GeneratedImage {
        try await generate(prompt: configuration.prompt(subject: subject), options: options)
    }

    /// プロンプトをそのまま使って生成する。
    public func generate(prompt: String, options: Options = Options()) async throws -> GeneratedImage {
        let scheduler = try LCMScheduler(
            configuration: LCMScheduler.Configuration(pipeline: configuration),
            inferenceSteps: options.steps
        )

        // 1. テキストエンコーダは生成ごとに 1 回だけ
        let conditioning = try await backend.encodeText(inputIDs: try tokenizer.tokenize(prompt))
        try conditioning.requireShape(
            configuration.textEmbeddingShape, context: "encodeText(prompt)"
        )

        var unconditional: Tensor?
        if options.guidanceScale > 1.0 {
            let embedding = try await backend.encodeText(
                inputIDs: try tokenizer.tokenize(options.negativePrompt)
            )
            try embedding.requireShape(
                configuration.textEmbeddingShape, context: "encodeText(negativePrompt)"
            )
            unconditional = embedding
        }

        // 2. UNet をステップ数だけ回す
        var random = SeededRandomSource(seed: options.seed)
        var latents = random
            .gaussianTensor(shape: configuration.latentShape)
            .scaled(by: scheduler.initialNoiseSigma)

        // 条件を使うのは序盤だけ。終盤まで効かせると輪郭をなぞるだけになる
        let controlSteps = options.control == nil ? 0
            : max(1, Int((Float(scheduler.timesteps.count) * options.controlEnd).rounded()))

        for (index, timestep) in scheduler.timesteps.enumerated() {
            let time = Float(timestep)
            var noisePrediction: Tensor
            if let control = options.control, index < controlSteps, backend.supportsControl {
                noisePrediction = try await backend.predictNoise(
                    sample: latents, timestep: time, encoderHiddenStates: conditioning,
                    control: control
                )
            } else {
                noisePrediction = try await backend.predictNoise(
                    sample: latents, timestep: time, encoderHiddenStates: conditioning
                )
            }
            try noisePrediction.requireShape(latents.shape, context: "predictNoise")

            if let unconditional {
                // CFG はバッチ 2 で 1 回ではなく、バッチ 1 で 2 回回す。
                // 計算量は同じで、既定の変換（--batch 1）で出した UNet をそのまま使える。
                let unguided = try await backend.predictNoise(
                    sample: latents, timestep: time, encoderHiddenStates: unconditional
                )
                try unguided.requireShape(latents.shape, context: "predictNoise(unconditional)")
                let scale = options.guidanceScale
                noisePrediction = try Tensor.combine(
                    unguided, noisePrediction, context: "classifierFreeGuidance"
                ) { unguidedValue, guidedValue in
                    unguidedValue + scale * (guidedValue - unguidedValue)
                }
            }

            // 中間ステップではノイズを注ぎ直す。ここを省くと 4 ステップでも眠い絵になる。
            let isFinalStep = index == scheduler.timesteps.count - 1
            let noise = isFinalStep
                ? Tensor.zeros(shape: latents.shape)
                : random.gaussianTensor(shape: latents.shape)

            latents = try scheduler.step(
                modelOutput: noisePrediction,
                sample: latents,
                stepIndex: index,
                noise: noise
            ).previousSample

            options.onProgress?(index + 1, scheduler.timesteps.count)
        }

        // 3. デコード
        let decodeInput = configuration.vaeScalingAppliedInModel
            ? latents
            : latents.scaled(by: Float(1.0 / configuration.vaeScalingFactor))
        let decoded = try await backend.decodeLatent(decodeInput)
        try decoded.requireShape(configuration.decodedImageShape, context: "decodeLatent")

        return try GeneratedImage(planarCHW: decoded)
    }
}
