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

    private let configuration: PipelineConfiguration
    private let textEncoder: InferenceFunction
    private let unet: InferenceFunction
    private let vaeDecoder: InferenceFunction

    private let textEncoderIO: (inputs: [String], output: String)
    private let unetIO: (inputs: [String], output: String)
    private let vaeDecoderIO: (inputs: [String], output: String)

    /// - Parameter preferredComputeUnit: 省略すると Core AI に任せる。
    ///   `.neuralEngine` を指定すると ANE を優先する（`ComputeUnitKind.availableKinds` で確認できる）。
    public init(
        resources: PipelineResources,
        configuration: PipelineConfiguration,
        preferredComputeUnit: ComputeUnitKind? = nil
    ) async throws {
        try resources.validateModels()
        self.configuration = configuration

        let options = preferredComputeUnit.map { SpecializationOptions(preferredComputeUnitKind: $0) }
            ?? .default

        // ローカル関数にすると self を暗黙に捕まえたと見なされ、全プロパティの初期化前に
        // self を使ったことになってビルドが通らない。static に出す。
        self.textEncoder = try await Self.load(resources.textEncoderURL, "TextEncoder", options)
        self.unet = try await Self.load(resources.unetURL, "UNet", options)
        self.vaeDecoder = try await Self.load(resources.vaeDecoderURL, "VAEDecoder", options)

        self.textEncoderIO = Self.names(configuration, "TextEncoder", DefaultIO.textEncoder)
        self.unetIO = Self.names(configuration, "UNet", DefaultIO.unet)
        self.vaeDecoderIO = Self.names(configuration, "VAEDecoder", DefaultIO.vaeDecoder)
    }

    private static func load(
        _ url: URL, _ label: String, _ options: SpecializationOptions
    ) async throws -> InferenceFunction {
        let model: AIModel
        do {
            model = try await AIModel(contentsOf: url, options: options)
        } catch {
            throw IconDiffusionError.unsupportedDevice(
                "\(label) を読み込めませんでした: \(error.localizedDescription)"
            )
        }
        // 変換時の entrypoint 名は "main" 固定（05_export_coreai.py）
        guard let function = try model.loadFunction(named: "main") else {
            throw IconDiffusionError.backendFailure(
                "\(label) に main がありません（あるのは \(model.functionNames)）"
            )
        }
        return function
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
        let inputs = [
            unetIO.inputs[0]: NDArray(scalars: sample.values, shape: sample.shape),
            unetIO.inputs[1]: NDArray(scalars: [timestep], shape: [1]),
            unetIO.inputs[2]: NDArray(scalars: encoderHiddenStates.values,
                                      shape: encoderHiddenStates.shape),
        ]
        return try await run(
            unet, inputs: inputs, output: unetIO.output,
            expecting: sample.shape, context: "predictNoise"
        )
    }

    public func decodeLatent(_ latent: Tensor) async throws -> Tensor {
        return try await run(
            vaeDecoder,
            inputs: [vaeDecoderIO.inputs[0]: NDArray(scalars: latent.values, shape: latent.shape)],
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
        return try Tensor(shape: shape, values: scalars(of: array), context: "CoreAIBackend.\(context)")
    }

    private func scalars(of array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        return array.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }
}

#endif
