//
//  CoreAIBackend.swift
//
//  ★ このファイルだけが Core AI に依存する。ほかはすべて純 Swift。
//
//  既定ではビルド対象から外してある。中身が WWDC26 の実 API に対して**未検証**だからで、
//  推測で書いたコードが既定で混ざるとビルドが通らない・静かにズレる、のどちらかになる。
//
//  有効にするには Package.swift のターゲットに定義を足す:
//
//      .target(
//          name: "IconDiffusionKit",
//          swiftSettings: [.define("ICON_DIFFUSION_COREAI")]
//      )
//
//  有効にする前に、下の TODO(core-ai) を実物と突き合わせること:
//    - WWDC26「Meet Core AI」(324) / 「Dive into Core AI model authoring and optimization」(325)
//    - apple/coreai-models（パッケージの組み方の見本）
//    - apple/ml-stable-diffusion（旧 Core ML 版。モデルの分け方と Swift ループの下敷き）
//
//  ---- 2026-08-23 に Python 側（coreai-core 1.0.0b2）で確認できたこと ----
//
//  Python ランタイムでは 3 モデルすべて実際に推論でき、PyTorch と数値が一致した。
//  ただし **Python の API と Swift の API は別物**なので、下は「モデルの契約」であって
//  「Swift の呼び出し方」ではない。Swift 側の API 名は依然として未検証。
//
//  モデルの入出力（名前・形・dtype）は Resources/pipeline_config.json の `io` に入っている。
//  **ここにハードコードしないこと。** 変換側を変えたときに静かにズレる。
//
//    TextEncoder  in  input_ids             [1, 77]          int32
//                 out text_embeddings       [1, 77, 768]     float32
//    UNet         in  sample                [1, 4, 64, 64]   float32
//                 in  timestep              [1]              float32  ★ int ではない
//                 in  encoder_hidden_states [1, 77, 768]     float32
//                 out noise_pred            [1, 4, 64, 64]   float32
//    VAEDecoder   in  latent                [1, 4, 64, 64]   float32
//                 out image                 [1, 3, 512, 512] float32（値域 -1〜1）
//
//  ★ timestep を整数で渡すとランタイムが弾く（Python 側で実際に踏んだ）。
//
//  Python 側で踏んだ落とし穴（Swift でも似た形で出るかもしれない）:
//    - 出力名は明示しないとグラフ内部の自動生成名（convolution_97 等）になる。
//      Stage 5 で input_names / output_names を明示的に渡して固定してある
//    - モデルの読み込みも推論も非同期だった
//
//  API がこちらの想定とずれていても、直すのはこのファイルだけで済むようにしてある。
//  DiffusionBackend さえ満たせば、パイプラインもスケジューラも変更不要。
//

#if canImport(CoreAI) && ICON_DIFFUSION_COREAI

import CoreAI
import Foundation

@available(iOS 26.0, macOS 26.0, *)
public actor CoreAIBackend: DiffusionBackend {
    private let configuration: PipelineConfiguration
    private let textEncoder: AIModel     // TODO(core-ai): 型名を実物に合わせる
    private let unet: AIModel
    private let vaeDecoder: AIModel

    public init(resources: PipelineResources, configuration: PipelineConfiguration) async throws {
        try resources.validateModels()
        self.configuration = configuration
        do {
            // TODO(core-ai): 読み込み API を実物に合わせる。
            // 計算ユニットの指定（ANE 優先など）もここで行う想定。
            self.textEncoder = try await AIModel(contentsOf: resources.textEncoderURL)
            self.unet = try await AIModel(contentsOf: resources.unetURL)
            self.vaeDecoder = try await AIModel(contentsOf: resources.vaeDecoderURL)
        } catch {
            throw IconDiffusionError.unsupportedDevice(
                "Core AI のモデルを読み込めませんでした: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - DiffusionBackend

    public func encodeText(inputIDs: [Int32]) async throws -> Tensor {
        let input = try makeNDArray(values: inputIDs.map(Float.init), shape: [1, inputIDs.count])
        let output = try await run(textEncoder, inputs: ["input_ids": input])
        return try makeTensor(output, expecting: configuration.textEmbeddingShape, context: "encodeText")
    }

    public func predictNoise(
        sample: Tensor,
        timestep: Float,
        encoderHiddenStates: Tensor
    ) async throws -> Tensor {
        let output = try await run(unet, inputs: [
            "sample": try makeNDArray(values: sample.values, shape: sample.shape),
            "timestep": try makeNDArray(values: [timestep], shape: [1]),
            "encoder_hidden_states": try makeNDArray(
                values: encoderHiddenStates.values, shape: encoderHiddenStates.shape
            ),
        ])
        return try makeTensor(output, expecting: sample.shape, context: "predictNoise")
    }

    public func decodeLatent(_ latent: Tensor) async throws -> Tensor {
        let output = try await run(
            vaeDecoder,
            inputs: ["latent": try makeNDArray(values: latent.values, shape: latent.shape)]
        )
        return try makeTensor(output, expecting: configuration.decodedImageShape, context: "decodeLatent")
    }

    // MARK: - Core AI との受け渡し

    /// TODO(core-ai): 推論の呼び出し方と、出力の取り出し方を実物に合わせる。
    /// 入力名（"sample" / "timestep" / "encoder_hidden_states" 等）は
    /// `05_export_coreai.py` のラッパーの引数名に対応する。Xcode のモデルビューアで実際の名前を確認すること。
    private func run(_ model: AIModel, inputs: [String: NDArray]) async throws -> NDArray {
        do {
            let outputs = try await model.prediction(from: inputs)
            guard let first = outputs.values.first else {
                throw IconDiffusionError.backendFailure("モデルが出力を返しませんでした")
            }
            return first
        } catch let error as IconDiffusionError {
            throw error
        } catch {
            throw IconDiffusionError.backendFailure(error.localizedDescription)
        }
    }

    /// TODO(core-ai): NDArray の生成方法を実物に合わせる。
    private func makeNDArray(values: [Float], shape: [Int]) throws -> NDArray {
        do {
            return try NDArray(shape: shape, values: values)
        } catch {
            throw IconDiffusionError.backendFailure("NDArray を作れませんでした: \(error.localizedDescription)")
        }
    }

    /// TODO(core-ai): NDArray から要素を取り出す方法を実物に合わせる。
    /// 形の検証だけはここで必ず行う。ズレたまま先に進むと、絵が壊れる形でしか気づけない。
    private func makeTensor(_ array: NDArray, expecting shape: [Int], context: String) throws -> Tensor {
        let values = array.scalars(of: Float.self)
        return try Tensor(shape: shape, values: values, context: "CoreAIBackend.\(context)")
    }
}

#endif
