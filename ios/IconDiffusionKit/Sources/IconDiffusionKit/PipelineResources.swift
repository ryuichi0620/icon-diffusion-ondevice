import Foundation

/// 生成に必要なファイルの置き場所。
///
/// モデルは合計で GB 級になるので Swift Package には同梱せず、アプリのバンドル
/// （またはダウンロード先のディレクトリ）に置いて、その URL をここで渡す。
/// ファイル名は `05_export_coreai.py` の出力に合わせてある。
public struct PipelineResources: Sendable {
    public var textEncoderURL: URL
    public var unetURL: URL
    public var vaeDecoderURL: URL
    /// 輪郭を条件に与える経路。載せていなければ nil のまま
    public var controlNetURL: URL?
    public var controlledUNetURL: URL?
    public var vocabularyURL: URL
    public var mergesURL: URL
    public var configurationURL: URL

    public init(
        textEncoderURL: URL,
        unetURL: URL,
        vaeDecoderURL: URL,
        vocabularyURL: URL,
        mergesURL: URL,
        configurationURL: URL,
        controlNetURL: URL? = nil,
        controlledUNetURL: URL? = nil
    ) {
        self.textEncoderURL = textEncoderURL
        self.unetURL = unetURL
        self.vaeDecoderURL = vaeDecoderURL
        self.vocabularyURL = vocabularyURL
        self.mergesURL = mergesURL
        self.configurationURL = configurationURL
        self.controlNetURL = controlNetURL
        self.controlledUNetURL = controlledUNetURL
    }

    public static let textEncoderFileName = "IconTextEncoder.aimodel"
    public static let unetFileName = "IconUNet.aimodel"
    public static let vaeDecoderFileName = "IconVAEDecoder.aimodel"
    /// 輪郭を条件に与える経路。**無くても動く**ので、必須の検証には入れない
    public static let controlNetFileName = "IconControlNet.aimodel"
    public static let controlledUNetFileName = "IconUNetControlled.aimodel"
    public static let vocabularyFileName = "vocab.json"
    public static let mergesFileName = "merges.txt"
    public static let configurationFileName = "pipeline_config.json"

    /// 1 つのディレクトリに Stage 5 の出力をそのまま置いた場合。
    public static func directory(_ directory: URL) -> PipelineResources {
        PipelineResources(
            textEncoderURL: directory.appendingPathComponent(textEncoderFileName),
            unetURL: directory.appendingPathComponent(unetFileName),
            vaeDecoderURL: directory.appendingPathComponent(vaeDecoderFileName),
            vocabularyURL: directory.appendingPathComponent(vocabularyFileName),
            mergesURL: directory.appendingPathComponent(mergesFileName),
            configurationURL: directory.appendingPathComponent(configurationFileName),
            controlNetURL: directory.appendingPathComponent(controlNetFileName),
            controlledUNetURL: directory.appendingPathComponent(controlledUNetFileName)
        )
    }

    /// アプリのバンドルに直接入れた場合。
    public static func bundled(in bundle: Bundle) throws -> PipelineResources {
        func locate(_ name: String) throws -> URL {
            let base = (name as NSString).deletingPathExtension
            let ext = (name as NSString).pathExtension
            guard let url = bundle.url(forResource: base, withExtension: ext) else {
                throw IconDiffusionError.missingResource(
                    bundle.bundleURL.appendingPathComponent(name)
                )
            }
            return url
        }
        return PipelineResources(
            textEncoderURL: try locate(textEncoderFileName),
            unetURL: try locate(unetFileName),
            vaeDecoderURL: try locate(vaeDecoderFileName),
            vocabularyURL: try locate(vocabularyFileName),
            mergesURL: try locate(mergesFileName),
            configurationURL: try locate(configurationFileName)
        )
    }

    /// トークナイザと設定だけを検証する。バックエンドを差し替えて疎通確認するときに使う。
    public func validateTokenizer() throws {
        try Self.require(vocabularyURL)
        try Self.require(mergesURL)
    }

    public func validateModels() throws {
        try Self.require(textEncoderURL)
        try Self.require(unetURL)
        try Self.require(vaeDecoderURL)
    }

    private static func require(_ url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw IconDiffusionError.missingResource(url)
        }
    }
}
