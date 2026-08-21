import Foundation

public enum IconDiffusionError: Error, Sendable, Equatable {
    /// Core AI が使えない端末（Apple Intelligence 非対応機など）。アプリ側で逃げ道に分岐する。
    case unsupportedDevice(String)
    case missingResource(URL)
    case invalidConfiguration(String)
    case shapeMismatch(expected: [Int], actual: [Int], context: String)
    case backendFailure(String)
    case tokenizerFailure(String)
}

extension IconDiffusionError: LocalizedError, CustomStringConvertible {
    public var description: String {
        switch self {
        case .unsupportedDevice(let detail):
            return "この端末では Core AI を使えません: \(detail)"
        case .missingResource(let url):
            return "必要なファイルが見つかりません: \(url.path)"
        case .invalidConfiguration(let detail):
            return "設定が不正です: \(detail)"
        case .shapeMismatch(let expected, let actual, let context):
            return "テンソルの形が合いません (\(context)): expected \(expected), actual \(actual)"
        case .backendFailure(let detail):
            return "モデルの実行に失敗しました: \(detail)"
        case .tokenizerFailure(let detail):
            return "トークナイズに失敗しました: \(detail)"
        }
    }

    public var errorDescription: String? { description }
}
