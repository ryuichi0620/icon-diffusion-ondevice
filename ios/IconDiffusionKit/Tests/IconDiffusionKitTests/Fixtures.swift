import Foundation
import Testing

/// テスト用の期待値は `Fixtures/` に JSON で置いてある。
///
/// 中身は Python 側の参照実装（diffusers の LCMScheduler / openai の CLIP tokenizer と
/// 同じ式）で生成したもの。Swift 実装がそれと一致することを確かめるためのオラクル。
enum Fixtures {
    static func url(_ name: String) throws -> URL {
        guard let base = Bundle.module.resourceURL else {
            throw FixtureError.noResourceBundle
        }
        let url = base.appendingPathComponent("Fixtures").appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FixtureError.missing(url)
        }
        return url
    }

    static func decode<T: Decodable>(_ type: T.Type, from name: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: try url(name)))
    }

    enum FixtureError: Error, CustomStringConvertible {
        case missing(URL)
        case noResourceBundle

        var description: String {
            switch self {
            case .missing(let url): return "フィクスチャがありません: \(url.path)"
            case .noResourceBundle: return "テストバンドルに resourceURL がありません"
            }
        }
    }
}

/// 浮動小数の比較。相対誤差と絶対誤差の大きいほうを許容する。
///
/// 参照実装は Double、Swift 実装は Float で計算しているので、桁の大きい値では
/// 絶対誤差だけ見ても意味がない。
func isClose(_ lhs: Double, _ rhs: Double, relative: Double = 1e-4, absolute: Double = 1e-6) -> Bool {
    abs(lhs - rhs) <= max(absolute, relative * max(abs(lhs), abs(rhs)))
}

func expectClose(
    _ lhs: [Float],
    _ rhs: [Double],
    relative: Double = 1e-4,
    absolute: Double = 1e-6,
    _ label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(lhs.count == rhs.count, "\(label): 要素数が違います", sourceLocation: sourceLocation)
    guard lhs.count == rhs.count else { return }
    for index in lhs.indices where !isClose(Double(lhs[index]), rhs[index], relative: relative, absolute: absolute) {
        Issue.record(
            "\(label)[\(index)]: \(lhs[index]) != \(rhs[index])",
            sourceLocation: sourceLocation
        )
    }
}
