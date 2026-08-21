import Foundation

/// CLIP の byte-level BPE トークナイザ。
///
/// テキストエンコーダに渡す `input_ids` を作る。企画メモが「難所」に挙げた
/// 「3 モデル＋ループ＋トークナイザ」のトークナイザ部分がこれ。
///
/// HF の `tokenizer/vocab.json`（token -> id）と `tokenizer/merges.txt`（マージ規則）を読む。
/// 両ファイルは Stage 5 の `05_export_coreai.py` がベースモデルから書き出す。
///
/// 分割は正規表現ではなく手書きの走査で行っている。CLIP のパターンは
/// `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
/// で、選択肢の優先順をそのまま順番に試せば同じ結果になる。正規表現エンジンの
/// Unicode プロパティ解釈に結果が左右されないほうが、学習側 (Python) と揃えやすい。
public struct CLIPTokenizer: Sendable {
    private struct BytePair: Hashable, Sendable {
        let first: String
        let second: String
    }

    public let bosTokenID: Int
    public let eosTokenID: Int
    public let maxLength: Int
    public var vocabularySize: Int { encoder.count }

    private let encoder: [String: Int]
    private let bpeRanks: [BytePair: Int]
    private let byteEncoder: [UInt8: Character]

    // MARK: - 読み込み

    public init(
        vocabularyURL: URL,
        mergesURL: URL,
        maxLength: Int = 77,
        bosToken: String = "<|startoftext|>",
        eosToken: String = "<|endoftext|>"
    ) throws {
        guard FileManager.default.fileExists(atPath: vocabularyURL.path) else {
            throw IconDiffusionError.missingResource(vocabularyURL)
        }
        guard FileManager.default.fileExists(atPath: mergesURL.path) else {
            throw IconDiffusionError.missingResource(mergesURL)
        }
        guard maxLength >= 2 else {
            throw IconDiffusionError.invalidConfiguration("maxLength は 2 以上にしてください: \(maxLength)")
        }

        let vocabularyData = try Data(contentsOf: vocabularyURL)
        let encoder = try JSONDecoder().decode([String: Int].self, from: vocabularyData)

        guard let bos = encoder[bosToken], let eos = encoder[eosToken] else {
            throw IconDiffusionError.tokenizerFailure(
                "vocab.json に特殊トークンがありません: \(bosToken) / \(eosToken)"
            )
        }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var ranks: [BytePair: Int] = [:]
        var rank = 0
        for rawLine in mergesText.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") { continue }   // 先頭の "#version: 0.2" を飛ばす
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            ranks[BytePair(first: String(parts[0]), second: String(parts[1]))] = rank
            rank += 1
        }
        guard !ranks.isEmpty else {
            throw IconDiffusionError.tokenizerFailure("merges.txt からマージ規則を読めませんでした")
        }

        self.encoder = encoder
        self.bpeRanks = ranks
        self.byteEncoder = Self.makeByteEncoder()
        self.bosTokenID = bos
        self.eosTokenID = eos
        self.maxLength = maxLength
    }

    // MARK: - 公開 API

    /// 特殊トークンもパディングも付けない生のトークン列。テストと調査用。
    public func encode(_ text: String) throws -> [Int] {
        var ids: [Int] = []
        for rawToken in Self.split(text.lowercased()) {
            for piece in bpe(byteEncoded(rawToken)) {
                guard let id = encoder[piece] else {
                    throw IconDiffusionError.tokenizerFailure(
                        "vocab に無いピースが出ました: \(piece)（token=\(rawToken)）。"
                        + "vocab.json と merges.txt が同じモデル由来か確認してください"
                    )
                }
                ids.append(id)
            }
        }
        return ids
    }

    /// テキストエンコーダにそのまま渡せる `[maxLength]` の入力。
    /// BOS を付け、`maxLength - 1` で切り詰め、EOS を付けてから EOS でパディングする
    /// （SD1.5 の CLIPTokenizer は pad_token が `<|endoftext|>`）。
    public func tokenize(_ text: String) throws -> [Int32] {
        var ids = [bosTokenID]
        ids.append(contentsOf: try encode(text))
        if ids.count > maxLength - 1 {
            ids = Array(ids.prefix(maxLength - 1))
        }
        ids.append(eosTokenID)
        if ids.count < maxLength {
            ids.append(contentsOf: repeatElement(eosTokenID, count: maxLength - ids.count))
        }
        return ids.map(Int32.init)
    }

    // MARK: - 分割

    /// CLIP の正規表現と同じ優先順で 1 文字ずつ走査する。
    static func split(_ text: String) -> [String] {
        let scalars = Array(text.unicodeScalars)
        var tokens: [String] = []
        var index = 0

        while index < scalars.count {
            // 1. 特殊トークンのリテラル
            if let special = matchSpecial(scalars, at: index) {
                tokens.append(special)
                index += special.unicodeScalars.count
                continue
            }
            // 2. 短縮形（'s / 't / 're / 've / 'm / 'll / 'd）
            if let contraction = matchContraction(scalars, at: index) {
                tokens.append(contraction)
                index += contraction.unicodeScalars.count
                continue
            }

            let scalar = scalars[index]
            if Self.isWhitespace(scalar) {
                index += 1
                continue
            }
            // 3. 文字の連続
            if Self.isLetter(scalar) {
                var end = index
                while end < scalars.count, Self.isLetter(scalars[end]) { end += 1 }
                tokens.append(String(String.UnicodeScalarView(scalars[index..<end])))
                index = end
                continue
            }
            // 4. 数字は 1 文字ずつ
            if Self.isNumber(scalar) {
                tokens.append(String(scalar))
                index += 1
                continue
            }
            // 5. 空白でも文字でも数字でもないものの連続
            var end = index
            while end < scalars.count,
                  !Self.isWhitespace(scalars[end]),
                  !Self.isLetter(scalars[end]),
                  !Self.isNumber(scalars[end]) {
                end += 1
            }
            tokens.append(String(String.UnicodeScalarView(scalars[index..<end])))
            index = end
        }
        return tokens
    }

    private static let specialTokens = ["<|startoftext|>", "<|endoftext|>"]
    private static let contractions = ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"]

    private static func matchSpecial(_ scalars: [UnicodeScalar], at index: Int) -> String? {
        matchAny(specialTokens, scalars, at: index)
    }

    private static func matchContraction(_ scalars: [UnicodeScalar], at index: Int) -> String? {
        guard scalars[index] == "'" else { return nil }
        // 長いものから試す。'll を 'l と取り違えないため（contractions は長さ順に並べてある）
        return matchAny(contractions, scalars, at: index)
    }

    private static func matchAny(_ candidates: [String], _ scalars: [UnicodeScalar], at index: Int) -> String? {
        for candidate in candidates {
            let candidateScalars = Array(candidate.unicodeScalars)
            guard index + candidateScalars.count <= scalars.count else { continue }
            if Array(scalars[index..<(index + candidateScalars.count)]) == candidateScalars {
                return candidate
            }
        }
        return nil
    }

    private static func isWhitespace(_ scalar: UnicodeScalar) -> Bool {
        scalar.properties.isWhitespace
    }

    private static func isLetter(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            return true   // \p{L}
        default:
            return false
        }
    }

    private static func isNumber(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber:
            return true   // \p{N}
        default:
            return false
        }
    }

    // MARK: - バイト -> 疑似文字

    /// GPT-2 / CLIP の `bytes_to_unicode()`。任意のバイト列を「空白を含まない印字可能な文字列」に写す。
    static func makeByteEncoder() -> [UInt8: Character] {
        let base = Array(33...126) + Array(161...172) + Array(174...255)
        let baseSet = Set(base)

        var mapping: [UInt8: Character] = [:]
        for value in base {
            mapping[UInt8(value)] = Character(UnicodeScalar(UInt32(value))!)
        }
        var offset = 0
        for value in 0...255 where !baseSet.contains(value) {
            mapping[UInt8(value)] = Character(UnicodeScalar(UInt32(256 + offset))!)
            offset += 1
        }
        return mapping
    }

    private func byteEncoded(_ token: String) -> String {
        var result = ""
        result.reserveCapacity(token.utf8.count)
        for byte in token.utf8 {
            if let mapped = byteEncoder[byte] {
                result.append(mapped)
            }
        }
        return result
    }

    // MARK: - BPE

    /// 1 トークンをマージ規則に従って分解する。
    ///
    /// **毎回いちばんランクの低いペアを選ぶ**のが肝で、左から貪欲にマージすると結果が変わる。
    /// 例: マージ表に `c o`(rank 2) と `i c`(rank 16) があるとき、"icon" は
    /// `["i", "co", "n</w>"]` になる（`["icon</w>"]` にはならない）。
    private func bpe(_ token: String) -> [String] {
        var word = token.unicodeScalars.map { String($0) }
        guard let last = word.last else { return [] }
        word[word.count - 1] = last + "</w>"
        guard word.count > 1 else { return word }

        while true {
            var bestRank = Int.max
            var bestPair: BytePair?
            for index in 0..<(word.count - 1) {
                let pair = BytePair(first: word[index], second: word[index + 1])
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            guard let pair = bestPair else { break }

            // 選んだペアの出現箇所を左から順に（重ならないように）すべてマージする
            var merged: [String] = []
            merged.reserveCapacity(word.count)
            var index = 0
            while index < word.count {
                if index < word.count - 1, word[index] == pair.first, word[index + 1] == pair.second {
                    merged.append(pair.first + pair.second)
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }
            word = merged
            if word.count == 1 { break }
        }
        return word
    }
}
