import Foundation
import Testing

@testable import IconDiffusionKit

/// `Fixtures/expected_tokens.json` の型。
struct TokenFixture: Decodable {
    struct Case: Decodable {
        let text: String
        let ids: [Int]
        let paddedHead: [Int]
        let paddedLength: Int
        enum CodingKeys: String, CodingKey {
            case text, ids
            case paddedHead = "padded_head"
            case paddedLength = "padded_len"
        }
    }

    struct Truncation: Decodable {
        let rawLength: Int
        let paddedLength: Int
        let lastID: Int
        let head: [Int]
        enum CodingKeys: String, CodingKey {
            case rawLength = "raw_len"
            case paddedLength = "padded_len"
            case lastID = "last_id"
            case head
        }
    }

    let bos: Int
    let eos: Int
    let maxLength: Int
    let vocabularySize: Int
    let cases: [Case]
    let truncation: Truncation

    enum CodingKeys: String, CodingKey {
        case bos, eos, cases, truncation
        case maxLength = "max_length"
        case vocabularySize = "vocab_size"
    }
}

@Suite("CLIP トークナイザ")
struct CLIPTokenizerTests {
    private func makeTokenizer(maxLength: Int = 77) throws -> CLIPTokenizer {
        try CLIPTokenizer(
            vocabularyURL: try Fixtures.url("vocab.json"),
            mergesURL: try Fixtures.url("merges.txt"),
            maxLength: maxLength
        )
    }

    // MARK: - 分割

    /// CLIP の正規表現
    /// `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
    /// と同じ結果になること。期待値は Python の `regex` モジュール（ICU と同じ Unicode プロパティ）
    /// で実際に照合して得たもの。
    /// 分割の期待値。**型を明示してあるのは推論を軽くするため。**
    /// 素のリテラルのままだと、タプルと入れ子配列の推論が重く、機械の負荷次第で
    /// 「the compiler is unable to type-check this expression in reasonable time」で
    /// ビルドが落ちる（2026-09-24 に実際に落ちた）。中身は変えていない。
    static let splitCases: [(String, [String])] = [
            (
                "an icon in ICONST style, a coffee cup, flat, minimal",
                ["an", "icon", "in", "iconst", "style", ",", "a", "coffee", "cup", ",", "flat", ",", "minimal"]
            ),
            ("hello!!", ["hello", "!!"]),
            ("abc 123", ["abc", "1", "2", "3"]),                          // 数字は 1 文字ずつ
            ("a_b", ["a", "_", "b"]),                                     // _ は記号側
            ("it's a cat's toy", ["it", "'s", "a", "cat", "'s", "toy"]),
            (
                "we'll go, they've gone, I'm here, you'd know, don't",
                ["we", "'ll", "go", ",", "they", "'ve", "gone", ",", "i", "'m", "here", ",",
                 "you", "'d", "know", ",", "don", "'t"]
            ),
            ("  Multiple    SPACES  ", ["multiple", "spaces"]),           // 小文字化と空白畳み込み
            ("...!!??", ["...!!??"]),                                     // 記号は連続でひとまとまり
            ("café naïve", ["café", "naïve"]),                            // 非 ASCII の文字も \p{L}
            ("tab\tand\nnewline", ["tab", "and", "newline"]),
            ("3.14 and -2", ["3", ".", "1", "4", "and", "-", "2"]),
            ("", []),
    ]

    @Test("分割が CLIP の正規表現と一致する", arguments: splitCases)
    func splitMatchesCLIPPattern(input: String, expected: [String]) {
        #expect(CLIPTokenizer.split(input.lowercased()) == expected)
    }

    // MARK: - バイト -> 疑似文字

    @Test("bytes_to_unicode が 256 バイトすべてを写す")
    func byteEncoderCoversAllBytes() {
        let encoder = CLIPTokenizer.makeByteEncoder()
        #expect(encoder.count == 256)
        #expect(Set(encoder.values).count == 256, "写像先が衝突しています")
        // GPT-2 / CLIP で知られている代表値
        #expect(encoder[97] == "a")
        #expect(encoder[32] == "Ġ")
        #expect(encoder[0] == "Ā")
        #expect(encoder[255] == "ÿ")
    }

    // MARK: - BPE

    @Test("BPE は毎回いちばんランクの低いペアを選ぶ")
    func bpePicksLowestRankPair() throws {
        let tokenizer = try makeTokenizer()
        // フィクスチャのマージ表は `c o`(rank 2) が `i c`(rank 16) より先。
        // 左から貪欲にマージする実装だと "icon</w>" 1 個になってしまい、ここで落ちる。
        #expect(try tokenizer.encode("icon") == [72, 514, 333])   // i / co / n</w>
        // 一方 "coffee" と "cup" はマージ連鎖が最後まで通って 1 トークンになる
        #expect(try tokenizer.encode("coffee") == [518])
        #expect(try tokenizer.encode("cup") == [520])
    }

    // MARK: - 参照実装との突き合わせ

    @Test("エンコード結果が参照実装と一致する")
    func encodeMatchesReference() throws {
        let fixture = try Fixtures.decode(TokenFixture.self, from: "expected_tokens.json")
        let tokenizer = try makeTokenizer(maxLength: fixture.maxLength)

        #expect(tokenizer.bosTokenID == fixture.bos)
        #expect(tokenizer.eosTokenID == fixture.eos)
        #expect(tokenizer.vocabularySize == fixture.vocabularySize)

        for testCase in fixture.cases {
            #expect(try tokenizer.encode(testCase.text) == testCase.ids, "encode: \(testCase.text)")

            let padded = try tokenizer.tokenize(testCase.text).map(Int.init)
            #expect(padded.count == testCase.paddedLength, "長さ: \(testCase.text)")
            #expect(Array(padded.prefix(16)) == testCase.paddedHead, "tokenize: \(testCase.text)")
        }
    }

    @Test("BOS / EOS / パディングの付き方")
    func specialTokenLayout() throws {
        let fixture = try Fixtures.decode(TokenFixture.self, from: "expected_tokens.json")
        let tokenizer = try makeTokenizer(maxLength: fixture.maxLength)

        let ids = try tokenizer.tokenize("a coffee cup").map(Int.init)
        #expect(ids.first == fixture.bos)
        #expect(Array(ids[1...3]) == [320, 518, 520])
        // 残りはすべて EOS（SD1.5 の CLIPTokenizer は pad_token が <|endoftext|>）
        #expect(ids[4...].allSatisfy { $0 == fixture.eos })
    }

    @Test("空文字列は BOS + EOS + パディングだけになる")
    func emptyPromptIsValid() throws {
        let tokenizer = try makeTokenizer()
        let ids = try tokenizer.tokenize("").map(Int.init)
        #expect(ids.count == 77)
        #expect(ids[0] == tokenizer.bosTokenID)
        #expect(ids[1...].allSatisfy { $0 == tokenizer.eosTokenID })
    }

    @Test("長すぎる入力は maxLength に切り詰められ、末尾は必ず EOS")
    func longPromptIsTruncated() throws {
        let fixture = try Fixtures.decode(TokenFixture.self, from: "expected_tokens.json")
        let tokenizer = try makeTokenizer(maxLength: fixture.maxLength)
        let longPrompt = Array(repeating: "coffee cup flat minimal", count: 40).joined(separator: " ")

        #expect(try tokenizer.encode(longPrompt).count == fixture.truncation.rawLength)

        let ids = try tokenizer.tokenize(longPrompt).map(Int.init)
        #expect(ids.count == fixture.truncation.paddedLength)
        #expect(ids.last == fixture.truncation.lastID, "切り詰めても末尾は EOS")
        #expect(Array(ids.prefix(8)) == fixture.truncation.head)
    }

    // MARK: - 異常系

    @Test("資産が無ければ missingResource で落ちる")
    func missingResourcesThrow() throws {
        let missing = URL(fileURLWithPath: "/tmp/icon-diffusion-does-not-exist.json")
        #expect(throws: IconDiffusionError.self) {
            _ = try CLIPTokenizer(vocabularyURL: missing, mergesURL: try Fixtures.url("merges.txt"))
        }
        #expect(throws: IconDiffusionError.self) {
            _ = try CLIPTokenizer(vocabularyURL: try Fixtures.url("vocab.json"), mergesURL: missing)
        }
    }
}
