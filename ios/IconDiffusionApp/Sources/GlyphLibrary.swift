import Foundation
import IconDiffusionKit

/// 被写体から輪郭グリフを引き当てる。**形は外から、スタイルはモデルから。**
///
/// なぜ端末に判定器を載せないのか（docs/architecture.md「引き当てをどこで動かすか」）:
/// Mac 側は SigLIP で「被写体の文章 → グリフの絵」を引いているが、その テキスト側は
/// **282M パラメータ**（うち 197M が語彙 25.6 万の埋め込み表）でトークナイザも
/// SentencePiece。端末に載せるには大きすぎる。CLIP で代用する道も測って潰した
/// （1 位の一致が 40 件中 5 件）。
///
/// そこで **Mac 側で先に全部引いておいた表**を同梱し、端末は文字列で引くだけにする。
/// 引き当ての質は SigLIP のまま、表に無い語だけが「条件なし」に落ちる。
struct GlyphLibrary {
    /// 語 → グリフ名。Mac 側で関門 2 段（絵 0.14 / 名前 0.90）を通ったものだけが入っている
    private let table: [String: String]
    private let directory: URL

    /// 補完に出す候補。利用者が「引ける語」を選べるようにするための一覧
    let words: [String]

    init?(directory: URL) {
        let archive = directory.appendingPathComponent("table.tsv")
        guard let text = try? String(contentsOf: archive, encoding: .utf8) else { return nil }
        var table: [String: String] = [:]
        table.reserveCapacity(24_000)
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2 else { continue }
            table[String(parts[0])] = String(parts[1])
        }
        guard !table.isEmpty else { return nil }
        self.table = table
        self.directory = directory
        self.words = table.keys.sorted()
    }

    /// 被写体に対応するグリフ名。引けなければ nil（＝条件を付けない）。
    ///
    /// **語の組み合わせは弱い。** 「a cactus in a pot」が `pot` に引かれて `cooking` に
    /// なるような取り違えが起きる（Mac で直接引けば `potted_plant`）。表に入っている
    /// 語句と単語の範囲では正確なので、入力欄には補完を出して選ばせるのが素直。
    func resolve(_ subject: String) -> String? {
        let normalized = subject.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        for key in [normalized, Self.dropArticle(normalized)] where !key.isEmpty {
            if let name = table[key] { return name }
        }
        // 語ごとに引く。長い語ほど被写体を特定しているとみなす
        let tokens = Self.dropArticle(normalized)
            .split(whereSeparator: { $0 == " " || $0 == "-" })
            .map(String.init)
            .filter { $0.count >= 3 }
            .sorted { $0.count > $1.count }
        for token in tokens {
            if let name = table[token] { return name }
        }
        return nil
    }

    /// 入力の途中から候補を出す。**表に無い語を打たせないための仕掛け。**
    func suggestions(for prefix: String, limit: Int = 8) -> [String] {
        let key = Self.dropArticle(prefix.lowercased().trimmingCharacters(in: .whitespaces))
        guard key.count >= 2 else { return [] }
        var starts: [String] = []
        var contains: [String] = []
        for word in words {
            if word.hasPrefix(key) {
                starts.append(word)
                if starts.count >= limit { break }
            } else if contains.count < limit, word.contains(key) {
                contains.append(word)
            }
        }
        return Array((starts + contains).prefix(limit))
    }

    /// グリフの輪郭を、ControlNet に渡す `[1, 3, size, size]` にする。
    ///
    /// **白線・黒地に反転する**（ControlNet 系はその向きを期待している）。境界は
    /// Canny ではなく形態学的に取る — 入力が既に二値なので、しきい値調整もノイズも要らず
    /// 1〜2 画素の綺麗な線になる。
    func controlTensor(for glyph: String, size: Int, lineWidth: Int = 2) -> Tensor? {
        let url = directory.appendingPathComponent("\(glyph).png")
        guard let image = GlyphRaster.load(url, size: size) else { return nil }
        let edge = GlyphRaster.outline(image, width: size, height: size, lineWidth: lineWidth)
        var values = [Float](repeating: 0, count: 3 * size * size)
        for index in 0..<(size * size) where edge[index] {
            values[index] = 1
            values[size * size + index] = 1
            values[2 * size * size + index] = 1
        }
        return Tensor(shape: [1, 3, size, size], values: values)
    }

    private static func dropArticle(_ text: String) -> String {
        for article in ["a ", "an ", "the "] where text.hasPrefix(article) {
            return String(text.dropFirst(article.count))
        }
        return text
    }
}
