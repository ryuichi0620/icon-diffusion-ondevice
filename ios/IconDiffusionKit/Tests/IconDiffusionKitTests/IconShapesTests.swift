import Testing
@testable import IconDiffusionKit

/// 形の整形と採点が、参照実装（training/src/icon_diffusion/shapes.py）と一致することを見る。
///
/// 期待値は同じ入力を Python の `shapes.polish()` に通して得た値をそのまま置いてある。
/// **片方だけ直すと Mac で測った数字と端末の候補選びがずれる**ので、どちらかを触ったら
/// この値も取り直すこと。
struct IconShapesTests {
    /// 128x128 の合成入力: 中央のネイビー正方形 + 消えるべき破片 + 埋まるべき穴
    private func fixture() -> GeneratedImage {
        let size = 128
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        func paint(_ x: Int, _ y: Int, _ colour: (UInt8, UInt8, UInt8), _ alpha: UInt8) {
            let offset = (y * size + x) * 4
            rgba[offset] = colour.0
            rgba[offset + 1] = colour.1
            rgba[offset + 2] = colour.2
            rgba[offset + 3] = alpha
        }
        for y in 32..<96 {
            for x in 32..<96 { paint(x, y, (0x1F, 0x2A, 0x44), 255) }
        }
        for y in 60..<62 {
            for x in 60..<62 { paint(x, y, (0x1F, 0x2A, 0x44), 0) }   // 小さな穴
        }
        for y in 5..<8 {
            for x in 5..<8 { paint(x, y, (0xFF, 0x6B, 0x4A), 255) }   // 3x3 の破片
        }
        return GeneratedImage(width: size, height: size, rgba: rgba)
    }

    @Test func 採点が参照実装と一致する() {
        let polished = IconShapes.polish(fixture(), size: 256)
        let report = polished.report
        #expect(abs(report.coverage - 0.25) < 1e-6)
        #expect(report.parts == 1)
        #expect(abs(report.stray) < 1e-6)
        #expect(abs(report.rough - 1.128379) < 1e-5)
        #expect(abs(report.mush) < 1e-6)
        #expect(abs(report.margin - 0.25) < 1e-6)
        #expect(abs(report.symmetry - 1.0) < 1e-6)
        // **この合成入力は「のっぺりした正方形」なので、粗さの下限を割って減点される。**
        // 1.0 -> 0.659933 に変わったのは 2026-09-24 に粗さを両側の帯にしたため
        // （中身の無い絵が満点になる逆転を直した）。値は参照実装から取り直したもの。
        #expect(abs(report.score - 0.659933) < 1e-5)
    }

    @Test func 破片が消えて穴が埋まる() {
        let polished = IconShapes.polish(fixture(), size: 256)
        let image = polished.image
        func alpha(_ x: Int, _ y: Int) -> UInt8 { image.rgba[(y * image.width + x) * 4 + 3] }
        func rgb(_ x: Int, _ y: Int) -> [UInt8] {
            let offset = (y * image.width + x) * 4
            return [image.rgba[offset], image.rgba[offset + 1], image.rgba[offset + 2]]
        }
        #expect(alpha(0, 0) == 0)                       // 背景は透過のまま
        #expect(alpha(12, 12) == 0)                     // 3x3 の破片は消えている
        #expect(alpha(128, 128) == 255)                 // 本体は残っている
        #expect(rgb(128, 128) == [0x1F, 0x2A, 0x44])    // パレットの色に量子化されている
        #expect(alpha(122, 122) == 255)                 // 小さな穴は埋まっている
    }

    @Test func 輪郭が数画素で立ち上がる() {
        // 二値のままだと階段が残る。濃度場を拡大してから切っているので、
        // 境界は数画素かけて 0 -> 255 へ滑らかに立ち上がる。
        // ここは描き直しだけを見たいので、円クロップ向けの変形は切る
        let image = IconShapes.polish(fixture(), size: 256, fit: false).image
        let column = 128
        let ramp = (60...70).map { image.rgba[($0 * image.width + column) * 4 + 3] }
        #expect(ramp.first == 0)
        #expect(ramp.last == 255)
        #expect(zip(ramp, ramp.dropFirst()).allSatisfy { $0 <= $1 })   // 単調増加
        #expect(ramp.filter { $0 > 0 && $0 < 255 }.count <= 3)         // 立ち上がりは数画素
    }

    @Test func 空の入力でも落ちない() {
        let empty = GeneratedImage(width: 32, height: 32,
                                   rgba: [UInt8](repeating: 0, count: 32 * 32 * 4))
        let polished = IconShapes.polish(empty)
        #expect(polished.report.score == 0)
        #expect(polished.image.rgba.allSatisfy { $0 == 0 })
    }
}

/// 円クロップ向けの整形と背景選びが、参照実装（training/src/icon_diffusion/icon_fit.py）と
/// 一致することを見る。期待値は同じ入力を Python に通して得た値をそのまま置いてある。
struct IconFitTests {
    /// 128x128: 中心からずれたネイビーの矩形（30..99, 20..69）と、角に接する切れ端
    private func fixture() -> GeneratedImage {
        let size = 128
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        func paint(_ x: Int, _ y: Int, _ colour: (UInt8, UInt8, UInt8)) {
            let offset = (y * size + x) * 4
            rgba[offset] = colour.0; rgba[offset + 1] = colour.1
            rgba[offset + 2] = colour.2; rgba[offset + 3] = 255
        }
        for y in 20..<70 { for x in 30..<100 { paint(x, y, (0x1F, 0x2A, 0x44)) } }
        for y in 0..<12 { for x in 0..<12 { paint(x, y, (0xFF, 0x6B, 0x4A)) } }
        return GeneratedImage(width: size, height: size, rgba: rgba)
    }

    @Test func 端に接する切れ端を落とす() {
        let image = fixture()
        let labels = IconShapes.clean(IconShapes.labelMap(image), width: 128, height: 128)
        #expect(labels.filter { $0 >= 0 }.count == 3644)
        let dropped = IconFit.dropScraps(labels, width: 128, height: 128)
        #expect(dropped.filter { $0 >= 0 }.count == 3500)   // 50x70 の矩形だけが残る
    }

    @Test func 変形が参照実装と一致する() {
        let image = fixture()
        let labels = IconShapes.clean(IconShapes.labelMap(image), width: 128, height: 128)
        let dropped = IconFit.dropScraps(labels, width: 128, height: 128)
        let plan = IconFit.plan(dropped, width: 128, height: 128)
        #expect(abs(plan.scale - 1.505051) < 1e-4)
        #expect(abs(plan.centerX - 64.5) < 1e-6)
        #expect(abs(plan.centerY - 44.5) < 1e-6)
    }

    @Test func コントラストの計算がWCAGと一致する() {
        let navy = (r: 0x1F as Double, g: 0x2A as Double, b: 0x44 as Double)
        let coral = (r: 0xFF as Double, g: 0x6B as Double, b: 0x4A as Double)
        let white = (r: 255.0, g: 255.0, b: 255.0)
        #expect(abs(IconFit.contrastRatio(navy, white) - 14.257412) < 1e-4)
        #expect(abs(IconFit.contrastRatio(navy, coral) - 5.060676) < 1e-4)
        // **コーラルと白は 2.82:1 で、WCAG 1.4.11 の 3:1 を満たさない**
        #expect(IconFit.contrastRatio(coral, white) < 3.0)
    }

    @Test func 背景を敷くと角まで不透明になる() {
        let polished = IconShapes.polish(fixture(), size: 256, background: "auto")
        #expect(polished.report.background == "white")
        #expect(abs(polished.report.backgroundContrast - 14.257412) < 1e-4)
        let image = polished.image
        func pixel(_ x: Int, _ y: Int) -> [UInt8] {
            let offset = (y * image.width + x) * 4
            return Array(image.rgba[offset..<(offset + 4)])
        }
        #expect(pixel(0, 0) == [255, 255, 255, 255])          // 角も塗られている
        #expect(pixel(128, 128) == [0x1F, 0x2A, 0x44, 255])   // 中心は主役の色
    }

    @Test func 縮小しても半透明の灰色にならない() {
        // 出力が元より小さいとき、不透明度の傾きが 1 を下回ると全面が半透明になる。
        // Mac 側で実際に出た不具合なので、両方の実装で見張る
        let image = IconShapes.polish(fixture(), size: 48).image
        let alphas = stride(from: 3, to: image.rgba.count, by: 4).map { image.rgba[$0] }
        #expect(alphas.contains(0))
        #expect(alphas.contains(255))
    }
}

/// 色を差し替えても形が変わらないこと、色の選択が参照実装と一致することを見る。
/// 期待値は同じ入力を Python の `shapes.polish(..., theme=...)` に通して得た値。
struct IconThemeTests {
    /// 64x64: ネイビーの正方形（10..53）の中心にコーラルの丸（半径 9）
    private func fixture() -> GeneratedImage {
        let size = 64
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        func paint(_ x: Int, _ y: Int, _ colour: (UInt8, UInt8, UInt8)) {
            let offset = (y * size + x) * 4
            rgba[offset] = colour.0; rgba[offset + 1] = colour.1
            rgba[offset + 2] = colour.2; rgba[offset + 3] = 255
        }
        for y in 10..<54 { for x in 10..<54 { paint(x, y, (0x1F, 0x2A, 0x44)) } }
        for y in 0..<64 {
            for x in 0..<64 where (x - 32) * (x - 32) + (y - 32) * (y - 32) <= 81 {
                paint(x, y, (0xFF, 0x6B, 0x4A))
            }
        }
        return GeneratedImage(width: size, height: size, rgba: rgba)
    }

    private func pixel(_ image: GeneratedImage, _ x: Int, _ y: Int) -> (Int, Int, Int) {
        let offset = (y * image.width + x) * 4
        return (Int(image.rgba[offset]), Int(image.rgba[offset + 1]), Int(image.rgba[offset + 2]))
    }

    @Test func 色を替えても形の点は変わらない() {
        let image = fixture()
        let base = IconShapes.polish(image, size: 64, background: "auto").report
        for theme in [IconShapes.Theme.mono((0x1F, 0x2A, 0x44)),
                      IconShapes.Theme(ink: (20, 83, 45), accent: (245, 183, 0))] {
            let other = IconShapes.polish(image, size: 64, background: "auto", theme: theme).report
            // 整形も採点もラベル番号の上で動くので、色を替えても 1 画素も変わらないはず
            #expect(abs(other.score - base.score) < 1e-9)
            #expect(other.parts == base.parts)
            #expect(abs(other.coverage - base.coverage) < 1e-9)
        }
    }

    @Test func 塗る色が参照実装と一致する() {
        let image = fixture()
        let cases: [(IconShapes.Theme?, (Int, Int, Int), (Int, Int, Int), Double)] = [
            (nil, (31, 42, 68), (255, 107, 74), 14.2574),
            (.mono((0x1F, 0x2A, 0x44)), (31, 42, 68), (31, 42, 68), 14.2574),
            (IconShapes.Theme(ink: (20, 83, 45), accent: (245, 183, 0)),
             (20, 83, 45), (245, 183, 0), 9.1109),
        ]
        for (theme, body, centre, contrast) in cases {
            let out = IconShapes.polish(image, size: 64, background: "auto",
                                        theme: theme ?? .default)
            #expect(out.report.background == "white")
            #expect(abs(out.report.backgroundContrast - contrast) < 1e-3)
            #expect(pixel(out.image, 0, 0) == (255, 255, 255))   // 角は地の色
            #expect(pixel(out.image, 16, 16) == body)
            #expect(pixel(out.image, 32, 32) == centre)
        }
    }

    @Test func 明るさの近い2色は弾かれる() {
        // 2 色は必ず隣り合うので、互いに 3:1 を満たさない組み合わせは使えない
        #expect(!IconShapes.Theme(ink: (37, 99, 235), accent: (59, 130, 246)).isUsable)
        #expect(IconShapes.Theme(ink: (20, 83, 45), accent: (245, 183, 0)).isUsable)
        #expect(IconShapes.Theme.default.isUsable)
        #expect(IconShapes.Theme.mono((37, 99, 235)).isUsable)   // 単色は対象外
    }
}
