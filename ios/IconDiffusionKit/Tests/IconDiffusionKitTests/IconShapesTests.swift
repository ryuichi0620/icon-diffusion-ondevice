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
        #expect(abs(report.score - 1.0) < 1e-6)
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
        // 境界は数画素かけて 0 -> 255 へ滑らかに立ち上がる
        let image = IconShapes.polish(fixture(), size: 256).image
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
