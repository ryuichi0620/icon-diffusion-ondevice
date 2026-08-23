// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "IconDiffusionKit",
    // Core AI 依存は CoreAIBackend.swift に閉じ込め、#if canImport(CoreAI) で切っている。
    // ここの下限はスケジューラ・トークナイザ（純 Swift）が動く範囲に合わせてある。
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "IconDiffusionKit", targets: ["IconDiffusionKit"])
    ],
    targets: [
        .target(
            name: "IconDiffusionKit",
            swiftSettings: [
                // CoreAIBackend.swift を有効にする。前段に #if canImport(CoreAI) があるので、
                // Core AI の無い環境（iOS 26 以前・素の macOS）では従来どおり除外される。
                //
                // 以前は「中身が WWDC26 の実 API に対して未検証」という理由で既定 off に
                // していたが、2026-08-23 に iOS 27.0 SDK に対して型チェックを通したので
                // その前提は無くなった。実機での動作確認はまだ（docs/architecture.md 6.6 節）。
                .define("ICON_DIFFUSION_COREAI")
            ]
        ),
        .testTarget(
            name: "IconDiffusionKitTests",
            dependencies: ["IconDiffusionKit"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
