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
        .target(name: "IconDiffusionKit"),
        .testTarget(
            name: "IconDiffusionKitTests",
            dependencies: ["IconDiffusionKit"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
