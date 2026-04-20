// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LiveKitWakeWord",
    platforms: [
        .iOS(.v16),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "LiveKitWakeWord",
            targets: ["LiveKitWakeWord"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            from: "1.20.0"
        ),
    ],
    targets: [
        .target(
            name: "LiveKitWakeWord",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Sources/LiveKitWakeWord",
            resources: [
                .copy("Resources/melspectrogram.onnx"),
                .copy("Resources/embedding_model.onnx"),
            ]
        ),
        .testTarget(
            name: "LiveKitWakeWordTests",
            dependencies: ["LiveKitWakeWord"],
            path: "Tests/LiveKitWakeWordTests",
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
