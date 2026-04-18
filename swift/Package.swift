// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LiveKitWakeWord",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "LiveKitWakeWord",
            targets: ["LiveKitWakeWord"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "LiveKitWakeWord",
            path: "Sources/LiveKitWakeWord",
            resources: [
                // Xcode 15+ is buggy at `.process`-ing .mlpackage directories
                // from inside a Swift package (it collapses nested
                // ``Manifest.json`` / ``coremldata.bin`` and produces
                // duplicate-resource errors). Use ``.copy`` so the whole
                // package directory is shipped verbatim, and compile it at
                // first use via ``MLModel.compileModel(at:)``.
                .copy("Resources/melspectrogram.mlpackage"),
                .copy("Resources/embedding_model.mlpackage"),
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
