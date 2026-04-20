// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import XCTest

@testable import LiveKitWakeWord

/// Micro-benchmarks for ``WakeWordModel/predict(_:)`` across execution
/// providers. These are expensive, so they're guarded by the
/// ``LIVEKIT_WAKEWORD_BENCH`` environment variable — set it to anything
/// truthy when you want to run them:
///
/// ```
/// LIVEKIT_WAKEWORD_BENCH=1 swift test --filter BenchmarkTests
/// ```
final class BenchmarkTests: XCTestCase {

    private func fixtureURL(_ name: String) throws -> URL {
        if let url = Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures") {
            return url
        }
        if let url = Bundle.module.url(forResource: name, withExtension: nil) {
            return url
        }
        throw WakeWordError.bundledResourceMissing(name: name)
    }

    private func skipUnlessBenchmarkEnabled() throws {
        if ProcessInfo.processInfo.environment["LIVEKIT_WAKEWORD_BENCH"] == nil {
            throw XCTSkip("Set LIVEKIT_WAKEWORD_BENCH=1 to run predict benchmarks")
        }
    }

    func testPredictPerformanceCoreML() throws {
        try skipUnlessBenchmarkEnabled()
        try runBenchmark(provider: .coreML, label: "coreML")
    }

    func testPredictPerformanceCoreMLCPUAndGPU() throws {
        try skipUnlessBenchmarkEnabled()
        try runBenchmark(provider: .coreMLCPUAndGPU, label: "coreMLCPUAndGPU")
    }

    func testPredictPerformanceCPU() throws {
        try skipUnlessBenchmarkEnabled()
        try runBenchmark(provider: .cpu, label: "cpu")
    }

    private func runBenchmark(provider: ExecutionProvider, label: String) throws {
        let classifier = try fixtureURL("hey_livekit.onnx")
        let model = try WakeWordModel(
            models: [classifier],
            sampleRate: 16_000,
            executionProvider: provider
        )

        // 2-second audio chunk matches the usual window size fed by the
        // listener path.
        let audio = Array(repeating: Int16.zero, count: 32_000)

        // Warm up to JIT-compile kernels / move weights to the ANE.
        for _ in 0..<10 {
            _ = try model.predict(audio)
        }

        let iterations = 200
        var timings = [Double]()
        timings.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try model.predict(audio)
            timings.append(CFAbsoluteTimeGetCurrent() - start)
        }
        timings.sort()

        func ms(_ i: Int) -> Double { timings[i] * 1000 }
        let p50 = ms(iterations / 2)
        let p95 = ms(Int(Double(iterations) * 0.95))
        let p99 = ms(Int(Double(iterations) * 0.99))
        print(String(format: "[bench] %@: p50=%.3f ms  p95=%.3f ms  p99=%.3f ms", label, p50, p95, p99))
    }
}
