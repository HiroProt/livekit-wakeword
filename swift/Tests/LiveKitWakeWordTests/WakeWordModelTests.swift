// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import AVFoundation
import Foundation
import XCTest

@testable import LiveKitWakeWord

/// Mirrors `rust-sdks/livekit-wakeword/tests/integration.rs`: sanity pipeline,
/// positive-wav-above-threshold, negative-wav-below-threshold. The fixtures
/// (hey_livekit.onnx, positive.wav, negative.wav) are copied from the
/// Rust crate's `tests/fixtures/` directory.
final class WakeWordModelTests: XCTestCase {
    private static let threshold: Float = 0.5

    private func fixtureURL(_ name: String) throws -> URL {
        let bundle = Bundle.module
        // The Fixtures directory is declared as a .copy resource, so it
        // survives verbatim in Bundle.module with a stable path.
        if let url = bundle.url(forResource: name, withExtension: nil, subdirectory: "Fixtures") {
            return url
        }
        if let url = bundle.url(forResource: name, withExtension: nil) {
            return url
        }
        throw WakeWordError.bundledResourceMissing(name: name)
    }

    func testPredictPipelineShape() throws {
        let classifier = try fixtureURL("hey_livekit.onnx")
        let model = try WakeWordModel(
            models: [classifier],
            sampleRate: 16_000,
            executionProvider: .cpu
        )

        let sine = Self.generateSine(freq: 440, sampleRate: 16_000, duration: 2.0)
        let predictions = try model.predict(sine)
        XCTAssertNotNil(predictions["hey_livekit"])
        if let score = predictions["hey_livekit"] {
            XCTAssertTrue((0.0...1.0).contains(score))
        }

        // Too-short audio returns zero (matches Rust + Python semantics).
        let tiny = Self.generateSine(freq: 440, sampleRate: 16_000, duration: 0.1)
        let tinyPredictions = try model.predict(tiny)
        XCTAssertEqual(tinyPredictions["hey_livekit"], 0.0)
    }

    func testPositiveWavAboveThreshold() throws {
        let classifier = try fixtureURL("hey_livekit.onnx")
        let wav = try fixtureURL("positive.wav")
        let (sampleRate, samples) = try Self.readWav(url: wav)

        let model = try WakeWordModel(
            models: [classifier],
            sampleRate: sampleRate,
            executionProvider: .cpu
        )

        let score = try model.predict(samples)["hey_livekit"] ?? 0
        XCTAssertGreaterThanOrEqual(
            score, Self.threshold,
            "positive.wav score (\(score)) should be >= threshold (\(Self.threshold))"
        )
    }

    func testNegativeWavBelowThreshold() throws {
        let classifier = try fixtureURL("hey_livekit.onnx")
        let wav = try fixtureURL("negative.wav")
        let (sampleRate, samples) = try Self.readWav(url: wav)

        let model = try WakeWordModel(
            models: [classifier],
            sampleRate: sampleRate,
            executionProvider: .cpu
        )

        let score = try model.predict(samples)["hey_livekit"] ?? 0
        XCTAssertLessThan(
            score, Self.threshold,
            "negative.wav score (\(score)) should be < threshold (\(Self.threshold))"
        )
    }

    func testLoadAndUnloadClassifier() throws {
        let classifier = try fixtureURL("hey_livekit.onnx")
        let model = try WakeWordModel(models: [], sampleRate: 16_000, executionProvider: .cpu)
        XCTAssertEqual(model.modelNames.count, 0)

        try model.loadModel(url: classifier, name: "custom")
        XCTAssertEqual(model.modelNames, ["custom"])

        let audio = Self.generateSine(freq: 440, sampleRate: 16_000, duration: 2.0)
        let scores = try model.predict(audio)
        XCTAssertNotNil(scores["custom"])
        XCTAssertNil(scores["hey_livekit"])

        XCTAssertTrue(model.unloadModel(name: "custom"))
        XCTAssertEqual(model.modelNames.count, 0)
    }

    // MARK: - Helpers

    private static func generateSine(freq: Double, sampleRate: Int, duration: Double) -> [Int16] {
        let count = Int(Double(sampleRate) * duration)
        var out = [Int16](repeating: 0, count: count)
        let twoPi = 2.0 * Double.pi
        for i in 0..<count {
            let t = Double(i) / Double(sampleRate)
            out[i] = Int16(clamping: Int(sin(twoPi * freq * t) * 32767.0))
        }
        return out
    }

    /// Read a 16-bit PCM WAV file; downmix stereo to mono by keeping the
    /// first channel. Mirrors `tests/common/mod.rs` in the Rust crate.
    private static func readWav(url: URL) throws -> (UInt32, [Int16]) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw WakeWordError.resamplingFailed(underlying: nil)
        }
        try file.read(into: buffer)

        let sampleRate = UInt32(format.sampleRate)
        let channels = Int(format.channelCount)
        let length = Int(buffer.frameLength)

        if let int16 = buffer.int16ChannelData {
            var mono = [Int16](repeating: 0, count: length)
            let first = int16[0]
            for i in 0..<length {
                mono[i] = first[i * (channels > 1 ? channels : 1)]
            }
            return (sampleRate, mono)
        }
        if let floats = buffer.floatChannelData {
            var mono = [Int16](repeating: 0, count: length)
            let first = floats[0]
            for i in 0..<length {
                let clamped = max(-1.0, min(1.0, first[i]))
                mono[i] = Int16(clamping: Int(clamped * 32767.0))
            }
            return (sampleRate, mono)
        }
        throw WakeWordError.resamplingFailed(underlying: nil)
    }
}
