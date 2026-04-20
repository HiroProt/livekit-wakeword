// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// ONNX Runtime wrapper around the frozen ``melspectrogram.onnx`` model.
///
/// Input: mono 16 kHz float32 audio in `[-1, 1]` shaped `(1, samples)`.
/// Output: `(frames, 32)` normalized mel spectrogram, where `frames`
/// depends on the audio length (~`samples / 160 - 2`). The ONNX model
/// produces raw dB-scale mel output; this wrapper applies the openWakeWord
/// `x/10 + 2` normalization so callers can feed the result straight into
/// the embedding model.
final class MelFrontend {
    private let session: ORTSession

    init(env: ORTEnv, options: ORTSessionOptions) throws {
        let url = try ResourceLoader.resourceURL(name: "melspectrogram", extension: "onnx")
        do {
            self.session = try ORTSession(
                env: env,
                modelPath: url.path,
                sessionOptions: options
            )
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }
    }

    /// Run the mel frontend on a single audio chunk.
    ///
    /// - Parameter audio: mono float samples at 16 kHz. Must contain between
    ///   ``WakeWordConstants/minMelSamples`` and
    ///   ``WakeWordConstants/maxMelSamples`` samples (currently 1–3 s).
    func predict(audio: UnsafeBufferPointer<Float>) throws -> MelOutput {
        let sampleCount = audio.count
        guard sampleCount >= WakeWordConstants.minMelSamples,
              sampleCount <= WakeWordConstants.maxMelSamples else {
            throw WakeWordError.audioOutOfRange(samples: sampleCount)
        }

        // ORT takes ownership semantics over the NSMutableData we hand it
        // (tensor data must outlive the run call). Copy the caller's buffer
        // in — the caller's pointer is not guaranteed to remain valid.
        let byteCount = sampleCount * MemoryLayout<Float>.size
        guard let inputData = NSMutableData(length: byteCount) else {
            throw WakeWordError.runtimeFailure(
                underlying: NSError(domain: "LiveKitWakeWord", code: -1)
            )
        }
        inputData.mutableBytes
            .assumingMemoryBound(to: Float.self)
            .update(from: audio.baseAddress!, count: sampleCount)

        let outputs: [String: ORTValue]
        do {
            let input = try ORTValue(
                tensorData: inputData,
                elementType: .float,
                shape: [NSNumber(value: 1), NSNumber(value: sampleCount)]
            )
            outputs = try session.run(
                withInputs: ["input": input],
                outputNames: ["output"],
                runOptions: nil
            )
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }

        guard let out = outputs["output"] else {
            throw WakeWordError.modelOutputMissing(key: "output")
        }

        let shape: [Int]
        let outData: NSMutableData
        do {
            shape = try out.tensorTypeAndShapeInfo().shape.map(\.intValue)
            outData = try out.tensorData()
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }

        // Expected layout: (batch=1, 1, frames, 32).
        guard shape.count == 4,
              shape[0] == 1,
              shape[1] == 1,
              shape[3] == WakeWordConstants.melBins else {
            throw WakeWordError.unexpectedOutputShape(
                expected: "(1, 1, frames, \(WakeWordConstants.melBins))",
                actual: shape
            )
        }
        let frameCount = shape[2]
        let elementCount = frameCount * WakeWordConstants.melBins

        var samples = [Float](repeating: 0, count: elementCount)
        samples.withUnsafeMutableBufferPointer { dst in
            let src = outData.bytes.assumingMemoryBound(to: Float.self)
            // Apply openWakeWord's x/10 + 2 mel normalization on the copy.
            let inv10: Float = 0.1
            let bias: Float = 2.0
            for i in 0..<elementCount {
                dst[i] = src[i] * inv10 + bias
            }
        }
        return MelOutput(samples: samples, frameCount: frameCount)
    }
}

/// A `(frameCount, 32)` mel spectrogram stored row-major as a flat
/// `[Float]`. Consumers slice contiguous 76-frame windows out of
/// ``samples`` via ``MelOutput`` to build the embedding batch.
struct MelOutput {
    let samples: [Float]
    let frameCount: Int
}
