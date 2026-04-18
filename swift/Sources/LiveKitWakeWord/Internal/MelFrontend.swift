// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import CoreML
import Foundation

/// Core ML wrapper around the frozen ``melspectrogram.mlpackage`` model.
///
/// Input: mono 16 kHz float32 audio in `[-1, 1]`.
/// Output: a `(frames, 32)` normalized mel spectrogram, where `frames`
/// depends on the audio length (roughly `samples / 160 - 2`).
///
/// The Core ML model already applies the openWakeWord `x/10 + 2`
/// normalization, so callers can feed the output straight into the
/// embedding model.
final class MelFrontend {
    private let model: MLModel

    init(computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let url = try ResourceLoader.compiledModelURL(forMLPackage: "melspectrogram")
        self.model = try MLModel(contentsOf: url, configuration: config)
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

        let input = try MLMultiArray(shape: [1, NSNumber(value: sampleCount)], dataType: .float32)
        input.dataPointer
            .assumingMemoryBound(to: Float.self)
            .update(from: audio.baseAddress!, count: sampleCount)

        let features = try MLDictionaryFeatureProvider(dictionary: ["audio": input])
        let result = try model.prediction(from: features)
        guard let melArray = result.featureValue(for: "mel")?.multiArrayValue else {
            throw WakeWordError.modelOutputMissing(key: "mel")
        }
        return MelOutput(array: melArray)
    }
}

/// Lightweight view over a `(frames, 32)` mel spectrogram tensor. Holds a
/// reference to the underlying ``MLMultiArray`` and exposes direct float
/// access so we can build embedding windows without another copy.
struct MelOutput {
    let array: MLMultiArray

    var frameCount: Int {
        array.shape[0].intValue
    }

    var melBins: Int {
        array.shape.last!.intValue
    }

    /// Copy a contiguous `[start, start + 76)` window into a new
    /// ``MLMultiArray`` of shape `(76, 32, 1)` ready for embedding input.
    ///
    /// The embedding model expects a channels-last layout `(N, 76, 32, 1)`;
    /// this returns a single `(76, 32, 1)` slab the caller can stack.
    func copyWindow(startFrame: Int, length: Int) throws -> MLMultiArray {
        precondition(startFrame + length <= frameCount,
                     "window out of range: start=\(startFrame) len=\(length) frames=\(frameCount)")

        let dst = try MLMultiArray(
            shape: [NSNumber(value: length), NSNumber(value: melBins), 1],
            dataType: .float32
        )
        let src = array.dataPointer.assumingMemoryBound(to: Float.self)
        let dstPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)

        // Both arrays are dense float32, row-major; the stride between
        // consecutive mel frames is `melBins` floats. Copy in one shot.
        let elements = length * melBins
        let srcOffset = startFrame * melBins
        memcpy(dstPtr, src + srcOffset, elements * MemoryLayout<Float>.size)
        return dst
    }
}
