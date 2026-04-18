// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import CoreML
import Foundation

/// Stateless wake-word detector — pass PCM audio, get back per-classifier
/// confidence scores.
///
/// Mirrors the semantics of the Rust `WakeWordModel` (and the Python
/// `WakeWordModel`): mel frontend → embedding CNN → one or more
/// classifier heads. The two frontend models are bundled with the Swift
/// package; classifier heads are loaded at runtime from disk so apps can
/// ship multiple wake words or swap them without rebuilding the package.
///
/// `WakeWordModel` is `@unchecked Sendable`: the underlying Core ML models
/// are safe to share across queues, but successive calls to ``predict(_:)``
/// must not overlap on the same instance (Core ML's `predict` is not
/// reentrant on a single `MLModel`). Use ``WakeWordListener`` for the
/// typical "tap the microphone and serialise inference" use case.
public final class WakeWordModel: @unchecked Sendable {
    public let sampleRate: UInt32
    public let computeUnits: MLComputeUnits

    private let melFrontend: MelFrontend
    private let embedding: EmbeddingModel
    private let resampler: AudioResampler?

    private let classifierLock = NSLock()
    private var classifiers: [String: Classifier] = [:]

    /// Create a detector with ``classifierURLs`` loaded up front.
    ///
    /// - Parameters:
    ///   - classifiers: URLs of `.mlpackage` or `.mlmodelc` classifier files.
    ///     Each file's name (minus the extension) is used as the key returned
    ///     by ``predict(_:)``.
    ///   - sampleRate: Sample rate of the PCM the caller will feed in.
    ///     Anything other than 16 kHz is resampled internally.
    ///   - computeUnits: Which accelerators Core ML may use. ``.all`` lets
    ///     it pick ANE/GPU/CPU per-layer, which is usually fastest and most
    ///     power-efficient.
    public init(
        classifiers classifierURLs: [URL] = [],
        sampleRate: UInt32 = 16_000,
        computeUnits: MLComputeUnits = .all
    ) throws {
        self.sampleRate = sampleRate
        self.computeUnits = computeUnits
        self.melFrontend = try MelFrontend(computeUnits: computeUnits)
        self.embedding = try EmbeddingModel(computeUnits: computeUnits)
        self.resampler = sampleRate == UInt32(WakeWordConstants.modelSampleRate)
            ? nil
            : try AudioResampler(inputSampleRate: sampleRate)

        for url in classifierURLs {
            try loadClassifier(url: url, name: nil)
        }
    }

    /// Add a classifier to the set consulted on every ``predict(_:)``.
    ///
    /// - Parameters:
    ///   - url: Path to a `.mlpackage` or `.mlmodelc` file.
    ///   - name: Optional key under which the result appears in the score
    ///     dictionary. Defaults to the filename stem.
    public func loadClassifier(url: URL, name: String? = nil) throws {
        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) {
            throw WakeWordError.classifierNotFound(url: url)
        }
        let resolvedName = name ?? url.deletingPathExtension().lastPathComponent
        let classifier = try Classifier(name: resolvedName, url: url, computeUnits: computeUnits)
        classifierLock.lock()
        classifiers[resolvedName] = classifier
        classifierLock.unlock()
    }

    /// Remove a classifier by name. Returns `true` if something was removed.
    @discardableResult
    public func unloadClassifier(name: String) -> Bool {
        classifierLock.lock()
        let removed = classifiers.removeValue(forKey: name) != nil
        classifierLock.unlock()
        return removed
    }

    /// Names of all currently loaded classifiers.
    public var classifierNames: [String] {
        classifierLock.lock()
        defer { classifierLock.unlock() }
        return Array(classifiers.keys)
    }

    /// Predict wake-word confidence across all loaded classifiers.
    ///
    /// Pass ~2 s of audio. Shorter chunks that produce fewer than 16
    /// embeddings return 0 for every classifier (same semantics as the
    /// Rust crate). Zero-copy overload for callers that already have a
    /// buffer pointer handy.
    public func predict(_ pcm: UnsafeBufferPointer<Int16>) throws -> [String: Float] {
        classifierLock.lock()
        let snapshot = classifiers
        classifierLock.unlock()
        if snapshot.isEmpty { return [:] }

        let audio16k: [Float]
        if let resampler {
            audio16k = try resampler.resample(samples: pcm)
        } else {
            audio16k = Self.int16ToFloat(pcm)
        }

        guard audio16k.count >= WakeWordConstants.minMelSamples else {
            return Self.zeroScores(snapshot)
        }

        let melOutput = try audio16k.withUnsafeBufferPointer { buf -> MelOutput in
            let capped = min(buf.count, WakeWordConstants.maxMelSamples)
            let slice = UnsafeBufferPointer(start: buf.baseAddress, count: capped)
            return try melFrontend.predict(audio: slice)
        }

        let frames = melOutput.frameCount
        if frames < WakeWordConstants.embeddingWindow {
            return Self.zeroScores(snapshot)
        }

        // Slide 76-frame windows with stride 8; the final 16 windows feed the
        // classifier. If we have fewer than 16 windows worth of audio the
        // result is undefined, so we return zeros (matches Rust behavior).
        let windowCount = (frames - WakeWordConstants.embeddingWindow) / WakeWordConstants.embeddingStride + 1
        if windowCount < WakeWordConstants.classifierEmbeddings {
            return Self.zeroScores(snapshot)
        }

        let startWindow = windowCount - WakeWordConstants.classifierEmbeddings
        let mlWindows = try makeEmbeddingBatch(from: melOutput, startWindow: startWindow)

        let embeddings = try embedding.predict(windows: mlWindows)
        // embeddings shape: (16, 96). Reshape to (1, 16, 96) for classifiers.
        let clsInput = try reshapeForClassifier(embeddings: embeddings)

        var results = [String: Float]()
        results.reserveCapacity(snapshot.count)
        for (name, classifier) in snapshot {
            results[name] = try classifier.predict(embeddings: clsInput)
        }
        return results
    }

    /// Convenience `Array<Int16>` overload.
    public func predict(_ pcm: [Int16]) throws -> [String: Float] {
        try pcm.withUnsafeBufferPointer { try predict($0) }
    }

    // MARK: - Helpers

    private func makeEmbeddingBatch(from mel: MelOutput, startWindow: Int) throws -> MLMultiArray {
        let W = WakeWordConstants.embeddingWindow
        let S = WakeWordConstants.embeddingStride
        let bins = WakeWordConstants.melBins
        let batch = WakeWordConstants.classifierEmbeddings

        let dst = try MLMultiArray(
            shape: [
                NSNumber(value: batch),
                NSNumber(value: W),
                NSNumber(value: bins),
                1,
            ],
            dataType: .float32
        )
        let dstPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)
        let srcPtr = mel.array.dataPointer.assumingMemoryBound(to: Float.self)
        let windowStride = W * bins

        for b in 0..<batch {
            let startFrame = (startWindow + b) * S
            let srcOffset = startFrame * bins
            let dstOffset = b * windowStride
            memcpy(
                dstPtr.advanced(by: dstOffset),
                srcPtr.advanced(by: srcOffset),
                windowStride * MemoryLayout<Float>.size
            )
        }
        return dst
    }

    private func reshapeForClassifier(embeddings: MLMultiArray) throws -> MLMultiArray {
        let shaped = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: WakeWordConstants.classifierEmbeddings),
                NSNumber(value: WakeWordConstants.embeddingDim),
            ],
            dataType: .float32
        )
        let count = WakeWordConstants.classifierEmbeddings * WakeWordConstants.embeddingDim
        memcpy(
            shaped.dataPointer,
            embeddings.dataPointer,
            count * MemoryLayout<Float>.size
        )
        return shaped
    }

    private static func int16ToFloat(_ pcm: UnsafeBufferPointer<Int16>) -> [Float] {
        var out = [Float](repeating: 0, count: pcm.count)
        let inv = Float(1.0 / 32768.0)
        for i in 0..<pcm.count {
            out[i] = Float(pcm[i]) * inv
        }
        return out
    }

    private static func zeroScores(_ classifiers: [String: Classifier]) -> [String: Float] {
        var d: [String: Float] = [:]
        d.reserveCapacity(classifiers.count)
        for k in classifiers.keys { d[k] = 0 }
        return d
    }
}
