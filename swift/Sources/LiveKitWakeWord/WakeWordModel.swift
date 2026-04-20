// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// Controls which ONNX Runtime execution provider ``WakeWordModel`` uses.
///
/// The `coreML*` variants wrap the CoreML Execution Provider, which lets
/// ONNX Runtime dispatch supported ops to Apple's ML stack (ANE / GPU /
/// CPU) and fall back to the ORT CPU kernels for anything unsupported.
/// ``cpu`` skips CoreML entirely.
public enum ExecutionProvider: Sendable, Equatable {
    /// CoreML EP with no device restrictions — may use ANE, GPU, and CPU.
    /// This is usually the fastest and most power-efficient choice.
    case coreML
    /// CoreML EP restricted to CPU + GPU (no Apple Neural Engine).
    case coreMLCPUAndGPU
    /// CoreML EP restricted to CPU only.
    case coreMLCPUOnly
    /// ORT's built-in CPU provider — CoreML EP is not appended.
    case cpu
}

/// Stateless wake-word detector — pass PCM audio, get back per-model
/// confidence scores.
///
/// Matches the shape of the Python `WakeWordModel` and the Rust
/// `livekit_wakeword::WakeWordModel`: mel frontend → embedding CNN → one
/// or more classifier heads. The two frontend models are bundled with the
/// Swift package as `.onnx` files; classifier heads are loaded at runtime
/// from disk so apps can ship multiple wake words or swap them without
/// rebuilding the package.
///
/// `WakeWordModel` is `@unchecked Sendable`: the underlying ORT sessions
/// are safe to share across queues, but successive calls to
/// ``predict(_:)`` on the same instance must not overlap (a single ONNX
/// Runtime session is not reentrant). Use ``WakeWordListener`` for the
/// typical "tap the microphone and serialise inference" use case.
public final class WakeWordModel: @unchecked Sendable {
    /// Sample rate the frozen mel + embedding models were trained at.
    /// Audio fed into the mel frontend is always at this rate; resampling
    /// from any other rate happens transparently in ``init(models:sampleRate:executionProvider:)``.
    public static let modelSampleRate: UInt32 = 16_000

    public let sampleRate: UInt32
    public let executionProvider: ExecutionProvider

    private let env: ORTEnv
    private let sessionOptions: ORTSessionOptions
    private let melFrontend: MelFrontend
    private let embedding: EmbeddingModel
    private let resampler: AudioResampler?

    private let modelsLock = NSLock()
    private var models: [String: Classifier] = [:]

    /// Create a detector with the given classifier models loaded up front.
    ///
    /// - Parameters:
    ///   - models: URLs of `.onnx` classifier files. Each file's name
    ///     (minus the extension) is used as the key returned by
    ///     ``predict(_:)``.
    ///   - sampleRate: Sample rate of the PCM the caller will feed in.
    ///     Anything other than 16 kHz is resampled internally.
    ///   - executionProvider: Which ONNX Runtime execution provider to use.
    ///     Defaults to ``ExecutionProvider/coreML`` (ANE + GPU + CPU).
    public init(
        models modelURLs: [URL] = [],
        sampleRate: UInt32 = 16_000,
        executionProvider: ExecutionProvider = .coreML
    ) throws {
        self.sampleRate = sampleRate
        self.executionProvider = executionProvider
        self.env = try ORTRuntime.sharedEnv()
        self.sessionOptions = try ORTRuntime.sessionOptions(for: executionProvider)
        self.melFrontend = try MelFrontend(env: env, options: sessionOptions)
        self.embedding = try EmbeddingModel(env: env, options: sessionOptions)
        self.resampler = sampleRate == UInt32(WakeWordConstants.modelSampleRate)
            ? nil
            : try AudioResampler(inputSampleRate: sampleRate)

        for url in modelURLs {
            try loadModel(url: url, name: nil)
        }
    }

    /// Add a classifier model to the set consulted on every ``predict(_:)``.
    ///
    /// - Parameters:
    ///   - url: Path to a `.onnx` classifier file.
    ///   - name: Optional key under which the result appears in the score
    ///     dictionary. Defaults to the filename stem.
    public func loadModel(url: URL, name: String? = nil) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            throw WakeWordError.classifierNotFound(url: url)
        }
        let resolvedName = name ?? url.deletingPathExtension().lastPathComponent
        let classifier = try Classifier(
            name: resolvedName,
            url: url,
            env: env,
            options: sessionOptions
        )
        modelsLock.lock()
        models[resolvedName] = classifier
        modelsLock.unlock()
    }

    /// Remove a classifier model by name. Returns `true` if something was removed.
    @discardableResult
    public func unloadModel(name: String) -> Bool {
        modelsLock.lock()
        let removed = models.removeValue(forKey: name) != nil
        modelsLock.unlock()
        return removed
    }

    /// Names of all currently loaded classifier models.
    public var modelNames: [String] {
        modelsLock.lock()
        defer { modelsLock.unlock() }
        return Array(models.keys)
    }

    /// Predict wake-word confidence across all loaded models.
    ///
    /// Pass ~2 s of audio. Shorter chunks that don't produce 16 sliding
    /// embedding windows return 0 for every model (same semantics as the
    /// Python reference implementation).
    public func predict(_ pcm: UnsafeBufferPointer<Int16>) throws -> [String: Float] {
        modelsLock.lock()
        let snapshot = models
        modelsLock.unlock()
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

        let capped = min(audio16k.count, WakeWordConstants.maxMelSamples)
        let melOutput = try audio16k.withUnsafeBufferPointer { buf -> MelOutput in
            let slice = UnsafeBufferPointer(start: buf.baseAddress, count: capped)
            return try melFrontend.predict(audio: slice)
        }

        let frames = melOutput.frameCount
        if frames < WakeWordConstants.embeddingWindow {
            return Self.zeroScores(snapshot)
        }

        // Slide 76-frame windows with stride 8; classifier consumes the
        // final 16. If we have fewer than 16 windows the result is
        // undefined, so return zeros.
        let windowCount = (frames - WakeWordConstants.embeddingWindow) / WakeWordConstants.embeddingStride + 1
        if windowCount < WakeWordConstants.classifierEmbeddings {
            return Self.zeroScores(snapshot)
        }

        let startWindow = windowCount - WakeWordConstants.classifierEmbeddings
        let batchBuffer = Self.makeEmbeddingBatch(from: melOutput, startWindow: startWindow)
        let embeddings = try embedding.predict(
            windows: batchBuffer,
            batchSize: WakeWordConstants.classifierEmbeddings
        )

        var results = [String: Float]()
        results.reserveCapacity(snapshot.count)
        for (name, model) in snapshot {
            results[name] = try model.predict(embeddings: embeddings)
        }
        return results
    }

    /// Convenience `Array<Int16>` overload.
    public func predict(_ pcm: [Int16]) throws -> [String: Float] {
        try pcm.withUnsafeBufferPointer { try predict($0) }
    }

    // MARK: - Helpers

    private static func makeEmbeddingBatch(
        from mel: MelOutput,
        startWindow: Int
    ) -> [Float] {
        let W = WakeWordConstants.embeddingWindow
        let S = WakeWordConstants.embeddingStride
        let bins = WakeWordConstants.melBins
        let batch = WakeWordConstants.classifierEmbeddings
        let elementsPerWindow = W * bins

        var buffer = [Float](repeating: 0, count: batch * elementsPerWindow)
        mel.samples.withUnsafeBufferPointer { src in
            buffer.withUnsafeMutableBufferPointer { dst in
                for b in 0..<batch {
                    let startFrame = (startWindow + b) * S
                    let srcOffset = startFrame * bins
                    let dstOffset = b * elementsPerWindow
                    dst.baseAddress!.advanced(by: dstOffset).update(
                        from: src.baseAddress!.advanced(by: srcOffset),
                        count: elementsPerWindow
                    )
                }
            }
        }
        return buffer
    }

    private static func int16ToFloat(_ pcm: UnsafeBufferPointer<Int16>) -> [Float] {
        var out = [Float](repeating: 0, count: pcm.count)
        let inv = Float(1.0 / 32768.0)
        for i in 0..<pcm.count {
            out[i] = Float(pcm[i]) * inv
        }
        return out
    }

    private static func zeroScores(_ models: [String: Classifier]) -> [String: Float] {
        var d: [String: Float] = [:]
        d.reserveCapacity(models.count)
        for k in models.keys { d[k] = 0 }
        return d
    }
}
