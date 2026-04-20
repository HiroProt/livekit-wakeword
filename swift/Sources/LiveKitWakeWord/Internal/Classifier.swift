// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// One trained wake-word classifier — maps `(1, 16, 96)` embeddings to a
/// scalar confidence score in `[0, 1]`. Loaded from a standalone `.onnx`
/// file produced by `livekit-wakeword export`.
final class Classifier {
    let name: String
    private let session: ORTSession

    init(name: String, url: URL, env: ORTEnv, options: ORTSessionOptions) throws {
        self.name = name
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

    /// Run the classifier on a `(1, 16, 96)` embeddings batch.
    ///
    /// - Parameter embeddings: flat `[Float]` with exactly
    ///   `classifierEmbeddings * embeddingDim` elements (16 × 96 = 1536).
    func predict(embeddings: [Float]) throws -> Float {
        let expected = WakeWordConstants.classifierEmbeddings * WakeWordConstants.embeddingDim
        precondition(embeddings.count == expected,
                     "classifier expects \(expected) floats, got \(embeddings.count)")

        let byteCount = embeddings.count * MemoryLayout<Float>.size
        guard let inputData = NSMutableData(length: byteCount) else {
            throw WakeWordError.runtimeFailure(
                underlying: NSError(domain: "LiveKitWakeWord", code: -1)
            )
        }
        embeddings.withUnsafeBufferPointer {
            inputData.mutableBytes
                .assumingMemoryBound(to: Float.self)
                .update(from: $0.baseAddress!, count: embeddings.count)
        }

        let outputs: [String: ORTValue]
        do {
            let input = try ORTValue(
                tensorData: inputData,
                elementType: .float,
                shape: [
                    1,
                    NSNumber(value: WakeWordConstants.classifierEmbeddings),
                    NSNumber(value: WakeWordConstants.embeddingDim),
                ]
            )
            outputs = try session.run(
                withInputs: ["embeddings": input],
                outputNames: ["score"],
                runOptions: nil
            )
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }

        guard let out = outputs["score"] else {
            throw WakeWordError.modelOutputMissing(key: "score")
        }
        let outData: NSMutableData
        do {
            outData = try out.tensorData()
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }
        return outData.bytes.assumingMemoryBound(to: Float.self).pointee
    }
}
