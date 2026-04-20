// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// ONNX Runtime wrapper around the frozen ``embedding_model.onnx`` CNN.
///
/// The underlying graph accepts a dynamic batch size; we call it with a
/// batch of ``WakeWordConstants/classifierEmbeddings`` (16), producing 16
/// embedding vectors of ``WakeWordConstants/embeddingDim`` (96) floats
/// each.
final class EmbeddingModel {
    private let session: ORTSession

    init(env: ORTEnv, options: ORTSessionOptions) throws {
        let url = try ResourceLoader.resourceURL(name: "embedding_model", extension: "onnx")
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

    /// Run the embedding CNN on a batch of mel windows.
    ///
    /// - Parameters:
    ///   - windows: flat `[Float]` laid out as `(batchSize, 76, 32, 1)`
    ///     row-major. Must have exactly
    ///     `batchSize * 76 * 32` elements.
    ///   - batchSize: number of windows in ``windows``.
    /// - Returns: flat `[Float]` of `batchSize * 96` embedding values.
    func predict(windows: [Float], batchSize: Int) throws -> [Float] {
        let elementsPerWindow = WakeWordConstants.embeddingWindow * WakeWordConstants.melBins
        precondition(
            windows.count == batchSize * elementsPerWindow,
            "embedding input must be batchSize*76*32 floats (got \(windows.count))"
        )

        let byteCount = windows.count * MemoryLayout<Float>.size
        guard let inputData = NSMutableData(length: byteCount) else {
            throw WakeWordError.runtimeFailure(
                underlying: NSError(domain: "LiveKitWakeWord", code: -1)
            )
        }
        windows.withUnsafeBufferPointer {
            inputData.mutableBytes
                .assumingMemoryBound(to: Float.self)
                .update(from: $0.baseAddress!, count: windows.count)
        }

        let outputs: [String: ORTValue]
        do {
            let input = try ORTValue(
                tensorData: inputData,
                elementType: .float,
                shape: [
                    NSNumber(value: batchSize),
                    NSNumber(value: WakeWordConstants.embeddingWindow),
                    NSNumber(value: WakeWordConstants.melBins),
                    1,
                ]
            )
            // The frozen model was exported from TF with opaque layer names;
            // "input_1" / "conv2d_19" are not arbitrary — they match the
            // ONNX graph shipped in src/livekit/wakeword/resources/.
            outputs = try session.run(
                withInputs: ["input_1": input],
                outputNames: ["conv2d_19"],
                runOptions: nil
            )
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }

        guard let out = outputs["conv2d_19"] else {
            throw WakeWordError.modelOutputMissing(key: "conv2d_19")
        }

        let outData: NSMutableData
        do {
            outData = try out.tensorData()
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }

        // Output shape: (batch, 1, 1, 96) → flatten to (batch * 96) floats.
        let count = batchSize * WakeWordConstants.embeddingDim
        var embeddings = [Float](repeating: 0, count: count)
        embeddings.withUnsafeMutableBytes { dst in
            let bytes = count * MemoryLayout<Float>.size
            dst.copyMemory(from: UnsafeRawBufferPointer(start: outData.bytes, count: bytes))
        }
        return embeddings
    }
}
