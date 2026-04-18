// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import CoreML
import Foundation

/// Core ML wrapper around the frozen ``embedding_model.mlpackage`` CNN.
///
/// The underlying model has a fixed batch size of
/// ``WakeWordConstants/embeddingBatch`` (16). One inference maps 16 mel
/// windows `(76, 32, 1)` to 16 embedding vectors `(96,)`.
final class EmbeddingModel {
    private let model: MLModel

    init(computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let url = try ResourceLoader.compiledModelURL(forMLPackage: "embedding_model")
        self.model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Run the embedding CNN on a batch of 16 mel windows.
    ///
    /// - Parameter windows: `(16, 76, 32, 1)` ``MLMultiArray`` where the
    ///   leading dim is the batch and the inner dims are the mel window.
    /// - Returns: `(16, 96)` float embeddings.
    func predict(windows: MLMultiArray) throws -> MLMultiArray {
        let features = try MLDictionaryFeatureProvider(dictionary: ["mel_windows": windows])
        let result = try model.prediction(from: features)
        guard let embeddings = result.featureValue(for: "embeddings")?.multiArrayValue else {
            throw WakeWordError.modelOutputMissing(key: "embeddings")
        }
        return embeddings
    }
}
