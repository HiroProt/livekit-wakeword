// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import CoreML
import Foundation

/// One trained wake-word classifier — maps `(1, 16, 96)` embeddings to a
/// scalar confidence score in `[0, 1]`.
final class Classifier {
    let name: String
    private let model: MLModel

    init(name: String, url: URL, computeUnits: MLComputeUnits) throws {
        self.name = name

        // ``.mlpackage`` directories need compilation; ``.mlmodelc`` is
        // already compiled. Accept either for flexibility: apps typically
        // ship the pre-compiled flavor, while dev/test workflows hand us
        // raw ``.mlpackage`` files to keep their source tree simple.
        let compiledURL: URL
        if url.pathExtension == "mlmodelc" {
            compiledURL = url
        } else {
            compiledURL = try MLModel.compileModel(at: url)
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    }

    /// Run the classifier on `(1, 16, 96)` embeddings and return the score.
    func predict(embeddings: MLMultiArray) throws -> Float {
        let features = try MLDictionaryFeatureProvider(dictionary: ["embeddings": embeddings])
        let result = try model.prediction(from: features)
        guard let score = result.featureValue(for: "score")?.multiArrayValue else {
            throw WakeWordError.modelOutputMissing(key: "score")
        }
        return score[0].floatValue
    }
}
