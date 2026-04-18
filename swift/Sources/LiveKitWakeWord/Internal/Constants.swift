// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// Magic numbers baked into the frozen mel + embedding Core ML models.
/// Changing these requires re-running the Python export pipeline.
enum WakeWordConstants {
    /// Sample rate the frozen models were trained at.
    static let modelSampleRate: Double = 16_000

    /// Mel window length (frames) consumed by the embedding CNN.
    static let embeddingWindow: Int = 76
    /// Stride between consecutive mel windows (frames).
    static let embeddingStride: Int = 8

    /// Number of embeddings the classifier consumes.
    static let classifierEmbeddings: Int = 16

    /// The embedding model is exported with a fixed batch axis. This must
    /// match the ``batch_size`` passed to ``convert_embedding_model`` in
    /// the Python export pipeline.
    static let embeddingBatch: Int = 16

    /// 96-dim embedding vector output by the embedding CNN.
    static let embeddingDim: Int = 96

    /// 32-bin mel spectrogram output by the frontend.
    static let melBins: Int = 32

    /// Mel frontend input bounds (samples). Must match the ``RangeDim``
    /// passed to ``convert_mel_frontend`` in the Python export pipeline.
    static let minMelSamples: Int = 16_000  // 1.0 s at 16 kHz
    static let maxMelSamples: Int = 48_000  // 3.0 s at 16 kHz
}
