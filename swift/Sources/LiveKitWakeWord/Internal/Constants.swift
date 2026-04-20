// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// Magic numbers baked into the frozen mel + embedding ONNX models.
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

    /// 96-dim embedding vector output by the embedding CNN.
    static let embeddingDim: Int = 96

    /// 32-bin mel spectrogram output by the frontend.
    static let melBins: Int = 32

    /// Minimum mel input (samples). Shorter clips can't fill a single
    /// 76-frame embedding window, so the result would be undefined.
    static let minMelSamples: Int = 16_000  // 1.0 s at 16 kHz
    /// Soft cap on mel input length. The ONNX mel model is fully dynamic,
    /// but arbitrarily long buffers waste memory; the listener feeds 2 s
    /// windows so 3 s is plenty of headroom.
    static let maxMelSamples: Int = 48_000  // 3.0 s at 16 kHz
}
