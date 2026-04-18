// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// Errors raised by ``WakeWordModel`` / ``WakeWordListener``.
public enum WakeWordError: Error, LocalizedError, Sendable {
    /// A bundled Core ML resource (mel/embedding ``.mlmodelc``) was not
    /// found inside the ``LiveKitWakeWord`` resource bundle.
    case bundledResourceMissing(name: String)
    /// The classifier file at the given URL does not exist or is not a
    /// readable Core ML model.
    case classifierNotFound(url: URL)
    /// A Core ML model's output dictionary was missing an expected key.
    case modelOutputMissing(key: String)
    /// ``predict`` was called with a PCM buffer that resampled to an audio
    /// length outside the mel model's declared `RangeDim` (currently 1–3 s
    /// at 16 kHz).
    case audioOutOfRange(samples: Int)
    /// The requested sample rate cannot be resampled to 16 kHz by
    /// ``AVAudioConverter``.
    case unsupportedSampleRate(rate: UInt32)
    /// ``AVAudioConverter`` reported an error while resampling.
    case resamplingFailed(underlying: Error?)

    public var errorDescription: String? {
        switch self {
        case .bundledResourceMissing(let name):
            return "LiveKitWakeWord: bundled resource '\(name)' is missing."
        case .classifierNotFound(let url):
            return "LiveKitWakeWord: classifier not found at \(url.path)."
        case .modelOutputMissing(let key):
            return "LiveKitWakeWord: Core ML model output missing '\(key)'."
        case .audioOutOfRange(let samples):
            return "LiveKitWakeWord: audio chunk must resample to 1–3 s at 16 kHz (got \(samples) samples)."
        case .unsupportedSampleRate(let rate):
            return "LiveKitWakeWord: sample rate \(rate) Hz cannot be resampled to 16 kHz."
        case .resamplingFailed(let underlying):
            if let underlying {
                return "LiveKitWakeWord: resampling failed (\(underlying))."
            }
            return "LiveKitWakeWord: resampling failed."
        }
    }
}
