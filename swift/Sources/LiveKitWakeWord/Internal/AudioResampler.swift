// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import AVFoundation
import Foundation

/// Wraps ``AVAudioConverter`` to convert arbitrary-rate mono Int16 PCM to
/// 16 kHz mono Float32 PCM, matching what the mel model expects.
///
/// ``AVAudioConverter`` picks a reasonable quality/latency tradeoff for us
/// out of the box (internally it uses ``AudioToolbox`` / vImage FIRs). If
/// we ever need bit-exact parity with the Rust ``resampler::ResamplerFir``
/// path we'd have to do that manually with ``vDSP``; for on-device wake
/// word detection the default is good enough and much simpler.
final class AudioResampler {
    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat
    private let converter: AVAudioConverter

    init(inputSampleRate: UInt32) throws {
        guard let inFmt = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(inputSampleRate),
            channels: 1,
            interleaved: true
        ) else {
            throw WakeWordError.unsupportedSampleRate(rate: inputSampleRate)
        }
        guard let outFmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: WakeWordConstants.modelSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw WakeWordError.unsupportedSampleRate(rate: inputSampleRate)
        }
        guard let conv = AVAudioConverter(from: inFmt, to: outFmt) else {
            throw WakeWordError.unsupportedSampleRate(rate: inputSampleRate)
        }

        self.inputFormat = inFmt
        self.outputFormat = outFmt
        self.converter = conv
    }

    /// Resample ``samples`` to 16 kHz float32. Returns the resampled array.
    func resample(samples: UnsafeBufferPointer<Int16>) throws -> [Float] {
        guard let count = samples.count as Int?, count > 0 else { return [] }
        let ratio = outputFormat.sampleRate / inputFormat.sampleRate

        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: AVAudioFrameCount(count)
        ) else {
            throw WakeWordError.resamplingFailed(underlying: nil)
        }
        inputBuffer.frameLength = AVAudioFrameCount(count)
        if let dst = inputBuffer.int16ChannelData?[0] {
            dst.update(from: samples.baseAddress!, count: count)
        }

        // AVAudioConverter's output buffer must be sized for the expected
        // number of output frames. +8 gives slack for internal alignment.
        let outFrameCapacity = AVAudioFrameCount(ceil(Double(count) * ratio)) + 8
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outFrameCapacity
        ) else {
            throw WakeWordError.resamplingFailed(underlying: nil)
        }

        var consumed = false
        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if status == .error || error != nil {
            throw WakeWordError.resamplingFailed(underlying: error)
        }

        let outCount = Int(outputBuffer.frameLength)
        guard let ptr = outputBuffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }
}
