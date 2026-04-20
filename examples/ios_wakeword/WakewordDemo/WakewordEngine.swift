// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

@preconcurrency import AVFoundation
import Foundation
import LiveKitWakeWord
import SwiftUI

/// User-selectable ONNX Runtime execution provider. Wraps
/// ``LiveKitWakeWord/ExecutionProvider`` in a SwiftUI-friendly `Identifiable`
/// enum so the ``Picker`` can render it.
enum ComputeBackend: String, CaseIterable, Identifiable, Sendable {
    case coreML
    case coreMLCPUAndGPU
    case coreMLCPUOnly
    case cpu

    var id: String { rawValue }

    var label: String {
        switch self {
        case .coreML: return "CoreML (ANE+GPU+CPU)"
        case .coreMLCPUAndGPU: return "CoreML (GPU+CPU)"
        case .coreMLCPUOnly: return "CoreML (CPU)"
        case .cpu: return "ORT CPU"
        }
    }

    var executionProvider: ExecutionProvider {
        switch self {
        case .coreML: return .coreML
        case .coreMLCPUAndGPU: return .coreMLCPUAndGPU
        case .coreMLCPUOnly: return .coreMLCPUOnly
        case .cpu: return .cpu
        }
    }
}

/// Drives the microphone and runs wake-word inference on rolling 2-second windows.
///
/// Uses ``LiveKitWakeWord.WakeWordModel`` (ONNX Runtime + CoreML EP) under
/// the hood. Shape of the public API consumed by the UI is stable.
///
/// Design:
/// - `AVAudioEngine` taps the input node on the real-time audio thread.
/// - Each callback converts `Float32` -> `Int16` and appends to a ring buffer
///   protected by an `NSLock`.
/// - Every `predictInterval` seconds (tracked inside the lock), a snapshot
///   of the full ring is dispatched to a background queue where
///   `model.predict()` runs.
/// - Results are published back to the main actor for the UI.
final class WakewordEngine: ObservableObject, @unchecked Sendable {

    // MARK: - Published UI state (mutated on MainActor)

    @Published private(set) var isRunning = false
    @Published private(set) var score: Float = 0
    @Published private(set) var isTriggered = false
    @Published private(set) var lastError: String?
    @Published private(set) var volume: Float = 0
    @Published private(set) var tick: UInt64 = 0
    @Published var backend: ComputeBackend = .coreML {
        didSet {
            guard oldValue != backend else { return }
            Task { @MainActor in self.rebuildModel() }
        }
    }

    // MARK: - Tuning

    private let triggerThreshold: Float = 0.75
    private let triggerHoldDuration: TimeInterval = 1.5
    private let predictInterval: CFAbsoluteTime = 0.02
    private let windowSeconds: Double = 2.0

    // MARK: - Wake-word model

    private let classifierURLs: [URL]
    /// Current `WakeWordModel` instance. Rebuilt when the user changes the
    /// execution provider or the hardware sample rate, otherwise reused.
    private var model: WakeWordModel?
    private var modelSampleRate: UInt32 = 0

    // MARK: - Audio capture

    private var engine: AVAudioEngine?
    /// Run inference at `.userInteractive` so it preempts other background
    /// work and latency stays bounded when the Neural Engine is busy.
    private let workQueue = DispatchQueue(label: "io.livekit.wakeword.predict", qos: .userInteractive)

    private let ringLock = NSLock()
    private var ring: [Int16] = []
    private var writeIdx = 0
    private var samplesWritten = 0
    private var lastPredictAt: CFAbsoluteTime = 0
    private var predictInFlight = false

    private var triggerResetTask: Task<Void, Never>?

    // MARK: - Init

    init() throws {
        guard let url = Bundle.main.url(forResource: "hey_livekit", withExtension: "onnx") else {
            throw NSError(
                domain: "WakewordEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "hey_livekit.onnx classifier not found in app bundle"]
            )
        }
        self.classifierURLs = [url]
    }

    // MARK: - Public API

    @MainActor
    func toggle() {
        if isRunning {
            stop()
        } else {
            Task { await self.startAfterAuth() }
        }
    }

    // MARK: - Permission + start

    @MainActor
    private func startAfterAuth() async {
        let granted = await requestMicrophonePermission()
        guard granted else {
            lastError = "Microphone permission denied. Enable it in System Settings → Privacy & Security → Microphone."
            return
        }
        do {
            try start()
            lastError = nil
        } catch {
            lastError = "Start failed: \(error.localizedDescription)"
            stopInternal()
        }
    }

    private func requestMicrophonePermission() async -> Bool {
        if #available(iOS 17.0, macOS 14.0, *) {
            if AVAudioApplication.shared.recordPermission == .granted { return true }
            return await AVAudioApplication.requestRecordPermission()
        } else {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            if session.recordPermission == .granted { return true }
            return await withCheckedContinuation { cont in
                session.requestRecordPermission { cont.resume(returning: $0) }
            }
            #else
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized: return true
            case .notDetermined:
                return await withCheckedContinuation { cont in
                    AVCaptureDevice.requestAccess(for: .audio) { cont.resume(returning: $0) }
                }
            default: return false
            }
            #endif
        }
    }

    // MARK: - Start / Stop

    @MainActor
    private func start() throws {
        try configureAudioSession()

        let engine = AVAudioEngine()
        self.engine = engine

        let input = engine.inputNode
        let hwFormat = input.inputFormat(forBus: 0)

        guard hwFormat.sampleRate > 0 else {
            throw NSError(
                domain: "WakewordEngine",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Input has no valid sample rate (is a microphone connected?)"]
            )
        }

        let hwRate = UInt32(hwFormat.sampleRate)
        if model == nil || modelSampleRate != hwRate {
            model = try WakeWordModel(
                classifiers: classifierURLs,
                sampleRate: hwRate,
                executionProvider: backend.executionProvider
            )
            modelSampleRate = hwRate
            let ringSize = max(Int(hwFormat.sampleRate * windowSeconds), 1)
            ringLock.lock()
            ring = [Int16](repeating: 0, count: ringSize)
            writeIdx = 0
            samplesWritten = 0
            lastPredictAt = 0
            predictInFlight = false
            ringLock.unlock()
        } else {
            ringLock.lock()
            writeIdx = 0
            samplesWritten = 0
            lastPredictAt = 0
            predictInFlight = false
            ringLock.unlock()
        }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: hwFormat.sampleRate,
            channels: 1,
            interleaved: true
        ) else {
            throw NSError(
                domain: "WakewordEngine",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not create target Int16 format"]
            )
        }
        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else {
            throw NSError(
                domain: "WakewordEngine",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Could not create AVAudioConverter"]
            )
        }

        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            self?.handleInput(buffer: buffer, converter: converter, targetFormat: targetFormat)
        }

        engine.prepare()
        try engine.start()
        isRunning = true
    }

    @MainActor
    private func stop() {
        stopInternal()
    }

    @MainActor
    private func stopInternal() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil

        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif

        ringLock.lock()
        samplesWritten = 0
        writeIdx = 0
        lastPredictAt = 0
        predictInFlight = false
        ringLock.unlock()

        isRunning = false
        score = 0
        volume = 0
        tick = 0
        isTriggered = false
        triggerResetTask?.cancel()
        triggerResetTask = nil
    }

    /// Rebuild the model to pick up a new ``ComputeBackend``. Invoked from
    /// ``backend``'s `didSet`. If we're currently running, we quickly cycle
    /// the audio engine so the new model picks up the next tap.
    @MainActor
    private func rebuildModel() {
        let wasRunning = isRunning
        if wasRunning {
            stopInternal()
        }
        model = nil
        modelSampleRate = 0
        if wasRunning {
            Task { await self.startAfterAuth() }
        }
    }

    @MainActor
    private func configureAudioSession() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true, options: [])
        #endif
    }

    // MARK: - Audio tap (real-time thread)

    private func handleInput(
        buffer inputBuffer: AVAudioPCMBuffer,
        converter: AVAudioConverter,
        targetFormat: AVAudioFormat
    ) {
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: inputBuffer.frameCapacity
        ) else { return }

        var consumed = false
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        guard status != .error, error == nil,
              let channelData = outBuffer.int16ChannelData else {
            return
        }

        let frameCount = Int(outBuffer.frameLength)
        guard frameCount > 0 else { return }

        let level = computeLevel(samples: channelData[0], count: frameCount)
        Task { @MainActor [weak self] in
            self?.publish(volume: level)
        }

        let shouldRun = appendAndCheck(samples: channelData[0], count: frameCount)
        if shouldRun, let snapshot = snapshotRing() {
            workQueue.async { [weak self] in
                self?.runPredict(snapshot: snapshot)
            }
        }
    }

    private func appendAndCheck(samples: UnsafePointer<Int16>, count: Int) -> Bool {
        ringLock.lock()
        defer { ringLock.unlock() }

        let size = ring.count
        guard size > 0 else { return false }
        var idx = writeIdx
        for i in 0..<count {
            ring[idx] = samples[i]
            idx += 1
            if idx >= size { idx = 0 }
        }
        writeIdx = idx
        samplesWritten = min(samplesWritten + count, size)

        guard samplesWritten >= size else { return false }
        let now = CFAbsoluteTimeGetCurrent()
        guard (now - lastPredictAt) >= predictInterval else { return false }
        guard !predictInFlight else { return false }
        lastPredictAt = now
        predictInFlight = true
        return true
    }

    /// RMS-derived 0..1 loudness indicator over a ~90 ms capture buffer.
    /// We map a 60 dB range above -60 dBFS → 0..1 so speech reads strongly
    /// without pinning to the ceiling.
    private func computeLevel(samples: UnsafePointer<Int16>, count: Int) -> Float {
        guard count > 0 else { return 0 }
        var sumSquares: Double = 0
        for i in 0..<count {
            let v = Double(samples[i])
            sumSquares += v * v
        }
        let rms = sqrt(sumSquares / Double(count))
        let normalized = rms / Double(Int16.max)
        let floorDb: Double = -60
        let db: Double
        if normalized <= 1e-7 {
            db = floorDb
        } else {
            db = max(floorDb, 20.0 * log10(normalized))
        }
        let level = (db - floorDb) / -floorDb
        return Float(max(0, min(1, level)))
    }

    private func snapshotRing() -> [Int16]? {
        ringLock.lock()
        defer { ringLock.unlock() }
        let size = ring.count
        guard samplesWritten >= size, size > 0 else { return nil }
        var out = [Int16](repeating: 0, count: size)
        let tail = size - writeIdx
        out.withUnsafeMutableBufferPointer { dst in
            ring.withUnsafeBufferPointer { src in
                guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress else { return }
                dstBase.update(from: srcBase + writeIdx, count: tail)
                if writeIdx > 0 {
                    (dstBase + tail).update(from: srcBase, count: writeIdx)
                }
            }
        }
        return out
    }

    private func runPredict(snapshot: [Int16]) {
        defer {
            ringLock.lock()
            predictInFlight = false
            ringLock.unlock()
        }
        guard let model else { return }
        do {
            let scores = try model.predict(snapshot)
            let maxScore = scores.values.max() ?? 0
            Task { @MainActor [weak self] in
                self?.publish(score: maxScore)
            }
        } catch {
            Task { @MainActor [weak self] in
                self?.lastError = "predict failed: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - UI publish (MainActor)

    @MainActor
    private func publish(volume newVolume: Float) {
        let alpha: Float = 0.35
        volume = (alpha * newVolume) + ((1 - alpha) * volume)
        tick &+= 1
    }

    @MainActor
    private func publish(score newScore: Float) {
        score = newScore
        if newScore >= triggerThreshold {
            isTriggered = true
            triggerResetTask?.cancel()
            let hold = triggerHoldDuration
            triggerResetTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(hold * 1_000_000_000))
                if !Task.isCancelled {
                    await MainActor.run { self?.isTriggered = false }
                }
            }
        }
    }
}

// Expose a read-only view of whether a model is currently loaded so tests or
// debug UIs can observe readiness without touching the ML model directly.
extension WakewordEngine {
    /// `nil` when no model has been built yet (e.g. the user has not pressed
    /// "Unmute mic" since launch / since the backend changed).
    var modelLoadedSampleRate: UInt32? {
        model == nil ? nil : modelSampleRate
    }
}
