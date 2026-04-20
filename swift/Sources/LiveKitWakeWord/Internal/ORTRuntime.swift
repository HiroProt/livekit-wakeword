// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// Lazy process-wide ``ORTEnv`` plus a builder that converts a public
/// ``ExecutionProvider`` choice into the matching ``ORTSessionOptions``.
///
/// The ONNX Runtime docs say a single ``ORTEnv`` should be created per
/// process and shared across sessions; this type memoizes the first one
/// that loads and hands it out on subsequent requests.
enum ORTRuntime {
    private static let lock = NSLock()
    private static var cachedEnv: ORTEnv?

    static func sharedEnv() throws -> ORTEnv {
        lock.lock()
        defer { lock.unlock() }
        if let env = cachedEnv { return env }
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            cachedEnv = env
            return env
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }
    }

    static func sessionOptions(for provider: ExecutionProvider) throws -> ORTSessionOptions {
        let options: ORTSessionOptions
        do {
            options = try ORTSessionOptions()
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }

        switch provider {
        case .cpu:
            return options
        case .coreML, .coreMLCPUAndGPU, .coreMLCPUOnly:
            let coremlOptions = ORTCoreMLExecutionProviderOptions()
            switch provider {
            case .coreMLCPUAndGPU:
                coremlOptions.useCPUAndGPU = true
            case .coreMLCPUOnly:
                coremlOptions.useCPUOnly = true
            case .coreML, .cpu:
                break
            }
            do {
                try options.appendCoreMLExecutionProvider(with: coremlOptions)
            } catch {
                throw WakeWordError.runtimeFailure(underlying: error)
            }
            return options
        }
    }
}
