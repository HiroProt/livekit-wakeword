// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// Locates ONNX resources bundled with the ``LiveKitWakeWord`` module.
///
/// The package ships its mel + embedding frontend models as plain `.onnx`
/// files. ONNX Runtime loads them directly from the bundle URL — no
/// compilation or cache step is needed.
enum ResourceLoader {
    static func resourceURL(name: String, extension ext: String) throws -> URL {
        if let url = Bundle.module.url(forResource: name, withExtension: ext) {
            return url
        }
        if let url = Bundle.main.url(forResource: name, withExtension: ext) {
            return url
        }
        throw WakeWordError.bundledResourceMissing(name: "\(name).\(ext)")
    }
}
