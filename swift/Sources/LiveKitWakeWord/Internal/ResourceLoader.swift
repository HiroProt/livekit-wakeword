// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import CoreML
import Foundation

/// Locates and compiles Core ML resources bundled with the
/// ``LiveKitWakeWord`` module.
///
/// The package ships its frontend models as ``.mlpackage`` directories
/// (SPM's Xcode integration mishandles nested ``.mlmodelc`` resources on
/// `.process`, so we use `.copy`). Two consumption paths need to work:
///
/// 1. **Xcode app target consuming the SPM package.** Xcode compiles the
///    copied ``.mlpackage`` into ``.mlmodelc`` at app build time and bundles
///    the compiled form. We detect that and use it directly.
/// 2. **Plain ``swift test`` / ``swift build``.** The resource stays as a
///    raw ``.mlpackage``; we run ``MLModel.compileModel(at:)`` at first
///    use and cache the result in the user's caches directory so subsequent
///    process launches skip the compile step.
enum ResourceLoader {
    static func compiledModelURL(forMLPackage name: String) throws -> URL {
        // Fast path: a pre-compiled `.mlmodelc` exists in the bundle — this
        // is how it ends up when an Xcode app target owns the build.
        if let precompiled = resourceURL(named: name, extension: "mlmodelc") {
            return precompiled
        }

        let source = try resourceURL(named: name, extensionOrThrow: "mlpackage")

        let cache = try cacheDirectory()
        let cached = cache.appendingPathComponent("\(name).mlmodelc", isDirectory: true)

        // Reuse the cached compile if the source hasn't changed. We compare
        // modification dates so a new app version with updated resources
        // invalidates the cache automatically.
        if FileManager.default.fileExists(atPath: cached.path),
           let srcMod = modificationDate(url: source),
           let dstMod = modificationDate(url: cached),
           dstMod >= srcMod {
            return cached
        }

        let compiled = try MLModel.compileModel(at: source)
        // MLModel.compileModel writes to a temporary location; move it into
        // our cache so it survives past the current process.
        if FileManager.default.fileExists(atPath: cached.path) {
            try? FileManager.default.removeItem(at: cached)
        }
        try FileManager.default.createDirectory(
            at: cached.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.moveItem(at: compiled, to: cached)
        return cached
    }

    private static func resourceURL(named name: String, extension ext: String) -> URL? {
        // Check the SPM-generated bundle first, then fall back to the main
        // bundle (covers the non-SPM vendored case).
        if let url = Bundle.module.url(forResource: name, withExtension: ext) {
            return url
        }
        if let url = Bundle.main.url(forResource: name, withExtension: ext) {
            return url
        }
        return nil
    }

    private static func resourceURL(named name: String, extensionOrThrow ext: String) throws -> URL {
        if let url = resourceURL(named: name, extension: ext) {
            return url
        }
        throw WakeWordError.bundledResourceMissing(name: "\(name).\(ext)")
    }

    private static func cacheDirectory() throws -> URL {
        let fm = FileManager.default
        let base = try fm.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = base.appendingPathComponent("LiveKitWakeWord", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func modificationDate(url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }
}
