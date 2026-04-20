# Wake Word Demo (iOS + macOS)

Minimal SwiftUI app that runs the [`LiveKitWakeWord`](../../swift) detector
against the device microphone. Tap **Unmute mic** to start listening;
when the `hey_livekit` classifier score crosses 0.75 the status turns green
and shows `WAKE WORD DETECTED`. A segmented control lets you switch between
ONNX Runtime execution providers (CoreML with ANE+GPU+CPU, CoreML GPU+CPU,
CoreML CPU-only, or plain ORT CPU) at runtime to compare latency.

The same SwiftUI sources build as two apps:

- **`WakewordDemo`** — iOS 16+ (iPhone, iPad, iOS Simulator)
- **`WakewordDemoMac`** — macOS 14+ native app

The macOS build is handy for iterating on the detector without a device.

## Architecture

```
Mic  ─►  AVAudioEngine tap  ─►  Float32→Int16 convert  ─►  2 s Int16 ring buffer
                                                                        │
                                                                        ▼
                                            background queue runs WakeWordModel.predict()
                                                                        │
                                                                        ▼
                                                        @MainActor publishes score + level
                                                                        │
                                                                        ▼
                                                           SwiftUI ContentView (graphs)
```

The Swift side never runs the ML arithmetic; it just buffers audio and calls
into the `WakeWordModel` class from the `LiveKitWakeWord` Swift package. The
mel-spectrogram and embedding `.onnx` models are bundled inside the package
and loaded by ONNX Runtime; only the classifier (`hey_livekit.onnx`) ships
with this demo.

## Prerequisites

- Xcode 15+ (the project currently builds with Xcode 26)
- No Rust toolchain, no UniFFI, no extra CLI installs — the ONNX Runtime
  framework is pulled in transitively as a binary target by SPM
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) only if you want to
  regenerate the `.xcodeproj` from [`project.yml`](./project.yml):

  ```sh
  brew install xcodegen
  ```

## Run the app

Open `WakewordDemo.xcodeproj` in Xcode. The Swift package at
[`../../swift`](../../swift) is picked up as a local SPM dependency; Xcode
resolves it (and fetches the ONNX Runtime binary framework) automatically on
first open.

### macOS (quickest loop)

1. Pick the **`WakewordDemoMac`** scheme and **`My Mac`** as the destination.
2. `Cmd+R`. Grant microphone permission when prompted.
3. Click **Unmute mic** and say "Hey LiveKit". The score should jump toward
   1.0 and the UI flashes `WAKE WORD DETECTED`.

The macOS target is sandboxed with the hardened runtime and only the
`com.apple.security.device.audio-input` entitlement (see
[`WakewordDemo/WakewordDemoMac.entitlements`](./WakewordDemo/WakewordDemoMac.entitlements)).

### iOS

1. Pick the **`WakewordDemo`** scheme and an iOS Simulator or a connected
   device. For hardware, set the target's signing team.
2. `Cmd+R`. Tap **Unmute mic**, grant microphone permission, and say
   "Hey LiveKit".

## Regenerating the Xcode project

The Xcode project is generated from [`project.yml`](./project.yml). If you
edit the yml, regenerate with:

```sh
xcodegen generate
```

## Files

| Path                                         | Purpose                                                           |
| -------------------------------------------- | ----------------------------------------------------------------- |
| `project.yml`                                | XcodeGen spec. Defines the iOS + macOS targets and the local SPM dep. |
| `WakewordDemo/WakewordDemoApp.swift`         | SwiftUI `@main` entry point (shared between iOS + macOS).         |
| `WakewordDemo/ContentView.swift`             | UI: detection-score graph, mic-level UV meter, mic + provider controls. |
| `WakewordDemo/WakewordEngine.swift`          | `AVAudioEngine` tap, Int16 ring buffer, background `predict()`.   |
| `WakewordDemo/Resources/hey_livekit.onnx`    | ONNX wake-word classifier (loaded by default).                    |
| `WakewordDemo/Info.plist`                    | iOS Info.plist (microphone string, orientations, scene).          |
| `WakewordDemo/Info-Mac.plist`                | macOS Info.plist (microphone string, `LSMinimumSystemVersion`).   |
| `WakewordDemo/WakewordDemoMac.entitlements`  | Sandbox + `audio-input` entitlement for the mac target.           |

## Tuning

Constants in [`WakewordEngine.swift`](./WakewordDemo/WakewordEngine.swift):

- `triggerThreshold` (default 0.75): score at which the UI shows a detection.
- `triggerHoldDuration` (default 1.5 s): how long the UI stays green after a hit.
- `predictInterval` (default 0.02 s): minimum time between `predict()` calls.
- `windowSeconds` (default 2.0): size of the rolling window fed to the model.

## Loading additional classifiers

To detect more than just "Hey LiveKit", drop more `.onnx` classifier files
into `WakewordDemo/Resources/` and extend the `classifierURLs` array
constructed in `WakewordEngine.init()`. The `WakeWordModel` constructor
accepts an array of classifier URLs; `predict()` returns a `[String: Float]`
keyed by classifier name (filename stem).

See [`docs/export-and-inference.md`](../../docs/export-and-inference.md) for
how to export your own trained classifier to ONNX.
