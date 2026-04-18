# Core ML Export (Apple Platforms)

`livekit-wakeword` ships with a Core ML export path and a native Swift package
(`LiveKitWakeWord`) so the full wake-word pipeline — mel spectrogram →
speech embedding CNN → classifier head — can run directly on the Apple
Neural Engine, GPU, or CPU, without any ONNX runtime dependency.

This is the recommended path for shipping wake words in iOS / macOS apps: the
Swift SDK is ~1.5 MB of compiled code plus ~1.2 MB of bundled Core ML frontend
models. Classifier `.mlpackage` files are typically 300–1000 KB.

**Source:**
- `src/livekit/wakeword/export/coreml.py` — Python export
- `swift/` — Swift Package (`LiveKitWakeWord`)
- `examples/ios_wakeword/` — iOS + macOS demo app (under `rust-sdks/examples/`)

## Requirements

- macOS 13 or later for running conversion (Core ML Tools and the converted
  models only execute on Apple platforms)
- Python 3.11+
- `livekit-wakeword[coreml]` extra — pulls in `coremltools`, `onnx2torch`,
  `onnx`, and a pinned PyTorch

Runtime requirements for the Swift package: **iOS 16 / macOS 13** and Swift
5.9. The bundled frontend models use ML Program (`mlprogram`) format with fp16
weights, which is required for ANE scheduling.

## Pipeline overview

```
┌─────────────────┐   ┌─────────────────┐   ┌────────────────────┐
│  mel frontend   │──▶│  embedding CNN  │──▶│ classifier head(s) │
│  (bundled)      │   │  (bundled)      │   │  (per wake word)   │
└─────────────────┘   └─────────────────┘   └────────────────────┘
   melspectrogram.mlpackage   embedding_model.mlpackage   *.mlpackage
```

The first two models are frozen — they are identical across every wake word.
They ship inside the Swift package as `.mlpackage` resources. Apps only need
to bundle a **classifier** `.mlpackage` for each wake word they want to
detect.

## Exporting a classifier from training

If you are training a model with `livekit-wakeword`, the simplest path is the
CLI:

```bash
livekit-wakeword export configs/prod.yaml --format coreml
```

This runs the trained PyTorch head (`WakeWordClassifier`) through
`torch.export` and `coremltools.convert`, producing
`output/<model_name>/<model_name>.mlpackage`.

Use `--format both` to produce ONNX and Core ML side-by-side, or `--fp32` to
keep weights in float32 (useful for numerical parity testing against ONNX
Runtime — note that fp32 weights disable ANE scheduling).

### From Python

```python
from pathlib import Path
from livekit.wakeword import load_config
from livekit.wakeword.export import export_classifier_coreml

config = load_config("configs/prod.yaml")
export_classifier_coreml(
    config,
    model_path=Path(f"output/{config.model_name}/{config.model_name}.pt"),
    output_path=Path(f"output/{config.model_name}/{config.model_name}.mlpackage"),
)
```

## Converting an existing ONNX classifier

If you already have a trained `.onnx` classifier (e.g. shipped with an older
release or produced by openWakeWord), you can convert it without round-tripping
through training:

```python
from pathlib import Path
from livekit.wakeword.export import convert_classifier_from_onnx

convert_classifier_from_onnx(
    Path("hey_livekit.onnx"),
    Path("hey_livekit.mlpackage"),
)
```

Under the hood this normalizes a couple of opset-18 idioms that the current
`onnx2torch` release does not handle (the `ReduceMean` axes-as-input form and
`Reshape`'s `allowzero=1` attribute), then goes
ONNX → `onnx2torch` → `torch.jit.trace` → `coremltools`.

## Regenerating the bundled frontend models

The mel spectrogram and speech-embedding models ship as `.mlpackage` resources
inside the Swift package under `swift/Sources/LiveKitWakeWord/Resources/`. You
should rarely need to regenerate them, but if you update the underlying
frozen ONNX artifacts, call:

```python
from pathlib import Path
from livekit.wakeword.export import convert_frontend_models

convert_frontend_models(
    mel_onnx=Path("src/livekit/wakeword/inference/resources/melspectrogram.onnx"),
    embedding_onnx=Path("src/livekit/wakeword/inference/resources/embedding_model.onnx"),
    output_dir=Path("swift/Sources/LiveKitWakeWord/Resources"),
)
```

Conversion notes:

- The mel spectrogram is **reimplemented in pure PyTorch** (weights loaded
  from the ONNX initializers) because the original torchlibrosa graph uses a
  dynamic `Clip(max = max(x) - 80)` that `onnx2torch` cannot lower. The
  reimplementation is numerically byte-equivalent with the ONNX graph.
- The mel package uses a Core ML `RangeDim` so a single artifact handles 1–3 s
  audio chunks at 16 kHz (`min_samples=16000`, `max_samples=48000`).
- The embedding CNN is exported with a **fixed batch of 16**, which matches
  how the classifier is always called (16 sliding windows). Dynamic batch
  would require rewriting the ONNX `Reshape` nodes, which is not worth the
  complexity.

## Using the Swift package

`LiveKitWakeWord` exposes two entry points:

- `WakeWordModel` — stateless: feed PCM, get `[classifierName: score]`
- `WakeWordListener` — actor-wrapped `AVAudioEngine` pipeline that emits
  detections via `AsyncStream`

```swift
import LiveKitWakeWord

let classifier = Bundle.main.url(forResource: "hey_livekit", withExtension: "mlpackage")!
let model = try WakeWordModel(
    classifiers: [classifier],
    sampleRate: 48_000,
    computeUnits: .all    // .all | .cpuAndGPU | .cpuOnly
)
let scores = try model.predict(pcmInt16)

let listener = WakeWordListener(model: model, threshold: 0.5, debounce: 2.0)
try await listener.start()
for await detection in listener.detections() {
    print("\(detection.name) @ \(detection.confidence)")
}
```

### Resource loading

`.mlpackage` is a directory and Xcode/SPM doesn't auto-compile it on copy, so
the package declares its frontend models as `.copy` resources and runs
`MLModel.compileModel(at:)` on first use. Compiled `.mlmodelc` artifacts are
cached in the app's caches directory, so subsequent loads are effectively
instant.

### Compute units

`MLComputeUnits.all` lets Core ML pick per-layer between the ANE, GPU, and
CPU; this is usually fastest and most power-efficient. `.cpuAndGPU` is useful
for debugging or comparing numerics on ANE-less hardware; `.cpuOnly` is
primarily for determinism.

## Parity testing

The repository ships macOS-only Python tests that verify Core ML predictions
match ONNX Runtime within tight tolerances — see `tests/test_coreml_parity.py`:

- `test_mel_parity` — mel spectrogram outputs (≤ `1e-4` max diff)
- `test_embedding_parity` — embeddings, both fp32 and fp16 weights
- `test_classifier_parity` — classifier scores
- `test_end_to_end_wav_parity` — compares full mel + embedding sequences on
  the `positive.wav` / `negative.wav` fixtures

Run them with:

```bash
pytest tests/test_coreml_parity.py -v
```

## Benchmarks

The Swift test target includes a benchmark harness that measures
`WakeWordModel.predict` latency across the three `MLComputeUnits` values.
It's gated by an environment variable so it doesn't run in normal
`swift test` invocations:

```bash
cd swift && LIVEKIT_WAKEWORD_BENCH=1 swift test --filter BenchmarkTests
```

Typical latencies on an Apple M-series laptop (2 s audio window, 50 runs):

| Compute units | p50   | p95   |
| ------------- | ----- | ----- |
| `.all`        | ~4 ms | ~6 ms |
| `.cpuAndGPU`  | ~6 ms | ~9 ms |
| `.cpuOnly`    | ~8 ms | ~12 ms |

These numbers are dominated by the embedding CNN; the mel and classifier
stages are each < 1 ms. For comparison, ONNX Runtime on the same hardware
runs the same pipeline in roughly 15–25 ms depending on thread count.

## Gotchas

- **`@preconcurrency import AVFoundation`** is required in consumers if they
  target strict concurrency — `AVAudioPCMBuffer` isn't `Sendable` yet.
- When bundling a classifier `.mlpackage` inside an app, add it as a
  `.copy` resource (not `.process`) to avoid the Xcode 15+ "multiple
  resources named `Manifest.json`" SPM bug.
- If you feed audio at a non-16-kHz sample rate, `WakeWordModel` resamples
  internally using `AVAudioConverter`. This adds a small per-window overhead
  (~0.2 ms on M1) compared to feeding 16 kHz PCM directly.
