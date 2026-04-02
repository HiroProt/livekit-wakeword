# Export & Inference

The export stage converts the trained PyTorch classifier to ONNX for deployment. The inference API provides `WakeWordModel` for prediction and `WakeWordListener` for async microphone detection.

**Source:** `src/livekit/wakeword/export/onnx.py`, `src/livekit/wakeword/inference/model.py`, `src/livekit/wakeword/inference/listener.py`
**CLI:** `livekit-wakeword export <config>`

## ONNX Export

### Classifier Export

`export_classifier()` exports the trained PyTorch classifier head to ONNX format.

| Property | Value |
|----------|-------|
| Input name | `embeddings` |
| Input shape | `(1, 16, 96)` with dynamic batch axis |
| Output name | `score` |
| Output shape | `(1, 1)` with dynamic batch axis |
| Opset version | 18 |

### INT8 Quantization

`quantize_onnx()` applies dynamic INT8 quantization using `onnxruntime.quantization`:

- Weight type: `QuantType.QInt8`
- Output filename: `<model_name>.int8.onnx`

Enable via the `--quantize` flag:

```bash
livekit-wakeword export configs/hey_jarvis.yaml --quantize
```

### Export Entry Point

`run_export()` loads the trained model from `output/<model_name>/<model_name>.pt`, exports it to ONNX, and optionally quantizes it. Raises `FileNotFoundError` if the trained model doesn't exist.

## Inference API

**Source:** `src/livekit/wakeword/inference/model.py`, `src/livekit/wakeword/inference/listener.py`, `src/livekit/wakeword/inference/vad.py`

### WakeWordModel

The `WakeWordModel` class is a stateless prediction API for wake word detection. Pass a complete audio window (~2 seconds) and receive confidence scores. An optional Silero VAD gate can skip expensive inference when no speech is detected.

```python
from livekit.wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.onnx"])

# With VAD gate enabled
model = WakeWordModel(models=["hey_livekit.onnx"], vad_enabled=True, vad_threshold=0.5)

# Pass ~2 seconds of 16kHz audio
scores = model.predict(audio_chunk)
# Returns: {"hey_livekit": 0.95}
```

#### Initialization

```python
WakeWordModel(
    models: list[str | Path] | None = None,  # Paths to ONNX classifiers
    vad_enabled: bool = False,                # Gate inference with Silero VAD
    vad_threshold: float = 0.5,               # Min speech probability to run inference
)
```

Feature extraction models (`melspectrogram.onnx`, `embedding_model.onnx`) and the Silero VAD model (`silero_vad.onnx`) are bundled with the package and loaded automatically.

#### VAD Gate

When `vad_enabled=True`, `predict()` first runs Silero VAD on the audio chunk. If the peak speech probability is below `vad_threshold`, all models return 0.0 without running the full mel → embedding → classifier pipeline. This saves CPU during silence and reduces false positives from noise.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(audio_chunk)` | `dict[str, float]` | Scores for each loaded model (0-1) |
| `load_model(path, name)` | `None` | Load additional wake word model |

#### Audio Input

- **Format:** 16kHz mono, int16 or float32
- **Chunk size:** ~2 seconds (32,000 samples) recommended — yields 16 embeddings for the classifier
- **Stateless:** No internal audio buffering; the caller manages the audio window

### WakeWordListener

The `WakeWordListener` class provides async microphone detection with debouncing.

```python
import asyncio
from livekit.wakeword import WakeWordModel, WakeWordListener

model = WakeWordModel(models=["hey_livekit.onnx"])

async def main():
    async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")

asyncio.run(main())
```

#### Initialization

```python
WakeWordListener(
    model: WakeWordModel,           # WakeWordModel instance with loaded classifiers
    threshold: float = 0.5,         # Detection threshold (0-1)
    debounce: float = 2.0,          # Minimum seconds between detections
    noise_suppression: bool = True,  # WebRTC noise suppression via LiveKit APM
    high_pass_filter: bool = True,   # Remove low-frequency rumble
    auto_gain_control: bool = True,  # Normalize mic volume
)
```

#### Detection Result

```python
@dataclass
class Detection:
    name: str        # Model name that triggered
    confidence: float  # Score (0-1)
    timestamp: float   # Monotonic time
```

#### Lifecycle

The listener is designed as an async context manager. On each `__aenter__`, all internal state is reset — including the audio buffer, error state, and detection queue — so the same listener instance can be safely reused across multiple `async with` blocks without stale detections carrying over.

#### Audio Capture

Uses PyAudio to capture from the default microphone. Each frame is processed through LiveKit's `AudioProcessingModule` for noise suppression, high-pass filtering, and auto gain control before buffering.

| Parameter | Value |
|-----------|-------|
| Format | int16 (paInt16) |
| Channels | 1 (mono) |
| Sample rate | 16,000 Hz |
| Buffer size | 1,280 samples (80ms) |
| APM frame size | 160 samples (10ms) |

The 80ms capture frames are split into 8 × 10ms sub-frames for APM processing (WebRTC requires exactly 10ms frames), then reassembled before buffering.

#### Audio Processing (LiveKit APM)

The listener uses LiveKit's `AudioProcessingModule` (WebRTC-based) to clean captured audio before inference:

| Feature | Default | Description |
|---------|---------|-------------|
| `noise_suppression` | `True` | WebRTC noise suppression |
| `high_pass_filter` | `True` | Removes low-frequency rumble |
| `auto_gain_control` | `True` | Normalizes microphone volume |

All three can be disabled by passing `False` to the constructor. When all are disabled, the APM is not created and raw audio passes through unchanged.
