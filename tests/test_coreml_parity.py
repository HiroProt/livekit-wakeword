"""Numerical parity tests: ONNX runtime vs Core ML.

Every wake-word stage (mel frontend, embedding CNN, classifier, end-to-end)
must produce the same score regardless of backend, otherwise on-device
detection will silently disagree with training-time evaluation.

These tests are macOS-only because Core ML prediction requires the system
frameworks. They also require the ``coreml`` optional extra (``coremltools``
+ ``onnx2torch``) to be installed.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.models.pipeline import WakeWordClassifier

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Core ML prediction only runs on macOS",
)


def _require_coreml():
    try:
        import coremltools  # noqa: F401
        import onnx2torch  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Core ML parity tests need [coreml] extras: {e}")


@pytest.fixture(scope="module")
def mel_onnx_path() -> Path:
    return Path("src/livekit/wakeword/resources/melspectrogram.onnx").resolve()


@pytest.fixture(scope="module")
def embedding_onnx_path() -> Path:
    return Path("src/livekit/wakeword/resources/embedding_model.onnx").resolve()


@pytest.fixture(scope="module")
def classifier_onnx_path() -> Path:
    """A trained classifier ONNX that lives with the Rust integration tests."""
    candidates = [
        Path(__file__).resolve().parents[2]
        / "rust-sdks/livekit-wakeword/tests/fixtures/hey_livekit.onnx",
        Path("rust-sdks/livekit-wakeword/tests/fixtures/hey_livekit.onnx").resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("hey_livekit.onnx fixture not found")


@pytest.fixture(scope="module")
def positive_wav() -> Path:
    for c in [
        Path(__file__).resolve().parents[2]
        / "rust-sdks/livekit-wakeword/tests/fixtures/positive.wav",
        Path("rust-sdks/livekit-wakeword/tests/fixtures/positive.wav").resolve(),
    ]:
        if c.exists():
            return c
    pytest.skip("positive.wav fixture not found")


@pytest.fixture(scope="module")
def negative_wav() -> Path:
    for c in [
        Path(__file__).resolve().parents[2]
        / "rust-sdks/livekit-wakeword/tests/fixtures/negative.wav",
        Path("rust-sdks/livekit-wakeword/tests/fixtures/negative.wav").resolve(),
    ]:
        if c.exists():
            return c
    pytest.skip("negative.wav fixture not found")


def _load_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
    return sr, samples.astype(np.float32)


def test_mel_parity(mel_onnx_path: Path) -> None:
    _require_coreml()
    warnings.filterwarnings("ignore")
    import coremltools as ct
    import onnxruntime as ort

    from livekit.wakeword.export.coreml import convert_mel_frontend

    rng = np.random.default_rng(0)
    audios = [rng.standard_normal((1, n), dtype=np.float32) * 0.1 for n in (16000, 24000, 32000)]

    sess = ort.InferenceSession(str(mel_onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "mel.mlpackage"
        convert_mel_frontend(mel_onnx_path, out, fp32=True)
        mlmodel = ct.models.MLModel(str(out))

        for audio in audios:
            onnx_raw = sess.run(None, {in_name: audio})[0]
            # ONNX output is (1, 1, frames, 32); normalize + squeeze like
            # the Python inference path does.
            onnx_norm = (onnx_raw.reshape(-1, 32)) / 10.0 + 2.0

            cm_out = mlmodel.predict({"audio": audio})["mel"]

            # Shapes must agree
            assert cm_out.shape == onnx_norm.shape, (cm_out.shape, onnx_norm.shape)
            diff = np.abs(onnx_norm - cm_out)
            assert diff.max() < 1e-3, f"max diff {diff.max()} too large for fp32"


def test_embedding_parity(embedding_onnx_path: Path) -> None:
    _require_coreml()
    warnings.filterwarnings("ignore")
    import coremltools as ct
    import onnxruntime as ort

    from livekit.wakeword.export.coreml import convert_embedding_model

    sess = ort.InferenceSession(str(embedding_onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 76, 32, 1), dtype=np.float32)

    onnx_out = sess.run(None, {in_name: x})[0].squeeze((1, 2))  # (16, 96)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "emb.mlpackage"
        convert_embedding_model(embedding_onnx_path, out, fp32=True)
        mlmodel = ct.models.MLModel(str(out))
        cm_out = mlmodel.predict({"mel_windows": x})["embeddings"]  # (16, 96)
        diff = np.abs(onnx_out - cm_out)
        # fp32 should agree to ~1e-4; fp16 drifts by ~5e-2.
        assert diff.max() < 1e-3, f"max diff {diff.max()} too large"

        out16 = Path(d) / "emb16.mlpackage"
        convert_embedding_model(embedding_onnx_path, out16, fp32=False)
        mlmodel16 = ct.models.MLModel(str(out16))
        cm_out16 = mlmodel16.predict({"mel_windows": x})["embeddings"]
        diff16 = np.abs(onnx_out - cm_out16)
        assert diff16.max() < 0.2, f"fp16 max diff {diff16.max()} surprisingly large"


def test_classifier_parity(tmp_path: Path) -> None:
    _require_coreml()
    warnings.filterwarnings("ignore")
    import coremltools as ct
    import onnxruntime as ort

    from livekit.wakeword.export.coreml import export_classifier_coreml
    from livekit.wakeword.export.onnx import export_classifier

    cfg = WakeWordConfig(model_name="parity_test", target_phrases=["hey parity"])
    model = WakeWordClassifier(cfg)
    model.eval()

    pt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), pt_path)

    onnx_path = tmp_path / "model.onnx"
    export_classifier(cfg, pt_path, onnx_path)

    mlpkg_path = tmp_path / "model.mlpackage"
    export_classifier_coreml(cfg, pt_path, mlpkg_path, fp32=True)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name
    mlmodel = ct.models.MLModel(str(mlpkg_path))

    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.standard_normal((1, 16, 96), dtype=np.float32)
        onnx_score = sess.run(None, {onnx_in: x})[0]
        cm_score = mlmodel.predict({"embeddings": x})["score"]
        diff = np.abs(onnx_score - cm_score)
        assert diff.max() < 1e-4, f"classifier max diff {diff.max()}"


def _predict_end_to_end_onnx(
    audio: np.ndarray,
    mel_session,
    emb_session,
    cls_session,
) -> float:
    """Reference end-to-end implementation mirroring inference/model.py."""
    mel_in = mel_session.get_inputs()[0].name
    emb_in = emb_session.get_inputs()[0].name
    cls_in = cls_session.get_inputs()[0].name

    audio_f32 = audio.astype(np.float32)
    if audio_f32.ndim == 1:
        audio_f32 = audio_f32[np.newaxis, :]
    mel_raw = mel_session.run(None, {mel_in: audio_f32})[0]
    if mel_raw.ndim == 4:
        mel_raw = mel_raw[:, 0, :, :]
    mel = (mel_raw[0] / 10.0) + 2.0  # (frames, 32)
    if mel.shape[0] < 76:
        return 0.0

    embeddings = []
    for start in range(0, mel.shape[0] - 76 + 1, 8):
        w = mel[start : start + 76][np.newaxis, :, :, np.newaxis].astype(np.float32)
        e = emb_session.run(None, {emb_in: w})[0]
        embeddings.append(e.squeeze())

    if len(embeddings) < 16:
        return 0.0
    emb_seq = np.stack(embeddings[-16:], axis=0)[np.newaxis, :, :].astype(np.float32)
    score = cls_session.run(None, {cls_in: emb_seq})[0]
    return float(score[0, 0])


def _predict_end_to_end_coreml(
    audio: np.ndarray,
    mel_model,
    emb_model,
    cls_model,
) -> float:
    audio_f32 = audio.astype(np.float32)
    if audio_f32.ndim == 1:
        audio_f32 = audio_f32[np.newaxis, :]
    mel = mel_model.predict({"audio": audio_f32})["mel"]  # (frames, 32)
    if mel.shape[0] < 76:
        return 0.0

    windows = []
    for start in range(0, mel.shape[0] - 76 + 1, 8):
        windows.append(mel[start : start + 76])
    if len(windows) < 16:
        return 0.0
    # Core ML embedding input is a fixed batch of 16, so always pass the last 16.
    x = np.stack(windows[-16:], axis=0)[..., np.newaxis].astype(np.float32)
    emb = emb_model.predict({"mel_windows": x})["embeddings"]  # (16, 96)
    emb_seq = emb[np.newaxis, :, :].astype(np.float32)
    score = cls_model.predict({"embeddings": emb_seq})["score"]
    return float(score[0, 0])


def _embeddings_through_onnx(
    audio: np.ndarray,
    mel_session,
    emb_session,
) -> np.ndarray | None:
    """Return the last 16 96-dim embeddings (same ones the classifier sees)."""
    mel_in = mel_session.get_inputs()[0].name
    emb_in = emb_session.get_inputs()[0].name
    audio_f32 = audio.astype(np.float32)
    if audio_f32.ndim == 1:
        audio_f32 = audio_f32[np.newaxis, :]
    mel_raw = mel_session.run(None, {mel_in: audio_f32})[0]
    if mel_raw.ndim == 4:
        mel_raw = mel_raw[:, 0, :, :]
    mel = (mel_raw[0] / 10.0) + 2.0
    if mel.shape[0] < 76:
        return None

    out = []
    for start in range(0, mel.shape[0] - 76 + 1, 8):
        w = mel[start : start + 76][np.newaxis, :, :, np.newaxis].astype(np.float32)
        e = emb_session.run(None, {emb_in: w})[0]
        out.append(e.squeeze())
    if len(out) < 16:
        return None
    return np.stack(out[-16:], axis=0).astype(np.float32)


def _embeddings_through_coreml(
    audio: np.ndarray,
    mel_model,
    emb_model,
) -> np.ndarray | None:
    audio_f32 = audio.astype(np.float32)
    if audio_f32.ndim == 1:
        audio_f32 = audio_f32[np.newaxis, :]
    mel = mel_model.predict({"audio": audio_f32})["mel"]
    if mel.shape[0] < 76:
        return None
    wins = []
    for start in range(0, mel.shape[0] - 76 + 1, 8):
        wins.append(mel[start : start + 76])
    if len(wins) < 16:
        return None
    x = np.stack(wins[-16:], axis=0)[..., np.newaxis].astype(np.float32)
    return emb_model.predict({"mel_windows": x})["embeddings"].astype(np.float32)


def test_end_to_end_wav_parity(
    mel_onnx_path: Path,
    embedding_onnx_path: Path,
    positive_wav: Path,
    negative_wav: Path,
    tmp_path: Path,
) -> None:
    """Real audio through mel+embedding: Core ML must match ONNX closely.

    We compare the 16x96 embedding tensor that the classifier would see,
    which is the authoritative interface between the frontend and the
    classifier head. If this matches, any classifier will produce the same
    score regardless of backend (and ``test_classifier_parity`` covers the
    classifier itself).
    """
    _require_coreml()
    warnings.filterwarnings("ignore")
    import coremltools as ct
    import onnxruntime as ort

    from livekit.wakeword.export.coreml import (
        convert_embedding_model,
        convert_mel_frontend,
    )

    mel_pkg = tmp_path / "mel.mlpackage"
    emb_pkg = tmp_path / "emb.mlpackage"
    convert_mel_frontend(mel_onnx_path, mel_pkg, fp32=True)
    convert_embedding_model(embedding_onnx_path, emb_pkg, fp32=True)

    mel_o = ort.InferenceSession(str(mel_onnx_path), providers=["CPUExecutionProvider"])
    emb_o = ort.InferenceSession(str(embedding_onnx_path), providers=["CPUExecutionProvider"])
    mel_m = ct.models.MLModel(str(mel_pkg))
    emb_m = ct.models.MLModel(str(emb_pkg))

    for wav_path, label in ((positive_wav, "positive"), (negative_wav, "negative")):
        sr, samples = _load_wav(wav_path)
        assert sr == 16000, f"fixtures expected to be 16 kHz, got {sr}"

        onnx_embs = _embeddings_through_onnx(samples, mel_o, emb_o)
        cm_embs = _embeddings_through_coreml(samples, mel_m, emb_m)
        assert onnx_embs is not None and cm_embs is not None, f"{label} too short"
        diff = np.abs(onnx_embs - cm_embs)
        # fp32 end-to-end — expect tight parity on the final embeddings.
        assert diff.max() < 1e-2, (
            f"{label} embeddings diverged: max diff {diff.max():.4f}, "
            f"mean {diff.mean():.4f}"
        )
