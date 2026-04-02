"""Silero VAD wrapper for speech detection on audio chunks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..resources import get_vad_model_path

logger = logging.getLogger(__name__)

# Silero VAD v5 expects 512-sample windows at 16 kHz
_VAD_WINDOW = 512
_VAD_SAMPLE_RATE = 16000


class SileroVAD:
    """Stateless speech detector using the Silero VAD ONNX model.

    Each call to :meth:`check_speech` processes a full audio chunk with
    a fresh hidden state, keeping the detector stateless across calls.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else get_vad_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found: {path}\n"
                "Please reinstall livekit-wakeword or download silero_vad.onnx."
            )
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

    def _fresh_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return zeroed (sr, h, c) tensors for a fresh forward pass."""
        sr = np.array(_VAD_SAMPLE_RATE, dtype=np.int64)
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        return sr, h, c

    def check_speech(self, audio: np.ndarray) -> float:
        """Return the peak speech probability across the audio chunk.

        Processes the chunk in 512-sample windows with a fresh hidden
        state each call, so the detector is fully stateless.

        Args:
            audio: 16 kHz audio samples (int16 or float32).

        Returns:
            Peak speech probability in [0, 1].
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = audio.flatten()

        sr, h, c = self._fresh_state()
        peak = 0.0

        for start in range(0, len(audio) - _VAD_WINDOW + 1, _VAD_WINDOW):
            window = audio[start : start + _VAD_WINDOW][np.newaxis, :]
            out, h, c = self._session.run(
                None,
                {"input": window, "sr": sr, "h": h, "c": c},
            )
            prob = float(out[0])
            if prob > peak:
                peak = prob

        return peak
