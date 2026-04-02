"""Tests for SileroVAD speech detection."""

from __future__ import annotations

import numpy as np
import pytest

from livekit.wakeword.inference.vad import SileroVAD


@pytest.fixture(scope="module")
def vad() -> SileroVAD:
    return SileroVAD()


class TestSileroVAD:
    def test_silence_returns_low_probability(self, vad: SileroVAD) -> None:
        """Pure silence should score well below 0.5."""
        silence = np.zeros(32000, dtype=np.float32)  # 2 seconds
        prob = vad.check_speech(silence)
        assert prob < 0.3, f"Expected low speech prob for silence, got {prob}"

    def test_int16_input_accepted(self, vad: SileroVAD) -> None:
        """int16 audio should be accepted and converted internally."""
        silence = np.zeros(16000, dtype=np.int16)
        prob = vad.check_speech(silence)
        assert 0.0 <= prob <= 1.0

    def test_short_audio(self, vad: SileroVAD) -> None:
        """Audio shorter than one VAD window should return 0.0."""
        short = np.zeros(256, dtype=np.float32)
        prob = vad.check_speech(short)
        assert prob == 0.0

    def test_output_range(self, vad: SileroVAD) -> None:
        """Speech probability should always be in [0, 1]."""
        # Random noise
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(32000).astype(np.float32) * 0.1
        prob = vad.check_speech(noise)
        assert 0.0 <= prob <= 1.0

    def test_stateless_across_calls(self, vad: SileroVAD) -> None:
        """Each call should use fresh state — results should be deterministic."""
        audio = np.zeros(32000, dtype=np.float32)
        prob1 = vad.check_speech(audio)
        prob2 = vad.check_speech(audio)
        assert prob1 == pytest.approx(prob2), "Stateless calls should give same result"

    def test_nonexistent_model_raises(self, tmp_path) -> None:
        """Should raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            SileroVAD(model_path=tmp_path / "missing.onnx")
