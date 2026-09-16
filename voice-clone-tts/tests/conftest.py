"""Shared fixtures: synthetic reference audio and a fake accentor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_clone_tts import audio as audio_utils  # noqa: E402

SR = 24_000


def make_speech_like(
    seconds: float = 6.0,
    sample_rate: int = SR,
    f0: float = 120.0,
    *,
    silence_head: float = 0.5,
    gaps: tuple[tuple[float, float], ...] = ((2.0, 2.6),),
    noise: float = 0.001,
) -> np.ndarray:
    """A buzzy, amplitude-modulated tone with pauses — enough for VAD tests."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * 4.0 * t))  # ~4 syllables/second
    audio = (wave * envelope * 0.3).astype(np.float32)
    audio[: int(silence_head * sample_rate)] = 0.0
    for start, end in gaps:
        audio[int(start * sample_rate) : int(end * sample_rate)] = 0.0
    rng = np.random.default_rng(0)
    return (audio + rng.normal(0.0, noise, audio.size).astype(np.float32)).astype(np.float32)


@pytest.fixture
def sample_rate() -> int:
    return SR


@pytest.fixture
def speech_audio() -> np.ndarray:
    return make_speech_like()


@pytest.fixture
def reference_wav(tmp_path: Path, speech_audio: np.ndarray) -> Path:
    path = tmp_path / "reference.wav"
    audio_utils.save_audio(path, speech_audio, SR)
    return path


class FakeAccentor:
    """Deterministic stand-in for silero-stress (no torch needed in tests)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> str:
        self.calls.append(text)
        out = []
        for word in text.split(" "):
            lowered = word.lower()
            index = next((i for i, ch in enumerate(lowered) if ch in "аеиоуыэюяё"), None)
            out.append(word if index is None else word[:index] + "+" + word[index:])
        return " ".join(out)


@pytest.fixture
def fake_accentor() -> FakeAccentor:
    return FakeAccentor()
