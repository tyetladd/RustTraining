"""A model-free backend used for smoke tests and CI.

It does not clone anything and it does not speak: it renders a buzz at the
reference speaker's estimated pitch, with one amplitude bump per syllable. That
is enough to exercise the whole pipeline (profile building, normalization,
stress placement, chunking, mixing, encoding) on machines without torch.
"""

from __future__ import annotations

import numpy as np

from voice_clone_tts.backends.base import SynthesisRequest, TTSBackend, register_backend
from voice_clone_tts.profile import load_reference_audio
from voice_clone_tts.text.stress import StressStyle, strip_stress_marks

_VOWELS = set("аеиоуыэюяёaeiouy")


def estimate_f0(samples: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 350.0) -> float:
    """Rough autocorrelation pitch estimate over the loudest second of audio."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size < sample_rate // 4:
        return 120.0
    window = samples[: sample_rate]
    window = window - float(window.mean())
    correlation = np.correlate(window, window, mode="full")[window.size - 1:]
    lag_min, lag_max = int(sample_rate / fmax), int(sample_rate / fmin)
    if lag_max >= correlation.size:
        lag_max = correlation.size - 1
    if lag_min >= lag_max:
        return 120.0
    lag = int(np.argmax(correlation[lag_min:lag_max])) + lag_min
    return float(sample_rate / lag) if lag else 120.0


@register_backend
class DummyBackend(TTSBackend):
    """Deterministic placeholder synthesizer — no model downloads."""

    name = "dummy"
    display_name = "Dummy (test tone)"
    description = "Pitch-matched buzz; for pipeline testing without neural models"
    supported_languages = None
    requires_reference = False
    clones_voice = False
    native_stress_style = StressStyle.NONE
    sample_rate = 24_000
    install_extra = None

    def load(self) -> None:
        self._f0_cache: dict[str, float] = {}
        super().load()

    def _speaker_f0(self, request: SynthesisRequest) -> float:
        profile = request.profile
        if profile is None or not profile.reference_files:
            return 130.0
        key = str(profile.directory)
        if key not in self._f0_cache:
            samples = load_reference_audio(profile, self.sample_rate)
            self._f0_cache[key] = estimate_f0(samples, self.sample_rate)
        return self._f0_cache[key]

    def synthesize(self, request: SynthesisRequest) -> np.ndarray:
        self.ensure_loaded()
        self.seed_everything(request.seed)
        text = strip_stress_marks(request.text)
        syllables = max(1, sum(1 for char in text.lower() if char in _VOWELS))
        f0 = self._speaker_f0(request)

        duration = min(60.0, max(0.4, syllables * 0.18 / max(0.25, request.speed)))
        t = np.arange(int(duration * self.sample_rate), dtype=np.float32) / self.sample_rate
        # Buzz with a few harmonics, so it has a voice-like spectrum.
        wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
        syllable_rate = syllables / duration
        envelope = 0.5 * (1.0 - np.cos(2 * np.pi * syllable_rate * t))
        out = (wave * envelope * 0.1).astype(np.float32)
        return out
