import numpy as np
import pytest

from voice_clone_tts.backends import (
    SynthesisRequest,
    TTSBackend,
    backend_class,
    get_backend,
    iter_backend_info,
    register_backend,
)
from voice_clone_tts.backends.xtts import LICENSE_ENV, XTTSBackend
from voice_clone_tts.errors import BackendError
from voice_clone_tts.text.languages import get_language
from voice_clone_tts.text.stress import StressStyle


def test_builtin_backends_are_registered():
    names = {row["name"] for row in iter_backend_info()}
    assert {"dummy", "xtts", "silero"} <= names


def test_dummy_backend_is_always_available():
    assert backend_class("dummy").is_available()


def test_unknown_backend_raises():
    with pytest.raises(BackendError):
        get_backend("does-not-exist")


def test_backend_needs_a_name():
    with pytest.raises(BackendError):
        register_backend(type("Anon", (TTSBackend,), {"name": "base"}))


def test_dummy_backend_renders_audio():
    backend = get_backend("dummy")
    request = SynthesisRequest(text="Прив+ет, м+ир!", language=get_language("ru"))
    with backend:
        wav = backend.synthesize(request)
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert wav.size > 0 and np.abs(wav).max() > 0


def test_dummy_backend_tracks_the_speaker_pitch(reference_wav, tmp_path):
    from voice_clone_tts.backends.dummy import estimate_f0
    from voice_clone_tts.config import PipelineConfig
    from voice_clone_tts.profile import build_profile, load_reference_audio

    config = PipelineConfig(language="ru")
    config.asr.enabled = False
    profile = build_profile(reference_wav, out_dir=tmp_path / "p", config=config)
    f0 = estimate_f0(load_reference_audio(profile), profile.sample_rate)
    assert 100.0 < f0 < 140.0  # the fixture is a 120 Hz buzz


def test_xtts_requires_license_acknowledgement(monkeypatch):
    monkeypatch.delenv(LICENSE_ENV, raising=False)
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    with pytest.raises(BackendError, match="Coqui Public Model License"):
        XTTSBackend._check_license()
    monkeypatch.setenv(LICENSE_ENV, "1")
    XTTSBackend._check_license()


def test_xtts_language_support():
    backend = XTTSBackend()
    assert backend.supports_language(get_language("ru"))
    assert backend.supports_language(get_language("en"))
    assert not backend.supports_language(get_language("be"))


def test_xtts_prefers_unmarked_text():
    assert XTTSBackend.native_stress_style is StressStyle.NONE


def test_silero_consumes_plus_marks():
    from voice_clone_tts.backends.silero import SileroBackend

    assert SileroBackend.native_stress_style is StressStyle.PLUS
    assert not SileroBackend.clones_voice


def test_silero_rejects_bad_sample_rates():
    with pytest.raises(BackendError):
        get_backend("silero", sample_rate=44_100)
