import numpy as np
import pytest

from voice_clone_tts.backends import SynthesisRequest, TTSBackend
from voice_clone_tts.config import PipelineConfig
from voice_clone_tts.errors import BackendError, TextError
from voice_clone_tts.pipeline import prepare_text, synthesize
from voice_clone_tts.text.stress import StressStyle, convert_stress_marks


class RecordingBackend(TTSBackend):
    """Captures what the pipeline asks it to say."""

    name = "recording"
    supported_languages = {"ru", "en"}
    requires_reference = False
    clones_voice = True
    native_stress_style = StressStyle.PLUS
    sample_rate = 16_000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests: list[SynthesisRequest] = []

    def synthesize(self, request: SynthesisRequest) -> np.ndarray:
        self.requests.append(request)
        return np.full(self.sample_rate // 2, 0.1, dtype=np.float32)


def ru_config() -> PipelineConfig:
    config = PipelineConfig(language="ru")
    config.asr.enabled = False
    config.text.stress = False
    config.reference.min_segment_sec = 1.0
    return config


def test_prepare_text_normalizes_and_chunks(fake_accentor, monkeypatch):
    import voice_clone_tts.pipeline as synth_module

    monkeypatch.setattr(synth_module, "put_stress",
                        lambda text, spec, **kw: fake_accentor(text))
    config = PipelineConfig()
    prepared, chunks = prepare_text("У меня 2 кота. " * 20, "ru", config)
    assert "дв+а" in prepared  # "2" expanded, then stressed
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_prepare_text_rejects_empty_input():
    with pytest.raises(TextError):
        prepare_text("   ", "ru")


def test_pipeline_end_to_end(reference_wav, tmp_path):
    backend = RecordingBackend()
    out = tmp_path / "out.wav"
    result = synthesize(
        voice=reference_wav,
        text="Привет! Это тест синтеза речи.",
        out_path=out,
        config=ru_config(),
        backend=backend,
    )
    assert out.exists()
    assert result.language == "ru"
    assert result.sample_rate == backend.sample_rate
    assert result.duration >= 0.5
    assert len(backend.requests) == len(result.chunks)
    assert result.profile is not None and result.profile.reference_files
    assert set(result.timings) == {"profile", "text", "load", "synthesis"}


def test_pipeline_reuses_a_saved_profile(reference_wav, tmp_path):
    from voice_clone_tts.profile import build_profile

    profile = build_profile(reference_wav, out_dir=tmp_path / "p", config=ru_config())
    backend = RecordingBackend()
    result = synthesize(
        text="Ещё один тест.", profile=tmp_path / "p", config=ru_config(), backend=backend
    )
    assert result.profile is not None
    assert result.profile.directory == profile.directory
    assert backend.requests[0].profile is result.profile


def test_stress_style_follows_the_backend(reference_wav, monkeypatch, fake_accentor):
    import voice_clone_tts.pipeline as synth_module

    captured = {}

    def fake_put_stress(text, spec, *, style, **kwargs):
        captured["style"] = style
        return fake_accentor(text)

    monkeypatch.setattr(synth_module, "put_stress", fake_put_stress)
    config = ru_config()
    config.text.stress = True

    backend = RecordingBackend()
    synthesize(voice=reference_wav, text="Замок.", config=config, backend=backend)
    assert captured["style"] is StressStyle.PLUS
    assert "+" in backend.requests[0].text


def test_backend_language_mismatch_is_reported(reference_wav):
    class EnglishOnly(RecordingBackend):
        name = "english-only"
        supported_languages = {"en"}

    with pytest.raises(BackendError, match="does not support"):
        synthesize(voice=reference_wav, text="Привет", config=ru_config(), backend=EnglishOnly())


def test_cloning_backend_without_reference_is_reported():
    class NeedsVoice(RecordingBackend):
        name = "needs-voice"
        requires_reference = True

    with pytest.raises(BackendError, match="needs --voice"):
        synthesize(text="Привет", config=ru_config(), backend=NeedsVoice())


def test_language_can_be_overridden(reference_wav):
    backend = RecordingBackend()
    result = synthesize(
        voice=reference_wav, text="Hello world.", language="en",
        config=ru_config(), backend=backend,
    )
    assert result.language == "en"
    assert backend.requests[0].language.code == "en"


def test_output_can_be_resampled(reference_wav, tmp_path):
    config = ru_config()
    config.synthesis.output_sample_rate = 48_000
    result = synthesize(
        voice=reference_wav, text="Тест.", out_path=tmp_path / "o.wav",
        config=config, backend=RecordingBackend(),
    )
    assert result.sample_rate == 48_000


def test_explicit_stress_style_overrides_the_backend(reference_wav, monkeypatch, fake_accentor):
    import voice_clone_tts.pipeline as synth_module

    captured = {}

    def fake_put_stress(text, spec, *, style, **kwargs):
        captured["style"] = style
        return convert_stress_marks(fake_accentor(text), style)

    monkeypatch.setattr(synth_module, "put_stress", fake_put_stress)
    config = ru_config()
    config.text.stress = True
    config.text.stress_style = "none"  # RecordingBackend asks for "plus"

    backend = RecordingBackend()
    synthesize(voice=reference_wav, text="Замок.", config=config, backend=backend)
    assert captured["style"] is StressStyle.NONE
    assert "+" not in backend.requests[0].text
