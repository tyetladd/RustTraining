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


# --------------------------------------------------------------------------
# voice conversion stage
# --------------------------------------------------------------------------

def _register_fake_converter(name="fake-vc"):
    """A converter that records its calls and shifts amplitude, not pitch."""
    from voice_clone_tts.vc import VoiceConverter, register_converter

    class FakeConverter(VoiceConverter):
        calls: list = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            FakeConverter.calls = []

        @classmethod
        def is_available(cls):
            return True

        def train(self, dataset, out_dir, **kwargs):  # pragma: no cover - unused here
            raise NotImplementedError

        def convert(self, audio, sample_rate, model, *, transpose=0):
            FakeConverter.calls.append((audio.size, sample_rate, model.name, transpose))
            return audio * 0.5

    FakeConverter.name = name
    register_converter(FakeConverter)
    return FakeConverter


def _fake_voice_model(tmp_path, converter_name, median_f0=0.0):
    from voice_clone_tts.vc import VoiceModel

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    model = VoiceModel(
        name="anna", directory=tmp_path / "voice", converter=converter_name,
        checkpoint=checkpoint, median_f0=median_f0,
    )
    model.save()
    return model


def test_voice_conversion_runs_on_every_chunk(reference_wav, tmp_path):
    fake = _register_fake_converter("fake-vc-chunks")
    _fake_voice_model(tmp_path, "fake-vc-chunks")

    config = ru_config()
    config.synthesis.voice_model = str(tmp_path / "voice")
    backend = RecordingBackend()
    result = synthesize(
        voice=reference_wav, text="Первое предложение. Второе предложение.",
        config=config, backend=backend,
    )
    assert len(fake.calls) == len(result.chunks)
    assert result.voice_model == "anna"
    assert "conversion" in result.timings
    assert "anna" in result.describe()


def test_auto_transpose_matches_the_target_pitch(reference_wav, tmp_path, monkeypatch):
    fake = _register_fake_converter("fake-vc-auto")
    _fake_voice_model(tmp_path, "fake-vc-auto", median_f0=220.0)

    class TonalBackend(RecordingBackend):
        name = "tonal"

        def synthesize(self, request):
            self.requests.append(request)
            t = np.arange(self.sample_rate, dtype=np.float32) / self.sample_rate
            return (np.sin(2 * np.pi * 110.0 * t) * 0.3).astype(np.float32)

    config = ru_config()
    config.synthesis.voice_model = str(tmp_path / "voice")
    config.synthesis.transpose = "auto"
    synthesize(voice=reference_wav, text="Тест.", config=config, backend=TonalBackend())

    # 110 Hz -> 220 Hz is exactly one octave.
    assert fake.calls[0][3] == 12


def test_transpose_is_zero_without_a_reference_pitch(reference_wav, tmp_path):
    fake = _register_fake_converter("fake-vc-nopitch")
    _fake_voice_model(tmp_path, "fake-vc-nopitch", median_f0=0.0)

    config = ru_config()
    config.synthesis.voice_model = str(tmp_path / "voice")
    config.synthesis.transpose = "auto"
    synthesize(voice=reference_wav, text="Тест.", config=config, backend=RecordingBackend())
    assert fake.calls[0][3] == 0


def test_unavailable_converter_is_reported(reference_wav, tmp_path):
    from voice_clone_tts.vc import VoiceConverter, register_converter

    class Unavailable(VoiceConverter):
        name = "fake-vc-missing"
        install_hint = "pip install something"

        @classmethod
        def is_available(cls):
            return False

        def train(self, dataset, out_dir, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def convert(self, audio, sample_rate, model, *, transpose=0):  # pragma: no cover
            raise NotImplementedError

    register_converter(Unavailable)
    _fake_voice_model(tmp_path, "fake-vc-missing")

    config = ru_config()
    config.synthesis.voice_model = str(tmp_path / "voice")
    with pytest.raises(BackendError, match="pip install something"):
        synthesize(voice=reference_wav, text="Тест.", config=config, backend=RecordingBackend())
