import json

import pytest

from voice_clone_tts.asr import Transcript, TranscriptSegment, select_reference_segments
from voice_clone_tts.audio import SpeechSegment
from voice_clone_tts.config import PipelineConfig
from voice_clone_tts.errors import AudioError
from voice_clone_tts.profile import SpeakerProfile, build_profile, load_reference_audio


def no_asr_config(**kwargs) -> PipelineConfig:
    config = PipelineConfig(**kwargs)
    config.asr.enabled = False
    config.reference.min_segment_sec = 1.0
    return config


class FakeTranscriber:
    """Stands in for faster-whisper."""

    def __init__(self, language="ru", probability=0.99):
        self.language = language
        self.probability = probability
        self.calls = []

    def transcribe(self, path, *, language=None):
        self.calls.append((str(path), language))
        return Transcript(
            language=self.language,
            language_probability=self.probability,
            model="fake",
            duration=6.0,
            segments=[
                TranscriptSegment(0.5, 2.0, "Первый сегмент.", avg_logprob=-0.2),
                TranscriptSegment(2.6, 6.0, "Второй сегмент подлиннее.", avg_logprob=-0.1),
            ],
        )


def test_build_profile_without_asr(reference_wav, tmp_path):
    profile = build_profile(
        reference_wav, out_dir=tmp_path / "anna", name="anna", config=no_asr_config(language="ru")
    )
    assert profile.name == "anna"
    assert profile.language == "ru"
    assert profile.reference_files
    assert profile.total_duration > 0.5
    assert (tmp_path / "anna" / "profile.json").exists()
    assert all(path.exists() for path in profile.reference_files)


def test_build_profile_uses_asr_language_and_text(reference_wav, tmp_path):
    config = PipelineConfig()
    config.reference.min_segment_sec = 1.0
    transcriber = FakeTranscriber(language="ru")
    profile = build_profile(
        reference_wav, out_dir=tmp_path / "p", config=config, transcriber=transcriber
    )
    assert transcriber.calls
    assert profile.language == "ru"
    assert "сегмент" in profile.reference_text
    assert profile.asr["model"] == "fake"
    assert (tmp_path / "p" / "transcript.json").exists()


def test_unsupported_detected_language_falls_back(reference_wav, tmp_path):
    config = PipelineConfig(fallback_language="ru")
    config.reference.min_segment_sec = 1.0
    profile = build_profile(
        reference_wav, out_dir=tmp_path / "p", config=config,
        transcriber=FakeTranscriber(language="xx"),
    )
    assert profile.language == "ru"


def test_asr_failure_does_not_break_the_build(reference_wav, tmp_path):
    class Broken:
        def transcribe(self, path, *, language=None):
            raise RuntimeError("no model")

    profile = build_profile(
        reference_wav, out_dir=tmp_path / "p",
        config=PipelineConfig(language="ru"), transcriber=Broken(),
    )
    assert profile.reference_files


def test_profile_roundtrip(reference_wav, tmp_path):
    original = build_profile(
        reference_wav, out_dir=tmp_path / "p", config=no_asr_config(language="ru")
    )
    loaded = SpeakerProfile.load(tmp_path / "p")
    assert loaded.name == original.name
    assert loaded.language == "ru"
    assert [p.name for p in loaded.reference_files] == [p.name for p in original.reference_files]
    assert load_reference_audio(loaded).size > 0


def test_existing_profile_is_not_clobbered(reference_wav, tmp_path):
    build_profile(reference_wav, out_dir=tmp_path / "p", config=no_asr_config(language="ru"))
    with pytest.raises(AudioError):
        build_profile(reference_wav, out_dir=tmp_path / "p", config=no_asr_config(language="ru"))
    build_profile(
        reference_wav, out_dir=tmp_path / "p", config=no_asr_config(language="ru"), overwrite=True
    )


def test_temporary_profile_is_cleaned_up(reference_wav):
    profile = build_profile(reference_wav, config=no_asr_config(language="ru"))
    directory = profile.directory
    assert profile.is_temporary and directory.exists()
    profile.cleanup()
    assert not directory.exists()


def test_silent_recording_is_rejected(tmp_path):
    import numpy as np

    from voice_clone_tts import audio as audio_utils

    path = tmp_path / "silence.wav"
    audio_utils.save_audio(path, np.zeros(24_000, dtype=np.float32), 24_000)
    with pytest.raises(AudioError):
        build_profile(path, out_dir=tmp_path / "p", config=no_asr_config(language="ru"))


def test_profile_format_guard(reference_wav, tmp_path):
    build_profile(reference_wav, out_dir=tmp_path / "p", config=no_asr_config(language="ru"))
    path = tmp_path / "p" / "profile.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["format"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AudioError):
        SpeakerProfile.load(tmp_path / "p")


def test_select_reference_segments_merges_and_ranks():
    segments = [
        SpeechSegment(0.0, 1.0, "a", score=-0.5),
        SpeechSegment(1.2, 2.0, "b", score=-0.5),
        SpeechSegment(3.0, 12.0, "c", score=-0.1),
    ]
    picked = select_reference_segments(segments, min_sec=2.0, max_sec=8.0, target_total_sec=20.0)
    assert picked
    assert all(segment.duration <= 8.0 for segment in picked)
    assert picked == sorted(picked, key=lambda s: s.start)
