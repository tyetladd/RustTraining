import numpy as np
import pytest

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.errors import AudioError


def test_save_load_roundtrip(tmp_path, speech_audio, sample_rate):
    path = tmp_path / "a.wav"
    audio_utils.save_audio(path, speech_audio, sample_rate)
    loaded, sr = audio_utils.load_audio(path)
    assert sr == sample_rate
    assert loaded.dtype == np.float32
    assert loaded.shape == speech_audio.shape
    assert np.allclose(loaded, speech_audio, atol=1e-4)


def test_load_resamples_on_request(reference_wav, sample_rate):
    loaded, sr = audio_utils.load_audio(reference_wav, 16_000)
    assert sr == 16_000
    assert abs(loaded.size / 16_000 - 6.0) < 0.05


def test_missing_file_raises():
    with pytest.raises(AudioError):
        audio_utils.load_audio("/definitely/not/here.mp3")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(AudioError):
        audio_utils.load_audio(path)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    assert np.allclose(audio_utils.to_mono(stereo), [0.5, 0.5])


def test_resample_changes_length(speech_audio, sample_rate):
    out = audio_utils.resample(speech_audio, sample_rate, 12_000)
    assert abs(out.size - speech_audio.size // 2) < 50


def test_normalize_loudness_hits_the_target(speech_audio):
    out = audio_utils.normalize_loudness(speech_audio, -20.0)
    assert abs(audio_utils.dbfs(out) + 20.0) < 0.5
    assert audio_utils.peak_dbfs(out) <= -1.0 + 1e-3


def test_trim_silence_removes_the_head(speech_audio, sample_rate):
    trimmed = audio_utils.trim_silence(speech_audio, sample_rate)
    assert trimmed.size < speech_audio.size
    assert trimmed.size > sample_rate  # still more than a second of speech


def test_detect_speech_segments_finds_the_gap(speech_audio, sample_rate):
    segments = audio_utils.detect_speech_segments(speech_audio, sample_rate)
    assert len(segments) >= 2
    assert all(segment.duration > 0 for segment in segments)
    assert segments[0].start >= 0.0


def test_analyze_reports_plausible_stats(speech_audio, sample_rate):
    stats = audio_utils.analyze(speech_audio, sample_rate)
    assert abs(stats.duration - 6.0) < 0.05
    assert stats.estimated_snr_db > 10.0
    assert stats.clipping_ratio == 0.0


def test_stats_warn_about_short_references(sample_rate):
    from tests.conftest import make_speech_like

    stats = audio_utils.analyze(make_speech_like(1.5, gaps=()), sample_rate)
    assert any("only" in warning for warning in stats.warnings())


def test_concat_with_pause(sample_rate):
    a = np.ones(100, dtype=np.float32)
    joined = audio_utils.concat_with_pause([a, a], sample_rate, pause_sec=0.1)
    assert joined.size == 200 + int(0.1 * sample_rate)


def test_apply_fade_starts_at_zero(speech_audio, sample_rate):
    faded = audio_utils.apply_fade(speech_audio, sample_rate, 10.0)
    assert faded[0] == 0.0 and abs(faded[-1]) < 1e-6


def test_slice_segment(speech_audio, sample_rate):
    clip = audio_utils.slice_segment(speech_audio, sample_rate, 1.0, 2.0)
    assert clip.size == sample_rate
