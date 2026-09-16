"""Grading a candidate recording."""

import json

import numpy as np
import pytest

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.cli import main
from voice_clone_tts.quality import FAIL, OK, WARN, assess

from tests.conftest import make_speech_like


def write(tmp_path, samples, sample_rate=44_100, name="take.wav"):
    path = tmp_path / name
    audio_utils.save_audio(path, samples, sample_rate)
    return path


def long_take(minutes, sample_rate=44_100, f0=120.0, noise=0.0002):
    """A realistic long recording: speech with real pauses, tiled to length."""
    block = make_speech_like(
        12.0, sample_rate=sample_rate, f0=f0,
        gaps=((2.0, 2.6), (5.0, 5.7), (8.5, 9.2)), noise=noise,
    )
    return np.tile(block, int(np.ceil(minutes * 60 / 12)))


def test_short_recording_fails_for_vc_but_passes_for_xtts(tmp_path):
    path = write(tmp_path, long_take(1.5))

    vc = assess(path, purpose="vc")
    assert vc.verdict == FAIL
    assert any("нужно хотя бы" in f.message for f in vc.findings)

    xtts = assess(path, purpose="xtts")
    assert xtts.verdict in {OK, WARN}
    assert all("нужно хотя бы" not in f.message for f in xtts.findings)


def test_clean_long_recording_passes(tmp_path):
    report = assess(write(tmp_path, long_take(20.0, noise=0.00005)), purpose="vc")
    assert report.verdict == OK, [f.message for f in report.findings]
    assert report.speech_duration > 15 * 60
    assert report.snr_db > 30


def test_noise_is_reported(tmp_path):
    report = assess(write(tmp_path, long_take(2.0, noise=0.08)), purpose="vc")
    assert any("SNR" in f.message and f.level == FAIL for f in report.findings)


def test_clipping_is_reported(tmp_path):
    report = assess(write(tmp_path, np.clip(long_take(2.0) * 6.0, -1.0, 1.0)), purpose="vc")
    assert any("клиппинг" in f.message and f.level == FAIL for f in report.findings)


def test_telephone_sample_rate_fails(tmp_path):
    samples = long_take(2.0, sample_rate=16_000)
    report = assess(write(tmp_path, samples, sample_rate=16_000), purpose="vc")
    assert report.verdict == FAIL
    assert any("дискретизации" in f.message and f.level == FAIL for f in report.findings)


def test_register_and_voice_suggestions(tmp_path):
    low = long_take(1.0, f0=110.0)
    high = long_take(1.0, f0=210.0)

    male = assess(write(tmp_path, low, name="m.wav"))
    female = assess(write(tmp_path, high, name="f.wav"))

    assert male.register == "male" and "aidar" in male.suggested_voices()
    assert female.register == "female" and "xenia" in female.suggested_voices()


def test_report_serializes(tmp_path):
    report = assess(write(tmp_path, make_speech_like(90.0, sample_rate=44_100)), purpose="xtts")
    data = report.to_dict()
    assert data["verdict"] in {OK, WARN, FAIL}
    assert data["findings"] and "register" in data
    assert json.dumps(data)


def test_cli_check(tmp_path, capsys):
    path = write(tmp_path, make_speech_like(90.0, sample_rate=44_100))
    assert main(["check", str(path), "--purpose", "xtts", "-q"]) == 0
    out = capsys.readouterr().out
    assert "длительность" in out and "SNR" in out

    assert main(["check", str(path), "--json", "-q"]) == 0
    assert json.loads(capsys.readouterr().out)["purpose"] == "vc"


def test_cli_check_strict_exit_code(tmp_path, capsys):
    path = write(tmp_path, make_speech_like(20.0, sample_rate=44_100))
    assert main(["check", str(path), "-q"]) == 0
    capsys.readouterr()
    assert main(["check", str(path), "--strict", "-q"]) == 1
