import json

import pytest

from voice_clone_tts.cli import main


def run(argv) -> int:
    return main(argv)


def test_backends_listing_json(capsys):
    assert run(["backends", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in rows} >= {"dummy", "xtts", "silero"}


def test_languages_listing(capsys):
    assert run(["languages"]) == 0
    out = capsys.readouterr().out
    assert "ru" in out and "silero-stress" in out


def test_speak_with_dummy_backend(reference_wav, tmp_path, capsys):
    out = tmp_path / "speech.wav"
    code = run([
        "speak", "-v", str(reference_wav), "-t", "Привет, мир!", "-l", "ru",
        "-b", "dummy", "--no-asr", "--no-stress", "-o", str(out), "-q",
    ])
    assert code == 0
    assert out.exists() and out.stat().st_size > 1000
    assert "backend   : dummy" in capsys.readouterr().out


def test_speak_can_save_the_profile(reference_wav, tmp_path):
    profile_dir = tmp_path / "profiles" / "anna"
    code = run([
        "speak", "-v", str(reference_wav), "-t", "Тест.", "-l", "ru", "-b", "dummy",
        "--no-asr", "--no-stress", "-o", str(tmp_path / "a.wav"),
        "--save-profile", str(profile_dir), "-q",
    ])
    assert code == 0
    assert (profile_dir / "profile.json").exists()


def test_dry_run_prints_chunks_without_synthesis(tmp_path, capsys):
    code = run([
        "speak", "-t", "У меня 2 кота. И 3 собаки.", "-l", "ru", "-b", "dummy",
        "--no-stress", "--dry-run", "-q",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "два" in out and "три" in out
    assert not (tmp_path / "output.wav").exists()


def test_profile_build_and_show(reference_wav, tmp_path, capsys):
    directory = tmp_path / "prof"
    assert run(["profile", "build", str(reference_wav), "-o", str(directory),
                "-l", "ru", "--no-asr", "-q"]) == 0
    capsys.readouterr()
    assert run(["profile", "show", str(directory), "--json", "-q"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["language"] == "ru" and data["reference_files"]


def test_stress_command_without_the_model(capsys):
    assert run(["stress", "Привет, мир", "--no-stress", "-q"]) == 0
    assert "Привет" in capsys.readouterr().out


def test_missing_text_is_an_error(reference_wav, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert run(["speak", "-v", str(reference_wav), "-b", "dummy", "-q"]) == 1
    assert "no text to speak" in capsys.readouterr().err


def test_bad_backend_option_is_an_error(reference_wav, capsys):
    code = run(["speak", "-v", str(reference_wav), "-t", "x", "-b", "dummy",
                "--backend-option", "oops", "-q"])
    assert code == 1
    assert "KEY=VALUE" in capsys.readouterr().err


def test_unknown_language_is_an_error(capsys):
    assert run(["stress", "hi", "-l", "klingon", "-q"]) == 1
    assert "unsupported language" in capsys.readouterr().err


def test_converters_listing(capsys):
    assert run(["converters", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in rows} >= {"rvc", "sovits"}


def test_voice_train_dataset_only(reference_wav, tmp_path, capsys):
    code = run([
        "voice", "train", "-s", str(reference_wav), "-o", str(tmp_path / "voice"),
        "--speaker", "anna", "--dataset-only", "--max-clip-seconds", "2", "-q",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "clips" in out
    assert (tmp_path / "voice" / "dataset" / "dataset_raw" / "anna").exists()


def test_voice_show(tmp_path, capsys):
    from voice_clone_tts.vc import VoiceModel

    checkpoint = tmp_path / "m.pth"
    checkpoint.write_bytes(b"w")
    VoiceModel(name="anna", directory=tmp_path / "v", converter="rvc",
               checkpoint=checkpoint, median_f0=180.0).save()
    assert run(["voice", "show", str(tmp_path / "v"), "--json", "-q"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["name"] == "anna" and data["converter"] == "rvc"


def test_speak_with_a_voice_model(reference_wav, tmp_path, capsys):
    from voice_clone_tts.vc import VoiceConverter, VoiceModel, register_converter

    class CliFakeConverter(VoiceConverter):
        name = "fake-cli-vc"

        @classmethod
        def is_available(cls):
            return True

        def train(self, dataset, out_dir, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def convert(self, audio, sample_rate, model, *, transpose=0):
            return audio * 0.8

    register_converter(CliFakeConverter)
    checkpoint = tmp_path / "m.pth"
    checkpoint.write_bytes(b"w")
    VoiceModel(name="anna", directory=tmp_path / "v", converter="fake-cli-vc",
               checkpoint=checkpoint).save()

    out = tmp_path / "cloned.wav"
    code = run([
        "speak", "-v", str(reference_wav), "-t", "Привет.", "-l", "ru", "-b", "dummy",
        "--no-asr", "--no-stress", "--voice-model", str(tmp_path / "v"),
        "--transpose", "2", "-o", str(out), "-q",
    ])
    assert code == 0 and out.exists()
    assert "voice     : anna (transpose +2)" in capsys.readouterr().out


def test_bad_transpose_is_an_error(reference_wav, capsys):
    code = run(["speak", "-v", str(reference_wav), "-t", "x", "-b", "dummy",
                "--transpose", "sideways", "-q"])
    assert code == 1
    assert "--transpose" in capsys.readouterr().err
