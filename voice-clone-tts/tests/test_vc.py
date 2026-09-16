"""Voice conversion layer: dataset, model metadata, drivers, pipeline wiring."""

import json
from pathlib import Path

import numpy as np
import pytest

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.config import PipelineConfig
from voice_clone_tts.errors import AudioError
from voice_clone_tts.vc import (
    VoiceConversionError,
    VoiceConverter,
    VoiceModel,
    build_training_dataset,
    converter_class,
    get_converter,
    iter_converter_info,
    register_converter,
)
from voice_clone_tts.vc.runner import CommandResult, ToolchainError, run_command
from voice_clone_tts.vc.rvc import RVCConverter, find_applio
from voice_clone_tts.vc.sovits import SoVitsConverter


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def test_dataset_is_cut_into_clips(reference_wav, tmp_path):
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna",
        max_clip_sec=1.0, min_clip_sec=0.4,
    )
    assert dataset.speaker == "anna"
    assert dataset.speaker_dir == tmp_path / "ds" / "dataset_raw" / "anna"
    assert len(dataset.clips) >= 3
    assert all(path.exists() for path in dataset.clips)
    for path in dataset.clips:
        samples, sr = audio_utils.load_audio(path)
        assert samples.size <= int(1.05 * sr)
    assert dataset.median_f0 > 0


def test_dataset_warns_about_short_recordings(reference_wav, tmp_path):
    dataset = build_training_dataset(reference_wav, out_dir=tmp_path / "ds", speaker="anna")
    assert any("min of speech" in warning for warning in dataset.warnings())
    assert "warning" in dataset.describe()


def test_dataset_refuses_to_clobber(reference_wav, tmp_path):
    build_training_dataset(reference_wav, out_dir=tmp_path / "ds", speaker="anna")
    with pytest.raises(AudioError, match="overwrite"):
        build_training_dataset(reference_wav, out_dir=tmp_path / "ds", speaker="anna")
    build_training_dataset(reference_wav, out_dir=tmp_path / "ds", speaker="anna", overwrite=True)


def test_dataset_falls_back_to_profile_clips(reference_wav, tmp_path):
    from voice_clone_tts.profile import build_profile

    config = PipelineConfig(language="ru")
    config.asr.enabled = False
    config.reference.min_segment_sec = 1.0
    profile = build_profile(reference_wav, out_dir=tmp_path / "p", config=config)
    profile.source_audio = "/gone/missing.mp3"

    dataset = build_training_dataset(
        profile=profile, out_dir=tmp_path / "ds", max_clip_sec=2.0, min_clip_sec=0.4
    )
    assert dataset.clips
    assert dataset.source is None


def test_dataset_needs_some_input(tmp_path):
    with pytest.raises(AudioError):
        build_training_dataset(out_dir=tmp_path / "ds")


# --------------------------------------------------------------------------
# model metadata
# --------------------------------------------------------------------------

def test_voice_model_roundtrip(tmp_path):
    checkpoint = tmp_path / "G_1000.pth"
    checkpoint.write_bytes(b"weights")
    model = VoiceModel(
        name="anna", directory=tmp_path, converter="sovits", checkpoint=checkpoint,
        median_f0=198.0, train_stats={"epochs": 300},
    )
    model.save()
    loaded = VoiceModel.load(tmp_path)
    assert loaded.name == "anna"
    assert loaded.converter == "sovits"
    assert loaded.checkpoint == checkpoint
    assert loaded.median_f0 == 198.0
    assert loaded.train_stats["epochs"] == 300
    assert "anna" in loaded.describe()


def test_voice_model_reports_a_missing_checkpoint(tmp_path):
    checkpoint = tmp_path / "G_1.pth"
    checkpoint.write_bytes(b"x")
    VoiceModel(name="a", directory=tmp_path, converter="rvc", checkpoint=checkpoint).save()
    checkpoint.unlink()
    with pytest.raises(VoiceConversionError, match="missing checkpoint"):
        VoiceModel.load(tmp_path)


def test_voice_model_format_guard(tmp_path):
    VoiceModel(name="a", directory=tmp_path, converter="rvc").save()
    path = tmp_path / "voice_model.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["format"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(VoiceConversionError, match="newer version"):
        VoiceModel.load(tmp_path)


def test_missing_voice_model_directory(tmp_path):
    with pytest.raises(VoiceConversionError, match="voice train"):
        VoiceModel.load(tmp_path / "nope")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_builtin_converters_are_registered():
    assert {row["name"] for row in iter_converter_info()} >= {"rvc", "sovits"}


def test_unknown_converter_raises():
    with pytest.raises(VoiceConversionError):
        get_converter("nope")


def test_converter_needs_a_name():
    with pytest.raises(VoiceConversionError):
        register_converter(type("Anon", (VoiceConverter,), {"name": "base"}))


# --------------------------------------------------------------------------
# so-vits-svc driver
# --------------------------------------------------------------------------

class FakeSvc:
    """Records `svc` invocations and fakes the artifacts each step produces."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, args, *, cwd=None, env=None, timeout=None, label="", check=True):
        args = [str(arg) for arg in args]
        self.commands.append(args)
        step = args[1]
        if step == "pre-config":
            config_path = Path(args[args.index("-c") + 1])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"train": {"epochs": 10000, "batch_size": 4}}),
                                   encoding="utf-8")
        elif step == "train":
            model_dir = Path(args[args.index("-m") + 1])
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "G_800.pth").write_bytes(b"g")
            (model_dir / "G_1600.pth").write_bytes(b"g")
        elif step == "infer":
            output = Path(args[args.index("-o") + 1])
            audio_utils.save_audio(output, np.full(4410, 0.05, dtype=np.float32), 44_100)
        return CommandResult(args=args, returncode=0)


@pytest.fixture
def fake_svc(monkeypatch, tmp_path):
    runner = FakeSvc()
    monkeypatch.setattr("voice_clone_tts.vc.sovits.run_command", runner)
    monkeypatch.setattr(SoVitsConverter, "executable", classmethod(lambda cls: "/usr/bin/svc"))
    return runner


def test_sovits_training_runs_the_documented_pipeline(fake_svc, reference_wav, tmp_path):
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )
    converter = SoVitsConverter()
    model = converter.train(dataset, tmp_path / "voice", epochs=42)

    steps = [command[1] for command in fake_svc.commands]
    assert steps == ["pre-resample", "pre-config", "pre-hubert", "train"]
    assert model.converter == "sovits"
    assert model.checkpoint.name == "G_1600.pth"  # the newest checkpoint wins
    assert model.train_stats["epochs"] == 42
    assert model.median_f0 > 0
    config = json.loads(model.config.read_text(encoding="utf-8"))
    assert config["train"]["epochs"] == 42  # the config was patched, not left at 10000
    assert VoiceModel.load(tmp_path / "voice").name == model.name


def test_sovits_training_fails_without_a_checkpoint(fake_svc, reference_wav, tmp_path, monkeypatch):
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )

    original = fake_svc.__call__

    def no_checkpoint(args, **kwargs):
        args = [str(a) for a in args]
        return original(args, **kwargs) if args[1] != "train" else CommandResult(args, 0)

    monkeypatch.setattr("voice_clone_tts.vc.sovits.run_command", no_checkpoint)
    with pytest.raises(VoiceConversionError, match="no G_.*checkpoint"):
        SoVitsConverter().train(dataset, tmp_path / "voice")


def test_sovits_conversion_passes_the_transpose(fake_svc, tmp_path):
    checkpoint = tmp_path / "logs" / "G_1.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"g")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    model = VoiceModel(name="anna", directory=tmp_path, converter="sovits",
                       checkpoint=checkpoint, config=config, speaker="anna")

    out = SoVitsConverter().convert(
        np.full(24_000, 0.1, dtype=np.float32), 24_000, model, transpose=-3
    )
    command = fake_svc.commands[-1]
    assert command[1] == "infer"
    assert command[command.index("-t") + 1] == "-3"
    assert command[command.index("-s") + 1] == "anna"
    assert out.dtype == np.float32 and out.size > 0


def test_sovits_reports_a_missing_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(SoVitsConverter, "executable", classmethod(lambda cls: None))
    model = VoiceModel(name="a", directory=tmp_path, converter="sovits")
    with pytest.raises(VoiceConversionError, match="pip install"):
        SoVitsConverter().convert(np.zeros(100, dtype=np.float32), 24_000, model)


# --------------------------------------------------------------------------
# RVC / Applio driver
# --------------------------------------------------------------------------

@pytest.fixture
def applio(tmp_path, monkeypatch):
    checkout = tmp_path / "Applio"
    checkout.mkdir()
    (checkout / "core.py").write_text("# fake", encoding="utf-8")
    monkeypatch.setenv("VCTTS_APPLIO_DIR", str(checkout))
    return checkout


class FakeApplio:
    def __init__(self, checkout: Path):
        self.checkout = checkout
        self.commands: list[list[str]] = []

    def __call__(self, args, *, cwd=None, env=None, timeout=None, label="", check=True):
        args = [str(arg) for arg in args]
        self.commands.append(args)
        step = args[2]
        if step == "train":
            name = args[args.index("--model-name") + 1]
            logs = self.checkout / "logs" / name
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{name}.pth").write_bytes(b"w")
        elif step == "index":
            name = args[args.index("--model-name") + 1]
            (self.checkout / "logs" / name / f"{name}.index").write_bytes(b"i")
        elif step == "infer":
            output = Path(args[args.index("--output-path") + 1])
            audio_utils.save_audio(output, np.full(4000, 0.05, dtype=np.float32), 40_000)
        return CommandResult(args=args, returncode=0)


def test_applio_is_discovered_from_the_environment(applio):
    assert find_applio() == applio
    assert RVCConverter.is_available()


def test_rvc_training_runs_the_documented_pipeline(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )
    model = RVCConverter().train(dataset, tmp_path / "voice", epochs=120)

    assert [command[2] for command in runner.commands] == ["preprocess", "extract", "train", "index"]
    assert model.converter == "rvc"
    assert model.checkpoint.parent == tmp_path / "voice"  # copied out of the checkout
    assert model.index is not None and model.index.exists()
    assert model.train_stats["epochs"] == 120


def test_rvc_conversion_passes_pitch_and_index(applio, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    checkpoint = tmp_path / "anna.pth"
    checkpoint.write_bytes(b"w")
    index = tmp_path / "anna.index"
    index.write_bytes(b"i")
    model = VoiceModel(name="anna", directory=tmp_path, converter="rvc",
                       checkpoint=checkpoint, index=index, sample_rate=40_000)

    out = RVCConverter().convert(np.full(24_000, 0.1, dtype=np.float32), 24_000, model, transpose=5)
    command = runner.commands[-1]
    assert command[command.index("--pitch") + 1] == "5"
    assert command[command.index("--index-path") + 1] == str(index)
    assert out.size > 0


def test_rvc_rejects_an_absurd_transpose(applio, tmp_path):
    checkpoint = tmp_path / "a.pth"
    checkpoint.write_bytes(b"w")
    model = VoiceModel(name="a", directory=tmp_path, converter="rvc", checkpoint=checkpoint)
    with pytest.raises(VoiceConversionError, match="24 semitones"):
        RVCConverter().convert(np.zeros(100, dtype=np.float32), 24_000, model, transpose=99)


def test_rvc_reports_a_missing_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("VCTTS_APPLIO_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    converter = RVCConverter()
    with pytest.raises(VoiceConversionError, match="VCTTS_APPLIO_DIR"):
        converter.train(None, tmp_path)


def test_rvc_rejects_unsupported_sample_rates():
    with pytest.raises(VoiceConversionError, match="RVC supports"):
        get_converter("rvc", sample_rate=22_050)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def test_run_command_raises_with_the_tail_of_the_output():
    with pytest.raises(ToolchainError) as excinfo:
        run_command(["python3", "-c", "import sys; print('boom'); sys.exit(3)"], label="test")
    assert "code 3" in str(excinfo.value) and "boom" in str(excinfo.value)


def test_run_command_reports_a_missing_binary():
    with pytest.raises(ToolchainError, match="command not found"):
        run_command(["definitely-not-a-real-binary-xyz"])
