"""Voice conversion layer: dataset, model metadata, drivers, pipeline wiring."""

import json
import os
import time
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

    def __call__(self, args, *, cwd=None, env=None, timeout=None, label="", check=True, **kwargs):
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
        elif step == "prerequisites":
            if not self.skip_prerequisites:
                for relative in ("rvc/models/predictors/rmvpe.pt",
                                 "rvc/models/embedders/contentvec/pytorch_model.bin",
                                 "rvc/models/pretraineds/hifi-gan/f0G40k.pth",
                                 "rvc/models/pretraineds/hifi-gan/f0D40k.pth"):
                    target = self.checkout / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"weights")
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
    """Stands in for an Applio checkout, writing what each step really leaves behind.

    `slices` and `entries` let a test starve a stage the way a broken
    preprocess or extract would.
    """

    def __init__(self, checkout: Path, *, slices: int = 40, entries: int = 40,
                 skip_dirs: tuple[str, ...] = (), skip_prerequisites: bool = False):
        self.skip_prerequisites = skip_prerequisites
        self.checkout = checkout
        self.commands: list[list[str]] = []
        self.slices = slices
        self.entries = entries
        self.skip_dirs = skip_dirs

    def __call__(self, args, *, cwd=None, env=None, timeout=None, label="", check=True, **kwargs):
        args = [str(arg) for arg in args]
        self.commands.append(args)
        step = args[2]
        name = args[args.index("--model-name") + 1] if "--model-name" in args else "anna"
        logs = self.checkout / "logs" / name
        if step == "preprocess":
            sliced = logs / "sliced_audios"
            sliced.mkdir(parents=True, exist_ok=True)
            for index in range(self.slices):
                (sliced / f"0_{index}.wav").write_bytes(b"wav")
        elif step == "extract":
            for directory, suffix in (("extracted", ".npy"), ("f0", ".wav.npy"),
                                      ("f0_voiced", ".wav.npy")):
                if directory in self.skip_dirs:
                    (logs / directory).mkdir(parents=True, exist_ok=True)
                    continue
                target = logs / directory
                target.mkdir(parents=True, exist_ok=True)
                for index in range(self.entries):
                    (target / f"0_{index}{suffix}").write_bytes(b"data")
            written = 0 if self.skip_dirs or not self.entries else self.entries
            (logs / "filelist.txt").write_text(
                "\n".join(f"line {i}" for i in range(written)), encoding="utf-8"
            )
        elif step == "train":
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{name}.pth").write_bytes(b"w")
        elif step == "index":
            (logs / f"{name}.index").write_bytes(b"i")
        elif step == "prerequisites":
            if not self.skip_prerequisites:
                for relative in ("rvc/models/predictors/rmvpe.pt",
                                 "rvc/models/embedders/contentvec/pytorch_model.bin",
                                 "rvc/models/pretraineds/hifi-gan/f0G40k.pth",
                                 "rvc/models/pretraineds/hifi-gan/f0D40k.pth"):
                    target = self.checkout / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"weights")
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

    assert steps(runner) == ["preprocess", "extract", "train", "index"]
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


# --------------------------------------------------------------------------
# cloud training: checkpoints must survive an ephemeral checkout
# --------------------------------------------------------------------------

def test_rvc_links_logs_to_persistent_storage(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    persistent = tmp_path / "drive" / "logs"
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )

    model = RVCConverter(logs_dir=str(persistent)).train(dataset, tmp_path / "voice", epochs=1)

    link = applio / "logs" / "anna"
    assert link.is_symlink()
    assert link.resolve() == (persistent / "anna").resolve()
    assert (persistent / "anna" / "anna.pth").exists()  # written through the link
    assert model.checkpoint.exists()


def test_rvc_moves_existing_logs_into_persistent_storage(applio, tmp_path):
    stale = applio / "logs" / "anna"
    stale.mkdir(parents=True)
    (stale / "G_100.pth").write_bytes(b"old")
    persistent = tmp_path / "drive" / "logs"

    linked = RVCConverter(logs_dir=str(persistent))._prepare_logs("anna")
    assert linked.is_symlink()
    assert (persistent / "anna" / "G_100.pth").exists()


def test_rvc_refuses_to_merge_two_populated_log_dirs(applio, tmp_path):
    stale = applio / "logs" / "anna"
    stale.mkdir(parents=True)
    (stale / "G_100.pth").write_bytes(b"old")
    persistent = tmp_path / "drive" / "logs" / "anna"
    persistent.mkdir(parents=True)
    (persistent / "G_200.pth").write_bytes(b"new")

    with pytest.raises(VoiceConversionError, match="hold training data"):
        RVCConverter(logs_dir=str(tmp_path / "drive" / "logs"))._prepare_logs("anna")


def test_dataset_is_reused_when_resuming(reference_wav, tmp_path):
    first = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )
    stamps = {clip: clip.stat().st_mtime_ns for clip in first.clips}

    again = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", reuse_existing=True
    )
    assert [c.name for c in again.clips] == [c.name for c in first.clips]
    assert {clip: clip.stat().st_mtime_ns for clip in again.clips} == stamps  # not rewritten
    assert abs(again.total_duration - first.total_duration) < 0.1
    assert again.median_f0 > 0


def test_reuse_falls_through_to_a_normal_build(reference_wav, tmp_path):
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna",
        max_clip_sec=2.0, min_clip_sec=0.4, reuse_existing=True,
    )
    assert dataset.clips  # nothing on disk yet, so it is built from the recording


# --------------------------------------------------------------------------
# time budget: a cloud session ends on a hard limit
# --------------------------------------------------------------------------

def test_run_command_streams_and_returns_output():
    result = run_command(
        ["python3", "-c", "print('hello'); print('world')"], label="test", timeout=30
    )
    assert result.ok and result.tail == ["hello", "world"]


def test_run_command_stops_a_chatty_process_at_the_deadline():
    """The old implementation blocked in the read loop and never timed out."""
    script = "import time\nwhile True:\n    print('working', flush=True)\n    time.sleep(0.05)\n"
    started = time.monotonic()
    with pytest.raises(ToolchainError, match="time budget"):
        run_command(["python3", "-c", script], label="chatty", timeout=1.0)
    assert time.monotonic() - started < 15.0
    
    
def test_run_command_stops_a_silent_process_at_the_deadline():
    started = time.monotonic()
    with pytest.raises(ToolchainError, match="time budget"):
        run_command(["python3", "-c", "import time; time.sleep(60)"], label="quiet", timeout=1.0)
    assert time.monotonic() - started < 15.0


def test_timeout_kills_the_whole_process_tree(tmp_path):
    """Applio spawns its trainer as a child; orphaning it would keep the GPU busy."""
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "while True:\n    print('parent alive', flush=True)\n    time.sleep(0.1)\n"
    )
    with pytest.raises(ToolchainError, match="time budget"):
        run_command(["python3", "-c", script], label="tree", timeout=2.0)

    grandchild = int(pid_file.read_text())
    for _ in range(40):  # give the signal a moment to land
        try:
            os.kill(grandchild, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.25)
    else:
        os.kill(grandchild, 9)
        pytest.fail(f"grandchild {grandchild} survived the timeout")


def test_rvc_uses_every_cpu_core_by_default(monkeypatch):
    """Preprocessing on a cloud box has no one to share cores with."""
    monkeypatch.setattr("voice_clone_tts.vc.rvc.os.cpu_count", lambda: 4)
    assert RVCConverter().cpu_cores == 4
    assert RVCConverter(cpu_cores=2).cpu_cores == 2


# --------------------------------------------------------------------------
# Applio reports failure in its output while still exiting 0
# --------------------------------------------------------------------------

def test_run_command_catches_a_failure_reported_with_exit_code_zero():
    """core.py swallows its child's exit code and just prints the failure."""
    script = (
        "print('Starting training...')\n"
        "print('Training failed for model andrei. Please check the console logs.')\n"
    )
    with pytest.raises(ToolchainError, match="exiting 0"):
        run_command(
            ["python3", "-c", script],
            label="applio train",
            failure_patterns=("failed for model",),
        )


def test_run_command_ignores_failure_patterns_in_ordinary_output():
    result = run_command(
        ["python3", "-c", "print('epoch 1: loss 3.2')"],
        label="applio train",
        failure_patterns=("failed for model",),
    )
    assert result.ok


def test_run_command_keeps_only_the_last_state_of_a_progress_line():
    script = "import sys; sys.stdout.write('10%\\r50%\\r100% done\\n')"
    result = run_command(["python3", "-c", script], label="progress")
    assert result.tail == ["100% done"]


def test_rvc_stops_when_training_produced_no_weights(applio, reference_wav, tmp_path, monkeypatch):
    """The index step must not run and hide a training failure."""

    class SilentlyFailing(FakeApplio):
        """train exits 0 without writing weights — Applio's actual behaviour."""

        def __call__(self, args, **kwargs):
            args = [str(a) for a in args]
            if args[2] == "train":
                self.commands.append(args)
                return CommandResult(args=args, returncode=0)
            return super().__call__(args, **kwargs)

    runner = SilentlyFailing(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    dataset = build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )

    with pytest.raises(VoiceConversionError) as excinfo:
        RVCConverter().train(dataset, tmp_path / "voice", epochs=10)

    message = str(excinfo.value)
    assert "no .pth weights" in message
    assert "core.py train --model-name anna" in message  # как воспроизвести
    assert steps(runner) == ["preprocess", "extract", "train"]  # index не запускался


def test_rvc_passes_failure_patterns_to_the_runner(applio, reference_wav, tmp_path, monkeypatch):
    seen = []

    def spy(args, **kwargs):
        seen.append(kwargs.get("failure_patterns"))
        return FakeApplio(applio)(args, **kwargs)

    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", spy)
    converter = RVCConverter()
    converter._core("preprocess", "--model-name", "anna", label="applio preprocess")
    assert seen and "failed for model" in seen[0]


# --------------------------------------------------------------------------
# where the data disappears between Applio's steps
# --------------------------------------------------------------------------

def steps(runner) -> list[str]:
    """Стадии конвейера без служебного шага скачивания весов."""
    return [command[2] for command in runner.commands if command[2] != "prerequisites"]


def _dataset(reference_wav, tmp_path):
    return build_training_dataset(
        reference_wav, out_dir=tmp_path / "ds", speaker="anna", max_clip_sec=2.0, min_clip_sec=0.4
    )


def test_rvc_reports_when_preprocess_sliced_nothing(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio, slices=0)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    with pytest.raises(VoiceConversionError, match="не нарезал ни одного фрагмента"):
        RVCConverter().train(_dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10)
    assert steps(runner) == ["preprocess"]  # extract даже не запускался


def test_rvc_reports_when_extract_produced_nothing(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio, entries=0)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    with pytest.raises(VoiceConversionError, match="не подготовил данные"):
        RVCConverter().train(_dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10)
    assert steps(runner) == ["preprocess", "extract"]


def test_rvc_explains_too_little_data_for_the_batch_size(applio, reference_wav, tmp_path, monkeypatch):
    """train.py stops below three batches; say so before hours are spent."""
    runner = FakeApplio(applio, slices=12, entries=12)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    with pytest.raises(VoiceConversionError) as excinfo:
        RVCConverter(batch_size=8).train(
            _dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10
        )
    message = str(excinfo.value)
    assert "12 фрагментов" in message and "минимум 24" in message
    assert "batch_size=4" in message  # предложенное значение


def test_rvc_trains_when_there_is_enough_data(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    model = RVCConverter(batch_size=8).train(
        _dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10
    )
    assert steps(runner) == ["preprocess", "extract", "train", "index"]
    assert model.checkpoint.exists()


def test_applio_failure_patterns_cover_the_silent_stops():
    from voice_clone_tts.vc.rvc import APPLIO_FAILURE_PATTERNS

    assert any("not enough data" in p.lower() for p in APPLIO_FAILURE_PATTERNS)
    assert any("no audio files found" in p.lower() for p in APPLIO_FAILURE_PATTERNS)


# --------------------------------------------------------------------------
# stopping before the GPU-only stage (for machines without CUDA)
# --------------------------------------------------------------------------

def test_rvc_can_stop_after_preprocess(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    model = RVCConverter().train(
        _dataset(reference_wav, tmp_path), tmp_path / "voice", stop_after="preprocess"
    )
    assert steps(runner) == ["preprocess"]
    assert model.checkpoint is None
    assert model.train_stats == {"stopped_after": "preprocess", "slices": 40}
    assert (tmp_path / "voice" / "voice_model.json").exists()


def test_rvc_can_stop_after_extract(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    model = RVCConverter().train(
        _dataset(reference_wav, tmp_path), tmp_path / "voice", stop_after="extract"
    )
    assert steps(runner) == ["preprocess", "extract"]
    assert model.train_stats["stopped_after"] == "extract"
    assert model.train_stats["filelist_entries"] == 40


def test_rvc_rejects_an_unknown_stage(applio, reference_wav, tmp_path):
    with pytest.raises(VoiceConversionError, match="unknown stage"):
        RVCConverter().train(None, tmp_path / "voice", stop_after="nonsense")


def test_partial_model_loads_back(applio, reference_wav, tmp_path, monkeypatch):
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", FakeApplio(applio))
    RVCConverter().train(
        _dataset(reference_wav, tmp_path), tmp_path / "voice", stop_after="extract"
    )
    loaded = VoiceModel.load(tmp_path / "voice")
    assert loaded.checkpoint is None and loaded.train_stats["stopped_after"] == "extract"


def test_rvc_names_the_empty_stage_directory(applio, reference_wav, tmp_path, monkeypatch):
    """The real failure: 268 slices, 268 features, but f0 empty -> empty filelist."""
    runner = FakeApplio(applio, skip_dirs=("f0", "f0_voiced"))
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    with pytest.raises(VoiceConversionError) as excinfo:
        RVCConverter().train(_dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10)

    message = str(excinfo.value)
    assert "пусто в f0, f0_voiced" in message
    assert "sliced_audios: 40" in message and "extracted: 40" in message
    assert "rmvpe" in message
    assert steps(runner) == ["preprocess", "extract"]  # train не запускался


def test_resume_still_validates_the_prepared_data(applio, reference_wav, tmp_path, monkeypatch):
    """--resume skips preparation, so the checks matter there most of all."""
    runner = FakeApplio(applio, skip_dirs=("f0", "f0_voiced"))
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    dataset = _dataset(reference_wav, tmp_path)

    # Оставляем состояние прошлого прогона: нарезка есть, f0 нет.
    RVCConverter().train(dataset, tmp_path / "voice", stop_after="preprocess")
    (applio / "logs" / "anna" / "filelist.txt").write_text("", encoding="utf-8")
    runner.commands.clear()

    with pytest.raises(VoiceConversionError, match="f0_voiced") as excinfo:
        RVCConverter().train(dataset, tmp_path / "voice", epochs=10, resume=True)
    assert "sliced_audios: 40" in str(excinfo.value)  # видно, что уцелело
    assert steps(runner) == []  # ни одной стадии: остановились на проверке


# --------------------------------------------------------------------------
# Applio ships no weights: `git clone` alone leaves it unable to train
# --------------------------------------------------------------------------

def test_rvc_downloads_missing_weights_first(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    RVCConverter().train(_dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10)

    assert runner.commands[0][2] == "prerequisites"  # до всего остального
    assert (applio / "rvc/models/predictors/rmvpe.pt").exists()


def test_rvc_skips_the_download_when_weights_are_there(applio, reference_wav, tmp_path, monkeypatch):
    for relative in ("rvc/models/predictors/rmvpe.pt",
                     "rvc/models/embedders/contentvec/pytorch_model.bin",
                     "rvc/models/pretraineds/hifi-gan/f0G40k.pth"):
        target = applio / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")

    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    RVCConverter().train(_dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10)
    assert "prerequisites" not in [command[2] for command in runner.commands]


def test_rvc_reports_a_download_that_did_not_help(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio, skip_prerequisites=True)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)

    with pytest.raises(VoiceConversionError, match="весов всё ещё нет"):
        RVCConverter().train(_dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10)


def test_missing_weights_are_named_precisely(applio):
    missing = RVCConverter(sample_rate=40_000)._missing_prerequisites()
    assert any("предикторы f0" in item for item in missing)
    assert any("contentvec" in item for item in missing)
    assert any("f0G40k.pth" in item for item in missing)


def test_prerequisites_can_be_disabled(applio, reference_wav, tmp_path, monkeypatch):
    runner = FakeApplio(applio)
    monkeypatch.setattr("voice_clone_tts.vc.rvc.run_command", runner)
    RVCConverter(prerequisites=False).train(
        _dataset(reference_wav, tmp_path), tmp_path / "voice", epochs=10
    )
    assert "prerequisites" not in [command[2] for command in runner.commands]
