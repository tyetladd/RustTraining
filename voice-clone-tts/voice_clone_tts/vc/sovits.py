"""so-vits-svc-fork driver.

``pip install -U so-vits-svc-fork`` gives a single ``svc`` console script that
covers the whole training pipeline, which makes it the easiest voice-conversion
toolchain to automate::

    svc pre-resample -i dataset_raw -o dataset/44k
    svc pre-config   -i dataset/44k -f filelists/44k -c configs/44k/config.json
    svc pre-hubert   -i dataset/44k -c configs/44k/config.json
    svc train        -c configs/44k/config.json -m logs/44k
    svc infer input.wav -o out.wav -m logs/44k -c configs/44k/config.json

Upstream is explicitly no longer maintained (the authors point at the RVC
family instead), but the released version installs and trains as documented.
Everything runs inside one working directory, so a trained voice is a single
self-contained folder.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.vc.base import (
    VoiceConversionError,
    VoiceConverter,
    VoiceModel,
    register_converter,
)
from voice_clone_tts.vc.dataset import TrainingDataset
from voice_clone_tts.vc.runner import find_executable, run_command

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_EPOCHS = 300


@register_converter
class SoVitsConverter(VoiceConverter):
    """Trains and applies a so-vits-svc-fork model."""

    name = "sovits"
    display_name = "so-vits-svc-fork"
    description = "SVC voice conversion with a pip-installable CLI (upstream unmaintained)"
    install_hint = "pip install -U so-vits-svc-fork"
    supports_training = True
    needs_gpu = True

    def __init__(self, *, device: str = "auto", **options) -> None:
        super().__init__(device=device, **options)
        self.sample_rate = int(options.get("sample_rate", DEFAULT_SAMPLE_RATE))
        self.config_type = options.get("config_type", "so-vits-svc-4.0v1")
        self.f0_method = options.get("f0_method", "dio")
        self.batch_size = options.get("batch_size")
        self.n_jobs = int(options.get("n_jobs", -1))
        self.auto_predict_f0 = bool(options.get("auto_predict_f0", True))
        self.db_thresh = int(options.get("db_thresh", -25))
        self.timeout = options.get("timeout")

    @classmethod
    def executable(cls) -> str | None:
        return find_executable("svc")

    @classmethod
    def is_available(cls) -> bool:
        return cls.executable() is not None

    def _svc(self) -> str:
        binary = self.executable()
        if binary is None:
            raise VoiceConversionError(
                "the 'svc' command was not found.\nInstall it with:  " + self.install_hint
            )
        return binary

    # -- training ---------------------------------------------------------
    def _patch_config(self, config_path: Path, epochs: int) -> dict:
        """Set epoch count and batch size in the generated config."""
        config = json.loads(config_path.read_text(encoding="utf-8"))
        train = config.setdefault("train", {})
        train["epochs"] = epochs
        if self.batch_size:
            train["batch_size"] = int(self.batch_size)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return train

    @staticmethod
    def _latest_checkpoint(model_dir: Path) -> Path:
        checkpoints = sorted(
            model_dir.glob("G_*.pth"),
            key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
        )
        if not checkpoints:
            raise VoiceConversionError(
                f"training finished but no G_*.pth checkpoint appeared in {model_dir}"
            )
        return checkpoints[-1]

    def train(
        self,
        dataset: TrainingDataset,
        out_dir: str | Path,
        *,
        name: str | None = None,
        epochs: int | None = None,
        resume: bool = False,
    ) -> VoiceModel:
        binary = self._svc()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = name or dataset.speaker
        epochs = int(epochs or DEFAULT_EPOCHS)

        work = out_dir / "work"
        prepared = work / "dataset" / str(self.sample_rate)
        filelists = work / "filelists"
        config_path = work / "configs" / "config.json"
        model_dir = out_dir / "logs"
        for directory in (prepared.parent, filelists, config_path.parent, model_dir):
            directory.mkdir(parents=True, exist_ok=True)

        if not resume:
            run_command(
                [binary, "pre-resample", "-i", dataset.raw_dir, "-o", prepared,
                 "-s", self.sample_rate, "-n", self.n_jobs],
                label="svc pre-resample", timeout=self.timeout,
            )
            run_command(
                [binary, "pre-config", "-i", prepared, "-f", filelists,
                 "-c", config_path, "-t", self.config_type],
                label="svc pre-config", timeout=self.timeout,
            )
            train_config = self._patch_config(config_path, epochs)
            log.info("training config: %s", {k: train_config.get(k) for k in ("epochs", "batch_size")})
            run_command(
                [binary, "pre-hubert", "-i", prepared, "-c", config_path, "-fm", self.f0_method],
                label="svc pre-hubert", timeout=self.timeout,
            )
        else:
            if not config_path.exists():
                raise VoiceConversionError(
                    f"cannot resume: {config_path} is missing, run the training without --resume"
                )
            self._patch_config(config_path, epochs)

        log.info("training %s for %d epochs — this is the long part", name, epochs)
        run_command(
            [binary, "train", "-c", config_path, "-m", model_dir],
            label="svc train", timeout=self.timeout,
        )

        checkpoint = self._latest_checkpoint(model_dir)
        model = VoiceModel(
            name=name,
            directory=out_dir,
            converter=self.name,
            checkpoint=checkpoint,
            config=config_path,
            speaker=dataset.speaker,
            sample_rate=self.sample_rate,
            source_profile=dataset.source,
            median_f0=dataset.median_f0,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            train_stats={
                "epochs": epochs,
                "clips": len(dataset.clips),
                "dataset_minutes": round(dataset.total_duration / 60, 2),
                "f0_method": self.f0_method,
                "config_type": self.config_type,
            },
        )
        model.save()
        log.info("trained voice model saved to %s", out_dir)
        return model

    # -- inference --------------------------------------------------------
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: VoiceModel,
        *,
        transpose: int = 0,
    ) -> np.ndarray:
        binary = self._svc()
        if model.checkpoint is None or model.config is None:
            raise VoiceConversionError(f"voice model '{model.name}' has no checkpoint or config")

        workdir = Path(tempfile.mkdtemp(prefix="vctts-svc-"))
        source = workdir / "in.wav"
        target = workdir / "out.wav"
        try:
            audio_utils.save_audio(source, audio, sample_rate)
            args = [
                binary, "infer", source, "-o", target,
                "-m", model.checkpoint.parent, "-c", model.config,
                "-t", int(transpose), "-db", self.db_thresh,
                "-d", self.resolve_device(),
                "-a" if self.auto_predict_f0 else "-na",
            ]
            if model.speaker:
                args += ["-s", model.speaker]
            run_command(args, label="svc infer", timeout=self.timeout)
            if not target.exists():
                raise VoiceConversionError("svc infer produced no output file")
            converted, converted_sr = audio_utils.load_audio(target, mono=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        if converted_sr != sample_rate:
            converted = audio_utils.resample(converted, converted_sr, sample_rate)
        return converted.astype(np.float32)
