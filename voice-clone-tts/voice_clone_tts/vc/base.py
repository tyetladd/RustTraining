"""Voice conversion: interface, trained-model metadata and registry.

Silero TTS speaks with its own stock voices, so cloning is done in a second
stage: the synthesized audio is passed through a voice-conversion model that
was *trained* on the target speaker. That is real training (minutes of audio,
GPU time), unlike the few-shot conditioning XTTS uses.

A converter is a driver around an external toolchain; adding one means
subclassing :class:`VoiceConverter` and registering it.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type

import numpy as np

from voice_clone_tts.errors import VoiceCloneError
from voice_clone_tts.vc.dataset import TrainingDataset

log = logging.getLogger(__name__)

MODEL_FILE = "voice_model.json"
MODEL_FORMAT = 1

STAGES = ("dataset", "preprocess", "extract", "train")
"""Стадии обучения по порядку; на любой можно остановиться.

Нужно для проверки конвейера там, где нет CUDA: нарезка и извлечение
признаков считаются на CPU, а обучение — нет.
"""


class VoiceConversionError(VoiceCloneError):
    """A voice conversion model is unusable, unknown or failed."""


@dataclass
class VoiceModel:
    """A voice-conversion model trained on one speaker."""

    name: str
    directory: Path
    converter: str
    checkpoint: Path | None = None
    config: Path | None = None
    index: Path | None = None
    speaker: str = "target"
    sample_rate: int = 44_100
    source_profile: str | None = None
    median_f0: float = 0.0
    """Median pitch of the target speaker; used for automatic transposition."""
    created_at: str = ""
    train_stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def relative(path: Path | None) -> str | None:
            if path is None:
                return None
            try:
                return str(Path(path).relative_to(self.directory))
            except ValueError:
                return str(path)

        return {
            "format": MODEL_FORMAT,
            "name": self.name,
            "converter": self.converter,
            "checkpoint": relative(self.checkpoint),
            "config": relative(self.config),
            "index": relative(self.index),
            "speaker": self.speaker,
            "sample_rate": self.sample_rate,
            "source_profile": self.source_profile,
            "median_f0": round(self.median_f0, 2),
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_stats": self.train_stats,
            "metadata": self.metadata,
        }

    def save(self, directory: str | Path | None = None) -> Path:
        target = Path(directory) if directory else self.directory
        target.mkdir(parents=True, exist_ok=True)
        self.directory = target
        (target / MODEL_FILE).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "VoiceModel":
        directory = Path(directory)
        path = directory / MODEL_FILE
        if not path.exists():
            raise VoiceConversionError(
                f"no {MODEL_FILE} in {directory}. Train one with:  vctts voice train …"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format", 1) > MODEL_FORMAT:
            raise VoiceConversionError(
                f"voice model {directory} was written by a newer version "
                f"(format {raw['format']} > {MODEL_FORMAT})"
            )

        def absolute(value: str | None) -> Path | None:
            if not value:
                return None
            path = Path(value)
            return path if path.is_absolute() else directory / path

        model = cls(
            name=raw.get("name", directory.name),
            directory=directory,
            converter=raw.get("converter", ""),
            checkpoint=absolute(raw.get("checkpoint")),
            config=absolute(raw.get("config")),
            index=absolute(raw.get("index")),
            speaker=raw.get("speaker", "target"),
            sample_rate=int(raw.get("sample_rate", 44_100)),
            source_profile=raw.get("source_profile"),
            median_f0=float(raw.get("median_f0", 0.0)),
            created_at=raw.get("created_at", ""),
            train_stats=raw.get("train_stats", {}),
            metadata=raw.get("metadata", {}),
        )
        if model.checkpoint is not None and not model.checkpoint.exists():
            raise VoiceConversionError(
                f"voice model {directory} points at a missing checkpoint: {model.checkpoint}"
            )
        return model

    def describe(self) -> str:
        lines = [
            f"voice model : {self.name}",
            f"converter   : {self.converter}",
            f"directory   : {self.directory}",
            f"checkpoint  : {self.checkpoint}",
            f"speaker     : {self.speaker} @ {self.sample_rate} Hz",
        ]
        if self.median_f0:
            lines.append(f"median F0   : {self.median_f0:.0f} Hz")
        if self.source_profile:
            lines.append(f"from profile: {self.source_profile}")
        for key, value in self.train_stats.items():
            lines.append(f"  {key:<10}: {value}")
        return "\n".join(lines)


class VoiceConverter(ABC):
    """Base class for a voice-conversion toolchain driver."""

    name: str = "base"
    display_name: str = "Base converter"
    description: str = ""
    install_hint: str = ""
    supports_training: bool = True
    needs_gpu: bool = True

    def __init__(self, *, device: str = "auto", **options: Any) -> None:
        self.device = device
        self.options = options
        self._loaded_model: VoiceModel | None = None

    @classmethod
    def is_available(cls) -> bool:
        """True when the external toolchain can be invoked."""
        return False

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:  # noqa: BLE001 - torch is optional
            pass
        return "cpu"

    @abstractmethod
    def train(
        self,
        dataset: TrainingDataset,
        out_dir: str | Path,
        *,
        name: str | None = None,
        epochs: int | None = None,
        resume: bool = False,
        stop_after: str | None = None,
    ) -> VoiceModel:
        """Train a model on `dataset` and return it.

        With `stop_after` set to a stage from :data:`STAGES`, the run stops
        once that stage is done and returns a model without a checkpoint.
        """

    @abstractmethod
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: VoiceModel,
        *,
        transpose: int = 0,
    ) -> np.ndarray:
        """Convert `audio` into the model's voice, returning float32 mono."""

    def load(self, model: VoiceModel) -> None:
        """Optional warm-up before the first :meth:`convert`."""
        self._loaded_model = model

    def close(self) -> None:
        self._loaded_model = None

    def info(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "supports_training": self.supports_training,
            "needs_gpu": self.needs_gpu,
            "available": type(self).is_available(),
            "install_hint": self.install_hint,
        }


_CONVERTERS: dict[str, Type[VoiceConverter]] = {}


def register_converter(cls: Type[VoiceConverter]) -> Type[VoiceConverter]:
    if not getattr(cls, "name", None) or cls.name == "base":
        raise VoiceConversionError(f"{cls.__name__} must define a unique 'name'")
    _CONVERTERS[cls.name] = cls
    return cls


def converter_class(name: str) -> Type[VoiceConverter]:
    try:
        return _CONVERTERS[name]
    except KeyError:
        raise VoiceConversionError(
            f"unknown voice converter '{name}'. Available: {', '.join(sorted(_CONVERTERS))}"
        ) from None


def get_converter(name: str, **kwargs: Any) -> VoiceConverter:
    return converter_class(name)(**kwargs)


def list_converters() -> list[Type[VoiceConverter]]:
    return [_CONVERTERS[name] for name in sorted(_CONVERTERS)]


def iter_converter_info() -> list[dict]:
    return [cls().info() for cls in list_converters()]
