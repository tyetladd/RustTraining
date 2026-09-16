"""TTS backend interface and registry.

A backend is the only part of the pipeline that knows about a concrete neural
model. Adding one means subclassing :class:`TTSBackend`, declaring what it
supports and registering it::

    @register_backend
    class MyBackend(TTSBackend):
        name = "my-tts"
        supported_languages = {"ru", "en"}
        clones_voice = True
        native_stress_style = StressStyle.NONE
        sample_rate = 24_000

        def load(self): ...
        def synthesize(self, request): return numpy_float32_waveform
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Type

import numpy as np

from voice_clone_tts.errors import BackendError
from voice_clone_tts.profile import SpeakerProfile
from voice_clone_tts.text.languages import LanguageSpec
from voice_clone_tts.text.stress import StressStyle

log = logging.getLogger(__name__)


@dataclass
class SynthesisRequest:
    """One chunk of text to render with one voice."""

    text: str
    language: LanguageSpec
    profile: SpeakerProfile | None = None
    speed: float = 1.0
    temperature: float = 0.7
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


class TTSBackend(ABC):
    """Base class for every speech synthesis engine."""

    name: str = "base"
    display_name: str = "Base backend"
    description: str = ""
    supported_languages: set[str] | None = None
    """``None`` means "ask the language spec instead" (any language works)."""
    requires_reference: bool = True
    clones_voice: bool = True
    native_stress_style: StressStyle = StressStyle.NONE
    sample_rate: int = 24_000
    install_extra: str | None = None

    def __init__(self, *, device: str = "auto", **options: Any) -> None:
        self.device = device
        self.options = options
        self._loaded = False

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def is_available(cls) -> bool:
        """True when the backend's dependencies are importable."""
        return True

    def load(self) -> None:
        """Load models. Called once before the first :meth:`synthesize`."""
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def close(self) -> None:
        """Release models (optional)."""
        self._loaded = False

    def __enter__(self) -> "TTSBackend":
        self.ensure_loaded()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- capabilities -----------------------------------------------------
    def supports_language(self, language: LanguageSpec) -> bool:
        if self.supported_languages is None:
            return True
        return language.code in self.supported_languages

    def resolve_device(self) -> str:
        """Turn ``"auto"`` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001 - torch is optional
            pass
        return "cpu"

    def seed_everything(self, seed: int | None) -> None:
        if seed is None:
            return
        np.random.seed(seed % (2**32))
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:  # noqa: BLE001
            pass

    # -- synthesis --------------------------------------------------------
    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> np.ndarray:
        """Render one chunk and return float32 mono at :attr:`sample_rate`."""

    def info(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "languages": sorted(self.supported_languages) if self.supported_languages else "any",
            "clones_voice": self.clones_voice,
            "requires_reference": self.requires_reference,
            "stress_style": self.native_stress_style.value,
            "sample_rate": self.sample_rate,
            "available": self.__class__.is_available(),
            "install_extra": self.install_extra,
        }


_BACKENDS: dict[str, Type[TTSBackend]] = {}


def register_backend(cls: Type[TTSBackend]) -> Type[TTSBackend]:
    """Class decorator that adds a backend to the registry."""
    if not getattr(cls, "name", None) or cls.name == "base":
        raise BackendError(f"{cls.__name__} must define a unique 'name'")
    _BACKENDS[cls.name] = cls
    return cls


def backend_class(name: str) -> Type[TTSBackend]:
    try:
        return _BACKENDS[name]
    except KeyError:
        raise BackendError(
            f"unknown backend '{name}'. Available: {', '.join(sorted(_BACKENDS))}"
        ) from None


def get_backend(name: str, **kwargs: Any) -> TTSBackend:
    """Instantiate a registered backend."""
    cls = backend_class(name)
    if not cls.is_available():
        raise BackendError(
            f"backend '{name}' is registered but its dependencies are missing. "
            + (f"Install them with:  pip install 'voice-clone-tts[{cls.install_extra}]'"
               if cls.install_extra else "")
        )
    return cls(**kwargs)


def list_backends() -> list[Type[TTSBackend]]:
    return [_BACKENDS[name] for name in sorted(_BACKENDS)]


def available_backends() -> list[str]:
    return [name for name, cls in sorted(_BACKENDS.items()) if cls.is_available()]


def iter_backend_info() -> Iterable[dict]:
    for cls in list_backends():
        yield {
            "name": cls.name,
            "display_name": cls.display_name,
            "description": cls.description,
            "languages": sorted(cls.supported_languages) if cls.supported_languages else "any",
            "clones_voice": cls.clones_voice,
            "stress_style": cls.native_stress_style.value,
            "sample_rate": cls.sample_rate,
            "available": cls.is_available(),
            "install_extra": cls.install_extra,
        }


def _lazy_import(module: str, purpose: str, extra: str) -> Callable[[], Any]:
    """Helper for backends: import a heavy module with a helpful error."""

    def _import() -> Any:
        from importlib import import_module

        from voice_clone_tts.errors import MissingDependencyError

        try:
            return import_module(module)
        except ImportError as exc:
            raise MissingDependencyError(module.split(".")[0], extra, purpose) from exc

    return _import
