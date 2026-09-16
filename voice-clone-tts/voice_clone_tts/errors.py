"""Exception hierarchy shared by the whole pipeline."""

from __future__ import annotations


class VoiceCloneError(Exception):
    """Base class for every error raised by this package."""


class MissingDependencyError(VoiceCloneError):
    """An optional dependency is required for the requested feature.

    The message always names the extra that installs it, because the heavy
    parts (torch, whisper, XTTS) are deliberately optional.
    """

    def __init__(self, package: str, extra: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} requires the '{package}' package.\n"
            f"Install it with:  pip install 'voice-clone-tts[{extra}]'   "
            f"(or:  pip install {package})"
        )
        self.package = package
        self.extra = extra


class AudioError(VoiceCloneError):
    """The reference audio cannot be read, decoded or used."""


class LanguageError(VoiceCloneError):
    """The requested language is unknown or unsupported by a component."""


class BackendError(VoiceCloneError):
    """A TTS backend is unknown, unusable or failed during synthesis."""


class TextError(VoiceCloneError):
    """The text to speak is empty or cannot be processed."""
