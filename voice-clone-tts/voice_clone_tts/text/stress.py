"""Stress and ``ё`` placement built on silero-stress.

`silero-stress <https://github.com/snakers4/silero-stress>`_ covers ~4M Russian
word forms plus ~2.2K homographs, and restores ``ё``. It marks the stressed
vowel with a ``+`` *before* the vowel::

    "Я из готов" -> "+Я +из г+отов"

Backends disagree about what to do with those marks — Silero TTS consumes them
natively, XTTS has never seen a ``+`` in training — so the pipeline keeps the
``+`` form as its internal representation and each backend asks for the
:class:`StressStyle` it understands.
"""

from __future__ import annotations

import logging
import re
import threading
from enum import Enum
from typing import Iterable, Protocol

from voice_clone_tts.errors import MissingDependencyError
from voice_clone_tts.text.languages import LanguageSpec, get_language

log = logging.getLogger(__name__)

STRESS_MARK = "+"
"""Marker silero-stress puts before a stressed vowel."""

COMBINING_ACUTE = "́"
"""Unicode combining acute accent, placed *after* the vowel."""

_VOWELS = "аеиоуыэюяёaeiouy"
_PLUS_BEFORE_VOWEL = re.compile(rf"\+([{_VOWELS}])", re.IGNORECASE)
_ACUTE_AFTER_VOWEL = re.compile(rf"([{_VOWELS}]){COMBINING_ACUTE}", re.IGNORECASE)


class StressStyle(str, Enum):
    """How stress marks should look in the text handed to a TTS backend."""

    PLUS = "plus"
    """``гот+ов`` — Silero TTS and most Russian G2P front-ends."""

    ACUTE = "acute"
    """``гото́в`` — combining U+0301, used by some multilingual front-ends."""

    NONE = "none"
    """Marks removed; ``ё`` restoration still survives."""

    @classmethod
    def parse(cls, value: "str | StressStyle") -> "StressStyle":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


def convert_stress_marks(text: str, style: str | StressStyle) -> str:
    """Rewrite ``+``-marked text into the requested style."""
    style = StressStyle.parse(style)
    if style is StressStyle.PLUS:
        return text
    if style is StressStyle.ACUTE:
        return _PLUS_BEFORE_VOWEL.sub(rf"\1{COMBINING_ACUTE}", text).replace(STRESS_MARK, "")
    return strip_stress_marks(text)


def strip_stress_marks(text: str) -> str:
    """Remove every stress marker, keeping the letters (including ``ё``)."""
    text = _ACUTE_AFTER_VOWEL.sub(r"\1", text)
    return _PLUS_BEFORE_VOWEL.sub(r"\1", text).replace(STRESS_MARK, "")


class Accentor(Protocol):
    """Anything that turns plain text into ``+``-marked text."""

    def __call__(self, text: str) -> str:  # pragma: no cover - protocol
        ...


class SileroStressAccentor:
    """Lazy wrapper around ``silero_stress.load_accentor``.

    The 56 MB model is loaded on first use and then kept for the lifetime of
    the object; loading takes a few seconds, a sentence takes tens of
    milliseconds.
    """

    def __init__(
        self,
        lang: str = "ru",
        *,
        device: str = "cpu",
        put_stress: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = True,
        put_yo_homo: bool = True,
        stress_single_vowel: bool = True,
        words_to_ignore: Iterable[str] | None = None,
    ) -> None:
        self.lang = lang
        self.device = device
        self.flags = {
            "put_stress": put_stress,
            "put_yo": put_yo,
            "put_stress_homo": put_stress_homo,
            "put_yo_homo": put_yo_homo,
            "stress_single_vowel": stress_single_vowel,
        }
        self.words_to_ignore = list(words_to_ignore) if words_to_ignore else None
        self._model = None
        self._lock = threading.Lock()

    @staticmethod
    def is_available() -> bool:
        """True when silero-stress (and torch) can be imported."""
        try:
            import silero_stress  # noqa: F401
        except Exception:
            return False
        return True

    def load(self):
        """Load the accentor model (idempotent, thread-safe)."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from silero_stress import load_accentor
            except ImportError as exc:  # pragma: no cover - depends on env
                raise MissingDependencyError(
                    "silero-stress", "stress", "Stress placement"
                ) from exc
            log.info("loading silero-stress accentor (lang=%s)", self.lang)
            model = load_accentor(lang=self.lang)
            if model is None:
                raise MissingDependencyError(
                    "silero-stress", "stress",
                    f"Stress placement for language '{self.lang}' "
                    "(silero-stress ships ru/ukr/bel accentors)",
                )
            if self.device != "cpu":
                model.to(device=self.device)
            self._model = model
            return model

    def __call__(self, text: str) -> str:
        model = self.load()
        kwargs = dict(self.flags)
        if self.words_to_ignore:
            kwargs["words_to_ignore"] = self.words_to_ignore
        # ukr/bel accentors have no homograph solver and reject *_homo flags.
        if self.lang != "ru":
            kwargs.pop("put_stress_homo", None)
            kwargs.pop("put_yo_homo", None)
        try:
            return model(text, **kwargs)
        except TypeError:
            # Be tolerant of flag renames in future silero-stress releases.
            log.debug("accentor rejected flags %s, retrying bare", sorted(kwargs))
            return model(text)


_ACCENTOR_CACHE: dict[tuple[str, str], SileroStressAccentor] = {}
_CACHE_LOCK = threading.Lock()


def get_accentor(language: str | LanguageSpec = "ru", *, device: str = "cpu", **kwargs):
    """Return a cached accentor for `language`, or ``None`` if it has none.

    Extra keyword arguments are forwarded to :class:`SileroStressAccentor`;
    passing any of them bypasses the cache so callers cannot poison it.
    """
    spec = language if isinstance(language, LanguageSpec) else get_language(language)
    if not spec.has_stress:
        return None
    if spec.stress_provider != "silero":
        raise MissingDependencyError(
            spec.stress_provider or "?", "stress",
            f"Stress provider '{spec.stress_provider}' for language '{spec.code}'",
        )
    stress_lang = spec.stress_lang or spec.code
    if kwargs:
        return SileroStressAccentor(stress_lang, device=device, **kwargs)
    key = (stress_lang, device)
    with _CACHE_LOCK:
        accentor = _ACCENTOR_CACHE.get(key)
        if accentor is None:
            accentor = SileroStressAccentor(stress_lang, device=device)
            _ACCENTOR_CACHE[key] = accentor
    return accentor


def put_stress(
    text: str,
    language: str | LanguageSpec = "ru",
    *,
    style: str | StressStyle = StressStyle.PLUS,
    accentor: Accentor | None = None,
    device: str = "cpu",
    strict: bool = False,
) -> str:
    """Place stress marks and ``ё`` in `text`.

    `accentor` can be injected (tests, custom models). When the language has no
    accentor, or silero-stress is not installed and ``strict`` is False, the
    text is returned unchanged — synthesis still works, it just loses the
    homograph disambiguation.
    """
    spec = language if isinstance(language, LanguageSpec) else get_language(language)
    if accentor is None:
        try:
            accentor = get_accentor(spec, device=device)
        except MissingDependencyError:
            if strict:
                raise
            log.warning("stress placement unavailable for '%s', continuing without it", spec.code)
            return text
    if accentor is None:
        return text
    try:
        stressed = accentor(text)
    except MissingDependencyError:
        if strict:
            raise
        log.warning("silero-stress is not installed, continuing without stress marks")
        return text
    return convert_stress_marks(stressed, style)
