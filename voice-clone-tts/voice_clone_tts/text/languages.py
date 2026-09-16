"""Language registry.

Everything language specific is described by a :class:`LanguageSpec`, so adding
a new language is a data change, not a code change::

    from voice_clone_tts.text.languages import LanguageSpec, register_language

    register_language(LanguageSpec(
        code="uk", name="Ukrainian", aliases=("ukr", "ua"),
        whisper_code="uk", xtts_code=None,
        stress_provider="silero", stress_lang="ukr",
        normalizer="ru",   # Cyrillic digits/abbreviation rules are close enough
    ))

The same can be done from the command line with ``--language-config file.json``
(see :func:`load_language_config`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from voice_clone_tts.errors import LanguageError


@dataclass(frozen=True)
class LanguageSpec:
    """Static description of one supported language."""

    code: str
    """Canonical ISO-639-1 code used everywhere inside the pipeline."""

    name: str
    """Human readable name, used in CLI listings."""

    aliases: tuple[str, ...] = ()
    """Extra spellings accepted from users (``rus``, ``ru-RU``, ``русский``…)."""

    whisper_code: str | None = None
    """Language code passed to Whisper; ``None`` means "let Whisper decide"."""

    xtts_code: str | None = None
    """Language code understood by the XTTS backend, if it supports it at all."""

    stress_provider: str | None = None
    """Name of the accentor factory (currently only ``"silero"``)."""

    stress_lang: str | None = None
    """Language argument handed to the accentor factory (``ru``/``ukr``/``bel``)."""

    normalizer: str = "generic"
    """Which text normalizer to use: ``ru``, ``en`` or ``generic``."""

    sentence_max_chars: int = 220
    """Soft limit for one synthesis chunk; XTTS degrades on longer inputs."""

    extra: dict[str, str] = field(default_factory=dict)
    """Free-form slot for user defined languages."""

    @property
    def has_stress(self) -> bool:
        return self.stress_provider is not None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "aliases": list(self.aliases),
            "whisper_code": self.whisper_code,
            "xtts_code": self.xtts_code,
            "stress_provider": self.stress_provider,
            "stress_lang": self.stress_lang,
            "normalizer": self.normalizer,
            "sentence_max_chars": self.sentence_max_chars,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "LanguageSpec":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821 - dataclass attr
        unknown = set(raw) - known
        if unknown:
            raise LanguageError(f"unknown language spec fields: {sorted(unknown)}")
        if "code" not in raw or "name" not in raw:
            raise LanguageError("language spec needs at least 'code' and 'name'")
        data = dict(raw)
        data["aliases"] = tuple(data.get("aliases", ()))
        data["extra"] = dict(data.get("extra", {}))
        return cls(**data)


RUSSIAN = LanguageSpec(
    code="ru",
    name="Russian",
    aliases=("rus", "ru-ru", "russian", "русский"),
    whisper_code="ru",
    xtts_code="ru",
    stress_provider="silero",
    stress_lang="ru",
    normalizer="ru",
    sentence_max_chars=200,
)

ENGLISH = LanguageSpec(
    code="en",
    name="English",
    aliases=("eng", "en-us", "en-gb", "english", "английский"),
    whisper_code="en",
    xtts_code="en",
    stress_provider=None,  # English stress is handled by the TTS model itself
    stress_lang=None,
    normalizer="en",
    sentence_max_chars=240,
)

# Shipped but not enabled by default in the CLI help: silero-stress also has
# Ukrainian and Belarusian accentors, and XTTS speaks neither, so these are a
# demonstration of how far the registry stretches.
UKRAINIAN = LanguageSpec(
    code="uk",
    name="Ukrainian",
    aliases=("ukr", "ua", "uk-ua", "українська"),
    whisper_code="uk",
    xtts_code=None,
    stress_provider="silero",
    stress_lang="ukr",
    normalizer="generic",
)

BELARUSIAN = LanguageSpec(
    code="be",
    name="Belarusian",
    aliases=("bel", "by", "be-by", "беларуская"),
    whisper_code="be",
    xtts_code=None,
    stress_provider="silero",
    stress_lang="bel",
    normalizer="generic",
)

_REGISTRY: dict[str, LanguageSpec] = {}
_ALIASES: dict[str, str] = {}


def register_language(spec: LanguageSpec, *, overwrite: bool = True) -> LanguageSpec:
    """Add (or replace) a language in the registry."""
    code = spec.code.lower()
    if not overwrite and code in _REGISTRY:
        raise LanguageError(f"language '{code}' is already registered")
    _REGISTRY[code] = spec
    _ALIASES[code] = code
    for alias in spec.aliases:
        _ALIASES[alias.lower()] = code
    return spec


def normalize_code(code: str) -> str:
    """Map any accepted spelling to the canonical code."""
    key = code.strip().lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    # "ru-RU" -> "ru" as a last resort
    head = key.split("-", 1)[0]
    if head in _ALIASES:
        return _ALIASES[head]
    raise LanguageError(
        f"unsupported language '{code}'. Known: {', '.join(sorted(_REGISTRY))}. "
        "Add your own with --language-config."
    )


def get_language(code: str) -> LanguageSpec:
    """Look up a language by code or alias."""
    return _REGISTRY[normalize_code(code)]


def is_supported(code: str) -> bool:
    try:
        normalize_code(code)
    except LanguageError:
        return False
    return True


def list_languages() -> list[LanguageSpec]:
    return [_REGISTRY[code] for code in sorted(_REGISTRY)]


def load_language_config(path: str | Path) -> list[LanguageSpec]:
    """Register extra languages from a JSON file.

    The file holds either a list of language specs or an object with a
    ``"languages"`` key. Fields match :class:`LanguageSpec`.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: Iterable[dict]
    if isinstance(raw, dict):
        items = raw.get("languages", [])
    else:
        items = raw
    return [register_language(LanguageSpec.from_dict(item)) for item in items]


for _spec in (RUSSIAN, ENGLISH, UKRAINIAN, BELARUSIAN):
    register_language(_spec)
