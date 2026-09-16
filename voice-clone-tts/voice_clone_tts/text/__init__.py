"""Text front-end: normalization, stress placement, chunking."""

from voice_clone_tts.text.chunking import split_into_chunks, split_into_sentences
from voice_clone_tts.text.languages import (
    LanguageSpec,
    get_language,
    list_languages,
    load_language_config,
    normalize_code,
    register_language,
)
from voice_clone_tts.text.normalize import normalize_text, number_to_words
from voice_clone_tts.text.stress import (
    STRESS_MARK,
    SileroStressAccentor,
    StressStyle,
    convert_stress_marks,
    get_accentor,
    put_stress,
)

__all__ = [
    "LanguageSpec",
    "STRESS_MARK",
    "SileroStressAccentor",
    "StressStyle",
    "convert_stress_marks",
    "get_accentor",
    "get_language",
    "list_languages",
    "load_language_config",
    "normalize_code",
    "normalize_text",
    "number_to_words",
    "put_stress",
    "register_language",
    "split_into_chunks",
    "split_into_sentences",
]
