import pytest

from voice_clone_tts.errors import MissingDependencyError
from voice_clone_tts.text.stress import (
    COMBINING_ACUTE,
    SileroStressAccentor,
    StressStyle,
    convert_stress_marks,
    get_accentor,
    put_stress,
    strip_stress_marks,
)


def test_plus_style_is_identity():
    assert convert_stress_marks("гот+ов", StressStyle.PLUS) == "гот+ов"


def test_acute_style_moves_the_mark_after_the_vowel():
    assert convert_stress_marks("гот+ов", "acute") == "гото" + COMBINING_ACUTE + "в"


def test_none_style_strips_marks_but_keeps_yo():
    assert convert_stress_marks("Л+ёва Корол+ёв", StressStyle.NONE) == "Лёва Королёв"


def test_strip_removes_both_notations():
    assert strip_stress_marks("гото" + COMBINING_ACUTE + "в и г+отов") == "готов и готов"


def test_put_stress_uses_injected_accentor(fake_accentor):
    assert put_stress("замок", "ru", accentor=fake_accentor) == "з+амок"
    assert fake_accentor.calls == ["замок"]


def test_put_stress_converts_style(fake_accentor):
    out = put_stress("замок", "ru", style=StressStyle.NONE, accentor=fake_accentor)
    assert out == "замок"


def test_english_has_no_accentor():
    assert get_accentor("en") is None
    assert put_stress("hello", "en") == "hello"


def test_missing_dependency_is_not_fatal_by_default(monkeypatch):
    def explode(self):
        raise MissingDependencyError("silero-stress", "stress", "Stress placement")

    monkeypatch.setattr(SileroStressAccentor, "load", explode)
    assert put_stress("замок", "ru") == "замок"
    with pytest.raises(MissingDependencyError):
        put_stress("замок", "ru", strict=True)


def test_accentor_cache_is_shared():
    assert get_accentor("ru") is get_accentor("ru")
    assert get_accentor("ru") is not get_accentor("ru", put_yo=False)
