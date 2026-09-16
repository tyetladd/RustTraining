import json

import pytest

from voice_clone_tts.errors import LanguageError
from voice_clone_tts.text.languages import (
    LanguageSpec,
    get_language,
    is_supported,
    list_languages,
    load_language_config,
    normalize_code,
    register_language,
)


def test_builtin_languages():
    codes = {spec.code for spec in list_languages()}
    assert {"ru", "en"} <= codes


def test_aliases_and_region_codes():
    assert normalize_code("RUS") == "ru"
    assert normalize_code("ru-RU") == "ru"
    assert normalize_code("en_US") == "en"
    assert get_language("русский").code == "ru"


def test_unknown_language_raises():
    with pytest.raises(LanguageError):
        get_language("klingon")
    assert not is_supported("klingon")


def test_russian_has_a_stress_provider():
    spec = get_language("ru")
    assert spec.has_stress and spec.stress_provider == "silero" and spec.stress_lang == "ru"
    assert not get_language("en").has_stress


def test_register_custom_language():
    register_language(LanguageSpec(code="de", name="German", aliases=("deu",), xtts_code="de"))
    assert get_language("deu").xtts_code == "de"


def test_load_language_config(tmp_path):
    path = tmp_path / "langs.json"
    path.write_text(json.dumps({"languages": [
        {"code": "kk", "name": "Kazakh", "aliases": ["kaz"], "normalizer": "generic"}
    ]}), encoding="utf-8")
    load_language_config(path)
    assert get_language("kaz").name == "Kazakh"


def test_bad_language_config_is_reported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"code": "xx", "name": "X", "nonsense": 1}]), encoding="utf-8")
    with pytest.raises(LanguageError):
        load_language_config(path)
