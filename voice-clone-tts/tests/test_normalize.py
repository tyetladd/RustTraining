import pytest

from voice_clone_tts.text.normalize import (
    clean_whitespace,
    ru_ordinal_to_words,
    en_number_to_words,
    normalize_text,
    number_to_words,
    ru_number_to_words,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "ноль"),
        (1, "один"),
        (11, "одиннадцать"),
        (21, "двадцать один"),
        (100, "сто"),
        (247, "двести сорок семь"),
        (1000, "одна тысяча"),
        (2000, "две тысячи"),
        (5000, "пять тысяч"),
        (1_000_000, "один миллион"),
        (2_024, "две тысячи двадцать четыре"),
        (-5, "минус пять"),
        (1_234_567, "один миллион двести тридцать четыре тысячи пятьсот шестьдесят семь"),
    ],
)
def test_russian_numbers(value, expected):
    assert ru_number_to_words(value) == expected


def test_russian_feminine():
    assert ru_number_to_words(1, feminine=True) == "одна"
    assert ru_number_to_words(2, feminine=True) == "две"
    assert ru_number_to_words(22, feminine=True) == "двадцать две"


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "zero"),
        (13, "thirteen"),
        (42, "forty-two"),
        (100, "one hundred"),
        (1024, "one thousand twenty-four"),
        (-7, "minus seven"),
    ],
)
def test_english_numbers(value, expected):
    assert en_number_to_words(value) == expected


def test_number_to_words_dispatches_on_language():
    assert number_to_words(3, "ru") == "три"
    assert number_to_words(3, "en") == "three"


def test_clean_whitespace_normalizes_punctuation():
    assert clean_whitespace("«Привет»   —  мир …") == '"Привет" - мир...'
    assert clean_whitespace("Что?!! Да......") == "Что?! Да..."


def test_russian_normalization_expands_numbers_and_symbols():
    out = normalize_text("В 2024 г. рост составил 12,5 % и 3 млн руб.", "ru")
    assert "две тысячи двадцать четвёртом году" in out
    assert "процентов" in out
    assert "миллионов" in out
    assert not any(ch.isdigit() for ch in out)


def test_russian_grouped_numbers():
    assert "одна тысяча двести тридцать четыре" in normalize_text("1 234 рубля", "ru")


def test_russian_decimals():
    out = normalize_text("Число 3,14 важно", "ru")
    assert "три целых четырнадцать сотых" in out


def test_english_normalization():
    out = normalize_text("Dr. Smith paid $20 (i.e. 20%).", "en")
    assert "Doctor" in out and "twenty" in out and "percent" in out


def test_normalize_can_be_disabled():
    assert normalize_text("В 2024 г.", "ru", expand=False) == "В 2024 г."


def test_generic_language_keeps_digits():
    assert "2024" in normalize_text("У 2024 годзе", "be")


@pytest.mark.parametrize(
    "value,case,expected",
    [
        (1, "gen", "первого"),
        (3, "gen", "третьего"),
        (3, "prep", "третьем"),
        (8, "prep", "восьмом"),
        (40, "gen", "сорокового"),
        (100, "gen", "сотого"),
        (1990, "gen_pl", "тысяча девятьсот девяностых"),
        (2000, "gen", "двухтысячного"),
        (2024, "prep", "две тысячи двадцать четвёртом"),
    ],
)
def test_russian_ordinals(value, case, expected):
    assert ru_ordinal_to_words(value, case) == expected


def test_years_are_read_as_ordinals():
    assert "в тысяча девятьсот девяносто девятом году" in normalize_text("в 1999 году", "ru")
    assert "до двухтысячного года" in normalize_text("до 2000 года", "ru")
    assert "тысяча девятьсот девяностых годов" in normalize_text("с 1990 гг.", "ru")


def test_abbreviation_keeps_a_sentence_boundary():
    out = normalize_text("Стоит 1000 руб. Я ухожу.", "ru")
    assert "рублей. Я" in out


def test_street_abbreviation_does_not_gain_a_dot():
    assert "улица Ленина" in normalize_text("ул. Ленина", "ru")
    assert "в городе Москва" in normalize_text("в г. Москва", "ru")


def test_currency_sign_moves_behind_the_number():
    assert "twenty dollars" in normalize_text("$20", "en")
    assert "пятьсот рублей" in normalize_text("500 ₽", "ru")


def test_clock_times():
    assert "десять тридцать" in normalize_text("в 10:30", "ru")
    assert "nine fifteen" in normalize_text("at 9:15", "en")
