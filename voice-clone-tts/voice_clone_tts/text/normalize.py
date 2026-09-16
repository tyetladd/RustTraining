"""Text normalization: whitespace, punctuation, symbols and numbers.

TTS models read what you give them literally, so "в 2024 г. было +5 °C" has to
become words before it reaches the model. Normalizers are picked per language
through :attr:`LanguageSpec.normalizer`; the ``generic`` one only does the
language independent cleanup.
"""

from __future__ import annotations

import re
from typing import Callable

from voice_clone_tts.text.languages import LanguageSpec, get_language

# --------------------------------------------------------------------------
# Language independent cleanup
# --------------------------------------------------------------------------

_QUOTES = {
    "«": '"', "»": '"', "“": '"', "”": '"',
    "„": '"', "‘": "'", "’": "'", "‚": "'",
}
_DASHES = {"–": "-", "—": "-", "‒": "-", "−": "-", "‐": "-"}
_MISC = {"…": "...", " ": " ", " ": " ", "​": "", "﻿": ""}

_TRANSLATION = str.maketrans({**_QUOTES, **_DASHES, **_MISC})

_MULTI_SPACE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE = re.compile(r"\n{2,}")
_MULTI_PUNCT = re.compile(r"([!?,;:])\1{1,}")
_LONG_ELLIPSIS = re.compile(r"\.{4,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.!?,;:])")

# 1 234 567 / 1,234,567 -> 1234567 (digit grouping, not a list of numbers)
_GROUPED_NUMBER = re.compile(r"(?<=\d)[  ,](?=\d{3}\b)")


def clean_whitespace(text: str) -> str:
    """Normalize exotic unicode punctuation and collapse whitespace."""
    text = text.translate(_TRANSLATION)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_PUNCT.sub(r"\1", text)
    text = _LONG_ELLIPSIS.sub("...", text)
    return text.strip()


# --------------------------------------------------------------------------
# Russian numbers
# --------------------------------------------------------------------------

_RU_ONES_M = (
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь",
    "девять", "десять", "одиннадцать", "двенадцать", "тринадцать",
    "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать",
    "девятнадцать",
)
_RU_ONES_F = {1: "одна", 2: "две"}
_RU_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
    "семьдесят", "восемьдесят", "девяносто",
)
_RU_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
    "семьсот", "восемьсот", "девятьсот",
)
# (singular, few, many) for each 1000^n scale
_RU_SCALES = (
    None,
    ("тысяча", "тысячи", "тысяч"),
    ("миллион", "миллиона", "миллионов"),
    ("миллиард", "миллиарда", "миллиардов"),
    ("триллион", "триллиона", "триллионов"),
)
_RU_FRACTIONS = {
    1: ("десятая", "десятых"),
    2: ("сотая", "сотых"),
    3: ("тысячная", "тысячных"),
}


def _ru_plural(n: int, forms: tuple[str, str, str]) -> str:
    """Pick the Russian plural form for `n` (1 рубль / 2 рубля / 5 рублей)."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _ru_triplet(n: int, feminine: bool) -> list[str]:
    """Spell a number in 1..999."""
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_RU_HUNDREDS[hundreds])
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        words.append(_RU_TENS[tens])
        if ones:
            words.append(_RU_ONES_F[ones] if feminine and ones in _RU_ONES_F else _RU_ONES_M[ones])
    elif rest:
        words.append(_RU_ONES_F[rest] if feminine and rest in _RU_ONES_F else _RU_ONES_M[rest])
    return words


def ru_number_to_words(number: int, *, feminine: bool = False) -> str:
    """Spell an integer in Russian (nominative case)."""
    if number < 0:
        return "минус " + ru_number_to_words(-number, feminine=feminine)
    if number == 0:
        return _RU_ONES_M[0]

    triplets: list[int] = []
    rest = number
    while rest:
        rest, triplet = divmod(rest, 1000)
        triplets.append(triplet)
    if len(triplets) > len(_RU_SCALES):
        # Too large to name: read it digit by digit rather than lie about it.
        return " ".join(_RU_ONES_M[int(d)] for d in str(number))

    words: list[str] = []
    for scale in range(len(triplets) - 1, -1, -1):
        triplet = triplets[scale]
        if not triplet:
            continue
        # "одна тысяча", but "один миллион"
        words += _ru_triplet(triplet, feminine=feminine if scale == 0 else scale == 1)
        if scale:
            words.append(_ru_plural(triplet, _RU_SCALES[scale]))
    return " ".join(words)


def _ru_decimal_to_words(integer: str, fraction: str) -> str:
    """"3.14" -> "три целых четырнадцать сотых"."""
    fraction = fraction.rstrip("0") or "0"
    int_value = int(integer)
    head = ru_number_to_words(int_value, feminine=True)
    head += " целая" if _ru_plural(int_value, ("1", "2", "5")) == "1" else " целых"
    if fraction == "0":
        return head
    if len(fraction) in _RU_FRACTIONS:
        singular, plural = _RU_FRACTIONS[len(fraction)]
        frac_value = int(fraction)
        tail = ru_number_to_words(frac_value, feminine=True)
        unit = singular if _ru_plural(frac_value, ("1", "2", "5")) == "1" else plural
        return f"{head} {tail} {unit}"
    digits = " ".join(_RU_ONES_M[int(d)] for d in fraction)
    return f"{head} и {digits}"


# --------------------------------------------------------------------------
# English numbers
# --------------------------------------------------------------------------

_EN_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_EN_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
_EN_SCALES = (None, "thousand", "million", "billion", "trillion")


def _en_triplet(n: int) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += [_EN_ONES[hundreds], "hundred"]
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        words.append(_EN_TENS[tens] + (f"-{_EN_ONES[ones]}" if ones else ""))
    elif rest:
        words.append(_EN_ONES[rest])
    return words


def en_number_to_words(number: int) -> str:
    """Spell an integer in English."""
    if number < 0:
        return "minus " + en_number_to_words(-number)
    if number == 0:
        return _EN_ONES[0]

    triplets: list[int] = []
    rest = number
    while rest:
        rest, triplet = divmod(rest, 1000)
        triplets.append(triplet)
    if len(triplets) > len(_EN_SCALES):
        return " ".join(_EN_ONES[int(d)] for d in str(number))

    words: list[str] = []
    for scale in range(len(triplets) - 1, -1, -1):
        triplet = triplets[scale]
        if not triplet:
            continue
        words += _en_triplet(triplet)
        if scale:
            words.append(_EN_SCALES[scale])
    return " ".join(words)


def _en_decimal_to_words(integer: str, fraction: str) -> str:
    digits = " ".join(_EN_ONES[int(d)] for d in fraction)
    return f"{en_number_to_words(int(integer))} point {digits}"


def number_to_words(number: int, language: str = "ru", *, feminine: bool = False) -> str:
    """Spell an integer in the given language."""
    spec = get_language(language)
    if spec.normalizer == "en":
        return en_number_to_words(number)
    return ru_number_to_words(number, feminine=feminine)


# --------------------------------------------------------------------------
# Russian ordinals (years are read as ordinals: "в 2024 г." -> "в две тысячи
# двадцать четвёртого года")
# --------------------------------------------------------------------------

_RU_ORD_UNITS = {
    1: ("перв", False), 2: ("втор", False), 3: ("трет", True),
    4: ("четвёрт", False), 5: ("пят", False), 6: ("шест", False),
    7: ("седьм", False), 8: ("восьм", False), 9: ("девят", False),
    10: ("десят", False), 11: ("одиннадцат", False), 12: ("двенадцат", False),
    13: ("тринадцат", False), 14: ("четырнадцат", False), 15: ("пятнадцат", False),
    16: ("шестнадцат", False), 17: ("семнадцат", False), 18: ("восемнадцат", False),
    19: ("девятнадцат", False),
}
_RU_ORD_TENS = {
    20: "двадцат", 30: "тридцат", 40: "сороков", 50: "пятидесят",
    60: "шестидесят", 70: "семидесят", 80: "восьмидесят", 90: "девяност",
}
_RU_ORD_HUNDREDS = {
    100: "сот", 200: "двухсот", 300: "трёхсот", 400: "четырёхсот",
    500: "пятисот", 600: "шестисот", 700: "семисот", 800: "восьмисот",
    900: "девятисот",
}
_RU_ORD_THOUSANDS = {
    1000: "тысячн", 2000: "двухтысячн", 3000: "трёхтысячн",
    4000: "четырёхтысячн", 5000: "пятитысячн", 6000: "шеститысячн",
    7000: "семитысячн", 8000: "восьмитысячн", 9000: "девятитысячн",
}
# case -> (hard ending, soft ending); soft is only used by "трет-"
_RU_ORD_ENDINGS = {
    "gen": ("ого", "ьего"),
    "prep": ("ом", "ьем"),
    "gen_pl": ("ых", "ьих"),
}


def _ru_year_cardinal(number: int) -> str:
    """Cardinal part of a year: 1900 is "тысяча девятьсот", not "одна тысяча"."""
    if number == 0:
        return ""
    words = ru_number_to_words(number, feminine=False)
    return words[len("одна "):] if words.startswith("одна тысяча") else words


def ru_ordinal_to_words(number: int, case: str = "gen") -> str:
    """Spell an ordinal: only the last component takes the ordinal ending."""
    if number <= 0 or case not in _RU_ORD_ENDINGS:
        return ru_number_to_words(number)

    if number in _RU_ORD_THOUSANDS:
        stem, soft, prefix = _RU_ORD_THOUSANDS[number], False, 0
    else:
        last_two = number % 100
        if last_two == 0:
            hundreds = number % 1000
            if hundreds == 0:  # 10000, 120000, … — not worth guessing
                return ru_number_to_words(number)
            stem, soft, prefix = _RU_ORD_HUNDREDS[hundreds], False, number - hundreds
        elif last_two in _RU_ORD_UNITS:
            stem, soft = _RU_ORD_UNITS[last_two]
            prefix = number - last_two
        elif number % 10 == 0:
            stem, soft, prefix = _RU_ORD_TENS[last_two], False, number - last_two
        else:
            unit = number % 10
            stem, soft = _RU_ORD_UNITS[unit]
            prefix = number - unit

    hard_ending, soft_ending = _RU_ORD_ENDINGS[case]
    ordinal = stem + (soft_ending if soft else hard_ending)
    head = _ru_year_cardinal(prefix)
    return f"{head} {ordinal}".strip()


# "в 2024 г." / "с 1999 года" / "в 2024 году" / "в 1990 гг."
_RU_YEAR_PREP = re.compile(r"\b(\d{1,4})\s*(?:г\.|году)(?!\w)", re.IGNORECASE)
_RU_YEAR_GEN = re.compile(r"\b(\d{1,4})\s*года(?!\w)", re.IGNORECASE)
_RU_YEARS_PLURAL = re.compile(r"\b(\d{1,4})\s*(?:гг\.|годов)(?!\w)", re.IGNORECASE)
_RU_YEAR_CONTEXT = re.compile(r"\bв\s*$", re.IGNORECASE)


def _expand_ru_years(text: str) -> str:
    """Read year numbers as ordinals in the right case."""

    def plural(match: re.Match[str]) -> str:
        return f"{ru_ordinal_to_words(int(match.group(1)), 'gen_pl')} годов"

    def genitive(match: re.Match[str]) -> str:
        return f"{ru_ordinal_to_words(int(match.group(1)), 'gen')} года"

    def prepositional(match: re.Match[str]) -> str:
        # "в 2024 г." is prepositional ("в … году"), a bare "2024 г." is genitive.
        head = text[: match.start()]
        case = "prep" if _RU_YEAR_CONTEXT.search(head) or match.group(0).rstrip().endswith("году") else "gen"
        noun = "году" if case == "prep" else "года"
        return f"{ru_ordinal_to_words(int(match.group(1)), case)} {noun}"

    text = _RU_YEARS_PLURAL.sub(plural, text)
    text = _RU_YEAR_GEN.sub(genitive, text)
    return _RU_YEAR_PREP.sub(prepositional, text)


# --------------------------------------------------------------------------
# Symbols and abbreviations
# --------------------------------------------------------------------------

# (pattern, replacement, flags, may_end_sentence). Case sensitive rules must
# NOT get re.IGNORECASE, or an [А-ЯЁ] lookahead would match lowercase too.
# `may_end_sentence` restores a swallowed full stop before a capitalised word:
# "1000 руб. Я ушёл" stays two sentences, while "ул. Ленина" does not gain one.
_RU_ABBREVIATIONS = [
    # Case agreement needs a parser, so only the common "в г. X" gets inflected.
    (r"(?<=\bв )г\.\s*(?=[А-ЯЁ])", "городе ", 0, False),
    (r"\bг\.\s*(?=[А-ЯЁ])", "город ", 0, False),
    (r"\bт\.\s*е\.", "то есть", re.IGNORECASE, True),
    (r"\bт\.\s*д\.", "так далее", re.IGNORECASE, True),
    (r"\bт\.\s*п\.", "тому подобное", re.IGNORECASE, True),
    (r"\bт\.\s*к\.", "так как", re.IGNORECASE, True),
    (r"\bи\s*др\.", "и другие", re.IGNORECASE, True),
    (r"\bсм\.\s*(?=[А-ЯЁа-яё])", "смотри ", re.IGNORECASE, False),
    (r"\bстр\.", "страница", re.IGNORECASE, False),
    (r"\bул\.", "улица", re.IGNORECASE, False),
    (r"\bпр\.", "проспект", re.IGNORECASE, False),
    (r"\bд\.\s*(?=\d)", "дом ", re.IGNORECASE, False),
    (r"\bруб\.", "рублей", re.IGNORECASE, True),
    (r"\bкоп\.", "копеек", re.IGNORECASE, True),
    (r"\bмлн\b\.?", "миллионов", re.IGNORECASE, True),
    (r"\bмлрд\b\.?", "миллиардов", re.IGNORECASE, True),
    (r"\bтыс\b\.?", "тысяч", re.IGNORECASE, True),
]

_EN_ABBREVIATIONS = [
    (r"\be\.\s*g\.", "for example", re.IGNORECASE, False),
    (r"\bi\.\s*e\.", "that is", re.IGNORECASE, False),
    (r"\betc\.", "et cetera", re.IGNORECASE, True),
    (r"\bvs\.?\b", "versus", re.IGNORECASE, False),
    (r"\bMr\.", "Mister", 0, False),
    (r"\bMrs\.", "Missus", 0, False),
    (r"\bDr\.", "Doctor", 0, False),
    (r"\bSt\.", "Saint", 0, False),
    (r"\bapprox\.", "approximately", re.IGNORECASE, True),
]

_RU_SYMBOLS = {
    "%": " процентов",
    "‰": " промилле",
    "№": "номер ",
    "&": " и ",
    "€": " евро",
    "$": " долларов",
    "£": " фунтов",
    "₽": " рублей",
    "°C": " градусов Цельсия",
    "°": " градусов",
    "+": " плюс ",
    "=": " равно ",
}

_EN_SYMBOLS = {
    "%": " percent",
    "‰": " per mille",
    "№": "number ",
    "&": " and ",
    "€": " euro",
    "$": " dollars",
    "£": " pounds",
    "₽": " rubles",
    "°C": " degrees Celsius",
    "°": " degrees",
    "+": " plus ",
    "=": " equals ",
}

# "$20" is spoken "twenty dollars": move a leading currency sign behind its number.
_CURRENCY_PREFIX = re.compile(r"([$€£₽])\s*(-?\d[\d\s.,]*\d|-?\d)")

_NUMBER = re.compile(r"(?<![\w])(-?\d+)(?:[.,](\d+))?")
_MULTI_DOT_NUMBER = re.compile(r"\b\d+(?:[.:]\d+){2,}\b")
_CLOCK = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

_SENTENCE_CONTINUES = re.compile(r"\s+[А-ЯЁA-Z]")


def _expand_abbreviations(text: str, rules) -> str:
    """Replace abbreviations, keeping a sentence-final dot as a real full stop."""
    for pattern, replacement, flags, may_end_sentence in rules:
        source = text

        def repl(match: re.Match[str], replacement: str = replacement,
                 may_end_sentence: bool = may_end_sentence) -> str:
            if (
                may_end_sentence
                and match.group(0).rstrip().endswith(".")
                and _SENTENCE_CONTINUES.match(source, match.end())
            ):
                return replacement + "."
            return replacement

        text = re.sub(pattern, repl, source, flags=flags)
    return text


def _expand_numbers(text: str, language: str) -> str:
    """Spell out every number in `text` (currency signs move behind them)."""
    spec = get_language(language)
    english = spec.normalizer == "en"

    def repl(match: re.Match[str]) -> str:
        integer, fraction = match.group(1), match.group(2)
        if fraction:
            if english:
                return _en_decimal_to_words(integer.lstrip("-"), fraction)
            spelled = _ru_decimal_to_words(integer.lstrip("-"), fraction)
            return ("минус " + spelled) if integer.startswith("-") else spelled
        value = int(integer)
        return en_number_to_words(value) if english else ru_number_to_words(value)

    text = _CURRENCY_PREFIX.sub(r"\2 \1", text)
    text = _GROUPED_NUMBER.sub("", text)
    spell = en_number_to_words if english else ru_number_to_words
    text = _CLOCK.sub(lambda m: f"{spell(int(m.group(1)))} {spell(int(m.group(2)))}", text)
    # Version numbers and clock times keep their structure: read them group wise.
    text = _MULTI_DOT_NUMBER.sub(
        lambda m: " ".join(
            en_number_to_words(int(part)) if english else ru_number_to_words(int(part))
            for part in re.split(r"[.:]", m.group(0))
        ),
        text,
    )
    return _NUMBER.sub(repl, text)


def _apply_symbols(text: str, mapping: dict[str, str]) -> str:
    for symbol, replacement in mapping.items():
        text = text.replace(symbol, replacement)
    return text


def _normalize_ru(text: str) -> str:
    text = _expand_ru_years(text)
    text = _expand_abbreviations(text, _RU_ABBREVIATIONS)
    text = _expand_numbers(text, "ru")
    text = _apply_symbols(text, _RU_SYMBOLS)
    return text


def _normalize_en(text: str) -> str:
    text = _expand_abbreviations(text, _EN_ABBREVIATIONS)
    text = _expand_numbers(text, "en")
    text = _apply_symbols(text, _EN_SYMBOLS)
    return text


def _normalize_generic(text: str) -> str:
    return text


_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "ru": _normalize_ru,
    "en": _normalize_en,
    "generic": _normalize_generic,
}


def register_normalizer(name: str, fn: Callable[[str], str]) -> None:
    """Plug in a normalizer for a custom language."""
    _NORMALIZERS[name] = fn


def normalize_text(text: str, language: str | LanguageSpec = "ru", *, expand: bool = True) -> str:
    """Normalize `text` for synthesis in `language`.

    With ``expand=False`` only the language independent cleanup runs, which is
    handy when the caller already prepared the text.
    """
    spec = language if isinstance(language, LanguageSpec) else get_language(language)
    text = clean_whitespace(text)
    if expand:
        normalizer = _NORMALIZERS.get(spec.normalizer, _normalize_generic)
        text = normalizer(text)
    return clean_whitespace(text)
