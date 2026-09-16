"""Splitting text into synthesis-sized chunks.

Neural TTS models have an effective context window (XTTS starts to drift and
repeat past ~250 characters), so long input is cut into sentences and greedily
regrouped into chunks. Splitting on sentence boundaries keeps prosody natural;
each chunk is synthesized separately and the waveforms are joined with a short
pause.
"""

from __future__ import annotations

import re

# Abbreviations whose dot must not end a sentence.
_ABBREVIATIONS = {
    "т", "тт", "г", "гг", "ул", "д", "стр", "рис", "табл", "им", "проф", "акад",
    "руб", "коп", "мин", "сек", "см", "мм", "км", "кг",
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "fig", "no", "vol", "etc",
    "e.g", "i.e", "approx", "inc", "ltd",
}

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_INITIAL = re.compile(r"\b[A-ZА-ЯЁ]\.$")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:—-])\s+")
_TRAILING_ABBREV = re.compile(r"(?:^|\s)([^\s]+)\.$")


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _TRAILING_ABBREV.search(fragment)
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    return word in _ABBREVIATIONS or bool(_INITIAL.search(fragment))


def split_into_sentences(text: str) -> list[str]:
    """Split `text` into sentences, keeping terminal punctuation."""
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    buffer = ""
    for fragment in _SENTENCE_END.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        buffer = f"{buffer} {fragment}".strip() if buffer else fragment
        # A dot after "т." or an initial is not a sentence boundary.
        if _ends_with_abbreviation(buffer):
            continue
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence at clause boundaries, then at spaces."""
    pieces = [p for p in _CLAUSE_BREAK.split(sentence) if p.strip()]
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            out.append(current)
        if len(piece) <= max_chars:
            current = piece
            continue
        # Still too long: fall back to word-level packing.
        current = ""
        for word in piece.split():
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    out.append(current)
                current = word if len(word) <= max_chars else word[:max_chars]
    if current:
        out.append(current)
    return out


def split_into_chunks(text: str, max_chars: int = 220, *, min_chars: int = 0) -> list[str]:
    """Split `text` into chunks of at most `max_chars` characters.

    Sentences are never split unless a single one exceeds the limit. Chunks
    shorter than `min_chars` are merged with the next one when that keeps the
    result under the limit, which avoids one-word chunks that sound clipped.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[str] = []
    current = ""
    for sentence in split_into_sentences(text):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    if min_chars:
        merged: list[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) < min_chars and len(merged[-1]) + len(chunk) + 1 <= max_chars:
                merged[-1] = f"{merged[-1]} {chunk}"
            else:
                merged.append(chunk)
        chunks = merged
    return [c for c in chunks if c.strip()]
