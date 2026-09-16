#!/usr/bin/env python3
"""Собрать из блоков .md обычный текст для суфлёра и файл проверочных фраз.

    python3 recording-script/make_plaintext.py

Markdown хорош для чтения с экрана, но телепромптеры и `vctts speak -f` ждут
plain text без разметки — поэтому файлы .txt генерируются, а не пишутся руками.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).parent
SKIP_PREFIXES = ("#", ">", "|")
WORDS_PER_MINUTE = 140


def strip_markdown(text: str) -> str:
    """Оставить только то, что нужно читать вслух.

    Убираются заголовки, курсивные ремарки (в том числе многострочные),
    блоки кода целиком и разметка списков.
    """
    lines: list[str] = []
    in_code = False
    in_note = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith(SKIP_PREFIXES):
            continue
        if in_note:
            in_note = not line.endswith("*")
            continue
        if line.startswith("*") and not line.startswith("* "):
            # Курсивная ремарка: может занимать несколько строк.
            in_note = not line.endswith("*") or line == "*"
            continue
        line = re.sub(r"^[-*]\s+", "", line)        # маркированные списки
        line = re.sub(r"^\d+\.\s*", "", line)      # нумерованные списки
        line = re.sub(r"[*_`]", "", line)            # выделения
        lines.append(line)
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len([word for word in text.split() if any(ch.isalpha() for ch in word)])


def main() -> None:
    blocks = sorted(HERE.glob("[0-9][0-9]-*.md"))
    parts: list[str] = []
    print(f"{'блок':26} {'слов':>6} {'минут':>7}")
    total = 0
    for path in blocks:
        body = strip_markdown(path.read_text(encoding="utf-8"))
        words = word_count(body)
        total += words
        print(f"{path.name:26} {words:6} {words / WORDS_PER_MINUTE:7.1f}")
        parts.append(f"=== {path.stem} ===\n\n{body}\n")

    (HERE / "vsyo-podryad.txt").write_text("\n".join(parts), encoding="utf-8")
    print(f"{'ИТОГО':26} {total:6} {total / WORDS_PER_MINUTE:7.1f}")
    print(f"\nчитаемого текста: ~{total / WORDS_PER_MINUTE:.0f}-{total / 115:.0f} мин "
          f"(+ свободная речь из блока 11)")

    phrases = strip_markdown((HERE / "test-frazy.md").read_text(encoding="utf-8"))
    (HERE / "test-frazy.txt").write_text(phrases + "\n", encoding="utf-8")
    print(f"собрано: vsyo-podryad.txt, test-frazy.txt")


if __name__ == "__main__":
    main()
