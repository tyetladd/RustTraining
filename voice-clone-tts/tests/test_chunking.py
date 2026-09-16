from voice_clone_tts.text.chunking import split_into_chunks, split_into_sentences


def test_sentences_are_split_on_terminal_punctuation():
    assert split_into_sentences("Привет! Как дела? Хорошо.") == [
        "Привет!", "Как дела?", "Хорошо.",
    ]


def test_abbreviations_do_not_end_a_sentence():
    assert split_into_sentences("Это т. д. и прочее. Конец.") == [
        "Это т. д. и прочее.", "Конец.",
    ]


def test_initials_do_not_end_a_sentence():
    assert split_into_sentences("Пушкин А. С. написал это. Да.")[0].startswith("Пушкин А. С.")


def test_chunks_respect_the_limit():
    text = " ".join(f"Предложение номер {i}." for i in range(40))
    chunks = split_into_chunks(text, max_chars=80)
    assert chunks and all(len(chunk) <= 80 for chunk in chunks)
    assert "".join(chunks).count("Предложение") == 40


def test_long_sentence_is_split_at_clause_boundaries():
    sentence = "первая часть, вторая часть, третья часть, четвёртая часть, пятая часть"
    chunks = split_into_chunks(sentence, max_chars=30)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert len(chunks) > 1


def test_single_huge_word_is_cut():
    chunks = split_into_chunks("а" * 250, max_chars=100)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_stress_marks_survive_chunking():
    chunks = split_into_chunks("Мен+я зов+ут Л+ёва. Я г+отов.", max_chars=100)
    assert "+" in " ".join(chunks)


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("   ") == []
