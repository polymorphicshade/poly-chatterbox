import pytest

from poly_chatterbox.text import parse_markup, split_text


def test_pause_markup():
    segs = parse_markup("Hi [pause] there [pause 800ms] you [PAUSE 1.5s] ok [pause 2]")
    assert [s.pause for s in segs if s.is_pause] == [0.5, 0.8, 1.5, 2.0]
    assert [s.text for s in segs if s.text] == ["Hi", "there", "you", "ok"]


def test_other_tags_pass_through():
    assert [s.text for s in parse_markup("Oh [laugh] really? [clear throat]")] == ["Oh [laugh] really? [clear throat]"]


def test_only_markup_is_rejected():
    with pytest.raises(ValueError):
        parse_markup("[pause 1s]")


def test_split_prefers_sentences_and_keeps_punctuation():
    text = 'He said "Stop!" Then he left. ' * 5
    chunks = split_text(text, 70)
    assert all(len(c) <= 70 for c in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_split_cjk_and_long_clauses():
    zh = "今天天气很好。我们去公园吧！" * 10
    assert all(len(c) <= 30 for c in split_text(zh, 30))
    words = "word " * 100
    assert all(len(c) <= 40 for c in split_text(words, 40))
