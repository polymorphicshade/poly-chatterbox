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


def test_settings_span_applies_only_inside():
    segs = parse_markup("Calm. [settings exaggeration=0.9 cfg_weight=0.3]Wow![/settings] Calm again.")
    assert [(s.text, dict(s.settings)) for s in segs] == [
        ("Calm.", {}), ("Wow!", {"exaggeration": 0.9, "cfg_weight": 0.3}), ("Calm again.", {})]


def test_nested_settings_merge_and_voice_spans():
    segs = parse_markup("[settings temperature=0.5 top_k=20][voice: sample_de]a [settings temperature=0.9]b[/settings] c[/voice][/settings] d")
    assert [(s.text, dict(s.settings), s.voice) for s in segs] == [
        ("a", {"temperature": 0.5, "top_k": 20}, "sample_de"),
        ("b", {"temperature": 0.9, "top_k": 20}, "sample_de"),
        ("c", {"temperature": 0.5, "top_k": 20}, "sample_de"),
        ("d", {}, None)]


def test_settings_values_and_case():
    (seg,) = parse_markup("[SETTINGS Exaggeration=1, norm_loudness=off]hi[/Settings]")
    assert dict(seg.settings) == {"exaggeration": 1.0, "norm_loudness": False}


@pytest.mark.parametrize("markup, message", [
    ("[settings speed=2]hi[/settings]", "Unknown"),
    ("[settings seed=3]hi[/settings]", "seed"),
    ("[settings exaggeration=9]hi[/settings]", "out of range"),
    ("[settings exaggeration=loud]hi[/settings]", "number"),
    ("[settings norm_loudness=maybe]hi[/settings]", "true or false"),
    ("[settings exaggeration]hi[/settings]", "name=value"),
    ("[settings]hi[/settings]", "at least one"),
    ("[settings exaggeration=1]hi", "never closed"),
    ("[voice: podcast]hi", "never closed"),
    ("hi[/settings]", "no matching"),
    ("[settings exaggeration=1][/settings]", "only contains markup"),
])
def test_markup_errors(markup, message):
    with pytest.raises(ValueError, match=message):
        parse_markup(markup)
