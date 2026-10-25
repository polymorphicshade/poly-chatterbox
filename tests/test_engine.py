import numpy as np
import pytest

from poly_chatterbox.audio import load_audio
from poly_chatterbox.backends import GenerationParams

from .conftest import make_wav


def last_call(engine, name):
    return [c for c in engine.backend.calls if c[0] == name][-1][1]


def test_original_builtin_voice(engine):
    r = engine.original("Hello there")
    assert r.sample_rate == 24000 and r.duration > 0
    call = last_call(engine, "synthesize")
    assert call["model"] == "original" and not call["voice"]
    assert r.voice == "default" and r.mode == "original"


def test_turbo_and_nano_pick_models(engine):
    engine.turbo("Hi [laugh] there")
    assert last_call(engine, "synthesize")["model"] == "turbo"
    assert last_call(engine, "synthesize")["texts"] == ["Hi [laugh] there"]
    r = engine.turbo("Hi", nano=True)
    assert last_call(engine, "synthesize")["model"] == "nano" and r.details["model"] == "Chatterbox-Nano"


def test_unsupported_options_warn(engine):
    r = engine.turbo("Hi", params=GenerationParams(cfg_weight=0.3, exaggeration=0.7))
    assert any("ignores exaggeration, cfg_weight" in w for w in r.warnings)
    assert GenerationParams(cfg_weight=0.3, top_k=5).model_kwargs("original") == {"cfg_weight": 0.3}


def test_pauses_insert_silence(engine):
    plain = engine.original("one two")
    paused = engine.original("one [pause 1s] two")
    assert paused.duration > plain.duration + 0.5
    assert paused.segments == 2


def test_long_text_is_chunked(engine, settings):
    settings.max_chunk_chars = 60
    text = "This is a sentence that goes on for a while. " * 6
    r = engine.original(text)
    texts = last_call(engine, "synthesize")["texts"]
    assert r.segments == len(texts) > 1 and all(len(t) <= 60 for t in texts)


def test_clone_requires_five_seconds(engine):
    with pytest.raises(ValueError, match="at least 5"):
        engine.turbo("hi", load_audio(make_wav(2)))
    engine.turbo("hi", load_audio(make_wav(6)))
    assert last_call(engine, "synthesize")["voice"]


def test_multilingual_defaults_to_language_sample(engine):
    r = engine.multilingual("Bonjour", "French")
    assert r.language == "fr" and r.voice == "sample_fr"
    call = last_call(engine, "synthesize")
    assert call["language"] == "fr" and call["version"] == "v3" and call["voice"]
    r = engine.multilingual("Hola", "es", voice_id="default", version="v2")
    assert r.voice == "default" and last_call(engine, "synthesize")["version"] == "v2"


def test_multilingual_validation_and_accent_warning(engine):
    with pytest.raises(ValueError, match="Unsupported language"):
        engine.multilingual("x", "Klingon")
    with pytest.raises(ValueError, match="version"):
        engine.multilingual("x", "fr", version="v9")
    r = engine.multilingual("Bonjour", "fr", voice_id="sample_de")
    assert any("CFG / pace to 0" in w for w in r.warnings)
    r = engine.multilingual("Bonjour", "fr", voice_id="sample_de", params=GenerationParams(cfg_weight=0))
    assert not r.warnings


def test_voice_conversion(engine):
    src = load_audio(make_wav(3))
    r = engine.convert(src)
    assert r.mode == "vc" and r.voice == "default" and abs(r.duration - 3) < 0.01
    r = engine.convert(src, target_voice_id="podcast")
    assert r.voice == "podcast" and last_call(engine, "convert")["target"]
    with pytest.raises(ValueError, match="silent"):
        engine.convert((np.zeros(16000, dtype=np.float32), 16000))


def test_saved_voice_roundtrip(engine):
    voice = engine.save_voice("My Narrator", load_audio(make_wav(6)), "English")
    assert voice.id == "my_narrator" and voice.kind == "cloned" and voice.language == "en"
    r = engine.turbo("hi", voice_id="my_narrator")
    assert r.voice == "my_narrator" and last_call(engine, "synthesize")["voice"]
    engine.remove_voice("my_narrator")
    with pytest.raises(LookupError):
        engine.library.get("my_narrator")


def test_unknown_and_builtin_voices(engine):
    with pytest.raises(LookupError):
        engine.original("hi", voice_id="nobody")
    with pytest.raises(ValueError):
        engine.remove_voice("default")
    with pytest.raises(ValueError, match="at most one"):
        engine.original("hi", load_audio(make_wav(6)), "podcast")


def test_seed_is_deterministic(engine):
    a = engine.original("same", params=GenerationParams(seed=7))
    b = engine.original("same", params=GenerationParams(seed=7))
    assert np.array_equal(a.audio, b.audio)


def test_watermark(engine):
    assert engine.detect_watermark(load_audio(make_wav(2))) == {"score": 0.0, "watermarked": False}
