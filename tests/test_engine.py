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


def synth_calls(engine):
    return [c[1] for c in engine.backend.calls if c[0] == "synthesize"]


def test_settings_span_changes_params_per_span(engine):
    r = engine.original("Calm. [settings exaggeration=0.9 cfg_weight=0.3]Wow![/settings] Calm again.",
                        params=GenerationParams(temperature=0.6, seed=4))
    calls = synth_calls(engine)
    assert len(calls) == 2 and r.segments == 3
    plain = next(c for c in calls if c["texts"] == ["Calm.", "Calm again."])
    span = next(c for c in calls if c["texts"] == ["Wow!"])
    assert plain["params"].exaggeration is None and plain["params"].temperature == 0.6
    assert (span["params"].exaggeration, span["params"].cfg_weight) == (0.9, 0.3)
    assert span["params"].temperature == 0.6 and span["params"].seed == 4  # request options carry over


def test_span_order_is_preserved(engine):
    plain = engine.original("Aaa bbb. Ccc ddd. Eee fff.", params=GenerationParams(seed=1))
    spanned = engine.original("Aaa bbb. [settings temperature=0.8]Ccc ddd.[/settings] Eee fff.", params=GenerationParams(seed=1))
    assert spanned.segments == 3 and abs(spanned.duration - plain.duration) < 0.2


def test_voice_span_switches_voice(engine):
    r = engine.multilingual("Bonjour. [voice: sample_de]Guten Tag.[/voice] Au revoir.", "fr", params=GenerationParams(cfg_weight=0))
    assert r.voice == "sample_fr" and r.details["voices"] == ["sample_fr", "sample_de"]
    assert [c["texts"] for c in synth_calls(engine)] == [["Bonjour.", "Au revoir."], ["Guten Tag."]]
    assert not r.warnings
    r = engine.turbo("Hi [voice: default]there[/voice]", voice_id="podcast")
    assert r.details["voices"] == ["podcast", "default"] and not synth_calls(engine)[-1]["voice"]


def test_span_voice_unknown_fails_before_generating(engine):
    with pytest.raises(LookupError):
        engine.turbo("Hi [voice: ghost]there[/voice]")
    assert not synth_calls(engine)


def test_span_voice_accent_warning(engine):
    r = engine.multilingual("Bonjour [voice: sample_de]Guten Tag[/voice]", "fr")
    assert sum("accent" in w for w in r.warnings) == 1
    r = engine.multilingual("Bonjour [voice: sample_de][settings cfg_weight=0]Guten Tag[/settings][/voice]", "fr")
    assert not r.warnings


def test_span_options_unsupported_by_model_warn_once(engine):
    r = engine.turbo("[settings exaggeration=0.9]a[/settings] b [settings cfg_weight=0.2 top_k=50]c[/settings]")
    assert r.warnings == ["Chatterbox-Turbo ignores exaggeration, cfg_weight."]


def test_transcribe(engine):
    out = engine.transcribe(load_audio(make_wav(3)), "German")
    assert out["language"] == "de" and out["duration"] == 3.0 and out["text"] == out["segments"][0]["text"]
    assert engine.transcribe(load_audio(make_wav(3)))["language"] is None
    assert engine.transcribe(load_audio(make_wav(3)), "Auto")["language"] is None
    assert engine.backend.calls[-1] == ("transcribe", {"language": None})
    with pytest.raises(ValueError, match="silent"):
        engine.transcribe((np.zeros(16000, dtype=np.float32), 16000))


def test_emotion_presets(engine):
    from poly_chatterbox.presets import EMOTION_PRESETS, apply_preset, preset_options
    from poly_chatterbox.backends import GENERATION_OPTIONS

    # Every preset is within the option ranges and sets something each model uses.
    for name, preset in EMOTION_PRESETS.items():
        for key, value in preset["options"].items():
            opt = GENERATION_OPTIONS[key]
            assert opt["min"] <= value <= opt["max"] if opt["max"] is not None else opt["min"] <= value, (name, key)
        for model in ("original", "turbo", "nano", "multilingual"):
            assert preset_options(name, model)

    # Per model: Turbo ignores exaggeration / CFG, so they are left out rather than warned about.
    assert "exaggeration" in preset_options("angry", "original")
    assert "exaggeration" not in preset_options("angry", "turbo") and "top_k" in preset_options("angry", "turbo")
    with pytest.raises(ValueError, match="Unknown emotion preset"):
        preset_options("giddy")

    # Explicit options win over the preset.
    merged = apply_preset("excited", GenerationParams(temperature=0.5, seed=3), "original")
    assert merged.temperature == 0.5 and merged.exaggeration == 0.9 and merged.seed == 3 and merged.top_k is None

    res = engine.original("Hello", preset="Sad", params=GenerationParams(cfg_weight=0.1))
    sent = last_call(engine, "synthesize")["params"]
    assert (sent.exaggeration, sent.cfg_weight) == (0.4, 0.1) and res.details["preset"] == "sad" and not res.warnings
    res = engine.turbo("Hi", preset="angry")
    assert last_call(engine, "synthesize")["params"].exaggeration is None and not res.warnings
    with pytest.raises(ValueError, match="Unknown emotion preset"):
        engine.multilingual("Hola", "es", preset="giddy")


def test_preset_in_settings_markup(engine):
    engine.original("Calm. [settings preset=angry temperature=0.5]Loud![/settings]")
    calls = [c[1]["params"] for c in engine.backend.calls if c[0] == "synthesize"]
    assert calls[0].exaggeration is None
    assert (calls[1].exaggeration, calls[1].temperature) == (1.1, 0.5)
    res = engine.turbo("[settings preset=angry]Loud![/settings]")
    assert not res.warnings and last_call(engine, "synthesize")["params"].exaggeration is None
    with pytest.raises(ValueError, match="Unknown emotion preset"):
        engine.original("[settings preset=giddy]x[/settings]")


def test_translate(engine):
    out = engine.translate("Hello [pause 200ms] friend", "English", "ja")
    assert out["text"] == "(ja) Hello [pause 200ms] (ja) friend" and (out["source"], out["target"]) == ("en", "ja")
    assert engine.translate("Same", "fr", "French")["text"] == "Same"
    with pytest.raises(ValueError, match="Unsupported language"):
        engine.translate("Hi", "en", "Klingon")
    with pytest.raises(ValueError, match="Text is required"):
        engine.translate("  ", "en", "fr")


def test_translate_before_speaking(engine):
    res = engine.multilingual("Good morning [pause 100ms] everyone", "Japanese", translate_from="English")
    assert res.details["translated_from"] == "en" and res.language == "ja"
    assert last_call(engine, "synthesize")["texts"] == ["(ja) Good morning", "(ja) everyone"]
    # English models translate into English.
    res = engine.turbo("Bonjour", translate_from="fr")
    assert last_call(engine, "synthesize")["texts"] == ["(en) Bonjour"] and res.details["translated_from"] == "fr"
    # Already in the spoken language: nothing to do.
    res = engine.multilingual("Hola", "es", translate_from="Spanish")
    assert "translated_from" not in res.details and last_call(engine, "synthesize")["texts"] == ["Hola"]
    res = engine.original("Hi", translate_from="none")
    assert "translated_from" not in res.details
    with pytest.raises(ValueError, match="Unsupported language"):
        engine.original("Hi", translate_from="Klingon")


def test_process_windows():
    import numpy as np

    from poly_chatterbox.audio import process_windows

    wav = np.sin(np.arange(16000 * 7) / 40.0).astype(np.float32)
    seen = []

    def identity(chunk):
        seen.append(len(chunk))
        return chunk

    out = process_windows(wav, 16000, identity, window_s=3.0, overlap_s=0.5)
    assert len(out) == len(wav) and np.allclose(out, wav, atol=1e-5)
    assert len(seen) == 3 and max(seen) == 48000
    # Short clips go through in one piece.
    seen.clear()
    process_windows(wav[:16000], 16000, identity, window_s=3.0)
    assert seen == [16000]


def test_denoise(engine):
    wav = load_audio(make_wav(3))
    res = engine.denoise(wav)
    assert res.mode == "denoise" and res.sample_rate == wav[1] and len(res.audio) == len(wav[0])
    # The mock halves the level; the engine restores the loudness of the input.
    assert np.max(np.abs(res.audio)) == pytest.approx(np.max(np.abs(wav[0])), rel=1e-3)
    assert engine.backend.calls[-1][0] == "denoise"
    with pytest.raises(ValueError, match="silent"):
        engine.denoise((np.zeros(16000, dtype=np.float32), 16000))


def test_save_voice_denoised(engine):
    engine.save_voice("Noisy", load_audio(make_wav(6)), denoise=True)
    assert engine.backend.calls[-1][0] == "denoise"
    engine.save_voice("Quiet", load_audio(make_wav(6)))
    assert engine.backend.calls[-1][0] == "denoise"  # not called again for the second voice
    assert len([c for c in engine.backend.calls if c[0] == "denoise"]) == 1
