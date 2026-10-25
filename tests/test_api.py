import io
import json
from urllib.parse import unquote

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from poly_chatterbox.api import create_app

from .conftest import make_wav


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def meta(res):
    return json.loads(unquote(res.headers["X-TTS-Metadata"]))


def assert_wav(res):
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(res.content))
    assert sr == 24000 and len(data) > 0


def ref(seconds=6):
    return {"ref_audio": ("ref.wav", make_wav(seconds), "audio/wav")}


def test_info_and_voices(client):
    info = client.get("/api/info").json()
    assert info["languages"]["fr"] == "French" and "[laugh]" in info["event_tags"]
    assert info["voice_counts"] == {"builtin": 1, "sample": 24, "cloned": 0}
    assert set(info["models"]) == {"original", "turbo", "nano", "multilingual"}
    voices = client.get("/api/voices").json()
    assert len(voices) == 25 and "url" not in voices[0]
    assert [v["id"] for v in client.get("/api/voices", params={"language": "German"}).json()] == ["sample_de"]
    assert client.get("/api/voices/podcast").json()["kind"] == "sample"
    assert client.get("/api/voices/nope").status_code == 404


@pytest.mark.parametrize("path", ["/api/tts/original", "/api/tts/turbo", "/api/tts/nano"])
def test_english_models(client, path):
    assert_wav(client.post(path, data={"text": "Hello [pause 300ms] there"}))
    res = client.post(path, data={"text": "Cloned", "generation": json.dumps({"seed": 3, "temperature": 0.7})}, files=ref())
    assert_wav(res)
    assert meta(res)["voice"] == "upload" and meta(res)["mode"] == path.rsplit("/", 1)[1]
    assert_wav(client.post(path, data={"text": "Hi", "voice_id": "podcast"}))


def test_multilingual(client):
    res = client.post("/api/tts/multilingual", data={"text": "Bonjour", "language": "fr"})
    assert_wav(res)
    assert meta(res)["language"] == "fr" and meta(res)["voice"] == "sample_fr"
    res = client.post("/api/tts/multilingual", data={"text": "Hola", "language": "Spanish", "version": "v2"}, files=ref())
    assert_wav(res)
    assert meta(res)["model"] == "Chatterbox Multilingual v2"
    assert client.post("/api/tts/multilingual", data={"text": "x", "language": "xx"}).status_code == 400


def test_voice_conversion(client):
    src = {"source_audio": ("s.wav", make_wav(3), "audio/wav")}
    assert_wav(client.post("/api/vc", files=src))
    res = client.post("/api/vc", files={**src, "target_audio": ("t.wav", make_wav(6), "audio/wav")})
    assert_wav(res)
    assert meta(res)["voice"] == "upload"
    assert_wav(client.post("/api/vc", data={"target_voice_id": "sample_ja"}, files=src))
    assert client.post("/api/vc").status_code == 422


def test_validation_errors(client):
    short = client.post("/api/tts/turbo", data={"text": "x"}, files=ref(2))
    assert short.status_code == 400 and "at least" in short.json()["detail"]
    junk = client.post("/api/tts/turbo", data={"text": "x"}, files={"ref_audio": ("r.wav", b"not audio", "audio/wav")})
    assert junk.status_code == 400
    bad_gen = client.post("/api/tts/original", data={"text": "x", "generation": "{\"top_p\": 5}"})
    assert bad_gen.status_code == 422
    unknown_gen = client.post("/api/tts/original", data={"text": "x", "generation": "{\"speed\": 2}"})
    assert unknown_gen.status_code == 422
    assert client.post("/api/tts/original", data={"text": "x", "voice_id": "ghost"}).status_code == 404


def test_save_use_delete_voice(client):
    res = client.post("/api/voices", data={"name": "Studio Me", "language": "English"}, files=ref())
    assert res.status_code == 201
    vid = res.json()["id"]
    assert vid in {v["id"] for v in client.get("/api/voices", params={"kind": "cloned"}).json()}
    assert_wav(client.post("/api/tts/turbo", data={"text": "Mine", "voice_id": vid}))
    assert client.delete(f"/api/voices/{vid}").status_code == 204
    assert client.delete(f"/api/voices/{vid}").status_code == 404
    assert client.delete("/api/voices/default").status_code == 400


def test_watermark(client):
    res = client.post("/api/watermark", files={"audio": ("a.wav", make_wav(2), "audio/wav")})
    assert res.json() == {"score": 0.0, "watermarked": False}


def test_web_ui_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "workspace-template" in res.text and "Poly Chatterbox" in res.text
    assert client.get("/static/app.js").status_code == 200


def test_generation_option_descriptions(client):
    from dataclasses import fields

    from poly_chatterbox.backends import GenerationParams

    opts = client.get("/api/info").json()["generation_options"]
    assert list(opts) == [f.name for f in fields(GenerationParams)]
    for opt in opts.values():
        assert opt["label"] and len(opt["description"]) > 40 and opt["defaults"]
    schema = client.get("/openapi.json").json()["components"]["schemas"]["GenerationOptions"]["properties"]
    assert "Defaults: original 1.0, multilingual 1.0, turbo 0.95, nano 0.95." in schema["top_p"]["description"]


def test_markup_spans_via_api(client):
    res = client.post("/api/tts/multilingual", data={
        "text": "Bonjour [settings cfg_weight=0 temperature=0.6][voice: sample_de]Guten Tag[/voice][/settings]",
        "language": "fr"})
    assert_wav(res)
    assert meta(res)["voices"] == ["sample_fr", "sample_de"]
    bad = client.post("/api/tts/original", data={"text": "[settings exaggeration=9]x[/settings]"})
    assert bad.status_code == 400 and "out of range" in bad.json()["detail"]
    assert client.post("/api/tts/turbo", data={"text": "[voice: ghost]x[/voice]"}).status_code == 404


def test_web_ui_has_wrap_buttons(client):
    js = client.get("/static/app.js").text
    assert "wrap-settings" in js and "wrap-voice" in js
