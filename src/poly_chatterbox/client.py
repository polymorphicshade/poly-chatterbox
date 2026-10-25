"""Clients used by the CLI: run in-process (``LocalClient``) or against a server (``HttpClient``).

Both expose the same methods and return ``(wav_bytes, metadata)`` for synthesis calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

from .audio import load_audio

Result = tuple[bytes, dict[str, Any]]


def _load(path: Optional[Path]):
    return load_audio(path) if path else None


class LocalClient:
    def __init__(self, engine=None):
        from .engine import get_engine

        self.engine = engine or get_engine()

    @staticmethod
    def _out(result) -> Result:
        return result.wav_bytes(), result.metadata()

    @staticmethod
    def _params(gen: Optional[dict]):
        from .backends import GenerationParams

        return GenerationParams(**(gen or {}))

    def original(self, text, ref_audio: Optional[Path] = None, voice_id=None, generation=None) -> Result:
        return self._out(self.engine.original(text, _load(ref_audio), voice_id, self._params(generation)))

    def turbo(self, text, ref_audio: Optional[Path] = None, voice_id=None, generation=None, nano=False) -> Result:
        return self._out(self.engine.turbo(text, _load(ref_audio), voice_id, self._params(generation), nano))

    def multilingual(self, text, language, ref_audio: Optional[Path] = None, voice_id=None, version=None,
                     generation=None) -> Result:
        return self._out(self.engine.multilingual(text, language, _load(ref_audio), voice_id, version, self._params(generation)))

    def convert(self, source_audio: Path, target_audio: Optional[Path] = None, target_voice_id=None) -> Result:
        return self._out(self.engine.convert(load_audio(source_audio), _load(target_audio), target_voice_id))

    def convert_clip(self, audio, target_audio: Optional[Path] = None, target_voice_id=None):
        """Voice-convert an in-memory clip; returns ``(wav, sample_rate)``. Used by live streaming."""
        return self._audio_of(self.engine.convert(audio, _load(target_audio), target_voice_id))

    @staticmethod
    def _audio_of(result):
        return result.audio, result.sample_rate

    def voices(self) -> list[dict]:
        return [v.to_dict() for v in self.engine.library.list()]

    def save_voice(self, name, ref_audio: Path, language=None, description=None) -> dict:
        return self.engine.save_voice(name, load_audio(ref_audio), language, description).to_dict()

    def remove_voice(self, voice_id) -> None:
        self.engine.remove_voice(voice_id)

    def prompts(self) -> list[dict]:
        return [p.to_dict() for p in self.engine.prompts.list()]

    def prompt(self, prompt_id) -> dict:
        return self.engine.prompts.get(prompt_id).to_dict()

    def save_prompt(self, name, text, mode=None, language=None, version=None, voice_id=None, generation=None) -> dict:
        return self.engine.save_prompt(name, text, mode, language, version, voice_id, generation)[0].to_dict()

    def remove_prompt(self, prompt_id) -> None:
        self.engine.prompts.remove(prompt_id)

    def watermark(self, audio: Path) -> dict:
        return self.engine.detect_watermark(load_audio(audio))

    def info(self) -> dict:
        from .backends import GENERATION_OPTIONS
        from .engine import MODEL_LABELS
        from .languages import LANGUAGES

        return {"backend": self.engine.info(), "models": MODEL_LABELS, "languages": LANGUAGES,
                "multilingual_version": self.engine.settings.multilingual_version,
                "generation_options": GENERATION_OPTIONS}


class ApiError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 900.0, verify: bool = True):
        import httpx

        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, verify=verify)

    def _check(self, res):
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail")
            except Exception:
                detail = res.text
            raise ApiError(f"HTTP {res.status_code}: {detail}")
        return res

    def _audio(self, res) -> Result:
        self._check(res)
        meta = json.loads(unquote(res.headers.get("X-TTS-Metadata", "{}")))
        return res.content, meta

    @staticmethod
    def _files(**paths: Optional[Path]) -> Optional[dict]:
        files = {name: (Path(p).name, Path(p).read_bytes(), "application/octet-stream") for name, p in paths.items() if p}
        return files or None

    def _post_form(self, path, data: dict, **files: Optional[Path]) -> Result:
        data = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in data.items() if v is not None}
        return self._audio(self.http.post(path, data=data, files=self._files(**files)))

    def original(self, text, ref_audio: Optional[Path] = None, voice_id=None, generation=None) -> Result:
        return self._post_form("/api/tts/original", dict(text=text, voice_id=voice_id, generation=generation), ref_audio=ref_audio)

    def turbo(self, text, ref_audio: Optional[Path] = None, voice_id=None, generation=None, nano=False) -> Result:
        return self._post_form("/api/tts/nano" if nano else "/api/tts/turbo",
                               dict(text=text, voice_id=voice_id, generation=generation), ref_audio=ref_audio)

    def multilingual(self, text, language, ref_audio: Optional[Path] = None, voice_id=None, version=None,
                     generation=None) -> Result:
        return self._post_form("/api/tts/multilingual", dict(text=text, language=language, voice_id=voice_id,
                                                              version=version, generation=generation), ref_audio=ref_audio)

    def convert(self, source_audio: Path, target_audio: Optional[Path] = None, target_voice_id=None) -> Result:
        return self._post_form("/api/vc", dict(target_voice_id=target_voice_id),
                               source_audio=source_audio, target_audio=target_audio)

    def convert_clip(self, audio, target_audio: Optional[Path] = None, target_voice_id=None):
        """Voice-convert an in-memory clip on the server; returns ``(wav, sample_rate)``."""
        from .audio import to_wav_bytes

        files = {"source_audio": ("phrase.wav", to_wav_bytes(*audio), "audio/wav")}
        if target_audio:
            files["target_audio"] = (Path(target_audio).name, Path(target_audio).read_bytes(), "application/octet-stream")
        data = {"target_voice_id": target_voice_id} if target_voice_id else None
        content, _ = self._audio(self.http.post("/api/vc", data=data, files=files))
        return load_audio(content)

    def voices(self) -> list[dict]:
        return self._check(self.http.get("/api/voices")).json()

    def save_voice(self, name, ref_audio: Path, language=None, description=None) -> dict:
        data = {k: v for k, v in dict(name=name, language=language, description=description).items() if v is not None}
        return self._check(self.http.post("/api/voices", data=data, files=self._files(ref_audio=ref_audio))).json()

    def remove_voice(self, voice_id) -> None:
        self._check(self.http.delete(f"/api/voices/{voice_id}"))

    def prompts(self) -> list[dict]:
        return self._check(self.http.get("/api/prompts")).json()

    def prompt(self, prompt_id) -> dict:
        return self._check(self.http.get(f"/api/prompts/{prompt_id}")).json()

    def save_prompt(self, name, text, mode=None, language=None, version=None, voice_id=None, generation=None) -> dict:
        body = {k: v for k, v in dict(name=name, text=text, mode=mode, language=language, version=version,
                                      voice_id=voice_id, generation=generation).items() if v is not None}
        return self._check(self.http.post("/api/prompts", json=body)).json()

    def remove_prompt(self, prompt_id) -> None:
        self._check(self.http.delete(f"/api/prompts/{prompt_id}"))

    def watermark(self, audio: Path) -> dict:
        return self._check(self.http.post("/api/watermark", files=self._files(audio=audio))).json()

    def info(self) -> dict:
        return self._check(self.http.get("/api/info")).json()
