"""High-level synthesis engine. Implements the core features on top of a :class:`Backend`.

The web API and the CLI both call into this module, so behaviour is identical everywhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import audio as A
from .audio import Audio
from .backends import (MULTILINGUAL, MULTILINGUAL_VERSIONS, NANO, ORIGINAL, TURBO, Backend, BackendError,
                       GenerationParams, create_backend)
from .config import Settings, get_settings
from .languages import LANGUAGES, SAMPLES, normalize_language
from .library import DEFAULT_VOICE, Voice, VoiceLibrary
from .text import Segment, parse_markup

MODEL_LABELS = {ORIGINAL: "Chatterbox", TURBO: "Chatterbox-Turbo", NANO: "Chatterbox-Nano", MULTILINGUAL: "Chatterbox Multilingual"}


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    mode: str
    language: Optional[str] = None
    voice: Optional[str] = None  # library voice id, "upload" or "default"
    segments: int = 1
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)  # mode-specific metadata

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.sample_rate)

    def wav_bytes(self) -> bytes:
        return A.to_wav_bytes(self.audio, self.sample_rate)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "language": self.language, "voice": self.voice, "sample_rate": self.sample_rate,
            "duration": round(self.duration, 3), "segments": self.segments, "elapsed": round(self.elapsed, 3),
            "warnings": self.warnings, **self.details,
        }


class Engine:
    def __init__(self, settings: Optional[Settings] = None, backend: Optional[Backend] = None):
        self.settings = settings or get_settings()
        self.backend = backend or create_backend(self.settings)
        self.library = VoiceLibrary(self.settings.voices_dir)
        self._sample_lock = threading.Lock()

    # ================================================================ core features

    def original(self, text: str, ref_audio: Optional[Audio] = None, voice_id: Optional[str] = None,
                 params: Optional[GenerationParams] = None) -> SynthesisResult:
        """Chatterbox (English, 500M): zero-shot cloning with exaggeration and CFG/pace control."""
        return self._tts(ORIGINAL, text, ref_audio, voice_id, params or GenerationParams())

    def turbo(self, text: str, ref_audio: Optional[Audio] = None, voice_id: Optional[str] = None,
              params: Optional[GenerationParams] = None, nano: bool = False) -> SynthesisResult:
        """Chatterbox-Turbo (350M) or -Nano (110M): fast English TTS with paralinguistic tags like [laugh]."""
        return self._tts(NANO if nano else TURBO, text, ref_audio, voice_id, params or GenerationParams())

    def multilingual(self, text: str, language: str, ref_audio: Optional[Audio] = None,
                     voice_id: Optional[str] = None, version: Optional[str] = None,
                     params: Optional[GenerationParams] = None) -> SynthesisResult:
        """Chatterbox Multilingual (23 languages). Without a voice, the language's sample speaker is used."""
        lang = normalize_language(language)
        version = (version or self.settings.multilingual_version).strip().lower()
        if version not in MULTILINGUAL_VERSIONS:
            raise ValueError(f"Unknown multilingual version {version!r}: use {' or '.join(MULTILINGUAL_VERSIONS)}.")
        if ref_audio is None and not voice_id and lang in SAMPLES:
            voice_id = f"sample_{lang}"
        p = params or GenerationParams()
        result = self._tts(MULTILINGUAL, text, ref_audio, voice_id, p, language=lang, version=version)
        voice = self.library.get(voice_id) if voice_id else None
        cfg = p.cfg_weight if p.cfg_weight is not None else 0.5
        if voice and voice.language and voice.language != lang and cfg > 0:
            result.warnings.append(
                f"The reference speaks {LANGUAGES.get(voice.language, voice.language)} but the text is "
                f"{LANGUAGES[lang]}, so the output may carry its accent. Set CFG / pace to 0 to avoid that.")
        return result

    def convert(self, source: Audio, target_audio: Optional[Audio] = None,
                target_voice_id: Optional[str] = None) -> SynthesisResult:
        """Voice conversion: re-voice existing speech with a target voice's timbre."""
        if target_audio is not None and target_voice_id:
            raise ValueError("Give at most one of target_audio or target_voice_id.")
        wav, sr = source
        if not wav.size or float(np.max(np.abs(wav))) < 1e-3:
            raise ValueError("The source audio is silent: there is no speech to convert.")
        warnings: list[str] = []
        target, label = self._voice(target_audio, target_voice_id, warnings)
        t0 = time.perf_counter()
        out, out_sr = self.backend.convert(source, target)
        return SynthesisResult(audio=out, sample_rate=out_sr, mode="vc", voice=label, elapsed=time.perf_counter() - t0,
                               warnings=warnings, details={"model": "Chatterbox VC", "source_duration": round(len(wav) / sr, 3)})

    # ================================================================ library and tools

    def save_voice(self, name: str, ref_audio: Audio, language: Optional[str] = None,
                   description: Optional[str] = None) -> Voice:
        lang = normalize_language(language, default="") if language and language.lower() != "auto" else ""
        return self.library.add(name, self._prepare_reference(ref_audio, []), lang, description)

    def remove_voice(self, voice_id: str) -> None:
        self.library.remove(voice_id)

    def detect_watermark(self, audio: Audio) -> dict[str, Any]:
        """Look for the Perth watermark that Chatterbox embeds in everything it generates."""
        score = self.backend.detect_watermark(audio)
        return {"score": round(score, 3), "watermarked": score >= 0.5}

    def info(self) -> dict[str, Any]:
        return self.backend.status()

    # ================================================================ internals

    def _tts(self, model: str, text: str, ref_audio: Optional[Audio], voice_id: Optional[str],
             params: GenerationParams, language: Optional[str] = None, version: Optional[str] = None) -> SynthesisResult:
        if not text or not text.strip():
            raise ValueError("Text is required.")
        if ref_audio is not None and voice_id:
            raise ValueError("Give at most one of ref_audio or voice_id.")
        segments = parse_markup(text, self.settings.max_chunk_chars)
        warnings: list[str] = []
        if ignored := params.unsupported(model):
            warnings.append(f"{MODEL_LABELS[model]} ignores {', '.join(ignored)}.")
        voice, label = self._voice(ref_audio, voice_id, warnings)
        t0 = time.perf_counter()
        wav, sr = self._render(segments, lambda texts: self.backend.synthesize(model, texts, voice, params, language, version))
        details = {"model": MODEL_LABELS[model] + (f" {version}" if version else "")}
        return SynthesisResult(audio=wav, sample_rate=sr, mode=model, language=language, voice=label,
                               segments=sum(1 for s in segments if s.text), elapsed=time.perf_counter() - t0,
                               warnings=warnings, details=details)

    @staticmethod
    def _render(segments: list[Segment], call: Callable[[list[str]], tuple[list[np.ndarray], int]]) -> tuple[np.ndarray, int]:
        """Synthesize the speech chunks, then interleave the requested pauses."""
        wavs, sr = call([s.text for s in segments if s.text])
        it = iter(wavs)
        pieces = [A.silence(s.pause, sr) if s.is_pause else next(it) for s in segments]
        return A.concat(pieces, sr), sr

    def _voice(self, ref_audio: Optional[Audio], voice_id: Optional[str], warnings: list[str]) -> tuple[Optional[Audio], str]:
        """Reference clip for a request (None: the model's built-in voice) and a label for the metadata."""
        if ref_audio is not None:
            return self._prepare_reference(ref_audio, warnings), "upload"
        if not voice_id or voice_id.lower() == DEFAULT_VOICE:
            return None, DEFAULT_VOICE
        voice = self.library.get(voice_id)
        if voice.kind == "sample":
            return self._sample_reference(voice), voice.id
        return self.library.reference(voice.id), voice.id

    def _prepare_reference(self, ref_audio: Audio, warnings: list[str]) -> Audio:
        wav, sr = ref_audio
        wav = A.trim_silence(wav, sr)
        seconds = len(wav) / sr
        if seconds < self.settings.min_ref_seconds:
            raise ValueError(
                f"Reference audio has {seconds:.1f}s of speech; at least {self.settings.min_ref_seconds:g}s is required "
                "(about 10 s of clean speech works best)."
            )
        if seconds > self.settings.max_ref_seconds:
            wav = wav[: int(self.settings.max_ref_seconds * sr)]
            warnings.append(f"Reference audio was trimmed to {self.settings.max_ref_seconds:g}s.")
        return A.peak_normalize(wav), sr

    def _sample_reference(self, voice: Voice) -> Audio:
        """A demo voice's reference clip, downloaded once and cached in the samples folder."""
        if self.backend.name == "mock":  # no network in mock mode: a distinct placeholder clip per voice
            from .backends.mock import SAMPLE_RATE, _tone

            return _tone(8.0, voice.id, None), SAMPLE_RATE
        path = self.settings.samples_dir / (voice.id + "." + voice.url.rsplit(".", 1)[-1])
        with self._sample_lock:
            if not path.is_file():
                import httpx

                try:
                    res = httpx.get(voice.url, timeout=60.0, follow_redirects=True)
                    res.raise_for_status()
                except httpx.HTTPError as e:
                    raise BackendError(f"Could not download the {voice.name} voice from {voice.url}: {e}") from e
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(res.content)
        return A.load_audio(path)


_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine()
        return _engine


def set_engine(engine: Optional[Engine]) -> None:
    global _engine
    with _engine_lock:
        _engine = engine
