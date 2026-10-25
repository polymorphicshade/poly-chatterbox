"""Real backend running the Chatterbox checkpoints through the vendored ``chatterbox`` package."""

from __future__ import annotations

import logging
import os
import random
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import numpy as np

from .. import audio as A
from ..audio import Audio
from ..config import Settings
from .base import MULTILINGUAL, NANO, ORIGINAL, TURBO, Backend, BackendError, GenerationParams

log = logging.getLogger(__name__)

VC = "vc"
WHISPER = "whisper"

TRANSLATOR = "translator"

# NLLB-200 language codes for the 23 Chatterbox Multilingual languages.
NLLB_CODES = {
    "ar": "arb_Arab", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "fi": "fin_Latn", "fr": "fra_Latn", "he": "heb_Hebr", "hi": "hin_Deva", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ms": "zsm_Latn", "nl": "nld_Latn", "no": "nob_Latn", "pl": "pol_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "sv": "swe_Latn", "sw": "swh_Latn", "tr": "tur_Latn", "zh": "zho_Hans",
}
MAX_CACHED_VOICES = 32


@contextmanager
def _wav_file(audio: Audio) -> Iterator[str]:
    """The upstream models read reference clips from disk, so hand them a temporary WAV."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        A.save_wav(path, *audio)
        yield path
    finally:
        os.unlink(path)


class ChatterboxBackend(Backend):
    name = "chatterbox"

    def __init__(self, settings: Settings):
        try:
            import torch  # noqa: F401

            import chatterbox  # noqa: F401
        except ImportError as e:
            raise BackendError(
                "The Chatterbox backend needs the 'chatterbox' extra: `uv sync --extra chatterbox` "
                "(or `pip install poly-chatterbox[chatterbox]`). Use --backend mock to try the app without it."
            ) from e
        self.settings = settings
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._builtin: dict[str, Any] = {}  # model key -> its built-in voice (conds / VC ref_dict)
        self._voices: OrderedDict[tuple, Any] = OrderedDict()  # (model key, clip fingerprint, ...) -> conds
        self._watermarker = None
        self._aux: dict[str, Any] = {}  # helper models (Whisper, ...) that aren't part of the TTS LRU
        # One lock for loading and generation: a single GPU runs one job at a time, and the upstream
        # models keep the current voice as mutable state.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ model management

    @property
    def device(self) -> str:
        import torch

        device = self.settings.device
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "mps" if torch.backends.mps.is_available() else "cpu"
        return device

    @staticmethod
    def _key(model: str, version: Optional[str]) -> str:
        return f"{model}:{version}" if model == MULTILINGUAL else model

    def _load(self, key: str):
        device = self.device
        if key == ORIGINAL:
            from chatterbox.tts import ChatterboxTTS

            return ChatterboxTTS.from_pretrained(device)
        if key in (TURBO, NANO):
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(device, nano=key == NANO)
        if key.startswith(MULTILINGUAL + ":"):
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            return ChatterboxMultilingualTTS.from_pretrained(device, t3_model=key.split(":", 1)[1])
        if key == VC:
            from chatterbox.vc import ChatterboxVC

            return ChatterboxVC.from_pretrained(device)
        raise BackendError(f"Unknown model {key!r}.")

    def model(self, key: str):
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]
            while len(self._models) >= max(1, self.settings.max_loaded_models):
                evicted, _ = self._models.popitem(last=False)
                self._builtin.pop(evicted, None)
                self._voices = OrderedDict((k, v) for k, v in self._voices.items() if k[0] != evicted)
                log.info("Unloading %s model", evicted)
                self._free_memory()
            log.info("Loading %s on %s", key, self.device)
            t0 = time.perf_counter()
            try:
                model = self._load(key)
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(f"Failed to load the {key} model: {e}") from e
            log.info("Loaded %s in %.1fs", key, time.perf_counter() - t0)
            self._models[key] = model
            self._builtin[key] = model.ref_dict if key == VC else model.conds
            return model

    def aux_model(self, key: str, load):
        """A helper model, loaded on first use and kept resident (they are small next to the TTS models)."""
        with self._lock:
            if key not in self._aux:
                log.info("Loading %s on %s", key, self.device)
                t0 = time.perf_counter()
                try:
                    self._aux[key] = load()
                except BackendError:
                    raise
                except Exception as e:
                    raise BackendError(f"Failed to load the {key} model: {e}") from e
                log.info("Loaded %s in %.1fs", key, time.perf_counter() - t0)
            return self._aux[key]

    @staticmethod
    def _free_memory() -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _seed(params: GenerationParams) -> None:
        if params.seed is None:
            return
        import torch

        torch.manual_seed(params.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(params.seed)
        random.seed(params.seed)
        np.random.seed(params.seed % 2**32)

    def _cached_voice(self, key: tuple, build):
        if key in self._voices:
            self._voices.move_to_end(key)
            return self._voices[key]
        value = build()
        self._voices[key] = value
        while len(self._voices) > MAX_CACHED_VOICES:
            self._voices.popitem(last=False)
        return value

    def _use_voice(self, key: str, model, voice: Optional[Audio], params: GenerationParams) -> None:
        """Point the model's conditionals at ``voice`` (or its built-in voice)."""
        if voice is None:
            if self._builtin.get(key) is None:
                raise BackendError(f"The {key} model has no built-in voice; give a reference clip.")
            model.conds = self._builtin[key]
            return
        norm = key in (TURBO, NANO) and params.norm_loudness is not False

        def build():
            with _wav_file(voice) as path:
                if key in (TURBO, NANO):
                    model.prepare_conditionals(path, norm_loudness=norm)
                else:
                    model.prepare_conditionals(path, exaggeration=params.exaggeration or 0.5)
            return model.conds
        model.conds = self._cached_voice((key, A.fingerprint(voice), norm), build)

    # ------------------------------------------------------------------ generation

    def synthesize(self, model, texts, voice, params, language=None, version=None):
        key = self._key(model, version or self.settings.multilingual_version)
        kwargs = params.model_kwargs(model)
        if model == MULTILINGUAL:
            kwargs["language_id"] = language or "en"
        with self._lock:
            m = self.model(key)
            self._use_voice(key, m, voice, params)
            self._seed(params)
            wavs = [m.generate(text, **kwargs).squeeze(0).cpu().numpy().astype(np.float32) for text in texts]
        return wavs, int(m.sr)

    def convert(self, source, target):
        with self._lock:
            m = self.model(VC)
            if target is None:
                if self._builtin.get(VC) is None:
                    raise BackendError("The voice conversion model has no built-in voice; give a target clip.")
                m.ref_dict = self._builtin[VC]
            else:
                def build():
                    with _wav_file(target) as path:
                        m.set_target_voice(path)
                    return m.ref_dict
                m.ref_dict = self._cached_voice((VC, A.fingerprint(target)), build)
            with _wav_file(source) as path:
                wav = m.generate(path)
        return wav.squeeze(0).cpu().numpy().astype(np.float32), int(m.sr)

    def transcribe(self, audio, language=None):
        import librosa
        import torch

        def load():
            from transformers import pipeline

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            return pipeline("automatic-speech-recognition", model=self.settings.whisper_model, dtype=dtype,
                            device=self.device)

        wav, sr = audio
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        with self._lock:
            pipe = self.aux_model(WHISPER, load)
            out = pipe({"raw": np.ascontiguousarray(wav, dtype=np.float32), "sampling_rate": 16000},
                       chunk_length_s=30, return_timestamps=True, generate_kwargs=generate_kwargs)
        total = len(audio[0]) / audio[1]
        segments = []
        for chunk in out.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            segments.append({"start": round(float(start or 0.0), 2), "end": round(float(end if end is not None else total), 2),
                             "text": chunk["text"].strip()})
        return {"text": out["text"].strip(), "segments": segments}

    def translate(self, texts, source, target):
        import torch

        def load():
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            name = self.settings.translation_model
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
            return AutoTokenizer.from_pretrained(name), model

        out: list[str] = []
        with self._lock:
            tokenizer, model = self.aux_model(TRANSLATOR, load)
            tokenizer.src_lang = NLLB_CODES[source]
            target_id = tokenizer.convert_tokens_to_ids(NLLB_CODES[target])
            for i in range(0, len(texts), 8):
                batch = tokenizer(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.inference_mode():
                    ids = model.generate(**batch.to(self.device), forced_bos_token_id=target_id, num_beams=4,
                                         max_new_tokens=512)
                out.extend(s.strip() for s in tokenizer.batch_decode(ids, skip_special_tokens=True))
        return out

    def detect_watermark(self, audio):
        import perth

        with self._lock:
            if self._watermarker is None:
                self._watermarker = perth.PerthImplicitWatermarker()
            return float(self._watermarker.get_watermark(audio[0], sample_rate=audio[1]))

    def status(self) -> dict[str, Any]:
        import torch

        info: dict[str, Any] = {"backend": self.name, "device": self.device, "loaded": list(self._models),
                                "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        return info
