"""Mock backend: deterministic placeholder tones. No GPU, downloads or torch needed.

Useful for tests, UI development and trying the API surface. Each model/voice maps to a different
pitch and the output length scales with the text, so results are distinguishable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from ..audio import Audio
from .base import Backend, GenerationParams

SAMPLE_RATE = 24000


def _hash01(*parts: object) -> float:
    h = hashlib.md5("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") / 2**32


def _voice_key(voice: Optional[Audio]) -> str:
    return "builtin" if voice is None else hashlib.md5(np.ascontiguousarray(voice[0]).tobytes()).hexdigest()


def _tone(seconds: float, key: str, seed: Optional[int]) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    f0 = 110.0 + 220.0 * _hash01(key)
    jitter = 0.02 * _hash01(seed) if seed is not None else 0.0
    wave = sum(np.sin(2 * np.pi * f0 * (1 + jitter) * k * t) / k for k in (1, 2, 3))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2  # ~4 "syllables" per second
    return (0.25 * wave * syllables).astype(np.float32)


class MockBackend(Backend):
    name = "mock"

    def __init__(self, settings: Any = None):
        self.calls: list[tuple[str, dict]] = []  # inspected by tests

    def synthesize(self, model, texts, voice, params, language=None, version=None):
        self.calls.append(("synthesize", dict(model=model, texts=texts, voice=voice is not None, params=params,
                                              language=language, version=version)))
        key = f"{model}|{version}|{language}|{_voice_key(voice)}"
        return [_tone(min(30.0, max(0.5, 0.065 * len(t))), key, params.seed) for t in texts], SAMPLE_RATE

    def convert(self, source, target):
        self.calls.append(("convert", dict(target=target is not None)))
        return _tone(len(source[0]) / source[1], _voice_key(target), None), SAMPLE_RATE

    def detect_watermark(self, audio):
        self.calls.append(("detect_watermark", {}))
        return 0.0

    def status(self):
        return {"backend": self.name, "loaded": [], "note": "placeholder tones, not speech"}
