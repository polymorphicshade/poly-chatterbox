"""Backend interface: the thin layer that actually runs the Chatterbox models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from ..audio import Audio

# The four TTS models. Turbo and Nano share an architecture (and request options).
ORIGINAL, TURBO, NANO, MULTILINGUAL = "original", "turbo", "nano", "multilingual"
TTS_MODELS = (ORIGINAL, TURBO, NANO, MULTILINGUAL)
MULTILINGUAL_VERSIONS = ("v2", "v3")


@dataclass
class GenerationParams:
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    min_p: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    norm_loudness: Optional[bool] = None
    seed: Optional[int] = None

    def model_kwargs(self, model: str) -> dict[str, Any]:
        """Kwargs for the model's generate call: set values the model supports (unset ones use its defaults)."""
        return {k: v for k, v in asdict(self).items()
                if v is not None and model in GENERATION_OPTIONS[k]["defaults"] and k not in ("seed", "norm_loudness")}

    def unsupported(self, model: str) -> list[str]:
        """Options that were set but that ``model`` ignores."""
        return [k for k, v in asdict(self).items() if v is not None and model not in GENERATION_OPTIONS[k]["defaults"]]


def _all(value: Any) -> dict[str, Any]:
    return {m: value for m in TTS_MODELS}


# Labels, ranges, per-model defaults and user-facing tuning advice for each GenerationParams field.
# This is the single source for the API schema descriptions, ``/api/info`` (rendered by the web UI)
# and the CLI help. A model missing from ``defaults`` ignores the option. Defaults are those of the
# upstream ``generate()`` methods. ``primary`` options are shown as sliders in the main form.
GENERATION_OPTIONS: dict[str, dict[str, Any]] = {
    "exaggeration": {
        "label": "Exaggeration", "type": "float", "min": 0.25, "max": 2.0, "step": 0.05, "primary": True,
        "defaults": {ORIGINAL: 0.5, MULTILINGUAL: 0.5},
        "description": "How emotional and animated the delivery is. 0.5 is neutral; 0.7 and up is more dramatic and "
                       "tends to speed speech up (lower CFG/pace to compensate). Extreme values can be unstable.",
    },
    "cfg_weight": {
        "label": "CFG / pace", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "primary": True,
        "defaults": {ORIGINAL: 0.5, MULTILINGUAL: 0.5},
        "description": "How closely the model sticks to the reference voice and text; also sets the pace. Lower it to "
                       "about 0.3 for a fast-talking reference or dramatic reads. Set it to 0 when the reference is in a "
                       "different language, so the output doesn't inherit its accent.",
    },
    "temperature": {
        "label": "Temperature", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05,
        "defaults": _all(0.8),
        "description": "How adventurous the delivery is. Lower (0.5-0.7) gives a steadier, more consistent read; "
                       "higher gives livelier, more varied intonation but can slur words or add glitches.",
    },
    "min_p": {
        "label": "Min-p", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01,
        "defaults": {ORIGINAL: 0.05, MULTILINGUAL: 0.05},
        "description": "Drops sounds less likely than this fraction of the most likely one. Copes with high "
                       "temperatures better than top-p; 0.02-0.1 is recommended. 0 turns it off.",
    },
    "top_p": {
        "label": "Top-p", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01,
        "defaults": {ORIGINAL: 1.0, MULTILINGUAL: 1.0, TURBO: 0.95, NANO: 0.95},
        "description": "Samples only from the most likely sounds whose probabilities add up to this share. Lowering it "
                       "trims unlikely sounds for cleaner but flatter speech. 1.0 turns it off.",
    },
    "top_k": {
        "label": "Top-k", "type": "int", "min": 1, "max": 1000, "step": 1,
        "defaults": {TURBO: 1000, NANO: 1000},
        "description": "Samples only from the k most likely sounds at each step. Smaller (50-200) is safer and more "
                       "monotone; larger allows more variety.",
    },
    "repetition_penalty": {
        "label": "Repetition penalty", "type": "float", "min": 1.0, "max": 2.0, "step": 0.05,
        "defaults": _all(1.2),
        "description": "Discourages repeating the same sounds. Raise it if the voice stutters or loops; too high can "
                       "distort the voice. 1.0 turns it off.",
    },
    "norm_loudness": {
        "label": "Normalize reference loudness", "type": "bool", "min": None, "max": None, "step": None,
        "defaults": {TURBO: True, NANO: True},
        "description": "Brings the reference clip to -27 LUFS before cloning, so quiet or loud references give an "
                       "evenly levelled voice. Turn it off to keep the reference's own level.",
    },
    "seed": {
        "label": "Seed", "type": "int", "min": 0, "max": None, "step": 1,
        "defaults": _all(None),
        "description": "Makes a take repeatable: the same seed with the same text and settings gives the same audio. "
                       "Leave it empty for a fresh take each time. With a fixed seed, change one option at a time to "
                       "hear exactly what it does.",
    },
}


class BackendError(RuntimeError):
    """The backend could not run (missing dependencies, model load failure, …)."""


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def synthesize(self, model: str, texts: list[str], voice: Optional[Audio], params: GenerationParams,
                   language: Optional[str] = None, version: Optional[str] = None) -> tuple[list[np.ndarray], int]:
        """Speak each text with a TTS model. ``voice`` is a reference clip, or None for the model's built-in voice.

        ``language`` (a code) and ``version`` (v2/v3) apply to the multilingual model only.
        """

    @abstractmethod
    def convert(self, source: Audio, target: Optional[Audio]) -> Audio:
        """Voice conversion: re-voice ``source`` with ``target``'s timbre (None: the built-in voice)."""

    @abstractmethod
    def transcribe(self, audio: Audio, language: Optional[str] = None) -> dict[str, Any]:
        """Speech to text with Whisper. ``language`` is a code or name (None: detect it).

        Returns ``{"text": str, "segments": [{"start": float, "end": float, "text": str}, ...]}``.
        """

    @abstractmethod
    def detect_watermark(self, audio: Audio) -> float:
        """Perth watermark score: 1.0 for Chatterbox output, 0.0 for unwatermarked audio."""

    def status(self) -> dict[str, Any]:
        return {"backend": self.name}
