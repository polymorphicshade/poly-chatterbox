"""Emotion presets: named bundles of generation options that push the delivery towards a mood.

Chatterbox has no emotion switch. The closest controls are ``exaggeration`` (how animated the
delivery is), ``cfg_weight`` (lower is faster and looser, higher is slower and steadier) and the
sampling options (``temperature``, ``top_p``, ``top_k``). Each preset sets a combination of them;
options a model doesn't use (Turbo and Nano ignore exaggeration and CFG) are left out for that model.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Optional

from .backends.base import GENERATION_OPTIONS, GenerationParams

EMOTION_PRESETS: dict[str, dict[str, Any]] = {
    "neutral": {
        "label": "Neutral", "description": "Even, matter-of-fact delivery. The model defaults.",
        "options": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "top_p": 1.0, "top_k": 1000},
    },
    "happy": {
        "label": "Happy", "description": "Warm and upbeat, a little quicker and brighter.",
        "options": {"exaggeration": 0.65, "cfg_weight": 0.4, "temperature": 0.85, "top_k": 1000},
    },
    "excited": {
        "label": "Excited", "description": "Fast, high-energy and animated.",
        "options": {"exaggeration": 0.9, "cfg_weight": 0.3, "temperature": 0.95, "top_k": 1000},
    },
    "playful": {
        "label": "Playful", "description": "Teasing and lively, with plenty of variation in the intonation.",
        "options": {"exaggeration": 0.8, "cfg_weight": 0.35, "temperature": 1.0, "top_k": 1000},
    },
    "sad": {
        "label": "Sad", "description": "Slow, subdued and heavy.",
        "options": {"exaggeration": 0.4, "cfg_weight": 0.65, "temperature": 0.7, "top_k": 300},
    },
    "tired": {
        "label": "Tired", "description": "Flat, slow and low on energy.",
        "options": {"exaggeration": 0.25, "cfg_weight": 0.7, "temperature": 0.6, "top_k": 150},
    },
    "calm": {
        "label": "Calm", "description": "Relaxed, soothing and steady.",
        "options": {"exaggeration": 0.3, "cfg_weight": 0.6, "temperature": 0.6, "top_k": 200},
    },
    "serious": {
        "label": "Serious", "description": "Measured and authoritative, like a news reader.",
        "options": {"exaggeration": 0.55, "cfg_weight": 0.6, "temperature": 0.65, "top_p": 0.9, "top_k": 250},
    },
    "nervous": {
        "label": "Nervous", "description": "Hurried and unsteady, with an uneven rhythm.",
        "options": {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 1.05, "top_k": 1000},
    },
    "angry": {
        "label": "Angry", "description": "Forceful and tense, pushed hard.",
        "options": {"exaggeration": 1.1, "cfg_weight": 0.3, "temperature": 0.9, "top_k": 1000},
    },
    "dramatic": {
        "label": "Dramatic", "description": "Theatrical and intense. Extreme values can be unstable, so try a few seeds.",
        "options": {"exaggeration": 1.3, "cfg_weight": 0.25, "temperature": 0.9, "top_k": 1000},
    },
}


def preset_options(name: str, model: Optional[str] = None) -> dict[str, Any]:
    """The options of preset ``name``; with ``model``, only those that model uses."""
    key = (name or "").strip().lower()
    if key not in EMOTION_PRESETS:
        raise ValueError(f"Unknown emotion preset {name!r}. Use: {', '.join(EMOTION_PRESETS)}.")
    options = EMOTION_PRESETS[key]["options"]
    return {k: v for k, v in options.items() if model is None or model in GENERATION_OPTIONS[k]["defaults"]}


def apply_preset(name: Optional[str], params: GenerationParams, model: str) -> GenerationParams:
    """``params`` with the preset's options filled in underneath: options set explicitly win."""
    if not name:
        return params
    explicit = {k: v for k, v in asdict(params).items() if v is not None}
    return replace(GenerationParams(**preset_options(name, model)), **explicit)


def describe_presets() -> list[dict[str, Any]]:
    """For ``GET /api/presets`` and ``/api/info``."""
    return [{"id": key, **preset} for key, preset in EMOTION_PRESETS.items()]
