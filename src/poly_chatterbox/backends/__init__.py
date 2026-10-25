from __future__ import annotations

from ..config import Settings
from .base import (GENERATION_OPTIONS, MULTILINGUAL, MULTILINGUAL_VERSIONS, NANO, ORIGINAL, TTS_MODELS, TURBO, Backend,
                   BackendError, GenerationParams)


def create_backend(settings: Settings) -> Backend:
    if settings.backend == "mock":
        from .mock import MockBackend

        return MockBackend(settings)
    if settings.backend == "chatterbox":
        from .chatterbox import ChatterboxBackend

        return ChatterboxBackend(settings)
    raise BackendError(f"Unknown backend {settings.backend!r} (expected 'chatterbox' or 'mock').")


__all__ = ["GENERATION_OPTIONS", "MULTILINGUAL", "MULTILINGUAL_VERSIONS", "NANO", "ORIGINAL", "TTS_MODELS", "TURBO",
           "Backend", "BackendError", "GenerationParams", "create_backend"]
