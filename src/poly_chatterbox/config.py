"""Runtime settings, read from ``POLY_CHATTERBOX_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"POLY_CHATTERBOX_{name}", default)


@dataclass
class Settings:
    # "chatterbox" runs the real Chatterbox models; "mock" synthesizes placeholder tones (no GPU / downloads).
    backend: str = field(default_factory=lambda: _env("BACKEND", "chatterbox"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))  # auto | cuda | cuda:1 | mps | cpu
    # Multilingual checkpoint used when a request doesn't pick one: "v3" (recommended) or "v2".
    multilingual_version: str = field(default_factory=lambda: _env("MULTILINGUAL_VERSION", "v3"))
    # How many models may be resident at once (LRU eviction beyond this).
    max_loaded_models: int = field(default_factory=lambda: int(_env("MAX_LOADED_MODELS", "3")))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(Path.cwd() / "data"))))
    min_ref_seconds: float = field(default_factory=lambda: float(_env("MIN_REF_SECONDS", "5.0")))
    max_ref_seconds: float = field(default_factory=lambda: float(_env("MAX_REF_SECONDS", "60.0")))
    # Long text is split at sentence boundaries into chunks of at most this many characters.
    max_chunk_chars: int = field(default_factory=lambda: int(_env("MAX_CHUNK_CHARS", "300")))

    # Hugging Face id of the Whisper checkpoint used by the Transcribe tab / `transcribe` (loaded on first use).
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "openai/whisper-large-v3-turbo"))

    # Hugging Face id of the NLLB-200 checkpoint that translates dialog (loaded on first use).
    translation_model: str = field(default_factory=lambda: _env("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"))

    # SpeechBrain speech-enhancement checkpoint that removes background noise from clips (loaded on first use).
    denoise_model: str = field(default_factory=lambda: _env("DENOISE_MODEL", "speechbrain/sepformer-dns4-16k-enhancement"))

    # Where saved prompts live (default: <data dir>/prompts). Lets a deployment mount them on their own volume.
    prompts_path: str = field(default_factory=lambda: _env("PROMPTS_DIR", ""))

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def prompts_dir(self) -> Path:
        return Path(self.prompts_path) if self.prompts_path else self.data_dir / "prompts"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
