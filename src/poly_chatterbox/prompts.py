"""Saved prompts: text plus the mode, language, voice and generation options to speak it with.

Layout: ``<data_dir>/prompts/<id>.json``. Saving under an existing name replaces that prompt.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .library import _slug


@dataclass
class Prompt:
    id: str
    name: str
    text: str
    mode: Optional[str] = None  # original | turbo | nano | multilingual
    language: Optional[str] = None  # language code (multilingual)
    version: Optional[str] = None  # multilingual checkpoint: v2 | v3
    voice_id: Optional[str] = None  # library voice; None: the request's own voice
    generation: dict[str, Any] = field(default_factory=dict)
    created: float = 0.0
    updated: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class PromptStore:
    def __init__(self, prompts_dir: Path):
        self.dir = Path(prompts_dir)
        self._lock = threading.Lock()

    def _path(self, prompt_id: str) -> Path:
        return self.dir / f"{_slug(prompt_id)}.json"

    @staticmethod
    def _read(path: Path) -> Prompt:
        return Prompt(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[Prompt]:
        if not self.dir.is_dir():
            return []
        prompts = []
        for path in self.dir.glob("*.json"):
            try:
                prompts.append(self._read(path))
            except (OSError, ValueError, TypeError):
                continue  # not one of ours (or damaged): leave it alone
        return sorted(prompts, key=lambda p: p.name.lower())

    def get(self, prompt_id: str) -> Prompt:
        path = self._path(prompt_id)
        if not path.is_file():
            raise LookupError(f"Unknown prompt {prompt_id!r}. List prompts with `poly-chatterbox prompts list` or GET /api/prompts.")
        return self._read(path)

    def save(self, prompt: Prompt) -> tuple[Prompt, bool]:
        """Store a prompt (its id comes from its name). Returns it and whether it replaced an existing one."""
        prompt.id = _slug(prompt.name)
        path = self._path(prompt.id)
        with self._lock:
            existing = self._read(path) if path.is_file() else None
            now = time.time()
            prompt.created = existing.created if existing else now
            prompt.updated = now
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prompt.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return prompt, existing is not None

    def remove(self, prompt_id: str) -> None:
        path = self._path(prompt_id)
        if not path.is_file():
            raise LookupError(f"Unknown prompt {prompt_id!r}.")
        path.unlink()
