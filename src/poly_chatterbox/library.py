"""Voice library: the models' built-in voice, upstream demo voices and user-saved cloned voices.

* ``default``: each model's own built-in voice (no reference clip).
* ``sample_<code>`` / ``podcast``: the reference clips from the upstream demo apps, one per
  multilingual language plus Turbo's podcast host. Downloaded on first use into ``<data_dir>/samples``.
* cloned voices: ``<data_dir>/voices/<id>/reference.wav`` and ``meta.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .audio import Audio, load_audio, save_wav
from .languages import LANGUAGES, SAMPLES

DEFAULT_VOICE = "default"


@dataclass
class Voice:
    id: str
    name: str
    kind: str  # builtin | sample | cloned
    language: str  # language code, or "" when not tied to one
    gender: str = "unknown"
    description: str = ""
    url: Optional[str] = field(default=None, repr=False)  # where a sample voice is downloaded from

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("url")
        return d


BUILTIN_VOICES: list[Voice] = [
    Voice(DEFAULT_VOICE, "Built-in voice", "builtin", "", description="The voice that ships with each model. No reference clip needed."),
    Voice("podcast", "Podcast host", "sample", "en", "female",
          "Casual podcast host; the reference clip from the upstream Turbo demo.",
          "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"),
] + [
    Voice(f"sample_{code}", f"{LANGUAGES[code]} sample", "sample", code, s["gender"],
          f"Native {LANGUAGES[code]} speaker; the reference clip from the upstream multilingual demo.", s["audio"])
    for code, s in SAMPLES.items()
]
_BUILTIN = {v.id: v for v in BUILTIN_VOICES}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:40] or "voice"


class VoiceLibrary:
    def __init__(self, voices_dir: Path):
        self.dir = Path(voices_dir)
        self._lock = threading.Lock()

    def _meta(self, voice_id: str) -> Optional[dict]:
        path = self.dir / voice_id / "meta.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _voice_from_meta(meta: dict) -> Voice:
        return Voice(id=meta["id"], name=meta["name"], kind="cloned", language=meta.get("language", ""),
                     gender=meta.get("gender", "unknown"), description=meta.get("description") or "Cloned from reference audio.")

    def cloned(self) -> list[Voice]:
        if not self.dir.is_dir():
            return []
        voices = []
        for d in sorted(self.dir.iterdir()):
            meta = self._meta(d.name) if d.is_dir() else None
            if meta:
                voices.append(self._voice_from_meta(meta))
        return voices

    def list(self) -> list[Voice]:
        return BUILTIN_VOICES + self.cloned()

    def get(self, voice_id: str) -> Voice:
        voice = _BUILTIN.get((voice_id or "").lower())
        if voice:
            return voice
        meta = self._meta(_slug(voice_id)) if voice_id else None
        if meta:
            return self._voice_from_meta(meta)
        raise LookupError(f"Unknown voice {voice_id!r}. List voices with `poly-chatterbox voices list` or GET /api/voices.")

    def reference(self, voice_id: str) -> Audio:
        """Reference audio of a cloned voice."""
        meta = self._meta(_slug(voice_id))
        if not meta:
            raise LookupError(f"{voice_id!r} is not a saved cloned voice.")
        return load_audio(self.dir / meta["id"] / "reference.wav")

    def add(self, name: str, ref_audio: Audio, language: str = "", description: Optional[str] = None,
            gender: str = "unknown") -> Voice:
        name = name.strip()
        if not name:
            raise ValueError("Voice name is required.")
        with self._lock:
            base = _slug(name)
            voice_id, n = base, 2
            while voice_id in _BUILTIN or (self.dir / voice_id).exists():
                voice_id, n = f"{base}_{n}", n + 1
            folder = self.dir / voice_id
            folder.mkdir(parents=True)
            save_wav(folder / "reference.wav", *ref_audio)
            meta = {"id": voice_id, "name": name, "language": language, "description": description or "",
                    "gender": gender, "created": time.time()}
            (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._voice_from_meta(meta)

    def remove(self, voice_id: str) -> None:
        if (voice_id or "").lower() in _BUILTIN:
            raise ValueError("Built-in voices cannot be removed.")
        folder = self.dir / _slug(voice_id)
        if not (folder / "meta.json").is_file():
            raise LookupError(f"Unknown cloned voice {voice_id!r}.")
        shutil.rmtree(folder)
