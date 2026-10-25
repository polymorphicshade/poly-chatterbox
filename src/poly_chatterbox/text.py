"""Inline pause markup and splitting long text into chunks the models handle well.

Inline markup
-------------
``[pause]``                                         insert 0.5 s of silence
``[pause 800ms]`` / ``[pause 1.5s]`` / ``[pause 2]``  explicit length (bare numbers are seconds)

Anything else in square brackets is left as text, so Turbo's paralinguistic tags (``[laugh]``,
``[cough]``, ...) pass straight through to the model.

Each Chatterbox call produces at most ~40 s of audio, so text longer than ``max_chars`` is split at
sentence boundaries (then at commas, then at spaces) and the pieces are joined afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PAUSE_SECONDS = 0.5
MAX_PAUSE_SECONDS = 10.0

# Paralinguistic tags understood natively by Chatterbox-Turbo and -Nano.
EVENT_TAGS = ["[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]", "[sniff]", "[gasp]", "[chuckle]", "[laugh]"]

_PAUSE = re.compile(r"\[\s*pause(?:[\s=:]+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?)?\s*\]", re.IGNORECASE)
# Sentence ends: Latin, Arabic and Devanagari terminators (plus closing quotes/brackets) before a space,
# or CJK terminators, which need no space after them.
_SENTENCE_END = re.compile(r"[.!?…؟।]+[\"'”’»)\]]*(?:\s+|$)|[。！？]+[”’」』)]*")
_CLAUSE_END = re.compile(r"[,;:，；、]\s*")


@dataclass
class Segment:
    text: str = ""
    pause: float = 0.0  # seconds; a pause segment has empty text

    @property
    def is_pause(self) -> bool:
        return self.pause > 0 and not self.text


def _split_after(pattern: re.Pattern, text: str) -> list[str]:
    """Split text after each match of pattern, keeping the punctuation with the piece before it."""
    parts, pos = [], 0
    for m in pattern.finditer(text):
        parts.append(text[pos:m.end()])
        pos = m.end()
    parts.append(text[pos:])
    return parts


def _pack(parts: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join parts into chunks of at most max_chars (a single oversized part is kept whole)."""
    chunks: list[str] = []
    current = ""
    for part in (p.strip() for p in parts):
        if not part:
            continue
        candidate = f"{current}{joiner}{part}" if current else part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars``, preferring sentence, then clause, then word breaks."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    for sentence in _pack(_split_after(_SENTENCE_END, text), max_chars, " "):
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        for clause in _pack(_split_after(_CLAUSE_END, sentence), max_chars, " "):
            chunks.extend([clause] if len(clause) <= max_chars else _pack(clause.split(" "), max_chars, " "))
    return chunks


def parse_markup(text: str, max_chars: int = 300) -> list[Segment]:
    """Split text into speech chunks and pauses according to the inline markup."""
    segments: list[Segment] = []
    pos = 0
    for m in _PAUSE.finditer(text):
        segments.extend(Segment(text=c) for c in split_text(text[pos:m.start()], max_chars))
        seconds = DEFAULT_PAUSE_SECONDS
        if m.group("num"):
            seconds = float(m.group("num")) / (1000.0 if (m.group("unit") or "").lower() == "ms" else 1.0)
        if seconds > 0:
            segments.append(Segment(pause=min(seconds, MAX_PAUSE_SECONDS)))
        pos = m.end()
    segments.extend(Segment(text=c) for c in split_text(text[pos:], max_chars))
    if not any(s.text for s in segments):
        raise ValueError("Text is required (the input only contains pause markup).")
    return segments
