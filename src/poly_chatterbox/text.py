"""Inline markup and splitting long text into chunks the models handle well.

Inline markup
-------------
``[pause]``                                           insert 0.5 s of silence
``[pause 800ms]`` / ``[pause 1.5s]`` / ``[pause 2]``    explicit length (bare numbers are seconds)
``[settings exaggeration=0.9 cfg_weight=0.3]…[/settings]``   generation options for that span only
``[settings preset=excited temperature=0.7]…[/settings]``   an emotion preset (see presets.py), then overrides
``[voice: sample_de]…[/voice]``                       a library voice for that span only

Spans can be nested (inner ``[settings]`` add to or override outer ones) and the two kinds can be
mixed. Anything else in square brackets is left as text, so Turbo's paralinguistic tags (``[laugh]``,
``[cough]``, ...) pass straight through to the model.

Each Chatterbox call produces at most ~40 s of audio, so text longer than ``max_chars`` is split at
sentence boundaries (then at commas, then at spaces) and the pieces are joined afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .backends.base import GENERATION_OPTIONS
from .presets import preset_options

DEFAULT_PAUSE_SECONDS = 0.5
MAX_PAUSE_SECONDS = 10.0

# Paralinguistic tags understood natively by Chatterbox-Turbo and -Nano.
EVENT_TAGS = ["[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]", "[sniff]", "[gasp]", "[chuckle]", "[laugh]"]

# `seed` applies to the whole request, so it can't change inside a span.
SPAN_OPTIONS = [name for name in GENERATION_OPTIONS if name != "seed"]

_TAG = re.compile(
    r"\[\s*(?:"
    r"(?P<pause>pause(?:[\s=:]+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?)?)"
    r"|(?P<settings>settings(?:\s+(?P<opts>[^\]]*?))?)"
    r"|(?P<voice>voice\s*[:=]\s*(?P<vid>[^\]]+?))"
    r"|(?P<close>/(?:settings|voice))"
    r")\s*\]",
    re.IGNORECASE,
)
# Sentence ends: Latin, Arabic and Devanagari terminators (plus closing quotes/brackets) before a space,
# or CJK terminators, which need no space after them.
_SENTENCE_END = re.compile(r"[.!?…؟।]+[\"'”’»)\]]*(?:\s+|$)|[。！？]+[”’」』)]*")
_CLAUSE_END = re.compile(r"[,;:，；、]\s*")

_TRUE, _FALSE = {"true", "on", "yes", "1"}, {"false", "off", "no", "0"}


@dataclass
class Segment:
    text: str = ""
    pause: float = 0.0  # seconds; a pause segment has empty text
    settings: tuple[tuple[str, Any], ...] = ()  # generation option overrides from enclosing [settings] spans
    voice: Optional[str] = None  # library voice id from an enclosing [voice] span (None: the request's voice)

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


_BRACKETED = re.compile(r"\[[^\]]*\]")


def translate_markup(text: str, translate: Callable[[list[str]], list[str]], max_chars: int = 300) -> str:
    """Translate the prose of ``text`` and leave every ``[bracketed]`` tag (markup and Turbo's
    paralinguistic tags) exactly where it is. ``translate`` maps a list of sentences to their translations.
    """
    pieces: list[str] = []  # alternating: prose, tag, prose, tag, ..., prose
    pos = 0
    for m in _BRACKETED.finditer(text):
        pieces += [text[pos:m.start()], m.group()]
        pos = m.end()
    pieces.append(text[pos:])

    chunks = [split_text(p, max_chars) for p in pieces[::2]]
    flat = [c for group in chunks for c in group]
    translated = iter(translate(flat) if flat else [])
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i % 2:
            out.append(piece)
            continue
        group = chunks[i // 2]
        if not group:
            out.append(piece)
            continue
        lead = piece[:len(piece) - len(piece.lstrip())]
        trail = piece[len(piece.rstrip()):]
        out.append(lead + " ".join(next(translated) for _ in group).strip() + trail)
    return "".join(out)


def _convert(name: str, raw: str) -> Any:
    opt = GENERATION_OPTIONS[name]
    kind = opt["type"]
    if kind == "bool":
        if raw.lower() in _TRUE:
            return True
        if raw.lower() in _FALSE:
            return False
        raise ValueError(f"[settings] {name} must be true or false, not {raw!r}.")
    try:
        value = float(raw) if kind == "float" else int(raw)
    except ValueError:
        raise ValueError(f"[settings] {name} must be a number, not {raw!r}.") from None
    if (opt["min"] is not None and value < opt["min"]) or (opt["max"] is not None and value > opt["max"]):
        lo = opt["min"] if opt["min"] is not None else "-∞"
        hi = opt["max"] if opt["max"] is not None else "∞"
        raise ValueError(f"[settings] {name}={raw} is out of range ({lo} to {hi}).")
    return value


def parse_settings(opts: Optional[str], model: Optional[str] = None) -> dict[str, Any]:
    """Parse the ``name=value`` pairs of a ``[settings …]`` tag.

    ``preset=name`` expands to that emotion preset's options (those ``model`` uses, if given); options
    written in the same tag win over it, whatever their order.
    """
    out: dict[str, Any] = {}
    preset: dict[str, Any] = {}
    for token in filter(None, re.split(r"[\s,]+", (opts or "").strip())):
        name, sep, raw = token.partition("=")
        name = name.lower()
        if not sep or not raw:
            raise ValueError(f"Bad [settings] option {token!r}: write name=value, e.g. exaggeration=0.8.")
        if name == "seed":
            raise ValueError("seed can't be set inside [settings]: it applies to the whole request.")
        if name == "preset":
            preset = preset_options(raw, model)
            continue
        if name not in SPAN_OPTIONS:
            raise ValueError(f"Unknown [settings] option {name!r}. Use: preset, {', '.join(SPAN_OPTIONS)}.")
        out[name] = _convert(name, raw)
    out = {**preset, **out}
    if not out:
        raise ValueError("[settings] needs at least one option, e.g. [settings exaggeration=0.8].")
    return out


def parse_markup(text: str, max_chars: int = 300, model: Optional[str] = None) -> list[Segment]:
    """Split text into speech chunks and pauses according to the inline markup.

    ``model`` limits the options that ``[settings preset=…]`` expands to the ones that model uses.
    """
    segments: list[Segment] = []
    settings_stack: list[dict[str, Any]] = []  # each entry is the merged settings in effect
    voice_stack: list[str] = []

    def add_text(chunk: str) -> None:
        context = tuple(sorted(settings_stack[-1].items())) if settings_stack else ()
        voice = voice_stack[-1] if voice_stack else None
        segments.extend(Segment(text=c, settings=context, voice=voice) for c in split_text(chunk, max_chars))

    pos = 0
    for m in _TAG.finditer(text):
        add_text(text[pos:m.start()])
        pos = m.end()
        if m.group("pause") is not None:
            seconds = DEFAULT_PAUSE_SECONDS
            if m.group("num"):
                seconds = float(m.group("num")) / (1000.0 if (m.group("unit") or "").lower() == "ms" else 1.0)
            if seconds > 0:
                segments.append(Segment(pause=min(seconds, MAX_PAUSE_SECONDS)))
        elif m.group("settings") is not None:
            merged = {**(settings_stack[-1] if settings_stack else {}), **parse_settings(m.group("opts"), model)}
            settings_stack.append(merged)
        elif m.group("voice") is not None:
            voice_stack.append(m.group("vid").strip())
        else:
            kind = m.group("close").lower()[1:]
            stack = settings_stack if kind == "settings" else voice_stack
            if not stack:
                raise ValueError(f"[/{kind}] has no matching [{kind} …] before it.")
            stack.pop()
    add_text(text[pos:])
    if settings_stack:
        raise ValueError("A [settings …] tag is never closed: add [/settings] after the text it applies to.")
    if voice_stack:
        raise ValueError("A [voice: …] tag is never closed: add [/voice] after the text it applies to.")
    if not any(s.text for s in segments):
        raise ValueError("Text is required (the input only contains markup).")
    return segments
