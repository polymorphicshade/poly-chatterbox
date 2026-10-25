"""Audio I/O and small DSP helpers (numpy + soundfile only)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import soundfile as sf

Audio = tuple[np.ndarray, int]  # (mono float32 waveform, sample rate)


def load_audio(source: bytes | str | Path) -> Audio:
    """Decode WAV/FLAC/OGG/MP3 bytes or a file path into a mono float32 waveform."""
    try:
        if isinstance(source, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(str(source), dtype="float32", always_2d=True)
    except Exception as e:  # soundfile raises LibsndfileError / RuntimeError subclasses
        raise ValueError(f"Could not decode reference audio ({e}). Use WAV, FLAC, OGG or MP3.") from e
    return data.mean(axis=1).astype(np.float32), int(sr)


def to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def duration(audio: Audio) -> float:
    wav, sr = audio
    return len(wav) / float(sr)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def _fade(wav: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    n = min(len(wav) // 2, int(sr * ms / 1000))
    if n <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


def concat(pieces: list[np.ndarray], sr: int) -> np.ndarray:
    """Concatenate pieces with short fades at the joins to avoid clicks."""
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    if len(pieces) == 1:
        return pieces[0].astype(np.float32)
    return np.concatenate([_fade(p.astype(np.float32), sr) for p in pieces])


def trim_silence(wav: np.ndarray, sr: int, threshold: float = 0.01, pad_ms: float = 100.0) -> np.ndarray:
    """Trim leading/trailing near-silence (keeps a small pad)."""
    idx = np.flatnonzero(np.abs(wav) > threshold)
    if idx.size == 0:
        return wav
    pad = int(sr * pad_ms / 1000)
    return wav[max(0, idx[0] - pad): min(len(wav), idx[-1] + pad)]


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m < 1e-6:
        return wav
    return (wav * (peak / m)).astype(np.float32)


def fingerprint(audio: Audio, *extra: object) -> str:
    wav, sr = audio
    h = hashlib.sha1(np.ascontiguousarray(wav).tobytes())
    h.update(str(sr).encode())
    for e in extra:
        h.update(repr(e).encode())
    return h.hexdigest()
