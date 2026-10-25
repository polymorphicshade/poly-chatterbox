"""Live voice conversion: speak into a microphone and hear (or stream) the converted voice.

``poly-chatterbox stream`` runs on the machine that has the audio devices (your PC). It listens to the
mic, cuts your speech into phrases at the pauses, converts each phrase with Chatterbox voice
conversion (in-process, or on a ``--server``) and plays the result to an output device, typically a
virtual audio cable that Discord or another app uses as its microphone.

Voice conversion works on whole clips, so this is near-real-time rather than instant: a phrase is
heard shortly after you finish saying it.

The pieces that don't touch audio hardware (:class:`PhraseSegmenter`, :class:`PlaybackBuffer`,
:class:`LiveConverter`) are separate so they can be tested without a sound card.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CONVERT_SAMPLE_RATE = 16000  # what Chatterbox VC reads; sending phrases at this rate keeps uploads small
BLOCK_SECONDS = 0.02
MIN_THRESHOLD = 0.006  # never treat room noise below this RMS level as speech


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if len(block) else 0.0


def auto_threshold(noise_rms: float) -> float:
    """Speech threshold from the measured background noise level."""
    return max(MIN_THRESHOLD, noise_rms * 4.0)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or not len(wav):
        return wav.astype(np.float32)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(sr, target_sr)
        return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    except ImportError:  # the `stream` extra includes scipy; this keeps tests and odd installs working
        n = int(round(len(wav) * target_sr / sr))
        return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


class PhraseSegmenter:
    """Cuts a stream of mic blocks into phrases: speech plus a little lead-in, ended by a pause.

    Feed it consecutive blocks; it returns the phrases that finished. Phrases shorter than ``min_ms`` of
    speech (clicks, coughs) are dropped, and a phrase is cut after ``max_seconds`` even without a pause.
    """

    def __init__(self, sample_rate: int, threshold: float, silence_ms: int = 500, min_ms: int = 300,
                 max_seconds: float = 15.0, preroll_ms: int = 200, tail_ms: int = 150, start_blocks: int = 2):
        ms = sample_rate / 1000.0
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.start_blocks = start_blocks
        self._silence = int(silence_ms * ms)
        self._min = int(min_ms * ms)
        self._max = int(max_seconds * sample_rate)
        self._preroll = int(preroll_ms * ms)
        self._tail = int(tail_ms * ms)
        self._reset()

    def _reset(self) -> None:
        self._speaking = False
        self._pre: deque[np.ndarray] = deque()
        self._pre_len = 0
        self._loud_run = 0
        self._phrase: list[np.ndarray] = []
        self._len = 0
        self._silent = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, block: np.ndarray) -> list[np.ndarray]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        loud = rms(block) >= self.threshold
        if not self._speaking:
            self._pre.append(block)
            self._pre_len += len(block)
            while len(self._pre) > 1 and self._pre_len - len(self._pre[0]) >= self._preroll:
                self._pre_len -= len(self._pre.popleft())
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.start_blocks:
                self._phrase, self._len = list(self._pre), self._pre_len
                self._pre, self._pre_len, self._loud_run = deque(), 0, 0
                self._speaking, self._silent = True, 0
            return []
        self._phrase.append(block)
        self._len += len(block)
        self._silent = 0 if loud else self._silent + len(block)
        if self._silent >= self._silence or self._len >= self._max:
            phrase = self._finish()
            return [phrase] if phrase is not None else []
        return []

    def flush(self) -> list[np.ndarray]:
        phrase = self._finish() if self._speaking else None
        return [phrase] if phrase is not None else []

    def _finish(self) -> Optional[np.ndarray]:
        wav = np.concatenate(self._phrase)
        spoken = len(wav) - self._silent
        wav = wav[: len(wav) - max(0, self._silent - self._tail)]  # keep a short natural tail
        self._reset()
        return wav if spoken >= self._min else None


class PlaybackBuffer:
    """Thread-safe queue of audio that the output stream's callback drains frame by frame."""

    def __init__(self):
        self._chunks: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._queued = 0

    def put(self, wav: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(np.asarray(wav, dtype=np.float32))
            self._queued += len(wav)

    def read(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(len(chunk), frames - filled)
                out[filled:filled + take] = chunk[:take]
                filled += take
                self._queued -= take
                if take == len(chunk):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[take:]
        return out

    @property
    def queued(self) -> int:
        return self._queued


Audio = tuple[np.ndarray, int]


class LiveConverter:
    """Converts phrases one at a time, in order, on a background thread and hands the results to ``sink``."""

    def __init__(self, convert: Callable[[Audio], Audio], sink: Callable[[Audio], None],
                 log: Callable[[str], None] = print, warn_backlog: int = 2):
        self._convert, self._sink, self._log, self._warn = convert, sink, log, warn_backlog
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="live-converter", daemon=True)
        self._thread.start()

    def submit(self, wav: np.ndarray, sr: int) -> None:
        self._queue.put((wav, sr, time.perf_counter()))
        if self._queue.qsize() > self._warn:
            self._log(f"warning: conversion is falling behind ({self._queue.qsize()} phrases waiting). "
                      "Try a faster GPU model setup, shorter phrases, or a longer pause between them.")

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            wav, sr, submitted = item
            try:
                t0 = time.perf_counter()
                out = self._convert((wav, sr))
                took = time.perf_counter() - t0
                self._sink(out)
                self._log(f"  {len(wav) / sr:.1f} s of speech -> converted in {took:.2f} s, "
                          f"played {time.perf_counter() - submitted:.2f} s after you stopped talking")
            except Exception as e:  # keep streaming when one phrase fails
                self._log(f"  conversion failed: {e}")

    def close(self, timeout: float = 30.0) -> None:
        """Finish the phrases already queued, then stop."""
        self._queue.put(None)
        self._thread.join(timeout)


# ---------------------------------------------------------------------------- audio devices

def _sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise RuntimeError("Live streaming needs the 'stream' extra: `uv sync --extra stream` "
                           f"(plus `--extra chatterbox` for local models). Details: {e}") from e
    return sd


def list_devices(out=sys.stdout) -> int:
    sd = _sounddevice()
    apis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    print(f"{'#':>3}  {'NAME':<44} {'HOST API':<22} IN OUT", file=out)
    for i, d in enumerate(sd.query_devices()):
        mark = ("<- default input" if i == default_in else "") + ("<- default output" if i == default_out else "")
        print(f"{i:>3}  {d['name'][:44]:<44} {apis[d['hostapi']]['name'][:22]:<22} {d['max_input_channels']:>2} "
              f"{d['max_output_channels']:>3} {mark}", file=out)
    print("\nPass a number or part of a name to --input / --output. If a name appears under several host APIs, "
          "the system's default host API (on Windows, usually MME) is used.", file=out)
    return 0


def find_device(sd, spec: Optional[str], kind: str) -> int:
    """Device index for a number, a name fragment, or (None) the system default input/output."""
    field = f"max_{kind}_channels"
    devices = sd.query_devices()
    if spec is None or str(spec).strip() == "":
        index = sd.default.device[0 if kind == "input" else 1]
        if index is None or index < 0:
            raise ValueError(f"No default {kind} device. Pick one with --{kind} (see --list-devices).")
        return int(index)
    if str(spec).strip().isdigit():
        index = int(spec)
        if not 0 <= index < len(devices) or devices[index][field] < 1:
            raise ValueError(f"Device #{index} is not an {kind} device (see --list-devices).")
        return index
    needle = str(spec).strip().lower()
    matches = [i for i, d in enumerate(devices) if needle in d["name"].lower() and d[field] > 0]
    if not matches:
        raise ValueError(f"No {kind} device matches {spec!r} (see --list-devices).")
    default_api = sd.default.hostapi
    preferred = [i for i in matches if devices[i]["hostapi"] == default_api]
    return (preferred or matches)[0]


# ---------------------------------------------------------------------------- the live loop

def run(client, input_spec: Optional[str] = None, output_spec: Optional[str] = None,
        voice: Optional[str] = None, target_audio: Optional[Path] = None, threshold: Optional[float] = None,
        silence_ms: int = 500, min_ms: int = 300, max_seconds: float = 15.0, log=None) -> int:
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    sd = _sounddevice()
    in_idx, out_idx = find_device(sd, input_spec, "input"), find_device(sd, output_spec, "output")
    in_info, out_info = sd.query_devices(in_idx), sd.query_devices(out_idx)
    in_sr, out_sr = int(in_info["default_samplerate"]), int(out_info["default_samplerate"])
    out_channels = 2 if out_info["max_output_channels"] >= 2 else 1
    log(f"Input:  #{in_idx} {in_info['name']} ({in_sr} Hz)")
    log(f"Output: #{out_idx} {out_info['name']} ({out_sr} Hz)")
    log(f"Target voice: {voice or (target_audio.name if target_audio else 'built-in voice')}")

    def convert(audio: Audio) -> Audio:
        return client.convert_clip(audio, target_audio, voice)

    log("Loading the voice conversion model and target voice (the first run downloads the model)…")
    t0 = time.perf_counter()
    noise = (np.random.default_rng(0).standard_normal(CONVERT_SAMPLE_RATE) * 0.02).astype(np.float32)
    convert((noise, CONVERT_SAMPLE_RATE))  # warm-up: loads the model and caches the target voice
    log(f"Ready in {time.perf_counter() - t0:.1f} s.")

    blocks: queue.Queue = queue.Queue()
    playback = PlaybackBuffer()
    frames_per_block = int(in_sr * BLOCK_SECONDS)

    def on_input(indata, frames, time_info, status):
        blocks.put(indata[:, 0].copy())

    def on_output(outdata, frames, time_info, status):
        outdata[:] = playback.read(frames)[:, None]

    def sink(audio: Audio) -> None:
        wav, sr = audio
        playback.put(np.clip(resample(wav, sr, out_sr), -1.0, 1.0))

    converter = LiveConverter(convert, sink, log)
    try:
        with sd.InputStream(device=in_idx, channels=1, samplerate=in_sr, blocksize=frames_per_block,
                            dtype="float32", callback=on_input), \
             sd.OutputStream(device=out_idx, channels=out_channels, samplerate=out_sr, dtype="float32",
                             callback=on_output):
            if threshold is None:
                log("Stay quiet for a second to measure the background noise…")
                levels = [rms(blocks.get()) for _ in range(int(1.0 / BLOCK_SECONDS))]
                threshold = auto_threshold(float(np.median(levels)))
            log(f"Speech threshold {threshold:.4f} (change it with --threshold). Listening: talk, pause, and the "
                "converted voice plays on the output. Ctrl+C to stop.")
            segmenter = PhraseSegmenter(in_sr, threshold, silence_ms, min_ms, max_seconds)
            while True:
                try:
                    block = blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                for phrase in segmenter.feed(block):
                    converter.submit(resample(phrase, in_sr, CONVERT_SAMPLE_RATE), CONVERT_SAMPLE_RATE)
    except KeyboardInterrupt:
        log("Stopped.")
    finally:
        converter.close(timeout=5.0)
    return 0
