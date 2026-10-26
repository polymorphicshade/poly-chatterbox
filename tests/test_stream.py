import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poly_chatterbox.api import create_app
from poly_chatterbox.client import HttpClient, LocalClient
from poly_chatterbox.stream import (LiveConverter, PhraseSegmenter, PlaybackBuffer, auto_threshold, device_choices,
                                    find_device, resample)

from .conftest import make_wav

SR = 16000
BLOCK = 320  # 20 ms


def blocks(seconds, amplitude, freq=220.0, start=0.0):
    n = int(seconds * SR)
    t = (np.arange(n) + start * SR) / SR
    wav = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return [wav[i:i + BLOCK] for i in range(0, n - BLOCK + 1, BLOCK)]


def run_segmenter(seg, parts):
    phrases = []
    for part in parts:
        for b in part:
            phrases.extend(seg.feed(b))
    return phrases


def test_one_phrase_with_leadin_and_tail():
    seg = PhraseSegmenter(SR, threshold=0.05)
    phrases = run_segmenter(seg, [blocks(0.5, 0.001), blocks(1.0, 0.3), blocks(0.8, 0.001)])
    assert len(phrases) == 1
    # 1 s of speech + ~200 ms lead-in + ~150 ms tail
    assert 1.3 <= len(phrases[0]) / SR <= 1.5
    assert not seg.speaking


def test_two_phrases_split_at_the_pause():
    seg = PhraseSegmenter(SR, threshold=0.05, silence_ms=400)
    phrases = run_segmenter(seg, [blocks(0.6, 0.3), blocks(0.6, 0.001), blocks(0.6, 0.3), blocks(0.6, 0.001)])
    assert len(phrases) == 2


def test_short_clicks_are_dropped():
    seg = PhraseSegmenter(SR, threshold=0.05, min_ms=300)
    assert run_segmenter(seg, [blocks(0.1, 0.5), blocks(1.0, 0.001)]) == []


def test_long_speech_is_cut_at_max_seconds():
    seg = PhraseSegmenter(SR, threshold=0.05, max_seconds=3.0)
    phrases = run_segmenter(seg, [blocks(10.0, 0.3)])
    assert len(phrases) >= 3 and all(len(p) / SR <= 3.1 for p in phrases)


def test_flush_returns_unfinished_phrase():
    seg = PhraseSegmenter(SR, threshold=0.05)
    assert run_segmenter(seg, [blocks(1.0, 0.3)]) == []
    assert seg.speaking and len(seg.flush()) == 1 and not seg.speaking


def test_auto_threshold_has_a_floor():
    assert auto_threshold(0.0) == pytest.approx(0.006)
    assert auto_threshold(0.01) == pytest.approx(0.04)


def test_playback_buffer_streams_in_order_and_pads_with_silence():
    buf = PlaybackBuffer()
    buf.put(np.arange(1, 6, dtype=np.float32))
    buf.put(np.arange(6, 9, dtype=np.float32))
    assert list(buf.read(4)) == [1, 2, 3, 4]
    assert list(buf.read(6)) == [5, 6, 7, 8, 0, 0]
    assert buf.queued == 0 and list(buf.read(2)) == [0, 0]


def test_resample_lengths():
    wav = np.sin(np.linspace(0, 100, 24000)).astype(np.float32)
    assert len(resample(wav, 24000, 48000)) == 48000
    assert len(resample(wav, 24000, 16000)) == 16000
    assert resample(wav, 24000, 24000) is not None


def test_live_converter_keeps_order_and_survives_errors():
    logs, played = [], []

    def convert(audio):
        wav, sr = audio
        if wav[0] == 99:
            raise RuntimeError("boom")
        return wav * 2, sr

    live = LiveConverter(convert, lambda a: played.append(float(a[0][0])), logs.append)
    for first in (1.0, 99.0, 3.0):
        live.submit(np.full(1600, first, dtype=np.float32), SR)
    live.close()
    assert played == [2.0, 6.0]
    assert any("conversion failed: boom" in line for line in logs)


def test_live_converter_warns_when_falling_behind():
    logs = []
    live = LiveConverter(lambda a: (time.sleep(0.05), a)[1], lambda a: None, logs.append, warn_backlog=1)
    for _ in range(4):
        live.submit(np.ones(1600, dtype=np.float32), SR)
    live.close()
    assert any("falling behind" in line for line in logs)


class FakeSD:
    def __init__(self):
        self.default = SimpleNamespace(device=(1, 5), hostapi=0)
        self._devices = [
            dict(name="Microsoft Sound Mapper - Input", hostapi=0, max_input_channels=2, max_output_channels=0),
            dict(name="Microphone (Shure MV7+)", hostapi=0, max_input_channels=1, max_output_channels=0),
            dict(name="CABLE Output (VB-Audio Virtual", hostapi=0, max_input_channels=16, max_output_channels=0),
            dict(name="Mapper - Output", hostapi=0, max_input_channels=0, max_output_channels=2),
            dict(name="CABLE Input (VB-Audio Virtual C", hostapi=0, max_input_channels=0, max_output_channels=16),
            dict(name="Speakers", hostapi=0, max_input_channels=0, max_output_channels=2),
            dict(name="CABLE Input (VB-Audio Virtual Cable)", hostapi=1, max_input_channels=0, max_output_channels=16),
        ]

    def query_devices(self, index=None):
        return self._devices if index is None else self._devices[index]


def test_find_device():
    sd = FakeSD()
    assert find_device(sd, None, "input") == 1 and find_device(sd, None, "output") == 5
    assert find_device(sd, "cable input", "output") == 4  # the default host API wins over the duplicate
    assert find_device(sd, "6", "output") == 6
    assert find_device(sd, "shure", "input") == 1
    with pytest.raises(ValueError, match="No output device matches"):
        find_device(sd, "shure", "output")  # a mic is not an output
    with pytest.raises(ValueError, match="not an input"):
        find_device(sd, "4", "input")


def test_convert_clip_local_and_over_http(engine, ref_wav):
    from poly_chatterbox.audio import load_audio

    clip = load_audio(make_wav(2, sr=16000))
    wav, sr = LocalClient(engine).convert_clip(clip, ref_wav, None)
    assert sr == 24000 and abs(len(wav) / sr - 2) < 0.05

    http = HttpClient("http://testserver")
    http.http = TestClient(create_app(engine))
    wav, sr = http.convert_clip(clip, None, "podcast")
    assert sr == 24000 and abs(len(wav) / sr - 2) < 0.05
    wav, sr = http.convert_clip(clip, ref_wav, None)
    assert len(wav) > 0
    assert any(c[0] == "convert" and c[1]["target"] for c in engine.backend.calls)


def test_end_to_end_pipeline_without_hardware(engine, ref_wav):
    """Mic blocks -> phrases -> conversion -> playback buffer, the same wiring `stream` uses."""
    client = LocalClient(engine)
    playback = PlaybackBuffer()
    live = LiveConverter(lambda a: client.convert_clip(a, ref_wav, None),
                         lambda a: playback.put(resample(a[0], a[1], 48000)), lambda m: None)
    seg = PhraseSegmenter(SR, threshold=0.05)
    for part in (blocks(0.5, 0.001), blocks(1.0, 0.3), blocks(0.8, 0.001), blocks(1.0, 0.3, 330.0), blocks(0.8, 0.001)):
        for b in part:
            for phrase in seg.feed(b):
                live.submit(phrase, SR)
    live.close()
    assert 2 * 1.3 * 48000 < playback.queued < 2 * 1.5 * 48000  # two phrases, ~1.35 s each, at 48 kHz


def test_device_choices_lists_each_device_once_on_the_default_host_api():
    sd = FakeSD()
    assert [i for i, _ in device_choices(sd, "output")] == [3, 4, 5]
    assert [name for _, name in device_choices(sd, "input")][1] == "Microphone (Shure MV7+)"


class FakeStreamSD(FakeSD):
    """FakeSD whose input stream feeds a loud tone, then silence, to the callback in real time."""

    def __init__(self):
        super().__init__()
        for d in self._devices:
            d["default_samplerate"] = SR
        outer = self

        class InputStream:
            def __init__(self, callback, blocksize, **kw):
                self.callback, self.blocksize, self.running = callback, blocksize, False

            def __enter__(self):
                self.running = True
                threading.Thread(target=self._feed, daemon=True).start()
                return self

            def _feed(self):
                for b in blocks(0.6, 0.3) + blocks(5.0, 0.001):
                    if not self.running:
                        return
                    self.callback(b[:, None], len(b), None, None)
                    time.sleep(BLOCK / SR / 4)

            def __exit__(self, *exc):
                self.running = False
                outer.closed = True

        class OutputStream:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

        self.InputStream, self.OutputStream = InputStream, OutputStream
        self.closed = False


def test_run_converts_and_stops_when_the_stop_event_is_set(engine, monkeypatch):
    from poly_chatterbox import stream

    sd = FakeStreamSD()
    monkeypatch.setattr(stream, "_sounddevice", lambda: sd)
    logs, stop = [], threading.Event()

    def log(msg):
        logs.append(msg)
        if "converted in" in msg:
            stop.set()

    done = threading.Thread(target=stream.run, args=(LocalClient(engine), "1", "4"),
                            kwargs=dict(voice="podcast", threshold=0.05, log=log, stop=stop))
    done.start()
    done.join(timeout=10)
    assert not done.is_alive() and sd.closed
    assert any("converted in" in m for m in logs) and logs[-1] == "Stopped."
