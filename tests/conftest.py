import io

import numpy as np
import pytest
import soundfile as sf

from poly_chatterbox.backends.mock import MockBackend
from poly_chatterbox.config import Settings
from poly_chatterbox.engine import Engine, set_engine


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.backend = "mock"
    s.data_dir = tmp_path / "data"
    return s


@pytest.fixture
def engine(settings):
    e = Engine(settings, MockBackend())
    set_engine(e)
    yield e
    set_engine(None)


def make_wav(seconds: float = 6.0, sr: int = 16000, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def ref_wav(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(make_wav())
    return path
