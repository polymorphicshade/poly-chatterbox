try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # For Python <3.8

# Vendored into poly-chatterbox: the distribution is "poly-chatterbox", not "chatterbox-tts".
try:
    __version__ = version("chatterbox-tts")
except PackageNotFoundError:
    __version__ = "0.1.7"


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
