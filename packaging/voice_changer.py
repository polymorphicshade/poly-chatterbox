"""PyInstaller entry point for PolyVoiceChanger.exe (see PolyVoiceChanger.spec)."""

import multiprocessing
import sys

from poly_chatterbox.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
