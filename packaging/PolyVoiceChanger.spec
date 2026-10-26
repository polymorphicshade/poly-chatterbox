# PyInstaller spec for PolyVoiceChanger.exe: the live voice changer GUI with torch (CUDA) bundled.
# Build it with packaging/build.ps1. Output: dist/PolyVoiceChanger/ (a folder; share it zipped).
from pathlib import PurePath

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = SPECPATH + "/.."

datas = []
for package in ("perth", "s3tokenizer", "librosa"):  # model weights / mel filters / example registry
    datas += collect_data_files(package)
# diffusers and transformers check their dependencies' installed versions at import time.
for package in ("diffusers", "transformers", "torch", "librosa", "s3tokenizer", "resemble-perth"):
    datas += copy_metadata(package, recursive=True)

hiddenimports = (collect_submodules("chatterbox")
                 + ["poly_chatterbox.backends.chatterbox", "poly_chatterbox.backends.mock"])

a = Analysis(
    [ROOT + "/packaging/voice_changer.py"],
    pathex=[ROOT + "/src"],
    datas=datas,
    hiddenimports=hiddenimports,
    # The web server, the training/eval extras and the other app tools aren't used by the voice changer.
    excludes=["fastapi", "uvicorn", "starlette", "speechbrain", "matplotlib", "IPython", "pytest", "tensorboard"],
    # Keep the .py sources next to the bytecode: TorchScript, numba and transformers read them at runtime.
    module_collection_mode={name: "pyz+py" for name in
                            ("torch", "torchaudio", "transformers", "diffusers", "chatterbox", "s3tokenizer",
                             "librosa", "numba", "perth", "conformer", "einops")},
    noarchive=False,
)
# torch loads its CUDA DLLs from torch/lib; drop the second copies PyInstaller puts at the top level (~3.5 GB).
torch_libs = {PurePath(dest).name.lower() for dest, _, _ in a.binaries if PurePath(dest).parts[:2] == ("torch", "lib")}
a.binaries = [b for b in a.binaries if not (len(PurePath(b[0]).parts) == 1 and b[0].lower() in torch_libs)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PolyVoiceChanger",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PolyVoiceChanger", upx=False)
