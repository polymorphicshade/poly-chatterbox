# PyInstaller spec for the small PolyVoiceChanger launcher (one file): it installs the voice changer on first
# run with the uv and MinGit it carries. Build it with packaging/build-launcher.ps1, which downloads those two.
DOWNLOADS = SPECPATH + "/downloads"

a = Analysis(
    [SPECPATH + "/launcher.py"],
    datas=[(DOWNLOADS + "/uv.exe", "."), (DOWNLOADS + "/MinGit.zip", "."), (DOWNLOADS + "/tools-version.txt", ".")],
    excludes=["numpy", "pydoc", "unittest", "xmlrpc", "sqlite3", "lib2to3"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PolyVoiceChanger",
    console=False,
    upx=False,
)
