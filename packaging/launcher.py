"""PolyVoiceChanger launcher: a small EXE that installs the voice changer on first run, then starts it.

It carries uv (installs Python and the packages) and MinGit (uv needs git for the git dependencies).
Everything goes in %LOCALAPPDATA%\\PolyVoiceChanger: the tools, a managed Python, a venv with
poly-chatterbox[chatterbox,stream] from GitHub (torch with CUDA 12.8), and the app's voices and settings.
On later starts it checks the branch for a newer commit and offers to update.

Standard library only, so the launcher stays small. Built by packaging/build-launcher.ps1.
For testing, POLY_VOICE_CHANGER_HOME / _REPO / _BRANCH override where it installs and from what.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Optional

APP_NAME = "Poly Voice Changer"
REPO = os.environ.get("POLY_VOICE_CHANGER_REPO", "https://github.com/polymorphicshade/poly-chatterbox")
BRANCH = os.environ.get("POLY_VOICE_CHANGER_BRANCH", "dev")
HOME = Path(os.environ.get("POLY_VOICE_CHANGER_HOME")
            or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PolyVoiceChanger")
PYTHON = "3.12"
TORCH_BACKEND = "cu128"  # the CUDA build the app is tested with; needs NVIDIA driver 570+

TOOLS = HOME / "tools"
ENV = HOME / "env"
STATE = HOME / "install.json"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def bundled(name: str) -> Path:
    """A file packed into the EXE (or, from source, downloaded into packaging/downloads by the build script)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent / "downloads")) / name


def ensure_tools() -> None:
    """Unpack uv and MinGit next to the install, again whenever this launcher carries different versions."""
    stamp, wanted = TOOLS / "tools-version.txt", bundled("tools-version.txt").read_text().strip()
    if stamp.is_file() and stamp.read_text().strip() == wanted and (TOOLS / "uv.exe").is_file():
        return
    shutil.rmtree(TOOLS, ignore_errors=True)
    TOOLS.mkdir(parents=True)
    shutil.copy2(bundled("uv.exe"), TOOLS / "uv.exe")
    with zipfile.ZipFile(bundled("MinGit.zip")) as z:
        z.extractall(TOOLS / "git")
    stamp.write_text(wanted)


def tool_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{TOOLS / 'git' / 'cmd'};{env.get('PATH', '')}"
    env.update(UV_CACHE_DIR=str(HOME / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(HOME / "python"),
               UV_PYTHON_PREFERENCE="only-managed", UV_NO_PROGRESS="1", GIT_TERMINAL_PROMPT="0")
    return env


def remote_commit(timeout: float = 20.0) -> Optional[str]:
    """The branch's latest commit, or None when offline."""
    try:
        out = subprocess.run([str(TOOLS / "git" / "cmd" / "git.exe"), "ls-remote", REPO, f"refs/heads/{BRANCH}"],
                             capture_output=True, text=True, timeout=timeout, env=tool_env(), creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.split()[0] if out.returncode == 0 and out.stdout.strip() else None


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def installed() -> bool:
    return bool(load_state().get("commit")) and (ENV / "Scripts" / "pythonw.exe").is_file()


def launch_app() -> None:
    env = dict(os.environ, POLY_CHATTERBOX_DATA_DIR=str(HOME))
    subprocess.Popen([str(ENV / "Scripts" / "pythonw.exe"), "-m", "poly_chatterbox.gui"], env=env, cwd=str(HOME),
                     close_fds=True)


class Installer:
    """Window that shows the install (uv's output) and starts the app when it's done."""

    def __init__(self, root, commit: Optional[str], updating: bool):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.commit, self.updating = root, commit, updating
        self.lines: queue.Queue = queue.Queue()
        self.proc: Optional[subprocess.Popen] = None
        self.cancelled = False
        root.title(f"{APP_NAME} setup")
        root.minsize(620, 380)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        ttk.Label(frame, text=("Updating" if updating else "Installing") + f" {APP_NAME}",
                  font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.status = ttk.Label(frame, wraplength=590, text=(
            "Downloading the latest version…" if updating else
            "The first start downloads Python, PyTorch with CUDA and the voice changer: about 3 GB. "
            "This can take 10 minutes or more. It only happens once."))
        self.status.grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.bar = ttk.Progressbar(frame, mode="indeterminate")
        self.bar.grid(row=2, column=0, sticky="ew")
        self.bar.start(15)
        self.log = tk.Text(frame, height=12, wrap="none", state="disabled", font=("Consolas", 9))
        self.log.grid(row=3, column=0, sticky="nsew", pady=(8, 8))
        self.button = ttk.Button(frame, text="Cancel", command=self.cancel)
        self.button.grid(row=4, column=0, sticky="e")
        root.protocol("WM_DELETE_WINDOW", self.cancel)
        root.after(100, self.pump)
        threading.Thread(target=self.work, daemon=True).start()

    # -------------------------------------------------------------- worker thread

    def say(self, line: str) -> None:
        self.lines.put(("log", line))

    def run(self, *args) -> None:
        self.say("> " + " ".join(str(a) for a in args))
        self.proc = subprocess.Popen([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", env=tool_env(), cwd=str(HOME),
                                     creationflags=NO_WINDOW)
        for line in self.proc.stdout:
            self.say(line.rstrip())
        if self.proc.wait() != 0 and not self.cancelled:
            raise RuntimeError(f"{Path(str(args[0])).name} {args[1]} failed (exit code {self.proc.returncode}).")

    def work(self) -> None:
        try:
            HOME.mkdir(parents=True, exist_ok=True)
            self.say("Unpacking uv and git…")
            ensure_tools()
            commit = self.commit or remote_commit()
            if not commit:
                raise RuntimeError(f"Could not reach {REPO}. Check your internet connection and try again.")
            uv = TOOLS / "uv.exe"
            if not (ENV / "Scripts" / "python.exe").is_file():
                self.run(uv, "venv", ENV, "--python", PYTHON, "--clear")
            self.run(uv, "pip", "install", "--python", ENV / "Scripts" / "python.exe", "--torch-backend", TORCH_BACKEND,
                     f"poly-chatterbox[chatterbox,stream] @ git+{REPO}@{commit}")
            if self.cancelled:
                return
            STATE.write_text(json.dumps({"repo": REPO, "branch": BRANCH, "commit": commit}, indent=2), encoding="utf-8")
            self.lines.put(("done", None))
        except Exception as e:
            if not self.cancelled:
                self.lines.put(("failed", str(e)))

    # -------------------------------------------------------------- UI thread

    def pump(self) -> None:
        while True:
            try:
                kind, value = self.lines.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.append(value)
            elif kind == "done":
                self.append("Done. Starting the voice changer…")
                launch_app()
                self.root.after(1500, self.root.destroy)
                return
            elif kind == "failed":
                self.failed(value)
        self.root.after(100, self.pump)

    def append(self, line: str) -> None:
        with open(HOME / "install.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def failed(self, msg: str) -> None:
        self.bar.stop()
        self.append(f"Error: {msg}")
        self.status.configure(text=f"Setup failed: {msg}\nThe full log is in {HOME / 'install.log'}.")
        self.button.configure(text="Close", command=self.root.destroy)
        if self.updating and installed():
            from tkinter import ttk

            ttk.Button(self.button.master, text="Start the current version",
                       command=lambda: (launch_app(), self.root.destroy())).grid(row=4, column=0, sticky="w")

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.root.destroy()


def main() -> int:
    import tkinter as tk
    from tkinter import messagebox

    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    HOME.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    if installed():
        root.withdraw()
        ensure_tools()
        latest = remote_commit(timeout=8.0)
        if not latest or latest == load_state().get("commit") or not messagebox.askyesno(
                APP_NAME, "An update for the voice changer is available. Install it now?"):
            launch_app()
            root.destroy()
            return 0
        root.deiconify()
        Installer(root, latest, updating=True)
    else:
        Installer(root, None, updating=False)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
