"""Voice Changer: a small desktop window for live voice conversion (``poly-chatterbox stream``).

Pick a voice (or add one from a clip), a microphone and an output device, typically the "CABLE Input"
of a virtual audio cable, and press Start. Apps such as Discord then use "CABLE Output" as their
microphone and hear the converted voice. Everything runs in-process on this PC's GPU.

Run it with ``uv run poly-chatterbox-voice-changer``, or as the packaged ``PolyVoiceChanger.exe``
(see ``packaging/``). ``--check`` loads the model, converts a test clip and exits (results in the log file).
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import Callable, Optional

APP_NAME = "Poly Voice Changer"
VB_CABLE_URL = "https://vb-audio.com/Cable/"
FROZEN = getattr(sys, "frozen", False)
# A windowed EXE or pythonw (as the launcher starts it) has no stdout/stderr: log to a file instead.
LOG_TO_FILE = FROZEN or sys.stdout is None


def _app_dir() -> Path:
    """Where the packaged app keeps voices, settings and its log. From source, the usual ``data`` folder."""
    if FROZEN:
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PolyVoiceChanger"
    from .config import get_settings

    return get_settings().data_dir


def _setup_frozen_environment() -> Path:
    """Point the data dir at the user's profile (EXE) and send stdout/stderr to a log file when there are none."""
    app_dir = _app_dir()
    app_dir.mkdir(parents=True, exist_ok=True)
    log_path = app_dir / "voice-changer.log"
    if FROZEN:
        os.environ.setdefault("POLY_CHATTERBOX_DATA_DIR", str(app_dir))
    if LOG_TO_FILE:
        log = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    return log_path


class Prefs:
    """The last voice and devices, remembered between runs."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save(self, **values) -> None:
        self.data.update(values)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            pass


class VoiceChangerApp:
    def __init__(self, root, prefs: Prefs, log_path: Path):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root, self.prefs, self.log_path = root, prefs, log_path
        self._events: queue.Queue = queue.Queue()  # (callable,) posted from worker threads, run on the UI thread
        self._client = None
        self._client_lock = threading.Lock()
        self._stop: Optional[threading.Event] = None
        self._voice_ids: dict[str, str] = {}
        self._inputs: dict[str, int] = {}
        self._outputs: dict[str, int] = {}

        root.title(APP_NAME)
        root.minsize(560, 460)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, pad=12)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Voice").grid(row=0, column=0, sticky="w", pady=4)
        self.voice = ttk.Combobox(frame, state="readonly")
        self.voice.grid(row=0, column=1, sticky="ew", pady=4)
        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=2, sticky="e", padx=(6, 0))
        ttk.Button(buttons, text="Add voice…", command=self.add_voice).pack(side="left")
        ttk.Button(buttons, text="Remove", command=self.remove_voice).pack(side="left", padx=(4, 0))

        ttk.Label(frame, text="Microphone").grid(row=1, column=0, sticky="w", pady=4)
        self.input = ttk.Combobox(frame, state="readonly")
        self.input.grid(row=1, column=1, sticky="ew", pady=4)
        self.refresh_button = ttk.Button(frame, text="Refresh devices", command=self.refresh_devices)
        self.refresh_button.grid(row=1, column=2, sticky="e", padx=(6, 0))

        ttk.Label(frame, text="Output").grid(row=2, column=0, sticky="w", pady=4)
        self.output = ttk.Combobox(frame, state="readonly")
        self.output.grid(row=2, column=1, sticky="ew", pady=4)
        self.cable_hint = ttk.Label(frame, foreground="#1a5fb4", cursor="hand2", wraplength=520)
        self.cable_hint.grid(row=3, column=1, columnspan=2, sticky="w")
        self.cable_hint.bind("<Button-1>", lambda e: webbrowser.open(VB_CABLE_URL) if self._no_cable else None)
        self._no_cable = False

        advanced = ttk.LabelFrame(frame, text="Advanced", padding=8)
        advanced.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(advanced, text="Pause that ends a phrase (ms)").grid(row=0, column=0, sticky="w")
        self.silence = tk.StringVar(value=str(prefs.get("silence_ms", 500)))
        ttk.Spinbox(advanced, from_=200, to=2000, increment=50, width=7, textvariable=self.silence).grid(row=0, column=1, padx=(6, 18))
        ttk.Label(advanced, text="Speech threshold (blank = auto)").grid(row=0, column=2, sticky="w")
        self.threshold = tk.StringVar(value=prefs.get("threshold", ""))
        ttk.Entry(advanced, width=8, textvariable=self.threshold).grid(row=0, column=3, padx=(6, 0))

        controls = ttk.Frame(frame)
        controls.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.start_button = ttk.Button(controls, text="Start", command=self.toggle, state="disabled")
        self.start_button.pack(side="left")
        self.status = ttk.Label(controls, text="Loading…")
        self.status.pack(side="left", padx=10)

        self.log_box = tk.Text(frame, height=12, wrap="word", state="disabled", font=("Consolas", 9))
        self.log_box.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        frame.rowconfigure(6, weight=1)

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._pump)
        self.refresh_voices()
        self.refresh_devices()
        self.log("Tip: in Discord (or any app), choose \"CABLE Output\" as the microphone to hear the converted voice.")
        self._background(self._load_engine, self._engine_loaded)

    # ------------------------------------------------------------------ threading helpers

    def _post(self, fn: Callable, *args) -> None:
        self._events.put((fn, args))

    def _pump(self) -> None:
        while True:
            try:
                fn, args = self._events.get_nowait()
            except queue.Empty:
                break
            fn(*args)
        self.root.after(100, self._pump)

    def _background(self, work: Callable, done: Optional[Callable] = None, failed: Optional[Callable] = None) -> None:
        """Run ``work`` on a thread; ``done(result)`` or ``failed()`` then runs on the UI thread. Errors are shown."""
        def run():
            try:
                result = work()
            except Exception as e:
                traceback.print_exc()
                self._post(self._error, e)
                if failed:
                    self._post(failed)
                return
            if done:
                self._post(done, result)

        threading.Thread(target=run, daemon=True).start()

    def log(self, msg: str) -> None:
        """Thread-safe: append a line to the log box (and the log file)."""
        print(msg, flush=True)
        self._post(self._append_log, msg)

    def _append_log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _error(self, e: Exception) -> None:
        from tkinter import messagebox

        msg = e.args[0] if isinstance(e, (ValueError, LookupError)) and e.args else str(e) or type(e).__name__
        self._append_log(f"Error: {msg}")
        messagebox.showerror(APP_NAME, f"{msg}\n\nDetails are in {self.log_path}" if LOG_TO_FILE else msg)
        if self._stop is None:
            self.status.configure(text="")

    def client(self):
        with self._client_lock:
            if self._client is None:
                from .client import LocalClient

                self._client = LocalClient()
            return self._client

    # ------------------------------------------------------------------ startup

    def _load_engine(self):
        self.log("Loading the AI libraries…")
        client = self.client()
        import torch

        return client, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    def _engine_loaded(self, result) -> None:
        _, gpu = result
        if gpu:
            self.log(f"GPU: {gpu}")
        else:
            self.log("Warning: no NVIDIA GPU found. Conversion runs on the CPU, which is too slow for live use.")
        self.start_button.configure(state="normal")
        self.status.configure(text="Ready. Press Start (the first start downloads the model, about 1 GB).")

    # ------------------------------------------------------------------ voices

    def refresh_voices(self, select: Optional[str] = None) -> None:
        from .config import get_settings
        from .library import VoiceLibrary

        voices = VoiceLibrary(get_settings().voices_dir).list()
        voices.sort(key=lambda v: (v.kind != "cloned", v.kind != "builtin"))  # yours first, then built-in, then demos
        self._voice_ids = {}
        for v in voices:
            label = v.name if v.kind != "sample" else f"{v.name} (demo)"
            self._voice_ids[label] = v.id
        self.voice.configure(values=list(self._voice_ids))
        wanted = select or self.prefs.get("voice")
        labels = [l for l, vid in self._voice_ids.items() if vid == wanted] or list(self._voice_ids)
        self.voice.set(labels[0])

    def selected_voice(self) -> str:
        return self._voice_ids.get(self.voice.get(), "default")

    def add_voice(self) -> None:
        AddVoiceDialog(self)

    def save_voice(self, name: str, clip: Path, language: str, done: Callable[[], None],
                   failed: Callable[[], None]) -> None:
        self.log(f"Saving voice {name!r} from {clip.name}…")

        def saved(voice):
            self.log(f"Saved voice {voice['name']!r}.")
            self.refresh_voices(select=voice["id"])
            self.prefs.save(voice=voice["id"])
            done()

        self._background(lambda: self.client().save_voice(name, clip, language), saved, failed)

    def remove_voice(self) -> None:
        from tkinter import messagebox

        voice_id, label = self.selected_voice(), self.voice.get()
        if not voice_id or voice_id == "default" or label.endswith("(demo)"):
            messagebox.showinfo(APP_NAME, "Only voices you added can be removed.")
            return
        if messagebox.askyesno(APP_NAME, f"Remove the voice {label!r}?"):
            from .config import get_settings
            from .library import VoiceLibrary

            try:
                VoiceLibrary(get_settings().voices_dir).remove(voice_id)
            except (ValueError, LookupError) as e:
                self._error(e)
                return
            self.log(f"Removed voice {label!r}.")
            self.refresh_voices()

    # ------------------------------------------------------------------ devices

    def refresh_devices(self) -> None:
        try:
            from .stream import _sounddevice, device_choices

            sd = _sounddevice()
            sd._terminate()  # re-scan, so devices plugged in (or a cable installed) since startup show up
            sd._initialize()
            inputs, outputs = device_choices(sd, "input"), device_choices(sd, "output")
            default_in = sd.default.device[0]
        except Exception as e:
            self._error(e)
            return
        self._inputs = {name: i for i, name in inputs}
        self._outputs = {name: i for i, name in outputs}
        self.input.configure(values=list(self._inputs))
        self.output.configure(values=list(self._outputs))

        def pick(box, names: dict, saved, fallback):
            if saved in names:
                box.set(saved)
            elif fallback:
                box.set(fallback)
            elif names:
                box.set(next(iter(names)))

        default_in_name = next((n for n, i in self._inputs.items() if i == default_in), None)
        cable = next((n for n in self._outputs if "cable input" in n.lower()), None)
        pick(self.input, self._inputs, self.prefs.get("input"), default_in_name)
        pick(self.output, self._outputs, self.prefs.get("output"), cable)
        self._no_cable = cable is None
        self.cable_hint.configure(
            text="No virtual audio cable found. Click here to download VB-CABLE (free), install it, restart "
                 "your PC and press Refresh devices." if self._no_cable else
                 "The converted voice plays into the output; apps hear it on the matching \"CABLE Output\" mic.")

    # ------------------------------------------------------------------ streaming

    def toggle(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self.start_button.configure(state="disabled")
            self.status.configure(text="Stopping…")
            return
        try:
            silence_ms = int(self.silence.get())
            threshold = float(self.threshold.get()) if self.threshold.get().strip() else None
        except ValueError:
            self._error(ValueError("The pause must be a whole number of ms and the threshold a number like 0.02."))
            return
        mic, out = self.input.get(), self.output.get()
        if mic not in self._inputs or out not in self._outputs:
            self._error(ValueError("Pick a microphone and an output device."))
            return
        voice = self.selected_voice()
        self.prefs.save(voice=voice, input=mic, output=out, silence_ms=silence_ms, threshold=self.threshold.get().strip())

        from . import stream

        stop = self._stop = threading.Event()
        self.start_button.configure(text="Stop")
        self.status.configure(text="Starting… stay quiet for a moment while the background noise is measured.")
        self._set_inputs("disabled")

        def run():
            stream.run(self.client(), str(self._inputs[mic]), str(self._outputs[out]), voice=voice,
                       threshold=threshold, silence_ms=silence_ms, log=self._stream_log, stop=stop)

        self._background(run, lambda _: self._stopped(stop), lambda: self._stopped(stop))

    def _stream_log(self, msg: str) -> None:
        self.log(msg)
        if msg.startswith("Speech threshold"):
            self._post(lambda: self.status.configure(text="Live: talk, pause, and the converted voice plays."))

    def _stopped(self, stop: threading.Event) -> None:
        if self._stop is stop:
            self._stop = None
            self.start_button.configure(text="Start", state="normal")
            self.status.configure(text="Stopped.")
            self._set_inputs("readonly")

    def _set_inputs(self, state: str) -> None:
        for box in (self.voice, self.input, self.output):
            box.configure(state=state)
        self.refresh_button.configure(state="disabled" if state == "disabled" else "normal")

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        self.root.destroy()


class AddVoiceDialog:
    """Name + reference clip + language; saves the clip as a cloned voice."""

    def __init__(self, app: VoiceChangerApp):
        tk, ttk = app.tk, app.ttk
        from .languages import LANGUAGES

        self.app = app
        self.win = win = tk.Toplevel(app.root)
        win.title("Add voice")
        win.transient(app.root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Name").grid(row=0, column=0, sticky="w", pady=4)
        self.name = tk.StringVar()
        name_entry = ttk.Entry(frame, textvariable=self.name, width=36)
        name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(frame, text="Voice clip").grid(row=1, column=0, sticky="w", pady=4)
        self.clip = tk.StringVar()
        ttk.Entry(frame, textvariable=self.clip, width=36).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse…", command=self.browse).grid(row=1, column=2, padx=(6, 0))

        ttk.Label(frame, text="Language").grid(row=2, column=0, sticky="w", pady=4)
        self.languages = {name: code for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])}
        self.language = ttk.Combobox(frame, state="readonly", values=list(self.languages), width=20)
        self.language.set(LANGUAGES.get("en", next(iter(self.languages))))
        self.language.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(frame, text="Use 5 to 60 seconds of one person talking clearly, without music (WAV, MP3, FLAC or OGG).",
                  wraplength=380, foreground="#555").grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 8))
        row = ttk.Frame(frame)
        row.grid(row=4, column=0, columnspan=3, sticky="e")
        self.save_button = ttk.Button(row, text="Save", command=self.save)
        self.save_button.pack(side="left")
        ttk.Button(row, text="Cancel", command=win.destroy).pack(side="left", padx=(6, 0))
        name_entry.focus_set()
        win.grab_set()

    def browse(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.win, title="Choose a voice clip",
                                          filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")])
        if path:
            self.clip.set(path)
            if not self.name.get().strip():
                self.name.set(Path(path).stem.replace("_", " ").title())

    def save(self) -> None:
        from tkinter import messagebox

        name, clip = self.name.get().strip(), Path(self.clip.get().strip())
        if not name or not clip.is_file():
            messagebox.showerror("Add voice", "Enter a name and choose a voice clip.", parent=self.win)
            return
        self.save_button.configure(state="disabled", text="Saving…")
        # If saving fails, the error is shown and the dialog stays open to try again.
        self.app.save_voice(name, clip, self.languages[self.language.get()], self.win.destroy, self._reenable)

    def _reenable(self) -> None:
        if self.win.winfo_exists():
            self.save_button.configure(state="normal", text="Save")


def check() -> int:
    """Load the voice conversion model and convert a test clip: verifies a packaged build works on this PC."""
    import time

    import numpy as np

    from .client import LocalClient

    t0 = time.perf_counter()
    try:
        import torch

        print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
        noise = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.05).astype(np.float32)
        wav, sr = LocalClient().convert_clip((noise, 16000))
        print(f"check ok: converted 2.0 s -> {len(wav) / sr:.2f} s at {sr} Hz in {time.perf_counter() - t0:.1f} s", flush=True)
        import tempfile

        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:  # an MP3 target voice, as "Add voice…" would load
            clip = Path(tmp) / "voice.mp3"
            t = np.arange(16000 * 6) / 16000
            sf.write(str(clip), (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 16000, format="MP3")
            wav, sr = LocalClient().convert_clip((noise, 16000), clip)
        print(f"check ok: converted with an MP3 target voice -> {len(wav) / sr:.2f} s", flush=True)
        from . import stream

        sd = stream._sounddevice()
        print(f"audio devices: {len(stream.device_choices(sd, 'input'))} inputs, "
              f"{len(stream.device_choices(sd, 'output'))} outputs", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("check FAILED", flush=True)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    log_path = _setup_frozen_environment()
    if "--check" in argv:
        return check()
    import tkinter as tk

    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
    except Exception:
        pass
    root = tk.Tk()
    VoiceChangerApp(root, Prefs(_app_dir() / "voice-changer.json"), log_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
