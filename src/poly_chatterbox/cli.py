"""Command-line interface: ``poly-chatterbox <command> ...`` (also ``python -m poly_chatterbox``).

Every core feature has a subcommand. By default models run in-process; pass ``--server URL``
(or set ``POLY_CHATTERBOX_SERVER``) to send requests to a running ``poly-chatterbox serve``
instead, which avoids reloading models on every call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .backends.base import GENERATION_OPTIONS, MULTILINGUAL, MULTILINGUAL_VERSIONS, NANO, ORIGINAL, TTS_MODELS, TURBO
from .languages import LANGUAGES
from .presets import EMOTION_PRESETS


def _read_text(args) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text in (None, "-"):
        if sys.stdin.isatty():
            raise SystemExit("error: provide TEXT, --text-file, or pipe text on stdin")
        return sys.stdin.read()
    return args.text


def _generation(args) -> Optional[dict]:
    gen = {k: getattr(args, k, None) for k in GENERATION_OPTIONS}
    gen = {k: v for k, v in gen.items() if v is not None}
    return gen or None


def _write_output(args, wav: bytes, meta: dict) -> None:
    if args.output == "-":
        sys.stdout.buffer.write(wav)
        sys.stdout.buffer.flush()
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(wav)
        meta = {**meta, "output": str(out)}
    if args.json:
        print(json.dumps(meta, ensure_ascii=False), file=sys.stderr if args.output == "-" else sys.stdout)
    else:
        where = "stdout" if args.output == "-" else args.output
        print(f"wrote {where} ({meta.get('duration')}s audio, generated in {meta.get('elapsed')}s)", file=sys.stderr)
        for w in meta.get("warnings", []):
            print(f"warning: {w}", file=sys.stderr)


# ---------------------------------------------------------------------------- parser

def _add_output(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", default="output.wav", help="Output WAV path, or '-' for stdout (default: output.wav).")
    p.add_argument("--json", action="store_true", help="Print result metadata as JSON.")


def _add_tts(p: argparse.ArgumentParser, models: tuple[str, ...]) -> None:
    p.add_argument("text", nargs="?", help="Text to speak ('-' or omitted: read stdin). Supports [pause 500ms], "
                                           "[settings …]…[/settings] and [voice: id]…[/voice].")
    p.add_argument("--text-file", help="Read the text from a file.")
    p.add_argument("--prompt", metavar="ID", help="Start from a saved prompt (see `prompts list`): its text, voice and "
                                                  "generation options are used unless you give your own.")
    p.add_argument("--emotion", choices=list(EMOTION_PRESETS), metavar="PRESET",
                   help=f"Emotion preset ({', '.join(EMOTION_PRESETS)}): a bundle of generation options. "
                        "Options you give explicitly win over it. See `presets`.")

    p.add_argument("--translate-from", metavar="LANG",
                   help="Language the text is written in, code or name: it is translated into the spoken language first "
                        "(English for the English models, -l for multilingual). [tags] and markup are kept.")
    voice = p.add_mutually_exclusive_group()
    voice.add_argument("--ref-audio", type=Path, help="Reference clip of the voice to clone, >= 5 s (WAV/FLAC/OGG/MP3).")
    voice.add_argument("--voice", help="Library voice id (see `poly-chatterbox voices list`).")
    _add_output(p)
    g = p.add_argument_group("generation options", "Unset options use the model defaults.")
    _add_generation_options(g, models)


def _add_generation_options(g, models: tuple[str, ...]) -> None:
    for name, opt in GENERATION_OPTIONS.items():
        defaults = {m: opt["defaults"][m] for m in models if m in opt["defaults"]}
        if not defaults:
            continue
        values = {d for d in defaults.values() if d is not None}
        default = f" (default: {values.pop()})" if len(values) == 1 else ""
        # argparse treats '%' as a format character in help strings.
        help_text = opt["description"].replace("%", "%%") + default
        flag = f"--{name.replace('_', '-')}"
        if opt["type"] == "bool":
            g.add_argument(flag, action=argparse.BooleanOptionalAction, help=help_text)
        else:
            g.add_argument(flag, type=float if opt["type"] == "float" else int, metavar=name.upper(), help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poly-chatterbox",
                                     description="Chatterbox TTS: voice cloning in English and 23 languages, voice conversion.")
    parser.add_argument("--server", default=os.environ.get("POLY_CHATTERBOX_SERVER"),
                        help="Use a running poly-chatterbox server (e.g. http://127.0.0.1:8000) instead of loading models locally.")
    parser.add_argument("--insecure", action="store_true",
                        help="With --server: don't verify the TLS certificate (for the self-signed HTTPS stack).")
    parser.add_argument("--backend", choices=["chatterbox", "mock"], help="Local backend (default: chatterbox; env POLY_CHATTERBOX_BACKEND).")
    parser.add_argument("--device", help="Local device, e.g. cuda, cuda:1, mps or cpu (default: auto).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show model loading logs.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("original", help="Chatterbox (English, 500M): cloning with exaggeration and CFG/pace control.")
    _add_tts(p, (ORIGINAL,))

    p = sub.add_parser("turbo", help="Chatterbox-Turbo (English, 350M): fast, with tags like [laugh] and [cough].")
    _add_tts(p, (TURBO,))

    p = sub.add_parser("nano", help="Chatterbox-Nano (English, 110M): Turbo in a CPU-friendly size.")
    _add_tts(p, (NANO,))

    p = sub.add_parser("multilingual", help="Chatterbox Multilingual: 23 languages.")
    _add_tts(p, (MULTILINGUAL,))
    p.add_argument("-l", "--language", help=f"Language of the text, code or name ({', '.join(LANGUAGES)}). "
                                            "Required unless --prompt has one.")
    p.add_argument("--model-version", choices=MULTILINGUAL_VERSIONS, help="Checkpoint (default: v3, or POLY_CHATTERBOX_MULTILINGUAL_VERSION).")

    p = sub.add_parser("convert", help="Voice conversion: re-voice a recording with another voice.")
    p.add_argument("source", type=Path, help="Speech to convert (WAV/FLAC/OGG/MP3).")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--target-audio", type=Path, help="Clip of the target voice, >= 5 s.")
    target.add_argument("--voice", help="Library voice id as the target (default: the built-in voice).")
    _add_output(p)

    p = sub.add_parser("voices", help="List or manage the voice library.")
    vs = p.add_subparsers(dest="voices_command", required=True, metavar="ACTION")
    pl = vs.add_parser("list", help="List voices.")
    pl.add_argument("--kind", choices=["builtin", "sample", "cloned"])
    pl.add_argument("--language")
    pl.add_argument("--json", action="store_true")
    pa = vs.add_parser("add", help="Save a cloned voice from a reference clip.")
    pa.add_argument("name")
    pa.add_argument("--ref-audio", type=Path, required=True, help="Clip of the voice, >= 5 s.")
    pa.add_argument("-l", "--language", help="Language spoken in the clip.")
    pa.add_argument("--description")
    pa.add_argument("--denoise", action="store_true", help="Remove background noise from the clip before saving it.")
    pr = vs.add_parser("remove", help="Delete a saved cloned voice.")
    pr.add_argument("voice_id")

    p = sub.add_parser("prompts", help="List, save and remove saved prompts.")
    ps = p.add_subparsers(dest="prompts_command", required=True, metavar="ACTION")
    pl = ps.add_parser("list", help="List saved prompts.")
    pl.add_argument("--json", action="store_true")
    pw = ps.add_parser("show", help="Print a saved prompt.")
    pw.add_argument("prompt_id")
    pw.add_argument("--json", action="store_true", help="Print everything as JSON instead of just the text.")
    pa = ps.add_parser("save", help="Save a prompt (same name: replaces it).")
    pa.add_argument("name")
    pa.add_argument("text", nargs="?", help="Prompt text ('-' or omitted: read stdin).")
    pa.add_argument("--text-file", help="Read the text from a file.")
    pa.add_argument("--mode", choices=TTS_MODELS, help="Model to speak it with.")
    pa.add_argument("-l", "--language", help="Language (multilingual).")
    pa.add_argument("--model-version", choices=MULTILINGUAL_VERSIONS, help="Multilingual checkpoint.")
    pa.add_argument("--voice", help="Library voice id.")
    ga = pa.add_argument_group("generation options", "Saved with the prompt; unset options use the model defaults.")
    for name, opt in GENERATION_OPTIONS.items():
        flag = f"--{name.replace('_', '-')}"
        if opt["type"] == "bool":
            ga.add_argument(flag, action=argparse.BooleanOptionalAction)
        else:
            ga.add_argument(flag, type=float if opt["type"] == "float" else int, metavar=name.upper())
    pr = ps.add_parser("remove", help="Delete a saved prompt.")
    pr.add_argument("prompt_id")

    p = sub.add_parser("presets", help="List the emotion presets and the generation options they set.")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("translate", help="Translate text between the 23 supported languages (prints the translation).")
    p.add_argument("text", nargs="?", help="Text to translate ('-' or omitted: read stdin).")
    p.add_argument("--text-file", help="Read the text from a file.")
    p.add_argument("--from", dest="source", required=True, metavar="LANG", help="Language the text is written in, code or name.")
    p.add_argument("--to", dest="target", required=True, metavar="LANG", help="Language to translate into, code or name.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")

    p = sub.add_parser("translate-speech", help="Translate a recording: transcribe it, translate it and speak it in another "
                                                "language (Whisper + NLLB + Chatterbox Multilingual).")
    p.add_argument("source", type=Path, help="Recording to translate (WAV/FLAC/OGG/MP3).")
    p.add_argument("-l", "--language", required=True, help=f"Language to speak, code or name ({', '.join(LANGUAGES)}).")
    p.add_argument("--source-language", metavar="LANG", help="Language of the recording (default: detected).")
    p.add_argument("--model-version", choices=MULTILINGUAL_VERSIONS, help="Multilingual checkpoint.")
    p.add_argument("--emotion", choices=list(EMOTION_PRESETS), metavar="PRESET", help="Emotion preset, see `presets`.")
    voice = p.add_mutually_exclusive_group()
    voice.add_argument("--ref-audio", type=Path, help="Reference clip of the voice to speak with, >= 5 s.")
    voice.add_argument("--voice", help="Library voice id to speak with (default: the language's sample speaker).")
    voice.add_argument("--keep-voice", action="store_true", help="Speak in the voice of the recording itself (>= 5 s of speech).")
    _add_output(p)
    g = p.add_argument_group("generation options", "Unset options use the model defaults.")
    _add_generation_options(g, (MULTILINGUAL,))

    p = sub.add_parser("denoise", help="Remove background noise from a recording (e.g. a reference clip) with an AI model.")
    p.add_argument("audio", type=Path, help="Recording to clean (WAV/FLAC/OGG/MP3).")
    _add_output(p)

    p = sub.add_parser("vocals", help="Isolate the vocals of a music track (no drums, bass or instruments) with an AI model.")
    p.add_argument("audio", type=Path, help="Music track or recording (WAV/FLAC/OGG/MP3).")
    _add_output(p)

    p = sub.add_parser("watermark", help="Check audio for the Perth watermark Chatterbox embeds.")
    p.add_argument("audio", type=Path, help="Audio file (WAV/FLAC/OGG/MP3).")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")

    p = sub.add_parser("transcribe", help="Whisper speech-to-text: print the transcript of a recording.")
    p.add_argument("audio", type=Path, help="Audio file (WAV/FLAC/OGG/MP3).")
    p.add_argument("-l", "--language", help="Language spoken, code or name (default: detect it).")
    p.add_argument("--json", action="store_true", help="Print the transcript with timestamped segments as JSON.")

    p = sub.add_parser("stream", help="Live voice conversion: mic in, converted voice out to an audio device "
                                       "(e.g. a virtual cable for Discord).",
                       description="Listens to a microphone, converts each phrase you say with Chatterbox voice conversion "
                                   "and plays it on an output device. Run it on the PC that has the audio devices; "
                                   "use --server to do the conversion on another machine.")
    p.add_argument("--list-devices", action="store_true", help="List the audio devices and exit.")
    p.add_argument("--input", help="Microphone: a device number or part of its name (default: the system default input).")
    p.add_argument("--output", help="Where the converted voice plays: a device number or part of its name, e.g. "
                                    '"CABLE Input" (default: the system default output).')
    target = p.add_mutually_exclusive_group()
    target.add_argument("--voice", help="Library voice id to sound like (default: the built-in voice).")
    target.add_argument("--target-audio", type=Path, help="Clip of the voice to sound like, >= 5 s.")
    p.add_argument("--threshold", type=float, help="Speech level (RMS, 0-1) that counts as talking. "
                                                   "Default: measured from the background noise at startup.")
    p.add_argument("--silence-ms", type=int, default=500, help="Pause that ends a phrase (default: 500).")
    p.add_argument("--min-ms", type=int, default=300, help="Ignore sounds shorter than this (default: 300).")
    p.add_argument("--max-seconds", type=float, default=15.0, help="Cut a phrase after this long without a pause (default: 15).")

    sub.add_parser("info", help="Show backend, models, languages and generation options.")

    p = sub.add_parser("serve", help="Run the web UI + REST API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--preload", nargs="?", const="all", metavar="MODELS",
                   help="Load models at startup instead of on first use: 'all' (default) or a comma list of "
                        "original, turbo, nano, multilingual, vc.")
    return parser


# ---------------------------------------------------------------------------- commands

def _client(args):
    if args.server:
        from .client import HttpClient

        return HttpClient(args.server, verify=not args.insecure)
    from .client import LocalClient

    return LocalClient()


def _apply_local_settings(args) -> None:
    if args.backend:
        os.environ["POLY_CHATTERBOX_BACKEND"] = args.backend
    if args.device:
        os.environ["POLY_CHATTERBOX_DEVICE"] = args.device


def run(args) -> int:
    if args.command == "serve":
        return _serve(args)

    if args.command == "stream" and args.list_devices:
        from . import stream

        return stream.list_devices()

    client = _client(args)
    cmd = args.command
    if cmd == "stream":
        from . import stream

        return stream.run(client, args.input, args.output, args.voice, args.target_audio, args.threshold,
                          args.silence_ms, args.min_ms, args.max_seconds)
    if cmd == "info":
        print(json.dumps(client.info(), indent=2, ensure_ascii=False))
        return 0
    if cmd == "voices":
        return _voices(args, client)
    if cmd == "prompts":
        return _prompts(args, client)
    if cmd == "presets":
        presets = client.presets()
        if args.json:
            print(json.dumps(presets, indent=2, ensure_ascii=False))
        else:
            for pr in presets:
                options = " ".join(f"{k}={v}" for k, v in pr["options"].items())
                print(f"{pr['id']:<10} {pr['description']}\n{'':<10} {options}")
        return 0

    if cmd == "translate":
        result = client.translate(_read_text(args), args.source, args.target)
        print(json.dumps(result, ensure_ascii=False) if args.json else result["text"])
        return 0
    if cmd == "translate-speech":
        wav, meta = client.translate_speech(args.source, args.language, args.source_language, args.ref_audio, args.voice,
                                            args.keep_voice, args.model_version, _generation(args), args.emotion)
        _write_output(args, wav, meta)
        if not args.json:
            print(f"heard ({meta.get('source_language')}): {meta.get('transcript')}\n"
                  f"said  ({meta.get('language')}): {meta.get('translation')}", file=sys.stderr)
        return 0
    if cmd == "vocals":
        wav, meta = client.isolate_vocals(args.audio)
        _write_output(args, wav, meta)
        return 0
    if cmd == "denoise":
        wav, meta = client.denoise(args.audio)
        _write_output(args, wav, meta)
        return 0
    if cmd == "watermark":
        result = client.watermark(args.audio)
        if args.json:
            print(json.dumps(result))
        else:
            print(f"{'watermarked' if result['watermarked'] else 'no watermark'} (score {result['score']})")
        return 0
    if cmd == "transcribe":
        result = client.transcribe(args.audio, args.language)
        print(json.dumps(result, ensure_ascii=False) if args.json else result["text"])
        return 0
    if cmd == "convert":
        wav, meta = client.convert(args.source, args.target_audio, args.voice)
        _write_output(args, wav, meta)
        return 0

    gen = _generation(args)
    saved = client.prompt(args.prompt) if args.prompt else {}
    if saved and not (args.text or args.text_file):
        text = saved["text"]
    else:
        text = _read_text(args)
    if saved:
        gen = {**saved.get("generation", {}), **(gen or {})} or None
        if not args.ref_audio and not args.voice:
            args.voice = saved.get("voice_id")
    if cmd == "original":
        wav, meta = client.original(text, args.ref_audio, args.voice, gen, args.emotion, args.translate_from)
    elif cmd in ("turbo", "nano"):
        wav, meta = client.turbo(text, args.ref_audio, args.voice, gen, nano=cmd == "nano", preset=args.emotion,
                                 translate_from=args.translate_from)
    elif cmd == "multilingual":
        language = args.language or saved.get("language")
        if not language:
            raise ValueError("A language is required: use -l, or a saved prompt that has one.")
        wav, meta = client.multilingual(text, language, args.ref_audio, args.voice,
                                        args.model_version or saved.get("version"), gen, args.emotion, args.translate_from)
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown command {cmd}")
    _write_output(args, wav, meta)
    return 0


def _prompts(args, client) -> int:
    cmd = args.prompts_command
    if cmd == "list":
        prompts = client.prompts()
        if args.json:
            print(json.dumps(prompts, indent=2, ensure_ascii=False))
        else:
            print(f"{'ID':<24} {'MODE':<13} {'VOICE':<16} TEXT")
            for p in prompts:
                preview = " ".join(p["text"].split())
                ellipsis = "…" if len(preview) > 50 else ""
                print(f"{p['id']:<24} {p['mode'] or '-':<13} {p['voice_id'] or '-':<16} {preview[:50]}{ellipsis}")
            print(f"\n{len(prompts)} prompts", file=sys.stderr)
    elif cmd == "show":
        prompt = client.prompt(args.prompt_id)
        print(json.dumps(prompt, indent=2, ensure_ascii=False) if args.json else prompt["text"])
    elif cmd == "save":
        prompt = client.save_prompt(args.name, _read_text(args), args.mode, args.language, args.model_version,
                                    args.voice, _generation(args))
        print(prompt["id"])
    elif cmd == "remove":
        client.remove_prompt(args.prompt_id)
        print(f"removed {args.prompt_id}", file=sys.stderr)
    return 0


def _voices(args, client) -> int:
    if args.voices_command == "list":
        voices = client.voices()
        if args.kind:
            voices = [v for v in voices if v["kind"] == args.kind]
        if args.language:
            lang = args.language.lower()
            voices = [v for v in voices if v["language"] == lang or LANGUAGES.get(v["language"], "").lower() == lang]
        if args.json:
            print(json.dumps(voices, indent=2, ensure_ascii=False))
        else:
            print(f"{'ID':<20} {'KIND':<8} {'LANGUAGE':<11} {'GENDER':<8} NAME")
            for v in voices:
                print(f"{v['id']:<20} {v['kind']:<8} {LANGUAGES.get(v['language'], 'any'):<11} {v['gender']:<8} {v['name']}")
            print(f"\n{len(voices)} voices", file=sys.stderr)
    elif args.voices_command == "add":
        voice = client.save_voice(args.name, args.ref_audio, args.language, args.description, args.denoise)
        print(voice["id"])
    elif args.voices_command == "remove":
        client.remove_voice(args.voice_id)
        print(f"removed {args.voice_id}", file=sys.stderr)
    return 0


def _serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .engine import get_engine

    engine = get_engine()
    if args.preload:
        from .backends.chatterbox import VC, ChatterboxBackend

        if isinstance(engine.backend, ChatterboxBackend):
            keys = {"original": ORIGINAL, "turbo": TURBO, "nano": NANO, "vc": VC,
                    "multilingual": f"{MULTILINGUAL}:{engine.settings.multilingual_version}"}
            names = list(keys) if args.preload == "all" else [n.strip() for n in args.preload.split(",") if n.strip()]
            unknown = [n for n in names if n not in keys]
            if unknown:
                raise ValueError(f"Unknown model(s) to preload: {', '.join(unknown)}. Use {', '.join(keys)}.")
            for name in names:
                engine.backend.model(keys[name])
    print(f"Poly Chatterbox on http://{args.host}:{args.port}  (API docs: /docs, backend: {engine.backend.name})", file=sys.stderr)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    # Voice names and texts include CJK/Arabic/Cyrillic text; legacy Windows consoles default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    _apply_local_settings(args)
    import logging

    logging.basicConfig(level=logging.INFO if args.verbose or args.command == "serve" else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return run(args)
    except (ValueError, LookupError) as e:
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    except Exception as e:
        from .backends import BackendError
        from .client import ApiError

        if isinstance(e, (BackendError, ApiError)):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
