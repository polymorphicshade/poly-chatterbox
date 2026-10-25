# poly-chatterbox

A toolkit around [Chatterbox](https://github.com/resemble-ai/chatterbox) with a **web UI**, a **REST API** and a **CLI**:

| Feature | Web UI mode | API | CLI |
|---|---|---|---|
| Chatterbox-Turbo (English, 350M) with paralinguistic tags | Turbo | `POST /api/tts/turbo` | `poly-chatterbox turbo` |
| Chatterbox-Nano (English, 110M, CPU-friendly) | Nano | `POST /api/tts/nano` | `poly-chatterbox nano` |
| Chatterbox Multilingual v3 / v2 (23 languages) | Multilingual | `POST /api/tts/multilingual` | `poly-chatterbox multilingual` |
| Original Chatterbox (English, 500M) with exaggeration and CFG/pace | Original | `POST /api/tts/original` | `poly-chatterbox original` |
| Zero-shot voice cloning (≥ 5 s clip) | every TTS mode | `ref_audio` / `voice_id` on each endpoint | `--ref-audio` / `--voice` |
| Voice conversion | Voice Conversion | `POST /api/vc` | `poly-chatterbox convert` |
| Voice library (built-in, 24 demo voices, your saved clones) | voice picker in every mode | `GET/POST/DELETE /api/voices` | `poly-chatterbox voices` |
| Perth watermark detection | Watermark | `POST /api/watermark` | `poly-chatterbox watermark` |

Multilingual languages: Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish and Chinese.

The model code is the upstream `chatterbox` package, vendored under `src/chatterbox` (MIT, © Resemble AI).

## Setup

Runs best on an NVIDIA GPU. Each model takes a few GB of VRAM; on smaller GPUs keep fewer loaded with `POLY_CHATTERBOX_MAX_LOADED_MODELS`. Nano also runs well on CPU. Uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra chatterbox           # installs torch (CUDA 12.8 wheels) + the Chatterbox dependencies
uv run poly-chatterbox serve         # http://127.0.0.1:8000  (API docs at /docs)
```

The weights are downloaded from Hugging Face the first time each model is used.

To try the UI, API and CLI without a GPU, use the mock backend, which outputs placeholder tones instead of speech: `uv sync && uv run poly-chatterbox --backend mock serve`.

`START.bat` / `START.sh` launch the server from the `.venv` on port 8000; extra arguments go to `poly-chatterbox serve`.

## Docker (Linux server)

There are two compose files. Both run the app on an NVIDIA GPU. The host needs Docker, the NVIDIA driver (570 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Ubuntu or Debian, `sudo ./NVIDIA.sh` installs all three and checks that a container can see the GPU.

**Plain HTTP** (`docker-compose.yml`):

```bash
cp .env.example .env          # optional: port, model settings and serve flags
docker compose up -d --build
docker compose logs -f app
```

The UI is then at `http://<host>:8000`.

**HTTPS** (`docker-compose.https.yml`) runs the same app plus an nginx sidecar that terminates TLS:

```bash
cp .env.https.example .env    # list the host's LAN address in TLS_SAN
docker compose -f docker-compose.https.yml up -d --build
```

The UI is then at `https://<host>:8443`. Browsers only allow microphone recording on HTTPS pages (or `localhost`), so use this stack to record reference clips from another machine.

- **Certificate:** a self-signed certificate is generated into `./certs` on first boot, so the browser warns once until you accept it. Every address you type in the browser must be in `TLS_SAN`, e.g. `IP:192.168.1.99`. After changing it, delete `./certs` and restart to mint a new one. To use a CA-issued certificate, put `fullchain.pem` and `privkey.pem` in `./certs`.
- **Ports:** in the HTTPS stack, 8000 is the app's plain HTTP listener and is published on `127.0.0.1` only. `https://<host>:8000` fails with `SSL_ERROR_RX_RECORD_TOO_LONG` because nothing there speaks TLS. Change either port with `HTTPS_PORT` and `PORT` in `.env`.
- **Storage:** weights download on first use into the `models` volume. Saved voices and downloaded demo voices go to the `library` volume. Both stacks share these volumes.
- **Settings:** `POLY_CHATTERBOX_SERVE_ARGS` passes extra `poly-chatterbox serve` flags, for example `--preload` (all models) or `--preload turbo,multilingual`. Any other `POLY_CHATTERBOX_*` variable (see [Configuration](#configuration-environment-variables)) can be added under `environment:` in the compose file. `HF_TOKEN` speeds up downloads.

The image runs on its own too: `docker build -t poly-chatterbox .`, then `docker run --gpus all -p 8000:8000 -v chatterbox-cache:/data/cache -v chatterbox-library:/data/library poly-chatterbox`.

## Web UI

Each **tab is an independent workspace** with every control. Use `+` (or Alt+N) to open one and `×` (or middle-click, or Alt+W) to close it. Double-click a tab to rename it. Workspaces can generate in parallel, and their settings are remembered in the browser.

In every TTS mode you choose the voice: the model's built-in voice, a library voice, or a clip you upload or record (which you can save to the library). Turbo and Nano show the paralinguistic tags as buttons that insert at the cursor. Original and Multilingual show exaggeration and CFG / pace as sliders.

## Models

* **Turbo** is the fastest high-quality model: one-step decoding, lower VRAM, and native paralinguistic tags: `[clear throat]`, `[sigh]`, `[shush]`, `[cough]`, `[groan]`, `[sniff]`, `[gasp]`, `[chuckle]`, `[laugh]`.
* **Nano** shares Turbo's architecture and tags at 110M parameters, for CPU or tight memory budgets (about 3× faster than realtime on 8 CPU cores).
* **Multilingual** speaks 23 languages. v3 (the default) improves speaker similarity and reduces hallucinations over v2. Without a voice, each language uses its native demo speaker (`sample_<code>`).
* **Original** is the first Chatterbox: English, with **exaggeration** (emotion intensity) and **CFG / pace** controls.

Tips from upstream:

* The defaults (exaggeration 0.5, CFG 0.5) suit most prompts. For a fast-talking reference, lower CFG to about 0.3.
* For dramatic speech, lower CFG to about 0.3 and raise exaggeration to 0.7 or more. Higher exaggeration speeds speech up; lower CFG slows it back down.
* When the reference clip is in a different language from the text, set CFG to 0 or the output inherits the reference's accent. The API warns when a library voice's language doesn't match.

## Text

* `[pause]` inserts 0.5 s of silence. `[pause 800ms]`, `[pause 1.5s]` and `[pause 2]` set an explicit length. Other bracketed text (such as Turbo's tags) goes to the model unchanged.
* One Chatterbox call produces at most about 40 seconds of audio, so longer text is split at sentence boundaries (then clauses, then words) into chunks of at most `POLY_CHATTERBOX_MAX_CHUNK_CHARS` characters and joined back together.

## Voice library

* **`default`**: each model's built-in voice.
* **24 demo voices**: `podcast` (the upstream Turbo demo's podcast host) and one native speaker per multilingual language (`sample_fr`, `sample_ja`, …), from the upstream demo apps. Each is downloaded on first use into `data/samples/`.
* **Your clones**: save one with `poly-chatterbox voices add`, `POST /api/voices` or the UI. They are stored in `data/voices/`.

Any voice works with any model and as a voice conversion target. `poly-chatterbox voices list` lists every voice.

## CLI

```sh
poly-chatterbox turbo "Hi there [chuckle], got a minute to chat?" --ref-audio me.wav -o hi.wav
poly-chatterbox nano "Running on the CPU." --device cpu --voice podcast
poly-chatterbox multilingual "Bonjour à tous !" -l fr                    # French demo speaker
poly-chatterbox multilingual "你好，今天天气真不错。" -l zh --voice me --cfg-weight 0
poly-chatterbox original "This is incredible!" --exaggeration 0.8 --cfg-weight 0.3
poly-chatterbox convert speech.wav --target-audio someone.wav -o converted.wav
poly-chatterbox voices add "Me" --ref-audio me.wav -l en
poly-chatterbox watermark converted.wav
echo "Text from a pipe" | poly-chatterbox turbo --voice podcast -o - > out.wav
```

Common options: `--ref-audio` or `--voice`, `-o/--output` (`-` writes to stdout), `--json` (prints metadata), `--text-file`, and the generation options below (each command shows only the options its model uses).

By default the CLI loads models in-process. For scripting many calls, run `poly-chatterbox serve` once and add `--server http://127.0.0.1:8000` (or set `POLY_CHATTERBOX_SERVER`) so the models stay loaded between calls.

## Generation options

Leave them unset to use the model defaults. The web UI shows this advice next to each control, `poly-chatterbox <command> --help` prints it, and `GET /api/info` returns it as `generation_options`.

| option | models | default | what it does |
|---|---|---|---|
| `exaggeration` | original, multilingual | 0.5 | Emotion intensity. 0.7+ is more dramatic and faster; extreme values can be unstable. |
| `cfg_weight` | original, multilingual | 0.5 | Adherence to the voice and text, and the pace. ~0.3 for fast references or drama; 0 for cross-language references. |
| `temperature` | all | 0.8 | Lower is steadier, higher is livelier but can glitch. |
| `min_p` | original, multilingual | 0.05 | Drops unlikely sounds; handles high temperatures well. 0.02–0.1 recommended, 0 disables. |
| `top_p` | all | 1.0 (turbo/nano 0.95) | Nucleus sampling; lower is cleaner but flatter. 1.0 disables. |
| `top_k` | turbo, nano | 1000 | Samples from the k most likely sounds. |
| `repetition_penalty` | all | 1.2 | Raise it if the voice stutters or loops. |
| `norm_loudness` | turbo, nano | on | Normalizes the reference clip to -27 LUFS before cloning. |
| `seed` | all | random | The same seed, text and settings give the same take. |

Options a model doesn't use are ignored, with a warning in the result metadata.

## API

Synthesis endpoints take multipart form data and return `audio/wav`. Metadata (duration, model, voice, warnings, …) comes back in the `X-TTS-Metadata` header as URL-encoded JSON. Give the voice as an uploaded `ref_audio` clip or a library `voice_id` (neither uses the built-in voice), and sampling options as a JSON `generation` field.

```sh
curl -X POST localhost:8000/api/tts/turbo -F text="Oh, that's hilarious! [chuckle]" -F ref_audio=@me.wav -o turbo.wav
curl -X POST localhost:8000/api/tts/multilingual -F text="Hola a todos" -F language=es \
  -F generation='{"cfg_weight": 0.3, "seed": 7}' -o hola.wav
curl -X POST localhost:8000/api/tts/original -F text="Big news!" -F voice_id=podcast \
  -F generation='{"exaggeration": 0.8}' -o original.wav
curl -X POST localhost:8000/api/vc -F source_audio=@speech.wav -F target_voice_id=sample_de -o converted.wav
curl -X POST localhost:8000/api/watermark -F audio=@converted.wav
```

The full schema is at `/docs`.

## Watermarking

Every clip Chatterbox generates carries Resemble AI's [Perth](https://github.com/resemble-ai/perth) watermark, an imperceptible neural watermark that survives MP3 compression and common edits. The Watermark mode, `POST /api/watermark` and `poly-chatterbox watermark` report whether a clip carries it.

## Configuration (environment variables)

| variable | default |
|---|---|
| `POLY_CHATTERBOX_BACKEND` | `chatterbox` (`mock` for placeholder tones) |
| `POLY_CHATTERBOX_DEVICE` | `auto` (CUDA, then MPS, then CPU; or `cuda:1`, `cpu`, …) |
| `POLY_CHATTERBOX_MULTILINGUAL_VERSION` | `v3` (`v2` for the previous checkpoint) |
| `POLY_CHATTERBOX_MAX_LOADED_MODELS` | `3` (least recently used model is unloaded first; 3 fits a 16 GB GPU) |
| `POLY_CHATTERBOX_DATA_DIR` | `./data` |
| `POLY_CHATTERBOX_MIN_REF_SECONDS` / `_MAX_REF_SECONDS` | `5` / `60` |
| `POLY_CHATTERBOX_MAX_CHUNK_CHARS` | `300` |
| `HF_TOKEN` | unset (optional Hugging Face token) |

## Development

```sh
uv run pytest      # uses the mock backend; no GPU needed
```

Layout: `src/poly_chatterbox/` contains `engine.py` (the feature logic), `backends/` (chatterbox, mock), `text.py` (pause markup and chunking), `library.py` (the voice library), `languages.py`, `api/` (FastAPI), `web/` (UI) and `cli.py`. `src/chatterbox/` is the vendored upstream model code.

## Disclaimer

Don't use this to do bad things: clone only voices you have permission to use.
