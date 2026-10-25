# poly-chatterbox

A toolkit around [Chatterbox](https://github.com/resemble-ai/chatterbox) with a **web UI**, a **REST API** and a **CLI**:

| Feature | Web UI mode | API | CLI |
|---|---|---|---|
| Chatterbox-Turbo (English, 350M) with paralinguistic tags | Turbo | `POST /api/tts/turbo` | `poly-chatterbox turbo` |
| Chatterbox-Nano (English, 110M, CPU-friendly) | Nano | `POST /api/tts/nano` | `poly-chatterbox nano` |
| Chatterbox Multilingual v3 / v2 (23 languages) | Multilingual | `POST /api/tts/multilingual` | `poly-chatterbox multilingual` |
| Original Chatterbox (English, 500M) with exaggeration and CFG/pace | Original | `POST /api/tts/original` | `poly-chatterbox original` |
| Emotion presets (mood bundles of generation options) | "Emotion preset" under the text | `GET /api/presets`, `preset` on each TTS endpoint | `poly-chatterbox presets`, `--emotion` |

| Dialog translation, on the fly or on its own | "Translate from" in every TTS mode | `POST /api/translate`, `translate_from` on each TTS endpoint | `poly-chatterbox translate`, `--translate-from` |

| Background-noise removal for reference clips (AI speech enhancement) | Clean Audio, and "Remove background noise" on a reference clip | `POST /api/denoise`, `denoise` on `POST /api/voices` | `poly-chatterbox denoise`, `voices add --denoise` |
| Translate a recording (speech in, translated speech out) | Speech Translation | `POST /api/translate/speech` | `poly-chatterbox translate-speech` |
| Zero-shot voice cloning (≥ 5 s clip) | every TTS mode | `ref_audio` / `voice_id` on each endpoint | `--ref-audio` / `--voice` |
| Voice conversion | Voice Conversion | `POST /api/vc` | `poly-chatterbox convert` |
| Live voice conversion from a microphone (e.g. into Discord) | n/a (runs on your PC) | `POST /api/vc` per phrase | `poly-chatterbox stream` |
| Voice library (built-in, 24 demo voices, your saved clones) | voice picker in every mode | `GET/POST/DELETE /api/voices` | `poly-chatterbox voices` |
| Perth watermark detection | Watermark | `POST /api/watermark` | `poly-chatterbox watermark` |
| Speech to text with Whisper | Whisper | `POST /api/transcribe` | `poly-chatterbox transcribe` |

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
- **Storage:** weights download on first use into the `models` volume. Saved voices and downloaded demo voices go to the `library` volume, and saved prompts to the `prompts` volume. Both stacks share these volumes. To keep prompts in a host folder instead (to edit, back up or share them), set `PROMPTS_DIR=./prompts` in `.env`.
- **Settings:** `POLY_CHATTERBOX_SERVE_ARGS` passes extra `poly-chatterbox serve` flags, for example `--preload` (all models) or `--preload turbo,multilingual`. Any other `POLY_CHATTERBOX_*` variable (see [Configuration](#configuration-environment-variables)) can be added under `environment:` in the compose file. `HF_TOKEN` speeds up downloads.

The image runs on its own too: `docker build -t poly-chatterbox .`, then `docker run --gpus all -p 8000:8000 -v chatterbox-cache:/data/cache -v chatterbox-library:/data/library -v chatterbox-prompts:/data/prompts poly-chatterbox`.

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
* `[settings name=value …]…[/settings]` gives just that span its own [generation options](#generation-options), for example `Calm. [settings exaggeration=0.9 cfg_weight=0.3]I can't believe it![/settings] Anyway…`. Options you don't name keep the request's values, and `seed` can't be set per span.
* `[voice: id]…[/voice]` has a library voice (see `poly-chatterbox voices list`, or `default` for the built-in voice) speak just that span. The voice uploaded or chosen for the request speaks everything else.
* Spans can be nested and mixed; inner `[settings]` add to or override outer ones. They work in every TTS mode through the API and CLI, since they are part of the text. Options a model doesn't use are ignored with a warning, as usual.
* In the web UI, change some sliders or generation options, highlight text and click **Wrap selection in settings** to wrap it with your changed values. Pick a voice next to **Wrap selection in voice** to do the same for a voice. With nothing highlighted, an empty pair is inserted at the cursor.
* One Chatterbox call produces at most about 40 seconds of audio, so longer text is split at sentence boundaries (then clauses, then words) into chunks of at most `POLY_CHATTERBOX_MAX_CHUNK_CHARS` characters and joined back together.

## Voice library

* **`default`**: each model's built-in voice.
* **24 demo voices**: `podcast` (the upstream Turbo demo's podcast host) and one native speaker per multilingual language (`sample_fr`, `sample_ja`, …), from the upstream demo apps. Each is downloaded on first use into `data/samples/`.
* **Your clones**: save one with `poly-chatterbox voices add`, `POST /api/voices` or the UI. They are stored in `data/voices/`.

Any voice works with any model and as a voice conversion target. `poly-chatterbox voices list` lists every voice.

## Live voice conversion (Discord and other apps)

`poly-chatterbox stream` listens to your microphone, cuts your speech into phrases at the pauses, converts each one with Chatterbox voice conversion and plays it on an output device. Point that output at a **virtual audio cable** and any app (Discord, OBS, a game, a call) can use the cable as its microphone, so it hears the converted voice instead of yours.

```
your mic ──► poly-chatterbox stream ──► virtual cable (output) ═══► Discord (input)
                  │
        Chatterbox voice conversion (your GPU, or a --server)
```

It is near-real-time: each phrase is heard about a second after you stop talking (the pause that ends the phrase, plus roughly 0.4 s of conversion on an RTX 4070 Ti Super). The converted voice keeps your timing and intonation; only the voice changes. Use only voices you have permission to use.

The command runs on the PC that has the microphone and the virtual cable. By default it converts in-process on that PC's GPU. With `--server https://host:8443` (and `--insecure` for the self-signed certificate) it sends each phrase to a server's `/api/vc` endpoint instead.

### What you need

| | Windows | macOS | Linux |
|---|---|---|---|
| [uv](https://docs.astral.sh/uv/) (installs Python) | `winget install astral-sh.uv` | `brew install uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| git (the watermark library installs from git) | `winget install Git.Git` | `xcode-select --install` | `sudo apt install git` |
| GPU | NVIDIA, driver 570 or newer | Apple silicon (uses MPS) | NVIDIA, driver 570 or newer |
| PortAudio (audio I/O) | bundled | bundled | `sudo apt install libportaudio2` |
| Virtual audio cable | [VB-CABLE](https://vb-audio.com/Cable/) | [BlackHole](https://existential.audio/blackhole/) (`brew install blackhole-2ch`) | a PulseAudio/PipeWire null sink (no install) |

Without a GPU it still runs (`--device cpu`), but conversion is much slower, which may be too slow to feel live.

The Windows steps below were tested end to end. The macOS and Linux setups use each platform's standard virtual-device tools but have not been tried on real hardware, so check the device names with `--list-devices`.

### Example: stream your voice into Discord (Windows)

1. **Install the virtual cable.** Download VB-CABLE, unzip it, run `VBCABLE_Setup_x64.exe` as administrator, then reboot. Windows now has a playback device called **CABLE Input** and a recording device called **CABLE Output**: audio sent to the first comes out of the second.

2. **Install poly-chatterbox's dependencies.** In the repo folder:
   ```
   uv sync --extra chatterbox --extra stream
   ```

3. **Save the voice you want to sound like.** Use a clean clip of at least 5 seconds (about 10 s of one speaker works best; WAV, FLAC, OGG or MP3):
   ```
   uv run poly-chatterbox voices add "Target" --ref-audio C:\path\to\target_voice.mp3 -l en
   ```
   This stores it in the voice library as `target`. To try things first without your own clip, use a demo voice such as `podcast` or `sample_de` (`uv run poly-chatterbox voices list` shows them all).

4. **Find your devices** (optional but useful):
   ```
   uv run poly-chatterbox stream --list-devices
   ```

5. **Start streaming.** `--input` is your microphone and `--output` is the cable; both take a device number or part of its name:
   ```
   uv run poly-chatterbox stream --input "Shure" --output "CABLE Input" --voice target
   ```
   Stay quiet for the one-second background-noise measurement. When it prints "Listening", talk, pause, and the converted phrase plays on the cable. Each phrase's timing is printed. Press Ctrl+C to stop. The first run downloads the model.

6. **Point Discord at the cable.** In User Settings → Voice & Video, set **Input Device** to **CABLE Output (VB-Audio Virtual Cable)**. Turn off **Noise Suppression**, **Echo Cancellation** and **Automatic Gain Control** (they can clip converted speech), and use **Let's Check** to confirm the level moves when you talk. Wear headphones so Discord doesn't pick up your speakers.

7. **Optional: hear what Discord hears.** Windows Sound settings → Recording → **CABLE Output** → Properties → Listen → tick **Listen to this device** and choose your headphones. This adds a little delay.

### macOS

1. `brew install uv blackhole-2ch` (restart after installing BlackHole), then `xcode-select --install` if git is missing.
2. `uv sync --extra chatterbox --extra stream`, then save a voice as in step 3 above.
3. `uv run poly-chatterbox stream --input "MacBook" --output "BlackHole" --voice target` (use `--list-devices` for your mic's name).
4. In Discord, choose **BlackHole 2ch** as the Input Device. To hear the output yourself, create a Multi-Output Device in Audio MIDI Setup that includes your headphones and BlackHole, and select it as `--output` instead.

### Linux

1. `sudo apt install git libportaudio2`, install uv, then `uv sync --extra chatterbox --extra stream` and save a voice as in step 3 above.
2. Create a virtual microphone (it lasts until you log out; put the commands in a startup script to keep it):
   ```
   pactl load-module module-null-sink sink_name=chatterbox sink_properties=device.description=Chatterbox
   pactl load-module module-remap-source master=chatterbox.monitor source_name=chatterbox_mic source_properties=device.description=ChatterboxMic
   ```
3. Send the output to that sink and start streaming:
   ```
   PULSE_SINK=chatterbox uv run poly-chatterbox stream --input default --output pulse --voice target
   ```
   If `--input default` isn't accepted, give your microphone's number or name from `--list-devices`.
4. In Discord, choose **ChatterboxMic** as the Input Device.

### Tuning and troubleshooting

| Problem | Try |
|---|---|
| Sentences are split in the middle | `--silence-ms 700` (a longer pause ends a phrase) |
| Too much delay | `--silence-ms 350` (may split sentences) |
| Nothing happens when you talk | Check the "Speech threshold" it printed; lower it with `--threshold 0.004`, and confirm `--input` is the right mic |
| Room noise or breathing triggers conversion | Raise `--threshold`, or raise `--min-ms` to ignore short sounds |
| "conversion is falling behind" | Conversion is slower than your speech: pause longer between phrases, free GPU memory, or use a `--server` with a faster GPU |
| Discord hears nothing | Input Device must be the cable's **output** side ("CABLE Output" / "BlackHole" / "ChatterboxMic"), while `--output` is its **input** side |
| Robotic or clipped audio | Turn off Discord's noise suppression, echo cancellation and automatic gain |
| Voice sounds off | Use a longer, cleaner target clip (10 s of one speaker, no music), or try a target close to your own pitch |

More options: `--target-audio clip.wav` (use a clip directly instead of a library voice), `--max-seconds` (cut a very long phrase) and `--threshold` (what counts as talking; by default measured from the background noise at startup).

## Saved prompts

A prompt is a name, the text (with any markup), and optionally the mode, language, multilingual version, library voice and generation options to speak it with. They are stored as JSON in `data/prompts/` (the `prompts` volume in Docker). Saving under an existing name replaces that prompt.

* **Web UI:** above the text box, pick a prompt and click **Load**, or click **Save…** to store the current text, mode, language, voice and the generation options you've changed. A voice from an uploaded clip isn't stored: save the clip to the voice library first.
* **API:** `GET /api/prompts`, `GET /api/prompts/{id}`, `POST /api/prompts` (JSON: `name`, `text`, and optional `mode`, `language`, `version`, `voice_id`, `generation`; 201 when created, 200 when replaced) and `DELETE /api/prompts/{id}`.
* **CLI:** `poly-chatterbox prompts list|show|save|remove`, and `--prompt ID` on `original`, `turbo`, `nano` and `multilingual` to speak a saved prompt. Its text, voice, language and generation options are the defaults; anything you pass explicitly wins.

```sh
poly-chatterbox prompts save "Dramatic intro" "Welcome. [settings exaggeration=0.9]It begins tonight.[/settings]" --mode original --voice podcast
poly-chatterbox original --prompt dramatic_intro -o intro.wav
```

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
poly-chatterbox transcribe speech.wav -l en
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

### Emotion presets

Chatterbox has no emotion switch, so the presets set the controls that move the delivery towards a mood: `exaggeration`, `cfg_weight` and the sampling options. Available: `neutral`, `happy`, `excited`, `playful`, `sad`, `tired`, `calm`, `serious`, `nervous`, `angry` and `dramatic`. They are a starting point, not a guarantee: results vary with the voice, so pair a preset with a `seed` and nudge the sliders.

* **Web UI:** pick one under *Emotion preset*; it fills in the sliders and options for the current model, which you can keep adjusting. (Turbo and Nano have no exaggeration or CFG, so for them a preset only changes the sampling.) It also works with *Wrap selection in settings* to give just one line a mood.
* **Markup:** `[settings preset=sad]…[/settings]` gives a span a preset; options in the same tag win over it, e.g. `[settings preset=angry temperature=0.6]`.
* **API:** `GET /api/presets` lists them with their options; the `preset` form field on `/api/tts/*` applies one. Options in `generation` win over the preset.
* **CLI:** `poly-chatterbox presets` lists them, and `--emotion NAME` applies one on `original`, `turbo`, `nano` and `multilingual`; explicit options like `--temperature` win.

```sh
poly-chatterbox original "I can't believe you did that." --emotion angry --seed 4
curl -X POST localhost:8000/api/tts/original -F text="We won!" -F preset=excited -o won.wav
```

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
curl -X POST localhost:8000/api/transcribe -F audio=@speech.wav -F language=fr
```

The full schema is at `/docs`.

## Watermarking

Every clip Chatterbox generates carries Resemble AI's [Perth](https://github.com/resemble-ai/perth) watermark, an imperceptible neural watermark that survives MP3 compression and common edits. The Watermark mode, `POST /api/watermark` and `poly-chatterbox watermark` report whether a clip carries it.

## Whisper transcription

The Whisper mode is a plain speech-to-text tool: upload or record audio, optionally pick the spoken language (otherwise it is detected), and copy the transcript. `POST /api/transcribe` (multipart `audio`, optional `language`) returns `{"text", "segments": [{"start", "end", "text"}], "language", "duration", "elapsed"}`, and `poly-chatterbox transcribe FILE [-l LANG] [--json]` does the same from the shell. The checkpoint (`openai/whisper-large-v3-turbo` by default, about 1.6 GB) is downloaded on first use and set with `POLY_CHATTERBOX_WHISPER_MODEL`; it stays loaded next to the TTS models without counting toward `MAX_LOADED_MODELS`.

## Translation

Dialog can be translated between any two of the 23 supported languages and spoken in the voice you chose, so "English text, in this cloned voice, in Japanese" is one request. Translation uses [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (distilled 600M, about 1.2 GB, downloaded on first use; the model card's licence is CC-BY-NC 4.0, so check it fits your use).

* **Web UI:** every TTS mode has a *Translate from* menu. Pick the language your text is written in and it is translated into the spoken language when you generate: English for Turbo, Nano and Original, or the *Language* above for Multilingual. *Translate text now* swaps the text for its translation instead, so you can read and fix it before generating.
* **API:** `POST /api/translate` (form fields `text`, `source`, `target`; returns `{"text", "source", "target", "elapsed"}`), and a `translate_from` form field on `/api/tts/original`, `/turbo`, `/nano` and `/multilingual`. The result metadata then has `translated_from`.
* **CLI:** `poly-chatterbox translate "Hello" --from en --to ja`, and `--translate-from LANG` on `original`, `turbo`, `nano` and `multilingual`.

`[pause]`, `[settings]`, `[voice]` and `[laugh]`-style tags are left where they are and only the text between them is translated. Multilingual doesn't understand the Turbo tags, so don't expect `[laugh]` to be acted on there.

```sh
poly-chatterbox multilingual "Good morning, everyone. [pause 400ms] Let's begin." -l ja --translate-from en --voice me -o ja.wav
curl -X POST localhost:8000/api/tts/multilingual -F text="Good morning" -F language=ja -F translate_from=en -F voice_id=me -o ja.wav
```

Translation needs the target to be one of the 23 languages.

### Translating recordings

Speech Translation chains the three models: Whisper transcribes the recording, NLLB-200 translates the words, and Chatterbox Multilingual speaks them. Choose the voice it speaks with: a library voice, an uploaded clip, the language's sample speaker, or *Original speaker*, which clones the voice of the recording itself so the translation sounds like them. The result shows what was heard and what was said.

* **Web UI:** the *Speech Translation* mode. *Spoken in* is detected unless you set it, which helps with short or noisy clips.
* **API:** `POST /api/translate/speech` with `source_audio`, `language` (the language to speak), and optionally `source_language`, one of `ref_audio` / `voice_id` / `keep_voice=true`, `version`, `generation` and `preset`. It returns `audio/wav`; the metadata has `source_language`, `transcript` and `translation` (each cut to 400 characters, since it travels in a header; use `POST /api/transcribe` and `POST /api/translate` for the full text).
* **CLI:** `poly-chatterbox translate-speech talk.wav -l ja [--source-language en] [--voice ID | --ref-audio clip.wav | --keep-voice]`, which also prints what was heard and said.

```sh
poly-chatterbox translate-speech talk.wav -l ja --voice me -o talk_ja.wav
curl -X POST localhost:8000/api/translate/speech -F source_audio=@talk.wav -F language=ja -F keep_voice=true -o talk_ja.wav
```

It speaks the whole recording as one passage: pauses and the original timing and intonation are not carried over (voice conversion keeps those, but can't change the language). For a noisy recording, clean it first with `denoise`.

## Cleaning noisy reference clips

A reference clip recorded in a noisy room clones the noise along with the voice. The speech-enhancement model ([SpeechBrain SepFormer trained on the DNS4 challenge data](https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement), about 100 MB, downloaded on first use) separates the talker from the background so you can clone from the cleaned clip.

* **Web UI:** the *Clean Audio* mode cleans a recording and lets you download it or click *Use as reference clip*. Anywhere you upload a reference clip there is also a *Remove background noise* button that replaces your clip with the cleaned one (you can listen to it before generating or saving).
* **API:** `POST /api/denoise` (multipart `audio`) returns the cleaned `audio/wav`; `denoise=true` on `POST /api/voices` cleans the clip before saving it to the library.
* **CLI:** `poly-chatterbox denoise noisy.wav -o clean.wav`, and `poly-chatterbox voices add "Me" --ref-audio noisy.wav --denoise`.

The model works at 16 kHz, so cleaned clips are 16 kHz mono with their original loudness. Long clips are processed in 15 s windows and cross-faded. It is trained for speech against everyday noise (fans, traffic, chatter, room tone). Music or very loud noise can leave artifacts, and a second voice talking in the background may be kept, so use the cleanest 10 seconds you have. Set another SpeechBrain enhancement checkpoint with `POLY_CHATTERBOX_DENOISE_MODEL`.

```sh
poly-chatterbox denoise noisy_room.wav -o clean.wav
poly-chatterbox voices add "Narrator" --ref-audio noisy_room.wav --denoise -l en
curl -X POST localhost:8000/api/denoise -F audio=@noisy_room.wav -o clean.wav
```

## Configuration (environment variables)

| variable | default |
|---|---|
| `POLY_CHATTERBOX_BACKEND` | `chatterbox` (`mock` for placeholder tones) |
| `POLY_CHATTERBOX_DEVICE` | `auto` (CUDA, then MPS, then CPU; or `cuda:1`, `cpu`, …) |
| `POLY_CHATTERBOX_MULTILINGUAL_VERSION` | `v3` (`v2` for the previous checkpoint) |
| `POLY_CHATTERBOX_MAX_LOADED_MODELS` | `3` (least recently used model is unloaded first; 3 fits a 16 GB GPU) |
| `POLY_CHATTERBOX_WHISPER_MODEL` | `openai/whisper-large-v3-turbo` (any Hugging Face Whisper checkpoint) |

| `POLY_CHATTERBOX_TRANSLATION_MODEL` | `facebook/nllb-200-distilled-600M` (any NLLB-200 checkpoint) |

| `POLY_CHATTERBOX_DENOISE_MODEL` | `speechbrain/sepformer-dns4-16k-enhancement` (any SpeechBrain SepFormer enhancement checkpoint) |
| `POLY_CHATTERBOX_DATA_DIR` | `./data` |
| `POLY_CHATTERBOX_PROMPTS_DIR` | `<data dir>/prompts` (`/data/prompts` in Docker) |
| `POLY_CHATTERBOX_MIN_REF_SECONDS` / `_MAX_REF_SECONDS` | `5` / `60` |
| `POLY_CHATTERBOX_MAX_CHUNK_CHARS` | `300` |
| `HF_TOKEN` | unset (optional Hugging Face token) |

## Development

```sh
uv run pytest      # uses the mock backend; no GPU needed
```

Layout: `src/poly_chatterbox/` contains `engine.py` (the feature logic), `backends/` (chatterbox, mock), `text.py` (pause markup and chunking), `presets.py` (emotion presets), `library.py` (the voice library), `languages.py`, `api/` (FastAPI), `web/` (UI) and `cli.py`. `stream.py` is the live voice conversion client, and `prompts.py` the saved prompts. `src/chatterbox/` is the vendored upstream model code.

## Disclaimer

Don't use this to do bad things: clone only voices you have permission to use.

# Scratchpad
```powershell
uv run poly-chatterbox voices add "Target" --ref-audio C:\some\path\margot_robbie_ex.mp3 -l en

uv run poly-chatterbox stream --input "Shure" --output "CABLE Input" --voice target
```