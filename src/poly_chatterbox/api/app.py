"""FastAPI application: REST API under ``/api`` and the web UI at ``/``.

Synthesis endpoints return ``audio/wav``. Generation metadata (duration, model, warnings, …) is in
the ``X-TTS-Metadata`` header as URL-encoded JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .. import __version__
from ..audio import Audio, load_audio
from ..backends import GENERATION_OPTIONS, MULTILINGUAL_VERSIONS, BackendError
from ..engine import MODEL_LABELS, Engine, SynthesisResult, get_engine
from ..languages import LANGUAGES, SAMPLES
from ..text import EVENT_TAGS
from .schemas import GenerationOptions, PromptRequest, params

STATIC_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
METADATA_HEADER = "X-TTS-Metadata"
WAV = {200: {"content": {"audio/wav": {}}}}

TEXT = Form(..., description="Text to speak. Long text is split at sentences. Markup: [pause 500ms] inserts silence; "
                             "[settings exaggeration=0.9]…[/settings] and [voice: sample_de]…[/voice] change the "
                             "generation options or voice for just that span.")
REF_AUDIO = File(None, description="Reference clip of the voice to clone, 5 s or longer (WAV/FLAC/OGG/MP3).")
VOICE_ID = Form(None, description="Or a library voice (see GET /api/voices). Neither: the model's built-in voice.")
GENERATION = Form(None, description="GenerationOptions (see Schemas) as JSON, e.g. {\"temperature\": 0.7, \"seed\": 1}.")


def _audio_response(result: SynthesisResult) -> Response:
    meta = quote(json.dumps(result.metadata(), ensure_ascii=False))
    return Response(result.wav_bytes(), media_type="audio/wav", headers={METADATA_HEADER: meta})


async def _read_upload(upload: Optional[UploadFile], what: str = "Reference audio") -> Optional[Audio]:
    if upload is None:
        return None
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{what} is larger than 25 MB.")
    if not data:
        return None
    return load_audio(data)


def _generation_form(raw: Optional[str]) -> Optional[GenerationOptions]:
    """Parse the JSON-encoded ``generation`` multipart field."""
    if not raw:
        return None
    try:
        return GenerationOptions.model_validate_json(raw)
    except ValidationError as e:
        raise HTTPException(422, f"Invalid 'generation' JSON: {e.errors()}") from e


def create_app(engine: Optional[Engine] = None) -> FastAPI:
    app = FastAPI(
        title="Poly Chatterbox",
        version=__version__,
        description="Chatterbox text-to-speech: zero-shot voice cloning in English (Chatterbox, Turbo, Nano) and "
                    "23 languages (Multilingual), paralinguistic tags, voice conversion and a voice library.",
    )
    app.state.engine = engine

    # `generation` arrives as a JSON string inside multipart forms, so FastAPI never sees the model;
    # publish it in the schema anyway so /docs documents every option.
    base_openapi = app.openapi

    def openapi():
        if app.openapi_schema is None:
            schema = base_openapi()
            schema.setdefault("components", {}).setdefault("schemas", {})["GenerationOptions"] = \
                GenerationOptions.model_json_schema(ref_template="#/components/schemas/{model}")
        return app.openapi_schema

    app.openapi = openapi

    def eng() -> Engine:
        if app.state.engine is None:
            app.state.engine = get_engine()
        return app.state.engine

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(LookupError)
    async def _lookup_error(_: Request, exc: LookupError):
        return JSONResponse({"detail": str(exc.args[0] if exc.args else exc)}, status_code=404)

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    # ------------------------------------------------------------------ metadata

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    @app.get("/api/info", tags=["meta"])
    def info():
        e = eng()
        voices = e.library.list()
        return {
            "version": __version__,
            "backend": e.info(),
            "models": MODEL_LABELS,
            "languages": LANGUAGES,
            "sample_texts": {code: s["text"] for code, s in SAMPLES.items()},
            "multilingual_versions": {"default": e.settings.multilingual_version, "choices": list(MULTILINGUAL_VERSIONS)},
            "event_tags": EVENT_TAGS,
            "generation_options": GENERATION_OPTIONS,
            "voice_counts": {k: sum(v.kind == k for v in voices) for k in ("builtin", "sample", "cloned")},
            "reference_audio": {"min_seconds": e.settings.min_ref_seconds, "max_seconds": e.settings.max_ref_seconds},
            "max_chunk_chars": e.settings.max_chunk_chars,
        }

    @app.post("/api/watermark", tags=["meta"])
    async def watermark(audio: UploadFile = File(..., description="Audio to check (WAV/FLAC/OGG/MP3).")):
        """Detect the Perth watermark that Chatterbox embeds in all generated audio."""
        clip = await _read_upload(audio, "Audio")
        if clip is None:
            raise HTTPException(400, "audio is empty.")
        return await run_in_threadpool(eng().detect_watermark, clip)

    @app.post("/api/transcribe", tags=["meta"])
    async def transcribe(
        audio: UploadFile = File(..., description="Speech to transcribe (WAV/FLAC/OGG/MP3)."),
        language: Optional[str] = Form(None, description="Language spoken: a code (fr) or name (French). Default: detect it."),
    ):
        """Whisper speech-to-text: the transcript, plus timestamped segments."""
        clip = await _read_upload(audio, "Audio")
        if clip is None:
            raise HTTPException(400, "audio is empty.")
        return await run_in_threadpool(eng().transcribe, clip, language or None)

    # ------------------------------------------------------------------ voice library

    @app.get("/api/voices", tags=["voices"])
    def list_voices(kind: Optional[str] = None, language: Optional[str] = None, q: Optional[str] = None):
        voices = eng().library.list()
        if kind:
            voices = [v for v in voices if v.kind == kind]
        if language:
            voices = [v for v in voices if v.language.lower() == language.lower()
                      or LANGUAGES.get(v.language, "").lower() == language.lower()]
        if q:
            ql = q.lower()
            voices = [v for v in voices if ql in f"{v.id} {v.name} {v.description}".lower()]
        return [v.to_dict() for v in voices]

    @app.get("/api/voices/{voice_id}", tags=["voices"])
    def get_voice(voice_id: str):
        return eng().library.get(voice_id).to_dict()

    @app.post("/api/voices", tags=["voices"], status_code=201)
    async def save_voice(
        name: str = Form(...),
        ref_audio: UploadFile = File(..., description="Clip of the voice, 5 s or longer."),
        language: Optional[str] = Form(None, description="Language spoken in the clip (code or name)."),
        description: Optional[str] = Form(None),
    ):
        """Save a cloned voice to the library so it can be reused by id."""
        ref = await _read_upload(ref_audio)
        if ref is None:
            raise HTTPException(400, "ref_audio is empty.")
        voice = await run_in_threadpool(eng().save_voice, name, ref, language, description)
        return voice.to_dict()

    @app.delete("/api/voices/{voice_id}", tags=["voices"], status_code=204)
    def delete_voice(voice_id: str):
        eng().remove_voice(voice_id)
        return Response(status_code=204)

    # ------------------------------------------------------------------ saved prompts

    @app.get("/api/prompts", tags=["prompts"])
    def list_prompts():
        return [p.to_dict() for p in eng().prompts.list()]

    @app.get("/api/prompts/{prompt_id}", tags=["prompts"])
    def get_prompt(prompt_id: str):
        return eng().prompts.get(prompt_id).to_dict()

    @app.post("/api/prompts", tags=["prompts"], status_code=201)
    def save_prompt(req: PromptRequest, response: Response):
        """Save a prompt (text, mode, language, voice and generation options). Same name: replaces it (200)."""
        generation = req.generation.model_dump(exclude_none=True) if req.generation else None
        prompt, replaced = eng().save_prompt(req.name, req.text, req.mode, req.language, req.version, req.voice_id, generation)
        if replaced:
            response.status_code = 200
        return prompt.to_dict()

    @app.delete("/api/prompts/{prompt_id}", tags=["prompts"], status_code=204)
    def delete_prompt(prompt_id: str):
        eng().prompts.remove(prompt_id)
        return Response(status_code=204)

    # ------------------------------------------------------------------ core features

    @app.post("/api/tts/original", tags=["tts"], response_class=Response, responses=WAV)
    async def original(text: str = TEXT, ref_audio: Optional[UploadFile] = REF_AUDIO,
                       voice_id: Optional[str] = VOICE_ID, generation: Optional[str] = GENERATION):
        """Chatterbox (English, 500M): voice cloning with exaggeration and CFG/pace control."""
        ref, gen = await _read_upload(ref_audio), _generation_form(generation)
        return _audio_response(await run_in_threadpool(eng().original, text, ref, voice_id or None, params(gen)))

    async def _turbo(text, ref_audio, voice_id, generation, nano: bool) -> Response:
        ref, gen = await _read_upload(ref_audio), _generation_form(generation)
        result = await run_in_threadpool(eng().turbo, text, ref, voice_id or None, params(gen), nano)
        return _audio_response(result)

    @app.post("/api/tts/turbo", tags=["tts"], response_class=Response, responses=WAV)
    async def turbo(text: str = TEXT, ref_audio: Optional[UploadFile] = REF_AUDIO,
                    voice_id: Optional[str] = VOICE_ID, generation: Optional[str] = GENERATION):
        """Chatterbox-Turbo (English, 350M): low-latency TTS with paralinguistic tags such as [laugh] and [cough]."""
        return await _turbo(text, ref_audio, voice_id, generation, nano=False)

    @app.post("/api/tts/nano", tags=["tts"], response_class=Response, responses=WAV)
    async def nano(text: str = TEXT, ref_audio: Optional[UploadFile] = REF_AUDIO,
                   voice_id: Optional[str] = VOICE_ID, generation: Optional[str] = GENERATION):
        """Chatterbox-Nano (English, 110M): Turbo's architecture in a CPU-friendly size, with the same tags."""
        return await _turbo(text, ref_audio, voice_id, generation, nano=True)

    @app.post("/api/tts/multilingual", tags=["tts"], response_class=Response, responses=WAV)
    async def multilingual(
        text: str = TEXT,
        language: str = Form(..., description="Language of the text: a code (fr) or name (French)."),
        ref_audio: Optional[UploadFile] = REF_AUDIO,
        voice_id: Optional[str] = Form(None, description="Or a library voice. Neither: the language's sample speaker."),
        version: Optional[str] = Form(None, description="Checkpoint: v3 (recommended) or v2. Default: server setting."),
        generation: Optional[str] = GENERATION,
    ):
        """Chatterbox Multilingual (23 languages): voice cloning across languages."""
        ref, gen = await _read_upload(ref_audio), _generation_form(generation)
        result = await run_in_threadpool(eng().multilingual, text, language, ref, voice_id or None, version or None, params(gen))
        return _audio_response(result)

    @app.post("/api/vc", tags=["tts"], response_class=Response, responses=WAV)
    async def voice_conversion(
        source_audio: UploadFile = File(..., description="Speech to re-voice (WAV/FLAC/OGG/MP3)."),
        target_audio: Optional[UploadFile] = File(None, description="Clip of the target voice, 5 s or longer."),
        target_voice_id: Optional[str] = Form(None, description="Or a library voice. Neither: the built-in voice."),
    ):
        """Voice conversion: keep the words and timing of the source, change the voice to the target's."""
        source = await _read_upload(source_audio, "Source audio")
        if source is None:
            raise HTTPException(400, "source_audio is empty.")
        target = await _read_upload(target_audio, "Target audio")
        return _audio_response(await run_in_threadpool(eng().convert, source, target, target_voice_id or None))

    # ------------------------------------------------------------------ web UI

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app
