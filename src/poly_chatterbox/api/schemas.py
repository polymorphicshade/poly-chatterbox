from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..backends import GENERATION_OPTIONS, GenerationParams


def _describe(name: str) -> str:
    opt = GENERATION_OPTIONS[name]
    defaults = {m: d for m, d in opt["defaults"].items() if d is not None}
    if not defaults:
        return opt["description"]
    if len(set(defaults.values())) == 1:
        models = "all models" if len(defaults) == 4 else ", ".join(defaults)
        return f"{opt['description']} Default: {next(iter(defaults.values()))} ({models})."
    return f"{opt['description']} Defaults: " + ", ".join(f"{m} {d}" for m, d in defaults.items()) + "."


def _field(name: str, **kw):
    opt = GENERATION_OPTIONS[name]
    if opt["min"] is not None:
        kw.setdefault("ge", opt["min"])
    if opt["max"] is not None:
        kw.setdefault("le", opt["max"])
    return Field(None, description=_describe(name), **kw)


class GenerationOptions(BaseModel):
    """Sampling options. Unset fields use the model's defaults; options a model doesn't use are ignored."""

    model_config = ConfigDict(extra="forbid")

    exaggeration: Optional[float] = _field("exaggeration")
    cfg_weight: Optional[float] = _field("cfg_weight")
    temperature: Optional[float] = _field("temperature")
    min_p: Optional[float] = _field("min_p")
    top_p: Optional[float] = _field("top_p")
    top_k: Optional[int] = _field("top_k")
    repetition_penalty: Optional[float] = _field("repetition_penalty")
    norm_loudness: Optional[bool] = _field("norm_loudness")
    seed: Optional[int] = _field("seed")

    def to_params(self) -> GenerationParams:
        return GenerationParams(**self.model_dump())


def params(opts: Optional[GenerationOptions]) -> GenerationParams:
    return opts.to_params() if opts else GenerationParams()


class PromptRequest(BaseModel):
    """A prompt to save. Saving under an existing name replaces that prompt."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=20000, description="The text, including any [pause]/[settings]/[voice] markup.")
    mode: Optional[Literal["original", "turbo", "nano", "multilingual"]] = Field(None, description="Model to speak it with.")
    language: Optional[str] = Field(None, description="Language code or name (multilingual).")
    version: Optional[Literal["v2", "v3"]] = Field(None, description="Multilingual checkpoint.")
    voice_id: Optional[str] = Field(None, description="Library voice to speak it with (see GET /api/voices).")
    generation: Optional[GenerationOptions] = None
