# poly-chatterbox: web UI + REST API on an NVIDIA GPU.
#
# The PyTorch CUDA wheels bundle the CUDA runtime, so a slim Python base is enough. The host
# only needs the NVIDIA driver and the NVIDIA Container Toolkit. Model weights are not baked in:
# they download on first use into /data/cache (mount a volume there).
#
#   docker build -t poly-chatterbox .
#   docker run --gpus all -p 8000:8000 -v chatterbox-cache:/data/cache -v chatterbox-library:/data/library poly-chatterbox

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

# git: resemble-perth (the watermarker) is installed from its git repository.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv

# Install into /opt/venv with the image's Python; torch comes from the CUDA 12.8 index pinned in
# uv.lock (driver >= 570, every GPU from Turing to Blackwell).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /src
# Dependencies first, in their own layer: torch is by far the largest and changes least.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra chatterbox --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra chatterbox --no-editable \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm

# ffmpeg: lets librosa decode compressed reference clips (MP3, M4A etc.).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data/cache /data/library \
    && chown -R app:app /data

COPY --from=build /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    NUMBA_CACHE_DIR=/data/cache/numba \
    HF_HUB_DISABLE_TELEMETRY=1 \
    POLY_CHATTERBOX_DATA_DIR=/data/library \
    POLY_CHATTERBOX_SERVE_ARGS=""

USER app
WORKDIR /data
VOLUME ["/data/cache", "/data/library"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Extra `serve` flags (e.g. "--preload") go in POLY_CHATTERBOX_SERVE_ARGS; everything else is a
# POLY_CHATTERBOX_* variable (see README.md).
CMD ["sh", "-c", "exec poly-chatterbox serve --host 0.0.0.0 --port 8000 $POLY_CHATTERBOX_SERVE_ARGS"]
