# ==============================================================================
# Dockerfile for VoxPulse Voice Attribute Inference Service
# Production-ready, baked ECAPA-TDNN weights, uv-powered build
# ==============================================================================

FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install essential system dependencies (FFmpeg, libsndfile1, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official Astral binary
COPY --from=ghcr.io/astral-sh/uv:0.6.2 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency specifications first for optimal layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies into virtualenv (excluding project itself for caching)
RUN uv sync --frozen --no-install-project

# Copy application source code, scripts, model weights, and documentation
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY model_weights/ ./model_weights/
COPY README.md ./

# Sync project package
RUN uv sync --frozen

# Download and bake pretrained SpeechBrain ECAPA-TDNN model into image during build
RUN uv run python scripts/download_model.py

# Create unprivileged user for security and set file ownership
RUN useradd -m -u 1000 voxpulse && \
    chown -R voxpulse:voxpulse /app

USER voxpulse

# Expose FastAPI application port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn server via uv run
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
