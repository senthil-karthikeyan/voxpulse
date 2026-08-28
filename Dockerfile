# ==============================================================================
# Dockerfile for VoxPulse Voice Attribute Inference Service
# Production-ready, ephemeral audio handling, uv-powered build
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

# Install project dependencies into virtualenv
RUN uv sync --frozen --no-install-project

# Copy application source code and scripts
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY README.md ./

# Sync project wheel
RUN uv sync --frozen

# Create unprivileged user for security
RUN useradd -m -u 1000 voxpulse && \
    mkdir -p /app/model_weights /app/pretrained_models && \
    chown -R voxpulse:voxpulse /app

USER voxpulse

# Expose FastAPI application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn via uv run
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
