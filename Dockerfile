# syntax=docker/dockerfile:1

# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

# Install official Astral uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and copy link mode for hermetic cross-stage transfer
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Install locked production dependencies into /app/.venv
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime
FROM python:3.11-slim-bookworm AS runtime

# Install curl for container healthchecks and clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy uv binaries to runtime for uniform developer CLI operations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy isolated virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application sources and artifacts with appuser ownership
COPY --chown=appuser:appgroup pyproject.toml uv.lock ./
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/processed/ ./data/processed/

# Hardened non-root execution
USER appuser

# Expose API (8000) and Dashboard (8501) ports
EXPOSE 8000 8501

# Default command: FastAPI API backend via uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

