---
phase: 5
plan: 1
wave: 1
status: complete
files_created:
  - Dockerfile
  - .dockerignore
  - docker-compose.yml
  - Makefile
files_modified: []
verification:
  dockerfile_check: "PASS (multi-stage build, uv binary, non-root appuser, ports 8000 & 8501)"
  dockerignore_check: "PASS (.venv, caches, test artifacts, git directories excluded)"
  compose_check: "PASS (services 'api' and 'dashboard', healthcheck, dependency condition, bridge network)"
  makefile_check: "PASS (all 12 required targets verified with tab indentation)"
  regression_suite: "88 passed in 22.84s (uv run pytest tests/ -v)"
---

# Plan 5.1 Summary: Containerization & Operational Tooling

## Objective
Package the production ML forecasting and backtesting system into an optimized, secure multi-stage Docker container utilizing official Astral `uv` binaries (`ghcr.io/astral-sh/uv:latest`), orchestrate multi-container serving with `docker-compose.yml` (FastAPI backend and Streamlit dashboard with automated healthcheck validation), and provide an ergonomic developer `Makefile` encapsulating all project lifecycles under strict `uv run` execution.

## Deliverables & Architecture

### 1. Multi-Stage Dockerfile & .dockerignore (`Dockerfile`, `.dockerignore`)
- **Stage 1 (`builder`)**:
  - Base: `python:3.11-slim-bookworm AS builder`
  - Leverages official `ghcr.io/astral-sh/uv:latest` binary (`/uv`, `/uvx`)
  - Configures `UV_COMPILE_BYTECODE=1` and `UV_LINK_MODE=copy`
  - Installs locked production dependencies hermetically via `uv sync --frozen --no-install-project --no-dev`
- **Stage 2 (`runtime`)**:
  - Base: `python:3.11-slim-bookworm AS runtime`
  - Installs `curl` for container healthcheck execution and purges apt cache
  - Hardens security with dedicated non-root user `appuser` (UID 1000) and `appgroup` (GID 1000)
  - Copies compiled virtual environment (`/app/.venv`) with `chown=appuser:appgroup`
  - Copies project directories (`configs/`, `src/`, `scripts/`, `models/`, `data/processed/`)
  - Exposes port 8000 (FastAPI REST backend) and 8501 (Streamlit dashboard)
  - Configures default entrypoint: `CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]`
- **Build Exclusions (`.dockerignore`)**:
  - Systematically excludes `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.coverage`, `data/raw/`, `.gsd`, `.agents`, and IDE metadata while preserving models and processed features.

### 2. Multi-Container Orchestration (`docker-compose.yml`)
- Orchestrates two core application services with bridge networking:
  - **`api` Service**:
    - Runs `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
    - Binds host port `8000:8000`
    - Configures active HTTP healthcheck against `http://localhost:8000/health` (15s interval, 5s timeout, 3 retries, 10s start period)
  - **`dashboard` Service**:
    - Runs `streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0`
    - Binds host port `8501:8501`
    - Injects backend connection URL `API_BASE_URL=http://api:8000`
    - Enforces healthcheck dependency condition: `api` must be `service_healthy` before dashboard initialization
  - **Networking**:
    - Connects both services via isolated `stock_network` bridge driver.

### 3. Developer Operational Tooling (`Makefile`)
- Encapsulates every operational workflow using strict `uv run` and `uv sync` commands:
  - `make help`: Displays clean overview of all operational targets.
  - `make install`: Hermetic dependency installation via `uv sync`.
  - `make prepare-data`: Downloads and preprocesses market data (parameterized with `SYMBOL`, `START_DATE`, `END_DATE`).
  - `make train`: Trains GRU neural network with early stopping and checkpointing.
  - `make backtest`: Executes vectorized financial backtest with metric calculations.
  - `make test`: Runs full pytest test suite.
  - `make api`: Launches FastAPI ASGI server with auto-reload.
  - `make dashboard`: Launches Streamlit analytics dashboard.
  - `make docker-build`: Builds multi-stage Docker container image.
  - `make docker-up`: Launches containerized services in background.
  - `make docker-down`: Stops and tears down containerized services.
  - `make clean`: Cross-platform cleanup of bytecode, pytest caches, and coverage artifacts.
- Enforces strict Make hard tab indentation across all recipes.

## Verification Evidence
1. **Dockerfile & .dockerignore Verification**:
   - `python -c "assert 'FROM python:3.11-slim-bookworm AS builder' in df; assert 'USER appuser' in df; ..."` passed cleanly.
2. **Docker Compose Verification**:
   - `yaml.safe_load` passed cleanly, asserting `api`, `dashboard`, ports `8000:8000`, `8501:8501`, and `service_healthy` dependency condition.
3. **Makefile Verification**:
   - Checked all 12 targets present and validated tab indentation compliance.
4. **Full Test Suite Regression**:
   - `uv run pytest tests/ -v`: 88 passed in 22.84s (100% pass rate).

## Notes on Commit Protocol
On Windows host sandbox, `.git/index.lock` creation is restricted by permission sandbox. All target files (`Dockerfile`, `.dockerignore`, `docker-compose.yml`, `Makefile`) have been created, verified, and validated against the plan requirements for the orchestrator to commit outside the sandbox:
- `feat(phase-5): multi-stage Dockerfile and .dockerignore with uv and non-root user`
- `feat(phase-5): multi-service docker-compose with healthchecks and networking`
- `feat(phase-5): developer workflow Makefile wrapping uv execution`

