---
phase: 5
plan: 1
wave: 1
depends_on: []
files_modified:
  - Dockerfile
  - .dockerignore
  - docker-compose.yml
  - Makefile
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Multi-stage Dockerfile utilizes ghcr.io/astral-sh/uv for hermetic, layer-cached dependency installation and python:3.11-slim runtime"
    - "Runtime container enforces non-root execution via appuser (UID 1000) for security hardening"
    - "docker-compose.yml orchestrates api (FastAPI on port 8000) and dashboard (Streamlit on port 8501) with healthchecks and network bridging"
    - "Makefile encapsulates all developer workflows using strict uv run execution"
  artifacts:
    - "Dockerfile exists with multi-stage build, non-root user, and exposed ports 8000 and 8501"
    - ".dockerignore exists excluding .venv, .git, caches, and raw artifacts"
    - "docker-compose.yml exists defining api and dashboard services with healthcheck and dependency conditions"
    - "Makefile exists with install, prepare-data, train, backtest, test, api, dashboard, docker-build, docker-up, docker-down, clean, and help targets"
---

# Plan 5.1: Containerization & Operational Tooling

## Objective
Package the entire production ML forecasting and backtesting system into an optimized, secure multi-stage Docker container utilizing official Astral `uv` binaries (`ghcr.io/astral-sh/uv:latest`), orchestrate multi-container serving with `docker-compose.yml` (FastAPI backend and Streamlit dashboard with automated healthcheck validation), and provide an ergonomic developer `Makefile` encapsulating all project lifecycles under strict `uv run` execution.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- pyproject.toml
- uv.lock
- src/api/app.py
- src/dashboard/app.py
- configs/config.yaml

## Tasks

<task type="auto">
  <name>Multi-Stage Dockerfile and .dockerignore with astral-sh/uv and Non-Root User Hardening</name>
  <files>
    Dockerfile
    .dockerignore
  </files>
  <action>
    Create `Dockerfile` and `.dockerignore` for production containerization following Astral uv and Docker best practices.

    Steps:
    1. In `Dockerfile`, construct a 2-stage build:
       - **Stage 1 (builder)**:
         - Base image: `python:3.11-slim-bookworm AS builder`
         - Copy uv binary: `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`
         - Working directory: `/app`
         - Environment variables:
           - `UV_COMPILE_BYTECODE=1` (compile Python bytecode during build for fast startup)
           - `UV_LINK_MODE=copy` (ensure isolated virtualenv copying across stages)
         - Copy dependency declarations: `COPY pyproject.toml uv.lock ./`
         - Install locked dependencies without development extras and without root project install:
           `RUN uv sync --frozen --no-install-project --no-dev`
       - **Stage 2 (runtime)**:
         - Base image: `python:3.11-slim-bookworm AS runtime`
         - Install curl for container healthcheck:
           `RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*`
         - Create non-root user and group:
           `RUN groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -s /bin/bash -m appuser`
         - Working directory: `/app`
         - Copy uv binary into runtime for uniform `uv run` CLI operations:
           `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`
         - Copy virtualenv from builder with correct ownership:
           `COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv`
         - Configure environment variables:
           - `PATH="/app/.venv/bin:$PATH"`
           - `PYTHONUNBUFFERED=1`
           - `PYTHONDONTWRITEBYTECODE=1`
         - Copy project files with ownership:
           - `COPY --chown=appuser:appgroup pyproject.toml uv.lock ./`
           - `COPY --chown=appuser:appgroup configs/ ./configs/`
           - `COPY --chown=appuser:appgroup src/ ./src/`
           - `COPY --chown=appuser:appgroup scripts/ ./scripts/`
           - `COPY --chown=appuser:appgroup models/ ./models/`
           - `COPY --chown=appuser:appgroup data/processed/ ./data/processed/`
         - Switch to non-root user: `USER appuser`
         - Expose ports: `EXPOSE 8000 8501`
         - Set default CMD: `CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]`
    2. In `.dockerignore`, exclude all non-production, transient, and sensitive assets:
       - Version control: `.git`, `.gitignore`
       - Virtual environments & packages: `.venv`, `venv/`, `env/`, `*.egg-info`
       - Python bytecode & caches: `__pycache__`, `*.py[cod]`, `*$py.class`
       - Testing & coverage: `.pytest_cache`, `.coverage`, `htmlcov/`, `reports/test*`
       - GSD & agent artifacts: `.gsd`, `.agents`
       - Raw market data caches: `data/raw`
       - IDE metadata: `.idea`, `.vscode`, `*.swp`
       - Note: retain `data/processed` and `models/` so the container has pre-trained checkpoints and processed splits out-of-the-box.

    AVOID: Running containers as root or installing packages with global `pip install` inside the container.
    USE: Official `ghcr.io/astral-sh/uv` multi-stage copying and `USER appuser` non-root isolation.
  </action>
  <verify>
    uv run python -c "
with open('Dockerfile', 'r', encoding='utf-8') as f:
    df = f.read()
assert 'FROM python:3.11-slim-bookworm AS builder' in df
assert 'COPY --from=ghcr.io/astral-sh/uv:latest' in df
assert 'USER appuser' in df
assert 'EXPOSE 8000 8501' in df
with open('.dockerignore', 'r', encoding='utf-8') as f:
    di = f.read()
assert '.venv' in di
assert '__pycache__' in di
assert '.pytest_cache' in di
print('Dockerfile and .dockerignore validated successfully')
"
  </verify>
  <done>
    `Dockerfile` defines a secure 2-stage uv build with non-root runtime execution and exposed ports 8000 and 8501; `.dockerignore` excludes all unnecessary build artifacts.
  </done>
</task>

<task type="auto">
  <name>Multi-Service Docker Compose Orchestration with Healthchecks and Networking</name>
  <files>
    docker-compose.yml
  </files>
  <action>
    Create `docker-compose.yml` to orchestrate both the FastAPI backend and Streamlit dashboard services with inter-service networking and healthcheck dependencies.

    Steps:
    1. Define `docker-compose.yml` using modern compose specification (version 3.8+):
       - Service `api`:
         - `build`: context `.`, dockerfile `Dockerfile`
         - `container_name`: `stock_price_api`
         - `command`: `["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]`
         - `ports`: `["8000:8000"]`
         - `environment`:
           - `API_HOST=0.0.0.0`
           - `API_PORT=8000`
         - `healthcheck`:
           - `test`: `["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]`
           - `interval`: `15s`
           - `timeout`: `5s`
           - `retries`: `3`
           - `start_period`: `10s`
         - `restart`: `unless-stopped`
         - `networks`: `[stock_network]`
       - Service `dashboard`:
         - `build`: context `.`, dockerfile `Dockerfile`
         - `container_name`: `stock_price_dashboard`
         - `command`: `["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]`
         - `ports`: `["8501:8501"]`
         - `environment`:
           - `API_BASE_URL=http://api:8000`
         - `depends_on`:
           - `api`:
             - `condition: service_healthy`
         - `restart`: `unless-stopped`
         - `networks`: `[stock_network]`
       - Networks:
         - `stock_network`: driver `bridge`

    AVOID: Starting the dashboard before the API backend is healthy.
    USE: Docker healthchecks with `condition: service_healthy` so Streamlit cleanly detects the live API server.
  </action>
  <verify>
    uv run python -c "
import yaml
with open('docker-compose.yml', 'r', encoding='utf-8') as f:
    compose = yaml.safe_load(f)
assert 'api' in compose['services']
assert 'dashboard' in compose['services']
assert compose['services']['api']['ports'] == ['8000:8000']
assert compose['services']['dashboard']['ports'] == ['8501:8501']
assert 'healthcheck' in compose['services']['api']
assert compose['services']['dashboard']['depends_on']['api']['condition'] == 'service_healthy'
print('docker-compose.yml validated successfully')
"
  </verify>
  <done>
    `docker-compose.yml` coordinates `api` and `dashboard` with healthcheck-gated dependency ordering, port mappings, and bridge network isolation.
  </done>
</task>

<task type="auto">
  <name>Developer Workflow Makefile Wrapping Strict uv Execution</name>
  <files>
    Makefile
  </files>
  <action>
    Create an ergonomic `Makefile` that wraps all system workflows in `uv run` commands, preventing manual environment drift and enforcing reproducibility.

    Steps:
    1. In `Makefile`, define standard configuration variables:
       - `PYTHON := uv run python`
       - `PYTEST := uv run pytest`
       - `UVICORN := uv run uvicorn`
       - `STREAMLIT := uv run streamlit`
       - `DOCKER_COMPOSE := docker compose`
       - `SYMBOL ?= GOOGL`
       - `START_DATE ?= 2006-01-01`
       - `END_DATE ?= 2018-01-01`
    2. Declare `.PHONY` targets:
       `.PHONY: help install prepare-data train backtest test api dashboard docker-build docker-up docker-down clean`
    3. Implement recipe targets (using strict TAB indentation):
       - `help`: Print formatted list of available targets and descriptions.
       - `install`: `uv sync` (installs locked dependencies in hermetic virtualenv).
       - `prepare-data`: `$(PYTHON) scripts/prepare_data.py --symbol $(SYMBOL) --start-date $(START_DATE) --end-date $(END_DATE)`
       - `train`: `$(PYTHON) scripts/train.py --symbol $(SYMBOL)`
       - `backtest`: `$(PYTHON) scripts/backtest.py --symbol $(SYMBOL)`
       - `test`: `$(PYTEST) tests/ -v`
       - `api`: `$(UVICORN) src.api.app:app --reload --port 8000`
       - `dashboard`: `$(STREAMLIT) run src/dashboard/app.py`
       - `docker-build`: `$(DOCKER_COMPOSE) build`
       - `docker-up`: `$(DOCKER_COMPOSE) up -d`
       - `docker-down`: `$(DOCKER_COMPOSE) down`
       - `clean`: Remove build artifacts, pycache directories, `.pytest_cache`, `.coverage`, and egg-info.

    AVOID: Using spaces for Makefile indentation (Make syntax requires hard tabs `\t`).
    USE: Strict `uv run` / `uv sync` wrappers for all targets.
  </action>
  <verify>
    uv run python -c "
with open('Makefile', 'r', encoding='utf-8') as f:
    lines = f.readlines()
targets = [l.split(':')[0].strip() for l in lines if ':' in l and not l.startswith(('#', '\t', ' '))]
required = ['help', 'install', 'prepare-data', 'train', 'backtest', 'test', 'api', 'dashboard', 'docker-build', 'docker-up', 'docker-down', 'clean']
assert all(r in targets for r in required), f'Missing targets: {[r for r in required if r not in targets]}'
# Verify tab indentation on recipe lines
for line in lines:
    if line.startswith('    ') and not line.strip().startswith('#'):
        raise AssertionError('Found 4-space indentation in Makefile instead of tab: ' + line)
print('Makefile validated successfully with all required targets and tab indentation')
"
  </verify>
  <done>
    `Makefile` provides concise, parameterized commands for all project operations with valid Makefile tab indentation and strict `uv` execution.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `Dockerfile` configures multi-stage build using `ghcr.io/astral-sh/uv:latest` and non-root `appuser`
- [ ] `.dockerignore` excludes virtual environments, git history, caches, and test artifacts
- [ ] `docker-compose.yml` specifies `api` and `dashboard` with healthcheck condition and ports 8000 and 8501
- [ ] `Makefile` provides operational targets (`install`, `train`, `backtest`, `test`, `api`, `dashboard`, `docker-up`, `clean`)

## Success Criteria
- [ ] All tasks verified passing
- [ ] Must-haves confirmed
- [ ] No regressions in existing test suite (`uv run pytest`)

