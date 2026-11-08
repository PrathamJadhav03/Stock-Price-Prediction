# Makefile for Production ML Stock Forecasting & Operational Tooling
# Enforces strict uv execution and reproducible workflows.

PYTHON := uv run python
PYTEST := uv run pytest
UVICORN := uv run uvicorn
STREAMLIT := uv run streamlit
DOCKER_COMPOSE := docker compose
SYMBOL ?= GOOGL
START_DATE ?= 2006-01-01
END_DATE ?= 2018-01-01

.PHONY: help install prepare-data train backtest test api dashboard docker-build docker-up docker-down clean

help:
	@echo "Production ML Stock Forecasting - Operational Commands"
	@echo "======================================================"
	@echo "  make install        - Install locked dependencies via uv sync"
	@echo "  make prepare-data   - Ingest & process market data (SYMBOL=GOOGL)"
	@echo "  make train          - Train GRU model with early stopping (SYMBOL=GOOGL)"
	@echo "  make backtest       - Run vectorized backtest engine (SYMBOL=GOOGL)"
	@echo "  make test           - Run full test suite via pytest"
	@echo "  make api            - Run FastAPI server with reload (port 8000)"
	@echo "  make dashboard      - Launch Streamlit interactive dashboard (port 8501)"
	@echo "  make docker-build   - Build multi-stage container image"
	@echo "  make docker-up      - Run containerized api and dashboard services"
	@echo "  make docker-down    - Stop and remove containerized services"
	@echo "  make clean          - Clean transient caches, bytecode, and artifacts"

install:
	uv sync

prepare-data:
	$(PYTHON) scripts/prepare_data.py --symbol $(SYMBOL) --start-date $(START_DATE) --end-date $(END_DATE)

train:
	$(PYTHON) scripts/train.py --symbol $(SYMBOL)

backtest:
	$(PYTHON) scripts/backtest.py --symbol $(SYMBOL)

test:
	$(PYTEST) tests/ -v

api:
	$(UVICORN) src.api.app:app --reload --port 8000

dashboard:
	$(STREAMLIT) run src/dashboard/app.py

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for pattern in ['**/__pycache__', '**/.pytest_cache', '**/*.egg-info', 'build', 'dist', 'htmlcov'] for p in pathlib.Path('.').glob(pattern) if '.venv' not in p.parts and '.git' not in p.parts]; [p.unlink() for pattern in ['**/*.pyc', '**/*.pyo', '.coverage'] for p in pathlib.Path('.').glob(pattern) if p.is_file() and '.venv' not in p.parts and '.git' not in p.parts]; print('Clean completed.')"

