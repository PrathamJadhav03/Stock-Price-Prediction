---
phase: 5
plan: 2
wave: 2
depends_on:
  - "1"
files_modified:
  - README.md
  - tests/test_ops.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "README.md delivers comprehensive enterprise-grade production documentation featuring architecture diagrams, mathematical equations, CLI guide, REST API specs, Streamlit guide, and container deployment"
    - "System Architecture Mermaid diagram visualizes the complete end-to-end pipeline from data ingestion to model training, evaluation, REST API, and Streamlit dashboard"
    - "Mathematical foundations rigorously document log-return stationarity, trajectory reconstruction, Monte Carlo Dropout 95% confidence intervals, and basis-point transaction fee modeling"
    - "tests/test_ops.py validates Dockerfile, docker-compose.yml, and Makefile structure, syntax, and operational contract"
    - "Full test suite achieves 100% pass rate with 0 failures across all test modules (>=92 tests) satisfying NFR-05"
  artifacts:
    - "README.md exists with comprehensive production guide and Mermaid diagrams"
    - "tests/test_ops.py exists and validates Dockerfile, docker-compose.yml, and Makefile programmatically"
    - "Full pytest run executes with 100% pass rate across all tests"
---

# Plan 5.2: Production Documentation & Full Suite Verification

## Objective
Author comprehensive, enterprise-grade production documentation in `README.md` (superseding legacy student notebook notes with architectural flowcharts, rigorous mathematical formulation, CLI guides, REST API contracts, dashboard guides, and Docker orchestration), implement an automated operational test suite in `tests/test_ops.py` to programmatically validate container and tooling specifications, and execute full-suite regression testing across all 12 test modules to guarantee a 100% test pass rate (REQ-13, NFR-01, NFR-05).

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- Dockerfile
- docker-compose.yml
- Makefile
- src/api/schemas.py
- src/dashboard/app.py
- models/metrics_GOOGL.json
- reports/backtest/backtest_GOOGL.json

## Tasks

<task type="auto">
  <name>Comprehensive Enterprise Production Documentation in README.md</name>
  <files>
    README.md
  </files>
  <action>
    Transform `README.md` into exhaustive enterprise documentation for the production ML forecasting and financial backtesting system.

    Sections to include:
    1. **Header & Badges**:
       - Title: Production Stock Price Forecasting & Quantitative Backtesting System.
       - Badges: Python 3.11+, uv Package Manager, PyTorch 2.x, FastAPI REST, Streamlit Dashboard, Docker Multi-Stage, 100% Pytest Coverage.
    2. **System Architecture**:
       - Mermaid flowchart (`flowchart TD`) visualizing:
         - Market Ingestion (`src.data.ingestion`, yfinance cache & CSV fallback)
         - Feature Engineering (`src.features`, log returns, RSI, MACD, Bollinger Bands, ATR, EMA, Volatility, zero-leakage RobustScaler pipeline)
         - Model Training & Inference (`src.models.gru`, PyTorch GRU with LayerNorm, MC Dropout, NaivePersistenceBaseline)
         - Evaluation & Backtesting (`src.evaluation`, RMSE, MAE, MAPE, Directional Accuracy, WalkForwardBacktester with basis point transaction costs)
         - Serving Layer (`src.api.app`, FastAPI REST endpoints `/health`, `/predict`, `/backtest` with in-memory checkpoint caching)
         - UI Layer (`src.dashboard.app`, Streamlit interactive financial analytics dashboard)
    3. **Mathematical Foundations**:
       - **Stationarity & Log Returns**:
         Explain why modeling raw Close prices is statistically flawed (non-stationary random walk with time-varying variance).
         Log return formulation:
         $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
         Exact price trajectory reconstruction from initial observed price $P_0$:
         $$P_t = P_0 \exp\left(\sum_{\tau=1}^t r_\tau\right)$$
       - **Predictive Uncertainty via Monte Carlo Dropout**:
         Explain epistemic uncertainty quantification via stochastic forward passes at inference time ($T=50$ passes with active Dropout $p=0.2$):
         $$\hat{\mu}_t = \frac{1}{T}\sum_{k=1}^T \hat{r}_t^{(k)}, \quad \hat{\sigma}_t^2 = \frac{1}{T}\sum_{k=1}^T (\hat{r}_t^{(k)} - \hat{\mu}_t)^2$$
         95% Confidence Interval:
         $$[\hat{r}_{t, 2.5\%}, \hat{r}_{t, 97.5\%}]$$
         Reconstructed price confidence bounds:
         $$P_{t, 2.5\%} = P_0 \exp(\hat{r}_{t, 2.5\%}), \quad P_{t, 97.5\%} = P_0 \exp(\hat{r}_{t, 97.5\%})$$
       - **Walk-Forward Financial Backtesting & Frictions**:
         Trading decision rule with threshold $\theta$:
         $$\text{pos}_t = \begin{cases} 1 & \text{if } \hat{r}_t > \theta \\ 0 & \text{otherwise} \end{cases}$$
         Transaction friction modeling in basis points:
         $$\text{fee}_t = \text{bps} \times 10^{-4} \times |\Delta \text{pos}_t|$$
         Strategy net return:
         $$R_t^{\text{strat}} = \text{pos}_{t-1} R_t^{\text{asset}} - \text{fee}_t$$
         Annualized Sharpe Ratio, Sortino Ratio, and Maximum Drawdown formulas.
    4. **Comparative Benchmark Results**:
       - Performance table across all 4 tickers (`GOOGL`, `AMZN`, `IBM`, `MSFT`) comparing GRU against `NaivePersistenceBaseline` on Directional Accuracy (Sign Hit Rate %), RMSE, MAE, and Annualized Sharpe.
    5. **Quickstart & Hermetic Installation**:
       - Clone repository and install dependencies with `uv sync`.
    6. **CLI Pipeline Guide**:
       - Data preparation: `uv run python scripts/prepare_data.py --symbol GOOGL`
       - Model training: `uv run python scripts/train.py --symbol GOOGL`
       - Financial backtesting: `uv run python scripts/backtest.py --symbol GOOGL --fee-bps 10.0`
    7. **REST API Reference**:
       - Documentation of endpoints (`GET /health`, `POST /predict`, `POST /backtest`) with request parameters, schemas, and complete JSON response examples.
    8. **Interactive Streamlit Dashboard**:
       - Visual tour and user guide for dashboard tabs (Overview & Candlesticks, Forecast & MC Fan Chart, Strategy Backtester & Equity Curve).
    9. **Docker & Compose Deployment**:
       - Instructions for `docker compose build` and `docker compose up -d`.
       - Port mapping (8000 for FastAPI, 8501 for Streamlit) and healthcheck verification.
    10. **Developer Tooling (Makefile)**:
        - Table of Makefile targets (`install`, `train`, `backtest`, `test`, `api`, `dashboard`, `docker-up`, `clean`).
    11. **Testing & Verification**:
        - Instructions for running the complete `pytest` test suite via `uv run pytest tests/ -v`.

    AVOID: Retaining obsolete notebook-style personal notes or vague instructions.
    USE: Exact, polished technical documentation with LaTeX math expressions, Mermaid diagrams, and reproducible CLI/API snippets.
  </action>
  <verify>
    uv run python -c "
with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()
assert len(content.splitlines()) > 250, 'README.md is too short'
assert '```mermaid' in content, 'Missing Mermaid diagram'
assert 'r_t = \ln' in content or 'r_t =' in content, 'Missing return equation'
assert '/predict' in content and '/backtest' in content, 'Missing API endpoint docs'
assert 'docker compose' in content, 'Missing Docker compose docs'
assert 'Makefile' in content, 'Missing Makefile docs'
print('README.md enterprise documentation verified successfully')
"
  </verify>
  <done>
    `README.md` is a complete, production-grade guide with architecture diagrams, mathematical equations, CLI guides, API schemas, and deployment instructions.
  </done>
</task>

<task type="auto">
  <name>Operational Configuration and Container Contract Test Suite</name>
  <files>
    tests/test_ops.py
  </files>
  <action>
    Create `tests/test_ops.py` to programmatically test and validate all operational artifacts (`Dockerfile`, `.dockerignore`, `docker-compose.yml`, and `Makefile`).

    Steps:
    1. In `tests/test_ops.py`, implement test functions:
       - `test_dockerfile_multi_stage_structure`:
         - Verify `Dockerfile` exists.
         - Verify stage 1: `FROM python:3.11-slim-bookworm AS builder`.
         - Verify `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/` appears.
         - Verify `UV_COMPILE_BYTECODE=1` is set.
         - Verify stage 2: `FROM python:3.11-slim-bookworm AS runtime`.
         - Verify `groupadd -g 1000 appgroup` and `useradd -u 1000 -g appgroup -s /bin/bash -m appuser` create non-root user.
         - Verify `USER appuser` switches away from root.
         - Verify `EXPOSE 8000 8501`.
       - `test_dockerignore_rules`:
         - Verify `.dockerignore` exists.
         - Verify exclusion of `.venv`, `.git`, `__pycache__`, `.pytest_cache`, `.coverage`, `.gsd`.
       - `test_docker_compose_validity`:
         - Read and parse `docker-compose.yml` with `yaml.safe_load`.
         - Verify services `api` and `dashboard` exist.
         - Verify `api` exposes `8000:8000` and configures healthcheck (`/health`).
         - Verify `dashboard` exposes `8501:8501` and has `depends_on` with `api: condition: service_healthy`.
         - Verify bridge network `stock_network` is configured.
       - `test_makefile_targets`:
         - Read `Makefile` and parse all target declarations.
         - Verify presence of required targets: `help`, `install`, `prepare-data`, `train`, `backtest`, `test`, `api`, `dashboard`, `docker-build`, `docker-up`, `docker-down`, `clean`.
         - Verify recipe lines use hard tab indentation (`\t`), not 4 spaces.
         - Verify targets invoke `uv run` or `uv sync` or `docker compose`.

    AVOID: Relying on a live Docker daemon inside test runners that might not have Docker installed.
    USE: Deterministic static AST and parser verification of configuration files.
  </action>
  <verify>
    uv run pytest tests/test_ops.py -v
  </verify>
  <done>
    `tests/test_ops.py` passes 100% across all operational configuration tests.
  </done>
</task>

<task type="auto">
  <name>Full-Suite End-to-End Regression Verification across All Test Modules</name>
  <files>
    tests/
  </files>
  <action>
    Execute and verify the complete automated test suite across all 12 test modules in `tests/`:
    - `tests/test_ingestion.py`
    - `tests/test_features.py`
    - `tests/test_pipeline.py`
    - `tests/test_dataset_baseline.py`
    - `tests/test_gru.py`
    - `tests/test_trainer.py`
    - `tests/test_metrics.py`
    - `tests/test_backtester.py`
    - `tests/test_backtest_cli.py`
    - `tests/test_api.py`
    - `tests/test_dashboard.py`
    - `tests/test_ops.py`

    Steps:
    1. Run `uv run pytest tests/ -v`.
    2. Confirm:
       - 100% test pass rate with 0 failures, 0 errors, and 0 skipped.
       - Total test count >= 92 tests.
       - No regressions across data ingestion, feature extraction, stationarity reconstruction, model inference, backtester execution, FastAPI endpoints, Streamlit dashboard, or operational configurations.

    AVOID: Skipping tests or using partial filter flags (`-k`).
    USE: Exhaustive `uv run pytest tests/ -v` execution confirming NFR-05.
  </action>
  <verify>
    uv run pytest tests/ -v
  </verify>
  <done>
    100% of all tests in the repository pass with 0 failures and 0 regressions.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `README.md` provides complete production documentation with Mermaid diagrams, mathematical foundations, CLI instructions, API specs, and Docker commands
- [ ] `tests/test_ops.py` programmatically validates `Dockerfile`, `.dockerignore`, `docker-compose.yml`, and `Makefile`
- [ ] `uv run pytest tests/ -v` passes 100% with >= 92 tests across all 12 test suites with 0 failures (NFR-05)

## Success Criteria
- [ ] All tasks verified passing
- [ ] Must-haves confirmed
- [ ] 100% test pass rate achieved across full repository

