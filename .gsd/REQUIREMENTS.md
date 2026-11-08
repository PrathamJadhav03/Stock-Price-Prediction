---
milestone: v1.0
updated: 2026-11-08T00:28:00+05:30
---

# Requirements

## Overview
Traceable requirement catalog linking SPEC goals to engineering verification across all five phases.

---

## Functional Requirements

| ID | Requirement | Source | Phase | Status |
|----|-------------|--------|-------|--------|
| REQ-01 | Configure `pyproject.toml` and generate `uv.lock` with core and dev dependencies using `uv` | SPEC Goal 1 | 1 | Complete |
| REQ-02 | Ingestion module downloading live `yfinance` daily bars with local disk caching in `data/raw/` and fallback to `Data/*.csv` | SPEC Goal 1 | 1 | Complete |
| REQ-03 | Compute Log Returns $\ln(P_t / P_{t-1})$ as model target with trajectory inversion utility $\hat{P}_t = P_0 \exp(\sum \hat{r}_\tau)$ | SPEC Goal 1 | 1 | Complete |
| REQ-04 | Feature engineering computing RSI, MACD, Bollinger Bands, ATR, SMA/EMA, and rolling volatility | SPEC Goal 1 | 1 | Complete |
| REQ-05 | Zero-leakage temporal split and serialized scaling (`joblib` export of fitted `RobustScaler`/`StandardScaler`) | SPEC Goal 1 | 1 | Complete |
| REQ-06 | PyTorch GRU architecture featuring LayerNorm, Dropout, and Monte Carlo (MC) Dropout inference for 95% confidence intervals | SPEC Goal 2 | 2 | Complete |
| REQ-07 | Naive Persistence Baseline ($\hat{P}_t = P_{t-1}$) implementing uniform interface for statistical benchmarking | SPEC Goal 2 | 2 | Complete |
| REQ-08 | Modular PyTorch trainer with EarlyStopping, ReduceLROnPlateau, gradient clipping, driven by `configs/config.yaml` | SPEC Goal 2 | 2 | Complete |
| REQ-09 | Metric calculation module for RMSE, MAE, MAPE, and Directional Accuracy (Sign Hit Rate %) | SPEC Goal 3 | 3 | Pending |
| REQ-10 | Walk-Forward financial backtester simulating long/flat strategy with basis point transaction costs, Sharpe, and Max Drawdown | SPEC Goal 3 | 3 | Pending |
| REQ-11 | FastAPI application exposing `/health`, `/predict` (predictions + 95% CI), and `/backtest` endpoints with Pydantic schemas | SPEC Goal 4 | 4 | Pending |
| REQ-12 | Interactive Streamlit dashboard porting Plotly candlestick, indicator overlays, forecast fan charts, and strategy equity curves | SPEC Goal 4 | 4 | Pending |
| REQ-13 | Automated test suite verifying leakage prevention, tensor dimensions, and API contracts executed via `uv run pytest` | SPEC Goal 5 | 5 | Pending |
| REQ-14 | Multi-stage `Dockerfile` (`ghcr.io/astral-sh/uv`), `docker-compose.yml`, and `Makefile` wrapping `uv run` commands | SPEC Goal 5 | 5 | Pending |

---

## Non-Functional Requirements

| ID | Requirement | Category | Phase | Status |
|----|-------------|----------|-------|--------|
| NFR-01 | Strict `uv` execution across all scripts, tests, servers, and tooling | Tooling / Env | All | Pending |
| NFR-02 | Device-agnostic PyTorch execution (CUDA enabled if available, fallback CPU) | Performance | 2 | Complete |
| NFR-03 | Model inference latency $< 100\text{ ms}$ for 50-step MC Dropout simulation | Performance | 4 | Pending |
| NFR-04 | Modular package structure in `src/` following standard Python packaging guidelines | Maintainability | 1-5 | Pending |
| NFR-05 | 100% test pass rate for automated test suite | Reliability | 5 | Pending |

---

## Constraints

| ID | Constraint | Source | Impact |
|----|------------|--------|--------|
| CON-01 | Exclusive usage of `uv` (`uv run`, `pyproject.toml`, `uv.lock`) | Strict Constraint | Eliminates `pip` and direct `venv` |
| CON-02 | Stationary modeling on Log Returns ($\ln(P_t / P_{t-1})$) | Financial ML | Prevents spurious correlation from raw Close |
| CON-03 | Zero lookahead leakage via strict temporal splitting and scaler serialization | Data Science | Prevents data leakage between train and test |
