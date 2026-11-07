# SPEC.md — Project Specification

> **Status**: `FINALIZED`
>
> ⚠️ **Planning Lock**: Spec is finalized. Phase planning may proceed.

## Vision
Transform the proof-of-concept Jupyter Notebook (`Stock_Price_Prediction_gru.ipynb`) and historical equity dataset (`Data/`) into an enterprise-grade, production Machine Learning system for financial time-series forecasting and quantitative backtesting. The architecture replaces fragile ad-hoc notebook routines with a modular, reproducible Python package managed strictly by `uv`, enforcing stationarity via log-returns, eliminating feature leakage, providing uncertainty quantification via Monte Carlo Dropout, benchmarking against a naive persistence baseline, and exposing capabilities through both a FastAPI service and an interactive Streamlit analytics dashboard.

---

## Goals

1. **Phase 1: Environment, Ingestion & Leak-Free Feature Engineering**
   - Initialize package and lockfile using `uv` (`pyproject.toml`, `uv.lock`) with pinned core and dev dependencies.
   - Implement resilient data ingestion with live `yfinance` download, local disk caching (`data/raw/`), and fallback to existing `Data/*.csv` records.
   - Address time-series non-stationarity by modeling Log Returns ($\ln(P_t / P_{t-1})$) instead of raw Close prices, accompanied by exact price trajectory reconstruction.
   - Compute a comprehensive technical indicator suite (RSI, MACD, Bollinger Bands, ATR, SMA/EMA, rolling volatility).
   - Enforce zero lookahead leakage by fitting scalers strictly on train splits within serialized `joblib` pipelines.

2. **Phase 2: PyTorch GRU Architecture & Baseline Benchmark**
   - Modularize the PyTorch GRU architecture with LayerNorm, Dropout, and Monte Carlo (MC) Dropout to estimate 95% predictive confidence intervals.
   - Establish a `NaivePersistenceBaseline` ($\hat{P}_t = P_{t-1}$) to empirically prove predictive superiority over a random walk.
   - Develop a robust, configuration-driven PyTorch `trainer.py` supporting EarlyStopping, `ReduceLROnPlateau`, and gradient clipping driven by `configs/config.yaml`.

3. **Phase 3: Quantitative Evaluation & Financial Backtesting**
   - Implement point-forecast and directional metrics: RMSE, MAE, MAPE, and Directional Accuracy (Sign Hit Rate %).
   - Build a Walk-Forward Backtester simulating a realistic long/flat trading strategy with configurable transaction costs (basis points), computing Sharpe Ratio, Maximum Drawdown, and Cumulative Return relative to Buy-and-Hold.

4. **Phase 4: API Serving & Interactive Dashboard**
   - Construct a high-performance FastAPI service providing `/health`, `/predict` (predictions + 95% confidence intervals), and `/backtest` endpoints.
   - Deliver a Streamlit web application porting and expanding notebook Plotly visualizations into interactive financial charts (Candlestick with overlays, fan forecast charts, and backtest performance scorecards).

5. **Phase 5: Automated Testing, Multi-Stage Dockerization & Operational Tooling**
   - Deliver an exhaustive `pytest` suite executed strictly via `uv run pytest` verifying leak-free preprocessing, tensor dimensions, model inference, and REST endpoints.
   - Package the application into a minimal, reproducible multi-stage `Dockerfile` using `ghcr.io/astral-sh/uv`, accompanied by `docker-compose.yml` and a `Makefile` wrapping `uv run` developer workflows.

---

## Non-Goals (Out of Scope)

- **Direct PIP/VENV Usage**: No manual `requirements.txt`, `pip install`, or standard virtualenv activation. All operations must run via `uv run`.
- **Live Brokerage Execution**: System performs simulated walk-forward backtesting; live order routing and real-money brokerage integration are strictly out of scope for v1.
- **High-Frequency Tick Streaming**: System focuses on daily bar intervals and swing horizons; sub-second order book and tick data are excluded.
- **Paid Market Data Feeds**: Exclusively utilizes open market data (`yfinance`) and static historical benchmark CSVs in `Data/`.

---

## Technical Constraints

- **Package Management**: Exclusive use of `uv` (`pyproject.toml` and `uv.lock`). Execution via `uv run ...`.
- **Stationarity Protocol**: Model target is strictly log returns ($\ln(P_t / P_{t-1})$); prices must be reconstructed deterministically.
- **Zero-Leakage Guarantee**: Feature transforms, rolling calculations, and scaling must prevent lookahead bias; fit on train, transform on test.
- **Hardware Agnostic**: Dynamic device selection (`cuda` if available, falling back gracefully to `cpu`).
- **Reproducibility**: Global seed initialization across Python `random`, `numpy`, and `torch`.

---

## Success Criteria

- [ ] `uv sync` resolves and locks all dependencies cleanly in a hermetic environment.
- [ ] Ingestion fetches ticker data or seamlessly reads existing `Data/*.csv` fallback.
- [ ] Technical indicators and log return calculations contain 0 NaN artifacts in modeled windows.
- [ ] Reconstruction utility converts predicted log returns back into price trajectories with error $< 10^{-5}$ on identity test.
- [ ] PyTorch GRU trains via `uv run python -m src.models.trainer` and triggers early stopping on plateau.
- [ ] Monte Carlo Dropout outputs empirical mean and 95% confidence intervals (2.5th and 97.5th percentiles).
- [ ] Model outperforms `NaivePersistenceBaseline` on directional accuracy (> 50% hit rate).
- [ ] Backtest engine computes Sharpe Ratio, Max Drawdown, and equity curves factoring transaction fees.
- [ ] FastAPI serves requests with Pydantic validation and OpenAPI documentation.
- [ ] Streamlit dashboard renders candlestick, forecast fan charts, and backtesting comparison.
- [ ] `uv run pytest` executes with 100% pass rate across unit and integration tests.
- [ ] Multi-stage Docker image builds cleanly and runs both FastAPI and Streamlit services.

---

*Last updated: 2026-11-08*
