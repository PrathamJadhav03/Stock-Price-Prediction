---
milestone: v1.0
version: 1.0.0
updated: 2026-11-08T00:28:00+05:30
---

# Roadmap

> **Current Phase:** Not started (Ready for Phase 1)  
> **Status:** ⬜ Ready for Planning

## Must-Haves (from SPEC)

- [x] Exclusive `uv` environment and dependency management (`pyproject.toml`, `uv.lock`)
- [x] Ingestion with `yfinance` caching and `Data/*.csv` fallback
- [x] Log-returns modeling with exact price reconstruction
- [x] Leakage-free technical indicators and serialized `joblib` scalers
- [ ] PyTorch GRU with LayerNorm, Dropout, and 95% MC Dropout uncertainty estimation
- [ ] Naive Persistence Baseline ($\hat{P}_t = P_{t-1}$) comparison
- [ ] Point forecast & directional accuracy metrics (Sign Hit Rate %)
- [ ] Walk-forward financial backtester with transaction costs (bps), Sharpe Ratio, and Drawdowns
- [ ] FastAPI serving `/health`, `/predict`, and `/backtest`
- [ ] Streamlit interactive dashboard with candlestick, forecast fan chart, and backtesting scorecard
- [ ] Pytest suite passing via `uv run pytest`
- [ ] Multi-stage Docker packaging (`ghcr.io/astral-sh/uv`), `docker-compose.yml`, and `Makefile`

---

## Phases

### Phase 1: `uv` Setup, Data Ingestion & Leak-Free Feature Engineering
**Status:** ✅ Complete  
**Objective:** Establish the hermetic `uv` environment, build cached `yfinance` ingestion with `Data/*.csv` fallback, compute log returns and technical indicators, and implement serialized leak-free scaling.  
**Requirements:** REQ-01, REQ-02, REQ-03, REQ-04, REQ-05, NFR-01, CON-01, CON-02, CON-03  

**Plans:**
- [x] Plan 1.1: `uv` project initialization, dependency locking, and resilient cached data ingestion with CSV fallback (`.gsd/phases/1/1-PLAN.md` - Wave 1)
- [x] Plan 1.2: Stationary log returns with price trajectory reconstruction and zero-lookahead technical indicator feature engine (`.gsd/phases/1/2-PLAN.md` - Wave 2)
- [x] Plan 1.3: Zero-leakage temporal data splitting, serialized scaler pipeline, and end-to-end data preparation CLI (`.gsd/phases/1/3-PLAN.md` - Wave 3)

**Deliverables:**
- `pyproject.toml` and `uv.lock`
- `src/data/ingestion.py` (caching and historical fallback)
- `src/features/returns.py` (log returns & trajectory reconstruction)
- `src/features/technical.py` (RSI, MACD, Bollinger Bands, ATR, EMA/SMA, Volatility)
- `src/features/pipeline.py` (temporal train/test splitting and serialized scaling)

---

### Phase 2: PyTorch GRU Architecture & Baseline Engine
**Status:** ✅ Complete  
**Objective:** Modularize the PyTorch GRU architecture with LayerNorm, Dropout, and Monte Carlo Dropout for uncertainty estimation; implement the naive persistence baseline and a config-driven PyTorch trainer with early stopping.  
**Depends on:** Phase 1  
**Requirements:** REQ-06, REQ-07, REQ-08, NFR-02  

**Plans:**
- [x] Plan 2.1: Configuration schema, sliding window sequence dataset, and Naive Persistence Baseline (`.gsd/phases/2/1-PLAN.md` - Wave 1)
- [x] Plan 2.2: PyTorch StockGRU architecture with LayerNorm, Dropout, and Monte Carlo Dropout 95% confidence intervals (`.gsd/phases/2/2-PLAN.md` - Wave 2)
- [x] Plan 2.3: Config-driven PyTorch trainer with EarlyStopping, gradient clipping, checkpoint persistence, and baseline comparison (`.gsd/phases/2/3-PLAN.md` - Wave 3)

**Deliverables:**
- `configs/config.yaml`
- `src/models/gru.py` (GRU with LayerNorm and MC Dropout)
- `src/models/baseline.py` (NaivePersistenceBaseline)
- `src/models/dataset.py` (Time-series sequence DataLoader)
- `src/models/trainer.py` (EarlyStopping, ReduceLROnPlateau, gradient clipping)
- `scripts/train.py` (End-to-end training and benchmark comparison runner)
- `models/gru_GOOGL.pt` (Trained model checkpoint)
- `models/metrics_GOOGL.json` (Comparative evaluation report)

---

### Phase 3: Quantitative Evaluation & Financial Backtesting
**Status:** ✅ Complete  
**Objective:** Implement standard forecasting error metrics along with directional accuracy (Sign Hit Rate %); construct a Walk-Forward backtesting engine simulating a long/flat strategy with basis point transaction costs, Sharpe Ratio, and Max Drawdown against Buy-and-Hold.  
**Depends on:** Phase 2  
**Requirements:** REQ-09, REQ-10  

**Plans:**
- [x] Plan 3.1: Forecasting & Financial Performance Metrics Engine (`.gsd/phases/3/1-PLAN.md` - Wave 1)
- [x] Plan 3.2: Event-Driven Walk-Forward Backtester with Transaction Cost Modeling (`.gsd/phases/3/2-PLAN.md` - Wave 2)
- [x] Plan 3.3: Backtesting CLI Runner, Artifact Serialization & Scorecard Generation (`.gsd/phases/3/3-PLAN.md` - Wave 3)

**Deliverables:**
- `src/evaluation/metrics.py` (Forecasting and financial portfolio metrics)
- `src/evaluation/backtester.py` (Walk-forward simulator and BacktestResult)
- `src/evaluation/__init__.py` (Evaluation exports)
- `scripts/backtest.py` (Walk-forward backtest CLI runner)
- `reports/backtest/backtest_{symbol}.json` (Serialized KPIs)
- `reports/backtest/equity_curve_{symbol}.csv` (Time-series equity trajectories)

---

### Phase 4: FastAPI Serving & Interactive Streamlit Dashboard
**Status:** 🔄 Ready for Planning  
**Objective:** Expose the model through high-performance REST API endpoints (`/health`, `/predict`, `/backtest`), and provide a rich Streamlit analytics dashboard porting and upgrading Plotly visualizations with candlestick charts, uncertainty fan charts, and backtest scorecards.  
**Depends on:** Phase 3  
**Requirements:** REQ-11, REQ-12, NFR-03  
**Deliverables:**
- `src/api/app.py`, `src/api/schemas.py`, `src/api/routes.py`
- `src/dashboard/app.py`
- `src/dashboard/components.py` (Candlestick + technical overlays, MC Dropout fan charts, performance curves)

---

### Phase 5: Testing, Multi-Stage Dockerization & Operational Tooling
**Status:** ⬜ Not Started  
**Objective:** Deliver an end-to-end `pytest` verification suite run via `uv run pytest` covering feature leakage prevention, tensor dimensions, and API contracts; package the production application into an optimized multi-stage Docker container with `docker-compose.yml` and `Makefile`.  
**Depends on:** Phase 4  
**Requirements:** REQ-13, REQ-14, NFR-04, NFR-05  
**Deliverables:**
- `tests/test_features.py` (leakage checks, return reconstruction)
- `tests/test_models.py` (tensor shapes, MC Dropout reproducibility)
- `tests/test_api.py` (FastAPI test client contract checks)
- `Dockerfile` (`ghcr.io/astral-sh/uv` multi-stage build)
- `docker-compose.yml` (API and Dashboard service orchestration)
- `Makefile` (Developer ergonomics for `uv run` commands)
- `README.md` (Production documentation)

