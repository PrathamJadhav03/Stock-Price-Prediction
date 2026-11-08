---
phase: 4
plan: 2
wave: 2
status: complete
files_created:
  - src/dashboard/__init__.py
  - src/dashboard/components.py
  - src/dashboard/app.py
  - tests/test_dashboard.py
files_modified: []
verification:
  test_dashboard: "11 passed in 12.01s (uv run pytest tests/test_dashboard.py -v)"
  full_suite: "88 passed in 28.89s (uv run pytest tests/ -v)"
---

# Plan 4.2 Summary: Plotly Analytics Components & Interactive Streamlit Dashboard

## Objective
Deliver an interactive, enterprise-grade Streamlit financial analytics dashboard in `src/dashboard/` featuring modular Plotly visualization components, responsive quantitative sidebar controls, epistemic uncertainty modeling via 95% Monte Carlo Dropout fan charts, walk-forward compounding equity curves with underwater drawdowns, and dual-mode connection fallback (FastAPI REST service or in-process `ModelService`).

## Deliverables & Architecture

### 1. Modular Plotly Visualization Engine (`src/dashboard/components.py`, `src/dashboard/__init__.py`)
- `create_candlestick_chart`: Dual-panel interactive chart (75% / 25% height split) containing:
  - Row 1: Candlestick OHLC trace with conditional Bollinger Bands (`bb_upper`, `bb_middle`, `bb_lower` with translucent tonexty fill) and EMAs (10, 20, 50).
  - Row 2: Volume bars dynamically color-coded by daily price action (green for $Close \ge Open$, red for $Close < Open$).
  - Graceful degradation when optional technical indicator columns are missing.
- `create_forecast_fan_chart`: Interactive single-step price forecast visualization:
  - Historical price trajectory line.
  - Distribution-free 95% Monte Carlo Dropout confidence interval fan (`rgba(0, 245, 212, 0.25)` envelope).
  - Expected price trajectory with point forecast annotation and expected percentage delta.
- `create_equity_curve_chart`: Dual-panel backtest performance visualizer:
  - Row 1: Compounding portfolio equity curves for Strategy ($) and Buy & Hold benchmark ($).
  - Row 2: Underwater percentage drawdown trajectories ($\le 0\%$) with filled `tozeroy` strategy risk area.
- `format_kpi_cards`: Extracts and formats backtesting summary metrics into structured cards with benchmark comparison deltas:
  - Cumulative Return, Annualized Return (CAGR), Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, Profit Factor, Directional Accuracy, and Total Transaction Fees.

### 2. Interactive Streamlit Web Application (`src/dashboard/app.py`)
- **Dual-Mode Backend Client (`BackendClient`)**:
  - Automatically probes `http://localhost:8000/health`.
  - Connects to remote FastAPI REST endpoints when online, and seamlessly falls back to local in-process `ModelService` when offline.
- **Sidebar Controls**:
  - Backend operational mode and compute device status badge.
  - Stock Ticker selector (`GOOGL`, `AMZN`, `IBM`, `MSFT`).
  - Lookback Window slider (30 to 365 trading days).
  - Monte Carlo Dropout Samples slider (10 to 200 passes).
  - Trading Strategy selector (`long_flat` vs `long_short`).
  - Transaction Cost slider (0.0 to 50.0 bps).
  - Hurdle Threshold $\tau$ slider (0.000 to 0.010).
  - Partition Split selector (`test`, `val`, `all`).
  - Initial Capital input (default $10,000.00) and Run Backtest action trigger.
- **Three Operational Tabs**:
  - **Tab 1 (Market Overview & Technicals)**: Metric bar (Close, Daily Return, Volatility, RSI), Candlestick chart with technical overlays, and expandable tabular feature matrix.
  - **Tab 2 (Model Forecast & Uncertainty)**: Directional Bullish/Bearish signal badge, 4 KPI cards ($P_0$, $\hat{P}_1$, 95% CI, $\sigma_{\text{MC}}$), 95% MC Dropout fan chart, and epistemic uncertainty callout.
  - **Tab 3 (Backtesting Scorecard)**: Executive KPI cards comparing Strategy against Buy & Hold, compounding equity and underwater drawdown chart, and trade audit log expander with CSV export.

### 3. Comprehensive Unit & Smoke Test Suite (`tests/test_dashboard.py`)
- Complete unit testing across all figure builders, edge cases (empty DataFrames, missing indicator columns), KPI card formatters, zero-division resiliency, module loading, and `BackendClient` local fallback behavior.

## Empirical Verification
1. **Dashboard Component & App Unit Tests**:
   - Command: `uv run pytest tests/test_dashboard.py -v`
   - Output: `11 passed in 12.01s` (100% pass rate).
2. **Streamlit App Module Compilation & Load**:
   - Command: `uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('app', 'src/dashboard/app.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); assert hasattr(mod, 'st'); print('Streamlit app module loaded successfully')"`
   - Output: `Streamlit app module loaded successfully` (exit code 0).
3. **Full Workspace Test Suite Regression Check**:
   - Command: `uv run pytest tests/ -v`
   - Output: `88 passed in 28.89s` (zero regressions across all phases).
4. **All 4 Equity Checkpoints Verified**:
   - `models/gru_GOOGL.pt`, `models/gru_AMZN.pt`, `models/gru_IBM.pt`, `models/gru_MSFT.pt` present and available in `ModelService.get_available_models()`.

## Deviations & Notes
- On Windows sandbox environment, direct `git commit` encounters a lock permission issue on `.git/index.lock`. In accordance with invocation instructions, implementation files have been staged and saved to workspace; git commits will be managed outside sandbox by the orchestrator.
