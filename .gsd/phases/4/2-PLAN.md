---
phase: 4
plan: 2
wave: 2
depends_on:
  - 1-PLAN.md
files_modified:
  - src/dashboard/__init__.py
  - src/dashboard/components.py
  - src/dashboard/app.py
  - tests/test_dashboard.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Plotly component builders generate valid, non-empty interactive candlestick charts, forecast fan charts, and equity curves"
    - "Candlestick chart renders dual subplots with OHLC bars, Bollinger Bands, EMA 10/20/50 overlays, and color-coded volume bars"
    - "Forecast fan chart renders historical price trajectory, single-step forecast point, and shaded 95% Monte Carlo Dropout confidence interval envelope"
    - "Equity curve chart displays compounding Strategy vs Buy-and-Hold portfolio equity alongside an underwater drawdown subplot"
    - "Streamlit dashboard exposes responsive sidebar controls for all 4 equities, lookback windows, MC samples, strategy types, and transaction fees"
    - "Dashboard supports dual-mode connection: querying live FastAPI REST endpoints with seamless in-process ModelService fallback"
  artifacts:
    - "src/dashboard/__init__.py exists and exports visualization and KPI card builders"
    - "src/dashboard/components.py exists with create_candlestick_chart, create_forecast_fan_chart, create_equity_curve_chart, and format_kpi_cards"
    - "src/dashboard/app.py exists as the full Streamlit web application entrypoint"
    - "tests/test_dashboard.py exists and passes 100% via uv run pytest tests/test_dashboard.py"
---

# Plan 4.2: Plotly Analytics Components & Interactive Streamlit Dashboard

## Objective
Deliver an interactive, enterprise-grade Streamlit financial analytics dashboard in `src/dashboard/` that ports and elevates proof-of-concept notebook visualizations into reusable Plotly components. The dashboard provides three rich operational views: (1) Market Overview with candlestick charts, Bollinger Bands, EMAs, and volume subplots; (2) Model Forecast & Uncertainty Analysis with 95% Monte Carlo Dropout fan charts and direction badges; and (3) Walk-Forward Backtesting Scorecards with compounding equity trajectories, underwater drawdowns, and executive KPI comparison cards.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- src/api/schemas.py
- src/api/service.py
- src/features/technical.py
- src/evaluation/backtester.py
- scripts/backtest.py

## Tasks

<task type="auto">
  <name>Implement Modular Plotly Visualization Engine and KPI Card Builders</name>
  <files>
    src/dashboard/__init__.py
    src/dashboard/components.py
    tests/test_dashboard.py
  </files>
  <action>
    Create `src/dashboard/components.py`, `src/dashboard/__init__.py`, and `tests/test_dashboard.py` implementing modular Plotly chart builders and metric card formatters with comprehensive unit tests.

    Steps:
    1. In `src/dashboard/components.py`, implement visualization functions:
       - `create_candlestick_chart(df: pd.DataFrame, title: str = "Market Overview & Technical Overlays") -> go.Figure`:
         - Use `plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])`.
         - Row 1: Candlestick trace (`go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC')`).
         - Technical overlays in Row 1:
           - Bollinger Bands: upper (`bb_upper`), middle (`bb_middle`), lower (`bb_lower`). If present, render with subtle lines and translucent fill (`fill='tonexty'`, `rgba(100, 149, 237, 0.1)`) between upper and lower.
           - Moving Averages: EMA 10 (`ema_10`, gold #f39c12), EMA 20 (`ema_20`, cyan #00f5d4), EMA 50 (`ema_50`, magenta #d63384) if present in DataFrame.
         - Row 2: Volume bar chart (`go.Bar(x=df.index, y=df['Volume'], name='Volume')`).
           - Color volume bars green (`rgba(46, 204, 113, 0.7)`) when `df['Close'] >= df['Open']` and red (`rgba(231, 76, 60, 0.7)`) when `df['Close'] < df['Open']`.
         - Layout: template="plotly_dark", `xaxis_rangeslider_visible=False`, title=title, margin=dict(l=40, r=40, t=50, b=30), hovermode="x unified".
       - `create_forecast_fan_chart(historical_dates: Any, historical_prices: Any, forecast_date: Any, current_price: float, pred_price: float, lower_95_price: float, upper_95_price: float, symbol: str = "GOOGL") -> go.Figure`:
         - Historical line: `go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Close', line=dict(color='#3498db', width=2))`.
         - Upper confidence fan: `go.Scatter(x=[historical_dates[-1], forecast_date], y=[current_price, upper_95_price], mode='lines', line=dict(width=0), showlegend=False)`.
         - Lower confidence fan: `go.Scatter(x=[historical_dates[-1], forecast_date], y=[current_price, lower_95_price], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 245, 212, 0.25)', name='95% MC Dropout Fan')`.
         - Expected trajectory: `go.Scatter(x=[historical_dates[-1], forecast_date], y=[current_price, pred_price], mode='lines+markers', line=dict(color='#00f5d4', width=2, dash='dash'), name='Predicted Price')`.
         - Point forecast annotation: text with predicted price and expected delta percentage.
         - Layout: template="plotly_dark", title=f"{symbol.upper()} 1-Step Price Forecast & 95% Uncertainty Fan", hovermode="x unified".
       - `create_equity_curve_chart(equity_df: pd.DataFrame, symbol: str = "GOOGL") -> go.Figure`:
         - Dual subplot: Row 1 (height 70%) = Compounding Portfolio Equity; Row 2 (height 30%) = Underwater Drawdown %.
         - Row 1 traces:
           - Strategy Equity line (`go.Scatter(x=dates, y=equity_df['strategy_equity'], name='Strategy', line=dict(color='#2ecc71', width=2.5))`).
           - Buy & Hold Equity line (`go.Scatter(x=dates, y=equity_df['buy_and_hold_equity'], name='Buy & Hold', line=dict(color='#9b59b6', width=2, dash='dot'))`).
         - Row 2 traces:
           - Calculate drawdowns if not in columns: $\text{DD}_t = (\text{Equity}_t - \text{HWM}_t) / \text{HWM}_t \times 100\%$.
           - Strategy Drawdown filled area (`go.Scatter(x=dates, y=strat_dd, fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.3)', line=dict(color='#e74c3c', width=1.5), name='Strategy Drawdown')`).
           - Buy & Hold Drawdown line (`go.Scatter(x=dates, y=bh_dd, line=dict(color='#7f8c8d', width=1, dash='dot'), name='B&H Drawdown')`).
         - Layout: template="plotly_dark", title=f"{symbol.upper()} Walk-Forward Compounding Equity & Underwater Drawdown", yaxis1=dict(title="Portfolio Equity ($)"), yaxis2=dict(title="Drawdown (%)").
       - `format_kpi_cards(summary: Dict[str, Any]) -> List[Dict[str, Any]]`:
         - Extracts and formats metrics: Cumulative Return, Annualized Return (CAGR), Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, Profit Factor, Directional Accuracy, and Total Fees.
         - Returns structured list: `[{"label": "Cumulative Return", "strategy": "+35.2%", "benchmark": "+18.4%", "delta": "+16.8%", "is_positive": True}, ...]`.
    2. In `src/dashboard/__init__.py`, export the four functions.
    3. In `tests/test_dashboard.py`, create unit tests:
       - `test_create_candlestick_chart`: Passes synthetic OHLCV DataFrame with technical indicators; verifies figure is instance of `plotly.graph_objects.Figure` and contains candlestick and volume traces.
       - `test_create_candlestick_chart_missing_indicators`: Passes plain OHLCV DataFrame without BB/EMA columns; confirms graceful rendering without exceptions.
       - `test_create_forecast_fan_chart`: Passes historical price series, forecast date, and price bounds; checks valid trace count, confidence envelope fill, and marker text.
       - `test_create_equity_curve_chart`: Passes synthetic equity DataFrame; checks 2 subplot rows and verified trace labels.
       - `test_format_kpi_cards`: Tests summary dictionary formatting, zero division handling, and missing metric defaults.

    AVOID: Hardcoding static colors without theme support or crashing when optional indicator columns are missing.
    USE: Resilient column checks (`if "bb_upper" in df.columns`) and Plotly dark template.
  </action>
  <verify>
    uv run pytest tests/test_dashboard.py -v
  </verify>
  <done>
    `tests/test_dashboard.py` passes 100% verifying that all Plotly figure builders and KPI formatters produce valid visual artifacts.
  </done>
</task>

<task type="auto">
  <name>Construct Streamlit Dashboard with Sidebar Controls, Multi-Tab Analytics, and Live Service Integration</name>
  <files>
    src/dashboard/app.py
  </files>
  <action>
    Create `src/dashboard/app.py` implementing the Streamlit web application with sidebar controls, multi-tab analytics, and dual-mode backend integration (FastAPI REST or in-process ModelService).

    Steps:
    1. In `src/dashboard/app.py`:
       - Configure page layout:
         ```python
         st.set_page_config(
             page_title="Stock Price Forecasting & Quantitative Backtesting",
             page_icon="📈",
             layout="wide",
             initial_sidebar_state="expanded",
         )
         ```
       - Implement backend communication helper:
         - Provide `get_service()` function cached via `@st.cache_resource`.
         - Check if FastAPI is running at `http://localhost:8000/health`.
         - If reachable, make HTTP requests via `requests.post` to `/predict` and `/backtest`.
         - If unreachable, instantiate local `ModelService()` in-process. This ensures the dashboard operates seamlessly whether or not `uvicorn` is active.
       - Sidebar Controls:
         - Header: "Quantitative Controls" with system status badge (Backend mode, Compute device).
         - Stock Ticker selectbox: `["GOOGL", "AMZN", "IBM", "MSFT"]` (default "GOOGL").
         - Lookback Window slider: `30` to `365` trading days (default 120).
         - Monte Carlo Dropout Samples slider: `10` to `200` (default 50).
         - Trading Strategy selectbox: `["long_flat", "long_short"]` (default "long_flat").
         - Transaction Cost (bps) slider: `0.0` to `50.0` bps (default 10.0).
         - Hurdle Threshold $\tau$ slider: `0.000` to `0.010` (step 0.001, default 0.0).
         - Partition Split selectbox: `["test", "val", "all"]` (default "test").
         - Initial Capital number input: default `10000.0`.
         - "Run Backtest" action button.
       - Main Application Header:
         - Title: "Enterprise Stock Forecasting & Walk-Forward Backtesting Engine"
         - Subtitle: "Stationary Log-Return Modeling • PyTorch GRU • Monte Carlo Dropout Uncertainty • Execution Cost Simulation"
       - Implement 3 Main Tabs:
         - **Tab 1: Market Overview & Technical Analysis**:
           - Load ticker historical data from `data/processed/{symbol}/`.
           - Top Metric Bar: 4 columns displaying Latest Close Price, Daily Return %, 20-Day Volatility %, and 14-Day RSI.
           - Chart: Interactive Plotly Candlestick chart with Bollinger Bands, EMAs (10, 20, 50), and volume subplots via `create_candlestick_chart`.
           - Expander: "View Technical Indicators & Scaled Feature Matrix" displaying raw tabular data.
         - **Tab 2: Model Forecast & Uncertainty Analysis**:
           - Run forecast via service (`predict(PredictRequest(symbol=symbol, mc_samples=mc_samples, return_ci=True))`).
           - Direction Badge: Large HTML card with bold styling:
             - If UP: Green banner `↗ SIGNAL: UP (Bullish) | Expected Return: +X.XX%`
             - If DOWN: Red banner `↘ SIGNAL: DOWN (Bearish) | Expected Return: -X.XX%`
           - 4 KPI columns:
             1. Current Observed Price ($P_0$)
             2. Predicted Forward Price ($\hat{P}_1$)
             3. 95% Confidence Interval ($[P_{\text{lower}}, P_{\text{upper}}]$)
             4. Predictive Uncertainty ($\sigma_{\text{MC}}$)
           - Chart: Interactive Plotly 95% Uncertainty Fan Chart via `create_forecast_fan_chart`.
           - Callout Box: Epistemic uncertainty explanation detailing how Monte Carlo Dropout sampling quantifies model risk without assuming Gaussian return distributions.
         - **Tab 3: Walk-Forward Backtesting Scorecard**:
           - Run walk-forward backtest via service with selected strategy, fee, threshold, split, and capital.
           - Top KPI Cards: Formatted using `format_kpi_cards` into a responsive grid comparing Strategy vs Buy & Hold for Cumulative Return, CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, and Total Fees.
           - Chart: Interactive Plotly Compounding Equity Curve & Underwater Drawdown Chart via `create_equity_curve_chart`.
           - Expander: "Detailed Step-by-Step Trade & Position Audit Log" with `st.dataframe` and a CSV download button.
    2. Add unit/smoke verification test to confirm script execution and clean module loading without syntax or import errors.

    AVOID: Requiring an external FastAPI server to be manually started before Streamlit can launch.
    USE: Dual-backend fallback: query FastAPI HTTP if running, fall back directly to in-process `ModelService` if offline.
  </action>
  <verify>
    uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('app', 'src/dashboard/app.py'); mod = importlib.util.module_from_spec(spec); assert hasattr(mod, 'st'), 'Streamlit module missing'; print('Streamlit app module loaded successfully')" && uv run pytest tests/ -v
  </verify>
  <done>
    `src/dashboard/app.py` compiles and loads cleanly, connects to `ModelService`, and all unit and integration tests pass across the entire repository.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `src/dashboard/components.py` generates valid Plotly candlestick, fan, and equity curve figures
- [ ] `src/dashboard/app.py` implements all 3 tabs with responsive sidebar controls
- [ ] Dashboard supports all 4 equity tickers (`GOOGL`, `AMZN`, `IBM`, `MSFT`)
- [ ] Monte Carlo Dropout fan chart visually bounds the predicted price within 95% CI
- [ ] Backtest scorecard compares Strategy against Buy & Hold factoring transaction fees
- [ ] `tests/test_dashboard.py` passes 100% via `uv run pytest`

## Success Criteria
- [ ] All tasks verified passing
- [ ] Must-haves confirmed
- [ ] Full test suite passes via `uv run pytest` without regressions

