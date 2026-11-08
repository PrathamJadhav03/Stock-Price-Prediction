---
phase: 3
plan: 2
wave: 2
depends_on:
  - 1-PLAN.md
files_modified:
  - src/evaluation/backtester.py
  - src/evaluation/__init__.py
  - tests/test_backtester.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "WalkForwardBacktester simulates realistic step-by-step trading with configurable transaction costs in basis points"
    - "Turnover and transaction fees are deducted only when positions change; holding position incurs zero fee"
    - "Strategy equity and Buy-and-Hold benchmark equity compound accurately over time"
    - "BacktestResult provides an audit DataFrame with 14 step-by-step columns and a comprehensive summary scorecard"
  artifacts:
    - "src/evaluation/backtester.py exists with WalkForwardBacktester and BacktestResult"
    - "src/evaluation/__init__.py exports WalkForwardBacktester and BacktestResult"
    - "tests/test_backtester.py exists and passes via uv run pytest"
---

# Plan 3.2: Event-Driven Walk-Forward Backtester with Transaction Cost Modeling

## Objective
Implement an event-driven, walk-forward financial backtesting simulator in `src/evaluation/backtester.py` that translates predicted log returns into discrete trading positions (long/flat or long/short with threshold $\tau$), models realistic transaction costs (basis points per unit turnover), tracks step-by-step compounding equity curves, and benchmarks strategy performance against a Buy-and-Hold baseline.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- src/evaluation/metrics.py
- src/models/dataset.py
- data/processed/GOOGL/test.csv

## Tasks

<task type="auto">
  <name>Implement WalkForwardBacktester and BacktestResult data structures</name>
  <files>
    src/evaluation/backtester.py
    src/evaluation/__init__.py
  </files>
  <action>
    Create `src/evaluation/backtester.py` implementing the step-by-step walk-forward simulator and performance aggregator.

    Steps:
    1. Define `BacktestResult` dataclass:
       - `summary: dict[str, Any]`: High-level metrics (cumulative return, annualized return, volatility, Sharpe, Sortino, max drawdown, win rate, profit factor, total trades, total transaction costs, outperformance vs Buy-and-Hold).
       - `audit_df: pd.DataFrame`: Step-by-step audit record.
       - `equity_curve: pd.DataFrame`: Time-series containing `date`, `strategy_equity`, and `buy_and_hold_equity`.
       - Helper method `to_dict() -> dict`: Serializable summary representation.
    2. Implement `WalkForwardBacktester` class:
       - Constructor `__init__(fee_bps: float = 10.0, threshold: float = 0.0, strategy: str = "long_flat", initial_capital: float = 10000.0, risk_free_rate: float = 0.0, periods_per_year: int = 252)`:
         - `fee_bps`: Transaction fee in basis points ($10\text{ bps} = 0.0010 = 0.1\%$).
         - `threshold`: Minimum predicted return magnitude $|\hat{r}_t| > \tau$ required to enter a position.
         - `strategy`: Trading logic mode:
           - `"long_flat"`: If $\hat{r}_t > \tau \implies \text{signal} = 1.0$; else $0.0$.
           - `"long_short"`: If $\hat{r}_t > \tau \implies 1.0$; elif $\hat{r}_t < -\tau \implies -1.0$; else $0.0$.
       - Method `generate_signals(predicted_returns: np.ndarray) -> np.ndarray`:
         Vectorized signal generation applying threshold $\tau$ and strategy rules.
       - Method `simulate(actual_returns: np.ndarray, predicted_returns: np.ndarray, dates: Optional[pd.DatetimeIndex | list] = None, base_prices: Optional[np.ndarray] = None, target_prices: Optional[np.ndarray] = None) -> BacktestResult`:
         - Step-by-step simulation loop:
           - Initialize position $s_0 = 0.0$, initial equity = `initial_capital`.
           - At step $t \in [0, N-1]$:
             - Determine target position $s_t$ from predicted signal.
             - Calculate turnover: $\text{turnover}_t = |s_t - s_{t-1}|$ where $s_{-1} = 0.0$.
             - Calculate fee: $\text{fee}_t = \text{turnover}_t \times (\text{fee\_bps} \times 10^{-4})$.
             - Strategy daily return:
               For simple returns: $R_{\text{strat}, t} = s_t \times R_{\text{actual}, t} - \text{fee}_t$.
               (If log returns are provided, convert to simple returns $R = \exp(r) - 1$, or handle exact log compounding: $R_{\text{strat}, t} = s_t \times (\exp(r_{\text{actual}, t}) - 1) - \text{fee}_t$).
             - Update strategy equity:
               $\text{Equity}_{\text{strat}, t} = \text{Equity}_{\text{strat}, t-1} \times (1 + R_{\text{strat}, t})$.
             - Buy-and-Hold return at step $t$:
               $R_{\text{bh}, t} = \exp(r_{\text{actual}, t}) - 1$ (or raw price change $(P_t - P_{t-1}) / P_{t-1}$).
               First step may account for one-time entry fee $\text{fee\_bps} \times 10^{-4}$.
             - Update Buy-and-Hold equity:
               $\text{Equity}_{\text{bh}, t} = \text{Equity}_{\text{bh}, t-1} \times (1 + R_{\text{bh}, t})$.
             - Update High-Water Mark and Drawdown:
               $\text{HWM}_{\text{strat}, t} = \max(\text{HWM}_{\text{strat}, t-1}, \text{Equity}_{\text{strat}, t})$, $\text{DD}_{\text{strat}, t} = (\text{HWM}_t - \text{Equity}_t) / \text{HWM}_t$.
         - Construct `audit_df` with 14 columns:
           `["date", "base_price", "target_price", "actual_return", "predicted_return", "signal", "position", "turnover", "fee", "strategy_daily_return", "strategy_equity", "buy_and_hold_daily_return", "buy_and_hold_equity", "strategy_drawdown"]`.
         - Compute summary statistics via `evaluate_financial_metrics` from `src.evaluation.metrics`:
           Compute strategy metrics, Buy-and-Hold metrics, trade count, total fee paid %, and difference / alpha.
         - Return `BacktestResult`.
    3. Export `WalkForwardBacktester` and `BacktestResult` in `src/evaluation/__init__.py`.

    AVOID: Looking ahead into step $t+1$ during step $t$ position allocation or applying transaction costs on unchanged positions.
    USE: Turnover-based fee deduction: fee applied strictly proportional to $|\Delta s_t| = |s_t - s_{t-1}|$.
  </action>
  <verify>
    uv run python -c "import numpy as np; from src.evaluation import WalkForwardBacktester; r_act = np.array([0.01, -0.02, 0.03, -0.01]); r_pred = np.array([0.015, -0.01, 0.02, 0.005]); bt = WalkForwardBacktester(fee_bps=10.0); res = bt.simulate(r_act, r_pred); print('Simulated trades:', res.summary['total_trades']); print('Strategy return:', res.summary['strategy_cumulative_return'])"
  </verify>
  <done>
    `WalkForwardBacktester` simulates walk-forward execution, deducts transaction costs on turnover, tracks compounding equity, and produces a populated `BacktestResult`.
  </done>
</task>

<task type="auto">
  <name>Build comprehensive backtester unit and property tests</name>
  <files>
    tests/test_backtester.py
  </files>
  <action>
    Create `tests/test_backtester.py` validating transaction cost accounting, position transitions, flat strategies, and real-data test split execution.

    Steps:
    1. Transaction Cost & Turnover Accounting Tests:
       - Entry test: Initial step going from $0 \to 1$ incurs exactly `fee_bps * 1e-4` fee.
       - Hold test: Step maintaining $1 \to 1$ incurs exactly $0.0$ fee.
       - Exit test: Step transitioning from $1 \to 0$ incurs exactly `fee_bps * 1e-4` fee.
       - Reversal test (`long_short`): Step transitioning from $+1 \to -1$ incurs $2 \times \text{fee\_bps} \times 10^{-4}$ fee.
    2. Strategy Invariant Tests:
       - Pure Flat Strategy: When predictions are all below threshold $\tau$, positions are all 0, fees are 0, equity remains equal to initial capital throughout.
       - Always Long Strategy: When predictions are always positive, strategy matches Buy-and-Hold equity curve (offset only by initial fee).
       - Threshold Invariance: Increasing threshold $\tau$ monotonically reduces or keeps equal the total number of trades.
    3. Perfect Foresight / Oracle Benchmark Test:
       - Construct an oracle predicting $\hat{r}_t = \text{sign}(r_{\text{actual}, t}) \times 0.01$.
       - Verify that under alternating volatile returns (e.g., $+2\%, -2\%, +2\%, -2\%$), the long/flat strategy captures the upside, goes flat on down days, avoids negative compounding, and beats Buy-and-Hold.
    4. End-to-End Real Data Test:
       - Load `data/processed/GOOGL/test.csv`.
       - Extract actual log returns, mock predictions, and base/target prices.
       - Run `WalkForwardBacktester.simulate(...)`.
       - Assert `audit_df` has the same number of rows as test data, has no NaN values, and equity curves compound smoothly without negative values.
       - Assert `summary` dictionary contains all expected keys: Sharpe, Max Drawdown, Cumulative Return, Win Rate, Profit Factor, Total Trades.

    AVOID: Testing backtesting logic only on real market data where edge cases might not occur.
    USE: Deterministic hand-crafted arrays to verify exact fee deductions down to $10^{-6}$ precision.
  </action>
  <verify>
    uv run pytest tests/test_backtester.py -v
  </verify>
  <done>
    `tests/test_backtester.py` passes 100%, verifying transaction cost deduction, signal generation, equity compounding, and real-data walk-forward simulation.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `WalkForwardBacktester` executes event-driven walk-forward backtesting with configurable fees and thresholds.
- [ ] Turnover logic properly charges transaction fees only on position changes.
- [ ] `BacktestResult` contains both high-level summary KPIs and 14-column audit time series.
- [ ] `src/evaluation/__init__.py` exposes `WalkForwardBacktester` and `BacktestResult`.
- [ ] `tests/test_backtester.py` passes via `uv run pytest tests/test_backtester.py -v`.
- [ ] Full test suite passes without regressions (`uv run pytest tests/ -v`).

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.
