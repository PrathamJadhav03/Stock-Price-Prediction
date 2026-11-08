---
phase: 3
plan: 2
wave: 2
status: completed
files_modified:
  - src/evaluation/backtester.py
  - src/evaluation/__init__.py
  - tests/test_backtester.py
---

# Phase 3 Plan 2 Summary: Event-Driven Walk-Forward Backtester with Transaction Cost Modeling

## Overview

Plan 3.2 delivered the quantitative backtesting engine in `src/evaluation/backtester.py`. The engine translates model-forecasted returns (point predictions) into discrete trading signals, models realistic transaction frictions (turnover-based fees in basis points), tracks step-by-step compounding portfolio equity curves, and benchmarks strategy performance against a Buy-and-Hold asset baseline.

## Implementation Details

### 1. `WalkForwardBacktester` (`src/evaluation/backtester.py`)
- **Configurable Execution Parameters:**
  - `fee_bps`: Basis points charged per unit of portfolio turnover ($10\text{ bps} = 0.0010 = 0.1\%$).
  - `threshold`: Prediction hurdle threshold $\tau$ required to take a position ($|\hat{r}_t| > \tau$).
  - `strategy`: Trading logic modes:
    - `"long_flat"`: If $\hat{r}_t > \tau \implies s_t = 1.0$, else $s_t = 0.0$.
    - `"long_short"`: If $\hat{r}_t > \tau \implies s_t = 1.0$; if $\hat{r}_t < -\tau \implies s_t = -1.0$; else $s_t = 0.0$.
  - `initial_capital`: Starting portfolio capital in currency units (default: $\$10,000.00$).
  - `risk_free_rate`: Risk-free rate for Sharpe and Sortino ratios.
  - `periods_per_year`: Annualization factor (default: $252$).
  - `include_benchmark_fee`: Optional toggle for whether Buy-and-Hold absorbs initial entry fee.
- **Signal Generation (`generate_signals`):**
  - Vectorized generation applying threshold $\tau$ and strategy rules with strictly zero lookahead bias.
- **Walk-Forward Simulation (`simulate`):**
  - Simulates step-by-step position allocation:
    - Calculates turnover: $\text{turnover}_t = |s_t - s_{t-1}|$ where $s_{-1} = 0.0$.
    - Deducts transaction fee strictly on turnover: $\text{fee}_t = \text{turnover}_t \times (\text{fee\_bps} \times 10^{-4})$.
    - Computes strategy daily return: $R_{\text{strat}, t} = s_t \times R_{\text{act}, t} - \text{fee}_t$.
    - Compounds portfolio equity: $\text{Equity}_{\text{strat}, t} = \text{Equity}_{\text{strat}, t-1} \times (1 + R_{\text{strat}, t})$.
    - Computes Buy-and-Hold daily return: $R_{\text{bh}, t} = R_{\text{act}, t} - \text{bnh\_fee}$.
    - Compounds Buy-and-Hold equity: $\text{Equity}_{\text{bh}, t} = \text{Equity}_{\text{bh}, t-1} \times (1 + R_{\text{bh}, t})$.
    - Tracks running high-water mark ($\text{HWM}_t$) and percentage drawdown ($\text{DD}_t = (\text{HWM}_t - \text{Equity}_t) / \text{HWM}_t$).

### 2. `BacktestResult` Dataclass (`src/evaluation/backtester.py`)
- **`summary: Dict[str, Any]`:** Comprehensive executive scorecard containing 37 KPIs:
  - Strategy metrics: cumulative return, annualized return (CAGR), annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor.
  - Buy-and-Hold benchmark metrics: cumulative return, CAGR, volatility, Sharpe, Sortino, max drawdown, win rate, profit factor.
  - Alpha / Outperformance: outperformance, alpha cumulative return, alpha annualized return, alpha Sharpe ratio.
  - Trading & Execution stats: total trades, total turnover, total transaction fees %, total transaction costs in dollars, directional accuracy %.
- **`audit_df: pd.DataFrame`:** 14-column step-by-step audit record:
  1. `date`
  2. `base_price`
  3. `target_price`
  4. `actual_return`
  5. `predicted_return`
  6. `signal`
  7. `position`
  8. `turnover`
  9. `fee`
  10. `strategy_daily_return`
  11. `strategy_equity`
  12. `buy_and_hold_daily_return`
  13. `buy_and_hold_equity`
  14. `strategy_drawdown`
- **`equity_curve: pd.DataFrame`:** Time series containing `["date", "strategy_equity", "buy_and_hold_equity"]`.
- **`to_dict() -> Dict[str, Any]`:** Serialization helper guaranteeing clean JSON-compliant output.

### 3. Package Exports (`src/evaluation/__init__.py`)
- Exported `WalkForwardBacktester` and `BacktestResult` alongside all forecasting and financial metrics.

## Empirical Verification

1. **Deterministic Verification Snippet:**
   - Ran 4-step walk-forward simulation:
     - `Simulated trades: 3`
     - `Strategy return: 0.027406887340660546` (+2.74%)
     - Exit code: 0

2. **Unit, Property, and Integration Test Suite (`tests/test_backtester.py`):**
   - `test_backtester_initialization_defaults`: Verified default parameters.
   - `test_backtester_invalid_parameters`: Verified `ValueError` on invalid strategy, negative fees, negative threshold, non-positive capital.
   - `test_simulate_input_validation`: Verified length matching, empty inputs, dimension checks.
   - `test_turnover_and_transaction_costs_long_flat`: Verified exact fee accounting ($0 \to 1$ entry = fee, $1 \to 1$ hold = 0 fee, $1 \to 0$ exit = fee, $0 \to 0$ flat = 0 fee).
   - `test_turnover_and_reversal_long_short`: Verified full reversal $+1 \to -1$ incurs $2 \times \text{fee}$ turnover charge.
   - `test_pure_flat_strategy`: Verified zero turnover, zero fees, and exact capital preservation when predictions are below threshold.
   - `test_always_long_strategy_matches_buy_and_hold`: Verified zero fee gives exact identity with Buy-and-Hold, and positive fee offsets only by initial entry fee.
   - `test_threshold_invariance_monotonic_trade_count`: Verified monotonically non-increasing trade count as threshold $\tau$ increases.
   - `test_oracle_benchmark_avoids_negative_compounding`: Verified oracle strategy captures upside, stays flat on down days, avoids negative compounding, and beats Buy-and-Hold.
   - `test_end_to_end_real_data_googl_test_split`: Loaded `data/processed/GOOGL/test.csv`, verified 14-column audit DataFrame with zero NaNs, positive equity compounding, bounded drawdowns, and complete summary KPIs.
   - Result: **10 passed in 3.72s**.

3. **Full Project Regression Suite:**
   - Ran `uv run pytest tests/ -v`.
   - Result: **64 passed in 9.09s** with 0 warnings/failures.

## Deviations & Notes

- As noted in the execution constraints on Windows sandbox, `.git/index.lock` has restricted permission from inside the subagent sandbox. The modified and untracked files are staged/ready for git commits outside the sandbox by the parent orchestrator.

## Next Steps

Wave 3 (Plan 3.3):
- Build backtesting CLI runner in `scripts/backtest.py`.
- Format executive comparison scorecard in console (Strategy vs Buy & Hold).
- Serialize structured reports to `reports/backtest/backtest_{symbol}.json` and `reports/backtest/equity_curve_{symbol}.csv`.
- Create integration tests in `tests/test_backtest_cli.py`.
