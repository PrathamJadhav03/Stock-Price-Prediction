---
phase: 3
plan: 1
wave: 1
depends_on: []
files_modified:
  - src/evaluation/__init__.py
  - src/evaluation/metrics.py
  - tests/test_metrics.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Regression metrics compute exact RMSE, MAE, and safe MAPE handling near-zero denominators"
    - "Directional accuracy calculates percentage of directional sign agreement between forecasts and actual returns"
    - "Financial metrics calculate cumulative return, annualized return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, win rate, and profit factor"
    - "All metrics handle edge cases gracefully: empty arrays, zero division, constant returns, and single-element inputs"
  artifacts:
    - "src/evaluation/__init__.py exists and exports public evaluation functions"
    - "src/evaluation/metrics.py exists with comprehensive forecasting and portfolio performance metrics"
    - "tests/test_metrics.py exists and passes via uv run pytest"
---

# Plan 3.1: Forecasting & Financial Performance Metrics Engine

## Objective
Implement a robust, production-grade statistical and financial metrics engine in `src/evaluation/metrics.py` that computes standard point-forecast regression errors (RMSE, MAE, MAPE), directional classification metrics (Sign Hit Rate %), and portfolio performance indicators (Cumulative Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Volatility, Win Rate, Profit Factor) with zero-division safeguards and complete test coverage.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- src/models/baseline.py

## Tasks

<task type="auto">
  <name>Implement comprehensive forecasting and financial metrics module</name>
  <files>
    src/evaluation/__init__.py
    src/evaluation/metrics.py
  </files>
  <action>
    Create `src/evaluation/metrics.py` and `src/evaluation/__init__.py` implementing point-forecasting regression metrics, directional metrics, and financial risk/return performance indicators.

    Steps:
    1. In `src/evaluation/metrics.py`, implement array conversion and sanitization helpers:
       - Accept `Union[np.ndarray, pd.Series, torch.Tensor, list]` and safely convert to 1D `np.ndarray` of `float64`.
       - Check for empty inputs, NaN/Inf handling, and length equality.
    2. Implement Point-Forecasting / Regression Metrics:
       - `calculate_rmse(y_true, y_pred) -> float`: Root Mean Squared Error $\sqrt{\frac{1}{N}\sum(y - \hat{y})^2}$.
       - `calculate_mae(y_true, y_pred) -> float`: Mean Absolute Error $\frac{1}{N}\sum|y - \hat{y}|$.
       - `calculate_mape(y_true, y_pred, eps: float = 1e-8) -> float`: Mean Absolute Percentage Error. Because daily financial log returns cluster around zero, standard MAPE explodes ($y \to 0$). Implement safe MAPE:
         $$\text{MAPE} = \frac{100}{N} \sum \frac{|y_i - \hat{y}_i|}{|y_i| + \epsilon}$$
         or symmetric MAPE (sMAPE).
    3. Implement Directional / Trading Metrics (DECISION-005):
       - `calculate_directional_accuracy(y_true, y_pred) -> float`:
         Directional Sign Hit Rate %:
         $$\text{Hit Rate \%} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{sign}(\hat{y}_i) == \text{sign}(y_i)) \times 100.0$$
         Where $\text{sign}(0) = 0$; if predicted sign is 0 (neutral), it matches only if actual return is 0, or strictly matches non-zero directions ($\text{sign}(\hat{y}) == \text{sign}(y) \land \text{sign}(\hat{y}) \neq 0$). Support parameter `zero_handling: str = "strictly_positive"` to allow both conventions.
    4. Implement Portfolio / Financial Performance Metrics:
       - `calculate_cumulative_return(returns: np.ndarray, is_log_returns: bool = False) -> float`:
         If simple arithmetic returns: $\prod (1 + r_t) - 1$. If log returns: $\exp(\sum r_t) - 1$.
       - `calculate_annualized_return(returns: np.ndarray, periods_per_year: int = 252, is_log_returns: bool = False) -> float`:
         Compound Annual Growth Rate (CAGR) normalized by length $N / 252$.
       - `calculate_annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float`:
         $\sigma_{\text{ann}} = \text{std}(r) \times \sqrt{252}$. Return 0.0 if $N \le 1$ or constant.
       - `calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float`:
         Annualized Sharpe Ratio:
         $$\text{Sharpe} = \frac{\text{mean}(r) - r_f / \text{periods\_per\_year}}{\text{std}(r) + \epsilon} \times \sqrt{\text{periods\_per\_year}}$$
         If $\text{std}(r) < 1e-9$, return 0.0.
       - `calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float`:
         Downside risk deviation $\sigma_{\text{down}} = \sqrt{\frac{1}{N}\sum (\min(0, r_t - r_f / \text{periods\_per\_year}))^2}$.
         $$\text{Sortino} = \frac{\text{mean}(r) - r_f / \text{periods\_per\_year}}{\sigma_{\text{down}} + \epsilon} \times \sqrt{\text{periods\_per\_year}}$$
         If $\sigma_{\text{down}} < 1e-9$, return 0.0 (or inf if mean excess return $> 0$).
       - `calculate_max_drawdown(equity_curve: np.ndarray) -> tuple[float, int, int]`:
         Computes peak-to-trough drop from running High-Water Mark (HWM):
         $$\text{HWM}_t = \max_{0 \le \tau \le t} \text{Equity}_\tau, \quad \text{DD}_t = \frac{\text{HWM}_t - \text{Equity}_t}{\text{HWM}_t}$$
         Returns `(max_drawdown_pct, peak_idx, trough_idx)` where `max_drawdown_pct` is positive percentage (e.g., 24.5%).
       - `calculate_win_rate(returns: np.ndarray) -> float`:
         Percentage of positive return periods: $\frac{\sum \mathbb{I}(r_t > 0)}{\sum \mathbb{I}(r_t \neq 0)} \times 100.0$.
       - `calculate_profit_factor(returns: np.ndarray) -> float`:
         Ratio of gross profits to gross losses:
         $$\text{Profit Factor} = \frac{\sum_{r_t > 0} r_t}{\sum_{r_t < 0} |r_t|}$$
         Handle edge cases: if no losses, return `float("inf")` or 100.0; if no profits, return 0.0.
    5. Implement high-level summary aggregator functions:
       - `evaluate_forecasting_metrics(y_true, y_pred) -> dict[str, float]`:
         Returns `{"rmse": ..., "mae": ..., "mape": ..., "directional_accuracy": ...}`.
       - `evaluate_financial_metrics(returns, equity_curve=None, risk_free_rate=0.0, periods_per_year=252) -> dict[str, float]`:
         Returns `{"cumulative_return": ..., "annualized_return": ..., "annualized_volatility": ..., "sharpe_ratio": ..., "sortino_ratio": ..., "max_drawdown": ..., "win_rate": ..., "profit_factor": ...}`.
    6. In `src/evaluation/__init__.py`, export all public metric functions and aggregators.

    AVOID: Unprotected division by zero when calculating Sharpe, Sortino, MAPE, or Profit Factor on edge-case return distributions.
    USE: Safe epsilon scaling and explicit guards returning 0.0 on zero-volatility inputs.
  </action>
  <verify>
    uv run python -c "import numpy as np; from src.evaluation import evaluate_forecasting_metrics, evaluate_financial_metrics; y = np.array([0.01, -0.02, 0.015, 0.03]); p = np.array([0.008, -0.015, 0.012, 0.025]); print('Forecast:', evaluate_forecasting_metrics(y, p)); print('Financial:', evaluate_financial_metrics(y))"
  </verify>
  <done>
    `src/evaluation/metrics.py` implements all regression, directional, and financial portfolio metrics cleanly with typed signatures and zero-division safeguards.
  </done>
</task>

<task type="auto">
  <name>Build exhaustive unit test suite for all metrics and numerical edge cases</name>
  <files>
    tests/test_metrics.py
  </files>
  <action>
    Create `tests/test_metrics.py` verifying mathematical precision against hand-calculated synthetic cases, invariant properties, and defensive stability under extreme inputs.

    Steps:
    1. Point-Forecast / Regression Tests:
       - Hand-calculated synthetic arrays: $y = [1.0, 2.0, 3.0]$, $\hat{y} = [1.0, 2.0, 4.0]$. Verify MAE = 1/3, RMSE = $\sqrt{1/3}$.
       - Perfect forecast test: $y == \hat{y} \implies \text{RMSE} = 0, \text{MAE} = 0, \text{MAPE} = 0, \text{Directional} = 100\%$.
       - Denominator stability: test MAPE with $y = [0.0, 0.0001, -0.0001]$; verify no NaN or Inf raised.
    2. Directional Accuracy Tests:
       - Test exact sign matching: $y = [1, -1, 1, -1]$, $\hat{y} = [2, -0.5, 0.1, -3] \implies 100\%$.
       - Inverted sign: $y = [1, -1]$, $\hat{y} = [-1, 1] \implies 0\%$.
       - Zero predictions: test behavior of flat forecast against varying market moves.
    3. Portfolio / Financial Metrics Tests:
       - Known Drawdown: `equity = [100, 120, 90, 110, 80, 130]`.
         - Peak is 120, trough is 80 $\implies \text{Drawdown} = (120 - 80) / 120 = 33.333\%$.
         - Monotonically increasing equity $\implies \text{Max Drawdown} = 0.0\%$.
       - Sharpe & Sortino tests:
         - Symmetrical positive returns $\implies \text{Sharpe} > 0$.
         - Constant returns (zero variance) $\implies \text{Sharpe} = 0.0$, does not crash or raise `ZeroDivisionError`.
         - All positive returns $\implies \text{Sortino}$ handles zero downside risk cleanly without exception.
       - Win Rate & Profit Factor tests:
         - All wins: `returns = [0.01, 0.02, 0.03]` $\implies \text{Win Rate} = 100\%$, $\text{Profit Factor} = \text{inf}$ or capped float.
         - All losses: `returns = [-0.01, -0.02]` $\implies \text{Win Rate} = 0\%$, $\text{Profit Factor} = 0.0$.
         - Known mixed trades: gains = $0.06$, losses = $0.02 \implies \text{Profit Factor} = 3.0$.
    4. Type and Input Invariance Tests:
       - Pass NumPy arrays, PyTorch tensors, pandas Series, and raw Python lists; assert identical numerical results across types.
       - Mismatched length assertions: assert `ValueError` raised when `len(y_true) != len(y_pred)`.
       - Empty input assertions: assert graceful handling (empty dict or `ValueError`).

    AVOID: Relying on third-party black-box metrics; test against clear analytical formulas.
    USE: `pytest` parameterized tests with exact floating-point tolerances (`pytest.approx`).
  </action>
  <verify>
    uv run pytest tests/test_metrics.py -v
  </verify>
  <done>
    `tests/test_metrics.py` passes 100% of test cases across regression, directional, and financial metric calculations and edge cases.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `src/evaluation/metrics.py` exists with all specified point-forecast and financial portfolio metrics.
- [ ] Safe zero-division handling prevents NaN/Inf across all metrics.
- [ ] `src/evaluation/__init__.py` exposes `evaluate_forecasting_metrics` and `evaluate_financial_metrics`.
- [ ] `tests/test_metrics.py` passes via `uv run pytest tests/test_metrics.py -v`.
- [ ] No regressions in existing test suite (`uv run pytest tests/ -v`).

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.
