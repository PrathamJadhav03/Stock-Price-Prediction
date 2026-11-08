"""Forecasting and financial portfolio performance metrics engine."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch


def _to_1d_numpy(arr: Any, name: str = "Input array") -> np.ndarray:
    """Safely convert tensor, Series, DataFrame, list, or array to a 1D float64 NumPy array."""
    if arr is None:
        raise ValueError(f"{name} cannot be None.")

    if isinstance(arr, torch.Tensor):
        np_arr = arr.detach().cpu().numpy().astype(np.float64).flatten()
    elif isinstance(arr, (pd.Series, pd.DataFrame)):
        np_arr = arr.to_numpy(dtype=np.float64).flatten()
    else:
        np_arr = np.asarray(arr, dtype=np.float64).flatten()

    if len(np_arr) == 0:
        raise ValueError(f"{name} cannot be empty.")

    if np.isnan(np_arr).any() or np.isinf(np_arr).any():
        raise ValueError(f"{name} contains NaN or Inf values.")

    return np_arr


def _validate_pair(
    y_true: Any, y_pred: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert paired ground truth and predictions."""
    y_t = _to_1d_numpy(y_true, name="y_true")
    y_p = _to_1d_numpy(y_pred, name="y_pred")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_t)}) and y_pred ({len(y_p)}) must have same length."
        )

    return y_t, y_p


# =====================================================================
# 1. Point-Forecasting / Regression Metrics
# =====================================================================


def calculate_rmse(
    y_true: Union[np.ndarray, pd.Series, torch.Tensor, list],
    y_pred: Union[np.ndarray, pd.Series, torch.Tensor, list],
) -> float:
    """Compute Root Mean Squared Error (RMSE): sqrt(mean((y - y_hat)^2))."""
    y_t, y_p = _validate_pair(y_true, y_pred)
    diff = y_t - y_p
    return float(np.sqrt(np.mean(diff ** 2)))


def calculate_mae(
    y_true: Union[np.ndarray, pd.Series, torch.Tensor, list],
    y_pred: Union[np.ndarray, pd.Series, torch.Tensor, list],
) -> float:
    """Compute Mean Absolute Error (MAE): mean(|y - y_hat|)."""
    y_t, y_p = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_mape(
    y_true: Union[np.ndarray, pd.Series, torch.Tensor, list],
    y_pred: Union[np.ndarray, pd.Series, torch.Tensor, list],
    eps: float = 1e-8,
) -> float:
    """Compute safe Mean Absolute Percentage Error (MAPE) with epsilon stabilization.

    Financial returns frequently cluster around zero where standard MAPE explodes.
    Safe MAPE uses (|y - y_hat| / (|y| + eps)) * 100.0.
    """
    y_t, y_p = _validate_pair(y_true, y_pred)
    denominator = np.abs(y_t) + eps
    return float(np.mean(np.abs(y_t - y_p) / denominator) * 100.0)


# =====================================================================
# 2. Directional / Classification Metrics
# =====================================================================


def calculate_directional_accuracy(
    y_true: Union[np.ndarray, pd.Series, torch.Tensor, list],
    y_pred: Union[np.ndarray, pd.Series, torch.Tensor, list],
    zero_handling: str = "strictly_positive",
) -> float:
    """Compute Directional Accuracy (Sign Hit Rate %).

    Parameters
    ----------
    y_true : array-like
        Actual returns or targets.
    y_pred : array-like
        Forecasted returns or targets.
    zero_handling : str, default="strictly_positive"
        Defines how zero/neutral predictions and actuals are evaluated:
        - "strictly_positive": Matches only non-zero sign agreement
          (sign(y_pred) == sign(y_true) and sign(y_pred) != 0).
          Zero predictions are treated as non-hits.
        - "include_zero": Exact sign match (sign(y_pred) == sign(y_true)),
          where 0 matches 0.

    Returns
    -------
    float
        Directional hit rate percentage in [0.0, 100.0].
    """
    if zero_handling not in ("strictly_positive", "include_zero"):
        raise ValueError(
            f"Invalid zero_handling '{zero_handling}'. Must be 'strictly_positive' or 'include_zero'."
        )

    y_t, y_p = _validate_pair(y_true, y_pred)
    true_sign = np.sign(y_t)
    pred_sign = np.sign(y_p)

    if zero_handling == "strictly_positive":
        hits = (pred_sign == true_sign) & (pred_sign != 0)
    else:
        hits = pred_sign == true_sign

    return float(np.mean(hits) * 100.0)


# =====================================================================
# 3. Portfolio & Financial Risk / Return Metrics
# =====================================================================


def calculate_cumulative_return(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
    is_log_returns: bool = False,
) -> float:
    """Compute total cumulative return over the period.

    For arithmetic returns: prod(1 + r_t) - 1.
    For log returns: exp(sum(r_t)) - 1.
    """
    r = _to_1d_numpy(returns, name="returns")
    if is_log_returns:
        return float(np.exp(np.sum(r)) - 1.0)
    return float(np.prod(1.0 + r) - 1.0)


def calculate_annualized_return(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
    periods_per_year: int = 252,
    is_log_returns: bool = False,
) -> float:
    """Compute Compound Annual Growth Rate (CAGR) normalized to periods_per_year."""
    r = _to_1d_numpy(returns, name="returns")
    n = len(r)
    cum_ret = calculate_cumulative_return(r, is_log_returns=is_log_returns)

    total_multiplier = 1.0 + cum_ret
    if total_multiplier <= 0.0:
        return -1.0

    exponent = periods_per_year / n
    return float((total_multiplier ** exponent) - 1.0)


def calculate_annualized_volatility(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
    periods_per_year: int = 252,
) -> float:
    """Compute annualized volatility of returns (sample standard deviation * sqrt(periods))."""
    r = _to_1d_numpy(returns, name="returns")
    if len(r) <= 1:
        return 0.0

    std = float(np.std(r, ddof=1))
    if std < 1e-9:
        return 0.0

    return float(std * np.sqrt(periods_per_year))


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe Ratio with zero-volatility safeguards.

    Sharpe = (mean(r) - r_f / periods) / std(r) * sqrt(periods).
    Returns 0.0 if standard deviation is below 1e-9.
    """
    r = _to_1d_numpy(returns, name="returns")
    if len(r) <= 1:
        return 0.0

    std = float(np.std(r, ddof=1))
    if std < 1e-9:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_mean = float(np.mean(r) - rf_per_period)
    return float((excess_mean / std) * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino Ratio focusing on downside risk deviation.

    Sortino = (mean(r) - r_f / periods) / downside_std * sqrt(periods).
    If downside risk is zero: returns inf if mean excess return > 0, else 0.0.
    """
    r = _to_1d_numpy(returns, name="returns")
    if len(r) <= 1:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    downside = np.minimum(0.0, excess)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))

    excess_mean = float(np.mean(excess))
    if downside_dev < 1e-9:
        if excess_mean > 0.0:
            return float("inf")
        return 0.0

    return float((excess_mean / downside_dev) * np.sqrt(periods_per_year))


def calculate_max_drawdown(
    equity_curve: Union[np.ndarray, pd.Series, torch.Tensor, list],
) -> Tuple[float, int, int]:
    """Calculate maximum peak-to-trough drawdown from running high-water mark.

    Parameters
    ----------
    equity_curve : array-like
        Portfolio equity values over time.

    Returns
    -------
    Tuple[float, int, int]
        (max_drawdown_pct, peak_index, trough_index)
        where max_drawdown_pct is expressed as a positive percentage (e.g. 25.0 for a 25% drop).
    """
    eq = _to_1d_numpy(equity_curve, name="equity_curve")
    if len(eq) <= 1:
        return 0.0, 0, 0

    running_max = eq[0]
    running_max_idx = 0
    max_dd_pct = 0.0
    peak_idx = 0
    trough_idx = 0

    for i, val in enumerate(eq):
        if val > running_max:
            running_max = val
            running_max_idx = i

        if running_max > 0.0:
            dd = ((running_max - val) / running_max) * 100.0
            if dd > max_dd_pct:
                max_dd_pct = dd
                peak_idx = running_max_idx
                trough_idx = i

    return float(max_dd_pct), peak_idx, trough_idx


def calculate_win_rate(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
) -> float:
    """Calculate percentage of positive return periods among all active (non-zero) periods."""
    r = _to_1d_numpy(returns, name="returns")
    non_zero = r != 0.0
    total_active = int(np.sum(non_zero))
    if total_active == 0:
        return 0.0

    wins = int(np.sum(r > 0.0))
    return float((wins / total_active) * 100.0)


def calculate_profit_factor(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
) -> float:
    """Calculate Profit Factor (ratio of gross profits to gross losses).

    Returns float('inf') if there are profits but zero losses.
    Returns 0.0 if there are no profits.
    """
    r = _to_1d_numpy(returns, name="returns")
    gross_profits = float(np.sum(r[r > 0.0]))
    gross_losses = float(np.sum(np.abs(r[r < 0.0])))

    if gross_losses < 1e-12:
        if gross_profits > 1e-12:
            return float("inf")
        return 0.0

    if gross_profits < 1e-12:
        return 0.0

    return float(gross_profits / gross_losses)


# =====================================================================
# 4. Aggregators
# =====================================================================


def evaluate_forecasting_metrics(
    y_true: Union[np.ndarray, pd.Series, torch.Tensor, list],
    y_pred: Union[np.ndarray, pd.Series, torch.Tensor, list],
    eps: float = 1e-8,
    zero_handling: str = "strictly_positive",
) -> Dict[str, float]:
    """Compute all standard point-forecast regression and directional accuracy metrics."""
    return {
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred, eps=eps),
        "directional_accuracy": calculate_directional_accuracy(
            y_true, y_pred, zero_handling=zero_handling
        ),
    }


def evaluate_financial_metrics(
    returns: Union[np.ndarray, pd.Series, torch.Tensor, list],
    equity_curve: Optional[Union[np.ndarray, pd.Series, torch.Tensor, list]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    is_log_returns: bool = False,
) -> Dict[str, float]:
    """Compute comprehensive portfolio financial performance and risk indicators."""
    r = _to_1d_numpy(returns, name="returns")

    if equity_curve is not None:
        eq = _to_1d_numpy(equity_curve, name="equity_curve")
    else:
        if is_log_returns:
            cum_log = np.cumsum(r)
            eq = np.insert(np.exp(cum_log), 0, 1.0)
        else:
            cum_prod = np.cumprod(1.0 + r)
            eq = np.insert(cum_prod, 0, 1.0)

    max_dd, _, _ = calculate_max_drawdown(eq)

    return {
        "cumulative_return": calculate_cumulative_return(r, is_log_returns=is_log_returns),
        "annualized_return": calculate_annualized_return(
            r, periods_per_year=periods_per_year, is_log_returns=is_log_returns
        ),
        "annualized_volatility": calculate_annualized_volatility(
            r, periods_per_year=periods_per_year
        ),
        "sharpe_ratio": calculate_sharpe_ratio(
            r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "sortino_ratio": calculate_sortino_ratio(
            r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "max_drawdown": max_dd,
        "win_rate": calculate_win_rate(r),
        "profit_factor": calculate_profit_factor(r),
    }

