"""Comprehensive unit test suite for forecasting and financial performance metrics."""

import math
import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_return,
    calculate_directional_accuracy,
    calculate_mae,
    calculate_mape,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_rmse,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    evaluate_financial_metrics,
    evaluate_forecasting_metrics,
)


# =====================================================================
# 1. Point-Forecast / Regression Metrics Tests
# =====================================================================


def test_regression_metrics_hand_calculated():
    """Verify hand-calculated regression errors on known synthetic arrays."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    expected_mae = 1.0 / 3.0
    expected_rmse = math.sqrt(1.0 / 3.0)

    assert calculate_mae(y_true, y_pred) == pytest.approx(expected_mae, abs=1e-7)
    assert calculate_rmse(y_true, y_pred) == pytest.approx(expected_rmse, abs=1e-7)


def test_regression_metrics_perfect_forecast():
    """Verify perfect predictions yield exactly zero error and 100% directional accuracy."""
    y = np.array([0.05, -0.02, 0.01, -0.04])
    metrics = evaluate_forecasting_metrics(y, y)

    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mape"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["directional_accuracy"] == pytest.approx(100.0, abs=1e-9)


def test_mape_denominator_stability():
    """Verify safe MAPE prevents division-by-zero explosions on zero and near-zero values."""
    y_true = np.array([0.0, 0.0001, -0.0001])
    y_pred = np.array([0.0001, 0.0, -0.0002])

    mape_val = calculate_mape(y_true, y_pred)
    assert not math.isnan(mape_val)
    assert not math.isinf(mape_val)
    assert mape_val >= 0.0


# =====================================================================
# 2. Directional Accuracy Tests
# =====================================================================


def test_directional_accuracy_exact_sign_matching():
    """Verify 100% directional accuracy when signs match regardless of magnitude."""
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([2.0, -0.5, 0.1, -3.0])

    acc = calculate_directional_accuracy(y_true, y_pred)
    assert acc == pytest.approx(100.0)


def test_directional_accuracy_inverted_signs():
    """Verify 0% directional accuracy when signs are completely inverted."""
    y_true = np.array([1.0, -1.0])
    y_pred = np.array([-1.0, 1.0])

    acc = calculate_directional_accuracy(y_true, y_pred)
    assert acc == pytest.approx(0.0)


def test_directional_accuracy_flat_predictions():
    """Verify flat (zero) predictions are treated as non-hits under strictly_positive mode."""
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.zeros_like(y_true)

    # In strictly_positive mode, predicting 0 gives 0% hit rate against market moves
    acc_strict = calculate_directional_accuracy(y_true, y_pred, zero_handling="strictly_positive")
    assert acc_strict == pytest.approx(0.0)

    # In include_zero mode, 0 only matches when actual return is 0
    y_true_with_zero = np.array([0.01, 0.0, 0.03, -0.01])
    acc_include = calculate_directional_accuracy(
        y_true_with_zero, y_pred, zero_handling="include_zero"
    )
    assert acc_include == pytest.approx(25.0)  # 1 out of 4 matches exactly


def test_directional_accuracy_invalid_mode():
    """Verify invalid zero_handling raises ValueError."""
    with pytest.raises(ValueError, match="Invalid zero_handling"):
        calculate_directional_accuracy([1.0], [1.0], zero_handling="invalid_mode")


# =====================================================================
# 3. Portfolio & Financial Risk / Return Metrics Tests
# =====================================================================


def test_max_drawdown_known_sequence():
    """Verify peak-to-trough drawdown calculation against a known equity path."""
    equity = np.array([100.0, 120.0, 90.0, 110.0, 80.0, 130.0])
    # Peak is 120 at index 1, lowest trough is 80 at index 4 -> (120 - 80) / 120 = 33.333%
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

    expected_dd = (120.0 - 80.0) / 120.0 * 100.0
    assert max_dd == pytest.approx(expected_dd, rel=1e-5)
    assert peak_idx == 1
    assert trough_idx == 4


def test_max_drawdown_monotonic_growth():
    """Verify monotonically increasing equity has 0.0% max drawdown."""
    equity = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

    assert max_dd == pytest.approx(0.0)
    assert peak_idx == 0
    assert trough_idx == 0


def test_sharpe_and_sortino_symmetrical_returns():
    """Verify positive excess returns yield positive Sharpe and Sortino ratios."""
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.005])

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)

    assert sharpe > 0.0
    assert sortino > 0.0
    # Downside deviation is smaller than total volatility, so Sortino > Sharpe
    assert sortino > sharpe


def test_sharpe_zero_variance():
    """Verify constant returns (zero variance) safely return 0.0 Sharpe without ZeroDivisionError."""
    returns = np.array([0.01, 0.01, 0.01, 0.01])

    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0


def test_sortino_all_positive_returns():
    """Verify all positive returns yield infinite Sortino ratio without crashing."""
    returns = np.array([0.01, 0.02, 0.015, 0.03])

    sortino = calculate_sortino_ratio(returns)
    assert math.isinf(sortino)
    assert sortino > 0.0


def test_win_rate_and_profit_factor_scenarios():
    """Verify win rate and profit factor across all-win, all-loss, and mixed trades."""
    # All wins
    all_wins = np.array([0.01, 0.02, 0.03])
    assert calculate_win_rate(all_wins) == pytest.approx(100.0)
    assert math.isinf(calculate_profit_factor(all_wins))

    # All losses
    all_losses = np.array([-0.01, -0.02])
    assert calculate_win_rate(all_losses) == pytest.approx(0.0)
    assert calculate_profit_factor(all_losses) == pytest.approx(0.0)

    # Known mixed trades: gains = 0.06, losses = -0.02 -> profit factor = 3.0
    mixed = np.array([0.02, -0.01, 0.04, -0.01, 0.0])  # Note 0.0 is neutral
    assert calculate_win_rate(mixed) == pytest.approx(50.0)  # 2 wins out of 4 active trades
    assert calculate_profit_factor(mixed) == pytest.approx(3.0)


def test_cumulative_and_annualized_returns():
    """Verify arithmetic and log cumulative return and CAGR calculations."""
    simple_returns = np.array([0.10, -0.05])
    # Total arithmetic return: (1.10 * 0.95) - 1 = 0.045 (4.5%)
    cum_ret = calculate_cumulative_return(simple_returns, is_log_returns=False)
    assert cum_ret == pytest.approx(0.045)

    # Annualized CAGR: (1 + 0.045) ** (252 / 2) - 1
    cagr = calculate_annualized_return(simple_returns, periods_per_year=252, is_log_returns=False)
    expected_cagr = (1.045 ** (252.0 / 2.0)) - 1.0
    assert cagr == pytest.approx(expected_cagr)

    # Log returns: sum of log returns = 0.02 + (-0.01) = 0.01 -> exp(0.01) - 1
    log_returns = np.array([0.02, -0.01])
    cum_log_ret = calculate_cumulative_return(log_returns, is_log_returns=True)
    assert cum_log_ret == pytest.approx(math.exp(0.01) - 1.0)


def test_evaluate_financial_metrics_aggregator():
    """Verify aggregator returns all required keys with proper values."""
    returns = np.array([0.01, -0.02, 0.015, 0.03])
    metrics = evaluate_financial_metrics(returns)

    expected_keys = {
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
    }
    assert expected_keys.issubset(metrics.keys())
    for k, v in metrics.items():
        assert not math.isnan(v), f"Metric '{k}' evaluated to NaN"


# =====================================================================
# 4. Type & Input Invariance Tests
# =====================================================================


@pytest.mark.parametrize(
    "cast_fn",
    [
        lambda x: np.array(x),
        lambda x: pd.Series(x),
        lambda x: torch.tensor(x, dtype=torch.float32),
        lambda x: list(x),
    ],
)
def test_input_type_invariance(cast_fn):
    """Verify identical numerical outputs across numpy, pandas, torch, and list inputs."""
    raw_true = [0.01, -0.02, 0.015, 0.03]
    raw_pred = [0.008, -0.015, 0.012, 0.025]

    y_true = cast_fn(raw_true)
    y_pred = cast_fn(raw_pred)

    mae = calculate_mae(y_true, y_pred)
    assert mae == pytest.approx(0.00375)

    cum_ret = calculate_cumulative_return(y_true)
    assert cum_ret == pytest.approx(np.prod(1.0 + np.array(raw_true)) - 1.0)


def test_mismatched_length_raises_value_error():
    """Verify ValueError is raised when input array lengths do not match."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_rmse(y_true, y_pred)


def test_empty_input_raises_value_error():
    """Verify ValueError is raised on empty input arrays."""
    empty = np.array([])
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_rmse(empty, empty)

    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_cumulative_return(empty)


def test_nan_or_inf_input_raises_value_error():
    """Verify ValueError is raised when inputs contain NaN or Inf values."""
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        calculate_rmse(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        calculate_cumulative_return(np.array([0.01, np.inf]))

