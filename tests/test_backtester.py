"""Unit, property, and integration tests for WalkForwardBacktester."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtester import BacktestResult, WalkForwardBacktester


# =====================================================================
# 1. Constructor and Validation Tests
# =====================================================================


def test_backtester_initialization_defaults():
    """Verify default parameters of WalkForwardBacktester."""
    bt = WalkForwardBacktester()
    assert bt.fee_bps == 10.0
    assert bt.fee_rate == 0.0010
    assert bt.threshold == 0.0
    assert bt.strategy == "long_flat"
    assert bt.initial_capital == 10000.0
    assert bt.risk_free_rate == 0.0
    assert bt.periods_per_year == 252


def test_backtester_invalid_parameters():
    """Verify constructor raises ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        WalkForwardBacktester(strategy="invalid_mode")

    with pytest.raises(ValueError, match="fee_bps must be non-negative"):
        WalkForwardBacktester(fee_bps=-5.0)

    with pytest.raises(ValueError, match="threshold must be non-negative"):
        WalkForwardBacktester(threshold=-0.01)

    with pytest.raises(ValueError, match="initial_capital must be positive"):
        WalkForwardBacktester(initial_capital=0.0)

    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        WalkForwardBacktester(periods_per_year=0)


def test_simulate_input_validation():
    """Verify input dimension and shape checks in simulate()."""
    bt = WalkForwardBacktester()

    # Mismatched lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        bt.simulate(
            actual_returns=np.array([0.01, 0.02]),
            predicted_returns=np.array([0.01]),
        )

    # Empty inputs
    with pytest.raises(ValueError, match="cannot be empty"):
        bt.simulate(actual_returns=[], predicted_returns=[])

    # Dates mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        bt.simulate(
            actual_returns=np.array([0.01, 0.02]),
            predicted_returns=np.array([0.01, 0.02]),
            dates=["2020-01-01"],
        )

    # Base prices mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        bt.simulate(
            actual_returns=np.array([0.01, 0.02]),
            predicted_returns=np.array([0.01, 0.02]),
            base_prices=np.array([100.0]),
        )


# =====================================================================
# 2. Transaction Cost & Turnover Accounting Tests
# =====================================================================


def test_turnover_and_transaction_costs_long_flat():
    """Verify exact fee deductions across entry, hold, and exit transitions."""
    fee_bps = 20.0
    fee_rate = 20.0 * 1e-4  # 0.0020 = 0.2%
    bt = WalkForwardBacktester(fee_bps=fee_bps, strategy="long_flat", threshold=0.0)

    # Step 0: Entry (0 -> 1)
    # Step 1: Hold  (1 -> 1)
    # Step 2: Exit  (1 -> 0)
    # Step 3: Flat  (0 -> 0)
    actual_r = np.array([0.01, 0.01, 0.01, 0.01])
    predicted_r = np.array([0.05, 0.05, -0.05, -0.05])

    res = bt.simulate(actual_r, predicted_r, is_log_returns=False)
    audit = res.audit_df

    # Entry test: Initial step going from 0 -> 1 incurs exactly fee_bps * 1e-4 fee
    assert audit.loc[0, "position"] == 1.0
    assert audit.loc[0, "turnover"] == 1.0
    assert np.isclose(audit.loc[0, "fee"], fee_rate, atol=1e-8)

    # Hold test: Step maintaining 1 -> 1 incurs exactly 0.0 fee
    assert audit.loc[1, "position"] == 1.0
    assert audit.loc[1, "turnover"] == 0.0
    assert np.isclose(audit.loc[1, "fee"], 0.0, atol=1e-8)

    # Exit test: Step transitioning from 1 -> 0 incurs exactly fee_bps * 1e-4 fee
    assert audit.loc[2, "position"] == 0.0
    assert audit.loc[2, "turnover"] == 1.0
    assert np.isclose(audit.loc[2, "fee"], fee_rate, atol=1e-8)

    # Stay flat test: Step maintaining 0 -> 0 incurs exactly 0.0 fee
    assert audit.loc[3, "position"] == 0.0
    assert audit.loc[3, "turnover"] == 0.0
    assert np.isclose(audit.loc[3, "fee"], 0.0, atol=1e-8)


def test_turnover_and_reversal_long_short():
    """Verify reversal transition (+1 -> -1) incurs 2 * fee_bps * 1e-4 fee."""
    fee_bps = 15.0
    fee_rate = 15.0 * 1e-4  # 0.0015
    bt = WalkForwardBacktester(fee_bps=fee_bps, strategy="long_short", threshold=0.0)

    # Step 0: Entry Long (0 -> 1)
    # Step 1: Full Reversal (1 -> -1)
    # Step 2: Exit to Flat (-1 -> 0)
    actual_r = np.array([0.01, -0.02, 0.01])
    predicted_r = np.array([0.02, -0.03, 0.0])

    res = bt.simulate(actual_r, predicted_r, is_log_returns=False)
    audit = res.audit_df

    # Step 0: Entry Long
    assert audit.loc[0, "position"] == 1.0
    assert audit.loc[0, "turnover"] == 1.0
    assert np.isclose(audit.loc[0, "fee"], fee_rate, atol=1e-8)

    # Step 1: Reversal Long to Short
    assert audit.loc[1, "position"] == -1.0
    assert audit.loc[1, "turnover"] == 2.0
    assert np.isclose(audit.loc[1, "fee"], 2.0 * fee_rate, atol=1e-8)

    # Step 2: Exit Short to Flat
    assert audit.loc[2, "position"] == 0.0
    assert audit.loc[2, "turnover"] == 1.0
    assert np.isclose(audit.loc[2, "fee"], fee_rate, atol=1e-8)


# =====================================================================
# 3. Strategy Invariant Tests
# =====================================================================


def test_pure_flat_strategy():
    """When predictions never exceed threshold, positions and fees remain 0 and capital unchanged."""
    initial_cap = 50000.0
    bt = WalkForwardBacktester(
        fee_bps=25.0,
        threshold=0.01,
        strategy="long_flat",
        initial_capital=initial_cap,
    )

    # Predictions all below threshold 0.01
    actual_r = np.array([0.02, -0.03, 0.04, -0.01, 0.05])
    predicted_r = np.array([0.005, -0.002, 0.009, -0.015, 0.000])

    res = bt.simulate(actual_r, predicted_r)
    audit = res.audit_df

    assert (audit["position"] == 0.0).all()
    assert (audit["turnover"] == 0.0).all()
    assert (audit["fee"] == 0.0).all()
    assert (audit["strategy_daily_return"] == 0.0).all()
    assert np.allclose(audit["strategy_equity"], initial_cap)
    assert res.summary["total_trades"] == 0
    assert res.summary["strategy_cumulative_return"] == 0.0
    assert res.summary["strategy_max_drawdown"] == 0.0


def test_always_long_strategy_matches_buy_and_hold():
    """When always long, strategy matches Buy-and-Hold equity (offset only by initial fee)."""
    # 1. Zero fees: exactly identical
    bt_zero = WalkForwardBacktester(fee_bps=0.0, strategy="long_flat", initial_capital=10000.0)
    actual_r = np.array([0.01, -0.015, 0.02, 0.005, -0.008])
    predicted_r = np.array([0.02, 0.01, 0.03, 0.01, 0.02])  # Always positive

    res_zero = bt_zero.simulate(actual_r, predicted_r, is_log_returns=False)
    audit_zero = res_zero.audit_df

    assert np.allclose(audit_zero["strategy_equity"], audit_zero["buy_and_hold_equity"])
    assert np.isclose(
        res_zero.summary["strategy_cumulative_return"],
        res_zero.summary["buy_and_hold_cumulative_return"],
    )

    # 2. Positive fees: offset by initial entry fee only
    fee_bps = 10.0
    fee_rate = fee_bps * 1e-4
    bt_fee = WalkForwardBacktester(fee_bps=fee_bps, strategy="long_flat", initial_capital=10000.0)
    res_fee = bt_fee.simulate(actual_r, predicted_r, is_log_returns=False)
    audit_fee = res_fee.audit_df

    # First step ratio is (1 + r0 - fee) / (1 + r0)
    expected_offset = (1.0 + actual_r[0] - fee_rate) / (1.0 + actual_r[0])
    for t in range(len(actual_r)):
        ratio = audit_fee.loc[t, "strategy_equity"] / audit_fee.loc[t, "buy_and_hold_equity"]
        assert np.isclose(ratio, expected_offset, atol=1e-8)


def test_threshold_invariance_monotonic_trade_count():
    """Increasing threshold tau monotonically reduces or keeps equal total trades."""
    np.random.seed(42)
    n = 100
    actual_r = np.random.normal(0.0005, 0.015, n)
    predicted_r = np.random.normal(0.0005, 0.012, n)

    thresholds = [0.0, 0.002, 0.005, 0.010, 0.020, 0.050]
    trades_list = []

    for tau in thresholds:
        bt = WalkForwardBacktester(threshold=tau, strategy="long_flat")
        res = bt.simulate(actual_r, predicted_r)
        trades_list.append(res.summary["total_trades"])

    # Monotonically non-increasing: trades[i] >= trades[i+1]
    for i in range(len(trades_list) - 1):
        assert (
            trades_list[i] >= trades_list[i + 1]
        ), f"Failed monotonicity: tau={thresholds[i]} ({trades_list[i]}) < tau={thresholds[i+1]} ({trades_list[i+1]})"


# =====================================================================
# 4. Perfect Foresight / Oracle Benchmark Test
# =====================================================================


def test_oracle_benchmark_avoids_negative_compounding():
    """Oracle long/flat strategy captures upside, stays flat on down days, and beats Buy-and-Hold."""
    # Alternating volatile returns: +2%, -2%, +2%, -2%, +2%, -2%
    actual_r = np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02])
    # Oracle predicts correct sign
    oracle_pred = np.sign(actual_r) * 0.01

    bt = WalkForwardBacktester(fee_bps=5.0, threshold=0.0, strategy="long_flat")
    res = bt.simulate(actual_r, oracle_pred, is_log_returns=False)

    # Strategy captures positive days (+2% minus fee), goes flat on negative days (0 minus fee)
    strat_cum = res.summary["strategy_cumulative_return"]
    bnh_cum = res.summary["buy_and_hold_cumulative_return"]

    # Buy and hold suffers from volatility decay: (1+0.02)(1-0.02) = 0.9996 < 1.0
    assert strat_cum > bnh_cum
    assert strat_cum > 0.0
    assert bnh_cum < 0.0
    assert res.summary["strategy_max_drawdown"] < res.summary["buy_and_hold_max_drawdown"]
    assert res.summary["outperformance"] > 0.0


# =====================================================================
# 5. End-to-End Real Data Test
# =====================================================================


def test_end_to_end_real_data_googl_test_split():
    """Run walk-forward simulation on data/processed/GOOGL/test.csv and verify audit DF and summary."""
    test_csv_path = Path("data/processed/GOOGL/test.csv")
    assert test_csv_path.exists(), f"Missing real test file: {test_csv_path}"

    df = pd.read_csv(test_csv_path)
    assert len(df) > 0

    dates = df["Date"].values
    actual_returns = df["log_return"].values
    target_prices = df["Close"].values

    # Base price P_{t-1}: use shifted Close, backfilled with first Close
    base_prices = df["Close"].shift(1).bfill().values

    # Realistic mock predictions: 60% correlated with actuals + noise
    np.random.seed(123)
    noise = np.random.normal(0, 0.005, len(df))
    predicted_returns = 0.6 * actual_returns + noise

    bt = WalkForwardBacktester(
        fee_bps=10.0,
        threshold=0.0005,
        strategy="long_flat",
        initial_capital=10000.0,
    )

    res = bt.simulate(
        actual_returns=actual_returns,
        predicted_returns=predicted_returns,
        dates=dates,
        base_prices=base_prices,
        target_prices=target_prices,
        is_log_returns=True,
    )

    # 1. Audit DataFrame assertions
    audit = res.audit_df
    assert len(audit) == len(df)
    assert len(audit.columns) == 14
    expected_cols = [
        "date",
        "base_price",
        "target_price",
        "actual_return",
        "predicted_return",
        "signal",
        "position",
        "turnover",
        "fee",
        "strategy_daily_return",
        "strategy_equity",
        "buy_and_hold_daily_return",
        "buy_and_hold_equity",
        "strategy_drawdown",
    ]
    assert list(audit.columns) == expected_cols
    assert not audit.isna().any().any(), "Audit DataFrame contains unexpected NaN values"

    # Equity curves must remain positive
    assert (audit["strategy_equity"] > 0.0).all()
    assert (audit["buy_and_hold_equity"] > 0.0).all()

    # Drawdown must be in [0, 1]
    assert (audit["strategy_drawdown"] >= 0.0).all()
    assert (audit["strategy_drawdown"] <= 1.0).all()

    # 2. Equity Curve DataFrame assertions
    eq = res.equity_curve
    assert len(eq) == len(df)
    assert list(eq.columns) == ["date", "strategy_equity", "buy_and_hold_equity"]

    # 3. Summary metrics assertions
    s = res.summary
    required_keys = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
        "total_turnover",
        "total_transaction_costs",
        "strategy_cumulative_return",
        "buy_and_hold_cumulative_return",
        "outperformance",
        "directional_accuracy",
    ]
    for k in required_keys:
        assert k in s, f"Missing key in summary: {k}"
        assert s[k] is not None, f"Key {k} is None"

    # Verify trades occurred
    assert s["total_trades"] > 0
    assert s["total_turnover"] > 0

    # 4. Serialization test
    d = res.to_dict()
    assert "summary" in d
    assert "equity_curve" in d
    assert len(d["equity_curve"]) == len(df)
    assert isinstance(d["summary"]["sharpe_ratio"], (float, int))
