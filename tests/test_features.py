"""Tests for feature engineering modules (returns & technical indicators)."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import fetch_stock_data
from src.features.returns import (
    compute_log_returns,
    reconstruct_price_trajectory,
    reconstruct_prices_from_base,
)
from src.features.technical import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_moving_averages,
    compute_rolling_volatility,
    add_all_technical_features,
)


def test_log_returns_invertibility_on_real_data():
    """Verify log return inversion matches true price series within 1e-5 error."""
    df = fetch_stock_data(symbol="GOOGL", fallback_dir="Data")
    prices = df["Close"]

    # Compute log returns
    log_ret = compute_log_returns(prices, drop_first=True)
    assert not log_ret.empty
    assert not log_ret.isna().any()

    # Reconstruct from initial price P_0
    initial_price = float(prices.iloc[0])
    reconstructed = reconstruct_price_trajectory(initial_price, log_ret)

    # Actual prices from t=1..T
    actual = prices.iloc[1:].values

    assert len(reconstructed) == len(actual)
    max_error = np.max(np.abs(reconstructed - actual))
    assert max_error < 1e-5, f"Maximum reconstruction error {max_error} exceeds 1e-5 threshold"


def test_log_returns_constant_prices():
    """Verify log returns of constant price sequence are strictly zero and reversible."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    returns = compute_log_returns(prices, drop_first=True)
    assert (returns == 0.0).all()

    reconstructed = reconstruct_price_trajectory(100.0, returns)
    assert np.allclose(reconstructed, [100.0, 100.0, 100.0])


def test_reconstruct_prices_from_base():
    """Verify 1-step point-in-time price reconstruction."""
    prices = pd.Series([100.0, 105.0, 102.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    log_ret = compute_log_returns(prices)

    reconstructed = reconstruct_prices_from_base(prices, log_ret)
    assert np.isnan(reconstructed.iloc[0])
    assert np.isclose(reconstructed.iloc[1], 105.0)
    assert np.isclose(reconstructed.iloc[2], 102.0)


def test_invalid_prices_raise_error():
    """Verify non-positive prices raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        compute_log_returns(pd.Series([100.0, 0.0, -5.0]))

    with pytest.raises(ValueError, match="strictly positive"):
        reconstruct_price_trajectory(-10.0, np.array([0.01, 0.02]))


def test_technical_indicator_bounds():
    """Verify mathematical bounds for RSI and Bollinger Bands."""
    df = fetch_stock_data(symbol="AMZN", fallback_dir="Data")

    rsi = compute_rsi(df["Close"], period=14).dropna()
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all(), "RSI must strictly remain in [0, 100]"

    bb = compute_bollinger_bands(df["Close"], period=20).dropna()
    assert (bb["bb_upper"] >= bb["bb_middle"]).all(), "Bollinger Upper must be >= Middle"
    assert (bb["bb_middle"] >= bb["bb_lower"]).all(), "Bollinger Middle must be >= Lower"


def test_zero_lookahead_leakage():
    """Verify modifying future prices does NOT alter indicators computed at past timesteps."""
    df1 = fetch_stock_data(symbol="IBM", fallback_dir="Data").iloc[:100].copy()
    df2 = df1.copy()

    # Modify prices in the future (from index 80 onwards)
    df2.iloc[80:, df2.columns.get_loc("Close")] = df2.iloc[80:, df2.columns.get_loc("Close")] * 2.0
    df2.iloc[80:, df2.columns.get_loc("High")] = df2.iloc[80:, df2.columns.get_loc("High")] * 2.0

    feat1 = add_all_technical_features(df1, drop_burn_in=False)
    feat2 = add_all_technical_features(df2, drop_burn_in=False)

    # Features up to index 79 must be IDENTICAL
    pd.testing.assert_frame_equal(
        feat1.iloc[:80],
        feat2.iloc[:80],
        check_exact=True,
    )


def test_add_all_technical_features_zero_nans():
    """Verify full feature extraction returns a clean DataFrame with zero NaNs."""
    df = fetch_stock_data(symbol="GOOGL", fallback_dir="Data")
    features = add_all_technical_features(df, drop_burn_in=True)

    assert not features.empty
    assert features.isna().sum().sum() == 0, "Enriched feature matrix must contain exactly 0 NaNs after burn-in drop"
    assert "log_return" in features.columns
    assert "rsi_14" in features.columns
    assert "macd_hist" in features.columns
    assert "bb_percent" in features.columns
    assert "atr_14" in features.columns
    assert "volatility_20" in features.columns
    assert "sma_50" in features.columns
