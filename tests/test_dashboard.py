"""Unit tests for Plotly dashboard visualization components and KPI card formatting."""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard.components import (
    create_candlestick_chart,
    create_equity_curve_chart,
    create_forecast_fan_chart,
    format_kpi_cards,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Fixture providing synthetic OHLCV and indicator DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
    high = close + np.random.uniform(0.5, 2.0, 100)
    low = close - np.random.uniform(0.5, 2.0, 100)
    open_ = low + np.random.uniform(0.1, 1.5, 100)
    volume = np.random.randint(100_000, 1_000_000, 100)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "bb_upper": close + 3.0,
            "bb_middle": close,
            "bb_lower": close - 3.0,
            "ema_10": close - 0.5,
            "ema_20": close - 1.0,
            "ema_50": close - 1.5,
        },
        index=dates,
    )
    return df


def test_create_candlestick_chart(sample_ohlcv_df: pd.DataFrame) -> None:
    """Verify candlestick chart generation with full technical overlays and volume."""
    fig = create_candlestick_chart(sample_ohlcv_df, title="GOOGL Market Overview")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 5  # OHLC, BB upper, BB lower, BB middle, EMAs, Volume

    trace_types = [t.type for t in fig.data]
    assert "candlestick" in trace_types
    assert "bar" in trace_types
    assert "scatter" in trace_types

    trace_names = [t.name for t in fig.data]
    assert "OHLC" in trace_names
    assert "BB Upper" in trace_names
    assert "BB Lower" in trace_names
    assert "EMA 10" in trace_names
    assert "Volume" in trace_names


def test_create_candlestick_chart_missing_indicators() -> None:
    """Verify graceful candlestick chart generation when technical indicators are absent."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0] * 30,
            "High": [105.0] * 30,
            "Low": [95.0] * 30,
            "Close": [102.0] * 30,
            "Volume": [500_000] * 30,
        },
        index=dates,
    )

    fig = create_candlestick_chart(df)
    assert isinstance(fig, go.Figure)
    trace_types = [t.type for t in fig.data]
    assert "candlestick" in trace_types
    assert "bar" in trace_types
    # No scatter traces for missing indicators
    assert "scatter" not in trace_types


def test_create_candlestick_chart_empty() -> None:
    """Verify empty DataFrame handling."""
    empty_df = pd.DataFrame()
    fig = create_candlestick_chart(empty_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_create_forecast_fan_chart() -> None:
    """Verify Monte Carlo fan chart generation with 95% confidence envelope."""
    dates = [
        (datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(20)
    ]
    prices = [150.0 + i * 0.5 for i in range(20)]
    forecast_date = "2024-01-22"
    current_price = prices[-1]
    pred_price = 161.0
    lower_95 = 157.0
    upper_95 = 165.0

    fig = create_forecast_fan_chart(
        historical_dates=dates,
        historical_prices=prices,
        forecast_date=forecast_date,
        current_price=current_price,
        pred_price=pred_price,
        lower_95_price=lower_95,
        upper_95_price=upper_95,
        symbol="GOOGL",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # Historical, Upper, Lower (Fan), Predicted

    names = [t.name for t in fig.data]
    assert "Historical Close" in names
    assert "95% MC Dropout Fan" in names
    assert "Predicted Price" in names

    # Verify fill mode on confidence fan
    fan_trace = [t for t in fig.data if t.name == "95% MC Dropout Fan"][0]
    assert fan_trace.fill == "tonexty"

    # Verify forecast point annotation exists
    assert len(fig.layout.annotations) >= 1
    assert "Pred: $161.00" in fig.layout.annotations[0].text


def test_create_equity_curve_chart() -> None:
    """Verify equity curve dual subplot with portfolio trajectories and underwater drawdowns."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    strat_eq = 10000.0 * np.cumprod(1.0 + np.random.normal(0.001, 0.01, 50))
    bh_eq = 10000.0 * np.cumprod(1.0 + np.random.normal(0.0005, 0.012, 50))

    equity_df = pd.DataFrame(
        {
            "date": dates,
            "strategy_equity": strat_eq,
            "buy_and_hold_equity": bh_eq,
        }
    )

    fig = create_equity_curve_chart(equity_df, symbol="AMZN")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # Strategy eq, B&H eq, Strat DD, B&H DD

    names = [t.name for t in fig.data]
    assert "Strategy" in names
    assert "Buy & Hold" in names
    assert "Strategy Drawdown" in names
    assert "B&H Drawdown" in names

    # Verify drawdown trace is filled to zero
    strat_dd_trace = [t for t in fig.data if t.name == "Strategy Drawdown"][0]
    assert strat_dd_trace.fill == "tozeroy"


def test_create_equity_curve_chart_empty() -> None:
    """Verify empty equity DataFrame handling."""
    fig = create_equity_curve_chart(pd.DataFrame(), symbol="MSFT")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_format_kpi_cards() -> None:
    """Verify KPI card formatters for complete backtest summary metrics."""
    summary = {
        "strategy_cumulative_return": 0.285,
        "buy_and_hold_cumulative_return": 0.150,
        "strategy_annualized_return": 0.321,
        "buy_and_hold_annualized_return": 0.165,
        "strategy_annualized_volatility": 0.185,
        "buy_and_hold_annualized_volatility": 0.220,
        "strategy_sharpe_ratio": 1.735,
        "buy_and_hold_sharpe_ratio": 0.750,
        "strategy_sortino_ratio": 2.450,
        "buy_and_hold_sortino_ratio": 0.980,
        "strategy_max_drawdown": 8.5,
        "buy_and_hold_max_drawdown": 15.5,
        "strategy_win_rate": 56.2,
        "buy_and_hold_win_rate": 51.0,
        "strategy_profit_factor": 1.48,
        "buy_and_hold_profit_factor": 1.12,
        "directional_accuracy": 57.8,
        "total_trades": 42,
        "final_strategy_equity": 12850.0,
        "final_buy_and_hold_equity": 11500.0,
        "total_transaction_costs": 142.50,
    }

    cards = format_kpi_cards(summary)
    assert len(cards) == 12

    labels = [c["label"] for c in cards]
    assert "Cumulative Return" in labels
    assert "Annualized Return (CAGR)" in labels
    assert "Sharpe Ratio" in labels
    assert "Maximum Drawdown" in labels
    assert "Win Rate" in labels
    assert "Profit Factor" in labels
    assert "Total Transaction Fees" in labels
    assert "Executed Trades" in labels
    assert "Final Portfolio Equity" in labels

    # Validate Cumulative Return card content
    cum_card = [c for c in cards if c["label"] == "Cumulative Return"][0]
    assert cum_card["strategy"] == "+28.50%"
    assert cum_card["benchmark"] == "+15.00%"
    assert cum_card["delta"] == "+13.50%"
    assert cum_card["is_positive"] is True

    # Validate percentage-native metrics are NOT multiplied by 100 twice
    dd_card = [c for c in cards if c["label"] == "Maximum Drawdown"][0]
    assert dd_card["strategy"] == "8.50%"
    assert dd_card["delta_color"] == "inverse"

    win_card = [c for c in cards if c["label"] == "Win Rate"][0]
    assert win_card["strategy"] == "56.20%"

    dir_card = [c for c in cards if c["label"] == "Directional Accuracy"][0]
    assert dir_card["strategy"] == "57.80%"

    fee_card = [c for c in cards if c["label"] == "Total Transaction Fees"][0]
    assert fee_card["delta_color"] == "inverse"


def test_format_kpi_cards_missing_and_zero_division() -> None:
    """Verify KPI card formatters handle missing keys, None, and zero values gracefully."""
    sparse_summary = {
        "cumulative_return": 0.05,
        "strategy_max_drawdown": 0.0,
        "buy_and_hold_max_drawdown": None,
        "strategy_sharpe_ratio": None,
    }

    cards = format_kpi_cards(sparse_summary)
    assert isinstance(cards, list)
    assert len(cards) >= 1

    cum_card = [c for c in cards if c["label"] == "Cumulative Return"][0]
    assert cum_card["strategy"] == "+5.00%"
    assert cum_card["benchmark"] == "N/A"


def test_dashboard_app_module_load() -> None:
    """Verify that src.dashboard.app module loads without syntax or import errors."""
    import src.dashboard.app as app_mod

    assert hasattr(app_mod, "main")
    assert hasattr(app_mod, "BackendClient")
    assert hasattr(app_mod, "get_service")
    assert hasattr(app_mod, "load_ticker_data")


def test_backend_client_local_fallback() -> None:
    """Verify BackendClient initializes in-process fallback when remote REST is unreachable."""
    from src.dashboard.app import BackendClient

    # Point to an unreachable mock port
    client = BackendClient(base_url="http://localhost:59999")
    assert client.is_remote is False
    assert client.mode_name == "In-Process Service"

    models = client.get_available_models()
    assert isinstance(models, list)
    assert "GOOGL" in models


def test_load_ticker_data() -> None:
    """Verify load_ticker_data returns unscaled market OHLCV and technical indicators."""
    from src.dashboard.app import load_ticker_data

    df = load_ticker_data("GOOGL")
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    for col in ["Close", "Open", "High", "Low", "Volume", "rsi_14", "volatility_20", "atr_14"]:
        assert col in df.columns
    assert (df["Volume"] >= 0).all()
    assert (df["High"] >= df["Low"]).all()
    assert (df["rsi_14"] >= 0.0).all() and (df["rsi_14"] <= 100.0).all()
    assert (df["volatility_20"] >= 0.0).all()

