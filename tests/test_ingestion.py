"""Tests for resilient data ingestion module."""

from pathlib import Path
from unittest.mock import patch
import pytest
import pandas as pd

from src.data.ingestion import fetch_stock_data, _normalize_ohlcv_dataframe


def test_schema_normalization():
    """Verify normalization standardizes column names, index, and types."""
    raw_df = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "HIGH": [12.0, 13.0],
            "low": [9.5, 10.5],
            "close": [11.5, 12.5],
            "volume": [1000, 2000],
            "date": ["2020-01-02", "2020-01-03"],
        }
    )
    norm = _normalize_ohlcv_dataframe(raw_df)
    assert list(norm.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(norm.index, pd.DatetimeIndex)
    assert norm.index.name == "Date"
    assert norm["Close"].dtype == "float64"
    assert norm["Volume"].dtype == "float64"
    assert norm.index.is_monotonic_increasing


def test_fallback_ingestion_when_offline(tmp_path):
    """Verify offline fallback loads from data/raw/*.csv when network fails."""
    with patch("yfinance.download", side_effect=ConnectionError("Offline simulation")):
        df = fetch_stock_data(
            symbol="GOOGL",
            cache_dir=str(tmp_path / "cache"),
            fallback_dir="data/raw",
            force_refresh=True,
        )
        assert not df.empty
        assert len(df) > 2000
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.name == "Date"
        assert df.index.is_monotonic_increasing


def test_caching_behavior(tmp_path):
    """Verify data is cached locally and re-read from cache on subsequent calls."""
    cache_dir = tmp_path / "cache"

    # First call uses fallback and writes to cache
    with patch("yfinance.download", side_effect=ConnectionError("Offline")):
        df1 = fetch_stock_data(
            symbol="MSFT",
            cache_dir=str(cache_dir),
            fallback_dir="data/raw",
        )
    cache_file = cache_dir / "MSFT.csv"
    assert cache_file.exists()

    # Second call should load directly from cache without hitting yfinance or fallback
    with patch("yfinance.download", side_effect=AssertionError("Network should not be called")):
        df2 = fetch_stock_data(
            symbol="MSFT",
            cache_dir=str(cache_dir),
            fallback_dir=str(tmp_path / "nonexistent"),
        )
        assert len(df1) == len(df2)
        pd.testing.assert_frame_equal(df1, df2)


def test_live_download_caching(tmp_path):
    """Verify mock live download saves data to cache directory."""
    cache_dir = tmp_path / "cache"
    mock_data = pd.DataFrame(
        {
            "Open": [150.0, 152.0],
            "High": [155.0, 156.0],
            "Low": [149.0, 151.0],
            "Close": [153.0, 155.0],
            "Volume": [50000.0, 60000.0],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D", name="Date"),
    )

    with patch("yfinance.download", return_value=mock_data):
        df = fetch_stock_data(
            symbol="AAPL",
            start_date="2020-01-01",
            end_date="2020-01-03",
            cache_dir=str(cache_dir),
            fallback_dir="data/raw",
        )
        assert len(df) == 2
        assert (cache_dir / "AAPL.csv").exists()


def test_missing_ticker_raises_file_not_found(tmp_path):
    """Verify requesting a nonexistent ticker raises FileNotFoundError."""
    with patch("yfinance.download", return_value=pd.DataFrame()):
        with pytest.raises(FileNotFoundError, match="Unable to load market data for 'INVALID_TICKER_XYZ'"):
            fetch_stock_data(
                symbol="INVALID_TICKER_XYZ",
                cache_dir=str(tmp_path / "cache"),
                fallback_dir=str(tmp_path / "empty_dir"),
                force_refresh=True,
            )
