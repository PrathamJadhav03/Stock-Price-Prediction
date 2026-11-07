"""Comprehensive zero-lookahead technical indicator feature engine."""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from src.features.returns import compute_log_returns


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing.
    
    Formula:
    RSI = 100 - (100 / (1 + RS))
    where RS = EMA(Gains, alpha=1/period) / EMA(Losses, alpha=1/period)
    """
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = f"rsi_{period}"
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute Moving Average Convergence Divergence (MACD)."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        },
        index=series.index,
    )


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands with trailing window."""
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()

    bb_upper = rolling_mean + (num_std * rolling_std)
    bb_lower = rolling_mean - (num_std * rolling_std)
    band_range = (bb_upper - bb_lower).replace(0, np.nan)
    bb_percent = (series - bb_lower) / (band_range + 1e-9)
    bb_width = (bb_upper - bb_lower) / (rolling_mean + 1e-9)

    return pd.DataFrame(
        {
            "bb_middle": rolling_mean,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_percent": bb_percent,
            "bb_width": bb_width,
        },
        index=series.index,
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR) with Wilder's exponential smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr.name = f"atr_{period}"
    return atr


def compute_moving_averages(
    series: pd.Series,
    windows: List[int] = [10, 20, 50],
) -> pd.DataFrame:
    """Compute simple and exponential moving averages across trailing windows."""
    ma_dict = {}
    for w in windows:
        ma_dict[f"sma_{w}"] = series.rolling(window=w).mean()
        ma_dict[f"ema_{w}"] = series.ewm(span=w, adjust=False).mean()
    return pd.DataFrame(ma_dict, index=series.index)


def compute_rolling_volatility(
    log_returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """Compute rolling volatility of log returns."""
    vol = log_returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = f"volatility_{window}"
    return vol


def add_all_technical_features(
    df: pd.DataFrame,
    drop_burn_in: bool = True,
    burn_in_window: int = 50,
) -> pd.DataFrame:
    """Extract full technical indicator feature suite from normalized OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized OHLCV DataFrame with Date index and ['Open', 'High', 'Low', 'Close', 'Volume'].
    drop_burn_in : bool
        If True, drops initial rows up to `burn_in_window` that contain rolling warmup NaNs.
    burn_in_window : int
        Maximum lookback window across indicators (default 50 for SMA_50).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with target 'log_return' and all technical indicators.
    """
    df = df.copy()

    # 1. Target: 1-step forward log return and contemporaneous log return
    close = df["Close"]
    df["log_return"] = compute_log_returns(close)

    # 2. RSI
    df["rsi_14"] = compute_rsi(close, period=14)

    # 3. MACD
    macd_df = compute_macd(close, fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_df], axis=1)

    # 4. Bollinger Bands
    bb_df = compute_bollinger_bands(close, period=20, num_std=2.0)
    df = pd.concat([df, bb_df], axis=1)

    # 5. ATR
    df["atr_14"] = compute_atr(df["High"], df["Low"], close, period=14)

    # 6. Moving Averages
    ma_df = compute_moving_averages(close, windows=[10, 20, 50])
    df = pd.concat([df, ma_df], axis=1)

    # 7. Rolling Volatility
    df["volatility_20"] = compute_rolling_volatility(df["log_return"], window=20, annualize=True)

    # 8. Volume Momentum
    df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"] + 1e-9)

    if drop_burn_in:
        # Drop the warmup rows and ensure no NaNs remain
        df = df.iloc[burn_in_window:].copy()
        df = df.dropna()

    return df
