"""Resilient market data ingestion with local disk caching and offline CSV fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe to standard OHLCV schema with DatetimeIndex.
    
    Standard schema:
    - Index: DatetimeIndex named 'Date', monotonically increasing, timezone-naive
    - Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    - Types: float64
    """
    df = df.copy()

    # Handle multi-index columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        # Extract first level or level matching OHLCV names
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Normalize column names (strip whitespace and title-case)
    df.columns = [str(c).strip().title() for c in df.columns]

    # If 'Date' is a column, set it as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index.name = "Date"
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Sort index monotonically and drop duplicate dates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required OHLCV columns: {missing}. Present: {list(df.columns)}")

    # Keep only OHLCV and cast to float64
    df = df[required_cols].astype("float64")

    # Clean forward fill / backward fill for any minor gaps (e.g. IBM dataset has 1 missing value)
    df = df.ffill().bfill()

    return df


def fetch_stock_data(
    symbol: str,
    start_date: str = "2006-01-01",
    end_date: str = "2018-01-01",
    cache_dir: str = "data/raw",
    fallback_dir: str = "data/raw",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch equity OHLCV data via cache, yfinance, or local repository fallback.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., 'AAPL', 'GOOGL', 'AMZN', 'IBM', 'MSFT').
    start_date : str
        Start date formatted as YYYY-MM-DD.
    end_date : str
        End date formatted as YYYY-MM-DD.
    cache_dir : str
        Directory to store and read cached CSVs.
    fallback_dir : str
        Directory containing historical repository CSV records.
    force_refresh : bool
        If True, ignore cache and attempt fresh download.
        
    Returns
    -------
    pd.DataFrame
        Normalized OHLCV DataFrame indexed by Date.
        
    Raises
    ------
    FileNotFoundError
        If data could not be retrieved via cache, network, or fallback directory.
    """
    clean_sym = symbol.strip().upper()
    cache_path = Path(cache_dir) / f"{clean_sym}.csv"

    # 1. Check local cache
    if not force_refresh and cache_path.exists():
        try:
            logger.info("Loading %s from local cache: %s", clean_sym, cache_path)
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            norm_df = _normalize_ohlcv_dataframe(cached_df)
            filtered = norm_df.loc[start_date:end_date]
            if not filtered.empty:
                return filtered
        except Exception as exc:
            logger.warning("Failed reading cache for %s (%s). Falling back.", clean_sym, exc)

    # 2. Try yfinance download
    try:
        logger.info("Attempting live yfinance download for %s (%s to %s)...", clean_sym, start_date, end_date)
        raw_yf = yf.download(clean_sym, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if raw_yf is not None and not raw_yf.empty:
            norm_df = _normalize_ohlcv_dataframe(raw_yf)
            if not norm_df.empty:
                # Save to cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                norm_df.to_csv(cache_path)
                return norm_df
    except Exception as exc:
        logger.info("yfinance download unavailable for %s: %s", clean_sym, exc)

    # 3. Search fallback directory
    fallback_path = Path(fallback_dir)
    if fallback_path.exists():
        candidates = list(fallback_path.glob(f"{clean_sym}_*.csv")) + list(fallback_path.glob(f"{clean_sym}.csv"))
        if candidates:
            cand = candidates[0]
            logger.info("Found fallback dataset for %s at %s", clean_sym, cand)
            fb_df = pd.read_csv(cand)
            norm_df = _normalize_ohlcv_dataframe(fb_df)
            filtered = norm_df.loc[start_date:end_date]
            if not filtered.empty:
                # Cache the normalized copy
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                norm_df.to_csv(cache_path)
                return filtered
            return norm_df

    raise FileNotFoundError(
        f"Unable to load market data for '{clean_sym}'. "
        f"Cache missing at '{cache_path}', yfinance failed/offline, and no matching CSV found in '{fallback_dir}'."
    )
