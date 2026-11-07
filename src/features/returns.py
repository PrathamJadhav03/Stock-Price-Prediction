"""Stationary log returns calculation and price trajectory reconstruction."""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd


def compute_log_returns(
    prices: Union[pd.Series, np.ndarray],
    drop_first: bool = False,
) -> Union[pd.Series, np.ndarray]:
    """Compute daily log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Array or Series of strictly positive price levels.
    drop_first : bool
        If True, drops the first element (which is NaN for Series or undefined).

    Returns
    -------
    pd.Series or np.ndarray
        Log returns matching input type.
    """
    if isinstance(prices, pd.Series):
        if (prices <= 0).any():
            raise ValueError("All prices must be strictly positive to compute log returns.")
        returns = np.log(prices / prices.shift(1))
        if drop_first:
            returns = returns.dropna()
        return returns
    elif isinstance(prices, np.ndarray):
        prices = np.asarray(prices, dtype=np.float64)
        if (prices <= 0).any():
            raise ValueError("All prices must be strictly positive to compute log returns.")
        if len(prices) == 0:
            return np.array([], dtype=np.float64)
        if len(prices) == 1:
            return np.array([] if drop_first else [np.nan], dtype=np.float64)
        r = np.empty_like(prices)
        r[0] = np.nan
        r[1:] = np.log(prices[1:] / prices[:-1])
        if drop_first:
            return r[1:]
        return r
    else:
        raise TypeError(f"Expected pd.Series or np.ndarray, got {type(prices)}")


def reconstruct_price_trajectory(
    initial_price: float,
    log_returns: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Deterministically convert a sequence of log returns back into price levels:
    P_t = P_0 * exp(sum_{tau=1}^t r_tau) = P_0 * exp(cumsum(r)).

    Parameters
    ----------
    initial_price : float
        Base price P_0 immediately preceding the return sequence.
    log_returns : np.ndarray or pd.Series
        Log return sequence [r_1, r_2, ..., r_T]. If the first element is NaN,
        it will be omitted during reconstruction.

    Returns
    -------
    np.ndarray
        Reconstructed price trajectory of length T (or T-1 if first was NaN).
    """
    if initial_price <= 0:
        raise ValueError(f"initial_price must be strictly positive, got {initial_price}")

    if isinstance(log_returns, pd.Series):
        arr = log_returns.values
    else:
        arr = np.asarray(log_returns, dtype=np.float64)

    # If first element is NaN, discard it
    if len(arr) > 0 and np.isnan(arr[0]):
        arr = arr[1:]

    if len(arr) == 0:
        return np.array([], dtype=np.float64)

    if np.isnan(arr).any():
        raise ValueError("log_returns contains NaN values; cannot reconstruct price trajectory.")

    cumulative_returns = np.cumsum(arr)
    reconstructed = initial_price * np.exp(cumulative_returns)
    return reconstructed


def reconstruct_prices_from_base(
    base_prices: pd.Series,
    log_returns: pd.Series,
) -> pd.Series:
    """Reconstruct an aligned price Series where each point at index t is
    computed from base_prices at t-1: P_t = base_price_{t-1} * exp(r_t).

    Parameters
    ----------
    base_prices : pd.Series
        Series of historical prices.
    log_returns : pd.Series
        Series of 1-step predicted or actual log returns aligned with base_prices index.

    Returns
    -------
    pd.Series
        Reconstructed 1-step price predictions.
    """
    prev_prices = base_prices.shift(1)
    reconstructed = prev_prices * np.exp(log_returns)
    return reconstructed
