"""Statistical baseline benchmarks for time-series forecasting."""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import torch


class NaivePersistenceBaseline:
    """Random walk benchmark model predicting zero log return (constant price level: P_t+1 = P_t).

    Financial time-series returns often follow a martingale difference sequence
    E[r_{t+1} | F_t] = 0. Any deep learning model must empirically beat this baseline
    in regression error (RMSE/MAE) and directional accuracy (> 50%).
    """

    def __init__(self, strategy: str = "zero_return"):
        if strategy not in ["zero_return", "last_return"]:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'zero_return' or 'last_return'.")
        self.strategy = strategy
        self.train_mean_return: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
    ) -> NaivePersistenceBaseline:
        """Fit baseline on training targets (computes training mean)."""
        if y is not None:
            if isinstance(y, torch.Tensor):
                y_arr = y.detach().cpu().numpy()
            else:
                y_arr = np.asarray(y)
            self.train_mean_return = float(np.nanmean(y_arr))
        else:
            self.train_mean_return = 0.0

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        last_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """Generate return forecasts.

        Parameters
        ----------
        X : array-like or Tensor
            Input sequence or feature matrix of length N.
        last_returns : array-like, optional
            Recent returns for 'last_return' strategy.

        Returns
        -------
        np.ndarray
            Predicted log returns of shape (N, 1).
        """
        if isinstance(X, torch.Tensor):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)

        if self.strategy == "zero_return":
            return np.zeros((n_samples, 1), dtype=np.float32)
        elif self.strategy == "last_return":
            if last_returns is None:
                raise ValueError("Strategy 'last_return' requires last_returns array.")
            return np.asarray(last_returns, dtype=np.float32).reshape(-1, 1)

        return np.zeros((n_samples, 1), dtype=np.float32)

    @staticmethod
    def evaluate(
        y_true: Union[np.ndarray, pd.Series, torch.Tensor],
        y_pred: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute standard point forecast and directional metrics.

        Parameters
        ----------
        y_true : array-like
            Actual log returns of shape (N, 1) or (N,).
        y_pred : array-like, optional
            Predicted log returns. If None, assumes zero-return baseline.

        Returns
        -------
        Dict[str, float]
            Dictionary containing RMSE, MAE, and Directional Accuracy.
        """
        if isinstance(y_true, torch.Tensor):
            y_t = y_true.detach().cpu().numpy().flatten()
        else:
            y_t = np.asarray(y_true, dtype=np.float64).flatten()

        if y_pred is None:
            y_p = np.zeros_like(y_t)
        else:
            if isinstance(y_pred, torch.Tensor):
                y_p = y_pred.detach().cpu().numpy().flatten()
            else:
                y_p = np.asarray(y_pred, dtype=np.float64).flatten()

        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

        # Point forecast errors
        diff = y_t - y_p
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))

        # Directional Accuracy (Sign Hit Rate %)
        # When y_pred is 0 (neutral), directional match is 0.0
        # Positive prediction hits when actual return is > 0
        # Negative prediction hits when actual return is < 0
        true_sign = np.sign(y_t)
        pred_sign = np.sign(y_p)

        # Hits are defined as predicting the correct non-zero direction
        directional_hits = (pred_sign == true_sign) & (pred_sign != 0)
        valid_eval_steps = len(y_t)
        directional_accuracy = float(np.sum(directional_hits) / max(1, valid_eval_steps) * 100.0)

        return {
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": directional_accuracy,
        }
