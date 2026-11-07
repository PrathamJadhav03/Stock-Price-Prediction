"""Chronological temporal splitter and serialized zero-leakage feature pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class TemporalSplitter:
    """Strict chronological time-series data partitioner with zero shuffling."""

    @staticmethod
    def split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train, val, and test subsets chronologically.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame indexed by monotonic DatetimeIndex.
        train_ratio : float
            Fraction of historical data allocated to training (default 0.70).
        val_ratio : float
            Fraction allocated to validation (default 0.15).
        test_ratio : float
            Fraction allocated to test (default 0.15).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, val_df, test_df)
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0 (got train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be monotonically increasing for temporal splitting.")

        n_total = len(df)
        if n_total < 3:
            raise ValueError(f"DataFrame too small for 3-way split (rows={n_total})")

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Ensure every partition gets at least 1 sample
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n_total:
            n_train = n_total - 2
            n_val = 1

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()

        # Enforce temporal boundaries
        if len(val_df) > 0 and len(train_df) > 0:
            assert train_df.index.max() < val_df.index.min(), "Leakage detected: train overlap with val"
        if len(test_df) > 0 and len(val_df) > 0:
            assert val_df.index.max() < test_df.index.min(), "Leakage detected: val overlap with test"

        return train_df, val_df, test_df


class FeaturePipeline:
    """Zero-leakage feature scaling pipeline with artifact serialization."""

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> FeaturePipeline:
        """Fit scaler strictly on training features.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature DataFrame.
        """
        self.feature_names = list(X_train.columns)
        self.scaler.fit(X_train.values)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters without updating statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Scaled features with original index and column names.
        """
        if not self.is_fitted or self.feature_names is None:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform.")

        # Ensure column order matches training order
        X_aligned = X[self.feature_names]
        scaled_array = self.scaler.transform(X_aligned.values)
        return pd.DataFrame(scaled_array, index=X.index, columns=self.feature_names)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training features and return transformed DataFrame."""
        return self.fit(X_train).transform(X_train)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted pipeline state to disk using joblib.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted FeaturePipeline.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> FeaturePipeline:
        """Reconstruct fitted FeaturePipeline from serialized joblib artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"FeaturePipeline artifact not found at: {path}")

        payload = joblib.load(path)
        instance = cls(scaler_type=payload["scaler_type"])
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = payload["is_fitted"]
        return instance
