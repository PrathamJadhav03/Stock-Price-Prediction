"""PyTorch Dataset and DataLoader generators for financial time-series sequence modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class StockSequenceDataset(Dataset):
    """Sliding-window sequence dataset for multi-feature financial forecasting.

    Generates windows of shape (seq_length, feature_dim) predicting the single-step
    forward log return at horizon t + seq_length, with zero lookahead bias.
    """

    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        seq_length: int = 30,
        dates: Optional[pd.DatetimeIndex] = None,
        prices: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        if isinstance(features, pd.DataFrame):
            self.features = features.values.astype(np.float32)
        else:
            self.features = np.asarray(features, dtype=np.float32)

        if isinstance(targets, pd.Series):
            self.targets = targets.values.astype(np.float32).reshape(-1, 1)
        else:
            self.targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

        if len(self.features) != len(self.targets):
            raise ValueError(
                f"Features length ({len(self.features)}) does not match targets length ({len(self.targets)})"
            )

        if len(self.features) <= seq_length:
            raise ValueError(
                f"Dataset length ({len(self.features)}) must be greater than seq_length ({seq_length})"
            )

        self.seq_length = seq_length
        self.dates = dates

        if prices is not None:
            if isinstance(prices, pd.Series):
                self.prices = prices.values.astype(np.float64)
            else:
                self.prices = np.asarray(prices, dtype=np.float64)
        else:
            self.prices = None

    def __len__(self) -> int:
        return len(self.features) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input features: window of length seq_length [idx : idx + seq_length]
        x = self.features[idx : idx + self.seq_length]
        # Target: single-step forward return at timestep idx + seq_length
        y = self.targets[idx + self.seq_length]

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_metadata(self, idx: int) -> Dict[str, Union[str, float]]:
        """Retrieve evaluation metadata for sample idx."""
        target_idx = idx + self.seq_length
        base_idx = target_idx - 1

        meta: Dict[str, Union[str, float]] = {
            "target_index": target_idx,
            "base_index": base_idx,
        }

        if self.dates is not None and len(self.dates) > target_idx:
            meta["date"] = str(self.dates[target_idx])

        if self.prices is not None and len(self.prices) > target_idx:
            meta["base_price"] = float(self.prices[base_idx])
            meta["target_price"] = float(self.prices[target_idx])

        return meta


def create_dataloaders(
    data_dir: Union[str, Path] = "data/processed",
    symbol: str = "GOOGL",
    seq_length: int = 30,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Load processed CSV partitions and return PyTorch DataLoaders.

    Parameters
    ----------
    data_dir : str or Path
        Base directory containing processed equity data.
    symbol : str
        Stock ticker symbol.
    seq_length : int
        Lookback sequence length (timesteps).
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of DataLoader worker processes.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, List[str]]
        (train_loader, val_loader, test_loader, feature_names)
    """
    symbol_dir = Path(data_dir) / symbol.strip().upper()
    train_path = symbol_dir / "train.csv"
    val_path = symbol_dir / "val.csv"
    test_path = symbol_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Processed dataset not found at: {p}. Run scripts/prepare_data.py first.")

    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

    feature_cols = [c for c in train_df.columns if c not in ["log_return", "Close"]]

    # Build datasets
    train_ds = StockSequenceDataset(
        features=train_df[feature_cols],
        targets=train_df["log_return"],
        seq_length=seq_length,
        dates=train_df.index,
        prices=train_df["Close"],
    )

    val_ds = StockSequenceDataset(
        features=val_df[feature_cols],
        targets=val_df["log_return"],
        seq_length=seq_length,
        dates=val_df.index,
        prices=val_df["Close"],
    )

    test_ds = StockSequenceDataset(
        features=test_df[feature_cols],
        targets=test_df["log_return"],
        seq_length=seq_length,
        dates=test_df.index,
        prices=test_df["Close"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, feature_cols
