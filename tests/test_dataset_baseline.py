"""Tests for PyTorch sequence dataset and Naive Persistence Baseline."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.returns import reconstruct_prices_from_base
from src.models.dataset import StockSequenceDataset, create_dataloaders
from src.models.baseline import NaivePersistenceBaseline


def test_sequence_dataset_shapes_and_alignment():
    """Verify sliding window shapes and strict temporal target alignment."""
    n_samples = 100
    n_features = 5
    seq_length = 10

    features = np.arange(n_samples * n_features, dtype=np.float32).reshape(n_samples, n_features)
    targets = np.arange(n_samples, dtype=np.float32)

    ds = StockSequenceDataset(features, targets, seq_length=seq_length)
    assert len(ds) == n_samples - seq_length

    # Check first item (idx = 0)
    x0, y0 = ds[0]
    assert x0.shape == (seq_length, n_features)
    assert y0.shape == (1,)
    assert y0.item() == float(seq_length)
    np.testing.assert_array_equal(x0.numpy(), features[:seq_length])

    # Check last item
    x_last, y_last = ds[len(ds) - 1]
    assert y_last.item() == float(n_samples - 1)


def test_create_dataloaders_real_data():
    """Verify DataLoader creation on processed GOOGL dataset."""
    train_loader, val_loader, test_loader, features = create_dataloaders(
        data_dir="data/processed",
        symbol="GOOGL",
        seq_length=30,
        batch_size=64,
    )

    assert len(features) == 23
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    # Inspect first batch
    for batch_x, batch_y in train_loader:
        assert batch_x.ndim == 3
        assert batch_x.shape[1] == 30
        assert batch_x.shape[2] == len(features)
        assert batch_y.ndim == 2
        assert batch_y.shape[1] == 1
        break


def test_naive_persistence_baseline_predictions():
    """Verify NaivePersistenceBaseline prediction dimensions and evaluations."""
    test_df = pd.read_csv("data/processed/GOOGL/test.csv", index_col=0, parse_dates=True)
    y_test = test_df["log_return"].values

    baseline = NaivePersistenceBaseline(strategy="zero_return")
    baseline.fit(test_df, y_test)

    preds = baseline.predict(test_df)
    assert preds.shape == (len(test_df), 1)
    assert (preds == 0.0).all()

    metrics = baseline.evaluate(y_test, preds)
    assert "rmse" in metrics and metrics["rmse"] > 0
    assert "mae" in metrics and metrics["mae"] > 0
    assert "directional_accuracy" in metrics
    assert isinstance(metrics["rmse"], float)


def test_naive_baseline_random_walk_price_invariance():
    """Verify zero return prediction corresponds to price persistence: P_{t+1} = P_t."""
    prices = pd.Series([100.0, 102.5, 99.8, 105.1], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    zero_returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=prices.index)

    reconstructed = reconstruct_prices_from_base(prices, zero_returns)
    # reconstructed at t is prev price * exp(0) = prev price
    assert np.isnan(reconstructed.iloc[0])
    for i in range(1, len(prices)):
        assert np.isclose(reconstructed.iloc[i], prices.iloc[i - 1])
