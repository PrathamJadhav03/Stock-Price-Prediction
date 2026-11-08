"""Tests for PyTorch sequence dataset and Naive Persistence Baseline."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dataset import StockSequenceDataset, create_dataloaders


def test_sequence_dataset_shapes_and_alignment():
    """Verify sliding window shapes and strict temporal target alignment."""
    n_samples = 100
    n_features = 5
    seq_length = 10

    # Create dummy monotonic feature matrix and target
    features = np.arange(n_samples * n_features, dtype=np.float32).reshape(n_samples, n_features)
    # Target is simple sequence [0, 1, 2, ..., 99]
    targets = np.arange(n_samples, dtype=np.float32)

    ds = StockSequenceDataset(features, targets, seq_length=seq_length)
    assert len(ds) == n_samples - seq_length

    # Check first item (idx = 0)
    x0, y0 = ds[0]
    assert x0.shape == (seq_length, n_features)
    assert y0.shape == (1,)
    # Target at idx=0 must be targets[seq_length] = 10.0
    assert y0.item() == float(seq_length)
    # x0 must match features[0:seq_length]
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
