"""Tests for PyTorch StockGRU architecture and Monte Carlo Dropout uncertainty."""

import pytest
import torch
import torch.nn as nn

from src.models.dataset import create_dataloaders
from src.models.gru import StockGRU


def test_gru_tensor_shapes():
    """Verify forward pass tensor dimensions across varied batch sizes and sequence lengths."""
    for b in [1, 16, 64]:
        for seq_len in [10, 30]:
            for d in [5, 23]:
                model = StockGRU(input_dim=d, hidden_dim=32, num_layers=2, output_dim=1)
                x = torch.randn(b, seq_len, d)
                out = model(x)
                assert out.shape == (b, 1)


def test_gru_eval_determinism():
    """Verify deterministic predictions when in standard eval mode."""
    model = StockGRU(input_dim=10, hidden_dim=32, num_layers=2, dropout=0.5)
    x = torch.randn(8, 20, 10)

    p1 = model.predict(x)
    p2 = model.predict(x)

    torch.testing.assert_close(p1, p2)


def test_mc_dropout_uncertainty_and_confidence_intervals():
    """Verify Monte Carlo Dropout generates positive variance and strictly bounded 95% CIs."""
    torch.manual_seed(42)
    model = StockGRU(input_dim=10, hidden_dim=64, num_layers=2, dropout=0.3)
    x = torch.randn(16, 25, 10)

    mean, std, lower_95, upper_95 = model.predict_mc(x, n_samples=50)

    assert mean.shape == (16, 1)
    assert std.shape == (16, 1)
    assert lower_95.shape == (16, 1)
    assert upper_95.shape == (16, 1)

    # Uncertainty must be strictly positive with active dropout
    assert (std > 0).all(), "MC Dropout standard deviation must be strictly positive"

    # 95% Confidence Interval must strictly enclose or equal the mean
    assert (lower_95 <= mean + 1e-6).all(), "Lower 95% bound must be <= mean"
    assert (mean <= upper_95 + 1e-6).all(), "Upper 95% bound must be >= mean"
    assert (lower_95 < upper_95).all(), "Lower 95% bound must be strictly < upper 95% bound"


def test_gru_gradient_backprop():
    """Verify backward gradient computation flows through all recurrent and linear parameters."""
    model = StockGRU(input_dim=8, hidden_dim=16, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(4, 15, 8)
    y_true = torch.randn(4, 1)

    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y_true)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_gru_real_dataloader_integration():
    """Verify model forward pass and loss evaluation with real data loader batch."""
    train_loader, _, _, features = create_dataloaders(
        data_dir="data/processed",
        symbol="GOOGL",
        seq_length=30,
        batch_size=32,
    )

    batch_x, batch_y = next(iter(train_loader))
    model = StockGRU(input_dim=len(features), hidden_dim=64, num_layers=2)
    pred = model(batch_x)

    assert pred.shape == batch_y.shape
    criterion = nn.MSELoss()
    loss = criterion(pred, batch_y)
    assert not torch.isnan(loss)
    assert loss.item() > 0
