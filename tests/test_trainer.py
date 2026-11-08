"""Unit and integration tests for ModelTrainer, EarlyStopping, and gradient clipping."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import (
    EarlyStopping,
    ModelTrainer,
    StockGRU,
    load_config,
)


class DummyLinearModel(nn.Module):
    """Simple linear regression model for unit testing trainer behaviors."""

    def __init__(self, input_dim: int = 4, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.fc(x)


@pytest.fixture
def mock_trainer_config() -> dict:
    return {
        "model": {
            "input_dim": 4,
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.1,
            "use_layer_norm": True,
        },
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 10,
            "patience": 3,
            "min_delta": 1e-4,
            "lr_factor": 0.5,
            "lr_patience": 2,
            "max_grad_norm": 1.0,
            "device": "cpu",
            "loss_function": "huber",
        },
    }


def test_early_stopping_trigger():
    """Verify EarlyStopping triggers strictly after patience stagnant evaluations."""
    model = DummyLinearModel(4, 1)
    patience = 3
    es = EarlyStopping(patience=patience, min_delta=1e-3, mode="min")

    # Initial epoch sets best score
    stop_0 = es(0.50, model)
    assert not stop_0
    assert es.counter == 0
    assert es.best_score == 0.50

    # Epoch 1: worse loss
    stop_1 = es(0.51, model)
    assert not stop_1
    assert es.counter == 1

    # Epoch 2: marginal improvement below min_delta
    stop_2 = es(0.4995, model)
    assert not stop_2
    assert es.counter == 2

    # Epoch 3: exceeds patience -> triggers stop
    stop_3 = es(0.52, model)
    assert stop_3
    assert es.counter == 3
    assert es.should_stop


def test_early_stopping_restores_best_weights():
    """Verify EarlyStopping accurately captures and restores the lowest loss weights."""
    model = DummyLinearModel(2, 1)
    es = EarlyStopping(patience=2, min_delta=1e-3, mode="min")

    # Set initial weights to 1.0
    with torch.no_grad():
        model.fc.weight.fill_(1.0)
        model.fc.bias.fill_(1.0)

    # First evaluation records best weights (1.0)
    es(0.20, model)

    # Modify weights
    with torch.no_grad():
        model.fc.weight.fill_(99.0)
        model.fc.bias.fill_(99.0)

    # Worse evaluation does not overwrite best weights
    es(0.80, model)

    # Restore best weights and assert
    es.restore_best_weights(model)
    assert torch.allclose(model.fc.weight, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(model.fc.bias, torch.tensor([1.0]))


def test_gradient_clipping(mock_trainer_config):
    """Verify that gradient norm clipping bounds gradients to max_norm."""
    model = DummyLinearModel(4, 1)
    trainer = ModelTrainer(model, mock_trainer_config, device="cpu")

    x = torch.randn(8, 4)
    y = torch.randn(8, 1)

    trainer.optimizer.zero_grad()
    preds = model(x)
    loss = trainer.criterion(preds, y)
    loss.backward()

    # Manually amplify gradients to massive values
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.fill_(1000.0)

    # Compute unclipped norm
    total_norm_before = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]),
        2,
    ).item()
    assert total_norm_before > 10.0

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=trainer.max_grad_norm)

    # Verify clipped norm
    total_norm_after = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]),
        2,
    ).item()
    assert total_norm_after <= trainer.max_grad_norm + 1e-5


def test_checkpoint_roundtrip(tmp_path, mock_trainer_config):
    """Verify saving and restoring model checkpoint preserves identical outputs."""
    model = StockGRU(
        input_dim=4,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        use_layer_norm=True,
    )
    trainer = ModelTrainer(model, mock_trainer_config, device="cpu")

    checkpoint_path = tmp_path / "checkpoints" / "gru_test.pt"
    trainer.save_checkpoint(checkpoint_path, extra_metadata={"epoch": 5, "symbol": "TEST"})
    assert checkpoint_path.exists()

    # Create fresh model with random weights
    new_model = StockGRU(
        input_dim=4,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        use_layer_norm=True,
    )
    new_trainer = ModelTrainer(new_model, mock_trainer_config, device="cpu")

    # Ensure weights are initially different
    sample_input = torch.randn(4, 10, 4)
    init_out_1 = model.predict(sample_input)
    init_out_2 = new_model.predict(sample_input)
    assert not torch.allclose(init_out_1, init_out_2)

    # Restore from checkpoint
    loaded_data = new_trainer.load_checkpoint(checkpoint_path)
    assert loaded_data["metadata"]["epoch"] == 5
    assert loaded_data["metadata"]["symbol"] == "TEST"

    # Restored predictions must now match exactly
    restored_out = new_model.predict(sample_input)
    assert torch.allclose(init_out_1, restored_out, atol=1e-6)


def test_trainer_fit_and_metrics(mock_trainer_config):
    """Verify ModelTrainer.fit executes epochs, decays LR or triggers early stopping, and evaluates metrics."""
    # Synthetic small dataset
    torch.manual_seed(42)
    n_samples = 64
    seq_len = 10
    input_dim = 4

    x = torch.randn(n_samples, seq_len, input_dim)
    y = torch.randn(n_samples, 1) * 0.01

    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = StockGRU(
        input_dim=input_dim,
        hidden_dim=16,
        num_layers=1,
        dropout=0.1,
        use_layer_norm=True,
    )
    trainer = ModelTrainer(model, mock_trainer_config, device="cpu")

    history = trainer.fit(train_loader, val_loader, max_epochs=4, verbose=False)

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4

    metrics = trainer.evaluate_metrics(val_loader)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "directional_accuracy" in metrics
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert 0.0 <= metrics["directional_accuracy"] <= 100.0


def test_train_pipeline_integration(tmp_path):
    """Verify scripts/train.py train_pipeline runs end-to-end on real processed data."""
    import math
    from scripts.train import train_pipeline

    output_dir = tmp_path / "models"
    report = train_pipeline(
        config_path="configs/config.yaml",
        symbol="GOOGL",
        epochs=2,
        batch_size=64,
        output_dir=str(output_dir),
        data_dir="data/processed",
        mc_samples=5,  # small sample count for fast unit test
        device="cpu",
        verbose=False,
    )

    # Check files created
    checkpoint_file = output_dir / "gru_GOOGL.pt"
    metrics_file = output_dir / "metrics_GOOGL.json"

    assert checkpoint_file.exists()
    assert checkpoint_file.stat().st_size > 0
    assert metrics_file.exists()
    assert metrics_file.stat().st_size > 0

    # Check metrics structure and validity
    assert "gru_point_metrics" in report
    assert "gru_mc_metrics" in report
    assert "baseline_metrics" in report
    assert "comparison" in report

    for k in ["rmse", "mae", "directional_accuracy"]:
        val = report["gru_mc_metrics"][k]
        assert isinstance(val, (int, float))
        assert not math.isnan(val)
        assert not math.isinf(val)

        base_val = report["baseline_metrics"][k]
        assert isinstance(base_val, (int, float))
        assert not math.isnan(base_val)
        assert not math.isinf(base_val)

    assert 0.0 <= report["gru_mc_metrics"]["directional_accuracy"] <= 100.0
    assert 0.0 <= report["gru_mc_metrics"]["coverage_95_pct"] <= 100.0

