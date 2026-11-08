"""PyTorch training engine with EarlyStopping, ReduceLROnPlateau, and Gradient Clipping."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.baseline import NaivePersistenceBaseline


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve target compute device (CUDA if available and requested, otherwise CPU)."""
    dev = str(device_str).lower().strip()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


class EarlyStopping:
    """Early stopping handler with best model weight caching.

    Monitors validation loss and signals termination when performance fails
    to improve by at least `min_delta` after `patience` consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        if mode not in ["min", "max"]:
            raise ValueError(f"Unsupported mode '{mode}'. Choose 'min' or 'max'.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.should_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Evaluate validation metric and update early stopping counter.

        Parameters
        ----------
        val_loss : float
            Current epoch validation loss.
        model : nn.Module
            PyTorch model whose state dictionary is cached on improvements.

        Returns
        -------
        bool
            True if early stopping patience was exceeded, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.should_stop = False
            return False

        if self.mode == "min":
            improved = val_loss < (self.best_score - self.min_delta)
        else:
            improved = val_loss > (self.best_score + self.min_delta)

        if improved:
            self.best_score = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best cached state dictionary into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelTrainer:
    """Configuration-driven PyTorch training orchestrator.

    Integrates AdamW, Huber loss, ReduceLROnPlateau, gradient norm clipping,
    and EarlyStopping for stable training of recurrent architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[Union[torch.device, str]] = None,
    ):
        target_device = device if device is not None else config.get("training", {}).get("device", "auto")
        if isinstance(target_device, str):
            self.device = resolve_device(target_device)
        else:
            self.device = target_device

        self.model = model.to(self.device)
        self.config = config

        training_cfg = config.get("training", {})
        lr = float(training_cfg.get("learning_rate", 1e-3))
        weight_decay = float(training_cfg.get("weight_decay", 1e-4))

        loss_choice = str(training_cfg.get("loss_function", "huber")).lower()
        if loss_choice == "mse":
            self.criterion: nn.Module = nn.MSELoss()
        else:
            self.criterion = nn.HuberLoss(delta=1.0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        lr_factor = float(training_cfg.get("lr_factor", 0.5))
        lr_patience = int(training_cfg.get("lr_patience", 5))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
        )

        es_patience = int(training_cfg.get("patience", 10))
        min_delta = float(training_cfg.get("min_delta", 1e-4))
        self.early_stopping = EarlyStopping(
            patience=es_patience,
            min_delta=min_delta,
            mode="min",
        )

        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Execute one complete training epoch with gradient clipping."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm,
                )

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return float(total_loss / max(1, num_batches))

    def validate(self, val_loader: DataLoader) -> float:
        """Compute mean validation loss under torch.no_grad()."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item()
                num_batches += 1

        return float(total_loss / max(1, num_batches))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Orchestrate training loop across epochs with scheduling and early stopping."""
        epochs = max_epochs or int(self.config.get("training", {}).get("max_epochs", 100))
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            curr_lr = float(self.optimizer.param_groups[0]["lr"])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rates"].append(curr_lr)

            self.scheduler.step(val_loss)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch {epoch:3d}/{epochs:3d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {curr_lr:.2e}"
                )

            if self.early_stopping(val_loss, self.model):
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        "Restoring best model weights."
                    )
                self.early_stopping.restore_best_weights(self.model)
                break

        # Ensure best weights are restored if max_epochs completed without triggering early stopping
        if self.early_stopping.best_weights is not None:
            self.early_stopping.restore_best_weights(self.model)

        return history

    def predict_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract deterministic point predictions and actual targets from a DataLoader."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                preds = self.model(x_batch)
                all_preds.append(preds.detach().cpu().numpy())
                all_targets.append(y_batch.detach().cpu().numpy())

        if not all_preds:
            return np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

        y_pred = np.vstack(all_preds).flatten()
        y_true = np.vstack(all_targets).flatten()
        return y_pred, y_true

    def predict_mc_loader(
        self,
        loader: DataLoader,
        n_samples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Monte Carlo Dropout predictions across DataLoader batches."""
        if not hasattr(self.model, "predict_mc"):
            raise AttributeError("Loaded model does not implement 'predict_mc'")

        all_means = []
        all_stds = []
        all_lowers = []
        all_uppers = []
        all_targets = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            mean, std, lower_95, upper_95 = self.model.predict_mc(x_batch, n_samples=n_samples)

            all_means.append(mean.detach().cpu().numpy())
            all_stds.append(std.detach().cpu().numpy())
            all_lowers.append(lower_95.detach().cpu().numpy())
            all_uppers.append(upper_95.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

        y_mean = np.vstack(all_means).flatten()
        y_std = np.vstack(all_stds).flatten()
        y_lower = np.vstack(all_lowers).flatten()
        y_upper = np.vstack(all_uppers).flatten()
        y_true = np.vstack(all_targets).flatten()

        return y_mean, y_std, y_lower, y_upper, y_true

    def evaluate_metrics(self, test_loader: DataLoader) -> Dict[str, float]:
        """Compute standard point forecast and directional metrics on test loader."""
        y_pred, y_true = self.predict_loader(test_loader)
        return NaivePersistenceBaseline.evaluate(y_true=y_true, y_pred=y_pred)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist model state dictionary, configuration, and metadata."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "metadata": extra_metadata or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Restore model weights from a persisted checkpoint."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
