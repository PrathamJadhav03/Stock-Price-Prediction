"""PyTorch GRU forecasting architecture with LayerNorm and Monte Carlo Dropout."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class StockGRU(nn.Module):
    """Multi-layer Gated Recurrent Unit (GRU) model with LayerNorm and Monte Carlo Dropout.

    Accepts sequential input tensors of shape (Batch, Seq_Length, Input_Dim)
    and predicts single-step forward log returns of shape (Batch, Output_Dim).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.use_layer_norm = use_layer_norm

        # Recurrent core
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer normalization to stabilize internal representations
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = nn.Identity()

        # Regularization and projection
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted return of shape (batch_size, output_dim).
        """
        # gru_out shape: (batch_size, seq_length, hidden_dim)
        gru_out, _ = self.gru(x)

        # Extract last timestep representation: (batch_size, hidden_dim)
        last_hidden = gru_out[:, -1, :]

        # Normalize and regularize
        normed = self.layer_norm(last_hidden)
        dropped = self.dropout(normed)

        # Project to target dimension
        pred = self.fc(dropped)
        return pred

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference evaluation in eval mode."""
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def predict_mc(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo Dropout sampling to quantify predictive uncertainty.

        Executes N stochastic forward passes by keeping dropout active at inference.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_dim).
        n_samples : int
            Number of stochastic forward samples (default: 50).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (mean, std, lower_95, upper_95) where:
            - mean: Expected prediction of shape (batch_size, output_dim)
            - std: Predictive uncertainty / volatility of shape (batch_size, output_dim)
            - lower_95: 2.5th empirical percentile of shape (batch_size, output_dim)
            - upper_95: 97.5th empirical percentile of shape (batch_size, output_dim)
        """
        self.eval()
        # Explicitly enable dropout layers for stochastic inference
        self.dropout.train(True)
        if self.num_layers > 1 and self.dropout_p > 0:
            self.gru.train(True)

        samples = []
        for _ in range(n_samples):
            pred = self.forward(x)
            samples.append(pred)

        # Reset model back to standard evaluation mode
        self.eval()

        # samples_tensor shape: (n_samples, batch_size, output_dim)
        samples_tensor = torch.stack(samples, dim=0)

        mean = samples_tensor.mean(dim=0)
        std = samples_tensor.std(dim=0)
        lower_95 = torch.quantile(samples_tensor, q=0.025, dim=0)
        upper_95 = torch.quantile(samples_tensor, q=0.975, dim=0)

        return mean, std, lower_95, upper_95
