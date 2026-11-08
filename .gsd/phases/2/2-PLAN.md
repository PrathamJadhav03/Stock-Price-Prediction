---
phase: 2
plan: 2
wave: 2
depends_on:
  - 1-PLAN.md
files_modified:
  - src/models/gru.py
  - src/models/__init__.py
  - tests/test_gru.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "StockGRU incorporates multi-layer GRU, LayerNorm, and Dropout layers with dynamic device placement"
    - "Model forward pass accepts (B, L, D) sequential tensors and outputs (B, 1) predicted log returns"
    - "Monte Carlo Dropout generates empirical distributions across N stochastic passes without disabling dropout at inference"
    - "Empirical 95% confidence intervals bound the predictive mean strictly (lower_95 <= mean <= upper_95)"
  artifacts:
    - "src/models/gru.py exists with StockGRU and predict_mc"
    - "src/models/__init__.py exports StockGRU"
    - "tests/test_gru.py exists and passes via uv run pytest"
---

# Plan 2.2: PyTorch GRU Architecture & Monte Carlo Dropout Uncertainty Engine

## Objective
Build and validate the core PyTorch `StockGRU` forecasting model featuring multi-layer GRU dynamics, `nn.LayerNorm` hidden state stabilization, and Monte Carlo (MC) Dropout inference to quantify predictive uncertainty via 95% confidence intervals.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- configs/config.yaml
- src/models/dataset.py
- .gsd/phases/2/1-PLAN.md

## Tasks

<task type="auto">
  <name>Implement PyTorch StockGRU Architecture with LayerNorm and Dropout</name>
  <files>
    src/models/gru.py
    src/models/__init__.py
  </files>
  <action>
    Construct the modular PyTorch GRU architecture designed for multivariate time-series return forecasting with internal layer normalization.

    Steps:
    1. In `src/models/gru.py`, implement `StockGRU(nn.Module)`:
       - Initialization parameters:
         - `input_dim: int`: Number of input features per timestep (e.g., 24).
         - `hidden_dim: int = 64`: Hidden state size for GRU cells.
         - `num_layers: int = 2`: Number of stacked GRU layers.
         - `output_dim: int = 1`: Forecasting horizon (single-step log return).
         - `dropout: float = 0.2`: Dropout probability between GRU layers and before linear head.
         - `use_layer_norm: bool = True`: Whether to apply LayerNorm to GRU representations.
       - Architecture components:
         - `self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)`
         - `self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()`
         - `self.dropout = nn.Dropout(p=dropout)`
         - `self.fc = nn.Linear(hidden_dim, output_dim)`
       - Forward pass:
         - Input: $X \in \mathbb{R}^{B \times L \times D}$
         - GRU forward: `out, h_n = self.gru(X)` where `out` has shape $(B, L, H)$.
         - Extract last timestep: `last_step = out[:, -1, :]` of shape $(B, H)$.
         - Normalization: `normed = self.layer_norm(last_step)`
         - Regularization: `dropped = self.dropout(normed)`
         - Projection: `pred = self.fc(dropped)` of shape $(B, 1)$.
         - Return `pred`.
    2. Export `StockGRU` in `src/models/__init__.py`.

    AVOID: Unstabilized deep RNNs without LayerNorm that produce vanishing or exploding gradients across volatile market regimes.
    USE: `nn.LayerNorm(hidden_dim)` applied to the final hidden representations before the output linear layer.
  </action>
  <verify>
    uv run python -c "import torch; from src.models.gru import StockGRU; m = StockGRU(input_dim=24, hidden_dim=64, num_layers=2); x = torch.randn(8, 30, 24); y = m(x); assert y.shape == (8, 1); print('StockGRU module verified. Output shape:', y.shape)"
  </verify>
  <done>
    `StockGRU` initializes cleanly, accepts $(B, 30, 24)$ input tensors, applies LayerNorm and Dropout, and returns $(B, 1)$ predictions.
  </done>
</task>

<task type="auto">
  <name>Implement Monte Carlo Dropout uncertainty quantification and comprehensive unit tests</name>
  <files>
    src/models/gru.py
    tests/test_gru.py
  </files>
  <action>
    Implement Monte Carlo (MC) Dropout inference methods directly on `StockGRU` to generate predictive distributions and 95% confidence intervals, backed by exhaustive unit tests.

    Steps:
    1. In `src/models/gru.py`, add inference methods to `StockGRU`:
       - `predict(self, x: torch.Tensor) -> torch.Tensor`:
         Standard deterministic inference: sets model to `eval()` mode, executes `torch.no_grad()`, and returns $(B, 1)$ tensor.
       - `predict_mc(self, x: torch.Tensor, n_samples: int = 50) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`:
         - Enables dropout sampling while keeping recurrent and normalization weights deterministic:
           - Sets model to `eval()` mode, but explicitly enables dropout:
             `self.dropout.train(True)`
             (If GRU has internal dropout, set `self.gru.train(True)` or maintain internal dropout behavior).
         - Under `torch.no_grad()`, executes $N = \text{n\_samples}$ forward passes:
           Stack outputs to form tensor of shape $(N, B, 1)$.
         - Calculate statistical metrics across the $N$ sample dimension:
           - Mean prediction: $\mu = \text{samples.mean(dim=0)}$ shape $(B, 1)$
           - Standard deviation: $\sigma = \text{samples.std(dim=0)}$ shape $(B, 1)$
           - 95% Confidence Interval:
             - Lower bound (2.5th percentile): `lower_95 = torch.quantile(samples, q=0.025, dim=0)`
             - Upper bound (97.5th percentile): `upper_95 = torch.quantile(samples, q=0.975, dim=0)`
         - Reset model back to standard `eval()` mode.
         - Returns `(mean, std, lower_95, upper_95)`.
    2. In `tests/test_gru.py`, implement automated test suite:
       - Tensor dimension test: Test varying batch sizes ($B=1, 16, 64$), sequence lengths ($L=10, 30, 60$), input dims ($D=5, 24$).
       - Determinism test: In standard `eval()` mode, two sequential calls to `predict(x)` return identical values with diff $0.0$.
       - MC Dropout stochasticity test: For `dropout > 0`, assert that individual MC passes differ, and `std > 0` everywhere.
       - Confidence interval sanity test: Verify `lower_95 <= mean <= upper_95` holds elementwise for all batch items.
       - Device-agnostic test (NFR-02): Verify `StockGRU` moves and computes cleanly on `torch.device("cpu")` and `torch.device("cuda")` (if available).
       - Integration test with `StockSequenceDataset`: Feed a real batch from `create_dataloaders("data/processed", "GOOGL")` through `StockGRU` and verify output shapes and loss calculation with `nn.MSELoss`.

    AVOID: Calling `model.eval()` without explicitly enabling dropout modules during MC Dropout inference (which reduces variance to zero).
    USE: Explicit `self.dropout.train(True)` during MC sampling and empirical quantile calculation (`torch.quantile`).
  </action>
  <verify>
    uv run pytest tests/test_gru.py -v
  </verify>
  <done>
    `tests/test_gru.py` passes 100%, verifying tensor shapes, eval determinism, MC Dropout uncertainty variance, and 95% confidence interval bounds.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `StockGRU` executes forward passes accepting input $(B, L, D)$ and returning $(B, 1)$.
- [ ] `StockGRU` incorporates `nn.LayerNorm` to stabilize recurrent hidden representations.
- [ ] `predict_mc(x, n_samples=50)` returns empirical mean, std, lower_95, and upper_95 tensors.
- [ ] Elementwise relation `lower_95 <= mean <= upper_95` holds across all predictions.
- [ ] All tests in `tests/test_gru.py` pass via `uv run pytest tests/test_gru.py`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.
