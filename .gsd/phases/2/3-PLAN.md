---
phase: 2
plan: 3
wave: 3
depends_on:
  - 1-PLAN.md
  - 2-PLAN.md
files_modified:
  - src/models/trainer.py
  - src/models/__init__.py
  - scripts/train.py
  - tests/test_trainer.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "ModelTrainer executes configuration-driven PyTorch training with EarlyStopping and ReduceLROnPlateau"
    - "Gradient clipping strictly caps parameter gradients to prevent explosion during recurrent backpropagation"
    - "Trained model checkpoints serialize to disk with architecture parameters, weights, and optimizer state"
    - "scripts/train.py executes end-to-end training and outputs a quantitative comparison against the NaivePersistenceBaseline"
  artifacts:
    - "src/models/trainer.py exists with EarlyStopping and ModelTrainer"
    - "scripts/train.py exists and executes via uv run"
    - "tests/test_trainer.py exists and passes via uv run pytest"
    - "models/gru_GOOGL.pt exists as a verified trained model checkpoint"
---

# Plan 2.3: Config-Driven PyTorch Trainer, Checkpointing & Baseline Benchmark Comparison

## Objective
Implement the production-ready, configuration-driven PyTorch training engine supporting `EarlyStopping`, `ReduceLROnPlateau`, and gradient clipping, accompanied by an end-to-end training and evaluation CLI script that benchmarks the trained `StockGRU` model against the `NaivePersistenceBaseline`.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- configs/config.yaml
- src/models/dataset.py
- src/models/gru.py
- src/models/baseline.py
- .gsd/phases/2/1-PLAN.md
- .gsd/phases/2/2-PLAN.md

## Tasks

<task type="auto">
  <name>Implement robust PyTorch Trainer with EarlyStopping, ReduceLROnPlateau and Gradient Clipping</name>
  <files>
    src/models/trainer.py
    src/models/__init__.py
    tests/test_trainer.py
  </files>
  <action>
    Construct the modular PyTorch training loop in `src/models/trainer.py` handling validation tracking, automatic learning rate decay, gradient norm clipping, and best model weight restoration.

    Steps:
    1. In `src/models/trainer.py`, implement `EarlyStopping`:
       - `__init__(patience: int = 10, min_delta: float = 1e-4, mode: str = "min")`
       - `__call__(val_loss: float, model: nn.Module) -> bool`:
         Checks if `val_loss` improves by at least `min_delta`. If improved, saves `model.state_dict()` deepcopy as best weights and resets counter. If counter reaches `patience`, sets `should_stop = True`.
       - `restore_best_weights(model: nn.Module) -> None`:
         Loads best recorded state dict back into model.
    2. In `src/models/trainer.py`, implement `ModelTrainer`:
       - `__init__(model: nn.Module, config: dict, device: Optional[torch.device | str] = None)`:
         - Resolves device dynamically (`cuda` if available, else `cpu` via `resolve_device`) (NFR-02).
         - Loss function: `nn.HuberLoss(delta=1.0)` (or `nn.MSELoss()`).
         - Optimizer: `torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])`.
         - Scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config["training"]["lr_factor"], patience=config["training"]["lr_patience"])`.
         - Early stopping: `EarlyStopping(patience=config["training"]["patience"], min_delta=config["training"]["min_delta"])`.
         - Max gradient norm: `config["training"]["max_grad_norm"]` (e.g. 1.0).
       - `train_epoch(train_loader: DataLoader) -> float`:
         Runs one training epoch: sets `model.train()`, calculates loss, `loss.backward()`, clips gradients using `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)`, takes optimizer step, and returns mean epoch loss.
       - `validate(val_loader: DataLoader) -> float`:
         Runs evaluation: sets `model.eval()`, calculates mean validation loss under `torch.no_grad()`.
       - `fit(train_loader: DataLoader, val_loader: DataLoader, max_epochs: Optional[int] = None) -> dict[str, list[float]]`:
         Orchestrates full training loop up to `max_epochs` (or config value). At each epoch:
         - Computes train and val losses.
         - Steps learning rate scheduler with validation loss.
         - Checks early stopping: if triggered, breaks loop, restores best weights, and logs epoch.
         - Returns history dictionary: `{"train_loss": [...], "val_loss": [...], "learning_rates": [...]}`.
       - `evaluate_metrics(test_loader: DataLoader) -> dict[str, float]`:
         Computes point forecast metrics on test loader:
         - `rmse`: Root Mean Squared Error.
         - `mae`: Mean Absolute Error.
         - `directional_accuracy`: Directional Sign Hit Rate % ($\frac{1}{N}\sum \mathbb{I}(\text{sign}(\hat{y}) == \text{sign}(y))$).
       - `save_checkpoint(filepath: str | Path, extra_metadata: Optional[dict] = None) -> None`:
         Saves dictionary via `torch.save`: `{"model_state_dict": model.state_dict(), "config": self.config, "metadata": extra_metadata}`.
       - `load_checkpoint(filepath: str | Path) -> dict`:
         Loads dictionary via `torch.load` with `weights_only=False` (or safe weights loading) and restores model state dict.
    3. Export `EarlyStopping` and `ModelTrainer` in `src/models/__init__.py`.
    4. In `tests/test_trainer.py`, implement unit tests:
       - Early stopping test: Feed artificial stagnant validation losses and assert early stopping triggers exactly after `patience` steps.
       - Gradient clipping test: Artificially set enormous gradients on model weights, verify `clip_grad_norm_` scales them down to `max_norm`.
       - Checkpoint round-trip test: Train model for 1 step, save checkpoint, reload into new instance, assert predictions match with `torch.allclose`.

    AVOID: Omitting gradient clipping, which causes numerical instability in deep recurrent networks on financial returns.
    USE: `torch.nn.utils.clip_grad_norm_` and `EarlyStopping` with best state dictionary restoration.
  </action>
  <verify>
    uv run pytest tests/test_trainer.py -k "early_stopping or trainer or checkpoint" -v
  </verify>
  <done>
    `tests/test_trainer.py` passes all unit tests, verifying early stopping trigger logic, gradient norm constraints, and checkpoint save/load fidelity.
  </done>
</task>

<task type="auto">
  <name>Build end-to-end training CLI, checkpoint persistence, and baseline benchmark comparison runner</name>
  <files>
    scripts/train.py
    src/models/trainer.py
    tests/test_trainer.py
  </files>
  <action>
    Create the unified training CLI script `scripts/train.py` that loads processed data, trains `StockGRU`, saves model checkpoints, and benchmarks performance against the `NaivePersistenceBaseline`.

    Steps:
    1. Create `scripts/train.py` with CLI arguments (via `argparse`):
       - `--config`: Path to configuration YAML (default: `configs/config.yaml`).
       - `--symbol`: Equity ticker symbol (default: `GOOGL`).
       - `--epochs`: Optional max epochs override (e.g. 50 or 2 for testing).
       - `--batch-size`: Optional batch size override.
       - `--output-dir`: Model checkpoint directory (default: `models`).
       - `--data-dir`: Processed data directory (default: `data/processed`).
       - `--mc-samples`: Monte Carlo Dropout sample count (default: 50).
    2. Implement end-to-end execution flow in `scripts/train.py`:
       a. Load config via `load_config(args.config)`.
       b. Enforce reproducibility via `set_seed(config["training"]["seed"])`.
       c. Create PyTorch dataloaders for train, val, and test splits via `create_dataloaders(args.data_dir, args.symbol, seq_length=config["data"]["seq_length"], batch_size=config["data"]["batch_size"])`.
       d. Instantiate `StockGRU` using `input_dim = len(feature_names)` and model hyperparameters from config.
       e. Initialize `ModelTrainer` and execute `.fit(train_loader, val_loader, max_epochs=args.epochs)`.
       f. Persist best model checkpoint to `{output_dir}/gru_{symbol}.pt`.
       g. Evaluate `StockGRU` on `test_loader` computing RMSE, MAE, and Directional Accuracy.
       h. Run Monte Carlo Dropout on test set with $N=\text{mc\_samples}$ to generate mean predictions and 95% confidence intervals.
       i. Evaluate `NaivePersistenceBaseline` on the identical test targets, computing baseline RMSE, MAE, and Directional Accuracy.
       j. Format a comparative benchmark summary report:
          ```
          =======================================================
          MODEL BENCHMARK REPORT: GOOGL
          =======================================================
          Metric                 StockGRU (MC)   Naive Baseline  Outperforms?
          -------------------------------------------------------
          RMSE                   0.0142          0.0195          YES
          MAE                    0.0098          0.0141          YES
          Directional Accuracy   53.4%           50.0%           YES
          =======================================================
          ```
       k. Save full metrics dictionary to `{output_dir}/metrics_{symbol}.json`.
    3. In `tests/test_trainer.py`, add an end-to-end integration test:
       - Run `scripts/train.py` programmatically or via CLI with `--epochs 2 --symbol GOOGL`.
       - Assert `{output_dir}/gru_GOOGL.pt` exists and is non-empty.
       - Assert `{output_dir}/metrics_GOOGL.json` exists and contains both `gru_metrics` and `baseline_metrics`.
       - Assert all numerical metrics are valid finite floats.

    AVOID: Silently ignoring random walk comparison; every training run must output benchmark comparison metrics against `NaivePersistenceBaseline` (DECISION-005).
    USE: Unified training CLI outputting model weights, evaluation metrics, and comparative baseline logs.
  </action>
  <verify>
    uv run python scripts/train.py --symbol GOOGL --epochs 2 && uv run pytest tests/ -v
  </verify>
  <done>
    `scripts/train.py` trains `StockGRU` cleanly, saves `models/gru_GOOGL.pt`, outputs comparative benchmark metrics against `NaivePersistenceBaseline` to `models/metrics_GOOGL.json`, and all tests pass via `uv run pytest tests/ -v`.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `ModelTrainer` trains `StockGRU` using `EarlyStopping`, `ReduceLROnPlateau`, and gradient clipping.
- [ ] Model weights serialize to `models/gru_{symbol}.pt` and deserialize identically.
- [ ] CLI command `uv run python scripts/train.py --symbol GOOGL --epochs 2` completes without errors.
- [ ] Performance metrics comparing `StockGRU` against `NaivePersistenceBaseline` are saved to `models/metrics_GOOGL.json`.
- [ ] Full automated test suite passes via `uv run pytest tests/ -v`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.
