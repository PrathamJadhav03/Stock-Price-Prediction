---
phase: 2
plan: 1
wave: 1
depends_on: []
files_modified:
  - configs/config.yaml
  - src/models/__init__.py
  - src/models/dataset.py
  - src/models/baseline.py
  - tests/test_dataset_baseline.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Hierarchical configuration YAML defines reproducible hyperparameters, paths, and training settings"
    - "StockSequenceDataset slices multi-feature historical data into sliding sequence tensors without target lookahead leakage"
    - "create_dataloaders produces PyTorch DataLoaders for train, val, and test splits with aligned (B, L, D) and (B, 1) tensor shapes"
    - "NaivePersistenceBaseline implements the random walk benchmark predicting zero log returns (constant price) and computes baseline metrics"
  artifacts:
    - "configs/config.yaml exists with complete data, model, and training configurations"
    - "src/models/__init__.py exists and exports load_config, set_seed, and public model classes"
    - "src/models/dataset.py exists with StockSequenceDataset and create_dataloaders"
    - "src/models/baseline.py exists with NaivePersistenceBaseline"
    - "tests/test_dataset_baseline.py exists and passes via uv run pytest"
---

# Plan 2.1: Configuration Specification, Sequence Dataset & Naive Persistence Baseline

## Objective
Establish the foundational data ingestion and benchmarking infrastructure for Phase 2: define the centralized YAML configuration schema, implement the time-series sliding window PyTorch `Dataset` and `DataLoader` generator, and construct the `NaivePersistenceBaseline` to enforce statistical validation against a random walk.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- scripts/prepare_data.py
- src/features/pipeline.py
- data/processed/GOOGL/train.csv

## Tasks

<task type="auto">
  <name>Create YAML configuration schema and loader utility with global seed initialization</name>
  <files>
    configs/config.yaml
    src/models/__init__.py
  </files>
  <action>
    Create a centralized configuration file in `configs/config.yaml` and utility functions in `src/models/__init__.py` for loading configs and enforcing cross-library reproducibility.

    Steps:
    1. Create `configs/config.yaml` specifying:
       - `data`:
         - `symbol`: "GOOGL"
         - `data_dir`: "data/processed"
         - `models_dir`: "models"
         - `target_col`: "log_return"
         - `price_col`: "Close"
         - `seq_length`: 30
         - `batch_size`: 64
         - `train_ratio`: 0.70
         - `val_ratio`: 0.15
         - `test_ratio`: 0.15
       - `model`:
         - `input_dim`: 24 # 24 technical features excluding target and price
         - `hidden_dim`: 64
         - `num_layers`: 2
         - `dropout`: 0.2
         - `mc_samples`: 50
       - `training`:
         - `learning_rate`: 0.001
         - `weight_decay`: 0.0001
         - `max_epochs`: 100
         - `patience`: 10
         - `min_delta`: 0.0001
         - `lr_factor`: 0.5
         - `lr_patience`: 5
         - `max_grad_norm`: 1.0
         - `seed`: 42
         - `device`: "auto" # Dynamic device selection: cuda if available, else cpu (NFR-02)
    2. In `src/models/__init__.py`, implement:
       - `load_config(config_path: str | Path = "configs/config.yaml") -> dict`:
         Loads and parses YAML configuration safely using `yaml.safe_load`. Validates presence of `data`, `model`, and `training` sections.
       - `set_seed(seed: int = 42) -> None`:
         Seeds Python `random`, NumPy `np.random.seed`, PyTorch `torch.manual_seed`, and if CUDA is available, `torch.cuda.manual_seed_all`, setting deterministic flags (`torch.backends.cudnn.deterministic = True`).
       - `resolve_device(device_str: str = "auto") -> torch.device`:
         Returns `torch.device("cuda")` if available and requested/auto, else `torch.device("cpu")`.

    AVOID: Hardcoding hyperparameters across multiple scripts; all parameters must stem from `configs/config.yaml`.
    USE: `yaml.safe_load` and explicit device resolution supporting CPU and CUDA.
  </action>
  <verify>
    uv run python -c "from src.models import load_config, set_seed, resolve_device; cfg = load_config(); set_seed(cfg['training']['seed']); dev = resolve_device(cfg['training']['device']); print('Config loaded successfully. Device:', dev)"
  </verify>
  <done>
    `configs/config.yaml` is parsed cleanly by `load_config()`, and `set_seed` initializes all RNGs deterministically.
  </done>
</task>

<task type="auto">
  <name>Implement time-series sliding window PyTorch Dataset and DataLoader factory</name>
  <files>
    src/models/dataset.py
    tests/test_dataset_baseline.py
  </files>
  <action>
    Construct the PyTorch dataset pipeline transforming processed CSV partitions into fixed-length sequential sliding windows with aligned single-step return targets.

    Steps:
    1. In `src/models/dataset.py`, implement `StockSequenceDataset(torch.utils.data.Dataset)`:
       - Signature: `__init__(features: np.ndarray, targets: np.ndarray, seq_length: int = 30, dates: Optional[pd.DatetimeIndex] = None, prices: Optional[np.ndarray] = None)`
       - Validates `len(features) == len(targets)` and `len(features) >= seq_length + 1`.
       - Windowing logic:
         - For index $i \in [0, N - \text{seq\_length} - 1]$:
           - Input window $X_i = \text{features}[i : i + \text{seq\_length}]$ of shape $(\text{seq\_length}, D)$
           - Target $y_i = \text{targets}[i + \text{seq\_length}]$ of shape $(1,)$ representing the log return of the timestep immediately following the window.
           - Base price $P_i = \text{prices}[i + \text{seq\_length} - 1]$ (the last known price in the window, needed for price reconstruction).
           - Target price $P_{target, i} = \text{prices}[i + \text{seq\_length}]$.
         - `__len__`: Returns $N - \text{seq\_length}$.
         - `__getitem__(idx)`: Returns tuple `(torch.tensor(X_i, dtype=torch.float32), torch.tensor(y_i, dtype=torch.float32))`.
         - Provide helper methods:
           - `get_metadata(idx: int) -> dict`: Returns date, base_price, target_price for backtest tracing.
    2. In `src/models/dataset.py`, implement `create_dataloaders`:
       - Signature: `create_dataloaders(data_dir: str | Path, symbol: str, seq_length: int = 30, batch_size: int = 64, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]`
       - Loads `train.csv`, `val.csv`, and `test.csv` from `{data_dir}/{symbol}/`.
       - Identifies feature columns: all columns except `Date`, `log_return`, and `Close`.
       - Instantiates `StockSequenceDataset` for train, val, and test.
       - Returns `(train_loader, val_loader, test_loader, feature_names)` where `train_loader` uses `shuffle=True` and val/test use `shuffle=False`.
    3. In `tests/test_dataset_baseline.py`, implement unit tests:
       - Sequence shape test: Verify batch output shapes $(B, L, D)$ and $(B, 1)$.
       - Temporal alignment test: Ensure target at index $i$ corresponds strictly to timestep $i + L$ (zero lookahead into input window).
       - Monotonicity test: Ensure dataloaders stream chronological batches when `shuffle=False`.
       - Real data test: Call `create_dataloaders` on `data/processed/GOOGL/` and verify dataloader lengths and batches.

    AVOID: Including target `log_return` or future prices inside the feature matrix of the sliding window.
    USE: Strict temporal offset where window features $X[i : i+L]$ predict $y[i+L]$.
  </action>
  <verify>
    uv run pytest tests/test_dataset_baseline.py -k "dataset or dataloader" -v
  </verify>
  <done>
    `StockSequenceDataset` and `create_dataloaders` pass all tests, streaming batches of shape $(B, 30, 24)$ and $(B, 1)$ without leakage.
  </done>
</task>

<task type="auto">
  <name>Implement Naive Persistence Baseline and benchmark evaluation engine</name>
  <files>
    src/models/baseline.py
    src/models/__init__.py
    tests/test_dataset_baseline.py
  </files>
  <action>
    Implement the Naive Persistence Baseline ($\hat{P}_{t+1} = P_t$, corresponding to $\hat{r}_{t+1} = 0$) to establish the statistical benchmark that the deep learning model must outperform.

    Steps:
    1. In `src/models/baseline.py`, implement `NaivePersistenceBaseline`:
       - Methods:
         - `__init__(strategy: str = "zero_return")`:
           - `"zero_return"`: predicts $\hat{r}_{t+1} = 0.0$ (random walk martingale assumption where expected return is zero, meaning $\hat{P}_{t+1} = P_t$).
           - `"last_return"`: predicts $\hat{r}_{t+1} = r_t$ (momentum persistence).
         - `fit(X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray] = None) -> NaivePersistenceBaseline`:
           Records baseline training mean for logging; baseline requires no iterative training.
         - `predict(X: np.ndarray | torch.Tensor) -> np.ndarray`:
           Returns array of shape $(N, 1)$ with predicted returns (all $0.0$ for `"zero_return"`).
         - `evaluate(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> dict[str, float]`:
           Computes point forecast and directional metrics:
           - `rmse`: $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$
           - `mae`: $\frac{1}{N}\sum |y - \hat{y}|$
           - `directional_accuracy`: Percentage of steps where $\text{sign}(\hat{y}) == \text{sign}(y)$. (For zero return prediction, define neutral convention or evaluate against sign of actual return).
       - Export `NaivePersistenceBaseline` in `src/models/__init__.py`.
    2. In `tests/test_dataset_baseline.py`, add tests for `NaivePersistenceBaseline`:
       - Verification test: Predict returns on `data/processed/GOOGL/test.csv`, assert shape matches target length.
       - Metric verification test: Verify computed RMSE, MAE, and Directional Accuracy are finite floats.
       - Random walk invariance: Assert $\hat{P}_{t+1} = P_t$ holds identically under zero log return inversion via `reconstruct_prices_from_base`.

    AVOID: Deploying complex neural network architectures without measuring performance against the zero-return persistence benchmark (DECISION-005).
    USE: Uniform evaluation metrics (RMSE, MAE, Directional Accuracy) shared across baseline and deep learning models.
  </action>
  <verify>
    uv run pytest tests/test_dataset_baseline.py -k "baseline" -v
  </verify>
  <done>
    `NaivePersistenceBaseline` passes unit tests and outputs baseline RMSE, MAE, and Directional Accuracy on test splits.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `configs/config.yaml` exists and is loaded cleanly by `load_config()`.
- [ ] `StockSequenceDataset` generates input windows $X \in \mathbb{R}^{B \times 30 \times 24}$ and targets $y \in \mathbb{R}^{B \times 1}$ with zero lookahead leakage.
- [ ] `create_dataloaders` creates valid PyTorch DataLoaders from `data/processed/GOOGL/`.
- [ ] `NaivePersistenceBaseline` implements uniform `.predict()` and `.evaluate()` methods.
- [ ] All tests in `tests/test_dataset_baseline.py` pass via `uv run pytest tests/test_dataset_baseline.py`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.
