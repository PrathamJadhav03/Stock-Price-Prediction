# Plan 2.1 Summary: Configuration Specification, Sequence Dataset & Naive Persistence Baseline

> **Phase**: 2  
> **Plan**: 1  
> **Status**: Complete  

## Overview
Defined the centralized YAML configuration schema, built the sliding window sequence dataset and PyTorch DataLoader generator, and established the Naive Persistence Baseline random walk benchmark.

## Tasks Completed

1. **Create YAML configuration schema and loader utility with global seed initialization**
   - Configured `configs/config.yaml` with data, model, and training parameters (seed=42, auto device resolution).
   - Implemented `load_config`, `set_seed`, and `resolve_device` in `src/models/__init__.py`.
   - **Commit**: `b33ad38`

2. **Implement time-series sliding window PyTorch Dataset and DataLoader factory**
   - Implemented `StockSequenceDataset` in `src/models/dataset.py` generating $(seq\_len, D)$ windows predicting 1-step forward log return without lookahead leakage.
   - Implemented `create_dataloaders` streaming $(B, 30, 23)$ batches from `data/processed/{symbol}/`.
   - Verified shapes and temporal alignment in `tests/test_dataset_baseline.py`.
   - **Commit**: `d1b5357`

3. **Implement Naive Persistence Baseline and benchmark evaluation engine**
   - Implemented `NaivePersistenceBaseline` in `src/models/baseline.py` predicting zero return ($\hat{P}_{t+1} = P_t$) with RMSE, MAE, and Directional Accuracy metrics.
   - Verified random walk invariance and metrics in `tests/test_dataset_baseline.py`.
   - **Verification**: `uv run pytest tests/ -v` (21/21 tests passing).
   - **Commit**: `4235094`

## Key Verification Proof
- `uv run pytest tests/test_dataset_baseline.py -v`: 4 passed.
- `uv run pytest tests/ -v`: 21 passed.
- Config loader cleanly resolved device to `cpu` with seed 42.
- Sliding window shapes verified at $(B, 30, 23)$ and targets $(B, 1)$.
