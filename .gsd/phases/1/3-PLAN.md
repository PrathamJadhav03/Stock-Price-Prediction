---
phase: 1
plan: 3
wave: 3
depends_on:
  - 2-PLAN.md
files_modified:
  - src/features/pipeline.py
  - scripts/prepare_data.py
  - tests/test_pipeline.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Temporal splitting strictly partitions time series chronologically into train, val, and test without shuffling"
    - "Feature scalers are fitted strictly on the train partition, preventing distribution leakage"
    - "Fitted scalers serialize to disk via joblib and deserialize with zero transform discrepancy"
    - "CLI script prepare_data.py executes end-to-end and outputs processed datasets ready for Phase 2 modeling"
  artifacts:
    - "src/features/pipeline.py exists with TemporalSplitter and FeaturePipeline"
    - "scripts/prepare_data.py exists and runs cleanly via uv run"
    - "tests/test_pipeline.py exists and passes via uv run pytest"
---

# Plan 1.3: Zero-Leakage Preprocessing Pipeline & Feature Artifacts

## Objective
Implement strict chronological dataset splitting and serialized feature scaling (`RobustScaler`) to guarantee zero lookahead leakage, accompanied by an end-to-end data preparation CLI script exporting ready-to-train datasets and persisted scaler artifacts.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- .gsd/phases/1/1-PLAN.md
- .gsd/phases/1/2-PLAN.md
- src/data/ingestion.py
- src/features/returns.py
- src/features/technical.py

## Tasks

<task type="auto">
  <name>Implement chronological temporal splitter and serialized leak-free scaler pipeline</name>
  <files>
    src/features/pipeline.py
    tests/test_pipeline.py
  </files>
  <action>
    Implement zero-leakage temporal data splitting and scikit-learn feature scaling with persistent `joblib` serialization.
    
    Steps:
    1. In `src/features/pipeline.py`, implement:
       - `TemporalSplitter`:
         - Signature: `split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
         - Splits data strictly in chronological order:
           - Train: first $70\%$ rows
           - Val: next $15\%$ rows
           - Test: final $15\%$ rows
         - Validates that ratios sum to $1.0$ and index is monotonically increasing.
         - Enforces $\max(\text{train.index}) < \min(\text{val.index})$ and $\max(\text{val.index}) < \min(\text{test.index})$.
       - `FeaturePipeline`:
         - Configurable scaler: default `scaler_type="robust"` (`sklearn.preprocessing.RobustScaler` for outlier tolerance), or `"standard"` (`StandardScaler`).
         - `fit(X_train: pd.DataFrame) -> FeaturePipeline`: fits scaler strictly on training features, records feature column names.
         - `transform(X: pd.DataFrame) -> np.ndarray`: transforms features using fitted parameters without re-estimating statistics.
         - `fit_transform(X_train: pd.DataFrame) -> np.ndarray`: fits on train and transforms in single call.
         - Target handling: Separates target (`log_return`) from input features; keeps an optional separate `target_scaler` if target normalization is requested, while preserving original return series for unscaled evaluation.
         - `save(filepath: str | Path)`: serializes pipeline state (`scaler`, `feature_names`, `scaler_type`) using `joblib.dump(..., compress=3)`.
         - `load(filepath: str | Path) -> FeaturePipeline`: class method reconstructing pipeline from disk artifact.
    2. In `tests/test_pipeline.py`, implement tests:
       - Chronological ordering test: Assert `train.index.max() < val.index.min() < test.index.min()`.
       - Zero-leakage verification: Assert `scaler.center_` and `scaler.scale_` match `X_train` statistics and differ from `X_test` statistics. Verify that changing `X_test` values has 0 impact on `scaler.center_`.
       - Scaler serialization test: Fit scaler on `X_train`, save to temp file, load back, assert `loaded_pipeline.transform(X_test)` is identical to `pipeline.transform(X_test)` within machine precision.
    
    AVOID: Any random shuffling (`shuffle=True`) or fitting scalers on the whole dataset prior to splitting.
    USE: Chronological indexing and strictly fitting `RobustScaler` on `train` partition only.
  </action>
  <verify>
    uv run pytest tests/test_pipeline.py -k "split or scaler or leakage" -v
  </verify>
  <done>
    `TemporalSplitter` and `FeaturePipeline` pass all tests, proving zero lookahead leakage and round-trip serialization fidelity.
  </done>
</task>

<task type="auto">
  <name>Build end-to-end data preparation CLI script and integration verification</name>
  <files>
    scripts/prepare_data.py
    tests/test_pipeline.py
  </files>
  <action>
    Create the unified pipeline script `scripts/prepare_data.py` that orchestrates data ingestion, feature generation, temporal splitting, and scaler artifact generation.
    
    Steps:
    1. Create `scripts/prepare_data.py` with CLI arguments (via `argparse`):
       - `--symbol`: Stock ticker (default: `GOOGL`).
       - `--start-date`: Start date (default: `2006-01-01`).
       - `--end-date`: End date (default: `2018-01-01`).
       - `--output-dir`: Processed data directory (default: `data/processed`).
       - `--models-dir`: Scaler / artifact directory (default: `models`).
       - `--scaler`: Scaler type (`robust` or `standard`, default: `robust`).
    2. Implement end-to-end execution flow in `scripts/prepare_data.py`:
       a. Ingest raw bars via `src.data.ingestion.fetch_stock_data`.
       b. Compute log returns and technical indicators via `src.features.technical.add_all_technical_features`.
       c. Split chronologically via `src.features.pipeline.TemporalSplitter` (70% train, 15% val, 15% test).
       d. Instantiate and fit `src.features.pipeline.FeaturePipeline` strictly on train features.
       e. Transform train, val, and test feature sets.
       f. Persist fitted scaler to `{models_dir}/scaler_{symbol}.joblib`.
       g. Persist split DataFrames / arrays to `{output_dir}/{symbol}/` (e.g. `train.csv`, `val.csv`, `test.csv` or `.parquet`).
       h. Output clear logging of execution: row counts, date ranges, feature count, verify 0 NaNs.
    3. In `tests/test_pipeline.py`, add an integration test:
       - Run `prepare_data` pipeline programmatically on `GOOGL` fallback data.
       - Verify output directory contains processed splits and `models/scaler_GOOGL.joblib`.
       - Verify processed splits load without NaN values and scaler can be reloaded to transform new data.
    
    AVOID: Hardcoded Windows or Unix paths; use `pathlib.Path`.
    USE: Modular components from `src.data` and `src.features` connected in a clean CLI workflow.
  </action>
  <verify>
    uv run python scripts/prepare_data.py --symbol GOOGL && uv run pytest tests/ -v
  </verify>
  <done>
    `scripts/prepare_data.py` executes cleanly generating processed splits and scaler artifact, and full test suite passes with 100% success rate via `uv run pytest tests/ -v`.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `TemporalSplitter` partitions data strictly chronologically with no overlap or shuffling.
- [ ] `FeaturePipeline` fits strictly on training data without lookahead leakage.
- [ ] Scaler artifact serializes to `models/scaler_{symbol}.joblib` and reloads with lossless precision.
- [ ] CLI command `uv run python scripts/prepare_data.py --symbol GOOGL` generates processed data and scaler without errors.
- [ ] Entire test suite passes via `uv run pytest tests/ -v`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.

