# Plan 3.3 Summary: Backtesting CLI Runner, Artifact Serialization & Scorecard Generation

> **Phase**: 3  
> **Plan**: 3  
> **Status**: Complete  

## Overview
Implemented the end-to-end financial backtesting CLI runner in `scripts/backtest.py` that loads trained PyTorch model checkpoints, generates out-of-sample Monte Carlo Dropout forecasts, executes walk-forward simulation with basis point transaction costs, prints an executive terminal comparison scorecard against Buy-and-Hold in the console, and serializes structured JSON reports and CSV equity curves to `reports/backtest/`.

## Tasks Completed

1. **Build backtesting CLI runner with formatted terminal scorecard and artifact persistence**
   - Created `scripts/backtest.py` supporting CLI flags: `--symbol`, `--model-path`, `--config`, `--data-dir`, `--output-dir`, `--bps`, `--threshold`, `--strategy`, `--mc-samples`, `--split`, `--initial-capital`, and `--device`.
   - Implemented `run_backtest_pipeline(...)`:
     - Loads model weights from `models/gru_{symbol}.pt` and reconstructs `StockGRU` architecture dynamically.
     - Generates out-of-sample Monte Carlo Dropout forecasts (mean return, uncertainty std, and 95% confidence intervals).
     - Extracts exact price trajectories and chronological dates from the test dataset.
     - Evaluates point forecasting metrics (`evaluate_forecasting_metrics`).
     - Simulates event-driven execution using `WalkForwardBacktester`.
     - Formats and outputs an executive comparison scorecard against Buy-and-Hold.
     - Serializes summary JSON report to `{output_dir}/backtest_{symbol}.json` with strictly finite, sanitized metrics.
     - Serializes full 14-column audit time series to `{output_dir}/equity_curve_{symbol}.csv`.
   - **Verification**: `uv run python scripts/backtest.py --symbol GOOGL --output-dir reports/backtest` executed cleanly and generated verified scorecard and disk artifacts.

2. **Build CLI integration tests and artifact verification**
   - Implemented `tests/test_backtest_cli.py` covering:
     - `test_backtest_pipeline_execution`: Programmatic execution verifying JSON keys and CSV column schema (416 rows matching test split).
     - `test_backtest_cli_subprocess`: Subprocess invocation verifying exit code 0, scorecard printing, and parameter overrides (`--bps 25.0 --threshold 0.001 --strategy long_short`).
     - `test_baseline_artifacts_exist`: Validates that permanent baseline artifacts in `reports/backtest/` exist with non-null, finite KPIs.
     - `test_missing_model_checkpoint_raises_error`: Validates informative `FileNotFoundError` with clear training command guidance.
     - `test_invalid_strategy_or_split_raises_error`: Validates argument error boundaries.
   - **Verification**: `uv run pytest tests/test_backtest_cli.py -v` (5/5 passed).
   - **Full Test Suite**: `uv run pytest tests/ -v` (69/69 passed across entire project).

## Key Verification Proof
- `uv run pytest tests/test_backtest_cli.py -v`: 5 passed in 13.20s.
- `uv run pytest tests/ -v`: 69 passed in 21.38s (100% pass rate).
- `uv run python scripts/backtest.py --symbol GOOGL`:
  - Scorecard successfully printed in console.
  - `reports/backtest/backtest_GOOGL.json` generated with 38 finite summary KPIs and forecasting metrics.
  - `reports/backtest/equity_curve_GOOGL.csv` generated with 416 step records and 14 tracking columns.
