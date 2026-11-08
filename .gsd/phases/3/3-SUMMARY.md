# Plan 3.3 Summary: Backtesting CLI Runner, Artifact Serialization & Scorecard Generation

> **Phase**: 3  
> **Plan**: 3  
> **Status**: Complete  

## Overview
Implemented the end-to-end financial backtesting CLI runner in `scripts/backtest.py` that loads trained PyTorch model checkpoints, generates out-of-sample Monte Carlo Dropout forecasts, executes walk-forward simulation with basis point transaction costs, prints an executive terminal comparison scorecard against Buy-and-Hold in the console, and serializes structured JSON reports and CSV equity curves to `reports/backtest/`.

## Tasks Completed

1. **Build backtesting CLI runner with formatted terminal scorecard and artifact persistence**
   - Created `scripts/backtest.py` supporting arguments `--symbol`, `--model-path`, `--config`, `--data-dir`, `--output-dir`, `--bps`, `--threshold`, `--strategy`, `--mc-samples`, `--split`, `--initial-capital`, and `--device`.
   - Implemented `run_backtest_pipeline(...)` loading model weights from `models/gru_{symbol}.pt`, generating Monte Carlo Dropout predictions, executing `WalkForwardBacktester.simulate()`, and printing an executive scorecard comparing Strategy vs Buy & Hold.
   - Serialized report artifacts to `reports/backtest/backtest_{symbol}.json` (KPIs, forecasting metrics, parameters) and `reports/backtest/equity_curve_{symbol}.csv` (14-column step-by-step audit DataFrame).
   - **Verification**: `uv run python scripts/backtest.py --symbol GOOGL --output-dir reports/backtest` executed cleanly and generated verified scorecard.

2. **Build CLI integration tests and artifact verification**
   - Implemented `tests/test_backtest_cli.py` covering:
     - `test_backtest_pipeline_execution`: Programmatic execution verifying JSON keys and CSV column schema (416 rows).
     - `test_backtest_cli_subprocess`: Subprocess invocation verifying exit code 0 and scorecard printing.
     - `test_missing_model_checkpoint_raises_error`: Validating clear `FileNotFoundError` on missing models.
     - `test_invalid_strategy_or_split_raises_error`: Validating errors on invalid strategy modes or partitions.
   - **Verification**: `uv run pytest tests/test_backtest_cli.py -v` (4/4 passed).

## Key Verification Proof
- `uv run pytest tests/ -v`: 68 passed in 18.04s.
- `uv run python scripts/backtest.py --symbol GOOGL`:
  - Scorecard successfully printed in console.
  - `reports/backtest/backtest_GOOGL.json` generated with finite metrics.
  - `reports/backtest/equity_curve_GOOGL.csv` generated with 416 step records.

