---
phase: 3
plan: 3
wave: 3
depends_on:
  - 1-PLAN.md
  - 2-PLAN.md
files_modified:
  - scripts/backtest.py
  - tests/test_backtest_cli.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "scripts/backtest.py executes walk-forward financial backtesting from CLI on trained StockGRU models"
    - "CLI prints a formatted executive scorecard in console comparing Strategy vs Buy-and-Hold"
    - "Backtest results serialize to reports/backtest/backtest_{symbol}.json with full metrics and hyperparameters"
    - "Step-by-step time series serialize to reports/backtest/equity_curve_{symbol}.csv"
  artifacts:
    - "scripts/backtest.py exists and executes cleanly via uv run"
    - "tests/test_backtest_cli.py exists and passes via uv run pytest"
    - "reports/backtest/backtest_GOOGL.json is generated with verified KPIs"
    - "reports/backtest/equity_curve_GOOGL.csv is generated with step-by-step equity trajectories"
---

# Plan 3.3: Backtesting CLI Runner, Artifact Serialization & Scorecard Generation

## Objective
Implement the end-to-end financial backtesting CLI in `scripts/backtest.py` that loads trained PyTorch model checkpoints, generates out-of-sample Monte Carlo Dropout forecasts, runs the walk-forward backtester, prints an executive comparison scorecard against the Buy-and-Hold benchmark in the console, and serializes structured JSON reports and CSV equity curve time series to `reports/backtest/`.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- configs/config.yaml
- src/evaluation/metrics.py
- src/evaluation/backtester.py
- src/models/gru.py
- src/models/trainer.py
- models/gru_GOOGL.pt
- scripts/train.py

## Tasks

<task type="auto">
  <name>Build backtesting CLI runner with formatted terminal scorecard and artifact persistence</name>
  <files>
    scripts/backtest.py
  </files>
  <action>
    Create `scripts/backtest.py` integrating data loading, model inference, walk-forward simulation, console scorecard printing, and disk serialization.

    Steps:
    1. Parse CLI arguments using `argparse`:
       - `--symbol`: Stock ticker symbol (default: `GOOGL`).
       - `--model-path`: Path to PyTorch model checkpoint (default: `models/gru_{symbol}.pt`).
       - `--config`: Path to YAML configuration (default: `configs/config.yaml`).
       - `--data-dir`: Processed data directory (default: `data/processed`).
       - `--output-dir`: Target directory for serialized reports (default: `reports/backtest`).
       - `--bps`: Transaction fee in basis points (default: `10.0`).
       - `--threshold`: Prediction hurdle threshold $\tau$ (default: `0.0`).
       - `--strategy`: Trading strategy mode (`"long_flat"` or `"long_short"`, default: `"long_flat"`).
       - `--mc-samples`: Number of Monte Carlo Dropout forward passes (default: `50`).
       - `--split`: Partition to evaluate (`"test"`, `"val"`, or `"all"`, default: `"test"`).
       - `--initial-capital`: Starting portfolio capital in dollars (default: `10000.0`).
    2. Implement `run_backtest_pipeline(...)`:
       - Load configuration via `load_config(config_path)`.
       - Seed RNGs via `set_seed(config["training"]["seed"])`.
       - Load sequence dataset or dataloaders for the specified split.
       - Load model checkpoint:
         - Inspect checkpoint metadata (feature names, seq_length, input_dim).
         - Reconstruct `StockGRU` architecture and load weights with `torch.load(..., weights_only=False)`.
         - Set model to eval mode on dynamic device (`resolve_device`).
       - Generate predictions:
         - If `mc_samples > 1`: Run Monte Carlo Dropout to obtain mean prediction $\hat{r}$, standard deviation $\sigma$, and 95% confidence intervals.
         - Else: Run deterministic forward pass.
       - Extract chronological dates, base prices ($P_{t-1}$), actual target prices ($P_t$), and actual target returns ($r_t$).
       - Instantiate `WalkForwardBacktester(fee_bps=bps, threshold=threshold, strategy=strategy, initial_capital=initial_capital)`.
       - Run `.simulate(actual_returns, predicted_returns, dates=dates, base_prices=base_prices, target_prices=target_prices)`.
       - Format and print an executive terminal comparison scorecard:
         ```
         ========================================================================
         QUANTITATIVE BACKTEST SCORECARD: GOOGL (Split: TEST)
         Strategy: long_flat | Cost: 10.0 bps | Threshold: 0.0000
         ========================================================================
         Metric                      Strategy        Buy & Hold      Diff / Alpha
         ------------------------------------------------------------------------
         Cumulative Return           +18.42%         +12.10%         +6.32%
         Annualized Return (CAGR)    +12.25%         +8.15%          +4.10%
         Annualized Volatility       14.20%          21.50%          -7.30%
         Sharpe Ratio (Rf=0%)        0.86            0.38            +0.48
         Sortino Ratio               1.24            0.52            +0.72
         Max Drawdown                -11.35%         -24.80%         +13.45%
         Win Rate (Daily)            54.20%          51.10%          +3.10%
         Profit Factor               1.35            1.08            +0.27
         Total Trades / Rebalances   42              1               --
         Total Transaction Fees      0.42%           0.01%           --
         Directional Sign Hit Rate   53.40%          50.00%          +3.40%
         ========================================================================
         ```
       - Serialize disk artifacts to `output_dir`:
         - Ensure `output_dir` exists (`mkdir(parents=True, exist_ok=True)`).
         - Save summary metrics dictionary to `{output_dir}/backtest_{symbol}.json`.
         - Save full step-by-step audit DataFrame to `{output_dir}/equity_curve_{symbol}.csv`.
       - Return `BacktestResult` and filepaths.

    AVOID: Crashing if a model checkpoint has not been trained yet; provide clear error messaging and instructions to run `scripts/train.py`.
    USE: Structured JSON and standard CSV formats allowing downstream ingestion by FastAPI and Streamlit in Phase 4.
  </action>
  <verify>
    uv run python scripts/backtest.py --symbol GOOGL --output-dir reports/backtest
  </verify>
  <done>
    `scripts/backtest.py` executes walk-forward backtesting from CLI, prints terminal scorecard, and saves `reports/backtest/backtest_GOOGL.json` and `reports/backtest/equity_curve_GOOGL.csv`.
  </done>
</task>

<task type="auto">
  <name>Build CLI integration tests and artifact verification</name>
  <files>
    tests/test_backtest_cli.py
  </files>
  <action>
    Create `tests/test_backtest_cli.py` to verify command-line invocation, argument parsing, artifact persistence, and end-to-end pipeline integrity.

    Steps:
    1. CLI Subprocess / Main Runner Test:
       - Run `scripts/backtest.py` with `--symbol GOOGL --output-dir reports/backtest --bps 10.0 --threshold 0.0`.
       - Verify exit code 0.
    2. Artifact Structure Verification:
       - Load `reports/backtest/backtest_GOOGL.json`:
         - Verify JSON contains expected root keys: `symbol`, `strategy`, `fee_bps`, `threshold`, `initial_capital`, `summary`, `forecasting_metrics`.
         - Verify all numeric metrics are non-null and finite (`is_finite`).
         - Verify `sharpe_ratio`, `max_drawdown`, `cumulative_return` exist for both strategy and buy_and_hold.
       - Load `reports/backtest/equity_curve_GOOGL.csv`:
         - Verify required columns exist: `date`, `actual_return`, `predicted_return`, `signal`, `position`, `turnover`, `fee`, `strategy_daily_return`, `strategy_equity`, `buy_and_hold_equity`, `strategy_drawdown`.
         - Verify row count matches sequence dataset test sample count.
         - Verify no NaN or Inf values in equity curves.
    3. Custom Parameter Override Test:
       - Run backtester with alternative settings in temporary test directory:
         `--bps 25 --threshold 0.001 --strategy long_short --output-dir <tmp_path>`.
       - Verify new artifacts are generated in `<tmp_path>` and reflect the overridden parameters.
    4. Run full test suite:
       - Execute `uv run pytest tests/ -v` to ensure 100% test pass rate across all project tests.

    AVOID: Leaving temporary test artifacts in tracked repository directories.
    USE: `pytest`'s `tmp_path` fixture for test-generated files while preserving permanent baseline reports in `reports/backtest/`.
  </action>
  <verify>
    uv run pytest tests/test_backtest_cli.py -v
  </verify>
  <done>
    `tests/test_backtest_cli.py` passes all integration checks, and `uv run pytest tests/ -v` confirms 100% project-wide test passage with zero regressions.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `scripts/backtest.py` executes via `uv run python scripts/backtest.py --symbol GOOGL`.
- [ ] Terminal scorecard prints formatted performance metrics (Strategy vs Buy & Hold).
- [ ] `reports/backtest/backtest_GOOGL.json` is generated with complete summary metrics.
- [ ] `reports/backtest/equity_curve_GOOGL.csv` is generated with 14-column audit time series.
- [ ] `tests/test_backtest_cli.py` passes via `uv run pytest tests/test_backtest_cli.py -v`.
- [ ] Full test suite passes via `uv run pytest tests/ -v`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.
