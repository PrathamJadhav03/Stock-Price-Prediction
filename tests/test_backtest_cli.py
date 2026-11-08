"""Integration tests for backtesting CLI, argument parsing, and artifact persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from scripts.backtest import run_backtest_pipeline


def test_backtest_pipeline_execution(tmp_path: Path):
    """Verify programmatic execution of backtest pipeline on real test split."""
    output_dir = tmp_path / "reports" / "backtest"

    output = run_backtest_pipeline(
        symbol="GOOGL",
        model_path="models/gru_GOOGL.pt",
        config_path="configs/config.yaml",
        data_dir="data/processed",
        output_dir=str(output_dir),
        bps=10.0,
        threshold=0.0,
        strategy="long_flat",
        mc_samples=5,  # small sample count for fast test execution
        split="test",
        initial_capital=10000.0,
        device="cpu",
        verbose=False,
    )

    # 1. Output file existence
    json_path = output.json_path
    csv_path = output.csv_path
    assert json_path.exists()
    assert csv_path.exists()
    assert json_path.stat().st_size > 0
    assert csv_path.stat().st_size > 0

    # 2. JSON artifact content validation
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    expected_root_keys = [
        "symbol",
        "strategy",
        "fee_bps",
        "threshold",
        "initial_capital",
        "split",
        "mc_samples",
        "summary",
        "forecasting_metrics",
    ]
    for key in expected_root_keys:
        assert key in payload, f"Missing root key: {key}"

    assert payload["symbol"] == "GOOGL"
    assert payload["strategy"] == "long_flat"
    assert payload["fee_bps"] == 10.0
    assert payload["initial_capital"] == 10000.0

    # Summary metrics sanity
    summary = payload["summary"]
    required_metrics = [
        "strategy_cumulative_return",
        "strategy_annualized_return",
        "strategy_sharpe_ratio",
        "strategy_sortino_ratio",
        "strategy_max_drawdown",
        "buy_and_hold_cumulative_return",
        "buy_and_hold_annualized_return",
        "buy_and_hold_sharpe_ratio",
        "buy_and_hold_max_drawdown",
        "directional_accuracy",
        "total_trades",
    ]
    for metric in required_metrics:
        assert metric in summary, f"Missing metric in summary: {metric}"
        val = summary[metric]
        if isinstance(val, (int, float)):
            assert math.isfinite(val), f"Metric {metric} is not finite: {val}"

    # Forecasting metrics sanity
    forecast_metrics = payload["forecasting_metrics"]
    for m in ["rmse", "mae", "mape", "directional_accuracy"]:
        assert m in forecast_metrics
        assert math.isfinite(forecast_metrics[m])

    # 3. CSV equity curve validation
    df = pd.read_csv(csv_path)
    required_cols = [
        "date",
        "actual_return",
        "predicted_return",
        "signal",
        "position",
        "turnover",
        "fee",
        "strategy_daily_return",
        "strategy_equity",
        "buy_and_hold_equity",
        "strategy_drawdown",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column in CSV: {col}"

    assert len(df) == 416  # 446 rows in test split - 30 lookback window
    assert not df["strategy_equity"].isna().any()
    assert not df["buy_and_hold_equity"].isna().any()
    assert df["strategy_equity"].iloc[0] > 0.0


def test_backtest_cli_subprocess(tmp_path: Path):
    """Verify CLI runner script works via standard Python invocation."""
    output_dir = tmp_path / "cli_reports"

    cmd = [
        sys.executable,
        "scripts/backtest.py",
        "--symbol",
        "GOOGL",
        "--model-path",
        "models/gru_GOOGL.pt",
        "--output-dir",
        str(output_dir),
        "--bps",
        "25.0",
        "--threshold",
        "0.001",
        "--strategy",
        "long_short",
        "--mc-samples",
        "2",
        "--split",
        "test",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"CLI command failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "QUANTITATIVE BACKTEST SCORECARD" in result.stdout

    json_file = output_dir / "backtest_GOOGL.json"
    csv_file = output_dir / "equity_curve_GOOGL.csv"
    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["fee_bps"] == 25.0
    assert data["threshold"] == 0.001
    assert data["strategy"] == "long_short"

    df = pd.read_csv(csv_file)
    assert len(df) == 416
    assert not df["strategy_equity"].isna().any()


def test_baseline_artifacts_exist():
    """Verify that reports/backtest contains valid baseline artifacts."""
    reports_dir = Path("reports/backtest")
    json_path = reports_dir / "backtest_GOOGL.json"
    csv_path = reports_dir / "equity_curve_GOOGL.csv"

    # If not already generated, run backtest CLI to produce baseline
    if not (json_path.exists() and csv_path.exists()):
        run_backtest_pipeline(
            symbol="GOOGL",
            output_dir=str(reports_dir),
            mc_samples=5,
            verbose=False,
        )

    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    for k in ["symbol", "strategy", "fee_bps", "threshold", "initial_capital", "summary", "forecasting_metrics"]:
        assert k in report

    summary = report["summary"]
    for k in ["sharpe_ratio", "max_drawdown", "cumulative_return", "buy_and_hold_sharpe_ratio", "buy_and_hold_max_drawdown", "buy_and_hold_cumulative_return"]:
        assert k in summary
        assert math.isfinite(summary[k])

    df = pd.read_csv(csv_path)
    for col in ["date", "actual_return", "predicted_return", "signal", "position", "turnover", "fee", "strategy_daily_return", "strategy_equity", "buy_and_hold_equity", "strategy_drawdown"]:
        assert col in df.columns
    assert not df["strategy_equity"].isna().any()
    assert not df["buy_and_hold_equity"].isna().any()


def test_missing_model_checkpoint_raises_error(tmp_path: Path):
    """Verify informative FileNotFoundError when model checkpoint is missing."""
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        run_backtest_pipeline(
            symbol="NONEXISTENT",
            model_path=str(tmp_path / "missing.pt"),
            output_dir=str(tmp_path),
        )


def test_invalid_strategy_or_split_raises_error():
    """Verify validation errors on invalid strategy modes or dataset partitions."""
    with pytest.raises(ValueError, match="split"):
        run_backtest_pipeline(
            symbol="GOOGL",
            split="invalid_split",
        )

    with pytest.raises(ValueError, match="strategy"):
        run_backtest_pipeline(
            symbol="GOOGL",
            strategy="invalid_strat",
        )
