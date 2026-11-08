"""End-to-end walk-forward financial backtesting CLI runner and report generator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation import (
    BacktestResult,
    WalkForwardBacktester,
    evaluate_forecasting_metrics,
)
from src.models import (
    ModelTrainer,
    StockGRU,
    StockSequenceDataset,
    create_dataloaders,
    load_config,
    resolve_device,
    set_seed,
)


class BacktestPipelineOutput(NamedTuple):
    """Container for backtest pipeline execution artifacts and results."""

    result: BacktestResult
    json_path: Path
    csv_path: Path


def _sanitize_metric(val: Any) -> Any:
    """Ensure numeric values are strictly finite and non-null for JSON serialization."""
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return 0.0
        if math.isinf(val):
            return 999.0 if val > 0 else -999.0
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, dict):
        return {k: _sanitize_metric(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_metric(v) for v in val]
    return val


def format_scorecard(
    symbol: str,
    split: str,
    strategy: str,
    fee_bps: float,
    threshold: float,
    summary: Dict[str, Any],
) -> str:
    """Format an executive terminal comparison scorecard (Strategy vs Buy & Hold)."""
    strat_cum = summary.get("strategy_cumulative_return", 0.0)
    bh_cum = summary.get("buy_and_hold_cumulative_return", 0.0)
    diff_cum = strat_cum - bh_cum

    strat_cagr = summary.get("strategy_annualized_return", 0.0)
    bh_cagr = summary.get("buy_and_hold_annualized_return", 0.0)
    diff_cagr = strat_cagr - bh_cagr

    strat_vol = summary.get("strategy_annualized_volatility", 0.0)
    bh_vol = summary.get("buy_and_hold_annualized_volatility", 0.0)
    diff_vol = strat_vol - bh_vol

    strat_sharpe = summary.get("strategy_sharpe_ratio", 0.0)
    bh_sharpe = summary.get("buy_and_hold_sharpe_ratio", 0.0)
    diff_sharpe = strat_sharpe - bh_sharpe

    strat_sortino = summary.get("strategy_sortino_ratio", 0.0)
    bh_sortino = summary.get("buy_and_hold_sortino_ratio", 0.0)
    diff_sortino = strat_sortino - bh_sortino

    strat_dd = summary.get("strategy_max_drawdown", 0.0)
    bh_dd = summary.get("buy_and_hold_max_drawdown", 0.0)
    diff_dd = bh_dd - strat_dd

    strat_win = summary.get("strategy_win_rate", 0.0)
    bh_win = summary.get("buy_and_hold_win_rate", 0.0)
    diff_win = strat_win - bh_win

    strat_pf = summary.get("strategy_profit_factor", 0.0)
    bh_pf = summary.get("buy_and_hold_profit_factor", 0.0)
    diff_pf = strat_pf - bh_pf

    strat_trades = summary.get("total_trades", 0)
    bh_trades = 1

    strat_fees = summary.get("total_fees_pct", 0.0)
    bh_fees = 0.00

    strat_dir = summary.get("directional_accuracy", 50.0)
    bh_dir = summary.get("benchmark_directional_accuracy", 50.0)
    diff_dir = strat_dir - bh_dir

    lines = [
        "=" * 72,
        f"QUANTITATIVE BACKTEST SCORECARD: {symbol.upper()} (Split: {split.upper()})",
        f"Strategy: {strategy} | Cost: {fee_bps:.1f} bps | Threshold: {threshold:.4f}",
        "=" * 72,
        f"{'Metric':<27} {'Strategy':<15} {'Buy & Hold':<15} {'Diff / Alpha':<13}",
        "-" * 72,
        f"{'Cumulative Return':<27} {f'{strat_cum * 100:+.2f}%':<15} {f'{bh_cum * 100:+.2f}%':<15} {f'{diff_cum * 100:+.2f}%':<13}",
        f"{'Annualized Return (CAGR)':<27} {f'{strat_cagr * 100:+.2f}%':<15} {f'{bh_cagr * 100:+.2f}%':<15} {f'{diff_cagr * 100:+.2f}%':<13}",
        f"{'Annualized Volatility':<27} {f'{strat_vol * 100:.2f}%':<15} {f'{bh_vol * 100:.2f}%':<15} {f'{diff_vol * 100:+.2f}%':<13}",
        f"{'Sharpe Ratio (Rf=0%)':<27} {f'{strat_sharpe:.2f}':<15} {f'{bh_sharpe:.2f}':<15} {f'{diff_sharpe:+.2f}':<13}",
        f"{'Sortino Ratio':<27} {f'{strat_sortino:.2f}':<15} {f'{bh_sortino:.2f}':<15} {f'{diff_sortino:+.2f}':<13}",
        f"{'Max Drawdown':<27} {f'-{strat_dd:.2f}%':<15} {f'-{bh_dd:.2f}%':<15} {f'{diff_dd:+.2f}%':<13}",
        f"{'Win Rate (Daily)':<27} {f'{strat_win:.2f}%':<15} {f'{bh_win:.2f}%':<15} {f'{diff_win:+.2f}%':<13}",
        f"{'Profit Factor':<27} {f'{strat_pf:.2f}':<15} {f'{bh_pf:.2f}':<15} {f'{diff_pf:+.2f}':<13}",
        f"{'Total Trades / Rebalances':<27} {f'{strat_trades}':<15} {f'{bh_trades}':<15} {'--':<13}",
        f"{'Total Transaction Fees':<27} {f'{strat_fees:.2f}%':<15} {f'{bh_fees:.2f}%':<15} {'--':<13}",
        f"{'Directional Sign Hit Rate':<27} {f'{strat_dir:.2f}%':<15} {f'{bh_dir:.2f}%':<15} {f'{diff_dir:+.2f}%':<13}",
        "=" * 72,
    ]
    return "\n".join(lines)


def run_backtest_pipeline(
    symbol: str = "GOOGL",
    model_path: Optional[Union[str, Path]] = None,
    config_path: Union[str, Path] = "configs/config.yaml",
    data_dir: Union[str, Path] = "data/processed",
    output_dir: Union[str, Path] = "reports/backtest",
    bps: float = 10.0,
    threshold: float = 0.0,
    strategy: str = "long_flat",
    mc_samples: int = 50,
    split: str = "test",
    initial_capital: float = 10000.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> BacktestPipelineOutput:
    """Execute complete walk-forward financial backtesting, scorecarding, and serialization.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g. 'GOOGL', 'AMZN', 'IBM', 'MSFT').
    model_path : str or Path, optional
        Path to PyTorch checkpoint. Defaults to 'models/gru_{symbol}.pt'.
    config_path : str or Path
        Path to YAML configuration file.
    data_dir : str or Path
        Directory containing processed stock CSV partitions.
    output_dir : str or Path
        Target directory for serialized reports and equity curves.
    bps : float
        Transaction fee in basis points (10.0 = 0.1%).
    threshold : float
        Prediction hurdle threshold tau for taking market positions.
    strategy : str
        Strategy mode ('long_flat' or 'long_short').
    mc_samples : int
        Number of Monte Carlo Dropout forward passes (<= 1 for deterministic).
    split : str
        Partition to backtest ('test', 'val', or 'all').
    initial_capital : float
        Starting portfolio capital in currency units.
    device : str, optional
        Target device override ('cuda', 'cpu', 'auto').
    verbose : bool
        Whether to print the terminal scorecard.

    Returns
    -------
    BacktestPipelineOutput
        Named tuple containing (result, json_path, csv_path).
    """
    clean_symbol = symbol.strip().upper()
    model_file = Path(model_path) if model_path is not None else Path("models") / f"gru_{clean_symbol}.pt"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_file}'. "
            f"Please train a model first using: uv run python scripts/train.py --symbol {clean_symbol}"
        )

    symbol_data_dir = Path(data_dir) / clean_symbol
    if not symbol_data_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found at '{symbol_data_dir}'. "
            f"Please prepare data first using: uv run python scripts/prepare_data.py --symbols {clean_symbol}"
        )

    # 1. Load configuration and seed
    config = load_config(config_path)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    target_device = resolve_device(device if device is not None else config.get("training", {}).get("device", "auto"))

    # 2. Load model checkpoint
    checkpoint = torch.load(model_file, map_location=target_device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model_config = checkpoint.get("config", {}).get("model", config.get("model", {}))

    input_dim = int(metadata.get("input_dim", model_config.get("input_dim", 23)))
    hidden_dim = int(model_config.get("hidden_dim", 64))
    num_layers = int(model_config.get("num_layers", 2))
    dropout = float(model_config.get("dropout", 0.2))
    use_layer_norm = bool(model_config.get("use_layer_norm", True))

    model = StockGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()

    # 3. Prepare dataset and dataloader for the requested split
    seq_length = int(metadata.get("seq_length", config.get("data", {}).get("seq_length", 30)))
    batch_size = int(config.get("data", {}).get("batch_size", 64))

    split_clean = split.strip().lower()
    if split_clean == "test":
        test_path = symbol_data_dir / "test.csv"
        df_split = pd.read_csv(test_path, index_col=0, parse_dates=True)
    elif split_clean == "val":
        val_path = symbol_data_dir / "val.csv"
        df_split = pd.read_csv(val_path, index_col=0, parse_dates=True)
    elif split_clean == "all":
        train_path = symbol_data_dir / "train.csv"
        val_path = symbol_data_dir / "val.csv"
        test_path = symbol_data_dir / "test.csv"
        tr_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        va_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        te_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        df_split = pd.concat([tr_df, va_df, te_df])
    else:
        raise ValueError(f"Unknown split '{split}'. Must be one of: 'test', 'val', 'all'.")

    feature_cols = [c for c in df_split.columns if c not in ["log_return", "Close"]]
    dataset = StockSequenceDataset(
        features=df_split[feature_cols],
        targets=df_split["log_return"],
        seq_length=seq_length,
        dates=df_split.index,
        prices=df_split["Close"],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # 4. Generate predictions
    trainer = ModelTrainer(model=model, config=config, device=target_device)
    mean_uncertainty = 0.0
    coverage_95 = 0.0

    if mc_samples > 1:
        y_mean, y_std, y_lower, y_upper, y_true = trainer.predict_mc_loader(
            loader, n_samples=mc_samples
        )
        predicted_returns = y_mean
        mean_uncertainty = float(np.mean(y_std))
        coverage_95 = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100.0)
    else:
        y_pred, y_true = trainer.predict_loader(loader)
        predicted_returns = y_pred

    # 5. Extract series metadata for exact price reconstruction and step dates
    n_samples = len(dataset)
    dates = [
        pd.to_datetime(dataset.dates[i + seq_length]).strftime("%Y-%m-%d")
        if dataset.dates is not None
        else str(i)
        for i in range(n_samples)
    ]
    base_prices = np.array(
        [dataset.get_metadata(i)["base_price"] for i in range(n_samples)],
        dtype=np.float64,
    )
    target_prices = np.array(
        [dataset.get_metadata(i)["target_price"] for i in range(n_samples)],
        dtype=np.float64,
    )
    actual_returns = dataset.targets[seq_length:].flatten().astype(np.float64)

    # 6. Forecasting quality evaluation
    forecasting_metrics = evaluate_forecasting_metrics(actual_returns, predicted_returns)
    if mc_samples > 1:
        forecasting_metrics["mean_uncertainty_std"] = mean_uncertainty
        forecasting_metrics["coverage_95_pct"] = coverage_95

    # 7. Walk-forward backtesting simulation
    backtester = WalkForwardBacktester(
        fee_bps=bps,
        threshold=threshold,
        strategy=strategy,
        initial_capital=initial_capital,
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    result = backtester.simulate(
        actual_returns=actual_returns,
        predicted_returns=predicted_returns,
        dates=dates,
        base_prices=base_prices,
        target_prices=target_prices,
        is_log_returns=True,
    )

    # Benchmark directional accuracy (% of actual returns that are positive)
    bh_positive_pct = float(np.mean(actual_returns > 0.0) * 100.0)
    result.summary["benchmark_directional_accuracy"] = bh_positive_pct

    # 8. Terminal Scorecard
    scorecard_str = format_scorecard(
        symbol=clean_symbol,
        split=split_clean,
        strategy=strategy,
        fee_bps=bps,
        threshold=threshold,
        summary=result.summary,
    )
    if verbose:
        print("\n" + scorecard_str + "\n")

    # 9. Serialize artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_file = out_dir / f"backtest_{clean_symbol}.json"
    csv_file = out_dir / f"equity_curve_{clean_symbol}.csv"

    # Sanitize metrics to guarantee finite numbers and valid JSON
    sanitized_summary = _sanitize_metric(result.summary)
    sanitized_forecasting = _sanitize_metric(forecasting_metrics)

    report_payload = {
        "symbol": clean_symbol,
        "strategy": strategy,
        "fee_bps": float(bps),
        "threshold": float(threshold),
        "initial_capital": float(initial_capital),
        "split": split_clean,
        "mc_samples": int(mc_samples),
        "model_path": str(model_file),
        "config_path": str(config_path),
        "summary": sanitized_summary,
        "forecasting_metrics": sanitized_forecasting,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Save 14-column step-by-step audit DataFrame
    result.audit_df.to_csv(csv_file, index=False)

    if verbose:
        print(f"Artifacts successfully saved:")
        print(f"  Scorecard & Metrics: {json_file}")
        print(f"  Step-by-step Equity: {csv_file}\n")

    return BacktestPipelineOutput(result=result, json_path=json_file, csv_path=csv_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward quantitative financial backtester on StockGRU forecasts.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="GOOGL",
        help="Stock ticker symbol to evaluate (default: GOOGL).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PyTorch model checkpoint (default: models/gru_{symbol}.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/backtest",
        help="Target directory for reports (default: reports/backtest).",
    )
    parser.add_argument(
        "--bps",
        type=float,
        default=10.0,
        help="Transaction fee in basis points (default: 10.0 bps = 0.0010 fee rate).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Prediction hurdle threshold tau (default: 0.0).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="long_flat",
        choices=["long_flat", "long_short"],
        help="Trading strategy logic (default: long_flat).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="Number of Monte Carlo Dropout forward passes (default: 50).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "all"],
        help="Data split to evaluate (default: test).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Starting portfolio capital in currency units (default: 10000.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target compute device override ('cuda', 'cpu', 'auto').",
    )

    args = parser.parse_args()

    try:
        run_backtest_pipeline(
            symbol=args.symbol,
            model_path=args.model_path,
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            bps=args.bps,
            threshold=args.threshold,
            strategy=args.strategy,
            mc_samples=args.mc_samples,
            split=args.split,
            initial_capital=args.initial_capital,
            device=args.device,
            verbose=True,
        )
    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] {fnf_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] An unexpected error occurred during backtesting: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
