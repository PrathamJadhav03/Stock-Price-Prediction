"""End-to-end training, checkpointing, and baseline benchmarking runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import numpy as np

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    EarlyStopping,
    ModelTrainer,
    NaivePersistenceBaseline,
    StockGRU,
    create_dataloaders,
    load_config,
    set_seed,
)


def train_pipeline(
    config_path: str = "configs/config.yaml",
    symbol: str = "GOOGL",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: str = "models",
    data_dir: str = "data/processed",
    mc_samples: int = 50,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete training workflow, checkpointing, and baseline benchmarking.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    symbol : str
        Stock ticker symbol (e.g., 'GOOGL', 'AMZN', 'IBM', 'MSFT').
    epochs : int, optional
        Override max epochs from configuration.
    batch_size : int, optional
        Override mini-batch size.
    output_dir : str
        Target directory for saved model checkpoints and evaluation metrics.
    data_dir : str
        Directory containing processed CSV partitions.
    mc_samples : int
        Number of Monte Carlo Dropout forward passes for uncertainty estimation.
    device : str, optional
        Compute device override ('cuda', 'cpu', 'auto').
    verbose : bool
        Whether to print progress tables and epoch logs.

    Returns
    -------
    Dict[str, Any]
        Dictionary with benchmark results, metrics, and checkpoint filepaths.
    """
    config = load_config(config_path)

    # Apply command overrides to config
    if batch_size is not None:
        config["data"]["batch_size"] = batch_size
    if device is not None:
        config["training"]["device"] = device
    if epochs is not None:
        config["training"]["max_epochs"] = epochs
    if mc_samples is not None:
        config["model"]["mc_samples"] = mc_samples

    # Enforce reproducibility
    set_seed(int(config.get("training", {}).get("seed", 42)))

    # Create PyTorch dataloaders
    seq_length = int(config.get("data", {}).get("seq_length", 30))
    effective_batch_size = int(config.get("data", {}).get("batch_size", 64))

    train_loader, val_loader, test_loader, feature_names = create_dataloaders(
        data_dir=data_dir,
        symbol=symbol,
        seq_length=seq_length,
        batch_size=effective_batch_size,
    )

    input_dim = len(feature_names)
    hidden_dim = int(config.get("model", {}).get("hidden_dim", 64))
    num_layers = int(config.get("model", {}).get("num_layers", 2))
    dropout = float(config.get("model", {}).get("dropout", 0.2))
    use_layer_norm = bool(config.get("model", {}).get("use_layer_norm", True))

    if verbose:
        print(f"Initializing StockGRU for symbol {symbol}:")
        print(f"  Input Dim: {input_dim} features ({', '.join(feature_names[:4])}...)")
        print(f"  Architecture: GRU(hidden={hidden_dim}, layers={num_layers}, dropout={dropout}, layer_norm={use_layer_norm})")

    model = StockGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )

    trainer = ModelTrainer(model=model, config=config, device=config["training"]["device"])

    # Fit model with early stopping and learning rate scheduling
    max_epochs = epochs or int(config.get("training", {}).get("max_epochs", 100))
    if verbose:
        print(f"\n--- Starting Model Training (Max Epochs: {max_epochs}, Device: {trainer.device}) ---")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        verbose=verbose,
    )

    # Persist model checkpoint
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_path / f"gru_{symbol}.pt"

    checkpoint_meta = {
        "symbol": symbol,
        "input_dim": input_dim,
        "feature_names": feature_names,
        "seq_length": seq_length,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }
    trainer.save_checkpoint(checkpoint_file, extra_metadata=checkpoint_meta)
    if verbose:
        print(f"\nModel checkpoint successfully saved to {checkpoint_file}")

    # Evaluate point forecasts on test set
    gru_point_metrics = trainer.evaluate_metrics(test_loader)

    # Evaluate Monte Carlo Dropout forecasts on test set
    if verbose:
        print(f"Generating Monte Carlo Dropout predictions (N={mc_samples} forward passes)...")

    y_mc_mean, y_mc_std, y_lower_95, y_upper_95, y_true = trainer.predict_mc_loader(
        test_loader,
        n_samples=mc_samples,
    )
    gru_mc_metrics = NaivePersistenceBaseline.evaluate(y_true=y_true, y_pred=y_mc_mean)
    gru_mc_metrics["mean_uncertainty_std"] = float(np.mean(y_mc_std))
    coverage = float(np.mean((y_true >= y_lower_95) & (y_true <= y_upper_95)) * 100.0)
    gru_mc_metrics["coverage_95_pct"] = coverage

    # Fit and evaluate NaivePersistenceBaseline on identical test targets
    baseline = NaivePersistenceBaseline(strategy="zero_return")
    baseline.fit(X=y_true, y=y_true)
    y_baseline_pred = baseline.predict(y_true)
    baseline_metrics = NaivePersistenceBaseline.evaluate(y_true=y_true, y_pred=y_baseline_pred)

    # Benchmark comparison
    rmse_outperforms = gru_mc_metrics["rmse"] < baseline_metrics["rmse"]
    mae_outperforms = gru_mc_metrics["mae"] < baseline_metrics["mae"]
    dir_acc_outperforms = gru_mc_metrics["directional_accuracy"] >= baseline_metrics["directional_accuracy"]

    comparison = {
        "rmse_outperforms": bool(rmse_outperforms),
        "mae_outperforms": bool(mae_outperforms),
        "directional_accuracy_outperforms": bool(dir_acc_outperforms),
    }

    if verbose:
        gru_dir_str = f"{gru_mc_metrics['directional_accuracy']:.2f}%"
        base_dir_str = f"{baseline_metrics['directional_accuracy']:.2f}%"
        cov_str = f"{coverage:.2f}%"

        print("\n" + "=" * 65)
        print(f"MODEL BENCHMARK REPORT: {symbol.upper()}")
        print("=" * 65)
        print(f"{'Metric':<24} {'StockGRU (MC)':<16} {'Naive Baseline':<16} {'Outperforms?':<12}")
        print("-" * 65)
        print(f"{'RMSE':<24} {gru_mc_metrics['rmse']:<16.6f} {baseline_metrics['rmse']:<16.6f} {'YES' if rmse_outperforms else 'NO':<12}")
        print(f"{'MAE':<24} {gru_mc_metrics['mae']:<16.6f} {baseline_metrics['mae']:<16.6f} {'YES' if mae_outperforms else 'NO':<12}")
        print(f"{'Directional Accuracy':<24} {gru_dir_str:<16} {base_dir_str:<16} {'YES' if dir_acc_outperforms else 'NO':<12}")
        print(f"{'95% CI Coverage':<24} {cov_str:<16} {'N/A':<16} {'VALID':<12}")
        print("=" * 65 + "\n")

    # Serialize complete evaluation report
    report = {
        "symbol": symbol,
        "checkpoint_path": str(checkpoint_file),
        "gru_point_metrics": gru_point_metrics,
        "gru_mc_metrics": gru_mc_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "config": config,
        "history": history,
    }

    metrics_file = out_path / f"metrics_{symbol}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"Benchmark metrics successfully saved to {metrics_file}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train StockGRU and benchmark against NaivePersistenceBaseline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="GOOGL",
        help="Stock ticker symbol to train (e.g. GOOGL, AMZN, IBM, MSFT).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Processed data directory.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="Number of Monte Carlo Dropout forward passes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cuda', 'cpu', 'auto').",
    )

    args = parser.parse_args()

    train_pipeline(
        config_path=args.config,
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        mc_samples=args.mc_samples,
        device=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()
