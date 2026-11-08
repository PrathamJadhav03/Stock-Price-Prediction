"""End-to-end data preparation CLI pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import fetch_stock_data
from src.features.technical import add_all_technical_features
from src.features.pipeline import TemporalSplitter, FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_data")


def run_pipeline(
    symbol: str = "GOOGL",
    start_date: str = "2006-01-01",
    end_date: str = "2018-01-01",
    output_dir: str = "data/processed",
    models_dir: str = "models",
    scaler_type: str = "robust",
    cache_dir: str = "data/raw",
    fallback_dir: str = "data/raw",
) -> dict:
    """Execute end-to-end data ingestion, feature extraction, temporal split, and scaler serialization.

    Returns summary dictionary with dataset paths, row counts, and date boundaries.
    """
    clean_sym = symbol.strip().upper()
    logger.info("=== Starting Data Preparation Pipeline for %s ===", clean_sym)

    # 1. Fetch raw data
    raw_df = fetch_stock_data(
        symbol=clean_sym,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        fallback_dir=fallback_dir,
    )
    logger.info("Fetched %d raw OHLCV rows for %s (%s to %s)", len(raw_df), clean_sym, raw_df.index.min().date(), raw_df.index.max().date())

    # 2. Extract technical features and log returns
    features_df = add_all_technical_features(raw_df, drop_burn_in=True)
    nan_count = features_df.isna().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Feature engineering produced {nan_count} NaNs in {clean_sym} dataset")
    logger.info("Engineered %d features across %d rows (0 NaNs)", len(features_df.columns), len(features_df))

    # 3. Temporal split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = TemporalSplitter.split(
        features_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    logger.info(
        "Temporal Split -> Train: %d (%s to %s), Val: %d (%s to %s), Test: %d (%s to %s)",
        len(train_df),
        train_df.index.min().date(),
        train_df.index.max().date(),
        len(val_df),
        val_df.index.min().date(),
        val_df.index.max().date(),
        len(test_df),
        test_df.index.min().date(),
        test_df.index.max().date(),
    )

    # 4. Feature scaling strictly fit on train features
    # Exclude target 'log_return' and raw 'Close' from feature scaling matrix if we want separate feature scaling
    feature_cols = [c for c in features_df.columns if c not in ["log_return", "Close"]]

    pipeline = FeaturePipeline(scaler_type=scaler_type)
    train_scaled = pipeline.fit_transform(train_df[feature_cols])
    val_scaled = pipeline.transform(val_df[feature_cols])
    test_scaled = pipeline.transform(test_df[feature_cols])

    # Re-attach target and close price for modeling and financial backtesting
    for split_df, split_scaled in [(train_df, train_scaled), (val_df, val_scaled), (test_df, test_scaled)]:
        split_scaled["log_return"] = split_df["log_return"].values
        split_scaled["Close"] = split_df["Close"].values

    # 5. Save scaler artifact
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    scaler_artifact = models_path / f"scaler_{clean_sym}.joblib"
    pipeline.save(scaler_artifact)
    logger.info("Persisted %s scaler to %s", scaler_type, scaler_artifact)

    # 6. Save split datasets
    out_symbol_dir = Path(output_dir) / clean_sym
    out_symbol_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_symbol_dir / "train.csv"
    val_path = out_symbol_dir / "val.csv"
    test_path = out_symbol_dir / "test.csv"

    train_scaled.to_csv(train_path)
    val_scaled.to_csv(val_path)
    test_scaled.to_csv(test_path)
    logger.info("Persisted split datasets to %s", out_symbol_dir)

    summary = {
        "symbol": clean_sym,
        "scaler_artifact": str(scaler_artifact),
        "train_rows": len(train_scaled),
        "val_rows": len(val_scaled),
        "test_rows": len(test_scaled),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "train_range": (str(train_df.index.min().date()), str(train_df.index.max().date())),
        "val_range": (str(val_df.index.min().date()), str(val_df.index.max().date())),
        "test_range": (str(test_df.index.min().date()), str(test_df.index.max().date())),
    }
    logger.info("=== Pipeline Completed Successfully for %s ===", clean_sym)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stock price dataset preparation and leak-free scaling CLI")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Equity ticker symbol (default: GOOGL)")
    parser.add_argument("--start-date", type=str, default="2006-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2018-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Processed datasets directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for scaler artifacts")
    parser.add_argument("--scaler", type=str, default="robust", choices=["robust", "standard"], help="Scaler type")

    args = parser.parse_args()
    run_pipeline(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        scaler_type=args.scaler,
    )


if __name__ == "__main__":
    main()
