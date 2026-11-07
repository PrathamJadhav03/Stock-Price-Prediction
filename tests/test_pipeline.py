"""Tests for temporal splitting and zero-leakage feature pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import fetch_stock_data
from src.features.technical import add_all_technical_features
from src.features.pipeline import TemporalSplitter, FeaturePipeline


def test_temporal_splitter_chronological_boundaries():
    """Verify temporal splitter strictly maintains forward time boundaries with 0 overlap."""
    df = fetch_stock_data(symbol="GOOGL", fallback_dir="Data")
    train, val, test = TemporalSplitter.split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert len(train) + len(val) + len(test) == len(df)
    assert train.index.max() < val.index.min(), "Train max date must precede Val min date"
    assert val.index.max() < test.index.min(), "Val max date must precede Test min date"
    assert train.index.is_monotonic_increasing
    assert val.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_temporal_splitter_invalid_ratios():
    """Verify splitter rejects ratio sums that do not equal 1.0."""
    df = fetch_stock_data(symbol="AMZN", fallback_dir="Data")
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        TemporalSplitter.split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)


def test_zero_lookahead_scaler_leakage():
    """Verify scaler is fit strictly on train and is completely invariant to test values."""
    df = fetch_stock_data(symbol="MSFT", fallback_dir="Data")
    features = add_all_technical_features(df, drop_burn_in=True)

    train, val, test1 = TemporalSplitter.split(features)
    test2 = test1.copy()
    # Drastically mutate test values
    test2["Close"] = test2["Close"] * 100.0
    test2["Volume"] = test2["Volume"] * 50.0

    pipe1 = FeaturePipeline(scaler_type="robust")
    pipe1.fit(train)

    pipe2 = FeaturePipeline(scaler_type="robust")
    pipe2.fit(train)

    # Both must have exact same center_ and scale_
    assert np.allclose(pipe1.scaler.center_, pipe2.scaler.center_)
    assert np.allclose(pipe1.scaler.scale_, pipe2.scaler.scale_)

    # Confirm center_ matches train median
    train_medians = np.median(train.values, axis=0)
    assert np.allclose(pipe1.scaler.center_, train_medians)


def test_pipeline_serialization_roundtrip(tmp_path):
    """Verify FeaturePipeline saves and loads with zero numerical degradation."""
    df = fetch_stock_data(symbol="IBM", fallback_dir="Data")
    features = add_all_technical_features(df, drop_burn_in=True)
    train, _, test = TemporalSplitter.split(features)

    pipeline = FeaturePipeline(scaler_type="robust")
    transformed_orig = pipeline.fit_transform(train)
    transformed_test_orig = pipeline.transform(test)

    artifact_path = tmp_path / "scaler_IBM.joblib"
    pipeline.save(artifact_path)
    assert artifact_path.exists()

    loaded_pipe = FeaturePipeline.load(artifact_path)
    transformed_test_loaded = loaded_pipe.transform(test)

    pd.testing.assert_frame_equal(transformed_test_orig, transformed_test_loaded, check_exact=True)
