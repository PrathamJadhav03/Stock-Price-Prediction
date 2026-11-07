# Phase 1 Verification Report

> **Phase**: 1 - `uv` Setup, Data Ingestion & Leak-Free Feature Engineering  
> **Verdict**: PASS  
> **Date**: 2026-11-08  

---

## Must-Haves Verification

| Must-Have | Expected | Actual | Evidence | Status |
|-----------|----------|--------|----------|--------|
| **Strict `uv` Environment** | Hermetic `pyproject.toml` and `uv.lock` | Dependencies locked, virtualenv synced | `uv run pytest` executes cleanly | PASS |
| **Resilient Ingestion** | Local cache in `data/raw/` with `Data/*.csv` fallback | Ingestion loads from cache/fallback | `tests/test_ingestion.py` passes (5/5) | PASS |
| **Log Returns Stationarity** | Predict $r_t = \ln(P_t/P_{t-1})$ with $<10^{-5}$ price inversion error | Numerical inversion error $< 10^{-5}$ on real stock data | `tests/test_features.py` passes | PASS |
| **Zero-Leakage Indicators** | RSI, MACD, Bollinger, ATR, SMA/EMA, Volatility computed backward only | 0 NaNs after burn-in drop, future price mutation has 0 effect on past features | `tests/test_features.py` passes | PASS |
| **Temporal Splitting** | Strict chronological 70/15/15 partition | $\text{train} < \text{val} < \text{test}$ date boundaries | `tests/test_pipeline.py` passes | PASS |
| **Leak-Free Scaling** | `RobustScaler` fit strictly on train split | `scaler.center_` matches train median, invariant to test set | `tests/test_pipeline.py` passes | PASS |
| **Artifact Serialization** | Scalers saved via `joblib` | Saved and reloaded with 0 numerical deviation | `models/scaler_*.joblib` generated | PASS |
| **End-to-End CLI** | `prepare_data.py` processes all equities | 4 equities (`GOOGL`, `AMZN`, `IBM`, `MSFT`) processed into `data/processed/` | `train.csv`, `val.csv`, `test.csv` exist with 0 NaNs | PASS |

---

## Empirical Verification Evidence

### 1. Test Suite Execution (`uv run pytest tests/ -v`)
```
tests/test_features.py::test_log_returns_invertibility_on_real_data PASSED
tests/test_features.py::test_log_returns_constant_prices PASSED
tests/test_features.py::test_reconstruct_prices_from_base PASSED
tests/test_features.py::test_invalid_prices_raise_error PASSED
tests/test_features.py::test_technical_indicator_bounds PASSED
tests/test_features.py::test_zero_lookahead_leakage PASSED
tests/test_features.py::test_add_all_technical_features_zero_nans PASSED
tests/test_ingestion.py::test_schema_normalization PASSED
tests/test_ingestion.py::test_fallback_ingestion_when_offline PASSED
tests/test_ingestion.py::test_caching_behavior PASSED
tests/test_ingestion.py::test_live_download_caching PASSED
tests/test_ingestion.py::test_missing_ticker_raises_file_not_found PASSED
tests/test_pipeline.py::test_temporal_splitter_chronological_boundaries PASSED
tests/test_pipeline.py::test_temporal_splitter_invalid_ratios PASSED
tests/test_pipeline.py::test_zero_lookahead_scaler_leakage PASSED
tests/test_pipeline.py::test_pipeline_serialization_roundtrip PASSED
tests/test_pipeline.py::test_prepare_data_integration PASSED
============================= 17 passed in 3.97s ==============================
```

### 2. Dataset Generation Summary
- **GOOGL**: Train 2078 rows (2006-03-16 to 2014-06-18), Val 445 rows (2014-06-19 to 2016-03-24), Test 446 rows (2016-03-28 to 2017-12-29), 25 features, 0 NaNs.
- **AMZN**: Train 2078 rows, Val 445 rows, Test 446 rows, 25 features, 0 NaNs.
- **IBM**: Train 2079 rows, Val 445 rows, Test 446 rows, 25 features, 0 NaNs.
- **MSFT**: Train 2078 rows, Val 445 rows, Test 446 rows, 25 features, 0 NaNs.

---

## Verdict: PASS
Phase 1 goals and must-haves are fully satisfied. The project is ready to proceed to Phase 2 (PyTorch GRU Architecture & Baseline Engine).
