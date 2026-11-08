# Phase 4 Plan 1: FastAPI REST Serving Engine Summary

## Executive Overview

Successfully constructed and verified the enterprise-grade FastAPI REST serving layer in `src/api/` as specified in Plan 4.1. The serving engine exposes `/health`, `/predict`, and `/backtest` endpoints backed by strict Pydantic v2 schemas and an in-memory `ModelService`. Single-step predictive inference with 95% Monte Carlo Dropout confidence intervals executes in ~21 ms (well within the <100 ms NFR-03 requirement). All 9 API contract tests pass (100%), and the full repository test suite (77 tests) passes with zero regressions.

---

## Artifacts Created and Modified

| File | Status | Description |
|------|--------|-------------|
| `src/api/__init__.py` | Created | Package export for `app`, `ModelService`, and all Pydantic request/response schemas |
| `src/api/schemas.py` | Created | Pydantic v2 schemas: `HealthResponse`, `PredictRequest`, `PredictResponse`, `BacktestRequest`, `BacktestResponse` with field bounds and validations |
| `src/api/service.py` | Created | `ModelService` encapsulating in-memory PyTorch model caching, data partition loading, batched MC Dropout inference, and walk-forward backtest execution |
| `src/api/app.py` | Created | FastAPI application with lifespan model preloading, CORS middleware, OpenAPI metadata, and HTTP 200/400/404/422 status handling |
| `tests/test_api.py` | Created | Starlette `TestClient` contract suite covering `/health`, `/predict` (stochastic & deterministic), `/backtest`, validation rejections (422), and missing resources (404) |
| `tests/conftest.py` | Modified | Added Windows sandbox IOCP loopback compatibility patch so that `anyio` and `asyncio` event loops run reliably in sandboxed Windows environments |

---

## Empirical Verification Proofs

### 1. ModelService Inference Latency and Price Reconstruction
```powershell
uv run python -c "from src.api.service import ModelService; from src.api.schemas import PredictRequest; svc = ModelService(); print('Available models:', svc.get_available_models()); resp = svc.predict(PredictRequest(symbol='GOOGL', mc_samples=10)); print('Response:', resp.symbol, resp.direction, f'Price={resp.predicted_price:.2f}', f'Latency={resp.latency_ms:.2f}ms'); assert resp.latency_ms < 100.0, 'Latency exceeded 100ms'"
```
**Output:**
```
Available models: ['GOOGL']
Response: GOOGL DOWN Price=918.05 Latency=21.11ms
```
- Expected latency: < 100 ms (NFR-03)
- Achieved latency: 21.11 ms (approx. 5x faster than target threshold)

### 2. FastAPI Contract Test Suite (`tests/test_api.py`)
```powershell
uv run pytest tests/test_api.py -v
```
**Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.2.0, pluggy-1.7.0 -- D:\Projects\Stock-Price\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Projects\Stock-Price
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.5, mock-3.17.0
collecting ... collected 9 items

tests/test_api.py::test_health_endpoint PASSED                           [ 11%]
tests/test_api.py::test_predict_endpoint_success PASSED                  [ 22%]
tests/test_api.py::test_predict_endpoint_deterministic PASSED            [ 33%]
tests/test_api.py::test_predict_endpoint_unknown_symbol PASSED           [ 44%]
tests/test_api.py::test_predict_endpoint_validation_error PASSED         [ 55%]
tests/test_api.py::test_backtest_endpoint_success PASSED                 [ 66%]
tests/test_api.py::test_backtest_endpoint_invalid_strategy PASSED        [ 77%]
tests/test_api.py::test_backtest_endpoint_negative_capital PASSED        [ 88%]
tests/test_api.py::test_backtest_endpoint_unknown_symbol PASSED          [100%]

============================= 9 passed in 10.82s ==============================
```

### 3. Full Repository Regression Test Suite
```powershell
uv run pytest
```
**Output:**
```
============================= 77 passed in 22.58s =============================
```

---

## Deviations & In-Scope Engineering Fixes

1. **Batched MC Dropout Forward Pass**:
   - Sequential Python looping (`for _ in range(n_samples): model.forward(...)`) required ~70 ms on CPU.
   - Vectorized `model.forward(tensor.expand(n_samples, -1, -1))` across the batch dimension. Since PyTorch GRU and Dropout operate independently across batch items, this reduced MC inference time from ~70 ms to ~6 ms (total endpoint latency ~21 ms).
2. **Windows Sandbox Asyncio Compatibility**:
   - In restricted Windows environments where loopback TCP `127.0.0.1` is restricted (`[WinError 10013]`), standard `socket.socketpair()` deadlocks during self-pipe initialization.
   - Added an IOCP completion port patch in `tests/conftest.py` that bypasses socket self-pipes while preserving full asynchronous portal semantics.

---

## Deferred Items

None. All Phase 4 Plan 1 deliverables and must-haves are satisfied.

