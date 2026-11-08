---
phase: 4
plan: 1
wave: 1
depends_on: []
files_modified:
  - src/api/__init__.py
  - src/api/schemas.py
  - src/api/service.py
  - src/api/app.py
  - tests/test_api.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "FastAPI application exposes /health, /predict, and /backtest with strict Pydantic v2 schemas"
    - "ModelService maintains an in-memory cache of PyTorch checkpoints and preloads available models at startup via lifespan"
    - "POST /predict generates single-step log return forecasts and reconstructed price predictions with Monte Carlo Dropout 95% confidence intervals within 100 ms"
    - "POST /backtest simulates step-by-step walk-forward execution and returns complete financial KPIs and compounding equity trajectories"
    - "API returns 404 for missing equity models or datasets and 422 for schema validation violations"
  artifacts:
    - "src/api/__init__.py exists and exports ModelService, app, and schemas"
    - "src/api/schemas.py exists with HealthResponse, PredictRequest, PredictResponse, BacktestRequest, and BacktestResponse"
    - "src/api/service.py exists with ModelService implementing cached inference and backtest execution"
    - "src/api/app.py exists with FastAPI app, CORS middleware, lifespan model preloading, and REST route handlers"
    - "tests/test_api.py exists and passes 100% via uv run pytest tests/test_api.py"
---

# Plan 4.1: FastAPI REST Serving Engine

## Objective
Construct a high-performance, enterprise-grade FastAPI REST serving layer in `src/api/` that exposes `/health`, `/predict`, and `/backtest` endpoints. The service encapsulates PyTorch StockGRU inference and walk-forward backtesting, caches model checkpoints in memory to eliminate repeated disk I/O, generates single-step return forecasts with 95% Monte Carlo Dropout confidence intervals under 100 ms (NFR-03), and provides full test coverage using `starlette.testclient.TestClient`.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- src/models/gru.py
- src/models/trainer.py
- src/evaluation/backtester.py
- scripts/backtest.py
- configs/config.yaml

## Tasks

<task type="auto">
  <name>Implement Pydantic Request/Response Schemas and In-Memory Model Service Layer</name>
  <files>
    src/api/__init__.py
    src/api/schemas.py
    src/api/service.py
  </files>
  <action>
    Create `src/api/schemas.py`, `src/api/service.py`, and `src/api/__init__.py` providing strict data contracts and high-performance inference/backtesting service orchestration.

    Steps:
    1. In `src/api/schemas.py`, define Pydantic v2 schemas:
       - `HealthResponse`:
         - `status`: str (e.g. "ok")
         - `version`: str (e.g. "0.1.0")
         - `models_available`: List[str] (e.g. ["GOOGL"])
         - `device`: str (e.g. "cpu" or "cuda")
       - `PredictRequest`:
         - `symbol`: str = Field(default="GOOGL", description="Ticker symbol")
         - `mc_samples`: int = Field(default=50, ge=1, le=500, description="Monte Carlo Dropout sample passes")
         - `return_ci`: bool = Field(default=True, description="Whether to compute 95% predictive confidence intervals")
       - `PredictResponse`:
         - `symbol`: str
         - `current_price`: float = Field(description="Last observed Close price P_0")
         - `predicted_log_return`: float = Field(description="Expected single-step log return")
         - `predicted_price`: float = Field(description="Reconstructed price: P_0 * exp(r)")
         - `uncertainty_std`: float = Field(description="Predictive uncertainty standard deviation")
         - `lower_95_ci`: float = Field(description="2.5th percentile log return")
         - `upper_95_ci`: float = Field(description="97.5th percentile log return")
         - `lower_95_price`: float = Field(description="Lower 95% price bound")
         - `upper_95_price`: float = Field(description="Upper 95% price bound")
         - `direction`: str = Field(description="'UP' if return >= 0 else 'DOWN'")
         - `timestamp`: str = Field(description="ISO timestamp of inference execution")
         - `latency_ms`: float = Field(description="Inference elapsed time in milliseconds")
       - `BacktestRequest`:
         - `symbol`: str = Field(default="GOOGL")
         - `fee_bps`: float = Field(default=10.0, ge=0.0, le=200.0, description="Transaction fee in basis points")
         - `threshold`: float = Field(default=0.0, ge=0.0, le=0.1, description="Minimum absolute return to trigger trade")
         - `strategy`: str = Field(default="long_flat", pattern="^(long_flat|long_short)$")
         - `split`: str = Field(default="test", pattern="^(test|val|all)$")
         - `initial_capital`: float = Field(default=10000.0, gt=0.0)
       - `BacktestResponse`:
         - `symbol`: str
         - `strategy`: str
         - `fee_bps`: float
         - `threshold`: float
         - `split`: str
         - `initial_capital`: float
         - `summary`: Dict[str, Any]
         - `equity_curve`: List[Dict[str, Any]]
    2. In `src/api/service.py`, build `ModelService`:
       - Class initialization `__init__(models_dir="models", data_dir="data/processed", config_path="configs/config.yaml")`:
         - Store paths as `Path` objects.
         - Initialize internal model cache `_model_cache: Dict[str, Tuple[StockGRU, Dict[str, Any], int]]`.
         - Resolve compute device via `resolve_device()`.
       - Method `get_available_models() -> List[str]`:
         - Scan `models_dir` for files matching `gru_*.pt`. Extract uppercase symbol names.
       - Method `load_model(symbol: str) -> Tuple[StockGRU, Dict[str, Any], int]`:
         - If `symbol` in cache, return immediately (zero disk read).
         - Check if checkpoint file exists at `models/gru_{symbol}.pt`. If missing, raise `FileNotFoundError(f"Model checkpoint for '{symbol}' not found")`.
         - Load state dict, extract metadata (`input_dim`, `hidden_dim`, `num_layers`, `dropout`, `use_layer_norm`, `seq_length`).
         - Instantiate `StockGRU`, load state dict, move to device, set to eval mode.
         - Cache and return `(model, metadata, seq_length)`.
       - Method `predict(request: PredictRequest) -> PredictResponse`:
         - Record start timestamp via `time.perf_counter()`.
         - Normalize symbol `clean_symbol = request.symbol.strip().upper()`.
         - Retrieve model, metadata, seq_length via `load_model(clean_symbol)`.
         - Load feature partition for symbol (e.g. `data/processed/{clean_symbol}/test.csv` or concatenate full sequence if needed). If data directory or CSV is missing, raise `FileNotFoundError`.
         - Extract feature columns `[c for c in df.columns if c not in ["log_return", "Close"]]`.
         - Take the last `seq_length` rows as input window.
         - Construct input tensor of shape `(1, seq_length, input_dim)` on target device.
         - Obtain `current_price = float(df["Close"].iloc[-1])`.
         - If `request.mc_samples > 1` and `request.return_ci`:
           - Call `model.predict_mc(tensor, n_samples=request.mc_samples)`.
           - Unpack `mean, std, lower_95, upper_95`.
           - Extract scalar floats: `pred_ret = float(mean.item())`, `unc_std = float(std.item())`, `lower_ci = float(lower_95.item())`, `upper_ci = float(upper_95.item())`.
         - Else:
           - Call deterministic `model.predict(tensor)`.
           - `pred_ret = float(pred.item())`, `unc_std = 0.0`, `lower_ci = pred_ret`, `upper_ci = pred_ret`.
         - Compute reconstructed price trajectories:
           - `pred_price = current_price * math.exp(pred_ret)`
           - `lower_price = current_price * math.exp(lower_ci)`
           - `upper_price = current_price * math.exp(upper_ci)`
         - Determine direction: `"UP"` if `pred_ret >= 0` else `"DOWN"`.
         - Calculate `latency_ms = (time.perf_counter() - start_time) * 1000.0`.
         - Return validated `PredictResponse`.
       - Method `run_backtest(request: BacktestRequest) -> BacktestResponse`:
         - Normalize symbol `clean_symbol = request.symbol.strip().upper()`.
         - Ensure model and data partition exist (raise `FileNotFoundError` if missing).
         - Invoke `run_backtest_pipeline(symbol=clean_symbol, bps=request.fee_bps, threshold=request.threshold, strategy=request.strategy, split=request.split, initial_capital=request.initial_capital, verbose=False)`.
         - Convert result summary and equity curve DataFrame into list of record dictionaries.
         - Return validated `BacktestResponse`.
    3. In `src/api/__init__.py`, export `ModelService`, `HealthResponse`, `PredictRequest`, `PredictResponse`, `BacktestRequest`, and `BacktestResponse`.

    AVOID: Reloading `torch.load` from disk on every `/predict` request, which violates NFR-03 latency (<100ms).
    USE: In-memory dictionary caching of instantiated PyTorch models and metadata.
  </action>
  <verify>
    uv run python -c "from src.api.service import ModelService; from src.api.schemas import PredictRequest; svc = ModelService(); print('Available models:', svc.get_available_models()); resp = svc.predict(PredictRequest(symbol='GOOGL', mc_samples=10)); print('Response:', resp.symbol, resp.direction, f'Price=${resp.predicted_price:.2f}', f'Latency={resp.latency_ms:.2f}ms'); assert resp.latency_ms < 100.0, 'Latency exceeded 100ms'"
  </verify>
  <done>
    `ModelService` loads cached model, runs Monte Carlo Dropout forward passes, enforces <100ms latency, and returns fully validated Pydantic `PredictResponse`.
  </done>
</task>

<task type="auto">
  <name>Construct FastAPI App with Lifespan Preloading, CORS, REST Endpoints, and Contract Tests</name>
  <files>
    src/api/app.py
    tests/test_api.py
  </files>
  <action>
    Create `src/api/app.py` and `tests/test_api.py` exposing REST endpoints with lifespan initialization, CORS middleware, error status handlers, and comprehensive TestClient integration tests.

    Steps:
    1. In `src/api/app.py`:
       - Define lifespan async context manager:
         ```python
         @asynccontextmanager
         async def lifespan(app: FastAPI):
             service = ModelService()
             # Preload all discovered models into memory cache
             for sym in service.get_available_models():
                 try:
                     service.load_model(sym)
                 except Exception:
                     pass
             app.state.service = service
             yield
             # Clean up if needed
         ```
       - Instantiate `app = FastAPI(title="Stock Price Forecasting & Backtesting API", version="0.1.0", lifespan=lifespan)`.
       - Add `CORSMiddleware`:
         - `allow_origins=["*"]`
         - `allow_credentials=True`
         - `allow_methods=["*"]`
         - `allow_headers=["*"]`
       - Route `GET /health` -> `HealthResponse`:
         - Returns status "ok", version "0.1.0", `models_available=service.get_available_models()`, `device=str(service.device)`.
       - Route `POST /predict` -> `PredictResponse`:
         - Parameter: `request: PredictRequest`.
         - Call `app.state.service.predict(request)`.
         - Catch `FileNotFoundError` as `HTTPException(status_code=404, detail=str(e))`.
         - Catch `ValueError` as `HTTPException(status_code=400, detail=str(e))`.
       - Route `POST /backtest` -> `BacktestResponse`:
         - Parameter: `request: BacktestRequest`.
         - Call `app.state.service.run_backtest(request)`.
         - Catch `FileNotFoundError` as `HTTPException(status_code=404, detail=str(e))`.
         - Catch `ValueError` as `HTTPException(status_code=400, detail=str(e))`.
       - Add an entrypoint block `if __name__ == "__main__": uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)`.
    2. In `tests/test_api.py`:
       - Use `from fastapi.testclient import TestClient` and `from src.api.app import app`.
       - Setup fixture or context manager: `client = TestClient(app)`.
       - Test `test_health_endpoint`:
         - Call `client.get("/health")`.
         - Assert status_code == 200.
         - Assert payload["status"] == "ok".
         - Assert payload["version"] == "0.1.0".
         - Assert "GOOGL" in payload["models_available"].
       - Test `test_predict_endpoint_success`:
         - Call `client.post("/predict", json={"symbol": "GOOGL", "mc_samples": 25, "return_ci": True})`.
         - Assert status_code == 200.
         - Assert payload["symbol"] == "GOOGL".
         - Assert payload["current_price"] > 0.
         - Assert payload["predicted_price"] > 0.
         - Assert payload["lower_95_ci"] <= payload["predicted_log_return"] <= payload["upper_95_ci"].
         - Assert payload["lower_95_price"] <= payload["predicted_price"] <= payload["upper_95_price"].
         - Assert payload["direction"] in ["UP", "DOWN"].
         - Assert payload["latency_ms"] < 100.0.
       - Test `test_predict_endpoint_unknown_symbol`:
         - Call `client.post("/predict", json={"symbol": "NONEXISTENT_TICKER"})`.
         - Assert status_code == 404.
         - Assert "not found" in payload["detail"].lower().
       - Test `test_predict_endpoint_validation_error`:
         - Call `client.post("/predict", json={"symbol": "GOOGL", "mc_samples": 1000})`. # exceeds le=500
         - Assert status_code == 422.
       - Test `test_backtest_endpoint_success`:
         - Call `client.post("/backtest", json={"symbol": "GOOGL", "strategy": "long_flat", "fee_bps": 10.0, "threshold": 0.0, "split": "test", "initial_capital": 10000.0})`.
         - Assert status_code == 200.
         - Assert payload["symbol"] == "GOOGL".
         - Assert "strategy_cumulative_return" in payload["summary"].
         - Assert "buy_and_hold_cumulative_return" in payload["summary"].
         - Assert "strategy_sharpe_ratio" in payload["summary"].
         - Assert len(payload["equity_curve"]) > 0.
         - Assert "strategy_equity" in payload["equity_curve"][0].
       - Test `test_backtest_endpoint_invalid_strategy`:
         - Call `client.post("/backtest", json={"strategy": "invalid_momentum"})`.
         - Assert status_code == 422.
       - Test `test_backtest_endpoint_negative_capital`:
         - Call `client.post("/backtest", json={"initial_capital": -1000.0})`.
         - Assert status_code == 422.
       - Test `test_backtest_endpoint_unknown_symbol`:
         - Call `client.post("/backtest", json={"symbol": "UNKNOWN"})`.
         - Assert status_code == 404.

    AVOID: Running uvicorn subprocess in unit tests.
    USE: Starlette's synchronous `TestClient(app)` which runs the lifespan context and handles in-memory ASGI dispatch.
  </action>
  <verify>
    uv run pytest tests/test_api.py -v
  </verify>
  <done>
    `tests/test_api.py` passes 100% across all 8 endpoint contract tests with proper validation error handling and sub-100ms latency.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `src/api/schemas.py` defines valid Pydantic models with bounds checking
- [ ] `src/api/service.py` provides in-memory model caching and fast inference
- [ ] `src/api/app.py` exposes `/health`, `/predict`, `/backtest` with CORS and OpenAPI docs
- [ ] `tests/test_api.py` validates all endpoint status codes (200, 404, 422)
- [ ] `POST /predict` executes MC Dropout inference within 100ms (NFR-03)

## Success Criteria
- [ ] All tasks verified passing
- [ ] Must-haves confirmed
- [ ] No regressions across existing test suite (`uv run pytest`)

