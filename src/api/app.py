"""FastAPI REST application exposing health, predictive forecasting, and backtesting endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.service import ModelService


def _get_service(request: Request) -> ModelService:
    """Retrieve or lazily initialize ModelService from application state."""
    if not hasattr(request.app.state, "service") or request.app.state.service is None:
        request.app.state.service = ModelService()
    return request.app.state.service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager that initializes and preloads models into ModelService."""
    service = ModelService()
    for sym in service.get_available_models():
        try:
            service.load_model(sym)
        except Exception:
            pass
    app.state.service = service
    yield


app = FastAPI(
    title="Stock Price Forecasting & Backtesting API",
    description="High-performance serving engine for PyTorch GRU stock price forecasting and event-driven backtesting.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    tags=["System"],
)
async def health(request: Request) -> HealthResponse:
    """Return API operational status, version, available trained models, and compute device."""
    service = _get_service(request)
    return HealthResponse(
        status="ok",
        version="0.1.0",
        models_available=service.get_available_models(),
        device=str(service.device),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single-Step Return & Price Forecast with Uncertainty",
    tags=["Inference"],
)
async def predict(
    payload: PredictRequest,
    request: Request,
) -> PredictResponse:
    """Generate single-step log return forecast and reconstructed price with 95% MC Dropout CI."""
    service = _get_service(request)
    try:
        return service.predict(payload)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@app.post(
    "/backtest",
    response_model=BacktestResponse,
    summary="Walk-Forward Financial Backtest Simulation",
    tags=["Backtesting"],
)
async def backtest(
    payload: BacktestRequest,
    request: Request,
) -> BacktestResponse:
    """Simulate walk-forward strategy execution with fee friction and return financial KPIs."""
    service = _get_service(request)
    try:
        return service.run_backtest(payload)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

