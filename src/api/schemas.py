"""Pydantic schemas for FastAPI serving layer data contracts."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Schema for API health status check."""

    status: str = Field(default="ok", description="Operational status of the API")
    version: str = Field(default="0.1.0", description="API version")
    models_available: List[str] = Field(
        default_factory=list, description="Available trained model symbols"
    )
    device: str = Field(default="cpu", description="Compute device used for inference")

    model_config = ConfigDict(extra="ignore")


class PredictRequest(BaseModel):
    """Schema for single-step predictive inference."""

    symbol: str = Field(default="GOOGL", description="Ticker symbol")
    mc_samples: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Monte Carlo Dropout sample passes",
    )
    return_ci: bool = Field(
        default=True,
        description="Whether to compute 95% predictive confidence intervals",
    )

    model_config = ConfigDict(extra="forbid")


class PredictResponse(BaseModel):
    """Schema for predictive inference output and uncertainty estimates."""

    symbol: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Last observed Close price P_0")
    predicted_log_return: float = Field(description="Expected single-step log return")
    predicted_price: float = Field(description="Reconstructed price: P_0 * exp(r)")
    uncertainty_std: float = Field(
        description="Predictive uncertainty standard deviation"
    )
    lower_95_ci: float = Field(description="2.5th percentile log return")
    upper_95_ci: float = Field(description="97.5th percentile log return")
    lower_95_price: float = Field(description="Lower 95% price bound")
    upper_95_price: float = Field(description="Upper 95% price bound")
    direction: str = Field(description="'UP' if return >= 0 else 'DOWN'")
    timestamp: str = Field(description="ISO timestamp of inference execution")
    latency_ms: float = Field(description="Inference elapsed time in milliseconds")

    model_config = ConfigDict(extra="ignore")


class BacktestRequest(BaseModel):
    """Schema for walk-forward financial backtesting simulation."""

    symbol: str = Field(default="GOOGL", description="Stock ticker symbol")
    fee_bps: float = Field(
        default=10.0,
        ge=0.0,
        le=200.0,
        description="Transaction fee in basis points",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=0.1,
        description="Minimum absolute return to trigger trade",
    )
    strategy: str = Field(
        default="long_flat",
        pattern=r"^(long_flat|long_short)$",
        description="Strategy execution logic",
    )
    split: str = Field(
        default="test",
        pattern=r"^(test|val|all)$",
        description="Data partition to simulate",
    )
    initial_capital: float = Field(
        default=10000.0,
        gt=0.0,
        description="Initial portfolio cash",
    )

    model_config = ConfigDict(extra="forbid")


class BacktestResponse(BaseModel):
    """Schema for walk-forward financial backtesting output and KPIs."""

    symbol: str = Field(description="Stock ticker symbol")
    strategy: str = Field(description="Strategy execution logic")
    fee_bps: float = Field(description="Transaction fee in basis points")
    threshold: float = Field(description="Prediction hurdle threshold")
    split: str = Field(description="Data partition simulated")
    initial_capital: float = Field(description="Initial portfolio cash")
    summary: Dict[str, Any] = Field(
        description="Executive performance summary and metrics"
    )
    equity_curve: List[Dict[str, Any]] = Field(
        description="Compounding equity trajectory records"
    )

    model_config = ConfigDict(extra="ignore")

