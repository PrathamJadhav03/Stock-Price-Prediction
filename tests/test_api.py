"""Unit and contract tests for FastAPI REST serving endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing TestClient with lifespan context execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify GET /health returns operational status and available models."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert isinstance(payload["models_available"], list)
    assert "GOOGL" in payload["models_available"]
    assert "device" in payload


def test_predict_endpoint_success(client: TestClient) -> None:
    """Verify POST /predict returns valid log returns, prices, and 95% CI within 100ms."""
    # Warm-up pass to ensure cold initialization doesn't skew latency check
    client.post(
        "/predict",
        json={"symbol": "GOOGL", "mc_samples": 5, "return_ci": True},
    )

    response = client.post(
        "/predict",
        json={"symbol": "GOOGL", "mc_samples": 25, "return_ci": True},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["symbol"] == "GOOGL"
    assert payload["current_price"] > 0.0
    assert payload["predicted_price"] > 0.0
    assert payload["lower_95_ci"] <= payload["predicted_log_return"] <= payload["upper_95_ci"]
    assert payload["lower_95_price"] <= payload["predicted_price"] <= payload["upper_95_price"]
    assert payload["direction"] in ["UP", "DOWN"]
    assert payload["uncertainty_std"] >= 0.0
    assert "timestamp" in payload
    assert payload["latency_ms"] < 100.0, f"Expected < 100ms, got {payload['latency_ms']:.2f}ms"


def test_predict_endpoint_deterministic(client: TestClient) -> None:
    """Verify POST /predict with mc_samples=1 and return_ci=False generates deterministic output."""
    response = client.post(
        "/predict",
        json={"symbol": "GOOGL", "mc_samples": 1, "return_ci": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["uncertainty_std"] == 0.0
    assert payload["lower_95_ci"] == payload["predicted_log_return"]
    assert payload["upper_95_ci"] == payload["predicted_log_return"]


def test_predict_endpoint_unknown_symbol(client: TestClient) -> None:
    """Verify POST /predict returns 404 for unknown ticker symbols."""
    response = client.post(
        "/predict",
        json={"symbol": "NONEXISTENT_TICKER"},
    )
    assert response.status_code == 404
    payload = response.json()
    assert "not found" in payload["detail"].lower()


def test_predict_endpoint_validation_error(client: TestClient) -> None:
    """Verify POST /predict rejects payloads violating Pydantic schema constraints (mc_samples > 500)."""
    response = client.post(
        "/predict",
        json={"symbol": "GOOGL", "mc_samples": 1000},
    )
    assert response.status_code == 422


def test_backtest_endpoint_success(client: TestClient) -> None:
    """Verify POST /backtest executes walk-forward simulation and returns financial KPIs."""
    response = client.post(
        "/backtest",
        json={
            "symbol": "GOOGL",
            "strategy": "long_flat",
            "fee_bps": 10.0,
            "threshold": 0.0,
            "split": "test",
            "initial_capital": 10000.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["symbol"] == "GOOGL"
    assert payload["strategy"] == "long_flat"
    assert payload["fee_bps"] == 10.0
    assert payload["threshold"] == 0.0
    assert payload["split"] == "test"
    assert payload["initial_capital"] == 10000.0

    summary = payload["summary"]
    assert "strategy_cumulative_return" in summary
    assert "buy_and_hold_cumulative_return" in summary
    assert "strategy_sharpe_ratio" in summary
    assert "strategy_max_drawdown" in summary

    equity_curve = payload["equity_curve"]
    assert isinstance(equity_curve, list)
    assert len(equity_curve) > 0
    first_record = equity_curve[0]
    assert "strategy_equity" in first_record
    assert "buy_and_hold_equity" in first_record


def test_backtest_endpoint_invalid_strategy(client: TestClient) -> None:
    """Verify POST /backtest rejects unsupported strategy options."""
    response = client.post(
        "/backtest",
        json={"strategy": "invalid_momentum"},
    )
    assert response.status_code == 422


def test_backtest_endpoint_negative_capital(client: TestClient) -> None:
    """Verify POST /backtest rejects negative starting capital."""
    response = client.post(
        "/backtest",
        json={"initial_capital": -1000.0},
    )
    assert response.status_code == 422


def test_backtest_endpoint_unknown_symbol(client: TestClient) -> None:
    """Verify POST /backtest returns 404 for missing symbol checkpoint."""
    response = client.post(
        "/backtest",
        json={"symbol": "UNKNOWN"},
    )
    assert response.status_code == 404
    payload = response.json()
    assert "not found" in payload["detail"].lower()
