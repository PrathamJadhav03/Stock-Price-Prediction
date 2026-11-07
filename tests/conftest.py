"""Pytest configuration and global fixtures for fast, hermetic test execution."""

from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_network_yfinance(request):
    """Auto-mock yfinance.download so tests run offline immediately unless test manages it."""
    # If the test already patches yfinance, do nothing
    if "live_download" in request.keywords or "live_download" in request.node.name:
        yield
        return

    with patch("yfinance.download", side_effect=ConnectionError("Offline testing mode")):
        yield
