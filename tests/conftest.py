"""Pytest configuration and global fixtures for fast, hermetic test execution."""

from __future__ import annotations

import sys
from unittest.mock import patch
import pytest

# Windows sandbox environment compatibility patch:
# In restricted Windows sandbox environments, loopback TCP socket connection (127.0.0.1)
# is blocked with [WinError 10013], which causes standard asyncio BaseProactorEventLoop._make_self_pipe
# (calling socket.socketpair) to hang indefinitely. We patch the loop self-pipe using native IOCP.
if sys.platform == "win32":
    import asyncio.proactor_events
    import asyncio.windows_events
    import signal
    import _overlapped

    class _DummySock:
        def fileno(self) -> int:
            return -1

        def close(self) -> None:
            pass

        def setblocking(self, b: bool) -> None:
            pass

        def send(self, b: bytes) -> int:
            return len(b)

    signal.set_wakeup_fd = lambda fd: -1

    def _dummy_make_self_pipe(self):
        self._ssock = _DummySock()
        self._csock = _DummySock()

    def _dummy_loop_self_reading(self, f=None):
        pass

    def _dummy_write_to_self(self):
        if hasattr(self, "_proactor") and hasattr(self._proactor, "_iocp") and self._proactor._iocp:
            try:
                ov = _overlapped.Overlapped()
                _overlapped.PostQueuedCompletionStatus(self._proactor._iocp, 0, 0, ov.address)
            except Exception:
                pass

    def _dummy_close_self_pipe(self):
        self._self_reading_future = None
        self._ssock = None
        self._csock = None

    _orig_poll = asyncio.windows_events.IocpProactor._poll

    def _safe_poll(self, timeout=None):
        try:
            return _orig_poll(self, timeout)
        except TimeoutError:
            return None

    asyncio.windows_events.IocpProactor._poll = _safe_poll
    asyncio.proactor_events.BaseProactorEventLoop._make_self_pipe = _dummy_make_self_pipe
    asyncio.proactor_events.BaseProactorEventLoop._loop_self_reading = _dummy_loop_self_reading
    asyncio.proactor_events.BaseProactorEventLoop._write_to_self = _dummy_write_to_self
    asyncio.proactor_events.BaseProactorEventLoop._close_self_pipe = _dummy_close_self_pipe


@pytest.fixture(autouse=True)
def mock_network_yfinance(request):
    """Auto-mock yfinance.download so tests run offline immediately unless test manages it."""
    # If the test already patches yfinance, do nothing
    if "live_download" in request.keywords or "live_download" in request.node.name:
        yield
        return

    with patch("yfinance.download", side_effect=ConnectionError("Offline testing mode")):
        yield
