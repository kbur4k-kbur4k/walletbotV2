"""
tests/unit/test_oci_deployer.py

Unit tests for OCIDeployer, ProcessWatchdog, and HealthCheckAPI (Task 1.3).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from wallet_manager.strategies.oci_deployer import (
    MEMORY_LIMIT_MB,
    WEBSOCKET_TIMEOUT_S,
    HealthCheckAPI,
    ProcessWatchdog,
    SystemVitals,
    WatchdogEvent,
    _get_memory_rss_mb,
)


class TestProcessWatchdogInstantiation:
    def test_default_instantiation(self) -> None:
        """ProcessWatchdog can be instantiated with defaults."""
        wd = ProcessWatchdog()
        assert wd._memory_limit_mb == MEMORY_LIMIT_MB
        assert wd._ws_timeout_s == WEBSOCKET_TIMEOUT_S
        assert wd._running is False
        assert wd._total_restarts == 0

    def test_custom_thresholds(self) -> None:
        """ProcessWatchdog respects custom memory and WS timeout values."""
        wd = ProcessWatchdog(memory_limit_mb=512, ws_timeout_s=30)
        assert wd._memory_limit_mb == 512
        assert wd._ws_timeout_s == 30


class TestSystemVitalsSchema:
    def test_build_vitals_returns_system_vitals(self) -> None:
        """build_vitals() returns a SystemVitals dataclass with correct types."""
        wd = ProcessWatchdog()
        wd.record_ws_heartbeat()  # mark as connected
        vitals = wd.build_vitals()

        assert isinstance(vitals, SystemVitals)
        assert isinstance(vitals.memory_usage_mb, float)
        assert isinstance(vitals.memory_pct, float)
        assert isinstance(vitals.cpu_pct, float)
        assert isinstance(vitals.uptime_seconds, float)
        assert isinstance(vitals.ws_connected, bool)
        assert isinstance(vitals.is_healthy, bool)
        assert vitals.memory_limit_mb == MEMORY_LIMIT_MB

    def test_memory_pct_within_range(self) -> None:
        """memory_pct is between 0 and 100 (or slightly above when limit exceeded)."""
        wd = ProcessWatchdog()
        vitals = wd.build_vitals()
        assert 0.0 <= vitals.memory_pct <= 200.0  # allow >100 when over limit


class TestWatchdogRssThreshold:
    def test_health_ok_below_rss_limit(self) -> None:
        """Watchdog reports healthy when RSS < limit."""
        wd = ProcessWatchdog(memory_limit_mb=9999)
        wd.record_ws_heartbeat()
        vitals = wd.build_vitals()
        # RSS should be well below 9999 MB in a test environment
        assert vitals.memory_pct < 100.0

    def test_unhealthy_when_rss_mocked_high(self) -> None:
        """is_healthy is False when simulated RSS exceeds limit."""
        wd = ProcessWatchdog(memory_limit_mb=1)
        wd.record_ws_heartbeat()
        with patch(
            "wallet_manager.strategies.oci_deployer._get_memory_rss_mb",
            return_value=2.0,  # 2 MB > 1 MB limit
        ):
            vitals = wd.build_vitals()
        assert vitals.memory_pct > 100.0
        assert not vitals.is_healthy


class TestWatchdogEvents:
    def test_get_events_empty_initially(self) -> None:
        """No events recorded at instantiation."""
        wd = ProcessWatchdog()
        assert wd.get_events() == []

    async def test_start_stop_lifecycle(self) -> None:
        """Watchdog can be started and stopped without error."""
        wd = ProcessWatchdog()
        await wd.start()
        assert wd._running is True
        await asyncio.sleep(0.05)
        await wd.stop()
        assert wd._running is False


class TestHealthCheckAPI:
    def test_instantiation(self) -> None:
        """HealthCheckAPI can be instantiated with a watchdog."""
        wd = ProcessWatchdog()
        api = HealthCheckAPI(watchdog=wd)
        assert api._started is False
        assert api._watchdog is wd

    async def test_health_endpoint_returns_healthy_schema(self) -> None:
        """_handle_health returns a JSON response with required keys."""
        wd = ProcessWatchdog(memory_limit_mb=9999)
        wd.record_ws_heartbeat()
        api = HealthCheckAPI(watchdog=wd)

        mock_request = MagicMock()

        from aiohttp.web import Response
        response = await api._handle_health(mock_request)

        assert isinstance(response, Response)
        import json
        body = json.loads(response.body)
        assert "status" in body
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert "uptime_s" in body
        assert "memory_mb" in body
        assert "ws_connected" in body

    async def test_vitals_endpoint_returns_full_schema(self) -> None:
        """_handle_vitals returns all SystemVitals fields."""
        wd = ProcessWatchdog()
        api = HealthCheckAPI(watchdog=wd)
        mock_request = MagicMock()

        from aiohttp.web import Response
        response = await api._handle_vitals(mock_request)

        assert isinstance(response, Response)
        import json
        body = json.loads(response.body)
        assert "memory_usage_mb" in body
        assert "is_healthy" in body
        assert "timestamp" in body


class TestGetMemoryRss:
    def test_returns_float(self) -> None:
        """_get_memory_rss_mb returns a non-negative float."""
        result = _get_memory_rss_mb()
        assert isinstance(result, float)
        assert result >= 0.0
