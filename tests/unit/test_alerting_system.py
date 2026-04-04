from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from wallet_manager.alerting import AlertEscalationSystem, AlertLevel


@pytest.mark.asyncio
async def test_warning_is_delivered_once_and_moved_to_history() -> None:
    system = AlertEscalationSystem()
    notifier = AsyncMock()

    system.warn("RiskCommittee", "Threshold exceeded")
    await system.tick(notifier)

    notifier.send_message.assert_awaited_once()
    snapshot = system.to_dict()
    assert snapshot["active_count"] == 0
    assert snapshot["history_count"] == 1


@pytest.mark.asyncio
async def test_critical_repeats_until_acknowledged() -> None:
    system = AlertEscalationSystem()
    notifier = AsyncMock()

    system.critical("Drawdown", "Accelerating")
    await system.tick(notifier)
    assert notifier.send_message.await_count == 1

    critical = system.get_active_criticals()[0]
    critical.last_sent = time.time() - (system._CRITICAL_REPEAT_INTERVAL + 1)
    await system.tick(notifier)
    assert notifier.send_message.await_count == 2

    acked = system.acknowledge("Drawdown")
    assert acked == 1
    await system.tick(notifier)
    assert system.to_dict()["critical_unacked"] == 0


def test_process_signals_routes_to_expected_levels() -> None:
    system = AlertEscalationSystem()

    system.process_signals(
        signal_alerts=["LOW_DIVERSITY: models converged", "IC_NOISE: unstable"],
        risk_alerts=["DRAWDOWN_ACCELERATING: 3 readings", "FAT_TAIL_VAR: elevated"],
    )

    active = system.to_dict()["active_alerts"]
    levels = [item["level"] for item in active]
    sources = [item["source"] for item in active]

    assert AlertLevel.CRITICAL.value in levels
    assert AlertLevel.WARNING.value in levels
    assert "EnsembleDiversity" in sources
    assert "DrawdownAcceleration" in sources
