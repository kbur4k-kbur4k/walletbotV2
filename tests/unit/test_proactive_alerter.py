"""
Tests for ProactiveAlerter.

Covers: alert triggering per condition, deduplication window,
multi-condition firing, notifier is never called for nominal state.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.telegram_notifier import ProactiveAlerter, TelegramNotifier

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def alerter() -> ProactiveAlerter:
    return ProactiveAlerter()


@pytest.fixture
def mock_notifier():
    n = MagicMock()
    n.send_event = AsyncMock()
    return n


def _settings_patch(
    yield_min_apy_pct: float = 1.0,
    drawdown_reduce_threshold: float = 0.05,
    alert_dedup_window_seconds: float = 3600.0,
    model_staleness_alert_days: float = 7.0,
    rpc_latency_alert_ms: float = 500.0,
):
    """Return a context manager that patches get_settings with safe defaults."""
    ms = MagicMock()
    ms.yield_min_apy_pct = yield_min_apy_pct
    ms.drawdown_reduce_threshold = drawdown_reduce_threshold
    ms.alert_dedup_window_seconds = alert_dedup_window_seconds
    ms.model_staleness_alert_days = model_staleness_alert_days
    ms.rpc_latency_alert_ms = rpc_latency_alert_ms
    return patch("wallet_manager.config.get_settings", return_value=ms)


# ── tests ─────────────────────────────────────────────────────────────────────

async def test_no_alerts_when_all_nominal(alerter, mock_notifier):
    """No alerts sent when all metrics are comfortably within bounds."""
    with _settings_patch():
        await alerter.check_and_alert(
            notifier=mock_notifier,
            drift_level="NOMINAL",
            apy=0.035,
            model_staleness_days=1.0,
            rpc_latency_ms=100.0,
            drawdown=0.01,
        )
    mock_notifier.send_event.assert_not_called()


async def test_severe_drift_triggers_alert(alerter, mock_notifier):
    """SEVERE drift level fires exactly one alert."""
    with _settings_patch():
        await alerter.check_and_alert(
            notifier=mock_notifier,
            drift_level="SEVERE",
            apy=0.035,
            model_staleness_days=1.0,
            rpc_latency_ms=100.0,
            drawdown=0.01,
        )
    mock_notifier.send_event.assert_called_once()


async def test_deduplication_suppresses_second_alert(alerter, mock_notifier):
    """A second identical SEVERE alert within the dedup window is silenced."""
    kwargs = {
        "notifier": mock_notifier,
        "drift_level": "SEVERE",
        "apy": 0.035,
        "model_staleness_days": 1.0,
        "rpc_latency_ms": 100.0,
        "drawdown": 0.01,
    }
    with _settings_patch(alert_dedup_window_seconds=3600.0):
        await alerter.check_and_alert(**kwargs)
        await alerter.check_and_alert(**kwargs)
    # Only the first call should reach send_event
    assert mock_notifier.send_event.call_count == 1


async def test_apy_below_threshold_fires(alerter, mock_notifier):
    """APY below yield_min_apy_pct threshold triggers an alert."""
    with _settings_patch(yield_min_apy_pct=1.0):   # 1.0% threshold
        await alerter.check_and_alert(
            notifier=mock_notifier,
            drift_level="NOMINAL",
            apy=0.005,           # 0.5% — below 1.0% threshold
            model_staleness_days=1.0,
            rpc_latency_ms=100.0,
            drawdown=0.01,
        )
    mock_notifier.send_event.assert_called_once()


async def test_stale_model_fires(alerter, mock_notifier):
    """Model older than model_staleness_alert_days triggers an alert."""
    with _settings_patch(model_staleness_alert_days=7.0):
        await alerter.check_and_alert(
            notifier=mock_notifier,
            drift_level="NOMINAL",
            apy=0.035,
            model_staleness_days=8.0,    # 8 > 7 days
            rpc_latency_ms=100.0,
            drawdown=0.01,
        )
    mock_notifier.send_event.assert_called_once()


async def test_drawdown_above_threshold_fires(alerter, mock_notifier):
    """Drawdown exceeding drawdown_reduce_threshold triggers an alert."""
    with _settings_patch(drawdown_reduce_threshold=0.05):
        await alerter.check_and_alert(
            notifier=mock_notifier,
            drift_level="NOMINAL",
            apy=0.035,
            model_staleness_days=1.0,
            rpc_latency_ms=100.0,
            drawdown=0.08,          # 8% > 5% threshold
        )
    mock_notifier.send_event.assert_called_once()


async def test_multiple_conditions_each_fire_once(mock_notifier):
    """When multiple distinct alert conditions are met, each fires once."""
    fresh_alerter = ProactiveAlerter()    # fresh so no prior dedup state
    with _settings_patch(
        yield_min_apy_pct=1.0,
        drawdown_reduce_threshold=0.05,
        alert_dedup_window_seconds=0.0,   # disable dedup to count all
    ):
        await fresh_alerter.check_and_alert(
            notifier=mock_notifier,
            drift_level="SEVERE",
            apy=0.005,            # below threshold
            model_staleness_days=1.0,
            rpc_latency_ms=100.0,
            drawdown=0.08,        # above threshold
        )
    # At minimum: drift + apy + drawdown = 3 conditions
    assert mock_notifier.send_event.call_count >= 3


async def test_send_event_alias_delegates_to_send():
    notifier = object.__new__(TelegramNotifier)
    notifier.send = AsyncMock()

    await TelegramNotifier.send_event(notifier, MagicMock())

    notifier.send.assert_awaited_once()
