"""Tests for Prometheus webhook server (Task 3.3)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

SAMPLE_PAYLOAD = {
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "PortfolioValueDrop5pct", "severity": "critical"},
            "annotations": {"summary": "Portfolio dropped >5% in 1h"},
        }
    ]
}


async def test_post_alert_calls_notifier():
    from wallet_manager.alerting.prometheus_webhook import PrometheusWebhookServer

    notifier = MagicMock()
    notifier.send_text = AsyncMock()

    server = PrometheusWebhookServer(notifier=notifier, port=19093)
    # Reset module-level rate-limit state
    import wallet_manager.alerting.prometheus_webhook as wh
    wh._last_sent.clear()

    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value=SAMPLE_PAYLOAD)

    response = await server._handle_alerts(mock_request)
    assert response.status == 200
    notifier.send_text.assert_called_once()


async def test_rate_limit_suppresses_duplicate():
    import wallet_manager.alerting.prometheus_webhook as wh
    from wallet_manager.alerting.prometheus_webhook import PrometheusWebhookServer

    notifier = MagicMock()
    notifier.send_text = AsyncMock()

    server = PrometheusWebhookServer(notifier=notifier, port=19094, rate_limit_s=3600.0)
    wh._last_sent.clear()

    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value=SAMPLE_PAYLOAD)

    # First call sends
    await server._handle_alerts(mock_request)
    assert notifier.send_text.call_count == 1

    # Second call within rate limit window must NOT send
    mock_request2 = MagicMock()
    mock_request2.json = AsyncMock(return_value=SAMPLE_PAYLOAD)
    await server._handle_alerts(mock_request2)
    assert notifier.send_text.call_count == 1


def test_format_alerts_contains_alert_name():
    from wallet_manager.alerting.prometheus_webhook import _format_alerts

    alerts = SAMPLE_PAYLOAD["alerts"]
    msg = _format_alerts(alerts)
    assert "PortfolioValueDrop5pct" in msg
    assert "firing" in msg
    assert "🔴" in msg
