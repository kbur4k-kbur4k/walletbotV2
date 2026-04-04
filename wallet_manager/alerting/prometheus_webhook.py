"""
wallet_manager/alerting/prometheus_webhook.py

Lightweight aiohttp webhook server that receives Alertmanager alerts
and forwards them to Telegram via TelegramNotifier.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from aiohttp import web

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# Rate-limit: max 1 message per 30s per alert name
_RATE_LIMIT_S: float = 30.0
_last_sent: dict[str, float] = {}


def _format_alerts(alerts: list[dict[str, Any]]) -> str:
    """Format Alertmanager alert dicts into a single Telegram message."""
    lines = ["🚨 *Prometheus Alert*"]
    for alert in alerts:
        name = alert.get("labels", {}).get("alertname", "Unknown")
        status = alert.get("status", "firing")
        summary = alert.get("annotations", {}).get("summary", "")
        severity = alert.get("labels", {}).get("severity", "unknown")
        emoji = "🔴" if severity == "critical" else "🟡" if severity == "warning" else "ℹ️"
        lines.append(f"{emoji} *{name}* ({status})")
        if summary:
            lines.append(f"  {summary}")
    return "\n".join(lines)


def _is_rate_limited(alert_name: str) -> bool:
    now = time.monotonic()
    last = _last_sent.get(alert_name, 0.0)
    if now - last < _RATE_LIMIT_S:
        return True
    _last_sent[alert_name] = now
    return False


class PrometheusWebhookServer:
    """Receives Alertmanager webhook POSTs and sends Telegram alerts."""

    def __init__(
        self,
        notifier: Any,
        port: int = 9093,
        host: str = "0.0.0.0",
        rate_limit_s: float = _RATE_LIMIT_S,
    ) -> None:
        self._notifier = notifier
        self._port = port
        self._host = host
        self._rate_limit_s = rate_limit_s
        self._app = web.Application()
        self._app.router.add_post("/alerts", self._handle_alerts)
        self._runner: web.AppRunner | None = None

    async def _handle_alerts(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            log.warning("Invalid webhook payload: %s", exc)
            return web.Response(status=400, text="Invalid JSON")

        alerts: list[dict[str, Any]] = payload.get("alerts", [])
        if not alerts:
            return web.Response(status=200, text="ok")

        # Group: only send non-rate-limited alerts
        active: list[dict[str, Any]] = []
        for a in alerts:
            name = a.get("labels", {}).get("alertname", "Unknown")
            now = time.monotonic()
            last = _last_sent.get(name, float("-inf"))
            if now - last >= self._rate_limit_s:
                _last_sent[name] = now
                active.append(a)

        if active:
            msg = _format_alerts(active)
            try:
                await self._notifier.send_text(msg)
            except Exception as exc:
                log.warning("Telegram send failed for alert: %s", exc)

        return web.Response(status=200, text="ok")

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        log.info("PrometheusWebhookServer started on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
