"""
wallet_manager/alerting.py

Sprint JS-7.1: Alert Escalation System
=========================================
Three-tier alert system:
  INFO     — log only, no Telegram
  WARNING  — single Telegram message
  CRITICAL — Telegram every 15min until acknowledged via /ack command

Replaces the single-level proactive_alerter.py for risk-critical notifications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    level: AlertLevel
    source: str
    message: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    sent_count: int = 0
    last_sent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "sent_count": self.sent_count,
            "age_minutes": round((time.time() - self.timestamp) / 60, 1),
        }


class AlertEscalationSystem:
    """
    Manages tiered alerts with Telegram delivery.

    INFO   : log only
    WARNING: send once via Telegram
    CRITICAL: send every 15 minutes until operator acknowledges

    Usage:
        alerter = get_alert_system()
        alerter.warn("RiskCommittee", "Risk score exceeded 70/100")
        alerter.critical("DrawdownAccel", "Drawdown accelerating — 3 consecutive readings")
        await alerter.run_delivery_loop(notifier)  # background task
    """

    _CRITICAL_REPEAT_INTERVAL = 15 * 60.0  # 15 minutes

    def __init__(self) -> None:
        self._active: list[Alert] = []
        self._history: list[Alert] = []
        self._max_history = 200

    def info(self, source: str, message: str) -> None:
        log.info(f"[ALERT:INFO] {source}: {message}")

    def warn(self, source: str, message: str) -> None:
        log.warning(f"[ALERT:WARNING] {source}: {message}")
        alert = Alert(AlertLevel.WARNING, source, message)
        self._active.append(alert)

    def critical(self, source: str, message: str) -> None:
        log.error(f"[ALERT:CRITICAL] {source}: {message}")
        for existing in self._active:
            if existing.level == AlertLevel.CRITICAL and existing.source == source and not existing.acknowledged:
                return
        alert = Alert(AlertLevel.CRITICAL, source, message)
        self._active.append(alert)

    def acknowledge(self, source: str | None = None) -> int:
        """Acknowledge CRITICAL alerts. Pass source to ACK specific, None to ACK all."""
        count = 0
        for alert in self._active:
            if (
                alert.level == AlertLevel.CRITICAL
                and not alert.acknowledged
                and (source is None or alert.source == source)
            ):
                alert.acknowledged = True
                count += 1
                log.info(f"Alert ACKed: {alert.source} — {alert.message[:60]}")
        return count

    def get_active_criticals(self) -> list[Alert]:
        return [a for a in self._active if a.level == AlertLevel.CRITICAL and not a.acknowledged]

    async def tick(self, notifier: Any) -> None:
        """
        Call periodically (e.g. every minute) to deliver pending alerts.
        notifier must have: async send_message(text, parse_mode=None)
        """
        now = time.time()
        delivered: list[Alert] = []
        for alert in self._active:
            if alert.acknowledged:
                delivered.append(alert)
                continue
            if alert.level == AlertLevel.WARNING:
                if alert.sent_count == 0:
                    await self._send(notifier, alert)
                    delivered.append(alert)
            elif (
                alert.level == AlertLevel.CRITICAL
                and (
                    alert.sent_count == 0
                    or (now - alert.last_sent) >= self._CRITICAL_REPEAT_INTERVAL
                )
            ):
                await self._send(notifier, alert)
        for a in delivered:
            if a in self._active:
                self._active.remove(a)
                self._history.append(a)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    async def _send(self, notifier: Any, alert: Alert) -> None:
        if notifier is None:
            return
        level_emoji = {"INFO": "[i]", "WARNING": "[!]", "CRITICAL": "[!!!]"}.get(alert.level.value, "[?]")
        repeat_note = f" \\(repeat #{alert.sent_count + 1}\\)" if alert.sent_count > 0 else ""
        msg = (
            f"{level_emoji} *{alert.level.value}*{repeat_note}\n"
            f"Source: `{alert.source}`\n"
            f"{alert.message}"
        )
        try:
            await notifier.send_message(msg, parse_mode="MarkdownV2")
            alert.sent_count += 1
            alert.last_sent = time.time()
        except Exception as exc:
            log.warning(f"AlertSystem: failed to send Telegram: {exc}")

    def process_signals(
        self,
        signal_alerts: list[str],
        risk_alerts: list[str],
    ) -> None:
        """
        Ingest alert strings from SignalIntelligence and RiskIntelligence.
        Routes to appropriate tier based on content.
        """
        for msg in signal_alerts:
            if "IC_NOISE" in msg:
                self.warn("SignalIntelligence", msg)
            elif "LOW_DIVERSITY" in msg:
                self.critical("EnsembleDiversity", msg)
            elif "REDUNDANT" in msg:
                self.warn("SignalCorrelation", msg)
        for msg in risk_alerts:
            if "DRAWDOWN_ACCELERATING" in msg:
                self.critical("DrawdownAcceleration", msg)
            elif "FAT_TAIL_VAR" in msg:
                self.warn("FatTailVaR", msg)

    def to_dict(self) -> dict[str, Any]:
        active_crits = self.get_active_criticals()
        return {
            "active_count": len(self._active),
            "critical_unacked": len(active_crits),
            "history_count": len(self._history),
            "active_alerts": [a.to_dict() for a in self._active[-10:]],
            "recent_history": [a.to_dict() for a in self._history[-5:]],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_alert_system: AlertEscalationSystem | None = None


def get_alert_system() -> AlertEscalationSystem:
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertEscalationSystem()
    return _alert_system
