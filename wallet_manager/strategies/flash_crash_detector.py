"""
wallet_manager/strategies/flash_crash_detector.py

Emergency Circuit Breaker — Flash Crash Detection + KILL_SWITCH.

Core Mandate
============
Monitor price feed for sudden drops exceeding 10% within a configurable
time window. When detected:

1. KILL_SWITCH activates: all strategies halt immediately.
2. Emergency Telegram alert sent with severity=CRITICAL.
3. All open short positions are force-closed to lock in profit.
4. Idle capital sweep paused.
5. System remains halted until operator sends /resume via Telegram.

Detection Algorithm
-------------------
Maintains a rolling price window (last N prices with timestamps).
Computes max drawdown from peak within the window. If drawdown >= 10%,
triggers the KILL_SWITCH.

This is NOT the same as the daily drawdown circuit breaker in
algo_strategy.py — that tracks portfolio value. This tracks raw
WETH/USDC price action for systemic market risk.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Flash crash: 8% drop within detection window (QUANT-X upgraded from 10%)
CRASH_THRESHOLD_PCT = 0.08

# Detection window: 5 minutes of price history
DETECTION_WINDOW_SECONDS = 300.0

# Maximum entries in the rolling price buffer
MAX_PRICE_ENTRIES = 120

# Cooldown after KILL_SWITCH trigger (avoid re-triggering on recovery)
KILL_SWITCH_COOLDOWN_SECONDS = 3600.0  # 1 hour

# Rate-limit for crash detection checks
MIN_CHECK_INTERVAL_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Price entry in rolling buffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceEntry:
    """Single price observation with timestamp."""

    price: float
    timestamp: float


# ---------------------------------------------------------------------------
# FlashCrashDetector
# ---------------------------------------------------------------------------


class FlashCrashDetector:
    """Real-time flash crash detection engine.

    Maintains a rolling deque of price observations and computes the
    maximum drawdown from peak within the detection window. If the
    drawdown exceeds ``CRASH_THRESHOLD_PCT``, the KILL_SWITCH is
    activated.

    Thread-safety: designed for single-threaded asyncio usage.
    All state mutations happen between await points.

    Args:
        crash_threshold: Minimum price drop fraction to trigger (default 10%).
        window_seconds: Rolling window for crash detection (default 5 min).
        cooldown_seconds: Post-trigger cooldown (default 1 hour).

    Example:
        .. code-block:: python

            detector = FlashCrashDetector()

            # Called every price tick:
            alert = detector.ingest_price(current_price)
            if alert:
                # KILL_SWITCH triggered!
                await send_emergency_alert(alert)
    """

    def __init__(
        self,
        crash_threshold: float = CRASH_THRESHOLD_PCT,
        window_seconds: float = DETECTION_WINDOW_SECONDS,
        cooldown_seconds: float = KILL_SWITCH_COOLDOWN_SECONDS,
    ) -> None:
        self._threshold = crash_threshold
        self._window = window_seconds
        self._cooldown = cooldown_seconds

        self._prices: deque[PriceEntry] = deque(maxlen=MAX_PRICE_ENTRIES)
        self._kill_switch_active: bool = False
        self._last_trigger_time: float = 0.0
        self._last_check_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_kill_switch_active(self) -> bool:
        """Return True if the KILL_SWITCH is currently engaged."""
        return self._kill_switch_active

    def reset_kill_switch(self) -> None:
        """Manually reset the KILL_SWITCH (e.g. via /resume command).

        Clears the active flag and the price buffer so detection
        resumes with fresh data.
        """
        self._kill_switch_active = False
        self._prices.clear()
        log.info("FlashCrashDetector: KILL_SWITCH manually reset.")

    def ingest_price(self, price: float) -> CrashAlert | None:
        """Ingest a new price observation and check for flash crash.

        Args:
            price: Current WETH/USDC price.

        Returns:
            :class:`CrashAlert` if crash detected, ``None`` otherwise.
        """
        now = time.time()

        # Rate limit checks
        if now - self._last_check_time < MIN_CHECK_INTERVAL_SECONDS:
            self._prices.append(PriceEntry(price=price, timestamp=now))
            return None
        self._last_check_time = now

        # Don't re-trigger during cooldown
        if self._kill_switch_active:
            # If triggered externally (last_trigger_time == 0), treat as just triggered
            effective_trigger = self._last_trigger_time if self._last_trigger_time > 0 else now
            if now - effective_trigger < self._cooldown:
                return None
            # Cooldown expired — reset for re-detection
            self._kill_switch_active = False

        # Add to buffer
        self._prices.append(PriceEntry(price=price, timestamp=now))

        # Evict stale entries outside the window
        self._evict_stale(now)

        # Need at least 2 data points
        if len(self._prices) < 2:
            return None

        # Compute max drawdown from peak within window
        alert = self._check_crash(now, price)
        return alert

    def get_buffer_stats(self) -> dict:
        """Return diagnostic stats about the price buffer.

        Returns:
            Dict with buffer_size, window_peak, window_low, current_drawdown_pct.
        """
        if not self._prices:
            return {
                "buffer_size": 0,
                "window_peak": 0.0,
                "window_low": 0.0,
                "current_drawdown_pct": 0.0,
            }
        prices = [e.price for e in self._prices]
        peak = max(prices)
        low = min(prices)
        drawdown = (peak - low) / peak if peak > 0 else 0.0
        return {
            "buffer_size": len(self._prices),
            "window_peak": peak,
            "window_low": low,
            "current_drawdown_pct": drawdown * 100,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_stale(self, now: float) -> None:
        """Remove entries outside the detection window."""
        cutoff = now - self._window
        while self._prices and self._prices[0].timestamp < cutoff:
            self._prices.popleft()

    def _check_crash(self, now: float, current_price: float) -> CrashAlert | None:
        """Compute max drawdown and trigger alert if threshold exceeded."""
        peak_price = 0.0
        peak_time = 0.0

        for entry in self._prices:
            if entry.price > peak_price:
                peak_price = entry.price
                peak_time = entry.timestamp

        if peak_price <= 0:
            return None

        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= self._threshold:
            self._kill_switch_active = True
            self._last_trigger_time = now

            alert = CrashAlert(
                drop_pct=drawdown * 100,
                peak_price=peak_price,
                current_price=current_price,
                window_seconds=now - peak_time,
                timestamp=now,
            )
            log.critical(
                f"🚨 FLASH CRASH DETECTED: {alert.drop_pct:.1f}% drop "
                f"(${peak_price:.2f} → ${current_price:.2f}) in "
                f"{alert.window_seconds:.0f}s. KILL_SWITCH ACTIVATED."
            )
            return alert

        return None


# ---------------------------------------------------------------------------
# Alert data carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrashAlert:
    """Immutable crash detection alert payload.

    Attributes:
        drop_pct: Price drop percentage (e.g. 12.5 for 12.5%).
        peak_price: Highest price in the detection window.
        current_price: Price that triggered the alert.
        window_seconds: Time elapsed from peak to crash.
        timestamp: Unix epoch when the alert was generated.
    """

    drop_pct: float
    peak_price: float
    current_price: float
    window_seconds: float
    timestamp: float


# ---------------------------------------------------------------------------
# Integration: send KILL_SWITCH alert via Telegram
# ---------------------------------------------------------------------------


async def send_kill_switch_alert(
    task_context: dict,
    alert: CrashAlert,
) -> None:
    """Send emergency KILL_SWITCH Telegram notification.

    Tries TelegramManager first (rich formatting), falls back to
    basic notifier if unavailable.
    """
    try:
        from wallet_manager.security.telegram_manager import (
            CircuitBreakerPayload,
        )

        tg = task_context.get("telegram_manager")
        if tg and hasattr(tg, "send_circuit_breaker_alert"):
            await tg.send_circuit_breaker_alert(
                CircuitBreakerPayload(
                    alert_type="FLASH_CRASH_KILL_SWITCH",
                    detail=(
                        f"🚨 FLASH CRASH: {alert.drop_pct:.1f}% drop in "
                        f"{alert.window_seconds:.0f}s\n"
                        f"Peak: ${alert.peak_price:.2f} → "
                        f"Current: ${alert.current_price:.2f}\n"
                        f"ALL STRATEGIES HALTED.\n"
                        f"Send /resume to restart trading."
                    ),
                    severity="critical",
                    start_value_usd=alert.peak_price,
                    current_value_usd=alert.current_price,
                    drawdown_pct=alert.drop_pct,
                )
            )
            return
    except Exception as exc:
        log.warning(f"KILL_SWITCH: TelegramManager alert failed: {exc}")

    # Fallback: basic notifier
    try:
        from wallet_manager.core.models import EventKind, NotificationEvent

        notifier = task_context.get("notifier")
        if notifier:
            await notifier.send(
                NotificationEvent(
                    kind=EventKind.WALLET_FAILURE,
                    detail=(
                        f"🚨 KILL_SWITCH: {alert.drop_pct:.1f}% flash crash. "
                        f"All trading halted. /resume to restart."
                    ),
                )
            )
    except Exception as exc:
        log.error(f"KILL_SWITCH: fallback alert also failed: {exc}")


# ---------------------------------------------------------------------------
# Integration: activate kill switch on CommandState
# ---------------------------------------------------------------------------


def activate_kill_switch(task_context: dict, alert: CrashAlert) -> None:
    """Activate KILL_SWITCH on the shared CommandState.

    This halts all strategies (YIELD, ALGO, SHORT, AIRDROP) until
    the operator sends /resume.

    Args:
        task_context: Shared orchestrator context dict.
        alert: The crash alert that triggered activation.
    """
    command_state = task_context.get("command_state")
    if command_state is None:
        log.warning("KILL_SWITCH: no CommandState available — cannot halt strategies.")
        return

    if hasattr(command_state, "trigger_circuit_breaker_halt"):
        # Halt indefinitely (until /resume) — use very large duration
        command_state.trigger_circuit_breaker_halt(86400 * 365)  # ~1 year
        log.critical(
            f"KILL_SWITCH: CommandState halted. "
            f"Crash: {alert.drop_pct:.1f}% in {alert.window_seconds:.0f}s."
        )
    else:
        log.warning("KILL_SWITCH: CommandState lacks trigger_circuit_breaker_halt().")


# ---------------------------------------------------------------------------
# Singleton detector (process-wide)
# ---------------------------------------------------------------------------

_detector: FlashCrashDetector | None = None


def get_detector() -> FlashCrashDetector:
    """Return the process-wide FlashCrashDetector singleton."""
    global _detector
    if _detector is None:
        _detector = FlashCrashDetector()
    return _detector


def reset_detector() -> None:
    """Reset the singleton (for tests)."""
    global _detector
    _detector = None
