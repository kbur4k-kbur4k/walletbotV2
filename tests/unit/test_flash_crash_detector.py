"""tests/unit/test_flash_crash_detector.py — FlashCrashDetector unit tests."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from wallet_manager.strategies.flash_crash_detector import (
    CRASH_THRESHOLD_PCT,
    KILL_SWITCH_COOLDOWN_SECONDS,
    FlashCrashDetector,
)


def _make_detector(**kwargs) -> FlashCrashDetector:
    return FlashCrashDetector(**kwargs)


def _inject_price(detector: FlashCrashDetector, price: float, ts: float | None = None) -> object:
    """Inject a price bypassing the rate-limit check by setting _last_check_time far in the past."""
    if ts is None:
        ts = time.time()
    detector._last_check_time = ts - 100.0  # ensure check fires
    return detector.ingest_price(price)


def test_no_trigger_small_drop():
    """A drop just below the threshold (e.g. 7.9% < 8%) must NOT trigger."""
    detector = _make_detector()
    base = 1000.0
    _inject_price(detector, base)
    # Drop by 7.9% — below CRASH_THRESHOLD_PCT (8%)
    low = base * (1.0 - 0.079)
    alert = _inject_price(detector, low)
    assert alert is None, f"Expected no alert, got {alert}"
    # is_kill_switch_active is a @property (not a method)
    assert not detector.is_kill_switch_active


def test_trigger_on_large_drop():
    """A rapid drop >= 8% MUST trigger a CrashAlert and activate kill-switch."""
    detector = _make_detector()
    base = 1000.0
    now = time.time()
    _inject_price(detector, base, ts=now)
    # Drop by exactly CRASH_THRESHOLD_PCT + a bit
    low = base * (1.0 - CRASH_THRESHOLD_PCT - 0.005)
    alert = _inject_price(detector, low, ts=now + 10.0)
    assert alert is not None, "Expected CrashAlert on large drop"
    assert detector.is_kill_switch_active
    # CrashAlert uses drop_pct field
    assert alert.drop_pct >= CRASH_THRESHOLD_PCT * 100


def test_cooldown_prevents_retriggering():
    """After a trigger, a subsequent drop within cooldown must NOT retrigger."""
    detector = _make_detector(cooldown_seconds=KILL_SWITCH_COOLDOWN_SECONDS)
    base = 1000.0
    now = time.time()
    _inject_price(detector, base, ts=now)
    # First crash
    low = base * (1.0 - CRASH_THRESHOLD_PCT - 0.01)
    first = _inject_price(detector, low, ts=now + 10.0)
    assert first is not None, "First crash should trigger"

    # Another large drop immediately — cooldown should prevent re-triggering
    even_lower = low * (1.0 - CRASH_THRESHOLD_PCT - 0.01)
    second = _inject_price(detector, even_lower, ts=now + 20.0)
    assert second is None, "Cooldown should prevent re-triggering"
