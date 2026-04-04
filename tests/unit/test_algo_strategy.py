"""tests/unit/test_algo_strategy.py — AlgoStrategy drawdown circuit-breaker and signal tests."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import wallet_manager.strategies.algo_strategy as algo_mod
from wallet_manager.strategies.algo_strategy import (
    _DAILY_DRAWDOWN_STATE,
    DAILY_DRAWDOWN_LIMIT_PCT,
    RSI_OVERSOLD_THRESHOLD,
    Z_SCORE_BUY_THRESHOLD,
    _check_daily_drawdown_circuit_breaker,
    compute_rsi,
    compute_z_score_ema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_command_state(halted: bool = False):
    cs = MagicMock()
    cs.is_circuit_breaker_halted = halted
    cs.trigger_circuit_breaker_halt = MagicMock()
    return cs


def _make_task_context(command_state=None):
    ctx = {"wallet_manager": MagicMock(), "portfolio_manager": MagicMock()}
    if command_state is not None:
        ctx["command_state"] = command_state
    return ctx


# ---------------------------------------------------------------------------
# Test: Drawdown circuit breaker
# ---------------------------------------------------------------------------

def test_drawdown_circuit_breaker():
    """When portfolio drops >= 5%, _check_daily_drawdown_circuit_breaker must trigger halt."""
    import time as _time
    algo_mod._DAILY_DRAWDOWN_STATE.clear()

    wallet_id = "test_wallet_drawdown"
    start_tvl = 10_000.0
    # Use today's actual day_start_ts so the function doesn't reset on day boundary
    now = _time.time()
    day_start_ts = (int(now) // 86400) * 86400
    # The function uses key = f"{wallet_id}_algo"
    key = f"{wallet_id}_algo"
    algo_mod._DAILY_DRAWDOWN_STATE[key] = (start_tvl, day_start_ts)

    # Simulate a 6% drop — exceeds 5% limit
    dropped_tvl = start_tvl * (1.0 - DAILY_DRAWDOWN_LIMIT_PCT - 0.01)
    command_state = _make_command_state()
    ctx = _make_task_context(command_state)

    triggered = asyncio.run(
        _check_daily_drawdown_circuit_breaker(wallet_id, dropped_tvl, ctx)
    )

    assert triggered is True, "Circuit breaker should trigger on large drawdown"
    command_state.trigger_circuit_breaker_halt.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Bullish signal leads to BUY logic
# ---------------------------------------------------------------------------

def test_strategy_handles_bullish_signal():
    """Confirm that a sudden price drop produces a negative z-score (buy direction)."""
    base = 2000.0
    # 20 stable prices, then a sharp drop for the last entry
    prices = [base] * 19 + [base * 0.70]  # 30% drop at the end

    _, _, z = compute_z_score_ema(prices, period=20)
    rsi = compute_rsi(prices, period=14)

    # Z-score should be negative (price below EMA mean → buy signal direction)
    assert z < 0, f"Expected negative z-score on price drop, got {z:.3f}"
    # RSI should be low (oversold condition)
    assert rsi < RSI_OVERSOLD_THRESHOLD, f"Expected rsi < {RSI_OVERSOLD_THRESHOLD}, got {rsi:.1f}"


# ---------------------------------------------------------------------------
# Test: Bearish signal leads to SELL logic
# ---------------------------------------------------------------------------

def test_strategy_handles_bearish_signal():
    """High RSI and positive z-score represent overbought / bearish conditions."""
    # Sharp rally → high RSI and high z-score
    base = 2000.0
    prices = [base] * 50
    prices += [base * 1.10] * 10  # +10% spike

    _, _, z = compute_z_score_ema(prices, period=20)
    rsi = compute_rsi(prices, period=14)

    # Z-score should be strongly positive (overbought)
    assert z > 0, f"Expected positive z-score for overbought, got {z:.3f}"
    # RSI should be elevated (not oversold)
    assert rsi > RSI_OVERSOLD_THRESHOLD, f"Expected rsi > {RSI_OVERSOLD_THRESHOLD}, got {rsi:.1f}"
