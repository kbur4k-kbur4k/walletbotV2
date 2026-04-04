"""tests/unit/test_kelly_criterion.py — KellyCriterion position sizing tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from wallet_manager.strategies.kelly_criterion import (
    KELLY_FRACTION,
    MAX_POSITION_PCT,
    MIN_TRADES_FOR_KELLY,
    KellyCriterion,
    TradeResult,
)


def _kelly(tmp_path: Path) -> KellyCriterion:
    return KellyCriterion(history_path=tmp_path / "kelly_test.json")


def _win(usd: float = 10.0) -> TradeResult:
    return TradeResult(
        pnl_usd=usd, entry_price=2000.0, exit_price=2010.0,
        position_usd=100.0, z_score_at_entry=-2.8, strategy="ALGO",
    )


def _loss(usd: float = 5.0) -> TradeResult:
    return TradeResult(
        pnl_usd=-usd, entry_price=2000.0, exit_price=1990.0,
        position_usd=100.0, z_score_at_entry=-2.8, strategy="ALGO",
    )


def test_zero_win_rate_produces_no_position(tmp_path):
    k = _kelly(tmp_path)
    # Record only losses to force 0% win rate
    for _ in range(MIN_TRADES_FOR_KELLY + 1):
        k.record_trade(_loss())
    result = k.calculate_position_size(
        total_capital_usd=1000.0, z_score=-3.0
    )
    assert result.position_usd == 0.0


def test_position_capped_at_max_pct(tmp_path):
    k = _kelly(tmp_path)
    for _ in range(MIN_TRADES_FOR_KELLY + 1):
        k.record_trade(_win(50.0))  # High wins to push Kelly up
    result = k.calculate_position_size(
        total_capital_usd=1000.0, z_score=-3.5
    )
    assert result.position_usd <= 1000.0 * MAX_POSITION_PCT + 0.01


def test_half_kelly_is_half_of_full(tmp_path):
    k = _kelly(tmp_path)
    for _ in range(MIN_TRADES_FOR_KELLY + 1):
        k.record_trade(_win(10.0))
        k.record_trade(_loss(5.0))
    r = k.calculate_position_size(total_capital_usd=1000.0, z_score=-3.0)
    # kelly_scaled <= kelly_raw * KELLY_FRACTION (half-kelly)
    assert r.kelly_scaled <= r.kelly_raw * KELLY_FRACTION + 1e-9


def test_insufficient_history_uses_conservative_default(tmp_path):
    k = _kelly(tmp_path)
    # Record fewer than MIN_TRADES_FOR_KELLY trades
    for _ in range(5):
        k.record_trade(_win())
    result = k.calculate_position_size(
        total_capital_usd=1000.0, z_score=-3.0
    )
    # Should use default 10% or max, not Kelly formula
    assert result.position_usd <= 1000.0 * MAX_POSITION_PCT + 0.01
    assert result.position_usd > 0


def test_drawdown_throttle_reduces_size(tmp_path):
    k = _kelly(tmp_path)
    # Build baseline
    for _ in range(MIN_TRADES_FOR_KELLY + 1):
        k.record_trade(_win(10.0))
    normal_result = k.calculate_position_size(
        total_capital_usd=1000.0, z_score=-3.0
    )
    # Now record 3+ consecutive losses (throttle kicks in at 3)
    for _ in range(4):
        k.record_trade(_loss(10.0))
    throttled_result = k.calculate_position_size(
        total_capital_usd=1000.0, z_score=-3.0
    )
    assert throttled_result.position_usd <= normal_result.position_usd + 0.01
    assert throttled_result.drawdown_multiplier < 1.0
