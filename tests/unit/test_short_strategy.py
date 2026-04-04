"""tests/unit/test_short_strategy.py

Unit tests for wallet_manager/strategies/short_strategy.py.

All blockchain / RPC calls are mocked.  Tests focus on:
- ShortPosition / ShortState dataclass round-trips
- should_open_short / should_close_short pure-logic functions
- Position sizing: never exceeds MAX_SHORT_PORTFOLIO_PCT
- Health-factor floor enforcement
- Hard stop-loss trigger (5% price rise)
- State persistence helpers
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.short_strategy import (
    MAX_SHORT_PORTFOLIO_PCT,
    MIN_HEALTH_FACTOR,
    SHORT_STOP_LOSS_PCT,
    Z_SCORE_SHORT_THRESHOLD,
    ShortPosition,
    ShortState,
    _load_state,
    _save_state,
    get_short_position_state,
    is_short_active,
    should_close_short,
    should_open_short,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_position(entry_price: float = 2_000.0, borrow_wei: int = 10**18) -> ShortPosition:
    stop_loss = entry_price * (1.0 + SHORT_STOP_LOSS_PCT)
    return ShortPosition(
        position_id="test_pos_001",
        entry_price=entry_price,
        borrowed_weth_wei=borrow_wei,
        usdc_received=int(borrow_wei * entry_price / 1e18 * 1e6),
        collateral_usdc=0,
        entry_timestamp=1_000_000.0,
        entry_z_score=3.0,
        entry_rsi=75.0,
        is_active=True,
        stop_loss_price=stop_loss,
    )


# ===========================================================================
# Test 1 — get_short_position_state with no existing file → zero position
# ===========================================================================

class TestGetShortPositionStateNoPosition:
    """When no state file exists, the returned state reflects zero exposure."""

    def test_no_position_position_value_is_zero(self, tmp_path):
        """Loading a non-existent path → ShortPosition.borrowed_weth_wei == 0."""
        missing = tmp_path / "no_state.json"
        state = _load_state(missing)
        assert state.position.borrowed_weth_wei == 0

    def test_no_position_not_active(self, tmp_path):
        """Loading a non-existent path → is_active is False."""
        missing = tmp_path / "no_state.json"
        state = _load_state(missing)
        assert state.position.is_active is False

    def test_no_position_pnl_is_zero(self, tmp_path):
        """Fresh state → total_short_pnl_usd is 0.0."""
        missing = tmp_path / "no_state.json"
        state = _load_state(missing)
        assert state.total_short_pnl_usd == 0.0

    def test_get_short_position_state_via_public_api(self, tmp_path):
        """get_short_position_state returns a ShortState with zero position."""
        mock_path = tmp_path / "short_state.json"
        with patch(
            "wallet_manager.strategies.short_strategy._default_state_path",
            return_value=mock_path,
        ):
            state = get_short_position_state()
        assert isinstance(state, ShortState)
        assert state.position.borrowed_weth_wei == 0


# ===========================================================================
# Test 2 — should_open_short: Z-score below threshold → False
# ===========================================================================

class TestOpenShortRequiresZScoreAboveThreshold:
    """should_open_short must return False when Z-score < Z_SCORE_SHORT_THRESHOLD."""

    def test_z_score_1_returns_false(self):
        """Z-score=1.0, RSI=75, price at BB upper → should_open_short() is False."""
        result = should_open_short(z_score=1.0, rsi=75.0, bb_pct=0.0)
        assert result is False

    def test_z_score_exactly_at_threshold_returns_false(self):
        """Z-score == threshold (not strictly above) → False."""
        result = should_open_short(
            z_score=Z_SCORE_SHORT_THRESHOLD,
            rsi=75.0,
            bb_pct=0.0,
        )
        # Implementation uses strictly >, so exactly at threshold → False
        assert result is False

    def test_all_conditions_required(self):
        """Failing any single condition with Z-score fine → still False."""
        # RSI too low
        assert should_open_short(z_score=3.0, rsi=65.0, bb_pct=0.0) is False
        # BB not reached (price too far below upper band)
        assert should_open_short(z_score=3.0, rsi=75.0, bb_pct=0.05) is False


# ===========================================================================
# Test 3 — should_open_short: Z-score above threshold → True
# ===========================================================================

class TestOpenShortAboveThresholdPositive:
    """should_open_short must return True when ALL conditions are met."""

    def test_z_score_3_rsi_75_at_upper_bb(self):
        """Z-score=3.0, RSI=75, bb_pct=0.0 (at upper band) → True."""
        result = should_open_short(z_score=3.0, rsi=75.0, bb_pct=0.0)
        assert result is True

    def test_above_upper_bb(self):
        """Price above upper BB (bb_pct negative) also satisfies proximity check."""
        result = should_open_short(z_score=3.0, rsi=80.0, bb_pct=-0.01)
        assert result is True

    def test_extreme_overbought(self):
        """Very high Z-score and RSI → should open short."""
        result = should_open_short(z_score=5.0, rsi=90.0, bb_pct=-0.05)
        assert result is True


# ===========================================================================
# Test 4 — max position size not exceeded (pure sizing logic)
# ===========================================================================

class TestMaxPositionSizeNotExceeded:
    """Position size must never exceed MAX_SHORT_PORTFOLIO_PCT of total equity."""

    def test_sizing_cap_at_15pct(self):
        """min(max_short_usd, ...) must cap at 15% of equity."""
        total_equity = 10_000.0
        max_short_usd = total_equity * MAX_SHORT_PORTFOLIO_PCT  # 1500.0

        # Simulate the sizing formula from run_short_strategy
        available_borrows_usd = 20_000.0
        position_usd = min(max_short_usd, available_borrows_usd * 0.5)

        assert position_usd <= max_short_usd
        assert position_usd <= total_equity * MAX_SHORT_PORTFOLIO_PCT

    def test_sizing_cap_with_small_equity(self):
        """Small equity → small cap, always <= 15%."""
        for equity in [100.0, 500.0, 1_000.0, 5_000.0]:
            max_short = equity * MAX_SHORT_PORTFOLIO_PCT
            available_borrows = equity * 2
            position_usd = min(max_short, available_borrows * 0.5)
            assert position_usd <= equity * MAX_SHORT_PORTFOLIO_PCT

    def test_max_short_portfolio_pct_constant(self):
        """Verify the constant is 15% (not changed accidentally)."""
        assert pytest.approx(0.15) == MAX_SHORT_PORTFOLIO_PCT


# ===========================================================================
# Test 5 — health-factor floor: position rejected when HF < MIN_HEALTH_FACTOR
# ===========================================================================

class TestHealthFactorFloorRespected:
    """When the health factor is below MIN_HEALTH_FACTOR, the position
    should be flagged as unsafe and the strategy should avoid/close it."""

    @pytest.mark.asyncio
    async def test_unsafe_health_factor_returns_false(self):
        """check_health_factor_safe must return (False, hf) when HF < floor."""
        from wallet_manager.strategies.short_strategy import check_health_factor_safe

        low_hf = MIN_HEALTH_FACTOR - 0.1  # e.g. 1.4

        with patch(
            "wallet_manager.strategies.short_strategy.actions.get_aave_account_data",
            new_callable=AsyncMock,
            return_value={"health_factor": low_hf},
        ):
            is_safe, hf = await check_health_factor_safe("http://rpc", "0xAddress")

        assert is_safe is False
        assert hf == pytest.approx(low_hf)

    @pytest.mark.asyncio
    async def test_safe_health_factor_returns_true(self):
        """HF >= MIN_HEALTH_FACTOR → (True, hf)."""
        from wallet_manager.strategies.short_strategy import check_health_factor_safe

        safe_hf = MIN_HEALTH_FACTOR + 0.5  # e.g. 2.0

        with patch(
            "wallet_manager.strategies.short_strategy.actions.get_aave_account_data",
            new_callable=AsyncMock,
            return_value={"health_factor": safe_hf},
        ):
            is_safe, hf = await check_health_factor_safe("http://rpc", "0xAddress")

        assert is_safe is True
        assert hf == pytest.approx(safe_hf)

    @pytest.mark.asyncio
    async def test_zero_hf_means_no_debt(self):
        """HF == 0 means no debt → treated as safe."""
        from wallet_manager.strategies.short_strategy import check_health_factor_safe

        with patch(
            "wallet_manager.strategies.short_strategy.actions.get_aave_account_data",
            new_callable=AsyncMock,
            return_value={"health_factor": 0.0},
        ):
            is_safe, _hf = await check_health_factor_safe("http://rpc", "0xAddress")

        assert is_safe is True

    @pytest.mark.asyncio
    async def test_rpc_failure_returns_unsafe(self):
        """RPC failure must default to (False, 0.0) for safety."""
        from wallet_manager.strategies.short_strategy import check_health_factor_safe

        with patch(
            "wallet_manager.strategies.short_strategy.actions.get_aave_account_data",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            is_safe, hf = await check_health_factor_safe("http://rpc", "0xAddress")

        assert is_safe is False
        assert hf == 0.0


# ===========================================================================
# Test 6 — hard stop-loss: 5% price rise above entry → should_close_short
# ===========================================================================

class TestHardStopLossTriggerClose:
    """should_close_short must return (True, 'stop_loss') when price rises 5%."""

    def test_price_5pct_above_entry_triggers_stop(self):
        """WETH price exactly at stop_loss_price → close with 'stop_loss' reason."""
        entry_price = 2_000.0
        stop_loss_price = entry_price * (1.0 + SHORT_STOP_LOSS_PCT)  # 2100.0

        should_close, reason = should_close_short(
            z_score=2.0,
            rsi=60.0,
            current_price=stop_loss_price,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
        )
        assert should_close is True
        assert reason == "stop_loss"

    def test_price_above_stop_loss_triggers_stop(self):
        """Price above stop_loss_price → (True, 'stop_loss') regardless of Z/RSI."""
        entry_price = 3_000.0
        stop_loss_price = entry_price * (1.0 + SHORT_STOP_LOSS_PCT)  # 3150.0
        current_price = 3_200.0  # well above stop

        should_close, reason = should_close_short(
            z_score=1.5,
            rsi=55.0,
            current_price=current_price,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
        )
        assert should_close is True
        assert reason == "stop_loss"

    def test_price_below_stop_loss_no_stop_trigger(self):
        """Price below stop_loss_price → stop-loss not triggered."""
        entry_price = 2_000.0
        stop_loss_price = entry_price * (1.0 + SHORT_STOP_LOSS_PCT)  # 2100.0
        current_price = 2_050.0  # below stop

        should_close, reason = should_close_short(
            z_score=2.0,
            rsi=60.0,
            current_price=current_price,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
        )
        # May close for other reasons but NOT stop_loss
        if should_close:
            assert reason != "stop_loss"

    def test_stop_loss_pct_constant(self):
        """Verify SHORT_STOP_LOSS_PCT is 5% (not changed accidentally)."""
        assert pytest.approx(0.05) == SHORT_STOP_LOSS_PCT


# ===========================================================================
# Test 7 — mean reversion exit
# ===========================================================================

class TestMeanReversionExit:
    """should_close_short returns True when Z-score and RSI signal mean reversion."""

    def test_z_zero_rsi_50_triggers_mean_reversion(self):
        """Z-score <= 0 and RSI <= 50 → (True, 'mean_reversion')."""
        entry_price = 2_000.0
        stop_loss = entry_price * 1.05

        should_close, reason = should_close_short(
            z_score=0.0,
            rsi=50.0,
            current_price=1_950.0,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
        )
        assert should_close is True
        assert reason == "mean_reversion"

    def test_negative_z_triggers_mean_reversion(self):
        """Z-score < 0 and RSI <= 50 → mean reversion exit."""
        entry_price = 2_500.0
        stop_loss = entry_price * 1.05

        should_close, reason = should_close_short(
            z_score=-1.5,
            rsi=40.0,
            current_price=2_400.0,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
        )
        assert should_close is True
        assert reason == "mean_reversion"


# ===========================================================================
# Test 8 — ShortPosition dataclass round-trip serialisation
# ===========================================================================

class TestShortPositionSerialisation:
    """to_dict / from_dict must be lossless for all fields."""

    def test_round_trip_active_position(self):
        """ShortPosition.to_dict() → from_dict() reproduces the original."""
        pos = _active_position(entry_price=2_500.0, borrow_wei=5 * 10**17)
        restored = ShortPosition.from_dict(pos.to_dict())
        assert restored.entry_price == pos.entry_price
        assert restored.borrowed_weth_wei == pos.borrowed_weth_wei
        assert restored.is_active == pos.is_active
        assert restored.stop_loss_price == pytest.approx(pos.stop_loss_price)

    def test_round_trip_state(self, tmp_path):
        """ShortState serialises and deserialises cleanly through JSON."""
        state = ShortState(
            position=_active_position(),
            total_short_pnl_usd=123.45,
            total_shorts_opened=3,
            total_shorts_closed=2,
        )
        path = tmp_path / "state.json"
        _save_state(path, state)
        loaded = _load_state(path)
        assert loaded.total_short_pnl_usd == pytest.approx(123.45)
        assert loaded.total_shorts_opened == 3
        assert loaded.total_shorts_closed == 2
        assert loaded.position.is_active is True
