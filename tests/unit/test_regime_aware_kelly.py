"""Tests for RegimeAwareKelly position sizing."""
import pytest

from wallet_manager.strategies.kelly_criterion import RegimeAwareKelly


@pytest.fixture
def kelly() -> RegimeAwareKelly:
    return RegimeAwareKelly()


def test_instantiation(kelly):
    assert kelly is not None


def test_bull_regime_positive_position(kelly):
    """Bull regime with no drawdown returns a positive position size."""
    result = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    assert result > 0.0


def test_crisis_regime_smaller_than_bull(kelly):
    """Crisis regime position must be strictly smaller than bull regime."""
    bull = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    crisis = kelly.compute_sized_position(
        base_kelly=0.5, regime="crisis",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    assert crisis < bull


def test_deep_drawdown_stops_trading(kelly):
    """Drawdown above 15% must produce zero position."""
    result = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL",
        current_drawdown=0.16,    # 16% > hard stop at 15%
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    assert result == 0.0


def test_moderate_drawdown_reduces_size(kelly):
    """5–15% drawdown must produce a smaller position than zero drawdown."""
    full = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    reduced = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL",
        current_drawdown=0.07,    # 7% — within taper zone
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    assert reduced < full


def test_max_position_cap_respected(kelly):
    """Result never exceeds max_position_usd regardless of kelly fraction."""
    result = kelly.compute_sized_position(
        base_kelly=1.0, regime="TRENDING_BULL",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=100.0,
    )
    assert result <= 100.0


def test_high_vol_smaller_than_bull(kelly):
    """HIGH_VOL regime must produce a smaller position than TRENDING_BULL."""
    bull = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    high_vol = kelly.compute_sized_position(
        base_kelly=0.5, regime="HIGH_VOL",
        current_drawdown=0.0,
        portfolio_value_usd=1_000.0, max_position_usd=500.0,
    )
    assert high_vol < bull


def test_result_non_negative(kelly):
    """Position size must never be negative."""
    for regime in ("TRENDING_BULL", "TRENDING_BEAR", "HIGH_VOL", "crisis", "LOW_VOL"):
        r = kelly.compute_sized_position(
            base_kelly=0.5, regime=regime,
            current_drawdown=0.0,
            portfolio_value_usd=1_000.0, max_position_usd=500.0,
        )
        assert r >= 0.0, f"Negative position for regime={regime}"
