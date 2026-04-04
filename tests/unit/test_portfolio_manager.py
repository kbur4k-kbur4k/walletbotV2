"""tests/unit/test_portfolio_manager.py — PortfolioManager tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.portfolio_manager import PortfolioManager, StrategyType


def _pm() -> PortfolioManager:
    return PortfolioManager(monthly_injection_usd=150.0, rebalance_threshold_pct=0.05)


def test_initial_total_value_is_zero():
    pm = _pm()
    assert pm.total_value_usd == 0.0


def test_update_pillar_value():
    pm = _pm()
    pm.update_pillar_value(StrategyType.ALGO, 500.0)
    assert pm.get_pillar_value(StrategyType.ALGO) == 500.0


def test_total_value_sums_all_pillars():
    pm = _pm()
    pm.update_pillar_value(StrategyType.ALGO, 100.0)
    pm.update_pillar_value(StrategyType.YIELD, 200.0)
    pm.update_pillar_value(StrategyType.SHORT, 50.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 150.0)
    assert pm.total_value_usd == 500.0


def test_no_initial_budget_constant():
    """INITIAL_BUDGET_USD must not exist — phantom $140 bug fix verification."""
    import wallet_manager.portfolio_manager as pm_module
    assert not hasattr(pm_module, "INITIAL_BUDGET_USD"), (
        "INITIAL_BUDGET_USD constant still exists in portfolio_manager.py"
    )


@pytest.mark.asyncio
async def test_sync_from_chain_updates_pillars():
    """sync_from_chain() must use real fetched values, not $140."""
    pm = _pm()

    with (
        patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(0.5e18))),
        patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=int(100e6))),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=int(200e6))),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=50.0)),
    ):
        await pm.sync_from_chain(
            wallet_address="0x1234",
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            eth_price_usd=2000.0,
        )

    # 0.5 ETH * $2000 = $1000 for ALGO
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(1000.0, abs=0.01)
    # 100 USDC for AIRDROP
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(100.0, abs=0.01)
    # 200 USDC Aave for YIELD
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(200.0, abs=0.01)
    # Not $140
    assert pm.total_value_usd != 140.0


@pytest.mark.asyncio
async def test_sync_from_chain_fallback_on_rpc_error():
    """When Aave RPC call fails, falls back to last known value without exception."""
    pm = _pm()
    pm.update_pillar_value(StrategyType.YIELD, 99.0)

    with (
        patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance",
              new=AsyncMock(side_effect=Exception("RPC error"))),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state",
              return_value=MagicMock(position_value_usd=0.0)),
    ):
        await pm.sync_from_chain(
            wallet_address="0x1234",
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            eth_price_usd=2000.0,
        )

    # Should fall back to last known value
    assert pm.get_pillar_value(StrategyType.YIELD) == 99.0


# ---------------------------------------------------------------------------
# New tests (ADIM 4.3)
# ---------------------------------------------------------------------------


def test_needs_rebalancing_returns_false_within_threshold():
    """Portfolio within drift threshold should NOT trigger rebalancing."""
    pm = PortfolioManager(monthly_injection_usd=150.0, rebalance_threshold_pct=0.05)
    # Set values very close to 40/35/10/15 weights on $1000 total
    pm.update_pillar_value(StrategyType.YIELD, 400.0)    # 40% exact
    pm.update_pillar_value(StrategyType.ALGO, 350.0)     # 35% exact
    pm.update_pillar_value(StrategyType.SHORT, 100.0)    # 10% exact
    pm.update_pillar_value(StrategyType.AIRDROP, 150.0)  # 15% exact
    assert pm.needs_rebalancing() is False


def test_needs_rebalancing_returns_true_above_threshold():
    """Portfolio that drifts beyond the threshold MUST trigger rebalancing."""
    pm = PortfolioManager(monthly_injection_usd=150.0, rebalance_threshold_pct=0.05)
    # YIELD over-allocated: 50% instead of 40% (10% drift)
    pm.update_pillar_value(StrategyType.YIELD, 500.0)
    pm.update_pillar_value(StrategyType.ALGO, 300.0)
    pm.update_pillar_value(StrategyType.SHORT, 100.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 100.0)
    assert pm.needs_rebalancing() is True


def test_weight_validation_rejects_weights_not_summing_to_one():
    """Settings with portfolio weights != 1.0 must raise ConfigurationError."""
    from wallet_manager.config import Settings
    from wallet_manager.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match=r"weights must sum to 1\.0"):
        Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="legacy",
            garch_alpha=0.10,
            garch_beta=0.85,
            yield_weight=0.50,
            algo_weight=0.30,
            short_weight=0.10,
            airdrop_weight=0.20,  # sum = 1.10
        )


def test_portfolio_manager_uses_config_weights():
    """PortfolioManager should load weights from config."""
    import os
    os.environ["YIELD_WEIGHT"] = "0.40"
    os.environ["ALGO_WEIGHT"] = "0.35"
    os.environ["SHORT_WEIGHT"] = "0.10"
    os.environ["AIRDROP_WEIGHT"] = "0.15"
    pm = PortfolioManager(monthly_injection_usd=150.0, rebalance_threshold_pct=0.05)
    assert abs(pm._weights[StrategyType.YIELD] - 0.40) < 1e-9
    assert abs(pm._weights[StrategyType.ALGO] - 0.35) < 1e-9
