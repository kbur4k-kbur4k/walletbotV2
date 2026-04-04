"""tests/integration/test_balance_sync.py — Balance sync integration tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.portfolio_manager import PortfolioManager, StrategyType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WALLET = "0xDeadBeef00000000000000000000000000001234"
_RPC = "https://mainnet.base.org"
_CHAIN = 8453


def _make_patches(
    eth_wei=0,
    usdc_raw=0,
    aave_raw=0,
    short_usd=0.0,
    eth_exc=None,
    usdc_exc=None,
    aave_exc=None,
    short_exc=None,
):
    """Return a dict of patch kwargs for the four on-chain calls."""
    eth_mock = AsyncMock(side_effect=eth_exc) if eth_exc else AsyncMock(return_value=eth_wei)
    usdc_mock = AsyncMock(side_effect=usdc_exc) if usdc_exc else AsyncMock(return_value=usdc_raw)
    aave_mock = AsyncMock(side_effect=aave_exc) if aave_exc else AsyncMock(return_value=aave_raw)
    short_mock_val = MagicMock(position_value_usd=short_usd)
    short_mock = MagicMock(side_effect=short_exc) if short_exc else MagicMock(return_value=short_mock_val)
    return {
        "wallet_manager.actions.fetch_native_balance": eth_mock,
        "wallet_manager.actions.fetch_usdc_balance": usdc_mock,
        "wallet_manager.strategies.yield_strategy.get_aave_usdc_balance": aave_mock,
        "wallet_manager.strategies.short_strategy.get_short_position_state": short_mock,
    }


async def _sync(pm: PortfolioManager, eth_price: float = 2000.0, patches: dict | None = None) -> None:
    p = patches or _make_patches()
    with (
        patch("wallet_manager.actions.fetch_native_balance", new=p["wallet_manager.actions.fetch_native_balance"]),
        patch("wallet_manager.actions.fetch_usdc_balance", new=p["wallet_manager.actions.fetch_usdc_balance"]),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance",
              new=p["wallet_manager.strategies.yield_strategy.get_aave_usdc_balance"]),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state",
              p["wallet_manager.strategies.short_strategy.get_short_position_state"]),
    ):
        await pm.sync_from_chain(
            wallet_address=_WALLET,
            rpc_url=_RPC,
            chain_id=_CHAIN,
            eth_price_usd=eth_price,
        )


# ---------------------------------------------------------------------------
# Existing tests (preserved, upgraded with stronger assertions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_from_chain_returns_real_values_not_140():
    """sync_from_chain() must return real on-chain values, not hardcoded $140."""
    pm = PortfolioManager()

    with (
        patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(0.05e18))),  # 0.05 ETH
        patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=int(50e6))),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=int(80e6))),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state",
              return_value=MagicMock(position_value_usd=0.0)),
    ):
        await pm.sync_from_chain(
            wallet_address="0x1234",
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            eth_price_usd=2000.0,
        )

    # 0.05 ETH * $2000 = $100
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(100.0, abs=0.01)
    assert pm.total_value_usd != 140.0


@pytest.mark.asyncio
async def test_sync_from_chain_zero_balance_returns_zero():
    """If wallet holds $0, total_value_usd must display $0.00, not $140.00."""
    pm = PortfolioManager()

    with (
        patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state",
              return_value=MagicMock(position_value_usd=0.0)),
    ):
        await pm.sync_from_chain(
            wallet_address="0x1234",
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            eth_price_usd=2000.0,
        )

    assert pm.total_value_usd == pytest.approx(0.0, abs=0.01)
    assert pm.total_value_usd != 140.0


@pytest.mark.asyncio
async def test_aave_rpc_failure_fallback():
    """When Aave RPC call fails, fall back to last known value, no exception."""
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.YIELD, 77.0)

    with (
        patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance",
              new=AsyncMock(side_effect=Exception("connection refused"))),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state",
              return_value=MagicMock(position_value_usd=0.0)),
    ):
        # Must not raise
        await pm.sync_from_chain(
            wallet_address="0x1234",
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            eth_price_usd=2000.0,
        )

    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(77.0, abs=0.01)


@pytest.mark.asyncio
async def test_get_snapshot_reflects_synced_values():
    """After sync_from_chain(), total_value_usd != 140.0."""
    pm = PortfolioManager()

    with (
        patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(0.1e18))),
        patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=int(200e6))),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=int(100e6))),
        patch("wallet_manager.strategies.short_strategy.get_short_position_state",
              return_value=MagicMock(position_value_usd=0.0)),
    ):
        await pm.sync_from_chain(
            wallet_address="0x1234",
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            eth_price_usd=3000.0,
        )

    assert pm.total_value_usd != 140.0
    # 0.1 ETH * $3000 = $300 + $200 USDC + $100 Aave = $600
    assert pm.total_value_usd == pytest.approx(600.0, abs=0.01)


# ---------------------------------------------------------------------------
# New tests — all four pillars with realistic values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_all_four_pillars_realistic_values():
    """All four pillars (YIELD, ALGO, SHORT, AIRDROP) updated with realistic data."""
    pm = PortfolioManager()

    # Realistic portfolio: 0.2 ETH @ $3500, 300 USDC spot, 500 USDC in Aave, $75 short
    patches = _make_patches(
        eth_wei=int(0.2e18),
        usdc_raw=int(300e6),
        aave_raw=int(500e6),
        short_usd=75.0,
    )
    await _sync(pm, eth_price=3500.0, patches=patches)

    algo_val = pm.get_pillar_value(StrategyType.ALGO)      # 0.2 * 3500 = $700
    airdrop_val = pm.get_pillar_value(StrategyType.AIRDROP)  # $300
    yield_val = pm.get_pillar_value(StrategyType.YIELD)     # $500
    short_val = pm.get_pillar_value(StrategyType.SHORT)     # $75

    assert algo_val == pytest.approx(700.0, abs=0.01)
    assert airdrop_val == pytest.approx(300.0, abs=0.01)
    assert yield_val == pytest.approx(500.0, abs=0.01)
    assert short_val == pytest.approx(75.0, abs=0.01)


@pytest.mark.asyncio
async def test_total_value_equals_sum_of_all_pillar_values():
    """total_value_usd must equal the arithmetic sum of all four pillar values."""
    pm = PortfolioManager()

    patches = _make_patches(
        eth_wei=int(0.15e18),   # 0.15 ETH
        usdc_raw=int(120e6),    # $120 USDC
        aave_raw=int(400e6),    # $400 Aave
        short_usd=50.0,         # $50 short
    )
    await _sync(pm, eth_price=2500.0, patches=patches)

    expected_algo = 0.15 * 2500.0        # $375
    expected_airdrop = 120.0
    expected_yield = 400.0
    expected_short = 50.0
    expected_total = expected_algo + expected_airdrop + expected_yield + expected_short

    assert pm.total_value_usd == pytest.approx(expected_total, abs=0.01)
    # Verify each pillar independently
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(expected_algo, abs=0.01)
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(expected_airdrop, abs=0.01)
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(expected_yield, abs=0.01)
    assert pm.get_pillar_value(StrategyType.SHORT) == pytest.approx(expected_short, abs=0.01)

    # Explicit sum cross-check
    manual_sum = sum(pm.get_pillar_value(st) for st in StrategyType)
    assert pm.total_value_usd == pytest.approx(manual_sum, abs=1e-9)


@pytest.mark.asyncio
async def test_partial_sync_eth_rpc_fails_others_succeed():
    """When ETH RPC fails, ALGO keeps last known value; others update normally."""
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.ALGO, 200.0)   # last known ETH value

    patches = _make_patches(
        eth_exc=ConnectionError("mainnet RPC timeout"),
        usdc_raw=int(80e6),
        aave_raw=int(160e6),
        short_usd=30.0,
    )
    await _sync(pm, eth_price=2000.0, patches=patches)

    # ALGO must retain its last known value
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(200.0, abs=0.01)
    # All other pillars updated successfully
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(80.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(160.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.SHORT) == pytest.approx(30.0, abs=0.01)


@pytest.mark.asyncio
async def test_partial_sync_usdc_rpc_fails_others_succeed():
    """When USDC RPC fails, AIRDROP keeps last known value; others update normally."""
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.AIRDROP, 55.0)

    patches = _make_patches(
        eth_wei=int(0.05e18),
        usdc_exc=OSError("USDC contract unreachable"),
        aave_raw=int(90e6),
        short_usd=0.0,
    )
    await _sync(pm, eth_price=2000.0, patches=patches)

    # AIRDROP retains fallback value
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(55.0, abs=0.01)
    # ALGO: 0.05 * 2000 = $100
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(100.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(90.0, abs=0.01)


@pytest.mark.asyncio
async def test_partial_sync_short_rpc_fails_others_succeed():
    """When SHORT position call fails, SHORT keeps last known value; others update."""
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.SHORT, 42.0)

    patches = _make_patches(
        eth_wei=int(0.1e18),
        usdc_raw=int(100e6),
        aave_raw=int(200e6),
        short_exc=RuntimeError("oracle offline"),
    )
    await _sync(pm, eth_price=2000.0, patches=patches)

    # SHORT retains last-known value
    assert pm.get_pillar_value(StrategyType.SHORT) == pytest.approx(42.0, abs=0.01)
    # ALGO: 0.1 * 2000 = $200
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(200.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(100.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(200.0, abs=0.01)


@pytest.mark.asyncio
async def test_partial_sync_two_rpc_calls_fail_partial_recorded():
    """Two RPC failures: failing pillars retain stale values, successes are fresh."""
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.ALGO, 111.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 22.0)

    patches = _make_patches(
        eth_exc=TimeoutError("eth rpc timeout"),
        usdc_exc=TimeoutError("usdc rpc timeout"),
        aave_raw=int(333e6),
        short_usd=15.0,
    )
    await _sync(pm, eth_price=2000.0, patches=patches)

    # Failing pillars: stale values preserved
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(111.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(22.0, abs=0.01)
    # Successful pillars: fresh values
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(333.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.SHORT) == pytest.approx(15.0, abs=0.01)

    # total_value_usd still reflects a consistent sum of all pillars
    expected_total = 111.0 + 22.0 + 333.0 + 15.0
    assert pm.total_value_usd == pytest.approx(expected_total, abs=0.01)


@pytest.mark.asyncio
async def test_sync_does_not_raise_on_complete_failure():
    """If every single RPC call fails, sync_from_chain() must not raise."""
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.ALGO, 10.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 20.0)
    pm.update_pillar_value(StrategyType.YIELD, 30.0)
    pm.update_pillar_value(StrategyType.SHORT, 5.0)

    patches = _make_patches(
        eth_exc=Exception("all RPCs down"),
        usdc_exc=Exception("all RPCs down"),
        aave_exc=Exception("all RPCs down"),
        short_exc=Exception("all RPCs down"),
    )
    # Must not raise
    await _sync(pm, eth_price=2000.0, patches=patches)

    # All pillars retain last-known stale values
    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(10.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(20.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(30.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.SHORT) == pytest.approx(5.0, abs=0.01)


@pytest.mark.asyncio
async def test_sync_overwrites_stale_values_on_success():
    """A successful sync must overwrite any pre-existing stale pillar value."""
    pm = PortfolioManager()
    # Pre-seed all pillars with stale values
    pm.update_pillar_value(StrategyType.ALGO, 9999.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 9999.0)
    pm.update_pillar_value(StrategyType.YIELD, 9999.0)
    pm.update_pillar_value(StrategyType.SHORT, 9999.0)

    patches = _make_patches(
        eth_wei=int(0.01e18),   # $20 @ $2000
        usdc_raw=int(5e6),      # $5
        aave_raw=int(10e6),     # $10
        short_usd=2.5,
    )
    await _sync(pm, eth_price=2000.0, patches=patches)

    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(20.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.AIRDROP) == pytest.approx(5.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.YIELD) == pytest.approx(10.0, abs=0.01)
    assert pm.get_pillar_value(StrategyType.SHORT) == pytest.approx(2.5, abs=0.01)
    assert pm.total_value_usd == pytest.approx(37.5, abs=0.01)


@pytest.mark.asyncio
async def test_sync_high_eth_price_scales_algo_correctly():
    """ETH price scaling: $10 000/ETH with 0.5 ETH must give $5 000 for ALGO."""
    pm = PortfolioManager()

    patches = _make_patches(
        eth_wei=int(0.5e18),
        usdc_raw=0,
        aave_raw=0,
        short_usd=0.0,
    )
    await _sync(pm, eth_price=10_000.0, patches=patches)

    assert pm.get_pillar_value(StrategyType.ALGO) == pytest.approx(5000.0, abs=0.01)
    assert pm.total_value_usd == pytest.approx(5000.0, abs=0.01)
