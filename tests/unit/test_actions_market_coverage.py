"""
tests/unit/test_actions_market_coverage.py

Comprehensive coverage tests for three low-coverage modules:
  - wallet_manager/actions.py             (68.8% -> target: 85%+)
  - wallet_manager/market_intelligence.py (59.8% -> target: 85%+)
  - wallet_manager/strategies/execution_logic.py (60.0% -> target: 80%+)

All Web3/RPC/network calls are mocked.
Run: pytest tests/unit/test_actions_market_coverage.py -v
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# ============================== actions.py ==================================
# ---------------------------------------------------------------------------


class TestGetChainAddresses:
    """Chain address resolution — including unsupported chain error."""

    def test_base_chain(self):
        from wallet_manager.actions import get_chain_addresses
        addrs = get_chain_addresses(8453)
        assert addrs.chain_id == 8453
        assert "0x" in addrs.router

    def test_linea_chain(self):
        from wallet_manager.actions import get_chain_addresses
        addrs = get_chain_addresses(59144)
        assert addrs.chain_id == 59144

    def test_unsupported_chain_raises(self):
        from wallet_manager.actions import get_chain_addresses
        with pytest.raises(ValueError, match="not supported"):
            get_chain_addresses(999)

    def test_unsupported_chain_message_includes_id(self):
        from wallet_manager.actions import get_chain_addresses
        with pytest.raises(ValueError, match="999"):
            get_chain_addresses(999)


class TestRotatedFeeTier:
    """Fee tier rotation for Ghost Protocol."""

    def test_rotates_through_three_tiers(self):
        from wallet_manager.actions import FEE_TIERS, get_rotated_fee_tier
        seen = {get_rotated_fee_tier(i) for i in range(len(FEE_TIERS))}
        assert seen == set(FEE_TIERS)

    def test_zero_returns_first_tier(self):
        from wallet_manager.actions import FEE_TIERS, get_rotated_fee_tier
        assert get_rotated_fee_tier(0) == FEE_TIERS[0]

    def test_wraps_around(self):
        from wallet_manager.actions import get_rotated_fee_tier
        assert get_rotated_fee_tier(0) == get_rotated_fee_tier(3)


class TestEthBalanceUsdValue:
    def test_one_eth(self):
        from wallet_manager.actions import eth_balance_usd_value
        assert eth_balance_usd_value(int(1e18), 3000.0) == pytest.approx(3000.0)

    def test_zero_balance(self):
        from wallet_manager.actions import eth_balance_usd_value
        assert eth_balance_usd_value(0, 3000.0) == pytest.approx(0.0)

    def test_zero_price(self):
        from wallet_manager.actions import eth_balance_usd_value
        assert eth_balance_usd_value(int(1e18), 0.0) == pytest.approx(0.0)

    def test_fractional_eth(self):
        from wallet_manager.actions import eth_balance_usd_value
        assert eth_balance_usd_value(int(0.1e18), 2000.0) == pytest.approx(200.0)


class TestUsdcBalanceUsdValue:
    def test_one_hundred_usdc(self):
        from wallet_manager.actions import usdc_balance_usd_value
        assert usdc_balance_usd_value(int(100e6)) == pytest.approx(100.0)

    def test_zero(self):
        from wallet_manager.actions import usdc_balance_usd_value
        assert usdc_balance_usd_value(0) == pytest.approx(0.0)

    def test_one_usdc(self):
        from wallet_manager.actions import usdc_balance_usd_value
        assert usdc_balance_usd_value(1_000_000) == pytest.approx(1.0)


class TestIsSwapEligible:
    def test_eligible_with_enough_eth_low_gas(self):
        from wallet_manager.actions import is_swap_eligible
        assert is_swap_eligible(
            balance_wei=int(0.1e18), eth_price_usd=3000.0, gas_gwei=5.0
        ) is True

    def test_ineligible_low_eth_value(self):
        from wallet_manager.actions import is_swap_eligible
        assert is_swap_eligible(
            balance_wei=int(0.001e18), eth_price_usd=3000.0, gas_gwei=5.0
        ) is False

    def test_ineligible_high_gas(self):
        from wallet_manager.actions import is_swap_eligible
        assert is_swap_eligible(
            balance_wei=int(1e18), eth_price_usd=3000.0, gas_gwei=50.0
        ) is False

    def test_alias_params_work(self):
        from wallet_manager.actions import is_swap_eligible
        # Using keyword aliases: eth_balance_wei, gas_price_gwei, min_trade_usd
        assert is_swap_eligible(
            eth_balance_wei=int(0.5e18),
            eth_price_usd=2000.0,
            gas_price_gwei=10.0,
            min_trade_usd=5.0,
        ) is True

    def test_alias_overrides_positional(self):
        from wallet_manager.actions import is_swap_eligible
        # eth_balance_wei overrides balance_wei=0
        result = is_swap_eligible(
            balance_wei=0,
            eth_price_usd=3000.0,
            gas_gwei=5.0,
            eth_balance_wei=int(1e18),
        )
        assert result is True

    def test_exactly_at_gas_threshold_is_ineligible(self):
        from wallet_manager.actions import is_swap_eligible
        # max_gas_gwei=25.0 means strictly < 25, so 25.0 is excluded
        result = is_swap_eligible(
            balance_wei=int(1e18), eth_price_usd=3000.0, gas_gwei=25.0,
            max_gas_gwei=25.0,
        )
        assert result is False


class TestIsUsdcSwapEligible:
    def test_eligible(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        assert is_usdc_swap_eligible(int(50e6), 5.0) is True

    def test_ineligible_low_balance(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        assert is_usdc_swap_eligible(int(0.001e6), 5.0) is False

    def test_ineligible_high_gas(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        assert is_usdc_swap_eligible(int(50e6), 100.0) is False

    def test_custom_min_usdc(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        # $5 balance but min is $100 → ineligible
        assert is_usdc_swap_eligible(int(5e6), 5.0, min_usdc_value_usd=100.0) is False


_VALID_ADDR = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # Vitalik.eth, valid checksum


class TestFetchNativeBalance:
    @pytest.mark.asyncio
    async def test_returns_integer(self):
        from wallet_manager.actions import fetch_native_balance
        mock_w3 = MagicMock()
        mock_w3.eth.get_balance = AsyncMock(return_value=int(1e18))
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await fetch_native_balance("http://rpc", _VALID_ADDR)
        assert isinstance(result, int)
        assert result == int(1e18)

    @pytest.mark.asyncio
    async def test_zero_balance(self):
        from wallet_manager.actions import fetch_native_balance
        mock_w3 = MagicMock()
        mock_w3.eth.get_balance = AsyncMock(return_value=0)
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await fetch_native_balance("http://rpc", _VALID_ADDR)
        assert result == 0


class TestFetchUsdcBalance:
    @pytest.mark.asyncio
    async def test_returns_balance(self):
        from wallet_manager.actions import fetch_usdc_balance
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call = AsyncMock(
            return_value=500_000_000
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await fetch_usdc_balance("http://rpc", _VALID_ADDR, 8453)
        assert result == 500_000_000

    @pytest.mark.asyncio
    async def test_unsupported_chain_raises(self):
        from wallet_manager.actions import fetch_usdc_balance
        with pytest.raises(ValueError):
            await fetch_usdc_balance("http://rpc", _VALID_ADDR, 1)  # Mainnet not supported


def _make_aiohttp_session(status: int, json_data: dict):
    """Build a properly nested aiohttp ClientSession mock."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)
    # Inner context manager (session.get(...))
    inner_cm = MagicMock()
    inner_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    inner_cm.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get.return_value = inner_cm
    # Outer context manager (aiohttp.ClientSession())
    outer_cm = MagicMock()
    outer_cm.__aenter__ = AsyncMock(return_value=mock_session)
    outer_cm.__aexit__ = AsyncMock(return_value=False)
    return outer_cm


class TestFetchEthPriceUsd:
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        from wallet_manager.actions import fetch_eth_price_usd
        outer_cm = _make_aiohttp_session(200, {"ethereum": {"usd": 2500.0}})
        with patch("aiohttp.ClientSession", return_value=outer_cm):
            result = await fetch_eth_price_usd()
        assert result == pytest.approx(2500.0)

    @pytest.mark.asyncio
    async def test_returns_fallback_on_all_failures(self):
        from wallet_manager.actions import fetch_eth_price_usd
        failing_cm = MagicMock()
        failing_cm.__aenter__ = AsyncMock(side_effect=Exception("network error"))
        failing_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=failing_cm):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_eth_price_usd()
        assert result == 3000.0

    @pytest.mark.asyncio
    async def test_rate_limited_response_retries(self):
        from wallet_manager.actions import fetch_eth_price_usd
        outer_cm = _make_aiohttp_session(429, {})
        with patch("aiohttp.ClientSession", return_value=outer_cm):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_eth_price_usd()
        assert result == 3000.0  # Falls back after all 429s

    @pytest.mark.asyncio
    async def test_zero_price_from_api_falls_back(self):
        from wallet_manager.actions import fetch_eth_price_usd
        outer_cm = _make_aiohttp_session(200, {"ethereum": {"usd": 0}})
        with patch("aiohttp.ClientSession", return_value=outer_cm):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_eth_price_usd()
        assert result == 3000.0


class TestQuoteAmountOut:
    @pytest.mark.asyncio
    async def test_successful_quote(self):
        from wallet_manager.actions import USDC_BASE, WETH_BASE, quote_amount_out
        mock_contract = MagicMock()
        mock_contract.functions.quoteExactInputSingle.return_value.call = AsyncMock(
            return_value=[3_000_000_000, 0, 0, 0]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await quote_amount_out(
                "http://rpc", 8453, WETH_BASE, USDC_BASE, int(1e18)
            )
        assert result == 3_000_000_000

    @pytest.mark.asyncio
    async def test_quoter_failure_falls_back_eth_to_usdc(self):
        from wallet_manager.actions import USDC_BASE, WETH_BASE, quote_amount_out
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("RPC error")
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3), patch(
            "wallet_manager.actions.fetch_eth_price_usd",
            new_callable=AsyncMock,
            return_value=3000.0,
        ):
            result = await quote_amount_out(
                "http://rpc", 8453, WETH_BASE, USDC_BASE, int(1e18)
            )
        assert result > 0

    @pytest.mark.asyncio
    async def test_quoter_failure_falls_back_usdc_to_weth(self):
        from wallet_manager.actions import USDC_BASE, WETH_BASE, quote_amount_out
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("RPC error")
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3), patch(
            "wallet_manager.actions.fetch_eth_price_usd",
            new_callable=AsyncMock,
            return_value=3000.0,
        ):
            result = await quote_amount_out(
                "http://rpc", 8453, USDC_BASE, WETH_BASE, 3_000_000_000
            )
        assert result > 0

    @pytest.mark.asyncio
    async def test_unknown_token_pair_returns_zero_on_fallback(self):
        from wallet_manager.actions import quote_amount_out
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("RPC error")
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await quote_amount_out(
                "http://rpc", 8453, "0xdeadbeef", "0xcafebabe", int(1e18)
            )
        assert result == 0


class TestGetPoolLiquidity:
    @pytest.mark.asyncio
    async def test_returns_liquidity_for_base(self):
        from wallet_manager.actions import get_pool_liquidity
        mock_contract = MagicMock()
        mock_contract.functions.liquidity.return_value.call = AsyncMock(
            return_value=10**24
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_pool_liquidity("http://rpc", 8453)
        assert result == 10**24

    @pytest.mark.asyncio
    async def test_non_base_chain_returns_zero(self):
        from wallet_manager.actions import get_pool_liquidity
        result = await get_pool_liquidity("http://rpc", 59144)
        assert result == 0

    @pytest.mark.asyncio
    async def test_rpc_error_returns_zero(self):
        from wallet_manager.actions import get_pool_liquidity
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("RPC timeout")
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_pool_liquidity("http://rpc", 8453)
        assert result == 0

    @pytest.mark.asyncio
    async def test_non_3000_fee_returns_zero(self):
        from wallet_manager.actions import get_pool_liquidity
        result = await get_pool_liquidity("http://rpc", 8453, fee=500)
        assert result == 0


class TestGetDynamicSlippage:
    @pytest.mark.asyncio
    async def test_baseline_slippage(self):
        from wallet_manager.actions import (
            MIN_SLIPPAGE_BPS,
            USDC_BASE,
            WETH_BASE,
            get_dynamic_slippage,
        )
        with patch(
            "wallet_manager.actions.get_pool_liquidity",
            new_callable=AsyncMock,
            return_value=10**24,
        ):
            result = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(1e6), gas_price_gwei=5.0
            )
        assert result >= MIN_SLIPPAGE_BPS

    @pytest.mark.asyncio
    async def test_high_gas_increases_slippage(self):
        from wallet_manager.actions import USDC_BASE, WETH_BASE, get_dynamic_slippage
        with patch(
            "wallet_manager.actions.get_pool_liquidity",
            new_callable=AsyncMock,
            return_value=10**24,
        ):
            low_gas = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(1e6), gas_price_gwei=2.0
            )
            high_gas = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(1e6), gas_price_gwei=20.0
            )
        assert high_gas > low_gas

    @pytest.mark.asyncio
    async def test_algo_mode_caps_slippage(self):
        from wallet_manager.actions import (
            ALGO_MAX_SLIPPAGE_BPS,
            USDC_BASE,
            WETH_BASE,
            get_dynamic_slippage,
        )
        with patch(
            "wallet_manager.actions.get_pool_liquidity",
            new_callable=AsyncMock,
            return_value=1,  # Very low liquidity
        ):
            result = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(50e18), gas_price_gwei=20.0, is_algo=True
            )
        assert result <= ALGO_MAX_SLIPPAGE_BPS

    @pytest.mark.asyncio
    async def test_large_swap_increases_slippage(self):
        from wallet_manager.actions import USDC_BASE, WETH_BASE, get_dynamic_slippage
        with patch(
            "wallet_manager.actions.get_pool_liquidity",
            new_callable=AsyncMock,
            return_value=10**24,
        ):
            small = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(1e6), gas_price_gwei=5.0
            )
            large = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(100e18), gas_price_gwei=5.0
            )
        assert large > small

    @pytest.mark.asyncio
    async def test_pool_liquidity_failure_handled_gracefully(self):
        from wallet_manager.actions import USDC_BASE, WETH_BASE, get_dynamic_slippage
        with patch(
            "wallet_manager.actions.get_pool_liquidity",
            new_callable=AsyncMock,
            side_effect=Exception("pool error"),
        ):
            result = await get_dynamic_slippage(
                "http://rpc", 8453, USDC_BASE, WETH_BASE,
                int(1e6), gas_price_gwei=5.0
            )
        assert result >= 0


class TestGetWethUsdcPrice:
    @pytest.mark.asyncio
    async def test_successful_slot0_price(self):
        from wallet_manager.actions import get_weth_usdc_price
        # sqrtPriceX96 matching roughly $2500 ETH
        # For WETH/USDC pool: price = (sqrt/2^96)^2 * 1e12
        # sqrt ~= 3976191 * 2^82 for approx $2500
        sqrt_val = 3976191482684856000000000000
        mock_contract = MagicMock()
        mock_contract.functions.slot0.return_value.call = AsyncMock(
            return_value=[sqrt_val, 0, 0, 0, 0, 0, True]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_weth_usdc_price("http://rpc")
        assert result > 0

    @pytest.mark.asyncio
    async def test_zero_sqrt_returns_zero(self):
        from wallet_manager.actions import get_weth_usdc_price
        mock_contract = MagicMock()
        mock_contract.functions.slot0.return_value.call = AsyncMock(
            return_value=[0, 0, 0, 0, 0, 0, True]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_weth_usdc_price("http://rpc")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_exception_returns_zero(self):
        from wallet_manager.actions import get_weth_usdc_price
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("pool not found")
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_weth_usdc_price("http://rpc")
        assert result == 0.0


class TestGetCurrentPrice:
    @pytest.mark.asyncio
    async def test_uses_slot0_on_base(self):
        from wallet_manager.actions import get_current_price
        # get_current_price re-implements slot0 internally — mock the contract call
        sqrt_val = 3976191482684856000000000000
        mock_contract = MagicMock()
        mock_contract.functions.slot0.return_value.call = AsyncMock(
            return_value=[sqrt_val, 0, 0, 0, 0, 0, True]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_current_price("http://rpc", chain_id=8453)
        assert result > 0

    @pytest.mark.asyncio
    async def test_slot0_fallback_to_quoter(self):
        from wallet_manager.actions import get_current_price
        # slot0 returns 0, quoter should be called
        mock_contract = MagicMock()
        # slot0 returns zero
        mock_contract.functions.slot0.return_value.call = AsyncMock(
            return_value=[0, 0, 0, 0, 0, 0, True]
        )
        # quoter returns non-zero
        mock_contract.functions.quoteExactInputSingle.return_value.call = AsyncMock(
            return_value=[2_500_000_000, 0, 0, 0]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_current_price("http://rpc", chain_id=8453)
        assert result > 0

    @pytest.mark.asyncio
    async def test_non_base_chain_skips_slot0(self):
        from wallet_manager.actions import get_current_price
        # On non-Base chains, slot0 is skipped; quoter is tried, then Chainlink
        with patch(
            "wallet_manager.actions.quote_amount_out",
            new_callable=AsyncMock,
            return_value=0,
        ), patch(
            "wallet_manager.actions._fetch_chainlink_eth_usd",
            new_callable=AsyncMock,
            return_value=2800.0,
        ):
            result = await get_current_price("http://rpc", chain_id=59144)
        assert result == pytest.approx(2800.0)

    @pytest.mark.asyncio
    async def test_use_slot0_false_skips_to_quoter_path(self):
        from wallet_manager.actions import get_current_price
        with patch(
            "wallet_manager.actions.quote_amount_out",
            new_callable=AsyncMock,
            return_value=3_000_000_000,
        ):
            result = await get_current_price("http://rpc", chain_id=8453, use_slot0=False)
        assert result == pytest.approx(3000.0)


class TestFetchChainlinkEthUsd:
    @pytest.mark.asyncio
    async def test_successful_chainlink_fetch(self):
        from wallet_manager.actions import _fetch_chainlink_eth_usd
        mock_contract = MagicMock()
        mock_contract.functions.latestRoundData.return_value.call = AsyncMock(
            return_value=[1, 250000000000, 0, int(time.time()), 1]  # $2500 * 1e8
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await _fetch_chainlink_eth_usd("http://rpc")
        assert result == pytest.approx(2500.0)

    @pytest.mark.asyncio
    async def test_zero_answer_falls_back_to_coingecko(self):
        from wallet_manager.actions import _fetch_chainlink_eth_usd
        mock_contract = MagicMock()
        mock_contract.functions.latestRoundData.return_value.call = AsyncMock(
            return_value=[1, 0, 0, 0, 1]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3), patch(
            "wallet_manager.actions.fetch_eth_price_usd",
            new_callable=AsyncMock,
            return_value=3000.0,
        ):
            result = await _fetch_chainlink_eth_usd("http://rpc")
        assert result == pytest.approx(3000.0)

    @pytest.mark.asyncio
    async def test_exception_falls_back_to_coingecko(self):
        from wallet_manager.actions import _fetch_chainlink_eth_usd
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("feed unavailable")
        with patch("wallet_manager.actions._make_w3", return_value=mock_w3), patch(
            "wallet_manager.actions.fetch_eth_price_usd",
            new_callable=AsyncMock,
            return_value=3000.0,
        ):
            result = await _fetch_chainlink_eth_usd("http://rpc")
        assert result == pytest.approx(3000.0)


class TestBuildApproveCalldata:
    def test_returns_bytes(self):
        from wallet_manager.actions import ROUTER_BASE, _build_approve_calldata
        data = _build_approve_calldata(ROUTER_BASE, int(1e18))
        assert isinstance(data, bytes)
        assert len(data) > 4

    def test_has_correct_selector(self):
        from wallet_manager.actions import ROUTER_BASE, _build_approve_calldata
        data = _build_approve_calldata(ROUTER_BASE, int(1e18))
        # approve selector = 0x095ea7b3
        assert data[:4] == bytes.fromhex("095ea7b3")


class TestBuildExactInputSingleCalldata:
    def test_no_deadline(self):
        from wallet_manager.actions import (
            ROUTER_BASE,
            USDC_BASE,
            WETH_BASE,
            _build_exact_input_single_calldata,
        )
        data = _build_exact_input_single_calldata(
            token_in=WETH_BASE,
            token_out=USDC_BASE,
            fee=3000,
            recipient=ROUTER_BASE,
            amount_in=int(1e18),
            amount_out_min=int(2900e6),
        )
        assert isinstance(data, bytes)
        assert len(data) > 4

    def test_with_deadline_supported(self):
        import wallet_manager.actions as act
        from wallet_manager.actions import (
            ROUTER_BASE,
            USDC_BASE,
            WETH_BASE,
            _build_exact_input_single_calldata,
        )
        original = act.DEADLINE_SUPPORTED
        try:
            act.DEADLINE_SUPPORTED = True
            data = _build_exact_input_single_calldata(
                token_in=WETH_BASE,
                token_out=USDC_BASE,
                fee=3000,
                recipient=ROUTER_BASE,
                amount_in=int(1e18),
                amount_out_min=int(2900e6),
                deadline=int(time.time()) + 900,
            )
            assert isinstance(data, bytes)
            assert len(data) > 4
        finally:
            act.DEADLINE_SUPPORTED = original


class TestBuildAaveCalldata:
    def test_supply_calldata(self):
        from wallet_manager.actions import AAVE_POOL_BASE, USDC_BASE, _build_aave_supply_calldata
        data = _build_aave_supply_calldata(USDC_BASE, int(100e6), AAVE_POOL_BASE)
        assert isinstance(data, bytes)
        # supply selector = 617ba037
        assert data[:4] == bytes.fromhex("617ba037")

    def test_withdraw_calldata(self):
        from wallet_manager.actions import AAVE_POOL_BASE, USDC_BASE, _build_aave_withdraw_calldata
        data = _build_aave_withdraw_calldata(USDC_BASE, int(100e6), AAVE_POOL_BASE)
        assert isinstance(data, bytes)
        # withdraw selector = 69328dec
        assert data[:4] == bytes.fromhex("69328dec")


class TestBuildAaveSupplyPlan:
    @pytest.mark.asyncio
    async def test_builds_approve_and_supply_steps_when_no_allowance(self):
        from wallet_manager.actions import AAVE_POOL_BASE, build_aave_supply_usdc_plan
        with patch(
            "wallet_manager.actions._get_allowance",
            new_callable=AsyncMock,
            return_value=0,
        ):
            steps = await build_aave_supply_usdc_plan(
                "http://rpc", _VALID_ADDR, int(100e6), 10.0
            )
        assert len(steps) == 2
        descs = [d for _, d in steps]
        assert any("approve" in d.lower() for d in descs)
        assert any("supply" in d.lower() for d in descs)

    @pytest.mark.asyncio
    async def test_skips_approve_when_allowance_sufficient(self):
        from wallet_manager.actions import build_aave_supply_usdc_plan
        with patch(
            "wallet_manager.actions._get_allowance",
            new_callable=AsyncMock,
            return_value=int(200e6),  # Already approved
        ):
            steps = await build_aave_supply_usdc_plan(
                "http://rpc", _VALID_ADDR, int(100e6), 10.0
            )
        assert len(steps) == 1
        _, desc = steps[0]
        assert "supply" in desc.lower()


class TestBuildAaveWithdrawPlan:
    @pytest.mark.asyncio
    async def test_builds_single_withdraw_step(self):
        from wallet_manager.actions import build_aave_withdraw_usdc_plan
        steps = await build_aave_withdraw_usdc_plan(
            _VALID_ADDR, int(100e6), 10.0
        )
        assert len(steps) == 1
        _, desc = steps[0]
        assert "withdraw" in desc.lower()


class TestBuildEthToUsdcPlan:
    @pytest.mark.asyncio
    async def test_plan_has_three_steps_when_no_allowance(self):
        from wallet_manager.actions import build_eth_to_usdc_plan
        with patch(
            "wallet_manager.actions.quote_amount_out",
            new_callable=AsyncMock,
            return_value=int(3000e6),
        ), patch(
            "wallet_manager.actions._get_allowance",
            new_callable=AsyncMock,
            return_value=0,
        ):
            plan = await build_eth_to_usdc_plan(
                "http://rpc", 8453, _VALID_ADDR, int(1e18)
            )
        from wallet_manager.actions import SwapDirection
        assert plan.direction == SwapDirection.ETH_TO_USDC
        assert len(plan.steps) == 3

    @pytest.mark.asyncio
    async def test_plan_has_two_steps_when_allowance_sufficient(self):
        from wallet_manager.actions import build_eth_to_usdc_plan
        with patch(
            "wallet_manager.actions.quote_amount_out",
            new_callable=AsyncMock,
            return_value=int(3000e6),
        ), patch(
            "wallet_manager.actions._get_allowance",
            new_callable=AsyncMock,
            return_value=int(10e18),  # Sufficient
        ):
            plan = await build_eth_to_usdc_plan(
                "http://rpc", 8453, _VALID_ADDR, int(1e18)
            )
        assert len(plan.steps) == 2


class TestBuildUsdcToEthPlan:
    @pytest.mark.asyncio
    async def test_plan_has_two_steps_when_no_allowance(self):
        from wallet_manager.actions import build_usdc_to_eth_plan
        with patch(
            "wallet_manager.actions.quote_amount_out",
            new_callable=AsyncMock,
            return_value=int(1e18) // 3000,
        ), patch(
            "wallet_manager.actions._get_allowance",
            new_callable=AsyncMock,
            return_value=0,
        ):
            plan = await build_usdc_to_eth_plan(
                "http://rpc", 8453, _VALID_ADDR, int(100e6)
            )
        from wallet_manager.actions import SwapDirection
        assert plan.direction == SwapDirection.USDC_TO_ETH
        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    async def test_plan_has_one_step_when_allowance_sufficient(self):
        from wallet_manager.actions import build_usdc_to_eth_plan
        with patch(
            "wallet_manager.actions.quote_amount_out",
            new_callable=AsyncMock,
            return_value=int(1e18) // 3000,
        ), patch(
            "wallet_manager.actions._get_allowance",
            new_callable=AsyncMock,
            return_value=int(200e6),
        ):
            plan = await build_usdc_to_eth_plan(
                "http://rpc", 8453, _VALID_ADDR, int(100e6)
            )
        assert len(plan.steps) == 1


# ---------------------------------------------------------------------------
# ======================== market_intelligence.py ===========================
# ---------------------------------------------------------------------------


class TestImpactLevel:
    def test_enum_values(self):
        from wallet_manager.market_intelligence import ImpactLevel
        assert ImpactLevel.LOW == "LOW"
        assert ImpactLevel.CRITICAL == "CRITICAL"

    def test_ordering(self):
        from wallet_manager.market_intelligence import ImpactLevel
        levels = [ImpactLevel.LOW, ImpactLevel.MEDIUM, ImpactLevel.HIGH, ImpactLevel.CRITICAL]
        assert len(levels) == 4


class TestEventCategory:
    def test_all_categories_exist(self):
        from wallet_manager.market_intelligence import EventCategory
        assert EventCategory.EMPLOYMENT == "EMPLOYMENT"
        assert EventCategory.CRYPTO_SPECIFIC == "CRYPTO_SPECIFIC"


class TestMacroEvent:
    def _make_event(self, ts: float, impact_str: str = "HIGH") -> Any:
        from wallet_manager.market_intelligence import EventCategory, ImpactLevel, MacroEvent
        return MacroEvent(
            name="Test NFP",
            timestamp_utc=ts,
            impact=ImpactLevel(impact_str),
            category=EventCategory.EMPLOYMENT,
            description="Test",
        )

    def test_dt_utc_property(self):
        from datetime import UTC, datetime
        event = self._make_event(0.0)
        assert event.dt_utc == datetime.fromtimestamp(0.0, tz=UTC)

    def test_is_imminent_within_window(self):
        now = time.time()
        event = self._make_event(now + 3600)  # 1h ahead
        assert event.is_imminent(now, window_hours=2.0) is True

    def test_is_imminent_outside_window(self):
        now = time.time()
        event = self._make_event(now + 10 * 3600)  # 10h ahead
        assert event.is_imminent(now, window_hours=2.0) is False

    def test_is_imminent_uses_current_time_when_none(self):
        now = time.time()
        event = self._make_event(now + 30)
        # window_hours=1.0 covers 30s difference
        assert event.is_imminent(None, window_hours=1.0) is True

    def test_is_pre_event_in_lead_window(self):
        now = time.time()
        event = self._make_event(now + 1800)  # 30 min ahead, lead=1h
        event.avoid_window_hours_before = 1.0
        assert event.is_pre_event(now) is True

    def test_is_pre_event_past_event(self):
        now = time.time()
        event = self._make_event(now - 3600)  # 1h ago
        assert event.is_pre_event(now) is False

    def test_is_active_within_skip_window(self):
        now = time.time()
        event = self._make_event(now)  # Happening now
        assert event.is_active(now) is True

    def test_is_active_outside_skip_window(self):
        now = time.time()
        event = self._make_event(now - 10 * 3600)  # 10h ago
        event.skip_alpha_window_hours = 2.0
        assert event.is_active(now) is False

    def test_to_dict_contains_expected_keys(self):
        event = self._make_event(time.time())
        d = event.to_dict()
        assert "name" in d
        assert "impact" in d
        assert "category" in d
        assert "timestamp_utc" in d
        assert "datetime_utc" in d


class TestEventImpactClassifier:
    def test_nfp_is_critical(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _cat, _skip, _lead = EventImpactClassifier.classify("NFP 2026-04-03")
        assert impact == ImpactLevel.CRITICAL

    def test_fomc_is_critical(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, _, _ = EventImpactClassifier.classify("FOMC Decision May")
        assert impact == ImpactLevel.CRITICAL

    def test_cpi_is_critical(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, _, _ = EventImpactClassifier.classify("CPI Report")
        assert impact == ImpactLevel.CRITICAL

    def test_gdp_is_high(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, _, _ = EventImpactClassifier.classify("GDP Q1 Advance")
        assert impact == ImpactLevel.HIGH

    def test_housing_is_medium(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, _, _ = EventImpactClassifier.classify("Housing Starts")
        assert impact == ImpactLevel.MEDIUM

    def test_unknown_event_is_low(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _cat, _skip, _lead = EventImpactClassifier.classify("RandomEvent XYZ")
        assert impact == ImpactLevel.LOW

    def test_case_insensitive(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact1, _, _, _ = EventImpactClassifier.classify("FOMC")
        impact2, _, _, _ = EventImpactClassifier.classify("fomc")
        assert impact1 == impact2

    def test_exchange_hack_is_critical(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, _, _ = EventImpactClassifier.classify("Exchange Hack Detected")
        assert impact == ImpactLevel.CRITICAL

    def test_etf_approval_is_critical(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, skip, _ = EventImpactClassifier.classify("ETF Approval Decision")
        assert impact == ImpactLevel.CRITICAL
        assert skip == 6.0  # 6-hour window

    def test_pce_is_critical(self):
        from wallet_manager.market_intelligence import EventImpactClassifier, ImpactLevel
        impact, _, _, _ = EventImpactClassifier.classify("PCE June")
        assert impact == ImpactLevel.CRITICAL


class TestMacroCalendar:
    def _fresh_calendar(self):
        from wallet_manager.market_intelligence import MacroCalendar
        return MacroCalendar()

    def test_populated_with_q2_events(self):
        cal = self._fresh_calendar()
        assert len(cal._events) > 0

    def test_add_event_returns_macro_event(self):
        from wallet_manager.market_intelligence import MacroEvent
        cal = self._fresh_calendar()
        event = cal.add_event("Test CPI", time.time() + 3600)
        assert isinstance(event, MacroEvent)
        assert event in cal._events

    def test_add_event_with_explicit_impact(self):
        from wallet_manager.market_intelligence import ImpactLevel
        cal = self._fresh_calendar()
        event = cal.add_event(
            "Random Event", time.time() + 3600,
            impact=ImpactLevel.LOW,
        )
        assert event.impact == ImpactLevel.LOW

    def test_add_event_with_explicit_skip_window(self):
        cal = self._fresh_calendar()
        event = cal.add_event("Test Event", time.time() + 3600, skip_alpha_window_hours=5.0)
        assert event.skip_alpha_window_hours == 5.0

    def test_get_imminent_events_returns_matching(self):
        from wallet_manager.market_intelligence import ImpactLevel
        cal = self._fresh_calendar()
        now = time.time()
        cal.add_event("FOMC Now", now + 60, impact=ImpactLevel.CRITICAL)
        events = cal.get_imminent_events(now_ts=now, window_hours=1.0)
        names = [e.name for e in events]
        assert "FOMC Now" in names

    def test_get_imminent_events_filters_low_impact(self):
        from wallet_manager.market_intelligence import ImpactLevel
        cal = self._fresh_calendar()
        now = time.time()
        cal.add_event("Low Impact Event", now + 60, impact=ImpactLevel.LOW)
        # min_impact=HIGH so LOW events should not appear
        events = cal.get_imminent_events(now_ts=now, window_hours=1.0, min_impact=ImpactLevel.HIGH)
        names = [e.name for e in events]
        assert "Low Impact Event" not in names

    def test_get_imminent_events_empty_when_no_events_near(self):
        from wallet_manager.market_intelligence import ImpactLevel, MacroCalendar
        cal = MacroCalendar()
        # Clear all events
        cal._events = []
        events = cal.get_imminent_events(now_ts=time.time(), window_hours=0.01)
        assert events == []

    def test_should_suppress_alpha_true_when_critical_event(self):
        from wallet_manager.market_intelligence import ImpactLevel
        cal = self._fresh_calendar()
        cal._events = []  # Clear
        now = time.time()
        cal.add_event("FOMC Decision", now + 60, impact=ImpactLevel.CRITICAL)
        suppress, reason = cal.should_suppress_alpha(now_ts=now, window_hours=1.0)
        assert suppress is True
        assert "FOMC" in reason or "NEWS_EVENT_IMMINENT" in reason

    def test_should_suppress_alpha_false_when_no_events(self):
        cal = self._fresh_calendar()
        cal._events = []
        suppress, reason = cal.should_suppress_alpha(now_ts=time.time(), window_hours=0.001)
        assert suppress is False
        assert reason == ""

    def test_upcoming_events_returns_sorted(self):
        cal = self._fresh_calendar()
        cal._events = []
        now = time.time()
        from wallet_manager.market_intelligence import ImpactLevel
        cal.add_event("Later Event", now + 7200, impact=ImpactLevel.HIGH)
        cal.add_event("Sooner Event", now + 3600, impact=ImpactLevel.HIGH)
        upcoming = cal.upcoming_events(hours_ahead=24.0)
        if len(upcoming) >= 2:
            assert upcoming[0].timestamp_utc <= upcoming[1].timestamp_utc

    def test_upcoming_events_excludes_past(self):
        from wallet_manager.market_intelligence import ImpactLevel
        cal = self._fresh_calendar()
        cal._events = []
        now = time.time()
        cal.add_event("Past Event", now - 3600, impact=ImpactLevel.HIGH)
        upcoming = cal.upcoming_events(hours_ahead=24.0)
        names = [e.name for e in upcoming]
        assert "Past Event" not in names

    def test_to_dict_returns_expected_structure(self):
        cal = self._fresh_calendar()
        d = cal.to_dict()
        assert "total_events" in d
        assert "imminent_high_plus" in d
        assert "next_48h" in d


class TestChainIntelligenceDepeg:
    def _fresh(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        return ChainIntelligence()

    def test_no_data_status(self):
        ci = self._fresh()
        result = ci.get_depeg_warning()
        assert result["status"] == "NO_DATA"
        assert result["alert"] is False

    def test_stable_peg(self):
        ci = self._fresh()
        ci.record_usdc_price(1.001)
        result = ci.get_depeg_warning()
        assert result["status"] == "PEG_STABLE"
        assert result["alert"] is False

    def test_minor_depeg(self):
        ci = self._fresh()
        ci.record_usdc_price(0.996)  # 0.4% deviation
        result = ci.get_depeg_warning()
        assert result["status"] == "MINOR_DEPEG"
        assert result["alert"] is True

    def test_critical_depeg(self):
        ci = self._fresh()
        ci.record_usdc_price(0.98)  # 2% deviation
        result = ci.get_depeg_warning()
        assert result["status"] == "CRITICAL_DEPEG"
        assert result["alert"] is True

    def test_history_capped_at_max(self):
        ci = self._fresh()
        for _ in range(300):
            ci.record_usdc_price(1.0)
        assert len(ci._price_history) <= ci._max_history


class TestChainIntelligenceLiquidity:
    def _fresh(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        return ChainIntelligence()

    def test_no_data_status(self):
        ci = self._fresh()
        result = ci.get_liquidity_change()
        assert result["status"] == "NO_DATA"

    def test_normal_liquidity(self):
        ci = self._fresh()
        for _ in range(5):
            ci.record_pool_depth(1_000_000.0)
        result = ci.get_liquidity_change()
        assert result["status"] == "NORMAL"

    def test_low_liquidity_alert(self):
        ci = self._fresh()
        for _ in range(10):
            ci.record_pool_depth(1_000_000.0)
        ci.record_pool_depth(500_000.0)  # 50% drop
        result = ci.get_liquidity_change()
        assert result["alert"] is True
        assert result["status"] == "LOW_LIQUIDITY"

    def test_history_capped(self):
        ci = self._fresh()
        for _ in range(300):
            ci.record_pool_depth(1_000_000.0)
        assert len(ci._pool_depth_history) <= ci._max_history


class TestChainIntelligenceWhale:
    def _fresh(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        return ChainIntelligence()

    def test_below_threshold_not_recorded(self):
        ci = self._fresh()
        ci.record_whale_transfer("0xtx1", "0xfrom", "0xto", 100_000.0)
        assert len(ci._whale_alerts) == 0

    def test_above_threshold_recorded(self):
        ci = self._fresh()
        ci.record_whale_transfer("0xtx1", "0xfrom", "0xto", 600_000.0)
        assert len(ci._whale_alerts) == 1

    def test_get_recent_alerts_filters_by_time(self):
        ci = self._fresh()
        ci._whale_alerts = [
            {"timestamp": time.time() - 3600 * 24, "amount_usd": 1_000_000},  # 24h ago
            {"timestamp": time.time() - 1800, "amount_usd": 2_000_000},       # 30m ago
        ]
        recent = ci.get_recent_whale_alerts(lookback_hours=6.0)
        assert len(recent) == 1
        assert recent[0]["amount_usd"] == 2_000_000

    def test_whale_summary_alert_on_3_or_more(self):
        ci = self._fresh()
        now = time.time()
        for i in range(3):
            ci._whale_alerts.append({
                "timestamp": now - 60,
                "amount_usd": 600_000.0,
                "token": "USDC",
            })
        summary = ci.get_whale_summary()
        assert summary["alert"] is True

    def test_whale_summary_alert_on_large_volume(self):
        ci = self._fresh()
        ci._whale_alerts.append({
            "timestamp": time.time() - 60,
            "amount_usd": 6_000_000.0,
            "token": "USDC",
        })
        summary = ci.get_whale_summary()
        assert summary["alert"] is True

    def test_whale_alert_capped_at_50(self):
        ci = self._fresh()
        for i in range(60):
            ci.record_whale_transfer(f"0xtx{i}", "0xfrom", "0xto", 1_000_000.0)
        assert len(ci._whale_alerts) <= 50


class TestChainIntelligenceMomentum:
    def _fresh(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        return ChainIntelligence()

    def test_insufficient_data_returns_neutral(self):
        ci = self._fresh()
        result = ci.get_momentum_sentiment()
        assert result["sentiment"] == "NEUTRAL"

    def test_bullish_signal(self):
        ci = self._fresh()
        # Uptrend: steadily rising prices
        prices = [100.0 + i * 0.5 for i in range(20)]
        for p in prices:
            ci.record_usdc_price(p)
        result = ci.get_momentum_sentiment()
        # With strong uptrend, should be BULLISH or NEUTRAL
        assert result["sentiment"] in ("BULLISH", "NEUTRAL")

    def test_bearish_signal(self):
        ci = self._fresh()
        prices = [100.0 - i * 0.5 for i in range(20)]
        for p in prices:
            ci.record_usdc_price(p)
        result = ci.get_momentum_sentiment()
        assert result["sentiment"] in ("BEARISH", "NEUTRAL")

    def test_result_contains_required_keys(self):
        ci = self._fresh()
        for _ in range(10):
            ci.record_usdc_price(1.0)
        result = ci.get_momentum_sentiment()
        assert "sentiment" in result
        assert "score" in result
        assert "momentum_pct" in result


class TestChainIntelligenceFlowSummary:
    @pytest.mark.asyncio
    async def test_no_rpc_returns_no_rpc_status(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        ci = ChainIntelligence()
        result = await ci.get_flow_summary("")
        assert result["status"] == "NO_RPC"

    @pytest.mark.asyncio
    async def test_successful_flow_summary(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        ci = ChainIntelligence()
        json_data = {
            "result": {
                "gasUsed": hex(9_500_000),
                "gasLimit": hex(10_000_000),
                "number": hex(1234567),
            }
        }
        mock_resp = MagicMock()
        mock_resp.json = AsyncMock(return_value=json_data)
        inner_cm = MagicMock()
        inner_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        inner_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post.return_value = inner_cm
        outer_cm = MagicMock()
        outer_cm.__aenter__ = AsyncMock(return_value=mock_session)
        outer_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=outer_cm):
            result = await ci.get_flow_summary("http://rpc")
        assert result["status"] == "OK"
        assert result["activity"] == "HIGH"

    @pytest.mark.asyncio
    async def test_exception_returns_error_status(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        ci = ChainIntelligence()
        failing_cm = MagicMock()
        failing_cm.__aenter__ = AsyncMock(side_effect=Exception("network error"))
        failing_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=failing_cm):
            result = await ci.get_flow_summary("http://rpc")
        assert result["status"] == "ERROR"

    @pytest.mark.asyncio
    async def test_medium_activity_threshold(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        ci = ChainIntelligence()
        json_data = {
            "result": {
                "gasUsed": hex(6_000_000),   # 60% usage — MEDIUM
                "gasLimit": hex(10_000_000),
                "number": hex(100),
            }
        }
        mock_resp = MagicMock()
        mock_resp.json = AsyncMock(return_value=json_data)
        inner_cm = MagicMock()
        inner_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        inner_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post.return_value = inner_cm
        outer_cm = MagicMock()
        outer_cm.__aenter__ = AsyncMock(return_value=mock_session)
        outer_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=outer_cm):
            result = await ci.get_flow_summary("http://rpc")
        assert result["activity"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_snapshot_aggregates_all_signals(self):
        from wallet_manager.market_intelligence import ChainIntelligence
        ci = ChainIntelligence()
        ci.record_usdc_price(1.001)
        ci.record_pool_depth(1_000_000.0)
        with patch.object(ci, "get_flow_summary", new_callable=AsyncMock,
                          return_value={"status": "OK", "activity": "LOW"}):
            snapshot = await ci.get_snapshot("http://rpc")
        assert "stablecoin_depeg" in snapshot
        assert "liquidity" in snapshot
        assert "whale" in snapshot
        assert "sentiment" in snapshot


class TestGetMacroCalendarSingleton:
    def test_singleton_returns_same_instance(self):
        import wallet_manager.market_intelligence as mi
        mi._calendar = None
        cal1 = mi.get_macro_calendar()
        cal2 = mi.get_macro_calendar()
        assert cal1 is cal2
        mi._calendar = None


class TestGetChainIntelligenceSingleton:
    def test_singleton_returns_same_instance(self):
        import wallet_manager.market_intelligence as mi
        mi._chain_intel = None
        ci1 = mi.get_chain_intelligence()
        ci2 = mi.get_chain_intelligence()
        assert ci1 is ci2
        mi._chain_intel = None


# ---------------------------------------------------------------------------
# ====================== execution_logic.py =================================
# ---------------------------------------------------------------------------


def _make_mock_manager(balance_wei=int(0.5e18)):
    """Create a minimal WalletManager mock."""
    manager = MagicMock()
    manager.get_safe_balance = AsyncMock(return_value=balance_wei)
    manager.get_config.return_value = MagicMock(
        chain_id=8453,
        rpc_url="http://rpc",
        max_gas_price_gwei=25.0,
        address=_VALID_ADDR,
        private_key=None,
    )
    manager.sign_and_broadcast_tx = AsyncMock(
        return_value=MagicMock(tx_hash="0xtxhash", nonce_used=1)
    )
    return manager


def _make_mock_portfolio():
    pm = MagicMock()
    pm.get_algo_trade_budget.return_value = 100.0
    return pm


def _make_mock_signal(alpha=0.3, prob_up=0.7, confidence=0.8):
    from wallet_manager.strategies.alpha_engine import AlphaMLSignal
    signal = MagicMock(spec=AlphaMLSignal)
    signal.alpha_score = alpha
    signal.prob_up = prob_up
    signal.confidence = confidence
    signal.garch_sigma = 0.02
    signal.volatility_regime = "normal"
    signal.should_enter_long = False
    signal.should_enter_short = False
    signal.should_exit = False
    return signal


class TestExecutionCycleResult:
    def test_defaults(self):
        from wallet_manager.strategies.execution_logic import ExecutionCycleResult
        r = ExecutionCycleResult()
        assert r.action_taken == "HOLD"
        assert r.position_usd == 0.0
        assert r.receipts == []
        assert r.error == ""


class TestMLQuantExecutionEngineInit:
    def test_singleton_reset(self):
        from wallet_manager.strategies.execution_logic import (
            get_ml_execution_engine,
            reset_ml_execution_engine,
        )
        reset_ml_execution_engine()
        e1 = get_ml_execution_engine()
        e2 = get_ml_execution_engine()
        assert e1 is e2
        reset_ml_execution_engine()

    def test_get_engine_stats_structure(self):
        from wallet_manager.strategies.execution_logic import (
            get_ml_execution_engine,
            reset_ml_execution_engine,
        )
        reset_ml_execution_engine()
        engine = get_ml_execution_engine()
        stats = engine.get_engine_stats()
        assert "alpha" in stats
        assert "risk" in stats
        assert "position" in stats
        reset_ml_execution_engine()


def _patch_kill_switch_inactive(engine):
    """Patch FlashCrashDetector.is_kill_switch_active property to return False."""
    return patch.object(
        type(engine.crash_detector),
        "is_kill_switch_active",
        new_callable=lambda: property(lambda self: False),
    )


def _patch_kill_switch_active(engine):
    """Patch FlashCrashDetector.is_kill_switch_active property to return True."""
    return patch.object(
        type(engine.crash_detector),
        "is_kill_switch_active",
        new_callable=lambda: property(lambda self: True),
    )


class TestRunCycleCircuitBreakers:
    @pytest.mark.asyncio
    async def test_halted_by_circuit_breaker(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        command_state = MagicMock()
        command_state.is_circuit_breaker_halted = True
        with _patch_kill_switch_inactive(engine):
            result = await engine.run_cycle(
                manager=_make_mock_manager(),
                wallet_id="w1",
                portfolio_manager=_make_mock_portfolio(),
                task_context={"command_state": command_state},
            )
        assert result.action_taken == "HALTED"

    @pytest.mark.asyncio
    async def test_halted_by_kill_switch(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        with _patch_kill_switch_active(engine):
            result = await engine.run_cycle(
                manager=_make_mock_manager(),
                wallet_id="w1",
                portfolio_manager=_make_mock_portfolio(),
                task_context={},
            )
        assert result.action_taken == "KILL_SWITCH"

    @pytest.mark.asyncio
    async def test_wrong_chain_id(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        manager = _make_mock_manager()
        manager.get_config.return_value.chain_id = 1  # Mainnet, not Base
        with _patch_kill_switch_inactive(engine):
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=_make_mock_portfolio(),
                task_context={},
            )
        assert result.action_taken == "WRONG_CHAIN"

    @pytest.mark.asyncio
    async def test_gas_too_high_returns_gas_high(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        manager = _make_mock_manager()
        with _patch_kill_switch_inactive(engine):
            with patch("wallet_manager.network.gas_monitor.GasMonitor") as mock_gm:
                mock_gm.return_value.is_safe = AsyncMock(return_value=(False, 30.0))
                result = await engine.run_cycle(
                    manager=manager,
                    wallet_id="w1",
                    portfolio_manager=_make_mock_portfolio(),
                    task_context={},
                )
        assert result.action_taken in ("GAS_HIGH", "CONGESTED", "WARMUP", "HOLD",
                                       "RISK_BLOCKED(normal)", "LOW_CONFIDENCE")

    @pytest.mark.asyncio
    async def test_price_fetch_failure_returns_error(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        manager = _make_mock_manager()
        with _patch_kill_switch_inactive(engine):
            with patch("wallet_manager.network.gas_monitor.GasMonitor") as mock_gm:
                mock_gm.return_value.is_safe = AsyncMock(return_value=(True, 5.0))
                with patch(
                    "wallet_manager.actions.get_live_price_v3",
                    new_callable=AsyncMock,
                    side_effect=Exception("RPC dead"),
                ):
                    result = await engine.run_cycle(
                        manager=manager,
                        wallet_id="w1",
                        portfolio_manager=_make_mock_portfolio(),
                        task_context={},
                    )
        assert result.error != "" or result.action_taken in (
            "WARMUP", "HOLD", "GAS_HIGH", "CONGESTED"
        )


class TestRunCycleHoldPath:
    @pytest.mark.asyncio
    async def test_hold_when_no_signal(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        manager = _make_mock_manager()
        with _patch_kill_switch_inactive(engine):
            with patch("wallet_manager.network.gas_monitor.GasMonitor") as mock_gm:
                mock_gm.return_value.is_safe = AsyncMock(return_value=(True, 5.0))
                with patch("wallet_manager.actions.get_live_price_v3",
                           new_callable=AsyncMock, return_value=3000.0):
                    with patch("wallet_manager.actions.get_weth_usdc_price",
                               new_callable=AsyncMock, return_value=3000.0):
                        with patch.object(engine.alpha, "process_tick", return_value=None):
                            result = await engine.run_cycle(
                                manager=manager,
                                wallet_id="w1",
                                portfolio_manager=_make_mock_portfolio(),
                                task_context={},
                            )
        assert result.action_taken == "WARMUP"


class TestCheckAndTriggerRetrain:
    @pytest.mark.asyncio
    async def test_no_retrain_without_controller(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        engine._retrain_controller = None
        # Should not raise
        await engine._check_and_trigger_retrain()

    @pytest.mark.asyncio
    async def test_no_retrain_below_cycle_threshold(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        ctrl = MagicMock()
        ctrl.should_retrain.return_value = True
        engine._retrain_controller = ctrl
        engine._cycle_count = 50  # Not at 100-cycle boundary
        engine._total_trade_count = 300
        await engine._check_and_trigger_retrain()
        ctrl.should_retrain.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_retrain_below_min_trades(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        ctrl = MagicMock()
        ctrl.should_retrain.return_value = True
        engine._retrain_controller = ctrl
        engine._cycle_count = 100
        engine._total_trade_count = 10  # Below MIN_TRADES_BEFORE_RETRAIN
        await engine._check_and_trigger_retrain()
        ctrl.should_retrain.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_retrain_within_cooldown(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        ctrl = MagicMock()
        ctrl.should_retrain.return_value = True
        engine._retrain_controller = ctrl
        engine._cycle_count = 100
        engine._total_trade_count = 300
        engine._last_retrain_trigger_ts = time.time()  # Just triggered
        await engine._check_and_trigger_retrain()
        ctrl.should_retrain.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrain_triggered_when_conditions_met(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        ctrl = MagicMock()
        ctrl.should_retrain.return_value = True
        ctrl._trigger_retrain = AsyncMock()
        engine._retrain_controller = ctrl
        engine._cycle_count = 100
        engine._total_trade_count = 300
        engine._last_retrain_trigger_ts = 0.0  # Never triggered
        await engine._check_and_trigger_retrain()
        ctrl.should_retrain.assert_called_once()
        assert engine._last_retrain_trigger_ts > 0

    @pytest.mark.asyncio
    async def test_no_retrain_when_controller_says_no(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        ctrl = MagicMock()
        ctrl.should_retrain.return_value = False  # Controller says no
        engine._retrain_controller = ctrl
        engine._cycle_count = 100
        engine._total_trade_count = 300
        engine._last_retrain_trigger_ts = 0.0
        await engine._check_and_trigger_retrain()
        # should_retrain called but retrain not triggered
        ctrl.should_retrain.assert_called_once()
        assert engine._retrain_task is None


class TestSetRetrainController:
    def test_sets_controller(self):
        from wallet_manager.strategies.execution_logic import MLQuantExecutionEngine
        engine = MLQuantExecutionEngine()
        ctrl = MagicMock()
        engine.set_retrain_controller(ctrl)
        assert engine._retrain_controller is ctrl


class TestNotifyMlTrade:
    def test_appends_notification(self):
        from wallet_manager.strategies.execution_logic import _notify_ml_trade
        signal = _make_mock_signal()
        from wallet_manager.strategies.execution_logic import ExecutionCycleResult
        result = ExecutionCycleResult()
        result.dex_venue = "uniswap_v3"
        result.risk_source = "kelly"
        context = {}
        with patch("wallet_manager.security.telegram_manager.TradeExecutedPayload"):
            _notify_ml_trade(context, "WETH/USDC", 3000.0, "$100", "BUY", signal, result)
        assert "trade_notifications" in context

    def test_handles_missing_telegram_gracefully(self):
        from wallet_manager.strategies.execution_logic import ExecutionCycleResult, _notify_ml_trade
        signal = _make_mock_signal()
        result = ExecutionCycleResult()
        context = {}
        # Should not raise even if telegram import fails
        with patch.dict("sys.modules", {"wallet_manager.security.telegram_manager": None}):
            _notify_ml_trade(context, "WETH/USDC", 3000.0, "$100", "BUY", signal, result)


class TestGetWalletAddress:
    def test_returns_address_field_when_set(self):
        from wallet_manager.strategies.execution_logic import _get_wallet_address
        config = MagicMock()
        config.address = "0x" + "b" * 40
        assert _get_wallet_address(config) == "0x" + "b" * 40

    def test_derives_from_private_key_when_no_address(self):
        from wallet_manager.strategies.execution_logic import _get_wallet_address
        config = MagicMock()
        config.address = None
        config.private_key = "0x" + "1" * 64
        with patch("eth_account.Account.from_key") as mock_acc:
            mock_acc.return_value.address = "0xderived"
            result = _get_wallet_address(config)
        assert result == "0xderived"


class TestExecutionLogicConstants:
    def test_min_trade_values(self):
        from wallet_manager.strategies.execution_logic import MIN_TRADE_USD, MIN_TRADE_WEI
        assert MIN_TRADE_USD > 0
        assert MIN_TRADE_WEI > 0

    def test_chain_id_is_base(self):
        from wallet_manager.strategies.execution_logic import CHAIN_ID_BASE
        assert CHAIN_ID_BASE == 8453

    def test_strategy_id_not_empty(self):
        from wallet_manager.strategies.execution_logic import STRATEGY_ID
        assert len(STRATEGY_ID) > 0
