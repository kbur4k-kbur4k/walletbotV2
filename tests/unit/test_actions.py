"""tests/unit/test_actions.py — actions.py pure fonksiyon testleri."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.actions import (
    _checksum,
    eth_balance_usd_value,
    get_chain_addresses,
    get_rotated_fee_tier,
    is_swap_eligible,
    is_usdc_swap_eligible,
    usdc_balance_usd_value,
)


class TestBalanceCalculations:
    def test_eth_balance_usd_value(self):
        """1 ETH = 2000 USD."""
        result = eth_balance_usd_value(int(1e18), eth_price_usd=2000.0)
        assert result == pytest.approx(2000.0)

    def test_eth_balance_usd_value_fractional(self):
        """0.5 ETH = 1000 USD."""
        result = eth_balance_usd_value(int(0.5e18), eth_price_usd=2000.0)
        assert result == pytest.approx(1000.0)

    def test_eth_balance_zero_price(self):
        result = eth_balance_usd_value(int(1e18), eth_price_usd=0.0)
        assert result == pytest.approx(0.0)

    def test_usdc_balance_usd_value(self):
        """100 USDC = $100.00."""
        result = usdc_balance_usd_value(int(100e6))
        assert result == pytest.approx(100.0)

    def test_usdc_balance_zero(self):
        result = usdc_balance_usd_value(0)
        assert result == pytest.approx(0.0)


class TestSwapEligibility:
    def test_eligible_swap(self):
        """Yeterli bakiye ve düşük gas ile swap eligible."""
        result = is_swap_eligible(
            eth_balance_wei=int(0.1e18),
            eth_price_usd=2000.0,
            gas_price_gwei=1.0,
            min_trade_usd=5.0,
            max_gas_gwei=50.0,
        )
        assert result is True

    def test_high_gas_prevents_swap(self):
        result = is_swap_eligible(
            eth_balance_wei=int(0.1e18),
            eth_price_usd=2000.0,
            gas_price_gwei=100.0,
            min_trade_usd=5.0,
            max_gas_gwei=50.0,
        )
        assert result is False

    def test_insufficient_balance_prevents_swap(self):
        result = is_swap_eligible(
            eth_balance_wei=int(0.001e18),  # $2 — altında min_trade
            eth_price_usd=2000.0,
            gas_price_gwei=1.0,
            min_trade_usd=5.0,
            max_gas_gwei=50.0,
        )
        assert result is False


class TestFeeTierRotation:
    def test_fee_tier_cycles(self):
        """Fee tier döngüsel olarak değişmeli."""
        tiers = {get_rotated_fee_tier(i) for i in range(10)}
        assert len(tiers) > 1

    def test_fee_tier_valid_values(self):
        """Fee tier Uniswap V3 geçerli değerlerinden biri olmalı."""
        valid_tiers = {100, 500, 3000, 10000}
        for i in range(20):
            assert get_rotated_fee_tier(i) in valid_tiers


class TestChainAddresses:
    def test_base_chain_addresses(self):
        addrs = get_chain_addresses(8453)
        assert hasattr(addrs, "usdc")
        assert hasattr(addrs, "weth")
        assert addrs.usdc.startswith("0x")
        assert addrs.weth.startswith("0x")

    def test_unsupported_chain_raises(self):
        with pytest.raises(ValueError):
            get_chain_addresses(99999)  # desteklenmeyen chain


class TestChecksumAddress:
    def test_lowercase_to_checksum(self):
        addr = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        result = _checksum(addr)
        assert result.startswith("0x")
        assert result != addr  # checksum farklı

    def test_already_checksum_unchanged(self):
        addr = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        result = _checksum(addr)
        assert result == addr


# ---------------------------------------------------------------------------
# New tests for async functions in actions.py
# ---------------------------------------------------------------------------

class TestFetchEthPriceUsd:
    """Tests for fetch_eth_price_usd with mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_successful_fetch_returns_price(self):
        from wallet_manager.actions import fetch_eth_price_usd

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"ethereum": {"usd": 3500.0}})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        with patch("aiohttp.ClientSession", return_value=mock_session):
            price = await fetch_eth_price_usd()

        assert price == pytest.approx(3500.0)

    @pytest.mark.asyncio
    async def test_api_failure_returns_fallback(self):
        from wallet_manager.actions import fetch_eth_price_usd

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            price = await fetch_eth_price_usd()

        # Should return fallback 3000.0
        assert price == pytest.approx(3000.0)

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(self):
        from wallet_manager.actions import fetch_eth_price_usd

        with patch("aiohttp.ClientSession", side_effect=Exception("network error")):
            price = await fetch_eth_price_usd()

        assert price == pytest.approx(3000.0)

    @pytest.mark.asyncio
    async def test_rate_limited_eventually_returns_fallback(self):
        from wallet_manager.actions import fetch_eth_price_usd

        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            price = await fetch_eth_price_usd()

        assert price == pytest.approx(3000.0)


class TestBuildExactInputSingleCalldata:
    """Tests for _build_exact_input_single_calldata."""

    def test_builds_bytes_without_deadline(self):
        from wallet_manager.actions import _build_exact_input_single_calldata
        data = _build_exact_input_single_calldata(
            token_in="0x4200000000000000000000000000000000000006",
            token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            fee=3000,
            recipient="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            amount_in=10**18,
            amount_out_min=0,
        )
        assert isinstance(data, bytes)
        assert len(data) > 4
        # selector = 04e45aaf
        assert data[:4] == bytes.fromhex("04e45aaf")

    def test_builds_bytes_with_deadline(self):
        import wallet_manager.actions as act
        from wallet_manager.actions import DEADLINE_SUPPORTED, _build_exact_input_single_calldata
        original = act.DEADLINE_SUPPORTED
        act.DEADLINE_SUPPORTED = True
        try:
            data = _build_exact_input_single_calldata(
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                fee=3000,
                recipient="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                amount_in=10**18,
                amount_out_min=0,
                deadline=9999999999,
            )
            assert isinstance(data, bytes)
            assert len(data) > 4
        finally:
            act.DEADLINE_SUPPORTED = original


class TestBuildApproveCalldata:
    def test_returns_bytes_with_selector(self):
        from wallet_manager.actions import _build_approve_calldata
        data = _build_approve_calldata(
            "0x2626664c2603336E57B271c5C0b26F421741e481",
            10**18,
        )
        assert isinstance(data, bytes)
        # approve selector: 095ea7b3
        assert data[:4] == bytes.fromhex("095ea7b3")


class TestGetDynamicSlippage:
    """Tests for get_dynamic_slippage."""

    @pytest.mark.asyncio
    async def test_low_gas_returns_min_slippage(self):
        from wallet_manager.actions import MIN_SLIPPAGE_BPS, get_dynamic_slippage

        with patch("wallet_manager.actions.get_pool_liquidity", new_callable=AsyncMock, return_value=0):
            result = await get_dynamic_slippage(
                rpc_url="http://rpc",
                chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=10**17,
                gas_price_gwei=5.0,
                fee=3000,
                is_algo=False,
            )
        assert result >= MIN_SLIPPAGE_BPS

    @pytest.mark.asyncio
    async def test_high_gas_increases_slippage(self):
        from wallet_manager.actions import MIN_SLIPPAGE_BPS, get_dynamic_slippage

        with patch("wallet_manager.actions.get_pool_liquidity", new_callable=AsyncMock, return_value=0):
            low_gas = await get_dynamic_slippage(
                rpc_url="http://rpc", chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=10**17, gas_price_gwei=5.0, fee=3000,
            )
            high_gas = await get_dynamic_slippage(
                rpc_url="http://rpc", chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=10**17, gas_price_gwei=20.0, fee=3000,
            )
        assert high_gas >= low_gas

    @pytest.mark.asyncio
    async def test_algo_mode_caps_slippage(self):
        from wallet_manager.actions import ALGO_MAX_SLIPPAGE_BPS, get_dynamic_slippage

        with patch("wallet_manager.actions.get_pool_liquidity", new_callable=AsyncMock, return_value=0):
            result = await get_dynamic_slippage(
                rpc_url="http://rpc", chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=10**17, gas_price_gwei=20.0, fee=3000,
                is_algo=True,
            )
        assert result <= ALGO_MAX_SLIPPAGE_BPS

    @pytest.mark.asyncio
    async def test_large_amount_adds_slippage(self):
        from wallet_manager.actions import MIN_SLIPPAGE_BPS, get_dynamic_slippage

        with patch("wallet_manager.actions.get_pool_liquidity", new_callable=AsyncMock, return_value=0):
            normal = await get_dynamic_slippage(
                rpc_url="http://rpc", chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=10**17, gas_price_gwei=5.0, fee=3000,
            )
            large = await get_dynamic_slippage(
                rpc_url="http://rpc", chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=100 * 10**18, gas_price_gwei=5.0, fee=3000,
            )
        assert large >= normal

    @pytest.mark.asyncio
    async def test_high_liquidity_low_amount_no_extra(self):
        from wallet_manager.actions import MIN_SLIPPAGE_BPS, get_dynamic_slippage

        # Very high liquidity: amount << liq/500 → no extra
        with patch("wallet_manager.actions.get_pool_liquidity", new_callable=AsyncMock, return_value=10**30):
            result = await get_dynamic_slippage(
                rpc_url="http://rpc", chain_id=8453,
                token_in="0x4200000000000000000000000000000000000006",
                token_out="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount_in=10**15, gas_price_gwei=5.0, fee=3000,
            )
        assert result >= MIN_SLIPPAGE_BPS


class TestGetWethUsdcPrice:
    """Tests for get_weth_usdc_price."""

    @pytest.mark.asyncio
    async def test_returns_float_on_success(self):
        from wallet_manager.actions import get_weth_usdc_price

        mock_pool = MagicMock()
        # sqrtPriceX96 for ~$2500 WETH
        # price = (sqrtPriceX96/2^96)^2 * 1e18/1e6
        # For 2500 USD: sqrtPriceX96 = sqrt(2500 * 1e6 / 1e18) * 2^96
        import math
        q96 = 2**96
        target_price = 2500.0
        # raw = target_price * 1e6 / 1e18
        raw = target_price * 1e6 / 1e18
        sqrt_val = math.sqrt(raw)
        sqrt_price_x96 = int(sqrt_val * q96)

        mock_pool.functions.slot0 = MagicMock()
        mock_pool.functions.slot0.return_value.call = AsyncMock(
            return_value=[sqrt_price_x96, 0, 0, 0, 0, 0, True]
        )

        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_pool

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            price = await get_weth_usdc_price("http://rpc")

        assert price > 0
        assert isinstance(price, float)

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self):
        from wallet_manager.actions import get_weth_usdc_price

        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("connection error")

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            price = await get_weth_usdc_price("http://rpc")

        assert price == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_sqrtprice_zero(self):
        from wallet_manager.actions import get_weth_usdc_price

        mock_pool = MagicMock()
        mock_pool.functions.slot0.return_value.call = AsyncMock(
            return_value=[0, 0, 0, 0, 0, 0, True]  # sqrtPriceX96 = 0
        )

        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_pool

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            price = await get_weth_usdc_price("http://rpc")

        assert price == 0.0


class TestGetCurrentPrice:
    """Tests for get_current_price (with slot0 fallback paths)."""

    @pytest.mark.asyncio
    async def test_chainlink_fallback_when_slot0_fails(self):
        from wallet_manager.actions import get_current_price

        mock_pool = MagicMock()
        mock_pool.functions.slot0.return_value.call = AsyncMock(side_effect=Exception("rpc error"))

        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_pool

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3), \
             patch("wallet_manager.actions.quote_amount_out", new_callable=AsyncMock, return_value=0), \
             patch("wallet_manager.actions._fetch_chainlink_eth_usd", new_callable=AsyncMock, return_value=2800.0):
            price = await get_current_price("http://rpc", chain_id=8453)

        assert price == pytest.approx(2800.0)

    @pytest.mark.asyncio
    async def test_quoter_fallback_when_slot0_zero(self):
        from wallet_manager.actions import get_current_price

        mock_pool = MagicMock()
        mock_pool.functions.slot0.return_value.call = AsyncMock(
            return_value=[0, 0, 0, 0, 0, 0, True]
        )
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_pool

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3), \
             patch("wallet_manager.actions.quote_amount_out", new_callable=AsyncMock, return_value=int(2600 * 1e6)), \
             patch("wallet_manager.actions._fetch_chainlink_eth_usd", new_callable=AsyncMock, return_value=2600.0):
            price = await get_current_price("http://rpc", chain_id=8453)

        assert price == pytest.approx(2600.0)

    @pytest.mark.asyncio
    async def test_non_base_chain_skips_slot0(self):
        from wallet_manager.actions import get_current_price

        with patch("wallet_manager.actions.quote_amount_out", new_callable=AsyncMock, return_value=int(2000 * 1e6)), \
             patch("wallet_manager.actions._fetch_chainlink_eth_usd", new_callable=AsyncMock, return_value=2000.0):
            price = await get_current_price("http://rpc", chain_id=59144)

        assert price > 0


class TestFetchNativeBalance:
    @pytest.mark.asyncio
    async def test_returns_int_balance(self):
        from wallet_manager.actions import fetch_native_balance

        mock_w3 = MagicMock()
        mock_w3.eth.get_balance = AsyncMock(return_value=int(1e18))

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            bal = await fetch_native_balance("http://rpc", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")

        assert bal == int(1e18)


class TestFetchUsdcBalance:
    @pytest.mark.asyncio
    async def test_returns_int_usdc_balance(self):
        from wallet_manager.actions import fetch_usdc_balance

        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call = AsyncMock(return_value=int(500e6))

        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            bal = await fetch_usdc_balance("http://rpc", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", 8453)

        assert bal == int(500e6)

    def test_unsupported_chain_raises(self):
        import asyncio

        from wallet_manager.actions import fetch_usdc_balance
        with pytest.raises(ValueError):
            asyncio.run(fetch_usdc_balance("http://rpc", "0xABCDEF", 1))


class TestGetPoolLiquidity:
    @pytest.mark.asyncio
    async def test_returns_zero_for_non_base(self):
        from wallet_manager.actions import get_pool_liquidity
        result = await get_pool_liquidity("http://rpc", chain_id=59144)
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_for_non_3000_fee(self):
        from wallet_manager.actions import get_pool_liquidity
        result = await get_pool_liquidity("http://rpc", chain_id=8453, fee=500)
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_liquidity_on_success(self):
        from wallet_manager.actions import get_pool_liquidity

        mock_pool = MagicMock()
        mock_pool.functions.liquidity.return_value.call = AsyncMock(return_value=10**20)

        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_pool

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_pool_liquidity("http://rpc", chain_id=8453, fee=3000)

        assert result == 10**20

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self):
        from wallet_manager.actions import get_pool_liquidity

        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("rpc error")

        with patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            result = await get_pool_liquidity("http://rpc", chain_id=8453, fee=3000)

        assert result == 0


class TestBuildAaveSupplyPlan:
    @pytest.mark.asyncio
    async def test_plan_includes_supply_step(self):
        from wallet_manager.actions import build_aave_supply_usdc_plan

        with patch("wallet_manager.actions._get_allowance", new_callable=AsyncMock, return_value=10**30):
            steps = await build_aave_supply_usdc_plan(
                rpc_url="http://rpc",
                from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                amount_usdc=int(100e6),
                gas_price_gwei=5.0,
            )

        assert len(steps) >= 1
        # Last step should be AavePool.supply
        assert "supply" in steps[-1][1].lower() or "aave" in steps[-1][1].lower()

    @pytest.mark.asyncio
    async def test_plan_includes_approve_when_allowance_insufficient(self):
        from wallet_manager.actions import build_aave_supply_usdc_plan

        with patch("wallet_manager.actions._get_allowance", new_callable=AsyncMock, return_value=0):
            steps = await build_aave_supply_usdc_plan(
                rpc_url="http://rpc",
                from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                amount_usdc=int(100e6),
                gas_price_gwei=5.0,
            )

        assert len(steps) == 2
        assert "approve" in steps[0][1].lower()


class TestBuildAaveWithdrawPlan:
    @pytest.mark.asyncio
    async def test_plan_has_single_withdraw_step(self):
        from wallet_manager.actions import build_aave_withdraw_usdc_plan

        steps = await build_aave_withdraw_usdc_plan(
            from_address="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            amount_usdc=int(100e6),
            gas_price_gwei=5.0,
        )

        assert len(steps) == 1
        assert "withdraw" in steps[0][1].lower()


class TestGetAaveStats:
    @pytest.mark.asyncio
    async def test_paper_mode_returns_mock_stats(self):
        from wallet_manager.actions import AaveStats, get_aave_stats

        mock_settings = MagicMock()
        mock_settings.execution_mode = "robust_paper"

        with patch("wallet_manager.config.get_settings", return_value=mock_settings):
            stats = await get_aave_stats("http://rpc", "0xABCDEF")

        assert isinstance(stats, AaveStats)
        assert stats.supply_apy_pct == pytest.approx(3.25)
        assert stats.supplied_usdc == 0.0

    @pytest.mark.asyncio
    async def test_on_chain_mode_exception_returns_empty_stats(self):
        from wallet_manager.actions import AaveStats, get_aave_stats

        mock_settings = MagicMock()
        mock_settings.execution_mode = "live"

        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = Exception("rpc error")

        with patch("wallet_manager.config.get_settings", return_value=mock_settings), \
             patch("wallet_manager.actions._make_w3", return_value=mock_w3):
            stats = await get_aave_stats("http://rpc", "0xABCDEF")

        assert isinstance(stats, AaveStats)
        assert stats.supply_apy_pct == 0.0


class TestGetAlternativeStableYieldApy:
    @pytest.mark.asyncio
    async def test_returns_compound_when_available(self):
        from wallet_manager.actions import get_alternative_stable_yield_apy

        with patch("wallet_manager.actions._get_compound_v3_apy", new_callable=AsyncMock, return_value=4.5):
            apy, name = await get_alternative_stable_yield_apy("http://rpc")

        assert apy == pytest.approx(4.5)
        assert "Compound" in name

    @pytest.mark.asyncio
    async def test_fallback_when_compound_fails(self):
        from wallet_manager.actions import (
            DEFAULT_ALTERNATIVE_APY_PCT,
            get_alternative_stable_yield_apy,
        )

        with patch("wallet_manager.actions._get_compound_v3_apy", new_callable=AsyncMock, return_value=None):
            apy, _name = await get_alternative_stable_yield_apy("http://rpc")

        assert apy == pytest.approx(DEFAULT_ALTERNATIVE_APY_PCT)


class TestGetHistoricalPrices:
    @pytest.mark.asyncio
    async def test_returns_list_of_prices(self):
        from wallet_manager.actions import get_historical_prices

        with patch("wallet_manager.actions.get_current_price", new_callable=AsyncMock, return_value=2500.0):
            prices = await get_historical_prices("http://rpc", chain_id=8453, count=20)

        assert isinstance(prices, list)
        assert len(prices) > 0
        assert all(p == pytest.approx(2500.0) for p in prices)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_price_zero(self):
        from wallet_manager.actions import get_historical_prices

        with patch("wallet_manager.actions.get_current_price", new_callable=AsyncMock, return_value=0.0):
            prices = await get_historical_prices("http://rpc", chain_id=8453)

        assert prices == []


class TestIsUsdcSwapEligible:
    def test_eligible_with_enough_usdc(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        assert is_usdc_swap_eligible(int(100e6), gas_gwei=5.0) is True

    def test_not_eligible_with_low_usdc(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        assert is_usdc_swap_eligible(int(1e6), gas_gwei=5.0) is False  # $1 < $10

    def test_not_eligible_high_gas(self):
        from wallet_manager.actions import is_usdc_swap_eligible
        assert is_usdc_swap_eligible(int(100e6), gas_gwei=30.0) is False


class TestSwapDirection:
    def test_enum_values_exist(self):
        from wallet_manager.actions import SwapDirection
        assert SwapDirection.ETH_TO_USDC is not None
        assert SwapDirection.USDC_TO_ETH is not None


class TestChainAddressesLinea:
    def test_linea_chain_addresses(self):
        addrs = get_chain_addresses(59144)
        assert hasattr(addrs, "usdc")
        assert hasattr(addrs, "weth")
        assert addrs.usdc.startswith("0x")
        assert addrs.weth.startswith("0x")


class TestMakeW3:
    def test_reuses_provider_for_same_url(self):
        from wallet_manager.actions import _make_w3, _shared_providers

        url = "http://test-reuse-provider.local"
        _shared_providers.pop(url, None)

        _make_w3(url)
        _make_w3(url)
        # Same provider is reused
        assert url in _shared_providers
        _shared_providers.pop(url, None)  # cleanup

    def test_different_urls_get_different_providers(self):
        from wallet_manager.actions import _make_w3, _shared_providers

        url1 = "http://test-url-a.local"
        url2 = "http://test-url-b.local"
        _shared_providers.pop(url1, None)
        _shared_providers.pop(url2, None)

        _make_w3(url1)
        _make_w3(url2)
        assert url1 in _shared_providers
        assert url2 in _shared_providers
        _shared_providers.pop(url1, None)
        _shared_providers.pop(url2, None)
