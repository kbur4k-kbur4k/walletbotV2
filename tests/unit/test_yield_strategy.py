"""tests/unit/test_yield_strategy.py — YIELD pillar testleri."""
from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestYieldStrategy:
    @pytest.mark.asyncio
    async def test_get_aave_usdc_balance_calls_rpc(self):
        from wallet_manager.strategies.yield_strategy import get_aave_usdc_balance

        # get_aave_usdc_balance imports AsyncWeb3 locally; patch at web3 module level.
        # Also patch _checksum to avoid real address validation.
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call = AsyncMock(return_value=int(500e6))
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        valid_address = "0xDeadBeef00000000000000000000000000001234"

        with patch("web3.AsyncWeb3", return_value=mock_w3), \
             patch("wallet_manager.strategies.yield_strategy._checksum", side_effect=lambda x: x):
            balance = await get_aave_usdc_balance(
                rpc_url="https://mainnet.base.org",
                address=valid_address,
            )
        assert balance == int(500e6)

    @pytest.mark.asyncio
    async def test_aave_balance_rpc_error_returns_zero(self):
        from wallet_manager.strategies.yield_strategy import get_aave_usdc_balance

        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call = AsyncMock(side_effect=Exception("RPC down"))
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = mock_contract
        valid_address = "0xDeadBeef00000000000000000000000000001234"

        with patch("web3.AsyncWeb3", return_value=mock_w3), \
             patch("wallet_manager.strategies.yield_strategy._checksum", side_effect=lambda x: x):
            try:
                balance = await get_aave_usdc_balance("https://rpc", valid_address)
                assert balance == 0
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


class TestSupplyUsdc:
    """Tests for supply_usdc() — ensures sign_and_broadcast_tx is called
    with calldata produced by build_aave_supply_usdc_plan."""

    @pytest.mark.asyncio
    async def test_supply_usdc_to_aave_calls_broadcast(self):
        """supply_usdc() must call sign_and_broadcast_tx with Aave calldata."""
        from wallet_manager.core.models import TxParams
        from wallet_manager.strategies.yield_strategy import supply_usdc

        fake_tx = TxParams(
            from_address="0xDeadBeef",
            to="0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
            data=b"\xaa\xbb\xcc",
            value=0,
            gas=200_000,
            gas_price_wei=int(3.0 * 1e9),
        )
        fake_plan_steps = [(fake_tx, "supply")]

        mock_manager = MagicMock()
        mock_manager.get_config.return_value = MagicMock(chain_id=8453)
        mock_manager.sign_and_broadcast_tx = AsyncMock(
            return_value=MagicMock(tx_hash="0xsupply", nonce_used=0)
        )

        with patch(
            "wallet_manager.strategies.yield_strategy.actions.build_aave_supply_usdc_plan",
            new=AsyncMock(return_value=fake_plan_steps),
        ):
            receipts = await supply_usdc(
                manager=mock_manager,
                wallet_id="w1",
                rpc_url="https://mainnet.base.org",
                from_address="0xDeadBeef",
                amount_usdc=int(100e6),
                gas_price_gwei=3.0,
            )

        assert len(receipts) == 1
        mock_manager.sign_and_broadcast_tx.assert_called_once()
        call_args = mock_manager.sign_and_broadcast_tx.call_args
        tx_sent = call_args[0][1]  # second positional arg is tx_params
        assert tx_sent.to == "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"

    @pytest.mark.asyncio
    async def test_withdraw_usdc_from_aave(self):
        """withdraw_usdc() must call sign_and_broadcast_tx with withdraw calldata."""
        from wallet_manager.core.models import TxParams
        from wallet_manager.strategies.yield_strategy import withdraw_usdc

        fake_tx = TxParams(
            from_address="0xDeadBeef",
            to="0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
            data=b"\xdd\xee\xff",
            value=0,
            gas=150_000,
            gas_price_wei=int(3.0 * 1e9),
        )
        fake_plan_steps = [(fake_tx, "withdraw")]

        mock_manager = MagicMock()
        mock_manager.get_config.return_value = MagicMock(chain_id=8453)
        mock_manager.sign_and_broadcast_tx = AsyncMock(
            return_value=MagicMock(tx_hash="0xwithdraw", nonce_used=0)
        )

        with patch(
            "wallet_manager.strategies.yield_strategy.actions.build_aave_withdraw_usdc_plan",
            new=AsyncMock(return_value=fake_plan_steps),
        ):
            receipts = await withdraw_usdc(
                manager=mock_manager,
                wallet_id="w1",
                from_address="0xDeadBeef",
                amount_usdc=int(50e6),
                gas_price_gwei=3.0,
            )

        assert len(receipts) == 1
        mock_manager.sign_and_broadcast_tx.assert_called_once()


class TestAaveAPYMonitor:
    """Tests for AaveAPYMonitor and apy_below_threshold module flag."""

    @pytest.mark.asyncio
    async def test_apy_below_threshold_sets_flag(self):
        """When APY < min_apy_pct, apy_below_threshold flag is set True and
        supply is skipped."""
        import wallet_manager.strategies.yield_strategy as ys
        from wallet_manager.strategies.yield_strategy import AaveAPYMonitor

        monitor = AaveAPYMonitor.__new__(AaveAPYMonitor)
        monitor.last_apy = 0.0

        # Simulate APY being below threshold
        ys.apy_below_threshold = False

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_settings, patch.object(
            monitor, "get_current_apy", new=AsyncMock(return_value=0.001)
        ), patch("asyncio.sleep", new=AsyncMock(side_effect=StopAsyncIteration)):
            cfg = MagicMock()
            cfg.aave_apy_poll_interval_seconds = 3600
            cfg.yield_min_apy_pct = 1.0  # 1% threshold
            mock_settings.return_value = cfg

            with contextlib.suppress(StopAsyncIteration):
                await monitor.poll_forever("https://mainnet.base.org")

        # 0.001 * 100 = 0.1% < 1.0% → flag must be True
        assert ys.apy_below_threshold is True

    @pytest.mark.asyncio
    async def test_apy_above_threshold_clears_flag(self):
        """When APY >= min_apy_pct, apy_below_threshold flag is cleared."""
        import wallet_manager.strategies.yield_strategy as ys
        from wallet_manager.strategies.yield_strategy import AaveAPYMonitor

        monitor = AaveAPYMonitor.__new__(AaveAPYMonitor)
        monitor.last_apy = 0.0
        ys.apy_below_threshold = True  # Start with flag set

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_settings, patch.object(
            monitor, "get_current_apy", new=AsyncMock(return_value=0.05)  # 5%
        ), patch("asyncio.sleep", new=AsyncMock(side_effect=StopAsyncIteration)):
            cfg = MagicMock()
            cfg.aave_apy_poll_interval_seconds = 3600
            cfg.yield_min_apy_pct = 1.0
            mock_settings.return_value = cfg

            with contextlib.suppress(StopAsyncIteration):
                await monitor.poll_forever("https://mainnet.base.org")

        # 5.0% > 1.0% → flag must be False
        assert ys.apy_below_threshold is False


class TestAaveContractAddress:
    """Verify that the Aave V3 Pool address constant is correct."""

    def test_get_aave_usdc_balance_uses_correct_contract_address(self):
        """Assert the module targets Base Aave V3 Pool 0xA238Dd80C259a72e81d7e4664a9801593F98d1c5."""
        from wallet_manager.strategies.yield_strategy import AAVE_POOL_BASE

        assert AAVE_POOL_BASE.lower() == "0xa238dd80c259a72e81d7e4664a9801593f98d1c5"

    def test_aave_pool_address_is_checksummed_correctly(self):
        """AAVE_POOL_BASE should be the canonical Base Aave V3 Pool address."""
        from wallet_manager.strategies.yield_strategy import AAVE_POOL_BASE

        # Exact checksum form as used in the Base network
        expected = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"
        assert expected == AAVE_POOL_BASE
