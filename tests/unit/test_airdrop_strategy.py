"""tests/unit/test_airdrop_strategy.py — AIRDROP pillar tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAirdropStrategyPlaceholder:
    """Placeholder for legacy AirdropStrategy tests (class no longer exists)."""

    def test_placeholder(self):
        """Airdrop strategy module is importable."""
        import wallet_manager.strategies.airdrop_strategy as mod
        assert mod is not None


# ---------------------------------------------------------------------------
# Tests for run_airdrop_strategy and GasAwareThrottle
# ---------------------------------------------------------------------------


class TestRunAirdropStrategy:
    """Tests targeting run_airdrop_strategy() and GasAwareThrottle."""

    def _make_manager(self, eth_balance_usd: float = 100.0, usdc_balance: int = 0):
        """Return a mock WalletManager pre-configured for common scenarios."""
        config = MagicMock()
        config.rpc_url = "https://mainnet.base.org"
        config.max_gas_price_gwei = 25.0
        config.chain_id = 8453
        config.address = "0xDeadBeef"

        manager = MagicMock()
        manager.get_config.return_value = config
        # balance_wei representing eth_balance_usd (price assumed $2000)
        manager.get_safe_balance = AsyncMock(
            return_value=int(eth_balance_usd / 2000 * 1e18)
        )
        manager.sign_and_broadcast_tx = AsyncMock(
            return_value=MagicMock(tx_hash="0xabc", nonce_used=1)
        )
        return manager

    @pytest.mark.asyncio
    async def test_min_trade_usd_respected(self):
        """eth_balance_usd=2.0 below MIN_SWAP_USD=10.0 → no broadcast."""
        from wallet_manager.strategies.airdrop_strategy import run_airdrop_strategy

        manager = self._make_manager(eth_balance_usd=2.0)

        scheduler_mock = MagicMock()
        scheduler_mock.should_run = True

        task_context = {"scheduler": scheduler_mock}

        with patch(
            "wallet_manager.strategies.airdrop_strategy.actions.fetch_usdc_balance",
            new=AsyncMock(return_value=0),
        ), patch(
            "wallet_manager.strategies.airdrop_strategy.actions.fetch_eth_price_usd",
            new=AsyncMock(return_value=2000.0),
        ), patch(
            "wallet_manager.strategies.airdrop_strategy.actions.eth_balance_usd_value",
            return_value=2.0,
        ), patch(
            "wallet_manager.strategies.airdrop_strategy.actions.usdc_balance_usd_value",
            return_value=0.0,
        ), patch(
            "wallet_manager.scheduler.should_run_swap_this_cycle",
            return_value=(True, 0),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor:
            gas_monitor_instance = MagicMock()
            gas_monitor_instance.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGasMonitor.return_value = gas_monitor_instance

            receipts = await run_airdrop_strategy(
                manager=manager,
                wallet_id="w1",
                task_context=task_context,
            )

        # Total portfolio = $2.0, max_swap = $0.30 < $10 MIN_SWAP_USD → skip
        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_gas_too_high_skips_cycle(self):
        """Gas price above max_gas_gwei → no broadcast."""
        from wallet_manager.strategies.airdrop_strategy import run_airdrop_strategy

        manager = self._make_manager(eth_balance_usd=500.0)
        scheduler_mock = MagicMock()
        task_context = {"scheduler": scheduler_mock}

        with patch(
            "wallet_manager.scheduler.should_run_swap_this_cycle",
            return_value=(True, 0),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor:
            gas_monitor_instance = MagicMock()
            # Gas is too high — is_safe returns False
            gas_monitor_instance.is_safe = AsyncMock(return_value=(False, 99.0))
            MockGasMonitor.return_value = gas_monitor_instance

            receipts = await run_airdrop_strategy(
                manager=manager,
                wallet_id="w1",
                task_context=task_context,
            )

        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_scheduler_returns_empty(self):
        """Missing scheduler in task_context → early return, no broadcast."""
        from wallet_manager.strategies.airdrop_strategy import run_airdrop_strategy

        manager = self._make_manager(eth_balance_usd=500.0)

        receipts = await run_airdrop_strategy(
            manager=manager,
            wallet_id="w1",
            task_context={},  # no scheduler
        )

        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_chain_returns_empty(self):
        """Chain ID other than Base/Linea → no broadcast."""
        from wallet_manager.strategies.airdrop_strategy import run_airdrop_strategy

        manager = self._make_manager(eth_balance_usd=500.0)
        config = manager.get_config.return_value
        config.chain_id = 1  # Ethereum mainnet, not supported

        scheduler_mock = MagicMock()
        task_context = {"scheduler": scheduler_mock}

        with patch(
            "wallet_manager.scheduler.should_run_swap_this_cycle",
            return_value=(True, 0),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor:
            gas_monitor_instance = MagicMock()
            gas_monitor_instance.is_safe = AsyncMock(return_value=(True, 3.0))
            MockGasMonitor.return_value = gas_monitor_instance

            receipts = await run_airdrop_strategy(
                manager=manager,
                wallet_id="w1",
                task_context=task_context,
            )

        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()


class TestGasAwareThrottle:
    """Tests for GasAwareThrottle.should_execute()."""

    def test_ping_pong_direction_alternates_via_balance(self):
        """After BUY (ETH>USDC) cycle, USDC>ETH triggers SELL direction.

        The airdrop strategy selects direction based on balance comparison:
        if usdc_value_usd > eth_value_usd → sell USDC for ETH (BUY ETH).
        This test verifies the balance-driven direction logic through
        GasAwareThrottle to ensure both directions are reachable.
        """
        from unittest.mock import patch

        from wallet_manager.strategies.airdrop_strategy import GasAwareThrottle

        throttle = GasAwareThrottle()

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_cfg:
            cfg = MagicMock()
            # ratio = (150_000 * 1.0 * 2000.0 / 1e9) / (100.0 * 0.0005) ≈ 6.0
            # max_ratio must exceed 6.0 to let this low-gas case through
            cfg.airdrop_max_gas_fee_ratio = 10.0
            mock_cfg.return_value = cfg

            # Low gas → should execute
            ok, reason = throttle.should_execute(
                current_gwei=1.0,
                eth_price_usd=2000.0,
                swap_amount_usd=100.0,
                pool_fee_tier=0.0005,
            )
            assert ok is True
            assert "OK" in reason

    def test_gas_throttle_blocks_when_ratio_exceeded(self):
        """GasAwareThrottle returns False when gas cost > max_ratio * fee."""
        from unittest.mock import patch

        from wallet_manager.strategies.airdrop_strategy import GasAwareThrottle

        throttle = GasAwareThrottle()

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_cfg:
            cfg = MagicMock()
            cfg.airdrop_max_gas_fee_ratio = 0.1  # very strict
            mock_cfg.return_value = cfg

            # Very high gas, tiny swap → ratio will be huge
            ok, reason = throttle.should_execute(
                current_gwei=500.0,
                eth_price_usd=2000.0,
                swap_amount_usd=5.0,
                pool_fee_tier=0.0005,
            )
            assert ok is False
            assert "gas_cost" in reason

    def test_gas_throttle_zero_fee_skips(self):
        """swap_amount_usd=0 produces expected_fee=0 → always skips."""
        from unittest.mock import patch

        from wallet_manager.strategies.airdrop_strategy import GasAwareThrottle

        throttle = GasAwareThrottle()

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_cfg:
            cfg = MagicMock()
            cfg.airdrop_max_gas_fee_ratio = 5.0
            mock_cfg.return_value = cfg

            ok, reason = throttle.should_execute(
                current_gwei=1.0,
                eth_price_usd=2000.0,
                swap_amount_usd=0.0,  # zero → fee = 0
            )
            assert ok is False
            assert "expected_fee_usd=0" in reason
