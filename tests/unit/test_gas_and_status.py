"""tests/unit/test_gas_and_status.py — Coverage for orchestrator_gas.py and orchestrator_status.py"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# orchestrator_gas.py — GasGuard
# ---------------------------------------------------------------------------

class TestGasGuard:
    """Tests for GasGuard class."""

    def _make_guard(self, poll_interval=0.001, wait_timeout=0.1):
        from wallet_manager.orchestrator_gas import GasGuard
        gas_monitor = MagicMock()
        gas_monitor.is_safe = AsyncMock(return_value=(True, 5.0))
        gas_monitor.current_gwei = AsyncMock(return_value=5.0)
        return GasGuard(gas_monitor, poll_interval, wait_timeout), gas_monitor

    def _make_config(self, max_gwei=50.0, wallet_id="w1"):
        return SimpleNamespace(max_gas_price_gwei=max_gwei, wallet_id=wallet_id)

    @pytest.mark.asyncio
    async def test_safe_gas_returns_immediately(self):
        guard, monitor = self._make_guard()
        monitor.is_safe = AsyncMock(return_value=(True, 5.0))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        await guard.wait_for_safe_gas(self._make_config(), notifier, safe_notify)
        safe_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_gas_too_high_sends_pause_then_resumes(self):
        guard, monitor = self._make_guard(poll_interval=0.001, wait_timeout=1.0)
        # First call: unsafe, second: safe
        monitor.is_safe = AsyncMock(side_effect=[(False, 100.0), (True, 10.0)])
        notifier = MagicMock()
        safe_notify = AsyncMock()
        await guard.wait_for_safe_gas(self._make_config(), notifier, safe_notify)
        assert safe_notify.call_count == 2  # GAS_PAUSE + GAS_RESUME

    @pytest.mark.asyncio
    async def test_gas_wait_timeout_raises(self):
        from wallet_manager.exceptions import GasPriceTooHighError
        guard, monitor = self._make_guard(poll_interval=0.001, wait_timeout=0.01)
        monitor.is_safe = AsyncMock(return_value=(False, 200.0))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        with pytest.raises(GasPriceTooHighError):
            await guard.wait_for_safe_gas(self._make_config(), notifier, safe_notify)

    @pytest.mark.asyncio
    async def test_gas_monitor_exception_proceeds_safe(self):
        guard, monitor = self._make_guard()
        monitor.is_safe = AsyncMock(side_effect=Exception("RPC down"))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        # Should NOT raise — treat as safe
        await guard.wait_for_safe_gas(self._make_config(), notifier, safe_notify)
        safe_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_current_gwei_returns_value(self):
        guard, monitor = self._make_guard()
        monitor.current_gwei = AsyncMock(return_value=7.5)
        result = await guard.current_gwei()
        assert result == 7.5

    @pytest.mark.asyncio
    async def test_current_gwei_exception_returns_zero(self):
        guard, monitor = self._make_guard()
        monitor.current_gwei = AsyncMock(side_effect=Exception("fail"))
        result = await guard.current_gwei()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_multiple_wallets_different_configs(self):
        guard, monitor = self._make_guard()
        monitor.is_safe = AsyncMock(return_value=(True, 3.0))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        # Test with different max_gwei values
        for max_gwei in [10.0, 30.0, 100.0]:
            await guard.wait_for_safe_gas(
                self._make_config(max_gwei=max_gwei), notifier, safe_notify
            )
        assert safe_notify.call_count == 0


# ---------------------------------------------------------------------------
# orchestrator_status.py — LiveStatusProvider
# ---------------------------------------------------------------------------

class TestLiveStatusProvider:
    """Tests for LiveStatusProvider class."""

    def _make_provider(self, configs=None, portfolio_manager=None):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        scheduler = MagicMock()
        scheduler.seconds_until_due = MagicMock(return_value=120)
        if configs is None:
            config = SimpleNamespace(
                wallet_id="w1",
                rpc_url="http://localhost:8545",
                chain_id=8453,
                address="0xabcdef1234567890" * 2,
                private_key="a" * 64,
            )
            configs = [config]
        return LiveStatusProvider(configs, scheduler, portfolio_manager)

    @pytest.mark.asyncio
    async def test_no_configs_returns_message(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        scheduler = MagicMock()
        provider = LiveStatusProvider([], scheduler)
        msg = await provider.build_status_message()
        assert "No wallets" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_all_exceptions(self):
        """All on-chain calls fail — returns fallback values."""
        provider = self._make_provider()
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(side_effect=Exception("rpc"))), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(side_effect=Exception("rpc"))), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(side_effect=Exception("rpc"))), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"):
            msg = await provider.build_status_message()
        assert "Portfolio Live Status" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_with_portfolio_manager(self):
        """Test with portfolio manager providing snapshot."""
        pm = MagicMock()
        pm.sync_from_chain = AsyncMock()
        pm.get_snapshot = MagicMock(return_value=SimpleNamespace(total_value_usd=5000.0))
        provider = self._make_provider(portfolio_manager=pm)
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(return_value=int(1e18))), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=int(1000 * 1e6))), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"), \
             patch("wallet_manager.actions.eth_balance_usd_value", return_value=2000.0):
            msg = await provider.build_status_message()
        assert "5000" in msg

    @pytest.mark.asyncio
    async def test_next_run_hours_format(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        scheduler = MagicMock()
        scheduler.seconds_until_due = MagicMock(return_value=7200)  # 2 hours
        config = SimpleNamespace(
            wallet_id="w1", rpc_url="http://localhost", chain_id=8453,
            address="0x1234", private_key="a" * 64,
        )
        provider = LiveStatusProvider([config], scheduler)
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"), \
             patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0):
            msg = await provider.build_status_message()
        assert "hours" in msg

    @pytest.mark.asyncio
    async def test_next_run_minutes_format(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        scheduler = MagicMock()
        scheduler.seconds_until_due = MagicMock(return_value=180)  # 3 mins
        config = SimpleNamespace(
            wallet_id="w1", rpc_url="http://localhost", chain_id=8453,
            address="0x1234", private_key="a" * 64,
        )
        provider = LiveStatusProvider([config], scheduler)
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"), \
             patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0):
            msg = await provider.build_status_message()
        assert "mins" in msg

    @pytest.mark.asyncio
    async def test_next_run_due_now_format(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        scheduler = MagicMock()
        scheduler.seconds_until_due = MagicMock(return_value=0)
        config = SimpleNamespace(
            wallet_id="w1", rpc_url="http://localhost", chain_id=8453,
            address="0x1234", private_key="a" * 64,
        )
        provider = LiveStatusProvider([config], scheduler)
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"), \
             patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0):
            msg = await provider.build_status_message()
        assert "due now" in msg

    @pytest.mark.asyncio
    async def test_next_run_seconds_format(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        scheduler = MagicMock()
        scheduler.seconds_until_due = MagicMock(return_value=45)
        config = SimpleNamespace(
            wallet_id="w1", rpc_url="http://localhost", chain_id=8453,
            address="0x1234", private_key="a" * 64,
        )
        provider = LiveStatusProvider([config], scheduler)
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=0)), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"), \
             patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0):
            msg = await provider.build_status_message()
        assert "secs" in msg

    @pytest.mark.asyncio
    async def test_without_portfolio_manager_uses_actions(self):
        """No portfolio manager: calculates total from individual balances."""
        provider = self._make_provider()  # No PM
        with patch("wallet_manager.actions.fetch_native_balance", AsyncMock(return_value=int(0.5e18))), \
             patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=int(500 * 1e6))), \
             patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
             patch("wallet_manager.task_factories._get_wallet_address", return_value="0xabc"), \
             patch("wallet_manager.actions.eth_balance_usd_value", return_value=1000.0):
            msg = await provider.build_status_message()
        assert "Portfolio Live Status" in msg
        assert "0xabc" in msg
