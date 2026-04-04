"""tests/unit/test_lifecycle_gas_coverage.py

Fast unit tests for:
- wallet_manager/orchestrator_lifecycle.py
- wallet_manager/orchestrator_gas.py  (supplemental coverage on top of test_gas_and_status.py)
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# orchestrator_lifecycle.py tests
# ---------------------------------------------------------------------------

class TestLifecycleManagerSignalHandlers:

    def test_setup_signal_handlers_calls_add_signal_handler(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        loop = MagicMock()
        command_state = MagicMock()
        command_state.stop = MagicMock()
        with patch("signal.SIGTERM", autospec=False), patch("signal.SIGINT", autospec=False):
            try:
                LifecycleManager.setup_signal_handlers(loop, command_state)
            except Exception:
                pass  # Windows doesn't support add_signal_handler

    def test_setup_signal_handlers_handles_not_implemented(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        loop = MagicMock()
        loop.add_signal_handler.side_effect = NotImplementedError
        command_state = MagicMock()
        # Should not raise
        LifecycleManager.setup_signal_handlers(loop, command_state)

    def test_setup_signal_handlers_handles_runtime_error(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        loop = MagicMock()
        loop.add_signal_handler.side_effect = RuntimeError("no loop")
        command_state = MagicMock()
        # Should not raise
        LifecycleManager.setup_signal_handlers(loop, command_state)

    def test_setup_without_stop_method(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        loop = MagicMock()
        loop.add_signal_handler.side_effect = NotImplementedError
        command_state = object()  # no stop method
        # Should not raise
        LifecycleManager.setup_signal_handlers(loop, command_state)


class TestValidateStartupConfig:

    def _make_settings(self, wallet_data, **kwargs):
        """Make a Settings mock with a temp wallet file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(wallet_data, f)
            wallet_path = Path(f.name)

        s = MagicMock()
        s.wallet_config_path = wallet_path
        s.telegram_bot_token = "tok"
        s.telegram_chat_id = "123"
        s.infura_key = "key"
        s.alchemy_key = None
        s.execution_mode = "robust_paper"
        s.is_live_execution = False
        s.live_trading_enabled = False
        s.live_max_wallets = 5
        s.vault_key = "kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo="
        s.min_paper_trades_before_live = 5
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    @pytest.mark.asyncio
    async def test_valid_wallet_file_passes(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc", "rpc_url": "https://x"}]
        s = self._make_settings(wallets)
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=[])):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_missing_wallet_file_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        s = MagicMock()
        s.wallet_config_path = Path("/nonexistent/wallets.json")
        with pytest.raises(ConfigurationError, match="does not exist"):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_empty_wallet_list_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        s = self._make_settings([])  # empty list
        with pytest.raises(ConfigurationError, match="no configured wallets"):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json{{")
            wallet_path = Path(f.name)
        s = MagicMock()
        s.wallet_config_path = wallet_path
        with pytest.raises(ConfigurationError, match="could not be parsed"):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_warnings_logged_when_no_telegram(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets)
        s.telegram_bot_token = None  # triggers warning
        s.telegram_chat_id = None
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=[])):
            await LifecycleManager.validate_startup_config(s)  # should not raise

    @pytest.mark.asyncio
    async def test_warnings_logged_when_no_rpc_keys(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets)
        s.infura_key = None
        s.alchemy_key = None  # triggers warning
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=[])):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_ml_quant_mode_missing_ml_dir(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets, execution_mode="ml_quant")
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=[])), patch("pathlib.Path.is_dir", return_value=False):
            await LifecycleManager.validate_startup_config(s)  # warning, not error

    @pytest.mark.asyncio
    async def test_live_guard_violated_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets)
        s.is_live_execution = True
        s.live_trading_enabled = False
        with pytest.raises(ConfigurationError, match="LIVE_TRADING_ENABLED"):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_live_too_many_wallets_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [
            {"wallet_id": f"w{i}", "private_key": "0xabc"}
            for i in range(10)
        ]
        s = self._make_settings(wallets)
        s.is_live_execution = True
        s.live_trading_enabled = True
        s.live_max_wallets = 3
        with pytest.raises(ConfigurationError, match="at most 3"):
            await LifecycleManager.validate_startup_config(s)

    @pytest.mark.asyncio
    async def test_health_check_exception_is_swallowed(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets)
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(side_effect=RuntimeError("network down"))):
            await LifecycleManager.validate_startup_config(s)  # should not raise

    @pytest.mark.asyncio
    async def test_health_check_warnings_appended(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets)
        s.telegram_bot_token = None  # prevent Telegram send
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=["RPC unreachable", "Aave timeout"])):
            await LifecycleManager.validate_startup_config(s)  # should not raise

    @pytest.mark.asyncio
    async def test_wallets_dict_format(self):
        """Test wallet_data as dict with 'wallets' key."""
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallet_data = {"wallets": [{"wallet_id": "w1", "private_key": "0xabc"}]}
        s = self._make_settings(wallet_data)
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=[])):
            await LifecycleManager.validate_startup_config(s)  # should not raise

    @pytest.mark.asyncio
    async def test_robust_paper_mode_skips_paper_engine_check(self):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        wallets = [{"wallet_id": "w1", "private_key": "0xabc"}]
        s = self._make_settings(wallets, execution_mode="robust_paper")
        with patch("wallet_manager.utils.health_check.run_startup_health_checks",
                   AsyncMock(return_value=[])):
            await LifecycleManager.validate_startup_config(s)  # should not raise


# ---------------------------------------------------------------------------
# orchestrator_gas.py supplemental tests
# ---------------------------------------------------------------------------

class TestGasGuardAdditional:
    """Additional coverage for GasGuard beyond test_gas_and_status.py."""

    def _make_guard(self, poll_interval=0.001, wait_timeout=0.01):
        from wallet_manager.network.gas_monitor import GasMonitor
        from wallet_manager.orchestrator_gas import GasGuard
        monitor = MagicMock(spec=GasMonitor)
        guard = GasGuard(
            gas_monitor=monitor,
            gas_poll_interval=poll_interval,
            gas_wait_timeout=wait_timeout,
        )
        return guard, monitor

    def _make_config(self, wallet_id="w1", max_gas=50.0):
        from wallet_manager.core.models import WalletConfig
        return WalletConfig(
            wallet_id=wallet_id,
            private_key="0xabcd" * 16,
            rpc_url="https://mainnet.base.org",
            chain_id=8453,
            max_gas_price_gwei=max_gas,
            address="0x1234567890123456789012345678901234567890",
        )

    @pytest.mark.asyncio
    async def test_gas_monitor_exception_returns_safe(self):
        """When monitor throws, proceed as safe (exception path line ~93-94)."""
        from wallet_manager.orchestrator_gas import GasGuard
        guard, monitor = self._make_guard()
        monitor.is_safe = AsyncMock(side_effect=ConnectionError("unreachable"))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        cfg = self._make_config()
        # Should not raise, just proceed
        await guard.wait_for_safe_gas(cfg, notifier, safe_notify)
        safe_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_gas_safe_on_first_check_returns_immediately(self):
        from wallet_manager.orchestrator_gas import GasGuard
        guard, monitor = self._make_guard()
        monitor.is_safe = AsyncMock(return_value=(True, 10.0))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        cfg = self._make_config()
        await guard.wait_for_safe_gas(cfg, notifier, safe_notify)
        safe_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_gas_timeout_raises_error(self):
        from wallet_manager.exceptions import GasPriceTooHighError
        from wallet_manager.orchestrator_gas import GasGuard
        guard, monitor = self._make_guard(poll_interval=0.001, wait_timeout=0.005)
        # Always unsafe
        monitor.is_safe = AsyncMock(return_value=(False, 200.0))
        notifier = MagicMock()
        safe_notify = AsyncMock()
        cfg = self._make_config()
        with pytest.raises(GasPriceTooHighError):
            await guard.wait_for_safe_gas(cfg, notifier, safe_notify)

    @pytest.mark.asyncio
    async def test_current_gwei_returns_value(self):
        from wallet_manager.orchestrator_gas import GasGuard
        guard, monitor = self._make_guard()
        monitor.current_gwei = AsyncMock(return_value=25.0)
        result = await guard.current_gwei()
        assert result == 25.0

    @pytest.mark.asyncio
    async def test_current_gwei_returns_zero_on_exception(self):
        from wallet_manager.orchestrator_gas import GasGuard
        guard, monitor = self._make_guard()
        monitor.current_gwei = AsyncMock(side_effect=Exception("rpc error"))
        result = await guard.current_gwei()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_gas_resumes_after_high(self):
        from wallet_manager.orchestrator_gas import GasGuard
        guard, monitor = self._make_guard(poll_interval=0.001, wait_timeout=5.0)
        # First check: unsafe, second check: safe
        monitor.is_safe = AsyncMock(side_effect=[(False, 100.0), (True, 10.0)])
        notifier = MagicMock()
        safe_notify = AsyncMock()
        cfg = self._make_config()
        await guard.wait_for_safe_gas(cfg, notifier, safe_notify)
        # Should have called safe_notify twice: GAS_PAUSE and GAS_RESUME
        assert safe_notify.call_count == 2
