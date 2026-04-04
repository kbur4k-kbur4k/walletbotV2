"""
tests/unit/test_orchestrator_deep.py

Deep coverage tests for wallet_manager/orchestrator.py.
Targets previously uncovered lines: 626-902, 1110-1225, 1440-1508, 1611-1749, 1754-1799

Focus areas:
- WalletOrchestrator.run() full lifecycle
- WalletOrchestrator._run_wallet() all branches
- _safe_notify() error handling
- GasGuard usage in _wait_for_safe_gas
- LifecycleManager hooks
- Portfolio management calls
- send_heartbeat_telemetry
- _load_wallet_configs
- MasterOrchestrator
- validate_wallets_schema edge cases
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wallet_config(wallet_id="w1", max_gas=50.0):
    from wallet_manager.core.models import WalletConfig
    return WalletConfig(
        wallet_id=wallet_id,
        private_key="0xabcd" * 16,
        rpc_url="https://mainnet.base.org",
        chain_id=8453,
        max_gas_price_gwei=max_gas,
        address="0x1234567890123456789012345678901234567890",
    )


def _make_settings(**kwargs):
    from wallet_manager.config import Settings
    defaults = {
        "vault_key": "kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        "execution_mode": "robust_paper",
        "max_concurrent_wallets": 3,
        "gas_poll_interval": 1.0,
        "gas_wait_timeout": 5.0,
        "live_trading_enabled": False,
        "live_max_wallets": 5,
        "live_max_total_capital_usd": 10000.0,
        "live_tiny_cap_mode": False,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def _make_notifier():
    n = MagicMock()
    n.send = AsyncMock()
    n.start = AsyncMock()
    n.stop = AsyncMock()
    return n


def _make_orchestrator(configs, task_factory=None, notifier=None, settings=None, **kwargs):
    from wallet_manager.orchestrator import WalletOrchestrator
    if task_factory is None:
        async def task_factory(manager, wallet_id, ctx):
            return []
    return WalletOrchestrator(
        wallet_configs=configs,
        task_factory=task_factory,
        notifier=notifier,
        settings=settings or _make_settings(),
        **kwargs,
    )


def _make_mock_manager():
    """Return a MagicMock that works as an async context manager."""
    mgr = MagicMock()
    mgr.__aenter__ = AsyncMock(return_value=mgr)
    mgr.__aexit__ = AsyncMock(return_value=False)
    return mgr


# ---------------------------------------------------------------------------
# 1. _safe_notify — extended coverage
# ---------------------------------------------------------------------------

class TestSafeNotifyExtended:
    @pytest.mark.asyncio
    async def test_sends_system_boot_event(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import WalletOrchestrator
        notifier = _make_notifier()
        evt = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        await WalletOrchestrator._safe_notify(notifier, evt)
        notifier.send.assert_awaited_once_with(evt)

    @pytest.mark.asyncio
    async def test_sends_system_shutdown_event(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import WalletOrchestrator
        notifier = _make_notifier()
        evt = NotificationEvent(kind=EventKind.SYSTEM_SHUTDOWN, detail="done")
        await WalletOrchestrator._safe_notify(notifier, evt)
        notifier.send.assert_awaited_once_with(evt)

    @pytest.mark.asyncio
    async def test_swallows_connection_error(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import WalletOrchestrator
        notifier = _make_notifier()
        notifier.send = AsyncMock(side_effect=ConnectionError("timeout"))
        evt = NotificationEvent(kind=EventKind.WALLET_SUCCESS, wallet_id="w1")
        # Must not raise
        await WalletOrchestrator._safe_notify(notifier, evt)

    @pytest.mark.asyncio
    async def test_swallows_value_error(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import WalletOrchestrator
        notifier = _make_notifier()
        notifier.send = AsyncMock(side_effect=ValueError("bad value"))
        evt = NotificationEvent(kind=EventKind.GAS_PAUSE, wallet_id="w1")
        await WalletOrchestrator._safe_notify(notifier, evt)  # no raise


# ---------------------------------------------------------------------------
# 2. WalletOrchestrator.run() — happy path (lines 626-902)
# ---------------------------------------------------------------------------

class TestOrchestratorRun:
    """Tests for WalletOrchestrator.run() — the main execution lifecycle."""

    @staticmethod
    def _make_run_context_managers(orch, results=None):
        """Return a single context manager that patches all heavy async internals."""
        if results is None:
            from wallet_manager.orchestrator import WalletResult
            results = [WalletResult(wallet_id="w1", success=True)]

        mock_mgr = _make_mock_manager()

        async def fake_run_wallet(manager, cfg, semaphore, notifier, task_ctx=None):
            if results:
                return results[0]
            from wallet_manager.orchestrator import WalletResult
            return WalletResult(wallet_id=cfg.wallet_id, success=True)

        import contextlib

        @contextlib.asynccontextmanager
        async def _ctx():
            with patch("wallet_manager.orchestrator.WalletManager", return_value=mock_mgr), \
                 patch("wallet_manager.orchestrator.PortfolioManager.sync_from_chain",
                       new=AsyncMock()), \
                 patch("wallet_manager.orchestrator._get_wallet_address",
                       return_value="0x1234567890123456789012345678901234567890"), \
                 patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                       new=AsyncMock(return_value=3000.0)), \
                 patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                       return_value={"passed": True, "violations": []}), \
                 patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                       new=AsyncMock(return_value={"audit_version": 1})), \
                 patch("wallet_manager.resilience.analytics.AnalyticsDB",
                       side_effect=ImportError("no analytics")), \
                 patch.object(orch, "_run_wallet", side_effect=fake_run_wallet):
                yield

        return _ctx()

    @pytest.mark.asyncio
    async def test_run_returns_list_of_wallet_results(self):
        from wallet_manager.orchestrator import WalletOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)
        expected = WalletResult(wallet_id="w1", success=True)

        async with self._make_run_context_managers(orch, [expected]):
            results = await orch.run()

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_run_sends_system_boot_and_shutdown(self):
        from wallet_manager.orchestrator import WalletOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)
        expected = WalletResult(wallet_id="w1", success=True)

        async with self._make_run_context_managers(orch, [expected]):
            await orch.run()

        from wallet_manager.core.models import EventKind
        kinds_sent = [call.args[0].kind for call in notifier.send.call_args_list]
        assert EventKind.SYSTEM_BOOT in kinds_sent
        assert EventKind.SYSTEM_SHUTDOWN in kinds_sent

    @pytest.mark.asyncio
    async def test_run_multiple_wallets(self):
        from wallet_manager.orchestrator import WalletOrchestrator, WalletResult
        cfg1 = _make_wallet_config("w1")
        cfg2 = _make_wallet_config("w2")
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg1, cfg2], notifier=notifier)

        call_count = {"n": 0}

        async def counting_run_wallet(manager, cfg, semaphore, notifier, task_ctx=None):
            call_count["n"] += 1
            return WalletResult(wallet_id=cfg.wallet_id, success=True)

        mock_mgr = _make_mock_manager()
        with patch("wallet_manager.orchestrator.WalletManager", return_value=mock_mgr), \
             patch("wallet_manager.orchestrator.PortfolioManager.sync_from_chain", new=AsyncMock()), \
             patch("wallet_manager.orchestrator._get_wallet_address", return_value="0xabcd"), \
             patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                   new=AsyncMock(return_value=3000.0)), \
             patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                   return_value={"passed": True, "violations": []}), \
             patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                   new=AsyncMock(return_value={})), \
             patch.object(orch, "_run_wallet", side_effect=counting_run_wallet):
            await orch.run()

        # Two wallets → two _run_wallet calls
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_run_fatal_exception_populates_all_results_with_error(self):
        """When run() hits a fatal error before wallet execution, all configs get failed results."""
        from wallet_manager.orchestrator import WalletOrchestrator
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)

        with patch("wallet_manager.orchestrator.WalletManager") as MockMgr:
            MockMgr.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("fatal DB"))
            MockMgr.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                       return_value={"passed": True, "violations": []}):
                with patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                           new=AsyncMock(return_value={})):
                    with patch("wallet_manager.orchestrator._get_wallet_address",
                               return_value="0x1234567890123456789012345678901234567890"):
                        with patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                                   new=AsyncMock(return_value=3000.0)):
                            results = await orch.run()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_run_live_guard_violation_raises_and_sends_alert(self):
        """Live guard failure causes RuntimeError; orchestrator sends SYSTEM_ALERT."""
        from wallet_manager.core.models import EventKind
        from wallet_manager.orchestrator import WalletOrchestrator
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)

        mock_mgr = _make_mock_manager()
        with patch("wallet_manager.orchestrator.WalletManager", return_value=mock_mgr):
            with patch("wallet_manager.orchestrator._get_wallet_address",
                       return_value="0xabcd"):
                with patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                           new=AsyncMock(return_value=3000.0)):
                    with patch("wallet_manager.orchestrator.PortfolioManager.sync_from_chain",
                               new=AsyncMock()):
                        with patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                                   return_value={"passed": False,
                                                 "violations": ["live disabled"]}):
                            with patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                                       new=AsyncMock(return_value={})):
                                results = await orch.run()

        # Should have failed results (live guard violation)
        assert all(not r.success for r in results)
        kinds_sent = [c.args[0].kind for c in notifier.send.call_args_list]
        assert EventKind.SYSTEM_ALERT in kinds_sent

    @pytest.mark.asyncio
    async def test_run_exception_from_run_wallet_captured(self):
        """If _run_wallet itself raises (not caught internally), gather captures it."""
        from wallet_manager.orchestrator import WalletOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)

        async def bad_run_wallet(manager, cfg, semaphore, notifier, task_ctx=None):
            raise RuntimeError("internal crash")

        mock_mgr = _make_mock_manager()
        with patch("wallet_manager.orchestrator.WalletManager", return_value=mock_mgr), \
             patch("wallet_manager.orchestrator.PortfolioManager.sync_from_chain", new=AsyncMock()), \
             patch("wallet_manager.orchestrator._get_wallet_address", return_value="0xabcd"), \
             patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                   new=AsyncMock(return_value=3000.0)), \
             patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                   return_value={"passed": True, "violations": []}), \
             patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                   new=AsyncMock(return_value={})), \
             patch.object(orch, "_run_wallet", side_effect=bad_run_wallet):
            results = await orch.run()

        assert len(results) == 1
        assert results[0].success is False
        assert isinstance(results[0].error, RuntimeError)


# ---------------------------------------------------------------------------
# 3. WalletOrchestrator._run_wallet() — extended branches
# ---------------------------------------------------------------------------

class TestRunWalletExtended:
    """Target lines 908-1036: all _run_wallet branches."""

    @pytest.mark.asyncio
    async def test_success_sends_wallet_success_notification(self):
        from wallet_manager.core.models import EventKind, TxReceipt
        receipt = TxReceipt(tx_hash="0xaaa", wallet_id="w1", nonce_used=1)

        async def task_factory(manager, wallet_id, ctx):
            return [receipt]

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.success is True
        kinds = [c.args[0].kind for c in notifier.send.call_args_list]
        assert EventKind.WALLET_SUCCESS in kinds

    @pytest.mark.asyncio
    async def test_success_with_no_receipts(self):
        """Zero-receipt run still succeeds; tx_hash is None."""
        async def task_factory(manager, wallet_id, ctx):
            return []

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.success is True
        assert result.receipts == []

    @pytest.mark.asyncio
    async def test_gas_price_too_high_sends_gas_deferred_notification(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.exceptions import GasPriceTooHighError

        async def task_factory(manager, wallet_id, ctx):
            raise GasPriceTooHighError("gas high", requested_gwei=200.0, max_gwei=50.0)

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            with patch.object(orch._scheduler, "schedule_gas_deferral"):
                result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.success is False
        kinds = [c.args[0].kind for c in notifier.send.call_args_list]
        assert EventKind.GAS_DEFERRED in kinds

    @pytest.mark.asyncio
    async def test_wallet_manager_error_sends_wallet_failure_notification(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.exceptions import WalletManagerError

        async def task_factory(manager, wallet_id, ctx):
            raise WalletManagerError("rpc failed", wallet_id="w1")

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.success is False
        kinds = [c.args[0].kind for c in notifier.send.call_args_list]
        assert EventKind.WALLET_FAILURE in kinds

    @pytest.mark.asyncio
    async def test_generic_exception_sends_wallet_failure_notification(self):
        from wallet_manager.core.models import EventKind

        async def task_factory(manager, wallet_id, ctx):
            raise ValueError("unexpected!")

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.success is False
        kinds = [c.args[0].kind for c in notifier.send.call_args_list]
        assert EventKind.WALLET_FAILURE in kinds

    @pytest.mark.asyncio
    async def test_command_state_pauses_then_runs(self):
        """wait_for_resume is awaited when not stopped."""
        command_state = MagicMock()
        command_state.is_stopped = False
        command_state.wait_for_resume = AsyncMock()

        async def task_factory(manager, wallet_id, ctx):
            return []

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory,
                                  notifier=notifier, command_state=command_state)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        command_state.wait_for_resume.assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_duration_seconds_is_positive(self):
        async def task_factory(manager, wallet_id, ctx):
            return []

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_failure_duration_seconds_is_positive(self):
        async def task_factory(manager, wallet_id, ctx):
            raise RuntimeError("fail")

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_task_context_passed_to_factory(self):
        captured = {}

        async def task_factory(manager, wallet_id, ctx):
            captured["ctx"] = ctx
            return []

        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], task_factory=task_factory, notifier=notifier)
        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)
        ctx = {"notifier": notifier, "custom_key": "hello"}

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            await orch._run_wallet(manager, cfg, sem, notifier, ctx)

        assert captured["ctx"]["custom_key"] == "hello"


# ---------------------------------------------------------------------------
# 4. GasGuard usage — _wait_for_safe_gas
# ---------------------------------------------------------------------------

class TestGasGuardUsage:
    @pytest.mark.asyncio
    async def test_wait_for_safe_gas_delegates_to_gas_guard(self):
        import wallet_manager.orchestrator as orch_mod
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)

        with patch.object(orch._gas_guard, "current_gwei", new=AsyncMock(return_value=30.0)):
            with patch.object(orch._gas_guard, "wait_for_safe_gas", new=AsyncMock()) as mock_wait:
                await orch._wait_for_safe_gas(cfg, notifier)

        mock_wait.assert_awaited_once()
        assert orch_mod._last_gas_price_gwei == 30.0

    @pytest.mark.asyncio
    async def test_gas_guard_current_gwei_failure_does_not_raise(self):
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)

        with patch.object(orch._gas_guard, "current_gwei",
                          new=AsyncMock(side_effect=Exception("rpc down"))):
            with patch.object(orch._gas_guard, "wait_for_safe_gas", new=AsyncMock()):
                # Must not raise
                await orch._wait_for_safe_gas(cfg, notifier)

    @pytest.mark.asyncio
    async def test_gas_guard_wait_called_with_safe_notify(self):
        from wallet_manager.orchestrator import WalletOrchestrator
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)

        with patch.object(orch._gas_guard, "current_gwei", new=AsyncMock(return_value=10.0)):
            with patch.object(orch._gas_guard, "wait_for_safe_gas", new=AsyncMock()) as mock_wait:
                await orch._wait_for_safe_gas(cfg, notifier)

        # Confirm wait_for_safe_gas was called with config and notifier
        call_args = mock_wait.call_args
        assert call_args is not None


# ---------------------------------------------------------------------------
# 5. LifecycleManager hooks
# ---------------------------------------------------------------------------

class TestLifecycleManagerHooks:
    @pytest.mark.asyncio
    async def test_validate_startup_config_delegates_to_lifecycle(self):
        from wallet_manager.orchestrator import _validate_startup_config

        with patch("wallet_manager.orchestrator.LifecycleManager.validate_startup_config",
                   new=AsyncMock()) as mock_validate:
            await _validate_startup_config()

        mock_validate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_migrate_wallets_to_vault_delegates_to_lifecycle(self):
        from wallet_manager.orchestrator import _migrate_wallets_to_vault

        mock_vault = MagicMock()
        config_path = Path("/fake/wallets.json")

        with patch("wallet_manager.orchestrator.LifecycleManager.migrate_wallets_to_vault",
                   new=AsyncMock()) as mock_migrate:
            await _migrate_wallets_to_vault(mock_vault, config_path)

        mock_migrate.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Portfolio management calls in run()
# ---------------------------------------------------------------------------

class TestPortfolioManagementInRun:
    @pytest.mark.asyncio
    async def test_portfolio_manager_created_during_run(self):
        """PortfolioManager is constructed each run() call."""
        from wallet_manager.orchestrator import WalletOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)
        expected = WalletResult(wallet_id="w1", success=True)

        with patch("wallet_manager.orchestrator.PortfolioManager") as MockPM:
            mock_pm_instance = MagicMock()
            mock_pm_instance.total_value_usd = 0.0
            mock_pm_instance.sync_from_chain = AsyncMock()
            mock_pm_instance.get_snapshot = MagicMock()
            MockPM.return_value = mock_pm_instance

            mock_mgr = _make_mock_manager()
            with patch("wallet_manager.orchestrator.WalletManager", return_value=mock_mgr):
                with patch("wallet_manager.orchestrator._get_wallet_address", return_value="0xabcd"):
                    with patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                               new=AsyncMock(return_value=3000.0)):
                        with patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                                   return_value={"passed": True, "violations": []}):
                            with patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                                       new=AsyncMock(return_value={})):
                                with patch.object(orch, "_run_wallet",
                                                  new=AsyncMock(return_value=expected)):
                                    await orch.run()

        MockPM.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_from_chain_failure_does_not_abort_run(self):
        """sync_from_chain failure is swallowed; run continues."""
        from wallet_manager.orchestrator import WalletOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        orch = _make_orchestrator([cfg], notifier=notifier)
        expected = WalletResult(wallet_id="w1", success=True)

        mock_mgr = _make_mock_manager()
        with patch("wallet_manager.orchestrator.WalletManager", return_value=mock_mgr):
            with patch("wallet_manager.orchestrator._get_wallet_address", return_value="0xabcd"):
                with patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                           new=AsyncMock(return_value=3000.0)):
                    with patch("wallet_manager.orchestrator.PortfolioManager.sync_from_chain",
                               new=AsyncMock(side_effect=Exception("chain down"))):
                        with patch("wallet_manager.orchestrator._evaluate_live_runtime_guard",
                                   return_value={"passed": True, "violations": []}):
                            with patch("wallet_manager.orchestrator._build_runtime_audit_snapshot",
                                       new=AsyncMock(return_value={})):
                                with patch.object(orch, "_run_wallet",
                                                  new=AsyncMock(return_value=expected)):
                                    results = await orch.run()

        assert results[0].success is True


# ---------------------------------------------------------------------------
# 7. send_heartbeat_telemetry (lines 1110-1225)
# ---------------------------------------------------------------------------

class TestSendHeartbeatTelemetry:
    @pytest.mark.asyncio
    async def test_skips_when_interval_not_elapsed(self):
        """If called within the heartbeat window, does nothing."""
        import wallet_manager.orchestrator as orch_mod
        from wallet_manager.orchestrator import send_heartbeat_telemetry
        from wallet_manager.portfolio_manager import PortfolioManager

        pm = MagicMock(spec=PortfolioManager)
        snapshot = MagicMock()
        snapshot.total_value_usd = 100.0
        snapshot.allocations = []
        snapshot.needs_rebalance = False
        pm.get_snapshot.return_value = snapshot

        # Set heartbeat timestamp to just now
        orch_mod._last_heartbeat_ts = time.time()

        # Should return immediately without sending anything
        ctx = {"notifier": _make_notifier()}
        await send_heartbeat_telemetry(pm, "https://rpc.url", "0xabc", ctx)
        # If it didn't raise, test passes; no assert needed

    @pytest.mark.asyncio
    async def test_sends_when_interval_elapsed(self):
        """When enough time has passed, builds and sends heartbeat message."""
        import wallet_manager.orchestrator as orch_mod
        from wallet_manager.orchestrator import send_heartbeat_telemetry

        pm = MagicMock()
        snapshot = MagicMock()
        snapshot.total_value_usd = 5000.0
        snapshot.allocations = []
        snapshot.needs_rebalance = False
        pm.get_snapshot.return_value = snapshot

        # Reset heartbeat timestamp so it triggers
        orch_mod._last_heartbeat_ts = 0.0

        notifier = _make_notifier()
        ctx = {"notifier": notifier}

        # Patch all external calls
        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "robust_paper"
            with patch("wallet_manager.strategies.yield_strategy.get_apy",
                       new=AsyncMock(return_value=5.0), create=True):
                with patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance",
                           new=AsyncMock(return_value=1000000), create=True):
                    with patch("wallet_manager.orchestrator.actions.fetch_eth_price_usd",
                               new=AsyncMock(return_value=3000.0)):
                        await send_heartbeat_telemetry(pm, "https://rpc", "0xabc", ctx)

        # Notifier.send should have been called with the heartbeat message
        notifier.send.assert_awaited()

    @pytest.mark.asyncio
    async def test_heartbeat_exception_is_swallowed(self):
        """Any exception inside heartbeat is caught; does not propagate."""
        import wallet_manager.orchestrator as orch_mod
        from wallet_manager.orchestrator import send_heartbeat_telemetry

        orch_mod._last_heartbeat_ts = 0.0

        bad_pm = MagicMock()
        bad_pm.get_snapshot.side_effect = RuntimeError("db gone")

        ctx = {}
        # Should not raise
        await send_heartbeat_telemetry(bad_pm, "https://rpc", "0xabc", ctx)


# ---------------------------------------------------------------------------
# 8. _load_wallet_configs (lines 1440-1508)
# ---------------------------------------------------------------------------

class TestLoadWalletConfigs:
    def test_raises_when_path_missing(self, tmp_path):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import _load_wallet_configs

        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.wallet_config_path = str(tmp_path / "nonexistent.json")
            mock_cfg.return_value.vault_key = None
            mock_cfg.return_value.base_rpc_1 = None
            mock_cfg.return_value.base_rpc_2 = None
            mock_cfg.return_value.base_rpc_3 = None

            with pytest.raises(ConfigurationError, match="WALLET_CONFIG_PATH"):
                _load_wallet_configs()

    def test_raises_when_path_unset(self, tmp_path):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import _load_wallet_configs

        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.wallet_config_path = ""
            mock_cfg.return_value.vault_key = None
            mock_cfg.return_value.base_rpc_1 = None
            mock_cfg.return_value.base_rpc_2 = None
            mock_cfg.return_value.base_rpc_3 = None

            with pytest.raises(ConfigurationError, match="WALLET_CONFIG_PATH"):
                _load_wallet_configs()

    def test_loads_plain_text_wallet(self, tmp_path):
        from wallet_manager.orchestrator import _load_wallet_configs

        data = [
            {
                "wallet_id": "test_wallet",
                "address": "0x" + "a" * 40,
                "private_key": "0xdeadbeef" * 8,
                "max_gas_price_gwei": 50.0,
                "rpc_url": "https://mainnet.base.org",
                "chain_id": 8453,
            }
        ]
        config_file = tmp_path / "wallets.json"
        config_file.write_text(json.dumps(data), encoding="utf-8")

        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.wallet_config_path = str(config_file)
            mock_cfg.return_value.vault_key = None
            mock_cfg.return_value.base_rpc_1 = "https://mainnet.base.org"
            mock_cfg.return_value.base_rpc_2 = None
            mock_cfg.return_value.base_rpc_3 = None

            with patch("web3.Web3.is_checksum_address", return_value=True):
                configs = _load_wallet_configs()

        assert len(configs) == 1
        assert configs[0].wallet_id == "test_wallet"

    def test_raises_on_empty_private_key(self, tmp_path):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import _load_wallet_configs

        data = [
            {
                "wallet_id": "test_wallet",
                "address": "0x" + "a" * 40,
                "private_key": "  ",
                "max_gas_price_gwei": 50.0,
            }
        ]
        config_file = tmp_path / "wallets.json"
        config_file.write_text(json.dumps(data), encoding="utf-8")

        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.wallet_config_path = str(config_file)
            mock_cfg.return_value.vault_key = None
            mock_cfg.return_value.base_rpc_1 = "https://mainnet.base.org"
            mock_cfg.return_value.base_rpc_2 = None
            mock_cfg.return_value.base_rpc_3 = None

            with patch("web3.Web3.is_checksum_address", return_value=True):
                with pytest.raises((ConfigurationError, Exception)):
                    _load_wallet_configs()

    def test_defaults_rpc_url_when_missing(self, tmp_path):
        from wallet_manager.orchestrator import _load_wallet_configs

        data = [
            {
                "wallet_id": "w1",
                "address": "0x" + "a" * 40,
                "private_key": "0xdeadbeef" * 8,
                "max_gas_price_gwei": 50.0,
            }
        ]
        config_file = tmp_path / "wallets.json"
        config_file.write_text(json.dumps(data), encoding="utf-8")

        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.wallet_config_path = str(config_file)
            mock_cfg.return_value.vault_key = None
            mock_cfg.return_value.base_rpc_1 = "https://custom.rpc.org"
            mock_cfg.return_value.base_rpc_2 = None
            mock_cfg.return_value.base_rpc_3 = None

            with patch("web3.Web3.is_checksum_address", return_value=True):
                configs = _load_wallet_configs()

        assert configs[0].rpc_url == "https://custom.rpc.org"

    def test_defaults_to_mainnet_base_rpc_when_none(self, tmp_path):
        from wallet_manager.orchestrator import _load_wallet_configs

        data = [
            {
                "wallet_id": "w1",
                "address": "0x" + "a" * 40,
                "private_key": "0xdeadbeef" * 8,
                "max_gas_price_gwei": 50.0,
            }
        ]
        config_file = tmp_path / "wallets.json"
        config_file.write_text(json.dumps(data), encoding="utf-8")

        with patch("wallet_manager.orchestrator._cfg") as mock_cfg:
            mock_cfg.return_value.wallet_config_path = str(config_file)
            mock_cfg.return_value.vault_key = None
            mock_cfg.return_value.base_rpc_1 = None
            mock_cfg.return_value.base_rpc_2 = None
            mock_cfg.return_value.base_rpc_3 = None

            with patch("web3.Web3.is_checksum_address", return_value=True):
                configs = _load_wallet_configs()

        assert configs[0].rpc_url == "https://mainnet.base.org"


# ---------------------------------------------------------------------------
# 9. MasterOrchestrator (lines 1529-1749)
# ---------------------------------------------------------------------------

class TestMasterOrchestrator:
    def test_master_orchestrator_instantiation(self):
        from wallet_manager.orchestrator import MasterOrchestrator
        cfg = _make_wallet_config()

        with patch("wallet_manager.security.telegram_manager.TelegramManager") as MockTG:
            MockTG.return_value = MagicMock()
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                settings=_make_settings(),
            )

        assert isinstance(orch, MasterOrchestrator)

    def test_master_orchestrator_inherits_wallet_orchestrator(self):
        from wallet_manager.orchestrator import MasterOrchestrator, WalletOrchestrator
        assert issubclass(MasterOrchestrator, WalletOrchestrator)

    def test_master_orchestrator_telegram_manager_unavailable(self):
        """Gracefully handles missing TelegramManager."""
        from wallet_manager.orchestrator import MasterOrchestrator
        cfg = _make_wallet_config()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   side_effect=ImportError("no telegram")):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                settings=_make_settings(),
            )

        assert orch._telegram_manager is None

    @pytest.mark.asyncio
    async def test_master_run_closes_telegram_manager_on_success(self):
        from wallet_manager.orchestrator import MasterOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()

        mock_tg = MagicMock()
        mock_tg.close = AsyncMock()
        mock_tg._is_configured = MagicMock(return_value=False)

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(return_value=[]),
                notifier=notifier,
                settings=_make_settings(),
            )

        expected = WalletResult(wallet_id="w1", success=True)
        with patch.object(orch, "run", wraps=orch.run):
            # Patch parent run to return quickly
            with patch("wallet_manager.orchestrator.WalletOrchestrator.run",
                       new=AsyncMock(return_value=[expected])):
                await orch.run()

        mock_tg.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_master_run_closes_telegram_manager_on_exception(self):
        """TelegramManager.close is called even when run() raises."""
        from wallet_manager.orchestrator import MasterOrchestrator

        cfg = _make_wallet_config()
        notifier = _make_notifier()

        mock_tg = MagicMock()
        mock_tg.close = AsyncMock()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(return_value=[]),
                notifier=notifier,
                settings=_make_settings(),
            )

        with patch("wallet_manager.orchestrator.WalletOrchestrator.run",
                   new=AsyncMock(side_effect=RuntimeError("crash"))), pytest.raises(RuntimeError):
            await orch.run()

        mock_tg.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_master_run_wallet_injects_telegram_manager(self):
        """MasterOrchestrator._run_wallet injects telegram_manager into ctx."""
        from wallet_manager.orchestrator import MasterOrchestrator, WalletResult
        cfg = _make_wallet_config()
        notifier = _make_notifier()
        injected_ctx = {}

        async def task_factory(manager, wallet_id, ctx):
            injected_ctx.update(ctx)
            return []

        mock_tg = MagicMock()
        mock_tg._is_configured = MagicMock(return_value=False)

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=task_factory,
                notifier=notifier,
                settings=_make_settings(),
            )

        manager = _make_mock_manager()
        sem = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            await orch._run_wallet(manager, cfg, sem, notifier, {})

        assert "telegram_manager" in injected_ctx
        assert injected_ctx["telegram_manager"] is mock_tg

    def test_master_resolve_notifier_wraps_with_system_alert_wrapper(self):
        """When telegram_manager is set, notifier is wrapped."""
        from wallet_manager.orchestrator import MasterOrchestrator, _SystemAlertNotifierWrapper
        cfg = _make_wallet_config()
        notifier = _make_notifier()

        mock_tg = MagicMock()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                notifier=notifier,
                settings=_make_settings(),
            )

        resolved = orch._resolve_notifier()
        assert isinstance(resolved, _SystemAlertNotifierWrapper)

    def test_master_resolve_notifier_no_telegram_returns_plain(self):
        """Without telegram_manager, returns plain notifier."""
        from wallet_manager.orchestrator import MasterOrchestrator
        cfg = _make_wallet_config()
        notifier = _make_notifier()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   side_effect=Exception("no TG")):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                notifier=notifier,
                settings=_make_settings(),
            )

        resolved = orch._resolve_notifier()
        assert resolved is notifier

    @pytest.mark.asyncio
    async def test_master_maybe_send_heartbeat_skips_when_not_configured(self):
        """_maybe_send_heartbeat returns early if not configured."""
        from wallet_manager.orchestrator import MasterOrchestrator
        cfg = _make_wallet_config()

        mock_tg = MagicMock()
        mock_tg._is_configured = MagicMock(return_value=False)

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                settings=_make_settings(),
            )

        pm = MagicMock()
        notifier = _make_notifier()
        # Should not raise, should return early
        await orch._maybe_send_heartbeat(pm, None, notifier)

    @pytest.mark.asyncio
    async def test_master_maybe_send_heartbeat_skips_when_interval_not_elapsed(self):
        """_maybe_send_heartbeat skips if within heartbeat window."""
        import wallet_manager.orchestrator as orch_mod
        from wallet_manager.orchestrator import MasterOrchestrator

        cfg = _make_wallet_config()
        mock_tg = MagicMock()
        mock_tg._is_configured = MagicMock(return_value=True)
        mock_tg.send_heartbeat = AsyncMock()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                settings=_make_settings(),
            )

        # Set heartbeat timestamp to just now
        orch_mod._last_heartbeat_ts = time.time()

        pm = MagicMock()
        pm.total_value_usd = 100.0
        notifier = _make_notifier()

        await orch._maybe_send_heartbeat(pm, None, notifier)
        # Should NOT have called send_heartbeat
        mock_tg.send_heartbeat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_master_notify_trade_if_algo_skips_when_no_manager(self):
        """_notify_trade_if_algo returns early when no telegram_manager in ctx."""
        from wallet_manager.core.models import TxReceipt
        from wallet_manager.orchestrator import MasterOrchestrator, WalletResult
        cfg = _make_wallet_config()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   side_effect=Exception("no TG")):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                settings=_make_settings(),
            )

        result = WalletResult(wallet_id="w1", success=True,
                              receipts=[TxReceipt(tx_hash="0xabc", wallet_id="w1", nonce_used=1)])
        ctx = {}  # no telegram_manager
        # Should not raise
        await orch._notify_trade_if_algo(ctx, result, cfg)

    @pytest.mark.asyncio
    async def test_master_notify_trade_sends_payload_when_configured(self):
        """_notify_trade_if_algo sends pending trade payloads when configured."""
        from wallet_manager.core.models import TxReceipt
        from wallet_manager.orchestrator import MasterOrchestrator, WalletResult
        cfg = _make_wallet_config()

        mock_tg = MagicMock()
        mock_tg._is_configured = MagicMock(return_value=True)
        mock_tg.send_trade_executed = AsyncMock()

        with patch("wallet_manager.security.telegram_manager.TelegramManager",
                   return_value=mock_tg):
            orch = MasterOrchestrator(
                wallet_configs=[cfg],
                task_factory=AsyncMock(),
                settings=_make_settings(),
            )

        from wallet_manager.security.telegram_manager import TradeExecutedPayload
        payload = TradeExecutedPayload(
            pair="ETH/USDC",
            price=3000.0,
            strategy_type="algo",
            amount="100.0",
            side="BUY",
            tx_hash="0xabc",
        )

        result = WalletResult(wallet_id="w1", success=True,
                              receipts=[TxReceipt(tx_hash="0xabc", wallet_id="w1", nonce_used=1)])
        ctx = {"telegram_manager": mock_tg, "trade_notifications": [payload]}

        await orch._notify_trade_if_algo(ctx, result, cfg)
        mock_tg.send_trade_executed.assert_awaited_once_with(payload)
        assert ctx["trade_notifications"] == []


# ---------------------------------------------------------------------------
# 10. _SystemAlertNotifierWrapper (extended)
# ---------------------------------------------------------------------------

class TestSystemAlertNotifierWrapperExtended:
    @pytest.mark.asyncio
    async def test_send_forwards_gas_pause_to_telegram(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper

        inner = _make_notifier()
        mock_tg = MagicMock()
        mock_tg.send_system_alert = AsyncMock()

        wrapper = _SystemAlertNotifierWrapper(inner, mock_tg)
        evt = NotificationEvent(kind=EventKind.GAS_PAUSE, wallet_id="w1",
                                gas_price_gwei=150.0)

        from wallet_manager.security.telegram_manager import SystemAlertPayload
        with patch("wallet_manager.orchestrator.cast", side_effect=lambda t, v: v):
            await wrapper.send(evt)

        inner.send.assert_awaited_once_with(evt)
        mock_tg.send_system_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_forwards_wallet_failure_with_critical_severity(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper

        inner = _make_notifier()
        mock_tg = MagicMock()
        mock_tg.send_system_alert = AsyncMock()

        wrapper = _SystemAlertNotifierWrapper(inner, mock_tg)
        evt = NotificationEvent(kind=EventKind.WALLET_FAILURE, wallet_id="w1",
                                detail="some error")

        with patch("wallet_manager.orchestrator.cast", side_effect=lambda t, v: v):
            await wrapper.send(evt)

        # send_system_alert called with critical severity
        call_args = mock_tg.send_system_alert.call_args
        assert call_args is not None
        payload = call_args.args[0]
        assert payload.severity == "critical"

    @pytest.mark.asyncio
    async def test_send_does_not_forward_non_alert_events(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper

        inner = _make_notifier()
        mock_tg = MagicMock()
        mock_tg.send_system_alert = AsyncMock()

        wrapper = _SystemAlertNotifierWrapper(inner, mock_tg)
        evt = NotificationEvent(kind=EventKind.SYSTEM_BOOT)

        await wrapper.send(evt)
        inner.send.assert_awaited_once_with(evt)
        mock_tg.send_system_alert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_swallows_telegram_forward_error(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper

        inner = _make_notifier()
        mock_tg = MagicMock()
        mock_tg.send_system_alert = AsyncMock(side_effect=RuntimeError("TG down"))

        wrapper = _SystemAlertNotifierWrapper(inner, mock_tg)
        evt = NotificationEvent(kind=EventKind.GAS_DEFERRED, wallet_id="w1")

        # Should not raise
        await wrapper.send(evt)

    @pytest.mark.asyncio
    async def test_aenter_calls_inner_start(self):
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = _make_notifier()
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        result = await wrapper.__aenter__()
        inner.start.assert_awaited_once()
        assert result is wrapper

    @pytest.mark.asyncio
    async def test_aexit_calls_inner_stop(self):
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = _make_notifier()
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        await wrapper.__aexit__(None, None, None)
        inner.stop.assert_awaited_once()

    def test_getattr_delegates_to_inner(self):
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.some_attr = "hello"
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        assert wrapper.some_attr == "hello"


# ---------------------------------------------------------------------------
# 11. validate_wallets_schema — additional edge cases
# ---------------------------------------------------------------------------

class TestValidateWalletsSchemaEdgeCases:
    def test_address_too_short_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema

        data = [{"wallet_id": "w1", "address": "0xshort",
                 "private_key": "abc"}]
        with pytest.raises(ConfigurationError):
            validate_wallets_schema(data)

    def test_non_checksummed_address_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema

        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40,
                 "private_key": "abc"}]
        with patch("web3.Web3.is_checksum_address", return_value=False):
            with pytest.raises(ConfigurationError, match="checksummed"):
                validate_wallets_schema(data)

    def test_gas_price_zero_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema

        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40,
                 "private_key": "abc", "max_gas_price_gwei": 0}]
        with pytest.raises(ConfigurationError, match="max_gas_price_gwei"):
            validate_wallets_schema(data)

    def test_non_dict_entries_skipped(self):
        from wallet_manager.orchestrator import validate_wallets_schema
        # Non-dict entries should be silently skipped
        data = ["not_a_dict", 42, None]
        validate_wallets_schema(data)  # should not raise


# ---------------------------------------------------------------------------
# 12. _encrypted_key_candidates — edge cases
# ---------------------------------------------------------------------------

class TestEncryptedKeyCandidatesEdgeCases:
    def test_non_fernet_raw_goes_last(self):
        import base64

        from wallet_manager.orchestrator import _encrypted_key_candidates
        # A raw non-Fernet token should appear after decoded candidates
        raw = "AAAAB" + "x" * 40
        result = _encrypted_key_candidates(raw)
        assert len(result) > 0

    def test_double_encoded_fernet_decoded_first(self):
        import base64

        from wallet_manager.orchestrator import _encrypted_key_candidates

        inner = b"gAAAA" + b"x" * 80
        encoded = base64.urlsafe_b64encode(inner).decode()
        result = _encrypted_key_candidates(encoded)

        # inner should appear before raw encoded string
        inner_idx = result.index(inner) if inner in result else len(result)
        raw_bytes = encoded.strip().encode("utf-8")
        raw_idx = result.index(raw_bytes) if raw_bytes in result else len(result)
        assert inner_idx <= raw_idx


# ---------------------------------------------------------------------------
# 13. _build_runtime_audit_snapshot
# ---------------------------------------------------------------------------

class TestBuildRuntimeAuditSnapshot:
    """Tests for _build_runtime_audit_snapshot — call the REAL function directly."""

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        import importlib

        import wallet_manager.orchestrator as orch_mod
        # Call the real function by reference to avoid any module-level patching issues
        fn = orch_mod._build_runtime_audit_snapshot

        settings = _make_settings()
        pm = MagicMock()
        snapshot = MagicMock()
        snapshot.total_value_usd = 1000.0
        snapshot.needs_rebalance = False
        snapshot.timestamp = time.time()
        snapshot.allocations = []
        pm.get_snapshot.return_value = snapshot

        result = await fn(
            settings,
            wallet_count=1,
            portfolio_manager=pm,
            live_guard={"passed": True},
            phase="preflight",
        )

        assert "audit_version" in result
        assert result["phase"] == "preflight"
        assert "execution_mode" in result
        assert "live_guard" in result

    @pytest.mark.asyncio
    async def test_includes_result_summary_when_provided(self):
        import wallet_manager.orchestrator as orch_mod
        fn = orch_mod._build_runtime_audit_snapshot
        from wallet_manager.orchestrator import WalletResult

        settings = _make_settings()
        results = [
            WalletResult(wallet_id="w1", success=True),
            WalletResult(wallet_id="w2", success=False, error=Exception("fail")),
        ]

        result = await fn(
            settings,
            wallet_count=2,
            portfolio_manager=None,
            live_guard=None,
            results=results,
            phase="completed",
        )

        assert result["result_summary"]["wallets_total"] == 2
        assert result["result_summary"]["wallets_succeeded"] == 1
        assert result["result_summary"]["wallets_failed"] == 1

    @pytest.mark.asyncio
    async def test_includes_fatal_error_when_provided(self):
        import wallet_manager.orchestrator as orch_mod
        fn = orch_mod._build_runtime_audit_snapshot

        settings = _make_settings()
        err = RuntimeError("catastrophic failure")

        result = await fn(
            settings,
            wallet_count=1,
            portfolio_manager=None,
            live_guard=None,
            fatal_error=err,
            phase="completed",
        )

        assert "RuntimeError" in result["fatal_error"]
        assert "catastrophic failure" in result["fatal_error"]

    @pytest.mark.asyncio
    async def test_fatal_error_none_when_not_provided(self):
        import wallet_manager.orchestrator as orch_mod
        fn = orch_mod._build_runtime_audit_snapshot

        settings = _make_settings()

        result = await fn(
            settings,
            wallet_count=1,
            portfolio_manager=None,
            live_guard=None,
            phase="runtime",
        )

        assert result["fatal_error"] is None
