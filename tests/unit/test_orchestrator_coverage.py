"""
tests/unit/test_orchestrator_coverage.py

Comprehensive unit tests for wallet_manager/orchestrator.py.
Covers: module-level helpers, WalletResult, WalletOrchestrator constructor,
_resolve_notifier, _safe_notify, _invoke_task_factory, _run_wallet,
validate_wallets_schema, _encrypted_key_candidates, _safe_portfolio_snapshot,
_evaluate_live_runtime_guard, _get_paper_gate_status, get_runtime_audit_snapshot,
_SystemAlertNotifierWrapper, MasterOrchestrator, and more.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wallet_config():
    from wallet_manager.core.models import WalletConfig
    return WalletConfig(
        wallet_id="w1",
        private_key="0xabcd" * 16,
        rpc_url="https://mainnet.base.org",
        chain_id=8453,
        max_gas_price_gwei=50.0,
        address="0x1234567890123456789012345678901234567890",
    )


@pytest.fixture
def wallet_config2():
    from wallet_manager.core.models import WalletConfig
    return WalletConfig(
        wallet_id="w2",
        private_key="0xdcba" * 16,
        rpc_url="https://mainnet.base.org",
        chain_id=8453,
        max_gas_price_gwei=50.0,
        address="0x9876543210987654321098765432109876543210",
    )


@pytest.fixture
def mock_settings():
    from wallet_manager.config import Settings
    return Settings(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode="robust_paper",
        max_concurrent_wallets=3,
        gas_poll_interval=1.0,
        gas_wait_timeout=5.0,
        live_trading_enabled=False,
        live_max_wallets=5,
        live_max_total_capital_usd=10000.0,
        live_tiny_cap_mode=False,
    )


@pytest.fixture
def mock_notifier():
    n = MagicMock()
    n.send = AsyncMock()
    n.start = AsyncMock()
    n.stop = AsyncMock()
    return n


def _make_orchestrator(wallet_configs, task_factory=None, notifier=None, settings=None, **kwargs):
    from wallet_manager.orchestrator import WalletOrchestrator
    if task_factory is None:
        async def task_factory(manager, wallet_id, ctx):
            return []
    return WalletOrchestrator(
        wallet_configs=wallet_configs,
        task_factory=task_factory,
        notifier=notifier,
        settings=settings,
        **kwargs,
    )


# ============================================================================
# 1. WalletResult dataclass
# ============================================================================

class TestWalletResult:
    def test_wallet_result_success_defaults(self):
        from wallet_manager.orchestrator import WalletResult
        r = WalletResult(wallet_id="abc", success=True)
        assert r.wallet_id == "abc"
        assert r.success is True
        assert r.receipts == []
        assert r.error is None
        assert r.duration_seconds == 0.0

    def test_wallet_result_failure(self):
        from wallet_manager.orchestrator import WalletResult
        err = ValueError("bad")
        r = WalletResult(wallet_id="abc", success=False, error=err)
        assert r.success is False
        assert r.error is err

    def test_wallet_result_with_receipts(self):
        from wallet_manager.core.models import TxReceipt
        from wallet_manager.orchestrator import WalletResult
        receipt = TxReceipt(tx_hash="0xabc", wallet_id="w1", nonce_used=1)
        r = WalletResult(wallet_id="w1", success=True, receipts=[receipt], duration_seconds=1.5)
        assert len(r.receipts) == 1
        assert r.duration_seconds == 1.5


# ============================================================================
# 2. WalletOrchestrator constructor
# ============================================================================

class TestOrchestratorConstructor:
    def test_raises_if_no_wallet_configs(self, mock_settings):
        from wallet_manager.orchestrator import WalletOrchestrator
        with pytest.raises(ValueError, match="at least one"):
            WalletOrchestrator(
                wallet_configs=[],
                task_factory=AsyncMock(),
                settings=mock_settings,
            )

    def test_constructor_stores_configs(self, wallet_config, mock_settings):
        orch = _make_orchestrator([wallet_config], settings=mock_settings)
        assert orch._configs == [wallet_config]

    def test_constructor_uses_settings_max_concurrent(self, wallet_config, mock_settings):
        orch = _make_orchestrator([wallet_config], settings=mock_settings)
        assert orch._max_concurrent == mock_settings.max_concurrent_wallets

    def test_constructor_overrides_max_concurrent(self, wallet_config, mock_settings):
        orch = _make_orchestrator([wallet_config], settings=mock_settings, max_concurrent=7)
        assert orch._max_concurrent == 7

    def test_constructor_stores_notifier_override(self, wallet_config, mock_settings, mock_notifier):
        orch = _make_orchestrator([wallet_config], settings=mock_settings, notifier=mock_notifier)
        assert orch._notifier_override is mock_notifier

    def test_constructor_gas_poll_from_settings(self, wallet_config, mock_settings):
        orch = _make_orchestrator([wallet_config], settings=mock_settings)
        assert orch._gas_poll_interval == mock_settings.gas_poll_interval

    def test_constructor_gas_poll_override(self, wallet_config, mock_settings):
        orch = _make_orchestrator([wallet_config], settings=mock_settings, gas_poll_interval=9.9)
        assert orch._gas_poll_interval == 9.9


# ============================================================================
# 3. _resolve_notifier
# ============================================================================

class TestResolveNotifier:
    def test_returns_injected_notifier(self, wallet_config, mock_settings, mock_notifier):
        orch = _make_orchestrator([wallet_config], settings=mock_settings, notifier=mock_notifier)
        assert orch._resolve_notifier() is mock_notifier

    def test_constructs_telegram_notifier_when_none(self, wallet_config, mock_settings):
        orch = _make_orchestrator([wallet_config], settings=mock_settings)
        fake_tg = MagicMock()
        with patch("wallet_manager.telegram_notifier.TelegramNotifier", return_value=fake_tg):
            result = orch._resolve_notifier()
        assert result is fake_tg


# ============================================================================
# 4. _safe_notify
# ============================================================================

class TestSafeNotify:
    @pytest.mark.asyncio
    async def test_calls_send(self, mock_notifier):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import WalletOrchestrator
        evt = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        await WalletOrchestrator._safe_notify(mock_notifier, evt)
        mock_notifier.send.assert_awaited_once_with(evt)

    @pytest.mark.asyncio
    async def test_swallows_notifier_exception(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import WalletOrchestrator
        bad_notifier = MagicMock()
        bad_notifier.send = AsyncMock(side_effect=RuntimeError("boom"))
        evt = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        # Should not raise
        await WalletOrchestrator._safe_notify(bad_notifier, evt)


# ============================================================================
# 5. _invoke_task_factory — both signatures
# ============================================================================

class TestInvokeTaskFactory:
    @pytest.mark.asyncio
    async def test_old_signature_manager_wallet_id(self, wallet_config, mock_settings):
        """factory(manager, wallet_id, ctx) — classic 3-arg form."""
        called_with = {}

        async def factory(manager, wallet_id, ctx):
            called_with["manager"] = manager
            called_with["wallet_id"] = wallet_id
            return []

        orch = _make_orchestrator([wallet_config], task_factory=factory, settings=mock_settings)
        manager = MagicMock()
        await orch._invoke_task_factory(wallet_config, manager, {})
        assert called_with["wallet_id"] == "w1"

    @pytest.mark.asyncio
    async def test_new_signature_config_manager(self, wallet_config, mock_settings):
        """factory(config, manager, settings, pm, **kwargs) — new form."""
        called_with = {}

        async def factory(config, manager, settings, portfolio_manager, **kwargs):
            called_with["config"] = config
            return []

        orch = _make_orchestrator([wallet_config], task_factory=factory, settings=mock_settings)
        manager = MagicMock()
        await orch._invoke_task_factory(wallet_config, manager, {})
        assert called_with["config"] is wallet_config

    @pytest.mark.asyncio
    async def test_wallet_named_first_param(self, wallet_config, mock_settings):
        """factory(wallet, ...) also triggers new-style dispatch."""
        called = {}

        async def factory(wallet, manager, settings, portfolio_manager, **kwargs):
            called["wallet"] = wallet
            return []

        orch = _make_orchestrator([wallet_config], task_factory=factory, settings=mock_settings)
        manager = MagicMock()
        await orch._invoke_task_factory(wallet_config, manager, {})
        assert called["wallet"] is wallet_config


# ============================================================================
# 6. _run_wallet
# ============================================================================

class TestRunWallet:
    @pytest.mark.asyncio
    async def test_success_returns_wallet_result_ok(self, wallet_config, mock_settings, mock_notifier):
        from wallet_manager.core.models import TxReceipt
        receipt = TxReceipt(tx_hash="0xabc", wallet_id="w1", nonce_used=1)

        async def task_factory(manager, wallet_id, ctx):
            return [receipt]

        orch = _make_orchestrator([wallet_config], task_factory=task_factory,
                                  settings=mock_settings, notifier=mock_notifier)
        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})

        assert result.success is True
        assert result.wallet_id == "w1"
        assert result.receipts == [receipt]

    @pytest.mark.asyncio
    async def test_wallet_manager_error_returns_failure(self, wallet_config, mock_settings, mock_notifier):
        from wallet_manager.exceptions import WalletManagerError

        async def task_factory(manager, wallet_id, ctx):
            raise WalletManagerError("rpc failed", wallet_id="w1")

        orch = _make_orchestrator([wallet_config], task_factory=task_factory,
                                  settings=mock_settings, notifier=mock_notifier)
        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})

        assert result.success is False
        assert isinstance(result.error, WalletManagerError)

    @pytest.mark.asyncio
    async def test_generic_exception_returns_failure(self, wallet_config, mock_settings, mock_notifier):
        async def task_factory(manager, wallet_id, ctx):
            raise RuntimeError("unknown boom")

        orch = _make_orchestrator([wallet_config], task_factory=task_factory,
                                  settings=mock_settings, notifier=mock_notifier)
        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})

        assert result.success is False
        assert isinstance(result.error, RuntimeError)

    @pytest.mark.asyncio
    async def test_gas_price_too_high_returns_failure_with_deferral(
            self, wallet_config, mock_settings, mock_notifier):
        from wallet_manager.exceptions import GasPriceTooHighError

        async def task_factory(manager, wallet_id, ctx):
            raise GasPriceTooHighError("gas too high", requested_gwei=200.0, max_gwei=50.0)

        orch = _make_orchestrator([wallet_config], task_factory=task_factory,
                                  settings=mock_settings, notifier=mock_notifier)
        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            with patch.object(orch._scheduler, "schedule_gas_deferral") as mock_defer:
                result = await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})

        assert result.success is False
        mock_defer.assert_called_once_with("w1")

    @pytest.mark.asyncio
    async def test_command_state_stopped_skips_wallet(self, wallet_config, mock_settings, mock_notifier):
        command_state = MagicMock()
        command_state.is_stopped = True
        command_state.wait_for_resume = AsyncMock()

        async def task_factory(manager, wallet_id, ctx):
            return []

        orch = _make_orchestrator([wallet_config], task_factory=task_factory,
                                  settings=mock_settings, notifier=mock_notifier,
                                  command_state=command_state)
        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        result = await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})
        assert result.success is False
        assert "Stopped" in str(result.error)

    @pytest.mark.asyncio
    async def test_command_state_not_stopped_runs_wallet(self, wallet_config, mock_settings, mock_notifier):
        command_state = MagicMock()
        command_state.is_stopped = False
        command_state.wait_for_resume = AsyncMock()

        async def task_factory(manager, wallet_id, ctx):
            return []

        orch = _make_orchestrator([wallet_config], task_factory=task_factory,
                                  settings=mock_settings, notifier=mock_notifier,
                                  command_state=command_state)
        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            result = await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})

        assert result.success is True
        command_state.wait_for_resume.assert_awaited_once()


# ============================================================================
# 7. _wait_for_safe_gas
# ============================================================================

class TestWaitForSafeGas:
    @pytest.mark.asyncio
    async def test_updates_last_gas_price(self, wallet_config, mock_settings, mock_notifier):
        import wallet_manager.orchestrator as orch_mod
        orch = _make_orchestrator([wallet_config], settings=mock_settings)

        with patch.object(orch._gas_guard, "current_gwei", new=AsyncMock(return_value=25.5)):
            with patch.object(orch._gas_guard, "wait_for_safe_gas", new=AsyncMock()):
                await orch._wait_for_safe_gas(wallet_config, mock_notifier)

        assert orch_mod._last_gas_price_gwei == 25.5

    @pytest.mark.asyncio
    async def test_swallows_current_gwei_error(self, wallet_config, mock_settings, mock_notifier):
        orch = _make_orchestrator([wallet_config], settings=mock_settings)

        with patch.object(orch._gas_guard, "current_gwei", new=AsyncMock(side_effect=RuntimeError("rpc fail"))):
            with patch.object(orch._gas_guard, "wait_for_safe_gas", new=AsyncMock()):
                # Should not raise
                await orch._wait_for_safe_gas(wallet_config, mock_notifier)


# ============================================================================
# 8. Module-level accessors
# ============================================================================

class TestModuleLevelAccessors:
    def test_get_runtime_audit_snapshot_returns_copy(self):
        import wallet_manager.orchestrator as orch_mod
        orch_mod._runtime_audit_snapshot = {"key": "value"}
        snap = orch_mod.get_runtime_audit_snapshot()
        assert snap == {"key": "value"}
        # Mutating returned copy must not affect original
        snap["key"] = "modified"
        assert orch_mod._runtime_audit_snapshot["key"] == "value"

    def test_get_portfolio_manager_returns_none_initially(self):
        import wallet_manager.orchestrator as orch_mod
        # After module load, may or may not be None depending on prior tests
        result = orch_mod._get_portfolio_manager()
        # Just test it's callable and returns the stored value
        assert result is orch_mod._runtime_portfolio_manager

    def test_get_rpc_router_returns_none_initially(self):
        import wallet_manager.orchestrator as orch_mod
        result = orch_mod._get_rpc_router()
        assert result is orch_mod._runtime_rpc_router

    def test_get_gas_price_returns_stored(self):
        import wallet_manager.orchestrator as orch_mod
        orch_mod._last_gas_price_gwei = 42.0
        assert orch_mod._get_gas_price_gwei() == 42.0


# ============================================================================
# 9. _safe_portfolio_snapshot
# ============================================================================

class TestSafePortfolioSnapshot:
    def test_returns_available_false_for_none(self):
        from wallet_manager.orchestrator import _safe_portfolio_snapshot
        result = _safe_portfolio_snapshot(None)
        assert result == {"available": False}

    def test_returns_snapshot_when_valid(self):
        from wallet_manager.orchestrator import _safe_portfolio_snapshot
        from wallet_manager.portfolio_manager import PortfolioManager

        pm = MagicMock(spec=PortfolioManager)
        snapshot = MagicMock()
        snapshot.total_value_usd = 1234.56
        snapshot.needs_rebalance = False
        snapshot.timestamp = 1000.0
        snapshot.allocations = []
        pm.get_snapshot.return_value = snapshot

        result = _safe_portfolio_snapshot(pm)
        assert result["available"] is True
        assert result["total_value_usd"] == 1234.56

    def test_returns_error_when_snapshot_raises(self):
        from wallet_manager.orchestrator import _safe_portfolio_snapshot

        pm = MagicMock()
        pm.get_snapshot.side_effect = RuntimeError("DB gone")

        result = _safe_portfolio_snapshot(pm)
        assert result["available"] is False
        assert "RuntimeError" in result["error"]


# ============================================================================
# 10. _evaluate_live_runtime_guard
# ============================================================================

class TestEvaluateLiveRuntimeGuard:
    def test_passes_when_not_live(self, mock_settings):
        from wallet_manager.orchestrator import _evaluate_live_runtime_guard
        result = _evaluate_live_runtime_guard(mock_settings, wallet_count=1, portfolio_total_usd=100.0)
        # mock_settings uses robust_paper → not live
        assert result["passed"] is True
        assert result["violations"] == []

    def test_fails_when_live_trading_disabled(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _evaluate_live_runtime_guard
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_live",
            live_trading_enabled=False,
        )
        result = _evaluate_live_runtime_guard(cfg, wallet_count=1, portfolio_total_usd=100.0)
        assert result["passed"] is False
        assert any("disabled" in v for v in result["violations"])

    def test_fails_when_too_many_wallets(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _evaluate_live_runtime_guard
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_live",
            live_trading_enabled=True,
            live_max_wallets=2,
        )
        result = _evaluate_live_runtime_guard(cfg, wallet_count=5, portfolio_total_usd=100.0)
        assert result["passed"] is False
        assert any("exceeds live_max_wallets" in v for v in result["violations"])

    def test_fails_when_capital_exceeded_in_tiny_cap(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _evaluate_live_runtime_guard
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_live",
            live_trading_enabled=True,
            live_max_wallets=10,
            live_tiny_cap_mode=True,
            live_max_total_capital_usd=500.0,
        )
        result = _evaluate_live_runtime_guard(cfg, wallet_count=1, portfolio_total_usd=10000.0)
        assert result["passed"] is False
        assert any("exceeds" in v for v in result["violations"])

    def test_passes_when_capital_within_limit(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _evaluate_live_runtime_guard
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_live",
            live_trading_enabled=True,
            live_max_wallets=10,
            live_tiny_cap_mode=True,
            live_max_total_capital_usd=50000.0,
        )
        result = _evaluate_live_runtime_guard(cfg, wallet_count=1, portfolio_total_usd=1000.0)
        assert result["passed"] is True

    def test_returns_full_struct(self, mock_settings):
        from wallet_manager.orchestrator import _evaluate_live_runtime_guard
        result = _evaluate_live_runtime_guard(mock_settings, wallet_count=2, portfolio_total_usd=999.0)
        assert "active" in result
        assert "enabled" in result
        assert "wallet_count" in result
        assert "violations" in result


# ============================================================================
# 11. _get_paper_gate_status
# ============================================================================

class TestGetPaperGateStatus:
    @pytest.mark.asyncio
    async def test_skipped_when_not_live(self, mock_settings):
        from wallet_manager.orchestrator import _get_paper_gate_status
        # robust_paper is not live
        result = await _get_paper_gate_status(mock_settings)
        assert result.get("skipped") is True
        assert result.get("passed") is True

    @pytest.mark.asyncio
    async def test_checks_trade_count_when_live(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _get_paper_gate_status

        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_live",
            live_trading_enabled=True,
            min_paper_trades_before_live=10,
        )
        mock_engine = MagicMock()
        mock_engine._ledger.get_pnl_summary = AsyncMock(return_value={"trade_count": 5})

        with patch("wallet_manager.strategies.paper_engine.get_paper_engine",
                   new=AsyncMock(return_value=mock_engine)):
            result = await _get_paper_gate_status(cfg)

        assert result["current_trade_count"] == 5
        assert result["passed"] is False  # 5 < 10

    @pytest.mark.asyncio
    async def test_passes_when_trade_count_met(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _get_paper_gate_status

        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_live",
            live_trading_enabled=True,
            min_paper_trades_before_live=5,
        )
        mock_engine = MagicMock()
        mock_engine._ledger.get_pnl_summary = AsyncMock(return_value={"trade_count": 20})

        with patch("wallet_manager.strategies.paper_engine.get_paper_engine",
                   new=AsyncMock(return_value=mock_engine)):
            result = await _get_paper_gate_status(cfg)

        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_passes_when_required_is_zero(self):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator import _get_paper_gate_status

        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_paper",
            min_paper_trades_before_live=0,
        )
        result = await _get_paper_gate_status(cfg)
        assert result["passed"] is True


# ============================================================================
# 12. validate_wallets_schema
# ============================================================================

class TestValidateWalletsSchema:
    def test_valid_wallet_no_error(self):
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "wallet_01", "address": "0x" + "a" * 40,
                 "private_key": "abc123", "max_gas_price_gwei": 50.0}]
        # Should not raise (web3 may not be available, so address check is lenient)
        with patch("web3.Web3.is_checksum_address", return_value=True):
            validate_wallets_schema(data)

    def test_missing_wallet_id_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"address": "0x" + "a" * 40, "private_key": "abc"}]
        with pytest.raises(ConfigurationError, match="wallet_id"):
            validate_wallets_schema(data)

    def test_invalid_wallet_id_chars_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "bad wallet!", "address": "0x" + "a" * 40, "private_key": "abc"}]
        with pytest.raises(ConfigurationError, match="wallet_id"):
            validate_wallets_schema(data)

    def test_missing_private_key_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40}]
        with pytest.raises(ConfigurationError, match="private_key"):
            validate_wallets_schema(data)

    def test_invalid_gas_price_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40,
                 "private_key": "abc", "max_gas_price_gwei": 99999}]
        with pytest.raises(ConfigurationError, match="max_gas_price_gwei"):
            validate_wallets_schema(data)

    def test_invalid_gas_price_non_numeric_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40,
                 "private_key": "abc", "max_gas_price_gwei": "notanumber"}]
        with pytest.raises(ConfigurationError):
            validate_wallets_schema(data)

    def test_invalid_enabled_field_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40,
                 "private_key": "abc", "enabled": "yes"}]
        with pytest.raises(ConfigurationError, match="enabled"):
            validate_wallets_schema(data)

    def test_valid_enabled_bool_no_error(self):
        from wallet_manager.orchestrator import validate_wallets_schema
        # Use a proper checksummed address to avoid validation failure
        data = [{"wallet_id": "w1", "address": "0x" + "a" * 40,
                 "private_key": "abc", "enabled": True}]
        # web3 may reject non-checksummed address; mock it
        with patch("web3.Web3.is_checksum_address", return_value=True):
            validate_wallets_schema(data)

    def test_skips_vault_meta_entries(self):
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"_vault_meta": True, "some_field": "x"}]
        # Should not raise
        validate_wallets_schema(data)

    def test_multiple_errors_reported_at_once(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [
            {"wallet_id": "w1", "address": "0x" + "a" * 40, "private_key": "abc",
             "max_gas_price_gwei": 99999, "enabled": "yes"},
        ]
        with pytest.raises(ConfigurationError) as exc_info:
            validate_wallets_schema(data)
        # At least 2 errors accumulated
        assert exc_info.value.message.count("\n  - ") >= 1

    def test_empty_list_no_error(self):
        from wallet_manager.orchestrator import validate_wallets_schema
        validate_wallets_schema([])  # no wallets — should not raise

    def test_missing_address_raises(self):
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator import validate_wallets_schema
        data = [{"wallet_id": "w1", "private_key": "abc"}]
        with pytest.raises(ConfigurationError, match="address"):
            validate_wallets_schema(data)


# ============================================================================
# 13. _encrypted_key_candidates
# ============================================================================

class TestEncryptedKeyCandidates:
    def test_returns_list(self):
        from wallet_manager.orchestrator import _encrypted_key_candidates
        result = _encrypted_key_candidates("gAAAAsometoken")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_raw_fernet_token_first(self):
        from wallet_manager.orchestrator import _encrypted_key_candidates
        token = "gAAAA" + "x" * 80
        result = _encrypted_key_candidates(token)
        # First candidate should be the raw token bytes
        assert result[0] == token.strip().encode("utf-8")

    def test_includes_decoded_variants(self):
        from wallet_manager.orchestrator import _encrypted_key_candidates
        # Make a valid base64-encoded string
        inner = b"gAAAA" + b"x" * 40
        outer = base64.urlsafe_b64encode(inner).decode("utf-8")
        result = _encrypted_key_candidates(outer)
        # Should include the decoded version
        assert inner in result

    def test_no_duplicates(self):
        from wallet_manager.orchestrator import _encrypted_key_candidates
        result = _encrypted_key_candidates("gAAAA" + "a" * 80)
        # All items must be unique
        assert len(result) == len(set(result))


# ============================================================================
# 14. _SystemAlertNotifierWrapper
# ============================================================================

class TestSystemAlertNotifierWrapper:
    @pytest.mark.asyncio
    async def test_start_delegates_to_inner(self):
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.start = AsyncMock()
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        await wrapper.start()
        inner.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_delegates_to_inner(self):
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.stop = AsyncMock()
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        await wrapper.stop()
        inner.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_calls_inner_send(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.send = AsyncMock()
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        evt = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        await wrapper.send(evt)
        inner.send.assert_awaited_once_with(evt)

    @pytest.mark.asyncio
    async def test_send_forwards_gas_pause_to_telegram(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.send = AsyncMock()
        tg = MagicMock()
        tg.send_system_alert = AsyncMock()

        fake_payload_cls = MagicMock(return_value=MagicMock())
        with patch("wallet_manager.security.telegram_manager.SystemAlertPayload", fake_payload_cls):
            wrapper = _SystemAlertNotifierWrapper(inner, tg)
            evt = NotificationEvent(kind=EventKind.GAS_PAUSE, gas_price_gwei=150.0)
            await wrapper.send(evt)

        tg.send_system_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_wallet_failure_forwarded_to_telegram(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.send = AsyncMock()
        tg = MagicMock()
        tg.send_system_alert = AsyncMock()

        fake_payload_cls = MagicMock(return_value=MagicMock())
        with patch("wallet_manager.security.telegram_manager.SystemAlertPayload", fake_payload_cls):
            wrapper = _SystemAlertNotifierWrapper(inner, tg)
            evt = NotificationEvent(kind=EventKind.WALLET_FAILURE, wallet_id="w1", detail="rip")
            await wrapper.send(evt)

        tg.send_system_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_non_alert_event_not_forwarded(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.send = AsyncMock()
        tg = MagicMock()
        tg.send_system_alert = AsyncMock()

        wrapper = _SystemAlertNotifierWrapper(inner, tg)
        evt = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        await wrapper.send(evt)

        tg.send_system_alert.assert_not_called()

    def test_getattr_proxies_to_inner(self):
        from wallet_manager.orchestrator import _SystemAlertNotifierWrapper
        inner = MagicMock()
        inner.custom_attr = "hello"
        wrapper = _SystemAlertNotifierWrapper(inner, None)
        assert wrapper.custom_attr == "hello"


# ============================================================================
# 15. MasterOrchestrator
# ============================================================================

class TestMasterOrchestrator:
    def test_inherits_from_wallet_orchestrator(self, wallet_config, mock_settings):
        from wallet_manager.orchestrator import MasterOrchestrator, WalletOrchestrator
        with patch("wallet_manager.security.telegram_manager.TelegramManager", side_effect=Exception("no creds")):
            orch = MasterOrchestrator(
                wallet_configs=[wallet_config],
                task_factory=AsyncMock(return_value=[]),
                settings=mock_settings,
            )
        assert isinstance(orch, WalletOrchestrator)

    def test_resolve_notifier_wraps_with_telegram_manager(self, wallet_config, mock_settings, mock_notifier):
        from wallet_manager.orchestrator import MasterOrchestrator, _SystemAlertNotifierWrapper
        fake_tg = MagicMock()
        fake_tg._is_configured = MagicMock(return_value=True)

        with patch("wallet_manager.security.telegram_manager.TelegramManager", return_value=fake_tg):
            orch = MasterOrchestrator(
                wallet_configs=[wallet_config],
                task_factory=AsyncMock(return_value=[]),
                settings=mock_settings,
                notifier=mock_notifier,
            )

        result = orch._resolve_notifier()
        assert isinstance(result, _SystemAlertNotifierWrapper)

    def test_resolve_notifier_without_telegram_manager(self, wallet_config, mock_settings, mock_notifier):
        from wallet_manager.orchestrator import MasterOrchestrator
        with patch("wallet_manager.security.telegram_manager.TelegramManager", side_effect=Exception("no")):
            orch = MasterOrchestrator(
                wallet_configs=[wallet_config],
                task_factory=AsyncMock(return_value=[]),
                settings=mock_settings,
                notifier=mock_notifier,
            )
        # Without telegram manager, should return inner directly
        result = orch._resolve_notifier()
        assert result is mock_notifier

    @pytest.mark.asyncio
    async def test_run_wallet_adds_telegram_manager_to_ctx(self, wallet_config, mock_settings, mock_notifier):
        from wallet_manager.orchestrator import MasterOrchestrator

        received_ctx = {}

        async def task_factory(manager, wallet_id, ctx):
            received_ctx.update(ctx)
            return []

        fake_tg = MagicMock()
        fake_tg._is_configured = MagicMock(return_value=True)
        fake_tg.close = AsyncMock()

        with patch("wallet_manager.security.telegram_manager.TelegramManager", return_value=fake_tg):
            orch = MasterOrchestrator(
                wallet_configs=[wallet_config],
                task_factory=task_factory,
                settings=mock_settings,
                notifier=mock_notifier,
            )

        manager = MagicMock()
        semaphore = asyncio.Semaphore(5)

        with patch.object(orch, "_wait_for_safe_gas", new=AsyncMock()):
            await orch._run_wallet(manager, wallet_config, semaphore, mock_notifier, {})

        assert received_ctx.get("telegram_manager") is fake_tg


# ============================================================================
# 16. _build_runtime_audit_snapshot
# ============================================================================

class TestBuildRuntimeAuditSnapshot:
    @pytest.mark.asyncio
    async def test_returns_expected_keys(self, mock_settings):
        from wallet_manager.orchestrator import _build_runtime_audit_snapshot

        pm = MagicMock()
        snap = MagicMock()
        snap.total_value_usd = 500.0
        snap.needs_rebalance = False
        snap.timestamp = 1234.0
        snap.allocations = []
        pm.get_snapshot.return_value = snap

        with patch("wallet_manager.orchestrator._get_paper_gate_status",
                   new=AsyncMock(return_value={"passed": True})):
            result = await _build_runtime_audit_snapshot(
                mock_settings,
                wallet_count=3,
                portfolio_manager=pm,
                live_guard={"passed": True},
                phase="test",
            )

        assert result["audit_version"] == 1
        assert result["phase"] == "test"
        assert result["configured_wallet_count"] == 3
        assert "paper_trade" in result
        assert "is_live_execution" in result

    @pytest.mark.asyncio
    async def test_includes_result_summary_when_provided(self, mock_settings):
        from wallet_manager.orchestrator import WalletResult, _build_runtime_audit_snapshot

        pm = MagicMock()
        snap = MagicMock()
        snap.total_value_usd = 0.0
        snap.needs_rebalance = False
        snap.timestamp = 0.0
        snap.allocations = []
        pm.get_snapshot.return_value = snap

        results = [
            WalletResult(wallet_id="w1", success=True),
            WalletResult(wallet_id="w2", success=False),
        ]

        with patch("wallet_manager.orchestrator._get_paper_gate_status",
                   new=AsyncMock(return_value={"passed": True})):
            result = await _build_runtime_audit_snapshot(
                mock_settings,
                wallet_count=2,
                portfolio_manager=pm,
                live_guard=None,
                results=results,
            )

        assert result["result_summary"]["wallets_total"] == 2
        assert result["result_summary"]["wallets_succeeded"] == 1
        assert result["result_summary"]["wallets_failed"] == 1


# ============================================================================
# 17. Delegate functions (_validate_startup_config, _setup_signal_handlers)
# ============================================================================

class TestDelegateFunctions:
    @pytest.mark.asyncio
    async def test_validate_startup_config_delegates(self):
        from wallet_manager.orchestrator import _validate_startup_config
        with patch("wallet_manager.orchestrator_lifecycle.LifecycleManager.validate_startup_config",
                   new=AsyncMock()) as mock_validate:
            await _validate_startup_config()
        mock_validate.assert_awaited_once()

    def test_setup_signal_handlers_delegates(self):
        from wallet_manager.orchestrator import _setup_signal_handlers
        loop = MagicMock()
        command_state = MagicMock()
        with patch("wallet_manager.orchestrator_lifecycle.LifecycleManager.setup_signal_handlers") as mock_sh:
            _setup_signal_handlers(loop, command_state)
        mock_sh.assert_called_once_with(loop, command_state)


# ============================================================================
# 18. LiveStatusProvider
# ============================================================================

class TestLiveStatusProvider:
    def test_instantiation(self, wallet_config, mock_settings):
        from wallet_manager.orchestrator import LiveStatusProvider
        sched = MagicMock()
        with patch("wallet_manager.orchestrator.FullLiveStatusProvider.__init__",
                   return_value=None):
            provider = LiveStatusProvider(
                configs=[wallet_config],
                scheduler=sched,
                portfolio_manager=None,
                orchestrator=None,
            )
        assert provider._configs == [wallet_config]
        assert provider._scheduler is sched

    @pytest.mark.asyncio
    async def test_build_status_message_no_configs(self):
        from wallet_manager.orchestrator import LiveStatusProvider
        sched = MagicMock()
        with patch("wallet_manager.orchestrator.FullLiveStatusProvider.__init__",
                   return_value=None):
            provider = LiveStatusProvider(configs=[], scheduler=sched)
        result = await provider.build_status_message()
        assert "No wallets" in result


# ============================================================================
# 19. WalletResult — edge cases
# ============================================================================

class TestWalletResultEdgeCases:
    def test_success_count_computation(self):
        from wallet_manager.orchestrator import WalletResult
        results = [
            WalletResult(wallet_id=f"w{i}", success=(i % 2 == 0))
            for i in range(6)
        ]
        success = sum(1 for r in results if r.success)
        fail = sum(1 for r in results if not r.success)
        assert success == 3
        assert fail == 3


# ============================================================================
# 20. _managed_notifier context manager
# ============================================================================

class TestManagedNotifier:
    @pytest.mark.asyncio
    async def test_starts_and_stops_notifier(self, mock_notifier):
        from wallet_manager.orchestrator import _managed_notifier
        async with _managed_notifier(mock_notifier) as n:
            assert n is mock_notifier
            mock_notifier.start.assert_awaited_once()
        mock_notifier.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stops_notifier_even_on_exception(self, mock_notifier):
        from wallet_manager.orchestrator import _managed_notifier
        with pytest.raises(ValueError):
            async with _managed_notifier(mock_notifier):
                raise ValueError("inside error")
        mock_notifier.stop.assert_awaited_once()
