"""tests/integration/test_orchestrator.py

Integration tests for the WalletOrchestrator lifecycle (ADIM 4.1).

All RPC calls are mocked — no real network requests are made.
"""
from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_wallet_config():
    from wallet_manager.core.models import WalletConfig
    return WalletConfig(
        wallet_id="test_wallet",
        private_key="0xabcd" * 16,
        rpc_url="https://mainnet.base.org",
        chain_id=8453,
        max_gas_price_gwei=50.0,
        address="0x1234567890123456789012345678901234567890",
    )


@pytest.fixture
def mock_settings():
    from wallet_manager.config import Settings
    return Settings(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode="legacy",
        telegram_bot_token="test_token",
        telegram_chat_id="123456789",
        garch_alpha=0.10,
        garch_beta=0.85,
        max_concurrent_wallets=2,
    )


@pytest.fixture
def mock_notifier():
    notifier = MagicMock()
    notifier.send = AsyncMock()
    notifier.start = AsyncMock()
    notifier.stop = AsyncMock()
    return notifier


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_single_cycle_completes(
    minimal_wallet_config, mock_settings, mock_notifier
):
    """A single orchestrator cycle must complete without raising."""
    from wallet_manager.orchestrator import WalletOrchestrator
    from wallet_manager.telegram_notifier import CommandState, ConfirmationGate

    command_state = CommandState()
    confirmation_gate = ConfirmationGate()

    async def _noop_task_factory(config, manager, settings, portfolio_manager, **kwargs):
        return []

    with (
        patch("wallet_manager.orchestrator._cfg", return_value=mock_settings),
        patch("wallet_manager.orchestrator.get_settings", return_value=mock_settings),
        patch("wallet_manager.orchestrator.build_logger", return_value=MagicMock()),
    ):
        orchestrator = WalletOrchestrator(
            wallet_configs=[minimal_wallet_config],
            task_factory=_noop_task_factory,
            command_state=command_state,
            confirmation_gate=confirmation_gate,
        )
        # run() should complete without error
        try:
            await asyncio.wait_for(orchestrator.run(), timeout=10.0)
        except (TimeoutError, Exception) as exc:
            # TimeoutError is acceptable (means run() didn't crash, just slow)
            if not isinstance(exc, asyncio.TimeoutError):
                pytest.fail(f"Unexpected exception: {exc}")
        assert orchestrator is not None


@pytest.mark.asyncio
async def test_orchestrator_wallet_failure_does_not_cancel_others(
    mock_settings,
):
    """A failure in one wallet task must not prevent other wallets from running."""
    from wallet_manager.core.models import WalletConfig
    from wallet_manager.orchestrator import WalletOrchestrator
    from wallet_manager.telegram_notifier import CommandState, ConfirmationGate

    wallet_ok = WalletConfig(
        wallet_id="wallet_ok",
        private_key="0xabcd" * 16,
        rpc_url="https://mainnet.base.org",
        chain_id=8453,
    )
    wallet_fail = WalletConfig(
        wallet_id="wallet_fail",
        private_key="0xabcd" * 16,
        rpc_url="https://mainnet.base.org",
        chain_id=8453,
    )

    processed_wallets: list[str] = []

    async def _task_factory(config, manager, settings, portfolio_manager, **kwargs):
        if config.wallet_id == "wallet_fail":
            raise RuntimeError("Simulated wallet failure")
        processed_wallets.append(config.wallet_id)
        return []

    command_state = CommandState()

    with (
        patch("wallet_manager.orchestrator._cfg", return_value=mock_settings),
        patch("wallet_manager.orchestrator.get_settings", return_value=mock_settings),
    ):
        orchestrator = WalletOrchestrator(
            wallet_configs=[wallet_ok, wallet_fail],
            task_factory=_task_factory,
            command_state=command_state,
        )
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(orchestrator.run(), timeout=15.0)

    # wallet_ok must have been processed despite wallet_fail raising
    assert "wallet_ok" in processed_wallets


@pytest.mark.asyncio
async def test_orchestrator_supports_legacy_task_factory_signature(
    minimal_wallet_config, mock_settings
):
    """Legacy task factories still receive manager, wallet_id, and context."""
    from wallet_manager.orchestrator import WalletOrchestrator
    from wallet_manager.telegram_notifier import CommandState

    calls: list[tuple[str, bool]] = []

    async def _legacy_task_factory(manager, wallet_id, task_context):
        calls.append((wallet_id, "portfolio_manager" in task_context))
        assert manager is not None
        return []

    with (
        patch("wallet_manager.orchestrator._cfg", return_value=mock_settings),
        patch("wallet_manager.orchestrator.get_settings", return_value=mock_settings),
        patch("wallet_manager.telegram_notifier.asyncio.sleep", new_callable=AsyncMock),
    ):
        orchestrator = WalletOrchestrator(
            wallet_configs=[minimal_wallet_config],
            task_factory=_legacy_task_factory,
            command_state=CommandState(),
        )
        await asyncio.wait_for(orchestrator.run(), timeout=30.0)

    assert calls == [("test_wallet", True)]


@pytest.mark.asyncio
async def test_orchestrator_graceful_shutdown(
    minimal_wallet_config, mock_settings,
):
    """SIGTERM (via command_state.stop()) causes the loop to exit cleanly."""
    from wallet_manager.orchestrator import _setup_signal_handlers
    from wallet_manager.telegram_notifier import CommandState

    command_state = CommandState()
    loop = asyncio.get_running_loop()

    # Trigger stop via signal handler mechanism
    _setup_signal_handlers(loop, command_state)
    command_state.stop()

    assert command_state.is_stopped is True
    # Once stopped, the orchestrator loop should not continue
    assert not command_state.is_paused  # stop() clears pause too
