"""tests/unit/test_telegram_commands.py — Telegram komut testleri."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTelegramCommands:
    def _make_notifier(self, mocker, chat_id="123456789"):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = chat_id
        mgr = TelegramManager(settings=cfg)
        mgr._send = AsyncMock(return_value=True)
        return mgr

    def test_wrong_chat_id_rejected(self, mocker):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = "ALLOWED_CHAT"
        mgr = TelegramManager(settings=cfg)
        update = MagicMock()
        update.message.chat_id = "EVIL_CHAT_ID"
        result = mgr.is_authorized(update)
        assert result is False

    def test_correct_chat_id_authorized(self, mocker):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = "123456789"
        mgr = TelegramManager(settings=cfg)
        update = MagicMock()
        update.message.chat_id = "123456789"
        result = mgr.is_authorized(update)
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess_commands(self, mocker):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = "123456789"
        mgr = TelegramManager(settings=cfg)
        for _ in range(3):
            mgr._record_confirm_attempt("123456789")
        assert mgr._is_confirm_rate_limited("123456789") is True


# ---------------------------------------------------------------------------
# New tests for CommandCenter commands (ADIM 4.2)
# ---------------------------------------------------------------------------


@pytest.fixture
def command_state():
    from wallet_manager.telegram_notifier import CommandState
    return CommandState()


@pytest.fixture
def credentials():
    from wallet_manager.telegram_notifier import _TelegramCredentials
    return _TelegramCredentials(_bot_token="test_token", _chat_id="123456789")


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.send_message = AsyncMock()
    return client


@pytest.fixture
def command_center(credentials, command_state, mock_client):
    from wallet_manager.telegram_notifier import CommandCenter
    return CommandCenter(
        credentials=credentials,
        command_state=command_state,
        client=mock_client,
    )


@pytest.mark.asyncio
async def test_setmode_command_validates_mode(command_center, mock_client):
    """_handle_setmode must reject unknown modes."""
    await command_center._handle_setmode("123456789", "invalid_mode_xyz")
    call_args = mock_client.send_message.call_args[0][0]
    assert "Unknown mode" in call_args or "invalid_mode" in call_args.lower()


@pytest.mark.asyncio
async def test_setmode_ml_quant_requires_confirm(command_center, mock_client):
    """Setting ml_quant mode without 'confirm' must ask for confirmation."""
    await command_center._handle_setmode("123456789", "ml_quant")
    call_args = mock_client.send_message.call_args[0][0]
    assert "confirm" in call_args.lower() or "production" in call_args.lower()


@pytest.mark.asyncio
async def test_setmode_ml_quant_with_confirm_succeeds(command_center, command_state, mock_client):
    """Setting ml_quant with 'confirm' must request mode change."""
    await command_center._handle_setmode("123456789", "ml_quant confirm")
    assert command_state.pop_mode_change() == "ml_quant"


@pytest.mark.asyncio
async def test_setmode_legacy_no_confirm_needed(command_center, command_state, mock_client):
    """Setting legacy mode must NOT require confirm."""
    await command_center._handle_setmode("123456789", "legacy")
    assert command_state.pop_mode_change() == "legacy"


@pytest.mark.asyncio
async def test_wallets_command_masks_addresses(command_center, mock_client):
    """_handle_wallets must mask wallet addresses."""
    mock_wallet = MagicMock()
    mock_wallet.wallet_id = "wallet_01"
    mock_wallet.address = "0xAbCd1234567890AbCd1234567890AbCd12345678"
    mock_wallet.max_gas_price_gwei = 50.0

    with patch("wallet_manager.orchestrator._get_wallet_configs", return_value=[mock_wallet]), \
         patch("wallet_manager.orchestrator._get_circuit_breakers", return_value={}):
        await command_center._handle_wallets("123456789")

    call_args = mock_client.send_message.call_args[0][0]
    # Full address must NOT appear
    assert "0xAbCd1234567890AbCd1234567890AbCd12345678" not in call_args


@pytest.mark.asyncio
async def test_allocation_command_shows_drift(command_center, mock_client):
    """_handle_allocation must display allocation info."""
    from wallet_manager.portfolio_manager import PortfolioManager, StrategyType
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.YIELD, 500.0)
    pm.update_pillar_value(StrategyType.ALGO, 300.0)
    pm.update_pillar_value(StrategyType.SHORT, 100.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 100.0)

    with patch("wallet_manager.orchestrator._get_portfolio_manager", return_value=pm):
        await command_center._handle_allocation("123456789")

    call_args = mock_client.send_message.call_args[0][0]
    assert "YIELD" in call_args or "allocation" in call_args.lower()


@pytest.mark.asyncio
async def test_reset_circuit_requires_confirm(command_center, mock_client):
    """_handle_reset_circuit without 'confirm' must ask for confirmation."""
    await command_center._handle_reset_circuit("123456789", "wallet_01")
    call_args = mock_client.send_message.call_args[0][0]
    assert "confirm" in call_args.lower()


@pytest.mark.asyncio
async def test_reset_circuit_with_confirm_calls_force_close(command_center, mock_client):
    """_handle_reset_circuit with 'confirm' must call force_close on the CB."""
    mock_cb = MagicMock()
    mock_cb.force_close = MagicMock()
    mock_cb.state = MagicMock()
    mock_cb.state.name = "CLOSED"

    with patch("wallet_manager.orchestrator._get_circuit_breakers", return_value={"wallet_01": mock_cb}):
        await command_center._handle_reset_circuit("123456789", "wallet_01 confirm")

    mock_cb.force_close.assert_called_once()


@pytest.mark.asyncio
async def test_rate_limiter_blocks_flood(command_center):
    """After 10 commands in 60s, the rate limiter must return False."""
    from wallet_manager.security.telegram_commands import _RATE_LIMIT_MAX
    chat_id = "flood_test_unique_42"

    for _ in range(_RATE_LIMIT_MAX):
        result = command_center._check_rate_limit(chat_id, "/status")
        assert result is True

    result = command_center._check_rate_limit(chat_id, "/status")
    assert result is False
    await asyncio.sleep(0)


def test_rate_limiter_allows_after_window_expires(command_center):
    """Commands after the rate limit window expires should be allowed again."""
    import time as _time

    from wallet_manager.security.telegram_commands import _RATE_LIMIT_MAX, _RATE_LIMIT_WINDOW
    chat_id = "window_test_unique_42"

    for _ in range(_RATE_LIMIT_MAX):
        command_center._check_rate_limit(chat_id, "/status")

    old_time = _time.monotonic() - _RATE_LIMIT_WINDOW - 1
    command_center._cmd_history[chat_id] = [old_time] * _RATE_LIMIT_MAX

    result = command_center._check_rate_limit(chat_id, "/status")
    assert result is True


# ---------------------------------------------------------------------------
# Pure function tests: _mask_address and _split_message
# ---------------------------------------------------------------------------


class TestMaskAddress:
    def test_masks_full_eth_address(self):
        from wallet_manager.security.telegram_commands import _mask_address
        addr = "0xAbCd1234567890AbCd1234567890AbCd12345678"
        result = _mask_address(addr)
        assert "****" in result
        assert result.startswith("0xAbCd")
        assert result.endswith("5678")
        assert "1234567890AbCd1234567890AbCd1234" not in result

    def test_no_address_unchanged(self):
        from wallet_manager.security.telegram_commands import _mask_address
        text = "hello world"
        assert _mask_address(text) == "hello world"

    def test_short_hex_not_masked(self):
        from wallet_manager.security.telegram_commands import _mask_address
        text = "0x1234"
        result = _mask_address(text)
        # Too short — should not be replaced
        assert "****" not in result

    def test_multiple_addresses_masked(self):
        from wallet_manager.security.telegram_commands import _mask_address
        a1 = "0xAbCd1234567890AbCd1234567890AbCd12345678"
        a2 = "0xDeAd1234567890DeAd1234567890DeAd12345678"
        text = f"from {a1} to {a2}"
        result = _mask_address(text)
        assert a1 not in result
        assert a2 not in result
        assert result.count("****") == 2

    def test_private_key_like_hex_masked(self):
        from wallet_manager.security.telegram_commands import _mask_address
        key = "0xaAbBcCdDeEfF0011223344556677889900aAbBcCdDeEfF0011223344556677889900"
        result = _mask_address(key)
        assert "****" in result


class TestSplitMessage:
    def test_short_message_not_split(self):
        from wallet_manager.security.telegram_commands import _split_message
        text = "hello world"
        parts = _split_message(text)
        assert parts == ["hello world"]

    def test_exactly_max_len_not_split(self):
        from wallet_manager.security.telegram_commands import _split_message
        text = "x" * 4096
        parts = _split_message(text, max_len=4096)
        assert len(parts) == 1

    def test_long_message_split(self):
        from wallet_manager.security.telegram_commands import _split_message
        text = "x" * 5000
        parts = _split_message(text, max_len=4096)
        assert len(parts) > 1
        for p in parts:
            assert len(p) <= 4096

    def test_split_prefers_section_separator(self):
        from wallet_manager.security.telegram_commands import _split_message
        # Build a string with a ━ line before max_len
        sep = "\n" + "\u2501" * 20
        chunk_a = "A" * 100 + sep
        chunk_b = "B" * 100
        text = chunk_a + chunk_b
        parts = _split_message(text, max_len=len(chunk_a) + 10)
        # First part should end before the B chunk
        assert "B" not in parts[0]

    def test_split_falls_back_to_double_newline(self):
        from wallet_manager.security.telegram_commands import _split_message
        chunk_a = "A" * 100 + "\n\n"
        chunk_b = "B" * 100
        text = chunk_a + chunk_b
        parts = _split_message(text, max_len=len(chunk_a) + 10)
        assert "B" not in parts[0]

    def test_split_falls_back_to_single_newline(self):
        from wallet_manager.security.telegram_commands import _split_message
        chunk_a = "A" * 100 + "\n"
        chunk_b = "B" * 100
        text = chunk_a + chunk_b
        parts = _split_message(text, max_len=len(chunk_a) + 10)
        assert "B" not in parts[0]

    def test_hard_split_when_no_newline(self):
        from wallet_manager.security.telegram_commands import _split_message
        text = "x" * 200
        parts = _split_message(text, max_len=100)
        assert len(parts) == 2

    def test_empty_remainder_not_appended(self):
        from wallet_manager.security.telegram_commands import _split_message
        text = "A" * 100 + "\n\n"
        parts = _split_message(text, max_len=50)
        # All parts must have non-whitespace content
        for p in parts:
            assert p.strip() != ""


# ---------------------------------------------------------------------------
# _MessageFormatter tests
# ---------------------------------------------------------------------------


class TestMessageFormatter:
    def _make_event(self, kind, **kwargs):
        from wallet_manager.core.models import NotificationEvent
        return NotificationEvent(kind=kind, **kwargs)

    def test_format_boot(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.SYSTEM_BOOT)
        result = _MessageFormatter.format(ev)
        assert "STARTED" in result or "Orchestrator" in result

    def test_format_shutdown_no_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.SYSTEM_SHUTDOWN)
        result = _MessageFormatter.format(ev)
        assert "COMPLETED" in result

    def test_format_shutdown_with_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.SYSTEM_SHUTDOWN, detail="clean exit")
        result = _MessageFormatter.format(ev)
        assert "clean exit" in result

    def test_format_gas_pause_with_gwei(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.GAS_PAUSE, gas_price_gwei=150.5)
        result = _MessageFormatter.format(ev)
        assert "150" in result

    def test_format_gas_pause_no_gwei(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.GAS_PAUSE)
        result = _MessageFormatter.format(ev)
        assert "unknown" in result

    def test_format_gas_resume(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.GAS_RESUME, gas_price_gwei=50.0)
        result = _MessageFormatter.format(ev)
        assert "NORMAL" in result or "Resumed" in result

    def test_format_gas_deferred(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.GAS_DEFERRED, gas_price_gwei=200.0)
        result = _MessageFormatter.format(ev)
        # Should mention gas or rescheduling
        assert "200" in result or "saat" in result or "iptal" in result

    def test_format_wallet_success_no_tx(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.WALLET_SUCCESS, wallet_id="wallet_01")
        result = _MessageFormatter.format(ev)
        assert "Success" in result

    def test_format_wallet_success_with_tx(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(
            EventKind.WALLET_SUCCESS,
            wallet_id="wallet_01",
            tx_hash="0xAbCd1234567890AbCd1234567890AbCd12345678",
        )
        result = _MessageFormatter.format(ev)
        assert "Success" in result

    def test_format_wallet_failure_with_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.WALLET_FAILURE, wallet_id="wallet_01", detail="timeout")
        result = _MessageFormatter.format(ev)
        assert "Failed" in result
        assert "timeout" in result

    def test_format_wallet_failure_no_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.WALLET_FAILURE, wallet_id="wallet_01")
        result = _MessageFormatter.format(ev)
        assert "Failed" in result

    def test_format_generic_event(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.SYSTEM_ALERT, detail="some alert")
        result = _MessageFormatter.format(ev)
        assert "SYSTEM_ALERT" in result or "some alert" in result

    def test_escape_md_preserves_code_spans(self):
        from wallet_manager.security.telegram_commands import _MessageFormatter
        text = "hello `world_test` end"
        result = _MessageFormatter._escape_md(text)
        # Inside code span underscores should NOT be escaped
        assert "`world_test`" in result

    def test_escape_md_outside_code_span(self):
        from wallet_manager.security.telegram_commands import _MessageFormatter
        text = "hello_world"
        result = _MessageFormatter._escape_md(text)
        assert r"\_" in result or "hello\\_world" in result or "\\_" in result

    def test_wallet_id_none_handled(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        ev = self._make_event(EventKind.WALLET_SUCCESS, wallet_id=None)
        result = _MessageFormatter.format(ev)
        assert "unknown" in result or "Success" in result


# ---------------------------------------------------------------------------
# CommandState tests
# ---------------------------------------------------------------------------


class TestCommandState:
    def test_initial_state_not_paused(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        assert state.is_paused is False

    def test_initial_state_not_stopped(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        assert state.is_stopped is False

    def test_pause_sets_paused(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.pause()
        assert state.is_paused is True

    def test_resume_clears_pause(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.pause()
        state.resume()
        assert state.is_paused is False

    def test_stop_sets_stopped(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.stop()
        assert state.is_stopped is True

    def test_stop_also_unpauses(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.pause()
        state.stop()
        # After stop, pause event should be set (unblocked) even if was paused
        assert state.is_stopped is True
        assert state.is_paused is False

    @pytest.mark.asyncio
    async def test_wait_for_resume_returns_immediately_when_not_paused(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        # Should return immediately since not paused
        await asyncio.wait_for(state.wait_for_resume(), timeout=1.0)

    def test_retrain_requested_initially_false(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        assert state.retrain_requested is False

    def test_request_retrain_sets_flag(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.request_retrain()
        assert state.retrain_requested is True

    def test_clear_retrain_clears_flag(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.request_retrain()
        state.clear_retrain()
        assert state.retrain_requested is False

    def test_pop_mode_change_initially_none(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        assert state.pop_mode_change() is None

    def test_request_mode_change_and_pop(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.request_mode_change("ml_quant")
        assert state.pop_mode_change() == "ml_quant"
        # Should be cleared after pop
        assert state.pop_mode_change() is None

    def test_circuit_breaker_initially_not_halted(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        assert state.is_circuit_breaker_halted is False

    def test_trigger_circuit_breaker_halts(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.trigger_circuit_breaker_halt(duration_seconds=86400.0)
        assert state.is_circuit_breaker_halted is True

    def test_resume_clears_circuit_breaker(self):
        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        state.trigger_circuit_breaker_halt(duration_seconds=86400.0)
        state.resume()
        assert state.is_circuit_breaker_halted is False

    def test_circuit_breaker_expires_naturally(self):
        import time

        from wallet_manager.security.telegram_commands import CommandState
        state = CommandState()
        # Set halt to expire in the past
        state._circuit_breaker_halt_until = time.time() - 1
        assert state.is_circuit_breaker_halted is False


# ---------------------------------------------------------------------------
# ConfirmationGate tests
# ---------------------------------------------------------------------------


class TestConfirmationGate:
    def test_initial_not_confirmed(self):
        from wallet_manager.security.telegram_commands import ConfirmationGate
        gate = ConfirmationGate()
        assert gate._confirmed is False

    def test_signal_evet_sets_confirmed(self):
        from wallet_manager.security.telegram_commands import ConfirmationGate
        gate = ConfirmationGate()
        gate.signal_evet()
        assert gate._confirmed is True

    def test_request_confirmation_resets(self):
        from wallet_manager.security.telegram_commands import ConfirmationGate
        gate = ConfirmationGate()
        gate.signal_evet()
        gate.request_confirmation()
        assert gate._confirmed is False

    @pytest.mark.asyncio
    async def test_wait_for_evet_returns_true_on_signal(self):
        from wallet_manager.security.telegram_commands import ConfirmationGate
        gate = ConfirmationGate()
        gate.signal_evet()
        result = await gate.wait_for_evet(timeout_seconds=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_evet_times_out(self):
        from wallet_manager.security.telegram_commands import ConfirmationGate
        gate = ConfirmationGate()
        result = await gate.wait_for_evet(timeout_seconds=0.05)
        assert result is False


# ---------------------------------------------------------------------------
# CommandCenter handler tests
# ---------------------------------------------------------------------------


@pytest.fixture
def live_provider():
    """A LiveStatusProvider backed by a mock orchestrator."""
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    orch = MagicMock()
    orch.portfolio_total_usd = 1000.0
    orch.portfolio_24h_change_usd = 50.0
    orch.portfolio_24h_change_pct = 5.0
    orch.wallet_address = "0xAbCd1234567890AbCd1234567890AbCd12345678"
    orch.eth_balance = 0.5
    orch.eth_price_usd = 3000.0
    orch.usdc_balance = 500.0
    orch.aave_balance = 200.0
    orch.gas_gwei = 30.0
    orch.execution_mode = "legacy"
    orch.current_regime = "BULL"
    orch.uptime_seconds = 3700.0
    orch.last_heartbeat_ago_minutes = 2.0
    orch.paper_sharpe = 1.5
    orch.real_sharpe = 1.2
    orch.last_tx_hash = "0xDeAd1234567890DeAd1234567890DeAd12345678"
    orch.last_tx_ago_minutes = 10.0
    orch.rebalance_needed = False
    orch.pillars = {
        "YIELD":   {"value": 400.0, "target": 400.0, "apy": 4.5},
        "ALGO":    {"value": 350.0, "target": 350.0, "open_positions": 2},
        "SHORT":   {"value": 100.0, "target": 100.0, "leverage": 1.5},
        "AIRDROP": {"value": 150.0, "target": 150.0, "next_swap_minutes": 30.0},
    }
    orch.pnl_summary = {"24h": 50.0, "7d": 200.0, "30d": 800.0}
    orch.last_trades = []
    orch.drawdown_pct = 1.5
    orch.var_99_usd = 100.0
    orch.cvar_99_usd = 120.0
    orch.kelly_fraction = 0.3
    orch.win_rate_50 = 0.6
    orch.sharpe_7d = 1.4
    orch.sortino_7d = 1.6
    orch.garch_sigma = 0.02
    orch.circuit_breaker_open = False
    orch.circuit_failures = 0
    orch.ir_zscore = 0.5
    orch.ir_ofi = 0.1
    orch.ir_ml = 0.2
    orch.signal_half_life_minutes = 15.0
    orch.signal_exec_seconds = 3.0
    return LiveStatusProvider(orchestrator=orch)


@pytest.fixture
def command_center_with_live(credentials, command_state, mock_client, live_provider):
    from wallet_manager.telegram_notifier import CommandCenter
    return CommandCenter(
        credentials=credentials,
        command_state=command_state,
        client=mock_client,
        status_provider=live_provider,
    )


@pytest.mark.asyncio
async def test_handle_pause_when_running(command_center, mock_client):
    """_handle_pause should pause state and confirm."""
    await command_center._handle_pause("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "PAUSED" in msg or "paused" in msg.lower()
    assert command_center._state.is_paused is True


@pytest.mark.asyncio
async def test_handle_pause_when_already_paused(command_center, mock_client):
    """_handle_pause when already paused should say so."""
    command_center._state.pause()
    await command_center._handle_pause("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "already" in msg.lower() or "paused" in msg.lower()


@pytest.mark.asyncio
async def test_handle_resume_when_paused(command_center, mock_client):
    """_handle_resume should clear pause and confirm."""
    command_center._state.pause()
    await command_center._handle_resume("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "RESUMED" in msg or "resumed" in msg.lower()
    assert command_center._state.is_paused is False


@pytest.mark.asyncio
async def test_handle_resume_when_already_running(command_center, mock_client):
    """_handle_resume when running should say already running."""
    await command_center._handle_resume("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "already" in msg.lower() or "running" in msg.lower()


@pytest.mark.asyncio
async def test_handle_resume_when_stopped(command_center, mock_client):
    """_handle_resume after stop should refuse."""
    command_center._state.stop()
    await command_center._handle_resume("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "stop" in msg.lower() or "Stop" in msg


@pytest.mark.asyncio
async def test_handle_stop_when_running(command_center, mock_client):
    """_handle_stop should signal stop and confirm."""
    await command_center._handle_stop("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "STOP" in msg or "stop" in msg.lower()
    assert command_center._state.is_stopped is True


@pytest.mark.asyncio
async def test_handle_stop_when_already_stopped(command_center, mock_client):
    """_handle_stop when already stopped should say so."""
    command_center._state.stop()
    await command_center._handle_stop("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "already" in msg.lower() or "Stop" in msg


@pytest.mark.asyncio
async def test_handle_help_returns_help_text(command_center, mock_client):
    """_handle_help should return a comprehensive help text."""
    await command_center._handle_help("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "/status" in msg
    assert "/pause" in msg
    assert "/stop" in msg


@pytest.mark.asyncio
async def test_handle_status_with_status_provider(command_center, mock_client):
    """_handle_status calls status_provider.build_status_message."""
    mock_provider = MagicMock()
    mock_provider.build_status_message = AsyncMock(return_value="Status OK")
    command_center._status_provider = mock_provider
    mock_client.send_message_with_keyboard = AsyncMock()
    await command_center._handle_status("123456789")
    mock_provider.build_status_message.assert_called_once()


@pytest.mark.asyncio
async def test_handle_status_provider_exception_sends_error(command_center, mock_client):
    """_handle_status should send an error message if provider raises."""
    mock_provider = MagicMock()
    mock_provider.build_status_message = AsyncMock(side_effect=RuntimeError("boom"))
    command_center._status_provider = mock_provider
    await command_center._handle_status("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "unavailable" in msg.lower() or "boom" in msg


@pytest.mark.asyncio
async def test_handle_pnl_with_live_provider(command_center_with_live, mock_client):
    """_handle_pnl should work with LiveStatusProvider."""
    await command_center_with_live._handle_pnl("123456789")
    assert mock_client.send_message.called
    msg = mock_client.send_message.call_args[0][0]
    assert len(msg) > 0


@pytest.mark.asyncio
async def test_handle_pnl_without_live_provider(command_center, mock_client):
    """_handle_pnl without LiveStatusProvider should say not available."""
    await command_center._handle_pnl("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "not available" in msg.lower() or "P" in msg


@pytest.mark.asyncio
async def test_handle_trades_with_live_provider(command_center_with_live, mock_client):
    """_handle_trades should work with LiveStatusProvider."""
    with patch("wallet_manager.config.get_settings") as mock_settings:
        mock_settings.return_value.paper_trade = False
        mock_settings.return_value.execution_mode = "legacy"
        await command_center_with_live._handle_trades("123456789")
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_handle_trades_without_live_provider(command_center, mock_client):
    """_handle_trades without LiveStatusProvider should say not available."""
    await command_center._handle_trades("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "not available" in msg.lower() or "history" in msg.lower()


@pytest.mark.asyncio
async def test_handle_regime_with_live_provider(command_center_with_live, mock_client):
    """_handle_regime should work with LiveStatusProvider."""
    await command_center_with_live._handle_regime("123456789")
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_handle_regime_without_live_provider(command_center, mock_client):
    """_handle_regime without LiveStatusProvider should say not available."""
    await command_center._handle_regime("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "not available" in msg.lower()


@pytest.mark.asyncio
async def test_handle_paper_with_live_provider(command_center_with_live, mock_client):
    """_handle_paper should work with LiveStatusProvider."""
    await command_center_with_live._handle_paper("123456789")
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_handle_paper_without_live_provider(command_center, mock_client):
    """_handle_paper without LiveStatusProvider should say not available."""
    await command_center._handle_paper("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "not available" in msg.lower()


@pytest.mark.asyncio
async def test_handle_risk_with_live_provider(command_center_with_live, mock_client):
    """_handle_risk should work with LiveStatusProvider."""
    await command_center_with_live._handle_risk("123456789")
    assert mock_client.send_message.called
    msg = mock_client.send_message.call_args[0][0]
    assert len(msg) > 0


@pytest.mark.asyncio
async def test_handle_risk_without_live_provider(command_center, mock_client):
    """_handle_risk without LiveStatusProvider should say not available."""
    await command_center._handle_risk("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "not available" in msg.lower()


@pytest.mark.asyncio
async def test_handle_retrain_first_call(command_center, mock_client):
    """First /retrain call should ask for confirmation."""
    await command_center._handle_retrain("123456789", confirmed=False)
    msg = mock_client.send_message.call_args[0][0]
    assert "retrain" in msg.lower() or "confirm" in msg.lower()
    assert command_center._pending_retrain is True


@pytest.mark.asyncio
async def test_handle_retrain_confirm_with_pending(command_center, mock_client):
    """Second /retrain confirm should trigger retrain."""
    command_center._pending_retrain = True
    await command_center._handle_retrain("123456789", confirmed=True)
    assert command_center._state.retrain_requested is True
    assert command_center._pending_retrain is False


@pytest.mark.asyncio
async def test_handle_retrain_confirm_without_pending(command_center, mock_client):
    """Confirm without prior request should warn."""
    await command_center._handle_retrain("123456789", confirmed=True)
    msg = mock_client.send_message.call_args[0][0]
    assert "pending" in msg.lower() or "No pending" in msg


@pytest.mark.asyncio
async def test_handle_mode_no_arg_shows_list(command_center, mock_client):
    """_handle_mode with no arg should show valid modes."""
    await command_center._handle_mode("123456789", "")
    msg = mock_client.send_message.call_args[0][0]
    assert "legacy" in msg or "valid" in msg.lower() or "mode" in msg.lower()


@pytest.mark.asyncio
async def test_handle_mode_invalid(command_center, mock_client):
    """_handle_mode with invalid mode should reject."""
    await command_center._handle_mode("123456789", "invalid_xyz")
    msg = mock_client.send_message.call_args[0][0]
    assert "Unknown" in msg or "unknown" in msg.lower()


@pytest.mark.asyncio
async def test_handle_mode_first_call_pending(command_center, mock_client):
    """First /mode legacy call should ask for re-confirmation."""
    await command_center._handle_mode("123456789", "legacy")
    msg = mock_client.send_message.call_args[0][0]
    assert "confirm" in msg.lower() or "Change" in msg
    assert command_center._pending_mode == "legacy"


@pytest.mark.asyncio
async def test_handle_mode_second_call_confirms(command_center, command_state, mock_client):
    """Second /mode legacy call should confirm mode change."""
    command_center._pending_mode = "legacy"
    await command_center._handle_mode("123456789", "legacy")
    assert command_state.pop_mode_change() == "legacy"


@pytest.mark.asyncio
async def test_dispatch_unauthorized_chat_ignored(command_center):
    """_dispatch should silently ignore messages from unauthorized chats."""
    update = {
        "message": {
            "chat": {"id": "EVIL_CHAT"},
            "text": "/stop",
        }
    }
    await command_center._dispatch(update)
    # command_state.is_stopped must remain False
    assert command_center._state.is_stopped is False


@pytest.mark.asyncio
async def test_dispatch_authorized_chat_processes(command_center, mock_client):
    """_dispatch should process messages from the authorized chat."""
    update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "/help",
        }
    }
    await command_center._dispatch(update)
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_dispatch_empty_message_ignored(command_center, mock_client):
    """_dispatch should ignore updates with no message."""
    update = {}
    await command_center._dispatch(update)
    assert not mock_client.send_message.called


@pytest.mark.asyncio
async def test_dispatch_callback_query_authorized(command_center, mock_client):
    """_dispatch_callback should handle cb_status from authorized chat."""
    mock_provider = MagicMock()
    mock_provider.build_status_message = AsyncMock(return_value="Status OK")
    command_center._status_provider = mock_provider
    mock_client.answer_callback_query = AsyncMock()
    mock_client.send_message_with_keyboard = AsyncMock()
    callback = {
        "id": "cb001",
        "message": {"chat": {"id": "123456789"}},
        "data": "cb_status",
    }
    await command_center._dispatch_callback(callback)
    mock_client.answer_callback_query.assert_called_once_with("cb001")


@pytest.mark.asyncio
async def test_dispatch_callback_unauthorized_acknowledged(command_center, mock_client):
    """_dispatch_callback should acknowledge but not process unauthorized chats."""
    mock_client.answer_callback_query = AsyncMock()
    callback = {
        "id": "cb002",
        "message": {"chat": {"id": "EVIL_CHAT"}},
        "data": "cb_status",
    }
    await command_center._dispatch_callback(callback)
    # Still acknowledges (removes spinner) but doesn't dispatch
    mock_client.answer_callback_query.assert_called_once_with("cb002")
    assert not mock_client.send_message.called


@pytest.mark.asyncio
async def test_dispatch_unknown_command(command_center, mock_client):
    """Unknown command should send help hint."""
    update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "/unknownxyz",
        }
    }
    await command_center._dispatch(update)
    msg = mock_client.send_message.call_args[0][0]
    assert "Unknown" in msg or "help" in msg.lower()


@pytest.mark.asyncio
async def test_dispatch_evet_with_gate(command_center, mock_client):
    """Sending EVET with a confirmation gate should signal it."""
    from wallet_manager.security.telegram_commands import ConfirmationGate
    gate = ConfirmationGate()
    command_center._confirmation_gate = gate
    update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "EVET",
        }
    }
    await command_center._dispatch(update)
    assert gate._confirmed is True


@pytest.mark.asyncio
async def test_dispatch_evet_without_gate(command_center, mock_client):
    """EVET without a confirmation gate should be treated as unknown command."""
    update = {
        "message": {
            "chat": {"id": "123456789"},
            "text": "EVET",
        }
    }
    # No gate set → falls to else unknown branch
    await command_center._dispatch(update)
    msg = mock_client.send_message.call_args[0][0]
    assert "Unknown" in msg or "help" in msg.lower()


@pytest.mark.asyncio
async def test_handle_gas_no_wallets(command_center, mock_client):
    """_handle_gas should report gas price."""
    with patch("wallet_manager.orchestrator._get_gas_price_gwei", return_value=45.5), \
         patch("wallet_manager.orchestrator._get_wallet_configs", return_value=[]):
        await command_center._handle_gas("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "45" in msg or "Gas" in msg


@pytest.mark.asyncio
async def test_handle_gas_with_wallet_suspended(command_center, mock_client):
    """_handle_gas shows suspended wallets when gas is over limit."""
    mock_wallet = MagicMock()
    mock_wallet.wallet_id = "wallet_01"
    mock_wallet.max_gas_price_gwei = 30.0
    with patch("wallet_manager.orchestrator._get_gas_price_gwei", return_value=50.0), \
         patch("wallet_manager.orchestrator._get_wallet_configs", return_value=[mock_wallet]):
        await command_center._handle_gas("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "suspended" in msg.lower() or "wallet_01" in msg


@pytest.mark.asyncio
async def test_handle_gas_with_wallet_active(command_center, mock_client):
    """_handle_gas shows active wallets when gas is under limit."""
    mock_wallet = MagicMock()
    mock_wallet.wallet_id = "wallet_01"
    mock_wallet.max_gas_price_gwei = 100.0
    with patch("wallet_manager.orchestrator._get_gas_price_gwei", return_value=20.0), \
         patch("wallet_manager.orchestrator._get_wallet_configs", return_value=[mock_wallet]):
        await command_center._handle_gas("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "active" in msg.lower() or "wallet_01" in msg


@pytest.mark.asyncio
async def test_handle_gas_exception(command_center, mock_client):
    """_handle_gas should handle import/execution errors gracefully."""
    with patch("wallet_manager.orchestrator._get_gas_price_gwei", side_effect=RuntimeError("rpc error")):
        await command_center._handle_gas("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "unavailable" in msg.lower()


@pytest.mark.asyncio
async def test_handle_report_invalid_period(command_center, mock_client):
    """_handle_report with invalid period should warn."""
    await command_center._handle_report("123456789", "quarterly")
    msg = mock_client.send_message.call_args[0][0]
    assert "Usage" in msg or "valid" in msg.lower() or "period" in msg.lower()


@pytest.mark.asyncio
async def test_handle_report_valid_period_exception(command_center, mock_client):
    """_handle_report with valid period should gracefully handle AnalyticsDB errors."""
    with patch("wallet_manager.config.get_settings") as mock_cfg, \
         patch("wallet_manager.resilience.analytics.AnalyticsDB") as mock_db_cls:
        mock_cfg.return_value.quant_analytics_db_path = "/tmp/test.db"
        mock_db_instance = MagicMock()
        mock_db_instance.get_daily_summary = AsyncMock(return_value={
            "trade_count": 5, "total_pnl_usd": 123.45, "avg_gas_gwei": 30.0
        })
        mock_db_cls.return_value = mock_db_instance
        await command_center._handle_report("123456789", "daily")
    msg = mock_client.send_message.call_args[0][0]
    assert "daily" in msg.lower() or "Report" in msg or "unavailable" in msg.lower()


@pytest.mark.asyncio
async def test_handle_reset_circuit_no_wallet(command_center, mock_client):
    """_handle_reset_circuit without wallet_id should show usage."""
    await command_center._handle_reset_circuit("123456789", "")
    msg = mock_client.send_message.call_args[0][0]
    assert "Usage" in msg or "usage" in msg.lower()


@pytest.mark.asyncio
async def test_handle_reset_circuit_wallet_not_found(command_center, mock_client):
    """_handle_reset_circuit with confirm but non-existent wallet should say not found."""
    with patch("wallet_manager.orchestrator._get_circuit_breakers", return_value={}):
        await command_center._handle_reset_circuit("123456789", "nonexistent confirm")
    msg = mock_client.send_message.call_args[0][0]
    assert "not found" in msg.lower() or "nonexistent" in msg


@pytest.mark.asyncio
async def test_handle_wallets_no_wallets(command_center, mock_client):
    """_handle_wallets with no wallets should say none configured."""
    with patch("wallet_manager.orchestrator._get_wallet_configs", return_value=[]), \
         patch("wallet_manager.orchestrator._get_circuit_breakers", return_value={}):
        await command_center._handle_wallets("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "No active" in msg or "no active" in msg.lower() or "wallet" in msg.lower()


@pytest.mark.asyncio
async def test_handle_wallets_exception(command_center, mock_client):
    """_handle_wallets should handle exceptions gracefully."""
    with patch("wallet_manager.orchestrator._get_wallet_configs", side_effect=RuntimeError("error")):
        await command_center._handle_wallets("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "unavailable" in msg.lower()


@pytest.mark.asyncio
async def test_handle_allocation_no_portfolio_manager(command_center, mock_client):
    """_handle_allocation with no portfolio manager should warn."""
    with patch("wallet_manager.orchestrator._get_portfolio_manager", return_value=None):
        await command_center._handle_allocation("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "not available" in msg.lower() or "portfolio" in msg.lower()


@pytest.mark.asyncio
async def test_handle_allocation_exception(command_center, mock_client):
    """_handle_allocation should handle exceptions gracefully."""
    with patch("wallet_manager.orchestrator._get_portfolio_manager", side_effect=RuntimeError("error")):
        await command_center._handle_allocation("123456789")
    msg = mock_client.send_message.call_args[0][0]
    assert "unavailable" in msg.lower()


@pytest.mark.asyncio
async def test_handle_rotate_vault_no_key(command_center, mock_client):
    """_handle_rotate_vault with no key should show usage."""
    await command_center._handle_rotate_vault("123456789", "")
    msg = mock_client.send_message.call_args[0][0]
    assert "Usage" in msg or "rotate_vault_key" in msg


@pytest.mark.asyncio
async def test_handle_rotate_vault_no_confirm(command_center, mock_client):
    """_handle_rotate_vault without confirm should ask for confirmation."""
    await command_center._handle_rotate_vault("123456789", "SomeKey123456")
    msg = mock_client.send_message.call_args[0][0]
    assert "confirm" in msg.lower() or "confirmation" in msg.lower()


@pytest.mark.asyncio
async def test_handle_rotate_vault_invalid_key(command_center, mock_client):
    """_handle_rotate_vault with confirm but invalid key should report failure."""
    await command_center._handle_rotate_vault("123456789", "InvalidKey confirm")
    msg = mock_client.send_message.call_args[0][0]
    assert "failed" in msg.lower() or "rotation" in msg.lower() or "invalid" in msg.lower()


@pytest.mark.asyncio
async def test_reply_retries_on_failure(command_center, mock_client):
    """_reply should retry up to 3 times on send failures."""
    mock_client.send_message = AsyncMock(side_effect=[
        Exception("network error"),
        Exception("network error"),
        None,  # Third attempt succeeds
    ])
    await command_center._reply("123456789", "test message")
    assert mock_client.send_message.call_count == 3


@pytest.mark.asyncio
async def test_reply_logs_after_3_failures(command_center, mock_client):
    """_reply should give up after 3 failures without raising."""
    mock_client.send_message = AsyncMock(side_effect=Exception("network error"))
    # Should NOT raise — just log warning
    await command_center._reply("123456789", "test message")
    assert mock_client.send_message.call_count == 3


@pytest.mark.asyncio
async def test_heavy_cmd_rate_limit(command_center):
    """_retrain (heavy command) should be rate limited to 1/hour."""
    from wallet_manager.security.telegram_commands import _HEAVY_CMD_MAX
    chat_id = "heavy_test_unique"
    # First should succeed
    for _ in range(_HEAVY_CMD_MAX):
        result = command_center._check_rate_limit(chat_id, "/retrain")
        assert result is True
    # Second within window should be blocked
    result = command_center._check_rate_limit(chat_id, "/retrain")
    assert result is False


@pytest.mark.asyncio
async def test_dispatch_callback_pnl(command_center_with_live, mock_client):
    """cb_pnl callback should trigger _handle_pnl."""
    mock_client.answer_callback_query = AsyncMock()
    callback = {
        "id": "cb003",
        "message": {"chat": {"id": "123456789"}},
        "data": "cb_pnl",
    }
    await command_center_with_live._dispatch_callback(callback)
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_dispatch_callback_risk(command_center_with_live, mock_client):
    """cb_risk callback should trigger _handle_risk."""
    mock_client.answer_callback_query = AsyncMock()
    callback = {
        "id": "cb004",
        "message": {"chat": {"id": "123456789"}},
        "data": "cb_risk",
    }
    await command_center_with_live._dispatch_callback(callback)
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_dispatch_callback_pause(command_center, mock_client):
    """cb_pause callback should trigger _handle_pause."""
    mock_client.answer_callback_query = AsyncMock()
    callback = {
        "id": "cb005",
        "message": {"chat": {"id": "123456789"}},
        "data": "cb_pause",
    }
    await command_center._dispatch_callback(callback)
    assert command_center._state.is_paused is True


@pytest.mark.asyncio
async def test_dispatch_callback_trades(command_center_with_live, mock_client):
    """cb_trades callback should trigger _handle_trades."""
    mock_client.answer_callback_query = AsyncMock()
    with patch("wallet_manager.config.get_settings") as mock_settings:
        mock_settings.return_value.paper_trade = False
        mock_settings.return_value.execution_mode = "legacy"
        callback = {
            "id": "cb006",
            "message": {"chat": {"id": "123456789"}},
            "data": "cb_trades",
        }
        await command_center_with_live._dispatch_callback(callback)
    assert mock_client.send_message.called


@pytest.mark.asyncio
async def test_dispatch_callback_regime(command_center_with_live, mock_client):
    """cb_regime callback should trigger _handle_regime."""
    mock_client.answer_callback_query = AsyncMock()
    callback = {
        "id": "cb007",
        "message": {"chat": {"id": "123456789"}},
        "data": "cb_regime",
    }
    await command_center_with_live._dispatch_callback(callback)
    assert mock_client.send_message.called


# ---------------------------------------------------------------------------
# LiveStatusProvider (telegram_commands version) builder tests
# ---------------------------------------------------------------------------


class TestLiveStatusProviderBuilders:
    """Tests for the telegram_commands.LiveStatusProvider (duck-typed orch version)."""

    def _make_provider(self, **orch_attrs):
        from wallet_manager.security.telegram_commands import LiveStatusProvider
        # Use spec=None so getattr(mock, unknown_key) returns the default (None)
        # by configuring __getattr__ to return None for unknown attrs
        orch = MagicMock(spec=[])  # empty spec → getattr raises AttributeError → default
        for k, v in orch_attrs.items():
            setattr(orch, k, v)
        return LiveStatusProvider(orchestrator=orch)

    def test_build_dashboard_returns_string(self):
        provider = self._make_provider(
            portfolio_total_usd=1000.0,
            portfolio_24h_change_usd=50.0,
            portfolio_24h_change_pct=5.0,
            pillars={},
            wallet_address="0x1234567890123456789012345678901234567890",
            eth_balance=0.5,
            eth_price_usd=2000.0,
            usdc_balance=100.0,
            aave_balance=200.0,
            gas_gwei=5.0,
            execution_mode="robust_paper",
            current_regime="BULL",
            uptime_seconds=3600.0,
            last_heartbeat_ago_minutes=2.0,
            paper_sharpe=1.2,
            real_sharpe=0.9,
            last_tx_hash="0xabc",
            last_tx_ago_minutes=10.0,
            rebalance_needed=False,
        )
        result = provider._build_dashboard()
        assert isinstance(result, str)
        assert "Portfolio" in result

    def test_build_dashboard_masks_wallet_address(self):
        provider = self._make_provider(
            portfolio_total_usd=0.0,
            portfolio_24h_change_usd=0.0,
            portfolio_24h_change_pct=0.0,
            pillars={},
            wallet_address="0xAbCd1234567890AbCd1234567890AbCd12345678",
            eth_balance=0.0, eth_price_usd=3000.0, usdc_balance=0.0,
            aave_balance=0.0, gas_gwei=0.0, execution_mode="paper",
            current_regime="UNKNOWN", uptime_seconds=0.0,
            last_heartbeat_ago_minutes=0.0, paper_sharpe=0.0,
            real_sharpe=0.0, last_tx_hash="0x0", last_tx_ago_minutes=0.0,
            rebalance_needed=False,
        )
        result = provider._build_dashboard()
        assert "0xAbCd1234567890AbCd1234567890AbCd12345678" not in result

    def test_build_risk_returns_string(self):
        provider = self._make_provider()
        result = provider._build_risk()
        assert isinstance(result, str)
        assert "Risk" in result or "Drawdown" in result

    def test_build_pnl_positive(self):
        provider = self._make_provider(pnl_summary={"24h": 50.0, "7d": 200.0, "30d": 500.0})
        result = provider._build_pnl()
        assert "+" in result
        assert "50" in result

    def test_build_pnl_negative(self):
        provider = self._make_provider(pnl_summary={"24h": -20.0, "7d": -50.0, "30d": -100.0})
        result = provider._build_pnl()
        assert "-" in result

    def test_build_trades_no_trades(self):
        provider = self._make_provider(last_trades=[])
        result = provider._build_trades()
        assert "No trades" in result

    def test_build_trades_with_trades(self):
        trades = [
            {"strategy": "YIELD", "venue": "Aave", "entry": 1.0, "exit": 1.1, "pnl": 10.0},
            {"strategy": "ALGO", "venue": "Uniswap", "entry": 2.0, "exit": 1.9, "pnl": -5.0},
        ]
        provider = self._make_provider(last_trades=trades)
        result = provider._build_trades()
        assert "YIELD" in result
        assert "ALGO" in result

    def test_build_regime_returns_string(self):
        provider = self._make_provider(
            current_regime="BULL",
            next_calendar_event="FOMC",
            next_calendar_event_hours=None,  # None → no format attempt
            velocity_ewm=0.001,
            hmm_state_probs={},
        )
        result = provider._build_regime()
        assert "BULL" in result or "Regime" in result

    def test_build_paper_returns_string(self):
        provider = self._make_provider(
            paper_vs_real={"paper_sharpe": 1.2, "real_sharpe": 0.8, "paper_pnl_6h": 30.0, "real_pnl_6h": 25.0, "drift_pct": 1.5}
        )
        result = provider._build_paper()
        assert "Paper" in result or "Real" in result

    @pytest.mark.asyncio
    async def test_build_status_message_calls_dashboard(self):
        from wallet_manager.security.telegram_commands import LiveStatusProvider
        orch = MagicMock(spec=[])
        orch.portfolio_total_usd = 500.0
        orch.portfolio_24h_change_usd = 10.0
        orch.portfolio_24h_change_pct = 2.0
        orch.pillars = {}
        orch.wallet_address = "0x0000000000000000000000000000000000000000"
        orch.eth_balance = 0.1
        orch.eth_price_usd = 3000.0
        orch.usdc_balance = 50.0
        orch.aave_balance = 100.0
        orch.gas_gwei = 3.0
        orch.execution_mode = "robust_paper"
        orch.current_regime = "NEUTRAL"
        orch.uptime_seconds = 7200.0
        orch.last_heartbeat_ago_minutes = 1.0
        orch.paper_sharpe = 1.0
        orch.real_sharpe = 0.8
        orch.last_tx_hash = "0xabc"
        orch.last_tx_ago_minutes = 5.0
        orch.rebalance_needed = False
        provider = LiveStatusProvider(orchestrator=orch)
        result = await provider.build_status_message()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_pnl_message_async(self):
        from wallet_manager.security.telegram_commands import LiveStatusProvider
        orch = MagicMock()
        orch.pnl_summary = {"24h": 100.0, "7d": 300.0, "30d": 1000.0}
        provider = LiveStatusProvider(orchestrator=orch)
        result = await provider.build_pnl_message()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_regime_message_async(self):
        from wallet_manager.security.telegram_commands import LiveStatusProvider
        orch = MagicMock(spec=[])
        orch.current_regime = "BEAR"
        orch.next_calendar_event = "NFP"
        orch.next_calendar_event_hours = None
        orch.velocity_ewm = 0.0
        orch.hmm_state_probs = {}
        provider = LiveStatusProvider(orchestrator=orch)
        result = await provider.build_regime_message()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_paper_message_async(self):
        from wallet_manager.security.telegram_commands import LiveStatusProvider
        orch = MagicMock()
        provider = LiveStatusProvider(orchestrator=orch)
        result = await provider.build_paper_message()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_trades_message_no_paper(self):
        from wallet_manager.security.telegram_commands import LiveStatusProvider
        orch = MagicMock()
        orch.last_trades = []
        provider = LiveStatusProvider(orchestrator=orch)
        with patch("wallet_manager.config.get_settings") as mock_cfg:
            mock_cfg.return_value.paper_trade = False
            mock_cfg.return_value.execution_mode = "legacy"
            result = await provider.build_trades_message()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# StatusProvider base class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_provider_base_default():
    """Base StatusProvider returns placeholder text."""
    from wallet_manager.security.telegram_commands import StatusProvider
    sp = StatusProvider()
    result = await sp.build_status_message()
    assert "not configured" in result.lower() or "ℹ️" in result
