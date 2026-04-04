"""tests/unit/test_telegram_commands_coverage.py

Comprehensive unit tests for wallet_manager/security/telegram_commands.py.
Covers: _mask_address, _split_message, _MessageFormatter, CommandState,
        ConfirmationGate, LiveStatusProvider builders, CommandCenter dispatch
        branches and error handling paths.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_creds(chat_id: str = "111222333") -> MagicMock:
    creds = MagicMock()
    creds.chat_id = chat_id
    return creds


def _make_client() -> MagicMock:
    client = MagicMock()
    client.send_message = AsyncMock()
    client.send_message_with_keyboard = AsyncMock()
    client.answer_callback_query = AsyncMock()
    client.get_updates = AsyncMock(return_value=[])
    return client


def _make_center(chat_id: str = "111222333", status_provider=None, gate=None):
    """Create CommandCenter with fresh CommandState inside same async scope."""
    from wallet_manager.security.telegram_commands import (
        CommandCenter,
        CommandState,
    )
    state = CommandState()
    creds = _make_creds(chat_id)
    client = _make_client()
    center = CommandCenter(
        credentials=creds,
        command_state=state,
        client=client,
        status_provider=status_provider,
        confirmation_gate=gate,
    )
    return center, state, client


def _upd(text: str, chat_id: str = "111222333") -> dict:
    return {
        "update_id": 1,
        "message": {
            "text": text,
            "chat": {"id": int(chat_id)},
        },
    }


# ===========================================================================
# _mask_address  (sync — no event loop needed)
# ===========================================================================

class TestMaskAddress:

    def test_masks_full_eth_address(self):
        from wallet_manager.security.telegram_commands import _mask_address
        addr = "0xAbCd1234567890abcdef1234567890ABCDEF1234"
        result = _mask_address(addr)
        assert "0xAbCd" in result
        assert "****" in result
        assert "1234" in result

    def test_passthrough_short_string(self):
        from wallet_manager.security.telegram_commands import _mask_address
        short = "hello world"
        assert _mask_address(short) == short

    def test_masks_private_key_in_sentence(self):
        from wallet_manager.security.telegram_commands import _mask_address
        text = "key=0xDEAD1234dead12345678901234567890dead1234 found"
        result = _mask_address(text)
        assert "0xDEAD" in result
        assert "****" in result

    def test_no_address_unchanged(self):
        from wallet_manager.security.telegram_commands import _mask_address
        text = "Gas: 42 Gwei"
        assert _mask_address(text) == text

    def test_multiple_addresses_all_masked(self):
        from wallet_manager.security.telegram_commands import _mask_address
        a1 = "0xAAAA1234567890abcdef1234567890ABCDAAAA1234"
        a2 = "0xBBBB1234567890abcdef1234567890ABCDBBBB5678"
        result = _mask_address(f"{a1} and {a2}")
        assert result.count("****") == 2

    def test_empty_string(self):
        from wallet_manager.security.telegram_commands import _mask_address
        assert _mask_address("") == ""


# ===========================================================================
# _split_message  (sync)
# ===========================================================================

class TestSplitMessage:

    def test_short_message_unchanged(self):
        from wallet_manager.security.telegram_commands import _split_message
        msg = "Hello, world!"
        assert _split_message(msg) == [msg]

    def test_exactly_max_len_no_split(self):
        from wallet_manager.security.telegram_commands import _split_message
        msg = "x" * 4096
        parts = _split_message(msg)
        assert len(parts) == 1

    def test_long_message_splits(self):
        from wallet_manager.security.telegram_commands import _split_message
        msg = "a" * 5000
        parts = _split_message(msg)
        assert len(parts) >= 2
        for p in parts:
            assert len(p) <= 4096

    def test_splits_on_double_newline(self):
        from wallet_manager.security.telegram_commands import _split_message
        half = "x" * 2048
        msg = half + "\n\n" + half
        parts = _split_message(msg, max_len=3000)
        assert len(parts) == 2

    def test_splits_on_section_separator(self):
        from wallet_manager.security.telegram_commands import _split_message
        half = "x" * 2048
        sep = "\n\u2501" * 2
        msg = half + sep + half
        parts = _split_message(msg, max_len=3000)
        assert len(parts) >= 2

    def test_hard_split_when_no_boundary(self):
        from wallet_manager.security.telegram_commands import _split_message
        msg = "z" * 9000
        parts = _split_message(msg, max_len=4096)
        assert all(len(p) <= 4096 for p in parts)

    def test_empty_string_returns_list(self):
        from wallet_manager.security.telegram_commands import _split_message
        result = _split_message("")
        assert result == [""]

    def test_custom_max_len(self):
        from wallet_manager.security.telegram_commands import _split_message
        msg = "ab\n\ncd" * 10
        parts = _split_message(msg, max_len=20)
        assert len(parts) > 1


# ===========================================================================
# _MessageFormatter  (sync)
# ===========================================================================

class TestMessageFormatter:

    def _evt(self, kind, **kwargs):
        from wallet_manager.core.models import NotificationEvent
        return NotificationEvent(kind=kind, **kwargs)

    def test_format_boot(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.SYSTEM_BOOT)
        result = _MessageFormatter.format(evt)
        assert "STARTED" in result

    def test_format_shutdown_no_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.SYSTEM_SHUTDOWN)
        result = _MessageFormatter.format(evt)
        assert "COMPLETED" in result

    def test_format_shutdown_with_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.SYSTEM_SHUTDOWN, detail="Done!")
        result = _MessageFormatter.format(evt)
        assert "Done" in result

    def test_format_gas_pause_with_gwei(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.GAS_PAUSE, gas_price_gwei=55.5)
        result = _MessageFormatter.format(evt)
        assert "55" in result and "Gwei" in result

    def test_format_gas_pause_unknown_gwei(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.GAS_PAUSE)
        result = _MessageFormatter.format(evt)
        assert "unknown" in result

    def test_format_gas_resume(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.GAS_RESUME, gas_price_gwei=30.0)
        result = _MessageFormatter.format(evt)
        assert "NORMAL" in result

    def test_format_gas_deferred(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.GAS_DEFERRED, gas_price_gwei=100.0)
        result = _MessageFormatter.format(evt)
        # Dot is escaped in MarkdownV2; check the number and unit fragments
        assert "100" in result and "Gwei" in result

    def test_format_wallet_success_with_tx(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(
            EventKind.WALLET_SUCCESS,
            wallet_id="0xAAAA1234567890abcdef1234567890ABCDAAAA5678",
            tx_hash="0xBBBB1234567890abcdef1234567890ABCDBBBB9012",
        )
        result = _MessageFormatter.format(evt)
        assert "Success" in result

    def test_format_wallet_failure_with_detail(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(
            EventKind.WALLET_FAILURE,
            wallet_id="0xCCCC1234567890abcdef1234567890ABCDCCCC3456",
            detail="Reverted",
        )
        result = _MessageFormatter.format(evt)
        assert "Failed" in result

    def test_format_generic_event(self):
        from wallet_manager.core.models import EventKind
        from wallet_manager.security.telegram_commands import _MessageFormatter
        evt = self._evt(EventKind.SYSTEM_ALERT, detail="alert msg")
        result = _MessageFormatter.format(evt)
        # Underscores are escaped in MarkdownV2 output
        assert "SYSTEM" in result and "ALERT" in result

    def test_escape_md_leaves_code_spans(self):
        from wallet_manager.security.telegram_commands import _MessageFormatter
        text = "Gas: `42 Gwei`"
        result = _MessageFormatter._escape_md(text)
        assert "`42 Gwei`" in result

    def test_escape_md_escapes_reserved_chars(self):
        from wallet_manager.security.telegram_commands import _MessageFormatter
        text = "hello. world!"
        result = _MessageFormatter._escape_md(text)
        assert r"\." in result
        assert r"\!" in result

    def test_escape_md_empty_string(self):
        from wallet_manager.security.telegram_commands import _MessageFormatter
        assert _MessageFormatter._escape_md("") == ""


# ===========================================================================
# CommandState  (sync — asyncio.Event created in running loop via pytest-asyncio)
# ===========================================================================

class TestCommandState:
    """All sync tests; asyncio.Event is attached to the running pytest loop."""

    def test_initial_state_not_paused(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        assert not s.is_paused

    def test_initial_state_not_stopped(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        assert not s.is_stopped

    def test_pause_sets_paused(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.pause()
        assert s.is_paused

    def test_resume_clears_pause(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.pause()
        s.resume()
        assert not s.is_paused

    def test_stop_sets_stopped(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.stop()
        assert s.is_stopped

    def test_stop_also_unblocks_pause(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.pause()
        s.stop()
        assert not s.is_paused

    def test_retrain_requested_default_false(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        assert not s.retrain_requested

    def test_request_retrain_sets_flag(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.request_retrain()
        assert s.retrain_requested

    def test_clear_retrain_resets_flag(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.request_retrain()
        s.clear_retrain()
        assert not s.retrain_requested

    def test_mode_change_request_and_pop(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.request_mode_change("quantx")
        assert s.pop_mode_change() == "quantx"
        assert s.pop_mode_change() is None

    def test_circuit_breaker_not_halted_by_default(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        assert not s.is_circuit_breaker_halted

    def test_trigger_circuit_breaker_halts(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.trigger_circuit_breaker_halt(duration_seconds=3600)
        assert s.is_circuit_breaker_halted

    def test_circuit_breaker_expires_naturally(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s._circuit_breaker_halt_until = time.time() - 1.0
        assert not s.is_circuit_breaker_halted

    def test_resume_clears_circuit_breaker(self):
        from wallet_manager.security.telegram_commands import CommandState
        s = CommandState()
        s.trigger_circuit_breaker_halt(duration_seconds=3600)
        s.resume()
        assert not s.is_circuit_breaker_halted


@pytest.mark.asyncio
async def test_command_state_wait_for_resume_immediate():
    from wallet_manager.security.telegram_commands import CommandState
    s = CommandState()
    await s.wait_for_resume()  # Should not hang since not paused


# ===========================================================================
# ConfirmationGate
# ===========================================================================

@pytest.mark.asyncio
async def test_confirmation_gate_timeout():
    from wallet_manager.security.telegram_commands import ConfirmationGate
    gate = ConfirmationGate()
    gate.request_confirmation()
    result = await gate.wait_for_evet(timeout_seconds=0.05)
    assert result is False


@pytest.mark.asyncio
async def test_confirmation_gate_signal_evet():
    from wallet_manager.security.telegram_commands import ConfirmationGate
    gate = ConfirmationGate()
    gate.request_confirmation()

    async def _signal():
        await asyncio.sleep(0.01)
        gate.signal_evet()

    signal_task = asyncio.create_task(_signal())
    result = await gate.wait_for_evet(timeout_seconds=2.0)
    await signal_task
    assert result is True


@pytest.mark.asyncio
async def test_confirmation_gate_reset():
    from wallet_manager.security.telegram_commands import ConfirmationGate
    gate = ConfirmationGate()
    gate.signal_evet()
    gate.request_confirmation()
    result = await gate.wait_for_evet(timeout_seconds=0.02)
    assert result is False


# ===========================================================================
# LiveStatusProvider builder methods
# ===========================================================================

def _make_provider(**attrs):
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    orch = MagicMock()
    for k, v in attrs.items():
        setattr(orch, k, v)
    return LiveStatusProvider(orchestrator=orch)


def _make_full_orch():
    """Return a MagicMock orchestrator with all numeric attrs as real numbers."""
    orch = MagicMock()
    # Numeric attrs used by _build_dashboard
    orch.portfolio_total_usd = 10000.0
    orch.portfolio_24h_change_usd = 100.0
    orch.portfolio_24h_change_pct = 1.0
    orch.eth_balance = 1.5
    orch.eth_price_usd = 3000.0
    orch.usdc_balance = 500.0
    orch.aave_balance = 200.0
    orch.gas_gwei = 20.0
    orch.uptime_seconds = 3700.0
    orch.last_heartbeat_ago_minutes = 2.0
    orch.paper_sharpe = 1.1
    orch.real_sharpe = 0.9
    orch.last_tx_ago_minutes = 5.0
    orch.execution_mode = "quantx"
    orch.current_regime = "BULL"
    orch.wallet_address = "0xAAAA1234567890abcdef1234567890ABCDAAAA5678"
    orch.last_tx_hash = "0xBBBB1234567890abcdef1234567890ABCDBBBB9012"
    orch.pillars = {
        "YIELD":   {"value": 4000.0, "target": 4000.0, "apy": 4.5},
        "ALGO":    {"value": 3500.0, "target": 3500.0, "open_positions": 2},
        "SHORT":   {"value": 1000.0, "target": 1000.0, "leverage": 1.5},
        "AIRDROP": {"value": 1500.0, "target": 1500.0, "next_swap_minutes": 30.0},
    }
    orch.rebalance_needed = False
    # For _build_risk
    orch.drawdown_pct = 2.0
    orch.var_99_usd = 500.0
    orch.cvar_99_usd = 600.0
    orch.kelly_fraction = 0.25
    orch.win_rate_50 = 0.55
    orch.sharpe_7d = 1.2
    orch.sortino_7d = 1.5
    orch.garch_sigma = 0.012
    orch.circuit_breaker_open = False
    orch.circuit_failures = 0
    orch.ir_zscore = 0.5
    orch.ir_ofi = 0.3
    orch.ir_ml = 0.7
    orch.signal_half_life_minutes = 45.0
    orch.signal_exec_seconds = 3.0
    # For _build_regime
    orch.velocity_ewm = 0.001
    orch.next_calendar_event = "FOMC"
    orch.next_calendar_event_hours = 2.5
    orch.hmm_state_probs = {"BULL": 0.7, "BEAR": 0.2, "NEUTRAL": 0.1}
    return orch


@pytest.mark.asyncio
async def test_live_status_build_dashboard_returns_string():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    p = LiveStatusProvider(orchestrator=_make_full_orch())
    msg = await p.build_status_message()
    assert isinstance(msg, str)
    assert "Portfolio" in msg


@pytest.mark.asyncio
async def test_live_status_build_dashboard_positive_change():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    orch = _make_full_orch()
    orch.portfolio_24h_change_usd = 100.0
    p = LiveStatusProvider(orchestrator=orch)
    msg = await p.build_status_message()
    assert "+" in msg


@pytest.mark.asyncio
async def test_live_status_build_dashboard_negative_change():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    orch = _make_full_orch()
    orch.portfolio_24h_change_usd = -100.0
    p = LiveStatusProvider(orchestrator=orch)
    msg = await p.build_status_message()
    assert "-" in msg


@pytest.mark.asyncio
async def test_live_status_build_risk_message():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    orch = _make_full_orch()
    orch.drawdown_pct = 3.5
    p = LiveStatusProvider(orchestrator=orch)
    msg = await p.build_risk_message()
    assert "Risk" in msg
    assert "3.50" in msg


@pytest.mark.asyncio
async def test_live_status_build_pnl_message():
    p = _make_provider(pnl_summary={"24h": 50.0, "7d": -20.0, "30d": 300.0})
    msg = await p.build_pnl_message()
    assert "P&L" in msg


@pytest.mark.asyncio
async def test_live_status_build_trades_no_trades():
    p = _make_provider(last_trades=[])
    msg = await p.build_trades_message()
    assert "No trades" in msg


@pytest.mark.asyncio
async def test_live_status_build_trades_with_data():
    p = _make_provider(last_trades=[
        {"strategy": "ALGO", "venue": "Uniswap", "entry": 1.0, "exit": 1.05, "pnl": 5.0}
    ])
    # Patch config so it does not try paper engine
    with patch("wallet_manager.security.telegram_commands.get_settings") as gs:
        cfg = MagicMock()
        cfg.paper_trade = False
        cfg.execution_mode = "legacy"
        gs.return_value = cfg
        msg = await p.build_trades_message()
    assert "ALGO" in msg


@pytest.mark.asyncio
async def test_live_status_build_regime_message():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    orch = _make_full_orch()
    p = LiveStatusProvider(orchestrator=orch)
    msg = await p.build_regime_message()
    assert "BULL" in msg


@pytest.mark.asyncio
async def test_live_status_build_paper_message():
    p = _make_provider(paper_sharpe=1.5, real_sharpe=1.2)
    msg = await p.build_paper_message()
    assert "Paper" in msg


@pytest.mark.asyncio
async def test_live_status_no_orchestrator():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    p = LiveStatusProvider(orchestrator=None)
    msg = await p.build_status_message()
    assert isinstance(msg, str)


def test_live_status_fmtu():
    from wallet_manager.security.telegram_commands import LiveStatusProvider
    assert LiveStatusProvider._fmtu(1234567.89) == "1,234,567.89"


# ===========================================================================
# CommandCenter — dispatch branches
# ===========================================================================

@pytest.mark.asyncio
async def test_cc_wrong_chat_id_ignored():
    center, _state, client = _make_center(chat_id="111222333")
    await center._dispatch(_upd("/status", chat_id="999999999"))
    client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_cc_unknown_command_replies():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/foobar"))
    client.send_message.assert_called()
    assert "Unknown" in client.send_message.call_args[0][0]


@pytest.mark.asyncio
async def test_cc_help_command():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/help"))
    client.send_message.assert_called()
    text = client.send_message.call_args[0][0]
    assert "Commands" in text or "status" in text.lower()


@pytest.mark.asyncio
async def test_cc_pause_command():
    center, state, client = _make_center()
    await center._dispatch(_upd("/pause"))
    assert state.is_paused
    client.send_message.assert_called()


@pytest.mark.asyncio
async def test_cc_pause_when_already_paused():
    center, state, client = _make_center()
    state.pause()
    await center._dispatch(_upd("/pause"))
    text = client.send_message.call_args[0][0]
    assert "already" in text.lower() or "paused" in text.lower()


@pytest.mark.asyncio
async def test_cc_resume_command():
    center, state, _client = _make_center()
    state.pause()
    await center._dispatch(_upd("/resume"))
    assert not state.is_paused


@pytest.mark.asyncio
async def test_cc_resume_when_stopped():
    center, state, client = _make_center()
    state.stop()
    client.send_message.reset_mock()
    await center._dispatch(_upd("/resume"))
    text = client.send_message.call_args[0][0]
    assert "Stop" in text or "stop" in text.lower()


@pytest.mark.asyncio
async def test_cc_stop_command():
    center, state, _client = _make_center()
    await center._dispatch(_upd("/stop"))
    assert state.is_stopped


@pytest.mark.asyncio
async def test_cc_stop_when_already_stopped():
    center, state, client = _make_center()
    state.stop()
    client.send_message.reset_mock()
    await center._dispatch(_upd("/stop"))
    text = client.send_message.call_args[0][0]
    assert "already" in text.lower() or "drain" in text.lower()


@pytest.mark.asyncio
async def test_cc_retrain_first_step():
    center, _state, _client = _make_center()
    await center._dispatch(_upd("/retrain"))
    assert center._pending_retrain is True


@pytest.mark.asyncio
async def test_cc_retrain_confirm_without_pending():
    center, state, client = _make_center()
    await center._dispatch(_upd("/retrain confirm"))
    assert not state.retrain_requested
    assert "No pending" in client.send_message.call_args[0][0]


@pytest.mark.asyncio
async def test_cc_retrain_confirm_with_pending():
    center, state, _client = _make_center()
    center._pending_retrain = True
    await center._dispatch(_upd("/retrain confirm"))
    assert state.retrain_requested
    assert center._pending_retrain is False


@pytest.mark.asyncio
async def test_cc_mode_unknown():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/mode bogusmode"))
    text = client.send_message.call_args[0][0]
    assert "Unknown" in text or "unknown" in text.lower()


@pytest.mark.asyncio
async def test_cc_mode_first_call_pending():
    center, _state, _client = _make_center()
    await center._dispatch(_upd("/mode quantx"))
    assert center._pending_mode == "quantx"


@pytest.mark.asyncio
async def test_cc_mode_second_call_confirms():
    center, state, _client = _make_center()
    center._pending_mode = "quantx"
    await center._dispatch(_upd("/mode quantx"))
    assert state.pop_mode_change() == "quantx"


@pytest.mark.asyncio
async def test_cc_setmode_no_arg_lists_modes():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/setmode"))
    text = client.send_message.call_args[0][0]
    assert "legacy" in text or "Valid" in text


@pytest.mark.asyncio
async def test_cc_setmode_invalid_mode():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/setmode invalidmode"))
    text = client.send_message.call_args[0][0]
    assert "Unknown" in text or "unknown" in text.lower()


@pytest.mark.asyncio
async def test_cc_setmode_dangerous_without_confirm():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/setmode robust_live"))
    text = client.send_message.call_args[0][0]
    assert "confirm" in text.lower() or "production" in text.lower()


@pytest.mark.asyncio
async def test_cc_setmode_dangerous_with_confirm():
    center, state, _client = _make_center()
    await center._dispatch(_upd("/setmode robust_live confirm"))
    assert state.pop_mode_change() == "robust_live"


@pytest.mark.asyncio
async def test_cc_setmode_safe_mode():
    center, state, _client = _make_center()
    await center._dispatch(_upd("/setmode legacy"))
    assert state.pop_mode_change() == "legacy"


@pytest.mark.asyncio
async def test_cc_evet_with_gate():
    from wallet_manager.security.telegram_commands import ConfirmationGate
    gate = ConfirmationGate()
    gate.request_confirmation()
    center, _state, _client = _make_center(gate=gate)
    await center._dispatch(_upd("EVET"))
    assert gate._confirmed is True


@pytest.mark.asyncio
async def test_cc_empty_message_no_crash():
    center, _state, client = _make_center()
    await center._dispatch({"update_id": 1, "message": {}})
    client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_cc_no_message_no_crash():
    center, _state, client = _make_center()
    await center._dispatch({"update_id": 1})
    client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_cc_reset_circuit_no_wallet_id():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/reset_circuit"))
    text = client.send_message.call_args[0][0]
    assert "Usage" in text


@pytest.mark.asyncio
async def test_cc_reset_circuit_without_confirm():
    center, _state, _client = _make_center()
    await center._dispatch(_upd("/reset_circuit wallet_1"))
    assert center._pending_reset_circuit == "wallet_1"


@pytest.mark.asyncio
async def test_cc_report_invalid_period():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/report yearly"))
    text = client.send_message.call_args[0][0]
    assert "Usage" in text or "Valid" in text


@pytest.mark.asyncio
async def test_cc_rotate_vault_no_key():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/rotate_vault_key"))
    text = client.send_message.call_args[0][0]
    assert "Usage" in text


@pytest.mark.asyncio
async def test_cc_rotate_vault_without_confirm():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/rotate_vault_key SOMEKEY"))
    text = client.send_message.call_args[0][0]
    assert "confirm" in text.lower()


@pytest.mark.asyncio
async def test_cc_callback_wrong_chat_ignored():
    center, _state, client = _make_center(chat_id="111222333")
    callback = {
        "id": "cb1",
        "data": "cb_status",
        "message": {"chat": {"id": 999999999}},
    }
    await center._dispatch_callback(callback)
    client.answer_callback_query.assert_called_once()
    client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_cc_callback_cb_pause():
    center, state, _client = _make_center()
    callback = {
        "id": "cb2",
        "data": "cb_pause",
        "message": {"chat": {"id": 111222333}},
    }
    await center._dispatch_callback(callback)
    assert state.is_paused


# ===========================================================================
# CommandCenter — rate limiting
# ===========================================================================

class TestCommandCenterRateLimit:

    def test_rate_limit_allows_first_commands(self):
        center, _state, _client = _make_center()
        for _ in range(3):
            assert center._check_rate_limit("111222333", "/status") is True

    def test_rate_limit_blocks_after_max(self):
        center, _state, _client = _make_center()
        from wallet_manager.security.telegram_commands import _RATE_LIMIT_MAX
        for _ in range(_RATE_LIMIT_MAX):
            center._check_rate_limit("111222333", "/status")
        result = center._check_rate_limit("111222333", "/status")
        assert result is False

    def test_rate_limit_separate_per_chat(self):
        center, _state, _client = _make_center()
        from wallet_manager.security.telegram_commands import _RATE_LIMIT_MAX
        for _ in range(_RATE_LIMIT_MAX):
            center._check_rate_limit("AAA", "/status")
        assert center._check_rate_limit("BBB", "/status") is True


# ===========================================================================
# CommandCenter — _reply retry behaviour
# ===========================================================================

@pytest.mark.asyncio
async def test_cc_reply_sends_message():
    center, _state, client = _make_center()
    await center._reply("111222333", "Hello!")
    client.send_message.assert_called_once_with("Hello!", chat_id="111222333")


@pytest.mark.asyncio
async def test_cc_reply_retries_on_failure():
    center, _state, client = _make_center()
    client.send_message = AsyncMock(side_effect=[Exception("fail"), None])
    with patch("wallet_manager.security.telegram_commands.asyncio.sleep", new=AsyncMock()):
        await center._reply("111222333", "test")
    assert client.send_message.call_count == 2


@pytest.mark.asyncio
async def test_cc_reply_splits_long_message():
    center, _state, client = _make_center()
    long_msg = "a" * 5000
    await center._reply("111222333", long_msg)
    assert client.send_message.call_count >= 2


# ===========================================================================
# CommandCenter — pnl / trades / regime / paper / risk with base provider
# ===========================================================================

@pytest.mark.asyncio
async def test_cc_pnl_with_base_provider():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/pnl"))
    text = client.send_message.call_args[0][0]
    assert "P&L" in text or "not available" in text.lower()


@pytest.mark.asyncio
async def test_cc_trades_with_base_provider():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/trades"))
    text = client.send_message.call_args[0][0]
    assert "Trade" in text or "not available" in text.lower()


@pytest.mark.asyncio
async def test_cc_regime_with_base_provider():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/regime"))
    text = client.send_message.call_args[0][0]
    assert "Regime" in text or "not available" in text.lower()


@pytest.mark.asyncio
async def test_cc_paper_with_base_provider():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/paper"))
    text = client.send_message.call_args[0][0]
    assert "Paper" in text or "not available" in text.lower()


@pytest.mark.asyncio
async def test_cc_risk_with_base_provider():
    center, _state, client = _make_center()
    await center._dispatch(_upd("/risk"))
    text = client.send_message.call_args[0][0]
    assert "Risk" in text or "not available" in text.lower()


# ===========================================================================
# CommandCenter — status command with keyboard
# ===========================================================================

@pytest.mark.asyncio
async def test_cc_status_calls_send_with_keyboard():
    from wallet_manager.security.telegram_commands import StatusProvider
    sp = StatusProvider()
    center, _state, client = _make_center(status_provider=sp)
    await center._handle_status("111222333")
    client.send_message_with_keyboard.assert_called()


@pytest.mark.asyncio
async def test_cc_status_error_path():
    sp = MagicMock()
    sp.build_status_message = AsyncMock(side_effect=RuntimeError("boom"))
    center, _state, client = _make_center(status_provider=sp)
    await center._handle_status("111222333")
    text = client.send_message.call_args[0][0]
    assert "unavailable" in text.lower() or "Status" in text
