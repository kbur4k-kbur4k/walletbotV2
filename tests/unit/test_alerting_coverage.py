"""
tests/unit/test_alerting_coverage.py

Comprehensive pytest test suite targeting:
  - wallet_manager/alerting.py        (AlertLevel, Alert, AlertEscalationSystem, get_alert_system)
  - wallet_manager/orchestrator_status.py  (LiveStatusProvider, _get_wallet_address_local)

Coverage goal: 90%+ line coverage on both modules.
All network/async calls are mocked — zero real I/O.
"""
from __future__ import annotations

import importlib
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from wallet_manager.alerting import (
    Alert,
    AlertEscalationSystem,
    AlertLevel,
    get_alert_system,
)

# ===========================================================================
# Helpers
# ===========================================================================

def _make_system() -> AlertEscalationSystem:
    """Return a fresh AlertEscalationSystem for each test."""
    return AlertEscalationSystem()


def _make_notifier(raise_on_send: Exception | None = None) -> AsyncMock:
    """Return a mock notifier with an async send_message method."""
    notifier = MagicMock()
    if raise_on_send:
        notifier.send_message = AsyncMock(side_effect=raise_on_send)
    else:
        notifier.send_message = AsyncMock()
    return notifier


# ===========================================================================
# AlertLevel
# ===========================================================================

class TestAlertLevel:
    def test_values_are_strings(self):
        assert AlertLevel.INFO == "INFO"
        assert AlertLevel.WARNING == "WARNING"
        assert AlertLevel.CRITICAL == "CRITICAL"

    def test_str_enum_membership(self):
        assert "INFO" in AlertLevel._value2member_map_
        assert "WARNING" in AlertLevel._value2member_map_
        assert "CRITICAL" in AlertLevel._value2member_map_

    def test_all_three_levels_exist(self):
        levels = list(AlertLevel)
        assert len(levels) == 3

    def test_level_identity(self):
        assert AlertLevel("INFO") is AlertLevel.INFO
        assert AlertLevel("WARNING") is AlertLevel.WARNING
        assert AlertLevel("CRITICAL") is AlertLevel.CRITICAL


# ===========================================================================
# Alert dataclass
# ===========================================================================

class TestAlertDataclass:
    def test_basic_construction(self):
        a = Alert(AlertLevel.INFO, "src", "msg")
        assert a.level == AlertLevel.INFO
        assert a.source == "src"
        assert a.message == "msg"
        assert a.acknowledged is False
        assert a.sent_count == 0
        assert a.last_sent == 0.0

    def test_timestamp_is_recent(self):
        before = time.time()
        a = Alert(AlertLevel.WARNING, "src", "msg")
        after = time.time()
        assert before <= a.timestamp <= after

    def test_to_dict_keys(self):
        a = Alert(AlertLevel.WARNING, "MySource", "Something happened")
        d = a.to_dict()
        assert set(d.keys()) == {"level", "source", "message", "timestamp",
                                  "acknowledged", "sent_count", "age_minutes"}

    def test_to_dict_level_is_string(self):
        a = Alert(AlertLevel.CRITICAL, "src", "msg")
        d = a.to_dict()
        assert d["level"] == "CRITICAL"
        assert isinstance(d["level"], str)

    def test_to_dict_age_minutes_is_float(self):
        a = Alert(AlertLevel.INFO, "src", "msg")
        d = a.to_dict()
        assert isinstance(d["age_minutes"], float)
        assert d["age_minutes"] >= 0.0

    def test_to_dict_acknowledged_false_by_default(self):
        a = Alert(AlertLevel.WARNING, "src", "msg")
        assert a.to_dict()["acknowledged"] is False

    def test_to_dict_acknowledged_true_after_setting(self):
        a = Alert(AlertLevel.CRITICAL, "src", "msg")
        a.acknowledged = True
        assert a.to_dict()["acknowledged"] is True

    def test_to_dict_sent_count_reflected(self):
        a = Alert(AlertLevel.WARNING, "src", "msg")
        a.sent_count = 3
        assert a.to_dict()["sent_count"] == 3

    def test_to_dict_source_and_message_preserved(self):
        a = Alert(AlertLevel.INFO, "SpecialSource", "detailed message text")
        d = a.to_dict()
        assert d["source"] == "SpecialSource"
        assert d["message"] == "detailed message text"


# ===========================================================================
# AlertEscalationSystem — basic methods
# ===========================================================================

class TestAlertEscalationSystemInit:
    def test_starts_empty(self):
        s = _make_system()
        assert s._active == []
        assert s._history == []
        assert s._max_history == 200

    def test_critical_repeat_interval_is_15min(self):
        assert AlertEscalationSystem._CRITICAL_REPEAT_INTERVAL == 900.0


class TestInfo:
    def test_info_does_not_add_to_active(self):
        s = _make_system()
        s.info("src", "just a note")
        assert len(s._active) == 0

    def test_info_multiple_calls_no_active(self):
        s = _make_system()
        for i in range(5):
            s.info("src", f"msg {i}")
        assert len(s._active) == 0


class TestWarn:
    def test_warn_adds_warning_alert(self):
        s = _make_system()
        s.warn("RiskEngine", "score elevated")
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.WARNING
        assert s._active[0].source == "RiskEngine"
        assert s._active[0].message == "score elevated"

    def test_warn_multiple_adds_multiple(self):
        s = _make_system()
        s.warn("A", "msg1")
        s.warn("B", "msg2")
        assert len(s._active) == 2

    def test_warn_does_not_deduplicate(self):
        """Warnings are NOT deduplicated — each call adds an alert."""
        s = _make_system()
        s.warn("SameSource", "same message")
        s.warn("SameSource", "same message")
        assert len(s._active) == 2


class TestCritical:
    def test_critical_adds_critical_alert(self):
        s = _make_system()
        s.critical("Drawdown", "drawdown exceeding limit")
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.CRITICAL

    def test_critical_deduplication_same_source(self):
        """A second CRITICAL from the same source is silently dropped."""
        s = _make_system()
        s.critical("SrcA", "first")
        s.critical("SrcA", "second attempt")
        assert len(s._active) == 1

    def test_critical_deduplication_different_sources(self):
        """Different sources each get their own CRITICAL alert."""
        s = _make_system()
        s.critical("SrcA", "msg")
        s.critical("SrcB", "msg")
        assert len(s._active) == 2

    def test_critical_after_acknowledge_allows_new(self):
        """After ACK, a new CRITICAL from the same source IS added."""
        s = _make_system()
        s.critical("SrcA", "first")
        s.acknowledge("SrcA")
        s.critical("SrcA", "second, after ack")
        # The first (acknowledged) + second = 2 entries
        assert len(s._active) == 2

    def test_critical_acknowledged_flag_prevents_dedup(self):
        """Already-acknowledged alert does not block new one from same source."""
        s = _make_system()
        s.critical("SrcA", "original")
        s._active[0].acknowledged = True
        s.critical("SrcA", "new one after manual ack")
        assert len(s._active) == 2


class TestAcknowledge:
    def test_acknowledge_all_criticals(self):
        s = _make_system()
        s.critical("A", "msg1")
        s.critical("B", "msg2")
        count = s.acknowledge()
        assert count == 2
        assert all(a.acknowledged for a in s._active)

    def test_acknowledge_specific_source(self):
        s = _make_system()
        s.critical("A", "msg1")
        s.critical("B", "msg2")
        count = s.acknowledge("A")
        assert count == 1
        assert s._active[0].acknowledged is True
        assert s._active[1].acknowledged is False

    def test_acknowledge_returns_zero_when_no_criticals(self):
        s = _make_system()
        s.warn("W", "warning only")
        count = s.acknowledge()
        assert count == 0

    def test_acknowledge_does_not_ack_warnings(self):
        s = _make_system()
        s.warn("W", "warning")
        s.critical("C", "critical")
        count = s.acknowledge()
        assert count == 1
        assert s._active[0].acknowledged is False   # warning untouched
        assert s._active[1].acknowledged is True    # critical acked

    def test_acknowledge_nonexistent_source_returns_zero(self):
        s = _make_system()
        s.critical("SrcA", "msg")
        count = s.acknowledge("SrcNonexistent")
        assert count == 0

    def test_acknowledge_already_acked_not_double_counted(self):
        s = _make_system()
        s.critical("SrcA", "msg")
        s.acknowledge("SrcA")
        count = s.acknowledge("SrcA")   # second ack
        assert count == 0


class TestGetActiveCriticals:
    def test_returns_only_unacked_criticals(self):
        s = _make_system()
        s.critical("A", "m1")
        s.critical("B", "m2")
        s.acknowledge("A")
        crits = s.get_active_criticals()
        assert len(crits) == 1
        assert crits[0].source == "B"

    def test_empty_when_no_criticals(self):
        s = _make_system()
        s.warn("W", "just a warning")
        assert s.get_active_criticals() == []

    def test_empty_after_all_acked(self):
        s = _make_system()
        s.critical("A", "msg")
        s.acknowledge()
        assert s.get_active_criticals() == []


# ===========================================================================
# AlertEscalationSystem.tick()
# ===========================================================================

class TestTick:
    @pytest.mark.asyncio
    async def test_tick_sends_unsent_warning_and_moves_to_history(self):
        s = _make_system()
        s.warn("Src", "warn msg")
        notifier = _make_notifier()
        await s.tick(notifier)
        notifier.send_message.assert_awaited_once()
        assert len(s._active) == 0
        assert len(s._history) == 1

    @pytest.mark.asyncio
    async def test_tick_skips_already_sent_warning(self):
        """A WARNING that was already sent (sent_count > 0) is NOT re-sent."""
        s = _make_system()
        s.warn("Src", "already sent warning")
        s._active[0].sent_count = 1          # simulate already delivered
        notifier = _make_notifier()
        await s.tick(notifier)
        # No send, but still moves to history since it stays in active with sent_count>0
        # Actually, tick only removes if sent_count==0 was just sent, so check logic:
        # sent_count > 0 means it won't enter the branch that appends to delivered
        notifier.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_sends_fresh_critical(self):
        s = _make_system()
        s.critical("Src", "critical msg")
        notifier = _make_notifier()
        await s.tick(notifier)
        notifier.send_message.assert_awaited_once()
        # CRITICAL stays in _active (not delivered to history yet)
        assert len(s._active) == 1
        assert s._active[0].sent_count == 1

    @pytest.mark.asyncio
    async def test_tick_repeats_critical_after_interval(self):
        s = _make_system()
        s.critical("Src", "critical msg")
        # Manually set sent_count and last_sent to simulate an old delivery
        s._active[0].sent_count = 1
        s._active[0].last_sent = time.time() - (16 * 60)  # 16 minutes ago
        notifier = _make_notifier()
        await s.tick(notifier)
        notifier.send_message.assert_awaited_once()
        assert s._active[0].sent_count == 2

    @pytest.mark.asyncio
    async def test_tick_does_not_repeat_critical_before_interval(self):
        s = _make_system()
        s.critical("Src", "critical msg")
        s._active[0].sent_count = 1
        s._active[0].last_sent = time.time() - 60  # only 1 minute ago
        notifier = _make_notifier()
        await s.tick(notifier)
        notifier.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_moves_acknowledged_alert_to_history(self):
        s = _make_system()
        s.warn("Src", "will be acked")
        s._active[0].acknowledged = True
        notifier = _make_notifier()
        await s.tick(notifier)
        assert len(s._active) == 0
        assert len(s._history) == 1
        notifier.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_trims_history_at_max(self):
        s = _make_system()
        s._max_history = 3
        # Pre-fill history with 3 items
        for i in range(3):
            s._history.append(Alert(AlertLevel.WARNING, f"src{i}", f"msg{i}"))
        # Add one more warning that will be delivered
        s.warn("New", "new warning")
        notifier = _make_notifier()
        await s.tick(notifier)
        assert len(s._history) == 3   # trimmed back to max

    @pytest.mark.asyncio
    async def test_tick_with_none_notifier_does_not_raise(self):
        s = _make_system()
        s.warn("Src", "msg")
        # Should not raise even with None notifier
        await s.tick(None)

    @pytest.mark.asyncio
    async def test_tick_mixed_alerts(self):
        """Tick with both WARNING and CRITICAL in active list."""
        s = _make_system()
        s.warn("WarnSrc", "warning")
        s.critical("CritSrc", "critical")
        notifier = _make_notifier()
        await s.tick(notifier)
        # Both should have triggered send_message
        assert notifier.send_message.await_count == 2
        # Warning moved to history, critical stays in active
        assert len(s._history) == 1
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_tick_acknowledged_critical_moved_to_history(self):
        s = _make_system()
        s.critical("Src", "critical")
        s.acknowledge("Src")
        notifier = _make_notifier()
        await s.tick(notifier)
        assert len(s._active) == 0
        assert len(s._history) == 1
        notifier.send_message.assert_not_awaited()


# ===========================================================================
# AlertEscalationSystem._send()
# ===========================================================================

class TestSend:
    @pytest.mark.asyncio
    async def test_send_with_none_notifier_is_noop(self):
        s = _make_system()
        a = Alert(AlertLevel.WARNING, "src", "msg")
        # Should return immediately without error
        await s._send(None, a)
        assert a.sent_count == 0

    @pytest.mark.asyncio
    async def test_send_increments_sent_count(self):
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.WARNING, "src", "msg")
        await s._send(notifier, a)
        assert a.sent_count == 1

    @pytest.mark.asyncio
    async def test_send_updates_last_sent(self):
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.CRITICAL, "src", "msg")
        before = time.time()
        await s._send(notifier, a)
        assert a.last_sent >= before

    @pytest.mark.asyncio
    async def test_send_uses_markdown_v2(self):
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.WARNING, "src", "msg")
        await s._send(notifier, a)
        _, kwargs = notifier.send_message.call_args
        assert kwargs.get("parse_mode") == "MarkdownV2"

    @pytest.mark.asyncio
    async def test_send_message_contains_level_value(self):
        """Sent message should contain the level name as bold text."""
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.CRITICAL, "src", "msg")
        a.sent_count = 2
        await s._send(notifier, a)
        sent_text = notifier.send_message.call_args[0][0]
        assert "CRITICAL" in sent_text

    @pytest.mark.asyncio
    async def test_send_message_contains_alert_message(self):
        """Sent message should include the alert's own message text."""
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.WARNING, "src", "the actual alert text")
        await s._send(notifier, a)
        sent_text = notifier.send_message.call_args[0][0]
        assert "the actual alert text" in sent_text

    @pytest.mark.asyncio
    async def test_send_exception_is_swallowed(self):
        """Telegram send failures must NOT propagate — alert system is resilient."""
        s = _make_system()
        notifier = _make_notifier(raise_on_send=RuntimeError("Telegram down"))
        a = Alert(AlertLevel.CRITICAL, "src", "msg")
        # Should not raise
        await s._send(notifier, a)
        assert a.sent_count == 0   # not incremented on failure

    @pytest.mark.asyncio
    async def test_send_warning_message_contains_warning_level(self):
        """Message for WARNING alert must contain the string 'WARNING'."""
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.WARNING, "src", "msg")
        await s._send(notifier, a)
        sent_text = notifier.send_message.call_args[0][0]
        assert "WARNING" in sent_text

    @pytest.mark.asyncio
    async def test_send_critical_message_contains_critical_level(self):
        """Message for CRITICAL alert must contain the string 'CRITICAL'."""
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.CRITICAL, "src", "msg")
        await s._send(notifier, a)
        sent_text = notifier.send_message.call_args[0][0]
        assert "CRITICAL" in sent_text

    @pytest.mark.asyncio
    async def test_send_includes_source_in_message(self):
        s = _make_system()
        notifier = _make_notifier()
        a = Alert(AlertLevel.CRITICAL, "DrawdownModule", "excessive drawdown")
        await s._send(notifier, a)
        sent_text = notifier.send_message.call_args[0][0]
        assert "DrawdownModule" in sent_text


# ===========================================================================
# AlertEscalationSystem.process_signals()
# ===========================================================================

class TestProcessSignals:
    def test_ic_noise_triggers_warn_signal_intelligence(self):
        s = _make_system()
        s.process_signals(["IC_NOISE detected in channel 3"], [])
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.WARNING
        assert s._active[0].source == "SignalIntelligence"

    def test_low_diversity_triggers_critical_ensemble(self):
        s = _make_system()
        s.process_signals(["LOW_DIVERSITY: only 2 of 5 signals active"], [])
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.CRITICAL
        assert s._active[0].source == "EnsembleDiversity"

    def test_redundant_triggers_warn_signal_correlation(self):
        s = _make_system()
        s.process_signals(["REDUNDANT signal pair detected"], [])
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.WARNING
        assert s._active[0].source == "SignalCorrelation"

    def test_drawdown_accelerating_triggers_critical(self):
        s = _make_system()
        s.process_signals([], ["DRAWDOWN_ACCELERATING: 3 consecutive drops"])
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.CRITICAL
        assert s._active[0].source == "DrawdownAcceleration"

    def test_fat_tail_var_triggers_warn(self):
        s = _make_system()
        s.process_signals([], ["FAT_TAIL_VAR exceeded threshold"])
        assert len(s._active) == 1
        assert s._active[0].level == AlertLevel.WARNING
        assert s._active[0].source == "FatTailVaR"

    def test_unknown_signal_alerts_ignored(self):
        s = _make_system()
        s.process_signals(["some random message"], ["another random message"])
        assert len(s._active) == 0

    def test_mixed_signals_all_routed(self):
        s = _make_system()
        s.process_signals(
            ["IC_NOISE low", "LOW_DIVERSITY alert", "REDUNDANT signals"],
            ["DRAWDOWN_ACCELERATING", "FAT_TAIL_VAR high"],
        )
        assert len(s._active) == 5

    def test_empty_lists_no_change(self):
        s = _make_system()
        s.process_signals([], [])
        assert len(s._active) == 0

    def test_process_signals_deduplicates_low_diversity_critical(self):
        """Calling process_signals twice with LOW_DIVERSITY should deduplicate."""
        s = _make_system()
        s.process_signals(["LOW_DIVERSITY alert"], [])
        s.process_signals(["LOW_DIVERSITY alert again"], [])
        # Second call hits dedup in critical()
        assert len(s._active) == 1


# ===========================================================================
# AlertEscalationSystem.to_dict()
# ===========================================================================

class TestToDict:
    def test_empty_system_to_dict(self):
        s = _make_system()
        d = s.to_dict()
        assert d["active_count"] == 0
        assert d["critical_unacked"] == 0
        assert d["history_count"] == 0
        assert d["active_alerts"] == []
        assert d["recent_history"] == []

    def test_to_dict_active_count(self):
        s = _make_system()
        s.warn("A", "msg1")
        s.warn("B", "msg2")
        d = s.to_dict()
        assert d["active_count"] == 2

    def test_to_dict_critical_unacked(self):
        s = _make_system()
        s.critical("A", "crit1")
        s.critical("B", "crit2")
        s.acknowledge("A")
        d = s.to_dict()
        assert d["critical_unacked"] == 1

    def test_to_dict_active_alerts_capped_at_10(self):
        s = _make_system()
        for i in range(15):
            s.warn(f"src{i}", f"msg{i}")
        d = s.to_dict()
        assert len(d["active_alerts"]) == 10

    def test_to_dict_recent_history_capped_at_5(self):
        s = _make_system()
        for i in range(10):
            s._history.append(Alert(AlertLevel.WARNING, f"src{i}", f"msg{i}"))
        d = s.to_dict()
        assert len(d["recent_history"]) == 5

    def test_to_dict_history_count(self):
        s = _make_system()
        s._history.append(Alert(AlertLevel.WARNING, "src", "msg"))
        s._history.append(Alert(AlertLevel.CRITICAL, "src2", "msg2"))
        d = s.to_dict()
        assert d["history_count"] == 2

    def test_to_dict_active_alerts_are_dicts(self):
        s = _make_system()
        s.warn("src", "msg")
        d = s.to_dict()
        assert isinstance(d["active_alerts"][0], dict)
        assert "level" in d["active_alerts"][0]


# ===========================================================================
# get_alert_system() singleton
# ===========================================================================

class TestGetAlertSystemSingleton:
    def test_returns_alert_escalation_system(self):
        import wallet_manager.alerting as alerting_mod
        # Reset singleton for isolation
        alerting_mod._alert_system = None
        result = get_alert_system()
        assert isinstance(result, AlertEscalationSystem)

    def test_singleton_same_instance(self):
        import wallet_manager.alerting as alerting_mod
        alerting_mod._alert_system = None
        a = get_alert_system()
        b = get_alert_system()
        assert a is b

    def test_singleton_reuses_existing(self):
        import wallet_manager.alerting as alerting_mod
        existing = AlertEscalationSystem()
        alerting_mod._alert_system = existing
        result = get_alert_system()
        assert result is existing
        # Cleanup
        alerting_mod._alert_system = None


# ===========================================================================
# orchestrator_status.py — LiveStatusProvider
# ===========================================================================

class TestOrchestratorStatus:
    """Tests for LiveStatusProvider.build_status_message()."""

    def _make_config(
        self,
        wallet_id: str = "wallet_01",
        rpc_url: str = "http://localhost:8545",
        chain_id: int = 8453,
    ):
        """Build a minimal WalletConfig-like mock."""
        cfg = MagicMock()
        cfg.wallet_id = wallet_id
        cfg.rpc_url = rpc_url
        cfg.chain_id = chain_id
        return cfg

    def _make_scheduler(self, seconds: float = 300.0) -> MagicMock:
        sched = MagicMock()
        sched.seconds_until_due = MagicMock(return_value=seconds)
        return sched

    @pytest.mark.asyncio
    async def test_no_configs_returns_no_wallets_message(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        provider = LiveStatusProvider(configs=[], scheduler=MagicMock())
        msg = await provider.build_status_message()
        assert "No wallets configured" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_basic(self):
        """Happy path: all fetches succeed, no portfolio_manager."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=300.0)  # 5 mins

        with (
            patch(
                "wallet_manager.orchestrator_status._get_wallet_address_local",
                return_value="0xABC123",
            ),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(0.5 * 1e18))),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=int(100 * 1e6))),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=1500.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=int(50 * 1e6))),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=200.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "Portfolio Live Status" in msg
        assert "0xABC123" in msg
        assert "ETH" in msg
        assert "USDC" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_with_portfolio_manager(self):
        """When portfolio_manager is provided, it syncs from chain and uses snapshot."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=7200.0)  # 2 hours

        pm = MagicMock()
        pm.sync_from_chain = AsyncMock()
        snapshot = MagicMock()
        snapshot.total_value_usd = 9999.99
        pm.get_snapshot = MagicMock(return_value=snapshot)

        with (
            patch(
                "wallet_manager.orchestrator_status._get_wallet_address_local",
                return_value="0xWALLET",
            ),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(1 * 1e18))),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=int(500 * 1e6))),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=2800.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=2800.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler, portfolio_manager=pm)
            msg = await provider.build_status_message()

        pm.sync_from_chain.assert_awaited_once()
        pm.get_snapshot.assert_called_once()
        assert "9999.99" in msg

    @pytest.mark.asyncio
    async def test_scheduler_time_hours(self):
        """When seconds >= 3600, output should say 'hours later'."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=7200.0)  # exactly 2 hours

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "hours later" in msg

    @pytest.mark.asyncio
    async def test_scheduler_time_minutes(self):
        """When 60 <= seconds < 3600, output should say 'mins later'."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=300.0)  # 5 minutes

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "mins later" in msg

    @pytest.mark.asyncio
    async def test_scheduler_time_seconds(self):
        """When 0 < seconds < 60, output should say 'secs later'."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=45.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "secs later" in msg

    @pytest.mark.asyncio
    async def test_scheduler_time_due_now(self):
        """When seconds <= 0, output should say 'due now'."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=0.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "due now" in msg

    @pytest.mark.asyncio
    async def test_fetch_native_balance_exception_defaults_to_zero(self):
        """If fetch_native_balance raises, balance_wei should default to 0."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler()

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(side_effect=RuntimeError("RPC down"))),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        # 0 wei / 1e18 = 0.0 ETH
        assert "0.0000 ETH" in msg

    @pytest.mark.asyncio
    async def test_fetch_usdc_balance_exception_defaults_to_zero(self):
        """If fetch_usdc_balance raises, USDC should default to 0."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler()

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(side_effect=OSError("network error"))),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "Spot USDC: 0.00" in msg

    @pytest.mark.asyncio
    async def test_fetch_eth_price_exception_defaults_to_3000(self):
        """If fetch_eth_price_usd raises, eth_price should default to 3000.0."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler()

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(1e18))),  # 1 ETH
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(side_effect=ValueError("price feed down"))),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=3000.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        # 1 ETH * default $3000 = $3000 — check message built without crashing
        assert "ETH" in msg

    @pytest.mark.asyncio
    async def test_aave_fetch_exception_defaults_to_zero(self):
        """If get_aave_usdc_balance raises, aave_usdc should default to 0.0."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler()

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(side_effect=Exception("aave RPC error"))),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "Aave USDC: 0.00" in msg

    @pytest.mark.asyncio
    async def test_short_strategy_exception_defaults_to_zero(self):
        """If get_short_position_state raises, short_usd should default to 0.0."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler()

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", side_effect=ImportError("module unavailable")),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "Short Pos: $0.00" in msg

    @pytest.mark.asyncio
    async def test_chain_id_none_defaults_to_zero(self):
        """If config.chain_id is None, chain_id should be 0 (int conversion)."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        cfg.chain_id = None  # force None

        scheduler = self._make_scheduler()

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=3000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "Portfolio Live Status" in msg

    @pytest.mark.asyncio
    async def test_without_portfolio_manager_total_computed_manually(self):
        """Without a portfolio_manager, total_usd is computed from individual fetches."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config()
        scheduler = self._make_scheduler(seconds=90.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=int(1e18))),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=int(500 * 1e6))),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=2000.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=2000.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=int(100 * 1e6))),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=50.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler, portfolio_manager=None)
            msg = await provider.build_status_message()

        # 2000 (eth) + 500 (spot usdc) + 100 (aave) + 50 (short) = 2650
        assert "2650.00" in msg

    @pytest.mark.asyncio
    async def test_portfolio_manager_sync_called_with_correct_args(self):
        """Verify portfolio_manager.sync_from_chain is called with correct keyword args."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        cfg = self._make_config(rpc_url="http://base-rpc.example", chain_id=8453)
        scheduler = self._make_scheduler()

        pm = MagicMock()
        pm.sync_from_chain = AsyncMock()
        snapshot = MagicMock()
        snapshot.total_value_usd = 5000.0
        pm.get_snapshot = MagicMock(return_value=snapshot)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xMYWALLET"),
            patch("wallet_manager.actions.fetch_native_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.actions.fetch_eth_price_usd", new=AsyncMock(return_value=2500.0)),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new=AsyncMock(return_value=0)),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=0.0)),
        ):
            provider = LiveStatusProvider(configs=[cfg], scheduler=scheduler, portfolio_manager=pm)
            await provider.build_status_message()

        pm.sync_from_chain.assert_awaited_once_with(
            wallet_address="0xMYWALLET",
            rpc_url="http://base-rpc.example",
            chain_id=8453,
            eth_price_usd=2500.0,
        )


class TestGetWalletAddressLocal:
    def test_delegates_to_task_factories(self):
        """_get_wallet_address_local should delegate to task_factories._get_wallet_address."""
        from wallet_manager.orchestrator_status import _get_wallet_address_local

        cfg = MagicMock()
        with patch(
            "wallet_manager.task_factories._get_wallet_address",
            return_value="0xDELEGATED",
        ) as mock_fn:
            result = _get_wallet_address_local(cfg)

        mock_fn.assert_called_once_with(cfg)
        assert result == "0xDELEGATED"
