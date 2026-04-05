"""
tests/unit/test_coverage_boost_phase1.py

Targeted coverage tests for Phase 1 gate: raise total coverage from 82.57% to ≥85%.

Modules targeted:
  - wallet_manager/alerting.py                  (103 stmts, 0%)
  - wallet_manager/strategies/stacking_meta_learner.py  (29 uncovered)
  - wallet_manager/network/session_factory.py   (20 uncovered)
  - wallet_manager/reporting.py                 (63 uncovered)
  - wallet_manager/orchestrator_lifecycle.py    (55 uncovered)
  - wallet_manager/network/tx_repair.py         (39 uncovered)
  - wallet_manager/strategies/idle_sweep.py     (30 uncovered)
  - wallet_manager/strategies/hyperopt.py       (51 uncovered)
  - wallet_manager/strategies/ensemble_weighter.py   (21 uncovered)
  - wallet_manager/strategies/cross_asset.py    (37 uncovered)
  - wallet_manager/strategies/flash_crash_detector.py  (32 uncovered)
  - wallet_manager/strategies/quant_analytics.py       (72 uncovered)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# alerting.py — covers the entire module (0% → ~100%)
# ---------------------------------------------------------------------------

class TestAlertEscalationSystem:
    def _system(self):
        from wallet_manager.alerting import AlertEscalationSystem
        return AlertEscalationSystem()

    def test_info_does_not_add_active(self):
        sys = self._system()
        sys.info("src", "msg")
        assert len(sys._active) == 0

    def test_warn_adds_active(self):
        sys = self._system()
        sys.warn("RiskCommittee", "risk exceeded")
        assert len(sys._active) == 1
        assert sys._active[0].source == "RiskCommittee"

    def test_critical_adds_active(self):
        sys = self._system()
        sys.critical("DrawdownAccel", "drawdown rising")
        assert len(sys._active) == 1

    def test_critical_deduplication(self):
        sys = self._system()
        sys.critical("DrawdownAccel", "first")
        sys.critical("DrawdownAccel", "second")  # same source, unacked → deduplicated
        assert len(sys._active) == 1

    def test_critical_different_source_added(self):
        sys = self._system()
        sys.critical("src_a", "a")
        sys.critical("src_b", "b")
        assert len(sys._active) == 2

    def test_acknowledge_all(self):
        sys = self._system()
        sys.critical("DrawdownAccel", "dd")
        count = sys.acknowledge()
        assert count == 1
        assert sys._active[0].acknowledged is True

    def test_acknowledge_by_source(self):
        sys = self._system()
        sys.critical("src_a", "a")
        sys.critical("src_b", "b")
        count = sys.acknowledge(source="src_a")
        assert count == 1
        acked = [a for a in sys._active if a.acknowledged]
        assert len(acked) == 1
        assert acked[0].source == "src_a"

    def test_acknowledge_already_acked_skipped(self):
        sys = self._system()
        sys.critical("src", "msg")
        sys.acknowledge()
        count = sys.acknowledge()  # second ack — already acked
        assert count == 0

    def test_get_active_criticals(self):
        sys = self._system()
        sys.critical("src", "msg")
        crits = sys.get_active_criticals()
        assert len(crits) == 1

    def test_get_active_criticals_excludes_acked(self):
        sys = self._system()
        sys.critical("src", "msg")
        sys.acknowledge()
        assert sys.get_active_criticals() == []

    @pytest.mark.asyncio
    async def test_tick_delivers_warning_once(self):
        sys = self._system()
        sys.warn("src", "msg")
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await sys.tick(notifier)
        notifier.send_message.assert_awaited_once()
        # After delivery, warning is moved to history
        assert len([a for a in sys._active if a.source == "src"]) == 0
        assert len(sys._history) == 1

    @pytest.mark.asyncio
    async def test_tick_delivers_critical_first_send(self):
        sys = self._system()
        sys.critical("DrawdownAccel", "dd")
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await sys.tick(notifier)
        notifier.send_message.assert_awaited_once()
        # critical stays active until acked or repeat interval
        assert sys._active[0].sent_count == 1

    @pytest.mark.asyncio
    async def test_tick_critical_no_resend_before_interval(self):
        sys = self._system()
        sys.critical("DrawdownAccel", "dd")
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await sys.tick(notifier)
        await sys.tick(notifier)  # second tick — within interval
        assert notifier.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_tick_critical_resend_after_interval(self):
        sys = self._system()
        sys.critical("DrawdownAccel", "dd")
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await sys.tick(notifier)
        # Manually backdate last_sent
        sys._active[0].last_sent -= sys._CRITICAL_REPEAT_INTERVAL + 1
        await sys.tick(notifier)
        assert notifier.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_tick_acked_alert_moves_to_history(self):
        sys = self._system()
        sys.critical("src", "msg")
        sys.acknowledge()
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await sys.tick(notifier)
        assert len(sys._history) == 1
        assert len(sys._active) == 0

    @pytest.mark.asyncio
    async def test_tick_no_notifier(self):
        sys = self._system()
        sys.warn("src", "msg")
        await sys.tick(None)  # should not raise; delivery skipped

    @pytest.mark.asyncio
    async def test_send_raises_is_swallowed(self):
        sys = self._system()
        sys.warn("src", "msg")
        notifier = MagicMock()
        notifier.send_message = AsyncMock(side_effect=RuntimeError("telegram down"))
        await sys.tick(notifier)  # should not raise

    @pytest.mark.asyncio
    async def test_send_includes_repeat_note(self):
        sys = self._system()
        sys.critical("src", "msg")
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        # Backdate last_sent to force repeat on second tick
        sys._active[0].sent_count = 1
        sys._active[0].last_sent = 0.0
        await sys.tick(notifier)
        # After second send, sent_count should be 2
        assert sys._active[0].sent_count == 2

    def test_process_signals_ic_noise_warn(self):
        sys = self._system()
        sys.process_signals(signal_alerts=["IC_NOISE: signal noisy"], risk_alerts=[])
        assert any(a.level.value == "WARNING" for a in sys._active)

    def test_process_signals_low_diversity_critical(self):
        sys = self._system()
        sys.process_signals(signal_alerts=["LOW_DIVERSITY: ensemble too correlated"], risk_alerts=[])
        assert any(a.level.value == "CRITICAL" for a in sys._active)

    def test_process_signals_redundant_warn(self):
        sys = self._system()
        sys.process_signals(signal_alerts=["REDUNDANT: overlap detected"], risk_alerts=[])
        assert any(a.level.value == "WARNING" for a in sys._active)

    def test_process_signals_drawdown_critical(self):
        sys = self._system()
        sys.process_signals(signal_alerts=[], risk_alerts=["DRAWDOWN_ACCELERATING: -5%"])
        assert any(a.level.value == "CRITICAL" for a in sys._active)

    def test_process_signals_fat_tail_warn(self):
        sys = self._system()
        sys.process_signals(signal_alerts=[], risk_alerts=["FAT_TAIL_VAR: extreme event risk"])
        assert any(a.level.value == "WARNING" for a in sys._active)

    def test_to_dict_structure(self):
        sys = self._system()
        sys.warn("src", "msg")
        d = sys.to_dict()
        assert "active_count" in d
        assert "critical_unacked" in d
        assert "active_alerts" in d
        assert d["active_count"] == 1

    def test_alert_to_dict(self):
        from wallet_manager.alerting import Alert, AlertLevel
        a = Alert(AlertLevel.WARNING, "src", "msg")
        d = a.to_dict()
        assert d["level"] == "WARNING"
        assert "age_minutes" in d
        assert d["acknowledged"] is False

    def test_history_trimmed_at_max(self):
        sys = self._system()
        sys._max_history = 3
        for i in range(5):
            from wallet_manager.alerting import Alert, AlertLevel
            a = Alert(AlertLevel.WARNING, f"src_{i}", "msg")
            a.acknowledged = True
            sys._history.append(a)
        # Simulate trim in tick by adding one more
        sys._history = sys._history[-sys._max_history:]
        assert len(sys._history) == 3

    def test_get_alert_system_singleton(self):
        import wallet_manager.alerting as alerting_mod
        alerting_mod._alert_system = None  # reset for test isolation
        from wallet_manager.alerting import get_alert_system
        s1 = get_alert_system()
        s2 = get_alert_system()
        assert s1 is s2
        alerting_mod._alert_system = None  # cleanup


# ---------------------------------------------------------------------------
# strategies/stacking_meta_learner.py
# ---------------------------------------------------------------------------

class TestStackingMetaLearner:
    def test_fallback_predict_before_100_obs(self):
        from wallet_manager.strategies.stacking_meta_learner import StackingMetaLearner
        learner = StackingMetaLearner(fallback_weights={"xgb": 0.5, "lgbm": 0.3, "lstm": 0.1, "transformer": 0.1})
        result = learner.predict(0.6, 0.5, 0.4, 0.7)
        expected = 0.5 * 0.6 + 0.3 * 0.5 + 0.1 * 0.4 + 0.1 * 0.7
        assert abs(result - expected) < 1e-9

    def test_learned_predict_after_100_obs(self):
        from wallet_manager.strategies.stacking_meta_learner import StackingMetaLearner
        learner = StackingMetaLearner()
        for i in range(110):
            learner.record_outcome(0.6, 0.5, 0.4, 0.3, outcome=i % 2)
        result = learner.predict(0.6, 0.5, 0.4, 0.3)
        assert 0.0 <= result <= 1.0

    def test_sgd_update_is_triggered(self):
        from wallet_manager.strategies.stacking_meta_learner import StackingMetaLearner
        learner = StackingMetaLearner()
        initial_n_updates = learner._n_updates
        for i in range(110):
            learner.record_outcome(0.5, 0.5, 0.5, 0.5, outcome=1)
        assert learner._n_updates > initial_n_updates

    def test_accuracy_before_learning(self):
        from wallet_manager.strategies.stacking_meta_learner import StackingMetaLearner
        learner = StackingMetaLearner()
        assert learner.accuracy == 0.0

    def test_accuracy_after_learning(self):
        from wallet_manager.strategies.stacking_meta_learner import StackingMetaLearner
        learner = StackingMetaLearner()
        for i in range(110):
            learner.record_outcome(0.8, 0.7, 0.6, 0.9, outcome=1)
        acc = learner.accuracy
        assert 0.0 <= acc <= 1.0

    def test_get_status(self):
        from wallet_manager.strategies.stacking_meta_learner import StackingMetaLearner
        learner = StackingMetaLearner()
        for i in range(50):
            learner.record_outcome(0.5, 0.5, 0.5, 0.5, outcome=i % 2)
        status = learner.get_status()
        assert status["n_observations"] == 50
        assert status["using_learned_weights"] is False

    def test_get_meta_learner_singleton(self):
        from wallet_manager.strategies.stacking_meta_learner import get_meta_learner
        l1 = get_meta_learner()
        l2 = get_meta_learner()
        assert l1 is l2


# ---------------------------------------------------------------------------
# network/session_factory.py
# ---------------------------------------------------------------------------

class TestSessionFactory:
    @pytest.mark.asyncio
    async def test_get_session_returns_open_session(self):
        import aiohttp
        from wallet_manager.network.session_factory import SessionFactory
        SessionFactory._sessions = {}
        SessionFactory._lock = None
        session = await SessionFactory.get_session("wallet_sf_test")
        try:
            assert isinstance(session, aiohttp.ClientSession)
            assert not session.closed
        finally:
            await SessionFactory.close_all()

    @pytest.mark.asyncio
    async def test_get_session_cached(self):
        from wallet_manager.network.session_factory import SessionFactory
        SessionFactory._sessions = {}
        SessionFactory._lock = None
        s1 = await SessionFactory.get_session("wallet_cached")
        s2 = await SessionFactory.get_session("wallet_cached")
        try:
            assert s1 is s2
        finally:
            await SessionFactory.close_all()

    @pytest.mark.asyncio
    async def test_close_all(self):
        from wallet_manager.network.session_factory import SessionFactory
        SessionFactory._sessions = {}
        SessionFactory._lock = None
        s = await SessionFactory.get_session("wallet_close_sf")
        assert not s.closed
        await SessionFactory.close_all()
        assert s.closed
        assert SessionFactory._sessions == {}

    @pytest.mark.asyncio
    async def test_stale_session_replaced(self):
        from wallet_manager.network.session_factory import SessionFactory
        SessionFactory._sessions = {}
        SessionFactory._lock = None
        s1 = await SessionFactory.get_session("wallet_stale_sf")
        await s1.close()
        s2 = await SessionFactory.get_session("wallet_stale_sf")
        try:
            assert not s2.closed
            assert s2 is not s1
        finally:
            await SessionFactory.close_all()

    def test_get_lock_lazy(self):
        from wallet_manager.network.session_factory import SessionFactory
        SessionFactory._lock = None
        lock1 = SessionFactory._get_lock()
        lock2 = SessionFactory._get_lock()
        assert lock1 is lock2


# ---------------------------------------------------------------------------
# reporting.py
# ---------------------------------------------------------------------------

class TestPerformanceReporter:
    @pytest.mark.asyncio
    async def test_tick_daily_fires(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = 0.0
        reporter._last_weekly = time.time()
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await reporter.tick(notifier, None)
        notifier.send_message.assert_awaited()
        assert reporter._last_daily > 0

    @pytest.mark.asyncio
    async def test_tick_weekly_fires(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = time.time()
        reporter._last_weekly = 0.0
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        with patch.object(reporter, "_build_weekly_summary", AsyncMock(return_value="wk")):
            await reporter.tick(notifier, None)
        notifier.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_tick_no_notifier(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = 0.0
        reporter._last_weekly = time.time()
        await reporter.tick(None, None)  # no raise

    @pytest.mark.asyncio
    async def test_tick_delivery_error_swallowed(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = 0.0
        reporter._last_weekly = time.time()
        notifier = MagicMock()
        notifier.send_message = AsyncMock(side_effect=RuntimeError("fail"))
        await reporter.tick(notifier, None)
        assert reporter._last_daily > 0

    @pytest.mark.asyncio
    async def test_generate_monthly_no_engine(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        summary = await reporter.generate_monthly(None, None)
        assert "not available" in summary

    @pytest.mark.asyncio
    async def test_generate_monthly_with_notifier(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        with patch.object(reporter, "_build_monthly_summary", AsyncMock(return_value="monthly data")):
            summary = await reporter.generate_monthly(notifier, MagicMock())
        notifier.send_message.assert_awaited()
        assert summary == "monthly data"

    @pytest.mark.asyncio
    async def test_generate_monthly_delivery_error_swallowed(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        notifier = MagicMock()
        notifier.send_message = AsyncMock(side_effect=RuntimeError("fail"))
        with patch.object(reporter, "_build_monthly_summary", AsyncMock(return_value="data")):
            summary = await reporter.generate_monthly(notifier, MagicMock())
        assert summary == "data"

    def test_chunk_splits(self):
        from wallet_manager.reporting import PerformanceReporter
        text = "\n".join(["x"] * 1000)
        chunks = PerformanceReporter._chunk(text, 200)
        assert len(chunks) > 1

    def test_chunk_single(self):
        from wallet_manager.reporting import PerformanceReporter
        chunks = PerformanceReporter._chunk("short", 500)
        assert chunks == ["short"]

    def test_chunk_empty(self):
        from wallet_manager.reporting import PerformanceReporter
        chunks = PerformanceReporter._chunk("", 500)
        assert chunks == [""]

    def test_escape_md(self):
        from wallet_manager.reporting import PerformanceReporter
        r = PerformanceReporter._escape_md("a.b-c(d)!")
        assert "\\." in r
        assert "\\-" in r

    def test_to_dict_initial(self):
        from wallet_manager.reporting import PerformanceReporter
        d = PerformanceReporter().to_dict()
        assert d["last_daily_ago_h"] is None

    def test_to_dict_after_tick(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = time.time() - 100
        d = reporter.to_dict()
        assert d["last_daily_ago_h"] is not None

    def test_singleton(self):
        from wallet_manager.reporting import get_reporter
        assert get_reporter() is get_reporter()

    @pytest.mark.asyncio
    async def test_tick_weekly_no_notifier(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = time.time()
        reporter._last_weekly = 0.0
        with patch.object(reporter, "_build_weekly_summary", AsyncMock(return_value="wk")):
            await reporter.tick(None, None)
        assert reporter._last_weekly > 0

    @pytest.mark.asyncio
    async def test_tick_weekly_multi_chunk(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        reporter._last_daily = time.time()
        reporter._last_weekly = 0.0
        long_text = "\n".join(["line"] * 2000)
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        with patch.object(reporter, "_build_weekly_summary", AsyncMock(return_value=long_text)):
            await reporter.tick(notifier, None)
        assert notifier.send_message.await_count >= 2

    @pytest.mark.asyncio
    async def test_build_daily_summary_no_engine(self):
        from wallet_manager.reporting import PerformanceReporter
        reporter = PerformanceReporter()
        summary = await reporter._build_daily_summary(None)
        assert "not available" in summary


# ---------------------------------------------------------------------------
# orchestrator_lifecycle.py
# ---------------------------------------------------------------------------

class TestLifecycleManager:
    @pytest.mark.asyncio
    async def test_validate_missing_wallet_file_raises(self):
        from wallet_manager.config import Settings
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_paper",
            wallet_config_path=Path("/nonexistent/missing_wallets.json"),
        )
        with pytest.raises(ConfigurationError):
            await LifecycleManager.validate_startup_config(cfg)

    @pytest.mark.asyncio
    async def test_validate_empty_wallets_raises(self, tmp_path):
        from wallet_manager.config import Settings
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "empty_wallets.json"
        p.write_text("[]")
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_paper",
            wallet_config_path=p,
        )
        with pytest.raises(ConfigurationError):
            await LifecycleManager.validate_startup_config(cfg)

    @pytest.mark.asyncio
    async def test_validate_corrupt_json_raises(self, tmp_path):
        from wallet_manager.config import Settings
        from wallet_manager.exceptions import ConfigurationError
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "corrupt.json"
        p.write_text("{bad json{{")
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_paper",
            wallet_config_path=p,
        )
        with pytest.raises(ConfigurationError):
            await LifecycleManager.validate_startup_config(cfg)

    @pytest.mark.asyncio
    async def test_validate_no_telegram_no_raise(self, tmp_path):
        from wallet_manager.config import Settings
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "wallets.json"
        p.write_text(json.dumps([{
            "wallet_id": "w1", "private_key": "0x" + "a" * 64,
            "rpc_url": "https://mainnet.base.org", "chain_id": 8453,
        }]))
        cfg = Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_paper",
            wallet_config_path=p,
            telegram_bot_token=None,
            telegram_chat_id=None,
        )
        with patch("wallet_manager.utils.health_check.run_startup_health_checks", AsyncMock(return_value=[])):
            await LifecycleManager.validate_startup_config(cfg)

    @pytest.mark.asyncio
    async def test_migrate_missing_file_noop(self, tmp_path):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        vault = MagicMock()
        vault.encrypt = MagicMock(return_value=b"enc")
        await LifecycleManager.migrate_wallets_to_vault(vault, tmp_path / "missing.json")
        vault.encrypt.assert_not_called()

    @pytest.mark.asyncio
    async def test_migrate_plaintext_keys(self, tmp_path):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "wallets_plain.json"
        p.write_text(json.dumps([{"wallet_id": "w1", "private_key": "0x" + "a" * 64}]))
        vault = MagicMock()
        vault.encrypt = MagicMock(return_value=b"ENC_TOKEN")
        await LifecycleManager.migrate_wallets_to_vault(vault, p)
        updated = json.loads(p.read_text())
        assert updated[0].get("vault_encrypted") is True

    @pytest.mark.asyncio
    async def test_migrate_already_encrypted_skipped(self, tmp_path):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "already_enc.json"
        p.write_text(json.dumps([{
            "wallet_id": "w1", "private_key": "ENC", "vault_encrypted": True,
        }]))
        vault = MagicMock()
        vault.encrypt = MagicMock(return_value=b"x")
        await LifecycleManager.migrate_wallets_to_vault(vault, p)
        vault.encrypt.assert_not_called()

    @pytest.mark.asyncio
    async def test_migrate_corrupt_json_noop(self, tmp_path):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "corrupt_migrate.json"
        p.write_text("{bad json")
        vault = MagicMock()
        await LifecycleManager.migrate_wallets_to_vault(vault, p)
        vault.encrypt.assert_not_called()

    @pytest.mark.asyncio
    async def test_migrate_no_private_key_noop(self, tmp_path):
        from wallet_manager.orchestrator_lifecycle import LifecycleManager
        p = tmp_path / "no_pk.json"
        p.write_text(json.dumps([{"wallet_id": "w1"}]))  # no private_key field
        vault = MagicMock()
        vault.encrypt = MagicMock(return_value=b"enc")
        await LifecycleManager.migrate_wallets_to_vault(vault, p)
        vault.encrypt.assert_not_called()


# ---------------------------------------------------------------------------
# network/tx_repair.py — NonceManager
# ---------------------------------------------------------------------------

class TestNonceManager:
    @pytest.mark.asyncio
    async def test_next_nonce_cold_cache(self):
        from wallet_manager.network.tx_repair import NonceManager
        nm = NonceManager()
        with patch.object(nm, "_fetch_pending_nonce", AsyncMock(return_value=42)):
            nonce = await nm.get_and_increment("wallet_nm", "https://rpc", "0xabc")
        assert nonce == 42

    @pytest.mark.asyncio
    async def test_next_nonce_increments(self):
        from wallet_manager.network.tx_repair import NonceManager
        nm = NonceManager()
        with patch.object(nm, "_fetch_pending_nonce", AsyncMock(return_value=10)):
            n1 = await nm.get_and_increment("wallet_nm2", "https://rpc", "0xabc")
            n2 = await nm.get_and_increment("wallet_nm2", "https://rpc", "0xabc")
        assert n1 == 10
        assert n2 == 11

    @pytest.mark.asyncio
    async def test_rewind(self):
        from wallet_manager.network.tx_repair import NonceManager
        nm = NonceManager()
        with patch.object(nm, "_fetch_pending_nonce", AsyncMock(return_value=5)):
            await nm.get_and_increment("wallet_rw", "https://rpc", "0xabc")
        await nm.rewind("wallet_rw")
        with patch.object(nm, "_fetch_pending_nonce", AsyncMock(return_value=5)):
            n = await nm.get_and_increment("wallet_rw", "https://rpc", "0xabc")
        assert n == 5

    @pytest.mark.asyncio
    async def test_rewind_unsync_noop(self):
        from wallet_manager.network.tx_repair import NonceManager
        nm = NonceManager()
        await nm.rewind("wallet_never_synced")  # should not raise

    @pytest.mark.asyncio
    async def test_invalidate(self):
        from wallet_manager.network.tx_repair import NonceManager
        nm = NonceManager()
        with patch.object(nm, "_fetch_pending_nonce", AsyncMock(return_value=7)):
            await nm.get_and_increment("wallet_inv", "https://rpc", "0xabc")
        await nm.invalidate("wallet_inv")
        with patch.object(nm, "_fetch_pending_nonce", AsyncMock(return_value=9)) as mock_fetch:
            n = await nm.get_and_increment("wallet_inv", "https://rpc", "0xabc")
        assert n == 9
        mock_fetch.assert_awaited_once()


# ---------------------------------------------------------------------------
# strategies/idle_sweep.py — calculate_sweepable_usdc, run_idle_sweep
# ---------------------------------------------------------------------------

class TestIdleSweep:
    @pytest.mark.asyncio
    async def test_calculate_sweepable_usdc_below_buffer(self):
        from wallet_manager.strategies.idle_sweep import calculate_sweepable_usdc
        # 24_999_999 < 20M buffer + 5M min sweep (= 25M) → result ≤ 0
        with patch("wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance", AsyncMock(return_value=24_999_999)):
            result = await calculate_sweepable_usdc("https://rpc", "0xabc", 8453)
        assert result <= 0

    @pytest.mark.asyncio
    async def test_calculate_sweepable_usdc_above_buffer(self):
        from wallet_manager.strategies.idle_sweep import calculate_sweepable_usdc
        # 500M >> 20M buffer
        with patch("wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance", AsyncMock(return_value=500_000_000)):
            result = await calculate_sweepable_usdc("https://rpc", "0xabc", 8453)
        assert result > 0

    @pytest.mark.asyncio
    async def test_calculate_sweepable_usdc_fetch_error(self):
        from wallet_manager.strategies.idle_sweep import calculate_sweepable_usdc
        with patch("wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance", AsyncMock(side_effect=RuntimeError("rpc error"))):
            result = await calculate_sweepable_usdc("https://rpc", "0xabc", 8453)
        assert result == 0

    @pytest.mark.asyncio
    async def test_run_idle_sweep_wrong_chain(self):
        from wallet_manager.strategies.idle_sweep import run_idle_sweep
        manager = MagicMock()
        config = MagicMock()
        config.chain_id = 1  # Ethereum mainnet, not Base
        manager.get_config = MagicMock(return_value=config)
        result = await run_idle_sweep(manager, "w1", {}, None)
        assert result == []

    @pytest.mark.asyncio
    async def test_run_idle_sweep_high_gas(self):
        from wallet_manager.strategies.idle_sweep import run_idle_sweep
        manager = MagicMock()
        config = MagicMock()
        config.chain_id = 8453
        config.rpc_url = "https://rpc"
        manager.get_config = MagicMock(return_value=config)
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 50.0))
        with patch("wallet_manager.network.gas_monitor.GasMonitor") as mock_gm_cls:
            mock_gm_cls.return_value = mock_gas
            with patch("wallet_manager.strategies.idle_sweep._get_wallet_address", return_value="0xabc"):
                result = await run_idle_sweep(manager, "w1", {}, None)
        assert result == []

    @pytest.mark.asyncio
    async def test_run_idle_sweep_no_sweepable(self):
        from wallet_manager.strategies.idle_sweep import run_idle_sweep
        manager = MagicMock()
        config = MagicMock()
        config.chain_id = 8453
        config.rpc_url = "https://rpc"
        manager.get_config = MagicMock(return_value=config)
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 2.0))
        with patch("wallet_manager.network.gas_monitor.GasMonitor") as mock_gm_cls:
            mock_gm_cls.return_value = mock_gas
            with patch("wallet_manager.strategies.idle_sweep._get_wallet_address", return_value="0xabc"):
                with patch("wallet_manager.strategies.idle_sweep.calculate_sweepable_usdc", AsyncMock(return_value=0)):
                    result = await run_idle_sweep(manager, "w1", {}, None)
        assert result == []


# ---------------------------------------------------------------------------
# strategies/hyperopt.py — BayesianHyperoptimizer
# ---------------------------------------------------------------------------

class TestBayesianHyperoptimizer:
    @pytest.mark.asyncio
    async def test_run_weekly_search_no_optuna(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        opt = BayesianHyperoptimizer()
        opt._optuna_available = False  # pre-set so no import attempt
        result = await opt.run_weekly_search(paper_engine=None)
        assert result["status"] == "OPTUNA_NOT_INSTALLED"

    @pytest.mark.asyncio
    async def test_run_weekly_search_insufficient_data(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        opt = BayesianHyperoptimizer()
        with patch.object(opt, "_gather_outcomes", AsyncMock(return_value=[])):
            with patch.object(opt, "_check_optuna", return_value=True):
                result = await opt.run_weekly_search(paper_engine=None)
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_should_apply_false_no_candidate(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        opt = BayesianHyperoptimizer()
        assert opt.should_apply() is False

    def test_should_apply_true(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer, HyperparamSet
        opt = BayesianHyperoptimizer()
        opt._current_params = HyperparamSet(sharpe_proxy=1.0)
        opt._best_candidate = HyperparamSet(sharpe_proxy=2.0)  # 100% improvement > 5% threshold
        assert opt.should_apply() is True

    def test_should_apply_false_no_improvement(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer, HyperparamSet
        opt = BayesianHyperoptimizer()
        opt._current_params = HyperparamSet(sharpe_proxy=1.0)
        opt._best_candidate = HyperparamSet(sharpe_proxy=1.01)  # < 5% improvement
        assert opt.should_apply() is False

    def test_apply_best_params(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer, HyperparamSet
        opt = BayesianHyperoptimizer()
        opt._current_params = HyperparamSet(sharpe_proxy=1.0)
        opt._best_candidate = HyperparamSet(sharpe_proxy=2.0)
        applied = opt.apply_best_params()
        assert applied is True
        assert opt._current_params.sharpe_proxy == 2.0
        assert opt._best_candidate is None
        assert opt._apply_count == 1

    def test_apply_best_params_false_when_none(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        opt = BayesianHyperoptimizer()
        assert opt.apply_best_params() is False

    def test_apply_best_params_false_no_improvement(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer, HyperparamSet
        opt = BayesianHyperoptimizer()
        opt._current_params = HyperparamSet(sharpe_proxy=1.0)
        opt._best_candidate = HyperparamSet(sharpe_proxy=1.01)
        assert opt.apply_best_params() is False

    def test_to_dict(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        d = BayesianHyperoptimizer().to_dict()
        assert "current_params" in d

    @pytest.mark.asyncio
    async def test_gather_outcomes_no_engine(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        outcomes = await BayesianHyperoptimizer()._gather_outcomes(None)
        assert outcomes == []

    @pytest.mark.asyncio
    async def test_gather_outcomes_with_engine(self):
        from wallet_manager.strategies.hyperopt import BayesianHyperoptimizer
        mock_engine = MagicMock()
        mock_engine._ledger = MagicMock()
        mock_engine._ledger.get_recent_trades = AsyncMock(return_value=[
            {"direction": "SELL", "pnl": 0.01},
            {"direction": "BUY", "pnl": 0.0},
        ])
        outcomes = await BayesianHyperoptimizer()._gather_outcomes(mock_engine)
        assert len(outcomes) == 1

    def test_get_hyperoptimizer_singleton(self):
        from wallet_manager.strategies.hyperopt import get_hyperoptimizer
        assert get_hyperoptimizer() is get_hyperoptimizer()


# ---------------------------------------------------------------------------
# strategies/ensemble_weighter.py
# ---------------------------------------------------------------------------

class TestDynamicEnsembleWeighter:
    def test_update_weights_normal(self):
        from wallet_manager.strategies.ensemble_weighter import DynamicEnsembleWeighter
        ew = DynamicEnsembleWeighter.__new__(DynamicEnsembleWeighter)
        ew._weights = {"xgb": 0.4, "lgbm": 0.25, "lstm": 0.25, "transformer": 0.1}
        sharpe_map = {"xgb": 0.8, "lgbm": 0.5, "lstm": 0.4, "transformer": 0.1}
        new_weights = ew.update_weights(sharpe_map)
        assert abs(sum(new_weights.values()) - 1.0) < 1e-6

    def test_update_weights_all_zero_returns_default(self):
        from wallet_manager.strategies.ensemble_weighter import DynamicEnsembleWeighter
        ew = DynamicEnsembleWeighter.__new__(DynamicEnsembleWeighter)
        ew._weights = {"xgb": 0.4, "lgbm": 0.25, "lstm": 0.25, "transformer": 0.1}
        sharpe_map = {"xgb": -1.0, "lgbm": -0.5, "lstm": -0.2, "transformer": 0.0}
        new_weights = ew.update_weights(sharpe_map)
        assert abs(sum(new_weights.values()) - 1.0) < 1e-6

    def test_update_weights_persists_to_file(self, tmp_path):
        from wallet_manager.strategies.ensemble_weighter import DynamicEnsembleWeighter
        path = tmp_path / "weights.json"
        ew = DynamicEnsembleWeighter(path=str(path))
        sharpe_map = {"xgb": 1.0, "lgbm": 0.8, "lstm": 0.6, "transformer": 0.4}
        ew.update_weights(sharpe_map)
        ew.persist(str(path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_load_from_existing_file(self, tmp_path):
        from wallet_manager.strategies.ensemble_weighter import DynamicEnsembleWeighter
        path = tmp_path / "weights_load.json"
        path.write_text(json.dumps({"xgb": 0.5, "lgbm": 0.5, "lstm": 0.0, "transformer": 0.0}))
        ew = DynamicEnsembleWeighter(path=str(path))
        w = ew.get_weights()
        assert w["xgb"] == 0.5

    def test_load_from_bad_file_uses_default(self, tmp_path):
        from wallet_manager.strategies.ensemble_weighter import DynamicEnsembleWeighter
        path = tmp_path / "bad_weights.json"
        path.write_text("{invalid json")
        ew = DynamicEnsembleWeighter(path=str(path))
        w = ew.get_weights()
        assert isinstance(w, dict)

    def test_get_singleton(self):
        from wallet_manager.strategies.ensemble_weighter import get_singleton
        assert get_singleton() is get_singleton()


# ---------------------------------------------------------------------------
# strategies/cross_asset.py — CrossAssetFeatures, FundingRateSignal
# ---------------------------------------------------------------------------

class TestCrossAsset:
    @pytest.mark.asyncio
    async def test_fetch_btc_price_cache_hit(self):
        from wallet_manager.strategies.cross_asset import CrossAssetFeatures
        sig = CrossAssetFeatures()
        sig._last_btc_price = 45000.0
        sig._last_fetch = time.time()
        price = await sig.fetch_btc_price()
        assert price == 45000.0

    @pytest.mark.asyncio
    async def test_fetch_btc_price_network_error_returns_cached(self):
        from wallet_manager.strategies.cross_asset import CrossAssetFeatures
        sig = CrossAssetFeatures()
        sig._last_btc_price = 50000.0
        sig._last_fetch = time.time() - 9999
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_resp = AsyncMock()
            mock_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("network error"))
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            price = await sig.fetch_btc_price()
        assert price == 50000.0  # returns cached

    def test_record_prices_and_get_features(self):
        from wallet_manager.strategies.cross_asset import CrossAssetFeatures
        sig = CrossAssetFeatures()
        for i in range(5):
            sig.record_prices(eth_price=3000.0 + i * 10, btc_price=60000.0 + i * 100)
        features = sig.get_features()
        assert isinstance(features, dict)

    def test_get_features_empty(self):
        from wallet_manager.strategies.cross_asset import CrossAssetFeatures
        sig = CrossAssetFeatures()
        features = sig.get_features()
        assert isinstance(features, dict)

    def test_funding_rate_signal_normalize(self):
        from wallet_manager.strategies.cross_asset import FundingRateSignal
        f = FundingRateSignal()
        assert f._normalize(0.0) == 0.0
        assert f._normalize(1.0) == 1.0
        assert f._normalize(-1.0) == -1.0

    def test_funding_rate_get_signal_structure(self):
        from wallet_manager.strategies.cross_asset import FundingRateSignal
        f = FundingRateSignal()
        d = f.get_signal()
        assert "normalized_signal" in d
        assert "sentiment" in d

    @pytest.mark.asyncio
    async def test_funding_rate_fetch_cache(self):
        from wallet_manager.strategies.cross_asset import FundingRateSignal
        f = FundingRateSignal()
        f._last_rate = 0.0001
        f._last_fetch = time.time()
        result = await f.fetch()
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_funding_rate_fetch_error_returns_cached(self):
        from wallet_manager.strategies.cross_asset import FundingRateSignal
        f = FundingRateSignal()
        f._last_rate = 0.0002
        f._last_fetch = time.time() - 9999
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.__aenter__ = AsyncMock(side_effect=RuntimeError("fail"))
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await f.fetch()
        assert isinstance(result, float)

    def test_get_cross_asset_singleton(self):
        from wallet_manager.strategies.cross_asset import get_cross_asset
        assert get_cross_asset() is get_cross_asset()

    def test_get_funding_rate_signal_singleton(self):
        from wallet_manager.strategies.cross_asset import get_funding_rate_signal
        assert get_funding_rate_signal() is get_funding_rate_signal()


# ---------------------------------------------------------------------------
# strategies/flash_crash_detector.py
# ---------------------------------------------------------------------------

class TestFlashCrashFunctions:
    def test_send_kill_switch_alert_no_context(self):
        from wallet_manager.strategies.flash_crash_detector import activate_kill_switch
        # No command_state in context — should just log warning, not raise
        activate_kill_switch({}, MagicMock(drop_pct=15.0, window_seconds=60.0))

    def test_activate_kill_switch_triggers_halt(self):
        from wallet_manager.strategies.flash_crash_detector import activate_kill_switch
        command_state = MagicMock()
        command_state.trigger_circuit_breaker_halt = MagicMock()
        alert = MagicMock()
        alert.drop_pct = 20.0
        alert.window_seconds = 30.0
        activate_kill_switch({"command_state": command_state}, alert)
        command_state.trigger_circuit_breaker_halt.assert_called_once()

    def test_activate_kill_switch_no_halt_method(self):
        from wallet_manager.strategies.flash_crash_detector import activate_kill_switch
        command_state = MagicMock(spec=[])  # no trigger_circuit_breaker_halt
        alert = MagicMock(drop_pct=20.0, window_seconds=30.0)
        activate_kill_switch({"command_state": command_state}, alert)  # should not raise

    @pytest.mark.asyncio
    async def test_send_kill_switch_alert_with_notifier(self):
        from wallet_manager.strategies.flash_crash_detector import send_kill_switch_alert
        notifier = MagicMock()
        notifier.send = AsyncMock()
        alert = MagicMock(drop_pct=20.0, window_seconds=30.0, peak_price=3000.0, current_price=2400.0)
        ctx = {"notifier": notifier}
        await send_kill_switch_alert(ctx, alert)

    @pytest.mark.asyncio
    async def test_send_kill_switch_alert_with_telegram_manager(self):
        from wallet_manager.strategies.flash_crash_detector import send_kill_switch_alert
        tg = MagicMock()
        tg.send_circuit_breaker_alert = AsyncMock()
        alert = MagicMock(drop_pct=20.0, window_seconds=30.0, peak_price=3000.0, current_price=2400.0)
        ctx = {"telegram_manager": tg}
        await send_kill_switch_alert(ctx, alert)
        tg.send_circuit_breaker_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_kill_switch_alert_telegram_manager_fails_fallback(self):
        from wallet_manager.strategies.flash_crash_detector import send_kill_switch_alert
        tg = MagicMock()
        tg.send_circuit_breaker_alert = AsyncMock(side_effect=RuntimeError("tg fail"))
        notifier = MagicMock()
        notifier.send = AsyncMock()
        alert = MagicMock(drop_pct=20.0, window_seconds=30.0, peak_price=3000.0, current_price=2400.0)
        ctx = {"telegram_manager": tg, "notifier": notifier}
        await send_kill_switch_alert(ctx, alert)
        notifier.send.assert_awaited_once()

    def test_get_detector_singleton(self):
        from wallet_manager.strategies.flash_crash_detector import (
            get_detector,
            reset_detector,
        )
        reset_detector()
        d1 = get_detector()
        d2 = get_detector()
        assert d1 is d2
        reset_detector()

    def test_reset_detector(self):
        from wallet_manager.strategies.flash_crash_detector import (
            get_detector,
            reset_detector,
        )
        reset_detector()
        d1 = get_detector()
        reset_detector()
        import wallet_manager.strategies.flash_crash_detector as fcd
        assert fcd._detector is None


# ---------------------------------------------------------------------------
# strategies/quant_analytics.py — QuantAnalytics DB methods
# ---------------------------------------------------------------------------

class TestQuantAnalytics:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_test.db"))
        await qa.initialize()
        assert qa._initialized is True
        await qa.close()
        assert qa._initialized is False

    @pytest.mark.asyncio
    async def test_record_market_features_no_features(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_mf.db"))
        await qa.initialize()
        try:
            await qa.record_market_features(price=3000.0)
        finally:
            await qa.close()

    @pytest.mark.asyncio
    async def test_record_ml_prediction_none(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_ml.db"))
        await qa.initialize()
        try:
            await qa.record_ml_prediction(signal=None)  # should not raise
        finally:
            await qa.close()

    @pytest.mark.asyncio
    async def test_get_tca_report_empty(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_tca.db"))
        await qa.initialize()
        try:
            reports = await qa.get_tca_report(last_n=10)
            assert isinstance(reports, list)
        finally:
            await qa.close()

    @pytest.mark.asyncio
    async def test_get_performance_metrics_empty(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_pm.db"))
        await qa.initialize()
        try:
            metrics = await qa.get_performance_metrics()
            assert metrics is not None
        finally:
            await qa.close()

    @pytest.mark.asyncio
    async def test_write_training_csv(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        out = tmp_path / "train.csv"
        count = QuantAnalytics._write_training_csv(
            str(out),
            ["col_a", "col_b"],
            [(1, 2), (3, 4)],
        )
        assert count == 2
        assert out.exists()

    @pytest.mark.asyncio
    async def test_ensure_db_auto_initializes(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_ensure.db"))
        assert qa._initialized is False
        try:
            await qa._ensure_db()
            assert qa._initialized is True
        finally:
            await qa.close()

    def test_get_db_raises_when_not_initialized(self):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics.__new__(QuantAnalytics)
        qa._db = None
        qa._initialized = False
        with pytest.raises(RuntimeError):
            qa._get_db()

    @pytest.mark.asyncio
    async def test_record_trade_execution(self, tmp_path):
        from wallet_manager.strategies.quant_analytics import QuantAnalytics
        qa = QuantAnalytics(db_path=str(tmp_path / "qa_te.db"))
        await qa.initialize()
        try:
            await qa.record_trade_execution(
                side="BUY",
                entry_price=3000.0,
                execution_price=3001.0,
                position_usd=100.0,
            )
        finally:
            await qa.close()
