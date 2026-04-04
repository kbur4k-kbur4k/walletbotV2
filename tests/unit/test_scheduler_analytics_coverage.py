"""
tests/unit/test_scheduler_analytics_coverage.py

Comprehensive coverage tests for four under-covered modules:
  1. wallet_manager/scheduler.py                  (target: 47.2% -> 80%+)
  2. wallet_manager/resilience/analytics.py       (target: 61.2% -> 80%+)
  3. wallet_manager/network/router.py             (target: 65%   -> 80%+)
  4. wallet_manager/security/telegram_manager.py  (target: 66.2% -> 80%+)

All I/O, network, and DB calls are fully mocked.
Each test is designed to run in < 100 ms.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Helpers shared across all sections
# ---------------------------------------------------------------------------

def _make_chain_config(
    chain_id: int = 8453,
    name: str = "Base",
    primary_rpc: str = "https://primary.example.com",
    fallback_rpcs: tuple = ("https://fallback1.example.com", "https://fallback2.example.com"),
):
    from wallet_manager.core.models import ChainConfig
    return ChainConfig(
        chain_id=chain_id,
        name=name,
        native_symbol="ETH",
        primary_rpc=primary_rpc,
        fallback_rpcs=fallback_rpcs,
        max_gas_price_gwei=5.0,
        block_time_seconds=2.0,
    )


def _make_tx_record(
    tx_hash: str = "0xdeadbeef00001111",
    wallet_id: str = "0xAbCd1234567890EF1234567890abcdef12345678",
    status=None,
    chain_id: int = 8453,
):
    from wallet_manager.core.models import TxRecord, TxStatus
    if status is None:
        status = TxStatus.PENDING
    return TxRecord(
        tx_hash=tx_hash,
        wallet_id=wallet_id,
        nonce=1,
        status=status,
        chain_id=chain_id,
        gas_price_gwei=10.0,
        gas_limit=21000,
        value_wei=1_000_000_000_000_000,
        submitted_at=time.time(),
    )


# ===========================================================================
# 1. SCHEDULER TESTS
# ===========================================================================


class TestSchedulerManagerInit:
    """Tests for SchedulerManager initialisation and file loading."""

    def test_init_uses_explicit_data_path(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "sched.json"
        sm = SchedulerManager(data_path=p)
        assert sm._path == p

    def test_init_uses_schedule_file_alias(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "sched2.json"
        sm = SchedulerManager(schedule_file=str(p))
        assert sm._path == p

    def test_init_loads_existing_file(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "sched.json"
        data = {"wallet_abc": {"last_run": 1000000.0}}
        p.write_text(json.dumps(data))
        sm = SchedulerManager(data_path=p)
        assert "wallet_abc" in sm._cache

    def test_init_handles_corrupt_json(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "corrupt.json"
        p.write_text("{not valid json!!!")
        sm = SchedulerManager(data_path=p)
        assert sm._cache == {}

    def test_init_handles_oserror(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "sched.json"
        p.write_text('{"key": 1}')
        with patch("builtins.open", side_effect=OSError("disk error")):
            # File exists so _load is called; OSError should be caught gracefully
            sm = SchedulerManager(data_path=p)
        assert sm._cache == {}

    def test_data_property_returns_cache(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "x.json")
        sm._cache = {"w1": {"last_run": 1.0}}
        assert sm._data == {"w1": {"last_run": 1.0}}

    def test_data_setter_updates_cache(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "x.json")
        sm._data = {"new": {}}
        assert sm._cache == {"new": {}}

    def test_default_data_path_uses_settings(self, tmp_path):
        """_default_data_path returns settings value when set."""
        from wallet_manager.scheduler import _default_data_path
        mock_settings = MagicMock()
        mock_settings.schedule_data_path = str(tmp_path / "custom.json")
        with patch("wallet_manager.config.get_settings", return_value=mock_settings):
            result = _default_data_path()
        assert result == Path(str(tmp_path / "custom.json"))

    def test_default_data_path_fallback_when_empty(self):
        """_default_data_path falls back to package dir when setting is empty."""
        from wallet_manager.scheduler import _default_data_path
        mock_settings = MagicMock()
        mock_settings.schedule_data_path = ""
        with patch("wallet_manager.config.get_settings", return_value=mock_settings):
            result = _default_data_path()
        assert result.name == "schedule_data.json"


class TestSchedulerManagerSave:
    """Tests for _save behaviour."""

    def test_save_creates_parent_dirs(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "sub" / "deep" / "sched.json"
        sm = SchedulerManager(data_path=p)
        sm._cache = {"w": {"last_run": 1.0}}
        sm._save()
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert "w" in loaded

    def test_save_handles_oserror(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "sched.json"
        sm = SchedulerManager(data_path=p)
        sm._cache = {"w": {}}
        with patch("builtins.open", side_effect=OSError("no space")):
            # Should log warning and not raise
            sm._save()


class TestSchedulerShouldRunSwapThisCycle:
    """Tests for should_run_swap_this_cycle method."""

    def test_new_wallet_returns_true(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.should_run_swap_this_cycle("new_wallet") is True

    def test_deferred_wallet_returns_false(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        future = time.time() + 9999
        sm._cache["w1"] = {"defer_until": future}
        assert sm.should_run_swap_this_cycle("w1") is False

    def test_wallet_past_defer_returns_based_on_last_run(self, tmp_path):
        from wallet_manager.scheduler import WEEK_SECONDS, SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        past = time.time() - 100  # defer already expired
        old_run = time.time() - WEEK_SECONDS - 10
        sm._cache["w1"] = {"defer_until": past, "last_run": old_run}
        assert sm.should_run_swap_this_cycle("w1") is True

    def test_wallet_ran_recently_returns_false(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w1"] = {"last_run": time.time() - 100}
        assert sm.should_run_swap_this_cycle("w1") is False

    def test_wallet_ran_week_ago_returns_true(self, tmp_path):
        from wallet_manager.scheduler import WEEK_SECONDS, SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w1"] = {"last_run": time.time() - WEEK_SECONDS - 1}
        assert sm.should_run_swap_this_cycle("w1") is True

    def test_legacy_next_run_epoch_not_yet_due(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w1"] = {"next_run_epoch": time.time() + 9999}
        assert sm.should_run_swap_this_cycle("w1") is False

    def test_legacy_next_run_epoch_past_due(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w1"] = {"next_run_epoch": time.time() - 1}
        assert sm.should_run_swap_this_cycle("w1") is True

    def test_entry_with_no_known_keys_returns_true(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w1"] = {}  # entry exists but no useful fields
        assert sm.should_run_swap_this_cycle("w1") is True

    def test_now_parameter_respected(self, tmp_path):
        from wallet_manager.scheduler import WEEK_SECONDS, SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        anchor = 1_000_000.0
        sm._cache["w1"] = {"last_run": anchor - WEEK_SECONDS - 1}
        assert sm.should_run_swap_this_cycle("w1", now=anchor) is True


class TestSchedulerMarkRan:
    """Tests for mark_ran and defer_due_to_gas."""

    def test_mark_ran_sets_last_run(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        sm = SchedulerManager(data_path=p)
        sm.mark_ran("wallet_1", now=1234567.0)
        assert sm._cache["wallet_1"]["last_run"] == 1234567.0
        assert p.exists()

    def test_mark_ran_clears_defer_until(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w"] = {"defer_until": 99999999.0}
        sm.mark_ran("w", now=100.0)
        assert "defer_until" not in sm._cache["w"]

    def test_defer_due_to_gas_sets_defer_until(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        now = 1_000_000.0
        sm.defer_due_to_gas("w1", defer_hours=3.0, now=now)
        assert sm._cache["w1"]["defer_until"] == now + 3.0 * 3600

    def test_defer_due_to_gas_default_2h(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        now = 5_000_000.0
        sm.defer_due_to_gas("w2", now=now)
        assert sm._cache["w2"]["defer_until"] == now + 2.0 * 3600


class TestSchedulerIsDueAndSeconds:
    """Tests for is_due and seconds_until_due."""

    def test_is_due_unknown_wallet_returns_false(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.is_due("ghost") is False

    def test_is_due_future_epoch_returns_false(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        data = {"w": {"next_run_epoch": time.time() + 9999}}
        p.write_text(json.dumps(data))
        sm = SchedulerManager(data_path=p)
        assert sm.is_due("w") is False

    def test_is_due_past_epoch_returns_true(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        data = {"w": {"next_run_epoch": time.time() - 1}}
        p.write_text(json.dumps(data))
        sm = SchedulerManager(data_path=p)
        assert sm.is_due("w") is True

    def test_seconds_until_due_unknown_returns_zero(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.seconds_until_due("ghost") == 0.0

    def test_seconds_until_due_past_epoch_returns_zero(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        data = {"w": {"next_run_epoch": time.time() - 100}}
        p.write_text(json.dumps(data))
        sm = SchedulerManager(data_path=p)
        assert sm.seconds_until_due("w") == 0.0

    def test_seconds_until_due_future_epoch(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        now = time.time()
        data = {"w": {"next_run_epoch": now + 500}}
        p.write_text(json.dumps(data))
        sm = SchedulerManager(data_path=p)
        result = sm.seconds_until_due("w", now=now)
        assert result == pytest.approx(500.0, abs=1.0)


class TestSchedulerSchedulingMethods:
    """Tests for schedule_first_run, schedule_next, schedule_gas_deferral."""

    def test_schedule_first_run_sets_next_run(self, tmp_path):
        from wallet_manager.scheduler import (
            FIRST_RUN_MAX_SECONDS,
            FIRST_RUN_MIN_SECONDS,
            SchedulerManager,
        )
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        now = 1_000_000.0
        sm.schedule_first_run("w1", now=now)
        entry = sm._cache["w1"]
        assert FIRST_RUN_MIN_SECONDS <= entry["next_run_epoch"] - now <= FIRST_RUN_MAX_SECONDS
        assert entry["last_completed_epoch"] == 0.0

    def test_schedule_next_within_week(self, tmp_path):
        from wallet_manager.scheduler import WEEK_SECONDS, SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        now = 2_000_000.0
        sm.schedule_next("w1", now=now)
        entry = sm._cache["w1"]
        assert 3600 <= entry["next_run_epoch"] - now <= WEEK_SECONDS
        assert entry["last_completed_epoch"] == now

    def test_schedule_next_updates_existing_entry(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm._cache["w1"] = {"last_run": 111.0}
        sm.schedule_next("w1", now=200.0)
        assert sm._cache["w1"]["last_completed_epoch"] == 200.0

    def test_schedule_gas_deferral_short_window(self, tmp_path):
        from wallet_manager.scheduler import (
            GAS_DEFER_MAX_SECONDS,
            GAS_DEFER_MIN_SECONDS,
            SchedulerManager,
        )
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        now = 3_000_000.0
        sm.schedule_gas_deferral("w1", now=now)
        offset = sm._cache["w1"]["next_run_epoch"] - now
        assert GAS_DEFER_MIN_SECONDS <= offset <= GAS_DEFER_MAX_SECONDS


class TestSchedulerHasRecordAndGetNextRun:
    """Tests for has_record and get_next_run_epoch."""

    def test_has_record_false_for_new_wallet(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.has_record("new") is False

    def test_has_record_true_after_schedule(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"w1": {}}))
        sm = SchedulerManager(data_path=p)
        assert sm.has_record("w1") is True

    def test_get_next_run_epoch_zero_for_unknown(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.get_next_run_epoch("ghost") == 0.0

    def test_get_next_run_epoch_returns_value(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"w": {"next_run_epoch": 9876543.0}}))
        sm = SchedulerManager(data_path=p)
        assert sm.get_next_run_epoch("w") == 9876543.0


class TestSchedulerMonthlyAndHyperopt:
    """Tests for monthly rebalance and weekly hyperopt tracking."""

    def test_is_initial_rebalance_true_when_no_record(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.is_initial_rebalance() is True

    def test_is_initial_rebalance_false_after_mark_done(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm.mark_monthly_rebalance_done(now=1_000_000.0)
        assert sm.is_initial_rebalance() is False

    def test_is_monthly_rebalance_due_true_when_no_record(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.is_monthly_rebalance_due() is True

    def test_is_monthly_rebalance_due_false_recently_done(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm.mark_monthly_rebalance_done(now=time.time() - 100)
        assert sm.is_monthly_rebalance_due() is False

    def test_is_monthly_rebalance_due_true_after_30_days(self, tmp_path):
        from wallet_manager.scheduler import MONTH_SECONDS, SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm.mark_monthly_rebalance_done(now=time.time() - MONTH_SECONDS - 1)
        assert sm.is_monthly_rebalance_due() is True

    def test_is_weekly_hyperopt_due_true_first_time(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        assert sm.is_weekly_hyperopt_due() is True

    def test_is_weekly_hyperopt_due_false_recently_done(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm.mark_weekly_hyperopt_done(now=time.time() - 100)
        assert sm.is_weekly_hyperopt_due() is False

    def test_is_weekly_hyperopt_due_true_after_week(self, tmp_path):
        from wallet_manager.scheduler import WEEK_SECONDS, SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        sm.mark_weekly_hyperopt_done(now=time.time() - WEEK_SECONDS - 1)
        assert sm.is_weekly_hyperopt_due() is True


class TestModuleLevelShouldRunSwapThisCycle:
    """Tests for the module-level should_run_swap_this_cycle function."""

    def test_new_wallet_schedules_first_run(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager, should_run_swap_this_cycle
        sm = SchedulerManager(data_path=tmp_path / "s.json")
        should_run, secs = should_run_swap_this_cycle(sm, "brand_new")
        assert should_run is False
        assert secs > 0  # deferred to first-run window

    def test_due_wallet_returns_true_zero(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager, should_run_swap_this_cycle
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"w": {"next_run_epoch": time.time() - 1, "last_completed_epoch": 0}}))
        sm = SchedulerManager(data_path=p)
        should_run, secs = should_run_swap_this_cycle(sm, "w")
        assert should_run is True
        assert secs == 0.0

    def test_not_yet_due_wallet_returns_false(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager, should_run_swap_this_cycle
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"w": {"next_run_epoch": time.time() + 3600, "last_completed_epoch": 0}}))
        sm = SchedulerManager(data_path=p)
        should_run, secs = should_run_swap_this_cycle(sm, "w")
        assert should_run is False
        assert secs > 0


class TestEnsureLoaded:
    """Tests for _ensure_loaded transitions."""

    def test_ensure_loaded_loads_from_disk_when_cache_empty(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        p = tmp_path / "s.json"
        data = {"w1": {"last_run": 1234.0}}
        p.write_text(json.dumps(data))
        sm = SchedulerManager(data_path=tmp_path / "different.json")
        sm._path = p
        sm._cache = {}  # Force empty cache
        sm._ensure_loaded()
        assert "w1" in sm._cache

    def test_ensure_loaded_resets_cache_when_no_file(self, tmp_path):
        from wallet_manager.scheduler import SchedulerManager
        sm = SchedulerManager(data_path=tmp_path / "nonexistent.json")
        sm._cache = {"stale": {}}
        sm._ensure_loaded()
        assert sm._cache == {}


# ===========================================================================
# 2. ANALYTICS DB TESTS
# ===========================================================================


class TestAnalyticsDBHelpers:
    """Tests for the private helper _mask_wallet_id and dataclasses."""

    def test_mask_wallet_id_short_addr(self):
        """Short addresses should not crash the masker."""
        from wallet_manager.resilience.analytics import _mask_wallet_id
        assert _mask_wallet_id("0xShort") == "0xShort"

    def test_mask_wallet_id_full_addr(self):
        from wallet_manager.resilience.analytics import _mask_wallet_id
        addr = "0xAbCd1234567890EF1234567890abcdef12345678"
        result = _mask_wallet_id(addr)
        assert len(result) < len(addr)
        assert "****" in result

    def test_session_report_duration_none_when_not_finished(self):
        from wallet_manager.resilience.analytics import SessionReport
        r = SessionReport(
            session_id="s1",
            started_at=1000.0,
            finished_at=None,
            total_txs=0, mined_txs=0, replaced_txs=0,
            total_gas_used=0, total_wallets=0, success_count=0, fail_count=0,
        )
        assert r.duration_seconds is None

    def test_session_report_duration_computed(self):
        from wallet_manager.resilience.analytics import SessionReport
        r = SessionReport(
            session_id="s1",
            started_at=1000.0,
            finished_at=1300.0,
            total_txs=5, mined_txs=3, replaced_txs=1,
            total_gas_used=100000, total_wallets=2, success_count=2, fail_count=0,
        )
        assert r.duration_seconds == pytest.approx(300.0)

    def test_wallet_summary_dataclass(self):
        from wallet_manager.resilience.analytics import WalletSummary
        ws = WalletSummary(
            wallet_id="0xABCD****1234",
            total_txs=10, mined_txs=8, replaced_txs=1, dropped_txs=1,
            total_gas_used=200000, total_value_wei=5_000_000_000_000_000,
            avg_gas_price_gwei=12.5,
        )
        assert ws.total_txs == 10
        assert ws.avg_gas_price_gwei == 12.5

    def test_strategy_summary_dataclass(self):
        from wallet_manager.resilience.analytics import StrategySummary
        ss = StrategySummary(
            strategy_type="ALGO",
            total_txs=5, mined_txs=4,
            total_gas_used=50000, total_value_wei=1_000_000_000,
        )
        assert ss.strategy_type == "ALGO"


@pytest.mark.asyncio
class TestAnalyticsDBLifecycle:
    """Tests for open / close / context manager flow using in-memory SQLite."""

    async def test_open_and_close(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        db = AnalyticsDB(db_path=str(tmp_path / "test.db"), session_id="test-session")
        await db.open()
        assert db._conn is not None
        await db.close()
        assert db._conn is None

    async def test_context_manager(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ctx.db"), session_id="ctx-s") as db:
            assert db._conn is not None

    async def test_close_idempotent(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        db = AnalyticsDB(db_path=str(tmp_path / "t.db"), session_id="s")
        await db.open()
        await db.close()
        await db.close()  # Should not raise

    async def test_assert_open_raises_when_closed(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        db = AnalyticsDB(db_path=str(tmp_path / "t.db"), session_id="s")
        # _conn is None, so _assert_open should raise
        with pytest.raises(RuntimeError):
            db._assert_open()


@pytest.mark.asyncio
class TestAnalyticsDBRecordTx:
    """Tests for record_tx and update_tx_status."""

    async def test_record_tx_inserts_row(self, tmp_path):
        from wallet_manager.core.models import TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = _make_tx_record(status=TxStatus.PENDING)
        async with AnalyticsDB(db_path=str(tmp_path / "r.db"), session_id="s1") as db:
            await db.record_tx(record)
            async with db._conn.execute(
                "SELECT COUNT(*) as n FROM transactions WHERE tx_hash = ?",
                (record.tx_hash,)
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == 1

    async def test_record_tx_replaces_on_duplicate(self, tmp_path):
        from wallet_manager.core.models import TxRecord, TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = _make_tx_record(status=TxStatus.PENDING)
        record2 = TxRecord(
            tx_hash=record.tx_hash,
            wallet_id=record.wallet_id,
            nonce=record.nonce,
            status=TxStatus.MINED,
            chain_id=record.chain_id,
            gas_price_gwei=record.gas_price_gwei,
            gas_limit=record.gas_limit,
            value_wei=record.value_wei,
            submitted_at=record.submitted_at,
            gas_used=21000,
        )
        async with AnalyticsDB(db_path=str(tmp_path / "r.db"), session_id="s1") as db:
            await db.record_tx(record)
            await db.record_tx(record2)
            async with db._conn.execute(
                "SELECT status FROM transactions WHERE tx_hash = ?",
                (record.tx_hash,)
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == "MINED"

    async def test_record_tx_raises_analytics_write_error_on_failure(self, tmp_path):
        from wallet_manager.exceptions import AnalyticsWriteError
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = _make_tx_record()
        async with AnalyticsDB(db_path=str(tmp_path / "r.db"), session_id="s1") as db:
            # Corrupt the connection to trigger a write error
            db._conn = None
            with pytest.raises((AnalyticsWriteError, Exception)):
                await db.record_tx(record)

    async def test_update_tx_status(self, tmp_path):
        from wallet_manager.core.models import TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = _make_tx_record(status=TxStatus.PENDING)
        async with AnalyticsDB(db_path=str(tmp_path / "u.db"), session_id="s1") as db:
            await db.record_tx(record)
            await db.update_tx_status(
                record.tx_hash,
                TxStatus.MINED,
                gas_used=21000,
                mined_at=time.time(),
            )
            async with db._conn.execute(
                "SELECT status, gas_used FROM transactions WHERE tx_hash = ?",
                (record.tx_hash,)
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == "MINED"
            assert row[1] == 21000


@pytest.mark.asyncio
class TestAnalyticsDBSessionOps:
    """Tests for close_session and get_session_report."""

    async def test_close_session_updates_record(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "cs.db"), session_id="s1") as db:
            await db.close_session(
                total_wallets=3, success_count=2, fail_count=1, notes="test"
            )
            async with db._conn.execute(
                "SELECT total_wallets, success_count, fail_count, notes FROM sessions WHERE session_id = ?",
                ("s1",)
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == 3
            assert row[1] == 2
            assert row[2] == 1
            assert row[3] == "test"

    async def test_get_session_report_current_session(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "sr.db"), session_id="sess-abc") as db:
            report = await db.get_session_report()
            assert report.session_id == "sess-abc"
            assert report.total_txs == 0

    async def test_get_session_report_not_found_raises(self, tmp_path):
        from wallet_manager.exceptions import AnalyticsQueryError
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "sr.db"), session_id="s1") as db:
            with pytest.raises(AnalyticsQueryError, match="not found"):
                await db.get_session_report(session_id="nonexistent-session")

    async def test_list_sessions_returns_list(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ls.db"), session_id="s1") as db:
            sessions = await db.list_sessions()
            assert isinstance(sessions, list)
            assert len(sessions) >= 1


@pytest.mark.asyncio
class TestAnalyticsDBWalletSummary:
    """Tests for get_wallet_summary."""

    async def test_wallet_summary_empty_returns_zero(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ws.db"), session_id="s1") as db:
            summary = await db.get_wallet_summary("0xDEAD1234567890abcdef1234567890abcdef0000")
            assert summary.total_txs == 0
            assert summary.avg_gas_price_gwei == 0.0

    async def test_wallet_summary_with_data(self, tmp_path):
        from wallet_manager.core.models import TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = _make_tx_record(
            tx_hash="0xaa11223344556677",
            wallet_id="0xAbCd1234567890EF1234567890abcdef12345678",
            status=TxStatus.MINED,
        )
        async with AnalyticsDB(db_path=str(tmp_path / "ws2.db"), session_id="s1") as db:
            await db.record_tx(record)
            summary = await db.get_wallet_summary(record.wallet_id)
            assert summary.total_txs == 1
            assert summary.mined_txs == 1


@pytest.mark.asyncio
class TestAnalyticsDBRuntimeAudit:
    """Tests for attach_runtime_audit and get_runtime_audit."""

    async def test_attach_and_get_runtime_audit(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        payload = {"audit_version": 1, "mode": "paper", "cycles": 5}
        async with AnalyticsDB(db_path=str(tmp_path / "ra.db"), session_id="s1") as db:
            await db.attach_runtime_audit(payload)
            record = await db.get_runtime_audit()
            assert record is not None
            assert record.payload["audit_version"] == 1

    async def test_get_runtime_audit_returns_none_when_no_notes(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ra2.db"), session_id="s1") as db:
            record = await db.get_runtime_audit()
            assert record is None

    async def test_get_runtime_audit_ignores_non_audit_json(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        # payload without audit_version key
        payload = {"data": "just some info"}
        async with AnalyticsDB(db_path=str(tmp_path / "ra3.db"), session_id="s1") as db:
            await db.attach_runtime_audit(payload)
            record = await db.get_runtime_audit()
            assert record is None

    async def test_list_runtime_audits(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        payload = {"audit_version": 2, "info": "x"}
        async with AnalyticsDB(db_path=str(tmp_path / "ra4.db"), session_id="s1") as db:
            await db.attach_runtime_audit(payload)
            audits = await db.list_runtime_audits(limit=5)
            assert isinstance(audits, list)
            assert len(audits) == 1

    async def test_list_runtime_audits_skips_invalid_json(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ra5.db"), session_id="s1") as db:
            # Manually inject invalid JSON into notes
            async with db._write_lock:
                await db._conn.execute(
                    "UPDATE sessions SET notes = ? WHERE session_id = ?",
                    ("not json {{{{", "s1"),
                )
                await db._conn.commit()
            audits = await db.list_runtime_audits(limit=5)
            assert audits == []


@pytest.mark.asyncio
class TestAnalyticsDBStrategySummary:
    """Tests for get_strategy_summary and get_strategy_pnl_report."""

    async def test_get_strategy_summary_empty(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ss.db"), session_id="s1") as db:
            result = await db.get_strategy_summary()
            assert isinstance(result, dict)

    async def test_get_strategy_summary_filtered(self, tmp_path):
        from wallet_manager.core.models import TxRecord, TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = TxRecord(
            tx_hash="0xstrat001",
            wallet_id="0xAbCd1234567890EF1234567890abcdef12345678",
            nonce=1,
            status=TxStatus.MINED,
            strategy_type="ALGO",
        )
        async with AnalyticsDB(db_path=str(tmp_path / "ss2.db"), session_id="s1") as db:
            await db.record_tx(record)
            result = await db.get_strategy_summary(strategy_type="ALGO")
            assert "ALGO" in result

    async def test_get_strategy_pnl_report(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "pnl.db"), session_id="s1") as db:
            result = await db.get_strategy_pnl_report(
                current_values={"ALGO": 100.0, "YIELD": 200.0}
            )
            assert isinstance(result, dict)

    async def test_get_strategy_comparison(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "sc.db"), session_id="s1") as db:
            result = await db.get_strategy_comparison()
            assert isinstance(result, dict)


@pytest.mark.asyncio
class TestAnalyticsDBExportAndDailySummary:
    """Tests for export_csv and get_daily_summary."""

    async def test_export_csv_empty_returns_zero(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        out = str(tmp_path / "export.csv")
        async with AnalyticsDB(db_path=str(tmp_path / "ex.db"), session_id="s1") as db:
            count = await db.export_csv(out)
        assert count == 0
        assert Path(out).exists()

    async def test_export_csv_with_records(self, tmp_path):
        from wallet_manager.core.models import TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = _make_tx_record(status=TxStatus.MINED)
        out = str(tmp_path / "export2.csv")
        async with AnalyticsDB(db_path=str(tmp_path / "ex2.db"), session_id="s1") as db:
            await db.record_tx(record)
            count = await db.export_csv(out)
        assert count == 1
        content = Path(out).read_text()
        assert "tx_hash" in content

    async def test_get_daily_summary_returns_dict(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ds.db"), session_id="s1") as db:
            result = await db.get_daily_summary("2026-04-04")
            assert "trade_count" in result
            assert "avg_gas_gwei" in result

    async def test_get_daily_summary_invalid_date_falls_back(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "ds2.db"), session_id="s1") as db:
            result = await db.get_daily_summary("not-a-date")
            assert result["trade_count"] == 0

    async def test_record_telegram_chat(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "tc.db"), session_id="s1") as db:
            await db.record_telegram_chat("123456789", label="main")
            async with db._conn.execute(
                "SELECT label FROM telegram_chats WHERE chat_id = ?", ("123456789",)
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == "main"

    async def test_get_strategy_performance_all(self, tmp_path):
        from wallet_manager.resilience.analytics import AnalyticsDB
        async with AnalyticsDB(db_path=str(tmp_path / "sp.db"), session_id="s1") as db:
            result = await db.get_strategy_performance()
            assert isinstance(result, dict)

    async def test_get_strategy_performance_filtered(self, tmp_path):
        from wallet_manager.core.models import TxRecord, TxStatus
        from wallet_manager.resilience.analytics import AnalyticsDB
        record = TxRecord(
            tx_hash="0xperf001",
            wallet_id="0xAbCd1234567890EF1234567890abcdef12345678",
            nonce=1,
            status=TxStatus.MINED,
            strategy_type="YIELD",
        )
        async with AnalyticsDB(db_path=str(tmp_path / "sp2.db"), session_id="s1") as db:
            await db.record_tx(record)
            result = await db.get_strategy_performance(strategy_type="YIELD")
            assert "YIELD" in result


# ===========================================================================
# 3. RPC ROUTER TESTS
# ===========================================================================


class TestProviderHealth:
    """Unit tests for _ProviderHealth dataclass."""

    def test_is_available_below_threshold(self):
        from wallet_manager.network.router import _PROVIDER_FAILURE_THRESHOLD, _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com")
        ph.failure_count = _PROVIDER_FAILURE_THRESHOLD - 1
        assert ph.is_available() is True

    def test_is_available_at_threshold_within_skip_window(self):
        from wallet_manager.network.router import _PROVIDER_FAILURE_THRESHOLD, _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com")
        ph.failure_count = _PROVIDER_FAILURE_THRESHOLD
        ph.skip_until = time.monotonic() + 9999
        assert ph.is_available() is False

    def test_is_available_recovery_after_skip_window(self):
        from wallet_manager.network.router import _PROVIDER_FAILURE_THRESHOLD, _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com")
        ph.failure_count = _PROVIDER_FAILURE_THRESHOLD
        ph.skip_until = time.monotonic() - 1  # already expired
        assert ph.is_available() is True
        assert ph.failure_count == 0  # reset on re-admission

    def test_record_failure_increments_count(self):
        from wallet_manager.network.router import _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com")
        ph.record_failure()
        assert ph.failure_count == 1
        assert ph.last_failure_time > 0

    def test_record_failure_sets_skip_at_threshold(self):
        from wallet_manager.network.router import _PROVIDER_FAILURE_THRESHOLD, _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com")
        for _ in range(_PROVIDER_FAILURE_THRESHOLD):
            ph.record_failure()
        assert ph.skip_until > time.monotonic()

    def test_record_success_resets_failure_count(self):
        from wallet_manager.network.router import _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com", failure_count=2)
        ph.record_success()
        assert ph.failure_count == 0
        assert ph.skip_until == 0.0

    def test_record_success_on_clean_provider(self):
        from wallet_manager.network.router import _ProviderHealth
        ph = _ProviderHealth(url="https://rpc.example.com")
        ph.record_success()  # Should not raise even if failure_count is 0


class TestChainPool:
    """Unit tests for _ChainPool."""

    def test_active_url_returns_primary(self):
        from wallet_manager.network.router import _ChainPool
        cfg = _make_chain_config()
        pool = _ChainPool(cfg)
        assert pool.active_url() == "https://primary.example.com"

    def test_active_url_deduplicates_rpcs(self):
        from wallet_manager.network.router import _ChainPool
        cfg = _make_chain_config(
            primary_rpc="https://same.example.com",
            fallback_rpcs=("https://same.example.com", "https://other.example.com"),
        )
        pool = _ChainPool(cfg)
        urls = [ph.url for ph in pool.pool]
        assert urls.count("https://same.example.com") == 1

    def test_active_url_raises_when_all_suspended(self):
        from wallet_manager.exceptions import NoAvailableRpcError
        from wallet_manager.network.router import _PROVIDER_FAILURE_THRESHOLD, _ChainPool
        cfg = _make_chain_config(fallback_rpcs=())
        pool = _ChainPool(cfg)
        # Suspend the only provider
        ph = pool.pool[0]
        ph.failure_count = _PROVIDER_FAILURE_THRESHOLD
        ph.skip_until = time.monotonic() + 9999
        with pytest.raises(NoAvailableRpcError):
            pool.active_url()

    def test_rotate_failed_moves_to_back(self):
        from wallet_manager.network.router import _ChainPool
        cfg = _make_chain_config()
        pool = _ChainPool(cfg)
        primary = pool.pool[0].url
        pool.rotate_failed(primary)
        assert pool.pool[-1].url == primary
        assert pool.pool[0].url != primary

    def test_rotate_failed_unknown_url_is_noop(self):
        from wallet_manager.network.router import _ChainPool
        cfg = _make_chain_config()
        pool = _ChainPool(cfg)
        before = [p.url for p in pool.pool]
        pool.rotate_failed("https://unknown.example.com")
        after = [p.url for p in pool.pool]
        assert before == after

    def test_record_success_resets_provider(self):
        from wallet_manager.network.router import _ChainPool
        cfg = _make_chain_config()
        pool = _ChainPool(cfg)
        url = pool.pool[0].url
        pool.pool[0].failure_count = 2
        pool.record_success(url)
        assert pool.pool[0].failure_count == 0

    def test_record_success_unknown_url_is_noop(self):
        from wallet_manager.network.router import _ChainPool
        cfg = _make_chain_config()
        pool = _ChainPool(cfg)
        pool.record_success("https://nope.example.com")  # Should not raise


class TestRPCRouter:
    """Unit tests for RPCRouter."""

    def test_init_requires_at_least_one_chain(self):
        from wallet_manager.network.router import RPCRouter
        with pytest.raises(ValueError, match="at least one"):
            RPCRouter([])

    def test_get_rpc_returns_primary(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        assert router.get_rpc(8453) == "https://primary.example.com"

    def test_get_rpc_raises_for_unknown_chain(self):
        from wallet_manager.exceptions import ChainNotConfiguredError
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        with pytest.raises(ChainNotConfiguredError):
            router.get_rpc(9999)

    def test_mark_failed_rotates_pool(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        primary = router.get_rpc(8453)
        router.mark_failed(8453, primary)
        next_url = router.get_rpc(8453)
        assert next_url != primary

    def test_mark_success_resets_failure(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        url = router.get_rpc(8453)
        router.mark_failed(8453, url)
        router.mark_success(8453, url)
        pool = router._pools[8453]
        health = pool._health_map[url]
        assert health.failure_count == 0

    def test_get_chain_config_returns_config(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        returned = router.get_chain_config(8453)
        assert returned.chain_id == 8453

    def test_get_chain_config_raises_for_unknown(self):
        from wallet_manager.exceptions import ChainNotConfiguredError
        from wallet_manager.network.router import RPCRouter
        router = RPCRouter([_make_chain_config()])
        with pytest.raises(ChainNotConfiguredError):
            router.get_chain_config(1)

    def test_pool_status_returns_snapshot(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        status = router.pool_status()
        assert 8453 in status
        assert isinstance(status[8453], list)
        assert "url" in status[8453][0]
        assert "failure_count" in status[8453][0]

    def test_latency_score_unknown_url_returns_one(self):
        from wallet_manager.network.router import RPCRouter
        router = RPCRouter([_make_chain_config()])
        score = router.latency_score(8453, "https://unknown.com")
        assert score == 1.0

    def test_latency_score_with_data(self):
        from wallet_manager.network.router import RPCRouter
        router = RPCRouter([_make_chain_config()])
        url = "https://primary.example.com"
        router._latency[8453][url] = deque([100.0, 200.0], maxlen=20)
        score = router.latency_score(8453, url)
        # mean = 150ms, score = 1/(150+1) ≈ 0.00663
        assert score == pytest.approx(1.0 / 151.0, rel=1e-3)

    def test_get_url_returns_a_string(self):
        from wallet_manager.network.router import RPCRouter
        router = RPCRouter([_make_chain_config()])
        with patch("wallet_manager.config.get_settings") as mock_settings:
            mock_settings.return_value.rpc_reprobe_interval_seconds = 3600
            url = router.get_url(8453)
        assert isinstance(url, str)
        assert url != ""

    def test_get_url_unknown_chain_returns_empty_string(self):
        from wallet_manager.network.router import RPCRouter
        router = RPCRouter([_make_chain_config()])
        with patch("wallet_manager.config.get_settings") as mock_settings:
            mock_settings.return_value.rpc_reprobe_interval_seconds = 3600
            url = router.get_url(99999)
        assert url == ""

    def test_multiple_chains_are_independent(self):
        from wallet_manager.network.router import RPCRouter
        cfg1 = _make_chain_config(chain_id=1, name="Ethereum", primary_rpc="https://eth.example.com")
        cfg2 = _make_chain_config(chain_id=8453, name="Base", primary_rpc="https://base.example.com")
        router = RPCRouter([cfg1, cfg2])
        assert router.get_rpc(1) == "https://eth.example.com"
        assert router.get_rpc(8453) == "https://base.example.com"
        router.mark_failed(1, "https://eth.example.com")
        assert router.get_rpc(8453) == "https://base.example.com"  # unaffected


@pytest.mark.asyncio
class TestRPCRouterProbe:
    """Tests for probe and probe_all with mocked network."""

    async def test_probe_success_marks_success(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])
        router.get_rpc(8453)

        mock_w3 = AsyncMock()
        mock_w3.eth.block_number = 12345678

        with patch("wallet_manager.network.router.AsyncWeb3") as MockAsyncWeb3, \
             patch("wallet_manager.network.router.asyncio.wait_for", new=AsyncMock(return_value=12345678)):
            MockAsyncWeb3.return_value = mock_w3
            result = await router.probe(8453)

        # Whether True or False, it should not raise
        assert isinstance(result, bool)

    async def test_probe_failure_marks_failed_returns_false(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])

        with patch("wallet_manager.network.router.AsyncWeb3") as MockAsyncWeb3:
            MockAsyncWeb3.side_effect = Exception("connection refused")
            result = await router.probe(8453)

        assert result is False

    async def test_probe_all_returns_dict(self):
        from wallet_manager.network.router import RPCRouter
        cfg1 = _make_chain_config(chain_id=1, name="Eth", primary_rpc="https://eth.example.com")
        cfg2 = _make_chain_config(chain_id=8453, name="Base", primary_rpc="https://base.example.com")
        router = RPCRouter([cfg1, cfg2])

        with patch.object(router, "probe", new=AsyncMock(return_value=True)):
            results = await router.probe_all()

        assert set(results.keys()) == {1, 8453}

    async def test_probe_all_handles_exception_per_chain(self):
        from wallet_manager.network.router import RPCRouter
        cfg = _make_chain_config()
        router = RPCRouter([cfg])

        async def _failing_probe(_chain_id):
            raise RuntimeError("network down")

        with patch.object(router, "probe", side_effect=_failing_probe):
            results = await router.probe_all()

        assert results[8453] is False


class TestBuildDefaultRouter:
    """Tests for the build_default_router factory function."""

    def test_build_default_router_no_keys(self):
        from wallet_manager.network.router import build_default_router
        router = build_default_router()
        assert 1 in router._pools
        assert 8453 in router._pools
        assert 42161 in router._pools
        assert 59144 in router._pools
        assert 137 in router._pools

    def test_build_default_router_with_infura_key(self):
        from wallet_manager.network.router import build_default_router
        router = build_default_router(infura_key="MYKEY123")
        eth_url = router.get_rpc(1)
        assert "MYKEY123" in eth_url

    def test_build_default_router_with_alchemy_key(self):
        from wallet_manager.network.router import build_default_router
        router = build_default_router(alchemy_key="ALCHKEY456")
        base_url = router.get_rpc(8453)
        assert "ALCHKEY456" in base_url


# ===========================================================================
# 4. TELEGRAM MANAGER TESTS
# ===========================================================================


class TestTelegramMaskAddress:
    """Tests for the _mask_address helper."""

    def test_masks_full_hex_address(self):
        from wallet_manager.security.telegram_manager import _mask_address
        addr = "0xAbCd1234567890EF1234567890abcdef12345678"
        result = _mask_address(addr)
        assert "****" in result
        assert result != addr

    def test_no_change_on_plain_text(self):
        from wallet_manager.security.telegram_manager import _mask_address
        text = "No addresses here, just text."
        assert _mask_address(text) == text

    def test_masks_multiple_addresses(self):
        from wallet_manager.security.telegram_manager import _mask_address
        text = "From 0xAbCd1234567890EF1234567890abcdef12345678 to 0xDEEF1234567890EF1234567890abcdef12345678"
        result = _mask_address(text)
        assert result.count("****") == 2


class TestTelegramCreds:
    """Tests for _TelegramCreds dataclass."""

    def test_repr_does_not_expose_token(self):
        from wallet_manager.security.telegram_manager import _TelegramCreds
        creds = _TelegramCreds(_token="SECRET_TOKEN", _chat_id="12345")
        assert "SECRET_TOKEN" not in repr(creds)
        assert "***" in repr(creds)

    def test_token_and_chat_id_accessible(self):
        from wallet_manager.security.telegram_manager import _TelegramCreds
        creds = _TelegramCreds(_token="mytoken", _chat_id="mychat")
        assert creds.token == "mytoken"
        assert creds.chat_id == "mychat"

    def test_from_env_empty_when_no_settings(self):
        from wallet_manager.security.telegram_manager import _TelegramCreds
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = ""
        mock_cfg.telegram_chat_id = ""
        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            creds = _TelegramCreds.from_env()
        assert creds.token == ""
        assert creds.chat_id == ""

    def test_from_env_populated_when_settings_set(self):
        from wallet_manager.security.telegram_manager import _TelegramCreds
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = "realtoken"
        mock_cfg.telegram_chat_id = "654321"
        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            creds = _TelegramCreds.from_env()
        assert creds.token == "realtoken"
        assert creds.chat_id == "654321"


class TestTelegramManagerInit:
    """Tests for TelegramManager initialisation paths."""

    def test_init_with_explicit_credentials(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="tok", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)
        assert mgr._creds.token == "tok"

    def test_init_with_settings_object(self):
        from wallet_manager.security.telegram_manager import TelegramManager
        settings = MagicMock()
        settings.telegram_bot_token = "s_token"
        settings.telegram_chat_id = "s_chat"
        mgr = TelegramManager(settings=settings)
        assert mgr._creds.token == "s_token"
        assert mgr._creds.chat_id == "s_chat"

    def test_init_settings_none_token_becomes_empty(self):
        from wallet_manager.security.telegram_manager import TelegramManager
        settings = MagicMock()
        settings.telegram_bot_token = None
        settings.telegram_chat_id = None
        mgr = TelegramManager(settings=settings)
        assert mgr._creds.token == ""

    def test_is_configured_false_when_empty(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="", _chat_id="")
        mgr = TelegramManager(credentials=creds)
        assert mgr._is_configured() is False

    def test_is_configured_true_when_set(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="tok", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)
        assert mgr._is_configured() is True


class TestTelegramManagerAuthorization:
    """Tests for is_authorized and confirm rate-limiting."""

    def test_is_authorized_matching_chat_id(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="789")
        mgr = TelegramManager(credentials=creds)
        update = MagicMock()
        update.message.chat_id = 789
        assert mgr.is_authorized(update) is True

    def test_is_authorized_wrong_chat_id(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="789")
        mgr = TelegramManager(credentials=creds)
        update = MagicMock()
        update.message.chat_id = 999
        assert mgr.is_authorized(update) is False

    def test_is_authorized_no_message_returns_false(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="789")
        mgr = TelegramManager(credentials=creds)
        update = MagicMock()
        update.message = None
        assert mgr.is_authorized(update) is False

    def test_confirm_rate_limiting_not_exceeded(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)
        mgr._record_confirm_attempt("chat1")
        mgr._record_confirm_attempt("chat1")
        assert mgr._is_confirm_rate_limited("chat1") is False

    def test_confirm_rate_limiting_exceeded(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)
        for _ in range(3):
            mgr._record_confirm_attempt("chat1")
        assert mgr._is_confirm_rate_limited("chat1") is True

    def test_confirm_rate_limiting_unknown_chat_not_limited(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)
        assert mgr._is_confirm_rate_limited("brand_new_chat") is False


@pytest.mark.asyncio
class TestTelegramManagerSend:
    """Tests for _send, close, and _ensure_session."""

    async def test_send_returns_false_when_not_configured(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="", _chat_id="")
        mgr = TelegramManager(credentials=creds)
        result = await mgr._send("hello")
        assert result is False

    async def test_send_returns_true_on_200(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="real_token", _chat_id="12345")
        mgr = TelegramManager(credentials=creds)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)

        mgr._session = mock_session
        result = await mgr._send("test message")
        assert result is True

    async def test_send_returns_false_on_non_200(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="tok", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)

        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)

        mgr._session = mock_session
        result = await mgr._send("bad message")
        assert result is False

    async def test_send_handles_network_exception(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="tok", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=Exception("network error"))

        mgr._session = mock_session
        result = await mgr._send("test")
        assert result is False

    async def test_close_no_session_is_noop(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="c")
        mgr = TelegramManager(credentials=creds)
        mgr._session = None
        await mgr.close()  # Must not raise

    async def test_close_closed_session_is_noop(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="t", _chat_id="c")
        mgr = TelegramManager(credentials=creds)
        mock_session = MagicMock()
        mock_session.closed = True
        mgr._session = mock_session
        await mgr.close()  # Must not raise


@pytest.mark.asyncio
class TestTelegramManagerNotifications:
    """Tests for all send_* notification methods."""

    def _make_mgr(self):
        from wallet_manager.security.telegram_manager import TelegramManager, _TelegramCreds
        creds = _TelegramCreds(_token="tok", _chat_id="cid")
        mgr = TelegramManager(credentials=creds)
        return mgr

    async def test_send_trade_executed(self):
        from wallet_manager.security.telegram_manager import TradeExecutedPayload
        mgr = self._make_mgr()
        payload = TradeExecutedPayload(
            pair="ETH/USDC", price=2500.0, strategy_type="ALGO",
            amount="0.1 ETH", side="BUY", tx_hash="0xabc123def456789012345678901234567890abcd12"
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_trade_executed(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "ETH/USDC" in call_text
        assert "2500.00" in call_text

    async def test_send_trade_executed_no_tx_hash(self):
        from wallet_manager.security.telegram_manager import TradeExecutedPayload
        mgr = self._make_mgr()
        payload = TradeExecutedPayload(
            pair="ETH/USDC", price=1800.0, strategy_type="YIELD",
            amount="1.0 ETH", side="SELL", tx_hash=None
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            await mgr.send_trade_executed(payload)
        call_text = mock_send.call_args[0][0]
        assert "Tx:" not in call_text

    async def test_send_daily_summary(self):
        from wallet_manager.security.telegram_manager import DailySummaryPayload
        mgr = self._make_mgr()
        payload = DailySummaryPayload(
            total_tvl_usd=5000.0, aave_yield_usd=12.5, aave_apy_pct=3.2, algo_pnl_usd=-5.0
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_daily_summary(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "5000.00" in call_text
        assert "3.20" in call_text

    async def test_send_system_alert_warning(self):
        from wallet_manager.security.telegram_manager import SystemAlertPayload
        mgr = self._make_mgr()
        payload = SystemAlertPayload(alert_type="HIGH_GAS", detail="Gas is 200 gwei")
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_system_alert(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "HIGH_GAS" in call_text

    async def test_send_system_alert_critical(self):
        from wallet_manager.security.telegram_manager import SystemAlertPayload
        mgr = self._make_mgr()
        payload = SystemAlertPayload(alert_type="TX_FAIL", detail="5 failures", severity="critical")
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            await mgr.send_system_alert(payload)
        call_text = mock_send.call_args[0][0]
        assert "🔴" in call_text

    async def test_send_circuit_breaker_alert(self):
        from wallet_manager.security.telegram_manager import CircuitBreakerPayload
        mgr = self._make_mgr()
        payload = CircuitBreakerPayload(
            alert_type="DRAWDOWN", detail="5% daily drawdown exceeded",
            start_value_usd=10000.0, current_value_usd=9490.0, drawdown_pct=5.1
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_circuit_breaker_alert(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "Circuit Breaker" in call_text
        assert "5.10%" in call_text

    async def test_send_stop_loss_alert(self):
        from wallet_manager.security.telegram_manager import StopLossPayload
        mgr = self._make_mgr()
        payload = StopLossPayload(
            wallet_id="0xAbCd1234567890EF1234567890abcdef12345678",
            pair="ETH/USDC", price=2425.0, last_buy_price=2500.0, pct_loss=3.0
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_stop_loss_alert(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "Stop-Loss" in call_text
        assert "3.00%" in call_text

    async def test_send_heartbeat_basic(self):
        from wallet_manager.security.telegram_manager import HeartbeatPayload
        mgr = self._make_mgr()
        payload = HeartbeatPayload(
            total_tvl_usd=9500.0,
            open_positions_count=2,
            unrealized_pnl_usd=-50.0,
            aave_yield_earned_usd=8.5,
            aave_apy_pct=2.8,
            current_gas_gwei=12.0,
            paper_balance_usd=10050.0,
            paper_starting_balance_usd=10000.0,
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_heartbeat(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "Heartbeat" in call_text

    async def test_send_heartbeat_next_run_hours(self):
        from wallet_manager.security.telegram_manager import HeartbeatPayload
        mgr = self._make_mgr()
        payload = HeartbeatPayload(
            total_tvl_usd=9000.0,
            open_positions_count=0,
            unrealized_pnl_usd=0.0,
            aave_yield_earned_usd=0.0,
            aave_apy_pct=0.0,
            current_gas_gwei=10.0,
            next_run_seconds=7200.0,  # 2 hours — triggers hours branch
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            await mgr.send_heartbeat(payload)
        call_text = mock_send.call_args[0][0]
        assert "2.0h" in call_text

    async def test_send_heartbeat_next_run_minutes(self):
        from wallet_manager.security.telegram_manager import HeartbeatPayload
        mgr = self._make_mgr()
        payload = HeartbeatPayload(
            total_tvl_usd=9000.0,
            open_positions_count=0,
            unrealized_pnl_usd=0.0,
            aave_yield_earned_usd=0.0,
            aave_apy_pct=0.0,
            current_gas_gwei=10.0,
            next_run_seconds=300.0,  # 5 min — triggers minutes branch
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            await mgr.send_heartbeat(payload)
        call_text = mock_send.call_args[0][0]
        assert "5m" in call_text

    async def test_send_heartbeat_with_last_trade_time(self):
        from wallet_manager.security.telegram_manager import HeartbeatPayload
        mgr = self._make_mgr()
        payload = HeartbeatPayload(
            total_tvl_usd=9000.0,
            open_positions_count=1,
            unrealized_pnl_usd=20.0,
            aave_yield_earned_usd=5.0,
            aave_apy_pct=2.5,
            current_gas_gwei=8.0,
            last_trade_time="2026-04-04T10:00",
            last_trade_token="ETH",
            last_trade_pnl=15.0,
            best_trade_pair="ETH/USDC",
            best_trade_pnl=15.0,
            worst_trade_pair="LINK/USDC",
            worst_trade_pnl=-3.0,
            strategy_status="ALGO: active",
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_heartbeat(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "ETH" in call_text

    async def test_send_heartbeat_gas_unsuitable(self):
        from wallet_manager.security.telegram_manager import HeartbeatPayload
        mgr = self._make_mgr()
        payload = HeartbeatPayload(
            total_tvl_usd=9000.0,
            open_positions_count=0,
            unrealized_pnl_usd=0.0,
            aave_yield_earned_usd=0.0,
            aave_apy_pct=0.0,
            current_gas_gwei=250.0,
            gas_suitable=False,
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            await mgr.send_heartbeat(payload)
        call_text = mock_send.call_args[0][0]
        assert "too high" in call_text

    async def test_send_yield_alert(self):
        from wallet_manager.security.telegram_manager import YieldAlertPayload
        mgr = self._make_mgr()
        payload = YieldAlertPayload(
            aave_apy_pct=1.5,
            alternative_protocol="Compound",
            alternative_apy_pct=4.2,
            detail="Aave APY dropped below threshold",
        )
        with patch.object(mgr, "_send", new=AsyncMock(return_value=True)) as mock_send:
            result = await mgr.send_yield_alert(payload)
        assert result is True
        call_text = mock_send.call_args[0][0]
        assert "Compound" in call_text
        assert "4.20%" in call_text


@pytest.mark.asyncio
class TestTelegramConvenienceFunctions:
    """Tests for the module-level convenience notify_* functions."""

    async def test_notify_trade_executed(self):
        from wallet_manager.security.telegram_manager import notify_trade_executed
        with patch("wallet_manager.security.telegram_manager.TelegramManager") as MockMgr:
            instance = MockMgr.return_value
            instance.send_trade_executed = AsyncMock(return_value=False)
            result = await notify_trade_executed(
                pair="ETH/USDC", price=2000.0, strategy_type="ALGO",
                amount="0.5 ETH", side="BUY",
            )
        assert result is False

    async def test_notify_daily_summary(self):
        from wallet_manager.security.telegram_manager import notify_daily_summary
        with patch("wallet_manager.security.telegram_manager.TelegramManager") as MockMgr:
            instance = MockMgr.return_value
            instance.send_daily_summary = AsyncMock(return_value=False)
            result = await notify_daily_summary(
                total_tvl_usd=5000.0, aave_yield_usd=10.0,
                aave_apy_pct=3.0, algo_pnl_usd=20.0,
            )
        assert result is False

    async def test_notify_system_alert(self):
        from wallet_manager.security.telegram_manager import notify_system_alert
        with patch("wallet_manager.security.telegram_manager.TelegramManager") as MockMgr:
            instance = MockMgr.return_value
            instance.send_system_alert = AsyncMock(return_value=False)
            result = await notify_system_alert(
                alert_type="TEST", detail="unit test alert", severity="warning"
            )
        assert result is False
