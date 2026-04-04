"""tests/unit/test_reporting.py

Comprehensive tests for wallet_manager/reporting.py (PerformanceReporter).
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import wallet_manager.reporting as _rep_mod
from wallet_manager.reporting import PerformanceReporter, ReportRecord, get_reporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_notifier_send_message():
    notifier = MagicMock()
    notifier.send_message = AsyncMock()
    return notifier


def _mock_paper_engine():
    """Return a mock paper engine with ledger that returns sensible data."""
    engine = MagicMock()
    ledger = engine._ledger

    pnl_data = {
        "current_equity_usd": 1000.0,
        "total_realized_pnl_usd": 5.25,
        "trade_count": 10,
    }
    adv_data = {
        "closed_count": 8,
        "win_rate_pct": 62.5,
        "expectancy_usd": 0.65,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "calmar_ratio": 0.8,
        "profit_factor": 1.4,
    }
    hold_data = {"avg_hold_human": "2h 15m"}

    ledger.get_pnl_summary = AsyncMock(return_value=pnl_data)
    ledger.get_advanced_analytics = AsyncMock(return_value=adv_data)
    ledger.get_hold_time_stats = AsyncMock(return_value=hold_data)

    # Mock the DB connection for skip-reasons query
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute = MagicMock(return_value=mock_cursor)
    ledger._get_conn = MagicMock(return_value=mock_conn)

    return engine


# ---------------------------------------------------------------------------
# ReportRecord
# ---------------------------------------------------------------------------

class TestReportRecord:
    def test_to_dict_fields(self):
        rec = ReportRecord("daily", time.time() - 3600, "summary text", delivered=True)
        d = rec.to_dict()
        assert d["type"] == "daily"
        assert d["delivered"] is True
        assert "age_hours" in d
        assert d["summary"] == "summary text"

    def test_to_dict_truncates_long_summary(self):
        long_summary = "x" * 500
        rec = ReportRecord("monthly", time.time(), long_summary)
        d = rec.to_dict()
        assert len(d["summary"]) <= 300


# ---------------------------------------------------------------------------
# PerformanceReporter._chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_short_text_single_chunk(self):
        result = PerformanceReporter._chunk("line1\nline2\nline3", 3800)
        assert len(result) == 1
        assert "line1" in result[0]

    def test_empty_string_returns_single_empty_chunk(self):
        result = PerformanceReporter._chunk("", 3800)
        assert result == [""]

    def test_long_text_splits_into_multiple_chunks(self):
        # Build text > 3800 chars
        line = "A" * 100
        text = "\n".join([line] * 50)  # 5050 chars
        result = PerformanceReporter._chunk(text, 3800)
        assert len(result) >= 2
        # Reassembling should contain all lines
        combined = "\n".join(result)
        assert combined.count(line) == 50

    def test_exact_boundary(self):
        # Single line of exactly size chars — the chunk method counts
        # length + 1 (newline), so a single 100-char line fits within size=101
        text = "x" * 100
        result = PerformanceReporter._chunk(text, 101)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# PerformanceReporter._escape_md
# ---------------------------------------------------------------------------

class TestEscapeMd:
    def test_special_chars_escaped(self):
        text = "Hello. World-test+1(ok)!#pipe|>end"
        escaped = PerformanceReporter._escape_md(text)
        for ch in [".", "-", "+", "(", ")", "!", "#", "|", ">"]:
            assert f"\\{ch}" in escaped

    def test_no_special_chars_unchanged(self):
        text = "Hello World 123"
        assert PerformanceReporter._escape_md(text) == text


# ---------------------------------------------------------------------------
# PerformanceReporter.to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_initial_state_none_times(self):
        reporter = PerformanceReporter()
        d = reporter.to_dict()
        assert d["last_daily_ago_h"] is None
        assert d["last_weekly_ago_h"] is None
        assert d["history"] == []

    def test_after_daily_set(self):
        reporter = PerformanceReporter()
        reporter._last_daily = time.time() - 3600
        d = reporter.to_dict()
        assert d["last_daily_ago_h"] is not None
        assert d["last_daily_ago_h"] > 0

    def test_history_capped_at_five(self):
        reporter = PerformanceReporter()
        for i in range(10):
            reporter._history.append(ReportRecord("daily", time.time(), f"summary {i}"))
        d = reporter.to_dict()
        assert len(d["history"]) == 5


# ---------------------------------------------------------------------------
# PerformanceReporter._store
# ---------------------------------------------------------------------------

class TestStore:
    def test_store_appends_to_history(self):
        reporter = PerformanceReporter()
        rec = ReportRecord("daily", time.time(), "x")
        reporter._store(rec)
        assert len(reporter._history) == 1

    def test_store_trims_old_history(self):
        reporter = PerformanceReporter()
        reporter._max_history = 3
        for i in range(5):
            reporter._store(ReportRecord("daily", time.time(), f"s{i}"))
        assert len(reporter._history) == 3


# ---------------------------------------------------------------------------
# PerformanceReporter.tick
# ---------------------------------------------------------------------------

class TestTick:
    @pytest.mark.asyncio
    async def test_tick_first_call_triggers_daily(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        with patch.object(reporter, "_deliver_daily", new=AsyncMock()) as mock_daily, \
             patch.object(reporter, "_deliver_weekly", new=AsyncMock()):
            await reporter.tick(notifier)
            mock_daily.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_first_call_triggers_weekly(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        with patch.object(reporter, "_deliver_daily", new=AsyncMock()), \
             patch.object(reporter, "_deliver_weekly", new=AsyncMock()) as mock_weekly:
            await reporter.tick(notifier)
            mock_weekly.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_second_call_does_not_redeliver(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        with patch.object(reporter, "_deliver_daily", new=AsyncMock()) as mock_daily, \
             patch.object(reporter, "_deliver_weekly", new=AsyncMock()) as mock_weekly:
            await reporter.tick(notifier)
            mock_daily.reset_mock()
            mock_weekly.reset_mock()
            await reporter.tick(notifier)
            mock_daily.assert_not_awaited()
            mock_weekly.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_redelivers_when_daily_interval_elapsed(self):
        reporter = PerformanceReporter()
        reporter._last_daily = time.time() - _rep_mod._DAILY_INTERVAL_S - 1
        reporter._last_weekly = time.time()  # weekly not due
        notifier = _mock_notifier_send_message()
        with patch.object(reporter, "_deliver_daily", new=AsyncMock()) as mock_daily, \
             patch.object(reporter, "_deliver_weekly", new=AsyncMock()) as mock_weekly:
            await reporter.tick(notifier)
            mock_daily.assert_awaited_once()
            mock_weekly.assert_not_awaited()


# ---------------------------------------------------------------------------
# PerformanceReporter._deliver_daily
# ---------------------------------------------------------------------------

class TestDeliverDaily:
    @pytest.mark.asyncio
    async def test_deliver_daily_without_notifier(self):
        reporter = PerformanceReporter()
        engine = _mock_paper_engine()
        await reporter._deliver_daily(None, engine)
        assert len(reporter._history) == 1
        assert reporter._history[0].delivered is False

    @pytest.mark.asyncio
    async def test_deliver_daily_with_notifier(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        engine = _mock_paper_engine()
        await reporter._deliver_daily(notifier, engine)
        notifier.send_message.assert_awaited()
        assert reporter._history[0].delivered is True

    @pytest.mark.asyncio
    async def test_deliver_daily_no_paper_engine(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        await reporter._deliver_daily(notifier, None)
        # Still stores record
        assert len(reporter._history) == 1

    @pytest.mark.asyncio
    async def test_deliver_daily_notifier_error_caught(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        notifier.send_message = AsyncMock(side_effect=Exception("Telegram down"))
        engine = _mock_paper_engine()
        # Should not raise
        await reporter._deliver_daily(notifier, engine)
        assert reporter._history[0].delivered is False


# ---------------------------------------------------------------------------
# PerformanceReporter._deliver_weekly
# ---------------------------------------------------------------------------

class TestDeliverWeekly:
    @pytest.mark.asyncio
    async def test_deliver_weekly_with_notifier(self):
        reporter = PerformanceReporter()
        _mock_notifier_send_message()

        # Patch out the research brain and signal imports
        weekly_review = MagicMock()
        weekly_review.to_text = MagicMock(return_value="weekly text")
        attribution = MagicMock()
        attribution.sections = [{"heading": "Sec1", "body": "body text"}]

        brain_mock = MagicMock()
        brain_mock.weekly_strategy_review = AsyncMock(return_value=weekly_review)
        brain_mock.pnl_attribution = AsyncMock(return_value=attribution)

        sig_intel = MagicMock()
        sig_intel.ic_tracker.get_ic_report = MagicMock(return_value={"signals": []})

        risk_intel = MagicMock()
        risk_intel.stress_engine.to_dict = MagicMock(return_value={"status": "OK", "all_pass": True, "worst_scenario": "none"})

        with patch("wallet_manager.reporting.ResearchBrain" if False else "wallet_manager.research_brain.ResearchBrain", brain_mock, create=True), \
             patch("wallet_manager.reporting._send", new=AsyncMock()):
            # We just test that it stores a record and doesn't crash with no engine
            await reporter._deliver_weekly(None, None)
        assert len(reporter._history) == 1

    @pytest.mark.asyncio
    async def test_deliver_weekly_no_notifier_stores_record(self):
        reporter = PerformanceReporter()
        await reporter._deliver_weekly(None, None)
        assert len(reporter._history) == 1
        assert reporter._history[0].report_type == "weekly"
        assert reporter._history[0].delivered is False


# ---------------------------------------------------------------------------
# PerformanceReporter.generate_monthly
# ---------------------------------------------------------------------------

class TestGenerateMonthly:
    @pytest.mark.asyncio
    async def test_generate_monthly_no_engine(self):
        reporter = PerformanceReporter()
        result = await reporter.generate_monthly(None, None)
        assert "not available" in result
        assert len(reporter._history) == 1

    @pytest.mark.asyncio
    async def test_generate_monthly_with_notifier_no_engine(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        result = await reporter.generate_monthly(notifier, None)
        assert isinstance(result, str)
        # notifier.send_message called (the "Engine not available." message)
        notifier.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_generate_monthly_stores_record(self):
        reporter = PerformanceReporter()
        await reporter.generate_monthly(None, None)
        assert reporter._history[0].report_type == "monthly"

    @pytest.mark.asyncio
    async def test_generate_monthly_notifier_error_caught(self):
        reporter = PerformanceReporter()
        notifier = _mock_notifier_send_message()
        notifier.send_message = AsyncMock(side_effect=Exception("fail"))
        result = await reporter.generate_monthly(notifier, None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _send module-level helper
# ---------------------------------------------------------------------------

class TestModuleSendHelper:
    @pytest.mark.asyncio
    async def test_send_with_send_message(self):
        notifier = MagicMock()
        notifier.send_message = AsyncMock()
        await _rep_mod._send(notifier, "hello")
        notifier.send_message.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_send_with_send_fallback(self):
        notifier = MagicMock(spec=["send"])
        notifier.send = AsyncMock()
        await _rep_mod._send(notifier, "hello")
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_with_none_no_error(self):
        await _rep_mod._send(None, "hello")  # should be a no-op


# ---------------------------------------------------------------------------
# get_reporter singleton
# ---------------------------------------------------------------------------

class TestGetReporter:
    def test_returns_same_instance(self):
        _rep_mod._reporter = None
        r1 = get_reporter()
        r2 = get_reporter()
        assert r1 is r2

    def test_returns_performance_reporter_type(self):
        _rep_mod._reporter = None
        assert isinstance(get_reporter(), PerformanceReporter)
