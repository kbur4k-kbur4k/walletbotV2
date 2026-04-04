"""tests/unit/test_tick_collector_coverage.py

Fast unit tests for wallet_manager/tick_collector.py
Uses mocked aiosqlite to avoid actual DB operations.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.tick_collector import TickCollector, get_tick_collector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collector(db_path=None):
    if db_path is None:
        db_path = Path("/tmp/test_ticks.db")
    return TickCollector(db_path)


def _mock_db_ctx(rows=None, rowcount=0):
    """Return a mock async context manager for aiosqlite.connect."""
    if rows is None:
        rows = []

    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=rows)
    mock_cursor.fetchone = AsyncMock(return_value=(len(rows), None, None) if not rows else (1, 1000000000, 2000000000))
    mock_cursor.rowcount = rowcount
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=mock_cursor)
    mock_db.executemany = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.row_factory = None
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    return mock_db


# ---------------------------------------------------------------------------
# TestTickCollectorInit
# ---------------------------------------------------------------------------

class TestTickCollectorInit:
    def test_default_db_path(self):
        tc = TickCollector()
        assert "ticks.db" in str(tc._db_path)

    def test_custom_db_path(self):
        p = Path("/custom/path/ticks.db")
        tc = TickCollector(p)
        assert tc._db_path == p

    def test_initial_state(self):
        tc = _make_collector()
        assert tc._initialized is False
        assert tc._tick_count == 0
        assert tc._last_price == 0.0
        assert tc._pending == []
        assert tc._flush_interval == 50


# ---------------------------------------------------------------------------
# TestRecordTick
# ---------------------------------------------------------------------------

class TestRecordTick:
    def test_record_positive_price(self):
        tc = _make_collector()
        tc.record_tick(3000.0)
        assert tc._tick_count == 1
        assert tc._last_price == 3000.0
        assert len(tc._pending) == 1

    def test_record_zero_price_ignored(self):
        tc = _make_collector()
        tc.record_tick(0.0)
        assert tc._tick_count == 0

    def test_record_negative_price_ignored(self):
        tc = _make_collector()
        tc.record_tick(-100.0)
        assert tc._tick_count == 0

    def test_uptick_direction(self):
        tc = _make_collector()
        tc.record_tick(3000.0)
        tc.record_tick(3100.0)
        # second tick should be direction=1 (uptick)
        _ts, _price, direction = tc._pending[1]
        assert direction == 1

    def test_downtick_direction(self):
        tc = _make_collector()
        tc.record_tick(3000.0)
        tc.record_tick(2900.0)
        _ts, _price, direction = tc._pending[1]
        assert direction == -1

    def test_unchanged_direction(self):
        tc = _make_collector()
        tc.record_tick(3000.0)
        tc.record_tick(3000.0)
        _ts, _price, direction = tc._pending[1]
        assert direction == 0

    def test_first_tick_direction_zero(self):
        tc = _make_collector()
        tc.record_tick(3000.0)
        _ts, _price, direction = tc._pending[0]
        assert direction == 0  # first tick, no last_price

    def test_multiple_ticks_count(self):
        tc = _make_collector()
        for i in range(10):
            tc.record_tick(float(3000 + i))
        assert tc._tick_count == 10
        assert len(tc._pending) == 10

    def test_spread_bps_accepted(self):
        tc = _make_collector()
        with patch("wallet_manager.tick_collector.get_tick_collector", side_effect=Exception):
            pass
        # Just ensure record_tick accepts spread_bps param
        tc.record_tick(3000.0, spread_bps=5.0)
        assert tc._tick_count == 1


# ---------------------------------------------------------------------------
# TestGetOfiFeaturesMethod
# ---------------------------------------------------------------------------

class TestGetOfiFeatures:
    def test_returns_dict_with_keys(self):
        tc = _make_collector()
        with patch("wallet_manager.strategies.order_flow.get_order_flow_imbalance") as mock_ofi:
            mock_ofi.return_value.get_features.return_value = {
                "ofi": 0.1, "ofi_momentum": 0.05, "spread_signal": 0.0, "ofi_composite": 0.08
            }
            result = tc.get_ofi_features()
        assert "ofi" in result

    def test_returns_defaults_on_exception(self):
        tc = _make_collector()
        with patch("wallet_manager.tick_collector.get_tick_collector", side_effect=ImportError):
            result = tc.get_ofi_features()
        # Should return default dict on exception
        assert isinstance(result, dict)
        assert "ofi" in result


# ---------------------------------------------------------------------------
# TestInitialize
# ---------------------------------------------------------------------------

class TestInitialize:
    @pytest.mark.asyncio
    async def test_sets_initialized_flag(self):
        tc = _make_collector()
        mock_db = _mock_db_ctx()
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "_ensure_dir"):
                await tc.initialize()
        assert tc._initialized is True

    @pytest.mark.asyncio
    async def test_ensure_dir_called(self):
        tc = _make_collector()
        mock_db = _mock_db_ctx()
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "_ensure_dir") as mock_dir:
                await tc.initialize()
        mock_dir.assert_called_once()


# ---------------------------------------------------------------------------
# TestFlush
# ---------------------------------------------------------------------------

class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_empty_pending_returns_zero(self):
        tc = _make_collector()
        tc._initialized = True
        result = await tc.flush()
        assert result == 0

    @pytest.mark.asyncio
    async def test_flush_writes_pending_ticks(self):
        tc = _make_collector()
        tc._initialized = True
        tc._pending = [(1000, 3000.0, 0), (2000, 3100.0, 1)]
        mock_db = _mock_db_ctx()
        with patch("aiosqlite.connect", return_value=mock_db):
            result = await tc.flush()
        assert result == 2
        assert tc._pending == []

    @pytest.mark.asyncio
    async def test_flush_initializes_if_not_initialized(self):
        tc = _make_collector()
        tc._initialized = False
        tc._pending = [(1000, 3000.0, 0)]
        mock_db = _mock_db_ctx()
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "_ensure_dir"):
                result = await tc.flush()
        assert result == 1

    @pytest.mark.asyncio
    async def test_maybe_flush_triggers_when_threshold_exceeded(self):
        tc = _make_collector()
        tc._initialized = True
        tc._flush_interval = 2
        tc._pending = [(i, 3000.0, 0) for i in range(3)]
        mock_db = _mock_db_ctx()
        with patch("aiosqlite.connect", return_value=mock_db):
            await tc.maybe_flush()
        # Pending should be cleared
        assert tc._pending == []

    @pytest.mark.asyncio
    async def test_maybe_flush_does_not_trigger_below_threshold(self):
        tc = _make_collector()
        tc._initialized = True
        tc._flush_interval = 50
        tc._pending = [(1, 3000.0, 0)]
        await tc.maybe_flush()
        assert len(tc._pending) == 1  # not flushed


# ---------------------------------------------------------------------------
# TestPruneOldTicks
# ---------------------------------------------------------------------------

class TestPruneOldTicks:
    @pytest.mark.asyncio
    async def test_prune_returns_zero_when_not_initialized(self):
        tc = _make_collector()
        result = await tc.prune_old_ticks()
        assert result == 0

    @pytest.mark.asyncio
    async def test_prune_returns_rowcount(self):
        tc = _make_collector()
        tc._initialized = True
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        with patch("aiosqlite.connect", return_value=mock_db):
            result = await tc.prune_old_ticks()
        assert result == 5

    @pytest.mark.asyncio
    async def test_prune_zero_rows_deleted(self):
        tc = _make_collector()
        tc._initialized = True
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        with patch("aiosqlite.connect", return_value=mock_db):
            result = await tc.prune_old_ticks()
        assert result == 0


# ---------------------------------------------------------------------------
# TestGetRecentTicks
# ---------------------------------------------------------------------------

class TestGetRecentTicks:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_initialized(self):
        tc = _make_collector()
        result = await tc.get_recent_ticks()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_formatted_rows(self):
        tc = _make_collector()
        tc._initialized = True
        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, k: {"timestamp_ns": 1000, "price": 3000.0, "direction": 1}[k]
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = lambda self, k: {"timestamp_ns": 2000, "price": 3100.0, "direction": 1}[k]

        # db.execute is used as context manager (not awaited), so use MagicMock not AsyncMock
        mock_cursor = MagicMock()
        mock_cursor.fetchall = AsyncMock(return_value=[mock_row1, mock_row2])
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_db = MagicMock()
        mock_db.execute = MagicMock(return_value=mock_cursor)  # returns ctx manager
        mock_db.row_factory = None
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("aiosqlite.connect", return_value=mock_db):
            result = await tc.get_recent_ticks(minutes=60)
        assert len(result) == 2
        assert result[0]["ts"] == 1000
        assert result[0]["price"] == 3000.0


# ---------------------------------------------------------------------------
# TestGetOfi
# ---------------------------------------------------------------------------

class TestGetOfi:
    @pytest.mark.asyncio
    async def test_returns_zero_when_insufficient_data(self):
        tc = _make_collector()
        tc._initialized = True
        # Mock get_recent_ticks to return < 10 ticks
        with patch.object(tc, "get_recent_ticks", AsyncMock(return_value=[{"direction": 1}] * 5)):
            result = await tc.get_ofi(5.0)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_computes_ofi_positive(self):
        tc = _make_collector()
        tc._initialized = True
        # 8 upticks, 2 downticks out of 10
        ticks = [{"direction": 1}] * 8 + [{"direction": -1}] * 2
        with patch.object(tc, "get_recent_ticks", AsyncMock(return_value=ticks)):
            result = await tc.get_ofi(5.0)
        # OFI = (8 - 2) / 10 = 0.6
        assert abs(result - 0.6) < 0.001

    @pytest.mark.asyncio
    async def test_computes_ofi_negative(self):
        tc = _make_collector()
        tc._initialized = True
        ticks = [{"direction": -1}] * 8 + [{"direction": 1}] * 2
        with patch.object(tc, "get_recent_ticks", AsyncMock(return_value=ticks)):
            result = await tc.get_ofi(5.0)
        # OFI = (2 - 8) / 10 = -0.6
        assert abs(result + 0.6) < 0.001


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------

class TestGetStats:
    @pytest.mark.asyncio
    async def test_returns_not_initialized_dict(self):
        tc = _make_collector()
        result = await tc.get_stats()
        assert result["initialized"] is False
        assert result["total_ticks"] == 0
        assert "session_ticks" in result

    def _make_stats_db_mock(self, row):
        """Build a mock for get_stats which uses 'async with connect() as db, db.execute() as cur'."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=row)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        mock_db = MagicMock()
        # db.execute is used as context manager (not awaited)
        mock_db.execute = MagicMock(return_value=mock_cursor)
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        return mock_db

    @pytest.mark.asyncio
    async def test_returns_full_stats_when_initialized(self):
        tc = _make_collector()
        tc._initialized = True
        now_ns = int(time.time() * 1e9)
        row = (600, now_ns - int(2 * 86400 * 1e9), now_ns)
        mock_db = self._make_stats_db_mock(row)
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "get_ofi", AsyncMock(return_value=0.5)):
                result = await tc.get_stats()
        assert result["initialized"] is True
        assert result["total_ticks_db"] == 600
        assert result["ofi_ready"] is True
        assert result["ofi_signal"] == "BUYING"

    @pytest.mark.asyncio
    async def test_ofi_signal_selling(self):
        tc = _make_collector()
        tc._initialized = True
        now_ns = int(time.time() * 1e9)
        row = (600, now_ns - int(2 * 86400 * 1e9), now_ns)
        mock_db = self._make_stats_db_mock(row)
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "get_ofi", AsyncMock(return_value=-0.5)):
                result = await tc.get_stats()
        assert result["ofi_signal"] == "SELLING"

    @pytest.mark.asyncio
    async def test_ofi_signal_neutral(self):
        tc = _make_collector()
        tc._initialized = True
        now_ns = int(time.time() * 1e9)
        row = (600, now_ns - int(2 * 86400 * 1e9), now_ns)
        mock_db = self._make_stats_db_mock(row)
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "get_ofi", AsyncMock(return_value=0.1)):
                result = await tc.get_stats()
        assert result["ofi_signal"] == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_not_ofi_ready_when_insufficient(self):
        tc = _make_collector()
        tc._initialized = True
        now_ns = int(time.time() * 1e9)
        row = (100, now_ns - int(0.5 * 86400 * 1e9), now_ns)  # 100 ticks, 0.5 days
        mock_db = self._make_stats_db_mock(row)
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "get_ofi", AsyncMock(return_value=0.0)):
                result = await tc.get_stats()
        assert result["ofi_ready"] is False
        assert "ACCUMULATING" in result["ofi_accumulation_status"]

    @pytest.mark.asyncio
    async def test_stats_null_row(self):
        tc = _make_collector()
        tc._initialized = True
        mock_db = self._make_stats_db_mock(None)
        with patch("aiosqlite.connect", return_value=mock_db):
            with patch.object(tc, "get_ofi", AsyncMock(return_value=0.0)):
                result = await tc.get_stats()
        assert result["total_ticks_db"] == 0


# ---------------------------------------------------------------------------
# TestEnsureDir
# ---------------------------------------------------------------------------

class TestEnsureDir:
    def test_ensure_dir_creates_parent(self):
        tc = _make_collector(Path("/tmp/testdir/subdir/ticks.db"))
        with patch.object(Path, "mkdir") as mock_mkdir:
            tc._ensure_dir()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TestSingleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_tick_collector_returns_instance(self):
        import wallet_manager.tick_collector as tc_mod
        tc_mod._collector = None
        with patch.dict("os.environ", {"TICK_DB_PATH": "/tmp/test_singleton.db"}):
            collector = get_tick_collector()
        assert isinstance(collector, TickCollector)

    def test_get_tick_collector_singleton(self):
        import wallet_manager.tick_collector as tc_mod
        tc_mod._collector = None
        c1 = get_tick_collector()
        c2 = get_tick_collector()
        assert c1 is c2

    def test_get_tick_collector_uses_env_var(self):
        import wallet_manager.tick_collector as tc_mod
        tc_mod._collector = None
        custom = "C:\\custom\\path.db" if __import__("os").name == "nt" else "/custom/path.db"
        with patch.dict("os.environ", {"TICK_DB_PATH": custom}):
            collector = get_tick_collector()
        assert str(collector._db_path) == custom
        tc_mod._collector = None  # cleanup
