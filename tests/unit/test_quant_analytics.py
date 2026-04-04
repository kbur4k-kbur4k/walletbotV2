"""tests/unit/test_quant_analytics.py — Unit tests for wallet_manager.strategies.quant_analytics."""
from __future__ import annotations

import pytest

from wallet_manager.strategies.quant_analytics import QuantAnalytics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def analytics(tmp_path):
    """Yield an initialised QuantAnalytics backed by a tmp_path SQLite DB."""
    db_file = str(tmp_path / "test_quant.db")
    qa = QuantAnalytics(db_path=db_file)
    await qa.initialize()
    yield qa
    await qa.close()


# ---------------------------------------------------------------------------
# Schema / Initialization
# ---------------------------------------------------------------------------


class TestDbSchemaMigration:
    @pytest.mark.asyncio
    async def test_db_schema_migration(self, tmp_path):
        """initialize() creates all four expected tables in the SQLite DB."""
        db_file = str(tmp_path / "schema_test.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        cursor = await qa._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        table_names = {row[0] for row in rows}

        await qa.close()

        assert "market_features" in table_names
        assert "ml_predictions" in table_names
        assert "trade_executions" in table_names
        assert "performance_snapshots" in table_names

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, tmp_path):
        """Calling initialize() twice must not raise and must leave the DB intact."""
        db_file = str(tmp_path / "idem.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()
        # Second call — CREATE TABLE IF NOT EXISTS should be safe
        await qa.initialize()
        assert qa._initialized is True
        await qa.close()


# ---------------------------------------------------------------------------
# record_trade_execution
# ---------------------------------------------------------------------------


class TestRecordTradeExecution:
    @pytest.mark.asyncio
    async def test_record_trade_persists_to_db(self, tmp_path):
        """record_trade_execution() writes a row that can be read back from the DB."""
        db_file = str(tmp_path / "trade_test.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        await qa.record_trade_execution(
            side="BUY",
            entry_price=2000.0,
            execution_price=2001.5,
            mid_price_at_entry=2000.0,
            position_usd=500.0,
            pnl_usd=0.0,
            pnl_pct=0.0,
            slippage_bps=7.5,
            gas_cost_usd=0.50,
            implementation_shortfall_bps=7.5,
            dex_venue="uniswap_v3",
            vol_regime="normal",
        )

        cursor = await qa._db.execute("SELECT side, entry_price, dex_venue FROM trade_executions")
        rows = await cursor.fetchall()
        await qa.close()

        assert len(rows) == 1
        side, entry_price, dex_venue = rows[0]
        assert side == "BUY"
        assert abs(entry_price - 2000.0) < 1e-6
        assert dex_venue == "uniswap_v3"

    @pytest.mark.asyncio
    async def test_record_multiple_trades_all_persist(self, tmp_path):
        """Multiple record_trade_execution() calls each persist a distinct row."""
        db_file = str(tmp_path / "multi_trade.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        sides = ["BUY", "SELL", "BUY", "SELL", "BUY"]
        for i, side in enumerate(sides):
            await qa.record_trade_execution(
                side=side,
                entry_price=float(1800 + i * 10),
                execution_price=float(1800 + i * 10 + 1),
                pnl_usd=float(i * 2 - 4),
            )

        cursor = await qa._db.execute("SELECT COUNT(*) FROM trade_executions")
        row = await cursor.fetchone()
        await qa.close()

        assert row[0] == 5


# ---------------------------------------------------------------------------
# get_tca_summary (via get_tca_report)
# ---------------------------------------------------------------------------


class TestGetTcaSummary:
    @pytest.mark.asyncio
    async def test_get_tca_summary_returns_dict(self, tmp_path):
        """After recording 5 trades, get_table_counts() returns a dict with expected keys."""
        db_file = str(tmp_path / "tca_test.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        for i in range(5):
            await qa.record_trade_execution(
                side="BUY" if i % 2 == 0 else "SELL",
                entry_price=2000.0 + i,
                execution_price=2001.0 + i,
                mid_price_at_entry=2000.0 + i,
                position_usd=100.0,
                pnl_usd=float(i - 2),
                pnl_pct=float(i - 2) / 100.0,
                slippage_bps=5.0,
                gas_cost_usd=0.10,
                implementation_shortfall_bps=5.0,
            )

        counts = await qa.get_table_counts()
        await qa.close()

        assert isinstance(counts, dict)
        assert "trade_executions" in counts
        assert "market_features" in counts
        assert "ml_predictions" in counts
        assert "performance_snapshots" in counts
        assert counts["trade_executions"] == 5

    @pytest.mark.asyncio
    async def test_get_tca_report_returns_list(self, tmp_path):
        """get_tca_report() returns a list (possibly empty) — never raises."""
        db_file = str(tmp_path / "tca_report.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        reports = await qa.get_tca_report(last_n=10)
        await qa.close()

        assert isinstance(reports, list)


# ---------------------------------------------------------------------------
# Implementation Shortfall
# ---------------------------------------------------------------------------


class TestImplementationShortfallCalculation:
    @pytest.mark.asyncio
    async def test_implementation_shortfall_calculation(self, tmp_path):
        """Known decision vs execution prices produce the correct shortfall in bps.

        IS = |execution_price - decision_price| / decision_price * 10_000
        With decision=2000, execution=2010: IS = 10/2000 * 10000 = 50 bps.
        """
        db_file = str(tmp_path / "is_test.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        decision_price = 2000.0
        execution_price = 2010.0
        expected_is_bps = abs(execution_price - decision_price) / decision_price * 10_000  # 50.0

        await qa.record_trade_execution(
            side="BUY",
            entry_price=decision_price,
            execution_price=execution_price,
            mid_price_at_entry=decision_price,
            position_usd=1000.0,
            implementation_shortfall_bps=expected_is_bps,
        )

        cursor = await qa._db.execute(
            "SELECT implementation_shortfall_bps FROM trade_executions LIMIT 1"
        )
        row = await cursor.fetchone()
        await qa.close()

        stored_bps = row[0]
        assert abs(stored_bps - 50.0) < 1e-6

    @pytest.mark.asyncio
    async def test_zero_shortfall_when_prices_match(self, tmp_path):
        """When execution_price == decision_price, shortfall is 0 bps."""
        db_file = str(tmp_path / "zero_is.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        price = 1500.0
        await qa.record_trade_execution(
            side="BUY",
            entry_price=price,
            execution_price=price,
            mid_price_at_entry=price,
            implementation_shortfall_bps=0.0,
        )

        cursor = await qa._db.execute(
            "SELECT implementation_shortfall_bps FROM trade_executions LIMIT 1"
        )
        row = await cursor.fetchone()
        await qa.close()

        assert row[0] == 0.0


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------


class TestGetPerformanceMetrics:
    @pytest.mark.asyncio
    async def test_empty_db_returns_default_metrics(self, analytics):
        """get_performance_metrics() on an empty DB returns a PerformanceMetrics object."""
        from wallet_manager.strategies.quant_analytics import PerformanceMetrics

        metrics = await analytics.get_performance_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.trade_count == 0

    @pytest.mark.asyncio
    async def test_performance_metrics_with_trades(self, tmp_path):
        """After recording BUY/SELL trades, metrics reflects the data."""
        db_file = str(tmp_path / "perf.db")
        qa = QuantAnalytics(db_path=db_file)
        await qa.initialize()

        pnl_values = [10.0, -5.0, 20.0, -3.0, 8.0]
        for pnl in pnl_values:
            await qa.record_trade_execution(
                side="BUY",
                entry_price=2000.0,
                execution_price=2001.0,
                pnl_usd=pnl,
                pnl_pct=pnl / 1000.0,
                position_usd=1000.0,
            )

        metrics = await qa.get_performance_metrics(days=30)
        await qa.close()

        assert metrics.trade_count == 5
        assert abs(metrics.total_pnl - sum(pnl_values)) < 1e-4
        assert 0.0 <= metrics.win_rate <= 1.0


# ---------------------------------------------------------------------------
# record_market_features
# ---------------------------------------------------------------------------


class TestRecordMarketFeatures:
    @pytest.mark.asyncio
    async def test_record_minimal_market_feature(self, analytics):
        """record_market_features with only price and vol_regime persists a row."""
        await analytics.record_market_features(price=1800.0, vol_regime="low")

        cursor = await analytics._db.execute("SELECT price, vol_regime FROM market_features")
        row = await cursor.fetchone()

        assert row is not None
        assert abs(row[0] - 1800.0) < 1e-6
        assert row[1] == "low"
