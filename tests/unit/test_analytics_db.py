"""tests/unit/test_analytics_db.py — AnalyticsDB in-memory testleri."""
from __future__ import annotations

import asyncio

import pytest

from wallet_manager.core.models import TxRecord, TxStatus
from wallet_manager.resilience.analytics import AnalyticsDB, RuntimeAuditRecord, WalletSummary


@pytest.fixture
async def db(tmp_path):
    async with AnalyticsDB(db_path=str(tmp_path / "test.db")) as database:
        yield database


class TestAnalyticsDBSchema:
    @pytest.mark.asyncio
    async def test_open_creates_schema(self, tmp_path):
        async with AnalyticsDB(db_path=str(tmp_path / "test.db")) as db:
            assert db is not None

    @pytest.mark.asyncio
    async def test_migration_idempotent(self, tmp_path):
        """İki kez açılınca migration tekrar çalışmamalı."""
        path = str(tmp_path / "mig.db")
        async with AnalyticsDB(db_path=path):
            pass
        async with AnalyticsDB(db_path=path) as db2:
            assert db2 is not None


class TestRecordTx:
    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, db):
        record = TxRecord(
            tx_hash="0xabc",
            wallet_id="w1",
            nonce=1,
            strategy="ALGO",
            direction="BUY",
            amount_wei=int(0.1e18),
            gas_used=21000,
            status=TxStatus.MINED,
        )
        await db.record_tx(record)
        summary = await db.get_wallet_summary("w1")
        assert summary.total_txs >= 1

    @pytest.mark.asyncio
    async def test_update_tx_status(self, db):
        record = TxRecord(
            tx_hash="0xdef",
            wallet_id="w2",
            nonce=2,
            strategy="YIELD",
            direction="SUPPLY",
            amount_wei=int(100e6),
            gas_used=0,
            status=TxStatus.PENDING,
        )
        await db.record_tx(record)
        await db.update_tx_status("0xdef", TxStatus.MINED, gas_used=45000)
        summary = await db.get_wallet_summary("w2")
        assert summary.total_txs >= 1

    @pytest.mark.asyncio
    async def test_wallet_summary_empty(self, db):
        summary = await db.get_wallet_summary("nonexistent_wallet")
        assert isinstance(summary, WalletSummary)
        assert summary.total_txs == 0


class TestSessionReport:
    @pytest.mark.asyncio
    async def test_session_report_empty(self, db):
        report = await db.get_session_report()
        assert report is not None

    @pytest.mark.asyncio
    async def test_close_session(self, db):
        await db.close_session(session_id="test_session_1", final_value_usd=1000.0)
        report = await db.get_session_report()
        assert report is not None

    @pytest.mark.asyncio
    async def test_attach_and_read_runtime_audit(self, db):
        payload = {"audit_version": 1, "execution_mode": "robust_paper", "live_guard": {"passed": True}}
        await db.attach_runtime_audit(payload)

        audit = await db.get_runtime_audit()
        assert isinstance(audit, RuntimeAuditRecord)
        assert audit.payload["execution_mode"] == "robust_paper"

        audits = await db.list_runtime_audits(limit=5)
        assert len(audits) == 1
        assert audits[0].payload["live_guard"]["passed"] is True
