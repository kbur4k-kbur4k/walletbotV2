"""tests/integration/test_analytics_db_migration.py — Schema migration tests."""
from __future__ import annotations

import time

import aiosqlite
import pytest

from wallet_manager.core.models import TxRecord, TxStatus
from wallet_manager.resilience.analytics import AnalyticsDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tx_record(tx_hash: str = "0xabc123", wallet_id: str = "0xDEADBEEF00001234") -> TxRecord:
    """Return a minimal valid TxRecord for insertion tests."""
    return TxRecord(
        tx_hash=tx_hash,
        wallet_id=wallet_id,
        chain_id=8453,
        nonce=1,
        gas_price_gwei=1.5,
        gas_limit=21000,
        value_wei=1000,
        status=TxStatus.MINED,
        submitted_at=time.time(),
        gas_used=21000,
        mined_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    @pytest.mark.asyncio
    async def test_fresh_db_creates_all_tables(self, tmp_path):
        async with AnalyticsDB(db_path=str(tmp_path / "fresh.db")):
            import aiosqlite
            async with aiosqlite.connect(str(tmp_path / "fresh.db")) as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] async for row in cursor}
        assert "transactions" in tables
        assert "sessions" in tables
        assert "schema_version" in tables

    @pytest.mark.asyncio
    async def test_migration_version_tracked(self, tmp_path):
        path = str(tmp_path / "versioned.db")
        async with AnalyticsDB(db_path=path):
            pass
        import aiosqlite
        async with aiosqlite.connect(path) as conn:
            cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
            row = await cursor.fetchone()
        assert row[0] >= 1

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path):
        path = str(tmp_path / "wal.db")
        async with AnalyticsDB(db_path=path):
            pass
        import aiosqlite
        async with aiosqlite.connect(path) as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
        assert row[0].lower() == "wal"


# ---------------------------------------------------------------------------
# New tests — schema columns, data preservation, re-initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transactions_table_has_required_columns(tmp_path):
    """transactions table must expose all schema columns after initialize."""
    path = str(tmp_path / "cols.db")
    async with AnalyticsDB(db_path=path):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("PRAGMA table_info(transactions)")
        rows = await cursor.fetchall()

    col_names = {row[1] for row in rows}
    required = {
        "tx_hash", "wallet_id", "chain_id", "nonce",
        "gas_price_gwei", "gas_limit", "gas_used",
        "value_wei", "status", "submitted_at",
        "mined_at", "replacement_hash", "session_id",
        "strategy_type", "strategy_id",
    }
    missing = required - col_names
    assert not missing, f"Missing columns in transactions: {missing}"


@pytest.mark.asyncio
async def test_sessions_table_has_required_columns(tmp_path):
    """sessions table must have all columns documented in the schema."""
    path = str(tmp_path / "sess_cols.db")
    async with AnalyticsDB(db_path=path):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("PRAGMA table_info(sessions)")
        rows = await cursor.fetchall()

    col_names = {row[1] for row in rows}
    required = {
        "session_id", "started_at", "finished_at",
        "total_wallets", "success_count", "fail_count", "notes",
    }
    missing = required - col_names
    assert not missing, f"Missing columns in sessions: {missing}"


@pytest.mark.asyncio
async def test_schema_version_table_has_required_columns(tmp_path):
    """schema_version table must have version, applied_at, and description."""
    path = str(tmp_path / "sv_cols.db")
    async with AnalyticsDB(db_path=path):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("PRAGMA table_info(schema_version)")
        rows = await cursor.fetchall()

    col_names = {row[1] for row in rows}
    assert "version" in col_names
    assert "applied_at" in col_names
    assert "description" in col_names


@pytest.mark.asyncio
async def test_telegram_chats_table_created(tmp_path):
    """telegram_chats table must be created alongside other tables."""
    path = str(tmp_path / "tg.db")
    async with AnalyticsDB(db_path=path):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='telegram_chats'"
        )
        row = await cursor.fetchone()

    assert row is not None, "telegram_chats table was not created"


@pytest.mark.asyncio
async def test_session_row_inserted_on_open(tmp_path):
    """Opening AnalyticsDB must insert exactly one row into sessions."""
    path = str(tmp_path / "sess.db")
    session_id = "test-session-001"
    async with AnalyticsDB(db_path=path, session_id=session_id):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT session_id, started_at FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()

    assert row is not None, "No session row was inserted"
    assert row[0] == session_id
    assert row[1] > 0  # started_at is a real Unix timestamp


@pytest.mark.asyncio
async def test_data_preserved_after_re_initialization(tmp_path):
    """Existing rows in sessions must survive a second open() call (idempotent init)."""
    path = str(tmp_path / "reopen.db")
    session_a = "session-alpha"
    session_b = "session-beta"

    # First open — inserts session_a
    async with AnalyticsDB(db_path=path, session_id=session_a):
        pass

    # Second open — inserts session_b; session_a must still be present
    async with AnalyticsDB(db_path=path, session_id=session_b):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT session_id FROM sessions ORDER BY started_at"
        )
        rows = await cursor.fetchall()

    session_ids = {row[0] for row in rows}
    assert session_a in session_ids, "First session was deleted on re-init"
    assert session_b in session_ids, "Second session was not inserted"


@pytest.mark.asyncio
async def test_record_tx_persists_to_sqlite(tmp_path):
    """record_tx() must write the row so it is readable via a raw aiosqlite query."""
    path = str(tmp_path / "record.db")
    record = _make_tx_record(tx_hash="0xdeadbeef01")

    async with AnalyticsDB(db_path=path, session_id="s1") as db:
        await db.record_tx(record)

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT tx_hash, chain_id, status FROM transactions WHERE tx_hash = ?",
            (record.tx_hash,),
        )
        row = await cursor.fetchone()

    assert row is not None, "TxRecord was not persisted"
    assert row[0] == record.tx_hash
    assert row[1] == 8453
    assert row[2] == TxStatus.MINED.name


@pytest.mark.asyncio
async def test_record_tx_data_survives_reopen(tmp_path):
    """Rows inserted in one session must still be readable after the DB is closed and reopened."""
    path = str(tmp_path / "survive.db")
    record = _make_tx_record(tx_hash="0xsurvive001")

    async with AnalyticsDB(db_path=path, session_id="first-session") as db:
        await db.record_tx(record)

    # Reopen — this runs initialization SQL again (IF NOT EXISTS) so must not wipe data
    async with AnalyticsDB(db_path=path, session_id="second-session"):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT tx_hash FROM transactions WHERE tx_hash = ?",
            (record.tx_hash,),
        )
        row = await cursor.fetchone()

    assert row is not None, "Transaction data was destroyed on re-initialization"


@pytest.mark.asyncio
async def test_multiple_sessions_isolated_but_share_db(tmp_path):
    """Two sessions in the same DB file must each have their own session row."""
    path = str(tmp_path / "multi.db")
    tx1 = _make_tx_record(tx_hash="0xaaaa0001")
    tx2 = _make_tx_record(tx_hash="0xbbbb0002")

    async with AnalyticsDB(db_path=path, session_id="session-one") as db1:
        await db1.record_tx(tx1)

    async with AnalyticsDB(db_path=path, session_id="session-two") as db2:
        await db2.record_tx(tx2)

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM transactions")
        count_row = await cursor.fetchone()

        cursor2 = await conn.execute("SELECT COUNT(*) FROM sessions")
        sess_count = await cursor2.fetchone()

    assert count_row[0] == 2, "Expected 2 transaction rows across both sessions"
    assert sess_count[0] == 2, "Expected 2 session rows"


@pytest.mark.asyncio
async def test_db_uses_tmp_path_not_real_db(tmp_path):
    """The DB must be created at the tmp_path location, not the production path."""
    db_path = tmp_path / "isolated.db"
    assert not db_path.exists(), "Pre-condition: file should not exist yet"

    async with AnalyticsDB(db_path=str(db_path)):
        pass

    assert db_path.exists(), "DB file was not created at tmp_path"


@pytest.mark.asyncio
async def test_schema_version_contains_baseline_migration(tmp_path):
    """After initialization the schema_version table must contain at least version 1."""
    path = str(tmp_path / "v1.db")
    async with AnalyticsDB(db_path=path):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT version, description FROM schema_version WHERE version = 1"
        )
        row = await cursor.fetchone()

    assert row is not None, "schema_version has no row for version 1 (baseline)"


@pytest.mark.asyncio
async def test_initialize_is_idempotent(tmp_path):
    """Opening the DB three times in succession must not raise or duplicate tables."""
    path = str(tmp_path / "idempotent.db")
    for i in range(3):
        async with AnalyticsDB(db_path=path, session_id=f"run-{i}"):
            pass

    # Confirm table list is stable (no duplication possible in SQLite, but verify)
    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] async for row in cursor}

    for expected in ("transactions", "sessions", "schema_version", "telegram_chats"):
        assert expected in tables


@pytest.mark.asyncio
async def test_indices_created(tmp_path):
    """Core indices on transactions must be present after initialization."""
    path = str(tmp_path / "idx.db")
    async with AnalyticsDB(db_path=path):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indices = {row[0] async for row in cursor}

    # At minimum these indices must exist (defined in _CREATE_INDICES_SQL)
    assert "idx_tx_wallet" in indices
    assert "idx_tx_chain" in indices
    assert "idx_tx_session" in indices
    assert "idx_tx_status" in indices
