"""
wallet_manager/resilience/analytics.py

Local Analytical SQLite Database for transaction history.

Responsibility
--------------
Persist every transaction lifecycle event to a local SQLite database so that
post-session analysis, reconciliation, and auditing are possible without
relying on external block explorers.

Schema
------
``transactions`` table — one row per unique ``tx_hash``:

    tx_hash          TEXT PRIMARY KEY
    wallet_id        TEXT NOT NULL
    chain_id         INTEGER NOT NULL
    nonce            INTEGER NOT NULL
    gas_price_gwei   REAL NOT NULL
    gas_limit        INTEGER NOT NULL
    gas_used         INTEGER            -- NULL until mined
    value_wei        TEXT NOT NULL      -- stored as text to handle uint256
    status           TEXT NOT NULL      -- TxStatus.name
    submitted_at     REAL NOT NULL      -- Unix epoch
    mined_at         REAL               -- NULL until mined
    replacement_hash TEXT               -- NULL unless RBF triggered
    session_id       TEXT NOT NULL      -- groups records by orchestrator run

``sessions`` table — one row per orchestrator run:

    session_id       TEXT PRIMARY KEY
    started_at       REAL NOT NULL
    finished_at      REAL
    total_wallets    INTEGER
    success_count    INTEGER
    fail_count       INTEGER
    notes            TEXT

Design decisions
----------------
* **aiosqlite**: All database operations are async (non-blocking).  The event
  loop is never blocked by disk I/O.
* **WAL mode**: Write-Ahead Logging is enabled so reads never block writes and
  vice versa.
* **Parameterised queries**: All queries use ``?`` placeholders; no string
  interpolation prevents SQL injection even from internal callers.
* **Wallet ID masking**: The wallet_id column stores the masked form (first 6 +
  last 4 chars) so the database itself never contains un-masked addresses.
* **Session grouping**: Every orchestrator run generates a ``session_id``
  (ISO timestamp) so records from different runs are trivially separable.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from wallet_manager.config import get_settings
from wallet_manager.core.models import TxRecord, TxStatus
from wallet_manager.exceptions import AnalyticsQueryError, AnalyticsWriteError
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH: str = str(get_settings().analytics_db_path)

_CREATE_TRANSACTIONS_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    tx_hash          TEXT PRIMARY KEY,
    wallet_id        TEXT NOT NULL,
    chain_id         INTEGER NOT NULL,
    nonce            INTEGER NOT NULL,
    gas_price_gwei   REAL NOT NULL,
    gas_limit        INTEGER NOT NULL,
    gas_used         INTEGER,
    value_wei        TEXT NOT NULL,
    status           TEXT NOT NULL,
    submitted_at     REAL NOT NULL,
    mined_at         REAL,
    replacement_hash TEXT,
    session_id       TEXT NOT NULL,
    strategy_type    TEXT,
    strategy_id      TEXT
)
"""

_CREATE_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    total_wallets INTEGER,
    success_count INTEGER,
    fail_count    INTEGER,
    notes         TEXT
)
"""

_CREATE_TELEGRAM_CHATS_SQL = """
CREATE TABLE IF NOT EXISTS telegram_chats (
    chat_id       TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    last_used_at  REAL,
    label         TEXT
)
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tx_wallet ON transactions(wallet_id)",
    "CREATE INDEX IF NOT EXISTS idx_tx_chain ON transactions(chain_id)",
    "CREATE INDEX IF NOT EXISTS idx_tx_session ON transactions(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_tx_status ON transactions(status)",
    "CREATE INDEX IF NOT EXISTS idx_tx_submitted ON transactions(submitted_at)",
    "CREATE INDEX IF NOT EXISTS idx_tx_strategy ON transactions(strategy_type)",
    "CREATE INDEX IF NOT EXISTS idx_tx_strategy_id ON transactions(strategy_id)",
]

# Migration: add strategy_type column if missing (for existing DBs)
_MIGRATE_STRATEGY_TYPE_SQL = (
    "ALTER TABLE transactions ADD COLUMN strategy_type TEXT DEFAULT 'UNKNOWN'"
)
_MIGRATE_STRATEGY_ID_SQL = (
    "ALTER TABLE transactions ADD COLUMN strategy_id TEXT"
)

# ---------------------------------------------------------------------------
# Schema versioning (Section 10.2)
# ---------------------------------------------------------------------------

_CREATE_SCHEMA_VERSION_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  REAL    NOT NULL,
    description TEXT    NOT NULL
)
"""

# Each tuple: (version, description, sql)
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "Initial schema",
     "-- baseline, no ALTER needed"),
    (2, "Add regime_label to trade_executions",
     "ALTER TABLE transactions ADD COLUMN regime_label TEXT DEFAULT ''"),
    (3, "Add concept_drift_level to trade_executions",
     "ALTER TABLE transactions ADD COLUMN concept_drift_level TEXT DEFAULT 'none'"),
]


# ---------------------------------------------------------------------------
# Summary dataclasses (returned by query methods)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalletSummary:
    """Aggregated stats for one wallet across all sessions.

    Attributes:
        wallet_id: Masked wallet identifier.
        total_txs: Total transactions submitted.
        mined_txs: Successfully mined transactions.
        replaced_txs: Transactions replaced via RBF.
        dropped_txs: Transactions dropped from mempool.
        total_gas_used: Cumulative gas consumed.
        total_value_wei: Sum of all transferred value in wei.
        avg_gas_price_gwei: Average gas price across all transactions.
    """

    wallet_id: str
    total_txs: int
    mined_txs: int
    replaced_txs: int
    dropped_txs: int
    total_gas_used: int
    total_value_wei: int
    avg_gas_price_gwei: float


@dataclass(frozen=True)
class StrategySummary:
    """Per-strategy analytics for PnL tracking.

    Attributes:
        strategy_type: YIELD, ALGO, or AIRDROP.
        total_txs: Total transactions for this strategy.
        mined_txs: Successfully mined transactions.
        total_gas_used: Cumulative gas consumed.
        total_value_wei: Sum of value_wei (capital flow proxy).
    """

    strategy_type: str
    total_txs: int
    mined_txs: int
    total_gas_used: int
    total_value_wei: int


@dataclass(frozen=True)
class SessionReport:
    """Full report for one orchestrator session.

    Attributes:
        session_id: Unique session identifier.
        started_at: Unix epoch start timestamp.
        finished_at: Unix epoch finish timestamp; ``None`` if still running.
        total_txs: Transactions recorded in this session.
        mined_txs: Mined in this session.
        replaced_txs: Replaced (RBF) in this session.
        total_gas_used: Gas consumed in this session.
        total_wallets: Wallets that executed in this session.
        success_count: Wallets that completed without error.
        fail_count: Wallets that raised an error.
        duration_seconds: Session wall-clock duration; ``None`` if ongoing.
    """

    session_id: str
    started_at: float
    finished_at: float | None
    total_txs: int
    mined_txs: int
    replaced_txs: int
    total_gas_used: int
    total_wallets: int
    success_count: int
    fail_count: int
    notes: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration of the session in seconds."""
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at


@dataclass(frozen=True)
class RuntimeAuditRecord:
    """Parsed runtime audit payload stored in ``sessions.notes``."""

    session_id: str
    started_at: float
    finished_at: float | None
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# AnalyticsDB
# ---------------------------------------------------------------------------


class AnalyticsDB:
    """Async SQLite analytics database for transaction history.

    Manages schema creation, record insertion, and reporting queries.
    All operations are non-blocking thanks to ``aiosqlite``.

    Args:
        db_path: Path to the SQLite file.  Defaults to ``ANALYTICS_DB_PATH``
            env var (``"wallet_analytics.db"`` if unset).
        session_id: Identifier for the current orchestrator run.  Defaults to
            a UUID4 string.  Pass an explicit value in tests for reproducibility.

    Example:
        .. code-block:: python

            async with AnalyticsDB() as db:
                await db.record_tx(tx_record)
                report = await db.get_session_report()
                await db.export_csv("session_export.csv")
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        session_id: str | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._session_id: str = session_id or str(uuid.uuid4())
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Async context manager / lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AnalyticsDB:
        """Open the database and initialise the schema."""
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the database connection."""
        await self.close()

    async def open(self) -> None:
        """Open the SQLite connection and create tables if absent.

        Enables WAL mode for concurrent read/write performance.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row

        # WAL mode: readers and the single writer do not block each other
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

        await self._conn.execute(_CREATE_SCHEMA_VERSION_SQL)
        await self._conn.execute(_CREATE_TRANSACTIONS_SQL)
        await self._conn.execute(_CREATE_SESSIONS_SQL)
        await self._conn.execute(_CREATE_TELEGRAM_CHATS_SQL)
        for idx_sql in _CREATE_INDICES_SQL:
            await self._conn.execute(idx_sql)
        # Migrate existing DBs: add strategy_type, strategy_id if columns missing
        async with self._conn.execute("PRAGMA table_info(transactions)") as cur:
            rows = await cur.fetchall()
        column_names = [row[1] for row in rows]
        if "strategy_type" not in column_names:
            await self._conn.execute(_MIGRATE_STRATEGY_TYPE_SQL)
            log.debug("AnalyticsDB: added strategy_type column to transactions.")
        if "strategy_id" not in column_names:
            await self._conn.execute(_MIGRATE_STRATEGY_ID_SQL)
            log.debug("AnalyticsDB: added strategy_id column to transactions.")
        await self._conn.commit()
        await self._apply_migrations()

        # Record this session
        await self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)",
            (self._session_id, time.time()),
        )
        await self._conn.commit()
        log.info(
            f"AnalyticsDB opened: '{self._db_path}', session='{self._session_id}'.",
        )

    async def _apply_migrations(self) -> None:
        """Apply pending schema migrations atomically.

        Reads current version from schema_version table (0 if empty).
        Applies all migrations with version > current in ascending order.
        Each migration is wrapped in BEGIN / COMMIT.
        On failure: rollback and re-raise — never leaves DB in partial state.
        """
        async with self._db.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ) as cur:
            row = await cur.fetchone()
        current_version: int = row[0] if row else 0

        for version, description, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
            if version <= current_version:
                continue
            try:
                await self._db.execute("BEGIN")
                if sql.strip() and not sql.strip().startswith("--"):
                    await self._db.execute(sql)
                await self._db.execute(
                    "INSERT INTO schema_version (version, applied_at, description) "
                    "VALUES (?, ?, ?)",
                    (version, __import__("time").time(), description),
                )
                await self._db.execute("COMMIT")
                log.info(
                    "AnalyticsDB: applied migration v%d — %s", version, description
                )
            except Exception as exc:
                await self._db.execute("ROLLBACK")
                log.error(
                    "AnalyticsDB: migration v%d FAILED — %s", version, exc
                )
                raise

    async def close(self) -> None:
        """Flush any pending writes and close the connection."""
        if self._conn:
            await self._conn.commit()
            await self._conn.close()
            self._conn = None
        log.info("AnalyticsDB closed.")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def record_tx(self, record: TxRecord) -> None:
        """Insert or update a transaction record in the database.

        Uses ``INSERT OR REPLACE`` so the same ``tx_hash`` can be updated
        (e.g. when a ``PENDING`` record transitions to ``MINED``).

        The ``wallet_id`` is masked before storage so the database file itself
        never contains un-masked on-chain identifiers.

        Args:
            record: :class:`~wallet_manager.core.models.TxRecord` to persist.

        Raises:
            AnalyticsWriteError: If the INSERT fails.
        """
        self._assert_open()
        masked_wallet = _mask_wallet_id(record.wallet_id)

        strategy_type_val = getattr(record, "strategy_type", None)
        strategy_id_val = getattr(record, "strategy_id", None)
        sql = """
            INSERT OR REPLACE INTO transactions (
                tx_hash, wallet_id, chain_id, nonce,
                gas_price_gwei, gas_limit, gas_used,
                value_wei, status, submitted_at,
                mined_at, replacement_hash, session_id, strategy_type, strategy_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        # Support amount_wei as alias for value_wei
        value_wei = record.value_wei or getattr(record, "amount_wei", None) or 0
        params = (
            record.tx_hash,
            masked_wallet,
            record.chain_id,
            record.nonce,
            record.gas_price_gwei,
            record.gas_limit,
            record.gas_used,
            str(value_wei),
            record.status.name,
            record.submitted_at,
            record.mined_at,
            record.replacement_hash,
            self._session_id,
            strategy_type_val,
            strategy_id_val,
        )
        try:
            async with self._write_lock:
                await self._db.execute(sql, params)
                await self._db.commit()
            log.debug(
                f"AnalyticsDB: recorded tx {record.tx_hash} ({record.status.name}) for wallet {masked_wallet}.",
            )
        except Exception as exc:
            raise AnalyticsWriteError(
                f"Failed to write tx record {record.tx_hash}: {exc}"
            ) from exc

    async def update_tx_status(
        self,
        tx_hash: str,
        status: TxStatus,
        gas_used: int | None = None,
        mined_at: float | None = None,
        replacement_hash: str | None = None,
    ) -> None:
        """Patch the status and mining data for an existing transaction record.

        Args:
            tx_hash: The transaction to update.
            status: New lifecycle status.
            gas_used: Actual gas consumed (filled on mining).
            mined_at: Unix epoch mining timestamp.
            replacement_hash: RBF replacement hash, if applicable.

        Raises:
            AnalyticsWriteError: If the UPDATE fails.
        """
        self._assert_open()
        sql = """
            UPDATE transactions
               SET status = ?,
                   gas_used = COALESCE(?, gas_used),
                   mined_at = COALESCE(?, mined_at),
                   replacement_hash = COALESCE(?, replacement_hash)
             WHERE tx_hash = ?
        """
        params = (status.name, gas_used, mined_at, replacement_hash, tx_hash)
        try:
            async with self._write_lock:
                await self._db.execute(sql, params)
                await self._db.commit()
        except Exception as exc:
            raise AnalyticsWriteError(
                f"Failed to update status for {tx_hash}: {exc}"
            ) from exc

    async def close_session(
        self,
        total_wallets: int = 0,
        success_count: int = 0,
        fail_count: int = 0,
        notes: str | None = None,
        session_id: str | None = None,
        final_value_usd: float | None = None,
    ) -> None:
        """Mark the current session as finished and record summary stats.

        Args:
            total_wallets: Number of wallets that participated.
            success_count: Wallets that completed without error.
            fail_count: Wallets that raised an error.
            notes: Optional freeform notes (e.g. operator comments).
        """
        self._assert_open()
        sql = """
            UPDATE sessions
               SET finished_at = ?,
                   total_wallets = ?,
                   success_count = ?,
                   fail_count = ?,
                   notes = ?
             WHERE session_id = ?
        """
        try:
            async with self._write_lock:
                await self._db.execute(
                    sql,
                    (time.time(), total_wallets, success_count, fail_count,
                     notes, self._session_id),
                )
                await self._db.commit()
        except Exception as exc:
            raise AnalyticsWriteError(
                f"Failed to close session {self._session_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def get_wallet_summary(self, wallet_id: str) -> WalletSummary:
        """Return aggregated statistics for *wallet_id* across all sessions.

        Args:
            wallet_id: Raw or masked wallet identifier to look up.

        Returns:
            :class:`WalletSummary` with cumulative stats.

        Raises:
            AnalyticsQueryError: If the query fails.
        """
        self._assert_open()
        masked = _mask_wallet_id(wallet_id)
        sql = """
            SELECT
                COUNT(*)                             AS total_txs,
                SUM(status = 'MINED')                AS mined_txs,
                SUM(status = 'REPLACED')             AS replaced_txs,
                SUM(status = 'DROPPED')              AS dropped_txs,
                COALESCE(SUM(gas_used), 0)           AS total_gas,
                COALESCE(SUM(CAST(value_wei AS INTEGER)), 0) AS total_value,
                COALESCE(AVG(gas_price_gwei), 0.0)   AS avg_gas_price
            FROM transactions
            WHERE wallet_id = ?
        """
        try:
            async with self._db.execute(sql, (masked,)) as cursor:
                row = await cursor.fetchone()
        except Exception as exc:
            raise AnalyticsQueryError(
                f"Failed to query summary for wallet '{masked}': {exc}"
            ) from exc

        if row is None or row["total_txs"] == 0:
            return WalletSummary(
                wallet_id=masked,
                total_txs=0, mined_txs=0, replaced_txs=0, dropped_txs=0,
                total_gas_used=0, total_value_wei=0, avg_gas_price_gwei=0.0,
            )

        return WalletSummary(
            wallet_id=masked,
            total_txs=row["total_txs"] or 0,
            mined_txs=row["mined_txs"] or 0,
            replaced_txs=row["replaced_txs"] or 0,
            dropped_txs=row["dropped_txs"] or 0,
            total_gas_used=row["total_gas"] or 0,
            total_value_wei=row["total_value"] or 0,
            avg_gas_price_gwei=round(row["avg_gas_price"] or 0.0, 4),
        )

    async def get_session_report(
        self, session_id: str | None = None
    ) -> SessionReport:
        """Return a :class:`SessionReport` for the specified (or current) session.

        Args:
            session_id: Session to query.  Defaults to the current session.

        Returns:
            :class:`SessionReport` with aggregate and session metadata.

        Raises:
            AnalyticsQueryError: If the query fails or session is not found.
        """
        self._assert_open()
        sid = session_id or self._session_id

        session_sql = "SELECT * FROM sessions WHERE session_id = ?"
        tx_sql = """
            SELECT
                COUNT(*)                    AS total_txs,
                SUM(status = 'MINED')       AS mined_txs,
                SUM(status = 'REPLACED')    AS replaced_txs,
                COALESCE(SUM(gas_used), 0)  AS total_gas
            FROM transactions
            WHERE session_id = ?
        """
        try:
            async with self._db.execute(session_sql, (sid,)) as cur:
                session_row = await cur.fetchone()
            async with self._db.execute(tx_sql, (sid,)) as cur:
                tx_row = await cur.fetchone()
        except Exception as exc:
            raise AnalyticsQueryError(
                f"Failed to query session report for '{sid}': {exc}"
            ) from exc

        if session_row is None:
            raise AnalyticsQueryError(f"Session '{sid}' not found in database.")

        _sr: Any = session_row
        _tx: Any = tx_row or {"total_txs": 0, "mined_txs": 0, "replaced_txs": 0, "total_gas": 0}

        return SessionReport(
            session_id=sid,
            started_at=_sr["started_at"],
            finished_at=_sr["finished_at"],
            total_txs=_tx["total_txs"] or 0,
            mined_txs=_tx["mined_txs"] or 0,
            replaced_txs=_tx["replaced_txs"] or 0,
            total_gas_used=_tx["total_gas"] or 0,
            total_wallets=_sr["total_wallets"] or 0,
            success_count=_sr["success_count"] or 0,
            fail_count=_sr["fail_count"] or 0,
            notes=_sr["notes"],
        )

    async def attach_runtime_audit(
        self,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        """Persist a JSON audit payload into the session notes column."""
        self._assert_open()
        sid = session_id or self._session_id
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        sql = "UPDATE sessions SET notes = ? WHERE session_id = ?"
        try:
            async with self._write_lock:
                await self._db.execute(sql, (encoded, sid))
                await self._db.commit()
        except Exception as exc:
            raise AnalyticsWriteError(
                f"Failed to attach runtime audit for session {sid}: {exc}"
            ) from exc

    async def get_runtime_audit(
        self,
        session_id: str | None = None,
    ) -> RuntimeAuditRecord | None:
        """Return the parsed runtime audit payload for a session if available."""
        report = await self.get_session_report(session_id=session_id)
        if not report.notes:
            return None
        try:
            payload = json.loads(report.notes)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("audit_version") is None:
            return None
        return RuntimeAuditRecord(
            session_id=report.session_id,
            started_at=report.started_at,
            finished_at=report.finished_at,
            payload=payload,
        )

    async def list_runtime_audits(self, limit: int = 10) -> list[RuntimeAuditRecord]:
        """Return recent runtime audit payloads from newest to oldest."""
        self._assert_open()
        sql = """
            SELECT session_id, started_at, finished_at, notes
            FROM sessions
            WHERE notes IS NOT NULL AND notes != ''
            ORDER BY started_at DESC
            LIMIT ?
        """
        try:
            async with self._db.execute(sql, (limit,)) as cur:
                rows = await cur.fetchall()
        except Exception as exc:
            raise AnalyticsQueryError(f"Failed to list runtime audits: {exc}") from exc

        audits: list[RuntimeAuditRecord] = []
        for row in rows:
            try:
                payload = json.loads(row["notes"])
            except Exception:
                continue
            if not isinstance(payload, dict) or payload.get("audit_version") is None:
                continue
            audits.append(
                RuntimeAuditRecord(
                    session_id=row["session_id"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    payload=payload,
                )
            )
        return audits

    async def get_strategy_summary(
        self, strategy_type: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return aggregated stats by strategy_type (YIELD, ALGO, AIRDROP) for PnL tracking.

        Args:
            strategy_type: Filter by strategy. If None, returns all strategies.

        Returns:
            Dict with strategy_type as key, values: total_txs, mined_txs,
            total_gas_used, total_value_wei. Use with current on-chain balances
            to compute PnL: pnl = current_value - sum(historical capital deployed).
        """
        self._assert_open()
        if strategy_type:
            sql = """
                SELECT strategy_type,
                       COUNT(*) AS total_txs,
                       SUM(status = 'MINED') AS mined_txs,
                       COALESCE(SUM(gas_used), 0) AS total_gas,
                       COALESCE(SUM(CAST(value_wei AS INTEGER)), 0) AS total_value
                FROM transactions
                WHERE strategy_type = ?
                GROUP BY strategy_type
            """
            params: tuple[Any, ...] = (strategy_type,)
        else:
            sql = """
                SELECT strategy_type,
                       COUNT(*) AS total_txs,
                       SUM(status = 'MINED') AS mined_txs,
                       COALESCE(SUM(gas_used), 0) AS total_gas,
                       COALESCE(SUM(CAST(value_wei AS INTEGER)), 0) AS total_value
                FROM transactions
                GROUP BY strategy_type
            """
            params = ()
        try:
            async with self._db.execute(sql, params) as cur:
                rows = await cur.fetchall()
            return {str(row["strategy_type"] or "UNKNOWN"): dict(row) for row in rows}
        except Exception as exc:
            raise AnalyticsQueryError(
                f"Failed to query strategy summary: {exc}"
            ) from exc

    async def get_strategy_pnl_report(
        self,
        current_values: dict[str, float] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return per-strategy report for PnL reconciliation.

        Combines historical tx data with optional current pillar values.
        PnL = current_value - capital_deployed (when current_values provided).

        Args:
            current_values: Optional dict mapping strategy_type to current USD value
                (e.g. {"YIELD": 125.0, "ALGO": 90.0, "AIRDROP": 35.0}).

        Returns:
            Dict mapping strategy_type to {total_txs, mined_txs, total_gas_used,
            total_value_wei, current_usd, pnl_usd (if current_values provided)}.
        """
        summary = await self.get_strategy_summary()
        report: dict[str, dict[str, Any]] = {}
        for stype, row in summary.items():
            entry = dict(row)
            if current_values and stype in current_values:
                entry["current_usd"] = current_values[stype]
                # Approximate PnL: current - (total_value_wei as USD proxy)
                # value_wei is in wei; rough USD = value_wei/1e18 * eth_price
                # Without eth_price we use total_value_wei as relative proxy
                entry["pnl_note"] = (
                    "Compute PnL as current_usd - sum(tx values in USD)"
                )
            report[stype] = entry
        return report

    async def record_telegram_chat(
        self,
        chat_id: str,
        label: str | None = None,
    ) -> None:
        """Upsert a Telegram chat ID for notifications.

        Args:
            chat_id: Telegram chat ID (from TELEGRAM_CHAT_ID env).
            label: Optional human-readable label.
        """
        self._assert_open()
        now = time.time()
        sql = """
            INSERT INTO telegram_chats (chat_id, created_at, last_used_at, label)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                last_used_at = excluded.last_used_at,
                label = COALESCE(excluded.label, label)
        """
        try:
            async with self._write_lock:
                await self._db.execute(sql, (chat_id, now, now, label))
                await self._db.commit()
        except Exception as exc:
            raise AnalyticsWriteError(
                f"Failed to record telegram chat: {exc}"
            ) from exc

    async def get_strategy_comparison(
        self,
        strategy_ids: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Side-by-side comparison: Algo vs Yield (or custom strategy_ids).
        Returns per-strategy stats for performance comparison.
        """
        self._assert_open()
        ids = strategy_ids or ["algo_vol_mean_reversion", "yield_aave_v3"]
        placeholders = ",".join("?" * len(ids))
        sql = f"""
            SELECT strategy_id, strategy_type,
                   COUNT(*) AS total_txs,
                   SUM(status = 'MINED') AS mined_txs,
                   COALESCE(SUM(gas_used), 0) AS total_gas,
                   COALESCE(SUM(CAST(value_wei AS INTEGER)), 0) AS total_value,
                   COALESCE(AVG(gas_price_gwei), 0) AS avg_gas
            FROM transactions
            WHERE strategy_id IN ({placeholders})
            GROUP BY strategy_id, strategy_type
        """
        try:
            async with self._db.execute(sql, ids) as cur:
                rows = await cur.fetchall()
            return {
                str(row["strategy_id"] or row["strategy_type"] or "UNKNOWN"): {
                    "strategy_type": row["strategy_type"],
                    "total_txs": row["total_txs"] or 0,
                    "mined_txs": row["mined_txs"] or 0,
                    "total_gas_used": row["total_gas"] or 0,
                    "total_value_wei": row["total_value"] or 0,
                    "avg_gas_price_gwei": round(row["avg_gas"] or 0, 4),
                }
                for row in rows
            }
        except Exception as exc:
            raise AnalyticsQueryError(
                f"Failed to query strategy comparison: {exc}"
            ) from exc

    async def get_strategy_performance(
        self,
        strategy_type: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return detailed strategy performance for PnL and analytics.

        Args:
            strategy_type: Filter by strategy. If None, returns all.

        Returns:
            Dict mapping strategy_type to {
                total_txs, mined_txs, total_gas_used, total_value_wei,
                avg_gas_price, first_tx_at, last_tx_at
            }.
        """
        self._assert_open()
        if strategy_type:
            sql = """
                SELECT strategy_type,
                       COUNT(*) AS total_txs,
                       SUM(status = 'MINED') AS mined_txs,
                       COALESCE(SUM(gas_used), 0) AS total_gas,
                       COALESCE(SUM(CAST(value_wei AS INTEGER)), 0) AS total_value,
                       COALESCE(AVG(gas_price_gwei), 0) AS avg_gas,
                       MIN(submitted_at) AS first_at,
                       MAX(submitted_at) AS last_at
                FROM transactions
                WHERE strategy_type = ?
                GROUP BY strategy_type
            """
            params: tuple[Any, ...] = (strategy_type,)
        else:
            sql = """
                SELECT strategy_type,
                       COUNT(*) AS total_txs,
                       SUM(status = 'MINED') AS mined_txs,
                       COALESCE(SUM(gas_used), 0) AS total_gas,
                       COALESCE(SUM(CAST(value_wei AS INTEGER)), 0) AS total_value,
                       COALESCE(AVG(gas_price_gwei), 0) AS avg_gas,
                       MIN(submitted_at) AS first_at,
                       MAX(submitted_at) AS last_at
                FROM transactions
                WHERE strategy_type IS NOT NULL
                GROUP BY strategy_type
            """
            params = ()
        try:
            async with self._db.execute(sql, params) as cur:
                rows = await cur.fetchall()
            return {
                str(row["strategy_type"] or "UNKNOWN"): {
                    "total_txs": row["total_txs"] or 0,
                    "mined_txs": row["mined_txs"] or 0,
                    "total_gas_used": row["total_gas"] or 0,
                    "total_value_wei": row["total_value"] or 0,
                    "avg_gas_price_gwei": round(row["avg_gas"] or 0, 4),
                    "first_tx_at": row["first_at"],
                    "last_tx_at": row["last_at"],
                }
                for row in rows
            }
        except Exception as exc:
            raise AnalyticsQueryError(
                f"Failed to query strategy performance: {exc}"
            ) from exc

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return a lightweight list of all recorded sessions.

        Returns:
            List of dicts with keys: ``session_id``, ``started_at``,
            ``finished_at``, ``total_wallets``, ``success_count``, ``fail_count``.
        """
        self._assert_open()
        sql = "SELECT * FROM sessions ORDER BY started_at DESC"
        try:
            async with self._db.execute(sql) as cur:
                rows = await cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise AnalyticsQueryError(f"Failed to list sessions: {exc}") from exc

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_csv(self, output_path: str) -> int:
        """Export the current session's transaction records to a CSV file.

        Wallet IDs in the export are already masked (as stored in the DB).
        No sensitive data leaves the database layer.

        Args:
            output_path: File path for the CSV output.

        Returns:
            Number of rows written.

        Raises:
            AnalyticsQueryError: If the database query fails.
        """
        self._assert_open()
        sql = """
            SELECT tx_hash, wallet_id, chain_id, nonce,
                   gas_price_gwei, gas_limit, gas_used,
                   value_wei, status, submitted_at, mined_at,
                   replacement_hash, session_id, strategy_type, strategy_id
              FROM transactions
             WHERE session_id = ?
             ORDER BY submitted_at ASC
        """
        try:
            async with self._db.execute(sql, (self._session_id,)) as cur:
                rows: list[Any] = list(await cur.fetchall())
        except Exception as exc:
            raise AnalyticsQueryError(f"Failed to fetch rows for CSV export: {exc}") from exc

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "tx_hash", "wallet_id", "chain_id", "nonce",
                    "gas_price_gwei", "gas_limit", "gas_used",
                    "value_wei", "status", "submitted_at", "mined_at",
                    "replacement_hash", "session_id", "strategy_type", "strategy_id",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        log.info(
            f"AnalyticsDB: exported {len(rows)} row(s) to '{output_path}'.",
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def get_daily_summary(self, date: str) -> dict[str, Any]:
        """Return P&L, trade count, and average gas summary for a given date.

        Used by the Telegram /report command.

        Args:
            date: ISO date string (e.g. ``"2026-03-25"``).

        Returns:
            Dict with keys: ``trade_count``, ``total_pnl_usd``, ``avg_gas_gwei``,
            ``total_gas_used``.
        """
        self._assert_open()
        # Convert date to Unix epoch range
        import datetime as _dt
        try:
            start_dt = _dt.datetime.fromisoformat(date)
        except ValueError:
            start_dt = _dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_ts = start_dt.timestamp()
        end_ts = start_ts + 86400.0

        sql = """
            SELECT
                COUNT(*) AS trade_count,
                COALESCE(AVG(gas_price_gwei), 0.0) AS avg_gas_gwei,
                COALESCE(SUM(gas_used), 0) AS total_gas_used
            FROM transactions
            WHERE submitted_at >= ? AND submitted_at < ?
              AND status IN ('MINED', 'SUCCESS')
        """
        try:
            async with self._db.execute(sql, (start_ts, end_ts)) as cur:
                row = await cur.fetchone()
            if row:
                return {
                    "trade_count": row["trade_count"] or 0,
                    "total_pnl_usd": 0.0,  # PnL requires price data; placeholder
                    "avg_gas_gwei": float(row["avg_gas_gwei"] or 0.0),
                    "total_gas_used": int(row["total_gas_used"] or 0),
                }
        except Exception as exc:
            log.warning("get_daily_summary failed: %s", exc)
        return {"trade_count": 0, "total_pnl_usd": 0.0, "avg_gas_gwei": 0.0, "total_gas_used": 0}

    async def get_strategy_window_performance(
        self,
        strategy: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Return performance metrics for a strategy over the last N days.

        Used by the Telegram /report command.

        Args:
            strategy: Strategy identifier (e.g. ``"ALGO"``, ``"YIELD"``).
            days: Number of days to look back.

        Returns:
            Dict with keys: ``trade_count``, ``avg_gas_gwei``, ``total_gas_used``,
            ``mined_count``.
        """
        self._assert_open()
        import time as _time
        cutoff = _time.time() - days * 86400.0

        sql = """
            SELECT
                COUNT(*) AS trade_count,
                SUM(status IN ('MINED', 'SUCCESS')) AS mined_count,
                COALESCE(AVG(gas_price_gwei), 0.0) AS avg_gas_gwei,
                COALESCE(SUM(gas_used), 0) AS total_gas_used
            FROM transactions
            WHERE submitted_at >= ?
              AND (strategy_type = ? OR strategy_id LIKE ?)
        """
        try:
            async with self._db.execute(
                sql, (cutoff, strategy, f"%{strategy.lower()}%")
            ) as cur:
                row = await cur.fetchone()
            if row:
                return {
                    "trade_count": row["trade_count"] or 0,
                    "mined_count": row["mined_count"] or 0,
                    "avg_gas_gwei": float(row["avg_gas_gwei"] or 0.0),
                    "total_gas_used": int(row["total_gas_used"] or 0),
                }
        except Exception as exc:
            log.warning("get_strategy_performance failed: %s", exc)
        return {"trade_count": 0, "mined_count": 0, "avg_gas_gwei": 0.0, "total_gas_used": 0}

    def _assert_open(self) -> None:
        """Raise if the database connection is not open.

        Raises:
            RuntimeError: If called before :meth:`open`.
        """
        if self._conn is None:
            raise RuntimeError(
                "AnalyticsDB is not open. Use it as an async context manager "
                "or call open() first."
            )

    @property
    def _db(self) -> aiosqlite.Connection:
        """Return the open connection; raises if not yet open."""
        assert self._conn is not None, "AnalyticsDB not open; call open() first"
        return self._conn


def _mask_wallet_id(wallet_id: str) -> str:
    """Reduce a wallet ID / address to a safe stored form.

    Keeps the first 6 and last 4 characters; masks the middle with ``****``.
    For IDs shorter than 12 characters the full string is returned unchanged.

    Args:
        wallet_id: Raw wallet identifier or Ethereum address.

    Returns:
        Masked string safe for database storage and log output.
    """
    if len(wallet_id) < 12:
        return wallet_id
    return f"{wallet_id[:6]}****{wallet_id[-4:]}"
