"""
wallet_manager/tick_collector.py

Sprint JS-4.4: Tick Data Collector
=====================================
Accumulates tick-level price data (timestamp_ns, price, direction) in SQLite.
Retains last 30 days. Enables order flow features once sufficient ticks accumulate.

Non-blocking: record_tick() is synchronous and fast (<1ms).
Async query methods for analytics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import aiosqlite

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_DEFAULT_TICK_DB = Path("data/ticks.db")
_RETENTION_DAYS = 30


class TickCollector:
    """
    Stores tick-level price updates in a SQLite WAL-mode database.
    Schema: ticks(timestamp_ns INTEGER, price REAL, direction INTEGER)
    direction: +1=uptick, -1=downtick, 0=unchanged
    """

    def __init__(self, db_path: Path = _DEFAULT_TICK_DB) -> None:
        self._db_path = db_path
        self._last_price: float = 0.0
        self._tick_count: int = 0
        self._initialized: bool = False
        self._pending: list[tuple[int, float, int]] = []
        self._flush_interval: int = 50

    def _ensure_dir(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        self._ensure_dir()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "CREATE TABLE IF NOT EXISTS ticks ("
                "timestamp_ns INTEGER NOT NULL,"
                "price REAL NOT NULL,"
                "direction INTEGER NOT NULL"
                ")"
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(timestamp_ns)")
            await db.commit()
        self._initialized = True
        log.info(f"TickCollector initialized at {self._db_path}")

    def record_tick(self, price: float, spread_bps: float = 0.0) -> None:
        """Record a price update. Call every time a new price is observed."""
        if price <= 0:
            return
        direction = 0
        if self._last_price > 0:
            if price > self._last_price:
                direction = 1
            elif price < self._last_price:
                direction = -1
        self._last_price = price
        ts_ns = int(time.time() * 1e9)
        self._pending.append((ts_ns, price, direction))
        self._tick_count += 1

        # Bridge: also feed OrderFlowImbalance for real-time in-memory OFI
        try:
            from wallet_manager.strategies.order_flow import get_order_flow_imbalance
            get_order_flow_imbalance().record_tick(price, spread_bps)
        except Exception:
            pass

    def get_ofi_features(self) -> dict[str, float]:
        """Return real-time OFI features from the in-memory order flow tracker."""
        try:
            from wallet_manager.strategies.order_flow import get_order_flow_imbalance
            return get_order_flow_imbalance().get_features()
        except Exception:
            return {"ofi": 0.0, "ofi_momentum": 0.0, "spread_signal": 0.0, "ofi_composite": 0.0}

    async def flush(self) -> int:
        """Flush pending ticks to SQLite. Returns count written."""
        if not self._pending:
            return 0
        if not self._initialized:
            await self.initialize()
        to_write = self._pending[:]
        self._pending = []
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executemany("INSERT INTO ticks VALUES (?, ?, ?)", to_write)
            await db.commit()
        return len(to_write)

    async def maybe_flush(self) -> None:
        """Flush if pending buffer exceeds threshold."""
        if len(self._pending) >= self._flush_interval:
            await self.flush()

    async def prune_old_ticks(self) -> int:
        """Delete ticks older than _RETENTION_DAYS. Returns rows deleted."""
        if not self._initialized:
            return 0
        cutoff_ns = int((time.time() - _RETENTION_DAYS * 86400) * 1e9)
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("DELETE FROM ticks WHERE timestamp_ns < ?", (cutoff_ns,))
            await db.commit()
            deleted = cursor.rowcount
        if deleted > 0:
            log.info(f"TickCollector pruned {deleted} old ticks.")
        return deleted

    async def get_recent_ticks(self, minutes: float = 60.0) -> list[dict[str, Any]]:
        """Return ticks from the last N minutes."""
        if not self._initialized:
            return []
        cutoff_ns = int((time.time() - minutes * 60) * 1e9)
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT timestamp_ns, price, direction FROM ticks WHERE timestamp_ns > ? ORDER BY timestamp_ns DESC LIMIT 5000",
                (cutoff_ns,),
            ) as cur:
                rows = await cur.fetchall()
        return [{"ts": r["timestamp_ns"], "price": r["price"], "direction": r["direction"]} for r in rows]

    async def get_ofi(self, minutes: float = 5.0) -> float:
        """
        Compute Order Flow Imbalance over last N minutes.
        OFI = (up_ticks - down_ticks) / total_ticks
        Returns 0.0 if insufficient data.
        """
        ticks = await self.get_recent_ticks(minutes)
        if len(ticks) < 10:
            return 0.0
        up = sum(1 for t in ticks if t["direction"] == 1)
        down = sum(1 for t in ticks if t["direction"] == -1)
        total = len(ticks)
        return round((up - down) / total, 4)

    async def get_stats(self) -> dict[str, Any]:
        """Return summary stats for the tick database."""
        if not self._initialized:
            return {"initialized": False, "total_ticks": 0, "session_ticks": self._tick_count}
        async with aiosqlite.connect(self._db_path) as db, db.execute(
            "SELECT COUNT(*) as cnt, MIN(timestamp_ns) as min_ts, MAX(timestamp_ns) as max_ts FROM ticks"
        ) as cur:
            row = await cur.fetchone()
        total = row[0] if row else 0
        min_ts = row[1] / 1e9 if row and row[1] else None
        max_ts = row[2] / 1e9 if row and row[2] else None
        span_days = (max_ts - min_ts) / 86400 if min_ts and max_ts else 0.0
        recent_ofi = await self.get_ofi(5.0)
        # Validation: OFI features need min 500 ticks, span >= 1 day
        ofi_ready = total >= 500 and span_days >= 1.0
        return {
            "initialized": True,
            "total_ticks_db": total,
            "session_ticks": self._tick_count,
            "pending_unflushed": len(self._pending),
            "span_days": round(span_days, 2),
            "ofi_5m": recent_ofi,
            "ofi_signal": "BUYING" if recent_ofi > 0.3 else "SELLING" if recent_ofi < -0.3 else "NEUTRAL",
            "ofi_ready": ofi_ready,
            "ofi_accumulation_status": "READY" if ofi_ready else f"ACCUMULATING ({int(total)}/500 ticks, {span_days:.1f}/1.0 days)",
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_collector: TickCollector | None = None


def get_tick_collector() -> TickCollector:
    global _collector
    if _collector is None:
        import os
        db_path = Path(os.getenv("TICK_DB_PATH", str(_DEFAULT_TICK_DB)))
        _collector = TickCollector(db_path)
    return _collector
