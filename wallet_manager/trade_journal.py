"""
wallet_manager/trade_journal.py

Structured Per-Trade Audit Log.

Appends a JSON line per completed trade to cfg.trade_journal_path
(default: trade_journal.jsonl). Rotates when cfg.trade_journal_max_mb
(default 100 MB) is reached. All writes are async (aiofiles).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


def _mask(value: str, prefix: int = 6, suffix: int = 4) -> str:
    """Mask a string, showing only prefix and suffix characters."""
    if not value or len(value) <= prefix + suffix:
        return value
    return f"{value[:prefix]}...{value[-suffix:]}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    timestamp_utc: str
    wallet_id: str          # masked: first6...last4
    strategy_type: str
    action: str
    entry_price: float
    exit_price: float
    position_usd: float
    pnl_usd: float
    pnl_pct: float
    execution_mode: str
    regime: str
    alpha_score: float
    confidence: float
    garch_sigma: float
    kelly_fraction_used: float
    slippage_bps: float
    gas_cost_usd: float
    dex_venue: str
    tx_hash: str            # masked: first6...last4
    drawdown_at_entry: float
    drawdown_at_exit: float
    ml_predictions: dict    # {xgb, lgbm, lstm, transformer, ensemble}


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------


class TradeJournal:
    """Appends a JSON line per completed trade to cfg.trade_journal_path.

    Rotates when cfg.trade_journal_max_mb is reached.
    All writes are async (aiofiles).
    """

    def __init__(
        self,
        journal_path: str | None = None,
        max_mb: float | None = None,
    ) -> None:
        self._rotation_index: int = 0
        self._journal_path: str | None = journal_path
        self._max_mb: float | None = max_mb
        # Expose _max_bytes so tests can override it directly
        self._max_bytes: int | None = int(max_mb * 1024 * 1024) if max_mb is not None else None

    def _get_path(self) -> Path:
        if self._journal_path is not None:
            return Path(self._journal_path)
        from wallet_manager.config import get_settings
        return Path(get_settings().trade_journal_path)

    def _get_max_bytes(self) -> int:
        if self._max_bytes is not None:
            return self._max_bytes
        if self._max_mb is not None:
            return int(self._max_mb * 1024 * 1024)
        from wallet_manager.config import get_settings
        return int(get_settings().trade_journal_max_mb * 1024 * 1024)

    def _rotated_path(self) -> Path:
        base = self._get_path()
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._rotation_index += 1
        return base.parent / f"trade_journal.{ts}.{self._rotation_index}.jsonl"

    def _rotate_sync_if_needed(self) -> None:
        path = self._get_path()
        max_bytes = self._get_max_bytes()
        try:
            if path.exists() and path.stat().st_size >= max_bytes:
                rotated = self._rotated_path()
                import os as _os
                _os.replace(str(path), str(rotated))
                log.info("TradeJournal: rotated to %s", rotated)
        except Exception as exc:
            log.warning("TradeJournal: rotation check failed: %s", exc)

    async def _rotate_if_needed(self) -> None:
        await asyncio.to_thread(self._rotate_sync_if_needed)

    @staticmethod
    def _write_line_sync(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _read_lines_sync(path: Path) -> list[str]:
        with path.open(encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def record_trade(
        self,
        wallet_id: str | TradeRecord | None = None,
        strategy: str | None = None,
        direction: str | None = None,
        amount_usd: float | None = None,
        tx_hash: str | None = None,
        extra: dict | None = None,
        trade: TradeRecord | None = None,
    ) -> None:
        """Append a JSON line for the completed trade.

        Accepts either a full TradeRecord object (legacy) or simple keyword args.
        """
        if isinstance(wallet_id, TradeRecord) and trade is None:
            trade = wallet_id
            wallet_id = None

        self._rotate_sync_if_needed()
        path = self._get_path()

        if trade is not None:
            record_dict = asdict(trade)
            record_dict["wallet_id"] = _mask(str(trade.wallet_id))
            record_dict["tx_hash"] = _mask(str(trade.tx_hash))
        else:
            record_dict = {
                "wallet_id": _mask(str(wallet_id or "")),
                "strategy": strategy,
                "direction": direction,
                "amount_usd": amount_usd,
                "tx_hash": _mask(str(tx_hash or "")),
            }
            if extra:
                # Mask any values that look like keys/secrets
                for k, v in extra.items():
                    if "key" in k.lower() or "secret" in k.lower() or "private" in k.lower():
                        record_dict[k] = "[REDACTED]"
                    else:
                        record_dict[k] = v

        line = json.dumps(record_dict, ensure_ascii=False)
        try:
            self._write_line_sync(path, line)
        except Exception as exc:
            log.warning("TradeJournal: write failed: %s", exc)

    def get_recent_trades(self, n: int = 50) -> list[dict]:
        """Return the last n trade records from the journal file (most recent first)."""
        path = self._get_path()
        if not path.exists():
            return []
        try:
            lines = self._read_lines_sync(path)
            recent = []
            for line in reversed(lines):
                with contextlib.suppress(json.JSONDecodeError):
                    recent.append(json.loads(line))
                if len(recent) >= n:
                    break
            return recent
        except Exception as exc:
            log.warning("TradeJournal: get_recent_trades failed: %s", exc)
            return []

    async def record_trade_async(self, trade: TradeRecord) -> None:
        """Async variant for legacy callers using full TradeRecord."""
        await self._rotate_if_needed()
        path = self._get_path()

        # Mask sensitive fields
        record_dict = asdict(trade)
        record_dict["wallet_id"] = _mask(trade.wallet_id)
        record_dict["tx_hash"] = _mask(trade.tx_hash)

        line = json.dumps(record_dict, ensure_ascii=False)
        try:
            await asyncio.to_thread(self._write_line_sync, path, line)
        except Exception as exc:
            log.warning("TradeJournal: write failed: %s", exc)

    async def get_recent_trades_async(self, n: int = 50) -> list[dict]:
        """Async variant: return the last n trade records from the journal file."""
        path = self._get_path()
        if not path.exists():
            return []

        try:
            lines = await asyncio.to_thread(self._read_lines_sync, path)
        except Exception as exc:
            log.warning("TradeJournal: read failed: %s", exc)
            return []

        records: list[dict] = []
        for raw in lines[-n:]:
            with contextlib.suppress(json.JSONDecodeError):
                records.append(json.loads(raw))
        return records


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_journal_instance: TradeJournal | None = None


def get_trade_journal() -> TradeJournal:
    """Return the process-wide TradeJournal singleton."""
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = TradeJournal()
    return _journal_instance
