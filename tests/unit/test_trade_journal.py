"""tests/unit/test_trade_journal.py - TradeJournal tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wallet_manager.trade_journal import TradeJournal, TradeRecord


class TestTradeJournal:
    @pytest.fixture
    def journal(self, tmp_path: Path) -> TradeJournal:
        return TradeJournal(journal_path=str(tmp_path / "trades.jsonl"), max_mb=1.0)

    def test_record_trade_writes_line(self, journal: TradeJournal, tmp_path: Path) -> None:
        journal.record_trade(
            wallet_id="w1",
            strategy="ALGO",
            direction="BUY",
            amount_usd=100.0,
            tx_hash="0xabc",
        )
        lines = Path(tmp_path / "trades.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["wallet_id"] == "w1"
        assert data["strategy"] == "ALGO"

    def test_private_key_masked(self, journal: TradeJournal, tmp_path: Path) -> None:
        journal.record_trade(
            wallet_id="w1",
            strategy="ALGO",
            direction="BUY",
            amount_usd=50.0,
            tx_hash="0xdef",
            extra={"private_key": "0xsecretkey"},
        )
        content = Path(tmp_path / "trades.jsonl").read_text(encoding="utf-8")
        assert "0xsecretkey" not in content

    def test_get_recent_trades_count(self, journal: TradeJournal) -> None:
        for i in range(10):
            journal.record_trade(f"w{i}", "ALGO", "BUY", 10.0, f"0x{i:04x}")
        recent = journal.get_recent_trades(n=5)
        assert len(recent) == 5

    def test_get_recent_trades_order(self, journal: TradeJournal) -> None:
        for i in range(5):
            journal.record_trade("w1", "ALGO", "BUY", float(i) * 10, f"0x{i:04x}")
        recent = journal.get_recent_trades(n=3)
        assert recent[0]["amount_usd"] == pytest.approx(40.0)

    def test_rotation_on_size_limit(self, journal: TradeJournal, tmp_path: Path) -> None:
        journal._max_bytes = 100
        for i in range(20):
            journal.record_trade("w1", "ALGO", "BUY", 100.0, f"0x{i:08x}")
        current_size = Path(tmp_path / "trades.jsonl").stat().st_size
        assert current_size <= 100 * 5

    @pytest.mark.asyncio
    async def test_record_trade_async_writes_trade_record(self, journal: TradeJournal) -> None:
        trade = TradeRecord(
            timestamp_utc="2026-04-02T00:00:00Z",
            wallet_id="wallet_primary_01",
            strategy_type="ALGO",
            action="SELL",
            entry_price=100.0,
            exit_price=110.0,
            position_usd=50.0,
            pnl_usd=10.0,
            pnl_pct=0.10,
            execution_mode="robust_paper",
            regime="normal",
            alpha_score=0.8,
            confidence=0.7,
            garch_sigma=0.02,
            kelly_fraction_used=0.25,
            slippage_bps=4.0,
            gas_cost_usd=0.5,
            dex_venue="uniswap",
            tx_hash="0xabcdef1234567890",
            drawdown_at_entry=0.01,
            drawdown_at_exit=0.02,
            ml_predictions={"ensemble": 0.8},
        )

        await journal.record_trade_async(trade)

        recent = journal.get_recent_trades(1)
        assert recent[0]["strategy_type"] == "ALGO"
        assert recent[0]["wallet_id"] != "wallet_primary_01"
