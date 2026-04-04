"""
wallet_manager/reporting.py

Sprint JS-7.2: Automated Performance Reports
===============================================
Daily   — equity, trades, PnL, top skip reasons (auto every 24h)
Weekly  — full strategy review + IC per signal + regime + experiments (auto every 7 days)
Monthly — Sharpe/Calmar trends, feature importance shifts (manual trigger /api/monthly_report)

All reports delivered via Telegram notifier and available at /api/reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from wallet_manager.core.models import EventKind, NotificationEvent
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


async def _send(notifier: Any, text: str) -> None:
    """Send a text message via notifier using the correct API."""
    if notifier is None:
        return
    if hasattr(notifier, "send_message"):
        await notifier.send_message(text)
    elif hasattr(notifier, "send"):
        await notifier.send(NotificationEvent(kind=EventKind.SYSTEM_ALERT, detail=text[:4000]))

_DAILY_INTERVAL_S = 24 * 3600.0
_WEEKLY_INTERVAL_S = 7 * 24 * 3600.0


@dataclass
class ReportRecord:
    report_type: str
    generated_at: float
    summary: str
    delivered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.report_type,
            "generated_at": self.generated_at,
            "summary": self.summary[:300],
            "delivered": self.delivered,
            "age_hours": round((time.time() - self.generated_at) / 3600, 1),
        }


class PerformanceReporter:
    """
    Generates and delivers automated performance reports.
    Call `tick(notifier, paper_engine)` from a background task every hour.
    """

    def __init__(self) -> None:
        self._last_daily: float = 0.0
        self._last_weekly: float = 0.0
        self._history: list[ReportRecord] = []
        self._max_history = 50

    async def tick(self, notifier: Any, paper_engine: Any | None = None) -> None:
        """Call every hour. Delivers reports when intervals elapsed."""
        now = time.time()
        if now - self._last_daily >= _DAILY_INTERVAL_S:
            await self._deliver_daily(notifier, paper_engine)
            self._last_daily = now
        if now - self._last_weekly >= _WEEKLY_INTERVAL_S:
            await self._deliver_weekly(notifier, paper_engine)
            self._last_weekly = now

    async def _deliver_daily(self, notifier: Any, paper_engine: Any | None) -> None:
        summary = await self._build_daily_summary(paper_engine)
        record = ReportRecord("daily", time.time(), summary)
        if notifier:
            try:
                msg = f"Daily Report {time.strftime('%Y-%m-%d', time.gmtime())}\n\n{summary}"
                await _send(notifier, msg)
                record.delivered = True
                log.info("Daily report delivered via Telegram.")
            except Exception as exc:
                log.warning(f"Daily report delivery failed: {exc}")
        self._store(record)

    async def _deliver_weekly(self, notifier: Any, paper_engine: Any | None) -> None:
        summary = await self._build_weekly_summary(paper_engine)
        record = ReportRecord("weekly", time.time(), summary)
        if notifier:
            chunks = self._chunk(summary, 3800)
            try:
                for i, chunk in enumerate(chunks):
                    header = f"Weekly Report {time.strftime('%Y-%m-%d', time.gmtime())}" if i == 0 else "(cont'd)"
                    await _send(notifier, f"{header}\n\n{chunk}")
                record.delivered = True
                log.info("Weekly report delivered via Telegram.")
            except Exception as exc:
                log.warning(f"Weekly report delivery failed: {exc}")
        self._store(record)

    async def generate_monthly(self, notifier: Any, paper_engine: Any | None) -> str:
        """Manually triggered monthly report. Returns summary string."""
        summary = await self._build_monthly_summary(paper_engine)
        record = ReportRecord("monthly", time.time(), summary)
        if notifier:
            try:
                chunks = self._chunk(summary, 3800)
                for i, chunk in enumerate(chunks):
                    header = f"Monthly Report {time.strftime('%Y-%m', time.gmtime())}" if i == 0 else "(cont'd)"
                    await _send(notifier, f"{header}\n\n{chunk}")
                record.delivered = True
            except Exception as exc:
                log.warning(f"Monthly report delivery failed: {exc}")
        self._store(record)
        return summary

    async def _build_daily_summary(self, paper_engine: Any | None) -> str:
        lines: list[str] = []
        if not paper_engine:
            return "Engine not available."
        try:
            ledger = paper_engine._ledger
            pnl = await ledger.get_pnl_summary()
            adv = await ledger.get_advanced_analytics(window_seconds=86400.0)
            conn = ledger._get_conn()
            async with conn.execute(
                "SELECT primary_reason, COUNT(*) as cnt FROM paper_decisions "
                "WHERE timestamp > ? GROUP BY primary_reason ORDER BY cnt DESC LIMIT 3",
                (time.time() - 86400,),
            ) as cur:
                skip_rows = await cur.fetchall()
            lines.append(f"Equity: ${pnl.get('current_equity_usd', 0):.2f}")
            lines.append(f"Daily PnL: ${pnl.get('total_realized_pnl_usd', 0):+.4f}")
            lines.append(f"Trades today: {adv.get('closed_count', 0)} | Win rate: {adv.get('win_rate_pct', 0):.1f}%")
            lines.append(f"Expectancy: ${adv.get('expectancy_usd', 0):+.4f}")
            if skip_rows:
                lines.append("Top skip reasons:")
                for r in skip_rows:
                    lines.append(f"  {r['primary_reason']}: {r['cnt']}")
        except Exception as exc:
            lines.append(f"Error collecting daily data: {exc}")
        return "\n".join(lines)

    async def _build_weekly_summary(self, paper_engine: Any | None) -> str:
        lines: list[str] = []
        if not paper_engine:
            return "Engine not available."
        try:
            from wallet_manager.research_brain import ResearchBrain
            brain = ResearchBrain(paper_engine=paper_engine)
            weekly = await brain.weekly_strategy_review()
            attribution = await brain.pnl_attribution()
            lines.append("=== WEEKLY STRATEGY REVIEW ===")
            lines.append(weekly.to_text())
            lines.append("\n=== P&L ATTRIBUTION ===")
            for section in attribution.sections[:2]:
                lines.append(f"[{section['heading']}]")
                lines.append(section.get("body", ""))
            from wallet_manager.signal_intelligence import get_signal_intelligence
            ic = get_signal_intelligence().ic_tracker.get_ic_report()
            if ic["signals"]:
                lines.append("\n=== IC PER SIGNAL ===")
                for s in ic["signals"][:5]:
                    lines.append(f"  {s['signal']}: IC={s['ic']:.4f} [{s['status']}]")
            from wallet_manager.risk_intelligence import get_risk_intelligence
            stress = get_risk_intelligence().stress_engine.to_dict()
            if stress.get("status") == "OK":
                lines.append("\n=== STRESS SCENARIOS ===")
                lines.append(f"All pass: {stress['all_pass']} | Worst: {stress.get('worst_scenario', '?')}")
        except Exception as exc:
            lines.append(f"Error collecting weekly data: {exc}")
        return "\n".join(lines)

    async def _build_monthly_summary(self, paper_engine: Any | None) -> str:
        lines: list[str] = []
        if not paper_engine:
            return "Engine not available."
        try:
            ledger = paper_engine._ledger
            adv = await ledger.get_advanced_analytics()
            hold = await ledger.get_hold_time_stats()
            lines.append("=== MONTHLY PERFORMANCE ===")
            lines.append(f"Closed trades: {adv.get('closed_count', 0)}")
            lines.append(f"Win rate: {adv.get('win_rate_pct', 0):.2f}%")
            lines.append(f"Sharpe: {adv.get('sharpe_ratio', 0):.3f}")
            lines.append(f"Sortino: {adv.get('sortino_ratio', 0):.3f}")
            lines.append(f"Calmar: {adv.get('calmar_ratio', 0):.3f}")
            lines.append(f"Expectancy: ${adv.get('expectancy_usd', 0):+.4f}")
            lines.append(f"Profit factor: {adv.get('profit_factor', 0):.3f}")
            lines.append(f"Avg hold: {hold.get('avg_hold_human', '?')}")
            from wallet_manager.signal_intelligence import get_signal_intelligence
            si = get_signal_intelligence()
            div = si.diversity_score.get_report()
            lines.append("\n=== MODEL DIVERSITY ===")
            lines.append(f"Diversity score: {div['diversity_score']:.3f} [{div['status']}]")
            corr = si.correlation_matrix.get_matrix()
            if corr.get("redundant_pairs"):
                lines.append(f"Redundant signal pairs: {len(corr['redundant_pairs'])}")
        except Exception as exc:
            lines.append(f"Error collecting monthly data: {exc}")
        return "\n".join(lines)

    def _store(self, record: ReportRecord) -> None:
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_daily_ago_h": round((time.time() - self._last_daily) / 3600, 1) if self._last_daily else None,
            "last_weekly_ago_h": round((time.time() - self._last_weekly) / 3600, 1) if self._last_weekly else None,
            "history": [r.to_dict() for r in self._history[-5:]],
        }

    @staticmethod
    def _escape_md(text: str) -> str:
        for ch in [".", "-", "+", "(", ")", "!", "#", "|", ">"]:
            text = text.replace(ch, f"\\{ch}")
        return text

    @staticmethod
    def _chunk(text: str, size: int) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for line in lines:
            if length + len(line) + 1 > size:
                chunks.append("\n".join(current))
                current = [line]
                length = len(line)
            else:
                current.append(line)
                length += len(line) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks or [""]


_reporter: PerformanceReporter | None = None


def get_reporter() -> PerformanceReporter:
    global _reporter
    if _reporter is None:
        _reporter = PerformanceReporter()
    return _reporter
