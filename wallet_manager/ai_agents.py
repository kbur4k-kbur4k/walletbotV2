"""
wallet_manager/ai_agents.py

Phase 8: Agentic AI Layer
==========================
Specialized AI agent roles for the WalletBot system.

Architecture:
- Each agent has a focused responsibility
- Agents produce structured text outputs (reports/alerts)
- No agent places orders directly
- All agents can be triggered manually or on a schedule

Agents:
1. ResearchAgent     — weekly edge analysis from paper ledger
2. RiskAgent         — guardrail review, anomaly warnings
3. NewsAgent         — market event summaries from macro calendar
4. TradeReviewAgent  — analyse recent wins/losses
5. OpsAgent          — detect health/log anomalies

Note: This module provides the framework and data-collection layer.
LLM API integration (gpt-4.1-mini via GitHub Models) is optional
and gracefully degrades to rule-based outputs when GITHUB_TOKEN is not set.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_OPENAI_AVAILABLE = False
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


@dataclass
class AgentReport:
    agent_name: str
    title: str
    body: str
    generated_at: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    used_llm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "title": self.title,
            "body": self.body,
            "generated_at": self.generated_at,
            "used_llm": self.used_llm,
        }

    def to_text(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self.generated_at))
        return f"[{self.agent_name}] {self.title}\nGenerated: {ts}\n\n{self.body}"


class BaseAgent:
    """Base class for all WalletBot AI agents."""

    name: str = "BaseAgent"
    description: str = ""

    _GITHUB_MODELS_BASE = "https://models.inference.ai.azure.com"

    def __init__(self) -> None:
        self._api_key = os.getenv("GITHUB_TOKEN", "")
        self._model = "gpt-4.1-mini"
        self._max_tokens = 1024

    def _has_llm(self) -> bool:
        return _OPENAI_AVAILABLE and bool(self._api_key)

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        """Call GitHub Models (gpt-4.1-mini) via OpenAI SDK. Returns empty string on failure."""
        if not self._has_llm():
            return ""
        try:
            client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._GITHUB_MODELS_BASE,
            )
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return str(response.choices[0].message.content) if response.choices else ""
        except Exception as exc:
            log.warning(f"{self.name}: LLM call failed: {exc}")
            return ""


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent):
    """Weekly edge analysis from paper trading data."""

    name = "ResearchAgent"
    description = "Analyzes paper trading data to assess edge quality and suggest improvements."

    async def run(self, paper_engine: Any | None = None) -> AgentReport:
        from wallet_manager.research_brain import ResearchBrain

        brain = ResearchBrain(paper_engine=paper_engine)
        weekly = await brain.weekly_strategy_review()
        threshold_report = await brain.threshold_tuning_suggestions()

        data = weekly.to_dict()
        base_body = weekly.to_text()

        # Optionally enhance with LLM
        llm_used = False
        llm_body = ""
        if self._has_llm():
            system = (
                "You are a quantitative trading analyst reviewing a paper trading bot's weekly performance. "
                "Be concise, data-driven, and constructive. Focus on what the numbers mean, "
                "not just what they are. Maximum 300 words."
            )
            user = f"Here is the weekly performance summary:\n\n{base_body}\n\nKey tuning suggestions:\n{threshold_report.to_text()}"
            llm_body = self._call_llm(system, user)
            llm_used = bool(llm_body)

        final_body = llm_body if llm_body else base_body
        return AgentReport(
            agent_name=self.name,
            title="Weekly Strategy Review",
            body=final_body,
            data=data,
            used_llm=llm_used,
        )


# ---------------------------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------------------------


class RiskAgent(BaseAgent):
    """Reviews risk committee state and issues guardrail warnings."""

    name = "RiskAgent"
    description = "Monitors risk brakes and warns when the system approaches danger thresholds."

    async def run(self, paper_engine: Any | None = None) -> AgentReport:
        from wallet_manager.risk_committee import get_risk_committee

        committee = get_risk_committee()
        verdict = await committee.evaluate(paper_engine=paper_engine)

        data = verdict.to_dict()
        base_body = (
            f"Risk Score: {verdict.risk_score}/100\n"
            f"Verdict: {verdict.overall_verdict.value}\n"
            f"Narrative: {verdict.narrative}\n\n"
        )
        if verdict.blocking_brakes:
            base_body += f"BLOCKING BRAKES ACTIVE: {', '.join(verdict.blocking_brakes)}\n"
        for r in verdict.brake_results:
            base_body += f"  {r.brake_name}: {r.verdict.value} — {r.reason}\n"

        llm_used = False
        llm_body = ""
        if self._has_llm() and verdict.risk_score > 30:
            system = (
                "You are a risk manager for a crypto trading bot. "
                "Review the risk committee output and explain the risk posture in plain language. "
                "Be specific about which risks are most concerning and why. Max 200 words."
            )
            llm_body = self._call_llm(system, base_body)
            llm_used = bool(llm_body)

        return AgentReport(
            agent_name=self.name,
            title=f"Risk Report (Score: {verdict.risk_score}/100)",
            body=llm_body if llm_body else base_body,
            data=data,
            used_llm=llm_used,
        )


# ---------------------------------------------------------------------------
# News Agent
# ---------------------------------------------------------------------------


class NewsAgent(BaseAgent):
    """Summarises upcoming macro events and their potential market impact."""

    name = "NewsAgent"
    description = "Scans the macro calendar and summarises upcoming high-impact events."

    def run(self) -> AgentReport:
        from wallet_manager.market_intelligence import get_macro_calendar

        calendar = get_macro_calendar()
        calendar_data = calendar.to_dict()
        upcoming = calendar.upcoming_events(hours_ahead=72.0)
        imminent = calendar.get_imminent_events(window_hours=4.0)

        lines: list[str] = []
        if imminent:
            lines.append(f"⚠️  IMMINENT EVENTS ({len(imminent)}):")
            for e in imminent:
                lines.append(f"  • {e.name} @ {e.dt_utc.strftime('%H:%M UTC')} — {e.impact.value}")
        if upcoming:
            lines.append("\nUpcoming (next 72h):")
            for e in upcoming[:8]:
                lines.append(f"  • {e.name} @ {e.dt_utc.strftime('%Y-%m-%d %H:%M UTC')} [{e.impact.value}]")
        else:
            lines.append("No high-impact events in the next 72 hours.")

        base_body = "\n".join(lines)

        llm_used = False
        llm_body = ""
        if self._has_llm() and upcoming:
            system = (
                "You are a market analyst for a crypto trading bot. "
                "Briefly explain which upcoming macro events are most relevant for ETH/USDC trading "
                "and what posture the bot should take. Max 150 words."
            )
            llm_body = self._call_llm(system, base_body)
            llm_used = bool(llm_body)

        return AgentReport(
            agent_name=self.name,
            title="Macro Event Calendar Summary",
            body=llm_body if llm_body else base_body,
            data=calendar_data,
            used_llm=llm_used,
        )


# ---------------------------------------------------------------------------
# Trade Review Agent
# ---------------------------------------------------------------------------


class TradeReviewAgent(BaseAgent):
    """Analyses recent wins and losses to identify patterns."""

    name = "TradeReviewAgent"
    description = "Reviews recent paper trades and identifies what differentiated wins from losses."

    async def run(self, paper_engine: Any | None = None) -> AgentReport:
        from wallet_manager.research_brain import ResearchBrain

        brain = ResearchBrain(paper_engine=paper_engine)
        postmortem = await brain.worst_trade_postmortem(n=3)
        best_setups = await brain.best_setup_summary()

        base_body = postmortem.to_text() + "\n\n" + best_setups.to_text()
        data = {
            "postmortem": postmortem.to_dict(),
            "best_setups": best_setups.to_dict(),
        }

        llm_used = False
        llm_body = ""
        if self._has_llm():
            system = (
                "You are a trading coach reviewing a paper bot's recent trades. "
                "Identify patterns in the worst trades and what made the best setups work. "
                "Be specific and actionable. Max 250 words."
            )
            llm_body = self._call_llm(system, base_body)
            llm_used = bool(llm_body)

        return AgentReport(
            agent_name=self.name,
            title="Trade Review: Wins, Losses, Patterns",
            body=llm_body if llm_body else base_body,
            data=data,
            used_llm=llm_used,
        )


# ---------------------------------------------------------------------------
# Ops Agent
# ---------------------------------------------------------------------------


class OpsAgent(BaseAgent):
    """Detects health and log anomalies in the runtime state."""

    name = "OpsAgent"
    description = "Monitors system health: session gaps, error rates, stale data, and anomalies."

    async def run(self, paper_engine: Any | None = None) -> AgentReport:
        issues: list[str] = []
        data: dict[str, Any] = {}

        # Check last trade age
        if paper_engine:
            conn = paper_engine._ledger._get_conn()
            async with conn.execute(
                "SELECT MAX(timestamp) AS last_ts FROM paper_trades"
            ) as cur:
                row = await cur.fetchone()
            last_ts = float(row["last_ts"]) if row and row["last_ts"] else 0.0
            age_h = (time.time() - last_ts) / 3600 if last_ts > 0 else 99.0
            data["last_trade_age_hours"] = round(age_h, 1)
            if age_h > 24:
                issues.append(f"No trade in {age_h:.1f}h — bot may be stuck or in permanent macro-event avoidance.")
            elif age_h > 6:
                issues.append(f"No trade in {age_h:.1f}h — check if regime filter is blocking all signals.")

            # Check decision journal health
            async with conn.execute(
                "SELECT COUNT(*) AS cnt FROM paper_decisions WHERE timestamp > ?",
                (time.time() - 3600,),
            ) as cur:
                row2 = await cur.fetchone()
            recent_decisions = int(row2["cnt"]) if row2 else 0
            data["decisions_last_hour"] = recent_decisions
            if recent_decisions == 0:
                issues.append("Zero decisions journaled in the last hour — orchestrator may not be running.")
            elif recent_decisions < 3:
                issues.append(f"Only {recent_decisions} decisions last hour — execution cycle may be slow.")

        if not issues:
            issues.append("All systems nominal. No anomalies detected.")

        base_body = "\n".join(f"• {i}" for i in issues)

        return AgentReport(
            agent_name=self.name,
            title="Ops Health Check",
            body=base_body,
            data=data,
            used_llm=False,
        )


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Central registry for all WalletBot AI agents."""

    def __init__(self) -> None:
        self.research = ResearchAgent()
        self.risk = RiskAgent()
        self.news = NewsAgent()
        self.trade_review = TradeReviewAgent()
        self.ops = OpsAgent()

    async def run_all(self, paper_engine: Any | None = None) -> list[AgentReport]:
        """Run all agents and return their reports."""
        reports: list[AgentReport] = []
        for agent_coro in [
            self.ops.run(paper_engine),
            self.risk.run(paper_engine),
            self.research.run(paper_engine),
            self.trade_review.run(paper_engine),
        ]:
            try:
                report = await agent_coro
                reports.append(report)
            except Exception as exc:
                log.warning(f"Agent run failed: {exc}")

        try:
            reports.append(self.news.run())
        except Exception as exc:
            log.warning(f"NewsAgent failed: {exc}")

        return reports

    async def run_all_to_dict(self, paper_engine: Any | None = None) -> list[dict[str, Any]]:
        reports = await self.run_all(paper_engine)
        return [r.to_dict() for r in reports]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Return the module-level AgentRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
