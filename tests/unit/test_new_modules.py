"""
tests/unit/test_new_modules.py

Unit tests for Phase 4–10 modules:
  - wallet_manager/market_intelligence.py  (Phase 5: MacroCalendar)
  - wallet_manager/risk_committee.py        (Phase 6: RiskCommittee)
  - wallet_manager/experiment_lab.py        (Phase 7: ExperimentLab)
  - wallet_manager/research_brain.py        (Phase 4: ResearchBrain)
  - wallet_manager/ai_agents.py             (Phase 8: AgentRegistry)
  - wallet_manager/continuous_learning.py   (Phase 10: ContinuousLearner)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Phase 5: MacroCalendar
# ---------------------------------------------------------------------------


class TestMacroCalendar:
    def setup_method(self) -> None:
        from wallet_manager.market_intelligence import MacroCalendar
        self.cal = MacroCalendar()

    def test_initial_events_loaded(self) -> None:
        assert len(self.cal._events) > 0

    def test_add_custom_event(self) -> None:
        from wallet_manager.market_intelligence import EventCategory, ImpactLevel
        before = len(self.cal._events)
        self.cal.add_event(
            name="Test Event",
            timestamp_utc=time.time() + 3600,
            impact=ImpactLevel.HIGH,
            category=EventCategory.OTHER,
        )
        assert len(self.cal._events) == before + 1

    def test_should_suppress_alpha_no_imminent(self) -> None:
        # No events within 2 hours → no suppression
        from wallet_manager.market_intelligence import MacroCalendar
        fresh_cal = MacroCalendar.__new__(MacroCalendar)
        fresh_cal._events = []
        suppress, reason = fresh_cal.should_suppress_alpha()
        assert suppress is False
        assert reason == ""

    def test_should_suppress_alpha_imminent_high_event(self) -> None:
        from wallet_manager.market_intelligence import (
            EventCategory,
            ImpactLevel,
            MacroCalendar,
        )
        cal = MacroCalendar.__new__(MacroCalendar)
        cal._events = []
        cal.add_event(
            name="FOMC Rate Decision",
            timestamp_utc=time.time() + 60,   # 1 minute away
            impact=ImpactLevel.CRITICAL,
            category=EventCategory.MONETARY_POLICY,
        )
        suppress, reason = cal.should_suppress_alpha()
        assert suppress is True
        assert "FOMC" in reason or "NEWS_EVENT_IMMINENT" in reason

    def test_to_dict_keys(self) -> None:
        d = self.cal.to_dict()
        assert "total_events" in d
        assert "next_48h" in d
        assert "imminent_high_plus" in d

    def test_upcoming_events_sorted(self) -> None:
        from wallet_manager.market_intelligence import EventCategory, ImpactLevel
        self.cal._events = []
        t = time.time()
        self.cal.add_event("Later", t + 7200, ImpactLevel.MEDIUM, EventCategory.OTHER)
        self.cal.add_event("Sooner", t + 1800, ImpactLevel.MEDIUM, EventCategory.OTHER)
        upcoming = self.cal.upcoming_events(hours_ahead=24)
        assert upcoming[0].name == "Sooner"

    def test_get_macro_calendar_singleton(self) -> None:
        from wallet_manager.market_intelligence import get_macro_calendar
        a = get_macro_calendar()
        b = get_macro_calendar()
        assert a is b


# ---------------------------------------------------------------------------
# Phase 6: RiskCommittee
# ---------------------------------------------------------------------------


class TestBrakeResult:
    def test_to_dict_fields(self) -> None:
        from wallet_manager.risk_committee import BrakeResult, BrakeVerdict
        br = BrakeResult("TestBrake", BrakeVerdict.ALLOW, "ok", 0, value=0.5, threshold=1.0)
        d = br.to_dict()
        assert d["brake"] == "TestBrake"
        assert d["verdict"] == "ALLOW"
        assert d["reason"] == "ok"
        assert d["value"] == 0.5

    def test_is_blocking(self) -> None:
        from wallet_manager.risk_committee import BrakeResult, BrakeVerdict
        blocking = BrakeResult("X", BrakeVerdict.BLOCK, "blocked", 2)
        allowing = BrakeResult("Y", BrakeVerdict.ALLOW, "ok", 0)
        assert blocking.is_blocking() is True
        assert allowing.is_blocking() is False


class TestSessionKillSwitch:
    def test_default_allows(self) -> None:
        from wallet_manager.risk_committee import SessionKillSwitch
        ks = SessionKillSwitch()
        result = ks.evaluate()
        assert result.verdict.value == "ALLOW"

    def test_activate_blocks(self) -> None:
        from wallet_manager.risk_committee import BrakeVerdict, SessionKillSwitch
        ks = SessionKillSwitch()
        ks.activate("manual stop")
        result = ks.evaluate()
        assert result.verdict == BrakeVerdict.BLOCK

    def test_deactivate_allows_again(self) -> None:
        from wallet_manager.risk_committee import BrakeVerdict, SessionKillSwitch
        ks = SessionKillSwitch()
        ks.activate("test")
        ks.deactivate()
        result = ks.evaluate()
        assert result.verdict == BrakeVerdict.ALLOW


class TestHighSpreadBrake:
    def test_allow_below_threshold(self) -> None:
        from wallet_manager.risk_committee import BrakeVerdict, HighSpreadBrake
        brake = HighSpreadBrake(max_spread_bps=30.0)
        result = brake.evaluate(spread_bps=10.0)
        assert result.verdict == BrakeVerdict.ALLOW

    def test_block_above_threshold(self) -> None:
        from wallet_manager.risk_committee import BrakeVerdict, HighSpreadBrake
        brake = HighSpreadBrake(max_spread_bps=30.0)
        result = brake.evaluate(spread_bps=50.0)
        assert result.verdict == BrakeVerdict.BLOCK


class TestCommitteeVerdict:
    def test_to_dict_fields(self) -> None:
        from wallet_manager.risk_committee import BrakeVerdict, CommitteeVerdict
        v = CommitteeVerdict(
            allowed=True,
            overall_verdict=BrakeVerdict.ALLOW,
            risk_score=10,
            brake_results=[],
            blocking_brakes=[],
            narrative="ok",
        )
        d = v.to_dict()
        assert "allowed" in d
        assert "overall_verdict" in d
        assert d["overall_verdict"] == "ALLOW"
        assert d["risk_score"] == 10


@pytest.mark.asyncio
class TestRiskCommitteeEvaluate:
    async def test_evaluate_no_engine_returns_verdict(self) -> None:
        from wallet_manager.risk_committee import RiskCommittee
        rc = RiskCommittee()
        verdict = await rc.evaluate(paper_engine=None)
        assert verdict.allowed is True or verdict.allowed is False
        assert verdict.overall_verdict.value in ("ALLOW", "BLOCK", "REDUCE")

    async def test_kill_switch_causes_block(self) -> None:
        from wallet_manager.risk_committee import BrakeVerdict, RiskCommittee
        rc = RiskCommittee()
        rc.kill_switch.activate("test block")
        verdict = await rc.evaluate(paper_engine=None)
        assert verdict.overall_verdict == BrakeVerdict.BLOCK
        assert verdict.allowed is False

    async def test_get_risk_committee_singleton(self) -> None:
        from wallet_manager.risk_committee import get_risk_committee
        a = get_risk_committee()
        b = get_risk_committee()
        assert a is b


# ---------------------------------------------------------------------------
# Phase 7: ExperimentLab
# ---------------------------------------------------------------------------


class TestThresholdVariant:
    def test_baseline_would_trade(self) -> None:
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(
            name="baseline",
            alpha_threshold=0.08,
            confidence_threshold=0.35,
        )
        assert v.would_trade(robust_alpha=0.10, confidence=0.40, adf_pvalue=0.04) is True
        assert v.would_trade(robust_alpha=0.02, confidence=0.40, adf_pvalue=0.04) is False
        assert v.would_trade(robust_alpha=0.10, confidence=0.20, adf_pvalue=0.04) is False

    def test_to_dict_fields(self) -> None:
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="test", alpha_threshold=0.1, confidence_threshold=0.4)
        d = v.to_dict()
        assert "alpha_threshold" in d
        assert "confidence_threshold" in d


class TestExperimentLab:
    def setup_method(self) -> None:
        from wallet_manager.experiment_lab import ExperimentLab
        self.lab = ExperimentLab()
        self.lab.register_threshold_variant("tight", alpha_threshold=0.15, confidence_threshold=0.5)

    def test_register_variant(self) -> None:
        variants = self.lab.active_variants()
        names = [v.name for v in variants]
        assert "baseline" in names
        assert "tight" in names

    def test_record_all_variants(self) -> None:
        self.lab.record_all_variants(
            robust_alpha=0.12, confidence=0.45, adf_pvalue=0.03,
            baseline_traded=True, pnl_usd=0.50,
        )
        baseline_records = self.lab._records.get("baseline", [])
        assert len(baseline_records) == 1
        assert baseline_records[0].baseline_traded is True

    def test_get_variant_report(self) -> None:
        for _ in range(5):
            self.lab.record_all_variants(0.10, 0.40, 0.04, baseline_traded=True)
        report = self.lab.get_variant_report()
        assert isinstance(report, list)
        assert any(r["name"] == "baseline" for r in report)

    def test_compare_variants_returns_winner(self) -> None:
        for _ in range(15):
            self.lab.record_all_variants(0.10, 0.40, 0.04, baseline_traded=True, pnl_usd=0.1)
        result = self.lab.compare_variants()
        assert "winner" in result
        assert "variants" in result

    def test_to_dict_keys(self) -> None:
        d = self.lab.to_dict()
        assert "total_variants" in d
        assert "comparison" in d

    def test_get_experiment_lab_singleton(self) -> None:
        from wallet_manager.experiment_lab import get_experiment_lab
        a = get_experiment_lab()
        b = get_experiment_lab()
        assert a is b


# ---------------------------------------------------------------------------
# Phase 4: ResearchBrain
# ---------------------------------------------------------------------------


class TestResearchBrain:
    def test_instantiation(self) -> None:
        from wallet_manager.research_brain import ResearchBrain
        brain = ResearchBrain(paper_engine=None)
        assert brain is not None

    @pytest.mark.asyncio
    async def test_weekly_review_no_engine(self) -> None:
        from wallet_manager.research_brain import ResearchBrain
        brain = ResearchBrain(paper_engine=None)
        report = await brain.weekly_strategy_review()
        assert report is not None
        assert hasattr(report, "sections")
        assert isinstance(report.sections, list)

    @pytest.mark.asyncio
    async def test_report_has_title(self) -> None:
        from wallet_manager.research_brain import ResearchBrain
        brain = ResearchBrain(paper_engine=None)
        report = await brain.weekly_strategy_review()
        assert report.title != ""


# ---------------------------------------------------------------------------
# Phase 8: AI Agents
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_instantiation(self) -> None:
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        assert registry.research is not None
        assert registry.risk is not None
        assert registry.ops is not None

    @pytest.mark.asyncio
    async def test_run_all_returns_list(self) -> None:
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        reports = await registry.run_all(paper_engine=None)
        assert isinstance(reports, list)

    @pytest.mark.asyncio
    async def test_run_all_to_dict(self) -> None:
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        dicts = await registry.run_all_to_dict(paper_engine=None)
        assert isinstance(dicts, list)
        for d in dicts:
            assert isinstance(d, dict)
            assert "agent_name" in d or "name" in d or len(d) > 0

    def test_get_agent_registry_singleton(self) -> None:
        from wallet_manager.ai_agents import get_agent_registry
        a = get_agent_registry()
        b = get_agent_registry()
        assert a is b


# ---------------------------------------------------------------------------
# Phase 10: ContinuousLearner
# ---------------------------------------------------------------------------


class TestContinuousLearner:
    def test_instantiation(self) -> None:
        from wallet_manager.continuous_learning import ContinuousLearner
        learner = ContinuousLearner()
        assert learner.drift_reviewer is not None
        assert learner.setup_decay is not None
        assert learner.retuner is not None
        assert learner.changelog is not None

    @pytest.mark.asyncio
    async def test_generate_report_no_engine(self) -> None:
        from wallet_manager.continuous_learning import ContinuousLearner
        learner = ContinuousLearner()
        report = await learner.generate_learning_report(paper_engine=None)
        assert isinstance(report, dict)
        assert "drift" in report
        assert "setup_decay" in report
        assert "threshold_suggestions" in report
        assert "alerts" in report

    def test_get_continuous_learner_singleton(self) -> None:
        from wallet_manager.continuous_learning import get_continuous_learner
        a = get_continuous_learner()
        b = get_continuous_learner()
        assert a is b

    def test_performance_changelog_record(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        from wallet_manager.continuous_learning import PerformanceChangelog
        with tempfile.TemporaryDirectory() as tmpdir:
            cl = PerformanceChangelog(path=Path(tmpdir) / "changelog.json")
            cl.record(
                category="threshold_update",
                description="Test change",
                before={"alpha_threshold": 0.08},
                after={"alpha_threshold": 0.10},
            )
            recent = cl.recent(1)
            assert len(recent) == 1
            assert recent[0]["description"] == "Test change"

    def test_drift_reviewer_trend(self) -> None:
        from wallet_manager.continuous_learning import DriftReviewer
        reviewer = DriftReviewer()
        summary = reviewer.trend_summary()
        assert "recent_level" in summary or isinstance(summary, dict)
