"""tests/unit/test_ai_agents_and_experiments.py — Coverage for ai_agents.py and experiment_lab.py"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ai_agents.py
# ---------------------------------------------------------------------------

class TestAgentReport:
    def test_to_dict(self):
        from wallet_manager.ai_agents import AgentReport
        r = AgentReport(agent_name="TestAgent", title="Test", body="body", used_llm=False)
        d = r.to_dict()
        assert d["agent"] == "TestAgent"
        assert d["title"] == "Test"
        assert d["body"] == "body"
        assert d["used_llm"] is False

    def test_to_text(self):
        from wallet_manager.ai_agents import AgentReport
        r = AgentReport(agent_name="TestAgent", title="Test", body="body text")
        txt = r.to_text()
        assert "TestAgent" in txt
        assert "body text" in txt

    def test_with_data(self):
        from wallet_manager.ai_agents import AgentReport
        r = AgentReport(agent_name="A", title="T", body="B", data={"key": "val"})
        assert r.data["key"] == "val"

    def test_used_llm_flag(self):
        from wallet_manager.ai_agents import AgentReport
        r = AgentReport(agent_name="A", title="T", body="B", used_llm=True)
        assert r.used_llm is True
        assert r.to_dict()["used_llm"] is True


class TestBaseAgent:
    def test_init(self):
        from wallet_manager.ai_agents import BaseAgent
        agent = BaseAgent()
        assert agent._model == "gpt-4.1-mini"
        assert agent._max_tokens == 1024

    def test_has_llm_no_key(self):
        from wallet_manager.ai_agents import BaseAgent
        with patch.dict("os.environ", {"GITHUB_TOKEN": ""}, clear=False):
            agent = BaseAgent()
            agent._api_key = ""
            assert agent._has_llm() is False

    def test_call_llm_no_llm(self):
        from wallet_manager.ai_agents import BaseAgent
        agent = BaseAgent()
        agent._api_key = ""
        result = agent._call_llm("sys", "user")
        assert result == ""


class TestNewsAgent:
    def test_run_no_events(self):
        from wallet_manager.ai_agents import NewsAgent
        agent = NewsAgent()
        mock_calendar = MagicMock()
        mock_calendar.to_dict = MagicMock(return_value={})
        mock_calendar.upcoming_events = MagicMock(return_value=[])
        mock_calendar.get_imminent_events = MagicMock(return_value=[])
        with patch("wallet_manager.market_intelligence.get_macro_calendar", return_value=mock_calendar):
            report = agent.run()
        assert report.agent_name == "NewsAgent"
        assert "No high-impact" in report.body

    def test_run_with_imminent_events(self):
        from wallet_manager.ai_agents import NewsAgent
        agent = NewsAgent()
        mock_event = MagicMock()
        mock_event.name = "FOMC Meeting"
        mock_event.impact = MagicMock(value="HIGH")
        import datetime
        mock_event.dt_utc = datetime.datetime(2026, 4, 5, 14, 0)
        mock_calendar = MagicMock()
        mock_calendar.to_dict = MagicMock(return_value={})
        mock_calendar.upcoming_events = MagicMock(return_value=[mock_event])
        mock_calendar.get_imminent_events = MagicMock(return_value=[mock_event])
        with patch("wallet_manager.market_intelligence.get_macro_calendar", return_value=mock_calendar):
            report = agent.run()
        assert "IMMINENT" in report.body or "FOMC" in report.body


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_run_no_paper_engine(self):
        from wallet_manager.ai_agents import ResearchAgent
        agent = ResearchAgent()
        mock_weekly = MagicMock()
        mock_weekly.to_dict = MagicMock(return_value={"sharpe": 0.5})
        mock_weekly.to_text = MagicMock(return_value="Weekly report")
        mock_threshold = MagicMock()
        mock_threshold.to_text = MagicMock(return_value="Thresholds OK")
        mock_brain = MagicMock()
        mock_brain.weekly_strategy_review = AsyncMock(return_value=mock_weekly)
        mock_brain.threshold_tuning_suggestions = AsyncMock(return_value=mock_threshold)
        with patch("wallet_manager.research_brain.ResearchBrain", return_value=mock_brain):
            report = await agent.run(paper_engine=None)
        assert report.agent_name == "ResearchAgent"
        assert report.used_llm is False


class TestRiskAgent:
    @pytest.mark.asyncio
    async def test_run(self):
        from wallet_manager.ai_agents import RiskAgent
        agent = RiskAgent()
        mock_verdict = MagicMock()
        mock_verdict.risk_score = 20
        mock_verdict.overall_verdict = MagicMock(value="ALLOW")
        mock_verdict.narrative = "All good"
        mock_verdict.blocking_brakes = []
        mock_verdict.brake_results = []
        mock_verdict.to_dict = MagicMock(return_value={})
        mock_committee = MagicMock()
        mock_committee.evaluate = AsyncMock(return_value=mock_verdict)
        with patch("wallet_manager.risk_committee.get_risk_committee", return_value=mock_committee):
            report = await agent.run()
        assert "Risk Score" in report.body

    @pytest.mark.asyncio
    async def test_run_with_blocking_brakes(self):
        from wallet_manager.ai_agents import RiskAgent
        agent = RiskAgent()
        mock_verdict = MagicMock()
        mock_verdict.risk_score = 80
        mock_verdict.overall_verdict = MagicMock(value="BLOCK")
        mock_verdict.narrative = "Drawdown exceeded"
        mock_verdict.blocking_brakes = ["DailyLossLimit"]
        mock_brake = MagicMock()
        mock_brake.brake_name = "DailyLossLimit"
        mock_brake.verdict = MagicMock(value="BLOCK")
        mock_brake.reason = "Loss > threshold"
        mock_verdict.brake_results = [mock_brake]
        mock_verdict.to_dict = MagicMock(return_value={})
        mock_committee = MagicMock()
        mock_committee.evaluate = AsyncMock(return_value=mock_verdict)
        with patch("wallet_manager.risk_committee.get_risk_committee", return_value=mock_committee):
            report = await agent.run()
        assert "DailyLossLimit" in report.body


class TestTradeReviewAgent:
    @pytest.mark.asyncio
    async def test_run(self):
        from wallet_manager.ai_agents import TradeReviewAgent
        agent = TradeReviewAgent()
        mock_postmortem = MagicMock()
        mock_postmortem.to_text = MagicMock(return_value="Postmortem text")
        mock_postmortem.to_dict = MagicMock(return_value={})
        mock_best = MagicMock()
        mock_best.to_text = MagicMock(return_value="Best setups")
        mock_best.to_dict = MagicMock(return_value={})
        mock_brain = MagicMock()
        mock_brain.worst_trade_postmortem = AsyncMock(return_value=mock_postmortem)
        mock_brain.best_setup_summary = AsyncMock(return_value=mock_best)
        with patch("wallet_manager.research_brain.ResearchBrain", return_value=mock_brain):
            report = await agent.run()
        assert "Postmortem text" in report.body or "Best setups" in report.body


class TestOpsAgent:
    @pytest.mark.asyncio
    async def test_run_no_paper_engine(self):
        from wallet_manager.ai_agents import OpsAgent
        agent = OpsAgent()
        report = await agent.run(paper_engine=None)
        assert "nominal" in report.body.lower()

    @pytest.mark.asyncio
    async def test_run_with_paper_engine_stuck(self):
        from wallet_manager.ai_agents import OpsAgent
        agent = OpsAgent()
        # Old timestamp = 48 hours ago
        old_ts = time.time() - 48 * 3600
        mock_row1 = MagicMock()
        mock_row1.__getitem__ = MagicMock(side_effect=lambda k: old_ts if k == "last_ts" else None)
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = MagicMock(return_value=0)
        mock_cursor1 = MagicMock()
        mock_cursor1.__aenter__ = AsyncMock(return_value=mock_cursor1)
        mock_cursor1.__aexit__ = AsyncMock(return_value=False)
        mock_cursor1.fetchone = AsyncMock(return_value=mock_row1)
        mock_cursor2 = MagicMock()
        mock_cursor2.__aenter__ = AsyncMock(return_value=mock_cursor2)
        mock_cursor2.__aexit__ = AsyncMock(return_value=False)
        mock_cursor2.fetchone = AsyncMock(return_value=mock_row2)
        call_count = [0]
        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            return mock_cursor1 if call_count[0] == 1 else mock_cursor2
        mock_conn = MagicMock()
        mock_conn.execute = execute_side_effect
        mock_ledger = MagicMock()
        mock_ledger._get_conn = MagicMock(return_value=mock_conn)
        mock_engine = MagicMock()
        mock_engine._ledger = mock_ledger
        report = await agent.run(paper_engine=mock_engine)
        assert "stuck" in report.body.lower() or "24" in report.body or "48" in report.body


class TestAgentRegistry:
    @pytest.mark.asyncio
    async def test_run_all_no_engine(self):
        from wallet_manager.ai_agents import AgentRegistry, AgentReport
        registry = AgentRegistry()
        # Mock all agent runs
        mock_report = AgentReport("Mock", "Title", "Body")
        with patch.object(registry.ops, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.risk, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.research, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.trade_review, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.news, "run", MagicMock(return_value=mock_report)):
            reports = await registry.run_all()
        assert len(reports) == 5

    @pytest.mark.asyncio
    async def test_run_all_handles_exceptions(self):
        from wallet_manager.ai_agents import AgentRegistry, AgentReport
        registry = AgentRegistry()
        mock_report = AgentReport("Mock", "Title", "Body")
        with patch.object(registry.ops, "run", AsyncMock(side_effect=Exception("fail"))), \
             patch.object(registry.risk, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.research, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.trade_review, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.news, "run", MagicMock(return_value=mock_report)):
            reports = await registry.run_all()
        assert len(reports) == 4  # ops failed

    @pytest.mark.asyncio
    async def test_run_all_to_dict(self):
        from wallet_manager.ai_agents import AgentRegistry, AgentReport
        registry = AgentRegistry()
        mock_report = AgentReport("Mock", "Title", "Body")
        with patch.object(registry.ops, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.risk, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.research, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.trade_review, "run", AsyncMock(return_value=mock_report)), \
             patch.object(registry.news, "run", MagicMock(return_value=mock_report)):
            result = await registry.run_all_to_dict()
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_get_agent_registry_singleton(self):
        import wallet_manager.ai_agents as _mod
        from wallet_manager.ai_agents import get_agent_registry
        _mod._registry = None
        r1 = get_agent_registry()
        r2 = get_agent_registry()
        assert r1 is r2


# ---------------------------------------------------------------------------
# experiment_lab.py
# ---------------------------------------------------------------------------

class TestThresholdVariant:
    def test_would_trade_all_conditions_met(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant("test", alpha_threshold=0.05, confidence_threshold=0.30)
        assert v.would_trade(0.10, 0.50) is True

    def test_would_trade_alpha_too_low(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant("test", alpha_threshold=0.10, confidence_threshold=0.30)
        assert v.would_trade(0.05, 0.50) is False

    def test_would_trade_confidence_too_low(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant("test", alpha_threshold=0.05, confidence_threshold=0.50)
        assert v.would_trade(0.10, 0.30) is False

    def test_would_trade_adf_pvalue_check(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant("test", alpha_threshold=0.05, confidence_threshold=0.30, adf_pvalue_threshold=0.05)
        assert v.would_trade(0.10, 0.50, adf_pvalue=0.10) is False  # adf too high

    def test_to_dict(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant("test", alpha_threshold=0.05, confidence_threshold=0.30, description="desc")
        d = v.to_dict()
        assert d["name"] == "test"
        assert d["alpha_threshold"] == 0.05


class TestFeatureToggleVariant:
    def test_feature_active_enabled(self):
        from wallet_manager.experiment_lab import FeatureToggleVariant
        v = FeatureToggleVariant("t", enabled_features=frozenset(["f1"]), disabled_features=frozenset())
        assert v.feature_active("f1") is True
        assert v.feature_active("f2") is True  # not disabled

    def test_feature_active_disabled(self):
        from wallet_manager.experiment_lab import FeatureToggleVariant
        v = FeatureToggleVariant("t", enabled_features=frozenset(), disabled_features=frozenset(["f1"]))
        assert v.feature_active("f1") is False

    def test_to_dict(self):
        from wallet_manager.experiment_lab import FeatureToggleVariant
        v = FeatureToggleVariant("t", enabled_features=frozenset(["f1"]), disabled_features=frozenset(["f2"]))
        d = v.to_dict()
        assert "f1" in d["enabled_features"]
        assert "f2" in d["disabled_features"]


class TestRegimeVariant:
    def test_would_trade_correct_regime(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant("rv", allowed_regimes=frozenset(["bull"]))
        assert v.would_trade(0.10, 0.50, "bull") is True

    def test_would_trade_wrong_regime(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant("rv", allowed_regimes=frozenset(["bull"]))
        assert v.would_trade(0.10, 0.50, "bear") is False

    def test_to_dict(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant("rv", allowed_regimes=frozenset(["bull", "normal"]))
        d = v.to_dict()
        assert "bull" in d["allowed_regimes"]


class TestExperimentLab:
    def setup_method(self):
        from wallet_manager.experiment_lab import ExperimentLab
        self.lab = ExperimentLab()

    def test_init_has_baseline(self):
        assert "baseline" in self.lab._variants

    def test_register_threshold_variant(self):
        v = self.lab.register_threshold_variant("test", 0.05, 0.30, description="test")
        assert v.name == "test"
        assert "test" in self.lab._variants

    def test_register_feature_toggle_variant(self):
        v = self.lab.register_feature_toggle_variant("ft", disabled_features=["funding_rate"], enabled_features=["ofi"])
        assert v.name == "ft"
        assert "ft" in self.lab._feature_toggles

    def test_register_regime_variant(self):
        v = self.lab.register_regime_variant("rv", allowed_regimes=["bull", "normal"])
        assert v.name == "rv"
        assert "rv" in self.lab._regime_variants

    def test_record_cycle(self):
        self.lab.record_cycle("baseline", True, True, 0.15, 0.70, pnl_usd=0.05)
        records = self.lab._records["baseline"]
        assert len(records) == 1
        assert records[0].would_trade is True

    def test_record_cycle_unknown_variant_ignored(self):
        self.lab.record_cycle("nonexistent", True, True, 0.15, 0.70)
        # Should not raise

    def test_record_all_variants(self):
        self.lab.register_threshold_variant("v2", 0.03, 0.25)
        self.lab.record_all_variants(0.15, 0.70, 0.05, True, pnl_usd=0.10)
        # baseline and v2 should both have records
        assert len(self.lab._records["baseline"]) > 0
        assert len(self.lab._records["v2"]) > 0

    def test_active_variants(self):
        self.lab.register_threshold_variant("active_v", 0.05, 0.30)
        variants = self.lab.active_variants()
        names = [v.name for v in variants]
        assert "baseline" in names
        assert "active_v" in names

    def test_get_variant_report_empty(self):
        report = self.lab.get_variant_report()
        # baseline exists but has no records initially
        assert any(r["name"] == "baseline" for r in report)

    def test_get_variant_report_with_data(self):
        for i in range(5):
            self.lab.record_cycle("baseline", i % 2 == 0, True, 0.10, 0.60, pnl_usd=0.01)
        report = self.lab.get_variant_report()
        baseline = next(r for r in report if r["name"] == "baseline")
        assert baseline["cycles"] == 5

    def test_compare_variants_empty(self):
        from wallet_manager.experiment_lab import ExperimentLab
        lab = ExperimentLab()
        # Clear all records
        lab._variants = {}
        lab._records = {}
        result = lab.compare_variants()
        assert result["winner"] is None

    def test_compare_variants_with_data(self):
        self.lab.register_threshold_variant("v2", 0.03, 0.25)
        for i in range(10):
            self.lab.record_cycle("baseline", True, True, 0.15, 0.70, pnl_usd=0.05)
            self.lab.record_cycle("v2", True, True, 0.15, 0.70, pnl_usd=0.05)
        result = self.lab.compare_variants()
        assert "winner" in result

    def test_record_regime_cycle(self):
        self.lab.register_regime_variant("bull_only", ["bull"])
        self.lab.record_regime_cycle("bull", 0.15, 0.70, True, pnl_usd=0.05)
        key = "regime:bull_only"
        assert key in self.lab._records
        assert len(self.lab._records[key]) == 1

    def test_get_feature_toggle_report(self):
        self.lab.register_feature_toggle_variant("ft1", disabled_features=["ofi"])
        report = self.lab.get_feature_toggle_report()
        assert any(r["name"] == "ft1" for r in report)

    def test_get_regime_variant_report(self):
        self.lab.register_regime_variant("bull_rv", ["bull"])
        report = self.lab.get_regime_variant_report()
        assert any(r["name"] == "bull_rv" for r in report)

    def test_get_regime_variant_report_with_data(self):
        self.lab.register_regime_variant("bull_rv2", ["bull"])
        self.lab.record_regime_cycle("bull", 0.15, 0.70, True, 0.05)
        report = self.lab.get_regime_variant_report()
        rv = next(r for r in report if r["name"] == "bull_rv2")
        assert rv["cycles"] == 1

    def test_max_records_trim(self):
        self.lab._max_records_per_variant = 5
        for i in range(10):
            self.lab.record_cycle("baseline", True, True, 0.10, 0.60)
        assert len(self.lab._records["baseline"]) == 5


class TestExperimentCycleRecord:
    def test_to_dict(self):
        from wallet_manager.experiment_lab import ExperimentCycleRecord
        r = ExperimentCycleRecord(
            timestamp=1000.0, variant_name="test", would_trade=True,
            baseline_traded=False, robust_alpha=0.15, confidence=0.70, pnl_usd=0.05
        )
        d = r.to_dict()
        assert d["variant_name"] == "test"
        assert d["would_trade"] is True
        assert d["pnl_usd"] == 0.05
