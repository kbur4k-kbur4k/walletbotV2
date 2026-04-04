from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from wallet_manager.research_brain import ResearchBrain, ResearchReport, get_research_brain


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _FakeConn:
    def __init__(self, responses: dict[str, list[dict[str, object]]]) -> None:
        self._responses = responses

    def execute(self, query: str, params: object | None = None) -> _FakeCursor:
        del params
        for fragment, rows in self._responses.items():
            if fragment in query:
                return _FakeCursor(rows)
        return _FakeCursor([])


class _FakeLedger:
    def __init__(
        self,
        *,
        analytics: dict[str, object] | None = None,
        responses: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        self._analytics = analytics or {}
        self._responses = responses or {}

    def _get_conn(self) -> _FakeConn:
        return _FakeConn(self._responses)

    async def get_advanced_analytics(self) -> dict[str, object]:
        return self._analytics.get("advanced_analytics", {})

    async def get_hold_time_stats(self) -> dict[str, object]:
        return self._analytics.get("hold_time_stats", {})

    async def get_pnl_summary(self) -> dict[str, object]:
        return self._analytics.get("pnl_summary", {})

    async def get_performance_by_regime(self) -> list[dict[str, object]]:
        return self._analytics.get("performance_by_regime", [])

    async def get_performance_by_setup(self) -> list[dict[str, object]]:
        return self._analytics.get("performance_by_setup", [])

    async def get_live_readiness_score(self) -> dict[str, object]:
        return self._analytics.get("live_readiness", {})

    async def get_decision_analytics(self) -> dict[str, object]:
        return self._analytics.get("decision_analytics", {})


def _analytics_fixture() -> dict[str, object]:
    return {
        "advanced_analytics": {
            "closed_count": 24,
            "net_after_gas_usd": -1.25,
            "expectancy_usd": 0.12,
            "win_rate_pct": 44.0,
            "profit_factor": 0.9,
            "payoff_ratio": 1.3,
            "avg_loss_usd": 1.4,
            "avg_win_usd": 0.5,
            "max_drawdown_pct": 4.2,
            "first_half_win_rate": 0.52,
            "second_half_win_rate": 0.44,
        },
        "hold_time_stats": {"avg_hold_seconds": 90},
        "pnl_summary": {"max_drawdown": 0.12},
        "performance_by_regime": [
            {"regime_label": "trend", "expectancy_usd": 0.35, "win_rate_pct": 60.0},
            {"regime_label": "chop", "expectancy_usd": -0.22, "win_rate_pct": 30.0},
        ],
        "performance_by_setup": [
            {
                "setup_tag": "breakout",
                "closed_count": 3,
                "decision_count": 6,
                "win_rate_pct": 66.7,
                "expectancy_usd": 0.4,
                "realized_pnl_usd": 1.2,
            },
            {
                "setup_tag": "mean_revert",
                "closed_count": 0,
                "decision_count": 4,
                "win_rate_pct": 0.0,
                "expectancy_usd": -0.1,
                "realized_pnl_usd": -0.2,
            },
        ],
        "live_readiness": {"score": 62, "max_score": 100, "recommendation": "Close to live-ready; accumulate more data"},
        "decision_analytics": {"summary": {"total_decisions": 40}},
    }


def _engine_with_rows(
    responses: dict[str, list[dict[str, object]]] | None = None,
    analytics: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(_ledger=_FakeLedger(analytics=analytics, responses=responses))


def test_research_report_to_dict_and_text() -> None:
    report = ResearchReport(
        report_type="weekly_review",
        generated_at=0.0,
        title="Weekly Strategy Review",
        sections=[{"heading": "Overview", "body": "Stable sample."}],
        raw_data={"sample": 1},
        recommendations=["Keep iterating."],
    )

    payload = report.to_dict()
    text = report.to_text()

    assert payload["report_type"] == "weekly_review"
    assert payload["raw_data"] == {"sample": 1}
    assert "# Weekly Strategy Review" in text
    assert "## Recommendations" in text
    assert "- Keep iterating." in text


@pytest.mark.asyncio
async def test_weekly_strategy_review_builds_sections_and_recommendations(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = ResearchBrain()
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=_analytics_fixture()))

    report = await brain.weekly_strategy_review()

    assert report.report_type == "weekly_review"
    assert len(report.sections) == 4
    assert report.sections[0]["heading"] == "Performance Overview"
    assert "Closed trades: 24" in report.sections[0]["body"]
    assert any("Do not advance to live mode" in rec for rec in report.recommendations)


@pytest.mark.asyncio
async def test_worst_trade_postmortem_without_engine_returns_error() -> None:
    report = await ResearchBrain().worst_trade_postmortem()
    assert report.sections == [{"heading": "Error", "body": "Paper engine not available."}]


@pytest.mark.asyncio
async def test_worst_trade_postmortem_identifies_likely_causes(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "FROM paper_trades WHERE direction = 'SELL'": [
            {
                "timestamp": 1710000000.0,
                "slippage_bps": 18.0,
                "pnl_usd": -12.0,
                "gas_cost_usd": 0.5,
                "simulated_fill_price": 1985.0,
                "entry_price": 2000.0,
                "size_usd": 150.0,
            }
        ]
    }
    brain = ResearchBrain(_engine_with_rows(responses=responses))
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=_analytics_fixture()))

    report = await brain.worst_trade_postmortem(n=1)

    assert report.report_type == "postmortem"
    assert "High slippage" in report.sections[0]["body"]
    assert "Large position size amplified the loss." in report.sections[0]["body"]


@pytest.mark.asyncio
async def test_threshold_tuning_suggestions_handles_empty_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = ResearchBrain(_engine_with_rows(responses={"FROM paper_decisions": []}))
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=_analytics_fixture()))

    report = await brain.threshold_tuning_suggestions()

    assert report.sections == [{"heading": "Insufficient Data", "body": "Need more decision journal rows."}]


@pytest.mark.asyncio
async def test_threshold_tuning_suggestions_generates_recommendations(monkeypatch: pytest.MonkeyPatch) -> None:
    decision_rows = [
        {"action": "BUY", "confidence": 0.35, "robust_alpha": 0.02, "adf_pvalue": 0.01, "decision_class": "EXECUTE"},
        {"action": "SELL", "confidence": 0.40, "robust_alpha": 0.03, "adf_pvalue": 0.01, "decision_class": "EXECUTE"},
        {"action": "HOLD", "confidence": 0.62, "robust_alpha": 0.08, "adf_pvalue": 0.02, "decision_class": "NO_TRADE"},
        {"action": "HOLD", "confidence": 0.58, "robust_alpha": 0.07, "adf_pvalue": 0.02, "decision_class": "NO_TRADE"},
        {"action": "HOLD", "confidence": 0.55, "robust_alpha": 0.06, "adf_pvalue": 0.03, "decision_class": "SKIP"},
        {"action": "HOLD", "confidence": 0.57, "robust_alpha": 0.05, "adf_pvalue": 0.03, "decision_class": "NO_TRADE"},
        {"action": "HOLD", "confidence": 0.56, "robust_alpha": 0.04, "adf_pvalue": 0.04, "decision_class": "NO_TRADE"},
        {"action": "HOLD", "confidence": 0.54, "robust_alpha": 0.03, "adf_pvalue": 0.05, "decision_class": "NO_TRADE"},
        {"action": "HOLD", "confidence": 0.53, "robust_alpha": 0.02, "adf_pvalue": 0.05, "decision_class": "NO_TRADE"},
    ]
    brain = ResearchBrain(_engine_with_rows(responses={"FROM paper_decisions": decision_rows}))
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=_analytics_fixture()))

    report = await brain.threshold_tuning_suggestions()

    assert report.report_type == "threshold_tuning"
    assert report.sections[0]["heading"] == "Signal Distribution"
    assert any("lowering ML_CONFIDENCE_THRESHOLD" in rec for rec in report.recommendations)
    assert any("raising ML_CONFIDENCE_THRESHOLD" in rec for rec in report.recommendations)


@pytest.mark.asyncio
async def test_best_setup_summary_handles_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = ResearchBrain()
    empty = _analytics_fixture()
    empty["performance_by_setup"] = []
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=empty))

    report = await brain.best_setup_summary()

    assert report.sections == [{"heading": "No Data", "body": "No setup analytics yet."}]


@pytest.mark.asyncio
async def test_best_setup_summary_ranks_closed_setups(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = ResearchBrain()
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=_analytics_fixture()))

    report = await brain.best_setup_summary()

    assert report.sections[0]["heading"] == "breakout"
    assert "Expectancy: $0.4000" in report.sections[0]["body"]


@pytest.mark.asyncio
async def test_model_performance_attribution_summarizes_models() -> None:
    rows = [
        {"robust_alpha": 0.20, "pnl_usd": 2.0, "confidence": 0.8, "regime_label": "trend"},
        {"robust_alpha": 0.10, "pnl_usd": -1.0, "confidence": 0.5, "regime_label": "chop"},
    ]
    brain = ResearchBrain(_engine_with_rows(responses={"FROM paper_trades WHERE action='SELL'": rows}))

    report = await brain.model_performance_attribution()

    assert report.sections[0]["heading"] == "Per-Model PnL Attribution"
    assert "xgboost" in report.sections[0]["body"]
    assert len(report.recommendations) == 2


@pytest.mark.asyncio
async def test_pnl_attribution_groups_contexts() -> None:
    rows = [
        {
            "robust_alpha": 0.03,
            "confidence": 0.4,
            "regime_label": "trend",
            "setup_tag": "breakout",
            "pnl_usd": 1.5,
            "hold_time_seconds": 120,
            "timestamp": 1710000000.0,
        },
        {
            "robust_alpha": 0.09,
            "confidence": 0.5,
            "regime_label": "trend",
            "setup_tag": "breakout",
            "pnl_usd": -0.2,
            "hold_time_seconds": 900,
            "timestamp": 1710018000.0,
        },
        {
            "robust_alpha": 0.18,
            "confidence": 0.8,
            "regime_label": "chop",
            "setup_tag": "mean_revert",
            "pnl_usd": 2.2,
            "hold_time_seconds": 8200,
            "timestamp": 1710046800.0,
        },
        {
            "robust_alpha": 0.14,
            "confidence": 0.7,
            "regime_label": "normal",
            "setup_tag": "momentum",
            "pnl_usd": -1.1,
            "hold_time_seconds": 2200,
            "timestamp": 1710072000.0,
        },
        {
            "robust_alpha": 0.06,
            "confidence": 0.6,
            "regime_label": "normal",
            "setup_tag": "momentum",
            "pnl_usd": 0.8,
            "hold_time_seconds": 350,
            "timestamp": 1710097200.0,
        },
    ]
    brain = ResearchBrain(_engine_with_rows(responses={"FROM paper_trades WHERE action='SELL'": rows}))

    report = await brain.pnl_attribution()

    assert [section["heading"] for section in report.sections] == [
        "P&L by Regime",
        "P&L by Signal Quintile (alpha strength)",
        "P&L by Time of Day",
        "P&L by Hold Time",
    ]
    assert report.raw_data["total_trades"] == 5
    assert any("Best regime:" in rec for rec in report.recommendations)


@pytest.mark.asyncio
async def test_trade_cluster_analysis_builds_clusters_and_recommendations() -> None:
    rows = [
        {"robust_alpha": 0.14, "pnl_usd": -1.8, "hold_time_seconds": 200, "regime_label": "trend", "setup_tag": "breakout", "confidence": 0.8},
        {"robust_alpha": 0.09, "pnl_usd": 0.6, "hold_time_seconds": 600, "regime_label": "normal", "setup_tag": "pullback", "confidence": 0.6},
        {"robust_alpha": 0.03, "pnl_usd": -0.9, "hold_time_seconds": 2400, "regime_label": "chop", "setup_tag": "fade", "confidence": 0.5},
        {"robust_alpha": 0.02, "pnl_usd": -1.2, "hold_time_seconds": 120, "regime_label": "chop", "setup_tag": "fade", "confidence": 0.4},
        {"robust_alpha": 0.18, "pnl_usd": 2.1, "hold_time_seconds": 3600, "regime_label": "trend", "setup_tag": "breakout", "confidence": 0.9},
    ]
    brain = ResearchBrain(_engine_with_rows(responses={"FROM paper_trades WHERE action='SELL' ORDER BY timestamp DESC LIMIT 200": rows}))

    report = await brain.trade_cluster_analysis()

    assert [section["heading"] for section in report.sections] == [
        "Alpha Strength Clusters",
        "Hold Time Clusters",
        "Regime Clusters",
    ]
    assert any("Raise alpha entry threshold" in rec for rec in report.recommendations)
    assert any("Widen exit bands" in rec for rec in report.recommendations)


@pytest.mark.asyncio
async def test_pattern_summaries_from_journal_reports_blocked_high_confidence() -> None:
    rows = [
        {"action": "BUY", "primary_reason": None, "regime_label": "trend", "setup_tag": "breakout", "confidence": 0.75, "robust_alpha": 0.12},
        {"action": "HOLD", "primary_reason": "RISK_BRAKE", "regime_label": "trend", "setup_tag": "breakout", "confidence": 0.82, "robust_alpha": 0.18},
        {"action": "SKIP", "primary_reason": "NEWS_EVENT", "regime_label": "normal", "setup_tag": "filter", "confidence": 0.35, "robust_alpha": 0.02},
    ]
    brain = ResearchBrain(_engine_with_rows(responses={"FROM paper_decisions ORDER BY timestamp DESC LIMIT 500": rows}))

    report = await brain.pattern_summaries_from_journal()

    assert report.sections[0]["heading"] == "Overview"
    assert "RISK_BRAKE" in report.sections[2]["body"]
    assert any("high-confidence signals were blocked" in rec for rec in report.recommendations)


@pytest.mark.asyncio
async def test_compute_live_readiness_scores_positive_case(monkeypatch: pytest.MonkeyPatch) -> None:
    analytics = _analytics_fixture()
    analytics["advanced_analytics"] = {
        "closed_count": 55,
        "expectancy_usd": 0.3,
        "max_drawdown_pct": 3.2,
        "first_half_win_rate": 0.54,
        "second_half_win_rate": 0.56,
        "win_rate_pct": 58.0,
    }
    analytics["live_readiness"] = {"error_rate": 0.0}
    brain = ResearchBrain()
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=analytics))

    readiness = await brain.compute_live_readiness()

    assert readiness["score"] == 80
    assert readiness["recommendation"] == "Ready for micro-live ($25)"
    assert readiness["checks"]["closed_trades_50"] is True


@pytest.mark.asyncio
async def test_feature_hypothesis_suggestions_emits_multiple_hypotheses(monkeypatch: pytest.MonkeyPatch) -> None:
    analytics = _analytics_fixture()
    analytics["advanced_analytics"] = {"win_rate_pct": 41.0, "expectancy_usd": 0.02}
    analytics["hold_time_stats"] = {"avg_hold_seconds": 60}
    brain = ResearchBrain()
    monkeypatch.setattr(brain, "_collect_analytics", AsyncMock(return_value=analytics))

    report = await brain.feature_hypothesis_suggestions()

    assert report.report_type == "hypothesis"
    assert len(report.sections) >= 3
    assert any("RSI confirmation filter" in section["body"] for section in report.sections)


@pytest.mark.asyncio
async def test_collect_analytics_reads_ledger_methods() -> None:
    analytics = _analytics_fixture()
    brain = ResearchBrain(_engine_with_rows(analytics=analytics))

    payload = await brain._collect_analytics()

    assert payload["advanced_analytics"]["closed_count"] == 24
    assert payload["live_readiness"]["score"] == 62


def test_build_recommendations_covers_low_sample_and_negative_metrics() -> None:
    brain = ResearchBrain()

    low_sample = brain._build_recommendations({"closed_count": 3}, {}, {}, [], [])
    detailed = brain._build_recommendations(
        {
            "closed_count": 12,
            "net_after_gas_usd": -0.1,
            "win_rate_pct": 40.0,
            "profit_factor": 0.8,
        },
        {"avg_hold_seconds": 120},
        {"max_drawdown": 0.11},
        [{"regime_label": "trend", "expectancy_usd": 0.2}],
        [],
    )

    assert "Only 3 closed trades." in low_sample[0]
    assert any("Max drawdown > 10%" in rec for rec in detailed)
    assert any("Positive expectancy observed in: trend." in rec for rec in detailed)


def test_fmt_s_formats_short_medium_and_long_ranges() -> None:
    assert ResearchBrain._fmt_s(12) == "12s"
    assert ResearchBrain._fmt_s(125) == "2m 5s"
    assert ResearchBrain._fmt_s(3720) == "1h 2m"


@pytest.mark.asyncio
async def test_get_research_brain_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import wallet_manager.research_brain as research_brain_module
    import wallet_manager.strategies.paper_engine as paper_engine_module

    fake_engine = _engine_with_rows()
    monkeypatch.setattr(research_brain_module, "_brain_instance", None)
    monkeypatch.setattr(paper_engine_module, "_engine_instance", fake_engine)

    first = await get_research_brain()
    second = await get_research_brain()

    assert first is second
    assert first._engine is fake_engine
