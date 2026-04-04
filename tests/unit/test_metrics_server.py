"""tests/unit/test_metrics_server.py — MetricsServer endpoint testleri."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from wallet_manager.metrics_server import (
    MetricsServer,
    _build_runtime_payload,
    _build_strategy_research_notes,
    _hydrate_runtime_payload,
)


@pytest.fixture
async def metrics_client():
    server = MetricsServer(port=0)  # port=0 → OS random port
    app = server.build_app()
    async with TestClient(TestServer(app)) as client:
        yield client


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, metrics_client):
        resp = await metrics_client.get("/metrics")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_metrics_content_type(self, metrics_client):
        resp = await metrics_client.get("/metrics")
        ct = resp.headers.get("Content-Type", "")
        assert "text/plain" in ct

    @pytest.mark.asyncio
    async def test_metrics_contains_wallet_prefix(self, metrics_client):
        resp = await metrics_client.get("/metrics")
        text = await resp.text()
        assert "wallet_" in text

    @pytest.mark.asyncio
    async def test_health_returns_200(self, metrics_client):
        resp = await metrics_client.get("/health")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_response_json(self, metrics_client):
        resp = await metrics_client.get("/health")
        data = await resp.json()
        assert data["status"] == "ok"
        assert "ts" in data

    @pytest.mark.asyncio
    async def test_runtime_api_returns_runtime_payload(self, metrics_client, monkeypatch):
        async def _fake_payload():
            return {
                "status": "ok",
                "execution_mode": "robust_paper",
                "live_guard": {"passed": True},
                "paper_gate": {"passed": True, "current_trade_count": 3, "required_trade_count": 25},
                "portfolio": {"total_value_usd": 1234.56},
                "result_summary": {"wallets_succeeded": 1, "wallets_total": 1},
                "fatal_error": None,
                "audit": {"phase": "completed"},
                "recent_audits": [],
            }

        monkeypatch.setattr("wallet_manager.metrics_server._build_runtime_payload", _fake_payload)

        resp = await metrics_client.get("/api/runtime")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["execution_mode"] == "robust_paper"
        assert data["portfolio"]["total_value_usd"] == 1234.56

    @pytest.mark.asyncio
    async def test_dashboard_renders_runtime_summary(self, metrics_client, monkeypatch):
        async def _fake_payload():
            return {
                "status": "ok",
                "execution_mode": "robust_live",
                "live_guard": {"passed": False},
                "paper_gate": {"passed": True, "current_trade_count": 30, "required_trade_count": 25},
                "portfolio": {"total_value_usd": 250.0},
                "result_summary": {"wallets_succeeded": 0, "wallets_total": 1},
                "fatal_error": "capital limit exceeded",
                "audit": {"phase": "preflight"},
                "recent_audits": [],
                "research": {
                    "trade_count": 12,
                    "closed_trade_count": 6,
                    "net_after_gas_usd": 42.5,
                    "win_rate_pct": 58.3,
                    "avg_slippage_bps": 4.2,
                    "last_trade_at": 1710000000.0,
                    "weth_exposure_pct": 12.0,
                    "notes": ["Enough trades to inspect."],
                    "equity_curve": [{"timestamp": 1.0, "total_equity_usd": 250.0}],
                    "decision_analytics": {
                        "summary": {
                            "total_decisions": 20,
                            "executed_decisions": 6,
                            "no_trade_decisions": 10,
                            "skip_decisions": 4,
                        },
                        "no_trade_reasons": [{"reason": "NEWS_EVENT_IMMINENT: NFP", "count": 3}],
                        "regime_breakdown": [
                            {
                                "regime_label": "normal",
                                "decision_count": 12,
                                "executed_count": 6,
                                "realized_pnl_usd": 4.2,
                            }
                        ],
                        "setup_breakdown": [
                            {
                                "setup_tag": "momentum_long",
                                "decision_count": 7,
                                "executed_count": 4,
                                "realized_pnl_usd": 3.5,
                            }
                        ],
                    },
                    "recent_decisions": [
                        {
                            "timestamp": 1710000000.0,
                            "action": "HOLD",
                            "primary_reason": "ENTRY_NOT_TRIGGERED",
                            "regime_label": "normal",
                            "setup_tag": "signal_wait",
                            "trade_budget_usd": 20.0,
                            "confidence": 0.22,
                        }
                    ],
                    "recent_trades": [
                        {
                            "direction": "BUY",
                            "pair": "WETH/USDC",
                            "size_usd": 25.0,
                            "fill_price": 2000.0,
                            "timestamp": 1710000000.0,
                            "slippage_bps": 3.0,
                            "pnl_usd": 1.2,
                        }
                    ],
                },
            }

        monkeypatch.setattr("wallet_manager.metrics_server._build_runtime_payload", _fake_payload)

        resp = await metrics_client.get("/dashboard")
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        text = await resp.text()
        assert "WalletBot Runtime" in text
        assert "robust_live" in text
        assert "capital limit exceeded" in text
        assert "Strategy Lab" in text
        assert "Trade Tape" in text
        assert "Decision Journal" in text
        assert "No-Trade Reasons" in text

    @pytest.mark.asyncio
    async def test_research_api_returns_research_payload(self, metrics_client, monkeypatch):
        async def _fake_payload():
            return {
                "status": "ok",
                "generated_at": 123.0,
                "execution_mode": "robust_paper",
                "research": {"trade_count": 7, "net_after_gas_usd": 3.5},
            }

        monkeypatch.setattr("wallet_manager.metrics_server._build_runtime_payload", _fake_payload)

        resp = await metrics_client.get("/api/research")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["execution_mode"] == "robust_paper"
        assert data["research"]["trade_count"] == 7


class TestMetricUpdates:
    def test_update_metric(self):
        server = MetricsServer(port=0)
        server.update_metric("wallet_cycle_total", 42.0)
        assert server._metrics.get("wallet_cycle_total") == 42.0

    def test_update_labeled_metric(self):
        server = MetricsServer(port=0)
        server.update_labeled_metric("wallet_pillar_value_usd", "ALGO", 500.0)
        key = ("wallet_pillar_value_usd", "ALGO")
        assert server._labeled_metrics.get(key) == 500.0


@pytest.mark.asyncio
async def test_runtime_payload_falls_back_to_latest_completed_audit(monkeypatch):
    monkeypatch.setattr("wallet_manager.orchestrator.get_runtime_audit_snapshot", lambda: {})

    async def _fake_recent(_limit: int):
        return [
            {
                "session_id": "sess-123",
                "started_at": 1.0,
                "finished_at": 2.0,
                "payload": {
                    "execution_mode": "robust_paper",
                    "live_guard": {"passed": True},
                    "paper_gate": {"passed": True, "current_trade_count": 4, "required_trade_count": 50},
                    "portfolio": {"total_value_usd": 9999.0, "available": True},
                    "result_summary": {"wallets_succeeded": 1, "wallets_total": 1},
                    "fatal_error": None,
                },
            }
        ]

    monkeypatch.setattr("wallet_manager.metrics_server._load_recent_runtime_audits", _fake_recent)

    class _Settings:
        audit_recent_sessions_limit = 10
        execution_mode = "robust_paper"
        paper_db_path = "paper_trades.db"
        min_paper_trades_before_live = 50

    monkeypatch.setattr("wallet_manager.config.get_settings", lambda: _Settings())

    payload = await _build_runtime_payload()
    assert payload["data_source"] == "latest_completed_audit"
    assert payload["latest_session_id"] == "sess-123"
    assert payload["execution_mode"] == "robust_paper"
    assert payload["portfolio"]["total_value_usd"] == 9999.0


def test_hydrate_runtime_payload_uses_paper_engine_state(monkeypatch):
    class _Ledger:
        _equity_history = [10012.5]
        _realized_pnl = 12.5
        _trade_count = 3
        _balances = {"USDC": 9980.0, "WETH": 32.5}

    class _Config:
        initial_balance_usd = 10000.0

    class _Engine:
        _ledger = _Ledger()
        _config = _Config()

    monkeypatch.setattr("wallet_manager.strategies.paper_engine._engine_instance", _Engine())

    payload = _hydrate_runtime_payload(
        {
            "execution_mode": "robust_paper",
            "paper_gate": {"required_trade_count": 50},
            "portfolio": {"total_value_usd": 0.0},
        },
        "robust_paper",
    )

    assert payload["paper_gate"]["current_trade_count"] == 3
    assert payload["portfolio"]["total_value_usd"] == 10012.5
    assert payload["portfolio"]["allocations"][0]["strategy"] == "ALGO"


def test_build_strategy_research_notes_flags_small_sample():
    notes = _build_strategy_research_notes(
        {
            "trade_count": 2,
            "closed_trade_count": 0,
            "win_rate_pct": 0.0,
            "avg_slippage_bps": 2.0,
            "avg_gas_cost_usd": 0.1,
            "realized_pnl_usd": 0.0,
            "net_after_gas_usd": -0.2,
            "max_drawdown_pct": 0.5,
            "last_trade_age_seconds": 120.0,
            "weth_exposure_pct": 10.0,
        }
    )
    assert any("round-trip" in note for note in notes)
    assert any("Orneklem" in note for note in notes)


class _DictWrapper:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return self._payload


@pytest.mark.asyncio
async def test_metrics_server_additional_api_endpoints(metrics_client, monkeypatch):
    import wallet_manager.ai_agents as ai_agents_module
    import wallet_manager.alerting as alerting_module
    import wallet_manager.continuous_learning as continuous_learning_module
    import wallet_manager.experiment_lab as experiment_lab_module
    import wallet_manager.market_intelligence as market_intelligence_module
    import wallet_manager.reporting as reporting_module
    import wallet_manager.risk_committee as risk_committee_module
    import wallet_manager.risk_intelligence as risk_intelligence_module
    import wallet_manager.signal_discovery as signal_discovery_module
    import wallet_manager.signal_intelligence as signal_intelligence_module
    import wallet_manager.strategies.adaptive_ensemble as adaptive_ensemble_module
    import wallet_manager.strategies.bayesian_kelly as bayesian_kelly_module
    import wallet_manager.strategies.cross_asset as cross_asset_module
    import wallet_manager.strategies.execution_quality as execution_quality_module
    import wallet_manager.strategies.funding_rate as funding_rate_module
    import wallet_manager.strategies.hyperopt as hyperopt_module
    import wallet_manager.strategies.model_registry as model_registry_module
    import wallet_manager.strategies.online_learning as online_learning_module
    import wallet_manager.strategies.order_flow as order_flow_module
    import wallet_manager.strategies.paper_engine as paper_engine_module
    import wallet_manager.strategies.shadow_live as shadow_live_module
    import wallet_manager.tick_collector as tick_collector_module

    class _Verdict:
        def to_dict(self) -> dict[str, object]:
            return {"decision": "allow"}

    class _RiskCommittee:
        async def evaluate(self, paper_engine=None) -> _Verdict:
            assert paper_engine is fake_engine
            return _Verdict()

    class _Calendar:
        def to_dict(self) -> dict[str, object]:
            return {"next_event": "CPI"}

        def should_suppress_alpha(self) -> tuple[bool, str]:
            return True, "macro_event"

    class _ExperimentLab:
        def compare_variants(self) -> dict[str, object]:
            return {"winner": "baseline"}

        def get_variant_report(self) -> list[dict[str, object]]:
            return [{"name": "baseline"}]

        def get_regime_variant_report(self) -> dict[str, object]:
            return {"trend": "baseline"}

        def get_feature_toggle_report(self) -> dict[str, object]:
            return {"use_rsi": True}

    class _AgentRegistry:
        async def run_all_to_dict(self, paper_engine=None) -> list[dict[str, object]]:
            assert paper_engine is fake_engine
            return [{"title": "weekly"}]

    class _FeatureDrift:
        def to_dict(self) -> dict[str, object]:
            return {"drift_score": 0.1}

        def get_recent_alerts(self, limit: int) -> list[str]:
            assert limit == 5
            return ["drift-warning"]

    class _Learner:
        feature_drift = _FeatureDrift()

        async def generate_learning_report(self, paper_engine) -> dict[str, object]:
            assert paper_engine is fake_engine
            return {"status": "fresh"}

    class _RiskIntelligence:
        def __init__(self) -> None:
            self.stress_engine = _DictWrapper({"scenario_count": 3})
            self._ran = False

        def to_dict(self, equity: float) -> dict[str, object]:
            return {"equity": equity}

        def run_stress_scenarios(self) -> None:
            self._ran = True

    class _SignalDiscovery:
        def __init__(self) -> None:
            self.ran = False

        async def run_discovery(self) -> None:
            self.ran = True

        def to_dict(self) -> dict[str, object]:
            return {"ran": self.ran}

    fake_engine = SimpleNamespace(_ledger=SimpleNamespace(get_total_equity_usd=AsyncMock(return_value=12345.0)))
    fake_risk_intelligence = _RiskIntelligence()
    fake_signal_discovery = _SignalDiscovery()

    monkeypatch.setattr(risk_committee_module, "get_risk_committee", lambda: _RiskCommittee())
    monkeypatch.setattr(market_intelligence_module, "get_macro_calendar", lambda: _Calendar())
    monkeypatch.setattr(market_intelligence_module, "get_chain_intelligence", lambda: SimpleNamespace(get_snapshot=AsyncMock(return_value={"rpc_url": "https://rpc"})))
    monkeypatch.setattr(experiment_lab_module, "get_experiment_lab", lambda: _ExperimentLab())
    monkeypatch.setattr(ai_agents_module, "get_agent_registry", lambda: _AgentRegistry())
    monkeypatch.setattr(continuous_learning_module, "get_continuous_learner", lambda: _Learner())
    monkeypatch.setattr(signal_intelligence_module, "get_signal_intelligence", lambda: _DictWrapper({"strength": 0.7}))
    monkeypatch.setattr(risk_intelligence_module, "get_risk_intelligence", lambda equity=10000.0: fake_risk_intelligence)
    monkeypatch.setattr(tick_collector_module, "get_tick_collector", lambda: SimpleNamespace(get_stats=AsyncMock(return_value={"count": 42})))
    monkeypatch.setattr(alerting_module, "get_alert_system", lambda: _DictWrapper({"active": 1}))
    monkeypatch.setattr(cross_asset_module, "get_cross_asset", lambda: SimpleNamespace(get_snapshot=lambda: {"beta": 0.5}))
    monkeypatch.setattr(adaptive_ensemble_module, "get_adaptive_ensemble", lambda: _DictWrapper({"models": 4}))
    monkeypatch.setattr(bayesian_kelly_module, "get_bayesian_kelly", lambda: _DictWrapper({"kelly_fraction": 0.25}))
    monkeypatch.setattr(execution_quality_module, "get_execution_quality", lambda: SimpleNamespace(to_dict=lambda: {"latency_ms": 12}, get_timing_snapshot=lambda: {"best_hour": 14}))
    monkeypatch.setattr(signal_discovery_module, "get_signal_discovery", lambda: fake_signal_discovery)
    monkeypatch.setattr(reporting_module, "get_reporter", lambda: SimpleNamespace(to_dict=lambda: {"reports": 2}, generate_monthly=AsyncMock(return_value={"month": "2026-04"})))
    monkeypatch.setattr(hyperopt_module, "get_hyperoptimizer", lambda: _DictWrapper({"trials": 8}))
    monkeypatch.setattr(order_flow_module, "get_order_flow_imbalance", lambda: SimpleNamespace(get_snapshot=lambda: {"ofi": 1.2}))
    monkeypatch.setattr(model_registry_module, "get_model_registry", lambda: _DictWrapper({"champion": "xgboost"}))
    monkeypatch.setattr(online_learning_module, "get_online_learning_hook", lambda: _DictWrapper({"updates": 5}))
    monkeypatch.setattr(funding_rate_module, "get_funding_rate_collector", lambda: _DictWrapper({"rate": 0.01}))
    monkeypatch.setattr(shadow_live_module, "get_shadow_live_engine", lambda: _DictWrapper({"matches": 4}))
    monkeypatch.setattr(paper_engine_module, "_engine_instance", fake_engine)

    endpoint_expectations = [
        ("/api/risk", "verdict"),
        ("/api/calendar", "calendar"),
        ("/api/experiments", "comparison"),
        ("/api/agents", "reports"),
        ("/api/learning", "learning"),
        ("/api/signals", "signals"),
        ("/api/risk_intelligence", "risk_intelligence"),
        ("/api/stress", "stress"),
        ("/api/ticks", "ticks"),
        ("/api/alerts", "alerts"),
        ("/api/cross_asset", "cross_asset"),
        ("/api/ensemble", "ensemble"),
        ("/api/bayesian_kelly", "bayesian_kelly"),
        ("/api/execution", "execution"),
        ("/api/discovery", "discovery"),
        ("/api/reports", "reports"),
        ("/api/monthly_report", "summary"),
        ("/api/hyperopt", "hyperopt"),
        ("/api/order_flow", "order_flow"),
        ("/api/model_registry", "model_registry"),
        ("/api/online_learning", "online_learning"),
        ("/api/timing", "timing"),
        ("/api/feature_drift", "feature_drift"),
        ("/api/funding_rate", "funding_rate"),
        ("/api/shadow_live", "shadow_live"),
        ("/api/chain", "chain"),
    ]

    for endpoint, key in endpoint_expectations:
        resp = await metrics_client.get(endpoint)
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert key in data


@pytest.mark.asyncio
async def test_root_redirects_to_dashboard(metrics_client):
    resp = await metrics_client.get("/", allow_redirects=False)
    assert resp.status == 302
    assert resp.headers["Location"] == "/dashboard"


@pytest.mark.asyncio
async def test_agents_api_surfaces_errors(metrics_client, monkeypatch):
    import wallet_manager.ai_agents as ai_agents_module

    class _BrokenRegistry:
        async def run_all_to_dict(self, paper_engine=None) -> list[dict[str, object]]:
            del paper_engine
            raise RuntimeError("registry offline")

    monkeypatch.setattr(ai_agents_module, "get_agent_registry", lambda: _BrokenRegistry())

    resp = await metrics_client.get("/api/agents")
    data = await resp.json()

    assert resp.status == 200
    assert data["status"] == "ok"
    assert data["error"] == "registry offline"
