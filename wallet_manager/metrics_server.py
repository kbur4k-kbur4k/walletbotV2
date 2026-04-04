"""
wallet_manager/metrics_server.py

Prometheus-Compatible /metrics Endpoint.

Lightweight async HTTP server exposing /metrics on cfg.metrics_port (default 9090).
Uses aiohttp.web + prometheus_client for proper OpenMetrics text generation.

Metrics:
  wallet_portfolio_total_usd
  wallet_pillar_value_usd{pillar}
  wallet_trade_count_total          (counter)
  wallet_pnl_usd
  wallet_sharpe_ratio
  wallet_max_drawdown
  wallet_ml_model_confidence
  wallet_concept_drift_level        (0=NOMINAL, 1=MILD, 2=MODERATE, 3=SEVERE)
  wallet_aave_apy_pct
  wallet_rpc_latency_ms{endpoint}
  wallet_gas_price_gwei

Runs as an asyncio background task; does not block the main loop.
If cfg.metrics_enabled = False, server never starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import aiohttp.web
import aiosqlite
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Metric registry — module-level prometheus_client objects
# ---------------------------------------------------------------------------

# Use a dedicated registry to avoid collisions with the default global one
# (important for tests that import this module multiple times).
_REGISTRY = CollectorRegistry(auto_describe=True)


def _gauge(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Gauge:
    return Gauge(name, doc, list(labelnames), registry=_REGISTRY)


def _counter(name: str, doc: str) -> Counter:
    return Counter(name, doc, registry=_REGISTRY)


# Scalar gauges
_G_PORTFOLIO_TOTAL   = _gauge("wallet_portfolio_total_usd",   "Total portfolio value in USD")
_G_PNL               = _gauge("wallet_pnl_usd",               "Current PnL in USD")
_G_SHARPE            = _gauge("wallet_sharpe_ratio",           "Current Sharpe ratio estimate")
_G_DRAWDOWN          = _gauge("wallet_max_drawdown",           "Maximum drawdown fraction")
_G_ML_CONFIDENCE     = _gauge("wallet_ml_model_confidence",    "ML model signal confidence 0-1")
_G_DRIFT_LEVEL       = _gauge("wallet_concept_drift_level",    "Concept drift level 0=NOMINAL 1=MILD 2=MODERATE 3=SEVERE")
_G_AAVE_APY          = _gauge("wallet_aave_apy_pct",           "Aave V3 USDC supply APY percent")
_G_GAS_PRICE         = _gauge("wallet_gas_price_gwei",         "Current gas price in Gwei")
_G_KILL_SWITCH       = _gauge("wallet_kill_switch_active",     "Kill switch active flag (1=active)")
_G_REGIME            = _gauge("wallet_regime_state",           "Market regime state")
_G_KELLY             = _gauge("wallet_kelly_fraction",         "Kelly fraction used for position sizing")
_G_VAR99             = _gauge("wallet_var_99_usd",             "Value-at-Risk 99th percentile in USD")
_G_OOS_SHARPE        = _gauge("wallet_oos_sharpe",             "Out-of-sample Sharpe ratio from walk-forward validation")

# Labeled gauges
_G_PILLAR_VALUE      = _gauge("wallet_pillar_value_usd",       "Pillar value in USD by strategy",       ("pillar",))
_G_RPC_LATENCY       = _gauge("wallet_rpc_latency_ms",         "RPC endpoint latency in milliseconds",  ("endpoint",))

# Counters
_C_TRADE_COUNT       = _counter("wallet_trade_count_total",    "Total number of completed trades")
_C_RPC_FAILOVER      = _counter("wallet_rpc_failover_total",   "Total RPC failover events")
_C_TX_REPAIR         = _counter("wallet_tx_repair_total",      "Total stuck-transaction repair events")
_C_CIRCUIT_OPEN      = _counter("wallet_circuit_open_total",   "Total circuit breaker open events")
_C_RETRAIN           = _counter("wallet_retrain_total",        "Total ML model retrain events")

# ---------------------------------------------------------------------------
# Mapping from legacy string names → prometheus_client objects
# ---------------------------------------------------------------------------

_SCALAR_GAUGES: dict[str, Gauge] = {
    "wallet_portfolio_total_usd":  _G_PORTFOLIO_TOTAL,
    "wallet_pnl_usd":              _G_PNL,
    "wallet_sharpe_ratio":         _G_SHARPE,
    "wallet_max_drawdown":         _G_DRAWDOWN,
    "wallet_ml_model_confidence":  _G_ML_CONFIDENCE,
    "wallet_concept_drift_level":  _G_DRIFT_LEVEL,
    "wallet_aave_apy_pct":         _G_AAVE_APY,
    "wallet_gas_price_gwei":       _G_GAS_PRICE,
    "wallet_kill_switch_active":   _G_KILL_SWITCH,
    "wallet_regime_state":         _G_REGIME,
    "wallet_kelly_fraction":       _G_KELLY,
    "wallet_var_99_usd":           _G_VAR99,
    "wallet_oos_sharpe":           _G_OOS_SHARPE,
}

_COUNTERS: dict[str, Counter] = {
    "wallet_trade_count_total":    _C_TRADE_COUNT,
    "wallet_rpc_failover_total":   _C_RPC_FAILOVER,
    "wallet_tx_repair_total":      _C_TX_REPAIR,
    "wallet_circuit_open_total":   _C_CIRCUIT_OPEN,
    "wallet_retrain_total":        _C_RETRAIN,
}

_last_update: float = 0.0


def update_metric(name: str, value: float) -> None:
    """Update a scalar metric value.

    Accepts both gauge and counter names.  For counters the value is treated
    as an *absolute* reading — the counter is incremented by the delta since
    the last call (never decremented).
    """
    global _last_update
    _last_update = time.time()

    if name in _SCALAR_GAUGES:
        _SCALAR_GAUGES[name].set(value)
    elif name in _COUNTERS:
        # prometheus_client counters only go up; call inc() with a positive delta
        counter = _COUNTERS[name]
        # We don't track previous values here, so just call inc(value) when > 0
        if value > 0:
            counter.inc(value)
    else:
        log.debug("update_metric: unknown metric %r (value=%s)", name, value)


def update_labeled_metric(name: str, label: str, value: float) -> None:
    """Update a labeled gauge value."""
    global _last_update
    _last_update = time.time()

    if name == "wallet_pillar_value_usd":
        _G_PILLAR_VALUE.labels(pillar=label).set(value)
    elif name == "wallet_rpc_latency_ms":
        _G_RPC_LATENCY.labels(endpoint=label).set(value)
    else:
        log.debug("update_labeled_metric: unknown metric %r", name)


# ---------------------------------------------------------------------------
# Async HTTP handlers
# ---------------------------------------------------------------------------


def _refresh_from_system() -> None:
    """Pull live values from running subsystems into the metrics registry."""
    # Portfolio manager
    with contextlib.suppress(Exception):
        from wallet_manager.orchestrator import _get_portfolio_manager
        pm = _get_portfolio_manager()
        if pm is not None:
            snap = pm.get_snapshot()
            _G_PORTFOLIO_TOTAL.set(snap.total_value_usd)
            for alloc in snap.allocations:
                _G_PILLAR_VALUE.labels(pillar=alloc.strategy.name).set(alloc.current_usd)

    # Concept drift
    with contextlib.suppress(Exception):
        from wallet_manager.task_factories import _drift_detector
        if _drift_detector is not None:
            dl = getattr(_drift_detector, "_drift_level", None)
            if dl is not None:
                _G_DRIFT_LEVEL.set(float(dl.value))

    # ML confidence — read from robust engine's inner engine (the active one)
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine
        robust = get_robust_alpha_engine()
        sig = robust._inner_engine.last_signal
        if sig is not None:
            _G_ML_CONFIDENCE.set(sig.confidence)

    # RPC latency
    with contextlib.suppress(Exception):
        from wallet_manager.orchestrator import _get_rpc_router
        router = _get_rpc_router()
        if router is not None:
            for chain_id, pool_info in router.pool_status().items():
                for p in pool_info:
                    url = p.get("url", "")
                    score = router.latency_score(chain_id, url)
                    if score > 0:
                        latency_ms = 1.0 / score - 1.0
                        _G_RPC_LATENCY.labels(endpoint=url[:40]).set(latency_ms)

    # Aave APY
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.yield_strategy import _apy_monitor
        if _apy_monitor is not None:
            _G_AAVE_APY.set(_apy_monitor.last_apy * 100.0)

    # Paper Trading (Overrides default portfolio amounts)
    with contextlib.suppress(Exception):
        from wallet_manager.config import get_settings
        _cfg = get_settings()
        if _cfg.paper_trade or _cfg.execution_mode == "robust_paper":
            from wallet_manager.strategies.paper_engine import _engine_instance
            if _engine_instance is not None and hasattr(_engine_instance, "_ledger"):
                _ledger = _engine_instance._ledger
                if _ledger._equity_history:
                    _equity = _ledger._equity_history[-1]
                else:
                    _equity = _engine_instance._config.initial_balance_usd + _ledger._realized_pnl
                _G_PORTFOLIO_TOTAL.set(_equity)
                _G_PNL.set(_ledger._realized_pnl)
                _COUNTERS["wallet_trade_count_total"]._value.set(float(_ledger._trade_count))
                # Pillar values from virtual balances so Grafana shows data in paper mode
                _usdc_bal = _ledger._balances.get("USDC", 0.0)
                _weth_bal = _ledger._balances.get("WETH", 0.0)
                _G_PILLAR_VALUE.labels(pillar="ALGO").set(_usdc_bal + _weth_bal)
                _G_PILLAR_VALUE.labels(pillar="YIELD").set(0.0)
                _G_PILLAR_VALUE.labels(pillar="AIRDROP").set(0.0)
                _G_PILLAR_VALUE.labels(pillar="SHORT").set(0.0)


async def _handle_metrics(request: aiohttp.web.Request) -> aiohttp.web.Response:
    _refresh_from_system()
    body = generate_latest(_REGISTRY)
    # Strip charset from content_type — aiohttp rejects compound content-type values
    ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
    return aiohttp.web.Response(
        body=body,
        content_type=ct,
    )


async def _handle_health(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"status": "ok", "ts": time.time()})


async def _handle_ready(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"status": "ready", "ts": time.time()})


async def _load_recent_runtime_audits(limit: int) -> list[dict[str, object]]:
    """Load recent audit snapshots for dashboard use."""
    try:
        from wallet_manager.resilience.analytics import AnalyticsDB

        async with AnalyticsDB() as db:
            audits = await db.list_runtime_audits(limit=limit)
        return [
            {
                "session_id": audit.session_id,
                "started_at": audit.started_at,
                "finished_at": audit.finished_at,
                "payload": audit.payload,
            }
            for audit in audits
        ]
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


async def _build_runtime_payload() -> dict[str, object]:
    """Assemble a user-facing runtime payload from live state and audit history."""
    from wallet_manager.config import get_settings
    from wallet_manager.orchestrator import get_runtime_audit_snapshot

    cfg = get_settings()
    audit = get_runtime_audit_snapshot()
    recent = await _load_recent_runtime_audits(cfg.audit_recent_sessions_limit)
    effective_audit, data_source, latest_session_id = _coalesce_runtime_audit(audit, recent)
    effective_audit = _hydrate_runtime_payload(effective_audit, cfg.execution_mode)
    research = await _load_strategy_research_snapshot(
        str(effective_audit.get("execution_mode", cfg.execution_mode)),
    )

    return {
        "status": "ok",
        "generated_at": time.time(),
        "data_source": data_source,
        "latest_session_id": latest_session_id,
        "execution_mode": effective_audit.get("execution_mode", cfg.execution_mode),
        "live_guard": effective_audit.get("live_guard"),
        "paper_gate": effective_audit.get("paper_gate"),
        "portfolio": effective_audit.get("portfolio"),
        "result_summary": effective_audit.get("result_summary"),
        "fatal_error": effective_audit.get("fatal_error"),
        "audit": effective_audit,
        "recent_audits": recent,
        "research": research,
    }


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    with contextlib.suppress(Exception):
        return cast(Mapping[str, object], dict(cast(Any, value)))
    return {}


def _coalesce_runtime_audit(
    audit: object,
    recent: list[dict[str, object]],
) -> tuple[Mapping[str, object], str, str]:
    """Return the best available audit payload for operator-facing endpoints."""
    audit_map = _as_mapping(audit)
    if audit_map:
        return audit_map, "live_snapshot", str(audit_map.get("session_id", "live"))

    if recent:
        latest = recent[0] if isinstance(recent[0], dict) else {}
        latest_map = _as_mapping(latest)
        payload_map = _as_mapping(latest_map.get("payload"))
        if payload_map:
            return payload_map, "latest_completed_audit", str(latest_map.get("session_id", "latest"))

    return {}, "empty", "n/a"


def _hydrate_runtime_payload(
    audit: Mapping[str, object],
    default_execution_mode: str,
) -> dict[str, object]:
    """Merge the best available paper-trading state into the operator payload."""
    hydrated = dict(audit)
    execution_mode = str(hydrated.get("execution_mode", default_execution_mode))
    if execution_mode not in {"robust_paper", "paper_trade"}:
        return hydrated

    with contextlib.suppress(Exception):
        from wallet_manager.strategies.paper_engine import _engine_instance

        if _engine_instance is None or not hasattr(_engine_instance, "_ledger"):
            return hydrated

        ledger = _engine_instance._ledger
        total_equity = (
            ledger._equity_history[-1]
            if ledger._equity_history
            else _engine_instance._config.initial_balance_usd + ledger._realized_pnl
        )
        portfolio = dict(_as_mapping(hydrated.get("portfolio")))
        allocations = [
            {
                "strategy": "ALGO",
                "current_usd": ledger._balances.get("USDC", 0.0) + ledger._balances.get("WETH", 0.0),
                "target_usd": portfolio.get("total_value_usd", total_equity),
                "delta_usd": 0.0,
            },
            {"strategy": "YIELD", "current_usd": 0.0, "target_usd": 0.0, "delta_usd": 0.0},
            {"strategy": "SHORT", "current_usd": 0.0, "target_usd": 0.0, "delta_usd": 0.0},
            {"strategy": "AIRDROP", "current_usd": 0.0, "target_usd": 0.0, "delta_usd": 0.0},
        ]
        portfolio.update(
            {
                "available": True,
                "needs_rebalance": False,
                "timestamp": time.time(),
                "total_value_usd": total_equity,
                "allocations": allocations,
            }
        )
        paper_gate = dict(_as_mapping(hydrated.get("paper_gate")))
        paper_gate["current_trade_count"] = ledger._trade_count
        result_summary = dict(_as_mapping(hydrated.get("result_summary")))
        if not result_summary:
            result_summary = {"wallets_succeeded": 1, "wallets_total": 1, "wallets_failed": 0}

        hydrated["portfolio"] = portfolio
        hydrated["paper_gate"] = paper_gate
        hydrated["result_summary"] = result_summary
    return hydrated


def _fmt_money(value: object) -> str:
    return f"${_safe_float(value):,.2f}"


def _fmt_count(value: object) -> str:
    return f"{_safe_int(value)}"


def _fmt_seconds(value: object) -> str:
    seconds = max(0, _safe_int(value, -1))
    if seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _fmt_relative_ts(value: object, now_ts: float) -> str:
    ts = _safe_float(value, -1.0)
    if ts < 0:
        return "-"
    delta = max(0, int(now_ts - ts))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _status_badge(label: str, tone: str) -> str:
    return f"<span class='badge {tone}'>{html.escape(label)}</span>"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return float(value)
    except Exception:
        pass
    return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str):
            return int(float(value))
    except Exception:
        pass
    return default


def _sparkline(values: list[float], stroke: str) -> str:
    width = 240
    height = 56
    if not values:
        return "<div class='sparkline-empty'>No data</div>"
    if len(values) == 1:
        values = values * 2

    low = min(values)
    high = max(values)
    spread = high - low or 1.0
    step = width / max(1, len(values) - 1)
    points: list[str] = []
    for idx, val in enumerate(values):
        x = idx * step
        y = height - (((val - low) / spread) * (height - 8) + 4)
        points.append(f"{x:.1f},{y:.1f}")
    joined = " ".join(points)
    return (
        f"<svg viewBox='0 0 {width} {height}' class='sparkline' preserveAspectRatio='none'>"
        f"<polyline fill='none' stroke='{stroke}' stroke-width='3' stroke-linecap='round' "
        f"stroke-linejoin='round' points='{joined}' />"
        "</svg>"
    )


def _fmt_pct(value: object, digits: int = 2) -> str:
    return f"{_safe_float(value):.{digits}f}%"


def _fmt_ts(value: object) -> str:
    ts = _safe_float(value, -1.0)
    if ts < 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def _build_strategy_research_notes(snapshot: Mapping[str, object]) -> list[str]:
    """Translate paper-trading telemetry into operator-facing recommendations."""
    notes: list[str] = []
    trade_count = _safe_int(snapshot.get("trade_count", 0))
    closed_trades = _safe_int(snapshot.get("closed_trade_count", 0))
    win_rate = _safe_float(snapshot.get("win_rate_pct", 0.0))
    max_drawdown = _safe_float(snapshot.get("max_drawdown_pct", 0.0))
    avg_slippage = _safe_float(snapshot.get("avg_slippage_bps", 0.0))
    avg_gas = _safe_float(snapshot.get("avg_gas_cost_usd", 0.0))
    realized = _safe_float(snapshot.get("realized_pnl_usd", 0.0))
    net_after_gas = _safe_float(snapshot.get("net_after_gas_usd", 0.0))
    last_trade_age = _safe_float(snapshot.get("last_trade_age_seconds", -1.0), -1.0)
    exposure_pct = _safe_float(snapshot.get("weth_exposure_pct", 0.0))

    if trade_count == 0:
        notes.append("Henuz paper trade yok. Bu asamada strateji kalitesinden cok sinyal uretilip uretilmedigini izle.")
    elif closed_trades == 0:
        notes.append("Acik pozisyon var ama tamamlanmis round-trip yok. Gercek edge yorumu icin en az birkaç kapanmis trade gerekli.")
    if 0 < trade_count < 10:
        notes.append("Orneklem henuz cok kucuk. Bu paneli karar vermek icin degil davranis toplamak icin kullan.")
    if closed_trades >= 5 and win_rate < 45.0:
        notes.append("Kapanmis trade win rate zayif. Entry filtresi veya cikis mantigi fazla gevsek olabilir.")
    if trade_count >= 5 and net_after_gas < 0 <= realized:
        notes.append("Brut PnL pozitif olsa bile gas net sonucu siliyor. Trade boyutu veya frekansi yeniden ayarlanabilir.")
    if avg_slippage >= 12.0:
        notes.append("Ortalama slippage yuksek. Likidite/giris zamani tarafini iyilestirmek faydali olur.")
    if avg_gas >= 1.0:
        notes.append("Ortalama gas maliyeti dikkat cekiyor. Yavas piyasa saatleri veya trade esigi denenebilir.")
    if max_drawdown >= 3.0:
        notes.append("Drawdown yukselmeye baslamis. Sizing veya ard arda kayip frenleri yeniden kontrol edilmeli.")
    if last_trade_age > 0 and last_trade_age >= 6 * 3600:
        notes.append("Uzun suredir yeni trade yok. Bu bug olmayabilir; macro filter veya confidence esikleri sebep olabilir.")
    if exposure_pct >= 35.0:
        notes.append("Spot WETH maruziyeti yukselmis. Paper olsa da cikis mantigi ve rebalancing davranisi izlenmeli.")
    if not notes:
        notes.append("Veri akisi saglikli gorunuyor. Simdi odak: daha cok kapanmis trade, net expectancy ve regime bazli tutarlilik.")
    return notes


async def _load_strategy_research_snapshot(execution_mode: str) -> dict[str, object]:
    """Build a paper-trading research snapshot for dashboard and API endpoints."""
    if execution_mode not in {"robust_paper", "paper_trade"}:
        return {"available": False, "mode": execution_mode}

    from wallet_manager.config import get_settings

    cfg = get_settings()
    db_path = Path(getattr(cfg, "paper_db_path", "paper_trades.db"))
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    snapshot: dict[str, object] = {
        "available": False,
        "mode": execution_mode,
        "db_path": str(db_path),
        "trade_count": 0,
        "closed_trade_count": 0,
        "buy_count": 0,
        "sell_count": 0,
        "win_rate_pct": 0.0,
        "avg_trade_size_usd": 0.0,
        "avg_slippage_bps": 0.0,
        "avg_gas_cost_usd": 0.0,
        "total_gas_cost_usd": 0.0,
        "realized_pnl_usd": 0.0,
        "net_after_gas_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "peak_equity_usd": 0.0,
        "current_equity_usd": 0.0,
        "paper_progress_pct": 0,
        "required_trade_count": cfg.min_paper_trades_before_live,
        "last_trade_at": None,
        "last_trade_age_seconds": None,
        "weth_exposure_pct": 0.0,
        "equity_curve": [],
        "recent_trades": [],
        "recent_decisions": [],
        "decision_analytics": {
            "summary": {
                "total_decisions": 0,
                "executed_decisions": 0,
                "no_trade_decisions": 0,
                "skip_decisions": 0,
            },
            "no_trade_reasons": [],
            "regime_breakdown": [],
            "setup_breakdown": [],
        },
        "best_trade": None,
        "worst_trade": None,
        "last_trade": None,
        "notes": [],
    }

    try:
        from wallet_manager.strategies.paper_engine import _engine_instance

        engine_ready = (
            _engine_instance is not None
            and hasattr(_engine_instance, "_ledger")
            and _engine_instance._ledger._conn is not None
        )
        if engine_ready:
            engine = _engine_instance
            if engine is None:
                return snapshot
            perf = await engine.get_performance_report()
            pnl_summary = _as_mapping(perf.get("virtual_pnl"))
            drawdown = _as_mapping(perf.get("drawdown"))
            balances = _as_mapping(perf.get("balances"))
            recent_trades = await engine._ledger.get_recent_trades(12)
            recent_decisions = await engine._ledger.get_recent_decisions(12)
            decision_analytics = await engine._ledger.get_decision_analytics()
            trade_extremes = await engine._ledger.get_best_worst_last_trade()
            # Phase 2: Advanced analytics
            advanced_analytics = await engine._ledger.get_advanced_analytics()
            hold_time_stats = await engine._ledger.get_hold_time_stats()
            performance_by_regime = await engine._ledger.get_performance_by_regime()
            performance_by_setup = await engine._ledger.get_performance_by_setup()
            equity_curve_7d = await engine._ledger.get_equity_curve_windowed(
                window_seconds=7 * 86400, max_points=100
            )
            equity_curve_1d = await engine._ledger.get_equity_curve_windowed(
                window_seconds=86400, max_points=48
            )
            # Phase 9: Live readiness
            live_readiness = await engine._ledger.get_live_readiness_score()

            # Phase 5-7/10: MacroCalendar, RiskCommittee, ExperimentLab, ContinuousLearner
            macro_calendar_data: dict[str, object] = {}
            risk_committee_data: dict[str, object] = {}
            experiment_lab_data: dict[str, object] = {}
            learning_data: dict[str, object] = {}
            with contextlib.suppress(Exception):
                from wallet_manager.market_intelligence import get_macro_calendar
                _cal = get_macro_calendar()
                macro_calendar_data = _cal.to_dict()
                _suppress, _sup_reason = _cal.should_suppress_alpha()
                macro_calendar_data["alpha_suppressed"] = _suppress
                macro_calendar_data["suppression_reason"] = _sup_reason
            with contextlib.suppress(Exception):
                from wallet_manager.risk_committee import get_risk_committee
                _rc_snapshot = await get_risk_committee().evaluate(paper_engine=engine)
                risk_committee_data = _rc_snapshot.to_dict()
            with contextlib.suppress(Exception):
                from wallet_manager.experiment_lab import get_experiment_lab
                experiment_lab_data = get_experiment_lab().to_dict()
            with contextlib.suppress(Exception):
                from wallet_manager.continuous_learning import get_continuous_learner
                learning_data = await get_continuous_learner().generate_learning_report(engine)

            conn = engine._ledger._get_conn()

            equity_curve: list[dict[str, object]] = []
            async with conn.execute(
                "SELECT timestamp, total_equity_usd FROM paper_balance_history "
                "ORDER BY timestamp DESC LIMIT 48"
            ) as cur:
                rows = list(await cur.fetchall())
            for row in reversed(rows):
                equity_curve.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "total_equity_usd": float(row["total_equity_usd"]),
                    }
                )

            async with conn.execute(
                "SELECT "
                "COUNT(*) AS trade_count, "
                "SUM(CASE WHEN direction = 'BUY' THEN 1 ELSE 0 END) AS buy_count, "
                "SUM(CASE WHEN direction = 'SELL' THEN 1 ELSE 0 END) AS sell_count, "
                "COALESCE(AVG(size_usd), 0.0) AS avg_trade_size_usd, "
                "COALESCE(AVG(gas_cost_usd), 0.0) AS avg_gas_cost_usd, "
                "COALESCE(SUM(gas_cost_usd), 0.0) AS total_gas_cost_usd, "
                "MAX(timestamp) AS last_trade_at "
                "FROM paper_trades"
            ) as cur:
                trade_row = _as_mapping(await cur.fetchone())

            snapshot.update(
                {
                    "available": True,
                    "trade_count": _safe_int(trade_row.get("trade_count", 0)),
                    "closed_trade_count": _safe_int(trade_row.get("sell_count", 0)),
                    "buy_count": _safe_int(trade_row.get("buy_count", 0)),
                    "sell_count": _safe_int(trade_row.get("sell_count", 0)),
                    "win_rate_pct": _safe_float(pnl_summary.get("win_rate", 0.0)),
                    "avg_trade_size_usd": _safe_float(trade_row.get("avg_trade_size_usd", 0.0)),
                    "avg_slippage_bps": _safe_float(pnl_summary.get("avg_slippage_bps", 0.0)),
                    "avg_gas_cost_usd": _safe_float(trade_row.get("avg_gas_cost_usd", 0.0)),
                    "total_gas_cost_usd": _safe_float(trade_row.get("total_gas_cost_usd", 0.0)),
                    "realized_pnl_usd": _safe_float(pnl_summary.get("realized_pnl", 0.0)),
                    "max_drawdown_pct": _safe_float(drawdown.get("max_drawdown", 0.0)) * 100.0,
                    "peak_equity_usd": _safe_float(drawdown.get("peak_equity", 0.0)),
                    "current_equity_usd": _safe_float(equity_curve[-1]["total_equity_usd"]) if equity_curve else _safe_float(pnl_summary.get("peak_equity", 0.0)),
                    "equity_curve": equity_curve,
                    "equity_curve_7d": equity_curve_7d,
                    "equity_curve_1d": equity_curve_1d,
                    "recent_trades": recent_trades,
                    "recent_decisions": recent_decisions,
                    "decision_analytics": decision_analytics,
                    "best_trade": trade_extremes.get("best_trade"),
                    "worst_trade": trade_extremes.get("worst_trade"),
                    "last_trade": trade_extremes.get("last_trade"),
                    "advanced_analytics": advanced_analytics,
                    "hold_time_stats": hold_time_stats,
                    "performance_by_regime": performance_by_regime,
                    "performance_by_setup": performance_by_setup,
                    "live_readiness": live_readiness,
                    "macro_calendar": macro_calendar_data,
                    "risk_committee": risk_committee_data,
                    "experiment_lab": experiment_lab_data,
                    "learning": learning_data,
                }
            )

            usdc_balance = _safe_float(balances.get("USDC", 0.0))
            weth_exposure_usd = max(0.0, _safe_float(snapshot["current_equity_usd"]) - usdc_balance)
            current_equity = max(_safe_float(snapshot["current_equity_usd"]), 1e-9)
            snapshot["weth_exposure_pct"] = (weth_exposure_usd / current_equity) * 100.0
            snapshot["last_trade_at"] = trade_row.get("last_trade_at")
        elif db_path.exists():
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    "SELECT "
                    "COUNT(*) AS trade_count, "
                    "SUM(CASE WHEN direction = 'BUY' THEN 1 ELSE 0 END) AS buy_count, "
                    "SUM(CASE WHEN direction = 'SELL' THEN 1 ELSE 0 END) AS sell_count, "
                    "COALESCE(AVG(size_usd), 0.0) AS avg_trade_size_usd, "
                    "COALESCE(AVG(slippage_bps), 0.0) AS avg_slippage_bps, "
                    "COALESCE(AVG(gas_cost_usd), 0.0) AS avg_gas_cost_usd, "
                    "COALESCE(SUM(gas_cost_usd), 0.0) AS total_gas_cost_usd, "
                    "COALESCE(SUM(pnl_usd), 0.0) AS realized_pnl_usd, "
                    "COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), 0) AS win_count, "
                    "MAX(timestamp) AS last_trade_at "
                    "FROM paper_trades"
                ) as cur:
                    trade_row = _as_mapping(await cur.fetchone())

                async with conn.execute(
                    "SELECT MAX(max_drawdown) AS max_drawdown, MAX(total_equity_usd) AS peak_equity "
                    "FROM paper_snapshots"
                ) as cur:
                    drawdown_row = _as_mapping(await cur.fetchone())

                async with conn.execute(
                    "SELECT timestamp, total_equity_usd, usdc_balance, weth_balance "
                    "FROM paper_balance_history ORDER BY timestamp DESC LIMIT 48"
                ) as cur:
                    history_rows = list(await cur.fetchall())

                async with conn.execute(
                    "SELECT direction, asset_pair, entry_price, simulated_fill_price, slippage_bps, "
                    "size_usd, pnl_usd, timestamp FROM paper_trades ORDER BY timestamp DESC LIMIT 12"
                ) as cur:
                    trade_rows = list(await cur.fetchall())

                has_decision_table = False
                async with conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'paper_decisions'"
                ) as cur:
                    has_decision_table = await cur.fetchone() is not None

                decision_rows: list[aiosqlite.Row] = []
                reason_rows: list[aiosqlite.Row] = []
                regime_rows: list[aiosqlite.Row] = []
                setup_rows: list[aiosqlite.Row] = []
                decision_summary_row: Mapping[str, object] = {}
                if has_decision_table:
                    async with conn.execute(
                        "SELECT timestamp, wallet_id, cycle_phase, action, regime_label, "
                        "setup_tag, decision_class, primary_reason, reason_codes, raw_alpha, "
                        "robust_alpha, confidence, adf_pvalue, stationarity_d, volatility_regime, "
                        "risk_position_usd, trade_budget_usd, gas_cost_usd, total_equity_usd, pnl_usd, notes "
                        "FROM paper_decisions ORDER BY timestamp DESC LIMIT 12"
                    ) as cur:
                        decision_rows = list(await cur.fetchall())
                    async with conn.execute(
                        "SELECT primary_reason, COUNT(*) AS cnt "
                        "FROM paper_decisions "
                        "WHERE decision_class IN ('NO_TRADE', 'SKIP', 'ERROR') "
                        "GROUP BY primary_reason ORDER BY cnt DESC, primary_reason ASC LIMIT 8"
                    ) as cur:
                        reason_rows = list(await cur.fetchall())
                    async with conn.execute(
                        "SELECT regime_label, COUNT(*) AS decision_count, "
                        "SUM(CASE WHEN action IN ('BUY', 'SELL') THEN 1 ELSE 0 END) AS executed_count, "
                        "COALESCE(SUM(pnl_usd), 0.0) AS realized_pnl_usd "
                        "FROM paper_decisions GROUP BY regime_label "
                        "ORDER BY decision_count DESC, regime_label ASC LIMIT 8"
                    ) as cur:
                        regime_rows = list(await cur.fetchall())
                    async with conn.execute(
                        "SELECT setup_tag, COUNT(*) AS decision_count, "
                        "SUM(CASE WHEN action IN ('BUY', 'SELL') THEN 1 ELSE 0 END) AS executed_count, "
                        "COALESCE(SUM(pnl_usd), 0.0) AS realized_pnl_usd "
                        "FROM paper_decisions GROUP BY setup_tag "
                        "ORDER BY decision_count DESC, setup_tag ASC LIMIT 8"
                    ) as cur:
                        setup_rows = list(await cur.fetchall())
                    async with conn.execute(
                        "SELECT COUNT(*) AS total_decisions, "
                        "SUM(CASE WHEN action IN ('BUY', 'SELL') THEN 1 ELSE 0 END) AS executed_decisions, "
                        "SUM(CASE WHEN decision_class = 'NO_TRADE' THEN 1 ELSE 0 END) AS no_trade_decisions, "
                        "SUM(CASE WHEN decision_class = 'SKIP' THEN 1 ELSE 0 END) AS skip_decisions "
                        "FROM paper_decisions"
                    ) as cur:
                        decision_summary_row = _as_mapping(await cur.fetchone())

            equity_curve = [
                {
                    "timestamp": float(row["timestamp"]),
                    "total_equity_usd": float(row["total_equity_usd"]),
                }
                for row in reversed(history_rows)
            ]
            recent_trades = [
                {
                    "direction": row["direction"],
                    "pair": row["asset_pair"],
                    "entry_price": float(row["entry_price"]),
                    "fill_price": float(row["simulated_fill_price"]),
                    "slippage_bps": float(row["slippage_bps"]),
                    "size_usd": float(row["size_usd"]),
                    "pnl_usd": float(row["pnl_usd"]),
                    "timestamp": float(row["timestamp"]),
                }
                for row in trade_rows
            ]
            recent_decisions = [
                {
                    "timestamp": float(row["timestamp"]),
                    "wallet_id": row["wallet_id"],
                    "cycle_phase": row["cycle_phase"],
                    "action": row["action"],
                    "regime_label": row["regime_label"],
                    "setup_tag": row["setup_tag"],
                    "decision_class": row["decision_class"],
                    "primary_reason": row["primary_reason"],
                    "reason_codes": json.loads(row["reason_codes"]),
                    "raw_alpha": float(row["raw_alpha"]),
                    "robust_alpha": float(row["robust_alpha"]),
                    "confidence": float(row["confidence"]),
                    "adf_pvalue": float(row["adf_pvalue"]),
                    "stationarity_d": float(row["stationarity_d"]),
                    "volatility_regime": row["volatility_regime"],
                    "risk_position_usd": float(row["risk_position_usd"]),
                    "trade_budget_usd": float(row["trade_budget_usd"]),
                    "gas_cost_usd": float(row["gas_cost_usd"]),
                    "total_equity_usd": float(row["total_equity_usd"]),
                    "pnl_usd": float(row["pnl_usd"]),
                    "notes": row["notes"],
                }
                for row in decision_rows
            ]
            decision_analytics = {
                "summary": {
                    "total_decisions": _safe_int(decision_summary_row.get("total_decisions", 0)),
                    "executed_decisions": _safe_int(decision_summary_row.get("executed_decisions", 0)),
                    "no_trade_decisions": _safe_int(decision_summary_row.get("no_trade_decisions", 0)),
                    "skip_decisions": _safe_int(decision_summary_row.get("skip_decisions", 0)),
                },
                "no_trade_reasons": [
                    {"reason": row["primary_reason"], "count": int(row["cnt"] or 0)}
                    for row in reason_rows
                ],
                "regime_breakdown": [
                    {
                        "regime_label": row["regime_label"],
                        "decision_count": int(row["decision_count"] or 0),
                        "executed_count": int(row["executed_count"] or 0),
                        "realized_pnl_usd": float(row["realized_pnl_usd"] or 0.0),
                    }
                    for row in regime_rows
                ],
                "setup_breakdown": [
                    {
                        "setup_tag": row["setup_tag"],
                        "decision_count": int(row["decision_count"] or 0),
                        "executed_count": int(row["executed_count"] or 0),
                        "realized_pnl_usd": float(row["realized_pnl_usd"] or 0.0),
                    }
                    for row in setup_rows
                ],
            }
            current_equity = _safe_float(equity_curve[-1]["total_equity_usd"]) if equity_curve else _safe_float(drawdown_row.get("peak_equity", 0.0))
            latest_history = _as_mapping(history_rows[0]) if history_rows else {}
            usdc_balance = _safe_float(latest_history.get("usdc_balance", current_equity))
            weth_exposure_usd = max(0.0, current_equity - usdc_balance)
            snapshot.update(
                {
                    "available": True,
                    "trade_count": _safe_int(trade_row.get("trade_count", 0)),
                    "closed_trade_count": _safe_int(trade_row.get("sell_count", 0)),
                    "buy_count": _safe_int(trade_row.get("buy_count", 0)),
                    "sell_count": _safe_int(trade_row.get("sell_count", 0)),
                    "win_rate_pct": (
                        (_safe_float(trade_row.get("win_count", 0)) / max(_safe_float(trade_row.get("sell_count", 0)), 1.0)) * 100.0
                        if _safe_int(trade_row.get("sell_count", 0)) > 0
                        else 0.0
                    ),
                    "avg_trade_size_usd": _safe_float(trade_row.get("avg_trade_size_usd", 0.0)),
                    "avg_slippage_bps": _safe_float(trade_row.get("avg_slippage_bps", 0.0)),
                    "avg_gas_cost_usd": _safe_float(trade_row.get("avg_gas_cost_usd", 0.0)),
                    "total_gas_cost_usd": _safe_float(trade_row.get("total_gas_cost_usd", 0.0)),
                    "realized_pnl_usd": _safe_float(trade_row.get("realized_pnl_usd", 0.0)),
                    "max_drawdown_pct": _safe_float(drawdown_row.get("max_drawdown", 0.0)) * 100.0,
                    "peak_equity_usd": _safe_float(drawdown_row.get("peak_equity", 0.0)),
                    "current_equity_usd": current_equity,
                    "equity_curve": equity_curve,
                    "recent_trades": recent_trades,
                    "recent_decisions": recent_decisions,
                    "decision_analytics": decision_analytics,
                    "weth_exposure_pct": (weth_exposure_usd / max(current_equity, 1e-9)) * 100.0,
                    "last_trade_at": trade_row.get("last_trade_at"),
                }
            )
    except Exception as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"

    snapshot["net_after_gas_usd"] = _safe_float(snapshot["realized_pnl_usd"]) - _safe_float(snapshot["total_gas_cost_usd"])
    required_trade_count = max(_safe_int(snapshot.get("required_trade_count", 0), 0), 1)
    snapshot["paper_progress_pct"] = min(
        100,
        int((_safe_float(snapshot.get("trade_count", 0.0)) / required_trade_count) * 100),
    )
    last_trade_at = _safe_float(snapshot.get("last_trade_at", -1.0), -1.0)
    if last_trade_at >= 0:
        snapshot["last_trade_age_seconds"] = max(0.0, time.time() - last_trade_at)
    snapshot["notes"] = _build_strategy_research_notes(snapshot)
    return snapshot


def _render_dashboard_html(payload: dict[str, object]) -> str:
    """Render a richer operator dashboard without requiring a frontend build."""
    execution_mode = html.escape(str(payload.get("execution_mode", "unknown")))
    data_source = html.escape(str(payload.get("data_source", "unknown")))
    latest_session_id = html.escape(str(payload.get("latest_session_id", "n/a")))
    live_guard = _as_mapping(payload.get("live_guard"))
    paper_gate = _as_mapping(payload.get("paper_gate"))
    portfolio = _as_mapping(payload.get("portfolio"))
    result_summary = _as_mapping(payload.get("result_summary"))
    audit = _as_mapping(payload.get("audit"))
    fatal_error = payload.get("fatal_error") or "none"
    recent = payload.get("recent_audits") or []
    if not isinstance(recent, list):
        recent = []
    research = _as_mapping(payload.get("research"))
    generated_at = _safe_float(payload.get("generated_at", time.time()) or time.time(), time.time())
    refresh_seconds = 15

    total_value = portfolio.get("total_value_usd", 0.0)
    live_pass = bool(live_guard.get("passed", True))
    paper_pass = bool(paper_gate.get("passed", True))
    wallets_ok = _fmt_count(result_summary.get("wallets_succeeded", 0))
    wallets_total = _fmt_count(result_summary.get("wallets_total", 0))
    paper_trades = _fmt_count(paper_gate.get("current_trade_count", 0))
    paper_needed = _fmt_count(paper_gate.get("required_trade_count", 0))
    portfolio_available = bool(portfolio.get("available", False))
    needs_rebalance = bool(portfolio.get("needs_rebalance", False))
    live_badge = _status_badge("PASS" if live_pass else "BLOCKED", "good" if live_pass else "bad")
    paper_badge = _status_badge("PASS" if paper_pass else "WAIT", "good" if paper_pass else "warn")
    availability_badge = _status_badge(
        "AVAILABLE" if portfolio_available else "UNAVAILABLE",
        "good" if portfolio_available else "warn",
    )
    rebalance_badge = _status_badge(
        "REBALANCE" if needs_rebalance else "STABLE",
        "warn" if needs_rebalance else "good",
    )
    paper_progress = min(
        100,
        int(
            (_safe_float(paper_gate.get("current_trade_count", 0.0)) / max(_safe_float(paper_gate.get("required_trade_count", 0.0), 1.0), 1.0))
            * 100
        ),
    )

    endpoint_cards = [
        ("Dashboard", "/dashboard"),
        ("Runtime API", "/api/runtime"),
        ("Research API", "/api/research"),
        ("Health", "/health"),
        ("Metrics", "/metrics"),
    ]
    endpoint_links = "".join(
        f"<a class='endpoint' href='{href}'>{html.escape(label)}</a>"
        for label, href in endpoint_cards
    )

    allocations = portfolio.get("allocations", [])
    allocation_rows: list[str] = []
    if isinstance(allocations, list) and allocations:
        for alloc in allocations:
            alloc_map = _as_mapping(alloc)
            allocation_rows.append(
                "<tr>"
                f"<td>{html.escape(str(alloc_map.get('strategy', 'unknown')))}</td>"
                f"<td>{_fmt_money(alloc_map.get('current_usd', 0.0))}</td>"
                f"<td>{_fmt_money(alloc_map.get('target_usd', 0.0))}</td>"
                f"<td>{_fmt_money(alloc_map.get('delta_usd', 0.0))}</td>"
                "</tr>"
            )
    if not allocation_rows:
        allocation_rows.append("<tr><td colspan='4'>No allocation snapshot yet.</td></tr>")

    recent_payloads = [
        _as_mapping(item.get("payload"))
        for item in recent
        if isinstance(item, dict) and _as_mapping(item.get("payload"))
    ]
    success_rate = 0
    if recent_payloads:
        success_hits = 0
        success_total = 0
        for payload_map in recent_payloads[:8]:
            summary_map = _as_mapping(payload_map.get("result_summary"))
            success_hits += _safe_int(summary_map.get("wallets_succeeded", 0))
            success_total += max(1, _safe_int(summary_map.get("wallets_total", 0)))
        success_rate = int((success_hits / success_total) * 100) if success_total else 0

    duration_values: list[float] = []
    portfolio_values: list[float] = []
    for item in recent[:8]:
        if not isinstance(item, dict):
            continue
        payload_map = _as_mapping(item.get("payload"))
        duration_values.append(max(0.0, _safe_float(item.get("finished_at")) - _safe_float(item.get("started_at"))))
        portfolio_values.append(_safe_float(_as_mapping(payload_map.get("portfolio")).get("total_value_usd", 0.0)))
    avg_duration = _fmt_seconds(sum(duration_values) / len(duration_values)) if duration_values else "-"
    duration_chart = _sparkline(duration_values, "#a85517")
    portfolio_chart = _sparkline(portfolio_values, "#166534")

    recommendations: list[str] = []
    if execution_mode == "robust_paper":
        recommendations.append("Bot guvenli paper modda. Canli moda gecmeden once daha fazla tamamlanmis round-trip trade toplanmali.")
    if not paper_pass:
        recommendations.append("Paper gate henuz dolmamis; canli gecis icin gereken islem sayisi tamamlanmamis.")
    if fatal_error not in (None, "", "none"):
        recommendations.append(f"Son oturumda hata gorulmus: {fatal_error}. Canli moda gecmeden once sebep kapatilmali.")
    if not recommendations:
        recommendations.append("Kritik hata gorunmuyor. Simdi izlenecek sey signal kalitesi, trade sayisi ve gas sonrasi net beklenti.")

    research_closed_count = _fmt_count(research.get("closed_trade_count", 0))
    research_net_after_gas = _fmt_money(research.get("net_after_gas_usd", 0.0))
    research_avg_slippage = _fmt_pct(research.get("avg_slippage_bps", 0.0), 1).replace("%", " bps")
    research_win_rate = _fmt_pct(research.get("win_rate_pct", 0.0))
    research_last_trade_at = _fmt_ts(research.get("last_trade_at"))
    research_exposure = _fmt_pct(research.get("weth_exposure_pct", 0.0))
    research_equity_curve = research.get("equity_curve", [])
    if not isinstance(research_equity_curve, list):
        research_equity_curve = []
    # Phase 2: Advanced analytics variables
    adv = _as_mapping(research.get("advanced_analytics"))
    hold = _as_mapping(research.get("hold_time_stats"))
    readiness = _as_mapping(research.get("live_readiness"))
    readiness_score = _safe_int(readiness.get("score", 0))
    readiness_max = max(_safe_int(readiness.get("max_score", 100)), 1)
    readiness_pct = int((readiness_score / readiness_max) * 100)
    readiness_rec = html.escape(str(readiness.get("recommendation", "Insufficient data.")))
    readiness_gates = readiness.get("gates") or {}
    if not isinstance(readiness_gates, dict):
        readiness_gates = {}

    # Live-readiness gate rows
    gate_rows_html: list[str] = []
    for gate_key, gate_val in readiness_gates.items():
        if not isinstance(gate_val, dict):
            continue
        passed = bool(gate_val.get("passed", False))
        tone = "good" if passed else "warn"
        icon = "✓" if passed else "○"
        gate_rows_html.append(
            "<tr>"
            f"<td><span class='badge {tone}'>{icon}</span></td>"
            f"<td>{html.escape(str(gate_val.get('label', gate_key)))}</td>"
            f"<td>{html.escape(str(gate_val.get('value', '-')))}</td>"
            f"<td>{html.escape(str(gate_val.get('target', '-')))}</td>"
            f"<td>{_safe_int(gate_val.get('earned', 0))}/{_safe_int(gate_val.get('points', 0))}</td>"
            "</tr>"
        )
    if not gate_rows_html:
        gate_rows_html.append("<tr><td colspan='5'>Readiness data will appear after closed trades accumulate.</td></tr>")

    # Phase 5-7/10: MacroCalendar, RiskCommittee, ExperimentLab, ContinuousLearner
    macro_cal = _as_mapping(research.get("macro_calendar"))
    risk_com = _as_mapping(research.get("risk_committee"))
    exp_lab = _as_mapping(research.get("experiment_lab"))

    # Risk committee brake rows
    brake_rows_html: list[str] = []
    for br in cast(list[object], risk_com.get("brake_results", [])):
        if not isinstance(br, dict):
            continue
        verdict_val = str(br.get("verdict", "ALLOW"))
        tone = "good" if verdict_val == "ALLOW" else "bad" if verdict_val == "BLOCK" else "warn"
        brake_rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(br.get('brake', '-')))}</td>"
            f"<td><span class='badge {tone}'>{html.escape(verdict_val)}</span></td>"
            f"<td>{html.escape(str(br.get('reason', '-')))}</td>"
            f"<td>{_safe_float(br.get('value', 0.0)):.3f}</td>"
            "</tr>"
        )
    if not brake_rows_html:
        brake_rows_html.append("<tr><td colspan='4'>Risk committee data will appear after the engine initializes.</td></tr>")

    rc_verdict = str(risk_com.get("overall_verdict", "ALLOW"))
    rc_score = _safe_int(risk_com.get("risk_score", 0))
    rc_narrative = html.escape(str(risk_com.get("narrative", "No data yet.")))
    rc_tone = "good" if rc_verdict == "ALLOW" else "bad" if rc_verdict == "BLOCK" else "warn"

    # Macro calendar: next 3 events
    upcoming_events = macro_cal.get("next_48h", [])
    if not isinstance(upcoming_events, list):
        upcoming_events = []
    macro_event_rows: list[str] = []
    for evt in cast(list[object], upcoming_events[:5]):
        if not isinstance(evt, dict):
            continue
        impact = str(evt.get("impact", "MEDIUM"))
        impact_tone = "bad" if impact == "CRITICAL" else "warn" if impact == "HIGH" else "muted-chip"
        macro_event_rows.append(
            "<tr>"
            f"<td>{html.escape(str(evt.get('name', '-')))}</td>"
            f"<td>{html.escape(str(evt.get('datetime_utc', '-')))}</td>"
            f"<td><span class='badge {impact_tone}'>{html.escape(impact)}</span></td>"
            f"<td>{html.escape(str(evt.get('category', '-')))}</td>"
            "</tr>"
        )
    if not macro_event_rows:
        macro_event_rows.append("<tr><td colspan='4'>No upcoming macro events in calendar.</td></tr>")
    alpha_suppressed = bool(macro_cal.get("alpha_suppressed", False))
    macro_badge_tone = "bad" if alpha_suppressed else "good"
    macro_badge_text = "SUPPRESSED" if alpha_suppressed else "ACTIVE"

    # Experiment lab variant rows
    exp_comparison = _as_mapping(exp_lab.get("comparison"))
    exp_winner = html.escape(str(exp_comparison.get("winner", "baseline") or "baseline"))
    exp_variant_rows: list[str] = []
    for variant in cast(list[object], exp_comparison.get("variants", [])):
        if not isinstance(variant, dict):
            continue
        is_winner = str(variant.get("name", "")) == str(exp_comparison.get("winner", ""))
        name_html = f"<b>{html.escape(str(variant.get('name', '-')))}</b>" if is_winner else html.escape(str(variant.get("name", "-")))
        agree_pct = _safe_float(variant.get("agreement_pct", 0.0))
        agree_tone = "good" if agree_pct >= 70 else "warn"
        exp_variant_rows.append(
            "<tr>"
            f"<td>{name_html}</td>"
            f"<td>{_fmt_count(variant.get('cycles', 0))}</td>"
            f"<td>{_fmt_count(variant.get('variant_would_trade', 0))}</td>"
            f"<td><span class='badge {agree_tone}'>{agree_pct:.1f}%</span></td>"
            f"<td>{_fmt_money(variant.get('shared_realized_pnl', 0.0))}</td>"
            "</tr>"
        )
    if not exp_variant_rows:
        exp_variant_rows.append("<tr><td colspan='5'>Variant data will appear after shadow cycles accumulate.</td></tr>")

    # Phase 2: Regime performance table (enhanced with win rate + expectancy)
    perf_regime_list = research.get("performance_by_regime", [])
    if not isinstance(perf_regime_list, list):
        perf_regime_list = []
    regime_perf_rows: list[str] = []
    for row in cast(list[object], perf_regime_list):
        if not isinstance(row, dict):
            continue
        wr = _safe_float(row.get("win_rate_pct", 0.0))
        wr_tone = "good" if wr >= 55 else "warn" if wr >= 45 else "bad"
        exp = _safe_float(row.get("expectancy_usd", 0.0))
        exp_tone = "good" if exp > 0 else "bad"
        regime_perf_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('regime_label', '-')))}</td>"
            f"<td>{_safe_int(row.get('decision_count', 0))}</td>"
            f"<td>{_safe_int(row.get('closed_count', 0))}</td>"
            f"<td><span class='badge {wr_tone}'>{wr:.1f}%</span></td>"
            f"<td><span class='badge {exp_tone}'>{_fmt_money(exp)}</span></td>"
            f"<td>{_fmt_money(row.get('realized_pnl_usd', 0.0))}</td>"
            "</tr>"
        )
    if not regime_perf_rows:
        regime_perf_rows.append("<tr><td colspan='6'>Regime performance will appear after trades accumulate.</td></tr>")

    # Phase 2: Setup performance table (enhanced with win rate + expectancy)
    perf_setup_list = research.get("performance_by_setup", [])
    if not isinstance(perf_setup_list, list):
        perf_setup_list = []
    setup_perf_rows: list[str] = []
    for row in cast(list[object], perf_setup_list):
        if not isinstance(row, dict):
            continue
        wr = _safe_float(row.get("win_rate_pct", 0.0))
        wr_tone = "good" if wr >= 55 else "warn" if wr >= 45 else "bad"
        exp = _safe_float(row.get("expectancy_usd", 0.0))
        exp_tone = "good" if exp > 0 else "bad"
        setup_perf_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('setup_tag', '-')))}</td>"
            f"<td>{_safe_int(row.get('decision_count', 0))}</td>"
            f"<td>{_safe_int(row.get('closed_count', 0))}</td>"
            f"<td><span class='badge {wr_tone}'>{wr:.1f}%</span></td>"
            f"<td><span class='badge {exp_tone}'>{_fmt_money(exp)}</span></td>"
            f"<td>{_fmt_money(row.get('realized_pnl_usd', 0.0))}</td>"
            "</tr>"
        )
    if not setup_perf_rows:
        setup_perf_rows.append("<tr><td colspan='6'>Setup performance will appear after trades accumulate.</td></tr>")

    # Equity curve JSON for JS chart (3 windows)
    def _curve_to_js(curve_data: object) -> str:
        if not isinstance(curve_data, list):
            return "[]"
        points = [
            {"t": _safe_float(p.get("timestamp", 0.0)), "v": _safe_float(p.get("total_equity_usd", 0.0))}
            for p in cast(list[object], curve_data)
            if isinstance(p, dict)
        ]
        return json.dumps(points)

    equity_curve_all_js = _curve_to_js(research.get("equity_curve", []))
    equity_curve_7d_js = _curve_to_js(research.get("equity_curve_7d", []))
    equity_curve_1d_js = _curve_to_js(research.get("equity_curve_1d", []))
    trade_notes = research.get("notes", [])
    if not isinstance(trade_notes, list):
        trade_notes = []
    trade_notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in cast(list[object], trade_notes))
    decision_analytics = _as_mapping(research.get("decision_analytics"))
    decision_summary = _as_mapping(decision_analytics.get("summary"))
    decision_total = _safe_int(decision_summary.get("total_decisions", 0))
    decision_executed = _safe_int(decision_summary.get("executed_decisions", 0))
    decision_no_trade = _safe_int(decision_summary.get("no_trade_decisions", 0))
    decision_skip = _safe_int(decision_summary.get("skip_decisions", 0))
    decision_execution_rate = (
        int((decision_executed / decision_total) * 100)
        if decision_total > 0
        else 0
    )

    trade_rows: list[str] = []
    research_trades = research.get("recent_trades", [])
    if isinstance(research_trades, list) and research_trades:
        for trade in research_trades[:10]:
            if not isinstance(trade, dict):
                continue
            pnl = _safe_float(trade.get("pnl_usd", 0.0))
            pnl_tone = "good" if pnl > 0 else "bad" if pnl < 0 else "muted-chip"
            trade_rows.append(
                "<tr>"
                f"<td>{html.escape(str(trade.get('direction', '-')))}</td>"
                f"<td>{html.escape(str(trade.get('pair', '-')))}</td>"
                f"<td>{_fmt_money(trade.get('size_usd', 0.0))}</td>"
                f"<td>{_fmt_money(trade.get('fill_price', 0.0))}</td>"
                f"<td>{_fmt_ts(trade.get('timestamp'))}</td>"
                f"<td>{_fmt_pct(trade.get('slippage_bps', 0.0), 1).replace('%', ' bps')}</td>"
                f"<td><span class='badge {pnl_tone}'>{_fmt_money(pnl)}</span></td>"
                "</tr>"
            )
    if not trade_rows:
        trade_rows.append("<tr><td colspan='7'>No paper trades captured yet.</td></tr>")

    no_trade_reason_rows: list[str] = []
    for row in cast(list[object], decision_analytics.get("no_trade_reasons", [])):
        if not isinstance(row, dict):
            continue
        no_trade_reason_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('reason', '-')))}</td>"
            f"<td>{_fmt_count(row.get('count', 0))}</td>"
            "</tr>"
        )
    if not no_trade_reason_rows:
        no_trade_reason_rows.append("<tr><td colspan='2'>No no-trade reasons recorded yet.</td></tr>")

    regime_rows_html: list[str] = []
    for row in cast(list[object], decision_analytics.get("regime_breakdown", [])):
        if not isinstance(row, dict):
            continue
        decision_count = max(_safe_int(row.get("decision_count", 0)), 1)
        execution_rate = int((_safe_int(row.get("executed_count", 0)) / decision_count) * 100)
        regime_rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('regime_label', '-')))}</td>"
            f"<td>{_fmt_count(row.get('decision_count', 0))}</td>"
            f"<td>{execution_rate}%</td>"
            f"<td>{_fmt_money(row.get('realized_pnl_usd', 0.0))}</td>"
            "</tr>"
        )
    if not regime_rows_html:
        regime_rows_html.append("<tr><td colspan='4'>Regime analytics will appear after decision journaling accumulates.</td></tr>")

    setup_rows_html: list[str] = []
    for row in cast(list[object], decision_analytics.get("setup_breakdown", [])):
        if not isinstance(row, dict):
            continue
        decision_count = max(_safe_int(row.get("decision_count", 0)), 1)
        execution_rate = int((_safe_int(row.get("executed_count", 0)) / decision_count) * 100)
        setup_rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('setup_tag', '-')))}</td>"
            f"<td>{_fmt_count(row.get('decision_count', 0))}</td>"
            f"<td>{execution_rate}%</td>"
            f"<td>{_fmt_money(row.get('realized_pnl_usd', 0.0))}</td>"
            "</tr>"
        )
    if not setup_rows_html:
        setup_rows_html.append("<tr><td colspan='4'>Setup analytics will appear after decision journaling accumulates.</td></tr>")

    decision_rows: list[str] = []
    research_decisions = research.get("recent_decisions", [])
    if isinstance(research_decisions, list) and research_decisions:
        for decision in research_decisions[:12]:
            if not isinstance(decision, dict):
                continue
            decision_rows.append(
                "<tr>"
                f"<td>{_fmt_ts(decision.get('timestamp'))}</td>"
                f"<td>{html.escape(str(decision.get('action', '-')))}</td>"
                f"<td>{html.escape(str(decision.get('primary_reason', '-')))}</td>"
                f"<td>{html.escape(str(decision.get('regime_label', '-')))}</td>"
                f"<td>{html.escape(str(decision.get('setup_tag', '-')))}</td>"
                f"<td>{_fmt_money(decision.get('trade_budget_usd', 0.0))}</td>"
                f"<td>{_fmt_pct(decision.get('confidence', 0.0) * 100.0, 1)}</td>"
                "</tr>"
            )
    if not decision_rows:
        decision_rows.append("<tr><td colspan='7'>No decision-journal rows yet.</td></tr>")

    extremes = [
        ("Best trade", _as_mapping(research.get("best_trade"))),
        ("Worst trade", _as_mapping(research.get("worst_trade"))),
        ("Last trade", _as_mapping(research.get("last_trade"))),
    ]
    extreme_rows: list[str] = []
    for label, trade in extremes:
        if trade:
            extreme_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(trade.get('direction', '-')))}</td>"
                f"<td>{html.escape(str(trade.get('pair', '-')))}</td>"
                f"<td>{_fmt_money(trade.get('size_usd', 0.0))}</td>"
                f"<td>{_fmt_money(trade.get('pnl_usd', 0.0))}</td>"
                f"<td>{_fmt_ts(trade.get('timestamp'))}</td>"
                "</tr>"
            )
    if not extreme_rows:
        extreme_rows.append("<tr><td colspan='6'>Trade extremes will appear after more paper history accumulates.</td></tr>")

    recent_rows = []
    for item in recent[:8]:
        if isinstance(item, dict) and item.get("error"):
            recent_rows.append(
                f"<tr><td colspan='6'>{html.escape(str(item['error']))}</td></tr>"
            )
            continue
        if not isinstance(item, dict):
            continue
        payload_data = _as_mapping(item.get("payload"))
        summary = _as_mapping(payload_data.get("result_summary"))
        started_at = _safe_float(item.get("started_at", 0.0))
        finished_at = _safe_float(item.get("finished_at", 0.0))
        duration = _fmt_seconds(finished_at - started_at)
        fatal = payload_data.get("fatal_error", "none")
        fatal_tone = "bad" if fatal not in (None, "", "none") else "muted-chip"
        recent_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('session_id', 'unknown')))[:12]}</td>"
            f"<td>{_fmt_relative_ts(started_at, generated_at)}</td>"
            f"<td>{html.escape(str(payload_data.get('execution_mode', 'unknown')))}</td>"
            f"<td>{html.escape(str(summary.get('wallets_succeeded', 0)))}/{html.escape(str(summary.get('wallets_total', 0)))}</td>"
            f"<td>{duration}</td>"
            f"<td>{_fmt_count(_as_mapping(payload_data.get('paper_gate')).get('current_trade_count', 0))}</td>"
            f"<td><span class='badge {fatal_tone}'>{html.escape(str(fatal))}</span></td>"
            "</tr>"
        )
    if not recent_rows:
        recent_rows.append("<tr><td colspan='6'>No audit snapshots yet.</td></tr>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WalletBot Runtime</title>
  <style>
    :root {{
      --bg: #f2ebe2;
      --panel: rgba(255, 250, 242, 0.78);
      --panel-strong: rgba(255, 253, 249, 0.92);
      --ink: #172033;
      --muted: #697385;
      --accent: #a85517;
      --good: #166534;
      --warn: #9a6700;
      --bad: #b42318;
      --line: rgba(170, 131, 78, 0.18);
      --shadow: 0 18px 50px rgba(82, 58, 18, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", "Helvetica Neue", sans-serif; color: var(--ink); background:
      radial-gradient(circle at top left, rgba(255,255,255,0.85), transparent 28%),
      radial-gradient(circle at top right, rgba(251, 191, 36, 0.12), transparent 20%),
      linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 56px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 22px;
    }}
    .hero-main, .hero-side, .card, table, .snapshot {{
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 24px;
    }}
    .hero-main {{ padding: 24px; position: relative; overflow: hidden; }}
    .hero-main::after {{
      content: "";
      position: absolute;
      inset: auto -40px -40px auto;
      width: 180px;
      height: 180px;
      background: radial-gradient(circle, rgba(234, 179, 8, 0.22), rgba(234, 179, 8, 0));
      pointer-events: none;
    }}
    .hero-side {{ padding: 20px; display: grid; gap: 14px; }}
    .eyebrow {{ letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); font-size: 11px; font-weight: 700; }}
    h1 {{ margin: 10px 0 8px; font-size: clamp(32px, 5vw, 58px); line-height: 0.92; letter-spacing: -0.03em; }}
    .sub {{ color: var(--muted); max-width: 760px; line-height: 1.55; }}
    .meta-row, .endpoint-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    .toolbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-top: 16px; flex-wrap: wrap; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 18px; }}
    .wide-grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; margin-top: 18px; }}
    .triple-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin-top: 18px; }}
    .card {{ padding: 18px; }}
    .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 10px; font-weight: 700; }}
    .value {{ font-size: clamp(28px, 4vw, 42px); font-weight: 750; line-height: 1; letter-spacing: -0.03em; }}
    .value.small {{ font-size: 22px; }}
    .detail {{ color: var(--muted); margin-top: 8px; line-height: 1.45; }}
    .badge {{ display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 8px 12px; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; }}
    .good {{ background: rgba(22,101,52,0.12); color: var(--good); }}
    .warn {{ background: rgba(146,64,14,0.12); color: var(--warn); }}
    .bad {{ background: rgba(153,27,27,0.12); color: var(--bad); }}
    .muted-chip {{ background: rgba(23,32,51,0.07); color: var(--muted); }}
    .endpoint {{
      text-decoration: none;
      color: var(--ink);
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 13px;
      font-weight: 600;
    }}
    .endpoint:hover {{ border-color: rgba(168,85,23,0.5); color: var(--accent); }}
    .refresh-note {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255,255,255,0.6);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 9px 14px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}
    .section {{ margin-top: 24px; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: end; margin-bottom: 12px; }}
    .section h2 {{ margin: 0; font-size: 22px; letter-spacing: -0.02em; }}
    .section-copy {{ color: var(--muted); font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; }}
    th, td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--line); font-size: 14px; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}
    .snapshot {{ padding: 18px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 13px; background: rgba(31,41,55,0.04); padding: 14px; border-radius: 14px; border: 1px solid var(--line); margin: 0; max-height: 420px; overflow: auto; }}
    .metric-list {{ display: grid; gap: 12px; }}
    .metric-row {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding-bottom: 10px; border-bottom: 1px dashed var(--line); }}
    .metric-row:last-child {{ padding-bottom: 0; border-bottom: none; }}
    .metric-name {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ font-size: 16px; font-weight: 700; }}
    .sparkline {{ width: 100%; height: 58px; display: block; margin-top: 10px; }}
    .sparkline-empty {{ color: var(--muted); font-size: 13px; margin-top: 18px; }}
    .progress-rail {{ width: 100%; height: 12px; border-radius: 999px; background: rgba(23,32,51,0.08); overflow: hidden; margin-top: 12px; }}
    .progress-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #f59e0b, #16a34a); }}
    .recos {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.6; }}
    a {{ color: var(--accent); }}
    .tab-bar {{ display: flex; gap: 8px; margin-bottom: 14px; }}
    .tab-btn {{ padding: 7px 16px; border-radius: 999px; border: 1px solid var(--line); background: var(--panel); cursor: pointer; font-size: 13px; font-weight: 600; color: var(--muted); transition: all .15s; }}
    .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .equity-canvas-wrap {{ position: relative; width: 100%; height: 180px; margin-top: 12px; }}
    canvas.equity-canvas {{ width: 100%; height: 100%; display: block; }}
    .score-ring {{ display: flex; align-items: center; gap: 20px; }}
    .score-num {{ font-size: 52px; font-weight: 800; letter-spacing: -0.04em; line-height: 1; }}
    .score-label {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
    @media (max-width: 960px) {{
      .hero, .wide-grid, .triple-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-main">
        <div class="eyebrow">WalletBot Operator Board</div>
        <h1>Runtime dashboard for human decisions.</h1>
        <div class="sub">This screen shows the latest runtime state, paper-trading progress, live guard posture, audit trail, and strategy-research telemetry in one place so you can decide whether the bot is healthy, active, and genuinely improving.</div>
        <div class="meta-row">
          {live_badge}
          {paper_badge}
          {availability_badge}
          {rebalance_badge}
        </div>
        <div class="toolbar">
          <div class="endpoint-row">
            {endpoint_links}
          </div>
          <div class="refresh-note">Auto refresh in <strong id="refresh-countdown">{refresh_seconds}</strong>s</div>
        </div>
      </div>
      <div class="hero-side">
        <div>
          <div class="label">Snapshot Generated</div>
          <div class="value small">{_fmt_relative_ts(generated_at, generated_at)}</div>
          <div class="detail">Execution mode: <strong>{execution_mode}</strong></div>
          <div class="detail">Source: <strong>{data_source}</strong></div>
        </div>
        <div class="metric-list">
          <div class="metric-row">
            <div class="metric-name">Wallet run status</div>
            <div class="metric-value">{wallets_ok}/{wallets_total}</div>
          </div>
          <div class="metric-row">
            <div class="metric-name">Paper trade count</div>
            <div class="metric-value">{paper_trades} / {paper_needed}</div>
          </div>
          <div class="metric-row">
            <div class="metric-name">Fatal error</div>
            <div class="metric-value">{html.escape(str(fatal_error))}</div>
          </div>
          <div class="metric-row">
            <div class="metric-name">Latest session</div>
            <div class="metric-value">{latest_session_id[:12]}</div>
          </div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <div class="label">Execution Mode</div>
        <div class="value">{execution_mode}</div>
        <div class="detail">Current orchestrator mode reported by the latest runtime audit.</div>
      </div>
      <div class="card">
        <div class="label">Portfolio Total</div>
        <div class="value">{_fmt_money(total_value)}</div>
        <div class="detail">Current portfolio equity from the latest runtime payload.</div>
      </div>
      <div class="card">
        <div class="label">Live Guard</div>
        {live_badge}
        <div class="detail">Tiny-cap and live-trading safety checks.</div>
      </div>
      <div class="card">
        <div class="label">Paper Gate</div>
        {paper_badge}
        <div class="detail">Minimum paper-trade gate before considering live mode.</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <div class="label">Paper Progress</div>
        <div class="value">{paper_progress}%</div>
        <div class="detail">Progress toward minimum paper trade history.</div>
        <div class="progress-rail"><div class="progress-fill" style="width:{paper_progress}%"></div></div>
      </div>
      <div class="card">
        <div class="label">Recent Success Rate</div>
        <div class="value">{success_rate}%</div>
        <div class="detail">Wallet success ratio across the latest completed sessions.</div>
      </div>
      <div class="card">
        <div class="label">Average Cycle Time</div>
        <div class="value small">{avg_duration}</div>
        <div class="detail">Mean completed runtime duration from recent sessions.</div>
      </div>
      <div class="card">
        <div class="label">Wallet Summary</div>
        <div class="value">{wallets_ok}/{wallets_total}</div>
        <div class="sub">Succeeded / total wallets in the latest completed run.</div>
      </div>
      <div class="card">
        <div class="label">Paper Trades</div>
        <div class="value">{paper_trades}</div>
        <div class="sub">Required before live: {paper_needed}</div>
      </div>
      <div class="card">
        <div class="label">Portfolio State</div>
        <div class="value small">{'Available' if portfolio_available else 'Unavailable'}</div>
        <div class="detail">Rebalance: {'Yes' if needs_rebalance else 'No'}</div>
      </div>
      <div class="card">
        <div class="label">Latest Fatal Error</div>
        <div class="value small">{html.escape(str(fatal_error))}</div>
        <div class="detail">`none` means the latest run completed without a fatal exception.</div>
      </div>
    </section>

    <section class="triple-grid">
      <div class="card">
        <div class="label">Portfolio Trend</div>
        <div class="detail">Recent portfolio totals from stored runtime audits.</div>
        {portfolio_chart}
      </div>
      <div class="card">
        <div class="label">Cycle Duration Trend</div>
        <div class="detail">Recent runtime lengths. Sudden spikes usually mean RPC or model loading pressure.</div>
        {duration_chart}
      </div>
      <div class="card">
        <div class="label">Operator Notes</div>
        <ul class="recos">
          {''.join(f"<li>{html.escape(note)}</li>" for note in recommendations)}
        </ul>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Strategy Lab</h2>
          <div class="section-copy">Paper-trading analytics designed to answer whether the bot is learning, trading, and improving enough to trust with the next step.</div>
        </div>
      </div>
      <section class="grid">
        <div class="card">
          <div class="label">Net After Gas</div>
          <div class="value">{research_net_after_gas}</div>
          <div class="detail">Realized paper PnL after subtracting recorded gas estimates.</div>
        </div>
        <div class="card">
          <div class="label">Closed Trades</div>
          <div class="value">{research_closed_count}</div>
          <div class="detail">Completed SELL-side exits. This is the minimum viable sample for judging edge.</div>
        </div>
        <div class="card">
          <div class="label">Win Rate</div>
          <div class="value">{research_win_rate}</div>
          <div class="detail">Current realized win rate across stored paper trades.</div>
        </div>
        <div class="card">
          <div class="label">Average Slippage</div>
          <div class="value small">{research_avg_slippage}</div>
          <div class="detail">Average simulated execution friction across recorded trades.</div>
        </div>
        <div class="card">
          <div class="label">Last Trade</div>
          <div class="value small">{research_last_trade_at}</div>
          <div class="detail">Useful for distinguishing “idle by design” from “not trading at all.”</div>
        </div>
        <div class="card">
          <div class="label">Spot Exposure</div>
          <div class="value">{research_exposure}</div>
          <div class="detail">Approximate WETH share of current paper equity.</div>
        </div>
        <div class="card">
          <div class="label">Decision Coverage</div>
          <div class="value">{decision_total}</div>
          <div class="detail">Logged trade + no-trade decisions used for research and Claude handoff.</div>
        </div>
        <div class="card">
          <div class="label">Execution Rate</div>
          <div class="value">{decision_execution_rate}%</div>
          <div class="detail">Share of journaled decisions that became BUY or SELL actions.</div>
        </div>
        <div class="card">
          <div class="label">No-Trade / Skip</div>
          <div class="value small">{decision_no_trade} / {decision_skip}</div>
          <div class="detail">No-trade decisions vs external/task-level skips.</div>
        </div>
      </section>
    </section>


    <!-- Phase 2: Advanced Strategy Analytics -->
    <section class="section">
      <div class="section-head">
        <div>
          <h2>Strategy Analytics</h2>
          <div class="section-copy">Expectancy, payoff ratio, profit factor, Calmar, Sortino — the numbers that answer "is the strategy actually good?"</div>
        </div>
      </div>
      <section class="grid">
        <div class="card">
          <div class="label">Expectancy</div>
          <div class="value {'good' if _safe_float(adv.get('expectancy_usd',0))>0 else 'bad'}">{_fmt_money(adv.get('expectancy_usd', 0.0))}</div>
          <div class="detail">Expected PnL per closed trade. Must be positive net of gas to justify live.</div>
        </div>
        <div class="card">
          <div class="label">Payoff Ratio</div>
          <div class="value">{_safe_float(adv.get('payoff_ratio', 0.0)):.2f}x</div>
          <div class="detail">Avg win / avg loss. Values above 1.5 give edge even at sub-50% win rate.</div>
        </div>
        <div class="card">
          <div class="label">Profit Factor</div>
          <div class="value">{_safe_float(adv.get('profit_factor', 0.0)):.2f}</div>
          <div class="detail">Gross profit / gross loss. Above 1.5 is healthy; below 1.0 means losing strategy.</div>
        </div>
        <div class="card">
          <div class="label">Calmar Ratio</div>
          <div class="value">{_safe_float(adv.get('calmar_ratio', 0.0)):.2f}</div>
          <div class="detail">Annualized return / max drawdown. Higher = better risk-adjusted return profile.</div>
        </div>
        <div class="card">
          <div class="label">Sortino Ratio</div>
          <div class="value">{_safe_float(adv.get('sortino_ratio', 0.0)):.2f}</div>
          <div class="detail">Like Sharpe but only penalizes downside deviation. Above 1.0 is good.</div>
        </div>
        <div class="card">
          <div class="label">Avg Hold Time</div>
          <div class="value small">{_fmt_seconds(hold.get('avg_hold_seconds', 0))}</div>
          <div class="detail">Median: {_fmt_seconds(hold.get('median_hold_seconds', 0))} · Max: {_fmt_seconds(hold.get('max_hold_seconds', 0))}</div>
        </div>
        <div class="card">
          <div class="label">Avg Win / Avg Loss</div>
          <div class="value small">{_fmt_money(adv.get('avg_win_usd', 0.0))} / {_fmt_money(adv.get('avg_loss_usd', 0.0))}</div>
          <div class="detail">Gross profit: {_fmt_money(adv.get('gross_profit_usd', 0.0))} · Gross loss: {_fmt_money(adv.get('gross_loss_usd', 0.0))}</div>
        </div>
        <div class="card">
          <div class="label">Net After Gas</div>
          <div class="value">{_fmt_money(adv.get('net_after_gas_usd', 0.0))}</div>
          <div class="detail">Total gas paid: {_fmt_money(adv.get('total_gas_usd', 0.0))} across {_safe_int(adv.get('trade_count', 0))} trades.</div>
        </div>
        <div class="card">
          <div class="label">Sample Sufficient?</div>
          <div class="value small">{'YES' if adv.get('sample_sufficient') else 'NO'}</div>
          <div class="detail">{_safe_int(adv.get('closed_count', 0))} closed trades. Need ≥ 10 for analytics to be meaningful.</div>
        </div>
      </section>
    </section>

    <!-- Phase 9: Live-Readiness Score -->
    <section class="wide-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <h2>Live-Readiness Score</h2>
            <div class="section-copy">Multi-gate scoring system that decides when it is safe to consider transitioning from paper to micro-live.</div>
          </div>
        </div>
        <div class="card">
          <div class="score-ring">
            <div>
              <div class="score-num">{readiness_score}</div>
              <div class="score-label">/ {readiness_max} pts</div>
            </div>
            <div style="flex:1">
              <div class="progress-rail" style="height:18px;margin-top:0">
                <div class="progress-fill" style="width:{readiness_pct}%"></div>
              </div>
              <div class="detail" style="margin-top:10px">{readiness_rec}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head"><div><h2>Gate Checklist</h2></div></div>
        <table>
          <thead>
            <tr><th></th><th>Gate</th><th>Current</th><th>Target</th><th>Score</th></tr>
          </thead>
          <tbody>
            {''.join(gate_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <!-- Phase 3: Equity Curve with time-window tabs -->
    <section class="wide-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <h2>Equity Curve</h2>
            <div class="section-copy">Paper equity history. Switch windows to compare today vs 7-day vs all-time.</div>
          </div>
        </div>
        <div class="card">
          <div class="label">Paper Equity</div>
          <div class="value small">{_fmt_money(research.get('current_equity_usd', 0.0))}</div>
          <div class="detail">Peak: {_fmt_money(research.get('peak_equity_usd', 0.0))} · Drawdown: {_fmt_pct(research.get('max_drawdown_pct', 0.0))}</div>
          <div class="tab-bar" id="equity-tabs">
            <button class="tab-btn" onclick="setEquityWindow('1d',this)">Today</button>
            <button class="tab-btn" onclick="setEquityWindow('7d',this)">7 Day</button>
            <button class="tab-btn active" onclick="setEquityWindow('all',this)">All Time</button>
          </div>
          <div class="equity-canvas-wrap">
            <canvas class="equity-canvas" id="equityCanvas"></canvas>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <div>
            <h2>Research Notes</h2>
            <div class="section-copy">Auto-generated observations from the paper ledger and runtime behaviour.</div>
          </div>
        </div>
        <div class="card">
          <ul class="recos">
            {trade_notes_html}
          </ul>
        </div>
      </div>
    </section>

    <!-- Phase 3: No-trade reasons + Enhanced regime + Enhanced setup -->
    <section class="triple-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <h2>No-Trade Reasons</h2>
            <div class="section-copy">Top blockers explaining why the bot stayed idle or deliberately skipped execution.</div>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>Reason</th><th>Count</th></tr>
          </thead>
          <tbody>
            {''.join(no_trade_reason_rows)}
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-head">
          <div>
            <h2>Regime Performance</h2>
            <div class="section-copy">Win rate and expectancy by regime — which market states actually produce edge.</div>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>Regime</th><th>Decisions</th><th>Closed</th><th>Win%</th><th>Expectancy</th><th>PnL</th></tr>
          </thead>
          <tbody>
            {''.join(regime_perf_rows)}
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-head">
          <div>
            <h2>Setup Performance</h2>
            <div class="section-copy">Which setup families actually produce closed-trade edge vs which are just noise.</div>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>Setup</th><th>Decisions</th><th>Closed</th><th>Win%</th><th>Expectancy</th><th>PnL</th></tr>
          </thead>
          <tbody>
            {''.join(setup_perf_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="wide-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <h2>Trade Tape</h2>
            <div class="section-copy">Most recent paper trades for fast manual review of size, timing, and friction.</div>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>Side</th><th>Pair</th><th>Size</th><th>Fill</th><th>Time</th><th>Slip</th><th>PnL</th></tr>
          </thead>
          <tbody>
            {''.join(trade_rows)}
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-head">
          <div>
            <h2>Trade Extremes</h2>
            <div class="section-copy">Best, worst, and latest trade snapshots to speed up post-trade review.</div>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>Bucket</th><th>Side</th><th>Pair</th><th>Size</th><th>PnL</th><th>Time</th></tr>
          </thead>
          <tbody>
            {''.join(extreme_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="wide-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <h2>Portfolio Allocation</h2>
            <div class="section-copy">Current, target, and drift values from the runtime portfolio snapshot.</div>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>Strategy</th><th>Current</th><th>Target</th><th>Delta</th></tr>
          </thead>
          <tbody>
            {''.join(allocation_rows)}
          </tbody>
        </table>
      </div>

      <div class="section">
        <div class="section-head">
          <div>
            <h2>Guard Detail</h2>
            <div class="section-copy">Quick readout of paper-gate and live-guard constraints.</div>
          </div>
        </div>
        <div class="card">
          <div class="metric-list">
            <div class="metric-row">
              <div class="metric-name">Live guard max capital</div>
              <div class="metric-value">{_fmt_money(live_guard.get('max_total_capital_usd', 0.0))}</div>
            </div>
            <div class="metric-row">
              <div class="metric-name">Live guard max wallets</div>
              <div class="metric-value">{_fmt_count(live_guard.get('max_wallets', 0))}</div>
            </div>
            <div class="metric-row">
              <div class="metric-name">Live guard observed capital</div>
              <div class="metric-value">{_fmt_money(live_guard.get('portfolio_total_usd', 0.0))}</div>
            </div>
            <div class="metric-row">
              <div class="metric-name">Paper gate current trades</div>
              <div class="metric-value">{paper_trades}</div>
            </div>
            <div class="metric-row">
              <div class="metric-name">Paper gate required</div>
              <div class="metric-value">{paper_needed}</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Decision Journal</h2>
          <div class="section-copy">Recent trade and no-trade decisions, preserved so research and handoff never depend on terminal logs alone.</div>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Time</th><th>Action</th><th>Primary Reason</th><th>Regime</th><th>Setup</th><th>Budget</th><th>Confidence</th></tr>
        </thead>
        <tbody>
          {''.join(decision_rows)}
        </tbody>
      </table>
    </section>

    <!-- Phase 6: Risk Committee -->
    <section class="wide-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <h2>Risk Committee</h2>
            <div class="section-copy">Phase 6 — multi-brake guardrail system. BLOCK means no new positions; REDUCE means size down.</div>
          </div>
          <span class="badge {rc_tone}">{rc_verdict} &nbsp;·&nbsp; Score {rc_score}/100</span>
        </div>
        <div class="card" style="margin-bottom:14px">
          <div class="label">Narrative</div>
          <div class="detail">{rc_narrative}</div>
        </div>
        <table>
          <thead><tr><th>Brake</th><th>Verdict</th><th>Reason</th><th>Value</th></tr></thead>
          <tbody>{''.join(brake_rows_html)}</tbody>
        </table>
      </div>

      <!-- Phase 5: Macro Calendar -->
      <div class="section">
        <div class="section-head">
          <div>
            <h2>Macro Calendar</h2>
            <div class="section-copy">Phase 5 — event blackout calendar. Suppresses alpha around FOMC, NFP, CPI, and other high-impact releases.</div>
          </div>
          <span class="badge {macro_badge_tone}">Alpha {macro_badge_text}</span>
        </div>
        <table>
          <thead><tr><th>Event</th><th>Date/Time UTC</th><th>Impact</th><th>Category</th></tr></thead>
          <tbody>{''.join(macro_event_rows)}</tbody>
        </table>
      </div>
    </section>

    <!-- Phase 7: Experiment Lab -->
    <section class="section">
      <div class="section-head">
        <div>
          <h2>Experiment Lab — Threshold Variants</h2>
          <div class="section-copy">Phase 7 — shadow tests competing threshold configs against the baseline. Winner: <b>{exp_winner}</b></div>
        </div>
      </div>
      <table>
        <thead><tr><th>Variant</th><th>Cycles</th><th>Would Trade</th><th>Agreement</th><th>Shared PnL</th></tr></thead>
        <tbody>{''.join(exp_variant_rows)}</tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Recent Audit Sessions</h2>
          <div class="section-copy">Last completed runtime sessions, with success ratio, elapsed time, and paper trade count.</div>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Session</th><th>Started</th><th>Mode</th><th>Success</th><th>Duration</th><th>Paper Trades</th><th>Fatal Error</th></tr>
        </thead>
        <tbody>
          {''.join(recent_rows)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Current Audit Snapshot</h2>
          <div class="section-copy">Raw runtime payload for debugging, automation, and deeper inspection.</div>
        </div>
      </div>
      <div class="snapshot">
        <pre>{html.escape(json.dumps(audit, indent=2, sort_keys=True))}</pre>
      </div>
    </section>
  </div>
  <script>
    // --- Countdown ---
    (() => {{
      let remaining = {refresh_seconds};
      const countdown = document.getElementById("refresh-countdown");
      window.setInterval(() => {{
        remaining -= 1;
        if (countdown) countdown.textContent = String(Math.max(0, remaining));
        if (remaining <= 0) window.location.reload();
      }}, 1000);
    }})();

    // --- Equity Canvas Chart ---
    const CURVES = {{
      all: {equity_curve_all_js},
      "7d": {equity_curve_7d_js},
      "1d": {equity_curve_1d_js},
    }};
    let activeWindow = "all";

    function drawEquityChart(windowKey) {{
      const canvas = document.getElementById("equityCanvas");
      if (!canvas) return;
      const data = CURVES[windowKey] || [];
      const ctx = canvas.getContext("2d");
      const W = canvas.offsetWidth || 600;
      const H = canvas.offsetHeight || 180;
      canvas.width = W;
      canvas.height = H;
      ctx.clearRect(0, 0, W, H);
      if (!data || data.length < 2) {{
        ctx.fillStyle = "#697385";
        ctx.font = "13px Segoe UI";
        ctx.fillText("No equity data for this window.", 16, H / 2);
        return;
      }}
      const values = data.map(p => p.v);
      const minV = Math.min(...values);
      const maxV = Math.max(...values);
      const spread = maxV - minV || 1;
      const pad = {{ t: 16, b: 24, l: 8, r: 8 }};
      const chartW = W - pad.l - pad.r;
      const chartH = H - pad.t - pad.b;

      // Grid lines
      ctx.strokeStyle = "rgba(170,131,78,0.15)";
      ctx.lineWidth = 1;
      [0.25, 0.5, 0.75, 1.0].forEach(frac => {{
        const y = pad.t + chartH * (1 - frac);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      }});

      // Fill gradient
      const grad = ctx.createLinearGradient(0, pad.t, 0, H);
      grad.addColorStop(0, "rgba(15,118,110,0.22)");
      grad.addColorStop(1, "rgba(15,118,110,0.01)");
      ctx.beginPath();
      data.forEach((p, i) => {{
        const x = pad.l + (i / (data.length - 1)) * chartW;
        const y = pad.t + chartH * (1 - (p.v - minV) / spread);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }});
      ctx.lineTo(pad.l + chartW, pad.t + chartH);
      ctx.lineTo(pad.l, pad.t + chartH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 2.5;
      ctx.lineJoin = "round";
      data.forEach((p, i) => {{
        const x = pad.l + (i / (data.length - 1)) * chartW;
        const y = pad.t + chartH * (1 - (p.v - minV) / spread);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }});
      ctx.stroke();

      // Axis labels
      ctx.fillStyle = "#697385";
      ctx.font = "11px Segoe UI";
      ctx.textAlign = "left";
      ctx.fillText("$" + minV.toFixed(0), pad.l + 2, H - 6);
      ctx.textAlign = "right";
      ctx.fillText("$" + maxV.toFixed(0), W - pad.r - 2, pad.t + 12);
    }}

    function setEquityWindow(key, btn) {{
      activeWindow = key;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      drawEquityChart(key);
    }}

    window.addEventListener("load", () => drawEquityChart(activeWindow));
    window.addEventListener("resize", () => drawEquityChart(activeWindow));
  </script>
</body>
</html>"""


async def _handle_runtime_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response(await _build_runtime_payload())


async def _handle_research_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    payload = await _build_runtime_payload()
    return aiohttp.web.json_response(
        {
            "status": "ok",
            "generated_at": payload.get("generated_at", time.time()),
            "execution_mode": payload.get("execution_mode"),
            "research": payload.get("research", {}),
        },
    )


async def _handle_risk_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.risk_committee import get_risk_committee
        from wallet_manager.strategies.paper_engine import _engine_instance
        verdict = await get_risk_committee().evaluate(paper_engine=_engine_instance)
        result["verdict"] = verdict.to_dict()
    return aiohttp.web.json_response(result)


async def _handle_calendar_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.market_intelligence import get_macro_calendar
        cal = get_macro_calendar()
        result["calendar"] = cal.to_dict()
        suppress, reason = cal.should_suppress_alpha()
        result["alpha_suppressed"] = suppress
        result["suppression_reason"] = reason
    return aiohttp.web.json_response(result)


async def _handle_experiments_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.experiment_lab import get_experiment_lab
        lab = get_experiment_lab()
        result["comparison"] = lab.compare_variants()
        result["variants"] = lab.get_variant_report()
        result["regime_variants"] = lab.get_regime_variant_report()
        result["feature_toggles"] = lab.get_feature_toggle_report()
    return aiohttp.web.json_response(result)


async def _handle_agents_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    try:
        from wallet_manager.ai_agents import get_agent_registry
        from wallet_manager.strategies.paper_engine import _engine_instance
        registry = get_agent_registry()
        reports = await registry.run_all_to_dict(paper_engine=_engine_instance)
        result["reports"] = reports
    except Exception as exc:
        result["error"] = str(exc)
    return aiohttp.web.json_response(result)


async def _handle_learning_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.continuous_learning import get_continuous_learner
        from wallet_manager.strategies.paper_engine import _engine_instance
        report = await get_continuous_learner().generate_learning_report(_engine_instance)
        result["learning"] = report
    return aiohttp.web.json_response(result)


async def _handle_signals_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.signal_intelligence import get_signal_intelligence
        result["signals"] = get_signal_intelligence().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_risk_intelligence_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.risk_intelligence import get_risk_intelligence
        from wallet_manager.strategies.paper_engine import _engine_instance
        equity = 10_000.0
        with contextlib.suppress(Exception):
            if _engine_instance:
                equity = await _engine_instance._ledger.get_total_equity_usd()
        result["risk_intelligence"] = get_risk_intelligence(equity).to_dict(equity)
    return aiohttp.web.json_response(result)


async def _handle_stress_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.risk_intelligence import get_risk_intelligence
        ri = get_risk_intelligence()
        ri.run_stress_scenarios()
        result["stress"] = ri.stress_engine.to_dict()
    return aiohttp.web.json_response(result)


async def _handle_ticks_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.tick_collector import get_tick_collector
        result["ticks"] = await get_tick_collector().get_stats()
    return aiohttp.web.json_response(result)


async def _handle_alerts_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.alerting import get_alert_system
        result["alerts"] = get_alert_system().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_cross_asset_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.cross_asset import get_cross_asset
        result["cross_asset"] = get_cross_asset().get_snapshot()
    return aiohttp.web.json_response(result)


async def _handle_ensemble_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.adaptive_ensemble import get_adaptive_ensemble
        result["ensemble"] = get_adaptive_ensemble().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_bayesian_kelly_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.bayesian_kelly import get_bayesian_kelly
        result["bayesian_kelly"] = get_bayesian_kelly().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_execution_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.execution_quality import get_execution_quality
        result["execution"] = get_execution_quality().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_discovery_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.signal_discovery import get_signal_discovery
        sd = get_signal_discovery()
        await sd.run_discovery()
        result["discovery"] = sd.to_dict()
    return aiohttp.web.json_response(result)


async def _handle_reports_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.reporting import get_reporter
        result["reports"] = get_reporter().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_hyperopt_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-3.4: Bayesian hyperparameter optimization status and recommendations."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.hyperopt import get_hyperoptimizer
        optimizer = get_hyperoptimizer()
        result["hyperopt"] = optimizer.to_dict()
    return aiohttp.web.json_response(result)


async def _handle_order_flow_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-4.2: Order Flow Imbalance (OFI) real-time signals."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.order_flow import get_order_flow_imbalance
        ofi = get_order_flow_imbalance()
        result["order_flow"] = ofi.get_snapshot()
    return aiohttp.web.json_response(result)


async def _handle_model_registry_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-7.3: Champion/Challenger model framework status."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.model_registry import get_model_registry
        registry = get_model_registry()
        result["model_registry"] = registry.to_dict()
    return aiohttp.web.json_response(result)


async def _handle_online_learning_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-3.3: Online learning hook status and update statistics."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.online_learning import get_online_learning_hook
        hook = get_online_learning_hook()
        result["online_learning"] = hook.to_dict()
    return aiohttp.web.json_response(result)


async def _handle_timing_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-5.2: Execution timing optimization by hour-of-day."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.execution_quality import get_execution_quality
        eq = get_execution_quality()
        result["timing"] = eq.get_timing_snapshot()
    return aiohttp.web.json_response(result)


async def _handle_feature_drift_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-7.4: Feature importance drift detection and alerts."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.continuous_learning import get_continuous_learner
        learner = get_continuous_learner()
        result["feature_drift"] = learner.feature_drift.to_dict()
        result["recent_alerts"] = learner.feature_drift.get_recent_alerts(5)
    return aiohttp.web.json_response(result)


async def _handle_funding_rate_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-4.3: Funding rate signal from perpetual futures exchanges."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.funding_rate import get_funding_rate_collector
        result["funding_rate"] = get_funding_rate_collector().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_shadow_live_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """JS-8.1: Shadow live mode — paper vs on-chain fill price comparisons."""
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.shadow_live import get_shadow_live_engine
        result["shadow_live"] = get_shadow_live_engine().to_dict()
    return aiohttp.web.json_response(result)


async def _handle_monthly_report_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.reporting import get_reporter
        from wallet_manager.strategies.paper_engine import _engine_instance
        summary = await get_reporter().generate_monthly(notifier=None, paper_engine=_engine_instance)
        result["summary"] = summary
    return aiohttp.web.json_response(result)


async def _handle_chain_api(request: aiohttp.web.Request) -> aiohttp.web.Response:
    result: dict[str, object] = {"status": "ok", "generated_at": time.time()}
    with contextlib.suppress(Exception):
        from wallet_manager.market_intelligence import get_chain_intelligence
        ci = get_chain_intelligence()
        rpc_url = ""
        with contextlib.suppress(Exception):
            from wallet_manager.config import get_settings
            cfg = get_settings()
            rpc_url = next(
                (
                    candidate
                    for candidate in (cfg.base_rpc_1, cfg.base_rpc_2, cfg.base_rpc_3)
                    if candidate
                ),
                "",
            )
        result["chain"] = await ci.get_snapshot(rpc_url=rpc_url)
    return aiohttp.web.json_response(result)


async def _handle_dashboard(request: aiohttp.web.Request) -> aiohttp.web.Response:
    payload = await _build_runtime_payload()
    return aiohttp.web.Response(text=_render_dashboard_html(payload), content_type="text/html")


async def _handle_root(request: aiohttp.web.Request) -> aiohttp.web.StreamResponse:
    raise aiohttp.web.HTTPFound("/dashboard")


# ---------------------------------------------------------------------------
# MetricsServer class (dependency-injectable for testing)
# ---------------------------------------------------------------------------


class MetricsServer:
    """Dependency-injectable MetricsServer for testing and production.

    Args:
        port: HTTP port to bind (0 = OS assigns a random port, for tests).
    """

    def __init__(self, port: int = 9090) -> None:
        self.port = port
        self._metrics: dict[str, float] = {}
        self._labeled_metrics: dict[tuple[str, str], float] = {}

    def update_metric(self, name: str, value: float) -> None:
        """Update a scalar metric by name."""
        self._metrics[name] = value
        update_metric(name, value)

    def update_labeled_metric(self, name: str, label: str, value: float) -> None:
        """Update a labeled metric."""
        self._labeled_metrics[(name, label)] = value
        update_labeled_metric(name, label, value)

    def build_app(self) -> aiohttp.web.Application:
        """Build and return the aiohttp Application with all routes."""
        app = aiohttp.web.Application()
        app.router.add_get("/", _handle_root)
        app.router.add_get("/metrics", _handle_metrics)
        app.router.add_get("/health", _handle_health)
        app.router.add_get("/ready", _handle_ready)
        app.router.add_get("/api/runtime", _handle_runtime_api)
        app.router.add_get("/api/research", _handle_research_api)
        app.router.add_get("/api/risk", _handle_risk_api)
        app.router.add_get("/api/calendar", _handle_calendar_api)
        app.router.add_get("/api/experiments", _handle_experiments_api)
        app.router.add_get("/api/agents", _handle_agents_api)
        app.router.add_get("/api/learning", _handle_learning_api)
        app.router.add_get("/api/chain", _handle_chain_api)
        app.router.add_get("/api/signals", _handle_signals_api)
        app.router.add_get("/api/risk_intelligence", _handle_risk_intelligence_api)
        app.router.add_get("/api/stress", _handle_stress_api)
        app.router.add_get("/api/ticks", _handle_ticks_api)
        app.router.add_get("/api/alerts", _handle_alerts_api)
        app.router.add_get("/api/cross_asset", _handle_cross_asset_api)
        app.router.add_get("/api/ensemble", _handle_ensemble_api)
        app.router.add_get("/api/bayesian_kelly", _handle_bayesian_kelly_api)
        app.router.add_get("/api/execution", _handle_execution_api)
        app.router.add_get("/api/discovery", _handle_discovery_api)
        app.router.add_get("/api/reports", _handle_reports_api)
        app.router.add_get("/api/monthly_report", _handle_monthly_report_api)
        app.router.add_get("/api/hyperopt", _handle_hyperopt_api)
        app.router.add_get("/api/order_flow", _handle_order_flow_api)
        app.router.add_get("/api/model_registry", _handle_model_registry_api)
        app.router.add_get("/api/online_learning", _handle_online_learning_api)
        app.router.add_get("/api/timing", _handle_timing_api)
        app.router.add_get("/api/feature_drift", _handle_feature_drift_api)
        app.router.add_get("/api/funding_rate", _handle_funding_rate_api)
        app.router.add_get("/api/shadow_live", _handle_shadow_live_api)
        app.router.add_get("/dashboard", _handle_dashboard)
        return app


# ---------------------------------------------------------------------------
# Background task entry point
# ---------------------------------------------------------------------------


async def start_metrics_server() -> asyncio.Task | None:
    """Start the /metrics HTTP server as a background asyncio task.

    Returns the task, or None if metrics_enabled = False.
    """
    from wallet_manager.config import get_settings
    cfg = get_settings()
    if not cfg.metrics_enabled:
        log.info("MetricsServer: disabled (metrics_enabled=False).")
        return None

    port = cfg.metrics_port
    host = getattr(cfg, "metrics_host", "0.0.0.0")

    async def _serve() -> None:
        try:
            server = MetricsServer(port=port)
            app = server.build_app()
            runner = aiohttp.web.AppRunner(app)
            await runner.setup()
            # reuse_address=True (maps to SO_REUSEADDR on Windows) allows the server
            # to bind even if the previous process didn't fully release the port yet.
            site = aiohttp.web.TCPSite(runner, host, port, reuse_address=True)
            await site.start()
            log.info("MetricsServer: listening on %s:%d/metrics", host, port)
            while True:
                await asyncio.sleep(3600)
        except Exception as exc:
            log.warning("MetricsServer: failed: %s", exc)

    task = asyncio.ensure_future(_serve())
    return task
