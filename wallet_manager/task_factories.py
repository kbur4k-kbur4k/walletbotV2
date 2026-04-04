"""
wallet_manager/task_factories.py

Per-execution-mode task factory coroutines.

Extracted from orchestrator.py to keep that module under 1500 lines.
All names are re-exported from orchestrator for backward compatibility.
"""
from __future__ import annotations

import asyncio
import contextlib
import random
import time
from typing import TYPE_CHECKING, Any

from wallet_manager import actions
from wallet_manager.config import get_settings
from wallet_manager.core.manager import WalletManager
from wallet_manager.core.models import (
    EventKind,
    NotificationEvent,
    TxReceipt,
    WalletConfig,
)
from wallet_manager.portfolio_manager import StrategyType
from wallet_manager.scheduler import should_run_swap_this_cycle
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

if TYPE_CHECKING:
    from wallet_manager.config import Settings


def _cfg() -> Settings:
    """Shorthand for the global settings singleton."""
    return get_settings()


# Module-level constants (same derivation as orchestrator.py)
_cfg_snapshot = get_settings()
_TASK_LOOP_CYCLES: int = _cfg_snapshot.default_task_loop_cycles
_GAS_POLL_INTERVAL: float = _cfg_snapshot.gas_poll_interval
_AAVE_APY_ALERT_THRESHOLD: float = _cfg_snapshot.aave_apy_alert_threshold_pct / 100.0
del _cfg_snapshot

# ConceptDriftDetector singleton — created once at module load
try:
    from wallet_manager.strategies.wf_validator import ConceptDriftDetector as _ConceptDriftDetector
    _drift_detector: _ConceptDriftDetector | None = _ConceptDriftDetector()
    log.info("ConceptDriftDetector (task_factories): initialized.")
except Exception as _cdd_exc:
    _drift_detector = None
    log.warning("ConceptDriftDetector (task_factories): init failed: %s", _cdd_exc)


_AGENT_RUN_INTERVAL_S: float = 6 * 3600.0   # Run agents every 6 hours
_last_agent_run_ts: float = 0.0


def _get_wallet_address(config: WalletConfig) -> str:
    """Return the Ethereum address for the wallet."""
    if config.address:
        return config.address
    from eth_account import Account
    return str(Account.from_key(config.private_key).address)


async def _maybe_run_agents(task_context: dict) -> None:
    """Run all Phase 8 AI agents if the interval has elapsed, then deliver via Telegram."""
    global _last_agent_run_ts
    now = time.time()
    if now - _last_agent_run_ts < _AGENT_RUN_INTERVAL_S:
        return
    _last_agent_run_ts = now
    try:
        from wallet_manager.ai_agents import get_agent_registry
        from wallet_manager.strategies.paper_engine import _engine_instance
        registry = get_agent_registry()
        reports = await registry.run_all(paper_engine=_engine_instance)
        if not reports:
            return
        lines: list[str] = [
            f"🤖 *WalletBot Agent Reports*  `{time.strftime('%H:%M UTC', time.gmtime(now))}`",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for report in reports:
            llm_flag = " \\(LLM\\)" if report.used_llm else ""
            safe_title = report.title.replace(".", "\\.").replace("(", "\\(").replace(")", "\\)").replace("-", "\\-").replace("+", "\\+")
            safe_body = report.body[:400].replace(".", "\\.").replace("(", "\\(").replace(")", "\\)").replace("-", "\\-").replace("+", "\\+").replace("$", "\\$")
            lines.append(f"\n*{safe_title}*{llm_flag}\n{safe_body}")
            if len(report.body) > 400:
                lines.append("_\\(truncated — full report at /api/agents\\)_")
        msg = "\n".join(lines)
        notifier = task_context.get("notifier")
        if notifier:
            with contextlib.suppress(Exception):
                await notifier.send_message(msg, parse_mode="MarkdownV2")
        log.info("Phase 8 agents: %d reports delivered (LLM=%s)", len(reports), any(r.used_llm for r in reports))
    except Exception as exc:
        log.warning("_maybe_run_agents failed (non-critical): %s", exc)


async def _collect_alerter_inputs(  # type: ignore[no-untyped-def]
    snapshot,           # PortfolioSnapshot — may be None
    cfg,                # Settings instance
) -> dict:
    """Collect all data ProactiveAlerter.check_and_alert() needs."""
    import time as _time

    # 1. Drift level
    _drift_name = "NOMINAL"
    try:
        if _drift_detector is not None and hasattr(
            _drift_detector, "_last_drift_level"
        ):
            _drift_name = _drift_detector._last_drift_level.name
    except Exception:
        pass

    # 2. Aave APY
    _apy = 0.0
    try:
        from wallet_manager.strategies.yield_strategy import get_apy_monitor
        _apy = get_apy_monitor().last_apy
    except Exception:
        pass

    # 3. Model staleness (days since ppo_model_path was last written)
    _staleness = 0.0
    try:
        import os as _os
        _model_path = cfg.ppo_model_path
        if _os.path.exists(_model_path):
            _staleness = (_time.time() - _os.path.getmtime(_model_path)) / 86400.0
    except Exception:
        pass

    # 4. RPC latency — worst-case across all tracked endpoints
    _rpc_latency = 0.0
    try:
        from wallet_manager.orchestrator import _get_rpc_router
        _latency_tracker = getattr(_get_rpc_router(), "_latency_tracker", {})
        _all = [
            max(v) for v in _latency_tracker.values()
            if isinstance(v, (list, tuple)) and v
        ]
        if _all:
            _rpc_latency = max(_all)
    except Exception:
        pass

    # 5. Current drawdown vs peak allocation
    _drawdown = 0.0
    try:
        if snapshot is not None and snapshot.total_value_usd > 0:
            _peak = max(
                (a.current_usd for a in snapshot.allocations), default=0.0
            )
            _drawdown = max(0.0, 1.0 - snapshot.total_value_usd / max(_peak, 1.0))
    except Exception:
        pass

    return dict(
        drift_level=_drift_name,
        apy=_apy,
        model_staleness_days=_staleness,
        rpc_latency_ms=_rpc_latency,
        drawdown=_drawdown,
    )


async def _default_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Task batch with active gas monitoring loop.

    Runs multiple cycles per wallet: each cycle waits for safe gas,
    fetches balance, then sleeps. Configurable via DEFAULT_TASK_LOOP_CYCLES.
    Replace with your own task_factory for production transactions.
    """
    from wallet_manager.network.gas_monitor import GasMonitor

    config = manager.get_config(wallet_id)
    gas_monitor = GasMonitor(rpc_url=config.rpc_url)
    max_gwei = float(config.max_gas_price_gwei)
    receipts: list[TxReceipt] = []

    for cycle in range(_TASK_LOOP_CYCLES):
        safe, current = await gas_monitor.is_safe(max_gwei)
        if not safe:
            log.info(
                f"Wallet '{wallet_id}' cycle {cycle + 1}/{_TASK_LOOP_CYCLES}: gas {current:.2f} Gwei > max {max_gwei:.2f}, waiting.",
            )
            await asyncio.sleep(_GAS_POLL_INTERVAL)
            continue

        try:
            await manager.get_safe_balance(wallet_id)
        except Exception as exc:
            log.info(f"Wallet '{wallet_id}' cycle {cycle + 1}: balance fetch skipped ({type(exc).__name__})")
            continue

        if cycle < _TASK_LOOP_CYCLES - 1:
            await asyncio.sleep(_GAS_POLL_INTERVAL)

    return receipts


async def _preload_robust_engine(engine: object, current_price: float) -> None:
    """Pre-seed RobustAlphaEngine with historical ETH prices to skip warmup.

    Fetches the last 24 h of prices from CoinGecko (≥ 288 5-min ticks).
    Falls back to synthetic Gaussian prices if the API is unavailable.
    Only runs when the engine has never ingested any price (_n_ticks == 0).
    """
    import math
    import aiohttp
    from wallet_manager.strategies.robust_alpha import FRACDIFF_MIN_OBSERVATIONS

    if getattr(engine, "_n_ticks", 1) > 0:
        return

    # 30 GARCH warmup + 60 LSTM buffer (SEQ_LEN_LSTM) + 60 extra = 150 ticks.
    # Transformer buffer (seq_len=30) also fills. data_coverage=150/288=0.52 → conf > 0.15.
    needed = max(FRACDIFF_MIN_OBSERVATIONS + 10, 150)
    prices: list[float] = []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0)
        ) as session:
            async with session.get(
                "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
                "?vs_currency=usd&days=1",
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    prices = [float(p[1]) for p in data.get("prices", [])]
    except Exception as exc:
        log.debug("Price history preload: CoinGecko unavailable (%s)", exc)

    if len(prices) < needed:
        rng = random.Random(42)
        p = current_price
        prices = []
        for _ in range(needed):
            p *= math.exp(rng.gauss(0.0, 0.001))
            prices.append(p)

    prices = prices[-needed:]
    ts = time.time() - len(prices) * 300.0
    _ingest = getattr(engine, "ingest_price", None) or getattr(engine, "process_tick", None)
    if _ingest is None:
        log.warning("_preload_robust_engine: engine has no ingest_price / process_tick method")
        return
    for p in prices:
        _ingest(price=p, volume=0.0, timestamp=ts)
        ts += 300.0
    log.info(
        "AlphaEngine: pre-seeded %d historical prices — warmup bypassed.",
        len(prices),
    )


async def _robust_paper_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Paper Trade task factory — high-fidelity simulation with full Robust pipeline.

    Runs the entire Robust execution pipeline (HMM regime filter, fracdiff alpha,
    walk-forward validation, Monte Carlo stress test, liquidity profiling, latency
    shield) but stops BEFORE send_transaction. Simulates slippage and gas costs.
    Records to a virtual ledger (paper_trades.db).

    EXECUTION_MODE=robust_paper activates this factory.
    """
    from wallet_manager.network.gas_monitor import GasMonitor
    from wallet_manager.orchestrator import send_heartbeat_telemetry
    from wallet_manager.strategies.latency_shield import get_latency_shield
    from wallet_manager.strategies.paper_engine import get_paper_engine
    from wallet_manager.strategies.regime_filter import get_regime_filter
    from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine
    from wallet_manager.strategies.stochastic_stress_test import get_stress_tester

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("PaperTrade: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    if chain_id != 8453:
        log.info("PaperTrade: Skipping — Base Mainnet only.")
        return receipts

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gas = config.max_gas_price_gwei if config.max_gas_price_gwei is not None else 100.0
    max_gwei = min(25.0, float(max_gas))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)

    try:
        # Initialize engines
        paper_engine = await get_paper_engine()
        stress_tester = get_stress_tester()
        latency_shield = get_latency_shield(rpc_url)
        regime_engine = get_regime_filter()
        robust_engine = get_robust_alpha_engine()

        # 1. Latency shield — validate quote freshness
        eth_price = await actions.fetch_eth_price_usd()
        quote_validation, latency_profile = await latency_shield.apply_latency_shield(
            price=eth_price,
            quote_timestamp=time.time(),
        )

        if quote_validation.should_invalidate_signals:
            log.info(
                f"PAPER: Stale quote ({quote_validation.quote_age_s:.1f}s) — "
                f"signals invalidated. Reason: {quote_validation.reason_code}"
            )
            with contextlib.suppress(Exception):
                await paper_engine.record_external_skip(
                    wallet_id=wallet_id,
                    primary_reason="QUOTE_INVALIDATED",
                    notes=(
                        f"quote_age_s={quote_validation.quote_age_s:.3f}; "
                        f"reason={quote_validation.reason_code}"
                    ),
                    regime_label="latency_guard",
                    reason_codes=("paper_shadow_only", "paper_signal_rejected"),
                )
            return receipts

        # 2a. Macro Calendar — Phase 5 event blackout
        with contextlib.suppress(Exception):
            from wallet_manager.market_intelligence import get_macro_calendar
            _macro_suppress, _macro_reason = get_macro_calendar().should_suppress_alpha()
            if _macro_suppress:
                log.info(f"PAPER: MacroCalendar suppressing alpha — {_macro_reason}")
                with contextlib.suppress(Exception):
                    await paper_engine.record_external_skip(
                        wallet_id=wallet_id,
                        primary_reason=_macro_reason[:64],
                        notes=_macro_reason,
                        regime_label="macro_event",
                        reason_codes=("paper_shadow_only", "paper_signal_rejected"),
                    )
                return receipts

        # 2b. Regime filter
        regime_engine.ingest_market_data(
            price=eth_price, volume=0.0, timestamp=time.time(),
        )
        override_to_safety, safety_reason = regime_engine.should_override_to_safety()
        if override_to_safety:
            log.info(
                f"PAPER REGIME OVERRIDE: {safety_reason} — "
                f"Would move to 100% USDC + Aave. Skipping alpha."
            )
            with contextlib.suppress(Exception):
                await paper_engine.record_external_skip(
                    wallet_id=wallet_id,
                    primary_reason=safety_reason,
                    notes="Would move to 100% USDC + Aave. Skipping alpha.",
                    regime_label=regime_engine.get_current_regime().regime.value,
                    reason_codes=("paper_shadow_only", "paper_signal_rejected"),
                )
            return receipts

        # Pre-seed engine on first run so warmup completes immediately
        if robust_engine._n_ticks == 0:
            await _preload_robust_engine(robust_engine, eth_price)

        # 3. Robust alpha signal
        robust_signal = robust_engine.ingest_price(
            price=eth_price, volume=0.0, timestamp=time.time(),
        )

        # 4. Monte Carlo stress test — chaos validation
        if robust_signal is not None and (
            robust_signal.should_enter_long or robust_signal.should_enter_short
        ):
            price_history = list(robust_engine._price_history)
            direction = 1 if robust_signal.should_enter_long else -1
            verdict = stress_tester.validate_signal(
                current_price=eth_price,
                signal_alpha=robust_signal.robust_alpha,
                signal_direction=direction,
                price_history=price_history,
            )
            if not verdict.approved:
                log.info(
                    f"PAPER STRESS TEST REJECTED: "
                    f"MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                    f"reasons={verdict.rejection_reasons}"
                )
                with contextlib.suppress(Exception):
                    await paper_engine.record_external_skip(
                        wallet_id=wallet_id,
                        primary_reason="STRESS_TEST_REJECTED",
                        notes=(
                            f"mc_win_rate={verdict.monte_carlo.win_rate:.4f}; "
                            f"confidence={verdict.confidence_score:.4f}; "
                            f"reasons={verdict.rejection_reasons}"
                        ),
                        regime_label=regime_engine.get_current_regime().regime.value,
                        reason_codes=("paper_shadow_only", "paper_signal_rejected"),
                    )
                return receipts
            log.info(
                f"PAPER STRESS TEST PASSED: "
                f"MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                f"confidence={verdict.confidence_score:.2f}"
            )

        # 5. Shadow execution via PaperTradingEngine
        paper_result = await paper_engine.run_shadow_cycle(
            manager=manager,
            wallet_id=wallet_id,
            portfolio_manager=portfolio_manager,
            task_context=ctx,
            analytics_db=analytics_db,
            quant_analytics=None,
        )

        # 6. Get performance report for logging
        await paper_engine.get_performance_report()
        telegram_report = await paper_engine.get_telegram_report()

        log.info(
            f"PAPER TRADE: action={paper_result.action} "
            f"alpha={paper_result.signal_alpha:+.3f} "
            f"fill={paper_result.simulated_fill_price:.2f} "
            f"slip={paper_result.slippage_bps:.1f}bps "
            f"gas=${paper_result.gas_cost_usd:.4f} "
            f"vPnL=${paper_result.virtual_pnl_usd:+.2f} "
            f"latency={latency_profile.total_latency_ms:.0f}ms "
            f"regime={regime_engine.get_current_regime().regime.value} "
            f"duration={paper_result.cycle_duration_ms:.0f}ms"
        )

        # 7. Report via Telegram (virtual P&L vs gas cost estimate)
        with contextlib.suppress(Exception):
            await send_heartbeat_telemetry(
                portfolio_manager=portfolio_manager,
                rpc_url=rpc_url,
                address=address,
                task_context={**ctx, "paper_report": telegram_report},
            )

    except Exception as exc:
        log.info(
            f"PaperTrade engine cycle skipped: {type(exc).__name__}: {exc}. "
            f"Will retry next cycle."
        )

    # 8. Phase 8: periodic AI agent run + Telegram delivery (every 6h)
    with contextlib.suppress(Exception):
        await _maybe_run_agents(ctx)

    # 9. JS-7.2: periodic performance reporter tick (daily/weekly)
    with contextlib.suppress(Exception):
        from wallet_manager.reporting import get_reporter
        from wallet_manager.strategies.paper_engine import _engine_instance as _pe_inst
        notifier = ctx.get("notifier")
        await get_reporter().tick(notifier=notifier, paper_engine=_pe_inst)

    # 10. JS-7.1: alert escalation delivery tick
    with contextlib.suppress(Exception):
        from wallet_manager.alerting import get_alert_system
        notifier = ctx.get("notifier")
        await get_alert_system().tick(notifier=notifier)

    return receipts


async def _robust_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Ultra-Robust Anti-Overfit task factory — Phase 4+5 combined.

    Layers on top of the ML-Quant pipeline with institutional defences:
    1. LatencyShield — Stale quote protection + RPC health monitoring
    2. RegimeFilter (HMM) — Market regime detection + economic calendar blackout
    3. RobustAlphaEngine — Fractional Differentiation + Permutation Importance
    4. StochasticStressTester — Monte Carlo chaos validation + RFE
    5. WalkForwardValidator — Purged K-Fold cross-validation + retrain control
    6. LiquidityAlpha — V3 tick-level profiling + split-order execution
    7. MLQuantExecutionEngine — Full ML pipeline (GARCH + Kalman + XGBoost/LSTM)

    If the HMM detects HIGH_VOL or NEWS_BLACKOUT, ALL alpha signals are overridden
    and capital moves to 100% USDC + Aave Supply.
    YIELD and AIRDROP pillars execute alongside.

    EXECUTION_MODE=robust or robust_live activates this factory.
    """
    from wallet_manager.network.gas_monitor import GasMonitor
    from wallet_manager.orchestrator import send_heartbeat_telemetry
    from wallet_manager.strategies.execution_logic import get_ml_execution_engine
    from wallet_manager.strategies.latency_shield import get_latency_shield
    from wallet_manager.strategies.liquidity_alpha import get_liquidity_alpha
    from wallet_manager.strategies.quant_analytics import get_quant_analytics
    from wallet_manager.strategies.regime_filter import get_regime_filter
    from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine
    from wallet_manager.strategies.stochastic_stress_test import get_stress_tester
    from wallet_manager.strategies.wf_validator import get_wf_validator

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("Robust: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Rebalance check
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"Robust {rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        _cfg_r = get_settings()
        _algo_w = _cfg_r.algo_weight
        _airdrop_w = _cfg_r.airdrop_weight
        liquid_weight_total = _algo_w + _airdrop_w
        algo_val = liquid_val * (_algo_w / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (_airdrop_w / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"Robust pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"Robust YIELD skip: {exc}")

    # 2. LATENCY SHIELD — Stale quote protection
    if chain_id == 8453:
        try:
            latency_shield = get_latency_shield(rpc_url)
            eth_price = await actions.fetch_eth_price_usd()
            quote_val, lat_prof = await latency_shield.apply_latency_shield(
                price=eth_price, quote_timestamp=time.time(),
            )
            if quote_val.should_invalidate_signals:
                log.info(
                    f"LATENCY SHIELD: Stale quote ({quote_val.quote_age_s:.1f}s) — "
                    f"ALL signals invalidated. Reason: {quote_val.reason_code}"
                )
                return receipts
        except Exception as exc:
            log.debug(f"Latency shield skip: {exc}")
            eth_price = await actions.fetch_eth_price_usd()

    # 3. REGIME FILTER — HMM + Economic Calendar guardrail
    if chain_id == 8453:
        regime_engine = get_regime_filter()
        try:
            regime_engine.ingest_market_data(
                price=eth_price, volume=0.0, timestamp=time.time(),
            )
        except Exception as exc:
            log.debug(f"Robust regime ingest skip: {exc}")

        override_to_safety, safety_reason = regime_engine.should_override_to_safety()
        if override_to_safety:
            log.info(
                f"REGIME OVERRIDE: {safety_reason} — "
                f"Moving to 100% USDC + Aave Supply. Skipping all alpha."
            )
            # Execute safety: supply all idle USDC to Aave
            try:
                from wallet_manager.strategies import yield_strategy
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                if free_usdc >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        free_usdc, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
                    log.info(f"REGIME OVERRIDE: Supplied {free_usdc / 1e6:.2f} USDC to Aave")
            except Exception as exc:
                log.info(f"Robust safety supply skipped: {type(exc).__name__}: {exc}")

            # Skip all alpha — go straight to heartbeat
            with contextlib.suppress(Exception):
                await send_heartbeat_telemetry(
                    portfolio_manager=portfolio_manager,
                    rpc_url=rpc_url,
                    address=address,
                    task_context=ctx,
                )
            return receipts

    # 4. ROBUST ALPHA + STRESS TEST + WALK-FORWARD + LIQUIDITY + ML-QUANT ENGINE
    if chain_id == 8453:
        try:
            robust_engine = get_robust_alpha_engine()
            liquidity_engine = get_liquidity_alpha()
            wf_validator = await get_wf_validator()
            stress_tester = get_stress_tester()

            # Pre-seed on first run to skip GARCH/LSTM warmup period
            if robust_engine._n_ticks == 0:
                await _preload_robust_engine(robust_engine, eth_price)

            # Walk-forward retrain check
            try:
                if wf_validator.should_retrain(wf_validator._controller._trade_count):
                    log.info("WALK-FORWARD: Retrain triggered — validating model OOS performance")
                    decision = await wf_validator.run_retrain_cycle()
                    log.info(
                        f"WALK-FORWARD: {decision.action} "
                        f"(mean_oos_sharpe={decision.mean_oos_sharpe:.3f})"
                    )
            except Exception as exc:
                log.debug(f"Walk-forward retrain skip: {exc}")

            # Ingest latest price into robust alpha
            try:
                robust_signal = robust_engine.ingest_price(
                    price=eth_price, volume=0.0, timestamp=time.time(),
                )
            except Exception as exc:
                log.debug(f"Robust alpha ingest skip: {exc}")
                robust_signal = None

            # Monte Carlo stress test — chaos gate
            if robust_signal is not None and (
                robust_signal.should_enter_long or robust_signal.should_enter_short
            ):
                try:
                    price_history = list(robust_engine._price_history)
                    direction = 1 if robust_signal.should_enter_long else -1
                    verdict = stress_tester.validate_signal(
                        current_price=eth_price,
                        signal_alpha=robust_signal.robust_alpha,
                        signal_direction=direction,
                        price_history=price_history,
                    )
                    if not verdict.approved:
                        log.info(
                            f"STRESS TEST REJECTED: MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                            f"reasons={verdict.rejection_reasons} — trade discarded"
                        )
                        robust_signal = None
                    else:
                        log.info(
                            f"STRESS TEST PASSED: MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                            f"confidence={verdict.confidence_score:.2f}"
                        )
                except Exception as exc:
                    log.debug(f"Stress test skip: {exc}")

            # Liquidity profiling — reduce size if thin
            liquidity_multiplier = 1.0
            try:
                # Profiling requires a pool-specific liquidity profile; until we
                # have one at this layer, keep neutral sizing instead of calling
                # the helper with an invalid signature.
                should_reduce = False
                liq_mult = 1.0
                reason = "profile unavailable"
                if should_reduce:
                    liquidity_multiplier = liq_mult
                    log.info(
                        f"LIQUIDITY: Size reduced to {liq_mult:.0%} — {reason}"
                    )
            except Exception as exc:
                log.debug(f"Liquidity profiling skip: {exc}")

            # ML-Quant engine with robust overlay
            engine = get_ml_execution_engine()
            quant_analytics = None
            with contextlib.suppress(Exception):
                quant_analytics = await get_quant_analytics()

            cycle_result = await engine.run_cycle(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                analytics_db=analytics_db,
                quant_analytics=quant_analytics,
            )
            receipts.extend(cycle_result.receipts)
            receipts.extend(cycle_result.sweep_receipts)

            # Record trade for walk-forward counter
            with contextlib.suppress(Exception):
                wf_validator.record_trade()

            log.info(
                f"Robust cycle: action={cycle_result.action_taken} "
                f"risk_src={cycle_result.risk_source} "
                f"drawdown={cycle_result.drawdown_state} "
                f"VaR99={cycle_result.var_99:.2f} "
                f"liq_mult={liquidity_multiplier:.0%} "
                f"robust_d={robust_engine._current_d:.2f} "
                f"stationary={robust_engine._is_stationary} "
                f"pnl=${cycle_result.pnl_usd:+.2f} "
                f"duration={cycle_result.cycle_duration_ms:.0f}ms"
            )
        except Exception as exc:
            log.info(f"Robust engine cycle skipped: {type(exc).__name__}: {exc}")

    # 4. AIRDROP (ping-pong, 15% cap)
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 5. HEARTBEAT
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
        )
    except Exception as exc:
        log.debug(f"Robust heartbeat skip: {exc}")

    return receipts


async def _ml_quant_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """ML-Quant Alpha Engine task factory — GARCH/Kalman/XGBoost/LSTM + PPO Kelly.

    Replaces QUANT-X with the full ML-integrated pipeline:
    1. Stochastic ML Alpha Engine (GARCH + Kalman + XGBoost/LSTM ensemble)
    2. Deep Kelly Risk Engine (PPO RL + VaR + Drawdown control)
    3. Smart Order Router (Multi-DEX aggregation)
    4. MEV Shield (Private RPC + Gaussian Jitter + Dynamic RBF)
    5. Flash Crash Circuit Breaker (8% / 5min)
    6. Quant Analytics (TCA + Implementation Shortfall)

    YIELD and AIRDROP pillars still execute alongside.
    """
    from wallet_manager.network.gas_monitor import GasMonitor
    from wallet_manager.orchestrator import send_heartbeat_telemetry
    from wallet_manager.strategies.execution_logic import get_ml_execution_engine
    from wallet_manager.strategies.quant_analytics import get_quant_analytics

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("ML-Quant: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Rebalance check
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"ML-Quant {rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        _cfg_mq = get_settings()
        _algo_w_mq = _cfg_mq.algo_weight
        _airdrop_w_mq = _cfg_mq.airdrop_weight
        liquid_weight_total = _algo_w_mq + _airdrop_w_mq
        algo_val = liquid_val * (_algo_w_mq / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (_airdrop_w_mq / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"ML-Quant pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"ML-Quant YIELD skip: {exc}")

    # 2. ML-QUANT ALPHA ENGINE (replaces QUANT-X + legacy algo)
    _severe_drift = False
    if chain_id == 8453:
        try:
            engine = get_ml_execution_engine()
            # Pre-seed inner alpha engine on first run to skip GARCH/LSTM warmup
            if engine.alpha._n_ticks == 0:
                await _preload_robust_engine(engine.alpha, eth_price)
            quant_analytics = None
            try:
                quant_analytics = await get_quant_analytics()
            except Exception as exc:
                log.debug(f"ML-Quant analytics init skip: {exc}")

            cycle_result = await engine.run_cycle(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                analytics_db=analytics_db,
                quant_analytics=quant_analytics,
            )
            receipts.extend(cycle_result.receipts)
            receipts.extend(cycle_result.sweep_receipts)

            log.info(
                f"ML-Quant cycle: action={cycle_result.action_taken} "
                f"risk_src={cycle_result.risk_source} "
                f"drawdown={cycle_result.drawdown_state} "
                f"VaR99={cycle_result.var_99:.2f} "
                f"dex={cycle_result.dex_venue or 'N/A'} "
                f"pnl=${cycle_result.pnl_usd:+.2f} "
                f"duration={cycle_result.cycle_duration_ms:.0f}ms"
            )

            # ConceptDrift check — after run_cycle() the feature vector is available
            if _drift_detector is not None:
                import time as _time
                try:
                    _fv = engine.alpha.get_last_feature_vector()
                    if _fv is not None:
                        _drift_level = _drift_detector.update(
                            prediction=0.0,
                            actual=0.0,
                            features=_fv,
                        )
                        if _drift_level.name in {"RETRAIN", "SEVERE"}:
                            log.warning(
                                "ConceptDriftDetector: drift_level=%s",
                                _drift_level.name,
                            )
                            if _drift_level.name == "SEVERE":
                                _severe_drift = True
                except Exception as _drift_exc:
                    log.debug("ConceptDriftDetector update skipped: %s", _drift_exc)

            # ── FeatureDriftMonitor — per-feature KL divergence tracking ──────────
            try:
                from wallet_manager.strategies.alpha_engine import FeatureStore
                from wallet_manager.strategies.robust_alpha import get_feature_drift_monitor
                _fdm = get_feature_drift_monitor()
                _fv2 = engine.alpha.get_last_feature_vector()
                if _fv2 is not None:
                    _fnames = getattr(FeatureStore, "FEATURE_NAMES", None) or [
                        f"feature_{i}" for i in range(len(_fv2))
                    ]
                    _fdm.update(_fv2, _fnames)
                    _report = _fdm.get_drift_report()
                    _drifted = _report.get("drifted_features", [])
                    if _drifted:
                        log.warning(
                            "FeatureDriftMonitor: %d feature(s) drifted | "
                            "worst=%s (KL=%.4f)",
                            len(_drifted),
                            _report.get("worst_feature", "?"),
                            float(_report.get("worst_kl", 0.0)),
                        )
            except Exception as _fdm_exc:
                log.debug("FeatureDriftMonitor update skipped: %s", _fdm_exc)
        except Exception as exc:
            log.info(f"ML-Quant engine cycle skipped: {type(exc).__name__}: {exc}")

    if _severe_drift:
        log.warning("ConceptDriftDetector: SEVERE drift — skipping remainder of wallet cycle.")
        return receipts

    # 3. AIRDROP (ping-pong, 15% cap)
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 4. HEARTBEAT
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
        )
    except Exception as exc:
        log.debug(f"ML-Quant heartbeat skip: {exc}")

    return receipts


async def _quantx_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """QUANT-X Engine task factory — institutional-grade DeFi execution.

    Replaces the legacy ALGO pillar with the QUANT-X pipeline:
    1. Signal Intelligence (Z-Score 2.0 + OFI + VWAP)
    2. Kelly Criterion position sizing
    3. Smart Order Router (Multi-DEX)
    4. MEV Shield (Private RPC + Gaussian Jitter)
    5. Flash Crash Circuit Breaker (8% / 5min)

    YIELD, SHORT, and AIRDROP pillars are still executed alongside.
    """
    from wallet_manager.network.gas_monitor import GasMonitor
    from wallet_manager.orchestrator import send_heartbeat_telemetry
    from wallet_manager.strategies.quantx_engine import get_quantx_engine

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("QUANT-X: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Rebalance check (same as legacy)
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"QUANT-X {rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        # SHORT pillar
        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        liquid_weight_total = 0.35 + 0.15
        algo_val = liquid_val * (0.35 / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (0.15 / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"QUANT-X pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"QUANT-X YIELD skip: {exc}")

    # 2. QUANT-X ALGO ENGINE (replaces legacy algo_strategy + short_strategy)
    if chain_id == 8453:
        try:
            engine = get_quantx_engine()
            cycle_result = await engine.run_cycle(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                analytics_db=analytics_db,
            )
            receipts.extend(cycle_result.receipts)
            receipts.extend(cycle_result.sweep_receipts)

            alpha_text = (
                f"{cycle_result.signal.alpha_score:+.3f}"
                if cycle_result.signal is not None
                else "N/A"
            )
            log.info(
                f"QUANT-X cycle: action={cycle_result.action_taken} "
                f"alpha={alpha_text} "
                f"dex={cycle_result.dex_venue or 'N/A'} "
                f"pnl=${cycle_result.pnl_usd:+.2f}"
            )
        except Exception as exc:
            log.info(f"QUANT-X engine cycle skipped: {type(exc).__name__}: {exc}")

    # 3. AIRDROP (ping-pong, 15% cap) — unchanged
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 4. HEARTBEAT
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
        )
    except Exception as exc:
        log.debug(f"QUANT-X heartbeat skip: {exc}")

    return receipts


async def _three_pillar_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """3-Pillar Portfolio Orchestrator: YIELD (50%) + ALGO (35%) + AIRDROP (15%).

    When QUANTX_ENGINE_MODE=quantx (env), delegates the ALGO pillar to the
    QUANT-X engine (signal_intelligence + kelly_criterion + smart_order_router
    + mev_shield). All other pillars (YIELD, SHORT, AIRDROP) remain unchanged.

    Delegates to yield_strategy, algo_strategy, and airdrop (ping-pong) based on
    PortfolioManager state. All txs routed through manager.sign_and_broadcast_tx.
    """
    from wallet_manager.network.gas_monitor import GasMonitor
    from wallet_manager.orchestrator import send_heartbeat_telemetry

    ctx = task_context or {}
    command_state = ctx.get("command_state")

    # Circuit breaker: halt ALL trading (YIELD, ALGO, AIRDROP) for 24h when triggered
    if command_state and getattr(command_state, "is_circuit_breaker_halted", False) and command_state.is_circuit_breaker_halted:
        log.info(f"Wallet '{wallet_id}': Circuit breaker halted — skip all pillars. Send /resume to override.")
        return []

    # Paper trade mode: Shadow execution with slippage sim + virtual ledger
    _exec_mode = _cfg().execution_mode
    if _exec_mode == "robust_paper":
        return await _robust_paper_task_factory(manager, wallet_id, task_context)

    # Robust live mode: Full anti-overfit pipeline (production)
    if _exec_mode in ("robust", "robust_live"):
        return await _robust_task_factory(manager, wallet_id, task_context)

    # ML-Quant mode: GARCH + Kalman + XGBoost/LSTM + PPO Kelly + TCA
    if _exec_mode == "ml_quant":
        return await _ml_quant_task_factory(manager, wallet_id, task_context)

    # QUANT-X mode: use institutional-grade engine for ALGO pillar
    _quantx_mode = _cfg().quantx_engine_mode
    if _quantx_mode == "quantx":
        return await _quantx_task_factory(manager, wallet_id, task_context)

    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Initial $250 or monthly $150 rebalance: distribute according to 50/35/15 when due
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"{rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain (first wallet as primary)
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        # Liquid balance: ETH + USDC not in Aave
        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        # SHORT pillar: value of active short position (unrealized P&L)
        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost  # Positive if price dropped
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        # Distribute liquid among ALGO + AIRDROP (proportional to weights)
        liquid_weight_total = 0.35 + 0.15  # ALGO + AIRDROP weights
        algo_val = liquid_val * (0.35 / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (0.15 / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"Pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only) — with APY optimization check
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            aave_apy = await yield_strategy.get_apy(rpc_url)
            if aave_apy < _AAVE_APY_ALERT_THRESHOLD:
                alt_apy, alt_protocol = await actions.get_alternative_stable_yield_apy(
                    rpc_url, chain_id
                )
                if alt_apy > aave_apy:
                    tg = ctx.get("telegram_manager")
                    if tg and tg._is_configured():
                        try:
                            from wallet_manager.security.telegram_manager import (
                                YieldAlertPayload,
                            )
                            await tg.send_yield_alert(
                                YieldAlertPayload(
                                    aave_apy_pct=aave_apy,
                                    alternative_protocol=alt_protocol,
                                    alternative_apy_pct=alt_apy,
                                    detail=f"Aave at {aave_apy:.2f}% APY. Consider {alt_protocol} ({alt_apy:.2f}%).",
                                )
                            )
                        except Exception as exc:
                            log.info(f"Yield alert skipped: {type(exc).__name__}: {exc}")

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:  # need to supply
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):  # min $5
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"YIELD strategy skip: {exc}")

    # --- Flash Crash Detection (before any trading) ---
    from wallet_manager.strategies.flash_crash_detector import (
        activate_kill_switch,
        get_detector,
        send_kill_switch_alert,
    )
    crash_detector = get_detector()
    if crash_detector.is_kill_switch_active:
        log.info("KILL_SWITCH active — all strategies halted.")
        return receipts
    try:
        crash_alert = crash_detector.ingest_price(
            await actions.get_current_price(rpc_url, chain_id)
        )
        if crash_alert is not None:
            activate_kill_switch(ctx, crash_alert)
            await send_kill_switch_alert(ctx, crash_alert)
            log.critical(
                f"🚨 KILL_SWITCH triggered: {crash_alert.drop_pct:.1f}% crash. "
                f"All trading halted."
            )
            return receipts
    except Exception as exc:
        log.debug(f"Flash crash check failed: {exc}")

    # 2. ALGO (Base only) — with Bollinger Bands + Z-Score + RSI
    algo_z_score = 0.0
    algo_rsi = 50.0
    algo_bb_upper = 0.0
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import algo_strategy

            recs = await algo_strategy.run_algo_strategy(
                manager, wallet_id, portfolio_manager, ctx, analytics_db=analytics_db,
            )
            receipts.extend(recs)

            # Capture signals for short strategy reuse
            state = algo_strategy._load_state(algo_strategy._default_history_path())
            if len(state.prices) >= algo_strategy.EMA_PERIOD:
                _, _, algo_z_score = algo_strategy.compute_z_score_ema(
                    state.prices, algo_strategy.EMA_PERIOD
                )
                algo_rsi = algo_strategy.compute_rsi(state.prices, algo_strategy.RSI_PERIOD)
                _, _, algo_bb_upper = algo_strategy.compute_bollinger_bands(
                    state.prices, algo_strategy.EMA_PERIOD
                )
        except Exception as exc:
            log.debug(f"ALGO strategy skip: {exc}")

    # 3. SHORT — Hedged Delta-Neutral (Base only, 10:1 model)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import short_strategy

            current_price = await actions.get_current_price(rpc_url, chain_id)

            # Pass alpha_signal when not in legacy mode
            _short_alpha = None
            _exec_mode_short = _cfg().execution_mode
            if _exec_mode_short != "legacy":
                try:
                    from wallet_manager.strategies.alpha_engine import get_alpha_engine
                    _ae = get_alpha_engine()
                    _short_alpha = _ae.last_signal
                except Exception:
                    pass

            recs = await short_strategy.run_short_strategy(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                z_score=algo_z_score,
                rsi=algo_rsi,
                current_price=current_price,
                bb_upper=algo_bb_upper,
                analytics_db=analytics_db,
                alpha_signal=_short_alpha,
            )
            receipts.extend(recs)
        except Exception as exc:
            log.debug(f"SHORT strategy skip: {exc}")

    # 4. IDLE SWEEP — Zero Lazy Capital (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies.idle_sweep import run_idle_sweep

            recs = await run_idle_sweep(
                manager=manager,
                wallet_id=wallet_id,
                task_context=ctx,
                analytics_db=analytics_db,
            )
            receipts.extend(recs)
        except Exception as exc:
            log.debug(f"Idle sweep skip: {exc}")

    # 5. AIRDROP (ping-pong, 15% cap)
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 6. HEARTBEAT TELEMETRY — every 6 hours
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
            analytics_db=analytics_db,
        )
    except Exception as exc:
        log.debug(f"Heartbeat telemetry skip: {exc}")

    return receipts


async def _airdrop_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Ping-Pong airdrop farming: swap ETH<->USDC based on which side is larger.

    If USDC value > $10 and USDC > ETH value: swap 20-50% USDC -> ETH.
    If ETH value > $10: swap 10-20% ETH -> USDC.
    When airdrop_budget_usd is in context, caps swap to stay within 15% allocation.
    """
    from wallet_manager.network.gas_monitor import GasMonitor

    ctx = task_context or {}
    notifier = ctx.get("notifier")
    confirmation_gate = ctx.get("confirmation_gate")
    scheduler = ctx.get("scheduler")
    receipts: list[TxReceipt] = []

    if scheduler is None:
        log.info("No scheduler in task_context — skip airdrop logic.")
        return receipts

    config = manager.get_config(wallet_id)
    gas_monitor = GasMonitor(rpc_url=config.rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))

    should_run, seconds_until_due = should_run_swap_this_cycle(scheduler, wallet_id)
    if not should_run:
        if seconds_until_due >= 3600:
            time_str = f"{seconds_until_due / 3600:.1f}h"
        else:
            time_str = f"{seconds_until_due / 60:.1f}m"
        log.info(
            f"Scheduler: next swap in {time_str} for wallet '{wallet_id}'.",
        )
        return receipts

    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        log.info(f"Wallet '{wallet_id}': gas {current_gwei:.2f} Gwei > {max_gwei} — skip.")
        return receipts

    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    if chain_id not in (59144, 8453):
        log.info(f"Wallet '{wallet_id}': chain {chain_id} not supported for swap.")
        return receipts

    address = _get_wallet_address(config)

    try:
        balance_wei = await manager.get_safe_balance(wallet_id)
    except Exception as exc:
        log.info(f"Wallet '{wallet_id}': balance fetch unavailable ({type(exc).__name__}: {exc})")
        return receipts

    try:
        balance_usdc = await actions.fetch_usdc_balance(config.rpc_url, address, chain_id)
    except Exception as exc:
        log.info(f"Wallet '{wallet_id}': USDC balance fetch unavailable ({type(exc).__name__})")
        balance_usdc = 0

    eth_price = await actions.fetch_eth_price_usd()
    eth_value_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
    usdc_value_usd = actions.usdc_balance_usd_value(balance_usdc)

    # Ping-Pong: choose direction
    do_usdc_to_eth = (
        actions.is_usdc_swap_eligible(balance_usdc, current_gwei, max_gas_gwei=max_gwei)
        and usdc_value_usd > eth_value_usd
    )
    do_eth_to_usdc = actions.is_swap_eligible(
        balance_wei, eth_price, current_gwei,
        min_eth_value_usd=actions.MIN_ETH_VALUE_USD,
        max_gas_gwei=max_gwei,
    )

    airdrop_budget_usd = (ctx or {}).get("airdrop_budget_usd")
    if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
        max_swap_usd = min(usdc_value_usd + eth_value_usd, airdrop_budget_usd)
        if max_swap_usd < 10:
            log.info(f"Wallet '{wallet_id}': airdrop budget ${max_swap_usd:.0f} < $10.")
            return receipts

    if do_usdc_to_eth:
        # USDC -> ETH: swap 20-50% of USDC (capped by airdrop budget if set)
        pct = random.uniform(0.20, 0.50)
        amount_usdc = int(balance_usdc * pct)
        if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
            max_usdc = int(airdrop_budget_usd * 0.5 * 1e6)
            amount_usdc = min(amount_usdc, max_usdc)
        min_usdc = int(10 * 1e6)
        if amount_usdc < min_usdc:
            log.info(f"Wallet '{wallet_id}': USDC amount too small.")
            return receipts

        return await _execute_usdc_to_eth(
            manager, wallet_id, config, address, chain_id,
            amount_usdc, current_gwei,
            notifier, confirmation_gate, scheduler,
            analytics_db=ctx.get("analytics_db"),
        )

    if do_eth_to_usdc:
        # ETH -> USDC: swap 10-20% of ETH (capped by airdrop budget if set)
        pct = random.uniform(0.10, 0.20)
        amount_wei = int(balance_wei * pct)
        if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
            max_eth_val_usd = airdrop_budget_usd * 0.5
            max_wei = int(max_eth_val_usd * 1e18 / eth_price)
            amount_wei = min(amount_wei, max_wei)
        if amount_wei < 10**15:
            log.info(f"Wallet '{wallet_id}': amount too small.")
            return receipts

        return await _execute_eth_to_usdc(
            manager, wallet_id, config, address, chain_id,
            amount_wei, current_gwei,
            notifier, confirmation_gate, scheduler,
            analytics_db=ctx.get("analytics_db"),
        )

    log.info(f"Wallet '{wallet_id}': not eligible for ping-pong (balances or gas).")
    return receipts


async def _execute_usdc_to_eth(
    manager: WalletManager,
    wallet_id: str,
    config: WalletConfig,
    address: str,
    chain_id: int,
    amount_usdc: int,
    current_gwei: float,
    notifier: Any = None,
    confirmation_gate: Any = None,
    scheduler: Any = None,
    analytics_db: Any = None,
) -> list[TxReceipt]:
    """Execute USDC -> ETH swap."""
    from wallet_manager.orchestrator import WalletOrchestrator

    receipts: list[TxReceipt] = []
    fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)  # Ghost: rotate path
    addrs = actions.get_chain_addresses(chain_id)
    slippage_bps = await actions.get_dynamic_slippage(
        rpc_url=config.rpc_url,
        chain_id=chain_id,
        token_in=addrs.usdc,
        token_out=addrs.weth,
        amount_in=amount_usdc,
        gas_price_gwei=current_gwei,
        fee=fee_tier,
        is_algo=False,
    )
    try:
        plan = await actions.build_usdc_to_eth_plan(
            rpc_url=config.rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_usdc=amount_usdc,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
    except Exception as exc:
        log.error(f"Wallet '{wallet_id}': USDC->ETH plan build failed: {exc}")
        if notifier and hasattr(notifier, "send"):
            await WalletOrchestrator._safe_notify(
                notifier,
                NotificationEvent(
                    kind=EventKind.WALLET_FAILURE,
                    wallet_id=wallet_id,
                    detail=f"Swap plan build failed: {exc}",
                ),
            )
        return receipts

    amount_str = f"{amount_usdc / 1e6:.2f} USDC"
    if notifier and hasattr(notifier, "send_confirmation_request") and confirmation_gate:
        confirmation_gate.request_confirmation()
        try:
            await notifier.send_confirmation_request(
                wallet_id, amount_str, "USDC -> ETH"
            )
        except Exception as exc:
            log.info(f"Confirmation request skipped: {type(exc).__name__}: {exc}")
        confirmed = await confirmation_gate.wait_for_evet(timeout_seconds=300.0)
        if not confirmed:
            log.info(f"Wallet '{wallet_id}': confirmation timeout or rejected.")
            scheduler.schedule_next(wallet_id)
            return receipts

    for i, step in enumerate(plan.steps):
        try:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
            if analytics_db:
                from wallet_manager.core.models import TxRecord, TxStatus

                record = TxRecord(
                    tx_hash=receipt.tx_hash,
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=receipt.nonce_used,
                    gas_price_gwei=step.tx_params.gas_price_gwei,
                    gas_limit=step.tx_params.gas,
                    value_wei=step.tx_params.value,
                    status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type=StrategyType.AIRDROP.value,
                    strategy_id="airdrop_ping_pong",
                )
                await analytics_db.record_tx(record)
        except Exception as exc:
            err_msg = f"Step {i + 1}/{len(plan.steps)} ({step.description}) failed: {exc}"
            log.error(f"Wallet '{wallet_id}': {err_msg}")
            if notifier and hasattr(notifier, "send"):
                await WalletOrchestrator._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_FAILURE,
                        wallet_id=wallet_id,
                        detail=f"Swap failed: {err_msg}",
                    ),
                )
            scheduler.schedule_next(wallet_id)
            return receipts

    scheduler.schedule_next(wallet_id)
    return receipts


async def _execute_eth_to_usdc(
    manager: WalletManager,
    wallet_id: str,
    config: WalletConfig,
    address: str,
    chain_id: int,
    amount_wei: int,
    current_gwei: float,
    notifier: Any = None,
    confirmation_gate: Any = None,
    scheduler: Any = None,
    analytics_db: Any = None,
) -> list[TxReceipt]:
    """Execute ETH -> USDC swap."""
    from wallet_manager.orchestrator import WalletOrchestrator

    receipts: list[TxReceipt] = []
    fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)  # Ghost: rotate path
    addrs = actions.get_chain_addresses(chain_id)
    slippage_bps = await actions.get_dynamic_slippage(
        rpc_url=config.rpc_url,
        chain_id=chain_id,
        token_in=addrs.weth,
        token_out=addrs.usdc,
        amount_in=amount_wei,
        gas_price_gwei=current_gwei,
        fee=fee_tier,
        is_algo=False,
    )
    try:
        plan = await actions.build_eth_to_usdc_plan(
            rpc_url=config.rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_wei=amount_wei,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
    except Exception as exc:
        log.error(f"Wallet '{wallet_id}': swap plan build failed: {exc}")
        if notifier and hasattr(notifier, "send"):
            await WalletOrchestrator._safe_notify(
                notifier,
                NotificationEvent(
                    kind=EventKind.WALLET_FAILURE,
                    wallet_id=wallet_id,
                    detail=f"Swap plan build failed: {exc}",
                ),
            )
        return receipts

    amount_eth = amount_wei / 1e18
    amount_str = f"{amount_eth:.4f} ETH"
    if notifier and hasattr(notifier, "send_confirmation_request") and confirmation_gate:
        confirmation_gate.request_confirmation()
        try:
            await notifier.send_confirmation_request(
                wallet_id, amount_str, "ETH -> USDC"
            )
        except Exception as exc:
            log.info(f"Confirmation request skipped: {type(exc).__name__}: {exc}")
        confirmed = await confirmation_gate.wait_for_evet(timeout_seconds=300.0)
        if not confirmed:
            log.info(f"Wallet '{wallet_id}': confirmation timeout or rejected.")
            scheduler.schedule_next(wallet_id)
            return receipts

    for i, step in enumerate(plan.steps):
        try:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
            if analytics_db:
                from wallet_manager.core.models import TxRecord, TxStatus

                record = TxRecord(
                    tx_hash=receipt.tx_hash,
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=receipt.nonce_used,
                    gas_price_gwei=step.tx_params.gas_price_gwei,
                    gas_limit=step.tx_params.gas,
                    value_wei=step.tx_params.value,
                    status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type=StrategyType.AIRDROP.value,
                    strategy_id="airdrop_ping_pong",
                )
                await analytics_db.record_tx(record)
        except Exception as exc:
            err_msg = f"Step {i + 1}/{len(plan.steps)} ({step.description}) failed: {exc}"
            log.error(f"Wallet '{wallet_id}': {err_msg}")
            if notifier and hasattr(notifier, "send"):
                await WalletOrchestrator._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_FAILURE,
                        wallet_id=wallet_id,
                        detail=f"Swap failed: {err_msg}",
                    ),
                )
            scheduler.schedule_next(wallet_id)
            return receipts

    scheduler.schedule_next(wallet_id)
    return receipts
