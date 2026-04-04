"""
wallet_manager/strategies/execution_logic.py

Surgical Execution Engine — ML-Integrated Order Execution Pipeline.

Top-level coordinator that replaces quantx_engine.py's execution loop
with an ML-aware pipeline:

1. ALPHA: StochasticMLAlphaEngine (GARCH + Kalman + XGBoost/LightGBM/LSTM/Transformer)
2. RISK: DeepKellyRiskEngine (PPO RL + Drawdown + VaR)
3. ROUTE: SmartOrderRouter (Multi-DEX aggregation)
4. PROTECT: MEVShield (Private RPC + Gaussian Jitter + Dynamic RBF)
5. CRASH: FlashCrashDetector (8% / 5min circuit breaker)
6. SWEEP: idle_sweep (Zero-Lazy-Capital → Aave V3)
7. ANALYTICS: QuantAnalytics (TCA + Implementation Shortfall)

The engine receives raw market data (price, volume, OFI) and produces
fully executed, risk-managed trades with complete post-trade analytics.

Integration
-----------
This module replaces _quantx_task_factory in the orchestrator when
EXECUTION_MODE=ml_quant (env var). All existing infrastructure
(WalletManager, Telegram, analytics DB) is preserved.
"""

# mypy: disable-error-code="no-untyped-def,no-any-return"

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wallet_manager import actions
from wallet_manager.core.models import TxReceipt, TxRecord, TxStatus
from wallet_manager.portfolio_manager import StrategyType
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.strategies.alpha_engine import (
    AlphaMLSignal,
    get_alpha_engine,
)
from wallet_manager.strategies.deep_kelly_risk import (
    TradeOutcome,
    get_deep_kelly_engine,
)
from wallet_manager.strategies.flash_crash_detector import (
    activate_kill_switch,
    send_kill_switch_alert,
)
from wallet_manager.strategies.flash_crash_detector import (
    get_detector as get_crash_detector,
)
from wallet_manager.strategies.idle_sweep import run_idle_sweep
from wallet_manager.strategies.kelly_criterion import RegimeAwareKelly
from wallet_manager.strategies.mev_shield import (
    get_mev_shield,
)
from wallet_manager.strategies.regime_filter import get_regime_filter
from wallet_manager.strategies.smart_order_router import (
    get_smart_order_router,
)

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from wallet_manager.portfolio_manager import PortfolioManager
    from wallet_manager.strategies.wf_validator import RetrainController

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAIN_ID_BASE = 8453
STRATEGY_ID = "ml_quant_alpha_v2"

MIN_TRADE_USD = 5.0
MIN_TRADE_WEI = 10**15
CONGESTION_GWEI = 15.0

def _get_weth_usdc_pool() -> str:
    from wallet_manager.config import get_settings
    return get_settings().weth_usdc_pool


# ---------------------------------------------------------------------------
# Execution cycle result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionCycleResult:
    """Complete result of one ML-Quant execution cycle."""

    signal: AlphaMLSignal | None = None
    action_taken: str = "HOLD"
    position_usd: float = 0.0
    receipts: list[TxReceipt] = field(default_factory=list)
    sweep_receipts: list[TxReceipt] = field(default_factory=list)
    pnl_usd: float = 0.0
    dex_venue: str = ""
    dex_savings_bps: float = 0.0
    jitter_ms: float = 0.0
    cycle_duration_ms: float = 0.0
    risk_source: str = ""
    var_99: float = 0.0
    drawdown_state: str = "normal"
    mid_price_at_entry: float = 0.0
    execution_price: float = 0.0
    implementation_shortfall_bps: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_wallet_address(config) -> str:
    if config.address:
        return config.address
    from eth_account import Account
    return Account.from_key(config.private_key).address


# ---------------------------------------------------------------------------
# ML-Quant Execution Engine
# ---------------------------------------------------------------------------


class MLQuantExecutionEngine:
    """Institutional-grade ML-integrated DeFi execution engine.

    Combines GARCH/Kalman/XGBoost/LSTM alpha generation with PPO-based
    risk management and multi-DEX smart order routing.

    This is the "brain" of the system — it decides WHAT to trade,
    HOW MUCH to risk, and WHERE to execute.

    Usage:
        engine = MLQuantExecutionEngine()
        result = await engine.run_cycle(
            manager=wallet_manager,
            wallet_id="wallet_1",
            portfolio_manager=pm,
            task_context=ctx,
        )
    """

    # Minimum total trades before the first retrain is triggered
    _MIN_TRADES_BEFORE_RETRAIN: int = 200
    # Deduplicate: at most one retrain per 24 hours
    _RETRAIN_COOLDOWN_SECONDS: float = 86_400.0

    def __init__(self) -> None:
        self.alpha = get_alpha_engine()
        self.risk = get_deep_kelly_engine()
        self.sor = get_smart_order_router()
        self.mev = get_mev_shield()
        self.crash_detector = get_crash_detector()
        self.regime_kelly = RegimeAwareKelly()

        # Override crash threshold for ML engine
        self.crash_detector._threshold = 0.08
        self.crash_detector._window = 300.0

        # Retrain integration
        self._cycle_count: int = 0
        self._total_trade_count: int = 0
        self._last_retrain_trigger_ts: float = 0.0
        self._retrain_controller: RetrainController | None = None
        self._retrain_task: asyncio.Task[None] | None = None

        # Position tracking
        self._in_position = False
        self._entry_price = 0.0
        self._entry_mid_price = 0.0
        self._position_size_usd = 0.0
        self._entry_alpha = 0.0
        self._entry_garch_sigma = 0.0
        self._entry_vol_regime = "normal"
        self._entry_timestamp = 0.0

    async def run_cycle(
        self,
        manager: WalletManager,
        wallet_id: str,
        portfolio_manager: PortfolioManager,
        task_context: dict,
        analytics_db=None,
        quant_analytics=None,
    ) -> ExecutionCycleResult:
        """Run one complete ML-Quant execution cycle.

        Full Pipeline:
        1. Pre-flight (chain, gas, circuit breakers)
        2. Price fetch + ML signal generation
        3. Flash crash screening
        4. Deep Kelly risk sizing (PPO or Kelly+VaR)
        5. Smart order routing (multi-DEX)
        6. MEV-protected execution
        7. Post-trade analytics (TCA)
        8. Idle capital sweep (Zero-Lazy-Capital)
        """
        from wallet_manager.network.gas_monitor import GasMonitor

        self._cycle_count += 1
        cycle_start = time.time()
        result = ExecutionCycleResult()

        # ==============================================================
        # 1. PRE-FLIGHT CHECKS
        # ==============================================================

        command_state = task_context.get("command_state")
        if command_state and getattr(command_state, "is_circuit_breaker_halted", False) and command_state.is_circuit_breaker_halted:
            result.action_taken = "HALTED"
            return result

        if self.crash_detector.is_kill_switch_active:
            result.action_taken = "KILL_SWITCH"
            return result

        config = manager.get_config(wallet_id)
        chain_id = int(config.chain_id)
        if chain_id != CHAIN_ID_BASE:
            result.action_taken = "WRONG_CHAIN"
            return result

        rpc_url = config.rpc_url
        address = _get_wallet_address(config)

        gas_monitor = GasMonitor(rpc_url=rpc_url)
        max_gwei = min(25.0, float(config.max_gas_price_gwei))
        safe, current_gwei = await gas_monitor.is_safe(max_gwei)
        if not safe:
            result.action_taken = "GAS_HIGH"
            return result

        if current_gwei > CONGESTION_GWEI:
            result.action_taken = "CONGESTED"
            return result

        # ==============================================================
        # 2. PRICE FETCH + ML SIGNAL GENERATION
        # ==============================================================

        try:
            price = await actions.get_live_price_v3(rpc_url)
            if price <= 0:
                price = await actions.get_weth_usdc_price(rpc_url)
            if price <= 0:
                price = await actions.get_current_price(rpc_url, chain_id)
        except Exception as exc:
            result.error = str(exc)
            return result

        if price <= 0:
            result.error = "Invalid price"
            return result

        # Update microstructure from OFI
        try:
            from wallet_manager.strategies.signal_intelligence import get_alpha_generator
            ofi_gen = get_alpha_generator()
            ofi_reading = await ofi_gen.read_ofi(rpc_url, _get_weth_usdc_pool())
            if ofi_reading:
                vol_imb = ofi_reading.imbalance_ratio - 1.0  # Center around 0
                vol_imb = max(-1.0, min(1.0, vol_imb))
                self.alpha.set_microstructure(
                    volume_imbalance=vol_imb,
                    spread_bps=5.0,  # Estimate; would come from pool data
                )
        except Exception as _ms_exc:
            log.warning("Microstructure update failed (non-critical): %s", _ms_exc)

        # Generate ML alpha signal
        signal = self.alpha.process_tick(price, volume=0.0)
        result.signal = signal

        if signal is None:
            result.action_taken = "WARMUP"
            return result

        log.info(
            f"ML-EXEC: price=${price:.2f} alpha={signal.alpha_score:+.3f} "
            f"prob_up={signal.prob_up:.3f} garch={signal.garch_sigma:.6f} "
            f"regime={signal.volatility_regime} conf={signal.confidence:.2f}"
        )

        # ==============================================================
        # 3. FLASH CRASH SCREENING
        # ==============================================================

        crash_alert = self.crash_detector.ingest_price(price)
        if crash_alert:
            activate_kill_switch(task_context, crash_alert)
            await send_kill_switch_alert(task_context, crash_alert)
            result.action_taken = "FLASH_CRASH"
            return result

        # ==============================================================
        # 4. BALANCE + EQUITY UPDATE
        # ==============================================================

        try:
            balance_wei = await manager.get_safe_balance(wallet_id)
            balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        except Exception as exc:
            result.error = str(exc)
            return result

        eth_value_usd = actions.eth_balance_usd_value(balance_wei, price)
        usdc_value_usd = actions.usdc_balance_usd_value(balance_usdc)
        total_tvl = eth_value_usd + usdc_value_usd

        try:
            from wallet_manager.strategies.yield_strategy import get_aave_usdc_balance
            aave_balance = await get_aave_usdc_balance(rpc_url, address)
            total_tvl += aave_balance / 1e6
        except Exception as _aave_exc:
            log.warning("Aave TVL enrichment failed (non-critical): %s", _aave_exc)

        # Update drawdown tracker
        dd_state = self.risk.update_equity(total_tvl)
        result.drawdown_state = dd_state

        self._in_position = eth_value_usd >= MIN_TRADE_USD

        # ==============================================================
        # 5. DEEP KELLY RISK SIZING
        # ==============================================================

        risk_decision = self.risk.calculate_position(
            total_capital_usd=total_tvl,
            alpha_score=signal.alpha_score,
            garch_sigma=signal.garch_sigma,
            vol_regime=signal.volatility_regime,
            confidence=signal.confidence,
        )

        result.risk_source = risk_decision.source
        result.var_99 = risk_decision.var_99_pct

        if risk_decision.position_usd <= 0:
            result.action_taken = f"RISK_BLOCKED({risk_decision.drawdown_state})"
            return result

        # Apply regime-aware Kelly adjustment
        try:
            _rf = get_regime_filter()
            _regime_signal = _rf.get_current_regime()
            _regime_name = _regime_signal.regime.name
        except Exception:
            _regime_name = "LOW_VOL"

        _current_dd = getattr(self.risk.drawdown, "current_drawdown", 0.0)
        _regime_position = self.regime_kelly.compute_sized_position(
            base_kelly=risk_decision.kelly_fraction if hasattr(risk_decision, "kelly_fraction") else (risk_decision.position_usd / max(total_tvl, 1.0)),
            regime=_regime_name,
            current_drawdown=_current_dd,
            portfolio_value_usd=total_tvl,
            max_position_usd=risk_decision.position_usd,
        )
        # Use regime-adjusted position if it is more conservative
        _base_position = risk_decision.position_usd
        risk_adjusted_position = min(_base_position, _regime_position) if _regime_position > 0 else _regime_position

        trade_budget_usd = min(
            risk_adjusted_position if risk_adjusted_position > 0 else _base_position,
            portfolio_manager.get_algo_trade_budget(),
        )
        result.position_usd = trade_budget_usd

        # ==============================================================
        # 5b. REGIME GATE — hard block during high-risk regimes
        # ==============================================================
        try:
            from wallet_manager.strategies.regime_filter import (
                MarketRegime,
            )
            _rf2 = get_regime_filter()
            _regime_sig2 = _rf2.get_current_regime()
            _regime2 = _regime_sig2.regime

            if _regime2 == MarketRegime.NEWS_BLACKOUT:
                result.action_taken = "REGIME_BLACKOUT"
                log.info("ML-EXEC: NEWS_BLACKOUT — all position opening blocked.")
                return result

            if _regime2 == MarketRegime.HIGH_VOL:
                result.action_taken = "REGIME_HIGH_VOL"
                log.info("ML-EXEC: HIGH_VOL regime — position opening blocked, idle sweep only.")
                sweep_recs = await run_idle_sweep(
                    manager=manager, wallet_id=wallet_id,
                    task_context=task_context, analytics_db=analytics_db,
                )
                result.sweep_receipts = sweep_recs
                return result

        except Exception as _regime_exc:
            log.debug("Regime gate check failed: %s — proceeding without gate", _regime_exc)

        # ==============================================================
        # 5c. CONFIDENCE GATE — block low-confidence signals
        # ==============================================================
        try:
            from wallet_manager.config import get_settings as _cfg_val
            _thresh = _cfg_val().ml_confidence_threshold
            if signal.confidence < _thresh:
                log.debug(
                    "ML-EXEC: confidence=%.3f < threshold=%.2f — HOLD.",
                    signal.confidence, _thresh,
                )
                result.action_taken = "LOW_CONFIDENCE"
                try:
                    sweep_recs = await run_idle_sweep(
                        manager=manager, wallet_id=wallet_id,
                        task_context=task_context, analytics_db=analytics_db,
                    )
                    result.sweep_receipts = sweep_recs
                except Exception as _sweep_exc:
                    log.warning("Idle sweep on low-confidence path failed: %s", _sweep_exc)
                return result
        except Exception as _conf_exc:
            log.debug("Confidence gate check failed: %s", _conf_exc)

        # ==============================================================
        # 6. TRADING LOGIC
        # ==============================================================

        should_buy = (
            signal.should_enter_long
            and not self._in_position
            and usdc_value_usd >= MIN_TRADE_USD
            and trade_budget_usd >= MIN_TRADE_USD
        )

        should_sell = (
            self._in_position
            and signal.should_exit
            and eth_value_usd >= MIN_TRADE_USD
        )

        should_short = (
            signal.should_enter_short
            and not self._in_position
        )

        if should_buy:
            result = await self._execute_buy(
                manager, wallet_id, rpc_url, address, chain_id, price,
                signal, trade_budget_usd, balance_usdc, current_gwei,
                task_context, analytics_db, quant_analytics, result,
            )
        elif should_sell:
            result = await self._execute_sell(
                manager, wallet_id, rpc_url, address, chain_id, price,
                signal, balance_wei, current_gwei,
                task_context, analytics_db, quant_analytics, result,
            )
        elif should_short:
            try:
                from wallet_manager.strategies.algo_strategy import (
                    compute_bollinger_bands,
                    compute_rsi,
                )
                from wallet_manager.strategies.short_strategy import run_short_strategy
                prices = self.alpha.feature_engineer._prices
                p_list = list(prices)
                rsi = compute_rsi(p_list) if len(p_list) > 15 else 50.0
                _, _, bb_upper = compute_bollinger_bands(p_list) if len(p_list) > 20 else (0, 0, 0)
                short_recs = await run_short_strategy(
                    manager=manager, wallet_id=wallet_id,
                    portfolio_manager=portfolio_manager, task_context=task_context,
                    z_score=signal.alpha_score * 2.8,  # Map alpha to Z-Score equivalent
                    rsi=rsi, current_price=price, bb_upper=bb_upper,
                    analytics_db=analytics_db,
                    alpha_signal=signal,          # ← ML signal drives short entry/exit
                )
                if short_recs:
                    result.action_taken = "SHORT_OPEN"
                    result.receipts.extend(short_recs)
            except Exception as exc:
                log.warning(f"ML-EXEC: short strategy failed: {exc}")
        else:
            result.action_taken = "HOLD"

        # ==============================================================
        # 7. ZERO-LAZY-CAPITAL SWEEP
        # ==============================================================

        try:
            sweep_recs = await run_idle_sweep(
                manager=manager, wallet_id=wallet_id,
                task_context=task_context, analytics_db=analytics_db,
            )
            result.sweep_receipts = sweep_recs
        except Exception as _sweep2_exc:
            log.warning("Idle sweep after trade failed: %s", _sweep2_exc)

        # ==============================================================
        # 8. POST-TRADE ANALYTICS
        # ==============================================================

        if quant_analytics and result.signal:
            try:
                await quant_analytics.record_cycle(
                    price=price,
                    signal=result.signal,
                    action=result.action_taken,
                    position_usd=result.position_usd,
                    pnl_usd=result.pnl_usd,
                    dex_venue=result.dex_venue,
                    gas_gwei=current_gwei,
                    mid_price=result.mid_price_at_entry,
                    execution_price=result.execution_price,
                    implementation_shortfall_bps=result.implementation_shortfall_bps,
                )
            except Exception as exc:
                log.debug(f"ML-EXEC: analytics record failed: {exc}")

        result.cycle_duration_ms = (time.time() - cycle_start) * 1000

        log.info(
            f"ML-EXEC: action={result.action_taken} pos=${result.position_usd:.2f} "
            f"risk={result.risk_source} var={result.var_99:.3f} "
            f"dd={result.drawdown_state} pnl=${result.pnl_usd:+.2f} "
            f"dex={result.dex_venue or 'N/A'} {result.cycle_duration_ms:.0f}ms"
        )

        return result

    # ------------------------------------------------------------------
    # BUY execution
    # ------------------------------------------------------------------

    async def _execute_buy(
        self, manager, wallet_id, rpc_url, address, chain_id, price,
        signal, trade_budget_usd, balance_usdc, current_gwei,
        task_context, analytics_db, quant_analytics, result,
    ) -> ExecutionCycleResult:
        """Execute BUY (USDC → WETH) with ML risk + SOR + MEV Shield."""

        amount_usdc = min(int(trade_budget_usd * 1e6), balance_usdc)
        if amount_usdc < int(MIN_TRADE_USD * 1e6):
            return result

        addrs = actions.get_chain_addresses(chain_id)
        result.mid_price_at_entry = price

        # MEV jitter
        jitter_ms = await self.mev.apply_jitter()
        result.jitter_ms = jitter_ms

        # SOR: find best route
        route = await self.sor.find_best_route(
            rpc_url=rpc_url, token_in=addrs.usdc, token_out=addrs.weth,
            amount_in=amount_usdc, gas_price_gwei=current_gwei,
        )
        if route:
            result.dex_venue = route.best_quote.venue.value
            result.dex_savings_bps = route.savings_bps
            fee_tier = route.best_quote.fee_tier
        else:
            fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
            result.dex_venue = "uniswap_v3_fallback"

        slippage_bps = await actions.get_dynamic_slippage(
            rpc_url=rpc_url, chain_id=chain_id,
            token_in=addrs.usdc, token_out=addrs.weth,
            amount_in=amount_usdc, gas_price_gwei=current_gwei,
            fee=fee_tier, is_algo=True,
        )

        plan = await actions.build_usdc_to_eth_plan(
            rpc_url=rpc_url, chain_id=chain_id, from_address=address,
            amount_usdc=amount_usdc, slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei, fee=fee_tier,
        )

        for step in plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            result.receipts.append(receipt)
            if analytics_db:
                record = TxRecord(
                    tx_hash=receipt.tx_hash, wallet_id=wallet_id,
                    chain_id=chain_id, nonce=receipt.nonce_used,
                    gas_price_gwei=step.tx_params.gas_price_gwei,
                    gas_limit=step.tx_params.gas,
                    value_wei=step.tx_params.value, status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type=StrategyType.ALGO.value,
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # Track position state
        self._in_position = True
        self._entry_price = price
        self._entry_mid_price = price
        self._position_size_usd = amount_usdc / 1e6
        self._entry_alpha = signal.alpha_score
        self._entry_garch_sigma = signal.garch_sigma
        self._entry_vol_regime = signal.volatility_regime
        self._entry_timestamp = time.time()
        result.action_taken = "BUY"
        result.execution_price = price

        # Calculate implementation shortfall (simplified: mid vs execution)
        result.implementation_shortfall_bps = abs(price - result.mid_price_at_entry) / price * 10000

        _notify_ml_trade(task_context, "WETH/USDC", price,
                         f"${amount_usdc / 1e6:.2f} USDC", "BUY", signal, result)

        log.info(
            f"ML-EXEC: BUY ${amount_usdc / 1e6:.2f} USDC → WETH @ ${price:.2f} "
            f"(alpha={signal.alpha_score:+.3f} prob={signal.prob_up:.3f} "
            f"dex={result.dex_venue})"
        )
        return result

    # ------------------------------------------------------------------
    # SELL execution
    # ------------------------------------------------------------------

    async def _execute_sell(
        self, manager, wallet_id, rpc_url, address, chain_id, price,
        signal, balance_wei, current_gwei,
        task_context, analytics_db, quant_analytics, result,
    ) -> ExecutionCycleResult:
        """Execute SELL (WETH → USDC) with post-trade analytics."""

        # Exit signals should fully flatten the live position. A partial sell here
        # would leave residual exposure while the engine clears its internal
        # position state below, causing position-tracking drift.
        amount_wei = int(balance_wei)

        if amount_wei < MIN_TRADE_WEI:
            return result

        addrs = actions.get_chain_addresses(chain_id)
        result.mid_price_at_entry = price

        jitter_ms = await self.mev.apply_jitter()
        result.jitter_ms = jitter_ms

        route = await self.sor.find_best_route(
            rpc_url=rpc_url, token_in=addrs.weth, token_out=addrs.usdc,
            amount_in=amount_wei, gas_price_gwei=current_gwei,
        )
        if route:
            result.dex_venue = route.best_quote.venue.value
            result.dex_savings_bps = route.savings_bps
            fee_tier = route.best_quote.fee_tier
        else:
            fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
            result.dex_venue = "uniswap_v3_fallback"

        slippage_bps = await actions.get_dynamic_slippage(
            rpc_url=rpc_url, chain_id=chain_id,
            token_in=addrs.weth, token_out=addrs.usdc,
            amount_in=amount_wei, gas_price_gwei=current_gwei,
            fee=fee_tier, is_algo=True,
        )

        plan = await actions.build_eth_to_usdc_plan(
            rpc_url=rpc_url, chain_id=chain_id, from_address=address,
            amount_wei=amount_wei, slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei, fee=fee_tier,
        )

        for step in plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            result.receipts.append(receipt)
            if analytics_db:
                record = TxRecord(
                    tx_hash=receipt.tx_hash, wallet_id=wallet_id,
                    chain_id=chain_id, nonce=receipt.nonce_used,
                    gas_price_gwei=step.tx_params.gas_price_gwei,
                    gas_limit=step.tx_params.gas,
                    value_wei=step.tx_params.value, status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type=StrategyType.ALGO.value,
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # P&L + trade recording
        if self._entry_price > 0:
            pnl_pct = (price - self._entry_price) / self._entry_price
            pnl_usd = self._position_size_usd * pnl_pct
            result.pnl_usd = pnl_usd
            result.execution_price = price

            holding_periods = max(1, int(
                (time.time() - self._entry_timestamp) / 300  # 5-min periods
            ))

            self.risk.record_trade(TradeOutcome(
                pnl_usd=pnl_usd, pnl_pct=pnl_pct,
                position_pct=self._position_size_usd / max(1.0, self.risk.drawdown._peak_equity),
                entry_alpha=self._entry_alpha,
                entry_garch_sigma=self._entry_garch_sigma,
                entry_vol_regime=self._entry_vol_regime,
                holding_periods=holding_periods,
            ))

            result.implementation_shortfall_bps = abs(
                price - result.mid_price_at_entry
            ) / price * 10000

        # Capture entry price before clearing (used by trade journal below)
        _captured_entry_price = self._entry_price
        _captured_position_usd = self._position_size_usd
        _captured_pnl_pct = (price - self._entry_price) / self._entry_price if self._entry_price > 0 else 0.0

        self._in_position = False
        self._entry_price = 0.0
        self._position_size_usd = 0.0
        result.action_taken = "SELL"

        # Record to trade journal
        try:
            import datetime

            from wallet_manager.trade_journal import TradeRecord, get_trade_journal
            _dd_entry = getattr(self.risk.drawdown, "_last_drawdown", 0.0)
            _dd_exit = getattr(self.risk.drawdown, "current_drawdown", 0.0)
            _ml_preds: dict = {}
            if self.alpha.last_signal is not None:
                sig = self.alpha.last_signal
                _ml_preds = {
                    "xgb": getattr(sig, "xgb_prediction", 0.0),
                    "lgbm": getattr(sig, "lgbm_prediction", 0.0),
                    "lstm": getattr(sig, "lstm_prediction", 0.0),
                    "transformer": 0.0,
                    "ensemble": sig.alpha_score,
                }
            _exec_mode_val = "ml_quant"
            try:
                from wallet_manager.config import get_settings as _gs
                _exec_mode_val = _gs().execution_mode
            except Exception as _mode_exc:
                log.warning("Could not read execution_mode from settings: %s", _mode_exc)
            trade_rec = TradeRecord(
                timestamp_utc=datetime.datetime.utcnow().isoformat() + "Z",
                wallet_id=wallet_id,
                strategy_type="ALGO",
                action="SELL",
                entry_price=_captured_entry_price,
                exit_price=price,
                position_usd=_captured_position_usd,
                pnl_usd=result.pnl_usd,
                pnl_pct=_captured_pnl_pct,
                execution_mode=_exec_mode_val,
                regime=self._entry_vol_regime,
                alpha_score=signal.alpha_score,
                confidence=signal.confidence,
                garch_sigma=signal.garch_sigma,
                kelly_fraction_used=getattr(result, "kelly_fraction", 0.0),
                slippage_bps=result.implementation_shortfall_bps,
                gas_cost_usd=0.0,
                dex_venue=result.dex_venue or "",
                tx_hash="",
                drawdown_at_entry=_dd_entry,
                drawdown_at_exit=_dd_exit,
                ml_predictions=_ml_preds,
            )
            await get_trade_journal().record_trade_async(trade_rec)
        except Exception as _tj_exc:
            log.debug("TradeJournal: record skipped: %s", _tj_exc)

        _notify_ml_trade(task_context, "WETH/USDC", price,
                         f"{amount_wei / 1e18:.6f} WETH (P&L: ${result.pnl_usd:+.2f})",
                         "SELL", signal, result)

        log.info(
            f"ML-EXEC: SELL {amount_wei / 1e18:.6f} WETH @ ${price:.2f} "
            f"P&L=${result.pnl_usd:+.2f} IS={result.implementation_shortfall_bps:.1f}bps"
        )
        self._total_trade_count += 1
        if self._retrain_controller is not None:
            self._retrain_controller.record_trade()
        await self._check_and_trigger_retrain()
        return result

    def set_retrain_controller(self, controller: RetrainController) -> None:
        """Inject a RetrainController to trigger periodic model retraining."""
        self._retrain_controller = controller

    async def _check_and_trigger_retrain(self) -> None:
        """Check retrain conditions and fire _trigger_retrain() if appropriate.

        Rules (Section 8.2):
          - Only fires after every 100 trade cycles.
          - Requires at least MIN_TRADES_BEFORE_RETRAIN total trades.
          - Deduplicates: at most one retrain per RETRAIN_COOLDOWN_SECONDS (24h).
        """
        if self._retrain_controller is None:
            return
        if self._cycle_count % 100 != 0:
            return
        if self._total_trade_count < self._MIN_TRADES_BEFORE_RETRAIN:
            return
        now = time.time()
        if now - self._last_retrain_trigger_ts < self._RETRAIN_COOLDOWN_SECONDS:
            return
        if not self._retrain_controller.should_retrain():
            return
        self._last_retrain_trigger_ts = now
        log.info(
            "MLQuantExecutionEngine: triggering retrain after %d cycles / %d trades",
            self._cycle_count, self._total_trade_count,
        )
        self._retrain_task = asyncio.create_task(
            self._retrain_controller._trigger_retrain()
        )

    def get_engine_stats(self) -> dict:
        """Return comprehensive engine diagnostics."""
        return {
            "alpha": self.alpha.get_engine_stats(),
            "risk": self.risk.get_stats(),
            "mev": self.mev.get_stats(),
            "crash": self.crash_detector.get_buffer_stats(),
            "position": {
                "in_position": self._in_position,
                "entry_price": self._entry_price,
                "position_usd": self._position_size_usd,
            },
        }


# ---------------------------------------------------------------------------
# Notification helper
# ---------------------------------------------------------------------------


def _notify_ml_trade(
    task_context: dict, pair: str, price: float, amount: str,
    side: str, signal: AlphaMLSignal, result: ExecutionCycleResult,
) -> None:
    """Append ML-Quant trade notification for TelegramManager."""
    try:
        from wallet_manager.security.telegram_manager import TradeExecutedPayload
        pending = task_context.get("trade_notifications", [])
        pending.append(TradeExecutedPayload(
            pair=pair, price=price,
            strategy_type=(
                f"ML-QUANT [a={signal.alpha_score:+.2f} "
                f"p={signal.prob_up:.2f} "
                f"σ={signal.garch_sigma:.5f} "
                f"risk={result.risk_source} "
                f"dex={result.dex_venue}]"
            ),
            amount=amount, side=side,
            tx_hash=result.receipts[-1].tx_hash if result.receipts else None,
        ))
        task_context["trade_notifications"] = pending
    except Exception as _notif_exc:
        log.warning("Trade notification queueing failed (non-critical): %s", _notif_exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: MLQuantExecutionEngine | None = None


def get_ml_execution_engine() -> MLQuantExecutionEngine:
    """Return the process-wide MLQuantExecutionEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MLQuantExecutionEngine()
    return _engine_instance


def reset_ml_execution_engine() -> None:
    """Reset the singleton (for tests)."""
    global _engine_instance
    _engine_instance = None
