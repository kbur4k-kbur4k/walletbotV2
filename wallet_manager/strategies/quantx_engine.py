"""
wallet_manager/strategies/quantx_engine.py

QUANT-X Engine — Institutional Grade DeFi Quantitative Execution Engine.

Top-level coordinator that integrates all QUANT-X modules into a unified
trading loop. Replaces the existing algo_strategy cycle with a fully
statistical, microstructure-aware execution pipeline.

Execution Pipeline (per cycle)
------------------------------
1. SIGNAL: AlphaGenerator produces composite signal (Z-Score 2.0 + OFI + VWAP).
2. RISK: FlashCrashDetector screens for systemic risk (8% / 5min halt).
3. SIZE: KellyCriterion computes optimal position size.
4. ROUTE: SmartOrderRouter finds best DEX venue.
5. PROTECT: MEVShield applies jitter + routes through private RPC.
6. EXECUTE: Build and broadcast transaction via WalletManager.
7. SWEEP: Auto-sweep idle capital to Aave V3 (Zero-Lazy-Capital).
8. RECORD: Log trade result for Kelly history + analytics DB.

Module Dependencies
-------------------
- signal_intelligence.py: AlphaGenerator (Z-Score 2.0, OFI, VWAP)
- smart_order_router.py: SmartOrderRouter (Multi-DEX aggregation)
- mev_shield.py: MEVShield (Private RPC, Gaussian Jitter, Dynamic RBF)
- kelly_criterion.py: KellyCriterion (Position sizing)
- flash_crash_detector.py: FlashCrashDetector (8% / 5min circuit breaker)
- idle_sweep.py: run_idle_sweep (Zero-Lazy-Capital)
"""

# mypy: disable-error-code="no-untyped-def,no-any-return"

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wallet_manager import actions
from wallet_manager.core.models import TxReceipt, TxRecord, TxStatus
from wallet_manager.portfolio_manager import StrategyType
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.strategies.flash_crash_detector import (
    FlashCrashDetector,
    activate_kill_switch,
    send_kill_switch_alert,
)
from wallet_manager.strategies.flash_crash_detector import (
    get_detector as get_crash_detector,
)
from wallet_manager.strategies.idle_sweep import run_idle_sweep
from wallet_manager.strategies.kelly_criterion import (
    KellyCriterion,
    TradeResult,
    get_kelly_criterion,
)
from wallet_manager.strategies.mev_shield import (
    MEVShield,
    get_mev_shield,
)
from wallet_manager.strategies.signal_intelligence import (
    AlphaGenerator,
    SignalOutput,
    get_alpha_generator,
)
from wallet_manager.strategies.smart_order_router import (
    SmartOrderRouter,
    get_smart_order_router,
)

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from wallet_manager.portfolio_manager import PortfolioManager

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAIN_ID_BASE = 8453
STRATEGY_ID = "quantx_alpha_engine"

# Flash crash: 8% in 5 minutes (upgraded from 10% / 5min)
QUANTX_CRASH_THRESHOLD = 0.08
QUANTX_CRASH_WINDOW = 300.0

# Minimum trade sizes
MIN_TRADE_USD = 5.0
MIN_TRADE_WEI = 10**15  # ~0.001 ETH

# Congestion threshold
CONGESTION_GWEI = 15.0

# Uniswap V3 WETH/USDC pool on Base (for OFI reads)
def _get_weth_usdc_pool() -> str:
    from wallet_manager.config import get_settings
    return get_settings().weth_usdc_pool


# ---------------------------------------------------------------------------
# Cycle result
# ---------------------------------------------------------------------------


@dataclass
class QuantXCycleResult:
    """Result of a single QUANT-X execution cycle."""

    signal: SignalOutput | None = None
    action_taken: str = "HOLD"          # HOLD, BUY, SELL, SHORT_OPEN, SHORT_CLOSE
    position_usd: float = 0.0
    receipts: list[TxReceipt] = field(default_factory=list)
    sweep_receipts: list[TxReceipt] = field(default_factory=list)
    pnl_usd: float = 0.0
    dex_venue: str = ""
    dex_savings_bps: float = 0.0
    jitter_ms: float = 0.0
    cycle_duration_ms: float = 0.0
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
# QUANT-X Engine
# ---------------------------------------------------------------------------


class QuantXEngine:
    """Institutional Grade DeFi Quantitative Execution Engine.

    Integrates all QUANT-X modules into a single execution pipeline.
    Designed for Base Mainnet (L2) with $140 initial + $150/month DCA.

    Usage:
        engine = QuantXEngine()
        result = await engine.run_cycle(
            manager=wallet_manager,
            wallet_id="wallet_1",
            portfolio_manager=pm,
            task_context=ctx,
        )
    """

    def __init__(self) -> None:
        self.alpha: AlphaGenerator = get_alpha_generator()
        self.sor: SmartOrderRouter = get_smart_order_router()
        self.mev: MEVShield = get_mev_shield()
        self.kelly: KellyCriterion = get_kelly_criterion()
        self.crash_detector: FlashCrashDetector = get_crash_detector()

        # Override crash detector thresholds for QUANT-X
        self.crash_detector._threshold = QUANTX_CRASH_THRESHOLD
        self.crash_detector._window = QUANTX_CRASH_WINDOW

        # Position tracking
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._position_size_usd: float = 0.0

    async def run_cycle(
        self,
        manager: WalletManager,
        wallet_id: str,
        portfolio_manager: PortfolioManager,
        task_context: dict,
        analytics_db=None,
    ) -> QuantXCycleResult:
        """Run one complete QUANT-X execution cycle.

        Pipeline:
        1. Pre-flight checks (chain, gas, circuit breakers)
        2. Price fetch + signal generation
        3. Flash crash screening
        4. Position sizing via Kelly Criterion
        5. Smart order routing
        6. MEV-protected execution
        7. Idle capital sweep
        8. Trade recording

        Returns:
            QuantXCycleResult with all execution details.
        """
        from wallet_manager.network.gas_monitor import GasMonitor

        cycle_start = time.time()
        result = QuantXCycleResult()

        # ----------------------------------------------------------
        # 1. PRE-FLIGHT CHECKS
        # ----------------------------------------------------------

        # Circuit breaker halt check
        command_state = task_context.get("command_state")
        if command_state and getattr(command_state, "is_circuit_breaker_halted", False) and command_state.is_circuit_breaker_halted:
            log.info("QUANT-X: Circuit breaker halted — skip. /resume to override.")
            result.action_taken = "HALTED"
            return result

        # Kill switch check
        if self.crash_detector.is_kill_switch_active:
            log.info("QUANT-X: KILL_SWITCH active — all trading halted.")
            result.action_taken = "KILL_SWITCH"
            return result

        config = manager.get_config(wallet_id)
        chain_id = int(config.chain_id)
        if chain_id != CHAIN_ID_BASE:
            log.debug(f"QUANT-X: chain {chain_id} not Base — skip.")
            result.action_taken = "WRONG_CHAIN"
            return result

        rpc_url = config.rpc_url
        address = _get_wallet_address(config)

        # Gas safety
        gas_monitor = GasMonitor(rpc_url=rpc_url)
        max_gwei = min(25.0, float(config.max_gas_price_gwei))
        safe, current_gwei = await gas_monitor.is_safe(max_gwei)
        if not safe:
            log.info(f"QUANT-X: gas {current_gwei:.2f} Gwei > max — skip.")
            result.action_taken = "GAS_HIGH"
            return result

        if current_gwei > CONGESTION_GWEI:
            log.info(f"QUANT-X: congested mempool ({current_gwei:.2f} Gwei) — skip for MEV protection.")
            result.action_taken = "CONGESTED"
            return result

        # ----------------------------------------------------------
        # 2. PRICE FETCH + SIGNAL GENERATION
        # ----------------------------------------------------------

        try:
            price = await actions.get_live_price_v3(rpc_url)
            if price <= 0:
                price = await actions.get_weth_usdc_price(rpc_url)
            if price <= 0:
                price = await actions.get_current_price(rpc_url, chain_id)
        except Exception as exc:
            log.warning(f"QUANT-X: price fetch failed: {exc}")
            result.error = str(exc)
            return result

        if price <= 0:
            result.error = "Invalid price"
            return result

        # Ingest price into all signal engines
        self.alpha.ingest_price(price)

        # Estimate volume from price change (proxy)
        estimated_volume = abs(price) * 0.01  # Rough estimate
        self.alpha.ingest_trade(price, estimated_volume, is_buy=True)

        # Read OFI (order flow imbalance) from Uniswap V3 pool
        try:
            ofi_reading = await self.alpha.read_ofi(rpc_url, _get_weth_usdc_pool())
        except Exception as exc:
            log.debug(f"QUANT-X: OFI read failed: {exc}")
            ofi_reading = None

        # Generate composite signal
        signal = self.alpha.generate_signal(price, ofi_reading)
        result.signal = signal

        log.info(
            f"QUANT-X: price=${price:.2f} alpha={signal.alpha_score:+.3f} "
            f"z={signal.z_score:+.2f} ofi={signal.ofi_ratio:.2f} "
            f"vwap=${signal.vwap:.2f} conf={signal.confidence:.2f}"
        )

        # ----------------------------------------------------------
        # 3. FLASH CRASH SCREENING
        # ----------------------------------------------------------

        crash_alert = self.crash_detector.ingest_price(price)
        if crash_alert:
            log.critical(
                f"QUANT-X: FLASH CRASH DETECTED — {crash_alert.drop_pct:.1f}% drop!"
            )
            activate_kill_switch(task_context, crash_alert)
            await send_kill_switch_alert(task_context, crash_alert)
            result.action_taken = "FLASH_CRASH"
            return result

        # ----------------------------------------------------------
        # 4. BALANCE CHECK + POSITION STATE
        # ----------------------------------------------------------

        try:
            balance_wei = await manager.get_safe_balance(wallet_id)
            balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        except Exception as exc:
            log.warning(f"QUANT-X: balance fetch failed: {exc}")
            result.error = str(exc)
            return result

        eth_value_usd = actions.eth_balance_usd_value(balance_wei, price)
        usdc_value_usd = actions.usdc_balance_usd_value(balance_usdc)
        total_tvl = eth_value_usd + usdc_value_usd

        # Aave supplied capital
        try:
            from wallet_manager.strategies.yield_strategy import get_aave_usdc_balance
            aave_balance = await get_aave_usdc_balance(rpc_url, address)
            aave_value_usd = aave_balance / 1e6
            total_tvl += aave_value_usd
        except Exception:
            aave_value_usd = 0.0

        # Update position state
        self._in_position = eth_value_usd >= MIN_TRADE_USD

        # ----------------------------------------------------------
        # 5. KELLY CRITERION POSITION SIZING
        # ----------------------------------------------------------

        sizing = self.kelly.calculate_position_size(
            total_capital_usd=total_tvl,
            z_score=signal.z_score,
            strategy="QUANTX",
            alpha_score=signal.alpha_score,
        )

        trade_budget_usd = min(
            sizing.position_usd,
            portfolio_manager.get_algo_trade_budget(),
        )
        result.position_usd = trade_budget_usd

        # ----------------------------------------------------------
        # 6. TRADING LOGIC
        # ----------------------------------------------------------

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

        # Check for short signal (delta-neutral hedging via existing short_strategy)
        should_short = (
            signal.should_enter_short
            and not self._in_position
        )

        if should_buy:
            result = await self._execute_buy(
                manager=manager,
                wallet_id=wallet_id,
                rpc_url=rpc_url,
                address=address,
                chain_id=chain_id,
                price=price,
                signal=signal,
                trade_budget_usd=trade_budget_usd,
                balance_usdc=balance_usdc,
                current_gwei=current_gwei,
                task_context=task_context,
                analytics_db=analytics_db,
                result=result,
            )

        elif should_sell:
            result = await self._execute_sell(
                manager=manager,
                wallet_id=wallet_id,
                rpc_url=rpc_url,
                address=address,
                chain_id=chain_id,
                price=price,
                signal=signal,
                balance_wei=balance_wei,
                current_gwei=current_gwei,
                task_context=task_context,
                analytics_db=analytics_db,
                result=result,
            )

        elif should_short:
            # Delegate to existing short_strategy for delta-neutral hedging
            try:
                from wallet_manager.strategies.algo_strategy import (
                    compute_bollinger_bands,
                    compute_rsi,
                )
                from wallet_manager.strategies.short_strategy import run_short_strategy

                prices = self.alpha.zscore_engine.get_prices()
                rsi = compute_rsi(prices) if len(prices) > 15 else 50.0
                _, _, bb_upper = compute_bollinger_bands(prices) if len(prices) > 20 else (0, 0, 0)

                short_receipts = await run_short_strategy(
                    manager=manager,
                    wallet_id=wallet_id,
                    portfolio_manager=portfolio_manager,
                    task_context=task_context,
                    z_score=signal.z_score,
                    rsi=rsi,
                    current_price=price,
                    bb_upper=bb_upper,
                    analytics_db=analytics_db,
                )
                if short_receipts:
                    result.action_taken = "SHORT_OPEN"
                    result.receipts.extend(short_receipts)
            except Exception as exc:
                log.warning(f"QUANT-X: short strategy failed: {exc}")

        else:
            result.action_taken = "HOLD"
            log.debug(
                f"QUANT-X: HOLD — alpha={signal.alpha_score:+.3f}, "
                f"z={signal.z_score:+.2f}, in_pos={self._in_position}"
            )

        # ----------------------------------------------------------
        # 7. ZERO-LAZY-CAPITAL: Auto-sweep idle USDC to Aave
        # ----------------------------------------------------------

        try:
            sweep_receipts = await run_idle_sweep(
                manager=manager,
                wallet_id=wallet_id,
                task_context=task_context,
                analytics_db=analytics_db,
            )
            result.sweep_receipts = sweep_receipts
        except Exception as exc:
            log.debug(f"QUANT-X: idle sweep failed: {exc}")

        # ----------------------------------------------------------
        # 8. CYCLE COMPLETE
        # ----------------------------------------------------------

        result.cycle_duration_ms = (time.time() - cycle_start) * 1000

        log.info(
            f"QUANT-X: cycle complete — action={result.action_taken} "
            f"position=${result.position_usd:.2f} "
            f"dex={result.dex_venue or 'N/A'} "
            f"jitter={result.jitter_ms:.0f}ms "
            f"duration={result.cycle_duration_ms:.0f}ms"
        )

        return result

    # ------------------------------------------------------------------
    # BUY execution
    # ------------------------------------------------------------------

    async def _execute_buy(
        self,
        manager: WalletManager,
        wallet_id: str,
        rpc_url: str,
        address: str,
        chain_id: int,
        price: float,
        signal: SignalOutput,
        trade_budget_usd: float,
        balance_usdc: int,
        current_gwei: float,
        task_context: dict,
        analytics_db,
        result: QuantXCycleResult,
    ) -> QuantXCycleResult:
        """Execute a BUY (USDC → WETH) using SOR + MEV Shield."""

        amount_usdc = min(int(trade_budget_usd * 1e6), balance_usdc)
        if amount_usdc < int(MIN_TRADE_USD * 1e6):
            return result

        addrs = actions.get_chain_addresses(chain_id)

        # MEV: Apply Gaussian jitter before trade
        jitter_ms = await self.mev.apply_jitter()
        result.jitter_ms = jitter_ms

        # SOR: Find best route across DEXes
        route = await self.sor.find_best_route(
            rpc_url=rpc_url,
            token_in=addrs.usdc,
            token_out=addrs.weth,
            amount_in=amount_usdc,
            gas_price_gwei=current_gwei,
        )

        if route:
            result.dex_venue = route.best_quote.venue.value
            result.dex_savings_bps = route.savings_bps
            fee_tier = route.best_quote.fee_tier
        else:
            # Fallback to Uniswap V3 default
            fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
            result.dex_venue = "uniswap_v3_fallback"

        # Get dynamic slippage
        slippage_bps = await actions.get_dynamic_slippage(
            rpc_url=rpc_url,
            chain_id=chain_id,
            token_in=addrs.usdc,
            token_out=addrs.weth,
            amount_in=amount_usdc,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
            is_algo=True,
        )

        # Build swap plan
        plan = await actions.build_usdc_to_eth_plan(
            rpc_url=rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_usdc=amount_usdc,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )

        # Execute
        for step in plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            result.receipts.append(receipt)
            if analytics_db:
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
                    strategy_type=StrategyType.ALGO.value,
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        self._in_position = True
        self._entry_price = price
        self._position_size_usd = amount_usdc / 1e6
        result.action_taken = "BUY"

        # Notification
        _notify_trade(
            task_context,
            pair="WETH/USDC",
            price=price,
            amount=f"${amount_usdc / 1e6:.2f} USDC",
            side="BUY",
            strategy="QUANT-X",
            alpha=signal.alpha_score,
            z_score=signal.z_score,
            dex=result.dex_venue,
            tx_hash=result.receipts[-1].tx_hash if result.receipts else None,
        )

        log.info(
            f"QUANT-X: BUY ${amount_usdc / 1e6:.2f} USDC → WETH "
            f"@ ${price:.2f} (alpha={signal.alpha_score:+.3f}, "
            f"z={signal.z_score:+.2f}, dex={result.dex_venue})"
        )

        return result

    # ------------------------------------------------------------------
    # SELL execution
    # ------------------------------------------------------------------

    async def _execute_sell(
        self,
        manager: WalletManager,
        wallet_id: str,
        rpc_url: str,
        address: str,
        chain_id: int,
        price: float,
        signal: SignalOutput,
        balance_wei: int,
        current_gwei: float,
        task_context: dict,
        analytics_db,
        result: QuantXCycleResult,
    ) -> QuantXCycleResult:
        """Execute a SELL (WETH → USDC) using SOR + MEV Shield."""

        # Sell Kelly-recommended fraction of ETH holdings
        kelly_pct = self.kelly.calculate_position_size(
            total_capital_usd=self._position_size_usd,
            z_score=signal.z_score,
            strategy="QUANTX",
        ).position_pct
        sell_fraction = max(kelly_pct, 0.5)  # At least sell half on exit
        amount_wei = int(balance_wei * sell_fraction)

        if amount_wei < MIN_TRADE_WEI:
            return result

        addrs = actions.get_chain_addresses(chain_id)

        # MEV: Apply Gaussian jitter
        jitter_ms = await self.mev.apply_jitter()
        result.jitter_ms = jitter_ms

        # SOR: Find best route
        route = await self.sor.find_best_route(
            rpc_url=rpc_url,
            token_in=addrs.weth,
            token_out=addrs.usdc,
            amount_in=amount_wei,
            gas_price_gwei=current_gwei,
        )

        if route:
            result.dex_venue = route.best_quote.venue.value
            result.dex_savings_bps = route.savings_bps
            fee_tier = route.best_quote.fee_tier
        else:
            fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
            result.dex_venue = "uniswap_v3_fallback"

        slippage_bps = await actions.get_dynamic_slippage(
            rpc_url=rpc_url,
            chain_id=chain_id,
            token_in=addrs.weth,
            token_out=addrs.usdc,
            amount_in=amount_wei,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
            is_algo=True,
        )

        plan = await actions.build_eth_to_usdc_plan(
            rpc_url=rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_wei=amount_wei,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )

        for step in plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            result.receipts.append(receipt)
            if analytics_db:
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
                    strategy_type=StrategyType.ALGO.value,
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # Calculate P&L
        if self._entry_price > 0:
            pnl_pct = (price - self._entry_price) / self._entry_price
            pnl_usd = self._position_size_usd * pnl_pct
            result.pnl_usd = pnl_usd

            # Record trade result for Kelly
            self.kelly.record_trade(
                TradeResult(
                    pnl_usd=pnl_usd,
                    entry_price=self._entry_price,
                    exit_price=price,
                    position_usd=self._position_size_usd,
                    z_score_at_entry=signal.z_score,
                    strategy="QUANTX",
                )
            )

        self._in_position = False
        self._entry_price = 0.0
        self._position_size_usd = 0.0
        result.action_taken = "SELL"

        _notify_trade(
            task_context,
            pair="WETH/USDC",
            price=price,
            amount=f"{amount_wei / 1e18:.6f} WETH (P&L: ${result.pnl_usd:+.2f})",
            side="SELL",
            strategy="QUANT-X",
            alpha=signal.alpha_score,
            z_score=signal.z_score,
            dex=result.dex_venue,
            tx_hash=result.receipts[-1].tx_hash if result.receipts else None,
        )

        log.info(
            f"QUANT-X: SELL {amount_wei / 1e18:.6f} WETH → USDC "
            f"@ ${price:.2f} (P&L: ${result.pnl_usd:+.2f}, "
            f"alpha={signal.alpha_score:+.3f}, dex={result.dex_venue})"
        )

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_engine_stats(self) -> dict:
        """Return comprehensive QUANT-X engine statistics."""
        return {
            "signal": {
                "z_buffer_size": self.alpha.zscore_engine.buffer_size,
                "z_data_sufficiency": self.alpha.zscore_engine.data_sufficiency,
                "vwap_trades": self.alpha.vwap_calculator.trade_count,
                "last_ofi": (
                    self.alpha.ofi_analyser.last_reading.imbalance_ratio
                    if self.alpha.ofi_analyser.last_reading
                    else None
                ),
            },
            "kelly": self.kelly.get_stats(),
            "mev": self.mev.get_stats(),
            "crash_detector": self.crash_detector.get_buffer_stats(),
            "position": {
                "in_position": self._in_position,
                "entry_price": self._entry_price,
                "position_size_usd": self._position_size_usd,
            },
        }


# ---------------------------------------------------------------------------
# Notification helper
# ---------------------------------------------------------------------------


def _notify_trade(
    task_context: dict,
    pair: str,
    price: float,
    amount: str,
    side: str,
    strategy: str,
    alpha: float,
    z_score: float,
    dex: str,
    tx_hash: str | None = None,
) -> None:
    """Append QUANT-X trade notification for TelegramManager."""
    try:
        from wallet_manager.security.telegram_manager import TradeExecutedPayload

        pending = task_context.get("trade_notifications", [])
        pending.append(
            TradeExecutedPayload(
                pair=pair,
                price=price,
                strategy_type=f"{strategy} [a={alpha:+.2f} z={z_score:+.1f} dex={dex}]",
                amount=amount,
                side=side,
                tx_hash=tx_hash,
            )
        )
        task_context["trade_notifications"] = pending
    except Exception as _notif_exc:
        log.warning("Trade notification queueing failed (non-critical): %s", _notif_exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: QuantXEngine | None = None


def get_quantx_engine() -> QuantXEngine:
    """Return the process-wide QuantXEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = QuantXEngine()
    return _engine_instance


def reset_quantx_engine() -> None:
    """Reset the singleton (for tests)."""
    global _engine_instance
    _engine_instance = None
