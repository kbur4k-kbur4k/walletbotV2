"""
wallet_manager/strategies/algo_strategy.py

Alpha Algo Engine — Volatility-Adjusted Mean Reversion on WETH/USDC (Base).

Statistical Superiority:
  - Z-Score computed using 20-period EMA (gives more weight to recent price action)
  - Volatility = EMA of squared deviations for adaptive thresholds

Extreme Edge Entry:
  - BUY only if Z-Score < -2.5 AND RSI(14) < 30 (filters "falling knife" scenarios)

Smart Exit (Trailing Take-Profit):
  - Once price hits Mean (SMA), don't sell immediately
  - Activate 0.5% trailing offset to capture extended rallies
  - Exit when price drops 0.5% from the peak since mean-cross

Risk Management:
  - Hard Stop-Loss at 3%
  - Daily Drawdown Circuit Breaker: if portfolio loses 5% in a day, kill all trades

Execution: Uniswap V3 swaps via actions. GasMonitor gates execution when congested.
"""

# mypy: disable-error-code="no-untyped-def,no-any-return"

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wallet_manager import actions
from wallet_manager.core.models import TxReceipt, TxRecord, TxStatus
from wallet_manager.portfolio_manager import StrategyType
from wallet_manager.resilience.circuit_breaker import CircuitBreaker
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from wallet_manager.portfolio_manager import PortfolioManager

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------------------------
EMA_PERIOD = 20
RSI_PERIOD = 14
Z_SCORE_BUY_THRESHOLD = -2.5
RSI_OVERSOLD_THRESHOLD = 30
STOP_LOSS_PCT = 0.03  # 3% hard stop
TRADE_BUDGET_PCT = 0.20
TRAILING_OFFSET_PCT = 0.005  # 0.5% trailing take-profit
DAILY_DRAWDOWN_LIMIT_PCT = 0.05  # 5% circuit breaker
CHAIN_ID_BASE = actions.CHAIN_ID_BASE
_MIN_TRADE_USD = 5.0

STRATEGY_ID = "algo_vol_mean_reversion"

# MEV protection: strict 0.5% slippage for algo swaps
ALGO_SLIPPAGE_BPS = 50  # 0.5%

def _default_history_path() -> Path:
    from wallet_manager.config import get_settings
    p = str(get_settings().algo_price_history_path)
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent / "algo_price_history.json"


_ALGO_CIRCUIT = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

# Daily drawdown tracking: (day_start_value, day_start_ts)
_DAILY_DRAWDOWN_STATE: dict[str, tuple[float, float]] = {}

# Circuit breaker halt duration (24h)
_CIRCUIT_BREAKER_HALT_SECONDS = 86400.0


@dataclass
class AlgoState:
    """Persistent state for the algo strategy."""

    prices: list[float] = field(default_factory=list)
    last_trade_ts: float = 0.0
    last_side: str = ""
    last_buy_price: float = 0.0
    max_history: int = 60  # Need enough for EMA(20) + RSI(14)
    # Trailing Take-Profit: peak price since mean-cross, and whether TTP is active
    ttp_active: bool = False
    ttp_peak_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prices": self.prices,
            "last_trade_ts": self.last_trade_ts,
            "last_side": self.last_side,
            "last_buy_price": self.last_buy_price,
            "ttp_active": self.ttp_active,
            "ttp_peak_price": self.ttp_peak_price,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AlgoState:
        return cls(
            prices=d.get("prices", [])[-cls.max_history:],
            last_trade_ts=float(d.get("last_trade_ts", 0)),
            last_side=str(d.get("last_side", "")),
            last_buy_price=float(d.get("last_buy_price", 0)),
            ttp_active=bool(d.get("ttp_active", False)),
            ttp_peak_price=float(d.get("ttp_peak_price", 0)),
        )


def _load_state(path: Path) -> AlgoState:
    if not path.exists():
        return AlgoState()
    try:
        with open(path, encoding="utf-8") as f:
            return AlgoState.from_dict(json.load(f))
    except Exception as exc:
        log.warning(f"AlgoState load failed: {exc}. Starting fresh.")
        return AlgoState()


def _save_state(path: Path, state: AlgoState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
    except OSError as exc:
        log.warning(f"AlgoState save failed: {exc}")


# ---------------------------------------------------------------------------
# EMA-based Z-Score (volatility-adjusted)
# ---------------------------------------------------------------------------
def _ema(values: np.ndarray, period: int) -> float:
    """Exponential Moving Average of last `period` values."""
    if len(values) < period:
        return float(np.mean(values)) if len(values) > 0 else 0.0
    arr = values[-period:].astype(np.float64)
    alpha = 2.0 / (period + 1)
    ema_val = arr[0]
    for i in range(1, len(arr)):
        ema_val = alpha * arr[i] + (1 - alpha) * ema_val
    return float(ema_val)


def _ema_std(values: np.ndarray, period: int, mean: float) -> float:
    """EMA of squared deviations for volatility (standard deviation proxy)."""
    if len(values) < period:
        return float(np.std(values)) if len(values) > 1 else 1e-10
    arr = values[-period:].astype(np.float64)
    sq_dev = (arr - mean) ** 2
    alpha = 2.0 / (period + 1)
    ema_sq = sq_dev[0]
    for i in range(1, len(sq_dev)):
        ema_sq = alpha * sq_dev[i] + (1 - alpha) * ema_sq
    std = np.sqrt(ema_sq)
    return float(std) if std > 0 else 1e-10


def compute_z_score_ema(
    prices: list[float],
    period: int = EMA_PERIOD,
) -> tuple[float, float, float]:
    """
    Compute Z-Score using 20-period EMA for mean and volatility.
    Returns (ema_mean, ema_volatility, z_score).
    """
    if len(prices) < period:
        return 0.0, 0.0, 0.0
    arr = np.array(prices, dtype=np.float64)
    ema_mean = _ema(arr, period)
    vol = _ema_std(arr, period, ema_mean)
    current = float(prices[-1])
    z = (current - ema_mean) / vol if vol > 0 else 0.0
    return ema_mean, vol, float(z)


def compute_sma(prices: list[float], period: int = EMA_PERIOD) -> float:
    """Simple Moving Average for exit mean reference."""
    if len(prices) < period:
        return 0.0
    return float(np.mean(prices[-period:]))


# ---------------------------------------------------------------------------
# Bollinger Bands (Mean ± 2σ)
# ---------------------------------------------------------------------------
def compute_bollinger_bands(
    prices: list[float],
    period: int = EMA_PERIOD,
    num_std: float = 2.0,
) -> tuple[float, float, float]:
    """Compute Bollinger Bands: (lower, middle, upper).

    Uses SMA as the middle band with ±num_std standard deviations.
    Bollinger Band proximity is used as a confluence filter:
    - Buy signal strengthened when price is near/below lower band.
    - Short signal strengthened when price is near/above upper band.

    Args:
        prices: Price history.
        period: Lookback period for SMA and std.
        num_std: Number of standard deviations for band width.

    Returns:
        (lower_band, middle_band, upper_band). All 0.0 if insufficient data.
    """
    if len(prices) < period:
        return 0.0, 0.0, 0.0
    arr = np.array(prices[-period:], dtype=np.float64)
    middle = float(np.mean(arr))
    std = float(np.std(arr))
    lower = middle - num_std * std
    upper = middle + num_std * std
    return lower, middle, upper


def bollinger_band_proximity(
    price: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Compute proximity to Bollinger Bands as a fraction.

    Returns:
        (lower_proximity, upper_proximity).
        lower_proximity < 0 means price is BELOW lower band (strong buy).
        upper_proximity < 0 means price is ABOVE upper band (strong short).
    """
    band_width = upper - lower if upper > lower else 1e-10
    lower_prox = (price - lower) / band_width
    upper_prox = (upper - price) / band_width
    return lower_prox, upper_prox


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------
def compute_rsi(prices: list[float], period: int = RSI_PERIOD) -> float:
    """
    Compute RSI(14). Returns 0-100.
    RSI < 30 = oversold (bullish signal when combined with Z-Score).
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral when insufficient data
    arr = np.array(prices[-period - 1 :], dtype=np.float64)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def _get_wallet_address(config) -> str:
    if config.address:
        return config.address
    from eth_account import Account
    return Account.from_key(config.private_key).address


async def _check_daily_drawdown_circuit_breaker(
    wallet_id: str,
    current_tvl: float,
    task_context: dict,
) -> bool:
    """
    Returns True if circuit breaker is TRIGGERED (must halt ALL trades for 24h).
    Daily drawdown limit: 5%. Triggers 24h halt; /resume overrides.
    """
    global _DAILY_DRAWDOWN_STATE
    now = time.time()
    day_start_ts = (int(now) // 86400) * 86400  # Midnight UTC
    key = f"{wallet_id}_algo"

    if key not in _DAILY_DRAWDOWN_STATE:
        _DAILY_DRAWDOWN_STATE[key] = (current_tvl, day_start_ts)
        return False

    start_val, start_day = _DAILY_DRAWDOWN_STATE[key]
    if start_day != day_start_ts:
        # New day: reset
        _DAILY_DRAWDOWN_STATE[key] = (current_tvl, day_start_ts)
        return False

    if start_val <= 0:
        return False

    drawdown_pct = (start_val - current_tvl) / start_val
    if drawdown_pct >= DAILY_DRAWDOWN_LIMIT_PCT:
        # Circuit breaker: halt ALL trading for 24h
        command_state = task_context.get("command_state")
        if command_state and hasattr(command_state, "trigger_circuit_breaker_halt"):
            command_state.trigger_circuit_breaker_halt(_CIRCUIT_BREAKER_HALT_SECONDS)
        try:
            from wallet_manager.security.telegram_manager import (
                CircuitBreakerPayload,
            )
            mgr = task_context.get("telegram_manager")
            if mgr and mgr._is_configured():
                await mgr.send_circuit_breaker_alert(
                    CircuitBreakerPayload(
                        alert_type="DAILY_DRAWDOWN",
                        detail=f"Portfolio lost {drawdown_pct*100:.2f}% today (limit {DAILY_DRAWDOWN_LIMIT_PCT*100:.0f}%). All trading halted for 24h.",
                        severity="critical",
                        start_value_usd=start_val,
                        current_value_usd=current_tvl,
                        drawdown_pct=drawdown_pct * 100,
                    )
                )
        except Exception as exc:
            log.warning(f"Circuit breaker alert failed: {exc}")
        log.error(
            f"ALGO Circuit Breaker: Daily drawdown {drawdown_pct*100:.2f}% >= {DAILY_DRAWDOWN_LIMIT_PCT*100:.0f}%. Halting all trading for 24h."
        )
        return True
    return False


async def run_algo_strategy(
    manager: WalletManager,
    wallet_id: str,
    portfolio_manager: PortfolioManager,
    task_context: dict,
    analytics_db=None,
) -> list[TxReceipt]:
    """
    Run one cycle of the Alpha volatility-adjusted mean-reversion strategy.

    Entry: Z-Score < -2.5 AND RSI(14) < 30.
    Exit: Trailing Take-Profit (0.5% below peak after mean-cross) OR 3% stop-loss.
    Circuit Breaker: 5% daily drawdown halts all trades.
    """
    from wallet_manager.network.gas_monitor import GasMonitor

    # Circuit breaker: skip if 24h halt active (daily drawdown or stop-loss)
    command_state = task_context.get("command_state")
    if command_state and getattr(command_state, "is_circuit_breaker_halted", False) and command_state.is_circuit_breaker_halted:
        log.info("ALGO: Circuit breaker halted — skip. Send /resume to override.")
        return []

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id)
    if chain_id != CHAIN_ID_BASE:
        log.info(f"ALGO: chain {chain_id} not Base — skip.")
        return []

    rpc_url = config.rpc_url
    address = _get_wallet_address(config)

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        log.info(f"ALGO: gas {current_gwei:.2f} Gwei > max — skip (MEV protection: avoid congested blocks).")
        return []

    # Congestion check: if gas is high, treat as congested — skip to avoid MEV
    CONGESTION_GWEI = 15.0
    if current_gwei > CONGESTION_GWEI:
        log.info(f"ALGO: gas {current_gwei:.2f} Gwei indicates congested mempool — skip for MEV protection.")
        return []

    if not _ALGO_CIRCUIT.allow_request():
        log.warning("ALGO: circuit OPEN — skipping this cycle.")
        return []

    try:
        price = await actions.get_live_price_v3(rpc_url)
        if price <= 0:
            price = await actions.get_weth_usdc_price(rpc_url)
        if price <= 0:
            price = await actions.get_current_price(rpc_url, chain_id)
    except Exception as exc:
        _ALGO_CIRCUIT.record_failure()
        log.warning(f"ALGO: price fetch failed: {exc}")
        raise
    if price <= 0:
        return []

    state = _load_state(_default_history_path())
    state.prices.append(price)
    if len(state.prices) > state.max_history:
        state.prices = state.prices[-state.max_history:]
    _save_state(_default_history_path(), state)

    ema_mean, _, z = compute_z_score_ema(state.prices, EMA_PERIOD)
    sma = compute_sma(state.prices, EMA_PERIOD)
    rsi = compute_rsi(state.prices, RSI_PERIOD)
    bb_lower, bb_middle, bb_upper = compute_bollinger_bands(state.prices, EMA_PERIOD)
    bb_lower_prox, bb_upper_prox = bollinger_band_proximity(price, bb_lower, bb_upper) if bb_upper > 0 else (0.5, 0.5)

    log.debug(
        f"ALGO: price={price:.2f}, EMA={ema_mean:.2f}, SMA={sma:.2f}, "
        f"z={z:.2f}, RSI={rsi:.1f}, BB=[{bb_lower:.2f}, {bb_upper:.2f}], "
        f"BB_prox_lower={bb_lower_prox:.3f}, BB_prox_upper={bb_upper_prox:.3f}."
    )

    if len(state.prices) < EMA_PERIOD:
        log.info(f"ALGO: need {EMA_PERIOD - len(state.prices)} more prices.")
        return []

    try:
        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
    except Exception as exc:
        _ALGO_CIRCUIT.record_failure()
        log.warning(f"ALGO: balance fetch failed: {exc}")
        raise
    _ALGO_CIRCUIT.record_success()

    eth_price = price
    eth_value_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
    usdc_value_usd = actions.usdc_balance_usd_value(balance_usdc)
    total_tvl = eth_value_usd + usdc_value_usd

    # Daily Drawdown Circuit Breaker
    if await _check_daily_drawdown_circuit_breaker(wallet_id, total_tvl, task_context):
        return []

    trade_budget_usd = portfolio_manager.get_algo_trade_budget()
    if trade_budget_usd < _MIN_TRADE_USD:
        log.info(f"ALGO: trade budget ${trade_budget_usd:.2f} < min.")
        return []

    in_position = eth_value_usd >= _MIN_TRADE_USD

    # Extreme Edge Entry: Z-Score < -2.5 AND RSI < 30 AND near lower Bollinger Band
    # Bollinger Band proximity adds confluence: buy only when price is within
    # the lower 20% of the band (bb_lower_prox < 0.20) or below the lower band.
    should_buy = (
        z < Z_SCORE_BUY_THRESHOLD
        and rsi < RSI_OVERSOLD_THRESHOLD
        and bb_lower_prox < 0.20  # Near or below lower Bollinger Band
        and usdc_value_usd >= _MIN_TRADE_USD
    )

    # Hard Stop-Loss at 3%
    stop_loss_triggered = False
    if in_position and state.last_buy_price > 0:
        pct_drop = (state.last_buy_price - price) / state.last_buy_price
        stop_loss_triggered = pct_drop >= STOP_LOSS_PCT

    # Trailing Take-Profit logic
    if in_position and price >= sma and not stop_loss_triggered:
        if not state.ttp_active:
            state.ttp_active = True
            state.ttp_peak_price = price
        else:
            state.ttp_peak_price = max(state.ttp_peak_price, price)
        _save_state(_default_history_path(), state)

    ttp_triggered = False
    if state.ttp_active and state.ttp_peak_price > 0:
        drop_from_peak = (state.ttp_peak_price - price) / state.ttp_peak_price
        if drop_from_peak >= TRAILING_OFFSET_PCT:
            ttp_triggered = True

    # Sell only on: (1) 3% stop-loss, or (2) 0.5% trailing take-profit after mean-cross
    should_sell = in_position and (stop_loss_triggered or ttp_triggered)

    # Stop-loss: trigger 24h circuit breaker and send high-priority alert
    if stop_loss_triggered and command_state:
        if hasattr(command_state, "trigger_circuit_breaker_halt"):
            command_state.trigger_circuit_breaker_halt(_CIRCUIT_BREAKER_HALT_SECONDS)
        try:
            from wallet_manager.security.telegram_manager import (
                StopLossPayload,
            )
            mgr = task_context.get("telegram_manager")
            if mgr and mgr._is_configured():
                pct_loss = (state.last_buy_price - price) / state.last_buy_price * 100
                await mgr.send_stop_loss_alert(
                    StopLossPayload(
                        wallet_id=wallet_id,
                        pair="WETH/USDC",
                        price=price,
                        last_buy_price=state.last_buy_price,
                        pct_loss=pct_loss,
                    )
                )
        except Exception as exc:
            log.warning(f"Stop-loss alert failed: {exc}")

    receipts: list[TxReceipt] = []

    if should_buy and not should_sell:
        amount_usdc = min(int(trade_budget_usd * 1e6), balance_usdc)
        if amount_usdc < int(_MIN_TRADE_USD * 1e6):
            return receipts

        addrs = actions.get_chain_addresses(chain_id)
        fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
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
        plan = await actions.build_usdc_to_eth_plan(
            rpc_url=rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_usdc=amount_usdc,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
        for step in plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
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
        state.last_side = "BUY"
        state.last_trade_ts = time.time()
        state.last_buy_price = price
        state.ttp_active = False
        state.ttp_peak_price = 0.0
        _save_state(_default_history_path(), state)
        _append_trade_notification(
            task_context,
            pair="WETH/USDC",
            price=price,
            strategy_type="ALGO",
            amount=f"${amount_usdc / 1e6:.2f} USDC",
            side="BUY",
            tx_hash=receipts[-1].tx_hash if receipts else None,
        )
        log.info(
            f"ALGO: BOUGHT ${amount_usdc / 1e6:.2f} USDC -> WETH (z={z:.2f}, RSI={rsi:.1f}, "
            f"price={price:.2f})."
        )

    elif should_sell:
        amount_wei = int(balance_wei * TRADE_BUDGET_PCT)
        if amount_wei < 10**15:
            return receipts

        addrs = actions.get_chain_addresses(chain_id)
        fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
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
            receipts.append(receipt)
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
        state.last_side = "SELL"
        state.last_trade_ts = time.time()
        state.last_buy_price = 0.0
        state.ttp_active = False
        state.ttp_peak_price = 0.0
        _save_state(_default_history_path(), state)
        _append_trade_notification(
            task_context,
            pair="WETH/USDC",
            price=price,
            strategy_type="ALGO",
            amount=f"{amount_wei / 1e18:.4f} WETH",
            side="SELL",
            tx_hash=receipts[-1].tx_hash if receipts else None,
        )
        reason = "stop-loss" if stop_loss_triggered else ("trailing-TP" if ttp_triggered else "mean-reversion")
        log.info(
            f"ALGO: SOLD {amount_wei / 1e18:.4f} WETH -> USDC "
            f"(price={price:.2f}, {reason})."
        )

    return receipts


def _append_trade_notification(
    task_context: dict,
    pair: str,
    price: float,
    strategy_type: str,
    amount: str,
    side: str,
    tx_hash: str | None = None,
) -> None:
    """Append trade executed payload for TelegramManager."""
    try:
        from wallet_manager.security.telegram_manager import (
            TradeExecutedPayload,
        )
        pending = task_context.get("trade_notifications", [])
        pending.append(
            TradeExecutedPayload(
                pair=pair,
                price=price,
                strategy_type=strategy_type,
                amount=amount,
                side=side,
                tx_hash=tx_hash,
            )
        )
        task_context["trade_notifications"] = pending
    except Exception as _notif_exc:
        log.warning("Trade notification queueing failed (non-critical): %s", _notif_exc)
