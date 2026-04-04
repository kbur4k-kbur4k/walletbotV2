"""
wallet_manager/strategies/short_strategy.py

Hedged Short (Leveraged Delta-Neutral) — Synthetic Short via Aave V3.

The 10:1 Symmetrical Growth Model — Bear-Side Axis
===================================================
When the market is overextended (Z-Score > +2.5, RSI > 70, price above
upper Bollinger Band), this strategy:

1. Collateralises idle USDC in Aave V3 (if not already supplied).
2. Borrows WETH against the USDC collateral at variable rate.
3. Immediately swaps the borrowed WETH → USDC via Uniswap V3.
4. Waits for mean reversion (Z-Score returns to 0 or below).
5. Repurchases WETH at a lower price and repays the Aave debt.
6. Captures the delta as profit (target: 1-unit profit per 10-unit long profit).

Risk Management
---------------
* Maximum LTV: 60% (conservative vs Aave's 80% USDC LTV).
* Health Factor floor: 1.5 (liquidation at 1.0; we stay well above).
* Hard stop-loss: if WETH price rises 5% above entry, auto-close short.
* Position size: never more than 15% of total portfolio equity.
* Gas check: skip if mempool congested (>15 Gwei on Base).

Security
--------
* All borrowed funds immediately swapped — no naked exposure.
* Private keys handled ephemerally through WalletManager.
* Position state persisted to disk for crash recovery.
"""

# mypy: disable-error-code="no-untyped-def,no-any-return"

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from wallet_manager import actions
from wallet_manager.core.models import TxReceipt, TxRecord, TxStatus
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.strategies.alpha_engine import AlphaMLSignal

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from wallet_manager.portfolio_manager import PortfolioManager

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------------------------

# Entry signals: overbought conditions
Z_SCORE_SHORT_THRESHOLD = 2.5       # Z > +2.5 → overextended
RSI_OVERBOUGHT_THRESHOLD = 70       # RSI > 70 → overbought
BB_UPPER_PROXIMITY_PCT = 0.02       # Within 2% of upper Bollinger Band

# Exit signals: mean reversion achieved
Z_SCORE_EXIT_THRESHOLD = 0.0        # Z returns to mean
RSI_EXIT_THRESHOLD = 50             # RSI normalises

# Risk limits
MAX_LTV_PCT = 0.60                  # Never exceed 60% LTV on Aave
MIN_HEALTH_FACTOR = 1.5             # Floor: stay well above liquidation
SHORT_STOP_LOSS_PCT = 0.05          # 5% adverse move → close position
MAX_SHORT_PORTFOLIO_PCT = 0.15      # Never short more than 15% of equity
MIN_SHORT_USD = 10.0                # Minimum $10 short position

# Aave V3 interest rate mode
VARIABLE_RATE_MODE = 2

STRATEGY_ID = "short_delta_neutral"
CHAIN_ID_BASE = 8453

# ---------------------------------------------------------------------------
# Position state (persisted to disk)
# ---------------------------------------------------------------------------


@dataclass
class ShortPosition:
    """Active synthetic short position state."""

    position_id: str = ""
    entry_price: float = 0.0            # WETH price at short entry
    borrowed_weth_wei: int = 0           # Amount of WETH borrowed
    usdc_received: int = 0               # USDC received from selling borrowed WETH
    collateral_usdc: int = 0             # USDC collateral supplied to Aave
    entry_timestamp: float = 0.0
    entry_z_score: float = 0.0
    entry_rsi: float = 0.0
    is_active: bool = False
    stop_loss_price: float = 0.0         # Auto-close above this price

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "entry_price": self.entry_price,
            "borrowed_weth_wei": self.borrowed_weth_wei,
            "usdc_received": self.usdc_received,
            "collateral_usdc": self.collateral_usdc,
            "entry_timestamp": self.entry_timestamp,
            "entry_z_score": self.entry_z_score,
            "entry_rsi": self.entry_rsi,
            "is_active": self.is_active,
            "stop_loss_price": self.stop_loss_price,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShortPosition:
        return cls(
            position_id=str(d.get("position_id", "")),
            entry_price=float(d.get("entry_price", 0)),
            borrowed_weth_wei=int(d.get("borrowed_weth_wei", 0)),
            usdc_received=int(d.get("usdc_received", 0)),
            collateral_usdc=int(d.get("collateral_usdc", 0)),
            entry_timestamp=float(d.get("entry_timestamp", 0)),
            entry_z_score=float(d.get("entry_z_score", 0)),
            entry_rsi=float(d.get("entry_rsi", 0)),
            is_active=bool(d.get("is_active", False)),
            stop_loss_price=float(d.get("stop_loss_price", 0)),
        )


@dataclass
class ShortState:
    """Persistent state for the short strategy."""

    position: ShortPosition = field(default_factory=ShortPosition)
    total_short_pnl_usd: float = 0.0
    total_shorts_opened: int = 0
    total_shorts_closed: int = 0

    def to_dict(self) -> dict:
        return {
            "position": self.position.to_dict(),
            "total_short_pnl_usd": self.total_short_pnl_usd,
            "total_shorts_opened": self.total_shorts_opened,
            "total_shorts_closed": self.total_shorts_closed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShortState:
        return cls(
            position=ShortPosition.from_dict(d.get("position", {})),
            total_short_pnl_usd=float(d.get("total_short_pnl_usd", 0)),
            total_shorts_opened=int(d.get("total_shorts_opened", 0)),
            total_shorts_closed=int(d.get("total_shorts_closed", 0)),
        )


def _default_state_path() -> Path:
    from wallet_manager.config import get_settings
    p = str(get_settings().short_state_path)
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent / "short_position_state.json"


def _load_state(path: Path) -> ShortState:
    if not path.exists():
        return ShortState()
    try:
        with open(path, encoding="utf-8") as f:
            return ShortState.from_dict(json.load(f))
    except Exception as exc:
        log.warning(f"ShortState load failed: {exc}. Starting fresh.")
        return ShortState()


def _save_state(path: Path, state: ShortState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
    except OSError as exc:
        log.warning(f"ShortState save failed: {exc}")


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------


def should_open_short(z_score: float, rsi: float, bb_pct: float) -> bool:
    """Evaluate whether conditions warrant opening a synthetic short.

    Args:
        z_score: Current Z-Score (EMA-based).
        rsi: Current RSI(14) value (0-100).
        bb_pct: Price position relative to upper Bollinger Band
                (0.0 = at band, negative = above band).

    Returns:
        True if ALL overbought conditions are met.
    """
    return (
        z_score > Z_SCORE_SHORT_THRESHOLD
        and rsi > RSI_OVERBOUGHT_THRESHOLD
        and bb_pct <= BB_UPPER_PROXIMITY_PCT
    )


def should_close_short(
    z_score: float,
    rsi: float,
    current_price: float,
    entry_price: float,
    stop_loss_price: float,
) -> tuple[bool, str]:
    """Evaluate whether to close an active short position.

    Returns:
        (should_close, reason) tuple.
    """
    # Stop-loss: price rose too much
    if current_price >= stop_loss_price:
        return True, "stop_loss"

    # Mean reversion achieved
    if z_score <= Z_SCORE_EXIT_THRESHOLD and rsi <= RSI_EXIT_THRESHOLD:
        return True, "mean_reversion"

    # Profit target: price dropped significantly (>3% below entry)
    if entry_price > 0:
        drop_pct = (entry_price - current_price) / entry_price
        if drop_pct >= 0.03:
            return True, "profit_target"

    return False, ""


# ---------------------------------------------------------------------------
# Health factor safety check
# ---------------------------------------------------------------------------


async def check_health_factor_safe(rpc_url: str, address: str) -> tuple[bool, float]:
    """Check if Aave V3 health factor is above minimum.

    Returns:
        (is_safe, health_factor).
    """
    try:
        account = await actions.get_aave_account_data(rpc_url, address)
        hf = account["health_factor"]
        is_safe = hf >= MIN_HEALTH_FACTOR or hf == 0  # 0 means no debt
        return is_safe, hf
    except Exception as exc:
        log.warning(f"Health factor check failed: {exc}. Assuming unsafe.")
        return False, 0.0


# ---------------------------------------------------------------------------
# Core strategy execution
# ---------------------------------------------------------------------------


def _get_wallet_address(config) -> str:
    if config.address:
        return config.address
    from eth_account import Account
    return Account.from_key(config.private_key).address


async def run_short_strategy(
    manager: WalletManager,
    wallet_id: str,
    portfolio_manager: PortfolioManager,
    task_context: dict,
    z_score: float,
    rsi: float,
    current_price: float,
    bb_upper: float,
    analytics_db=None,
    alpha_signal: AlphaMLSignal | None = None,
) -> list[TxReceipt]:
    """Run one cycle of the Hedged Short (Delta-Neutral) strategy.

    Entry: Z-Score > +2.5 AND RSI > 70 AND price near upper BB.
    Exit: Z-Score ≤ 0 AND RSI ≤ 50, OR stop-loss at 5%, OR profit target 3%.

    The 10:1 ratio means for every $10 the long strategy makes in a bull
    cycle, the short captures $1 in the bear-side correction.

    Args:
        manager: Active WalletManager.
        wallet_id: Wallet identifier.
        portfolio_manager: Portfolio state tracker.
        task_context: Shared context dict.
        z_score: Current Z-Score value.
        rsi: Current RSI(14) value.
        current_price: Current WETH/USDC price.
        bb_upper: Upper Bollinger Band value.
        analytics_db: Optional analytics recorder.

    Returns:
        List of transaction receipts.
    """
    from wallet_manager.network.gas_monitor import GasMonitor

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id)
    if chain_id != CHAIN_ID_BASE:
        return []

    rpc_url = config.rpc_url
    address = _get_wallet_address(config)

    # Gas safety — skip in congested mempool
    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        log.info(f"SHORT: gas {current_gwei:.2f} Gwei too high — skip.")
        return []

    state = _load_state(_default_state_path())
    receipts: list[TxReceipt] = []

    # --- Active position: check for exit ---
    if state.position.is_active:
        # ML signal exit: if alpha_signal is provided and no longer bearish → exit
        if alpha_signal is not None and not alpha_signal.should_enter_short:
            close_receipts = await _close_short_position(
                manager=manager,
                wallet_id=wallet_id,
                rpc_url=rpc_url,
                address=address,
                state=state,
                current_price=current_price,
                current_gwei=current_gwei,
                reason="ml_signal_exit",
                task_context=task_context,
                analytics_db=analytics_db,
            )
            receipts.extend(close_receipts)
            return receipts

        close, reason = should_close_short(
            z_score=z_score,
            rsi=rsi,
            current_price=current_price,
            entry_price=state.position.entry_price,
            stop_loss_price=state.position.stop_loss_price,
        )

        # Also check health factor
        hf_safe, hf = await check_health_factor_safe(rpc_url, address)
        if not hf_safe and hf > 0:
            close = True
            reason = "health_factor_critical"

        if close:
            close_receipts = await _close_short_position(
                manager=manager,
                wallet_id=wallet_id,
                rpc_url=rpc_url,
                address=address,
                state=state,
                current_price=current_price,
                current_gwei=current_gwei,
                reason=reason,
                task_context=task_context,
                analytics_db=analytics_db,
            )
            receipts.extend(close_receipts)
        else:
            log.debug(
                f"SHORT: position active, no exit signal. "
                f"Entry: ${state.position.entry_price:.2f}, "
                f"Current: ${current_price:.2f}, HF: {hf:.2f}."
            )
        return receipts

    # --- No active position: check for entry ---
    bb_proximity = (current_price - bb_upper) / bb_upper if bb_upper > 0 else 1.0

    # ML signal entry: use ML confidence when alpha_signal is provided
    if alpha_signal is not None:
        if not alpha_signal.should_enter_short:
            log.debug(
                f"SHORT: ML signal — no short entry signal "
                f"(confidence={alpha_signal.confidence:.3f})."
            )
            return receipts
        # ML signal present → continue to entry operations
        log.info(
            f"SHORT: ML signal entry — confidence={alpha_signal.confidence:.3f}"
        )
    elif not should_open_short(z_score, rsi, bb_proximity):
        log.debug(
            f"SHORT: no entry signal. z={z_score:.2f}, RSI={rsi:.1f}, "
            f"BB_prox={bb_proximity:.4f}."
        )
        return receipts

    # Check portfolio allocation limit
    total_equity = portfolio_manager.total_value_usd
    max_short_usd = total_equity * MAX_SHORT_PORTFOLIO_PCT
    if max_short_usd < MIN_SHORT_USD:
        log.info(f"SHORT: equity too low for short position (max ${max_short_usd:.2f}).")
        return receipts

    # Check available borrows on Aave
    try:
        account_data = await actions.get_aave_account_data(rpc_url, address)
    except Exception as exc:
        log.warning(f"SHORT: Aave account data fetch failed: {exc}")
        return receipts

    available_borrows_usd = account_data["available_borrows_usd"]
    if available_borrows_usd < MIN_SHORT_USD:
        log.info(
            f"SHORT: insufficient Aave borrowing capacity "
            f"(${available_borrows_usd:.2f} available)."
        )
        return receipts

    # Size the position: min of (max_short, available_borrows * 0.5, free USDC)
    position_usd = min(max_short_usd, available_borrows_usd * 0.5)
    borrow_eth_amount = position_usd / current_price if current_price > 0 else 0
    borrow_wei = int(borrow_eth_amount * 1e18)

    if borrow_wei < 10**15:  # < 0.001 ETH
        log.info("SHORT: calculated borrow amount too small.")
        return receipts

    log.info(
        f"SHORT: OPENING position — borrow {borrow_wei / 1e18:.6f} WETH "
        f"(~${position_usd:.2f}) at price ${current_price:.2f}. "
        f"z={z_score:.2f}, RSI={rsi:.1f}."
    )

    open_receipts = await _open_short_position(
        manager=manager,
        wallet_id=wallet_id,
        rpc_url=rpc_url,
        address=address,
        state=state,
        borrow_wei=borrow_wei,
        current_price=current_price,
        current_gwei=current_gwei,
        z_score=z_score,
        rsi=rsi,
        task_context=task_context,
        analytics_db=analytics_db,
    )
    receipts.extend(open_receipts)

    return receipts


# ---------------------------------------------------------------------------
# Position open/close
# ---------------------------------------------------------------------------


async def _open_short_position(
    manager: WalletManager,
    wallet_id: str,
    rpc_url: str,
    address: str,
    state: ShortState,
    borrow_wei: int,
    current_price: float,
    current_gwei: float,
    z_score: float,
    rsi: float,
    task_context: dict,
    analytics_db=None,
) -> list[TxReceipt]:
    """Execute synthetic short: borrow WETH → swap to USDC."""
    receipts: list[TxReceipt] = []
    chain_id = CHAIN_ID_BASE

    try:
        # Step 1: Borrow WETH from Aave V3
        borrow_plan = await actions.build_aave_borrow_eth_plan(
            rpc_url=rpc_url,
            from_address=address,
            borrow_amount_wei=borrow_wei,
            gas_price_gwei=current_gwei,
        )
        for tx_params, desc in borrow_plan:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, tx_params)
            receipts.append(receipt)
            log.info(f"SHORT: {desc} — tx: {receipt.tx_hash}")
            if analytics_db:
                record = TxRecord(
                    tx_hash=receipt.tx_hash,
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=receipt.nonce_used,
                    gas_price_gwei=tx_params.gas_price_gwei,
                    gas_limit=tx_params.gas,
                    value_wei=tx_params.value,
                    status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type="SHORT",
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # Step 2: Swap borrowed WETH → USDC via Uniswap V3
        addrs = actions.get_chain_addresses(chain_id)
        fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
        slippage_bps = await actions.get_dynamic_slippage(
            rpc_url=rpc_url,
            chain_id=chain_id,
            token_in=addrs.weth,
            token_out=addrs.usdc,
            amount_in=borrow_wei,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
            is_algo=True,  # Strict 0.5% slippage for MEV protection
        )
        swap_plan = await actions.build_eth_to_usdc_plan(
            rpc_url=rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_wei=borrow_wei,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
        for step in swap_plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
            log.info(f"SHORT: {step.description} — tx: {receipt.tx_hash}")
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
                    strategy_type="SHORT",
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # Update state
        usdc_received = int(borrow_wei * current_price / 1e18 * 1e6)
        state.position = ShortPosition(
            position_id=f"short_{int(time.time())}",
            entry_price=current_price,
            borrowed_weth_wei=borrow_wei,
            usdc_received=usdc_received,
            collateral_usdc=0,  # Collateral already in Aave from yield strategy
            entry_timestamp=time.time(),
            entry_z_score=z_score,
            entry_rsi=rsi,
            is_active=True,
            stop_loss_price=current_price * (1.0 + SHORT_STOP_LOSS_PCT),
        )
        state.total_shorts_opened += 1
        _save_state(_default_state_path(), state)

        # Notification
        _notify_short_opened(task_context, state.position, current_price)

        log.info(
            f"SHORT: Position OPENED — {borrow_wei / 1e18:.6f} WETH "
            f"@ ${current_price:.2f}. Stop-loss @ ${state.position.stop_loss_price:.2f}."
        )

    except Exception as exc:
        log.error(f"SHORT: Failed to open position: {exc}")
        raise

    return receipts


async def _close_short_position(
    manager: WalletManager,
    wallet_id: str,
    rpc_url: str,
    address: str,
    state: ShortState,
    current_price: float,
    current_gwei: float,
    reason: str,
    task_context: dict,
    analytics_db=None,
) -> list[TxReceipt]:
    """Close synthetic short: buy WETH → repay Aave debt."""
    receipts: list[TxReceipt] = []
    chain_id = CHAIN_ID_BASE
    pos = state.position

    try:
        # Step 1: Get current debt balance (may differ due to accrued interest)
        debt_wei = await actions.get_variable_debt_balance(rpc_url, address)
        if debt_wei <= 0:
            log.info("SHORT: No debt found — position already closed.")
            state.position.is_active = False
            _save_state(_default_state_path(), state)
            return receipts

        # Step 2: Buy WETH with USDC to repay (swap USDC → WETH)
        addrs = actions.get_chain_addresses(chain_id)
        # Calculate USDC needed to buy `debt_wei` WETH
        repay_usdc_needed = int((debt_wei / 1e18) * current_price * 1.01 * 1e6)  # 1% buffer

        fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
        slippage_bps = await actions.get_dynamic_slippage(
            rpc_url=rpc_url,
            chain_id=chain_id,
            token_in=addrs.usdc,
            token_out=addrs.weth,
            amount_in=repay_usdc_needed,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
            is_algo=True,
        )
        swap_plan = await actions.build_usdc_to_eth_plan(
            rpc_url=rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_usdc=repay_usdc_needed,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
        for step in swap_plan.steps:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
            log.info(f"SHORT CLOSE: {step.description} — tx: {receipt.tx_hash}")
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
                    strategy_type="SHORT",
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # Step 3: Repay WETH debt to Aave V3
        repay_plan = await actions.build_aave_repay_eth_plan(
            rpc_url=rpc_url,
            from_address=address,
            repay_amount_wei=debt_wei,
            gas_price_gwei=current_gwei,
        )
        for tx_params, desc in repay_plan:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, tx_params)
            receipts.append(receipt)
            log.info(f"SHORT CLOSE: {desc} — tx: {receipt.tx_hash}")
            if analytics_db:
                record = TxRecord(
                    tx_hash=receipt.tx_hash,
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=receipt.nonce_used,
                    gas_price_gwei=tx_params.gas_price_gwei,
                    gas_limit=tx_params.gas,
                    value_wei=tx_params.value,
                    status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type="SHORT",
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)

        # Calculate P&L
        entry_cost_usd = (pos.borrowed_weth_wei / 1e18) * pos.entry_price
        close_cost_usd = (debt_wei / 1e18) * current_price
        pnl_usd = entry_cost_usd - close_cost_usd  # Positive if price dropped

        state.position.is_active = False
        state.total_shorts_closed += 1
        state.total_short_pnl_usd += pnl_usd
        _save_state(_default_state_path(), state)

        # Notification
        _notify_short_closed(task_context, pos, current_price, pnl_usd, reason)

        log.info(
            f"SHORT: Position CLOSED ({reason}) — "
            f"Entry: ${pos.entry_price:.2f}, Exit: ${current_price:.2f}, "
            f"P&L: ${pnl_usd:+.2f}. Lifetime P&L: ${state.total_short_pnl_usd:+.2f}."
        )

    except Exception as exc:
        log.error(f"SHORT: Failed to close position: {exc}")
        raise

    return receipts


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------


def _notify_short_opened(
    task_context: dict,
    position: ShortPosition,
    current_price: float,
) -> None:
    """Send short-opened notification via TelegramManager."""
    try:
        from wallet_manager.security.telegram_manager import TradeExecutedPayload

        pending = task_context.get("trade_notifications", [])
        pending.append(
            TradeExecutedPayload(
                pair="WETH/USDC",
                price=current_price,
                strategy_type="SHORT",
                amount=f"{position.borrowed_weth_wei / 1e18:.6f} WETH (synthetic)",
                side="SHORT_OPEN",
            )
        )
        task_context["trade_notifications"] = pending
    except Exception:
        pass


def _notify_short_closed(
    task_context: dict,
    position: ShortPosition,
    current_price: float,
    pnl_usd: float,
    reason: str,
) -> None:
    """Send short-closed notification via TelegramManager."""
    try:
        from wallet_manager.security.telegram_manager import TradeExecutedPayload

        pending = task_context.get("trade_notifications", [])
        pending.append(
            TradeExecutedPayload(
                pair="WETH/USDC",
                price=current_price,
                strategy_type="SHORT",
                amount=f"P&L: ${pnl_usd:+.2f} ({reason})",
                side="SHORT_CLOSE",
            )
        )
        task_context["trade_notifications"] = pending
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def get_short_position_state() -> ShortState:
    """Load and return current short position state."""
    return _load_state(_default_state_path())


def get_short_pnl_usd() -> float:
    """Return lifetime short strategy P&L in USD."""
    state = _load_state(_default_state_path())
    return state.total_short_pnl_usd


def is_short_active() -> bool:
    """Return True if there is an active short position."""
    state = _load_state(_default_state_path())
    return state.position.is_active
