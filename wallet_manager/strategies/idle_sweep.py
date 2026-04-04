"""
wallet_manager/strategies/idle_sweep.py

Passive Yield — Zero Lazy Capital Enforcement.

Core Mandate
============
All idle USDC must be algorithmically swept into Aave V3 Supply to
maintain a baseline APR. Zero capital should sit idle in the wallet
when it could be earning yield.

Sweep Logic
-----------
1. Check wallet USDC balance (not already in Aave).
2. Reserve a configurable buffer for gas + upcoming swaps.
3. Supply everything above the buffer to Aave V3.
4. On withdrawal demand (algo trade, short close), withdraw from Aave first.

Safety
------
* Minimum sweep: $5 USDC (avoid dust transactions that waste gas).
* Gas check: only sweep when gas is cheap (<10 Gwei on Base).
* Buffer: keep $20 USDC liquid for potential algo/airdrop swaps.
* Health factor: never sweep if it would endanger Aave positions.
"""

# mypy: disable-error-code="no-untyped-def,no-any-return"

from __future__ import annotations

from typing import TYPE_CHECKING

from wallet_manager import actions
from wallet_manager.core.models import TxReceipt
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

# Minimum USDC to sweep (avoid dust tx)
MIN_SWEEP_USDC = 5_000_000      # $5 in 6-decimal units

# USDC buffer to keep liquid for swaps/gas
USDC_BUFFER = 20_000_000        # $20 buffer

# Max gas for sweep (Gwei) — only sweep when cheap
MAX_SWEEP_GAS_GWEI = 10.0

# Aave supply gas limits
GAS_LIMIT_APPROVE = 80_000
GAS_LIMIT_SUPPLY = 200_000

CHAIN_ID_BASE = 8453
STRATEGY_ID = "idle_sweep_aave_v3"


def _get_wallet_address(config) -> str:
    if config.address:
        return config.address
    from eth_account import Account
    return Account.from_key(config.private_key).address


async def calculate_sweepable_usdc(
    rpc_url: str,
    address: str,
    chain_id: int,
) -> int:
    """Calculate how much USDC can be swept to Aave.

    Returns amount in 6-decimal USDC units. Returns 0 if below minimum.
    """
    try:
        balance = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
    except Exception as exc:
        log.warning(f"Idle sweep: USDC balance fetch failed: {exc}")
        return 0

    sweepable = balance - USDC_BUFFER
    if sweepable < MIN_SWEEP_USDC:
        return 0

    return sweepable


async def run_idle_sweep(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
    analytics_db=None,
) -> list[TxReceipt]:
    """Sweep idle USDC into Aave V3 for passive yield.

    Called every orchestrator cycle. Only executes if:
    1. Chain is Base (8453).
    2. Gas is cheap (< MAX_SWEEP_GAS_GWEI).
    3. Sweepable USDC exceeds MIN_SWEEP_USDC.
    4. Aave health factor is safe (if existing positions).

    Returns:
        List of transaction receipts from the supply operation.
    """
    from wallet_manager.network.gas_monitor import GasMonitor
    from wallet_manager.strategies import yield_strategy

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id)
    if chain_id != CHAIN_ID_BASE:
        return []

    rpc_url = config.rpc_url
    address = _get_wallet_address(config)

    # Gas check — only sweep when cheap
    gas_monitor = GasMonitor(rpc_url=rpc_url)
    safe, current_gwei = await gas_monitor.is_safe(MAX_SWEEP_GAS_GWEI)
    if not safe:
        log.debug(
            f"Idle sweep: gas {current_gwei:.2f} Gwei > {MAX_SWEEP_GAS_GWEI} — skip."
        )
        return []

    # Calculate sweepable amount
    sweep_amount = await calculate_sweepable_usdc(rpc_url, address, chain_id)
    if sweep_amount <= 0:
        log.debug("Idle sweep: no sweepable USDC above buffer.")
        return []

    # Health factor check (if we have Aave positions)
    try:
        account = await actions.get_aave_account_data(rpc_url, address)
        hf = account["health_factor"]
        # If we have debt and HF is concerning, don't add more exposure
        if account["total_debt_usd"] > 0 and hf < 2.0:
            log.info(
                f"Idle sweep: skipping — Aave HF {hf:.2f} < 2.0 with active debt."
            )
            return []
    except Exception as exc:
        log.debug(f"Idle sweep: Aave account check failed ({exc}). Proceeding.")

    log.info(
        f"Idle sweep: supplying {sweep_amount / 1e6:.2f} USDC to Aave V3 "
        f"(gas: {current_gwei:.2f} Gwei)."
    )

    # Execute supply via yield strategy
    try:
        receipts = await yield_strategy.supply_usdc(
            manager=manager,
            wallet_id=wallet_id,
            rpc_url=rpc_url,
            from_address=address,
            amount_usdc=sweep_amount,
            gas_price_gwei=current_gwei,
            analytics_db=analytics_db,
        )

        log.info(
            f"Idle sweep: {sweep_amount / 1e6:.2f} USDC supplied to Aave V3 "
            f"({len(receipts)} tx)."
        )

        # Notify
        if task_context is not None:
            try:
                from wallet_manager.security.telegram_manager import (
                    TradeExecutedPayload,
                )
                pending = task_context.get("trade_notifications", [])
                pending.append(
                    TradeExecutedPayload(
                        pair="USDC/aUSDC",
                        price=1.0,
                        strategy_type="YIELD_SWEEP",
                        amount=f"${sweep_amount / 1e6:.2f} USDC → Aave V3",
                        side="SUPPLY",
                        tx_hash=receipts[-1].tx_hash if receipts else None,
                    )
                )
                task_context["trade_notifications"] = pending
            except Exception as _notif_exc:
                log.warning("Trade notification queueing failed (non-critical): %s", _notif_exc)

        return receipts

    except Exception as exc:
        log.error(f"Idle sweep: supply failed: {exc}")
        return []


async def emergency_withdraw_for_trade(
    manager: WalletManager,
    wallet_id: str,
    amount_usdc: int,
    gas_price_gwei: float,
    analytics_db=None,
) -> list[TxReceipt]:
    """Withdraw USDC from Aave V3 to fund a trade.

    Called by algo/short strategies when they need USDC but the wallet
    balance is insufficient and capital is locked in Aave.

    Args:
        manager: Active WalletManager.
        wallet_id: Wallet identifier.
        amount_usdc: Amount to withdraw in 6-decimal USDC units.
        gas_price_gwei: Current gas price.
        analytics_db: Optional analytics recorder.

    Returns:
        List of transaction receipts.
    """
    from wallet_manager.strategies import yield_strategy

    config = manager.get_config(wallet_id)
    address = _get_wallet_address(config)

    log.info(
        f"Emergency withdraw: pulling {amount_usdc / 1e6:.2f} USDC from Aave "
        f"for trade execution."
    )

    try:
        receipts = await yield_strategy.withdraw_usdc(
            manager=manager,
            wallet_id=wallet_id,
            from_address=address,
            amount_usdc=amount_usdc,
            gas_price_gwei=gas_price_gwei,
            analytics_db=analytics_db,
        )
        log.info(
            f"Emergency withdraw: {amount_usdc / 1e6:.2f} USDC withdrawn "
            f"({len(receipts)} tx)."
        )
        return receipts
    except Exception as exc:
        log.error(f"Emergency withdraw failed: {exc}")
        return []
