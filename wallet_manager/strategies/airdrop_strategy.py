"""
wallet_manager/strategies/airdrop_strategy.py

Airdrop pillar (15%) — Ping-Pong ETH<->USDC farming with 15% total balance limit.

Revised ping-pong: swap direction based on balances, capped at 15% of total
portfolio value (ETH + USDC) to stay within allocation.
"""

# mypy: disable-error-code="no-untyped-def,no-any-return"

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from wallet_manager import actions
from wallet_manager.core.models import TxReceipt, TxRecord, TxStatus
from wallet_manager.portfolio_manager import StrategyType
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager

log = build_logger(__name__)

AIRDROP_ALLOCATION_PCT = 0.15
MIN_SWAP_USD = 10.0
MAX_GWEI = 25.0
STRATEGY_ID = "airdrop_ping_pong"

# Estimated gas units for a Uniswap V3 swap on Base
_GAS_USED_ESTIMATE = 150_000


class GasAwareThrottle:
    """Skips the ping-pong cycle if gas cost exceeds a given ratio
    of the expected fee revenue.

    Logic:
      estimated_gas_cost_usd = gas_used_estimate * current_gwei * eth_price / 1e9
      expected_fee_usd = swap_amount_usd * pool_fee_tier  (0.0005 for 0.05%)

      if estimated_gas_cost_usd > cfg.airdrop_max_gas_fee_ratio * expected_fee_usd:
          skip cycle, log reason
    """

    def should_execute(
        self,
        current_gwei: float,
        eth_price_usd: float,
        swap_amount_usd: float,
        pool_fee_tier: float = 0.0005,
    ) -> tuple[bool, str]:
        """Returns (should_execute, reason_string)."""
        from wallet_manager.config import get_settings
        cfg = get_settings()
        max_ratio = cfg.airdrop_max_gas_fee_ratio

        estimated_gas_cost_usd = _GAS_USED_ESTIMATE * current_gwei * eth_price_usd / 1e9
        expected_fee_usd = swap_amount_usd * pool_fee_tier

        if expected_fee_usd <= 0:
            return False, "expected_fee_usd=0 — skip"

        ratio = estimated_gas_cost_usd / expected_fee_usd
        if ratio > max_ratio:
            reason = (
                f"gas_cost=${estimated_gas_cost_usd:.4f} > "
                f"{max_ratio:.1f}x fee=${expected_fee_usd:.4f} (ratio={ratio:.2f})"
            )
            return False, reason

        return True, f"gas_ratio={ratio:.2f} OK"


_gas_throttle = GasAwareThrottle()


def _get_wallet_address(config) -> str:
    if config.address:
        return config.address
    from eth_account import Account
    return Account.from_key(config.private_key).address


async def run_airdrop_strategy(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict,
) -> list[TxReceipt]:
    """
    Ping-Pong airdrop farming: swap ETH<->USDC based on which side is larger.

    Hard cap: total swap exposure never exceeds 15% of total balance (ETH + USDC).
    If airdrop_budget_usd is in context, further caps to that budget.
    """
    ctx = task_context or {}
    notifier = ctx.get("notifier")
    confirmation_gate = ctx.get("confirmation_gate")
    scheduler = ctx.get("scheduler")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if scheduler is None:
        log.warning("No scheduler in task_context — skip airdrop logic.")
        return receipts

    config = manager.get_config(wallet_id)
    from wallet_manager.network.gas_monitor import GasMonitor

    gas_monitor = GasMonitor(rpc_url=config.rpc_url)
    max_gwei = min(MAX_GWEI, float(config.max_gas_price_gwei))

    from wallet_manager.scheduler import should_run_swap_this_cycle
    should_run, seconds_until_due = should_run_swap_this_cycle(scheduler, wallet_id)
    if not should_run:
        return receipts

    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        log.info(f"AIRDROP: gas {current_gwei:.2f} Gwei > max — skip.")
        return receipts

    chain_id = int(config.chain_id)
    if chain_id not in (59144, 8453):
        log.warning(f"AIRDROP: chain {chain_id} not supported.")
        return receipts

    address = _get_wallet_address(config)

    try:
        balance_wei = await manager.get_safe_balance(wallet_id)
    except Exception as exc:
        log.warning(f"AIRDROP: balance fetch failed: {exc}")
        return receipts

    try:
        balance_usdc = await actions.fetch_usdc_balance(config.rpc_url, address, chain_id)
    except Exception as exc:
        log.warning(f"AIRDROP: USDC fetch failed: {exc}")
        balance_usdc = 0

    eth_price = await actions.fetch_eth_price_usd()
    eth_value_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
    usdc_value_usd = actions.usdc_balance_usd_value(balance_usdc)
    total_balance_usd = eth_value_usd + usdc_value_usd

    airdrop_budget_usd = ctx.get("airdrop_budget_usd")
    max_swap_usd = total_balance_usd * AIRDROP_ALLOCATION_PCT
    if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
        max_swap_usd = min(max_swap_usd, airdrop_budget_usd)

    if max_swap_usd < MIN_SWAP_USD:
        log.info(f"AIRDROP: budget ${max_swap_usd:.2f} < ${MIN_SWAP_USD} — skip.")
        return receipts

    # Gas-aware throttle check
    gas_ok, gas_reason = _gas_throttle.should_execute(
        current_gwei=current_gwei,
        eth_price_usd=eth_price,
        swap_amount_usd=max_swap_usd,
    )
    if not gas_ok:
        log.info(f"AIRDROP: GasAwareThrottle skipped — {gas_reason}")
        return receipts

    do_usdc_to_eth = (
        actions.is_usdc_swap_eligible(balance_usdc, current_gwei, max_gas_gwei=max_gwei)
        and usdc_value_usd > eth_value_usd
    )
    do_eth_to_usdc = actions.is_swap_eligible(
        balance_wei, eth_price, current_gwei,
        min_eth_value_usd=actions.MIN_ETH_VALUE_USD,
        max_gas_gwei=max_gwei,
    )

    if do_usdc_to_eth:
        pct = random.uniform(0.20, 0.50)
        amount_usdc = int(balance_usdc * pct)
        max_usdc = int(max_swap_usd * 0.5 * 1e6)
        amount_usdc = min(amount_usdc, max_usdc)
        if amount_usdc < int(MIN_SWAP_USD * 1e6):
            return receipts
        return await _execute_usdc_to_eth(
            manager, wallet_id, config, address, chain_id,
            amount_usdc, current_gwei,
            notifier, confirmation_gate, scheduler,
            analytics_db=analytics_db,
        )

    if do_eth_to_usdc:
        pct = random.uniform(0.10, 0.20)
        amount_wei = int(balance_wei * pct)
        max_eth_val = max_swap_usd * 0.5
        max_wei = int(max_eth_val * 1e18 / eth_price)
        amount_wei = min(amount_wei, max_wei)
        if amount_wei < 10**15:
            return receipts
        return await _execute_eth_to_usdc(
            manager, wallet_id, config, address, chain_id,
            amount_wei, current_gwei,
            notifier, confirmation_gate, scheduler,
            analytics_db=analytics_db,
        )

    log.info("AIRDROP: not eligible for ping-pong.")
    return receipts


async def _execute_usdc_to_eth(
    manager: WalletManager,
    wallet_id: str,
    config,
    address: str,
    chain_id: int,
    amount_usdc: int,
    current_gwei: float,
    notifier,
    confirmation_gate,
    scheduler,
    analytics_db=None,
) -> list[TxReceipt]:
    """Execute USDC -> ETH swap."""
    from wallet_manager.core.models import EventKind, NotificationEvent
    from wallet_manager.orchestrator import WalletOrchestrator  # Late import

    receipts: list[TxReceipt] = []
    fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
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
        log.error(f"AIRDROP: USDC->ETH plan failed: {exc}")
        if notifier:
            await WalletOrchestrator._safe_notify(
                notifier,
                NotificationEvent(kind=EventKind.WALLET_FAILURE, wallet_id=wallet_id, detail=str(exc)),
            )
        return receipts

    amount_str = f"{amount_usdc / 1e6:.2f} USDC"
    if notifier and hasattr(notifier, "send_confirmation_request") and confirmation_gate:
        confirmation_gate.request_confirmation()
        try:
            await notifier.send_confirmation_request(wallet_id, amount_str, "USDC -> ETH")
        except Exception as exc:
            log.warning(f"Confirmation request failed: {exc}")
        confirmed = await confirmation_gate.wait_for_evet(timeout_seconds=300.0)
        if not confirmed:
            log.warning("AIRDROP: confirmation timeout/rejected.")
            scheduler.schedule_next(wallet_id)
            return receipts

    for i, step in enumerate(plan.steps):
        try:
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
                    strategy_type=StrategyType.AIRDROP.value,
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)
        except Exception as exc:
            log.error(f"AIRDROP: step {i + 1} failed: {exc}")
            if notifier:
                await WalletOrchestrator._safe_notify(
                    notifier,
                    NotificationEvent(kind=EventKind.WALLET_FAILURE, wallet_id=wallet_id, detail=str(exc)),
                )
            scheduler.schedule_next(wallet_id)
            return receipts

    scheduler.schedule_next(wallet_id)
    return receipts


async def _execute_eth_to_usdc(
    manager: WalletManager,
    wallet_id: str,
    config,
    address: str,
    chain_id: int,
    amount_wei: int,
    current_gwei: float,
    notifier,
    confirmation_gate,
    scheduler,
    analytics_db=None,
) -> list[TxReceipt]:
    """Execute ETH -> USDC swap."""
    from wallet_manager.core.models import EventKind, NotificationEvent
    from wallet_manager.orchestrator import WalletOrchestrator  # Late import

    receipts: list[TxReceipt] = []
    fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)
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
        log.error(f"AIRDROP: ETH->USDC plan failed: {exc}")
        if notifier:
            await WalletOrchestrator._safe_notify(
                notifier,
                NotificationEvent(kind=EventKind.WALLET_FAILURE, wallet_id=wallet_id, detail=str(exc)),
            )
        return receipts

    amount_str = f"{amount_wei / 1e18:.4f} ETH"
    if notifier and hasattr(notifier, "send_confirmation_request") and confirmation_gate:
        confirmation_gate.request_confirmation()
        try:
            await notifier.send_confirmation_request(wallet_id, amount_str, "ETH -> USDC")
        except Exception as exc:
            log.warning(f"Confirmation request failed: {exc}")
        confirmed = await confirmation_gate.wait_for_evet(timeout_seconds=300.0)
        if not confirmed:
            log.warning("AIRDROP: confirmation timeout/rejected.")
            scheduler.schedule_next(wallet_id)
            return receipts

    for i, step in enumerate(plan.steps):
        try:
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
                    strategy_type=StrategyType.AIRDROP.value,
                    strategy_id=STRATEGY_ID,
                )
                await analytics_db.record_tx(record)
        except Exception as exc:
            log.error(f"AIRDROP: step {i + 1} failed: {exc}")
            if notifier:
                await WalletOrchestrator._safe_notify(
                    notifier,
                    NotificationEvent(kind=EventKind.WALLET_FAILURE, wallet_id=wallet_id, detail=str(exc)),
                )
            scheduler.schedule_next(wallet_id)
            return receipts

    scheduler.schedule_next(wallet_id)
    return receipts
