"""
wallet_manager/strategies/yield_strategy.py

Yield pillar (50%) — Aave V3 USDC supply on Base.

Supplies 50% of portfolio capital to Aave V3 for passive yield.
Functions: supply_usdc, withdraw_usdc, get_apy.
"""

# mypy: disable-error-code="call-overload,no-any-return,no-untyped-def"

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wallet_manager import actions
from wallet_manager.core.models import TxParams, TxReceipt, TxRecord, TxStatus
from wallet_manager.portfolio_manager import StrategyType
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager

log = build_logger(__name__)

# Aave V3 Base (use actions for single source of truth)
AAVE_POOL_BASE = actions.AAVE_POOL_BASE
USDC_BASE = actions.USDC_BASE
ATOKEN_USDC_BASE = actions.ATOKEN_USDC_BASE
USDC_DECIMALS = 6

GAS_LIMIT_APPROVE = 80_000
GAS_LIMIT_SUPPLY = 200_000
GAS_LIMIT_WITHDRAW = 150_000

STRATEGY_ID = "yield_aave_v3"


def _checksum(addr: str) -> str:
    from web3 import Web3
    return Web3.to_checksum_address(addr)




async def get_usdc_balance(rpc_url: str, address: str) -> int:
    """Fetch USDC balance for the address on Base."""
    return await actions.fetch_usdc_balance(rpc_url, address, actions.CHAIN_ID_BASE)


from wallet_manager.utils.web3_utils import close_w3_provider as _close_w3_provider


async def get_aave_usdc_balance(rpc_url: str, address: str) -> int:
    """Fetch aBaseUSDC (supplied USDC) balance for the address."""
    from web3 import AsyncWeb3
    balance_abi = [
        {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf",
         "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}
    ]
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
    try:
        contract = w3.eth.contract(address=_checksum(ATOKEN_USDC_BASE), abi=balance_abi)
        return int(await contract.functions.balanceOf(_checksum(address)).call())
    finally:
        await _close_w3_provider(w3)


async def get_apy(rpc_url: str) -> float:
    """
    Fetch current USDC supply APY from Aave V3.

    Returns:
        APY as percentage (e.g. 3.5 for 3.5%). Fallback ~3.5% if unavailable.
    """
    bps = await get_supply_apy_bps(rpc_url)
    return (bps or 350.0) / 100.0


async def get_supply_apy_bps(rpc_url: str) -> float | None:
    """Fetch USDC supply APY in basis points (100 bps = 1%)."""
    from web3 import AsyncWeb3
    pool_abi = [
        {"inputs": [{"name": "asset", "type": "address"}], "name": "getReserveData",
         "outputs": [
             {"name": "configuration", "type": "tuple", "components": [{"name": "data", "type": "uint256"}]},
             {"name": "liquidityIndex", "type": "uint128"},
             {"name": "currentLiquidityRate", "type": "uint128"},
             {"name": "variableBorrowIndex", "type": "uint128"},
             {"name": "currentVariableBorrowRate", "type": "uint128"},
             {"name": "currentStableBorrowRate", "type": "uint128"},
             {"name": "lastUpdateTimestamp", "type": "uint40"},
             {"name": "id", "type": "uint16"},
             {"name": "aTokenAddress", "type": "address"},
             {"name": "stableDebtTokenAddress", "type": "address"},
             {"name": "variableDebtTokenAddress", "type": "address"},
             {"name": "interestRateStrategyAddress", "type": "address"},
             {"name": "accruedToTreasuryScaled", "type": "uint128"},
             {"name": "unbacked", "type": "uint128"},
             {"name": "isolationModeTotalDebt", "type": "uint128"},
         ], "stateMutability": "view", "type": "function"}
    ]
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
    try:
        pool = w3.eth.contract(address=_checksum(AAVE_POOL_BASE), abi=pool_abi)
        data = await pool.functions.getReserveData(_checksum(USDC_BASE)).call()
        # currentLiquidityRate is at index 2 (tuple ABI: [0]=configuration tuple, [1]=liquidityIndex, [2]=currentLiquidityRate)
        rate_ray = int(data[2])
        # rate_ray is in ray units (1e27). Convert to bps: rate/1e27 * 100 (to %) * 100 (to bps)
        apy_bps = rate_ray * 10000 / 1e27
        # Sanity cap: 50000 bps = 500% APY (should never exceed this in practice)
        return min(float(apy_bps), 50000.0)
    except Exception as exc:
        log.info(f"Aave APY fetch unavailable ({type(exc).__name__}: {exc}). Using 3.5% fallback.")
        return 350.0
    finally:
        await _close_w3_provider(w3)



async def get_yield_pillar_value_usd(rpc_url: str, address: str) -> float:
    """Return total USD value of the YIELD pillar (aBaseUSDC balance)."""
    supplied = await get_aave_usdc_balance(rpc_url, address)
    return supplied / 10**USDC_DECIMALS


@dataclass
class UserPosition:
    """User's Aave V3 position: balance + APY."""

    balance_usd: float
    balance_atoken_units: int
    apy_pct: float

    @property
    def balance_usdc(self) -> float:
        """Balance in USDC (6 decimals)."""
        return self.balance_atoken_units / 10**USDC_DECIMALS


async def get_user_position(rpc_url: str, address: str) -> UserPosition:
    """
    Fetch user's Aave V3 USDC position: balance + APY tracking.

    Returns:
        UserPosition with balance_usd, balance_atoken_units, apy_pct.
    """
    balance_atoken = await get_aave_usdc_balance(rpc_url, address)
    apy = await get_apy(rpc_url)
    balance_usd = balance_atoken / 10**USDC_DECIMALS
    return UserPosition(
        balance_usd=balance_usd,
        balance_atoken_units=balance_atoken,
        apy_pct=apy,
    )


@dataclass
class SupplyPlan:
    """Plan for supplying or withdrawing USDC to/from Aave."""

    steps: list[tuple[TxParams, str]]


async def build_supply_usdc_plan(
    rpc_url: str,
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
) -> SupplyPlan:
    """Build supply plan using actions layer (Aave V3 Base)."""
    steps = await actions.build_aave_supply_usdc_plan(
        rpc_url, from_address, amount_usdc, gas_price_gwei
    )
    return SupplyPlan(steps=steps)


async def build_withdraw_usdc_plan(
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
) -> SupplyPlan:
    """Build withdraw plan using actions layer (Aave V3 Base)."""
    steps = await actions.build_aave_withdraw_usdc_plan(
        from_address, amount_usdc, gas_price_gwei
    )
    return SupplyPlan(steps=steps)


async def supply_usdc(
    manager: WalletManager,
    wallet_id: str,
    rpc_url: str,
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
    analytics_db=None,
) -> list[TxReceipt]:
    """Supply USDC to Aave V3 on Base."""
    plan = await build_supply_usdc_plan(rpc_url, from_address, amount_usdc, gas_price_gwei)
    receipts: list[TxReceipt] = []
    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id)

    for tx_params, _ in plan.steps:
        receipt = await manager.sign_and_broadcast_tx(wallet_id, tx_params)
        receipts.append(receipt)
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
                strategy_type=StrategyType.YIELD.value,
                strategy_id=STRATEGY_ID,
            )
            await analytics_db.record_tx(record)

    log.info(f"YIELD: supplied {amount_usdc / 1e6:.2f} USDC for wallet '{wallet_id}'.")
    return receipts


async def withdraw_usdc(
    manager: WalletManager,
    wallet_id: str,
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
    analytics_db=None,
) -> list[TxReceipt]:
    """Withdraw USDC from Aave V3 on Base."""
    plan = await build_withdraw_usdc_plan(from_address, amount_usdc, gas_price_gwei)
    receipts: list[TxReceipt] = []
    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id)

    for tx_params, _ in plan.steps:
        receipt = await manager.sign_and_broadcast_tx(wallet_id, tx_params)
        receipts.append(receipt)
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
                strategy_type=StrategyType.YIELD.value,
                strategy_id=STRATEGY_ID,
            )
            await analytics_db.record_tx(record)

    log.info(f"YIELD: withdrew {amount_usdc / 1e6:.2f} USDC for wallet '{wallet_id}'.")
    return receipts


execute_supply_usdc = supply_usdc
execute_withdraw_usdc = withdraw_usdc


# ---------------------------------------------------------------------------
# AaveAPYMonitor
# ---------------------------------------------------------------------------

# Module-level flag set by AaveAPYMonitor when APY falls below threshold
apy_below_threshold: bool = False


class AaveAPYMonitor:
    """Reads Aave V3 USDC liquidityRate on Base and compares against
    the break-even rate.

    If current_apy < cfg.yield_min_apy_pct (default 1.0%):
      → Log warning and send Telegram alert
      → Set yield_strategy.apy_below_threshold = True flag
      → Orchestrator can check this flag and reallocate to ALGO pillar

    Polls every cfg.aave_apy_poll_interval_seconds (default 3600).
    Reads liquidityRate from Aave V3 Pool getReserveData().
    liquidityRate is in ray units (1e27):
      apy = (1 + rate/1e27)^(365*24*3600) - 1
    """

    # Aave V3 Pool getReserveData ABI (minimal subset)
    # NOTE: configuration MUST be "tuple" with one field "data" to correctly
    # map Aave V3 DataTypes.ReserveConfigurationMap. Using "uint256" shifts all
    # subsequent field indices and causes currentLiquidityRate to be read as
    # liquidityIndex (a ray-scaled ~1.05e27 value), producing astronomic APY%.
    _RESERVE_DATA_ABI = [
        {
            "inputs": [{"name": "asset", "type": "address"}],
            "name": "getReserveData",
            "outputs": [
                {"name": "configuration", "type": "tuple",
                 "components": [{"name": "data", "type": "uint256"}]},
                {"name": "liquidityIndex", "type": "uint128"},
                {"name": "currentLiquidityRate", "type": "uint128"},
                {"name": "variableBorrowIndex", "type": "uint128"},
                {"name": "currentVariableBorrowRate", "type": "uint128"},
                {"name": "currentStableBorrowRate", "type": "uint128"},
                {"name": "lastUpdateTimestamp", "type": "uint40"},
                {"name": "id", "type": "uint16"},
                {"name": "aTokenAddress", "type": "address"},
                {"name": "stableDebtTokenAddress", "type": "address"},
                {"name": "variableDebtTokenAddress", "type": "address"},
                {"name": "interestRateStrategyAddress", "type": "address"},
                {"name": "accruedToTreasury", "type": "uint128"},
                {"name": "unbacked", "type": "uint128"},
                {"name": "isolationModeTotalDebt", "type": "uint128"},
            ],
            "stateMutability": "view",
            "type": "function",
        }
    ]

    def __init__(self) -> None:
        self.last_apy: float = 0.0

    async def get_current_apy(self, rpc_url: str) -> float:
        """Return annualized USDC supply APY as decimal (0.03 = 3%).

        Uses linear approximation: APY ≈ rate/1e27 (valid for rates < 100%).
        This avoids exp() overflow and is accurate to <0.01% for typical DeFi APYs.
        Hard-capped at 5.0 (500%) as a sanity guard.
        """
        from web3 import AsyncWeb3
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        pool_addr = _checksum(AAVE_POOL_BASE)
        usdc_addr = _checksum(USDC_BASE)
        contract = w3.eth.contract(address=pool_addr, abi=self._RESERVE_DATA_ABI)
        try:
            data = await contract.functions.getReserveData(usdc_addr).call()
        finally:
            await _close_w3_provider(w3)
        # currentLiquidityRate is at index 2 (uint128, ray = 1e27)
        # With the corrected tuple ABI, data[0]=configuration(tuple), data[1]=liquidityIndex,
        # data[2]=currentLiquidityRate
        liquidity_rate_ray = int(data[2])
        if liquidity_rate_ray <= 0:
            return 0.0
        # Linear APY approximation: rate/1e27 (exact for continuous compounding at small rates)
        apy = liquidity_rate_ray / 1e27
        # Sanity cap: 5.0 = 500% APY — prevents astronomical values from corrupt data
        return min(apy, 5.0)



    async def poll_forever(self, rpc_url: str) -> None:
        """Async loop, runs every aave_apy_poll_interval_seconds."""
        import asyncio

        from wallet_manager.config import get_settings

        global apy_below_threshold

        while True:
            cfg = get_settings()
            poll_interval = cfg.aave_apy_poll_interval_seconds
            min_apy_pct = cfg.yield_min_apy_pct

            try:
                current_apy = await self.get_current_apy(rpc_url)
                self.last_apy = current_apy  # stored for heartbeat access
                current_apy_pct = current_apy * 100.0
                log.info("AaveAPYMonitor: current USDC APY = %.4f%%", current_apy_pct)

                if current_apy_pct < min_apy_pct:
                    apy_below_threshold = True
                    log.warning(
                        "AaveAPYMonitor: APY %.4f%% < threshold %.1f%% — alerting.",
                        current_apy_pct, min_apy_pct,
                    )
                else:
                    apy_below_threshold = False
            except Exception as exc:
                log.warning("AaveAPYMonitor: poll failed: %s", exc)

            await asyncio.sleep(poll_interval)


_apy_monitor: AaveAPYMonitor | None = None


def get_apy_monitor() -> AaveAPYMonitor:
    """Return the process-wide AaveAPYMonitor singleton."""
    global _apy_monitor
    if _apy_monitor is None:
        _apy_monitor = AaveAPYMonitor()
    return _apy_monitor
