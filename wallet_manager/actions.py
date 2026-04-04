"""
wallet_manager/actions.py

Airdrop Farming & Swap actions — Uniswap V3 integration for Linea and Base.

Provides ETH <-> USDC swap functions with slippage protection and chain-specific
contract addresses. Designed for integration with the WalletManager and orchestrator.
"""

# mypy: disable-error-code="call-overload,arg-type"

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from wallet_manager.core.models import TxParams
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from web3 import AsyncWeb3

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Chain-specific contract addresses (Linea, Base)
# ---------------------------------------------------------------------------

# Uniswap V3 SwapRouter02
ROUTER_LINEA = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"
ROUTER_BASE = "0x2626664c2603336E57B271c5C0b26F421741e481"

# Uniswap V3 QuoterV2 (chain-specific)
QUOTER_LINEA = "0x42bE4D6527829FeFA1493e1fb9F3676d2425C3C1"
QUOTER_BASE = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"

# WETH (Wrapped Native)
WETH_LINEA = "0xe5D7C2a44FfDDf6b295A15c148167daaAf5Cf34f"
WETH_BASE = "0x4200000000000000000000000000000000000006"

# USDC (6 decimals)
USDC_LINEA = "0x176211869cA2b568f2A7D4EE941E073a821EE1ff"
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# ---------------------------------------------------------------------------
# Aave V3 Base (Chain ID 8453) - Production addresses from aave-address-book
# ---------------------------------------------------------------------------
AAVE_POOL_BASE = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"
ATOKEN_USDC_BASE = "0x4e65fe4dba92790696d040ac24aa414708f5c0ab"  # aBaseUSDC

# Aave V3 Pool ABI fragments (Base) — supply, withdraw, getReserveData
AAVE_V3_POOL_ABI = [
    {
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "onBehalfOf", "type": "address"},
            {"name": "referralCode", "type": "uint16"},
        ],
        "name": "supply",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "to", "type": "address"},
        ],
        "name": "withdraw",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "asset", "type": "address"}],
        "name": "getReserveData",
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
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# Uniswap V3 WETH/USDC 0.3% pool on Base (for slot0 price feed)
UNISWAP_V3_WETH_USDC_POOL_BASE = "0xd0b53d9277642d899df5c87a3966a349a798f224"
# FIX: pre-checksum the pool address to avoid per-call 'Unknown format' warnings
try:
    from web3 import Web3 as _Web3
    UNISWAP_V3_WETH_USDC_POOL_BASE = _Web3.to_checksum_address(UNISWAP_V3_WETH_USDC_POOL_BASE)
except Exception:
    pass  # web3 not available at import time; _checksum() will handle it at call time
CHAINLINK_WETH_USD_BASE = "0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70"

# Uniswap V3 pool fee tiers (basis points: 500=0.05%, 3000=0.3%, 10000=1%)
# Rotate for Ghost Protocol — different paths reduce fingerprint correlation
POOL_FEE = 3000
FEE_TIERS = (500, 3000, 10000)  # 0.05%, 0.3%, 1% — use for path rotation

# Default slippage 0.5%
DEFAULT_SLIPPAGE_BPS = 50  # 0.5% = 50 basis points

# Slippage bounds for dynamic calculation
MIN_SLIPPAGE_BPS = 20   # 0.2% floor
MAX_SLIPPAGE_BPS = 150  # 1.5% ceiling for airdrop
ALGO_MAX_SLIPPAGE_BPS = 50  # 0.5% cap for algo (MEV protection)

# ERC20 ABI fragments
ERC20_APPROVE_SELECTOR = "0x095ea7b3"  # approve(address,uint256)
ERC20_DECIMALS = 6  # USDC
WETH_DECIMALS = 18

# Minimum ETH balance threshold (wei) ~0.003 ETH for ~$10 at ~$3300
MIN_ETH_VALUE_USD = 10.0

# Mempool safety: deadline (now + 15 min) — only if router supports it.
# SwapRouter02 (used on Linea/Base) does NOT have deadline in exactInputSingle.
# Original ISwapRouter has it; set True if using a router that supports deadline.
DEADLINE_SUPPORTED = False
DEADLINE_BUFFER_SECONDS = 15 * 60


class SwapDirection(Enum):
    """Swap direction enumeration."""

    ETH_TO_USDC = "ETH_TO_USDC"
    USDC_TO_ETH = "USDC_TO_ETH"


@dataclass
class ChainAddresses:
    """Contract addresses for a single chain."""

    chain_id: int
    router: str
    quoter: str
    weth: str
    usdc: str


_CHAIN_ADDRESSES: dict[int, ChainAddresses] = {
    59144: ChainAddresses(
        chain_id=59144,
        router=ROUTER_LINEA,
        quoter=QUOTER_LINEA,
        weth=WETH_LINEA,
        usdc=USDC_LINEA,
    ),
    8453: ChainAddresses(
        chain_id=8453,
        router=ROUTER_BASE,
        quoter=QUOTER_BASE,
        weth=WETH_BASE,
        usdc=USDC_BASE,
    ),
}


def _get_addresses(chain_id: int) -> ChainAddresses:
    """Return chain addresses for the given chain_id."""
    addrs = _CHAIN_ADDRESSES.get(chain_id)
    if addrs is None:
        raise ValueError(f"Chain {chain_id} not supported. Supported: 59144 (Linea), 8453 (Base).")
    return addrs


def get_chain_addresses(chain_id: int) -> ChainAddresses:
    """Public accessor for chain addresses. Use for dynamic slippage and swap plans."""
    return _get_addresses(chain_id)


def _checksum(addr: str) -> str:
    """Return checksummed address."""
    from web3 import Web3

    return Web3.to_checksum_address(addr)


# Timeout for RPC calls (seconds)
_RPC_REQUEST_TIMEOUT: float = 15.0


_shared_providers: dict[str, object] = {}

def _make_w3(rpc_url: str) -> AsyncWeb3:
    """Create or retrieve an AsyncWeb3 instance with a reused HTTP provider cache to avoid session leaks."""
    from web3 import AsyncWeb3
    
    if rpc_url not in _shared_providers:
        _shared_providers[rpc_url] = AsyncWeb3.AsyncHTTPProvider(
            rpc_url,
            request_kwargs={"timeout": _RPC_REQUEST_TIMEOUT},
        )
    return AsyncWeb3(_shared_providers[rpc_url])


async def _get_allowance(
    rpc_url: str,
    token_address: str,
    owner: str,
    spender: str,
) -> int:
    """Return ERC20 allowance(owner, spender)."""

    allowance_abi = [
        {
            "inputs": [
                {"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"},
            ],
            "name": "allowance",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    w3 = _make_w3(rpc_url)
    contract = w3.eth.contract(
        address=_checksum(token_address),
        abi=allowance_abi,
    )
    return int(await contract.functions.allowance(
        _checksum(owner),
        _checksum(spender),
    ).call())


async def fetch_native_balance(rpc_url: str, address: str) -> int:
    """Fetch native (ETH) balance in wei."""

    w3 = _make_w3(rpc_url)
    return int(await w3.eth.get_balance(_checksum(address)))


async def fetch_usdc_balance(rpc_url: str, address: str, chain_id: int) -> int:
    """Fetch ERC20 USDC balance for the given address on the given chain.

    Returns:
        USDC balance in 6-decimal units (e.g. 1_000_000 = 1 USDC).
    """

    addrs = _get_addresses(chain_id)
    balance_abi = [
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    w3 = _make_w3(rpc_url)
    contract = w3.eth.contract(
        address=_checksum(addrs.usdc),
        abi=balance_abi,
    )
    return int(await contract.functions.balanceOf(_checksum(address)).call())


async def fetch_eth_price_usd() -> float:
    """Fetch current ETH price in USD with retry (fallback to conservative default)."""
    import aiohttp

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
              async with session.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
                timeout=aiohttp.ClientTimeout(total=8.0),
              ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data.get("ethereum", {}).get("usd", 0))
                    if price > 0:
                        return price
                if resp.status == 429:
                    # Rate limited — back off
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
        except Exception as exc:
            if attempt < 2:
                import asyncio
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            log.info(f"ETH price fetch unavailable after 3 attempts ({type(exc).__name__}), using fallback $3000")
    return 3000.0  # Conservative fallback


async def quote_amount_out(
    rpc_url: str,
    chain_id: int,
    token_in: str,
    token_out: str,
    amount_in: int,
    fee: int = POOL_FEE,
) -> int:
    """Quote expected amount out via Uniswap V3 QuoterV2 (chain-specific) or estimate."""

    addrs = _get_addresses(chain_id)
    quoter_abi = [
        {
            "inputs": [
                {
                    "components": [
                        {"name": "tokenIn", "type": "address"},
                        {"name": "tokenOut", "type": "address"},
                        {"name": "amountIn", "type": "uint256"},
                        {"name": "fee", "type": "uint24"},
                        {"name": "sqrtPriceLimitX96", "type": "uint160"},
                    ],
                    "name": "params",
                    "type": "tuple",
                }
            ],
            "name": "quoteExactInputSingle",
            "outputs": [
                {"name": "amountOut", "type": "uint256"},
                {"name": "sqrtPriceX96After", "type": "uint160"},
                {"name": "initializedTicksCrossed", "type": "uint32"},
                {"name": "gasEstimate", "type": "uint256"},
            ],
            "stateMutability": "nonpayable",
            "type": "function",
        }
    ]
    try:
        w3 = _make_w3(rpc_url)
        quoter = w3.eth.contract(
            address=_checksum(addrs.quoter),
            abi=quoter_abi,
        )
        params = (
            _checksum(token_in),
            _checksum(token_out),
            amount_in,
            fee,
            0,
        )
        result = await quoter.functions.quoteExactInputSingle(params).call()
        if result and len(result) > 0:
            return int(result[0])
    except Exception as exc:
        log.debug(f"Quoter call failed ({chain_id}), using estimate: {exc}")

    # Fallback: rough estimate from chain addresses
    addrs = _get_addresses(chain_id)
    if token_in.lower() == addrs.weth.lower():
        eth_price = await fetch_eth_price_usd()
        return int(amount_in * eth_price / 1e18 * 1e6)
    if token_in.lower() == addrs.usdc.lower():
        eth_price = await fetch_eth_price_usd()
        return int(amount_in / 1e6 * 1e18 / eth_price)
    return 0


async def get_pool_liquidity(rpc_url: str, chain_id: int, fee: int = POOL_FEE) -> int:
    """
    Fetch Uniswap V3 WETH/USDC pool liquidity (L) for Base.
    Lower L = thinner pool = need higher slippage.
    """
    if chain_id != CHAIN_ID_BASE:
        return 0

    # WETH/USDC 0.3% pool on Base
    pool_addr = UNISWAP_V3_WETH_USDC_POOL_BASE if fee == 3000 else None
    if not pool_addr:
        return 0
    try:
        liquidity_abi = [
            {"inputs": [], "name": "liquidity", "outputs": [{"name": "", "type": "uint128"}], "stateMutability": "view", "type": "function"},
        ]
        w3 = _make_w3(rpc_url)
        pool = w3.eth.contract(address=_checksum(pool_addr), abi=liquidity_abi)
        return int(await pool.functions.liquidity().call())
    except Exception as exc:
        log.debug(f"Pool liquidity fetch failed: {exc}")
    return 0


async def get_dynamic_slippage(
    rpc_url: str,
    chain_id: int,
    token_in: str,
    token_out: str,
    amount_in: int,
    gas_price_gwei: float,
    fee: int = POOL_FEE,
    is_algo: bool = False,
) -> int:
    """
    Compute slippage in bps based on pool liquidity and volatility.
    Never use fixed slippage — adapt to market conditions for MEV protection.

    Args:
        rpc_url: JSON-RPC endpoint.
        chain_id: EVM chain ID.
        token_in: Input token address.
        token_out: Output token address.
        amount_in: Swap amount in token's smallest units.
        gas_price_gwei: Current gas price (volatility proxy).
        fee: Uniswap fee tier.
        is_algo: If True, cap at 0.5% for stricter MEV protection.

    Returns:
        Slippage in basis points (e.g. 50 = 0.5%).
    """
    base_bps = MIN_SLIPPAGE_BPS
    # Volatility proxy: high gas = congested mempool = higher frontrun risk
    if gas_price_gwei > 15:
        base_bps += 25
    elif gas_price_gwei > 10:
        base_bps += 12

    # Pool liquidity: lower L = thinner pool = more price impact
    try:
        liq = await get_pool_liquidity(rpc_url, chain_id, fee)
        if liq > 0:
            # Heuristic: if amount_in is large relative to liquidity, add slippage
            # L in Uniswap V3 is sqrt(x*y); rough scale: 1e18 ~ $1M depth
            if amount_in > liq // 100:  # >1% of liquidity
                base_bps += 30
            elif amount_in > liq // 500:
                base_bps += 15
    except Exception:
        pass

    # Size factor: large swaps need more slippage
    if amount_in > 50 * 10**18 or amount_in > 50 * 10**6:  # ~50 ETH or 50 USDC
        base_bps += 20

    cap = ALGO_MAX_SLIPPAGE_BPS if is_algo else MAX_SLIPPAGE_BPS
    return min(max(base_bps, MIN_SLIPPAGE_BPS), cap)


@dataclass
class SwapStep:
    """A single transaction step within a swap flow."""

    tx_params: TxParams
    description: str


@dataclass
class SwapPlan:
    """Full swap plan — ordered list of transaction steps."""

    direction: SwapDirection
    chain_id: int
    steps: list[SwapStep] = field(default_factory=list)
    amount_in_wei: int = 0
    amount_out_min: int = 0


def get_rotated_fee_tier(trade_count: int = 0) -> int:
    """
    Rotate fee tier for Ghost Protocol — different paths reduce fingerprint correlation.
    Returns 0.05%, 0.3%, or 1% fee based on trade_count (round-robin).
    """
    return FEE_TIERS[trade_count % len(FEE_TIERS)]


async def build_eth_to_usdc_plan(
    rpc_url: str,
    chain_id: int,
    from_address: str,
    amount_wei: int,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    gas_price_gwei: float = 25.0,
    fee: int | None = None,
) -> SwapPlan:
    """
    Build a swap plan for ETH -> USDC.

    Steps: 1) WETH.deposit, 2) WETH.approve(router), 3) router.exactInputSingle
    """
    addrs = _get_addresses(chain_id)
    from_addr = _checksum(from_address)
    fee_tier = fee if fee is not None else POOL_FEE

    # Quote expected USDC output
    amount_out_estimate = await quote_amount_out(
        rpc_url, chain_id, addrs.weth, addrs.usdc, amount_wei, fee_tier
    )
    amount_out_min = int(amount_out_estimate * (10000 - slippage_bps) / 10000)
    if amount_out_min < 0:
        amount_out_min = 0

    gas_price_wei = int(gas_price_gwei * 1e9)
    gas_limit_deposit = 100_000
    gas_limit_approve = 80_000
    gas_limit_swap = 250_000

    steps: list[SwapStep] = []

    # Step 1: WETH.deposit{value: amount_wei}
    steps.append(
        SwapStep(
            tx_params=TxParams(
                from_address=from_addr,
                to=addrs.weth,
                value=amount_wei,
                gas=gas_limit_deposit,
                gas_price_wei=gas_price_wei,
                data=bytes.fromhex("d0e30db0"),  # deposit()
            ),
            description="WETH.deposit",
        )
    )

    # Step 2: WETH.approve(router) — only if allowance insufficient
    current_allowance = await _get_allowance(
        rpc_url, addrs.weth, from_addr, addrs.router
    )
    if current_allowance < amount_wei:
        approve_data = _build_approve_calldata(addrs.router, amount_wei)
        steps.append(
            SwapStep(
                tx_params=TxParams(
                    from_address=from_addr,
                    to=addrs.weth,
                    value=0,
                    gas=gas_limit_approve,
                    gas_price_wei=gas_price_wei,
                    data=approve_data,
                ),
                description="WETH.approve(router)",
            )
        )

    # Step 3: router.exactInputSingle (deadline = now + 15 min if router supports it)
    deadline = int(time.time()) + DEADLINE_BUFFER_SECONDS if DEADLINE_SUPPORTED else None
    swap_data = _build_exact_input_single_calldata(
        token_in=addrs.weth,
        token_out=addrs.usdc,
        fee=fee_tier,
        recipient=from_addr,
        amount_in=amount_wei,
        amount_out_min=amount_out_min,
        deadline=deadline,
    )
    steps.append(
        SwapStep(
            tx_params=TxParams(
                from_address=from_addr,
                to=addrs.router,
                value=0,
                gas=gas_limit_swap,
                gas_price_wei=gas_price_wei,
                data=swap_data,
            ),
            description="router.exactInputSingle(WETH->USDC)",
        )
    )

    return SwapPlan(
        direction=SwapDirection.ETH_TO_USDC,
        chain_id=chain_id,
        steps=steps,
        amount_in_wei=amount_wei,
        amount_out_min=amount_out_min,
    )


def _build_approve_calldata(spender: str, amount: int) -> bytes:
    """Build ERC20 approve calldata using web3."""
    from eth_abi import encode

    selector = bytes.fromhex("095ea7b3")
    encoded = encode(["address", "uint256"], [_checksum(spender), amount])
    return selector + encoded


def _build_exact_input_single_calldata(
    token_in: str,
    token_out: str,
    fee: int,
    recipient: str,
    amount_in: int,
    amount_out_min: int,
    sqrt_price_limit: int = 0,
    deadline: int | None = None,
) -> bytes:
    """Build exactInputSingle calldata using web3 ABI encoding.

    When DEADLINE_SUPPORTED is True and deadline is set, encodes the 8-parameter
    struct (ISwapRouter) for mempool safety: tx reverts if not mined before deadline.
    SwapRouter02 uses 7 params (no deadline); set DEADLINE_SUPPORTED=False for it.
    """
    from eth_abi import encode

    # Function selector for exactInputSingle
    selector = bytes.fromhex("04e45aaf")
    params: tuple[object, ...]
    if DEADLINE_SUPPORTED and deadline is not None:
        # Original ISwapRouter: tokenIn, tokenOut, fee, recipient, deadline, amountIn, amountOutMin, sqrtPriceLimit
        params = (
            _checksum(token_in),
            _checksum(token_out),
            fee,
            _checksum(recipient),
            deadline,
            amount_in,
            amount_out_min,
            sqrt_price_limit,
        )
        encoded = encode(
            ["address", "address", "uint24", "address", "uint256", "uint256", "uint256", "uint160"],
            list(params),
        )
    else:
        # SwapRouter02: no deadline
        params = (
            _checksum(token_in),
            _checksum(token_out),
            fee,
            _checksum(recipient),
            amount_in,
            amount_out_min,
            sqrt_price_limit,
        )
        encoded = encode(
            ["address", "address", "uint24", "address", "uint256", "uint256", "uint160"],
            list(params),
        )
    return selector + encoded


async def build_usdc_to_eth_plan(
    rpc_url: str,
    chain_id: int,
    from_address: str,
    amount_usdc: int,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    gas_price_gwei: float = 25.0,
    fee: int | None = None,
) -> SwapPlan:
    """
    Build a swap plan for USDC -> WETH.

    Steps: 1) USDC.approve(router) [only if allowance insufficient],
           2) router.exactInputSingle (USDC->WETH).
    WETH is kept (no unwrap) — airdrop farming benefits from WETH; saves gas.
    """
    addrs = _get_addresses(chain_id)
    from_addr = _checksum(from_address)
    fee_tier = fee if fee is not None else POOL_FEE

    # amount_usdc is in 6 decimals
    amount_out_estimate = await quote_amount_out(
        rpc_url, chain_id, addrs.usdc, addrs.weth, amount_usdc, fee_tier
    )
    amount_out_min = int(amount_out_estimate * (10000 - slippage_bps) / 10000)
    if amount_out_min < 0:
        amount_out_min = 0

    gas_price_wei = int(gas_price_gwei * 1e9)
    gas_limit_approve = 80_000
    gas_limit_swap = 300_000

    steps: list[SwapStep] = []

    # Step 1: USDC.approve(router) — only if allowance insufficient
    current_allowance = await _get_allowance(
        rpc_url, addrs.usdc, from_addr, addrs.router
    )
    if current_allowance < amount_usdc:
        steps.append(
            SwapStep(
                tx_params=TxParams(
                    from_address=from_addr,
                    to=addrs.usdc,
                    value=0,
                    gas=gas_limit_approve,
                    gas_price_wei=gas_price_wei,
                    data=_build_approve_calldata(addrs.router, amount_usdc),
                ),
                description="USDC.approve(router)",
            )
        )

    # Step 2: router.exactInputSingle(USDC->WETH) (deadline if supported)
    deadline = int(time.time()) + DEADLINE_BUFFER_SECONDS if DEADLINE_SUPPORTED else None
    steps.append(
        SwapStep(
            tx_params=TxParams(
                from_address=from_addr,
                to=addrs.router,
                value=0,
                gas=gas_limit_swap,
                gas_price_wei=gas_price_wei,
                data=_build_exact_input_single_calldata(
                    token_in=addrs.usdc,
                    token_out=addrs.weth,
                    fee=fee_tier,
                    recipient=from_addr,
                    amount_in=amount_usdc,
                    amount_out_min=amount_out_min,
                    deadline=deadline,
                ),
            ),
            description="router.exactInputSingle(USDC->WETH)",
        )
    )

    return SwapPlan(
        direction=SwapDirection.USDC_TO_ETH,
        chain_id=chain_id,
        steps=steps,
        amount_in_wei=amount_usdc,
        amount_out_min=amount_out_min,
    )


def eth_balance_usd_value(balance_wei: int, eth_price_usd: float) -> float:
    """Convert wei balance to approximate USD value."""
    return balance_wei * eth_price_usd / 1e18


def usdc_balance_usd_value(balance_usdc_units: int) -> float:
    """Convert USDC (6 decimals) balance to approximate USD value (1 USDC ≈ 1 USD)."""
    return balance_usdc_units / 1e6


def is_swap_eligible(
    balance_wei: int = 0,
    eth_price_usd: float = 0.0,
    gas_gwei: float = 0.0,
    min_eth_value_usd: float = MIN_ETH_VALUE_USD,
    max_gas_gwei: float = 25.0,
    eth_balance_wei: int | None = None,   # alias for balance_wei
    gas_price_gwei: float | None = None,  # alias for gas_gwei
    min_trade_usd: float | None = None,   # alias for min_eth_value_usd
) -> bool:
    """Check if wallet is eligible for ETH->USDC swap (balance > $10, gas < 25 Gwei)."""
    if eth_balance_wei is not None:
        balance_wei = eth_balance_wei
    if gas_price_gwei is not None:
        gas_gwei = gas_price_gwei
    if min_trade_usd is not None:
        min_eth_value_usd = min_trade_usd
    value_usd = eth_balance_usd_value(balance_wei, eth_price_usd)
    return value_usd >= min_eth_value_usd and gas_gwei < max_gas_gwei


def is_usdc_swap_eligible(
    balance_usdc: int,
    gas_gwei: float,
    min_usdc_value_usd: float = MIN_ETH_VALUE_USD,
    max_gas_gwei: float = 25.0,
) -> bool:
    """Check if wallet is eligible for USDC->ETH swap (USDC > $10, gas < 25 Gwei)."""
    value_usd = usdc_balance_usd_value(balance_usdc)
    return value_usd >= min_usdc_value_usd and gas_gwei < max_gas_gwei


# ---------------------------------------------------------------------------
# Price feeds for Algo strategy (Uniswap V3 slot0 + Chainlink fallback)
# ---------------------------------------------------------------------------

CHAIN_ID_BASE = 8453
CHAINLINK_USD_DECIMALS = 8


async def get_live_price_v3(rpc_url: str) -> float:
    """
    Fetch WETH/USDC live price from Uniswap V3 slot0 on Base.

    Zero-latency on-chain price for algo trading. Uses the WETH/USDC 0.3%
    pool on Base. Alias for get_weth_usdc_price for clarity in trading context.

    Args:
        rpc_url: JSON-RPC endpoint for Base (chain_id 8453).

    Returns:
        WETH price in USD (e.g. 2500.0), or 0.0 on failure.
    """
    return await get_weth_usdc_price(rpc_url)


async def get_weth_usdc_price(rpc_url: str) -> float:
    """
    Fetch WETH/USDC price directly from Uniswap V3 pool slot0 on Base.

    Uses the WETH/USDC 0.3% pool on Base (UNISWAP_V3_WETH_USDC_POOL_BASE).
    Pool: token0=WETH(18), token1=USDC(6).
    sqrtPriceX96 = sqrt(token1/token0) * 2^96.

    Args:
        rpc_url: JSON-RPC endpoint for Base.

    Returns:
        WETH price in USD (e.g. 2500.0), or 0.0 on failure.
    """

    slot0_abi = [
        {
            "inputs": [],
            "name": "slot0",
            "outputs": [
                {"name": "sqrtPriceX96", "type": "uint160"},
                {"name": "tick", "type": "int24"},
                {"name": "observationIndex", "type": "uint16"},
                {"name": "observationCardinality", "type": "uint16"},
                {"name": "observationCardinalityNext", "type": "uint16"},
                {"name": "feeProtocol", "type": "uint8"},
                {"name": "unlocked", "type": "bool"},
            ],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    try:
        w3 = _make_w3(rpc_url)
        pool = w3.eth.contract(
            address=_checksum(UNISWAP_V3_WETH_USDC_POOL_BASE),
            abi=slot0_abi,
        )
        result = await pool.functions.slot0().call()
        sqrt_price_x96 = int(result[0])
        if sqrt_price_x96 <= 0:
            return 0.0
        q96 = 2**96
        raw = (sqrt_price_x96 / q96) ** 2
        # USDC (6 dec) per WETH (18 dec)
        price = raw * (1e18 / 1e6)
        return float(price)
    except Exception as exc:
        log.info(f"WETH/USDC price from slot0 unavailable ({type(exc).__name__}), using fallback")
        return 0.0


async def get_current_price(
    rpc_url: str,
    chain_id: int = CHAIN_ID_BASE,
    use_slot0: bool = True,
) -> float:
    """
    Fetch WETH/USDC spot price (1 WETH in USD).

    Primary: Uniswap V3 slot0 on-chain (Base). Fallback: Quoter, then Chainlink.

    Args:
        rpc_url: JSON-RPC endpoint.
        chain_id: EVM chain (8453 for Base).
        use_slot0: If True, use Uniswap pool slot0 for on-chain price.

    Returns:
        WETH price in USD (e.g. 2500.0).
    """

    if chain_id == CHAIN_ID_BASE and use_slot0:
        try:
            slot0_abi = [
                {
                    "inputs": [],
                    "name": "slot0",
                    "outputs": [
                        {"name": "sqrtPriceX96", "type": "uint160"},
                        {"name": "tick", "type": "int24"},
                        {"name": "observationIndex", "type": "uint16"},
                        {"name": "observationCardinality", "type": "uint16"},
                        {"name": "observationCardinalityNext", "type": "uint16"},
                        {"name": "feeProtocol", "type": "uint8"},
                        {"name": "unlocked", "type": "bool"},
                    ],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
            w3 = _make_w3(rpc_url)
            pool = w3.eth.contract(
                address=_checksum(UNISWAP_V3_WETH_USDC_POOL_BASE),
                abi=slot0_abi,
            )
            result = await pool.functions.slot0().call()
            sqrt_price_x96 = int(result[0])
            if sqrt_price_x96 > 0:
                # Pool: token0=WETH(18), token1=USDC(6). sqrtPriceX96 = sqrt(token1/token0) * 2^96.
                # (sqrtPriceX96/2^96)^2 = USDC_base/WETH_base. USDC per 1 WETH = raw * 1e18/1e6.
                q96 = 2**96
                raw = (sqrt_price_x96 / q96) ** 2
                price = raw * (1e18 / 1e6)  # USDC (6 dec) per WETH (18 dec)
                return float(price)
        except Exception as exc:
            log.debug(f"slot0 price fetch failed: {exc}. Falling back to Quoter.")

    addrs = _get_addresses(chain_id)
    try:
        amount_weth = 10**18
        amount_usdc = await quote_amount_out(
            rpc_url, chain_id,
            addrs.weth, addrs.usdc,
            amount_weth, POOL_FEE,
        )
        if amount_usdc > 0:
            return amount_usdc / 1e6
    except Exception as exc:
        log.debug(f"Quoter price failed: {exc}. Falling back to Chainlink.")

    return await _fetch_chainlink_eth_usd(rpc_url)


async def _fetch_chainlink_eth_usd(rpc_url: str) -> float:
    """Fetch ETH/USD from Chainlink on Base."""

    abi = [
        {
            "inputs": [],
            "name": "latestRoundData",
            "outputs": [
                {"name": "roundId", "type": "uint80"},
                {"name": "answer", "type": "int256"},
                {"name": "startedAt", "type": "uint256"},
                {"name": "updatedAt", "type": "uint256"},
                {"name": "answeredInRound", "type": "uint80"},
            ],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    try:
        w3 = _make_w3(rpc_url)
        feed = w3.eth.contract(
            address=_checksum(CHAINLINK_WETH_USD_BASE),
            abi=abi,
        )
        result = await feed.functions.latestRoundData().call()
        answer = int(result[1])
        if answer > 0:
            return float(answer) / float(10**CHAINLINK_USD_DECIMALS)
    except Exception as exc:
        log.info(f"Chainlink WETH/USD unavailable ({type(exc).__name__}), using fallback")
    return await fetch_eth_price_usd()


# ---------------------------------------------------------------------------
# Aave V3 Base — supply_usdc / withdraw_usdc for Yield Strategy (50% allocation)
# ---------------------------------------------------------------------------

_AAVE_SUPPLY_SELECTOR = bytes.fromhex("617ba037")  # supply(address,uint256,address,uint16)
_AAVE_WITHDRAW_SELECTOR = bytes.fromhex("69328dec")  # withdraw(address,uint256,address)


def _build_aave_supply_calldata(asset: str, amount: int, on_behalf_of: str) -> bytes:
    """Build Aave Pool.supply(asset, amount, onBehalfOf, referralCode) calldata."""
    from eth_abi import encode
    return _AAVE_SUPPLY_SELECTOR + encode(
        ["address", "uint256", "address", "uint16"],
        [_checksum(asset), amount, _checksum(on_behalf_of), 0],
    )


def _build_aave_withdraw_calldata(asset: str, amount: int, to: str) -> bytes:
    """Build Aave Pool.withdraw(asset, amount, to) calldata."""
    from eth_abi import encode
    return _AAVE_WITHDRAW_SELECTOR + encode(
        ["address", "uint256", "address"],
        [_checksum(asset), amount, _checksum(to)],
    )


async def build_aave_supply_usdc_plan(
    rpc_url: str,
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
) -> list[tuple[TxParams, str]]:
    """
    Build plan to supply USDC to Aave V3 on Base.

    Steps: approve (if needed) + Pool.supply(USDC, amount, onBehalfOf, 0).
    Uses AAVE_POOL_BASE and USDC_BASE addresses.
    """
    from_addr = _checksum(from_address)
    gas_wei = int(gas_price_gwei * 1e9)
    steps: list[tuple[TxParams, str]] = []

    current_allowance = await _get_allowance(
        rpc_url, USDC_BASE, from_addr, AAVE_POOL_BASE
    )
    if current_allowance < amount_usdc:
        steps.append((
            TxParams(
                from_address=from_addr,
                to=_checksum(USDC_BASE),
                value=0,
                gas=80_000,
                gas_price_wei=gas_wei,
                data=_build_approve_calldata(AAVE_POOL_BASE, amount_usdc),
            ),
            "USDC.approve(AavePool)",
        ))

    steps.append((
        TxParams(
            from_address=from_addr,
            to=_checksum(AAVE_POOL_BASE),
            value=0,
            gas=200_000,
            gas_price_wei=gas_wei,
            data=_build_aave_supply_calldata(USDC_BASE, amount_usdc, from_addr),
        ),
        "AavePool.supply(USDC)",
    ))
    return steps


async def build_aave_withdraw_usdc_plan(
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
) -> list[tuple[TxParams, str]]:
    """
    Build plan to withdraw USDC from Aave V3 on Base.

    Step: Pool.withdraw(USDC, amount, to).
    """
    from_addr = _checksum(from_address)
    gas_wei = int(gas_price_gwei * 1e9)
    return [
        (
            TxParams(
                from_address=from_addr,
                to=_checksum(AAVE_POOL_BASE),
                value=0,
                gas=150_000,
                gas_price_wei=gas_wei,
                data=_build_aave_withdraw_calldata(USDC_BASE, amount_usdc, from_addr),
            ),
            "AavePool.withdraw(USDC)",
        )
    ]


@dataclass
class AaveStats:
    """Aave V3 yield statistics for tracking."""

    supply_apy_pct: float
    supplied_usdc: float
    address: str


# Aave yield optimization: threshold below which we alert (2% institutional floor)
AAVE_APY_ALERT_THRESHOLD_PCT = 2.0

# Fallback "alternative" yield when we can't fetch Compound/Stable-LP on-chain
# Set COMPOUND_USDC_BASE / STABLE_LP_ADDRESS in env for auto-switch if desired
DEFAULT_ALTERNATIVE_APY_PCT = 3.5


# Compound III USDC on Base (cUSDCv3)
COMPOUND_V3_CUSDC_BASE = "0xb125E6687d4313864e53df431d5425969c15Eb2"

# ABI fragments for Compound III comet (cUSDCv3)
COMPOUND_V3_ABI = [
    {
        "inputs": [],
        "name": "getUtilization",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "utilization", "type": "uint256"}],
        "name": "getSupplyRate",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


async def _get_compound_v3_apy(rpc_url: str) -> float | None:
    """
    Fetch Compound III (cUSDCv3) supply APY on Base.

    Returns APY as a percentage (e.g. 4.2 for 4.2%) or None on any failure.
    """
    from web3 import AsyncWeb3
    from web3.providers import AsyncHTTPProvider
    from wallet_manager.utils.web3_utils import close_w3_provider

    w3 = None
    try:
        w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
        comet = w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(COMPOUND_V3_CUSDC_BASE),
            abi=COMPOUND_V3_ABI,
        )
        utilization: int = await comet.functions.getUtilization().call()
        rate_per_second: int = await comet.functions.getSupplyRate(utilization).call()
        # rate_per_second is in 1e18 scale; annualise to percentage
        seconds_per_year = 365 * 24 * 3600
        apy_pct = (rate_per_second / 1e18) * seconds_per_year * 100
        return float(apy_pct)
    except Exception as exc:
        log.warning("Compound V3 APY fetch failed: %s", exc)
        return None
    finally:
        if w3 is not None:
            await close_w3_provider(w3)


async def get_alternative_stable_yield_apy(rpc_url: str, chain_id: int = CHAIN_ID_BASE) -> tuple[float, str]:
    """
    Fetch best alternative USDC yield when Aave is below threshold.
    Returns (apy_pct, protocol_name). On Base, tries Compound V3 first;
    falls back to conservative constant only on failure.
    """
    apy = await _get_compound_v3_apy(rpc_url)
    if apy is not None:
        return apy, "Compound V3"
    return DEFAULT_ALTERNATIVE_APY_PCT, "Compound/Stable-LP"


async def get_aave_stats(
    rpc_url: str,
    address: str,
) -> AaveStats | None:
    """
    Fetch Aave V3 yield statistics for the given address on Base.

    Returns supply APY, supplied USDC balance, for yield tracking in daily summary.

    Args:
        rpc_url: JSON-RPC endpoint for Base.
        address: User address to query.

    Returns:
        AaveStats with apy, supplied amount, or None on failure.
    """
    from wallet_manager.config import get_settings

    # Paper trade mode: return mock Aave stats without hitting on-chain
    _exec_mode = get_settings().execution_mode
    if _exec_mode == "robust_paper":
        return AaveStats(
            supply_apy_pct=3.25,
            supplied_usdc=0.0,
            address=address,
        )


    try:
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
        balance_abi = [
            {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf",
             "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}
        ]
        w3 = _make_w3(rpc_url)
        pool = w3.eth.contract(address=_checksum(AAVE_POOL_BASE), abi=pool_abi)
        data = await pool.functions.getReserveData(_checksum(USDC_BASE)).call()

        # data[2] is currentLiquidityRate (ray, 1e27 precision)
        rate_ray_raw = data[2] if len(data) > 2 else None
        if rate_ray_raw is None:
            log.info("get_aave_stats: getReserveData returned no liquidity rate")
            return AaveStats(supply_apy_pct=0.0, supplied_usdc=0.0, address=address)
        rate_ray = int(rate_ray_raw)
        apy_bps = float(rate_ray * 10000 / 1e27)
        apy_pct = apy_bps / 100.0

        a_token = w3.eth.contract(
            address=_checksum(ATOKEN_USDC_BASE), abi=balance_abi
        )
        supplied = await a_token.functions.balanceOf(_checksum(address)).call()
        supplied_usdc = int(supplied) / 1e6 if supplied is not None else 0.0
        return AaveStats(
            supply_apy_pct=apy_pct,
            supplied_usdc=supplied_usdc,
            address=address,
        )
    except Exception as exc:
        log.info(f"get_aave_stats: Aave connection unavailable ({type(exc).__name__}: {exc}), returning empty stats")
        return AaveStats(supply_apy_pct=0.0, supplied_usdc=0.0, address=address)


async def supply_usdc(
    manager: WalletManager,
    wallet_id: str,
    rpc_url: str,
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
    analytics_db: object | None = None,
) -> list:
    """Supply USDC to Aave V3 on Base. Delegates to yield_strategy for execution."""
    from wallet_manager.strategies import yield_strategy
    return await yield_strategy.supply_usdc(
        manager, wallet_id, rpc_url, from_address,
        amount_usdc, gas_price_gwei, analytics_db=analytics_db,
    )


async def withdraw_usdc(
    manager: WalletManager,
    wallet_id: str,
    from_address: str,
    amount_usdc: int,
    gas_price_gwei: float,
    analytics_db: object | None = None,
) -> list:
    """Withdraw USDC from Aave V3 on Base. Delegates to yield_strategy for execution."""
    from wallet_manager.strategies import yield_strategy
    return await yield_strategy.withdraw_usdc(
        manager, wallet_id, from_address,
        amount_usdc, gas_price_gwei, analytics_db=analytics_db,
    )


async def get_historical_prices(
    rpc_url: str,
    chain_id: int,
    count: int = 20,
) -> list[float]:
    """
    Fetch last N WETH/USDC prices for Bollinger Bands.

    On-chain slot0/Quoter only provide current price. This implementation
    returns [current] repeated count times when no history exists; the
    algo strategy maintains its own JSON-based price history. For a
    production system, consider a subgraph or block-scanned history.

    Args:
        rpc_url: JSON-RPC endpoint.
        chain_id: EVM chain ID.
        count: Number of prices to return (default 20 for BB period).

    Returns:
        List of prices, newest last. Length <= count.
    """
    current = await get_current_price(rpc_url, chain_id)
    if current <= 0:
        return []
    return [current] * min(count, 1)


# ---------------------------------------------------------------------------
# Aave V3 Borrow/Repay — Synthetic Short Support (Base)
# ---------------------------------------------------------------------------

# Variable-rate debt token for WETH on Aave V3 Base
VARIABLE_DEBT_WETH_BASE = "0x24e6e0795b3c7c71D965fCc4f371803d1c1DcA1E"

AAVE_V3_BORROW_ABI = [
    {
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "interestRateMode", "type": "uint256"},
            {"name": "referralCode", "type": "uint16"},
            {"name": "onBehalfOf", "type": "address"},
        ],
        "name": "borrow",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "interestRateMode", "type": "uint256"},
            {"name": "onBehalfOf", "type": "address"},
        ],
        "name": "repay",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

# ABI for getUserAccountData — returns health factor, LTV, etc.
AAVE_V3_ACCOUNT_ABI = [
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "getUserAccountData",
        "outputs": [
            {"name": "totalCollateralBase", "type": "uint256"},
            {"name": "totalDebtBase", "type": "uint256"},
            {"name": "availableBorrowsBase", "type": "uint256"},
            {"name": "currentLiquidationThreshold", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "healthFactor", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


async def get_aave_account_data(rpc_url: str, address: str) -> dict:
    """Fetch Aave V3 user account data: collateral, debt, health factor.

    Returns dict with keys: total_collateral_usd, total_debt_usd,
    available_borrows_usd, ltv, health_factor (1e18 scaled).
    """

    w3 = _make_w3(rpc_url)
    pool = w3.eth.contract(
        address=_checksum(AAVE_POOL_BASE),
        abi=AAVE_V3_ACCOUNT_ABI,
    )
    data = await pool.functions.getUserAccountData(_checksum(address)).call()
    # Aave returns values in Base currency units (8 decimals for USD)
    return {
        "total_collateral_usd": int(data[0]) / 1e8,
        "total_debt_usd": int(data[1]) / 1e8,
        "available_borrows_usd": int(data[2]) / 1e8,
        "ltv": int(data[4]),  # in basis points (e.g. 8000 = 80%)
        "health_factor": int(data[5]) / 1e18,  # >1 = safe, <1 = liquidatable
    }


async def get_variable_debt_balance(rpc_url: str, address: str) -> int:
    """Fetch variable-rate WETH debt balance from Aave V3 on Base.

    Returns debt in wei (18 decimals).
    """

    balance_abi = [
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    w3 = _make_w3(rpc_url)
    contract = w3.eth.contract(
        address=_checksum(VARIABLE_DEBT_WETH_BASE),
        abi=balance_abi,
    )
    return int(await contract.functions.balanceOf(_checksum(address)).call())


async def build_aave_borrow_eth_plan(
    rpc_url: str,
    from_address: str,
    borrow_amount_wei: int,
    gas_price_gwei: float,
) -> list[tuple[TxParams, str]]:
    """Build transaction plan to borrow WETH from Aave V3 on Base.

    Borrow uses variable rate (interestRateMode=2).
    Returns list of (TxParams, description) tuples.
    """

    w3 = _make_w3(rpc_url)
    pool = w3.eth.contract(
        address=_checksum(AAVE_POOL_BASE),
        abi=AAVE_V3_BORROW_ABI,
    )
    gas_price_wei = int(gas_price_gwei * 1e9)
    from_addr = _checksum(from_address)

    # Borrow WETH (variable rate = 2)
    borrow_data = pool.functions.borrow(
        _checksum(WETH_BASE),
        borrow_amount_wei,
        2,  # variable rate
        0,  # referral code
        from_addr,
    ).build_transaction({
        "from": from_addr,
        "gas": 300_000,
        "gasPrice": gas_price_wei,
        "value": 0,
    })

    steps = [
        (
            TxParams(
                from_address=from_addr,
                to=_checksum(AAVE_POOL_BASE),
                value=0,
                gas=300_000,
                gas_price_wei=gas_price_wei,
                data=bytes.fromhex(borrow_data["data"][2:]),
            ),
            "Aave V3: Borrow WETH (variable rate)",
        ),
    ]
    return steps


async def build_aave_repay_eth_plan(
    rpc_url: str,
    from_address: str,
    repay_amount_wei: int,
    gas_price_gwei: float,
) -> list[tuple[TxParams, str]]:
    """Build transaction plan to repay WETH debt on Aave V3 Base.

    Steps: 1) Approve WETH to Pool, 2) Repay.
    Use repay_amount_wei = 2**256 - 1 for full repayment.
    Returns list of (TxParams, description) tuples.
    """

    w3 = _make_w3(rpc_url)
    pool = w3.eth.contract(
        address=_checksum(AAVE_POOL_BASE),
        abi=AAVE_V3_BORROW_ABI,
    )
    gas_price_wei = int(gas_price_gwei * 1e9)
    from_addr = _checksum(from_address)
    steps: list[tuple[TxParams, str]] = []

    # Step 1: Approve WETH to Aave Pool (if needed)
    current_allowance = await _get_allowance(
        rpc_url, WETH_BASE, from_addr, AAVE_POOL_BASE
    )
    if current_allowance < repay_amount_wei:
        approve_abi = [
            {
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function",
            }
        ]
        weth_contract = w3.eth.contract(
            address=_checksum(WETH_BASE), abi=approve_abi
        )
        max_uint = 2**256 - 1
        approve_data = weth_contract.functions.approve(
            _checksum(AAVE_POOL_BASE), max_uint
        ).build_transaction({
            "from": from_addr,
            "gas": 80_000,
            "gasPrice": gas_price_wei,
            "value": 0,
        })
        steps.append((
            TxParams(
                from_address=from_addr,
                to=_checksum(WETH_BASE),
                value=0,
                gas=80_000,
                gas_price_wei=gas_price_wei,
                data=bytes.fromhex(approve_data["data"][2:]),
            ),
            "WETH: Approve Aave Pool for repay",
        ))

    # Step 2: Repay WETH debt
    repay_data = pool.functions.repay(
        _checksum(WETH_BASE),
        repay_amount_wei,
        2,  # variable rate
        from_addr,
    ).build_transaction({
        "from": from_addr,
        "gas": 250_000,
        "gasPrice": gas_price_wei,
        "value": 0,
    })
    steps.append((
        TxParams(
            from_address=from_addr,
            to=_checksum(AAVE_POOL_BASE),
            value=0,
            gas=250_000,
            gas_price_wei=gas_price_wei,
            data=bytes.fromhex(repay_data["data"][2:]),
        ),
        "Aave V3: Repay WETH debt",
    ))
    return steps
