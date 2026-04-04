"""
wallet_manager/strategies/liquidity_alpha.py

Advanced Execution & Liquidity Profiling for Uniswap V3.

Provides three core capabilities:

1. **Tick Liquidity Analysis** (`TickLiquidityProfiler`):
   - Queries on-chain tick data around the current price (+-50 ticks).
   - Computes liquidity depth at each tick level and an aggregate
     Liquidity Score (0-100).
   - If liquidity is "thin" (score < 30), downstream callers should
     reduce trade size regardless of Kelly signals.
   - Estimates expected slippage from the tick distribution.

2. **Split-Order Execution** (`SplitOrderEngine`):
   - If trade size > 1% of pool's estimated 1-minute volume, the order
     is split into 3 sub-orders sized [40%, 35%, 25%] (front-loaded).
   - Random Gaussian delays between sub-orders prevent timing leakage.
   - Each sub-order is independently routed through SmartOrderRouter.

3. **LiquidityAlpha** orchestrator:
   - Single entry-point for liquidity analysis, size reduction, and
     split execution.
   - Singleton via ``get_liquidity_alpha()`` / ``reset_liquidity_alpha()``.

Architecture
------------
- Python 3.11+, full asyncio.
- Uses web3.py for on-chain queries.
- Graceful RPC failure handling: returns conservative estimates.
- Pool address constants for Base Mainnet WETH/USDC pairs.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, cast

from web3 import AsyncWeb3, Web3

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default tick scan radius around the current tick
DEFAULT_TICK_RADIUS: int = 50

# Liquidity score thresholds
THIN_LIQUIDITY_THRESHOLD: int = 30
MODERATE_LIQUIDITY_THRESHOLD: int = 60

# Split-order parameters
SPLIT_VOLUME_THRESHOLD_PCT: float = 0.01   # 1% of estimated 1-min volume
DEFAULT_SPLIT_WEIGHTS: list[float] = [0.40, 0.35, 0.25]

# Gaussian jitter for inter-sub-order delays (milliseconds)
SPLIT_JITTER_MEAN_MS: float = 3000.0       # 3 seconds
SPLIT_JITTER_STD_MS: float = 1500.0        # 1.5 seconds
SPLIT_JITTER_MIN_MS: float = 1000.0        # 1 second minimum
SPLIT_JITTER_MAX_MS: float = 8000.0        # 8 seconds maximum

# Pool volume estimation
VOLUME_WINDOW_SECONDS: float = 60.0        # 1-minute rolling window
VOLUME_DEQUE_MAXLEN: int = 500             # Max swap events tracked

# Conservative fallback values when RPC fails
FALLBACK_LIQUIDITY_SCORE: int = 15         # Assume thin liquidity
FALLBACK_SLIPPAGE_BPS: float = 50.0        # 50 bps conservative estimate
FALLBACK_DEPTH_USD: float = 10_000.0       # $10k conservative depth

# Uniswap V3 math constants
Q96: int = 2**96
TICK_BASE: float = 1.0001

# Base Mainnet pool addresses
BASE_WETH_USDC_005: str = "0xd0b53D9277642d899DF5C87A3966A349A798F224"   # 0.05% fee
BASE_WETH_USDC_030: str = "0x4C36388bE6F416A29C8d8Ae5C112AB4cc73c0357"   # 0.30% fee

DEFAULT_POOL_ADDRESSES: dict[str, str] = {
    "WETH/USDC-500":  BASE_WETH_USDC_005,
    "WETH/USDC-3000": BASE_WETH_USDC_030,
}

# Reason codes
REASON_THIN_LIQUIDITY: str = "THIN_LIQUIDITY_REDUCTION"
REASON_SPLIT_TRIGGERED: str = "SPLIT_ORDER_TRIGGERED"
REASON_DEPTH_INSUFFICIENT: str = "POOL_DEPTH_INSUFFICIENT"
REASON_OK: str = "LIQUIDITY_OK"


# ---------------------------------------------------------------------------
# Minimal Uniswap V3 Pool ABI (inline)
# ---------------------------------------------------------------------------

UNISWAP_V3_POOL_ABI: list[dict[str, Any]] = [
    {
        "name": "liquidity",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint128"}],
    },
    {
        "name": "slot0",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [
            {"name": "sqrtPriceX96", "type": "uint160"},
            {"name": "tick", "type": "int24"},
            {"name": "observationIndex", "type": "uint16"},
            {"name": "observationCardinality", "type": "uint16"},
            {"name": "observationCardinalityNext", "type": "uint16"},
            {"name": "feeProtocol", "type": "uint8"},
            {"name": "unlocked", "type": "bool"},
        ],
    },
    {
        "name": "ticks",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "tick", "type": "int24"}],
        "outputs": [
            {"name": "liquidityGross", "type": "uint128"},
            {"name": "liquidityNet", "type": "int128"},
            {"name": "feeGrowthOutside0X128", "type": "uint256"},
            {"name": "feeGrowthOutside1X128", "type": "uint256"},
            {"name": "tickCumulativeOutside", "type": "int56"},
            {"name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
            {"name": "secondsOutside", "type": "uint32"},
            {"name": "initialized", "type": "bool"},
        ],
    },
    {
        "name": "fee",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint24"}],
    },
]


# ---------------------------------------------------------------------------
# Data Carriers (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickData:
    """Liquidity data for a single Uniswap V3 tick."""

    tick_idx: int
    liquidity_net: int
    liquidity_gross: int
    price_at_tick: float


@dataclass(frozen=True)
class LiquidityProfile:
    """Aggregate liquidity profile for a pool around the current price."""

    pool_address: str
    liquidity_score: int          # 0-100
    depth_usd: float              # Total USD depth in scanned range
    spread_bps: float             # Effective spread in basis points
    expected_slippage_bps: float  # Expected execution slippage
    tick_count: int               # Number of initialized ticks found
    is_thin: bool                 # True if liquidity_score < 30


@dataclass(frozen=True)
class SubOrderResult:
    """Execution result for a single sub-order."""

    sub_order_id: str
    size_usd: float
    size_pct: float
    delay_ms: float
    slippage_bps: float
    receipt: dict[str, Any] | None
    reason_code: str


@dataclass(frozen=True)
class SplitOrderResult:
    """Aggregate result for a split-order execution."""

    total_size_usd: float
    sub_orders: list[SubOrderResult]
    avg_slippage_bps: float
    total_delay_ms: float
    was_split: bool


# ---------------------------------------------------------------------------
# Uniswap V3 tick math helpers
# ---------------------------------------------------------------------------


def tick_to_price(tick: int) -> float:
    """Convert a Uniswap V3 tick index to a human-readable price ratio.

    Price = 1.0001^tick. Returns the price of token0 in terms of token1.
    """
    return TICK_BASE**tick


def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
    """Convert sqrtPriceX96 to a human-readable price.

    price = (sqrtPriceX96 / 2^96)^2
    """
    if sqrt_price_x96 <= 0:
        return 0.0
    ratio = sqrt_price_x96 / Q96
    return ratio * ratio


def price_to_usd_depth(
    liquidity_gross: int,
    tick_lower: int,
    tick_upper: int,
    current_price_usd: float,
) -> float:
    """Estimate USD depth contributed by liquidity between two ticks.

    Uses a simplified model: depth ~ liquidity * price_range * current_price.
    This is an approximation sufficient for scoring purposes.
    """
    if liquidity_gross <= 0 or current_price_usd <= 0.0:
        return 0.0
    price_low = tick_to_price(tick_lower)
    price_high = tick_to_price(tick_upper)
    if price_low >= price_high:
        return 0.0
    sqrt_low = math.sqrt(price_low)
    sqrt_high = math.sqrt(price_high)
    delta_sqrt = sqrt_high - sqrt_low
    # Approximate token0 and token1 amounts from liquidity
    # L * delta(sqrt(P)) gives token1 amount; L / delta(1/sqrt(P)) gives token0
    token1_amount = liquidity_gross * delta_sqrt
    depth = token1_amount * current_price_usd / 1e18  # Normalize from wei-scale
    return max(depth, 0.0)


# ---------------------------------------------------------------------------
# Gaussian Jitter (split-order variant)
# ---------------------------------------------------------------------------


class _SplitOrderJitter:
    """Gaussian jitter engine for inter-sub-order delays.

    Uses the same pattern as ``mev_shield.GaussianJitter`` but with parameters
    tuned for split-order execution: mean=3s, std=1.5s, clamped to [1s, 8s].
    """

    def __init__(
        self,
        mean_ms: float = SPLIT_JITTER_MEAN_MS,
        std_ms: float = SPLIT_JITTER_STD_MS,
        min_ms: float = SPLIT_JITTER_MIN_MS,
        max_ms: float = SPLIT_JITTER_MAX_MS,
    ) -> None:
        self._mean = mean_ms
        self._std = std_ms
        self._min = min_ms
        self._max = max_ms
        self._total_jitter_ms: float = 0.0
        self._jitter_count: int = 0

    async def apply(self) -> float:
        """Apply a random Gaussian jitter delay.

        Returns:
            Actual delay applied in milliseconds.
        """
        delay_ms = random.gauss(self._mean, self._std)
        delay_ms = max(self._min, min(self._max, delay_ms))

        # Add micro-offset for uniqueness (0.1-50ms)
        delay_ms += random.uniform(0.1, 50.0)

        await asyncio.sleep(delay_ms / 1000.0)

        self._total_jitter_ms += delay_ms
        self._jitter_count += 1
        return delay_ms

    @property
    def avg_jitter_ms(self) -> float:
        """Average jitter applied across all calls."""
        if self._jitter_count == 0:
            return 0.0
        return self._total_jitter_ms / self._jitter_count

    @property
    def total_jitter_seconds(self) -> float:
        """Total jitter applied in seconds."""
        return self._total_jitter_ms / 1000.0


# ---------------------------------------------------------------------------
# Pool Volume Estimator
# ---------------------------------------------------------------------------


class PoolVolumeEstimator:
    """Tracks recent swap events to estimate 1-minute trading volume.

    Maintains a rolling deque of ``(timestamp, volume_usd)`` entries.
    Volume older than ``VOLUME_WINDOW_SECONDS`` is pruned on every query.
    """

    def __init__(self, window_seconds: float = VOLUME_WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._events: deque[tuple[float, float]] = deque(
            maxlen=VOLUME_DEQUE_MAXLEN,
        )

    def record_swap(self, volume_usd: float, timestamp: float | None = None) -> None:
        """Record a swap event.

        Args:
            volume_usd: USD volume of the swap.
            timestamp:  Event timestamp (defaults to ``time.time()``).
        """
        ts = timestamp if timestamp is not None else time.time()
        self._events.append((ts, abs(volume_usd)))

    def _prune(self) -> None:
        """Remove entries older than the rolling window."""
        cutoff = time.time() - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def estimated_1min_volume(self) -> float:
        """Return estimated 1-minute volume in USD.

        If insufficient data, returns 0.0 (callers should treat this as
        unknown and fall back to conservative behaviour).
        """
        self._prune()
        if not self._events:
            return 0.0
        total = sum(vol for _, vol in self._events)
        # Scale to full minute if window is partially filled
        elapsed = time.time() - self._events[0][0]
        if elapsed < 1.0:
            return total
        scale = self._window / elapsed
        return total * min(scale, 5.0)  # Cap scaling to avoid wild extrapolation

    @property
    def event_count(self) -> int:
        """Number of swap events currently in the window."""
        self._prune()
        return len(self._events)


# ---------------------------------------------------------------------------
# Tick Liquidity Profiler
# ---------------------------------------------------------------------------


class TickLiquidityProfiler:
    """Analyzes Uniswap V3 concentrated liquidity around the current price.

    Queries on-chain tick data via web3 contract calls to the pool's
    ``liquidity()``, ``slot0()``, and ``ticks()`` functions.

    Computes:
    - Liquidity depth at each tick level.
    - Aggregate Liquidity Score (0-100): ratio of available liquidity
      vs. the intended trade size.
    - Expected slippage from the tick distribution.

    Gracefully handles RPC failures by returning conservative fallback
    estimates rather than raising exceptions.
    """

    def __init__(self, tick_radius: int = DEFAULT_TICK_RADIUS) -> None:
        self._tick_radius = tick_radius
        self._query_count: int = 0
        self._failure_count: int = 0

    async def profile(
        self,
        rpc_url: str,
        pool_address: str,
        trade_size_usd: float,
        eth_price_usd: float = 3000.0,
    ) -> LiquidityProfile:
        """Build a liquidity profile for the given pool.

        Args:
            rpc_url:         RPC endpoint URL.
            pool_address:    Checksummed Uniswap V3 pool address.
            trade_size_usd:  Intended trade size in USD.
            eth_price_usd:   Current ETH price for USD conversions.

        Returns:
            ``LiquidityProfile`` with score, depth, spread, and slippage.
        """
        self._query_count += 1
        try:
            return await self._profile_impl(
                rpc_url, pool_address, trade_size_usd, eth_price_usd,
            )
        except Exception as exc:
            self._failure_count += 1
            log.warning(
                "TickProfiler: RPC query failed for pool %s — returning "
                "conservative fallback. Error: %s",
                pool_address,
                exc,
            )
            return self._fallback_profile(pool_address)

    async def _profile_impl(
        self,
        rpc_url: str,
        pool_address: str,
        trade_size_usd: float,
        eth_price_usd: float,
    ) -> LiquidityProfile:
        """Internal implementation — may raise on RPC failure."""
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        pool = w3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=UNISWAP_V3_POOL_ABI,
        )

        # Fetch slot0 and global liquidity concurrently
        slot0_task = asyncio.create_task(pool.functions.slot0().call())
        liq_task = asyncio.create_task(pool.functions.liquidity().call())

        slot0_result, global_liquidity = await asyncio.gather(
            slot0_task, liq_task,
        )

        current_tick: int = slot0_result[1]

        # Scan ticks around current price
        tick_data_list = await self._scan_ticks(
            pool, current_tick, eth_price_usd,
        )

        # Compute depth and score
        total_depth_usd = self._compute_depth(
            tick_data_list, current_tick, eth_price_usd,
        )

        liquidity_score = self._compute_score(total_depth_usd, trade_size_usd)
        expected_slippage = self._estimate_slippage(
            tick_data_list, trade_size_usd, total_depth_usd, eth_price_usd,
        )

        # Spread approximation: fee tier from pool
        try:
            fee_raw: int = await pool.functions.fee().call()
            spread_bps = fee_raw / 100.0  # fee is in hundredths of a bip
        except Exception:
            spread_bps = 5.0  # Default 5 bps

        is_thin = liquidity_score < THIN_LIQUIDITY_THRESHOLD
        if is_thin:
            log.info(
                "TickProfiler: THIN liquidity detected for %s — "
                "score=%d, depth=$%.0f",
                pool_address,
                liquidity_score,
                total_depth_usd,
            )

        return LiquidityProfile(
            pool_address=pool_address,
            liquidity_score=liquidity_score,
            depth_usd=total_depth_usd,
            spread_bps=spread_bps,
            expected_slippage_bps=expected_slippage,
            tick_count=len(tick_data_list),
            is_thin=is_thin,
        )

    async def _scan_ticks(
        self,
        pool: Any,
        current_tick: int,
        eth_price_usd: float,
    ) -> list[TickData]:
        """Query tick data for ticks in [current - radius, current + radius].

        Ticks are queried in batched concurrent calls to limit RPC load.
        """
        tick_indices = list(
            range(
                current_tick - self._tick_radius,
                current_tick + self._tick_radius + 1,
            )
        )

        # Batch into groups of 20 to avoid overwhelming the RPC
        batch_size = 20
        results: list[TickData] = []

        for i in range(0, len(tick_indices), batch_size):
            batch = tick_indices[i : i + batch_size]
            tasks = [
                asyncio.create_task(self._query_single_tick(pool, t))
                for t in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for tick_idx, result in zip(batch, batch_results, strict=False):
                if isinstance(result, Exception):
                    log.debug(
                        "TickProfiler: Failed to query tick %d: %s",
                        tick_idx, result,
                    )
                    continue
                if result is not None:
                    results.append(cast(TickData, result))

        return results

    async def _query_single_tick(
        self,
        pool: Any,
        tick_idx: int,
    ) -> TickData | None:
        """Query a single tick from the pool contract."""
        try:
            tick_result = await pool.functions.ticks(tick_idx).call()
            liquidity_gross: int = tick_result[0]
            liquidity_net: int = tick_result[1]

            # Skip uninitialized ticks
            if liquidity_gross == 0 and liquidity_net == 0:
                return None

            price = tick_to_price(tick_idx)
            return TickData(
                tick_idx=tick_idx,
                liquidity_net=liquidity_net,
                liquidity_gross=liquidity_gross,
                price_at_tick=price,
            )
        except Exception:
            return None

    def _compute_depth(
        self,
        tick_data: list[TickData],
        current_tick: int,
        eth_price_usd: float,
    ) -> float:
        """Sum USD depth across all scanned ticks."""
        total_depth = 0.0
        for td in tick_data:
            depth = price_to_usd_depth(
                td.liquidity_gross,
                td.tick_idx,
                td.tick_idx + 1,
                eth_price_usd,
            )
            total_depth += depth
        return total_depth

    def _compute_score(
        self,
        depth_usd: float,
        trade_size_usd: float,
    ) -> int:
        """Compute Liquidity Score (0-100).

        Score = min(100, (depth_usd / trade_size_usd) * 10).

        A score of 100 means the pool has at least 10x the trade size in
        available liquidity. A score < 30 is considered "thin".
        """
        if trade_size_usd <= 0.0:
            return 100
        if depth_usd <= 0.0:
            return 0
        raw = (depth_usd / trade_size_usd) * 10.0
        return min(100, max(0, int(raw)))

    def _estimate_slippage(
        self,
        tick_data: list[TickData],
        trade_size_usd: float,
        total_depth_usd: float,
        eth_price_usd: float,
    ) -> float:
        """Estimate expected slippage in basis points.

        Uses a simplified model: slippage ~ (trade_size / depth)^0.6 * 100 bps.
        The exponent < 1 accounts for partial fill across multiple ticks.
        """
        if total_depth_usd <= 0.0 or trade_size_usd <= 0.0:
            return FALLBACK_SLIPPAGE_BPS
        ratio = trade_size_usd / total_depth_usd
        slippage_bps = (ratio ** 0.6) * 100.0
        return float(round(min(slippage_bps, 500.0), 2))  # Cap at 500 bps

    def _fallback_profile(self, pool_address: str) -> LiquidityProfile:
        """Return a conservative fallback profile on RPC failure."""
        return LiquidityProfile(
            pool_address=pool_address,
            liquidity_score=FALLBACK_LIQUIDITY_SCORE,
            depth_usd=FALLBACK_DEPTH_USD,
            spread_bps=30.0,
            expected_slippage_bps=FALLBACK_SLIPPAGE_BPS,
            tick_count=0,
            is_thin=True,
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Return profiler diagnostics."""
        return {
            "query_count": self._query_count,
            "failure_count": self._failure_count,
            "failure_rate": (
                self._failure_count / self._query_count
                if self._query_count > 0
                else 0.0
            ),
            "tick_radius": self._tick_radius,
        }


# ---------------------------------------------------------------------------
# Split-Order Execution Engine
# ---------------------------------------------------------------------------


class SplitOrderEngine:
    """Intelligent order splitting for large trades.

    If a trade exceeds 1% of the pool's estimated 1-minute volume, the
    engine splits it into sub-orders sized [40%, 35%, 25%] with random
    Gaussian delays between each.  Each sub-order is independently routed
    through ``SmartOrderRouter``.
    """

    def __init__(
        self,
        split_weights: list[float] | None = None,
        volume_threshold_pct: float = SPLIT_VOLUME_THRESHOLD_PCT,
    ) -> None:
        self._weights = split_weights or list(DEFAULT_SPLIT_WEIGHTS)
        self._volume_threshold = volume_threshold_pct
        self._jitter = _SplitOrderJitter()
        self._executions: int = 0
        self._splits_triggered: int = 0

    def should_split(
        self,
        trade_size_usd: float,
        estimated_1min_volume: float,
    ) -> bool:
        """Determine whether a trade should be split.

        Returns ``True`` if trade_size_usd > 1% of estimated_1min_volume,
        or if volume data is unavailable (conservative approach).
        """
        if estimated_1min_volume <= 0.0:
            # Unknown volume — split to be safe if trade is non-trivial
            return trade_size_usd > 500.0
        threshold = estimated_1min_volume * self._volume_threshold
        return trade_size_usd > threshold

    async def execute(
        self,
        rpc_url: str,
        token_in: str,
        token_out: str,
        total_amount_in: int,
        total_size_usd: float,
        gas_price_gwei: float = 0.5,
        estimated_1min_volume: float = 0.0,
    ) -> SplitOrderResult:
        """Execute a (potentially split) order.

        If splitting is not required, executes as a single order.

        Args:
            rpc_url:                RPC endpoint.
            token_in:               Input token address.
            token_out:              Output token address.
            total_amount_in:        Total input amount in smallest units.
            total_size_usd:         Total trade size in USD.
            gas_price_gwei:         Current gas price.
            estimated_1min_volume:  Pool's estimated 1-minute volume.

        Returns:
            ``SplitOrderResult`` with per-sub-order details.
        """
        from wallet_manager.strategies.smart_order_router import (
            get_smart_order_router,
        )

        self._executions += 1
        sor = get_smart_order_router()
        do_split = self.should_split(total_size_usd, estimated_1min_volume)

        if do_split:
            self._splits_triggered += 1
            log.info(
                "SplitOrder: Splitting $%.0f into %d sub-orders "
                "(1-min vol est: $%.0f)",
                total_size_usd,
                len(self._weights),
                estimated_1min_volume,
            )

        weights = self._weights if do_split else [1.0]
        sub_results: list[SubOrderResult] = []
        total_delay_ms: float = 0.0

        for idx, weight in enumerate(weights):
            sub_id = f"sub-{uuid.uuid4().hex[:8]}"
            sub_amount = int(total_amount_in * weight)
            sub_size_usd = total_size_usd * weight

            # Apply jitter delay between sub-orders (not before the first)
            delay_ms = 0.0
            if idx > 0:
                delay_ms = await self._jitter.apply()
                total_delay_ms += delay_ms
                log.debug(
                    "SplitOrder: Sub-order %s delayed %.0fms",
                    sub_id, delay_ms,
                )

            # Route through SOR
            receipt: dict[str, Any] | None = None
            slippage_bps: float = 0.0
            reason_code = REASON_SPLIT_TRIGGERED if do_split else REASON_OK

            try:
                route = await sor.find_best_route(
                    rpc_url=rpc_url,
                    token_in=token_in,
                    token_out=token_out,
                    amount_in=sub_amount,
                    gas_price_gwei=gas_price_gwei,
                )
                if route is not None and route.best_quote is not None:
                    # Estimate slippage from quote vs mid-price
                    expected = sub_amount  # Simplified: 1:1 baseline
                    actual = route.best_quote.amount_out
                    if expected > 0 and actual > 0:
                        slippage_bps = max(
                            0.0,
                            (1.0 - actual / expected) * 10_000,
                        )
                    receipt = {
                        "venue": str(route.best_quote.venue),
                        "amount_out": route.best_quote.amount_out,
                        "gas_estimate": getattr(
                            route.best_quote, "gas_estimate", None,
                        ),
                    }
                else:
                    reason_code = REASON_DEPTH_INSUFFICIENT
                    slippage_bps = FALLBACK_SLIPPAGE_BPS
                    log.warning(
                        "SplitOrder: No route found for sub-order %s "
                        "(size=$%.0f)",
                        sub_id, sub_size_usd,
                    )
            except Exception as exc:
                reason_code = REASON_DEPTH_INSUFFICIENT
                slippage_bps = FALLBACK_SLIPPAGE_BPS
                log.error(
                    "SplitOrder: Sub-order %s failed: %s", sub_id, exc,
                )

            sub_results.append(
                SubOrderResult(
                    sub_order_id=sub_id,
                    size_usd=sub_size_usd,
                    size_pct=weight,
                    delay_ms=delay_ms,
                    slippage_bps=round(slippage_bps, 2),
                    receipt=receipt,
                    reason_code=reason_code,
                )
            )

        # Aggregate
        total_slippage = sum(
            sr.slippage_bps * sr.size_pct for sr in sub_results
        )
        avg_slippage = round(total_slippage, 2)

        return SplitOrderResult(
            total_size_usd=total_size_usd,
            sub_orders=sub_results,
            avg_slippage_bps=avg_slippage,
            total_delay_ms=round(total_delay_ms, 1),
            was_split=do_split,
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Return engine diagnostics."""
        return {
            "executions": self._executions,
            "splits_triggered": self._splits_triggered,
            "split_rate": (
                self._splits_triggered / self._executions
                if self._executions > 0
                else 0.0
            ),
            "jitter_avg_ms": self._jitter.avg_jitter_ms,
            "jitter_total_s": self._jitter.total_jitter_seconds,
            "weights": list(self._weights),
        }


# ---------------------------------------------------------------------------
# LiquidityAlpha — Main Orchestrator
# ---------------------------------------------------------------------------


class LiquidityAlpha:
    """Unified entry-point for liquidity analysis and split-order execution.

    Combines ``TickLiquidityProfiler``, ``PoolVolumeEstimator``, and
    ``SplitOrderEngine`` into a single interface.

    Usage::

        alpha = get_liquidity_alpha()

        profile = await alpha.analyze_liquidity(
            rpc_url="https://mainnet.base.org",
            pool_address=BASE_WETH_USDC_005,
            trade_size_usd=5000.0,
        )

        reduce, multiplier = alpha.should_reduce_size(profile)
        if reduce:
            trade_size_usd *= multiplier

        result = await alpha.execute_split_order(
            rpc_url=rpc_url,
            pool_address=pool_address,
            token_in=WETH,
            token_out=USDC,
            total_amount_in=amount_wei,
            total_size_usd=trade_size_usd,
        )
    """

    def __init__(self) -> None:
        self._profiler = TickLiquidityProfiler()
        self._volume_estimators: dict[str, PoolVolumeEstimator] = {}
        self._split_engine = SplitOrderEngine()
        self._profiles_cache: dict[str, tuple[float, LiquidityProfile]] = {}
        self._cache_ttl: float = 15.0  # Cache profiles for 15 seconds
        self._analysis_count: int = 0
        self._reduction_count: int = 0

    # ----- Volume tracking -----

    def _get_volume_estimator(self, pool_address: str) -> PoolVolumeEstimator:
        """Get or create a volume estimator for the given pool."""
        key = pool_address.lower()
        if key not in self._volume_estimators:
            self._volume_estimators[key] = PoolVolumeEstimator()
        return self._volume_estimators[key]

    def record_swap_event(
        self,
        pool_address: str,
        volume_usd: float,
        timestamp: float | None = None,
    ) -> None:
        """Record a swap event for volume estimation.

        Should be called by upstream event listeners whenever a Swap
        event is observed on a tracked pool.

        Args:
            pool_address: Pool contract address.
            volume_usd:   USD volume of the swap.
            timestamp:    Event timestamp (defaults to now).
        """
        estimator = self._get_volume_estimator(pool_address)
        estimator.record_swap(volume_usd, timestamp)

    # ----- Liquidity analysis -----

    async def analyze_liquidity(
        self,
        rpc_url: str,
        pool_address: str,
        trade_size_usd: float,
        eth_price_usd: float = 3000.0,
    ) -> LiquidityProfile:
        """Analyze pool liquidity around the current price.

        Results are cached for ``_cache_ttl`` seconds to avoid redundant
        RPC queries during rapid-fire signal evaluation.

        Args:
            rpc_url:         RPC endpoint.
            pool_address:    Uniswap V3 pool address.
            trade_size_usd:  Intended trade size in USD.
            eth_price_usd:   Current ETH price for conversions.

        Returns:
            ``LiquidityProfile`` with score, depth, and slippage estimates.
        """
        self._analysis_count += 1
        cache_key = f"{pool_address.lower()}:{int(trade_size_usd)}"

        # Check cache
        cached = self._profiles_cache.get(cache_key)
        if cached is not None:
            ts, profile = cached
            if time.time() - ts < self._cache_ttl:
                log.debug(
                    "LiquidityAlpha: Cache hit for %s (score=%d)",
                    pool_address, profile.liquidity_score,
                )
                return profile

        profile = await self._profiler.profile(
            rpc_url, pool_address, trade_size_usd, eth_price_usd,
        )
        self._profiles_cache[cache_key] = (time.time(), profile)

        log.info(
            "LiquidityAlpha: Pool %s — score=%d, depth=$%.0f, "
            "slippage=%.1f bps, thin=%s",
            pool_address,
            profile.liquidity_score,
            profile.depth_usd,
            profile.expected_slippage_bps,
            profile.is_thin,
        )
        return profile

    def should_reduce_size(
        self,
        profile: LiquidityProfile,
    ) -> tuple[bool, float]:
        """Determine whether trade size should be reduced.

        Args:
            profile: Previously computed liquidity profile.

        Returns:
            Tuple of (should_reduce, multiplier).
            - ``multiplier`` is in [0.0, 1.0].
            - If ``should_reduce`` is False, ``multiplier`` is 1.0.
        """
        if not profile.is_thin:
            return (False, 1.0)

        self._reduction_count += 1

        # Linear scaling: score 0 -> multiplier 0.1, score 30 -> multiplier 1.0
        score = max(0, profile.liquidity_score)
        multiplier = 0.1 + (score / THIN_LIQUIDITY_THRESHOLD) * 0.9
        multiplier = round(min(1.0, max(0.1, multiplier)), 3)

        log.info(
            "LiquidityAlpha: %s — score=%d, reducing size by %.0f%% "
            "(multiplier=%.3f)",
            REASON_THIN_LIQUIDITY,
            profile.liquidity_score,
            (1.0 - multiplier) * 100,
            multiplier,
        )
        return (True, multiplier)

    # ----- Split-order execution -----

    async def execute_split_order(
        self,
        rpc_url: str,
        pool_address: str,
        token_in: str,
        token_out: str,
        total_amount_in: int,
        total_size_usd: float,
        gas_price_gwei: float = 0.5,
    ) -> SplitOrderResult:
        """Execute a trade with automatic split-order logic.

        Checks pool volume to decide whether splitting is needed, then
        delegates to ``SplitOrderEngine``.

        Args:
            rpc_url:          RPC endpoint.
            pool_address:     Pool address (for volume lookup).
            token_in:         Input token address.
            token_out:        Output token address.
            total_amount_in:  Total input amount in smallest units.
            total_size_usd:   Total trade size in USD.
            gas_price_gwei:   Current gas price.

        Returns:
            ``SplitOrderResult`` with per-sub-order details.
        """
        estimator = self._get_volume_estimator(pool_address)
        est_volume = estimator.estimated_1min_volume()

        log.info(
            "LiquidityAlpha: execute_split_order — size=$%.0f, "
            "est_1min_vol=$%.0f, pool=%s",
            total_size_usd,
            est_volume,
            pool_address,
        )

        result = await self._split_engine.execute(
            rpc_url=rpc_url,
            token_in=token_in,
            token_out=token_out,
            total_amount_in=total_amount_in,
            total_size_usd=total_size_usd,
            gas_price_gwei=gas_price_gwei,
            estimated_1min_volume=est_volume,
        )

        if result.was_split:
            log.info(
                "LiquidityAlpha: Split execution complete — "
                "%d sub-orders, avg_slippage=%.1f bps, "
                "total_delay=%.0f ms",
                len(result.sub_orders),
                result.avg_slippage_bps,
                result.total_delay_ms,
            )
        return result

    # ----- Diagnostics -----

    def get_liquidity_stats(self) -> dict[str, Any]:
        """Return comprehensive diagnostic statistics.

        Includes profiler, split-engine, and volume estimator metrics.
        """
        volume_stats: dict[str, Any] = {}
        for addr, estimator in self._volume_estimators.items():
            volume_stats[addr] = {
                "est_1min_volume": estimator.estimated_1min_volume(),
                "event_count": estimator.event_count,
            }

        return {
            "analysis_count": self._analysis_count,
            "reduction_count": self._reduction_count,
            "reduction_rate": (
                self._reduction_count / self._analysis_count
                if self._analysis_count > 0
                else 0.0
            ),
            "profiler": self._profiler.stats,
            "split_engine": self._split_engine.stats,
            "volume_estimators": volume_stats,
            "cache_size": len(self._profiles_cache),
            "cache_ttl_seconds": self._cache_ttl,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LiquidityAlpha | None = None


def get_liquidity_alpha() -> LiquidityAlpha:
    """Return the process-wide ``LiquidityAlpha`` singleton."""
    global _instance
    if _instance is None:
        _instance = LiquidityAlpha()
        log.info("LiquidityAlpha: Singleton initialized.")
    return _instance


def reset_liquidity_alpha() -> None:
    """Destroy the current singleton (useful for testing)."""
    global _instance
    if _instance is not None:
        log.info("LiquidityAlpha: Singleton reset.")
    _instance = None
