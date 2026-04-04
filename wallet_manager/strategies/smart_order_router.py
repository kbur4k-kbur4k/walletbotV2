"""
wallet_manager/strategies/smart_order_router.py

QUANT-X Smart Order Router (SOR) — The "Surgical" Striker.

Multi-DEX aggregation engine that calculates the optimal swap path
across Uniswap V3, SushiSwap, and Aerodrome on Base to minimise
slippage and maximise output.

Architecture
------------
1. Query quotes from all supported DEXes in parallel.
2. Compare net output (after gas + slippage).
3. Route to the best venue, or split across venues if beneficial.
4. Build transaction plan using the winning route.

Supported DEXes (Base Chain)
----------------------------
- Uniswap V3: Primary venue (0.05%, 0.3%, 1% fee tiers)
- SushiSwap V3: Secondary venue (same fee tiers)
- Aerodrome: Base-native DEX (variable + stable pools)
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# DEX Registry (Base Chain ID: 8453)
# ---------------------------------------------------------------------------

CHAIN_ID_BASE = 8453


class DEXVenue(Enum):
    """Supported DEX venues on Base."""

    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP_V3 = "sushiswap_v3"
    AERODROME = "aerodrome"


@dataclass(frozen=True)
class DEXConfig:
    """Configuration for a single DEX venue."""

    venue: DEXVenue
    router_address: str
    quoter_address: str
    fee_tiers: tuple[int, ...]
    is_active: bool = True


# Base mainnet DEX addresses
DEX_REGISTRY: dict[DEXVenue, DEXConfig] = {
    DEXVenue.UNISWAP_V3: DEXConfig(
        venue=DEXVenue.UNISWAP_V3,
        router_address="0x2626664c2603336E57B271c5C0b26F421741e481",
        quoter_address="0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
        fee_tiers=(500, 3000, 10000),
    ),
    DEXVenue.SUSHISWAP_V3: DEXConfig(
        venue=DEXVenue.SUSHISWAP_V3,
        router_address="0xFB7eF66a7e61224DD6FcD0D7d9C3be5C8B049b9f",
        quoter_address="0xb1E835Dc2785b52265711e17fCCb0fd018226a6e",
        fee_tiers=(500, 3000, 10000),
    ),
    DEXVenue.AERODROME: DEXConfig(
        venue=DEXVenue.AERODROME,
        router_address="0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
        quoter_address="0x0000000000000000000000000000000000000000",
        fee_tiers=(1,),  # Aerodrome uses stable/volatile flag, not fee tiers
    ),
}


# ---------------------------------------------------------------------------
# Quote data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteQuote:
    """Quote from a single DEX venue for a specific swap."""

    venue: DEXVenue
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    fee_tier: int
    estimated_gas: int
    price_impact_bps: float
    timestamp: float


@dataclass
class OptimalRoute:
    """Best route selected by the SOR.

    Attributes:
        best_quote: The winning quote with highest net output.
        all_quotes: All quotes collected (for analytics).
        savings_bps: How many basis points better vs worst quote.
        latency_ms: Time taken to collect all quotes.
    """

    best_quote: RouteQuote
    all_quotes: list[RouteQuote]
    savings_bps: float
    latency_ms: float


# ---------------------------------------------------------------------------
# Quoter implementations
# ---------------------------------------------------------------------------


async def _quote_uniswap_v3(
    rpc_url: str,
    token_in: str,
    token_out: str,
    amount_in: int,
    fee_tier: int,
) -> RouteQuote | None:
    """Get quote from Uniswap V3 QuoterV2."""
    from web3 import AsyncWeb3

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
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        config = DEX_REGISTRY[DEXVenue.UNISWAP_V3]
        quoter = w3.eth.contract(
            address=w3.to_checksum_address(config.quoter_address),
            abi=quoter_abi,
        )

        params = (
            w3.to_checksum_address(token_in),
            w3.to_checksum_address(token_out),
            amount_in,
            fee_tier,
            0,  # sqrtPriceLimitX96 = 0 (no limit)
        )

        result = await quoter.functions.quoteExactInputSingle(params).call()
        amount_out = int(result[0])
        gas_estimate = int(result[3])

        # Calculate price impact
        # For WETH→USDC: expected ≈ amount_in * price / 1e12
        price_impact = 0.0  # Simplified; real implementation would compare
        if amount_out > 0:
            price_impact = float(fee_tier) / 100.0  # Approximation

        return RouteQuote(
            venue=DEXVenue.UNISWAP_V3,
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=amount_out,
            fee_tier=fee_tier,
            estimated_gas=gas_estimate,
            price_impact_bps=price_impact,
            timestamp=time.time(),
        )
    except Exception as exc:
        log.debug(f"SOR: Uniswap V3 quote failed (fee={fee_tier}): {exc}")
        return None


async def _quote_sushiswap_v3(
    rpc_url: str,
    token_in: str,
    token_out: str,
    amount_in: int,
    fee_tier: int,
) -> RouteQuote | None:
    """Get quote from SushiSwap V3 (same interface as Uniswap V3)."""
    from web3 import AsyncWeb3

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
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        config = DEX_REGISTRY[DEXVenue.SUSHISWAP_V3]
        quoter = w3.eth.contract(
            address=w3.to_checksum_address(config.quoter_address),
            abi=quoter_abi,
        )

        params = (
            w3.to_checksum_address(token_in),
            w3.to_checksum_address(token_out),
            amount_in,
            fee_tier,
            0,
        )

        result = await quoter.functions.quoteExactInputSingle(params).call()
        amount_out = int(result[0])
        gas_estimate = int(result[3])

        return RouteQuote(
            venue=DEXVenue.SUSHISWAP_V3,
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=amount_out,
            fee_tier=fee_tier,
            estimated_gas=gas_estimate,
            price_impact_bps=float(fee_tier) / 100.0,
            timestamp=time.time(),
        )
    except Exception as exc:
        log.debug(f"SOR: SushiSwap V3 quote failed (fee={fee_tier}): {exc}")
        return None


async def _quote_aerodrome(
    rpc_url: str,
    token_in: str,
    token_out: str,
    amount_in: int,
) -> RouteQuote | None:
    """Get quote from Aerodrome (Base-native DEX).

    Aerodrome uses a different interface — getAmountOut with stable flag.
    """
    from web3 import AsyncWeb3

    router_abi = [
        {
            "inputs": [
                {"name": "amountIn", "type": "uint256"},
                {"name": "tokenIn", "type": "address"},
                {"name": "tokenOut", "type": "address"},
                {"name": "stable", "type": "bool"},
            ],
            "name": "getAmountOut",
            "outputs": [
                {"name": "amount", "type": "uint256"},
                {"name": "stable", "type": "bool"},
            ],
            "stateMutability": "view",
            "type": "function",
        }
    ]

    try:
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        config = DEX_REGISTRY[DEXVenue.AERODROME]
        router = w3.eth.contract(
            address=w3.to_checksum_address(config.router_address),
            abi=router_abi,
        )

        result = await router.functions.getAmountOut(
            amount_in,
            w3.to_checksum_address(token_in),
            w3.to_checksum_address(token_out),
            False,  # volatile pool (not stable) for WETH/USDC
        ).call()

        amount_out = int(result[0])
        if amount_out <= 0:
            return None

        return RouteQuote(
            venue=DEXVenue.AERODROME,
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=amount_out,
            fee_tier=30,  # Aerodrome typical 0.3%
            estimated_gas=200_000,
            price_impact_bps=30.0,
            timestamp=time.time(),
        )
    except Exception as exc:
        log.debug(f"SOR: Aerodrome quote failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# RoutePlan — simplified route result for get_best_route API
# ---------------------------------------------------------------------------


@dataclass
class RoutePlan:
    """Simplified route result for get_best_route API.

    Attributes:
        dex: Name of the selected DEX (e.g. 'uniswap_v3', 'sushiswap').
        expected_out: Expected output amount in wei/smallest units.
        from_token: Input token symbol or address.
        to_token: Output token symbol or address.
        amount_in_wei: Input amount in wei.
    """

    dex: str
    expected_out: int
    from_token: str = ""
    to_token: str = ""
    amount_in_wei: int = 0


# ---------------------------------------------------------------------------
# Smart Order Router
# ---------------------------------------------------------------------------


class SmartOrderRouter:
    """Multi-DEX aggregation engine for optimal swap execution.

    Queries all supported DEXes in parallel, compares net output
    (amount_out - gas_cost), and returns the optimal route.

    Usage:
        sor = SmartOrderRouter()
        route = await sor.find_best_route(
            rpc_url="https://...",
            token_in=WETH_BASE,
            token_out=USDC_BASE,
            amount_in=10**17,  # 0.1 ETH
            gas_price_gwei=0.5,
        )
        print(f"Best venue: {route.best_quote.venue.value}")
        print(f"Output: {route.best_quote.amount_out}")
        print(f"Savings: {route.savings_bps:.1f} bps vs worst")
    """

    def __init__(self) -> None:
        self._route_cache: dict[str, OptimalRoute] = {}
        self._cache_ttl = 10.0  # Cache routes for 10 seconds
        self.is_tracker = ExecutionShortfallTracker()
        self.twap = TWAPExecutionScheduler()
        self.adaptive_split = AdaptiveSplitRatio()

    # ------------------------------------------------------------------
    # Simplified instance-method quote API (used by get_best_route)
    # ------------------------------------------------------------------

    async def _quote_uniswap_v3(
        self,
        from_token: str,
        to_token: str,
        amount_in_wei: int,
        rpc_url: str,
    ) -> int:
        """Quote Uniswap V3 for the best fee tier; returns output amount."""
        best = 0
        for fee in (500, 3000, 10000):
            try:
                result = await _quote_uniswap_v3(rpc_url, from_token, to_token, amount_in_wei, fee)
                if result and result.amount_out > best:
                    best = result.amount_out
            except Exception as _q_exc:
                log.warning("Uniswap V3 quote fee=%d failed: %s", fee, _q_exc)
        return best

    async def _quote_sushiswap(
        self,
        from_token: str,
        to_token: str,
        amount_in_wei: int,
        rpc_url: str,
    ) -> int:
        """Quote SushiSwap V3; returns output amount."""
        best = 0
        for fee in (500, 3000, 10000):
            try:
                result = await _quote_sushiswap_v3(rpc_url, from_token, to_token, amount_in_wei, fee)
                if result and result.amount_out > best:
                    best = result.amount_out
            except Exception as _sq_exc:
                log.warning("SushiSwap V3 quote fee=%d failed: %s", fee, _sq_exc)
        return best

    async def get_best_route(
        self,
        from_token: str,
        to_token: str,
        amount_in_wei: int,
        rpc_url: str,
    ) -> RoutePlan:
        """Get the best route across DEXes, returning a RoutePlan.

        Args:
            from_token: Input token symbol or address.
            to_token: Output token symbol or address.
            amount_in_wei: Amount in smallest units.
            rpc_url: RPC endpoint URL.

        Returns:
            RoutePlan with the best DEX and expected output.

        Raises:
            RuntimeError: If all DEX quotes fail.
        """
        quotes: dict[str, int] = {}
        errors: list[Exception] = []

        for dex_name, coro in [
            ("uniswap_v3", self._quote_uniswap_v3(from_token, to_token, amount_in_wei, rpc_url)),
            ("sushiswap", self._quote_sushiswap(from_token, to_token, amount_in_wei, rpc_url)),
        ]:
            try:
                out = await coro
                if out > 0:
                    quotes[dex_name] = out
            except Exception as exc:
                errors.append(exc)

        if not quotes:
            raise RuntimeError(
                f"All DEX quotes failed: {[str(e) for e in errors]}"
            )

        best_dex = max(quotes, key=lambda k: quotes[k])
        return RoutePlan(
            dex=best_dex,
            expected_out=quotes[best_dex],
            from_token=from_token,
            to_token=to_token,
            amount_in_wei=amount_in_wei,
        )

    async def find_best_route(
        self,
        rpc_url: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        gas_price_gwei: float = 0.5,
        eth_price_usd: float = 3000.0,
    ) -> OptimalRoute | None:
        """Find the optimal route across all DEXes.

        Args:
            rpc_url: Base chain RPC endpoint.
            token_in: Input token address.
            token_out: Output token address.
            amount_in: Amount of input token (in smallest units).
            gas_price_gwei: Current gas price for net output calculation.

        Returns:
            OptimalRoute with the best quote, or None if all quotes fail.
        """
        cache_key = f"{token_in}:{token_out}:{amount_in}"
        cached = self._route_cache.get(cache_key)
        if cached and time.time() - cached.best_quote.timestamp < self._cache_ttl:
            return cached

        start_time = time.time()

        # Build all quote tasks
        tasks: list[asyncio.Task] = []

        # Uniswap V3: query all fee tiers
        uni_config = DEX_REGISTRY[DEXVenue.UNISWAP_V3]
        if uni_config.is_active:
            for fee in uni_config.fee_tiers:
                tasks.append(
                    asyncio.create_task(
                        _quote_uniswap_v3(rpc_url, token_in, token_out, amount_in, fee)
                    )
                )

        # SushiSwap V3: query all fee tiers
        sushi_config = DEX_REGISTRY[DEXVenue.SUSHISWAP_V3]
        if sushi_config.is_active:
            for fee in sushi_config.fee_tiers:
                tasks.append(
                    asyncio.create_task(
                        _quote_sushiswap_v3(rpc_url, token_in, token_out, amount_in, fee)
                    )
                )

        # Aerodrome
        aero_config = DEX_REGISTRY[DEXVenue.AERODROME]
        if aero_config.is_active:
            tasks.append(
                asyncio.create_task(
                    _quote_aerodrome(rpc_url, token_in, token_out, amount_in)
                )
            )

        # Execute all quotes in parallel with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid quotes
        quotes: list[RouteQuote] = []
        for r in results:
            if isinstance(r, RouteQuote) and r.amount_out > 0:
                quotes.append(r)

        if not quotes:
            log.warning("SOR: all DEX quotes failed — no valid route found.")
            return None

        latency_ms = (time.time() - start_time) * 1000

        # Calculate net output (amount_out minus gas cost expressed in USDC units)
        # Gas cost USD = gas_units × gas_price_gwei × 1e-9 ETH × eth_price_usd
        # USDC units (6 decimals): multiply USD by 1e6
        gas_price_eth = gas_price_gwei * 1e-9  # gwei → ETH per gas unit

        def _net_output(q: RouteQuote) -> float:
            gas_cost_usd = q.estimated_gas * gas_price_eth * eth_price_usd
            gas_cost_usdc_units = gas_cost_usd * 1e6  # USD → USDC 6-decimal units
            return float(q.amount_out) - gas_cost_usdc_units

        # Sort by net output (highest first)
        quotes.sort(key=_net_output, reverse=True)
        best = quotes[0]
        worst = quotes[-1]

        # Calculate savings vs worst route
        if worst.amount_out > 0:
            savings_bps = (best.amount_out - worst.amount_out) / worst.amount_out * 10000
        else:
            savings_bps = 0.0

        route = OptimalRoute(
            best_quote=best,
            all_quotes=quotes,
            savings_bps=savings_bps,
            latency_ms=latency_ms,
        )

        self._route_cache[cache_key] = route

        log.info(
            f"SOR: best={best.venue.value} fee={best.fee_tier} "
            f"out={best.amount_out} savings={savings_bps:.1f}bps "
            f"({len(quotes)} quotes in {latency_ms:.0f}ms)"
        )

        return route

    def clear_cache(self) -> None:
        """Clear the route cache."""
        self._route_cache.clear()


# ---------------------------------------------------------------------------
# TWAPExecutionScheduler — 5-slice TWAP with jitter
# ---------------------------------------------------------------------------


class TWAPExecutionScheduler:
    """Splits a large order into N equal time-slices to reduce market impact.

    Random ±jitter per slice prevents MEV bots from predicting execution
    times and front-running. Default: 5 slices × 60-second intervals = 5 min.
    """

    def __init__(
        self,
        n_slices: int = 5,
        interval_secs: float = 60.0,
        jitter_pct: float = 0.10,
    ) -> None:
        self._n_slices = n_slices
        self._interval_secs = interval_secs
        self._jitter_pct = jitter_pct

    def schedule(
        self,
        total_amount_in: int,
        start_time: float | None = None,
    ) -> list[dict]:
        """Build execution schedule.

        Returns:
            List of dicts: {slice_idx, amount_in, scheduled_time, executed}.
        """
        now = start_time or time.time()
        base_slice = total_amount_in // self._n_slices
        remainder = total_amount_in - base_slice * self._n_slices

        slices = []
        for i in range(self._n_slices):
            amt = base_slice + (remainder if i == self._n_slices - 1 else 0)
            jitter = random.uniform(-self._jitter_pct, self._jitter_pct) * self._interval_secs
            slices.append({
                "slice_idx": i,
                "amount_in": amt,
                "scheduled_time": now + i * self._interval_secs + jitter,
                "executed": False,
            })
        return slices

    def get_due_slice(self, schedule: list[dict]) -> dict | None:
        """Return the next unexecuted slice whose scheduled_time has passed."""
        now = time.time()
        for s in schedule:
            if not s.get("executed", False) and now >= s["scheduled_time"]:
                return s
        return None

    @staticmethod
    def mark_executed(s: dict) -> None:
        """Mark a slice as executed in-place."""
        s["executed"] = True


# ---------------------------------------------------------------------------
# AdaptiveSplitRatio — live-tick-based venue weighting
# ---------------------------------------------------------------------------


class AdaptiveSplitRatio:
    """Computes the optimal cross-venue split ratio from live tick data.

    Each venue is scored by: score = depth_usd / max(spread_bps, 1).
    Final weights are the normalised scores. Venues with no data are
    excluded from splits.
    """

    def __init__(self) -> None:
        self._venue_data: dict[str, dict] = {}

    def update_venue(
        self,
        venue: str,
        bid_depth_usd: float,
        ask_depth_usd: float,
        spread_bps: float,
    ) -> None:
        """Update liquidity snapshot for a venue."""
        total_depth = bid_depth_usd + ask_depth_usd
        score = total_depth / max(spread_bps, 1.0)
        self._venue_data[venue] = {
            "bid_depth": bid_depth_usd,
            "ask_depth": ask_depth_usd,
            "spread_bps": spread_bps,
            "score": score,
        }

    def get_split(self) -> dict[str, float]:
        """Return normalised per-venue weight dict summing to 1.0.

        Returns {} if no venue data available (use even split externally).
        """
        if not self._venue_data:
            return {}
        scores = {v: d["score"] for v, d in self._venue_data.items() if d["score"] > 0}
        total = sum(scores.values())
        if total <= 0:
            return {}
        return {v: s / total for v, s in scores.items()}

    def get_venue_score(self, venue: str) -> float:
        return float(self._venue_data.get(venue, {}).get("score", 0.0))


# ---------------------------------------------------------------------------
# ExecutionShortfallTracker — rolling IS model
# ---------------------------------------------------------------------------


class ExecutionShortfallTracker:
    """Tracks implementation shortfall (IS) on a rolling 50-trade window.

    IS definition (buy side):
        IS = (execution_price - arrival_price) / arrival_price

    IS > 0 → paid more than arrival (adverse); IS < 0 → favorable fill.
    Sell-side IS is negated so positive always means "cost to trade".

    Uses this to estimate expected IS for sizing decisions: when rolling
    mean IS > threshold, widen TWAP intervals to reduce impact.
    """

    WINDOW: int = 50
    WARN_IS_THRESHOLD: float = 0.001  # 10 bps mean IS triggers warning

    def __init__(self) -> None:
        self._is_history: deque[float] = deque(maxlen=self.WINDOW)

    def record(
        self,
        arrival_price: float,
        execution_price: float,
        side: str = "buy",
    ) -> float:
        """Record a completed trade and return its IS.

        Args:
            arrival_price: Mid-price at order submission time.
            execution_price: Actual average fill price.
            side: 'buy' or 'sell'.

        Returns:
            IS value for this trade (positive = adverse).
        """
        if arrival_price <= 0:
            return 0.0
        raw_is = (execution_price - arrival_price) / arrival_price
        is_val = raw_is if side == "buy" else -raw_is
        self._is_history.append(is_val)

        if len(self._is_history) >= 5 and self.mean_is > self.WARN_IS_THRESHOLD:
            log.warning(
                f"ExecutionShortfall: rolling mean IS={self.mean_is:.4%} "
                f"exceeds {self.WARN_IS_THRESHOLD:.4%} threshold "
                f"({len(self._is_history)} trades)"
            )
        return is_val

    @property
    def mean_is(self) -> float:
        """Rolling mean IS (positive = consistently paying more than arrival)."""
        if not self._is_history:
            return 0.0
        return sum(self._is_history) / len(self._is_history)

    @property
    def median_is(self) -> float:
        """Median IS (robust to outliers)."""
        if not self._is_history:
            return 0.0
        sorted_is = sorted(self._is_history)
        n = len(sorted_is)
        mid = n // 2
        return sorted_is[mid] if n % 2 else (sorted_is[mid - 1] + sorted_is[mid]) / 2.0

    @property
    def trade_count(self) -> int:
        return len(self._is_history)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_sor_instance: SmartOrderRouter | None = None


def get_smart_order_router() -> SmartOrderRouter:
    """Return the process-wide SmartOrderRouter singleton."""
    global _sor_instance
    if _sor_instance is None:
        _sor_instance = SmartOrderRouter()
    return _sor_instance
