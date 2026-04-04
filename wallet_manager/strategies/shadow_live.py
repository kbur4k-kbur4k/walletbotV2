"""
wallet_manager/strategies/shadow_live.py

Sprint JS-8.1: Shadow Live Mode
================================
Constructs real on-chain quotes but does NOT broadcast transactions.
Compares shadow fill prices (from Uniswap V3 QuoterV2) vs paper simulation
fill prices to measure model accuracy before live transition.

Activation gate (execution_mode = shadow_live):
  mean(|shadow - paper| / paper) < 0.1% over last 50 trade comparisons
  => model is accurate enough for live execution

Architecture: view-only. QuoterV2.quoteExactInputSingle is a staticcall
(reverts internally, no gas consumed). Safe to call on every trade cycle.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# Base chain contract addresses
WETH_BASE = "0x4200000000000000000000000000000000000006"
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
QUOTER_BASE = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"
POOL_FEE_005 = 500   # 0.05% Uniswap V3 pool fee tier
CHAIN_ID_BASE = 8453

# Readiness gate thresholds
SHADOW_GATE_DEVIATION_PCT = 0.10  # mean deviation must be < 0.1%
SHADOW_GATE_MIN_SAMPLES = 50      # need at least 50 comparisons


@dataclass
class ShadowComparison:
    """One paper vs shadow fill price comparison."""

    timestamp: float
    direction: str           # "BUY" | "SELL"
    size_usd: float
    paper_fill_price: float  # from slippage simulator
    shadow_fill_price: float # from Uniswap V3 QuoterV2
    deviation_pct: float     # |shadow - paper| / paper * 100
    rpc_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "direction": self.direction,
            "size_usd": round(self.size_usd, 4),
            "paper_fill_price": round(self.paper_fill_price, 4),
            "shadow_fill_price": round(self.shadow_fill_price, 4),
            "deviation_pct": round(self.deviation_pct, 4),
            "rpc_latency_ms": round(self.rpc_latency_ms, 1),
        }


class ShadowLiveEngine:
    """
    JS-8.1 Shadow Live Mode engine.

    When execution_mode = shadow_live, wraps every paper trade with a real
    on-chain Uniswap V3 quote to measure how well the slippage model
    approximates true on-chain fill prices.

    Usage in paper_engine.py (after simulated_fill_price is computed):
        from wallet_manager.strategies.shadow_live import get_shadow_live_engine
        sle = get_shadow_live_engine()
        shadow_price = await sle.get_shadow_quote(rpc_url, "BUY", size_usd, price)
        if shadow_price:
            sle.record_comparison("BUY", size_usd, simulated_fill_price, shadow_price)
    """

    def __init__(self, window_size: int = 200) -> None:
        self._comparisons: deque[ShadowComparison] = deque(maxlen=window_size)
        self._n_queries: int = 0
        self._n_failures: int = 0
        self._total_latency_ms: float = 0.0

    async def get_shadow_quote(
        self,
        rpc_url: str,
        direction: str,   # "BUY" | "SELL"
        size_usd: float,
        eth_price: float,
    ) -> float | None:
        """Get real on-chain fill price from Uniswap V3 QuoterV2.

        BUY  → USDC → WETH swap: returns effective ETH price (USD/ETH)
        SELL → WETH → USDC swap: returns effective ETH price (USD/ETH)

        QuoterV2 is a view/staticcall — no gas consumed.
        Returns None on any failure (graceful degradation).
        """
        if not rpc_url or size_usd <= 0 or eth_price <= 0:
            return None

        t0 = time.time()
        try:
            from wallet_manager.actions import quote_amount_out

            if direction == "BUY":
                # USDC → WETH: how much WETH do we get for size_usd USDC?
                amount_in_usdc = int(size_usd * 1_000_000)  # USDC: 6 decimals
                amount_out_weth = await quote_amount_out(
                    rpc_url=rpc_url,
                    chain_id=CHAIN_ID_BASE,
                    token_in=USDC_BASE,
                    token_out=WETH_BASE,
                    amount_in=amount_in_usdc,
                    fee=POOL_FEE_005,
                )
                if amount_out_weth and amount_out_weth > 0:
                    weth_out = amount_out_weth / 1e18  # WETH: 18 decimals
                    effective_price = float(size_usd / weth_out)
                    latency_ms = (time.time() - t0) * 1000
                    self._total_latency_ms += latency_ms
                    self._n_queries += 1
                    log.debug(
                        f"SHADOW_LIVE: BUY quote ${size_usd:.2f} USDC → "
                        f"{weth_out:.6f} WETH @ ${effective_price:.2f} "
                        f"(latency={latency_ms:.0f}ms)"
                    )
                    return effective_price

            elif direction == "SELL":
                # WETH → USDC: how much USDC do we get for size_usd worth of WETH?
                weth_in = size_usd / eth_price
                amount_in_weth = int(weth_in * 1e18)  # WETH: 18 decimals
                amount_out_usdc = await quote_amount_out(
                    rpc_url=rpc_url,
                    chain_id=CHAIN_ID_BASE,
                    token_in=WETH_BASE,
                    token_out=USDC_BASE,
                    amount_in=amount_in_weth,
                    fee=POOL_FEE_005,
                )
                if amount_out_usdc and amount_out_usdc > 0:
                    usdc_out = amount_out_usdc / 1e6  # USDC: 6 decimals
                    effective_price = float(usdc_out / weth_in) if weth_in > 0 else 0.0
                    latency_ms = (time.time() - t0) * 1000
                    self._total_latency_ms += latency_ms
                    self._n_queries += 1
                    log.debug(
                        f"SHADOW_LIVE: SELL quote {weth_in:.6f} WETH → "
                        f"${usdc_out:.2f} USDC @ ${effective_price:.2f} "
                        f"(latency={latency_ms:.0f}ms)"
                    )
                    return effective_price

        except Exception as exc:
            self._n_failures += 1
            log.warning(f"ShadowLive: QuoterV2 call failed ({direction} ${size_usd:.2f}): {exc}")

        return None

    def record_comparison(
        self,
        direction: str,
        size_usd: float,
        paper_fill_price: float,
        shadow_fill_price: float,
        rpc_latency_ms: float = 0.0,
    ) -> ShadowComparison:
        """Record a paper vs shadow fill price comparison."""
        deviation_pct = 0.0
        if paper_fill_price > 0:
            deviation_pct = abs(shadow_fill_price - paper_fill_price) / paper_fill_price * 100.0

        comparison = ShadowComparison(
            timestamp=time.time(),
            direction=direction,
            size_usd=size_usd,
            paper_fill_price=paper_fill_price,
            shadow_fill_price=shadow_fill_price,
            deviation_pct=deviation_pct,
            rpc_latency_ms=rpc_latency_ms,
        )
        self._comparisons.append(comparison)
        log.info(
            f"SHADOW_LIVE: {direction} ${size_usd:.2f} — "
            f"paper=${paper_fill_price:.2f} shadow=${shadow_fill_price:.2f} "
            f"deviation={deviation_pct:.3f}%"
        )
        return comparison

    def get_readiness_gate(self) -> dict[str, Any]:
        """Evaluate JS-8.1 readiness gate for micro-live transition.

        Gate: mean(|shadow - paper| / paper) < 0.1% over last 50 comparisons.
        This confirms the slippage model is accurate enough to trust for live.
        """
        recent = list(self._comparisons)[-SHADOW_GATE_MIN_SAMPLES:]
        n = len(recent)

        if n == 0:
            return {
                "ready": False,
                "n_comparisons": 0,
                "required": SHADOW_GATE_MIN_SAMPLES,
                "mean_deviation_pct": None,
                "gate_threshold_pct": SHADOW_GATE_DEVIATION_PCT,
                "status": "INSUFFICIENT_DATA",
                "interpretation": "No shadow comparisons yet. Enable shadow_live mode and run cycles.",
            }

        mean_dev = sum(c.deviation_pct for c in recent) / n
        max_dev = max(c.deviation_pct for c in recent)
        gate_passed = n >= SHADOW_GATE_MIN_SAMPLES and mean_dev < SHADOW_GATE_DEVIATION_PCT

        if gate_passed:
            status = "READY"
        elif mean_dev < SHADOW_GATE_DEVIATION_PCT * 2:
            status = "CLOSE"
        else:
            status = "BUILDING"

        return {
            "ready": gate_passed,
            "n_comparisons": n,
            "required": SHADOW_GATE_MIN_SAMPLES,
            "mean_deviation_pct": round(mean_dev, 4),
            "max_deviation_pct": round(max_dev, 4),
            "gate_threshold_pct": SHADOW_GATE_DEVIATION_PCT,
            "status": status,
            "interpretation": (
                "Shadow model matches on-chain quotes — ready for micro-live ($25)."
                if gate_passed
                else f"Need {SHADOW_GATE_MIN_SAMPLES - n} more comparisons, "
                f"mean dev {mean_dev:.3f}% vs gate {SHADOW_GATE_DEVIATION_PCT:.2f}%."
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        readiness = self.get_readiness_gate()
        recent = list(self._comparisons)[-10:]
        avg_latency = (
            self._total_latency_ms / self._n_queries
            if self._n_queries > 0
            else 0.0
        )

        return {
            "generated_at": time.time(),
            "readiness_gate": readiness,
            "n_queries_total": self._n_queries,
            "n_failures": self._n_failures,
            "failure_rate_pct": round(
                self._n_failures / max(self._n_queries + self._n_failures, 1) * 100, 1
            ),
            "avg_rpc_latency_ms": round(avg_latency, 1),
            "recent_comparisons": [c.to_dict() for c in reversed(recent)],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_shadow_live_engine: ShadowLiveEngine | None = None


def get_shadow_live_engine() -> ShadowLiveEngine:
    global _shadow_live_engine
    if _shadow_live_engine is None:
        _shadow_live_engine = ShadowLiveEngine()
    return _shadow_live_engine
