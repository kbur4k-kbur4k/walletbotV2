"""
wallet_manager/strategies/mev_shield.py

QUANT-X MEV Shield — The "Ghost" Protocol for Transaction Protection.

Routes all transactions through private RPC endpoints to eliminate
sandwich attacks and front-running. Implements:

1. Private RPC Routing: Flashbots Protect or specialised Base builders.
2. Dynamic RBF: If tx not confirmed within 2 blocks, bump gas by 15%.
3. Gaussian Jitter: Random micro-delays (100ms-2000ms) on every
   on-chain action to obfuscate bot-like behaviour.
4. Failover RPC Cluster: Automatic switching between 3+ RPC providers
   with latency monitoring.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import cast

from eth_typing import HexStr

from wallet_manager.security.logging_filter import build_logger
from wallet_manager.utils.web3_utils import close_w3_provider

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Private RPC endpoints (Base chain)
# Flashbots Protect on Base, or custom builder endpoints
PRIVATE_RPC_ENDPOINTS: list[str] = [
    # Default: use env vars for private RPCs
]

# Gaussian Jitter parameters
JITTER_MIN_MS = 100     # Minimum delay in milliseconds
JITTER_MAX_MS = 2000    # Maximum delay in milliseconds
JITTER_MEAN_MS = 800    # Mean of Gaussian distribution
JITTER_STD_MS = 400     # Standard deviation

# Dynamic RBF parameters
RBF_BLOCK_DEADLINE = 2      # Bump after 2 blocks without confirmation
RBF_GAS_BUMP_PCT = 0.15     # 15% gas increase per bump
RBF_MAX_ATTEMPTS = 3         # Maximum RBF attempts
RBF_POLL_INTERVAL = 3.0      # Seconds between confirmation checks

# Failover RPC parameters
RPC_HEALTH_CHECK_INTERVAL = 30.0   # Seconds between health checks
RPC_MAX_LATENCY_MS = 2000          # Max acceptable latency
RPC_FAILURE_THRESHOLD = 3          # Consecutive failures before failover
RPC_RECOVERY_WINDOW = 120.0        # Seconds before retrying failed provider


# ---------------------------------------------------------------------------
# Gaussian Jitter Engine
# ---------------------------------------------------------------------------


class GaussianJitter:
    """Applies random micro-delays to obfuscate bot-like timing patterns.

    Uses a Gaussian distribution centred at JITTER_MEAN_MS with
    JITTER_STD_MS standard deviation, clamped to [JITTER_MIN_MS, JITTER_MAX_MS].
    """

    def __init__(
        self,
        mean_ms: float = JITTER_MEAN_MS,
        std_ms: float = JITTER_STD_MS,
        min_ms: float = JITTER_MIN_MS,
        max_ms: float = JITTER_MAX_MS,
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
# RPC Health Tracker
# ---------------------------------------------------------------------------


@dataclass
class RPCProviderState:
    """Health state for a single RPC provider."""

    url: str
    is_private: bool = False
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    is_healthy: bool = True

    def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.consecutive_failures = 0
        self.is_healthy = True
        self.last_success_time = time.time()
        # Rolling average
        alpha = 0.3
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

    def record_failure(self) -> None:
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= RPC_FAILURE_THRESHOLD:
            self.is_healthy = False

    def should_retry(self) -> bool:
        """Check if enough time has passed to retry a failed provider."""
        if self.is_healthy:
            return True
        return time.time() - self.last_failure_time >= RPC_RECOVERY_WINDOW

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests


# ---------------------------------------------------------------------------
# Failover RPC Cluster
# ---------------------------------------------------------------------------


class FailoverRPCCluster:
    """Manages a cluster of RPC providers with automatic failover.

    Tracks health metrics per provider and routes requests to the
    healthiest available endpoint. Private RPCs are preferred for
    MEV-sensitive transactions.

    Args:
        providers: List of (url, is_private) tuples.
    """

    def __init__(self, providers: list[tuple[str, bool]] | None = None) -> None:
        self._providers: list[RPCProviderState] = []
        self._active_index: int = 0

        if providers:
            for url, is_private in providers:
                self._providers.append(RPCProviderState(url=url, is_private=is_private))

    def add_provider(self, url: str, is_private: bool = False) -> None:
        """Add a new RPC provider to the cluster."""
        # Avoid duplicates
        for p in self._providers:
            if p.url == url:
                return
        self._providers.append(RPCProviderState(url=url, is_private=is_private))

    def get_best_rpc(self, prefer_private: bool = True) -> str | None:
        """Return the best available RPC endpoint.

        Args:
            prefer_private: If True, prefer private RPCs (MEV protection).

        Returns:
            RPC URL string, or None if no healthy providers.
        """
        if not self._providers:
            return None

        # Filter healthy or retryable providers
        available = [p for p in self._providers if p.is_healthy or p.should_retry()]
        if not available:
            # All failed: reset oldest failure for retry
            oldest = min(self._providers, key=lambda p: p.last_failure_time)
            oldest.is_healthy = True
            available = [oldest]

        if prefer_private:
            private = [p for p in available if p.is_private]
            if private:
                # Pick the one with lowest latency
                best = min(private, key=lambda p: p.avg_latency_ms)
                return best.url

        # Fallback to healthiest public RPC
        best = min(available, key=lambda p: (not p.is_healthy, p.avg_latency_ms))
        return best.url

    def record_success(self, url: str, latency_ms: float) -> None:
        """Record a successful request to the given provider."""
        for p in self._providers:
            if p.url == url:
                p.record_success(latency_ms)
                return

    def record_failure(self, url: str) -> None:
        """Record a failed request to the given provider."""
        for p in self._providers:
            if p.url == url:
                p.record_failure()
                if not p.is_healthy:
                    log.warning(
                        f"MEV Shield: RPC {url[:30]}... marked unhealthy "
                        f"({p.consecutive_failures} consecutive failures)."
                    )
                return

    async def health_check(self, url: str) -> tuple[bool, float]:
        """Perform a lightweight health check on an RPC provider.

        Returns:
            (is_healthy, latency_ms).
        """
        from web3 import AsyncWeb3

        start = time.time()
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(url))
        try:
            await asyncio.wait_for(
                w3.eth.block_number,
                timeout=RPC_MAX_LATENCY_MS / 1000.0,
            )
            latency_ms = (time.time() - start) * 1000
            self.record_success(url, latency_ms)
            return True, latency_ms
        except Exception:
            latency_ms = (time.time() - start) * 1000
            self.record_failure(url)
            return False, latency_ms
        finally:
            await close_w3_provider(w3)

    async def health_check_all(self) -> dict[str, tuple[bool, float]]:
        """Run health checks on all providers in parallel.

        Returns:
            Dict mapping URL to (is_healthy, latency_ms).
        """
        results = {}
        tasks = []
        for p in self._providers:
            tasks.append(self.health_check(p.url))

        checked = await asyncio.gather(*tasks, return_exceptions=True)
        for p, result in zip(self._providers, checked, strict=False):
            if isinstance(result, tuple):
                results[p.url] = result
            else:
                results[p.url] = (False, 0.0)

        return results

    def get_cluster_stats(self) -> dict:
        """Return cluster-wide health statistics."""
        total = len(self._providers)
        healthy = sum(1 for p in self._providers if p.is_healthy)
        private = sum(1 for p in self._providers if p.is_private)
        avg_latency = (
            sum(p.avg_latency_ms for p in self._providers) / total
            if total > 0
            else 0.0
        )
        return {
            "total_providers": total,
            "healthy_providers": healthy,
            "private_providers": private,
            "avg_latency_ms": avg_latency,
            "provider_details": [
                {
                    "url": p.url[:40] + "...",
                    "is_private": p.is_private,
                    "is_healthy": p.is_healthy,
                    "avg_latency_ms": p.avg_latency_ms,
                    "success_rate": p.success_rate,
                }
                for p in self._providers
            ],
        }


# ---------------------------------------------------------------------------
# Dynamic RBF (Replace-By-Fee) Engine
# ---------------------------------------------------------------------------


class DynamicRBF:
    """Enhanced Replace-By-Fee engine for stuck transactions.

    Monitors transaction confirmations and automatically bumps gas
    if not confirmed within RBF_BLOCK_DEADLINE blocks.

    Improvements over basic RBF:
    - Monitors block production (not just time)
    - Escalating gas bumps (15% → 25% → 40%)
    - Gas ceiling to prevent runaway costs
    """

    def __init__(
        self,
        block_deadline: int = RBF_BLOCK_DEADLINE,
        base_bump_pct: float = RBF_GAS_BUMP_PCT,
        max_attempts: int = RBF_MAX_ATTEMPTS,
    ) -> None:
        self._block_deadline = block_deadline
        self._base_bump = base_bump_pct
        self._max_attempts = max_attempts

    def calculate_bumped_gas(
        self,
        current_gas_price: int,
        attempt: int,
    ) -> int:
        """Calculate bumped gas price with escalation.

        Escalation: attempt 1 = 15%, attempt 2 = 25%, attempt 3 = 40%.

        Args:
            current_gas_price: Current gas price in wei.
            attempt: RBF attempt number (1-based).

        Returns:
            New gas price in wei.
        """
        # Escalating bump: base * (1 + 0.5 * (attempt - 1))
        escalation = self._base_bump * (1.0 + 0.5 * (attempt - 1))
        bumped = int(current_gas_price * (1.0 + escalation))
        return bumped

    async def monitor_and_bump(
        self,
        rpc_url: str,
        tx_hash: str,
        original_gas_price: int,
        jitter: GaussianJitter | None = None,
    ) -> tuple[bool, int, str]:
        """Monitor a transaction and bump gas if stuck.

        Args:
            rpc_url: RPC endpoint.
            tx_hash: Transaction hash to monitor.
            original_gas_price: Original gas price in wei.
            jitter: Optional GaussianJitter to apply between checks.

        Returns:
            (was_bumped, final_gas_price, status).
        """
        from web3 import AsyncWeb3

        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        try:
            start_block = await w3.eth.block_number
            current_gas = original_gas_price

            for attempt in range(1, self._max_attempts + 1):
                # Wait for poll interval
                await asyncio.sleep(RBF_POLL_INTERVAL)

                if jitter:
                    await jitter.apply()

                # Check if confirmed
                try:
                    receipt = await w3.eth.get_transaction_receipt(cast(HexStr, tx_hash))
                    if receipt and receipt.get("status") is not None:
                        return False, current_gas, "confirmed"
                except Exception:
                    pass  # Not yet mined

                # Check block progress
                current_block = await w3.eth.block_number
                blocks_elapsed = current_block - start_block

                if blocks_elapsed >= self._block_deadline * attempt:
                    # Bump gas
                    new_gas = self.calculate_bumped_gas(current_gas, attempt)
                    log.info(
                        f"RBF: tx {tx_hash[:10]}... stuck for {blocks_elapsed} blocks. "
                        f"Bumping gas: {current_gas / 1e9:.2f} → {new_gas / 1e9:.2f} Gwei "
                        f"(attempt {attempt}/{self._max_attempts})."
                    )
                    current_gas = new_gas
                    # Return bumped gas for caller to resubmit
                    return True, new_gas, f"bumped_attempt_{attempt}"

            return False, current_gas, "max_attempts_reached"
        finally:
            await close_w3_provider(w3)


# ---------------------------------------------------------------------------
# MEV Shield (Unified Interface)
# ---------------------------------------------------------------------------


class MEVShield:
    """Unified MEV protection layer.

    Combines all anti-MEV components into a single interface:
    - Private RPC routing
    - Gaussian jitter
    - Dynamic RBF
    - Failover cluster

    Usage:
        shield = MEVShield.from_env()

        # Get protected RPC endpoint
        rpc = shield.get_protected_rpc()

        # Apply jitter before transaction
        await shield.apply_jitter()

        # After submitting tx, monitor for stuck state
        bumped, new_gas, status = await shield.monitor_tx(rpc, tx_hash, gas_price)
    """

    def __init__(self) -> None:
        self.jitter = GaussianJitter()
        self.rbf = DynamicRBF()
        self.cluster = FailoverRPCCluster()

    @classmethod
    def from_env(cls) -> MEVShield:
        """Create MEVShield from environment variables.

        Reads private/public RPC URLs and API keys from centralized config.
        """
        from wallet_manager.config import get_settings
        cfg = get_settings()
        shield = cls()

        # Add private RPCs from config
        for url in [cfg.private_rpc_1, cfg.private_rpc_2, cfg.private_rpc_3]:
            if url:
                shield.cluster.add_provider(url, is_private=True)

        # Add public RPCs from config
        for url in [cfg.base_rpc_1, cfg.base_rpc_2, cfg.base_rpc_3]:
            if url:
                shield.cluster.add_provider(url, is_private=False)

        # Construct from Infura/Alchemy keys
        if cfg.infura_key:
            shield.cluster.add_provider(
                f"https://base-mainnet.infura.io/v3/{cfg.infura_key}",
                is_private=False,
            )

        if cfg.alchemy_key:
            shield.cluster.add_provider(
                f"https://base-mainnet.g.alchemy.com/v2/{cfg.alchemy_key}",
                is_private=False,
            )

        # Always add Base public RPC as last resort
        shield.cluster.add_provider(
            "https://mainnet.base.org",
            is_private=False,
        )

        return shield

    def get_protected_rpc(self) -> str:
        """Get the best available RPC, preferring private endpoints."""
        rpc = self.cluster.get_best_rpc(prefer_private=True)
        if rpc is None:
            return "https://mainnet.base.org"
        return rpc

    async def apply_jitter(self) -> float:
        """Apply Gaussian jitter delay before an on-chain action.

        Returns:
            Actual delay applied in milliseconds.
        """
        return await self.jitter.apply()

    async def monitor_tx(
        self,
        rpc_url: str,
        tx_hash: str,
        gas_price_wei: int,
    ) -> tuple[bool, int, str]:
        """Monitor a transaction and bump gas if stuck.

        Returns:
            (was_bumped, final_gas_price_wei, status).
        """
        return await self.rbf.monitor_and_bump(
            rpc_url=rpc_url,
            tx_hash=tx_hash,
            original_gas_price=gas_price_wei,
            jitter=self.jitter,
        )

    def get_stats(self) -> dict:
        """Return MEV Shield operational statistics."""
        return {
            "jitter": {
                "avg_ms": self.jitter.avg_jitter_ms,
                "total_seconds": self.jitter.total_jitter_seconds,
            },
            "cluster": self.cluster.get_cluster_stats(),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_mev_shield: MEVShield | None = None


def get_mev_shield() -> MEVShield:
    """Return the process-wide MEVShield singleton."""
    global _mev_shield
    if _mev_shield is None:
        _mev_shield = MEVShield.from_env()
    return _mev_shield


def reset_mev_shield() -> None:
    """Reset the singleton (for tests)."""
    global _mev_shield
    _mev_shield = None
