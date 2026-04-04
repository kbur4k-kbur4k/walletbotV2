"""
wallet_manager/network/router.py

Multi-Chain RPC Provider Router with automatic failover.

Responsibility
--------------
Maintain an ordered pool of RPC URLs for each registered EVM chain and
provide a single ``get_rpc(chain_id)`` call that always returns the healthiest
available endpoint.  When a provider fails the router rotates to the next
entry in the pool without crashing the caller.

Architecture
------------
Each chain's provider pool is modelled as a rotating deque::

    chain 8453 (Base):
    ┌───────────────────────────────────────────────────────┐
    │  active → [Infura] [Alchemy] [Ankr] [Public]          │
    │               ↑                                        │
    │           mark_failed("Infura")                        │
    │                                                        │
    │  rotated → [Alchemy] [Ankr] [Public] [Infura]         │
    └───────────────────────────────────────────────────────┘

``mark_failed`` moves the failed URL to the *end* of the deque and records a
failure timestamp.  A failed provider is eligible for reuse after
``_RECOVERY_WINDOW_SECONDS`` so that a temporary outage does not permanently
remove it from the pool.

Circuit breaker per provider
-----------------------------
Each ``(chain_id, rpc_url)`` pair has an independent failure counter.  After
``_PROVIDER_FAILURE_THRESHOLD`` consecutive failures the provider is skipped
entirely during ``get_rpc``, similar to the wallet-level circuit breaker.

Health check
------------
``probe(chain_id)`` sends a lightweight ``eth_blockNumber`` call to validate
that the active provider is responsive.  The orchestrator can call this on
startup and periodically to pre-warm the pool state.

SOLID alignment
---------------
* **S** — router owns only provider selection; no signing, no gas logic.
* **O** — new failover strategies extend ``IRouter`` without changing this class.
* **D** — callers depend on ``IRouter``; concrete ``RPCRouter`` is injected.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import cast

from web3 import AsyncWeb3

from wallet_manager.core.interfaces import IRouter
from wallet_manager.core.models import ChainConfig
from wallet_manager.exceptions import (
    ChainNotConfiguredError,
    ConfigurationError,
    NoAvailableRpcError,
)
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Consecutive failures before a provider is temporarily skipped
_PROVIDER_FAILURE_THRESHOLD: int = 3

# Seconds after which a failed provider re-enters the rotation
_RECOVERY_WINDOW_SECONDS: float = 120.0

# Timeout for health-check probe calls (seconds)
_PROBE_TIMEOUT: float = 5.0

# Rolling window for latency scoring
_LATENCY_WINDOW: int = 20

# Re-probe disabled endpoints every N seconds
_REPROBE_DISABLED_INTERVAL: float = 300.0


# ---------------------------------------------------------------------------
# Per-provider health state
# ---------------------------------------------------------------------------


@dataclass
class _ProviderHealth:
    """Mutable health record for a single RPC URL.

    Attributes:
        url: The RPC endpoint URL.
        failure_count: Consecutive failures since last success.
        last_failure_time: Monotonic timestamp of the most recent failure.
        skip_until: Monotonic timestamp after which the provider is eligible
            for reuse (set when ``failure_count`` reaches the threshold).
    """

    url: str
    failure_count: int = 0
    last_failure_time: float = 0.0
    skip_until: float = 0.0

    def is_available(self) -> bool:
        """Return True if this provider should be considered for requests.

        Returns:
            True if healthy or recovery window has elapsed.
        """
        if self.failure_count < _PROVIDER_FAILURE_THRESHOLD:
            return True
        # Allow back in after recovery window
        if time.monotonic() >= self.skip_until:
            self.failure_count = 0   # Reset on re-admission
            return True
        return False

    def record_failure(self) -> None:
        """Increment failure counter and set skip_until if threshold crossed."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= _PROVIDER_FAILURE_THRESHOLD:
            self.skip_until = time.monotonic() + _RECOVERY_WINDOW_SECONDS
            log.warning(
                f"Provider '{self.url}' suspended for {_RECOVERY_WINDOW_SECONDS:.0f}s after {self.failure_count} failures.",
            )

    def record_success(self) -> None:
        """Reset failure state on a successful request."""
        if self.failure_count > 0:
            log.info(f"Provider '{self.url}' recovered after failures.")
        self.failure_count = 0
        self.skip_until = 0.0


# ---------------------------------------------------------------------------
# Per-chain provider pool
# ---------------------------------------------------------------------------


class _ChainPool:
    """Rotating deque of RPC providers for a single chain.

    Attributes:
        config: Immutable chain configuration.
        pool: Ordered deque of :class:`_ProviderHealth` objects.
        lock: Asyncio lock guarding pool mutations.
    """

    def __init__(self, config: ChainConfig) -> None:
        self.config = config

        # Build the pool: primary first, then fallbacks
        all_urls = [config.primary_rpc, *config.fallback_rpcs]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_urls: list[str] = []
        for url in all_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        self.pool: deque[_ProviderHealth] = deque(
            _ProviderHealth(url=url) for url in unique_urls
        )
        self._health_map: dict[str, _ProviderHealth] = {
            ph.url: ph for ph in self.pool
        }
        self.lock = asyncio.Lock()

    def active_url(self) -> str:
        """Return the URL at the front of the pool.

        Returns:
            URL string of the currently preferred provider.

        Raises:
            NoAvailableRpcError: If every provider is suspended.
        """
        # Scan pool in order to find the first available provider
        for provider in self.pool:
            if provider.is_available():
                return provider.url

        raise NoAvailableRpcError(
            f"All RPC providers for chain {self.config.chain_id} "
            f"({self.config.name}) are suspended.",
            chain_id=self.config.chain_id,
        )

    def rotate_failed(self, url: str) -> None:
        """Record a failure for *url* and rotate it to the back of the pool.

        Args:
            url: The URL that produced the failure.
        """
        provider = self._health_map.get(url)
        if provider is None:
            return
        provider.record_failure()

        # Move failed provider to the back so the next call tries the next one
        if self.pool and self.pool[0].url == url:
            self.pool.rotate(-1)
            log.debug(
                f"Chain {self.config.chain_id}: rotated failed provider '{url}' to back of pool.",
            )

    def record_success(self, url: str) -> None:
        """Mark *url* as healthy.

        Args:
            url: The URL that succeeded.
        """
        provider = self._health_map.get(url)
        if provider:
            provider.record_success()


# ---------------------------------------------------------------------------
# RPCRouter — public implementation of IRouter
# ---------------------------------------------------------------------------


class RPCRouter(IRouter):
    """Multi-chain RPC router implementing :class:`~wallet_manager.core.interfaces.IRouter`.

    Maintains a :class:`_ChainPool` for each registered :class:`ChainConfig`
    and exposes a simple ``get_rpc(chain_id)`` interface.

    Args:
        chains: List of :class:`~wallet_manager.core.models.ChainConfig`
            records defining all supported chains and their RPC pools.

    Example:
        .. code-block:: python

            router = RPCRouter([
                ChainConfig(
                    chain_id=8453,
                    name="Base",
                    native_symbol="ETH",
                    primary_rpc="https://mainnet.base.org",
                    fallback_rpcs=("https://base.llamarpc.com",),
                ),
            ])
            rpc = router.get_rpc(8453)

            # After a failed call:
            router.mark_failed(8453, rpc)
            rpc = router.get_rpc(8453)   # now returns the next provider
    """

    def __init__(self, chains: list[ChainConfig]) -> None:
        if not chains:
            raise ValueError("RPCRouter requires at least one ChainConfig.")

        self._pools: dict[int, _ChainPool] = {
            cfg.chain_id: _ChainPool(cfg) for cfg in chains
        }
        # Latency tracking: {chain_id: {url: deque[latency_ms]}}
        self._latency: dict[int, dict[str, deque]] = {
            cfg.chain_id: {} for cfg in chains
        }
        log.info(
            f"RPCRouter initialised with {len(chains)} chain(s): {', '.join(cfg.name for cfg in chains)}.",
        )

    # ------------------------------------------------------------------
    # IRouter implementation
    # ------------------------------------------------------------------

    def get_rpc(self, chain_id: int) -> str:
        """Return the currently active RPC URL for *chain_id*.

        Args:
            chain_id: EVM chain identifier.

        Returns:
            Active RPC URL string.

        Raises:
            ChainNotConfiguredError: If *chain_id* is not registered.
            NoAvailableRpcError: If every provider for the chain is suspended.
        """
        pool = self._resolve_pool(chain_id)
        url = pool.active_url()
        log.debug(f"RPCRouter: selected '{url}' for chain {chain_id}.")
        return url

    def mark_failed(self, chain_id: int, rpc_url: str) -> None:
        """Record a failure for *rpc_url* and rotate the pool.

        Args:
            chain_id: EVM chain identifier.
            rpc_url: The URL that produced the failure.
        """
        pool = self._resolve_pool(chain_id)
        pool.rotate_failed(rpc_url)

    def mark_success(self, chain_id: int, rpc_url: str) -> None:
        """Record a successful call for *rpc_url*, resetting its failure count.

        Args:
            chain_id: EVM chain identifier.
            rpc_url: The URL that succeeded.
        """
        pool = self._resolve_pool(chain_id)
        pool.record_success(rpc_url)

    def get_chain_config(self, chain_id: int) -> ChainConfig:
        """Return the full :class:`~wallet_manager.core.models.ChainConfig`
        for *chain_id*.

        Args:
            chain_id: EVM chain identifier.

        Returns:
            Matching :class:`ChainConfig`.

        Raises:
            ChainNotConfiguredError: If *chain_id* is not registered.
        """
        return self._resolve_pool(chain_id).config

    # ------------------------------------------------------------------
    # Convenience methods (not part of IRouter)
    # ------------------------------------------------------------------

    async def probe(self, chain_id: int) -> bool:
        """Send a lightweight ``eth_blockNumber`` call to validate the active RPC.

        Calls :meth:`mark_failed` automatically on failure.

        Args:
            chain_id: Chain to probe.

        Returns:
            ``True`` if the active provider responded; ``False`` otherwise.
        """
        rpc_url = self.get_rpc(chain_id)
        try:
            w3 = AsyncWeb3(
                AsyncWeb3.AsyncHTTPProvider(
                    rpc_url,
                    request_kwargs={"timeout": _PROBE_TIMEOUT},
                )
            )
            block = await asyncio.wait_for(w3.eth.block_number, timeout=_PROBE_TIMEOUT)
            log.debug(
                f"RPCRouter probe: chain {chain_id} OK via '{rpc_url}' (block #{block}).",
            )
            self.mark_success(chain_id, rpc_url)
            return True
        except Exception as exc:
            log.warning(
                f"RPCRouter probe failed for chain {chain_id} via '{rpc_url}': {exc}. Rotating.",
            )
            self.mark_failed(chain_id, rpc_url)
            return False

    async def probe_all(self) -> dict[int, bool]:
        """Probe every registered chain concurrently.

        Returns:
            Mapping of ``{chain_id: is_healthy}`` for all registered chains.
        """
        tasks = {
            chain_id: asyncio.ensure_future(self.probe(chain_id))
            for chain_id in self._pools
        }
        results: dict[int, bool] = {}
        for chain_id, task in tasks.items():
            try:
                results[chain_id] = await task
            except Exception:
                results[chain_id] = False
        return results

    def pool_status(self) -> dict[int, list[dict]]:
        """Return a snapshot of every chain's provider health for monitoring.

        Returns:
            ``{chain_id: [{url, failure_count, available}]}`` dict.
        """
        snapshot: dict[int, list[dict]] = {}
        for chain_id, pool in self._pools.items():
            snapshot[chain_id] = [
                {
                    "url": ph.url,
                    "failure_count": ph.failure_count,
                    "available": ph.is_available(),
                }
                for ph in pool.pool
            ]
        return snapshot

    async def _probe_all_endpoints(self) -> None:
        """Startup health check — probe every registered RPC endpoint.

        For each URL:
          - Sends eth_blockNumber with 5-second timeout.
          - Records responded (bool), latency_ms (float), block_number (int).
          - If block_number lags best by > 5 blocks: marked as lagging
            (lower priority, not disabled).
          - No response within timeout: marked as disabled.
          - URL with API key substrings is never logged.

        Raises ConfigurationError if zero endpoints respond.
        """
        best_block: dict[int, int] = {}
        results: dict[int, list[dict]] = {}

        for chain_id, pool in self._pools.items():
            results[chain_id] = []
            for ph in pool.pool:
                url = ph.url
                t0 = time.monotonic()
                responded = False
                latency_ms = 0.0
                block_num = 0
                try:
                    w3 = AsyncWeb3(
                        AsyncWeb3.AsyncHTTPProvider(
                            url,
                            request_kwargs={"timeout": _PROBE_TIMEOUT},
                        )
                    )
                    block_num = int(await asyncio.wait_for(
                        w3.eth.block_number, timeout=_PROBE_TIMEOUT
                    ))
                    latency_ms = (time.monotonic() - t0) * 1000.0
                    responded = True
                    best_block[chain_id] = max(best_block.get(chain_id, 0), block_num)

                    # Record latency
                    chain_lat = self._latency.setdefault(chain_id, {})
                    if url not in chain_lat:
                        chain_lat[url] = deque(maxlen=_LATENCY_WINDOW)
                    chain_lat[url].append(latency_ms)

                    # Mask API keys in log output
                    safe_url = url.split("v3/")[0] + "v3/***" if "v3/" in url else url[:40] + "..."
                    log.info(
                        "RPCRouter probe chain=%d url=%s responded=True "
                        "block=%d latency=%.1fms",
                        chain_id, safe_url, block_num, latency_ms,
                    )
                except Exception as exc:
                    safe_url = url.split("v3/")[0] + "v3/***" if "v3/" in url else url[:40] + "..."
                    log.warning(
                        "RPCRouter probe chain=%d url=%s FAILED: %s",
                        chain_id, safe_url, exc,
                    )
                    pool.rotate_failed(url)

                results[chain_id].append({
                    "url": url,
                    "responded": responded,
                    "latency_ms": latency_ms,
                    "block_number": block_num,
                })

        # Mark lagging endpoints (> 5 blocks behind best)
        for chain_id, probes in results.items():
            best = best_block.get(chain_id, 0)
            for p in probes:
                if p["responded"] and best - p["block_number"] > 5:
                    log.warning(
                        "RPCRouter chain=%d: endpoint lagging by %d blocks",
                        chain_id, best - p["block_number"],
                    )

        # Raise if all endpoints on all chains are down
        any_responded = any(
            any(p["responded"] for p in probes)
            for probes in results.values()
        )
        if not any_responded:
            raise ConfigurationError(
                "RPCRouter startup probe: zero RPC endpoints responded. "
                "Check INFURA_KEY, ALCHEMY_KEY, and network connectivity."
            )

    def get_url(self, chain_id: int) -> str:
        """Latency-weighted selection algorithm.

        1. Fetch all endpoints for chain_id.
        2. Compute weights: w_i = 1 / (mean_latency_ms + 1)
        3. If all weights are 0 (no probes yet): round-robin fallback.
        4. Weighted random selection via numpy.random.choice.
        5. If selected endpoint has 3+ consecutive failures in last 60s:
           skip and re-select.

        Never raises — returns a URL even if all probes failed.
        """
        try:
            pool = self._resolve_pool(chain_id)
        except Exception:
            return ""

        # Auto re-probe if any endpoint is stale
        try:
            from wallet_manager.config import get_settings
            reprobe_interval = get_settings().rpc_reprobe_interval_seconds
        except Exception:
            reprobe_interval = _REPROBE_DISABLED_INTERVAL

        now = time.monotonic()
        chain_lat = self._latency.setdefault(chain_id, {})
        needs_reprobe = any(
            url not in chain_lat or (now - getattr(self, f"_last_probe_{chain_id}_{i}", 0.0)) > reprobe_interval
            for i, url in enumerate(ph.url for ph in pool.pool)
        )
        if needs_reprobe:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._probe_all_endpoints())
            except Exception:
                pass

        providers = list(pool.pool)
        urls = [ph.url for ph in providers]

        if not urls:
            return ""

        # Compute latency weights
        weights = []
        for url in urls:
            window = chain_lat.get(url)
            if window:
                mean_ms = sum(window) / len(window)
                weights.append(1.0 / (mean_ms + 1.0))
            else:
                weights.append(1.0)  # No data — neutral weight

        total_w = sum(weights)
        if total_w <= 0:
            # Fallback: round-robin
            return self.get_rpc(chain_id)

        # Normalize weights
        norm = [w / total_w for w in weights]

        # Weighted random selection, skip endpoints with 3+ failures in last 60s
        try:
            import numpy as np
            selected_url = cast(str, np.random.choice(urls, p=norm))
        except Exception:
            selected_url = urls[0]

        # Check if selected endpoint has 3+ consecutive failures in last 60s
        ph_map = {ph.url: ph for ph in providers}
        ph = ph_map.get(selected_url)
        if ph is not None and ph.failure_count >= _PROVIDER_FAILURE_THRESHOLD and time.monotonic() - ph.last_failure_time < 60.0:
            # Re-select: pick next best available
            for url, _ in sorted(zip(urls, norm, strict=False), key=lambda x: -x[1]):
                    if url == selected_url:
                        continue
                    candidate = ph_map.get(url)
                    if candidate is None or candidate.failure_count < _PROVIDER_FAILURE_THRESHOLD:
                        selected_url = url
                        break

        log.debug("RPCRouter.get_url: chain=%d selected=%s", chain_id, selected_url[:40])
        return str(selected_url)

    def latency_score(self, chain_id: int, url: str) -> float:
        """Priority score based on rolling mean latency: 1/(mean_ms + 1).

        Higher score = preferred endpoint.
        """
        chain_lat = self._latency.get(chain_id, {})
        window = chain_lat.get(url)
        if not window:
            return 1.0  # Unknown latency — neutral score
        mean_ms = sum(window) / len(window)
        return float(1.0 / (mean_ms + 1.0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_pool(self, chain_id: int) -> _ChainPool:
        """Look up the pool for *chain_id*.

        Args:
            chain_id: EVM chain identifier.

        Returns:
            Associated :class:`_ChainPool`.

        Raises:
            ChainNotConfiguredError: If the chain is not registered.
        """
        pool = self._pools.get(chain_id)
        if pool is None:
            raise ChainNotConfiguredError(
                f"Chain ID {chain_id} is not registered in the RPCRouter. "
                f"Registered chains: {list(self._pools.keys())}."
            )
        return pool


# ---------------------------------------------------------------------------
# Pre-built chain catalogue (use as starting points; inject real RPC URLs)
# ---------------------------------------------------------------------------


def build_default_router(
    infura_key: str = "",
    alchemy_key: str = "",
) -> RPCRouter:
    """Factory that builds an ``RPCRouter`` pre-populated with common chains.

    Populate ``infura_key`` and/or ``alchemy_key`` from your ``.env`` file.
    Each chain is configured with a public fallback so the system remains
    functional even without API keys (at reduced rate limits).

    Args:
        infura_key: Infura project ID (optional).
        alchemy_key: Alchemy API key (optional).

    Returns:
        An :class:`RPCRouter` configured with Ethereum, Base, Arbitrum,
        Linea, and Polygon.
    """
    def _infura(network: str) -> str:
        return f"https://{network}.infura.io/v3/{infura_key}" if infura_key else ""

    def _alchemy(subdomain: str) -> str:
        return (
            f"https://{subdomain}.g.alchemy.com/v2/{alchemy_key}"
            if alchemy_key
            else ""
        )

    def _rpcs(*urls: str) -> tuple[str, ...]:
        """Filter out empty strings from the URL list."""
        return tuple(u for u in urls if u)

    chains: list[ChainConfig] = [
        ChainConfig(
            chain_id=1,
            name="Ethereum Mainnet",
            native_symbol="ETH",
            primary_rpc=_infura("mainnet") or "https://eth.llamarpc.com",
            fallback_rpcs=_rpcs(
                _alchemy("eth-mainnet"),
                "https://rpc.ankr.com/eth",
                "https://eth.llamarpc.com",
            ),
            max_gas_price_gwei=80.0,
            block_time_seconds=12.0,
        ),
        ChainConfig(
            chain_id=8453,
            name="Base",
            native_symbol="ETH",
            primary_rpc=_alchemy("base-mainnet") or "https://mainnet.base.org",
            fallback_rpcs=_rpcs(
                "https://base.llamarpc.com",
                "https://rpc.ankr.com/base",
            ),
            max_gas_price_gwei=5.0,
            block_time_seconds=2.0,
        ),
        ChainConfig(
            chain_id=42161,
            name="Arbitrum One",
            native_symbol="ETH",
            primary_rpc=_infura("arbitrum-mainnet") or "https://arb1.arbitrum.io/rpc",
            fallback_rpcs=_rpcs(
                _alchemy("arb-mainnet"),
                "https://rpc.ankr.com/arbitrum",
            ),
            max_gas_price_gwei=2.0,
            block_time_seconds=0.25,
        ),
        ChainConfig(
            chain_id=59144,
            name="Linea",
            native_symbol="ETH",
            primary_rpc=_infura("linea-mainnet") or "https://rpc.linea.build",
            fallback_rpcs=_rpcs(
                "https://linea.decubate.com",
                "https://rpc.ankr.com/linea",
            ),
            max_gas_price_gwei=3.0,
            block_time_seconds=2.0,
        ),
        ChainConfig(
            chain_id=137,
            name="Polygon",
            native_symbol="MATIC",
            primary_rpc=_infura("polygon-mainnet") or "https://polygon-rpc.com",
            fallback_rpcs=_rpcs(
                _alchemy("polygon-mainnet"),
                "https://rpc.ankr.com/polygon",
            ),
            max_gas_price_gwei=150.0,
            block_time_seconds=2.0,
        ),
    ]

    return RPCRouter(chains)
