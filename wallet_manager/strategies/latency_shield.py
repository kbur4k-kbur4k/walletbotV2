"""
wallet_manager/strategies/latency_shield.py

QUANT-X Latency Shield — Execution Latency Emulation & Stale Quote Protection.

Simulates realistic OCI Frankfurt -> Base Sequencer network latency for
paper-trade back-testing and protects live execution from acting on stale
market data.  Implements:

1. Network Latency Emulation: Gaussian delay N(mu=450ms, sigma=150ms) clamped
   to [200ms, 800ms] modelling real RPC round-trips.
2. Stale Quote Protection: Quote age > 2s invalidates all signals; > 1s
   triggers a warning and halves confidence.  Price divergence > 0.3%
   from a reference price also invalidates.
3. Connection Health Monitoring: Rolling heartbeat checks with degraded-
   connection warning when health rate drops below 80%.
4. Circuit Breaker: 5+ consecutive stale quotes triggers STALE_CIRCUIT_BREAK.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Network latency emulation (OCI Frankfurt -> Base Sequencer)
LATENCY_MIN_MS = 200
LATENCY_MAX_MS = 800
LATENCY_MEAN_MS = 450
LATENCY_STD_MS = 150

# Stale quote thresholds
STALE_QUOTE_THRESHOLD_S = 2.0
STALE_QUOTE_WARNING_S = 1.0
PRICE_DIVERGENCE_THRESHOLD_PCT = 0.003

# Connection health
HEARTBEAT_INTERVAL_S = 30.0
MAX_CONSECUTIVE_STALE = 5

# Internal
_LATENCY_HISTORY_SIZE = 100
_HEALTH_HISTORY_SIZE = 100
_FRESHNESS_FLOOR_S = 0.5
_RPC_TIMEOUT_S = 5.0
_HEALTH_RATE_MINIMUM = 0.80

# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

QUOTE_FRESH = "QUOTE_FRESH"
QUOTE_WARNING = "QUOTE_WARNING"
QUOTE_STALE = "QUOTE_STALE"
QUOTE_DIVERGED = "QUOTE_DIVERGED"
STALE_CIRCUIT_BREAK = "STALE_CIRCUIT_BREAK"
LATENCY_OK = "LATENCY_OK"
LATENCY_HIGH = "LATENCY_HIGH"
CONNECTION_HEALTHY = "CONNECTION_HEALTHY"
CONNECTION_DEGRADED = "CONNECTION_DEGRADED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyProfile:
    """Snapshot of a single latency measurement."""

    simulated_delay_ms: float
    actual_rpc_latency_ms: float
    total_latency_ms: float
    is_acceptable: bool
    timestamp: float


@dataclass(frozen=True)
class QuoteValidation:
    """Result of validating a market-data quote."""

    price: float
    quote_age_s: float
    is_stale: bool
    is_warning: bool
    price_divergence_pct: float
    should_invalidate_signals: bool
    reason_code: str
    timestamp: float


@dataclass(frozen=True)
class LatencyReport:
    """Aggregated latency and health statistics."""

    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    stale_quote_count: int
    invalidated_signal_count: int
    uptime_pct: float
    timestamp: float


# ---------------------------------------------------------------------------
# Network Latency Emulator
# ---------------------------------------------------------------------------


class NetworkLatencyEmulator:
    """Simulates OCI Frankfurt -> Base Sequencer network latency.

    Gaussian distribution centred at LATENCY_MEAN_MS with LATENCY_STD_MS
    standard deviation, clamped to [LATENCY_MIN_MS, LATENCY_MAX_MS].
    """

    def __init__(
        self,
        mean_ms: float = LATENCY_MEAN_MS,
        std_ms: float = LATENCY_STD_MS,
        min_ms: float = LATENCY_MIN_MS,
        max_ms: float = LATENCY_MAX_MS,
    ) -> None:
        self._mean = mean_ms
        self._std = std_ms
        self._min = min_ms
        self._max = max_ms
        self._history: deque[float] = deque(maxlen=_LATENCY_HISTORY_SIZE)
        self._total_delay_ms: float = 0.0
        self._measurement_count: int = 0

    async def simulate_network_delay(self) -> LatencyProfile:
        """Sample a Gaussian delay, sleep, and return a LatencyProfile."""
        delay_ms = random.gauss(self._mean, self._std)
        delay_ms = max(self._min, min(self._max, delay_ms))

        t0 = time.monotonic()
        await asyncio.sleep(delay_ms / 1000.0)
        actual_sleep_ms = (time.monotonic() - t0) * 1000.0

        self._history.append(delay_ms)
        self._total_delay_ms += delay_ms
        self._measurement_count += 1

        profile = LatencyProfile(
            simulated_delay_ms=delay_ms,
            actual_rpc_latency_ms=0.0,
            total_latency_ms=delay_ms,
            is_acceptable=delay_ms <= LATENCY_MAX_MS,
            timestamp=time.time(),
        )
        log.debug(
            "Simulated network delay: %.1f ms (sleep accuracy: %.1f ms)",
            delay_ms, actual_sleep_ms,
        )
        return profile

    async def measure_rpc_latency(self, rpc_url: str) -> float:
        """Send eth_blockNumber and measure round-trip.  Returns ms or -1."""
        if not rpc_url:
            return -1.0

        import aiohttp

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1,
        }
        t0 = time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    rpc_url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=_RPC_TIMEOUT_S),
                ) as resp:
                    await resp.json()
            latency_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:
            log.warning("RPC latency measurement failed: %s", exc)
            return -1.0

        self._history.append(latency_ms)
        self._total_delay_ms += latency_ms
        self._measurement_count += 1
        log.debug("RPC latency to %s: %.1f ms", rpc_url, latency_ms)
        return latency_ms

    def get_percentiles(self) -> dict[str, float]:
        """Return p50/p95/p99/avg from the rolling window."""
        if not self._history:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}
        arr = np.array(self._history)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "avg": float(np.mean(arr)),
        }

    @property
    def measurement_count(self) -> int:
        return self._measurement_count


# ---------------------------------------------------------------------------
# Stale Quote Protector
# ---------------------------------------------------------------------------


class StaleQuoteProtector:
    """Validates quote freshness and detects price divergence.

    Trips STALE_CIRCUIT_BREAK after MAX_CONSECUTIVE_STALE consecutive stale
    observations.
    """

    def __init__(self) -> None:
        self._consecutive_stale: int = 0
        self._total_stale_count: int = 0
        self._total_invalidated: int = 0
        self._circuit_broken: bool = False

    def validate_quote(
        self,
        price: float,
        quote_timestamp: float,
        reference_price: float | None = None,
    ) -> QuoteValidation:
        """Validate a market-data quote for freshness and price accuracy."""
        now = time.time()
        quote_age_s = now - quote_timestamp

        # Price divergence against reference
        divergence_pct = 0.0
        if reference_price is not None and reference_price > 0:
            divergence_pct = abs(price - reference_price) / reference_price

        is_stale = False
        is_warning = False
        should_invalidate = False
        reason_code = QUOTE_FRESH

        # Priority 1: Circuit-breaker already tripped
        if self._circuit_broken:
            is_stale = True
            should_invalidate = True
            reason_code = STALE_CIRCUIT_BREAK
            log.warning(
                "STALE_CIRCUIT_BREAK active — all signals invalidated "
                "(consecutive stale: %d)", self._consecutive_stale,
            )

        # Priority 2: Quote age exceeds stale threshold
        elif quote_age_s > STALE_QUOTE_THRESHOLD_S:
            is_stale = True
            should_invalidate = True
            reason_code = QUOTE_STALE
            self._consecutive_stale += 1
            self._total_stale_count += 1
            self._total_invalidated += 1

            if self._consecutive_stale >= MAX_CONSECUTIVE_STALE:
                self._circuit_broken = True
                reason_code = STALE_CIRCUIT_BREAK
                log.error(
                    "STALE_CIRCUIT_BREAK triggered after %d consecutive "
                    "stale quotes (age: %.2f s)",
                    self._consecutive_stale, quote_age_s,
                )
            else:
                log.warning(
                    "Stale quote — age: %.2f s, consecutive: %d/%d",
                    quote_age_s, self._consecutive_stale,
                    MAX_CONSECUTIVE_STALE,
                )

        # Priority 3: Price divergence beyond threshold
        elif (
            reference_price is not None
            and divergence_pct > PRICE_DIVERGENCE_THRESHOLD_PCT
        ):
            should_invalidate = True
            reason_code = QUOTE_DIVERGED
            self._total_invalidated += 1
            self._reset_on_fresh()
            log.warning(
                "Price divergence: %.4f %% (threshold: %.4f %%)",
                divergence_pct * 100, PRICE_DIVERGENCE_THRESHOLD_PCT * 100,
            )

        # Priority 4: Quote age exceeds warning threshold
        elif quote_age_s > STALE_QUOTE_WARNING_S:
            is_warning = True
            reason_code = QUOTE_WARNING
            self._reset_on_fresh()
            log.info(
                "Quote age warning — %.2f s (threshold: %.2f s)",
                quote_age_s, STALE_QUOTE_WARNING_S,
            )

        # Priority 5: Fresh quote
        else:
            self._reset_on_fresh()
            log.debug(
                "Quote fresh — age: %.3f s, divergence: %.4f %%",
                quote_age_s, divergence_pct * 100,
            )

        return QuoteValidation(
            price=price,
            quote_age_s=quote_age_s,
            is_stale=is_stale,
            is_warning=is_warning,
            price_divergence_pct=divergence_pct,
            should_invalidate_signals=should_invalidate,
            reason_code=reason_code,
            timestamp=now,
        )

    def get_freshness_multiplier(self, quote_timestamp: float) -> float:
        """Confidence multiplier based on quote age.

        Returns 1.0 for quotes < 0.5 s old, decays linearly to 0.0 at 2.0 s.
        """
        age_s = time.time() - quote_timestamp
        if age_s <= _FRESHNESS_FLOOR_S:
            return 1.0
        if age_s >= STALE_QUOTE_THRESHOLD_S:
            return 0.0
        decay_range = STALE_QUOTE_THRESHOLD_S - _FRESHNESS_FLOOR_S
        position = age_s - _FRESHNESS_FLOOR_S
        return max(0.0, 1.0 - (position / decay_range))

    def reset_stale_counter(self) -> None:
        """Reset the consecutive stale counter and circuit-breaker."""
        if self._circuit_broken:
            log.info(
                "STALE_CIRCUIT_BREAK cleared — fresh data received "
                "(was %d consecutive stale)", self._consecutive_stale,
            )
        self._consecutive_stale = 0
        self._circuit_broken = False

    @property
    def consecutive_stale(self) -> int:
        return self._consecutive_stale

    @property
    def total_stale_count(self) -> int:
        return self._total_stale_count

    @property
    def total_invalidated(self) -> int:
        return self._total_invalidated

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    def _reset_on_fresh(self) -> None:
        """Reset consecutive stale counter on non-stale observation."""
        if self._consecutive_stale > 0:
            log.debug("Stale counter reset (was %d)", self._consecutive_stale)
        self._consecutive_stale = 0
        self._circuit_broken = False


# ---------------------------------------------------------------------------
# Connection Health Monitor
# ---------------------------------------------------------------------------


class ConnectionHealthMonitor:
    """Monitors RPC endpoint health via periodic eth_blockNumber calls.

    Triggers CONNECTION_DEGRADED when success rate < 80%.
    """

    def __init__(self) -> None:
        self._health_history: deque[bool] = deque(maxlen=_HEALTH_HISTORY_SIZE)
        self._latency_history: deque[float] = deque(
            maxlen=_HEALTH_HISTORY_SIZE,
        )
        self._total_checks: int = 0
        self._total_failures: int = 0

    async def check_websocket_health(self, rpc_url: str) -> bool:
        """Send eth_blockNumber and verify response within timeout."""
        if not rpc_url:
            log.debug("No RPC URL configured — skipping health check")
            return False

        import aiohttp

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1,
        }
        t0 = time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    rpc_url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=_RPC_TIMEOUT_S),
                ) as resp:
                    data = await resp.json()
            latency_ms = (time.monotonic() - t0) * 1000.0

            if "result" not in data:
                log.warning("Health check: unexpected response — %s", data)
                self.record_health_check(is_healthy=False)
                return False

            self._latency_history.append(latency_ms)
            self.record_health_check(is_healthy=True)
            log.debug(
                "Health check OK — block %s, latency %.1f ms",
                data["result"], latency_ms,
            )
            return True

        except TimeoutError:
            log.warning("Health check timed out after %.1f s", _RPC_TIMEOUT_S)
            self.record_health_check(is_healthy=False)
            return False

        except Exception as exc:
            log.warning("Health check failed: %s", exc)
            self.record_health_check(is_healthy=False)
            return False

    def record_health_check(self, is_healthy: bool) -> None:
        """Record health-check outcome.  Warns if rate < 80%."""
        self._health_history.append(is_healthy)
        self._total_checks += 1
        if not is_healthy:
            self._total_failures += 1

        health_rate = self.get_health_rate()
        if (
            len(self._health_history) >= 5
            and health_rate < _HEALTH_RATE_MINIMUM
        ):
            log.warning(
                "CONNECTION_DEGRADED — health rate %.1f %% (min: %.1f %%)",
                health_rate * 100, _HEALTH_RATE_MINIMUM * 100,
            )

    def get_health_rate(self) -> float:
        """Fraction of successful checks in window.  1.0 if none recorded."""
        if not self._health_history:
            return 1.0
        return sum(self._health_history) / len(self._health_history)

    def get_health_report(self) -> dict:
        """Return uptime, avg_latency, health_rate, total_checks, status."""
        health_rate = self.get_health_rate()
        avg_latency = (
            float(np.mean(self._latency_history))
            if self._latency_history else 0.0
        )
        status = CONNECTION_HEALTHY
        if (
            len(self._health_history) >= 5
            and health_rate < _HEALTH_RATE_MINIMUM
        ):
            status = CONNECTION_DEGRADED

        return {
            "uptime": health_rate,
            "avg_latency": avg_latency,
            "health_rate": health_rate,
            "total_checks": self._total_checks,
            "total_failures": self._total_failures,
            "status": status,
        }

    @property
    def total_checks(self) -> int:
        return self._total_checks

    @property
    def total_failures(self) -> int:
        return self._total_failures


# ---------------------------------------------------------------------------
# Latency Shield — Main Coordinator
# ---------------------------------------------------------------------------


class LatencyShield:
    """Combines latency emulation, stale-quote protection, and connection
    health monitoring into a single unified shield.

    In paper-trade mode (default) network delays are simulated via Gaussian
    distribution.  In live mode the shield measures real RPC round-trip
    times and validates quote freshness before allowing signal generation.
    """

    def __init__(self, rpc_url: str = "") -> None:
        self._rpc_url = rpc_url
        self._emulator = NetworkLatencyEmulator()
        self._quote_protector = StaleQuoteProtector()
        self._health_monitor = ConnectionHealthMonitor()
        self._paper_trade: bool = True
        self._created_at: float = time.time()
        log.info(
            "LatencyShield initialised — RPC: %s, paper_trade: %s",
            rpc_url or "(none)", self._paper_trade,
        )

    async def apply_latency_shield(
        self,
        price: float,
        quote_timestamp: float,
        reference_price: float | None = None,
    ) -> tuple[QuoteValidation, LatencyProfile]:
        """Apply the full latency shield to a quote.

        1. Validate quote freshness.
        2. If paper-trade mode, simulate network delay.
        3. Return combined (QuoteValidation, LatencyProfile).
        """
        # Step 1: Validate quote freshness
        validation = self._quote_protector.validate_quote(
            price=price,
            quote_timestamp=quote_timestamp,
            reference_price=reference_price,
        )

        # Step 2: Simulate or measure network latency
        if self._paper_trade:
            profile = await self._emulator.simulate_network_delay()
        else:
            rpc_latency = await self._emulator.measure_rpc_latency(
                self._rpc_url,
            )
            profile = LatencyProfile(
                simulated_delay_ms=0.0,
                actual_rpc_latency_ms=max(rpc_latency, 0.0),
                total_latency_ms=max(rpc_latency, 0.0),
                is_acceptable=0 < rpc_latency <= LATENCY_MAX_MS,
                timestamp=time.time(),
            )

        # Step 3: Log combined result
        if validation.should_invalidate_signals:
            log.warning(
                "LatencyShield BLOCKED — reason: %s, age: %.2f s, "
                "latency: %.1f ms",
                validation.reason_code, validation.quote_age_s,
                profile.total_latency_ms,
            )
        else:
            log.debug(
                "LatencyShield OK — reason: %s, age: %.3f s, "
                "latency: %.1f ms",
                validation.reason_code, validation.quote_age_s,
                profile.total_latency_ms,
            )

        return validation, profile

    async def get_latency_report(self) -> LatencyReport:
        """Build a comprehensive latency and health report."""
        percentiles = self._emulator.get_percentiles()
        health = self._health_monitor.get_health_report()
        return LatencyReport(
            avg_latency_ms=percentiles["avg"],
            p50_latency_ms=percentiles["p50"],
            p95_latency_ms=percentiles["p95"],
            p99_latency_ms=percentiles["p99"],
            stale_quote_count=self._quote_protector.total_stale_count,
            invalidated_signal_count=self._quote_protector.total_invalidated,
            uptime_pct=health["uptime"] * 100.0,
            timestamp=time.time(),
        )

    def is_system_healthy(self) -> bool:
        """True when circuit-breaker is off and health rate >= 80%."""
        if self._quote_protector.is_circuit_broken:
            return False
        health_rate = self._health_monitor.get_health_rate()
        return not (
            self._health_monitor.total_checks >= 5
            and health_rate < _HEALTH_RATE_MINIMUM
        )

    @property
    def emulator(self) -> NetworkLatencyEmulator:
        return self._emulator

    @property
    def quote_protector(self) -> StaleQuoteProtector:
        return self._quote_protector

    @property
    def health_monitor(self) -> ConnectionHealthMonitor:
        return self._health_monitor

    @property
    def rpc_url(self) -> str:
        return self._rpc_url

    @property
    def paper_trade(self) -> bool:
        return self._paper_trade

    @paper_trade.setter
    def paper_trade(self, value: bool) -> None:
        self._paper_trade = value
        log.info("Paper-trade mode set to %s", value)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_latency_shield: LatencyShield | None = None


def get_latency_shield(rpc_url: str = "") -> LatencyShield:
    """Return the process-wide LatencyShield singleton."""
    global _latency_shield
    if _latency_shield is None:
        _latency_shield = LatencyShield(rpc_url=rpc_url)
    return _latency_shield


def reset_latency_shield() -> None:
    """Reset the singleton (for tests)."""
    global _latency_shield
    _latency_shield = None
