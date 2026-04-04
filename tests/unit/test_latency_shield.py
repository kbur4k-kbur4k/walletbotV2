"""
tests/unit/test_latency_shield.py

Unit tests for wallet_manager/strategies/latency_shield.py

Covers:
- NetworkLatencyEmulator: delays within bounds, percentiles from rolling window
- StaleQuoteProtector: fresh / warning / stale / diverged classification
- StaleQuoteProtector: circuit-breaker trips after MAX_CONSECUTIVE_STALE stales
- ConnectionHealthMonitor: health-rate calculation and degraded detection
- LatencyShield: full apply_latency_shield() flow, is_system_healthy()
- LatencyShield: rolling window discards old samples (maxlen)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from wallet_manager.strategies.latency_shield import (
    _LATENCY_HISTORY_SIZE,
    CONNECTION_DEGRADED,
    CONNECTION_HEALTHY,
    LATENCY_MAX_MS,
    LATENCY_MIN_MS,
    MAX_CONSECUTIVE_STALE,
    QUOTE_DIVERGED,
    QUOTE_FRESH,
    QUOTE_STALE,
    QUOTE_WARNING,
    STALE_CIRCUIT_BREAK,
    STALE_QUOTE_THRESHOLD_S,
    STALE_QUOTE_WARNING_S,
    ConnectionHealthMonitor,
    LatencyShield,
    NetworkLatencyEmulator,
    StaleQuoteProtector,
    reset_latency_shield,
)

# ---------------------------------------------------------------------------
# NetworkLatencyEmulator
# ---------------------------------------------------------------------------


class TestNetworkLatencyEmulator:
    """Tests for simulated network latency sampling."""

    @pytest.mark.asyncio
    async def test_records_latency_returns_profile(self):
        """simulate_network_delay() returns a LatencyProfile with positive delay."""
        emulator = NetworkLatencyEmulator()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            profile = await emulator.simulate_network_delay()

        assert profile.simulated_delay_ms > 0
        assert profile.is_acceptable is True

    @pytest.mark.asyncio
    async def test_delay_within_bounds(self):
        """Each simulated delay is clamped to [LATENCY_MIN_MS, LATENCY_MAX_MS]."""
        emulator = NetworkLatencyEmulator()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for _ in range(20):
                profile = await emulator.simulate_network_delay()
                assert LATENCY_MIN_MS <= profile.simulated_delay_ms <= LATENCY_MAX_MS

    @pytest.mark.asyncio
    async def test_measurement_count_increments(self):
        """measurement_count increments with each simulate_network_delay() call."""
        emulator = NetworkLatencyEmulator()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for i in range(1, 4):
                await emulator.simulate_network_delay()
                assert emulator.measurement_count == i

    @pytest.mark.asyncio
    async def test_get_percentiles_returns_positive_values(self):
        """After several samples, p50/p95/p99 are positive floats."""
        emulator = NetworkLatencyEmulator()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for _ in range(10):
                await emulator.simulate_network_delay()

        percentiles = emulator.get_percentiles()
        assert percentiles["p50"] > 0
        assert percentiles["p95"] >= percentiles["p50"]
        assert percentiles["p99"] >= percentiles["p95"]

    @pytest.mark.asyncio
    async def test_get_percentiles_empty_returns_zeros(self):
        """Before any samples, percentiles are all 0.0."""
        emulator = NetworkLatencyEmulator()
        result = emulator.get_percentiles()
        assert result == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}

    @pytest.mark.asyncio
    async def test_rolling_window_behavior(self):
        """History deque is bounded by _LATENCY_HISTORY_SIZE."""
        emulator = NetworkLatencyEmulator()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Fill beyond the window
            for _ in range(_LATENCY_HISTORY_SIZE + 20):
                await emulator.simulate_network_delay()

        assert len(emulator._history) == _LATENCY_HISTORY_SIZE


# ---------------------------------------------------------------------------
# StaleQuoteProtector
# ---------------------------------------------------------------------------


class TestStaleQuoteProtector:
    """Tests for quote freshness and stale-circuit-breaker logic."""

    def _fresh_ts(self) -> float:
        """Timestamp that is definitely fresh."""
        return time.time() - 0.1

    def _stale_ts(self) -> float:
        """Timestamp older than STALE_QUOTE_THRESHOLD_S."""
        return time.time() - (STALE_QUOTE_THRESHOLD_S + 0.5)

    def _warning_ts(self) -> float:
        """Timestamp between WARNING and STALE thresholds."""
        midpoint = (STALE_QUOTE_WARNING_S + STALE_QUOTE_THRESHOLD_S) / 2
        return time.time() - midpoint

    # -- Fresh quote

    def test_fresh_quote_is_not_stale(self):
        protector = StaleQuoteProtector()
        result = protector.validate_quote(price=100.0, quote_timestamp=self._fresh_ts())
        assert result.is_stale is False
        assert result.should_invalidate_signals is False
        assert result.reason_code == QUOTE_FRESH

    # -- Warning quote

    def test_warning_quote_sets_is_warning(self):
        protector = StaleQuoteProtector()
        result = protector.validate_quote(price=100.0, quote_timestamp=self._warning_ts())
        assert result.is_warning is True
        assert result.is_stale is False
        assert result.reason_code == QUOTE_WARNING

    # -- Stale quote

    def test_high_latency_triggers_alert(self):
        """Stale quote: is_stale=True, should_invalidate_signals=True."""
        protector = StaleQuoteProtector()
        result = protector.validate_quote(price=100.0, quote_timestamp=self._stale_ts())
        assert result.is_stale is True
        assert result.should_invalidate_signals is True
        assert result.reason_code in (QUOTE_STALE, STALE_CIRCUIT_BREAK)

    def test_low_latency_no_alert(self):
        """Fresh quote: should_invalidate_signals=False."""
        protector = StaleQuoteProtector()
        result = protector.validate_quote(price=100.0, quote_timestamp=self._fresh_ts())
        assert result.should_invalidate_signals is False

    # -- Price divergence

    def test_price_divergence_invalidates_signal(self):
        """Price > 0.3% from reference triggers QUOTE_DIVERGED."""
        protector = StaleQuoteProtector()
        ref = 1000.0
        diverged_price = ref * 1.005  # 0.5% divergence — above 0.3% threshold
        result = protector.validate_quote(
            price=diverged_price,
            quote_timestamp=self._fresh_ts(),
            reference_price=ref,
        )
        assert result.should_invalidate_signals is True
        assert result.reason_code == QUOTE_DIVERGED

    def test_small_divergence_does_not_invalidate(self):
        """Price < 0.3% from reference is QUOTE_FRESH."""
        protector = StaleQuoteProtector()
        ref = 1000.0
        close_price = ref * 1.001  # 0.1% — below threshold
        result = protector.validate_quote(
            price=close_price,
            quote_timestamp=self._fresh_ts(),
            reference_price=ref,
        )
        assert result.should_invalidate_signals is False
        assert result.reason_code == QUOTE_FRESH

    # -- Circuit-breaker

    def test_circuit_breaker_trips_after_consecutive_stales(self):
        """After MAX_CONSECUTIVE_STALE stale observations, circuit breaks."""
        protector = StaleQuoteProtector()
        for _ in range(MAX_CONSECUTIVE_STALE):
            protector.validate_quote(price=100.0, quote_timestamp=self._stale_ts())

        assert protector.is_circuit_broken is True

    def test_circuit_breaker_active_invalidates_all(self):
        """Once tripped, even non-stale observations are invalidated."""
        protector = StaleQuoteProtector()
        # Trip the breaker
        for _ in range(MAX_CONSECUTIVE_STALE):
            protector.validate_quote(price=100.0, quote_timestamp=self._stale_ts())

        # Now pass a technically fresh quote — still invalidated
        result = protector.validate_quote(price=100.0, quote_timestamp=self._fresh_ts())
        assert result.reason_code == STALE_CIRCUIT_BREAK
        assert result.should_invalidate_signals is True

    def test_reset_clears_circuit_breaker(self):
        """reset_stale_counter() clears circuit-breaker state."""
        protector = StaleQuoteProtector()
        for _ in range(MAX_CONSECUTIVE_STALE):
            protector.validate_quote(price=100.0, quote_timestamp=self._stale_ts())

        assert protector.is_circuit_broken is True
        protector.reset_stale_counter()
        assert protector.is_circuit_broken is False
        assert protector.consecutive_stale == 0

    # -- Freshness multiplier

    def test_freshness_multiplier_one_for_very_fresh(self):
        """Quote < 0.5 s old -> multiplier 1.0."""
        protector = StaleQuoteProtector()
        ts = time.time() - 0.1  # 100 ms old
        multiplier = protector.get_freshness_multiplier(ts)
        assert multiplier == 1.0

    def test_freshness_multiplier_zero_for_stale(self):
        """Quote >= STALE_QUOTE_THRESHOLD_S old -> multiplier 0.0."""
        protector = StaleQuoteProtector()
        ts = time.time() - (STALE_QUOTE_THRESHOLD_S + 1.0)
        multiplier = protector.get_freshness_multiplier(ts)
        assert multiplier == 0.0


# ---------------------------------------------------------------------------
# ConnectionHealthMonitor
# ---------------------------------------------------------------------------


class TestConnectionHealthMonitor:
    """Tests for rolling health-rate monitoring."""

    def test_initial_health_rate_is_one(self):
        monitor = ConnectionHealthMonitor()
        assert monitor.get_health_rate() == 1.0

    def test_all_healthy_checks_rate_one(self):
        monitor = ConnectionHealthMonitor()
        for _ in range(5):
            monitor.record_health_check(is_healthy=True)
        assert monitor.get_health_rate() == 1.0

    def test_health_rate_reflects_failures(self):
        monitor = ConnectionHealthMonitor()
        monitor.record_health_check(is_healthy=True)
        monitor.record_health_check(is_healthy=False)  # 50 % failure
        rate = monitor.get_health_rate()
        assert rate == pytest.approx(0.5)

    def test_get_health_report_healthy_status(self):
        monitor = ConnectionHealthMonitor()
        for _ in range(5):
            monitor.record_health_check(is_healthy=True)
        report = monitor.get_health_report()
        assert report["status"] == CONNECTION_HEALTHY

    def test_get_health_report_degraded_status(self):
        """After 5+ checks with >20% failure rate, status is CONNECTION_DEGRADED."""
        monitor = ConnectionHealthMonitor()
        # 2 healthy, 4 failed = 33% healthy < 80% threshold
        for _ in range(2):
            monitor.record_health_check(is_healthy=True)
        for _ in range(4):
            monitor.record_health_check(is_healthy=False)
        report = monitor.get_health_report()
        assert report["status"] == CONNECTION_DEGRADED

    def test_total_checks_counts_all_records(self):
        monitor = ConnectionHealthMonitor()
        for _ in range(7):
            monitor.record_health_check(is_healthy=True)
        assert monitor.total_checks == 7

    def test_total_failures_counts_only_failures(self):
        monitor = ConnectionHealthMonitor()
        monitor.record_health_check(is_healthy=True)
        monitor.record_health_check(is_healthy=False)
        monitor.record_health_check(is_healthy=False)
        assert monitor.total_failures == 2


# ---------------------------------------------------------------------------
# LatencyShield
# ---------------------------------------------------------------------------


class TestLatencyShield:
    """Tests for the unified latency shield coordinator."""

    def setup_method(self):
        reset_latency_shield()

    @pytest.mark.asyncio
    async def test_records_latency_returns_positive_float(self):
        """apply_latency_shield() returns a LatencyProfile with positive total_latency."""
        shield = LatencyShield()
        now = time.time()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            _validation, profile = await shield.apply_latency_shield(
                price=100.0,
                quote_timestamp=now - 0.05,  # fresh
            )

        assert profile.total_latency_ms > 0
        assert isinstance(profile.total_latency_ms, (int, float))

    @pytest.mark.asyncio
    async def test_high_latency_stale_triggers_alert(self):
        """Stale quote causes should_invalidate_signals=True in validation."""
        shield = LatencyShield()
        stale_ts = time.time() - (STALE_QUOTE_THRESHOLD_S + 1.0)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            validation, _ = await shield.apply_latency_shield(
                price=100.0,
                quote_timestamp=stale_ts,
            )

        assert validation.should_invalidate_signals is True

    @pytest.mark.asyncio
    async def test_low_latency_fresh_no_alert(self):
        """Fresh quote causes should_invalidate_signals=False."""
        shield = LatencyShield()
        fresh_ts = time.time() - 0.05

        with patch("asyncio.sleep", new_callable=AsyncMock):
            validation, _ = await shield.apply_latency_shield(
                price=100.0,
                quote_timestamp=fresh_ts,
            )

        assert validation.should_invalidate_signals is False

    @pytest.mark.asyncio
    async def test_rolling_window_behavior(self):
        """Latency history never exceeds _LATENCY_HISTORY_SIZE entries."""
        shield = LatencyShield()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            fresh_ts = time.time() - 0.05
            for _ in range(_LATENCY_HISTORY_SIZE + 30):
                await shield.apply_latency_shield(
                    price=100.0, quote_timestamp=fresh_ts
                )

        assert len(shield.emulator._history) == _LATENCY_HISTORY_SIZE

    @pytest.mark.asyncio
    async def test_get_latency_report_after_samples(self):
        """get_latency_report() returns non-zero p95 after several samples."""
        shield = LatencyShield()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            fresh_ts = time.time() - 0.05
            for _ in range(10):
                await shield.apply_latency_shield(
                    price=100.0, quote_timestamp=fresh_ts
                )

        report = await shield.get_latency_report()
        assert report.p95_latency_ms > 0
        assert report.p95_latency_ms >= report.p50_latency_ms

    def test_is_system_healthy_true_initially(self):
        """Freshly created shield is healthy (no stale quotes, no failed checks)."""
        shield = LatencyShield()
        assert shield.is_system_healthy() is True

    def test_is_system_healthy_false_after_circuit_break(self):
        """System is unhealthy when circuit-breaker is tripped."""
        shield = LatencyShield()
        stale_ts = time.time() - (STALE_QUOTE_THRESHOLD_S + 0.5)
        for _ in range(MAX_CONSECUTIVE_STALE):
            shield.quote_protector.validate_quote(
                price=100.0, quote_timestamp=stale_ts
            )

        assert shield.is_system_healthy() is False

    def test_paper_trade_default_true(self):
        """paper_trade defaults to True."""
        shield = LatencyShield()
        assert shield.paper_trade is True

    def test_paper_trade_setter(self):
        """paper_trade setter changes the mode."""
        shield = LatencyShield()
        shield.paper_trade = False
        assert shield.paper_trade is False
