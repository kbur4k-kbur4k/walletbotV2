"""
tests/unit/test_mev_shield.py

Unit tests for wallet_manager/strategies/mev_shield.py

Covers:
- GaussianJitter: delays within [JITTER_MIN_MS, JITTER_MAX_MS + 50ms micro-offset]
- GaussianJitter: delays are not constant across calls
- GaussianJitter: avg_jitter_ms and total_jitter_seconds properties
- DynamicRBF: calculate_bumped_gas() increases gas price by ~15% on attempt 1
- DynamicRBF: escalating bumps on subsequent attempts
- RPCProviderState: record_success / record_failure / success_rate
- RPCProviderState: unhealthy after RPC_FAILURE_THRESHOLD consecutive failures
- FailoverRPCCluster: get_best_rpc prefers private provider
- FailoverRPCCluster: no duplicate providers added
- MEVShield: get_protected_rpc falls back to public Base RPC when empty cluster
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from wallet_manager.strategies.mev_shield import (
    JITTER_MAX_MS,
    JITTER_MIN_MS,
    RBF_GAS_BUMP_PCT,
    RPC_FAILURE_THRESHOLD,
    DynamicRBF,
    FailoverRPCCluster,
    GaussianJitter,
    MEVShield,
    RPCProviderState,
    reset_mev_shield,
)

# ---------------------------------------------------------------------------
# GaussianJitter
# ---------------------------------------------------------------------------


class TestGaussianJitter:
    """Tests for random micro-delay generation."""

    @pytest.mark.asyncio
    async def test_jitter_delay_within_bounds(self):
        """Each call must return a delay within [JITTER_MIN_MS, JITTER_MAX_MS + 50]."""
        jitter = GaussianJitter()
        upper_bound = JITTER_MAX_MS + 50.0  # accounts for micro-offset up to 50 ms

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for _ in range(20):
                delay = await jitter.apply()
                assert delay >= JITTER_MIN_MS, (
                    f"Delay {delay:.2f} ms is below minimum {JITTER_MIN_MS} ms"
                )
                assert delay <= upper_bound, (
                    f"Delay {delay:.2f} ms exceeds maximum {upper_bound} ms"
                )

    @pytest.mark.asyncio
    async def test_jitter_delays_are_not_constant(self):
        """10 calls should produce at least 2 distinct values (random)."""
        jitter = GaussianJitter()
        delays: list[float] = []

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for _ in range(10):
                delays.append(await jitter.apply())

        distinct = set(delays)
        assert len(distinct) >= 2, (
            f"Expected at least 2 distinct delay values, got: {distinct}"
        )

    @pytest.mark.asyncio
    async def test_avg_jitter_ms_updates_after_calls(self):
        """avg_jitter_ms must be positive after at least one apply() call."""
        jitter = GaussianJitter()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await jitter.apply()

        assert jitter.avg_jitter_ms > 0.0

    @pytest.mark.asyncio
    async def test_avg_jitter_ms_zero_before_any_call(self):
        """avg_jitter_ms is 0.0 before any apply() call."""
        jitter = GaussianJitter()
        assert jitter.avg_jitter_ms == 0.0

    @pytest.mark.asyncio
    async def test_total_jitter_seconds_accumulates(self):
        """total_jitter_seconds accumulates across multiple apply() calls."""
        jitter = GaussianJitter()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for _ in range(3):
                await jitter.apply()

        assert jitter.total_jitter_seconds > 0.0
        # Sanity: total = avg * count (in seconds)
        expected = jitter.avg_jitter_ms * 3 / 1000.0
        assert pytest.approx(jitter.total_jitter_seconds, rel=1e-6) == expected

    @pytest.mark.asyncio
    async def test_custom_bounds_respected(self):
        """Custom min/max passed to constructor are respected as clamp bounds."""
        jitter = GaussianJitter(mean_ms=500, std_ms=10, min_ms=490, max_ms=510)
        upper_bound = 510 + 50.0  # plus micro-offset

        with patch("asyncio.sleep", new_callable=AsyncMock):
            for _ in range(20):
                delay = await jitter.apply()
                assert delay >= 490
                assert delay <= upper_bound


# ---------------------------------------------------------------------------
# DynamicRBF
# ---------------------------------------------------------------------------


class TestDynamicRBF:
    """Tests for the Replace-By-Fee gas bump calculation."""

    def test_rbf_bump_increases_gas_attempt_1(self):
        """Attempt 1: 15% bump on 10 Gwei -> ~11.5 Gwei."""
        rbf = DynamicRBF()
        gas_10_gwei = int(10e9)  # 10 Gwei in wei

        bumped = rbf.calculate_bumped_gas(gas_10_gwei, attempt=1)

        expected = int(gas_10_gwei * (1 + RBF_GAS_BUMP_PCT))  # 11.5 Gwei
        assert bumped == expected

    def test_rbf_bump_escalates_on_attempt_2(self):
        """Attempt 2 uses base * 1.5 escalation (15% * 1.5 = 22.5%)."""
        rbf = DynamicRBF()
        gas_price = int(10e9)

        bump1 = rbf.calculate_bumped_gas(gas_price, attempt=1)
        bump2 = rbf.calculate_bumped_gas(gas_price, attempt=2)

        assert bump2 > bump1, "Attempt 2 should produce a higher gas than attempt 1"

    def test_rbf_bump_escalates_on_attempt_3(self):
        """Attempt 3 uses base * 2.0 escalation (15% * 2 = 30%)."""
        rbf = DynamicRBF()
        gas_price = int(10e9)

        bump2 = rbf.calculate_bumped_gas(gas_price, attempt=2)
        bump3 = rbf.calculate_bumped_gas(gas_price, attempt=3)

        assert bump3 > bump2, "Attempt 3 should produce a higher gas than attempt 2"

    def test_rbf_bump_returns_integer(self):
        """calculate_bumped_gas must return an int (wei is always integer)."""
        rbf = DynamicRBF()
        result = rbf.calculate_bumped_gas(int(5e9), attempt=1)
        assert isinstance(result, int)

    def test_custom_base_bump_pct(self):
        """Custom base_bump_pct is honoured."""
        rbf = DynamicRBF(base_bump_pct=0.10)  # 10%
        gas = int(10e9)
        bumped = rbf.calculate_bumped_gas(gas, attempt=1)
        expected = int(gas * 1.10)
        assert bumped == expected


# ---------------------------------------------------------------------------
# RPCProviderState
# ---------------------------------------------------------------------------


class TestRPCProviderState:
    """Tests for individual provider health tracking."""

    def test_initial_state_is_healthy(self):
        state = RPCProviderState(url="https://rpc.example.com")
        assert state.is_healthy is True
        assert state.consecutive_failures == 0

    def test_record_success_resets_failures(self):
        state = RPCProviderState(url="https://rpc.example.com")
        state.consecutive_failures = 2
        state.record_success(latency_ms=100.0)
        assert state.consecutive_failures == 0
        assert state.is_healthy is True

    def test_record_failure_increments_counter(self):
        state = RPCProviderState(url="https://rpc.example.com")
        state.record_failure()
        assert state.consecutive_failures == 1
        assert state.total_failures == 1

    def test_marks_unhealthy_after_threshold(self):
        state = RPCProviderState(url="https://rpc.example.com")
        for _ in range(RPC_FAILURE_THRESHOLD):
            state.record_failure()
        assert state.is_healthy is False

    def test_success_rate_perfect_when_no_failures(self):
        state = RPCProviderState(url="https://rpc.example.com")
        state.record_success(latency_ms=50.0)
        state.record_success(latency_ms=60.0)
        assert state.success_rate == 1.0

    def test_success_rate_zero_when_all_fail(self):
        state = RPCProviderState(url="https://rpc.example.com")
        for _ in range(3):
            state.record_failure()
        assert state.success_rate == 0.0

    def test_success_rate_one_when_no_requests(self):
        """Undefined success rate defaults to 1.0."""
        state = RPCProviderState(url="https://rpc.example.com")
        assert state.success_rate == 1.0


# ---------------------------------------------------------------------------
# FailoverRPCCluster
# ---------------------------------------------------------------------------


class TestFailoverRPCCluster:
    """Tests for the multi-provider cluster logic."""

    def test_get_best_rpc_returns_none_when_empty(self):
        cluster = FailoverRPCCluster()
        assert cluster.get_best_rpc() is None

    def test_get_best_rpc_prefers_private(self):
        """When private providers are available, prefer_private=True picks one."""
        cluster = FailoverRPCCluster(
            providers=[
                ("https://public.rpc", False),
                ("https://private.rpc", True),
            ]
        )
        rpc = cluster.get_best_rpc(prefer_private=True)
        assert rpc == "https://private.rpc"

    def test_get_best_rpc_falls_back_to_public(self):
        """When no private providers exist, returns the public one."""
        cluster = FailoverRPCCluster(
            providers=[("https://public.rpc", False)]
        )
        rpc = cluster.get_best_rpc(prefer_private=True)
        assert rpc == "https://public.rpc"

    def test_no_duplicate_providers(self):
        """add_provider() with the same URL twice should not duplicate."""
        cluster = FailoverRPCCluster()
        cluster.add_provider("https://rpc.example.com", is_private=False)
        cluster.add_provider("https://rpc.example.com", is_private=False)
        assert len(cluster._providers) == 1

    def test_record_success_and_failure_update_state(self):
        cluster = FailoverRPCCluster(
            providers=[("https://rpc.example.com", False)]
        )
        cluster.record_success("https://rpc.example.com", latency_ms=200.0)
        cluster.record_failure("https://rpc.example.com")
        state = cluster._providers[0]
        assert state.total_requests == 2
        assert state.total_failures == 1

    def test_cluster_stats_structure(self):
        cluster = FailoverRPCCluster(
            providers=[("https://rpc.a.com", True), ("https://rpc.b.com", False)]
        )
        stats = cluster.get_cluster_stats()
        assert stats["total_providers"] == 2
        assert stats["healthy_providers"] == 2
        assert stats["private_providers"] == 1


# ---------------------------------------------------------------------------
# MEVShield
# ---------------------------------------------------------------------------


class TestMEVShield:
    def setup_method(self):
        reset_mev_shield()

    def test_mev_shield_init_has_components(self):
        """MEVShield must have jitter, rbf, and cluster attributes."""
        shield = MEVShield()
        assert isinstance(shield.jitter, GaussianJitter)
        assert isinstance(shield.rbf, DynamicRBF)
        assert isinstance(shield.cluster, FailoverRPCCluster)

    def test_get_protected_rpc_fallback_when_empty_cluster(self):
        """With an empty cluster, get_protected_rpc() returns the Base public RPC."""
        shield = MEVShield()  # cluster is empty
        rpc = shield.get_protected_rpc()
        assert rpc == "https://mainnet.base.org"

    def test_get_stats_returns_dict_with_expected_keys(self):
        shield = MEVShield()
        stats = shield.get_stats()
        assert "jitter" in stats
        assert "cluster" in stats
        assert "avg_ms" in stats["jitter"]
        assert "total_seconds" in stats["jitter"]

    @pytest.mark.asyncio
    async def test_apply_jitter_returns_positive_float(self):
        shield = MEVShield()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            delay = await shield.apply_jitter()
        assert isinstance(delay, float)
        assert delay >= JITTER_MIN_MS
