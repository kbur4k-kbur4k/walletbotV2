"""tests/unit/test_signal_discovery.py — SignalDiscoveryPipeline tests."""
from __future__ import annotations

import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest

from wallet_manager.signal_discovery import (
    CandidateSignal,
    SignalDiscoveryPipeline,
    _pearson,
    get_signal_discovery,
)


class TestPearson:
    def test_perfectly_correlated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_pearson(x, y) - 1.0) < 1e-10

    def test_perfectly_anticorrelated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(x, y) + 1.0) < 1e-10

    def test_insufficient_data_returns_zero(self):
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0]
        assert _pearson(x, y) == 0.0  # n < 5

    def test_zero_variance_returns_zero(self):
        x = [5.0, 5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _pearson(x, y) == 0.0

    def test_different_lengths_uses_min(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _pearson(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_uncorrelated(self):
        # Random-like, not perfectly correlated
        x = [1.0, -1.0, 1.0, -1.0, 1.0]
        y = [1.0, 1.0, -1.0, -1.0, 1.0]
        result = _pearson(x, y)
        assert -1.0 <= result <= 1.0


class TestCandidateSignal:
    def test_to_dict_structure(self):
        sig = CandidateSignal(
            name="test_sig",
            compute=lambda p: 0.5,
            description="Test signal",
        )
        d = sig.to_dict()
        assert d["name"] == "test_sig"
        assert d["description"] == "Test signal"
        assert d["ic"] == 0.0
        assert d["status"] == "UNTESTED"
        assert d["observations"] == 0


class TestSignalComputations:
    def test_rsi_insufficient_data(self):
        # Less than period+1 points returns 50.0
        result = SignalDiscoveryPipeline._rsi([1.0, 2.0, 3.0], 14)
        assert result == 50.0

    def test_rsi_all_gains(self):
        # All increasing prices — avg_loss near 0, should return +1.0
        prices = [100.0 + i for i in range(20)]
        result = SignalDiscoveryPipeline._rsi(prices, 14)
        assert result == 1.0

    def test_rsi_normal(self):
        import random
        rng = random.Random(42)
        prices = [100.0]
        for _ in range(20):
            prices.append(prices[-1] * (1 + rng.gauss(0, 0.01)))
        result = SignalDiscoveryPipeline._rsi(prices, 14)
        assert -1.0 <= result <= 1.0

    def test_bollinger_insufficient_data(self):
        result = SignalDiscoveryPipeline._bollinger_position([1.0, 2.0], 20)
        assert result == 0.0

    def test_bollinger_zero_std(self):
        # All same prices — std=0 returns 0
        prices = [100.0] * 25
        result = SignalDiscoveryPipeline._bollinger_position(prices, 20)
        assert result == 0.0

    def test_bollinger_normal(self):
        import random
        rng = random.Random(42)
        prices = [100.0 + rng.gauss(0, 1) for _ in range(25)]
        result = SignalDiscoveryPipeline._bollinger_position(prices, 20)
        assert -1.0 <= result <= 1.0

    def test_atr_ratio_insufficient(self):
        result = SignalDiscoveryPipeline._atr_ratio([100.0, 101.0], 14)
        assert result == 1.0

    def test_atr_ratio_zero_long(self):
        # All same prices — long ATR=0
        prices = [100.0] * 50
        result = SignalDiscoveryPipeline._atr_ratio(prices, 14)
        assert result == 1.0

    def test_atr_ratio_normal(self):
        import random
        rng = random.Random(42)
        prices = [100.0 + rng.gauss(0, 2) for _ in range(60)]
        result = SignalDiscoveryPipeline._atr_ratio(prices, 14)
        assert -1.0 <= result <= 1.0

    def test_vwap_deviation_insufficient(self):
        result = SignalDiscoveryPipeline._vwap_deviation([100.0] * 5, 20)
        assert result == 0.0

    def test_vwap_deviation_zero_vwap(self):
        prices = [0.0] * 25
        result = SignalDiscoveryPipeline._vwap_deviation(prices, 20)
        assert result == 0.0

    def test_vwap_deviation_normal(self):
        prices = [100.0] * 20 + [105.0]  # 5% above VWAP
        result = SignalDiscoveryPipeline._vwap_deviation(prices, 20)
        assert -1.0 <= result <= 1.0

    def test_momentum_insufficient(self):
        result = SignalDiscoveryPipeline._momentum([100.0, 101.0], 3)
        assert result == 0.0

    def test_momentum_zero_ref(self):
        # All zeros: ref price is 0.0 → returns 0.0
        prices = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = SignalDiscoveryPipeline._momentum(prices, 3)
        assert result == 0.0

    def test_momentum_normal(self):
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        result = SignalDiscoveryPipeline._momentum(prices, 3)
        assert -1.0 <= result <= 1.0


class TestSignalDiscoveryPipeline:
    def test_build_candidates_creates_seven(self):
        p = SignalDiscoveryPipeline()
        assert len(p._candidates) == 7
        names = [c.name for c in p._candidates]
        assert "rsi_14" in names
        assert "rsi_2" in names
        assert "bollinger_position" in names

    def test_record_cycle_updates_history(self):
        p = SignalDiscoveryPipeline()
        p.record_cycle(100.0, 0.5, 0.8)
        assert len(p._price_history) == 1
        assert len(p._existing_signals["robust_alpha"]) == 1
        assert len(p._existing_signals["confidence"]) == 1

    def test_record_cycle_computes_forward_returns(self):
        p = SignalDiscoveryPipeline()
        p.record_cycle(100.0, 0.5, 0.8)
        p.record_cycle(102.0, 0.6, 0.9)
        assert len(p._forward_returns) == 1
        assert abs(p._forward_returns[0] - 0.02) < 1e-10

    def test_record_cycle_zero_price_no_div_error(self):
        p = SignalDiscoveryPipeline()
        p.record_cycle(0.0, 0.5, 0.8)  # zero first price → no forward return computed
        p.record_cycle(100.0, 0.5, 0.8)
        # With first price=0.0 (<=0), _last_price stays 0, so return computed on 2nd call
        # prices[-2]=0.0 → condition `prices[-2] > 0` is False → ret=0.0
        assert len(p._forward_returns) == 1
        assert p._forward_returns[0] == 0.0

    @pytest.mark.asyncio
    async def test_run_discovery_insufficient_data_returns_empty(self):
        p = SignalDiscoveryPipeline()
        for i in range(10):  # less than _MIN_OBSERVATIONS=30
            p.record_cycle(100.0 + i, 0.5, 0.8)
        result = await p.run_discovery()
        assert result == []

    @pytest.mark.asyncio
    async def test_run_discovery_enough_data_returns_candidates(self):
        import random
        rng = random.Random(42)
        p = SignalDiscoveryPipeline()
        price = 100.0
        for _ in range(50):
            price *= (1 + rng.gauss(0, 0.01))
            p.record_cycle(price, rng.uniform(-1, 1), rng.uniform(0, 1))
        results = await p.run_discovery()
        assert len(results) == 7
        for c in results:
            assert c.status in ("PROMISING", "WEAK", "REDUNDANT", "MARGINAL", "UNTESTED")

    @pytest.mark.asyncio
    async def test_run_discovery_sets_last_run(self):
        import random
        import time
        rng = random.Random(0)
        p = SignalDiscoveryPipeline()
        price = 100.0
        for _ in range(50):
            price *= (1 + rng.gauss(0, 0.02))
            p.record_cycle(price, rng.uniform(-1, 1), rng.uniform(0, 1))
        before = time.time()
        await p.run_discovery()
        assert p._last_run >= before

    @pytest.mark.asyncio
    async def test_run_discovery_registers_promising_candidates(self):
        import random
        rng = random.Random(99)
        p = SignalDiscoveryPipeline()
        # Generate data that has a clearly WEAK signal for most candidates
        # We just need to exercise the registration path
        price = 100.0
        for _ in range(50):
            price *= (1 + rng.gauss(0, 0.01))
            p.record_cycle(price, rng.uniform(-1, 1), rng.uniform(0, 1))

        with patch("wallet_manager.signal_discovery.get_signal_discovery") as _:
            # Mock the ExperimentLab to avoid real lab registration
            mock_lab = MagicMock()
            mock_lab._feature_toggles = {}
            with patch("wallet_manager.experiment_lab.get_experiment_lab", return_value=mock_lab):
                results = await p.run_discovery()
        assert len(results) == 7

    def test_to_dict_structure(self):
        p = SignalDiscoveryPipeline()
        d = p.to_dict()
        assert "last_run" in d
        assert "n_prices" in d
        assert "candidates" in d
        assert "promising_count" in d

    def test_to_dict_uses_results_after_run(self):
        p = SignalDiscoveryPipeline()
        p._results = [CandidateSignal(name="x", compute=lambda _: 0, description="y", status="WEAK")]
        d = p.to_dict()
        assert len(d["candidates"]) == 1

    @pytest.mark.asyncio
    async def test_register_promising_handles_lab_error(self):
        """_register_promising should not crash on ExperimentLab exception."""
        p = SignalDiscoveryPipeline()
        c = CandidateSignal(name="test", compute=lambda _: 0.5, description="d", status="PROMISING", ic=0.05)
        with patch("wallet_manager.experiment_lab.get_experiment_lab", side_effect=Exception("lab down")):
            p._register_promising([c])  # Should not raise

    def test_get_signal_discovery_singleton(self):
        import wallet_manager.signal_discovery as sd_mod
        sd_mod._pipeline = None  # reset
        p1 = get_signal_discovery()
        p2 = get_signal_discovery()
        assert p1 is p2
        sd_mod._pipeline = None  # cleanup
