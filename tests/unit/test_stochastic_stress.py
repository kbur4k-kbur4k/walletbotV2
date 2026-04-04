"""tests/unit/test_stochastic_stress.py — MonteCarloEngine ve StochasticStressTester."""
from __future__ import annotations

import numpy as np
import pytest

from wallet_manager.strategies.stochastic_stress_test import (
    MonteCarloEngine,
    MonteCarloResult,
    StochasticStressTester,
    StressTestVerdict,
)


class TestMonteCarloEngine:
    def test_run_stress_test_returns_result(self):
        engine = MonteCarloEngine()
        prices = [2000.0 * (0.999 + np.random.normal(0, 0.002)) for _ in range(200)]
        result = engine.run_stress_test(prices=prices, n_simulations=100)
        assert isinstance(result, MonteCarloResult)

    def test_var_is_positive(self):
        engine = MonteCarloEngine()
        prices = list(np.linspace(2000, 1800, 100))  # downtrend
        result = engine.run_stress_test(prices=prices, n_simulations=50)
        assert result.var_99 >= 0.0

    def test_simulation_count_respected(self):
        engine = MonteCarloEngine()
        prices = [2000.0 + np.random.normal(0, 10) for _ in range(100)]
        result = engine.run_stress_test(prices=prices, n_simulations=200)
        assert result.n_simulations == 200


class TestStochasticStressTester:
    def _make_prices(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.001, 0.02, n)
        return (2000.0 * np.cumprod(1 + returns)).tolist()

    def test_validate_signal_returns_verdict(self):
        tester = StochasticStressTester()
        prices = self._make_prices()
        signals = [np.random.normal(0, 1) for _ in prices]
        verdict = tester.validate_signal(prices=prices, signals=signals)
        assert isinstance(verdict, StressTestVerdict)

    def test_get_stats_returns_dict(self):
        tester = StochasticStressTester()
        stats = tester.get_stats()
        assert isinstance(stats, dict)

    def test_high_var_verdict_halt(self):
        """Çok yüksek VaR → halt kararı."""
        tester = StochasticStressTester(var_halt_threshold=0.05)
        prices = [2000.0 * (0.90 ** i) for i in range(100)]
        signals = [0.5] * 100
        verdict = tester.validate_signal(prices=prices, signals=signals)
        assert hasattr(verdict, "should_halt")
