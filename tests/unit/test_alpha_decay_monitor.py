"""tests/unit/test_alpha_decay_monitor.py — Alpha decay monitor tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wallet_manager.strategies.alpha_decay_monitor import (
    CrowdingDetector,
    SignalHalfLifeEstimator,
)


def _ar1_series(n: int = 60, rho: float = 0.8) -> list[float]:
    """Generate an AR(1) series with known autocorrelation rho."""
    rng = np.random.default_rng(42)
    x = [0.0]
    for _ in range(n - 1):
        x.append(rho * x[-1] + rng.normal(0.0, 0.1))
    return x


def test_signal_half_life_estimator_positive_half_life():
    """AR(1) with rho=0.8 should produce a positive half-life."""
    estimator = SignalHalfLifeEstimator()
    series = _ar1_series(n=60, rho=0.8)
    for v in series:
        estimator.update("zscore", v)
    result = estimator.estimate("zscore")
    assert result is not None
    assert result.half_life_bars > 0


def test_signal_half_life_estimator_returns_none_insufficient_data():
    """With fewer than 20 observations, estimate() returns None."""
    estimator = SignalHalfLifeEstimator()
    for v in [1.0, 2.0, 3.0]:
        estimator.update("zscore", v)
    assert estimator.estimate("zscore") is None


def test_crowding_detector_instantiate():
    """CrowdingDetector should instantiate without error."""
    detector = CrowdingDetector()
    assert detector is not None


def test_crowding_detector_score_in_range():
    """get_crowding_signal().composite should be in [0, 1]."""
    detector = CrowdingDetector()
    for i in range(20):
        gas = 5.0 + i * 0.5
        detector.update_gas(gas)
        detector.update_signal(float(i % 2) * 2 - 1, gas)
        detector.update_is(float(i) * 0.01)
    signal = detector.get_crowding_signal()
    assert 0.0 <= signal.composite <= 1.0


def test_crowding_detector_update_does_not_raise():
    """All update methods should run without raising."""
    detector = CrowdingDetector()
    detector.update_gas(10.0)
    detector.update_signal(0.5, 10.0)
    detector.update_is(2.5)
    signal = detector.get_crowding_signal()
    assert 0.0 <= signal.composite <= 1.0
