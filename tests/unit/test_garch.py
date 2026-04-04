"""tests/unit/test_garch.py - AdaptiveGARCH (GJR-GARCH) tests."""
from __future__ import annotations

import math

import pytest

from wallet_manager.strategies.alpha_engine import AdaptiveGARCH


def test_asymmetry_negative_shock_increases_sigma_more():
    """GJR: after a negative shock, sigma > sigma after equal positive shock."""
    garch_neg = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85, gamma=0.05)
    garch_pos = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85, gamma=0.05)

    for _ in range(30):
        garch_neg.update(0.001)
        garch_pos.update(0.001)

    magnitude = 0.05
    garch_neg.update(-magnitude)
    garch_pos.update(+magnitude)

    sigma_after_neg = garch_neg.update(0.001)
    sigma_after_pos = garch_pos.update(0.001)

    assert sigma_after_neg > sigma_after_pos, (
        f"Expected sigma(neg) > sigma(pos); got sigma(neg)={sigma_after_neg:.6f}, "
        f"sigma(pos)={sigma_after_pos:.6f}"
    )


def test_stationarity_alpha_beta_gamma():
    """alpha + beta + gamma/2 must be < 1 for stationarity."""
    g = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85, gamma=0.05)
    assert g._alpha + g._beta + g._gamma / 2.0 < 1.0


def test_sigma_is_positive():
    g = AdaptiveGARCH()
    for ret in [0.01, -0.02, 0.005, -0.03, 0.015]:
        sigma = g.update(ret)
        assert sigma > 0


def test_long_run_sigma_formula():
    """sqrt(omega / (1 - alpha - beta - gamma/2)) matches theoretical formula."""
    omega, alpha, beta, gamma = 1e-6, 0.06, 0.85, 0.05
    AdaptiveGARCH(omega=omega, alpha=alpha, beta=beta, gamma=gamma)
    denom = 1.0 - alpha - beta - gamma / 2.0
    expected_long_run = math.sqrt(omega / denom)
    assert expected_long_run > 0
    assert abs(expected_long_run - 0.003162) < 0.001


def test_first_update_returns_initial_sigma():
    g = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85)
    sigma = g.update(0.01)
    assert sigma > 0
    assert math.isfinite(sigma)


def test_ewma_fallback_fewer_than_30_observations():
    """With < 30 observations, has_enough_data should be False."""
    from wallet_manager.strategies.alpha_engine import GARCH_MIN_OBSERVATIONS

    g = AdaptiveGARCH()
    for i in range(GARCH_MIN_OBSERVATIONS - 1):
        g.update(0.001 * (i % 5 - 2))
    assert not g.has_enough_data
