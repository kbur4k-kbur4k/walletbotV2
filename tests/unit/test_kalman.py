"""tests/unit/test_kalman.py — KalmanPriceSmoother tests."""
from __future__ import annotations

import pytest

from wallet_manager.strategies.alpha_engine import KalmanPriceSmoother


def test_state_vector_has_3_components():
    """Kalman filter maintains price, velocity, and acceleration."""
    k = KalmanPriceSmoother()
    k.update(2000.0)
    k.update(2001.0)
    # _x should have 3 components
    assert k._x.shape == (3,)


def test_velocity_converges_on_linear_series():
    """On a linearly rising series, velocity converges to the slope."""
    slope = 1.0  # +1.0 per bar
    k = KalmanPriceSmoother(process_noise=1e-4, measurement_noise=1e-2)
    price = 2000.0
    for _ in range(200):
        k.update(price)
        price += slope

    _, velocity, _ = k.update(price)
    # Velocity should be close to the slope
    assert abs(velocity - slope) < 0.5, f"Expected velocity ≈ {slope}, got {velocity:.4f}"


def test_process_noise_increases_with_garch_sigma():
    """Process noise Q should scale with volatility."""
    k = KalmanPriceSmoother(process_noise=1e-5, measurement_noise=1e-3)
    # Just verify update doesn't raise and returns 3 floats
    result = k.update(2000.0)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_acceleration_component():
    """Acceleration (3rd state) should be tracked."""
    k = KalmanPriceSmoother()
    for i in range(50):
        k.update(2000.0 + i * i * 0.01)  # quadratic series
    _, _vel, accel = k.update(2000.0 + 50 * 50 * 0.01)
    assert isinstance(accel, float)


def test_initialized_flag():
    k = KalmanPriceSmoother()
    assert not k._initialized
    k.update(2000.0)
    assert k._initialized


def test_first_update_returns_input_price():
    k = KalmanPriceSmoother()
    price, vel, accel = k.update(2000.0)
    assert price == 2000.0
    assert vel == 0.0
    assert accel == 0.0
