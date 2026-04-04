"""tests/unit/test_feature_expansion.py — Feature engineering expansion tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wallet_manager.strategies.alpha_engine import (
    FeatureStore,
    FeatureVector,
    KalmanPriceSmoother,
    OrderFlowImbalanceV2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_vector(price: float = 3000.0) -> FeatureVector:
    """Return a minimal FeatureVector with sensible defaults."""
    return FeatureVector(
        timestamp=1_700_000_000.0,
        price=price,
        log_return=0.001,
        kalman_price=price,
        kalman_trend=0.5,
        garch_sigma=0.02,
        realized_vol_5=0.015,
        realized_vol_30=0.018,
        vol_ratio=0.83,
        momentum_5=0.002,
        momentum_15=0.004,
        momentum_30=0.003,
        momentum_60=0.001,
        momentum_120=0.0,
        volume_imbalance=0.1,
        spread_bps=5.0,
        trade_intensity=2.0,
        skewness=-0.3,
        kurtosis=1.2,
    )


def _make_ofi_v2() -> OrderFlowImbalanceV2:
    """Return a warmed-up OrderFlowImbalanceV2."""
    ofi = OrderFlowImbalanceV2()
    for i in range(60):
        ofi.update(0.001 * ((-1) ** i), 1_000.0, 0.55)
    return ofi


def _make_smoother(price: float = 3000.0) -> KalmanPriceSmoother:
    """Return a warmed-up KalmanPriceSmoother."""
    k = KalmanPriceSmoother()
    for i in range(50):
        k.update(price + i * 0.1)
    return k


def _make_price_history(n: int = 50, base: float = 3000.0) -> np.ndarray:
    """Monotonically increasing price series."""
    return np.array([base + i * 0.5 for i in range(n)], dtype=np.float64)


def _make_volume_history(n: int = 30, base: float = 1_000.0) -> np.ndarray:
    return np.full(n, base, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_vector_shape():
    """FeatureStore.build() must return a vector of length 36."""
    store = FeatureStore()
    fv = _make_feature_vector()
    ofi = _make_ofi_v2()
    smoother = _make_smoother()
    prices = _make_price_history(50)
    vols = _make_volume_history(30)

    result = store.build(fv, ofi, smoother, 0.3, price_history=prices, volume_history=vols)

    assert result.shape == (36,), f"Expected shape (36,), got {result.shape}"
    assert len(result) == FeatureStore.N_FEATURES == 36


def test_no_nan_in_features():
    """No NaN values must appear in the output feature vector."""
    store = FeatureStore()
    fv = _make_feature_vector()
    ofi = _make_ofi_v2()
    smoother = _make_smoother()
    prices = _make_price_history(50)
    vols = _make_volume_history(30)

    result = store.build(fv, ofi, smoother, 0.3, price_history=prices, volume_history=vols)

    assert not np.any(np.isnan(result)), f"NaN found at indices: {np.where(np.isnan(result))}"
    assert not np.any(np.isinf(result)), f"Inf found at indices: {np.where(np.isinf(result))}"


def test_no_nan_in_features_without_history():
    """Feature vector must be NaN-free even when no price/volume history is supplied."""
    store = FeatureStore()
    fv = _make_feature_vector()
    ofi = _make_ofi_v2()
    smoother = _make_smoother()

    result = store.build(fv, ofi, smoother, 0.3, price_history=None, volume_history=None)

    assert result.shape == (36,)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_rsi_normalized_range():
    """rsi_14 feature must be in [0, 1]."""
    store = FeatureStore()
    fv = _make_feature_vector()
    ofi = _make_ofi_v2()
    smoother = _make_smoother()
    prices = _make_price_history(50)
    vols = _make_volume_history(30)

    result = store.build(fv, ofi, smoother, 0.3, price_history=prices, volume_history=vols)

    rsi_idx = FeatureStore.FEATURE_NAMES.index("rsi_14")
    rsi_val = result[rsi_idx]
    assert 0.0 <= rsi_val <= 1.0, f"rsi_14={rsi_val} is out of [0, 1]"


def test_bb_position_range():
    """bb_position feature must be in [0, 1]."""
    store = FeatureStore()
    fv = _make_feature_vector()
    ofi = _make_ofi_v2()
    smoother = _make_smoother()
    prices = _make_price_history(50)
    vols = _make_volume_history(30)

    result = store.build(fv, ofi, smoother, 0.3, price_history=prices, volume_history=vols)

    bb_idx = FeatureStore.FEATURE_NAMES.index("bb_position")
    bb_val = result[bb_idx]
    assert 0.0 <= bb_val <= 1.0, f"bb_position={bb_val} is out of [0, 1]"


def test_feature_names_length():
    """FEATURE_NAMES list must have exactly 36 entries."""
    assert len(FeatureStore.FEATURE_NAMES) == 36
    assert FeatureStore.N_FEATURES == 36


def test_new_feature_names_present():
    """All 6 new feature names must be present in FEATURE_NAMES."""
    expected = {"rsi_14", "bb_position", "volume_trend",
                "price_acceleration", "spread_ratio", "microstructure_score"}
    assert expected.issubset(set(FeatureStore.FEATURE_NAMES))


def test_spread_ratio_accumulates():
    """spread_ratio should stabilise after enough spread observations."""
    store = FeatureStore()
    fv = _make_feature_vector()
    ofi = _make_ofi_v2()
    smoother = _make_smoother()

    # Feed 60 ticks so spread history is full
    for i in range(60):
        store.build(fv, ofi, smoother, 0.3)

    result = store.build(fv, ofi, smoother, 0.3)
    sr_idx = FeatureStore.FEATURE_NAMES.index("spread_ratio")
    sr_val = result[sr_idx]
    assert 0.1 <= sr_val <= 10.0, f"spread_ratio={sr_val} out of clipped range"
