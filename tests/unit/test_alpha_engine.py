"""tests/unit/test_alpha_engine.py — Alpha engine component tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wallet_manager.strategies.alpha_engine import (
    ENSEMBLE_WEIGHT_LGBM,
    ENSEMBLE_WEIGHT_LSTM,
    ENSEMBLE_WEIGHT_TRANSFORMER,
    ENSEMBLE_WEIGHT_XGB,
    AdaptiveGARCH,
    AlphaConfidenceEstimator,
    EnsembleAlphaEngine,
    KalmanPriceSmoother,
)


def test_adaptive_garch_asymmetry():
    """γ > 0 causes asymmetric response to negative shocks."""
    g = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85, gamma=0.05)
    for _ in range(30):
        g.update(0.001)

    g_neg = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85, gamma=0.05)
    g_pos = AdaptiveGARCH(omega=1e-6, alpha=0.06, beta=0.85, gamma=0.05)
    for _ in range(30):
        g_neg.update(0.001)
        g_pos.update(0.001)

    # Apply shock — indicator uses _last_return (0.001 > 0), so asymmetry
    # kicks in on the NEXT update after the shock sets _last_return
    g_neg.update(-0.05)
    g_pos.update(+0.05)

    # Neutral step: now indicator=1 for neg instance, 0 for pos instance
    sigma_neg = g_neg.update(0.001)
    sigma_pos = g_pos.update(0.001)
    assert sigma_neg > sigma_pos


def test_adaptive_garch_stationarity():
    """alpha + beta + gamma/2 < 1 must hold."""
    g = AdaptiveGARCH(alpha=0.06, beta=0.85, gamma=0.05)
    assert g._alpha + g._beta + g._gamma / 2.0 < 1.0


def test_kalman_velocity_converges():
    """Velocity converges on a known linear price series."""
    k = KalmanPriceSmoother(process_noise=1e-4, measurement_noise=0.01)
    slope = 2.0
    p = 2000.0
    for _ in range(300):
        k.update(p)
        p += slope
    _, vel, _ = k.update(p)
    assert abs(vel - slope) < 1.0


def test_ensemble_output_probability_in_range():
    """EnsembleAlphaEngine output probability must be in [0, 1]."""
    from wallet_manager.strategies.alpha_engine import (
        LightGBMPredictor,
        LSTMPredictor,
        TransformerPredictor,
        XGBoostPredictor,
    )
    xgb = XGBoostPredictor()
    lgbm = LightGBMPredictor()
    lstm = LSTMPredictor()
    transformer = TransformerPredictor()

    engine = EnsembleAlphaEngine(xgb=xgb, lgbm=lgbm, lstm=lstm, transformer=transformer)
    features = np.random.rand(30).astype(np.float32)
    result = engine.predict(features)
    assert 0.0 <= result["ensemble"] <= 1.0


def test_ensemble_uses_4_models():
    """Ensemble must reference exactly 4 model keys with positive weights."""
    total = ENSEMBLE_WEIGHT_XGB + ENSEMBLE_WEIGHT_LGBM + ENSEMBLE_WEIGHT_LSTM + ENSEMBLE_WEIGHT_TRANSFORMER
    assert ENSEMBLE_WEIGHT_LGBM > 0, "LightGBM weight must be > 0"
    assert ENSEMBLE_WEIGHT_XGB > 0
    assert ENSEMBLE_WEIGHT_LSTM > 0
    # Transformer may be disabled (weight=0) if marked not accepted in training_report.json
    assert ENSEMBLE_WEIGHT_TRANSFORMER >= 0
    assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


def test_confidence_estimator_in_range():
    """AlphaConfidenceEstimator.estimate() returns a value in [0, 1]."""
    est = AlphaConfidenceEstimator()
    preds = {"xgb": 0.6, "lgbm": 0.55, "lstm": 0.65, "transformer": 0.58, "ensemble": 0.60}
    confidence = est.estimate(
        model_predictions=preds,
        data_coverage=0.9,
        vol_regime="normal",
        horizon_score=0.5,
    )
    assert 0.0 <= confidence <= 1.0
    assert math.isfinite(confidence)


def test_confidence_estimator_finite():
    """scale_factor must be finite."""
    est = AlphaConfidenceEstimator()
    for data_cov in [0.0, 0.5, 1.0]:
        preds = {"xgb": 0.5, "lgbm": 0.5, "lstm": 0.5, "transformer": 0.5, "ensemble": 0.5}
        conf = est.estimate(preds, data_cov, "normal", 0.0)
        assert math.isfinite(conf)
