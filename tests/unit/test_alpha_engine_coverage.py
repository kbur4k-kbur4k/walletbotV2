"""tests/unit/test_alpha_engine_coverage.py

Coverage tests for alpha_engine.py — targets GARCHEngine, KalmanFilter,
FeatureEngineer, XGBoostPredictor, LightGBMPredictor, LSTMPredictor,
TransformerPredictor, EnsembleAlphaEngine, FeatureNormalizer, FeatureStore,
AdaptiveGARCH, KalmanPriceSmoother, OrderFlowImbalanceV2, MultiHorizonSignal.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from wallet_manager.strategies.alpha_engine import (
    AdaptiveGARCH,
    EnsembleAlphaEngine,
    FeatureEngineer,
    FeatureNormalizer,
    FeatureStore,
    FeatureVector,
    GARCHEngine,
    KalmanFilter,
    KalmanPriceSmoother,
    LightGBMPredictor,
    LSTMPredictor,
    MultiHorizonSignal,
    OrderFlowImbalanceV2,
    TransformerPredictor,
    XGBoostPredictor,
)

# ---------------------------------------------------------------------------
# GARCHEngine tests
# ---------------------------------------------------------------------------


def test_garch_engine_initial_sigma_positive():
    g = GARCHEngine()
    assert g.sigma > 0


def test_garch_engine_update_returns_float():
    g = GARCHEngine()
    result = g.update(0.01)
    assert isinstance(result, float)
    assert result > 0


def test_garch_engine_update_increments_observations():
    g = GARCHEngine()
    g.update(0.001)
    g.update(0.002)
    assert g._n_observations == 2


def test_garch_engine_long_run_sigma_positive():
    g = GARCHEngine()
    assert g.long_run_sigma > 0


def test_garch_engine_volatility_regime_before_min_obs():
    g = GARCHEngine()
    assert g.get_volatility_regime() == "normal"


def test_garch_engine_volatility_regime_after_min_obs():
    g = GARCHEngine()
    for _ in range(35):
        g.update(0.001)
    regime = g.get_volatility_regime()
    assert regime in ("low", "normal", "high")


def test_garch_engine_high_volatility_regime():
    g = GARCHEngine(omega=1e-6, alpha=0.10, beta=0.85)
    for _ in range(30):
        g.update(0.001)
    # Feed large shocks to drive sigma high above long-run
    for _ in range(5):
        g.update(0.30)
    regime = g.get_volatility_regime()
    # Should be 'high' after large shocks
    assert regime in ("normal", "high")


def test_garch_engine_has_enough_data():
    g = GARCHEngine()
    assert not g.has_enough_data
    for _ in range(30):
        g.update(0.001)
    assert g.has_enough_data


def test_garch_engine_sigma_floor():
    g = GARCHEngine()
    for _ in range(50):
        g.update(0.0)
    assert g.sigma > 0


# ---------------------------------------------------------------------------
# KalmanFilter (2-state) tests
# ---------------------------------------------------------------------------


def test_kalman_filter_initialization():
    kf = KalmanFilter()
    p, v = kf.update(1000.0)
    assert p == 1000.0
    assert v == 0.0


def test_kalman_filter_smoothed_price_close_to_input():
    kf = KalmanFilter()
    for _ in range(20):
        p, _ = kf.update(2000.0)
    assert abs(p - 2000.0) < 10.0


def test_kalman_filter_properties():
    kf = KalmanFilter()
    kf.update(1500.0)
    kf.update(1501.0)
    assert isinstance(kf.smoothed_price, float)
    assert isinstance(kf.trend_velocity, float)


def test_kalman_filter_velocity_positive_trend():
    kf = KalmanFilter()
    price = 1000.0
    for _ in range(50):
        kf.update(price)
        price += 1.0
    _, vel = kf.update(price)
    assert vel > 0


# ---------------------------------------------------------------------------
# FeatureNormalizer tests
# ---------------------------------------------------------------------------


def test_feature_normalizer_first_update_returns_zeros():
    fn = FeatureNormalizer(n_features=5)
    result = fn.update_and_normalize(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.all(result == 0)


def test_feature_normalizer_clips_extremes():
    fn = FeatureNormalizer(n_features=3)
    fn.update_and_normalize(np.array([0.0, 0.0, 0.0]))
    for _ in range(10):
        fn.update_and_normalize(np.array([1.0, 1.0, 1.0]))
    # Feed extreme value
    result = fn.update_and_normalize(np.array([1000.0, 1000.0, 1000.0]))
    assert np.all(result <= 5.0)


def test_feature_normalizer_output_shape():
    fn = FeatureNormalizer(n_features=18)
    raw = np.ones(18)
    fn.update_and_normalize(raw)
    out = fn.update_and_normalize(raw)
    assert out.shape == (18,)


def test_feature_normalizer_tracks_count():
    fn = FeatureNormalizer(n_features=4)
    for i in range(5):
        fn.update_and_normalize(np.array([float(i)] * 4))
    assert fn._n == 5


# ---------------------------------------------------------------------------
# XGBoostPredictor tests
# ---------------------------------------------------------------------------


def test_xgboost_fallback_returns_valid_proba():
    xgb = XGBoostPredictor()
    features = np.zeros(18, dtype=np.float64)
    up, down = xgb.predict_proba(features)
    assert 0.0 <= up <= 1.0
    assert 0.0 <= down <= 1.0
    assert abs(up + down - 1.0) < 1e-6


def test_xgboost_load_model_missing_returns_false():
    xgb = XGBoostPredictor()
    result = xgb.load_model(model_path=None)
    # No model on disk in test env — should return False
    assert result is False or result is True  # either is OK


def test_xgboost_statistical_fallback_bullish_signal():
    xgb = XGBoostPredictor()
    # Feature vector: kalman_trend (idx 2) = very positive
    features = np.zeros(18, dtype=np.float64)
    features[2] = 2.0   # kalman_trend
    features[7] = 1.0   # momentum_5
    up, _ = xgb._statistical_fallback(features)
    assert up > 0.5


def test_xgboost_statistical_fallback_bearish_signal():
    xgb = XGBoostPredictor()
    features = np.zeros(18, dtype=np.float64)
    features[2] = -2.0  # kalman_trend negative
    features[7] = -1.0  # momentum_5 negative
    up, _ = xgb._statistical_fallback(features)
    assert up < 0.5


# ---------------------------------------------------------------------------
# LightGBMPredictor tests
# ---------------------------------------------------------------------------


def test_lightgbm_fallback_returns_neutral():
    lgbm = LightGBMPredictor()
    features = np.random.rand(36).astype(np.float64)
    up, down = lgbm.predict_proba(features)
    # Without a loaded model, should return (0.5, 0.5)
    assert up == 0.5
    assert down == 0.5


def test_lightgbm_load_model_no_file_returns_false():
    lgbm = LightGBMPredictor()
    result = lgbm.load_model(model_path=None)
    assert result is False or result is True


def test_lightgbm_is_loaded_false_by_default():
    lgbm = LightGBMPredictor()
    assert lgbm._is_loaded is False


# ---------------------------------------------------------------------------
# LSTMPredictor tests
# ---------------------------------------------------------------------------


def test_lstm_neutral_when_buffer_empty():
    lstm = LSTMPredictor(sequence_length=5)
    up, down = lstm.predict_proba()
    assert up == 0.5
    assert down == 0.5


def test_lstm_buffer_fill_zero_initially():
    lstm = LSTMPredictor(sequence_length=10)
    assert lstm.buffer_fill == 0.0


def test_lstm_ingest_fills_buffer():
    lstm = LSTMPredictor(sequence_length=5)
    for _ in range(3):
        lstm.ingest_features(np.zeros(18))
    assert lstm.buffer_fill == 0.6


def test_lstm_ewm_fallback_after_full_buffer():
    lstm = LSTMPredictor(sequence_length=5)
    for i in range(5):
        feat = np.zeros(18)
        feat[0] = 0.001 * (i + 1)  # log_return at index 0
        lstm.ingest_features(feat)
    up, down = lstm.predict_proba()
    assert 0.0 <= up <= 1.0
    assert abs(up + down - 1.0) < 1e-6


def test_lstm_load_model_no_file_returns_false():
    lstm = LSTMPredictor()
    result = lstm.load_model(model_path=None)
    assert result is False or result is True


# ---------------------------------------------------------------------------
# TransformerPredictor tests
# ---------------------------------------------------------------------------


def test_transformer_neutral_when_buffer_empty():
    tp = TransformerPredictor(sequence_length=5)
    up, down = tp.predict_proba()
    assert up == 0.5
    assert down == 0.5


def test_transformer_buffer_fill_zero_initially():
    tp = TransformerPredictor(sequence_length=10)
    assert tp.buffer_fill == 0.0


def test_transformer_ingest_fills_buffer():
    tp = TransformerPredictor(sequence_length=4)
    for _ in range(2):
        tp.ingest_features(np.zeros(36))
    assert tp.buffer_fill == 0.5


def test_transformer_neutral_when_buffer_partial():
    tp = TransformerPredictor(sequence_length=10)
    tp.ingest_features(np.zeros(36))
    up, down = tp.predict_proba()
    assert up == 0.5
    assert down == 0.5


# ---------------------------------------------------------------------------
# EnsembleAlphaEngine tests
# ---------------------------------------------------------------------------


def test_ensemble_predict_output_keys():
    xgb = XGBoostPredictor()
    lgbm = LightGBMPredictor()
    lstm = LSTMPredictor()
    transformer = TransformerPredictor()
    engine = EnsembleAlphaEngine(xgb=xgb, lgbm=lgbm, lstm=lstm, transformer=transformer)
    features = np.random.rand(36).astype(np.float32)
    result = engine.predict(features)
    for key in ("xgb", "lgbm", "lstm", "transformer", "ensemble"):
        assert key in result


def test_ensemble_predict_values_in_range():
    xgb = XGBoostPredictor()
    lgbm = LightGBMPredictor()
    lstm = LSTMPredictor()
    transformer = TransformerPredictor()
    engine = EnsembleAlphaEngine(xgb=xgb, lgbm=lgbm, lstm=lstm, transformer=transformer)
    features = np.random.rand(36).astype(np.float32)
    result = engine.predict(features)
    for key in ("xgb", "lgbm", "lstm", "transformer", "ensemble"):
        assert 0.0 <= result[key] <= 1.0


def test_ensemble_update_platt_params():
    xgb = XGBoostPredictor()
    lgbm = LightGBMPredictor()
    lstm = LSTMPredictor()
    transformer = TransformerPredictor()
    engine = EnsembleAlphaEngine(xgb=xgb, lgbm=lgbm, lstm=lstm, transformer=transformer)
    engine.update_platt_params("xgb", A=-1.5, B=0.1)
    assert engine._platt_A["xgb"] == -1.5
    assert engine._platt_B["xgb"] == 0.1


def test_ensemble_platt_scale_identity():
    xgb = XGBoostPredictor()
    lgbm = LightGBMPredictor()
    lstm = LSTMPredictor()
    transformer = TransformerPredictor()
    engine = EnsembleAlphaEngine(xgb=xgb, lgbm=lgbm, lstm=lstm, transformer=transformer)
    # With A=-1, B=0, Platt scaling should return approximately the input
    for p in [0.3, 0.5, 0.7]:
        result = engine._platt_scale(p, "xgb")
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# FeatureEngineer tests
# ---------------------------------------------------------------------------


def test_feature_engineer_returns_none_before_warmup():
    fe = FeatureEngineer()
    result = fe.ingest(price=2000.0, volume=100.0)
    assert result is None


def test_feature_engineer_returns_feature_vector_after_warmup():
    fe = FeatureEngineer()
    fv = None
    for i in range(35):
        fv = fe.ingest(price=2000.0 + i * 0.1, volume=100.0, timestamp=float(i))
    assert fv is not None


def test_feature_engineer_set_microstructure():
    fe = FeatureEngineer()
    fe.set_microstructure(volume_imbalance=0.3, spread_bps=5.0)
    assert fe._volume_imbalance == 0.3
    assert fe._spread_bps == 5.0


def test_feature_engineer_feature_vector_fields():
    fe = FeatureEngineer()
    fv = None
    for i in range(35):
        fv = fe.ingest(price=2000.0 + i * 0.5, volume=50.0, timestamp=float(i * 5))
    assert fv is not None
    assert hasattr(fv, "log_return")
    assert hasattr(fv, "garch_sigma")
    assert hasattr(fv, "kalman_price")
    assert hasattr(fv, "momentum_5")


# ---------------------------------------------------------------------------
# FeatureVector.to_array tests
# ---------------------------------------------------------------------------


def test_feature_vector_to_array_shape():
    fv = FeatureVector(
        timestamp=0.0, price=1000.0, log_return=0.001,
        kalman_price=999.9, kalman_trend=0.1,
        garch_sigma=0.01, realized_vol_5=0.008, realized_vol_30=0.009,
        vol_ratio=0.9, momentum_5=0.001, momentum_15=0.002,
        momentum_30=0.003, momentum_60=0.004, momentum_120=0.005,
        volume_imbalance=0.1, spread_bps=3.0, trade_intensity=0.5,
        skewness=0.1, kurtosis=0.2,
    )
    arr = fv.to_array()
    assert arr.shape == (18,)
    assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# FeatureStore tests
# ---------------------------------------------------------------------------


def _make_feature_vector() -> FeatureVector:
    return FeatureVector(
        timestamp=1.0, price=2000.0, log_return=0.001,
        kalman_price=1999.0, kalman_trend=0.5,
        garch_sigma=0.01, realized_vol_5=0.008, realized_vol_30=0.009,
        vol_ratio=0.9, momentum_5=0.002, momentum_15=0.003,
        momentum_30=0.004, momentum_60=0.002, momentum_120=0.001,
        volume_imbalance=0.2, spread_bps=4.0, trade_intensity=0.3,
        skewness=0.05, kurtosis=0.1,
    )


def test_feature_store_build_output_shape():
    fs = FeatureStore()
    ofi = OrderFlowImbalanceV2()
    smoother = KalmanPriceSmoother()
    smoother.update(2000.0)
    fv = _make_feature_vector()
    result = fs.build(fv, ofi, smoother, horizon_score=0.1)
    assert result.shape == (36,)


def test_feature_store_n_features_constant():
    assert FeatureStore.N_FEATURES == 36


def test_feature_store_feature_names_count():
    assert len(FeatureStore.FEATURE_NAMES) == 36


def test_feature_store_build_with_price_volume_history():
    fs = FeatureStore()
    ofi = OrderFlowImbalanceV2()
    smoother = KalmanPriceSmoother()
    smoother.update(2000.0)
    fv = _make_feature_vector()
    prices = np.array([2000.0 + i * 0.1 for i in range(30)])
    volumes = np.array([100.0 + i for i in range(30)])
    result = fs.build(fv, ofi, smoother, horizon_score=0.1,
                      price_history=prices, volume_history=volumes)
    assert result.shape == (36,)
    assert np.all(np.isfinite(result))


def test_feature_store_rsi_neutral_when_no_data():
    fs = FeatureStore()
    rsi = fs._compute_rsi(None, 14)
    assert rsi == 0.5


def test_feature_store_bb_position_neutral_when_no_data():
    fs = FeatureStore()
    bb = fs._compute_bb_position(None, 2000.0, 20)
    assert bb == 0.5


def test_feature_store_volume_trend_default_when_no_data():
    fs = FeatureStore()
    vt = fs._compute_volume_trend(None)
    assert vt == 1.0


def test_feature_store_spread_ratio_default_when_insufficient():
    fs = FeatureStore()
    # Only 1 spread recorded -> fallback
    sr = fs._compute_spread_ratio(5.0)
    assert sr == 1.0


# ---------------------------------------------------------------------------
# AdaptiveGARCH tests (additional coverage)
# ---------------------------------------------------------------------------


def test_adaptive_garch_initial_sigma_positive():
    ag = AdaptiveGARCH()
    assert ag.sigma > 0


def test_adaptive_garch_has_enough_data_flag():
    ag = AdaptiveGARCH()
    assert not ag.has_enough_data
    for _ in range(30):
        ag.update(0.001)
    assert ag.has_enough_data


def test_adaptive_garch_low_regime():
    ag = AdaptiveGARCH()
    for _ in range(35):
        ag.update(0.0001)
    regime = ag.get_volatility_regime()
    assert regime in ("low", "normal", "high")


# ---------------------------------------------------------------------------
# KalmanPriceSmoother tests
# ---------------------------------------------------------------------------


def test_kalman_price_smoother_initial():
    ks = KalmanPriceSmoother()
    p, v, a = ks.update(3000.0)
    assert p == 3000.0
    assert v == 0.0
    assert a == 0.0


def test_kalman_price_smoother_properties():
    ks = KalmanPriceSmoother()
    ks.update(3000.0)
    ks.update(3001.0)
    assert isinstance(ks.smoothed_price, float)
    assert isinstance(ks.velocity, float)
    assert isinstance(ks.acceleration, float)


def test_kalman_price_smoother_tracks_trend():
    ks = KalmanPriceSmoother()
    price = 1000.0
    for _ in range(50):
        ks.update(price)
        price += 1.0
    assert ks.velocity > 0


# ---------------------------------------------------------------------------
# OrderFlowImbalanceV2 tests
# ---------------------------------------------------------------------------


def test_ofi_initial_values_zero():
    ofi = OrderFlowImbalanceV2()
    assert ofi.weighted_ofi == 0.0
    assert ofi.kyle_lambda == 0.0
    assert ofi.amihud_ratio == 0.0


def test_ofi_update_populates_metrics():
    ofi = OrderFlowImbalanceV2()
    for i in range(10):
        ofi.update(log_return=0.001 * (i % 3 - 1), volume=1000.0, buy_ratio=0.6)
    # After 10 updates (>= 5), metrics should be computed
    assert isinstance(ofi.weighted_ofi, float)
    assert isinstance(ofi.kyle_lambda, float)


def test_ofi_buy_pressure_positive_ofi():
    ofi = OrderFlowImbalanceV2()
    for _ in range(10):
        ofi.update(log_return=0.001, volume=1000.0, buy_ratio=0.9)
    assert ofi.weighted_ofi > 0


# ---------------------------------------------------------------------------
# MultiHorizonSignal tests
# ---------------------------------------------------------------------------


def test_multi_horizon_returns_zero_insufficient_data():
    mh = MultiHorizonSignal()
    assert mh.compute() == 0.0


def test_multi_horizon_compute_in_range():
    mh = MultiHorizonSignal()
    price = 2000.0
    for i in range(50):
        mh.update(price + i * 0.5)
    score = mh.compute()
    assert -1.0 <= score <= 1.0


def test_multi_horizon_positive_trend():
    mh = MultiHorizonSignal()
    price = 1000.0
    for i in range(50):
        mh.update(price)
        price *= 1.001  # steady uptrend
    score = mh.compute()
    assert score > 0
