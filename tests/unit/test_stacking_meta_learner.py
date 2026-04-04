"""Tests for StackingMetaLearner — regime-conditional meta-model."""
import pytest

from wallet_manager.strategies.alpha_engine import StackingMetaLearner


@pytest.fixture
def stacker() -> StackingMetaLearner:
    return StackingMetaLearner()


def test_instantiation(stacker):
    assert stacker is not None


def test_predict_returns_two_floats(stacker):
    """predict() returns a (float, float) tuple."""
    result = stacker.predict(
        xgb_pred=0.6, lgbm_pred=0.55,
        lstm_pred=0.58, transformer_pred=0.52,
        regime="LOW_VOL",
    )
    assert isinstance(result, tuple) and len(result) == 2
    pred, conf = result
    assert isinstance(pred, float) and isinstance(conf, float)


def test_predict_output_in_unit_interval(stacker):
    """Both prediction and confidence are in [0, 1]."""
    pred, conf = stacker.predict(
        xgb_pred=0.7, lgbm_pred=0.65,
        lstm_pred=0.6, transformer_pred=0.55,
        regime="LOW_VOL",
    )
    assert 0.0 <= pred <= 1.0, f"pred out of range: {pred}"
    assert 0.0 <= conf <= 1.0, f"conf out of range: {conf}"


def test_unknown_regime_uses_fallback(stacker):
    """Unknown regime falls back to a weighted average without raising."""
    pred, conf = stacker.predict(
        xgb_pred=0.6, lgbm_pred=0.6,
        lstm_pred=0.6, transformer_pred=0.6,
        regime="COMPLETELY_UNKNOWN_REGIME",
    )
    assert isinstance(pred, float) and isinstance(conf, float)


def test_all_half_predictions_produce_near_half(stacker):
    """All-0.5 base predictions → meta output near 0.5 (within ±0.15)."""
    pred, _ = stacker.predict(
        xgb_pred=0.5, lgbm_pred=0.5,
        lstm_pred=0.5, transformer_pred=0.5,
        regime="LOW_VOL",
    )
    assert abs(pred - 0.5) < 0.15, f"Expected near 0.5, got {pred}"


def test_high_confidence_when_models_agree(stacker):
    """High model agreement (all preds close to 1.0) → higher confidence
    than when models disagree (spread predictions)."""
    _, conf_agree = stacker.predict(
        xgb_pred=0.9, lgbm_pred=0.88,
        lstm_pred=0.91, transformer_pred=0.89,
        regime="LOW_VOL",
    )
    _, conf_disagree = stacker.predict(
        xgb_pred=0.9, lgbm_pred=0.1,
        lstm_pred=0.8, transformer_pred=0.2,
        regime="LOW_VOL",
    )
    assert conf_agree >= conf_disagree
