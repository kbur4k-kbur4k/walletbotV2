"""
wallet_manager/strategies/stacking_meta_learner.py

Online logistic-regression meta-learner for ensemble blending.
Falls back to weighted average when fewer than 100 observations exist.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

@dataclass
class _Observation:
    features: list[float]   # [xgb_prob, lgbm_prob, lstm_prob, transformer_prob]
    label: int              # 1 if price went up, 0 otherwise


class StackingMetaLearner:
    """
    Online logistic regression meta-learner.
    
    Takes the four model probability outputs as features and learns
    to blend them optimally. Updates via SGD every 100 new observations
    using a sliding window of the last 500.
    """

    MIN_OBS_FOR_LEARNING = 100
    WINDOW_SIZE = 500
    UPDATE_EVERY = 100
    LEARNING_RATE = 0.01

    def __init__(self, fallback_weights: dict[str, float] | None = None) -> None:
        # FIX: load from training_report instead of hardcoded weights
        try:
            import json as _j
            from pathlib import Path as _P
            _rp = _P(__file__).resolve().parent.parent / "ml_models" / "training_report.json"
            _report = _j.loads(_rp.read_text())
            _minfo = _report.get("models", {})
            _k = ["xgboost", "lightgbm", "lstm", "transformer"]
            _acc = [i for i, k in enumerate(_k) if _minfo.get(k, {}).get("accepted", False)]
            if _acc:
                _w = [0.0, 0.0, 0.0, 0.0]
                for i in _acc:
                    _w[i] = 1.0 / len(_acc)
                self._weights = np.array(_w, dtype=float)
            else:
                self._weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        except Exception:
            self._weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # transformer-only fallback
        self._bias = 0.0
        self._obs: deque[_Observation] = deque(maxlen=self.WINDOW_SIZE)
        self._obs_since_update = 0
        self._n_updates = 0
        self._fallback_weights: dict[str, float] = fallback_weights or {
            "xgb": 0.40, "lgbm": 0.25, "lstm": 0.25, "transformer": 0.10
        }

    def predict(self, xgb_prob: float, lgbm_prob: float, lstm_prob: float, transformer_prob: float) -> float:
        """
        Blend four model probabilities into a single prediction in [0, 1].
        Falls back to weighted average when fewer than MIN_OBS_FOR_LEARNING observations.
        """
        features = [xgb_prob, lgbm_prob, lstm_prob, transformer_prob]
        
        if len(self._obs) < self.MIN_OBS_FOR_LEARNING:
            # Fallback: weighted average
            fw = self._fallback_weights
            return (
                fw["xgb"] * xgb_prob
                + fw["lgbm"] * lgbm_prob
                + fw["lstm"] * lstm_prob
                + fw["transformer"] * transformer_prob
            )
        
        # Logistic regression: sigmoid(w·x + b)
        x = np.array(features, dtype=float)
        z = float(np.dot(self._weights, x) + self._bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def record_outcome(self, xgb_prob: float, lgbm_prob: float, lstm_prob: float, transformer_prob: float, outcome: int) -> None:
        """Record a prediction and its actual outcome (1=up, 0=down)."""
        self._obs.append(_Observation(
            features=[xgb_prob, lgbm_prob, lstm_prob, transformer_prob],
            label=outcome,
        ))
        self._obs_since_update += 1
        
        if self._obs_since_update >= self.UPDATE_EVERY and len(self._obs) >= self.MIN_OBS_FOR_LEARNING:
            self._sgd_update()
            self._obs_since_update = 0
            self._n_updates += 1
            log.debug("Meta-learner updated (n_updates=%d, obs=%d)", self._n_updates, len(self._obs))

    def _sgd_update(self) -> None:
        """One epoch of SGD over the current sliding window."""
        obs_list = list(self._obs)
        for obs in obs_list:
            x = np.array(obs.features, dtype=float)
            z = float(np.dot(self._weights, x) + self._bias)
            pred = 1.0 / (1.0 + np.exp(-z))
            error = obs.label - pred
            self._weights += self.LEARNING_RATE * error * x
            self._bias += self.LEARNING_RATE * error

    @property
    def accuracy(self) -> float:
        """Accuracy over current sliding window (0.0 if no observations)."""
        if len(self._obs) < self.MIN_OBS_FOR_LEARNING:
            return 0.0
        correct = 0
        for obs in self._obs:
            pred = self.predict(*obs.features)
            correct += 1 if (pred >= 0.5) == bool(obs.label) else 0
        return correct / len(self._obs)

    def get_status(self) -> dict[str, object]:
        return {
            "n_observations": len(self._obs),
            "n_updates": self._n_updates,
            "using_learned_weights": len(self._obs) >= self.MIN_OBS_FOR_LEARNING,
            "accuracy": self.accuracy,
        }


_singleton: StackingMetaLearner | None = None


def get_meta_learner() -> StackingMetaLearner:
    global _singleton
    if _singleton is None:
        _singleton = StackingMetaLearner()
    return _singleton
