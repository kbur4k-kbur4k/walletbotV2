"""
wallet_manager/strategies/adaptive_ensemble.py

Sprint JS-3.1: Regime-Conditional Model Weights
=================================================
Different ML models perform better in different market regimes.
XGBoost/LightGBM → trending markets (non-linear, feature-rich)
LSTM            → volatile / mean-reverting markets (temporal patterns)

Tracks per-model IC by regime over last 50 cycles.
Applies softmax-normalized regime-specific weights to ensemble output.
Updated weekly via walk-forward IC measurement.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

MODEL_NAMES = ["xgboost", "lightgbm", "lstm", "sklearn"]
_IC_WINDOW_PER_REGIME = 50
_SOFTMAX_TEMP = 2.0   # Lower = sharper weight differentiation


@dataclass
class ModelCycleRecord:
    timestamp: float
    regime: str
    model_name: str
    prediction: float      # raw model output (e.g. alpha score)
    forward_return: float  # filled in after forward window
    resolved: bool = False


class RegimeModelTracker:
    """Tracks prediction accuracy per model per regime."""

    def __init__(self, forward_minutes: float = 30.0) -> None:
        self._forward_s = forward_minutes * 60.0
        self._pending: list[ModelCycleRecord] = []
        self._resolved: dict[str, dict[str, deque[ModelCycleRecord]]] = defaultdict(
            lambda: {m: deque(maxlen=_IC_WINDOW_PER_REGIME) for m in MODEL_NAMES}
        )

    def record(self, regime: str, predictions: dict[str, float], price: float) -> None:
        """Record per-model predictions for the current cycle."""
        for model_name, pred in predictions.items():
            self._pending.append(ModelCycleRecord(
                timestamp=time.time(),
                regime=regime,
                model_name=model_name,
                prediction=pred,
                forward_return=0.0,
            ))

    def resolve(self, current_price: float, ref_price: float, now: float | None = None) -> None:
        """Resolve pending records whose forward window has elapsed."""
        now = now or time.time()
        still_pending: list[ModelCycleRecord] = []
        for rec in self._pending:
            if now - rec.timestamp >= self._forward_s and ref_price > 0:
                rec.forward_return = (current_price - ref_price) / ref_price
                rec.resolved = True
                self._resolved[rec.regime][rec.model_name].append(rec)
            else:
                still_pending.append(rec)
        self._pending = still_pending

    def get_regime_weights(self, regime: str) -> dict[str, float]:
        """
        Return softmax-normalized weights for this regime.
        Falls back to equal weights when insufficient data.
        """
        regime_data = self._resolved.get(regime, {})
        ics: dict[str, float] = {}
        for model in MODEL_NAMES:
            records = list(regime_data.get(model, []))
            if len(records) < 5:
                ics[model] = 0.0
                continue
            preds = [r.prediction for r in records]
            rets = [r.forward_return for r in records]
            ics[model] = max(0.0, _pearson(preds, rets))

        total_ic = sum(ics.values())
        if total_ic < 0.01:
            equal = 1.0 / len(MODEL_NAMES)
            return dict.fromkeys(MODEL_NAMES, equal)

        # Softmax over ICs
        scaled = {m: ic * _SOFTMAX_TEMP for m, ic in ics.items()}
        exp_vals = {m: math.exp(v) for m, v in scaled.items()}
        total_exp = sum(exp_vals.values())
        return {m: round(exp_vals[m] / total_exp, 4) for m in MODEL_NAMES}

    def get_report(self) -> dict[str, Any]:
        regimes = list(self._resolved.keys())
        report: list[dict[str, Any]] = []
        for regime in regimes:
            weights = self.get_regime_weights(regime)
            report.append({
                "regime": regime,
                "weights": weights,
                "dominant_model": max(weights, key=lambda m: weights[m]),
                "n_records": {m: len(self._resolved[regime][m]) for m in MODEL_NAMES},
            })
        return {
            "regimes_tracked": len(regimes),
            "pending_count": len(self._pending),
            "regime_weights": report,
        }


class AdaptiveEnsemble:
    """
    Applies regime-conditional weights to the ML ensemble output.

    Usage in run_shadow_cycle:
        ensemble = get_adaptive_ensemble()
        weighted_alpha = ensemble.apply_weights(
            regime_label=current_regime,
            model_alphas={"xgboost": 0.12, "lightgbm": 0.10, "lstm": -0.05, "sklearn": 0.08},
        )
    """

    def __init__(self) -> None:
        self.tracker = RegimeModelTracker()
        self._last_price: float = 0.0
        self._last_regime: str = "unknown"

    def record_cycle(
        self,
        regime: str,
        model_predictions: dict[str, float],
        current_price: float,
    ) -> None:
        self._last_regime = regime
        self.tracker.record(regime, model_predictions, current_price)
        if self._last_price > 0:
            self.tracker.resolve(current_price, self._last_price)
        self._last_price = current_price

    def apply_weights(self, regime: str, model_alphas: dict[str, float]) -> float:
        """
        Apply regime-conditional weights to model alphas.
        Returns weighted ensemble alpha.
        """
        weights = self.tracker.get_regime_weights(regime)
        weighted = sum(model_alphas.get(m, 0.0) * weights.get(m, 0.25) for m in MODEL_NAMES)
        return round(weighted, 6)

    def get_current_weights(self) -> dict[str, float]:
        return self.tracker.get_regime_weights(self._last_regime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_regime": self._last_regime,
            "current_weights": self.get_current_weights(),
            "report": self.tracker.get_report(),
        }


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (dx * dy) if dx * dy > 1e-12 else 0.0


_ensemble: AdaptiveEnsemble | None = None


def get_adaptive_ensemble() -> AdaptiveEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = AdaptiveEnsemble()
    return _ensemble
