"""
wallet_manager/signal_intelligence.py

Sprint JS-1: Signal Intelligence Layer
=======================================
Implements four institutional-grade signal quality monitors:

JS-1.1  ICTracker          — Information Coefficient per signal (rolling Pearson)
JS-1.2  SignalDecayAnalyzer — PnL decay curve after signal generation
JS-1.3  SignalCorrelationMatrix — Detect redundant signals (|corr| > 0.70)
JS-1.4  EnsembleDiversityScore — Measure model disagreement rate

All components are stateful singletons, feed from the shadow cycle, and
expose to_dict() for dashboard + API consumption.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# JS-1.1  Information Coefficient (IC) Tracker
# ---------------------------------------------------------------------------

_IC_WINDOW = 100          # observations for rolling IC
_IC_FORWARD_MINUTES = 30  # measure forward return N minutes after signal


@dataclass
class ICObservation:
    timestamp: float
    signal_name: str
    signal_value: float
    price_at_signal: float
    forward_price: float = 0.0
    forward_return: float = 0.0
    resolved: bool = False


class ICTracker:
    """
    Tracks Information Coefficient (IC) = Pearson corr(signal, forward_return)
    for each named signal over a rolling window.

    Usage in shadow cycle:
        tracker.record_signal("robust_alpha", alpha_value, current_price)

    After forward_minutes have elapsed:
        tracker.resolve_pending(current_price, current_time)

    Weekly report:
        tracker.get_ic_report()
    """

    def __init__(self, forward_minutes: float = _IC_FORWARD_MINUTES, window: int = _IC_WINDOW) -> None:
        self._forward_seconds = forward_minutes * 60.0
        self._window = window
        self._pending: list[ICObservation] = []
        self._resolved: dict[str, deque[ICObservation]] = {}

    def record_signal(self, signal_name: str, signal_value: float, price: float) -> None:
        """Record a signal value and current price for future IC calculation."""
        self._pending.append(ICObservation(
            timestamp=time.time(),
            signal_name=signal_name,
            signal_value=signal_value,
            price_at_signal=price,
        ))

    def resolve_pending(self, current_price: float, now: float | None = None) -> int:
        """Resolve pending observations whose forward window has elapsed."""
        now = now or time.time()
        resolved_count = 0
        still_pending: list[ICObservation] = []
        for obs in self._pending:
            if now - obs.timestamp >= self._forward_seconds:
                if obs.price_at_signal > 0:
                    obs.forward_price = current_price
                    obs.forward_return = (current_price - obs.price_at_signal) / obs.price_at_signal
                obs.resolved = True
                if obs.signal_name not in self._resolved:
                    self._resolved[obs.signal_name] = deque(maxlen=self._window)
                self._resolved[obs.signal_name].append(obs)
                resolved_count += 1
            else:
                still_pending.append(obs)
        self._pending = still_pending
        return resolved_count

    def compute_ic(self, signal_name: str) -> float:
        """Return rolling IC for a signal. Returns 0.0 if insufficient data."""
        obs_list = list(self._resolved.get(signal_name, []))
        if len(obs_list) < 10:
            return 0.0
        signals = [o.signal_value for o in obs_list]
        returns = [o.forward_return for o in obs_list]
        return _pearson(signals, returns)

    def get_ic_status(self, signal_name: str) -> str:
        ic = self.compute_ic(signal_name)
        if ic >= 0.05:
            return "WORKING"
        if ic >= 0.02:
            return "WEAK"
        return "NOISE"

    def get_ic_report(self) -> dict[str, Any]:
        signals = list(self._resolved.keys())
        rows: list[dict[str, Any]] = []
        for name in signals:
            ic = self.compute_ic(name)
            n = len(self._resolved[name])
            rows.append({
                "signal": name,
                "ic": round(ic, 5),
                "n_observations": n,
                "status": self.get_ic_status(name),
            })
        rows.sort(key=lambda r: r["ic"], reverse=True)
        return {
            "generated_at": time.time(),
            "forward_minutes": self._forward_seconds / 60.0,
            "window": self._window,
            "signals": rows,
            "pending_count": len(self._pending),
            "alerts": [r["signal"] for r in rows if r["status"] == "NOISE" and r["n_observations"] >= 20],
        }


# ---------------------------------------------------------------------------
# JS-1.2  Signal Decay Analyzer
# ---------------------------------------------------------------------------

_DECAY_HORIZONS_S = [5 * 60, 15 * 60, 30 * 60, 60 * 60, 4 * 60 * 60]  # 5m,15m,30m,1h,4h


@dataclass
class DecayObservation:
    timestamp: float
    price_at_signal: float
    signal_direction: int        # +1 for long signal, -1 for short
    pnl_by_horizon: dict[int, float] = field(default_factory=dict)
    resolved_horizons: set[int] = field(default_factory=set)


class SignalDecayAnalyzer:
    """
    Measures cumulative PnL at multiple horizons after a signal is generated.
    Identifies the optimal hold time for the current strategy.
    """

    def __init__(self, horizons_s: list[int] | None = None) -> None:
        self._horizons = horizons_s or _DECAY_HORIZONS_S
        self._pending: list[DecayObservation] = []
        self._completed: deque[DecayObservation] = deque(maxlen=200)

    def record_signal(self, price: float, direction: int) -> None:
        """Record a new signal for decay analysis. direction: +1=long, -1=short."""
        self._pending.append(DecayObservation(
            timestamp=time.time(),
            price_at_signal=price,
            signal_direction=direction,
        ))

    def update(self, current_price: float, now: float | None = None) -> None:
        """Called each cycle to resolve pending horizon observations."""
        now = now or time.time()
        still_pending: list[DecayObservation] = []
        for obs in self._pending:
            elapsed = now - obs.timestamp
            for h in self._horizons:
                if h not in obs.resolved_horizons and elapsed >= h:
                    ret = (current_price - obs.price_at_signal) / obs.price_at_signal
                    obs.pnl_by_horizon[h] = ret * obs.signal_direction
                    obs.resolved_horizons.add(h)
            if obs.resolved_horizons == set(self._horizons):
                self._completed.append(obs)
            else:
                still_pending.append(obs)
        self._pending = still_pending

    def get_decay_curve(self) -> dict[str, Any]:
        """Return average PnL at each horizon across all completed observations."""
        if not self._completed:
            return {"status": "NO_DATA", "decay_curve": {}, "optimal_hold_minutes": None}
        horizon_pnl: dict[int, list[float]] = {h: [] for h in self._horizons}
        for obs in self._completed:
            for h, pnl in obs.pnl_by_horizon.items():
                horizon_pnl[h].append(pnl)
        curve: dict[str, float] = {}
        for h in self._horizons:
            vals = horizon_pnl[h]
            curve[f"{h // 60}m"] = round(sum(vals) / len(vals) * 100, 4) if vals else 0.0
        best_h = max(self._horizons, key=lambda h: sum(horizon_pnl[h]) / max(len(horizon_pnl[h]), 1))
        return {
            "status": "OK",
            "n_completed": len(self._completed),
            "decay_curve_pct": curve,
            "optimal_hold_minutes": best_h // 60,
            "recommendation": f"Most PnL accumulates by {best_h // 60} minutes after signal.",
        }


# ---------------------------------------------------------------------------
# JS-1.3  Signal Correlation Matrix
# ---------------------------------------------------------------------------

_CORR_WINDOW = 200


class SignalCorrelationMatrix:
    """
    Tracks rolling Pearson correlation between all signal dimensions.
    Flags pairs with |corr| > 0.70 as redundant (adding no Sharpe benefit).
    """

    SIGNAL_NAMES = ["robust_alpha", "confidence", "adf_pvalue", "garch_sigma", "spread_bps"]

    def __init__(self, window: int = _CORR_WINDOW) -> None:
        self._window = window
        self._history: dict[str, deque[float]] = {
            name: deque(maxlen=window) for name in self.SIGNAL_NAMES
        }

    def record(
        self,
        robust_alpha: float = 0.0,
        confidence: float = 0.0,
        adf_pvalue: float = 1.0,
        garch_sigma: float = 0.0,
        spread_bps: float = 0.0,
    ) -> None:
        self._history["robust_alpha"].append(robust_alpha)
        self._history["confidence"].append(confidence)
        self._history["adf_pvalue"].append(adf_pvalue)
        self._history["garch_sigma"].append(garch_sigma)
        self._history["spread_bps"].append(spread_bps)

    def get_matrix(self) -> dict[str, Any]:
        n = min(len(v) for v in self._history.values())
        if n < 20:
            return {"status": "INSUFFICIENT_DATA", "n_observations": n, "redundant_pairs": [], "matrix": {}}
        names = self.SIGNAL_NAMES
        matrix: dict[str, dict[str, float]] = {}
        redundant: list[dict[str, Any]] = []
        for i, a in enumerate(names):
            matrix[a] = {}
            for j, b in enumerate(names):
                if i == j:
                    matrix[a][b] = 1.0
                    continue
                xa = list(self._history[a])[-n:]
                xb = list(self._history[b])[-n:]
                corr = _pearson(xa, xb)
                matrix[a][b] = round(corr, 4)
                if i < j and abs(corr) > 0.70:
                    redundant.append({"signal_a": a, "signal_b": b, "correlation": round(corr, 4)})
        return {
            "status": "OK",
            "n_observations": n,
            "matrix": matrix,
            "redundant_pairs": redundant,
            "alert": len(redundant) > 0,
        }


# ---------------------------------------------------------------------------
# JS-1.4  Ensemble Diversity Score
# ---------------------------------------------------------------------------

_DIVERSITY_WINDOW = 100


class EnsembleDiversityScore:
    """
    Measures whether the 4 ML models are genuinely different from each other.
    Diversity = 1 - mean(pairwise agreement rate).
    Target: > 0.25. Below 0.15: ensemble is effectively a single model.
    """

    MODEL_NAMES = ["xgboost", "lightgbm", "lstm", "sklearn"]

    def __init__(self, window: int = _DIVERSITY_WINDOW) -> None:
        self._window = window
        self._predictions: dict[str, deque[int]] = {
            m: deque(maxlen=window) for m in self.MODEL_NAMES
        }

    def record(self, xgboost_signal: int, lightgbm_signal: int, lstm_signal: int, sklearn_signal: int) -> None:
        """Record per-model signal direction: +1=long, -1=short, 0=neutral."""
        self._predictions["xgboost"].append(xgboost_signal)
        self._predictions["lightgbm"].append(lightgbm_signal)
        self._predictions["lstm"].append(lstm_signal)
        self._predictions["sklearn"].append(sklearn_signal)

    def compute_diversity(self) -> float:
        n = min(len(v) for v in self._predictions.values())
        if n < 10:
            return 0.0
        models = self.MODEL_NAMES
        agreement_rates: list[float] = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = list(self._predictions[models[i]])[-n:]
                b = list(self._predictions[models[j]])[-n:]
                agree = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
                agreement_rates.append(agree)
        if not agreement_rates:
            return 0.0
        return round(1.0 - sum(agreement_rates) / len(agreement_rates), 4)

    def get_report(self) -> dict[str, Any]:
        n = min(len(v) for v in self._predictions.values())
        diversity = self.compute_diversity()
        if diversity >= 0.25:
            status = "HEALTHY"
        elif diversity >= 0.15:
            status = "LOW"
        else:
            status = "CRITICAL" if n >= 10 else "INSUFFICIENT_DATA"
        return {
            "diversity_score": diversity,
            "status": status,
            "n_observations": n,
            "target": 0.25,
            "alert": diversity < 0.15 and n >= 10,
            "interpretation": (
                "Models are sufficiently diverse." if status == "HEALTHY"
                else "Models are too similar — ensemble provides minimal Sharpe benefit."
                if status in ("LOW", "CRITICAL") else "Need more observations."
            ),
        }


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient. Returns 0.0 on degenerate input."""
    n = len(x)
    if n < 3 or len(y) != n:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    denom_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    denom = denom_x * denom_y
    return num / denom if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Aggregate SignalIntelligence facade
# ---------------------------------------------------------------------------


class SignalIntelligence:
    """Single facade that owns all four JS-1.x components."""

    def __init__(self) -> None:
        self.ic_tracker = ICTracker()
        self.decay_analyzer = SignalDecayAnalyzer()
        self.correlation_matrix = SignalCorrelationMatrix()
        self.diversity_score = EnsembleDiversityScore()

    def record_cycle(
        self,
        *,
        robust_alpha: float,
        confidence: float,
        adf_pvalue: float,
        garch_sigma: float,
        spread_bps: float,
        current_price: float,
        signal_direction: int = 0,
    ) -> None:
        """Call once per shadow cycle to feed all components."""
        self.ic_tracker.record_signal("robust_alpha", robust_alpha, current_price)
        self.ic_tracker.record_signal("confidence", confidence, current_price)
        self.ic_tracker.record_signal("adf_pvalue", adf_pvalue, current_price)
        self.ic_tracker.record_signal("garch_sigma", garch_sigma, current_price)
        self.correlation_matrix.record(robust_alpha, confidence, adf_pvalue, garch_sigma, spread_bps)
        if signal_direction != 0:
            self.decay_analyzer.record_signal(current_price, signal_direction)
        self.ic_tracker.resolve_pending(current_price)
        self.decay_analyzer.update(current_price)

    def record_model_predictions(
        self,
        xgboost: int,
        lightgbm: int,
        lstm: int,
        sklearn: int,
    ) -> None:
        self.diversity_score.record(xgboost, lightgbm, lstm, sklearn)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": time.time(),
            "ic_report": self.ic_tracker.get_ic_report(),
            "decay_curve": self.decay_analyzer.get_decay_curve(),
            "correlation_matrix": self.correlation_matrix.get_matrix(),
            "ensemble_diversity": self.diversity_score.get_report(),
        }

    def get_alerts(self) -> list[str]:
        alerts: list[str] = []
        ic = self.ic_tracker.get_ic_report()
        for s in ic.get("alerts", []):
            alerts.append(f"IC_NOISE: signal '{s}' IC < 0.02, consider deactivating.")
        div = self.diversity_score.get_report()
        if div.get("alert"):
            alerts.append(f"LOW_DIVERSITY: ensemble diversity {div['diversity_score']:.3f} < 0.15.")
        corr = self.correlation_matrix.get_matrix()
        for pair in corr.get("redundant_pairs", []):
            alerts.append(f"REDUNDANT_SIGNALS: {pair['signal_a']} / {pair['signal_b']} corr={pair['correlation']:.2f}")
        return alerts


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_signal_intel: SignalIntelligence | None = None


def get_signal_intelligence() -> SignalIntelligence:
    global _signal_intel
    if _signal_intel is None:
        _signal_intel = SignalIntelligence()
    return _signal_intel
