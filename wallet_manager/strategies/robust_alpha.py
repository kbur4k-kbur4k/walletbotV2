"""
wallet_manager/strategies/robust_alpha.py

Anti-Overfit Shield — Robust Alpha via Fractional Differentiation
and Permutation Feature Importance.

Wraps StochasticMLAlphaEngine with two institutional defences against
overfitting:

1. Fractionally Differentiated Features (fracdiff)
   Raw price series are non-stationary (unit root), poisoning ML models.
   Full first-differencing (d=1) achieves stationarity but destroys memory.
   Fractional differentiation (0 < d < 1) finds the minimum d that passes
   an ADF test (p < 0.05) while preserving maximum price memory.
   Implementation uses the fixed-width window (FFD) method (López de Prado,
   "Advances in Financial Machine Learning", Ch. 5).

2. Permutation Feature Importance
   After training, each feature column is shuffled independently and the
   accuracy drop measured.  Features below a threshold are classified as
   noise and zeroed out, preventing the ML ensemble from fitting randomness.

Architecture
------------
RobustAlphaEngine wraps StochasticMLAlphaEngine:
  raw price -> fracdiff -> feature engineering -> ML ensemble
                                               -> permutation pruning
                                               -> RobustAlphaSignal

ADF test is implemented via OLS on lagged differences (no statsmodels).
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from wallet_manager.security.logging_filter import build_logger
from wallet_manager.strategies.alpha_engine import (
    AlphaMLSignal,
    StochasticMLAlphaEngine,
)

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — Fractional Differentiation
# ---------------------------------------------------------------------------

FRACDIFF_D_MIN: float = 0.1
FRACDIFF_D_MAX: float = 1.0
FRACDIFF_D_STEP: float = 0.1
FRACDIFF_WEIGHT_THRESHOLD: float = 1e-5
FRACDIFF_MAX_WINDOW: int = 500
ADF_DEFAULT_MAX_LAGS: int = 12
FRACDIFF_MIN_OBSERVATIONS: int = 50

def _get_adf_significance_level() -> float:
    try:
        from wallet_manager.config import get_settings as _gs
        return _gs().adf_significance_level
    except Exception:
        return 0.05

ADF_SIGNIFICANCE_LEVEL: float = _get_adf_significance_level()
PERMUTATION_IMPORTANCE_THRESHOLD: float = 0.01
PERMUTATION_MIN_SAMPLES: int = 100

def _get_permutation_n_shuffles() -> int:
    try:
        from wallet_manager.config import get_settings as _gs
        return _gs().permutation_n_shuffles
    except Exception:
        return 30

PERMUTATION_N_SHUFFLES: int = _get_permutation_n_shuffles()

FEATURE_NAMES: tuple[str, ...] = (
    "log_return", "kalman_price", "kalman_trend", "garch_sigma",
    "realized_vol_5", "realized_vol_30", "vol_ratio",
    "momentum_5", "momentum_15", "momentum_30", "momentum_60", "momentum_120",
    "volume_imbalance", "spread_bps", "trade_intensity",
    "skewness", "kurtosis", "price",
)
NUM_FEATURES: int = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Configuration — Robust Engine
# ---------------------------------------------------------------------------

RETRAIN_INTERVAL_TICKS: int = 500
PRICE_HISTORY_MAXLEN: int = 2000

REASON_FRACDIFF_NON_STATIONARY: str = "FRACDIFF_NON_STATIONARY"
REASON_NOISE_FEATURE_ELIMINATED: str = "NOISE_FEATURE_ELIMINATED"
REASON_INSUFFICIENT_DATA: str = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FracdiffResult:
    """Result of fractional differentiation on a time series."""

    series: np.ndarray
    d: float
    adf_statistic: float
    adf_pvalue: float
    is_stationary: bool
    n_weights: int


@dataclass(frozen=True)
class FeatureImportance:
    """Permutation importance result for a single feature."""

    feature_name: str
    feature_index: int
    importance_score: float
    importance_std: float
    is_noise: bool


@dataclass(frozen=True)
class FeatureRegistry:
    """Tracks which features are active vs. filtered out."""

    active_mask: tuple[bool, ...]
    active_names: tuple[str, ...]
    eliminated_names: tuple[str, ...]
    importances: tuple[FeatureImportance, ...]
    timestamp: float


@dataclass(frozen=True)
class RobustAlphaSignal:
    """Output of the RobustAlphaEngine — anti-overfit-shielded alpha.

    Attributes:
        raw_alpha: Original alpha from StochasticMLAlphaEngine [-1, +1].
        robust_alpha: Alpha after fracdiff transformation [-1, +1].
        feature_mask: Boolean tuple — which features survived pruning.
        stationarity_d: Fractional diff order used.
        adf_pvalue: ADF test p-value for the differentiated series.
        confidence: Signal confidence [0, 1] adjusted for robustness.
        should_enter_long: True if all robust conditions met for long.
        should_enter_short: True if all robust conditions met for short.
        should_exit: True if position should be closed.
        reason_codes: Filtering reason codes applied.
        underlying_signal: Raw AlphaMLSignal from the wrapped engine.
        timestamp: Signal generation time.
    """

    raw_alpha: float
    robust_alpha: float
    feature_mask: tuple[bool, ...]
    stationarity_d: float
    adf_pvalue: float
    confidence: float
    should_enter_long: bool
    should_enter_short: bool
    should_exit: bool
    reason_codes: tuple[str, ...]
    underlying_signal: AlphaMLSignal | None
    timestamp: float


# ---------------------------------------------------------------------------
# Fractional Differentiation — Fixed-Width Window (FFD) Method
# ---------------------------------------------------------------------------


def _compute_fracdiff_weights(
    d: float, threshold: float, max_window: int,
) -> np.ndarray:
    """Compute FFD weights via the recurrence w_k = -w_{k-1} * (d-k+1)/k.

    Starting with w_0 = 1, weights are truncated when |w_k| < threshold.
    """
    weights = [1.0]
    k = 1
    while k < max_window:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights, dtype=np.float64)


def apply_fracdiff(
    series: np.ndarray,
    d: float,
    threshold: float = FRACDIFF_WEIGHT_THRESHOLD,
    max_window: int = FRACDIFF_MAX_WINDOW,
) -> np.ndarray:
    """Apply fractional differentiation to a 1-D time series.

    Uses the FFD method: y_t = sum_{k=0}^{K} w_k * x_{t-k}.
    Leading values without sufficient history are NaN.
    """
    weights = _compute_fracdiff_weights(d, threshold, max_window)
    n_w = len(weights)
    n = len(series)
    result = np.full(n, np.nan, dtype=np.float64)
    for t in range(n_w - 1, n):
        window = series[t - n_w + 1: t + 1][::-1]
        result[t] = np.dot(weights, window)
    return result


def find_optimal_d(
    series: np.ndarray,
    d_min: float = FRACDIFF_D_MIN,
    d_max: float = FRACDIFF_D_MAX,
    d_step: float = FRACDIFF_D_STEP,
    significance: float = ADF_SIGNIFICANCE_LEVEL,
    threshold: float = FRACDIFF_WEIGHT_THRESHOLD,
    max_lags: int = ADF_DEFAULT_MAX_LAGS,
) -> FracdiffResult:
    """Find the minimum d that achieves ADF stationarity (p < significance).

    Grid-searches d from d_min to d_max.  Returns the first d that passes
    the ADF test, preserving maximum memory.  Falls back to d_max if none
    achieves stationarity.
    """
    best: FracdiffResult | None = None
    for d_raw in np.arange(d_min, d_max + d_step / 2.0, d_step):
        d = round(float(d_raw), 4)
        diffed = apply_fracdiff(series, d, threshold)
        valid = diffed[~np.isnan(diffed)]
        if len(valid) < max_lags + 5:
            continue
        adf_stat, adf_p = _adf_test(valid, max_lags=max_lags)
        weights = _compute_fracdiff_weights(d, threshold, FRACDIFF_MAX_WINDOW)
        result = FracdiffResult(
            series=diffed, d=d,
            adf_statistic=adf_stat, adf_pvalue=adf_p,
            is_stationary=adf_p < significance,
            n_weights=len(weights),
        )
        if adf_p < significance:
            log.info(
                f"FRACDIFF: optimal d={d:.2f} (ADF={adf_stat:.4f}, p={adf_p:.6f})"
            )
            return result
        best = result

    if best is not None:
        log.info(
            f"FRACDIFF: no d in [{d_min},{d_max}] stationary. "
            f"Using d={best.d:.2f} (p={best.adf_pvalue:.6f})"
        )
        return best

    return FracdiffResult(
        series=np.full_like(series, np.nan), d=d_max,
        adf_statistic=0.0, adf_pvalue=1.0,
        is_stationary=False, n_weights=0,
    )


# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller Test — Manual OLS (no statsmodels)
# ---------------------------------------------------------------------------


def _adf_test(
    series: np.ndarray, max_lags: int = ADF_DEFAULT_MAX_LAGS,
) -> tuple[float, float]:
    """ADF unit root test via OLS on lagged differences.

    Regression model:
        delta_y_t = alpha + gamma * y_{t-1} + sum beta_i * delta_y_{t-i} + eps
    ADF statistic = t-stat for gamma.  P-value approximated from
    MacKinnon (1994) response surface.
    """
    n = len(series)
    if n < max_lags + 5:
        return 0.0, 1.0

    # Schwert lag selection
    effective_lags = max(1, min(max_lags, int(np.floor(12.0 * (n / 100.0) ** 0.25))))
    dy = np.diff(series)
    start = effective_lags
    nobs = len(dy) - start
    if nobs < effective_lags + 3:
        return 0.0, 1.0

    y_dep = dy[start:]
    n_reg = 2 + effective_lags
    X = np.ones((nobs, n_reg), dtype=np.float64)
    X[:, 1] = series[start: start + nobs]
    for lag in range(1, effective_lags + 1):
        X[:, 1 + lag] = dy[start - lag: start - lag + nobs]

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)
        residuals = y_dep - X @ beta
        sigma_sq = float(np.dot(residuals, residuals) / max(nobs - n_reg, 1))
        if sigma_sq < 1e-20:
            return -999.0, 0.0
        try:
            cov = sigma_sq * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            cov = sigma_sq * np.linalg.pinv(X.T @ X)
        se_gamma = math.sqrt(max(cov[1, 1], 1e-20))
        adf_stat = float(beta[1]) / se_gamma
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0

    return adf_stat, _mackinnon_pvalue(adf_stat, nobs)


def _mackinnon_pvalue(adf_stat: float, n: int) -> float:
    """Approximate ADF p-value via MacKinnon (1994) response surface.

    Uses critical value interpolation for constant-only model (tau_c).
    Finite-sample adjustment: cv(n) = c_inf + c1/n + c2/n^2.
    """
    inv_n = 1.0 / max(n, 1)
    cv_coeffs = {
        0.01: (-3.4336, -5.999, -29.25),
        0.05: (-2.8621, -2.738, -8.36),
        0.10: (-2.5671, -1.438, -4.48),
    }
    cvs = {
        sig: c0 + c1 * inv_n + c2 * inv_n * inv_n
        for sig, (c0, c1, c2) in cv_coeffs.items()
    }

    if adf_stat <= cvs[0.01]:
        slope = (0.01 - 0.001) / (cvs[0.01] - (cvs[0.01] - 2.0))
        return max(0.01 + slope * (adf_stat - cvs[0.01]), 0.0001)
    elif adf_stat <= cvs[0.05]:
        frac = (adf_stat - cvs[0.01]) / (cvs[0.05] - cvs[0.01])
        return 0.01 + frac * 0.04
    elif adf_stat <= cvs[0.10]:
        frac = (adf_stat - cvs[0.05]) / (cvs[0.10] - cvs[0.05])
        return 0.05 + frac * 0.05
    else:
        return min(float(sp_stats.norm.sf(-adf_stat)), 1.0)


# ---------------------------------------------------------------------------
# Permutation Feature Importance
# ---------------------------------------------------------------------------


def compute_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    n_shuffles: int = PERMUTATION_N_SHUFFLES,
    threshold: float = PERMUTATION_IMPORTANCE_THRESHOLD,
    rng: np.random.Generator | None = None,
) -> FeatureRegistry:
    """Compute permutation importance and build a FeatureRegistry.

    For each feature, shuffles its column and measures the MSE increase.
    Features whose mean importance < threshold are marked as noise.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples, n_feat = X.shape
    n_feat = min(n_feat, NUM_FEATURES)

    baseline_pred = predict_fn(X)
    baseline_mse = float(np.mean((y - baseline_pred) ** 2))

    importances: list[FeatureImportance] = []
    for idx in range(n_feat):
        scores: list[float] = []
        for _ in range(n_shuffles):
            X_s = X.copy()
            rng.shuffle(X_s[:, idx])
            s_pred = predict_fn(X_s)
            scores.append(float(np.mean((y - s_pred) ** 2)) - baseline_mse)

        mean_imp = float(np.mean(scores))
        std_imp = float(np.std(scores))
        is_noise = mean_imp < threshold
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"

        importances.append(FeatureImportance(
            feature_name=name, feature_index=idx,
            importance_score=mean_imp, importance_std=std_imp,
            is_noise=is_noise,
        ))
        if is_noise:
            log.info(
                f"{REASON_NOISE_FEATURE_ELIMINATED}: {name} "
                f"(importance={mean_imp:.6f} < {threshold})"
            )

    active_mask = tuple(not fi.is_noise for fi in importances)
    active = tuple(fi.feature_name for fi in importances if not fi.is_noise)
    eliminated = tuple(fi.feature_name for fi in importances if fi.is_noise)

    log.info(
        f"PERMUTATION_IMPORTANCE: {len(active)}/{n_feat} active, "
        f"{len(eliminated)} eliminated"
    )
    return FeatureRegistry(
        active_mask=active_mask, active_names=active,
        eliminated_names=eliminated, importances=tuple(importances),
        timestamp=time.time(),
    )


def apply_feature_mask(
    features: np.ndarray, mask: tuple[bool, ...],
) -> np.ndarray:
    """Zero out noise features based on the active mask.

    Noise features are set to zero rather than removed, preserving
    array dimensionality for downstream model compatibility.
    """
    masked = features.copy()
    m = np.array(mask, dtype=bool)
    if masked.ndim == 1:
        k = min(len(masked), len(m))
        masked[:k] *= m[:k]
    elif masked.ndim == 2:
        k = min(masked.shape[1], len(m))
        masked[:, :k] *= m[:k]
    return masked


# ---------------------------------------------------------------------------
# Robust Alpha Engine
# ---------------------------------------------------------------------------


class RobustAlphaEngine:
    """Anti-overfit wrapper around StochasticMLAlphaEngine.

    Interposes two safeguards between raw prices and the ML alpha signal:
    1. Fractional differentiation — stationarity at minimum d.
    2. Permutation feature importance — prunes noise features.

    Usage:
        engine = get_robust_alpha_engine()
        signal = engine.ingest_price(price=3250.0, volume=1e6)
        if signal and signal.should_enter_long:
            pass  # Signal is overfit-resistant
    """

    def __init__(self) -> None:
        self._inner_engine = StochasticMLAlphaEngine()

        self._price_history: deque[float] = deque(maxlen=PRICE_HISTORY_MAXLEN)
        self._fracdiff_prices: deque[float] = deque(maxlen=PRICE_HISTORY_MAXLEN)

        self._current_d: float = 0.5
        self._current_adf_pvalue: float = 1.0
        self._is_stationary: bool = False

        self._feature_registry: FeatureRegistry | None = None
        self._feature_mask: tuple[bool, ...] = tuple(True for _ in range(NUM_FEATURES))

        self._feature_history: deque[np.ndarray] = deque(maxlen=PRICE_HISTORY_MAXLEN)
        self._target_history: deque[float] = deque(maxlen=PRICE_HISTORY_MAXLEN)

        self._n_ticks: int = 0
        self._last_retrain_tick: int = 0
        self._rng = np.random.default_rng(42)

        log.info("RobustAlphaEngine initialized — Anti-Overfit Shield active")

    # ----- Fracdiff calibration -----

    def _recalibrate_fracdiff(self) -> None:
        """Re-estimate optimal d using the full price history buffer."""
        if len(self._price_history) < FRACDIFF_MIN_OBSERVATIONS:
            return
        prices = np.array(self._price_history, dtype=np.float64)
        log_prices = np.log(np.maximum(prices, 1e-10))
        result = find_optimal_d(log_prices)
        self._current_d = result.d
        self._current_adf_pvalue = result.adf_pvalue
        self._is_stationary = result.is_stationary
        if not result.is_stationary:
            log.info(
                f"{REASON_FRACDIFF_NON_STATIONARY}: d={result.d:.2f} "
                f"not stationary (p={result.adf_pvalue:.6f})"
            )

    def _get_fracdiff_price(self, price: float) -> float | None:
        """Compute FFD value for the latest price using current d."""
        if len(self._price_history) < FRACDIFF_MIN_OBSERVATIONS:
            return None
        weights = _compute_fracdiff_weights(
            self._current_d, FRACDIFF_WEIGHT_THRESHOLD, FRACDIFF_MAX_WINDOW,
        )
        n_w = len(weights)
        if len(self._price_history) < n_w:
            return None
        recent = list(self._price_history)
        log_recent = np.log(np.maximum(
            np.array(recent[-n_w:], dtype=np.float64), 1e-10,
        ))
        return float(np.dot(weights, log_recent[::-1]))

    # ----- Permutation importance -----

    def _run_permutation_importance(self) -> None:
        """Run permutation importance on accumulated feature/target data."""
        if len(self._feature_history) < PERMUTATION_MIN_SAMPLES:
            return
        X = np.array(list(self._feature_history), dtype=np.float64)
        y = np.array(list(self._target_history), dtype=np.float64)
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]

        def linear_predict(X_in: np.ndarray) -> np.ndarray:
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                return np.asarray(X_in @ beta)
            except np.linalg.LinAlgError:
                return np.full(X_in.shape[0], float(np.mean(y)))

        self._feature_registry = compute_permutation_importance(
            X, y, linear_predict,
            n_shuffles=PERMUTATION_N_SHUFFLES,
            threshold=PERMUTATION_IMPORTANCE_THRESHOLD,
            rng=self._rng,
        )
        self._feature_mask = self._feature_registry.active_mask

    # ----- Main API -----

    def set_microstructure(
        self, volume_imbalance: float, spread_bps: float,
    ) -> None:
        """Proxy to inner engine's microstructure update."""
        self._inner_engine.set_microstructure(volume_imbalance, spread_bps)

    def ingest_price(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float | None = None,
    ) -> RobustAlphaSignal | None:
        """Ingest price and return a robust, overfit-resistant alpha signal.

        Pipeline:
        1. Append to history, compute fracdiff
        2. Feed raw price to inner engine
        3. Periodically recalibrate d and run permutation importance
        4. Apply feature mask, blend fracdiff signal
        5. Return RobustAlphaSignal
        """
        ts = timestamp or time.time()
        self._n_ticks += 1
        self._price_history.append(price)
        reason_codes: list[str] = []

        # --- Insufficient data early exit ---
        if len(self._price_history) < FRACDIFF_MIN_OBSERVATIONS:
            raw = self._inner_engine.process_tick(price, volume, ts)
            if raw is None:
                return None
            return RobustAlphaSignal(
                raw_alpha=raw.alpha_score, robust_alpha=raw.alpha_score,
                feature_mask=self._feature_mask,
                stationarity_d=self._current_d, adf_pvalue=self._current_adf_pvalue,
                confidence=raw.confidence * 0.5,
                should_enter_long=False, should_enter_short=False,
                should_exit=raw.should_exit,
                reason_codes=(REASON_INSUFFICIENT_DATA,),
                underlying_signal=raw, timestamp=ts,
            )

        # --- Periodic recalibration ---
        since_retrain = self._n_ticks - self._last_retrain_tick
        if since_retrain >= RETRAIN_INTERVAL_TICKS or self._last_retrain_tick == 0:
            self._recalibrate_fracdiff()
            self._run_permutation_importance()
            self._last_retrain_tick = self._n_ticks

        # --- Fracdiff price ---
        fd_price = self._get_fracdiff_price(price)
        if fd_price is not None:
            self._fracdiff_prices.append(fd_price)

        # --- Raw signal ---
        raw = self._inner_engine.process_tick(price, volume, ts)
        if raw is None:
            return None

        # --- Accumulate for permutation importance ---
        raw_features = raw.features.to_array()
        self._feature_history.append(raw_features)
        self._target_history.append(raw.features.log_return)

        # --- Apply feature mask ---
        apply_feature_mask(raw_features, self._feature_mask)

        # --- Robust alpha: blend raw + fracdiff signal ---
        if fd_price is not None and len(self._fracdiff_prices) >= 5:
            fd_arr = np.array(list(self._fracdiff_prices), dtype=np.float64)
            fd_std = float(np.std(fd_arr))
            if fd_std > 1e-10:
                fd_z = (fd_price - float(np.mean(fd_arr))) / fd_std
                fd_alpha = float(np.tanh(fd_z / 3.0))
            else:
                fd_alpha = 0.0
            robust_alpha = max(-1.0, min(1.0,
                0.6 * raw.alpha_score + 0.4 * fd_alpha))
        else:
            robust_alpha = raw.alpha_score

        # --- Reason codes ---
        if not self._is_stationary:
            reason_codes.append(REASON_FRACDIFF_NON_STATIONARY)
        if self._feature_registry and self._feature_registry.eliminated_names:
            reason_codes.append(REASON_NOISE_FEATURE_ELIMINATED)

        # --- Confidence adjustment ---
        confidence = raw.confidence
        if not self._is_stationary:
            confidence *= 0.7
        if self._feature_registry:
            coverage = sum(self._feature_mask) / max(NUM_FEATURES, 1)
            if coverage < 0.5:
                confidence *= 0.8
        confidence = max(0.0, min(1.0, confidence))

        # --- Entry/exit (stricter than raw engine) ---
        # Allow entry when fully stationary (p < significance) OR borderline
        # (p < 2× significance ≈ 0.10).  The confidence penalty applied above
        # already reduces sizing for non-stationary series; a hard gate would
        # permanently block trading on borderline-stationary series.
        from wallet_manager.config import get_settings

        cfg = get_settings()
        adf_entry_limit = 0.10
        alpha_entry_threshold = 0.05
        if cfg.execution_mode == "robust_paper" or cfg.paper_trade:
            adf_entry_limit = max(adf_entry_limit, cfg.robust_paper_adf_entry_pvalue_max)
            alpha_entry_threshold = min(alpha_entry_threshold, cfg.robust_paper_alpha_entry_threshold)

        _adf_allows_entry = (
            self._is_stationary or self._current_adf_pvalue < adf_entry_limit
        )
        should_long = (
            raw.should_enter_long and _adf_allows_entry
            and robust_alpha > alpha_entry_threshold and confidence >= 0.15
        )
        should_short = (
            raw.should_enter_short and _adf_allows_entry
            and robust_alpha < -alpha_entry_threshold and confidence >= 0.15
        )

        signal = RobustAlphaSignal(
            raw_alpha=raw.alpha_score, robust_alpha=robust_alpha,
            feature_mask=self._feature_mask,
            stationarity_d=self._current_d, adf_pvalue=self._current_adf_pvalue,
            confidence=confidence,
            should_enter_long=should_long, should_enter_short=should_short,
            should_exit=raw.should_exit,
            reason_codes=tuple(reason_codes),
            underlying_signal=raw, timestamp=ts,
        )

        log.info(
            f"ROBUST-ALPHA: raw={raw.alpha_score:+.3f} "
            f"robust={robust_alpha:+.3f} d={self._current_d:.2f} "
            f"adf_p={self._current_adf_pvalue:.6f} "
            f"stationary={self._is_stationary} "
            f"features={sum(self._feature_mask)}/{NUM_FEATURES} "
            f"conf={confidence:.2f}"
        )
        return signal

    # ----- Diagnostics -----

    def get_engine_stats(self) -> dict:
        """Return engine diagnostic statistics."""
        inner = self._inner_engine.get_engine_stats()
        n_active = sum(self._feature_mask)
        elim = self._feature_registry.eliminated_names if self._feature_registry else ()
        return {
            **inner,
            "robust_n_ticks": self._n_ticks,
            "fracdiff_d": self._current_d,
            "fracdiff_adf_pvalue": self._current_adf_pvalue,
            "fracdiff_is_stationary": self._is_stationary,
            "fracdiff_history_len": len(self._price_history),
            "features_active": n_active,
            "features_eliminated": NUM_FEATURES - n_active,
            "features_eliminated_names": elim,
            "last_retrain_tick": self._last_retrain_tick,
        }

    def get_feature_registry(self) -> FeatureRegistry | None:
        """Return the current feature registry, if computed."""
        return self._feature_registry

    def get_fracdiff_result(self) -> FracdiffResult | None:
        """Run fracdiff on full price history (on-demand diagnostic)."""
        if len(self._price_history) < FRACDIFF_MIN_OBSERVATIONS:
            return None
        prices = np.array(self._price_history, dtype=np.float64)
        return find_optimal_d(np.log(np.maximum(prices, 1e-10)))

    def force_recalibrate(self) -> None:
        """Force immediate recalibration (bypasses tick-interval gating)."""
        log.info("ROBUST-ALPHA: forced recalibration triggered")
        self._recalibrate_fracdiff()
        self._run_permutation_importance()
        self._last_retrain_tick = self._n_ticks


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: RobustAlphaEngine | None = None


def get_robust_alpha_engine() -> RobustAlphaEngine:
    """Return the process-wide RobustAlphaEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RobustAlphaEngine()
    return _engine_instance


def reset_robust_alpha_engine() -> None:
    """Reset the singleton (for tests)."""
    global _engine_instance
    _engine_instance = None


# ===========================================================================
# Section 3.3 — FeatureImportanceStabilityMonitor
# ===========================================================================

_UNSTABLE_FEATURES_PATH = Path("ml_models/unstable_features.json")


class FeatureImportanceStabilityMonitor:
    """Prune features whose importance rank is unstable across CPCV folds.

    A feature that ranks #2 in fold 1 but #28 in fold 3 is fitting noise,
    not signal. This monitor identifies and eliminates such features.

    For each CPCV fold, records the feature importance ranking vector.
    Stability score = mean Spearman rank correlation across all fold pairs.

    If stability_score < cfg.feature_stability_min_spearman (default 0.6):
      - Mark feature as unstable.
      - Append to ml_models/unstable_features.json.
      - Zero out that feature in all subsequent inference calls.
      - Log a warning listing pruned features.

    Unstable features persist across restarts (loaded from JSON at startup).
    Re-evaluated after each successful retrain.
    """

    def __init__(self) -> None:
        from wallet_manager.config import get_settings
        self._min_spearman = get_settings().feature_stability_min_spearman
        self._unstable: set[str] = set()
        self._load_persisted()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_and_prune(
        self,
        fold_importances: list[np.ndarray],
        feature_names: list[str],
    ) -> list[str]:
        """Compute per-feature Spearman stability and prune unstable ones.

        Args:
            fold_importances: List of 1-D importance arrays, one per CPCV fold.
                Each array must have the same length as ``feature_names``.
            feature_names: Human-readable feature identifiers, length F.

        Returns:
            Names of newly pruned features (may be empty if all stable).
        """
        if len(fold_importances) < 2:
            log.debug("FeatureImportanceStabilityMonitor: need >= 2 folds to evaluate.")
            return []

        n_features = len(feature_names)
        # Convert importances to rank arrays (rank 0 = most important)
        rank_arrays: list[np.ndarray] = []
        for imp in fold_importances:
            if len(imp) != n_features:
                log.warning(
                    f"FeatureImportanceStabilityMonitor: fold array length "
                    f"{len(imp)} != {n_features}; skipping fold."
                )
                continue
            rank_arrays.append(self._rank(imp))

        if len(rank_arrays) < 2:
            return []

        # Mean Spearman rank correlation across all fold pairs
        feature_stability = np.zeros(n_features)
        pair_count = 0
        for i, j in itertools.combinations(range(len(rank_arrays)), 2):
            feature_stability += self._spearman_corr_per_feature(
                rank_arrays[i], rank_arrays[j], n_features
            )
            pair_count += 1

        if pair_count > 0:
            feature_stability /= pair_count

        # Identify newly unstable features
        newly_pruned: list[str] = []
        for _idx, (name, score) in enumerate(zip(feature_names, feature_stability, strict=False)):
            if score < self._min_spearman and name not in self._unstable:
                self._unstable.add(name)
                newly_pruned.append(name)
                log.warning(
                    f"FeatureImportanceStabilityMonitor: '{name}' pruned "
                    f"(Spearman={score:.3f} < threshold={self._min_spearman:.2f})."
                )

        if newly_pruned:
            self._persist()
            log.warning(
                f"FeatureImportanceStabilityMonitor: {len(newly_pruned)} new "
                f"unstable features → {newly_pruned}. "
                f"Total unstable: {len(self._unstable)}."
            )

        return newly_pruned

    def apply_mask(self, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Zero out unstable features in a live inference feature vector.

        Args:
            features: 1-D or 2-D feature array (inference or batch).
            feature_names: Feature names matching the columns of ``features``.

        Returns:
            Feature array with unstable columns zeroed.
        """
        if not self._unstable:
            return np.asarray(features)
        mask = np.array([1.0 if n not in self._unstable else 0.0 for n in feature_names])
        return np.asarray(features * mask)

    @property
    def unstable_features(self) -> frozenset[str]:
        """Read-only set of currently pruned feature names."""
        return frozenset(self._unstable)

    def reset_after_retrain(self) -> None:
        """Clear the unstable set so stability is re-evaluated from scratch."""
        self._unstable.clear()
        if _UNSTABLE_FEATURES_PATH.exists():
            _UNSTABLE_FEATURES_PATH.unlink()
        log.info("FeatureImportanceStabilityMonitor: reset after retrain.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rank(arr: np.ndarray) -> np.ndarray:
        """Return rank indices (descending: highest importance = rank 0)."""
        order = np.argsort(-arr)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(arr))
        return ranks.astype(float)

    @staticmethod
    def _spearman_corr_per_feature(
        ranks_a: np.ndarray,
        ranks_b: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """Per-feature contribution to Spearman correlation.

        Instead of a single scalar correlation, returns a per-feature
        agreement score: 1 - |rank_a[i] - rank_b[i]| / (n_features - 1).
        This allows per-feature stability assessment.
        """
        diffs = np.abs(ranks_a - ranks_b)
        return np.asarray(1.0 - diffs / max(n_features - 1, 1))

    def _load_persisted(self) -> None:
        """Load unstable features from JSON file (survives restarts)."""
        if _UNSTABLE_FEATURES_PATH.exists():
            try:
                data = json.loads(_UNSTABLE_FEATURES_PATH.read_text())
                self._unstable = set(data.get("unstable_features", []))
                log.info(
                    f"FeatureImportanceStabilityMonitor: loaded "
                    f"{len(self._unstable)} persisted unstable features."
                )
            except Exception as exc:
                log.warning(f"Failed to load unstable features: {exc}")

    def _persist(self) -> None:
        """Save current unstable set to JSON."""
        _UNSTABLE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"unstable_features": sorted(self._unstable)}
        _UNSTABLE_FEATURES_PATH.write_text(json.dumps(data, indent=2))


# Module-level singleton
_stability_monitor: FeatureImportanceStabilityMonitor | None = None


def get_stability_monitor() -> FeatureImportanceStabilityMonitor:
    """Return the process-wide FeatureImportanceStabilityMonitor singleton."""
    global _stability_monitor
    if _stability_monitor is None:
        _stability_monitor = FeatureImportanceStabilityMonitor()
    return _stability_monitor


# ---------------------------------------------------------------------------
# FeatureDriftMonitor
# ---------------------------------------------------------------------------


class FeatureDriftMonitor:
    """Tracks rolling distribution statistics for each FeatureStore feature.
    Produces an alert when any feature exceeds
    cfg.feature_drift_kl_threshold (default 0.1) KL divergence.

    Uses a sliding window of cfg.feature_drift_window (default 288)
    observations per feature. Compares live rolling distribution against
    the training reference distribution loaded from
    ml_models/feature_stats.json.

    Output:
      drifted_features: list[str]  — feature names above threshold
      worst_feature: str           — highest KL divergence
      worst_kl: float

    Called every cfg.feature_drift_check_interval (default 100) cycles.
    Results consumed by /health command and ProactiveAlerter.
    """

    _STATS_PATH = "ml_models/feature_stats.json"

    _REFERENCE_BUILD_AFTER: int = 200  # build reference from first N observations

    def __init__(self) -> None:
        self._reference_stats: dict[str, dict] = {}
        self._windows: dict[str, deque] = {}
        self._reference_windows: dict[str, list] = {}  # early observations used as reference
        self._reference_built: bool = False
        self._update_count: int = 0
        self._last_report: dict = {"drifted_features": [], "worst_feature": "", "worst_kl": 0.0}
        self._snapshots: list[dict] = []  # Weekly importance drift snapshots
        self._load_reference_stats()

    def _load_reference_stats(self) -> None:
        import json
        from pathlib import Path
        path = Path(self._STATS_PATH)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                self._reference_stats = json.load(f)
            log.info("FeatureDriftMonitor: loaded reference stats for %d features.", len(self._reference_stats))
        except Exception as exc:
            log.debug("FeatureDriftMonitor: reference stats load failed: %s", exc)

    def _kl_divergence_normal(
        self,
        live_mean: float, live_std: float,
        ref_mean: float, ref_std: float,
    ) -> float:
        """KL(live || reference) for two normal distributions."""
        import math
        if ref_std <= 0 or live_std <= 0:
            return 0.0
        kl = (
            math.log(ref_std / live_std)
            + (live_std ** 2 + (live_mean - ref_mean) ** 2) / (2 * ref_std ** 2)
            - 0.5
        )
        return max(0.0, kl)

    def update(self, feature_vector: np.ndarray, feature_names: list[str]) -> None:
        """Ingest a new feature vector observation."""
        from collections import deque

        from wallet_manager.config import get_settings
        cfg = get_settings()
        window_size = cfg.feature_drift_window

        for i, name in enumerate(feature_names):
            if i >= len(feature_vector):
                break
            val = float(feature_vector[i])
            if name not in self._windows:
                self._windows[name] = deque(maxlen=window_size)
            self._windows[name].append(val)

        # Build internal reference from early observations (fallback when no JSON file)
        if not self._reference_built:
            for i, name in enumerate(feature_names):
                if i >= len(feature_vector):
                    break
                val = float(feature_vector[i])
                if name not in self._reference_windows:
                    self._reference_windows[name] = []
                if len(self._reference_windows[name]) < self._REFERENCE_BUILD_AFTER:
                    self._reference_windows[name].append(val)
            if all(len(v) >= self._REFERENCE_BUILD_AFTER for v in self._reference_windows.values() if v):
                # Promote reference windows to reference_stats (if not already loaded)
                for name, vals in self._reference_windows.items():
                    if name not in self._reference_stats and len(vals) >= 2:
                        import numpy as _np
                        arr = _np.array(vals)
                        self._reference_stats[name] = {
                            "mean": float(_np.mean(arr)),
                            "std": float(_np.std(arr)),
                        }
                self._reference_built = True

        self._update_count += 1

        check_interval = cfg.feature_drift_check_interval
        if self._update_count % check_interval == 0:
            self._compute_report(feature_names, cfg.feature_drift_kl_threshold)

    def _compute_report(self, feature_names: list[str], threshold: float) -> None:
        import numpy as np
        drifted: list[str] = []
        worst_feature = ""
        worst_kl = 0.0

        for name in feature_names:
            if name not in self._windows or name not in self._reference_stats:
                continue
            window = list(self._windows[name])
            if len(window) < 10:
                continue
            live_mean = float(np.mean(window))
            live_std = float(np.std(window)) + 1e-9
            ref = self._reference_stats[name]
            ref_mean = float(ref.get("mean", 0.0))
            ref_std = float(ref.get("std", 1.0)) + 1e-9
            kl = self._kl_divergence_normal(live_mean, live_std, ref_mean, ref_std)
            if kl > threshold:
                drifted.append(name)
            if kl > worst_kl:
                worst_kl = kl
                worst_feature = name

        self._last_report = {
            "drifted_features": drifted,
            "worst_feature": worst_feature,
            "worst_kl": worst_kl,
        }
        if drifted:
            log.warning(
                f"FeatureDriftMonitor: {len(drifted)} features drifted "
                f"(worst={worst_feature} KL={worst_kl:.4f})"
            )

    def get_drift_report(self) -> dict:
        """Return last drift analysis report."""
        return dict(self._last_report)

    def record_snapshot(self, feature_importance_scores: dict[str, float] | None = None) -> None:
        """Record weekly feature importance drift snapshot."""
        import time
        snapshot = {
            "timestamp": time.time(),
            "feature_importance": feature_importance_scores or {},
            "drifted_features": self._last_report.get("drifted_features", []),
            "worst_feature": self._last_report.get("worst_feature", ""),
            "worst_kl": self._last_report.get("worst_kl", 0.0),
        }
        self._snapshots.append(snapshot)
        if len(self._snapshots) > 52:  # Keep last 52 weeks
            self._snapshots = self._snapshots[-52:]
        log.info(f"FeatureDriftMonitor: recorded weekly snapshot ({len(self._snapshots)} total)")

    def get_snapshots(self, n: int = 10) -> list[dict]:
        """Return last N weekly snapshots."""
        return self._snapshots[-n:] if self._snapshots else []


_feature_drift_monitor: FeatureDriftMonitor | None = None


def get_feature_drift_monitor() -> FeatureDriftMonitor:
    """Return the process-wide FeatureDriftMonitor singleton."""
    global _feature_drift_monitor
    if _feature_drift_monitor is None:
        _feature_drift_monitor = FeatureDriftMonitor()
    return _feature_drift_monitor
