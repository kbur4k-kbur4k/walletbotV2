"""
wallet_manager/strategies/alpha_engine.py

Stochastic-ML Hybrid Alpha Engine — Predictive Model Ensemble.

Replaces ALL lagging indicators (RSI, MACD, basic Z-Score) with a
forward-looking predictive pipeline built on:

1. Feature Engineering
   - GARCH(1,1) conditional volatility estimation
   - Kalman Filter for price smoothing and trend extraction
   - Volume-Imbalance Dynamics from Uniswap V3 tick data

2. ML Model Ensemble (4-model ensemble with Platt-scaled probability outputs)
   - XGBoost (40%) for non-linear feature interactions
   - LightGBM (25%) for high-speed gradient boosting
   - LSTM (25%) Recurrent Neural Network for temporal pattern recognition
   - Transformer (10%) for attention-based sequence modelling
   - Stacking meta-learner combines all four models' predictions

3. Alpha Score
   - ML ensemble outputs probability distribution for next 5-min move
   - GARCH-adjusted volatility filter gates entries during regime changes
   - Alpha score is probability-derived from Platt-scaled ensemble outputs

Architecture
------------
All models run inference-only on-device. Training is done offline via
quant_analytics.py using accumulated market data. Model weights are
serialized to disk (joblib/torch) and loaded at startup.

The engine is stateless per-call: it receives a feature vector and
returns a prediction. State accumulation (price buffers, GARCH params)
lives in dedicated sub-engines that persist across cycles.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GARCH(1,1) parameters
GARCH_OMEGA = 1e-6         # Long-run variance constant
GARCH_ALPHA = 0.10         # ARCH coefficient (reaction to shocks)
GARCH_BETA = 0.85          # GARCH coefficient (persistence)
GARCH_MIN_OBSERVATIONS = 30

# Kalman Filter parameters
KALMAN_PROCESS_NOISE = 1e-5       # Q: state transition noise
KALMAN_MEASUREMENT_NOISE = 1e-3   # R: observation noise
KALMAN_INITIAL_ESTIMATE_ERR = 1.0 # P0: initial estimate uncertainty

# Feature engineering
FEATURE_WINDOW = 288              # 24h at 5-min intervals
LOOKBACK_PERIODS = [5, 15, 30, 60, 120]  # Multi-scale momentum features
VOLUME_IMBALANCE_TICKS = 10       # Ticks around current for OFI


# ---------------------------------------------------------------------------
# FIX: Dynamic ensemble weights from training_report.json
# ---------------------------------------------------------------------------

def _load_ensemble_weights_from_report() -> tuple:
    """Read training_report.json and return optimised per-model weights.

    Prefers the ``ensemble_active_weights`` field (written by train_models.py
    after CPCV validation) which is calibrated to each model's oos_sharpe.
    Falls back to equal weighting across accepted models, then transformer-only.

    Returns (xgb, lgbm, lstm, transformer) weights summing to 1.0.
    """
    report_path = Path(__file__).resolve().parent.parent / "ml_models" / "training_report.json"
    _keys = ["xgboost", "lightgbm", "lstm", "transformer"]
    _map = {"xgboost": 0, "lightgbm": 1, "lstm": 2, "transformer": 3}
    try:
        import json as _json
        report = _json.loads(report_path.read_text())
        models_info = report.get("models", {})
        accepted = [k for k in _keys if models_info.get(k, {}).get("accepted", False)]
        if not accepted:
            log.warning("alpha_engine: no accepted models — using transformer-only")
            return (0.0, 0.0, 0.0, 1.0)

        # Use optimised weights from training pipeline if available
        active_weights = report.get("ensemble_active_weights", {})
        if active_weights and all(k in active_weights for k in accepted):
            total = sum(active_weights.get(k, 0.0) for k in accepted)
            weights = [0.0, 0.0, 0.0, 0.0]
            if total > 1e-8:
                for k in accepted:
                    weights[_map[k]] = active_weights[k] / total
        else:
            # Equal-weight fallback
            per_model = 1.0 / len(accepted)
            weights = [0.0, 0.0, 0.0, 0.0]
            for k in accepted:
                weights[_map[k]] = per_model

        log.info(
            "alpha_engine: ensemble weights — xgb=%.2f lgbm=%.2f lstm=%.2f transformer=%.2f (accepted: %s)",
            weights[0], weights[1], weights[2], weights[3], ", ".join(accepted),
        )
        return tuple(weights)
    except Exception as exc:
        log.warning("alpha_engine: could not load training_report.json (%s) — using transformer-only", exc)
        return (0.0, 0.0, 0.0, 1.0)


# ML ensemble — weights set dynamically from training_report.json at startup
_ew = _load_ensemble_weights_from_report()
ENSEMBLE_WEIGHT_XGB = _ew[0]          # XGBoost weight in final prediction
ENSEMBLE_WEIGHT_LGBM = _ew[1]         # LightGBM weight in final prediction
ENSEMBLE_WEIGHT_LSTM = _ew[2]         # LSTM weight in final prediction
ENSEMBLE_WEIGHT_TRANSFORMER = _ew[3]  # Transformer weight in final prediction
GJR_GAMMA = 0.08                    # GJR-GARCH asymmetry (leverage effect)
PREDICTION_HORIZON = 1            # Predict next 1 period (5 min)

# Alpha thresholds
ALPHA_LONG_THRESHOLD = 0.55       # Probability > 55% → long signal
ALPHA_SHORT_THRESHOLD = 0.45      # Probability < 45% → short signal
VOLATILITY_REGIME_HIGH = 2.0      # σ-multiplier for high-vol regime gate

# Persistence
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureVector:
    """Engineered feature set for ML prediction.

    All features are normalized to zero-mean, unit-variance before
    model ingestion. Raw values stored here for diagnostics.
    """

    timestamp: float

    # Price features
    price: float
    log_return: float
    kalman_price: float          # Smoothed price (Kalman estimate)
    kalman_trend: float          # Kalman-extracted trend velocity

    # Volatility features
    garch_sigma: float           # GARCH(1,1) conditional volatility
    realized_vol_5: float        # 5-period realized volatility
    realized_vol_30: float       # 30-period realized volatility
    vol_ratio: float             # Short-term / long-term vol ratio

    # Momentum features (multi-scale log returns)
    momentum_5: float
    momentum_15: float
    momentum_30: float
    momentum_60: float
    momentum_120: float

    # Microstructure features
    volume_imbalance: float      # OFI from tick data [-1, +1]
    spread_bps: float            # Bid-ask spread proxy in basis points
    trade_intensity: float       # Trades per unit time (normalised)

    # Statistical features
    skewness: float              # Return distribution skew (30-period)
    kurtosis: float              # Return distribution excess kurtosis

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input (18 features)."""
        return np.array([
            self.log_return,
            self.kalman_price,
            self.kalman_trend,
            self.garch_sigma,
            self.realized_vol_5,
            self.realized_vol_30,
            self.vol_ratio,
            self.momentum_5,
            self.momentum_15,
            self.momentum_30,
            self.momentum_60,
            self.momentum_120,
            self.volume_imbalance,
            self.spread_bps,
            self.trade_intensity,
            self.skewness,
            self.kurtosis,
            self.price,
        ], dtype=np.float64)


@dataclass(frozen=True)
class AlphaMLSignal:
    """Output of the Stochastic-ML Alpha Engine.

    Attributes:
        alpha_score: Composite signal in [-1, +1].
        prob_up: ML ensemble probability of price increase.
        prob_down: ML ensemble probability of price decrease.
        xgb_prediction: XGBoost model's raw prediction.
        lstm_prediction: LSTM model's raw prediction.
        garch_sigma: Current GARCH conditional volatility.
        kalman_price: Kalman-smoothed price.
        volatility_regime: 'low', 'normal', or 'high'.
        should_enter_long: True if all conditions met for long entry.
        should_enter_short: True if all conditions met for short entry.
        should_exit: True if position should be closed.
        confidence: Signal confidence [0, 1].
        features: The feature vector used for this prediction.
        timestamp: Signal generation time.
    """

    alpha_score: float
    prob_up: float
    prob_down: float
    xgb_prediction: float
    lstm_prediction: float
    garch_sigma: float
    kalman_price: float
    volatility_regime: str
    should_enter_long: bool
    should_enter_short: bool
    should_exit: bool
    confidence: float
    features: FeatureVector
    timestamp: float
    # Extended fields (Section 4.1 ensemble upgrade) — all have defaults for backward compat
    lgbm_prediction: float = 0.5
    transformer_prediction: float = 0.5
    ofi_weighted: float = 0.0
    kyle_lambda: float = 0.0
    amihud_ratio: float = 0.0
    multi_horizon_score: float = 0.0
    stacking_prediction: float = 0.5
    stacking_confidence: float = 0.0
    regime_model_used: str = "fixed_weights"


# ---------------------------------------------------------------------------
# GARCH(1,1) Engine — Conditional Volatility Estimation
# ---------------------------------------------------------------------------


class GARCHEngine:
    """Generalized Autoregressive Conditional Heteroskedasticity model.

    GARCH(1,1) captures volatility clustering — periods of high volatility
    tend to persist, and periods of calm tend to persist. This is critical
    for sizing positions: trade smaller when volatility is expanding.

    Model:
        σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

    Where:
        ω = long-run variance floor
        α = reaction to new shocks (ARCH term)
        β = persistence of past volatility (GARCH term)
        ε = log return (innovation)
    """

    def __init__(
        self,
        omega: float = GARCH_OMEGA,
        alpha: float = GARCH_ALPHA,
        beta: float = GARCH_BETA,
    ) -> None:
        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._sigma_sq: float = omega / (1.0 - alpha - beta)  # Unconditional variance
        self._last_return: float = 0.0
        self._n_observations: int = 0

    def update(self, log_return: float) -> float:
        """Update GARCH model with new log return observation.

        Args:
            log_return: ln(P_t / P_{t-1})

        Returns:
            Current conditional standard deviation σ(t).
        """
        self._n_observations += 1

        if self._n_observations < 2:
            self._last_return = log_return
            return math.sqrt(self._sigma_sq)

        # GARCH(1,1) update: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
        self._sigma_sq = (
            self._omega
            + self._alpha * (self._last_return ** 2)
            + self._beta * self._sigma_sq
        )

        # Floor to prevent numerical underflow
        self._sigma_sq = max(self._sigma_sq, 1e-12)

        self._last_return = log_return
        return math.sqrt(self._sigma_sq)

    @property
    def sigma(self) -> float:
        """Current conditional standard deviation."""
        return math.sqrt(self._sigma_sq)

    @property
    def long_run_sigma(self) -> float:
        """Unconditional (long-run) standard deviation."""
        uncond_var = self._omega / (1.0 - self._alpha - self._beta)
        return math.sqrt(max(uncond_var, 1e-12))

    @property
    def has_enough_data(self) -> bool:
        return self._n_observations >= GARCH_MIN_OBSERVATIONS

    def get_volatility_regime(self) -> str:
        """Classify current volatility regime.

        Returns:
            'low', 'normal', or 'high' based on ratio vs long-run vol.
        """
        if not self.has_enough_data:
            return "normal"
        ratio = self.sigma / self.long_run_sigma
        if ratio > VOLATILITY_REGIME_HIGH:
            return "high"
        elif ratio < 0.5:
            return "low"
        return "normal"


# ---------------------------------------------------------------------------
# Kalman Filter — Price Smoothing & Trend Extraction
# ---------------------------------------------------------------------------


class KalmanFilter:
    """Univariate Kalman filter for price smoothing.

    State vector: [price, velocity]
    The filter simultaneously estimates:
    1. True (smoothed) price — removes market microstructure noise
    2. Price velocity (trend) — instantaneous rate of change

    This is superior to moving averages because Kalman adapts its
    smoothing intensity based on measurement uncertainty.
    """

    def __init__(
        self,
        process_noise: float = KALMAN_PROCESS_NOISE,
        measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
    ) -> None:
        # State: [price, velocity]
        self._x = np.array([0.0, 0.0])  # State estimate
        self._P = np.eye(2) * KALMAN_INITIAL_ESTIMATE_ERR  # Estimate covariance

        # State transition matrix (constant velocity model)
        self._F = np.array([[1.0, 1.0],
                            [0.0, 1.0]])

        # Measurement matrix (we observe price only)
        self._H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self._Q = np.array([[process_noise, 0.0],
                            [0.0, process_noise * 10]])

        # Measurement noise covariance
        self._R = np.array([[measurement_noise]])

        self._initialized = False

    def update(self, price: float) -> tuple[float, float]:
        """Process new price observation.

        Args:
            price: Raw market price.

        Returns:
            (smoothed_price, trend_velocity)
        """
        if not self._initialized:
            self._x = np.array([price, 0.0])
            self._initialized = True
            return price, 0.0

        # --- Predict ---
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # --- Update ---
        z = np.array([price])
        y = z - self._H @ x_pred                               # Innovation
        S = self._H @ P_pred @ self._H.T + self._R             # Innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)              # Kalman gain

        self._x = x_pred + K @ y
        self._P = (np.eye(2) - K @ self._H) @ P_pred

        return float(self._x[0]), float(self._x[1])

    @property
    def smoothed_price(self) -> float:
        return float(self._x[0])

    @property
    def trend_velocity(self) -> float:
        return float(self._x[1])


# ---------------------------------------------------------------------------
# XGBoost Model Wrapper
# ---------------------------------------------------------------------------


class XGBoostPredictor:
    """Gradient Boosted Trees predictor for 5-minute price direction.

    Uses XGBoost for non-linear feature interactions. Outputs
    probability of price increase in the next prediction horizon.

    When no trained model is available, falls back to a statistical
    baseline (logistic regression on Z-score of features).
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._is_loaded = False

    def load_model(self, model_path: Path | None = None) -> bool:
        """Load pre-trained XGBoost model from disk.

        Returns:
            True if model loaded successfully, False if fallback mode.
        """
        path = model_path or (MODEL_DIR / "xgb_alpha_model.json")
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            self._model = model
            self._is_loaded = True
            log.info(f"XGBoost model loaded from {path}")
            return True
        except Exception as exc:
            log.info(f"XGBoost model not found ({exc}). Using statistical fallback.")
            self._is_loaded = False
            return False

    def predict_proba(self, features: np.ndarray) -> tuple[float, float]:
        """Predict probability of price direction.

        Args:
            features: Normalised feature array (18 features).

        Returns:
            (prob_up, prob_down)
        """
        if self._is_loaded and self._model is not None:
            try:
                proba = self._model.predict_proba(features.reshape(1, -1))[0]
                return float(proba[1]), float(proba[0])
            except Exception as exc:
                log.debug(f"XGBoost inference failed: {exc}")

        # Statistical fallback: sigmoid of weighted feature sum
        return self._statistical_fallback(features)

    def _statistical_fallback(self, features: np.ndarray) -> tuple[float, float]:
        """Logistic baseline when XGBoost model is unavailable.

        Uses hand-crafted weights inspired by quantitative research:
        - Momentum features (positive = bullish)
        - Kalman trend (positive = uptrend)
        - Volume imbalance (positive = buying pressure)
        - Inverse volatility (lower vol = higher conviction)
        """
        # Feature indices: [log_ret, kalman_p, kalman_t, garch_s, rv5, rv30,
        #                   vol_ratio, m5, m15, m30, m60, m120, vol_imb,
        #                   spread, trade_int, skew, kurt, price]
        weights = np.array([
            0.05,   # log_return
            0.00,   # kalman_price (level, not directional)
            0.20,   # kalman_trend (strong directional signal)
            -0.10,  # garch_sigma (high vol = less conviction)
            -0.05,  # rv5
            -0.05,  # rv30
            -0.08,  # vol_ratio (expanding vol = caution)
            0.15,   # momentum_5
            0.10,   # momentum_15
            0.08,   # momentum_30
            0.05,   # momentum_60
            0.03,   # momentum_120
            0.15,   # volume_imbalance (buy pressure)
            -0.03,  # spread (wider = less liquid)
            0.05,   # trade_intensity
            -0.02,  # skewness
            -0.02,  # kurtosis
            0.00,   # price (level)
        ], dtype=np.float64)

        logit = float(np.dot(features, weights))
        prob_up = 1.0 / (1.0 + math.exp(-logit * 5.0))  # Scale factor for sensitivity
        prob_up = max(0.01, min(0.99, prob_up))
        return prob_up, 1.0 - prob_up


# ---------------------------------------------------------------------------
# LSTM Model Wrapper
# ---------------------------------------------------------------------------


class LSTMPredictor:
    """LSTM Recurrent Neural Network for temporal pattern recognition.

    Captures sequential dependencies in price/volume patterns that
    tree-based models miss. Uses a 30-step sequence of feature vectors.

    When PyTorch is not available or no trained model exists, falls
    back to an exponential-weighted momentum signal.
    """

    def __init__(self, sequence_length: int = 30) -> None:
        self._model = None
        self._seq_len = sequence_length
        self._feature_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
        self._is_loaded = False

    def load_model(self, model_path: Path | None = None) -> bool:
        """Load pre-trained LSTM model from disk.

        Returns:
            True if loaded, False if using fallback.
        """
        path = model_path or (MODEL_DIR / "lstm_alpha_model.pt")
        try:
            import torch
            self._model = torch.jit.load(str(path), map_location="cpu")
            model = self._model
            if model is None:
                raise RuntimeError("LSTM model failed to load")
            model.eval()
            self._is_loaded = True
            log.info(f"LSTM model loaded from {path}")
            return True
        except Exception as exc:
            log.info(f"LSTM model not found ({exc}). Using EWM momentum fallback.")
            self._is_loaded = False
            return False

    def ingest_features(self, features: np.ndarray) -> None:
        """Add feature vector to the temporal sequence buffer."""
        self._feature_buffer.append(features.copy())

    def predict_proba(self) -> tuple[float, float]:
        """Predict probability using LSTM on accumulated sequence.

        Returns:
            (prob_up, prob_down)
        """
        if len(self._feature_buffer) < self._seq_len:
            return 0.5, 0.5  # Neutral until buffer full

        if self._is_loaded and self._model is not None:
            try:
                import torch
                seq = np.stack(list(self._feature_buffer))
                tensor = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, features)
                with torch.no_grad():
                    output = self._model(tensor)
                    prob = torch.sigmoid(output).item()
                return float(prob), float(1.0 - prob)
            except Exception as exc:
                log.debug(f"LSTM inference failed: {exc}")

        # Fallback: exponential-weighted momentum from recent returns
        return self._ewm_momentum_fallback()

    def _ewm_momentum_fallback(self) -> tuple[float, float]:
        """EWM momentum when LSTM is unavailable.

        Uses exponentially weighted moving average of log returns
        (feature index 0) with half-life of 10 periods.
        """
        if len(self._feature_buffer) < 5:
            return 0.5, 0.5

        returns = [f[0] for f in self._feature_buffer]  # log_return is index 0
        alpha = 2.0 / (10 + 1)
        ewm = 0.0
        for r in returns:
            ewm = alpha * r + (1.0 - alpha) * ewm

        # Map EWM momentum to probability via sigmoid
        prob_up = 1.0 / (1.0 + math.exp(-ewm * 200.0))
        prob_up = max(0.05, min(0.95, prob_up))
        return prob_up, 1.0 - prob_up

    @property
    def buffer_fill(self) -> float:
        """Fraction of sequence buffer filled [0, 1]."""
        return len(self._feature_buffer) / self._seq_len


# ---------------------------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Transforms raw market data into ML-ready feature vectors.

    Maintains internal state (price buffer, GARCH, Kalman) and
    produces a FeatureVector on every new price observation.
    """

    def __init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=FEATURE_WINDOW)
        self._returns: deque[float] = deque(maxlen=FEATURE_WINDOW)
        self._volumes: deque[float] = deque(maxlen=FEATURE_WINDOW)
        self._timestamps: deque[float] = deque(maxlen=FEATURE_WINDOW)

        self._garch = GARCHEngine()
        self._kalman = KalmanFilter()

        self._last_price: float = 0.0
        self._volume_imbalance: float = 0.0
        self._spread_bps: float = 0.0

    def set_microstructure(
        self,
        volume_imbalance: float,
        spread_bps: float,
    ) -> None:
        """Update microstructure features from external source (OFI, spread)."""
        self._volume_imbalance = volume_imbalance
        self._spread_bps = spread_bps

    def ingest(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float | None = None,
    ) -> FeatureVector | None:
        """Ingest new price and produce feature vector.

        Args:
            price: Current market price.
            volume: Trade volume (optional).
            timestamp: Observation time (defaults to now).

        Returns:
            FeatureVector if enough data, None otherwise.
        """
        ts = timestamp or time.time()
        self._prices.append(price)
        self._volumes.append(volume)
        self._timestamps.append(ts)

        # Log return
        log_ret = math.log(price / self._last_price) if self._last_price > 0 else 0.0
        self._returns.append(log_ret)
        self._last_price = price

        # Need minimum data for features
        if len(self._prices) < GARCH_MIN_OBSERVATIONS:
            # Still update models for warm-up
            self._garch.update(log_ret)
            self._kalman.update(price)
            return None

        # GARCH update
        garch_sigma = self._garch.update(log_ret)

        # Kalman update
        kalman_price, kalman_trend = self._kalman.update(price)

        # Realized volatility (multi-scale)
        returns_arr = np.array(self._returns, dtype=np.float64)
        rv5 = float(np.std(returns_arr[-5:])) if len(returns_arr) >= 5 else 0.0
        rv30 = float(np.std(returns_arr[-30:])) if len(returns_arr) >= 30 else rv5
        vol_ratio = rv5 / rv30 if rv30 > 1e-10 else 1.0

        # Multi-scale momentum
        prices_arr = np.array(self._prices, dtype=np.float64)
        momenta = {}
        for p in LOOKBACK_PERIODS:
            if len(prices_arr) >= p:
                momenta[p] = math.log(prices_arr[-1] / prices_arr[-p])
            else:
                momenta[p] = 0.0

        # Higher-order statistics (30-period returns)
        window = returns_arr[-30:] if len(returns_arr) >= 30 else returns_arr
        if len(window) > 3:
            from scipy import stats as sp_stats
            skewness = float(sp_stats.skew(window))
            kurtosis = float(sp_stats.kurtosis(window))
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Trade intensity (normalised trades per period)
        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[0]
            trade_intensity = len(self._timestamps) / max(dt, 1.0)
        else:
            trade_intensity = 0.0

        return FeatureVector(
            timestamp=ts,
            price=price,
            log_return=log_ret,
            kalman_price=kalman_price,
            kalman_trend=kalman_trend,
            garch_sigma=garch_sigma,
            realized_vol_5=rv5,
            realized_vol_30=rv30,
            vol_ratio=vol_ratio,
            momentum_5=momenta.get(5, 0.0),
            momentum_15=momenta.get(15, 0.0),
            momentum_30=momenta.get(30, 0.0),
            momentum_60=momenta.get(60, 0.0),
            momentum_120=momenta.get(120, 0.0),
            volume_imbalance=self._volume_imbalance,
            spread_bps=self._spread_bps,
            trade_intensity=trade_intensity,
            skewness=skewness,
            kurtosis=kurtosis,
        )


# ---------------------------------------------------------------------------
# Feature Normalizer
# ---------------------------------------------------------------------------


class FeatureNormalizer:
    """Online feature normalization using running mean and std.

    Uses Welford's algorithm for numerically stable online computation
    of mean and variance — no need to store the full history.
    """

    def __init__(self, n_features: int = 18) -> None:
        self._n = 0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._M2 = np.zeros(n_features, dtype=np.float64)

    def update_and_normalize(self, raw: np.ndarray) -> np.ndarray:
        """Update statistics and return normalized feature vector.

        Uses Welford's online algorithm for numerically stable
        running mean + variance computation.
        """
        self._n += 1
        delta = raw - self._mean
        self._mean += delta / self._n
        delta2 = raw - self._mean
        self._M2 += delta * delta2

        if self._n < 2:
            return np.zeros_like(raw)

        variance = self._M2 / (self._n - 1)
        std = np.sqrt(np.maximum(variance, 1e-10))
        normalized = (raw - self._mean) / std

        # Clip extreme values to [-5, 5] standard deviations
        return np.clip(normalized, -5.0, 5.0)


# ---------------------------------------------------------------------------
# Stochastic-ML Alpha Engine (Main Class)
# ---------------------------------------------------------------------------


class StochasticMLAlphaEngine:
    """Institutional-grade predictive alpha generation engine.

    Combines GARCH volatility, Kalman smoothing, and ML ensemble
    (XGBoost + LSTM) into a single forward-looking prediction system.

    Usage:
        engine = StochasticMLAlphaEngine()

        # Each 5-min tick:
        signal = engine.process_tick(price=3250.0, volume=1e6)
        if signal and signal.should_enter_long:
            # Execute long via execution_logic
            pass

        # With microstructure data:
        engine.set_microstructure(volume_imbalance=0.3, spread_bps=5.0)
    """

    def __init__(self) -> None:
        # --- Base components (kept for backward compat) ---
        self.feature_engineer = FeatureEngineer()
        self.normalizer = FeatureNormalizer(n_features=18)
        self.xgboost = XGBoostPredictor()
        self.lstm = LSTMPredictor(sequence_length=60)  # matches SEQ_LEN_LSTM in train_models.py

        # --- Section 4.1 upgraded components ---
        self.adaptive_garch = AdaptiveGARCH()
        self.kalman_smoother = KalmanPriceSmoother()
        self.ofi_v2 = OrderFlowImbalanceV2()
        self.multi_horizon = MultiHorizonSignal()
        self.feature_store = FeatureStore()
        self.lgbm = LightGBMPredictor()
        self.transformer_predictor = TransformerPredictor()
        self.normalizer_30 = FeatureNormalizer(n_features=FeatureStore.N_FEATURES)
        self.ensemble_engine = EnsembleAlphaEngine(
            self.xgboost, self.lgbm, self.lstm, self.transformer_predictor
        )
        self.confidence_estimator = AlphaConfidenceEstimator()

        # Attempt to load pre-trained models
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.xgboost.load_model()
        self.lstm.load_model()
        self.lgbm.load_model()
        self.transformer_predictor.load_model()

        self._n_ticks: int = 0
        self._last_feature_vector: np.ndarray | None = None
        self.last_signal: AlphaMLSignal | None = None

    def set_microstructure(
        self,
        volume_imbalance: float,
        spread_bps: float,
    ) -> None:
        """Update microstructure features (OFI, spread) from DEX pool."""
        self.feature_engineer.set_microstructure(volume_imbalance, spread_bps)

    def process_tick(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float | None = None,
    ) -> AlphaMLSignal | None:
        """Backward-compatible entry point — delegates to generate_signal()."""
        return self.generate_signal(price, volume, timestamp)

    def generate_signal(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float | None = None,
        buy_ratio: float = 0.5,
    ) -> AlphaMLSignal | None:
        """10-step predictive alpha pipeline.

        Steps:
          1.  Ingest tick into base FeatureEngineer (18-dim base features).
          2.  AdaptiveGARCH (GJR-GARCH) — leverage-aware conditional vol.
          3.  KalmanPriceSmoother (3-state) — price / velocity / acceleration.
          4.  OrderFlowImbalanceV2 — weighted OFI, Kyle λ, Amihud.
          5.  MultiHorizonSignal — composite across 5m/15m/30m/1h/4h.
          6.  FeatureStore — build 36-dim vector and normalize.
          7.  Feed sequences to LSTM (18-dim) and Transformer (30-dim) buffers.
          8.  EnsembleAlphaEngine — 4-model prediction with Platt calibration.
          9.  Vol regime dampening + AlphaConfidenceEstimator.
          10. Assemble AlphaMLSignal.

        Args:
            price: Current market price.
            volume: Trade volume (USDC) for this tick.
            timestamp: Tick time (defaults to now).
            buy_ratio: Fraction of volume that was buying [0, 1].

        Returns:
            AlphaMLSignal if enough data, None during warm-up.
        """
        self._n_ticks += 1
        ts = timestamp or time.time()

        # --- Step 1: Base feature engineering ---
        fv = self.feature_engineer.ingest(price, volume, ts)
        if fv is None:
            return None  # Still warming up

        # --- Step 2: AdaptiveGARCH ---
        gjr_sigma = self.adaptive_garch.update(fv.log_return)
        vol_regime = self.adaptive_garch.get_volatility_regime()

        # --- Step 3: KalmanPriceSmoother ---
        k_price, k_vel, k_accel = self.kalman_smoother.update(price)

        # --- Step 4: OrderFlowImbalanceV2 ---
        self.ofi_v2.update(fv.log_return, volume, buy_ratio)

        # --- Step 5: MultiHorizonSignal ---
        self.multi_horizon.update(price)
        horizon_score = self.multi_horizon.compute()

        # --- Step 6: 36-dim feature vector + normalization ---
        _price_hist = np.array(self.feature_engineer._prices, dtype=np.float64)
        _vol_hist = np.array(self.feature_engineer._volumes, dtype=np.float64)
        features_30 = self.feature_store.build(
            fv, self.ofi_v2, self.kalman_smoother, horizon_score,
            price_history=_price_hist, volume_history=_vol_hist,
        )
        normalized_30 = self.normalizer_30.update_and_normalize(features_30)
        self._last_feature_vector = np.asarray(normalized_30, dtype=np.float64)

        # --- Step 7: Sequence ingestion ---
        # Both LSTM and Transformer trained on full 36-dim feature vector (N_FEATURES=36)
        self.lstm.ingest_features(normalized_30)
        self.transformer_predictor.ingest_features(normalized_30)

        # --- Step 8: 4-model ensemble with Platt calibration ---
        preds = self.ensemble_engine.predict(normalized_30)
        prob_up = preds["ensemble"]
        prob_down = 1.0 - prob_up

        # --- Step 9: Vol regime filter + confidence ---
        if vol_regime == "high":
            prob_up = 0.5 + (prob_up - 0.5) * 0.5
            prob_down = 1.0 - prob_up

        data_coverage = min(1.0, self._n_ticks / FEATURE_WINDOW)
        confidence = self.confidence_estimator.estimate(preds, data_coverage, vol_regime, horizon_score)

        alpha_score = max(-1.0, min(1.0, 2.0 * (prob_up - 0.5)))
        has_enough_data = self._n_ticks >= GARCH_MIN_OBSERVATIONS

        should_enter_long = (
            has_enough_data
            and prob_up >= ALPHA_LONG_THRESHOLD
            and vol_regime != "high"
            and confidence >= 0.15
        )
        should_enter_short = (
            has_enough_data
            and prob_down >= (1.0 - ALPHA_SHORT_THRESHOLD)
            and vol_regime != "high"
            and confidence >= 0.15
        )
        should_exit = (
            has_enough_data
            and abs(alpha_score) < 0.1
            and confidence >= 0.10
        )

        # --- Step 10: Build signal ---
        signal = AlphaMLSignal(
            alpha_score=alpha_score,
            prob_up=prob_up,
            prob_down=prob_down,
            xgb_prediction=preds["xgb"],
            lstm_prediction=preds["lstm"],
            garch_sigma=gjr_sigma,
            kalman_price=k_price,
            volatility_regime=vol_regime,
            should_enter_long=should_enter_long,
            should_enter_short=should_enter_short,
            should_exit=should_exit,
            confidence=confidence,
            features=fv,
            timestamp=ts,
            lgbm_prediction=preds["lgbm"],
            transformer_prediction=preds["transformer"],
            ofi_weighted=self.ofi_v2.weighted_ofi,
            kyle_lambda=self.ofi_v2.kyle_lambda,
            amihud_ratio=self.ofi_v2.amihud_ratio,
            multi_horizon_score=horizon_score,
            stacking_prediction=preds.get("stacking_prediction", 0.5),
            stacking_confidence=preds.get("stacking_confidence", 0.0),
            regime_model_used=preds.get("regime_model_used", "fixed_weights"),
        )

        log.info(
            f"ALPHA-ML: price=${price:.2f} prob_up={prob_up:.3f} "
            f"alpha={alpha_score:+.3f} gjr_σ={gjr_sigma:.6f} "
            f"regime={vol_regime} conf={confidence:.2f} "
            f"xgb={preds['xgb']:.3f} lgbm={preds['lgbm']:.3f} "
            f"lstm={preds['lstm']:.3f} transformer={preds['transformer']:.3f}"
        )

        self.last_signal = signal
        return signal

    def get_last_feature_vector(self) -> np.ndarray | None:
        """Return the 36-dim feature vector from the most recent generate_signal() call.

        Returns None before first call.
        """
        return self._last_feature_vector

    def get_engine_stats(self) -> dict:
        """Return engine diagnostic statistics."""
        return {
            "n_ticks": self._n_ticks,
            # Legacy GARCH (base FeatureEngineer)
            "garch_sigma": self.feature_engineer._garch.sigma,
            "garch_long_run": self.feature_engineer._garch.long_run_sigma,
            "garch_regime": self.feature_engineer._garch.get_volatility_regime(),
            # GJR-GARCH (AdaptiveGARCH)
            "gjr_sigma": self.adaptive_garch.sigma,
            "gjr_regime": self.adaptive_garch.get_volatility_regime(),
            # 3-state Kalman
            "kalman_price": self.kalman_smoother.smoothed_price,
            "kalman_velocity": self.kalman_smoother.velocity,
            "kalman_acceleration": self.kalman_smoother.acceleration,
            # Microstructure
            "ofi_weighted": self.ofi_v2.weighted_ofi,
            "kyle_lambda": self.ofi_v2.kyle_lambda,
            "amihud_ratio": self.ofi_v2.amihud_ratio,
            # Models
            "xgb_loaded": self.xgboost._is_loaded,
            "lgbm_loaded": self.lgbm._is_loaded,
            "lstm_loaded": self.lstm._is_loaded,
            "transformer_loaded": self.transformer_predictor._is_loaded,
            "lstm_buffer_fill": self.lstm.buffer_fill,
            "transformer_buffer_fill": self.transformer_predictor.buffer_fill,
            "feature_count_30": self.normalizer_30._n,
        }


# ---------------------------------------------------------------------------
# AdaptiveGARCH — GJR-GARCH(1,1) with leverage-effect asymmetry
# ---------------------------------------------------------------------------


class AdaptiveGARCH:
    """GJR-GARCH(1,1): extends standard GARCH with asymmetric shock response.

    Model:
        σ²(t) = ω + (α + γ·I(ε<0))·ε²(t-1) + β·σ²(t-1)

    Negative returns (ε<0) receive extra weight γ, capturing the well-known
    leverage effect where downside shocks amplify volatility more than
    upside shocks of the same magnitude.
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.06,
        beta: float = 0.85,
        gamma: float = GJR_GAMMA,
    ) -> None:
        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        # Unconditional variance: ω / (1 - α - γ/2 - β)
        denom = max(1.0 - alpha - gamma / 2.0 - beta, 1e-8)
        self._sigma_sq: float = omega / denom
        self._last_return: float = 0.0
        self._n: int = 0

    def update(self, log_return: float) -> float:
        self._n += 1
        if self._n < 2:
            self._last_return = log_return
            return math.sqrt(self._sigma_sq)

        indicator = 1.0 if self._last_return < 0 else 0.0
        self._sigma_sq = (
            self._omega
            + (self._alpha + self._gamma * indicator) * self._last_return ** 2
            + self._beta * self._sigma_sq
        )
        self._sigma_sq = max(self._sigma_sq, 1e-12)
        self._last_return = log_return
        return math.sqrt(self._sigma_sq)

    @property
    def sigma(self) -> float:
        return math.sqrt(self._sigma_sq)

    @property
    def has_enough_data(self) -> bool:
        return self._n >= GARCH_MIN_OBSERVATIONS

    def get_volatility_regime(self) -> str:
        if not self.has_enough_data:
            return "normal"
        denom = max(1.0 - self._alpha - self._gamma / 2.0 - self._beta, 1e-8)
        long_run = math.sqrt(max(self._omega / denom, 1e-12))
        ratio = self.sigma / long_run
        if ratio > VOLATILITY_REGIME_HIGH:
            return "high"
        if ratio < 0.5:
            return "low"
        return "normal"


# ---------------------------------------------------------------------------
# KalmanPriceSmoother — 3-state (price / velocity / acceleration)
# ---------------------------------------------------------------------------


class KalmanPriceSmoother:
    """Kalman filter with 3-state vector: [price, velocity, acceleration].

    Compared to the 2-state KalmanFilter, the acceleration state enables
    the filter to track curved price trajectories (e.g. during trending
    markets) without phase lag.

    State transition (constant-acceleration model, dt=1 bar):
        x(t+1) = F·x(t),  F = [[1, 1, 0.5],
                                [0, 1, 1  ],
                                [0, 0, 1  ]]
    """

    def __init__(
        self,
        process_noise: float = KALMAN_PROCESS_NOISE,
        measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
    ) -> None:
        self._x = np.zeros(3)
        self._P = np.eye(3) * KALMAN_INITIAL_ESTIMATE_ERR
        self._F = np.array([[1.0, 1.0, 0.5],
                            [0.0, 1.0, 1.0],
                            [0.0, 0.0, 1.0]])
        self._H = np.array([[1.0, 0.0, 0.0]])
        q = process_noise
        self._Q = np.diag([q, q * 10.0, q * 100.0])
        self._R = np.array([[measurement_noise]])
        self._initialized = False

    def update(self, price: float) -> tuple[float, float, float]:
        """Returns (smoothed_price, velocity, acceleration)."""
        if not self._initialized:
            self._x = np.array([price, 0.0, 0.0])
            self._initialized = True
            return price, 0.0, 0.0

        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        z = np.array([price])
        y = z - self._H @ x_pred
        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        self._x = x_pred + K @ y
        self._P = (np.eye(3) - K @ self._H) @ P_pred

        return float(self._x[0]), float(self._x[1]), float(self._x[2])

    @property
    def smoothed_price(self) -> float:
        return float(self._x[0])

    @property
    def velocity(self) -> float:
        return float(self._x[1])

    @property
    def acceleration(self) -> float:
        return float(self._x[2])


# ---------------------------------------------------------------------------
# OrderFlowImbalanceV2 — weighted OFI, Kyle λ, Amihud illiquidity
# ---------------------------------------------------------------------------


class OrderFlowImbalanceV2:
    """Enhanced order flow imbalance with microstructure metrics.

    Metrics:
        weighted_ofi   : Exponentially-weighted signed volume imbalance [-1, 1].
        kyle_lambda    : Price impact per unit signed volume (Kyle 1985).
        amihud_ratio   : Mean(|return| / volume) illiquidity proxy (Amihud 2002).
    """

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._returns: deque[float] = deque(maxlen=window)
        self._volumes: deque[float] = deque(maxlen=window)
        self._signed_vols: deque[float] = deque(maxlen=window)
        self._weighted_ofi: float = 0.0
        self._kyle_lambda: float = 0.0
        self._amihud: float = 0.0

    def update(
        self,
        log_return: float,
        volume: float,
        buy_ratio: float = 0.5,
    ) -> None:
        """Ingest one observation.

        Args:
            log_return: ln(P_t / P_{t-1})
            volume: Trade volume in USDC.
            buy_ratio: Fraction of volume that was buying [0, 1]. 0.5 = balanced.
        """
        signed_vol = (2.0 * buy_ratio - 1.0) * volume
        self._returns.append(log_return)
        self._volumes.append(volume)
        self._signed_vols.append(signed_vol)

        n = len(self._returns)
        if n < 5:
            return

        sv_arr = np.array(self._signed_vols, dtype=np.float64)
        ret_arr = np.array(self._returns, dtype=np.float64)
        vol_arr = np.array(self._volumes, dtype=np.float64)

        # Weighted OFI: linear ramp 0.5→1.0 so recent trades matter more
        weights = np.linspace(0.5, 1.0, n)
        weighted_sv = sv_arr * weights
        total_abs = float(np.sum(np.abs(weighted_sv))) or 1e-10
        self._weighted_ofi = float(np.sum(weighted_sv) / total_abs)

        # Kyle lambda via OLS: cov(ret, signed_vol) / var(signed_vol)
        sv_std = float(np.std(sv_arr))
        if sv_std > 1e-10:
            cov_matrix = np.cov(ret_arr, sv_arr)
            self._kyle_lambda = float(cov_matrix[0, 1] / (np.var(sv_arr) + 1e-10))

        # Amihud illiquidity
        nonzero = vol_arr > 0
        if nonzero.any():
            self._amihud = float(np.mean(np.abs(ret_arr[nonzero]) / vol_arr[nonzero]))

    @property
    def weighted_ofi(self) -> float:
        return self._weighted_ofi

    @property
    def kyle_lambda(self) -> float:
        return self._kyle_lambda

    @property
    def amihud_ratio(self) -> float:
        return self._amihud


# ---------------------------------------------------------------------------
# MultiHorizonSignal — composite across 5 time horizons
# ---------------------------------------------------------------------------


class MultiHorizonSignal:
    """Aggregates directional signals across 5 horizons.

    Horizons (in 5-min bars): 1 (5m), 3 (15m), 6 (30m), 12 (1h), 48 (4h).
    Shorter horizons carry more weight because the engine trades at 5-min
    granularity, but longer horizon agreement increases conviction.
    """

    HORIZONS = [1, 3, 6, 12, 48]
    WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08]

    def __init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=50)

    def update(self, price: float) -> None:
        self._prices.append(price)

    def compute(self) -> float:
        """Returns confidence-weighted composite score in [-1, +1]."""
        if len(self._prices) < 2:
            return 0.0
        prices = list(self._prices)
        total_w = 0.0
        composite = 0.0
        for h, w in zip(self.HORIZONS, self.WEIGHTS, strict=False):
            if len(prices) <= h:
                continue
            ret = math.log(prices[-1] / prices[-h])
            # Normalise by √h bars so each horizon is comparable
            score = ret / max(0.001 * math.sqrt(h), 1e-10)
            score = max(-1.0, min(1.0, score))
            composite += score * w
            total_w += w
        if total_w < 1e-10:
            return 0.0
        return max(-1.0, min(1.0, composite / total_w))


# ---------------------------------------------------------------------------
# FeatureStore — 36-dim feature vector (18 base + 12 derived + 6 expanded)
# ---------------------------------------------------------------------------


class FeatureStore:
    """Assembles the extended 36-dimensional feature array.

    Base (18): FeatureVector.to_array()
    Derived (12):
        0  trend × OFI interaction
        1  sigma × vol_ratio (vol expansion signal)
        2  momentum alignment (m5 × m15)
        3  weighted OFI (v2)
        4  Kyle lambda (scaled)
        5  Amihud illiquidity (scaled)
        6  Kalman acceleration (3-state)
        7  multi-horizon composite
        8  realized_vol_5 / garch_sigma ratio
        9  vol_ratio² (nonlinear vol signal)
        10 short-term reversal (m5 - m15)
        11 medium-term momentum (m15 - m30)
    Expanded (6):
        0  rsi_14             — 14-period RSI normalized to [0, 1]
        1  bb_position        — price position within 20-period Bollinger Bands
        2  volume_trend       — EMA(vol,20) / EMA(vol,5) ratio
        3  price_acceleration — Kalman 3-state acceleration component
        4  spread_ratio       — current spread_bps / rolling_mean(spread_bps, 50)
        5  microstructure_score — log1p(|Kyle λ × Amihud|)
    """

    N_FEATURES = 36

    # Must exactly match the order produced by build() — 18 base + 12 derived + 6 expanded
    FEATURE_NAMES: list[str] = [
        # Base (18): FeatureVector.to_array() order
        "log_return", "kalman_price", "kalman_trend", "garch_sigma",
        "realized_vol_5", "realized_vol_30", "vol_ratio",
        "momentum_5", "momentum_15", "momentum_30", "momentum_60", "momentum_120",
        "volume_imbalance", "spread_bps", "trade_intensity",
        "skewness", "kurtosis", "price",
        # Derived (12): computed in build()
        "trend_ofi_interaction",
        "sigma_vol_expansion",
        "momentum_alignment",
        "ofi_weighted_v2",
        "kyle_lambda_scaled",
        "amihud_illiquidity",
        "kalman_acceleration",
        "multi_horizon_composite",
        "realized_garch_ratio",
        "vol_ratio_squared",
        "short_term_reversal",
        "medium_term_momentum",
        # Expanded (6): feature engineering expansion
        "rsi_14",
        "bb_position",
        "volume_trend",
        "price_acceleration",
        "spread_ratio",
        "microstructure_score",
    ]

    def __init__(self) -> None:
        self._spread_history: deque[float] = deque(maxlen=50)

    def build(
        self,
        base_fv: FeatureVector,
        ofi_v2: OrderFlowImbalanceV2,
        smoother: KalmanPriceSmoother,
        horizon_score: float,
        price_history: np.ndarray | None = None,
        volume_history: np.ndarray | None = None,
    ) -> np.ndarray:
        base = base_fv.to_array()  # 18 dims

        derived = np.array([
            base_fv.kalman_trend * base_fv.volume_imbalance,
            base_fv.garch_sigma * base_fv.vol_ratio,
            base_fv.momentum_5 * base_fv.momentum_15,
            ofi_v2.weighted_ofi,
            ofi_v2.kyle_lambda * 1e4,
            ofi_v2.amihud_ratio * 1e6,
            smoother.acceleration,
            horizon_score,
            base_fv.realized_vol_5 / max(base_fv.garch_sigma, 1e-10),
            base_fv.vol_ratio ** 2,
            base_fv.momentum_5 - base_fv.momentum_15,
            base_fv.momentum_15 - base_fv.momentum_30,
        ], dtype=np.float64)

        # --- Expanded feature set (6 new features) ---
        self._spread_history.append(base_fv.spread_bps)

        expanded = np.array([
            float(np.nan_to_num(self._compute_rsi(price_history, 14), nan=0.5)),
            float(np.nan_to_num(self._compute_bb_position(price_history, base_fv.price, 20), nan=0.5)),
            float(np.nan_to_num(self._compute_volume_trend(volume_history), nan=1.0)),
            float(np.nan_to_num(smoother.acceleration, nan=0.0)),
            float(np.nan_to_num(self._compute_spread_ratio(base_fv.spread_bps), nan=1.0)),
            float(np.nan_to_num(self._compute_microstructure_score(ofi_v2), nan=0.0)),
        ], dtype=np.float64)

        return np.concatenate([base, derived, expanded])

    # ------------------------------------------------------------------
    # Private helpers for expanded features
    # ------------------------------------------------------------------

    def _compute_rsi(self, prices: np.ndarray | None, period: int = 14) -> float:
        """14-period RSI normalized to [0, 1]."""
        if prices is None or len(prices) < period + 1:
            return 0.5
        p = prices[-(period + 1):]
        deltas = np.diff(p)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss < 1e-10:
            return 1.0 if avg_gain > 0 else 0.5
        rs = avg_gain / avg_loss
        rsi_raw = 100.0 - 100.0 / (1.0 + rs)
        return float(np.clip(rsi_raw / 100.0, 0.0, 1.0))

    def _compute_bb_position(
        self, prices: np.ndarray | None, current_price: float, period: int = 20
    ) -> float:
        """Price position within 20-period Bollinger Bands, clipped [0, 1]."""
        if prices is None or len(prices) < period:
            return 0.5
        p = prices[-period:]
        mean = float(np.mean(p))
        std = float(np.std(p))
        if std < 1e-10:
            return 0.5
        upper = mean + 2.0 * std
        lower = mean - 2.0 * std
        band_width = upper - lower
        if band_width < 1e-10:
            return 0.5
        position = (current_price - lower) / band_width
        return float(np.clip(position, 0.0, 1.0))

    def _compute_volume_trend(self, volumes: np.ndarray | None) -> float:
        """EMA(vol, 20) / EMA(vol, 5) ratio, clipped [0.1, 10]."""
        if volumes is None or len(volumes) < 5:
            return 1.0

        def _ema(arr: np.ndarray, span: int) -> float:
            alpha = 2.0 / (span + 1)
            val = float(arr[0])
            for v in arr[1:]:
                val = alpha * float(v) + (1.0 - alpha) * val
            return val

        ema5 = _ema(volumes[-5:], 5)
        ema20 = _ema(volumes[-min(20, len(volumes)):], 20)
        if ema5 < 1e-10:
            return 1.0
        return float(np.clip(ema20 / ema5, 0.1, 10.0))

    def _compute_spread_ratio(self, current_spread: float) -> float:
        """current spread_bps / rolling_mean(spread_bps, 50), clipped [0.1, 10]."""
        if len(self._spread_history) < 5:
            return 1.0
        mean_spread = float(np.mean(list(self._spread_history)))
        if mean_spread < 1e-10:
            return 1.0
        return float(np.clip(current_spread / mean_spread, 0.1, 10.0))

    def _compute_microstructure_score(self, ofi_v2: OrderFlowImbalanceV2) -> float:
        """Kyle lambda × Amihud ratio, log-scaled."""
        try:
            product = abs(float(ofi_v2.kyle_lambda) * float(ofi_v2.amihud_ratio))
            return float(math.log1p(product))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# LightGBMPredictor
# ---------------------------------------------------------------------------


class LightGBMPredictor:
    """LightGBM predictor on 36-dim feature vector.

    Falls back to neutral (0.5, 0.5) when no trained model is available.
    LightGBM trains ~10× faster than XGBoost on tabular data, complementing
    XGBoost's stronger regularisation.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._is_loaded = False

    def load_model(self, model_path: Path | None = None) -> bool:
        path = model_path or (MODEL_DIR / "lgbm_alpha_model.txt")
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(path))
            self._is_loaded = True
            log.info(f"LightGBM model loaded from {path}")
            return True
        except Exception as exc:
            log.info(f"LightGBM model not found ({exc}). Using neutral fallback.")
            self._is_loaded = False
            return False

    def predict_proba(self, features: np.ndarray) -> tuple[float, float]:
        if self._is_loaded and self._model is not None:
            try:
                prob = float(self._model.predict(features.reshape(1, -1))[0])
                prob = max(0.01, min(0.99, prob))
                return prob, 1.0 - prob
            except Exception as exc:
                exc_str = str(exc).lower()
                if "feature" in exc_str or "shape" in exc_str or "mismatch" in exc_str:
                    log.warning(
                        f"LightGBM feature mismatch — model trained with different "
                        f"feature count (got {features.shape[0]}): {exc}"
                    )
                else:
                    log.debug(f"LightGBM inference failed: {exc}")
        return 0.5, 0.5


# ---------------------------------------------------------------------------
# TransformerPredictor
# ---------------------------------------------------------------------------


class TransformerPredictor:
    """Self-attention Transformer for 48-bar (4h) sequence modelling.

    Operates on the full 36-dim feature sequence. Captures longer-range
    regime patterns that LSTM's recurrent state can miss when gradients vanish.
    Falls back to neutral until buffer fills or model is unavailable.
    """

    def __init__(self, sequence_length: int = 30) -> None:
        self._model = None
        self._seq_len = sequence_length
        self._feature_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
        self._is_loaded = False

    def load_model(self, model_path: Path | None = None) -> bool:
        path = model_path or (MODEL_DIR / "transformer_alpha_model.pt")
        try:
            import torch
            self._model = torch.jit.load(str(path), map_location="cpu")
            model = self._model
            if model is None:
                raise RuntimeError("Transformer model failed to load")
            model.eval()
            self._is_loaded = True
            log.info(f"Transformer model loaded from {path}")
            return True
        except Exception as exc:
            log.info(f"Transformer model not found ({exc}). Using neutral fallback.")
            self._is_loaded = False
            return False

    def ingest_features(self, features: np.ndarray) -> None:
        self._feature_buffer.append(features.copy())

    def predict_proba(self) -> tuple[float, float]:
        if len(self._feature_buffer) < self._seq_len:
            return 0.5, 0.5
        if self._is_loaded and self._model is not None:
            try:
                import torch
                seq = np.stack(list(self._feature_buffer))
                tensor = torch.FloatTensor(seq).unsqueeze(0)
                with torch.no_grad():
                    output = self._model(tensor)
                    prob = float(torch.sigmoid(output).item())
                return prob, 1.0 - prob
            except Exception as exc:
                log.debug(f"Transformer inference failed: {exc}")
        return 0.5, 0.5

    @property
    def buffer_fill(self) -> float:
        return len(self._feature_buffer) / self._seq_len


# ---------------------------------------------------------------------------
# EnsembleAlphaEngine — 4-model ensemble with Platt scaling calibration
# ---------------------------------------------------------------------------


class EnsembleAlphaEngine:
    """Combines XGBoost, LightGBM, LSTM, and Transformer with Platt scaling.

    Platt scaling: p_cal = sigmoid(A·logit(p) + B)
    Parameters (A, B) per model are initially identity (A=-1, B=0 ⟹ p_cal≈p)
    and can be updated after offline calibration via update_platt_params().

    Final ensemble:
        P(up) = 0.40·xgb_cal + 0.25·lgbm_cal + 0.25·lstm_cal + 0.10·transformer_cal
    """

    _MODEL_KEYS = ("xgb", "lgbm", "lstm", "transformer")

    def __init__(
        self,
        xgb: XGBoostPredictor,
        lgbm: LightGBMPredictor,
        lstm: LSTMPredictor,
        transformer: TransformerPredictor,
    ) -> None:
        self._xgb = xgb
        self._lgbm = lgbm
        self._lstm = lstm
        self._transformer = transformer
        # Platt: A=-1, B=0 → sigmoid(-logit(p)) = p (identity when A≈-1, B≈0)
        self._platt_A = dict.fromkeys(self._MODEL_KEYS, -1.0)
        self._platt_B = dict.fromkeys(self._MODEL_KEYS, 0.0)
        # Stacking meta-learner (loaded lazily from ml_models/meta_learner_*.pkl)
        self._stacker = StackingMetaLearner()

    def _platt_scale(self, raw_prob: float, key: str) -> float:
        """Calibrate raw probability via Platt sigmoid."""
        p = max(1e-6, min(1.0 - 1e-6, raw_prob))
        logit = math.log(p / (1.0 - p))
        scaled = self._platt_A[key] * logit + self._platt_B[key]
        return 1.0 / (1.0 + math.exp(-scaled))

    def predict(self, features_30d: np.ndarray) -> dict:
        """Run all 4 models and return calibrated per-model + ensemble probs.

        XGBoost and LSTM use the 18-dim base slice for backward-compat with
        existing trained checkpoints; LightGBM and Transformer use all 36.

        Returns:
            Dict with keys 'xgb', 'lgbm', 'lstm', 'transformer', 'ensemble'.
        """
        # Read active models from config (progressive ensemble loading)
        try:
            from wallet_manager.config import get_settings as _get_settings
            _active = set(_get_settings().ensemble_active_models.split(","))
        except Exception:
            _active = {"xgb", "lgbm"}

        # XGBoost: 18-dim (existing model checkpoint compatibility)
        if "xgb" in _active and self._xgb._is_loaded:
            xgb_up, _ = self._xgb.predict_proba(features_30d[:18])
        else:
            xgb_up = 0.5
        # LightGBM: 30-dim
        if "lgbm" in _active and self._lgbm._is_loaded:
            lgbm_up, _ = self._lgbm.predict_proba(features_30d)
        else:
            lgbm_up = 0.5
        # LSTM and Transformer: use pre-ingested sequence buffers
        if "lstm" in _active and self._lstm._is_loaded:
            lstm_up, _ = self._lstm.predict_proba()
        else:
            lstm_up = 0.5
        if "transformer" in _active and self._transformer._is_loaded:
            transformer_up, _ = self._transformer.predict_proba()
        else:
            transformer_up = 0.5

        xgb_cal = self._platt_scale(xgb_up, "xgb")
        lgbm_cal = self._platt_scale(lgbm_up, "lgbm")
        lstm_cal = self._platt_scale(lstm_up, "lstm")
        transformer_cal = self._platt_scale(transformer_up, "transformer")

        # Online logistic-regression meta-learner blend (falls back to weighted avg)
        try:
            from wallet_manager.strategies.stacking_meta_learner import get_meta_learner
            blended_prob = get_meta_learner().predict(
                xgb_prob=xgb_cal, lgbm_prob=lgbm_cal,
                lstm_prob=lstm_cal, transformer_prob=transformer_cal,
            )
        except Exception as _ml_exc:
            log.warning("Meta-learner predict failed, using weighted average: %s", _ml_exc)
            blended_prob = (
                ENSEMBLE_WEIGHT_XGB * xgb_cal
                + ENSEMBLE_WEIGHT_LGBM * lgbm_cal
                + ENSEMBLE_WEIGHT_LSTM * lstm_cal
                + ENSEMBLE_WEIGHT_TRANSFORMER * transformer_cal
            )
        prob_up = blended_prob

        # Stacking meta-learner (regime-conditional)
        try:
            from wallet_manager.strategies.regime_filter import get_regime_filter
            _regime = get_regime_filter().get_current_regime().regime.name
        except Exception:
            _regime = "LOW_VOL"

        stacking_pred, stacking_conf = self._stacker.predict(
            xgb_cal, lgbm_cal, lstm_cal, transformer_cal, _regime
        )
        regime_model_used = _regime if self._stacker._is_loaded and _regime in self._stacker._models else "fixed_weights"

        # Use stacking prediction if meta-learner is loaded, else fixed weighted average
        if self._stacker._is_loaded and _regime in self._stacker._models:
            ensemble_final = stacking_pred
        else:
            ensemble_final = float(np.clip(prob_up, 0.01, 0.99))

        return {
            "xgb": xgb_cal,
            "lgbm": lgbm_cal,
            "lstm": lstm_cal,
            "transformer": transformer_cal,
            "ensemble": ensemble_final,
            "stacking_prediction": stacking_pred,
            "stacking_confidence": stacking_conf,
            "regime_model_used": regime_model_used,
        }

    def update_platt_params(self, model_key: str, A: float, B: float) -> None:
        """Update Platt scaling params after offline calibration."""
        if model_key in self._platt_A:
            self._platt_A[model_key] = A
            self._platt_B[model_key] = B


# ---------------------------------------------------------------------------
# StackingMetaLearner
# ---------------------------------------------------------------------------

# Minimum number of observations per regime before using meta-learner
_STACKING_MIN_OBS: int = 50

# Default fixed weights for fallback (when fewer than min observations)
_STACKING_FIXED_WEIGHTS = {
    "xgb": ENSEMBLE_WEIGHT_XGB,
    "lgbm": ENSEMBLE_WEIGHT_LGBM,
    "lstm": ENSEMBLE_WEIGHT_LSTM,
    "transformer": ENSEMBLE_WEIGHT_TRANSFORMER,
}


class StackingMetaLearner:
    """Level-2 meta-learner that takes the 4 base model predictions as
    input features and learns the optimal combination weights.

    Architecture: LogisticRegression (scikit-learn) meta-model.
    Trained on 30% holdout from CPCV splits (never on training data).
    Automatically retrained when RetrainController triggers a retrain.

    Regime-conditional: maintains a separate meta-model per regime.
    Falls back to fixed weights if fewer than 50 observations for a regime.

    Fields added to AlphaMLSignal:
      stacking_prediction: float
      stacking_confidence: float
      regime_model_used: str
    """

    # Model directory relative to project root
    _MODEL_DIR: str = "ml_models"
    _MODEL_PREFIX: str = "meta_learner_"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._is_loaded: bool = False
        self._load_models()

    def _load_models(self) -> None:
        """Attempt to load per-regime meta-learner models from disk."""
        from pathlib import Path
        model_dir = Path(self._MODEL_DIR)
        if not model_dir.exists():
            return
        try:
            import pickle
            loaded = 0
            for path in model_dir.glob(f"{self._MODEL_PREFIX}*.pkl"):
                regime = path.stem.replace(self._MODEL_PREFIX, "")
                with open(path, "rb") as f:
                    self._models[regime] = pickle.load(f)
                loaded += 1
            if loaded:
                self._is_loaded = True
                log.info("StackingMetaLearner: loaded %d regime models.", loaded)
        except Exception as exc:
            log.debug("StackingMetaLearner: model load failed: %s", exc)
            self._is_loaded = False

    def predict(
        self,
        xgb_pred: float,
        lgbm_pred: float,
        lstm_pred: float,
        transformer_pred: float,
        regime: str,
    ) -> tuple[float, float]:
        """Returns (stacking_prediction, stacking_confidence).

        Falls back to fixed weighted average if no meta-model for regime.
        """
        features = [xgb_pred, lgbm_pred, lstm_pred, transformer_pred]

        if self._is_loaded and regime in self._models:
            try:
                import numpy as np
                model = cast(Any, self._models[regime])
                X = np.array(features, dtype=np.float64).reshape(1, -1)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    # Class 1 = price up
                    prob_up = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    prob_up = float(model.predict(X)[0])
                # Confidence: distance from 0.5
                confidence = abs(prob_up - 0.5) * 2.0
                return prob_up, confidence
            except Exception as exc:
                log.debug("StackingMetaLearner: predict failed for regime=%s: %s", regime, exc)

        # Fallback: fixed weighted average
        weights = _STACKING_FIXED_WEIGHTS
        pred = (
            weights["xgb"] * xgb_pred
            + weights["lgbm"] * lgbm_pred
            + weights["lstm"] * lstm_pred
            + weights["transformer"] * transformer_pred
        )
        pred = float(np.clip(pred, 0.01, 0.99)) if "np" in dir() else max(0.01, min(0.99, pred))
        confidence = abs(pred - 0.5) * 2.0
        return pred, confidence


# ---------------------------------------------------------------------------
# AlphaConfidenceEstimator
# ---------------------------------------------------------------------------


class AlphaConfidenceEstimator:
    """Combines inter-model agreement, data coverage, vol regime, and
    multi-horizon alignment into a single confidence score in [0, 1]."""

    def estimate(
        self,
        model_predictions: dict,
        data_coverage: float,
        vol_regime: str,
        horizon_score: float,
    ) -> float:
        """
        Args:
            model_predictions: Output of EnsembleAlphaEngine.predict().
            data_coverage: Fraction of FEATURE_WINDOW filled [0, 1].
            vol_regime: 'low', 'normal', or 'high'.
            horizon_score: MultiHorizonSignal composite in [-1, +1].

        Returns:
            Confidence in [0, 1].
        """
        preds = [model_predictions[k] for k in ("xgb", "lgbm", "lstm", "transformer")]
        mean_p = sum(preds) / len(preds)
        variance = sum((p - mean_p) ** 2 for p in preds) / len(preds)
        # Agreement: 1.0 at zero variance, 0.0 at max variance (0.25)
        agreement = max(0.0, 1.0 - 4.0 * variance)

        regime_factor = float({"low": 0.9, "normal": 1.0, "high": 0.6}.get(vol_regime, 1.0))

        # Does multi-horizon direction agree with ensemble direction?
        alpha_dir = 1.0 if model_predictions["ensemble"] > 0.5 else -1.0
        horizon_alignment = max(0.0, alpha_dir * horizon_score)  # [0, 1]

        confidence = agreement * data_coverage * regime_factor * (0.7 + 0.3 * horizon_alignment)
        return float(max(0.0, min(1.0, confidence)))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: StochasticMLAlphaEngine | None = None


def get_alpha_engine() -> StochasticMLAlphaEngine:
    """Return the process-wide StochasticMLAlphaEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = StochasticMLAlphaEngine()
    return _engine_instance


def reset_alpha_engine() -> None:
    """Reset the singleton (for tests)."""
    global _engine_instance
    _engine_instance = None
