"""
wallet_manager/strategies/alpha_decay_monitor.py

Alpha Decay Monitor — Section 4.7.

Two components:
  1. SignalHalfLifeEstimator — fits an exponential decay model to each
     signal type's autocorrelation and reports the half-life in bars.

  2. CrowdingDetector — detects when the alpha is being "crowded" by
     other market participants via three indicators:
       a. MEV detection: unusually high gas prices before our signal fires
       b. Gas spike correlation: gas and signal appear at same time
       c. Implementation shortfall (IS) trend: IS worsening over time
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# SignalHalfLifeEstimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HalfLifeResult:
    """Result of a half-life estimation for one signal."""

    signal_type: str
    half_life_bars: float      # Estimated half-life in 5-min bars
    decay_rate: float          # λ in e^(-λt)
    autocorr_lag1: float       # Lag-1 autocorrelation (raw)
    n_observations: int
    is_decaying: bool          # True if half-life < DECAY_WARN_BARS


class SignalHalfLifeEstimator:
    """Estimates signal half-life from lag-1 autocorrelation of signal values.

    Decay model: signal(t) = A × e^{-λt}

    From lag-1 AR(1): ρ₁ = e^{-λΔt}  → λ = -ln(ρ₁) / Δt
    Half-life T½ = ln(2) / λ = ln(2) × Δt / (-ln(ρ₁))

    A half-life < DECAY_WARN_BARS (default 12 bars = 1h) indicates the signal
    mean-reverts quickly and should not be held for more than 1 period.
    """

    DECAY_WARN_BARS: int = 12   # 1-hour half-life warning threshold
    WINDOW: int = 100

    def __init__(self) -> None:
        self._histories: dict[str, deque] = {}

    def update(self, signal_type: str, signal_value: float) -> None:
        """Record a new signal observation."""
        if signal_type not in self._histories:
            self._histories[signal_type] = deque(maxlen=self.WINDOW)
        self._histories[signal_type].append(signal_value)

    def estimate(self, signal_type: str) -> HalfLifeResult | None:
        """Estimate the half-life for one signal type.

        Returns None if insufficient data.
        """
        hist = self._histories.get(signal_type)
        if hist is None or len(hist) < 20:
            return None

        arr = np.array(hist, dtype=np.float64)
        n = len(arr)

        # Lag-1 autocorrelation
        mean = float(np.mean(arr))
        demeaned = arr - mean
        autocov_0 = float(np.dot(demeaned, demeaned) / n)
        autocov_1 = float(np.dot(demeaned[1:], demeaned[:-1]) / (n - 1))

        if autocov_0 < 1e-10:
            return HalfLifeResult(
                signal_type=signal_type,
                half_life_bars=float("inf"),
                decay_rate=0.0,
                autocorr_lag1=0.0,
                n_observations=n,
                is_decaying=False,
            )

        rho1 = autocov_1 / autocov_0
        rho1 = max(-0.9999, min(0.9999, rho1))

        if rho1 <= 0:
            # Negative autocorrelation → signal oscillates, immediate decay
            return HalfLifeResult(
                signal_type=signal_type,
                half_life_bars=0.5,
                decay_rate=float("inf"),
                autocorr_lag1=rho1,
                n_observations=n,
                is_decaying=True,
            )

        lam = -math.log(rho1)  # decay rate (λ), Δt=1 bar
        half_life = math.log(2.0) / lam if lam > 0 else float("inf")

        return HalfLifeResult(
            signal_type=signal_type,
            half_life_bars=half_life,
            decay_rate=lam,
            autocorr_lag1=rho1,
            n_observations=n,
            is_decaying=half_life < self.DECAY_WARN_BARS,
        )

    def get_all_half_lives(self) -> dict[str, HalfLifeResult | None]:
        return {name: self.estimate(name) for name in self._histories}


# ---------------------------------------------------------------------------
# CrowdingDetector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrowdingSignal:
    """Crowding assessment snapshot."""

    mev_pressure: float       # [0, 1]: elevated gas before our signals
    gas_correlation: float    # [-1, 1]: gas-signal co-movement
    is_trend: float           # [0, 1]: deteriorating IS trend
    composite: float          # Weighted composite [0, 1]
    is_crowded: bool          # True if composite > CROWDED_THRESHOLD


class CrowdingDetector:
    """Detects alpha crowding via three independent signals.

    MEV pressure:
        Measures whether gas prices spike shortly before our signals fire.
        High gas before signal → bots anticipating the same move.

    Gas-signal correlation:
        Positive correlation between gas and alpha direction → front-running.
        If gas rises when our alpha is bullish, someone else sees our signal.

    IS trend:
        Persistent worsening of implementation shortfall over rolling window
        → execution is getting costlier → crowding effect.
    """

    WINDOW: int = 50
    CROWDED_THRESHOLD: float = 0.6

    # Weights for composite score
    W_MEV: float = 0.40
    W_GAS_CORR: float = 0.35
    W_IS: float = 0.25

    def __init__(self) -> None:
        self._gas_prices: deque[float] = deque(maxlen=self.WINDOW)
        self._signal_values: deque[float] = deque(maxlen=self.WINDOW)
        self._is_values: deque[float] = deque(maxlen=self.WINDOW)
        self._pre_signal_gas: deque[float] = deque(maxlen=self.WINDOW)
        self._last_signal_gas: float = 0.0

    def update_gas(self, gas_gwei: float) -> None:
        """Record current gas price (call every bar)."""
        self._gas_prices.append(gas_gwei)

    def update_signal(self, alpha: float, gas_gwei: float) -> None:
        """Record that a signal fired with this alpha value.

        Args:
            alpha: Alpha signal value [-1, +1].
            gas_gwei: Gas price at signal time.
        """
        self._signal_values.append(alpha)
        self._last_signal_gas = gas_gwei
        # Record gas at signal time for MEV analysis
        if self._gas_prices:
            median_gas = float(np.median(self._gas_prices))
            gas_ratio = gas_gwei / max(median_gas, 0.001)
            self._pre_signal_gas.append(gas_ratio)

    def update_is(self, is_value: float) -> None:
        """Record an implementation shortfall observation."""
        self._is_values.append(is_value)

    def get_crowding_signal(self) -> CrowdingSignal:
        """Assess current crowding level."""
        mev = self._mev_pressure()
        gas_corr = self._gas_signal_correlation()
        is_trend = self._is_trend()

        composite = self.W_MEV * mev + self.W_GAS_CORR * gas_corr + self.W_IS * is_trend
        composite = max(0.0, min(1.0, composite))

        return CrowdingSignal(
            mev_pressure=mev,
            gas_correlation=gas_corr,
            is_trend=is_trend,
            composite=composite,
            is_crowded=composite >= self.CROWDED_THRESHOLD,
        )

    def _mev_pressure(self) -> float:
        """High gas at signal time vs baseline → MEV bots active."""
        if len(self._pre_signal_gas) < 5:
            return 0.0
        ratios = np.array(self._pre_signal_gas, dtype=np.float64)
        # Mean ratio > 1.5 → elevated gas = MEV pressure
        mean_ratio = float(np.mean(ratios))
        return min(1.0, max(0.0, (mean_ratio - 1.0) / 1.5))

    def _gas_signal_correlation(self) -> float:
        """Pearson correlation between gas prices and signal values."""
        if len(self._gas_prices) < 10 or len(self._signal_values) < 10:
            return 0.0
        n = min(len(self._gas_prices), len(self._signal_values))
        gas = np.array(list(self._gas_prices)[-n:], dtype=np.float64)
        sig = np.array(list(self._signal_values)[-n:], dtype=np.float64)
        if float(np.std(gas)) < 1e-10 or float(np.std(sig)) < 1e-10:
            return 0.0
        corr = float(np.corrcoef(gas, sig)[0, 1])
        # Positive correlation = front-running signal
        return max(0.0, corr)

    def _is_trend(self) -> float:
        """Linear trend in IS values. Rising IS = worsening execution."""
        if len(self._is_values) < 10:
            return 0.0
        vals = np.array(self._is_values, dtype=np.float64)
        x = np.arange(len(vals), dtype=np.float64)
        # Linear regression slope
        slope = float(np.polyfit(x, vals, 1)[0])
        # Scale: 1 bps/period = 0.0001/period → normalized to [0, 1] at 5 bps/period
        return min(1.0, max(0.0, slope / 0.0005))
