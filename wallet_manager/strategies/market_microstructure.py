"""
wallet_manager/strategies/market_microstructure.py

Market Microstructure Analysis — Section 4.6.

Four components:
  1. BidAskSpreadEstimator   — Roll (1984) implicit cost estimator.
  2. MarketImpactModel       — Kyle (1985) + Almgren-Chriss (2001) price impact.
  3. InformationRatioTracker — Per-signal IR with cooldown after degradation.
  4. OptimalExecutionScheduler — Adaptive execution horizon from impact model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# BidAskSpreadEstimator — Roll (1984) implicit spread
# ---------------------------------------------------------------------------


class BidAskSpreadEstimator:
    """Estimates the effective bid-ask spread using the Roll (1984) model.

    Roll's formula uses the covariance of successive price changes:
        s = 2 × √(-Cov(ΔP_t, ΔP_{t-1}))

    This works on transaction prices alone — no quote data needed.
    When the covariance is positive (trending market) the model clamps to 0.

    Attributes:
        spread_bps: Current spread estimate in basis points.
    """

    WINDOW: int = 50

    def __init__(self, window: int = WINDOW) -> None:
        self._window = window
        self._returns: deque[float] = deque(maxlen=window)
        self._spread_bps: float = 0.0
        self._last_price: float = 0.0

    def update(self, price: float) -> float:
        """Ingest a new price observation and return the current spread estimate.

        Returns:
            Spread estimate in basis points.
        """
        if self._last_price > 0:
            self._returns.append(math.log(price / self._last_price))
        self._last_price = price

        if len(self._returns) >= 10:
            self._spread_bps = self._roll_spread()

        return self._spread_bps

    def _roll_spread(self) -> float:
        ret = np.array(self._returns, dtype=np.float64)
        if len(ret) < 4:
            return 0.0
        # Cov(ΔP_t, ΔP_{t-1}) via dot product of lagged returns
        cov = float(np.cov(ret[1:], ret[:-1])[0, 1])
        if cov >= 0:
            return 0.0
        # Spread in log-price units → convert to bps
        spread_log = 2.0 * math.sqrt(-cov)
        return spread_log * 10_000  # bps

    @property
    def spread_bps(self) -> float:
        return self._spread_bps


# ---------------------------------------------------------------------------
# MarketImpactModel — Kyle λ + Almgren-Chriss permanent + temporary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactEstimate:
    """Price impact estimate for a given trade size."""

    kyle_impact_pct: float       # Kyle: λ × signed_volume
    permanent_impact_pct: float  # Almgren-Chriss permanent component
    temporary_impact_pct: float  # Almgren-Chriss temporary (intra-bar)
    total_cost_pct: float        # permanent + temporary
    trade_size_usd: float


class MarketImpactModel:
    """Combined Kyle (1985) and Almgren-Chriss (2001) market impact model.

    Kyle lambda is estimated from past return / signed-volume regressions.
    Almgren-Chriss uses:
        Permanent impact  = η × participation_rate
        Temporary impact  = ε × sign(x) × |x|^0.6   (power-law model)

    Where participation_rate = trade_size / ADV (avg daily volume).
    """

    # Default AC parameters (empirically calibrated for DEX)
    ETA: float = 0.10     # permanent impact coefficient
    EPS: float = 0.05     # temporary impact coefficient
    AC_ALPHA: float = 0.6  # power-law exponent

    ADV_WINDOW: int = 288  # 24h of 5-min bars

    def __init__(self) -> None:
        self._returns: deque[float] = deque(maxlen=50)
        self._signed_vols: deque[float] = deque(maxlen=50)
        self._daily_vols: deque[float] = deque(maxlen=self.ADV_WINDOW)
        self._kyle_lambda: float = 0.0

    def update(self, log_return: float, volume_usd: float, buy_ratio: float = 0.5) -> None:
        """Ingest one bar for parameter estimation."""
        signed_vol = (2.0 * buy_ratio - 1.0) * volume_usd
        self._returns.append(log_return)
        self._signed_vols.append(signed_vol)
        self._daily_vols.append(volume_usd)

        if len(self._returns) >= 10:
            sv = np.array(self._signed_vols, dtype=np.float64)
            sv_std = float(np.std(sv))
            if sv_std > 0:
                ret = np.array(self._returns, dtype=np.float64)
                cov = float(np.cov(ret, sv)[0, 1])
                self._kyle_lambda = cov / (float(np.var(sv)) + 1e-10)

    def estimate_impact(self, trade_size_usd: float, side: str = "buy") -> ImpactEstimate:
        """Estimate the price impact of executing trade_size_usd.

        Args:
            trade_size_usd: Absolute trade size in USD.
            side: 'buy' or 'sell'.

        Returns:
            ImpactEstimate with component breakdown.
        """
        adv = float(np.mean(self._daily_vols)) * 288 if self._daily_vols else 1e6

        participation = trade_size_usd / max(adv, 1.0)

        # Kyle permanent impact
        kyle_impact = abs(self._kyle_lambda) * trade_size_usd
        kyle_pct = kyle_impact / max(trade_size_usd, 1.0)

        # Almgren-Chriss
        permanent = self.ETA * participation
        temporary = self.EPS * (participation ** self.AC_ALPHA)

        return ImpactEstimate(
            kyle_impact_pct=float(np.clip(kyle_pct, 0.0, 0.05)),
            permanent_impact_pct=float(np.clip(permanent, 0.0, 0.05)),
            temporary_impact_pct=float(np.clip(temporary, 0.0, 0.05)),
            total_cost_pct=float(np.clip(permanent + temporary, 0.0, 0.10)),
            trade_size_usd=trade_size_usd,
        )

    @property
    def kyle_lambda(self) -> float:
        return self._kyle_lambda


# ---------------------------------------------------------------------------
# InformationRatioTracker — per-signal IR with cooldown
# ---------------------------------------------------------------------------


class InformationRatioTracker:
    """Tracks Information Ratio (IR) per signal component with cooldown.

    IR = mean(signal_return) / std(signal_return)

    When a component's rolling IR falls below IR_DISABLE_THRESHOLD for
    COOLDOWN_PERIODS consecutive periods, that component is disabled
    and its weight set to zero until IR recovers above IR_REENABLE_THRESHOLD.

    Cooldown prevents the system from abandoning a temporarily underperforming
    signal only to re-enable it right at the bottom.
    """

    IR_WINDOW: int = 50
    IR_DISABLE_THRESHOLD: float = -0.5
    IR_REENABLE_THRESHOLD: float = 0.3
    COOLDOWN_PERIODS: int = 10

    def __init__(self, signal_names: list[str]) -> None:
        self._names = signal_names
        self._returns: dict[str, deque] = {n: deque(maxlen=self.IR_WINDOW) for n in signal_names}
        self._ir: dict[str, float] = dict.fromkeys(signal_names, 0.0)
        self._disabled: dict[str, bool] = dict.fromkeys(signal_names, False)
        self._below_threshold_count: dict[str, int] = dict.fromkeys(signal_names, 0)

    def record(self, signal_name: str, signal_return: float) -> None:
        """Record the realised return for one signal component."""
        if signal_name not in self._names:
            return
        self._returns[signal_name].append(signal_return)
        self._update_ir(signal_name)

    def _update_ir(self, name: str) -> None:
        hist = np.array(self._returns[name], dtype=np.float64)
        if len(hist) < 5:
            return
        std = float(np.std(hist))
        mean = float(np.mean(hist))
        if std < 1e-10:
            self._ir[name] = -1.0 if mean < 0 else 0.0
        else:
            self._ir[name] = mean / std

        ir = self._ir[name]
        if ir < self.IR_DISABLE_THRESHOLD:
            self._below_threshold_count[name] += 1
            if self._below_threshold_count[name] >= self.COOLDOWN_PERIODS:
                if not self._disabled[name]:
                    log.warning(f"IR cooldown triggered for '{name}': IR={ir:.3f}")
                self._disabled[name] = True
        else:
            self._below_threshold_count[name] = 0
            if self._disabled[name] and ir >= self.IR_REENABLE_THRESHOLD:
                log.info(f"IR cooldown lifted for '{name}': IR={ir:.3f}")
                self._disabled[name] = False

    def get_ir(self, name: str) -> float:
        return self._ir.get(name, 0.0)

    def is_disabled(self, name: str) -> bool:
        return self._disabled.get(name, False)

    def get_active_weights(self, base_weights: dict[str, float]) -> dict[str, float]:
        """Return normalised weights with disabled signals zeroed out."""
        active = {n: (w if not self.is_disabled(n) else 0.0) for n, w in base_weights.items()}
        total = sum(active.values())
        if total <= 0:
            return {n: 1.0 / len(base_weights) for n in base_weights}
        return {n: w / total for n, w in active.items()}


# ---------------------------------------------------------------------------
# OptimalExecutionScheduler — adaptive horizon from impact model
# ---------------------------------------------------------------------------


class OptimalExecutionScheduler:
    """Derives the optimal TWAP interval from market impact and urgency.

    Uses a simplified Almgren-Chriss tradeoff:
        Optimal time T* = sqrt(trade_size_usd / (volatility × adv))

    T* is clamped to [1, 20] bars (5 min – 100 min at 5-min granularity).
    Higher urgency compresses T*; low liquidity extends it.
    """

    MIN_BARS: int = 1    # never split into more than 1 bar (instant)
    MAX_BARS: int = 20   # never wait longer than 100 min

    def __init__(self, impact_model: MarketImpactModel) -> None:
        self._impact = impact_model

    def optimal_n_slices(
        self,
        trade_size_usd: float,
        garch_sigma: float,
        urgency: float = 0.5,
    ) -> int:
        """Compute the recommended number of TWAP slices.

        Args:
            trade_size_usd: Size of the trade in USD.
            garch_sigma: Current GARCH conditional volatility (per bar).
            urgency: In [0, 1]. 1 = execute immediately (1 slice).

        Returns:
            Number of slices in [1, MAX_BARS].
        """
        adv = float(np.mean(self._impact._daily_vols)) * 288 if self._impact._daily_vols else 1e6
        participation = trade_size_usd / max(adv, 1.0)

        # Rough AC optimal horizon: sqrt of participation × vol ratio
        if garch_sigma > 0 and participation > 0:
            t_star = math.sqrt(participation / max(garch_sigma, 1e-6))
        else:
            t_star = 1.0

        # Urgency compression: high urgency reduces horizon
        t_adjusted = t_star * (1.0 - urgency * 0.9)

        n = max(self.MIN_BARS, min(self.MAX_BARS, int(round(t_adjusted))))
        log.debug(
            f"OptExec: size=${trade_size_usd:.0f} sigma={garch_sigma:.4f} "
            f"urgency={urgency:.2f} t*={t_star:.1f} → {n} slices"
        )
        return n
