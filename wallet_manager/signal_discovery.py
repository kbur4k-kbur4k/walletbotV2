"""
wallet_manager/signal_discovery.py

Sprint JS-6.2: Signal Discovery Pipeline
==========================================
Systematically tests candidate signals against paper trade history.
Pre-built candidates: RSI(14), RSI(2), Bollinger position, ATR ratio, VWAP deviation.

Workflow:
  1. Compute candidate signal value for each historical cycle
  2. Measure IC(candidate, forward_return)
  3. Measure correlation(candidate, existing_signals)
  4. If IC > 0.03 AND max_corr < 0.70: auto-register in ExperimentLab
  5. Report: "New candidate signals found: X"
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_IC_THRESHOLD = 0.03
_CORR_THRESHOLD = 0.70
_MIN_OBSERVATIONS = 30
_PRICE_WINDOW = 200


@dataclass
class CandidateSignal:
    name: str
    compute: Callable[[list[float]], float]
    description: str
    ic: float = 0.0
    max_corr_with_existing: float = 0.0
    status: str = "UNTESTED"   # UNTESTED | PROMISING | WEAK | REDUNDANT
    observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "ic": round(self.ic, 5),
            "max_corr_existing": round(self.max_corr_with_existing, 4),
            "status": self.status,
            "observations": self.observations,
        }


class SignalDiscoveryPipeline:
    """
    Tests candidate signals using paper trade price history.
    Reports which candidates show IC > 0.03 and are uncorrelated with existing signals.
    """

    def __init__(self) -> None:
        self._price_history: deque[float] = deque(maxlen=_PRICE_WINDOW)
        self._existing_signals: dict[str, deque[float]] = {
            "robust_alpha": deque(maxlen=_PRICE_WINDOW),
            "confidence": deque(maxlen=_PRICE_WINDOW),
        }
        self._forward_returns: deque[float] = deque(maxlen=_PRICE_WINDOW)
        self._last_run: float = 0.0
        self._results: list[CandidateSignal] = []
        self._candidates: list[CandidateSignal] = self._build_candidates()

    def _build_candidates(self) -> list[CandidateSignal]:
        return [
            CandidateSignal(
                name="rsi_14",
                compute=lambda prices: self._rsi(prices, 14),
                description="RSI(14) — momentum oscillator, mean-reverting signal",
            ),
            CandidateSignal(
                name="rsi_2",
                compute=lambda prices: self._rsi(prices, 2),
                description="RSI(2) — very short-term mean reversion, high sensitivity",
            ),
            CandidateSignal(
                name="bollinger_position",
                compute=lambda prices: self._bollinger_position(prices, 20),
                description="Bollinger band position [-1,+1] — where is price relative to 20-period band?",
            ),
            CandidateSignal(
                name="atr_ratio",
                compute=lambda prices: self._atr_ratio(prices, 14),
                description="ATR(14) / ATR(50) ratio — current volatility vs baseline volatility",
            ),
            CandidateSignal(
                name="vwap_deviation",
                compute=lambda prices: self._vwap_deviation(prices, 20),
                description="Price deviation from 20-period VWAP proxy — institutional entry signal",
            ),
            CandidateSignal(
                name="momentum_3",
                compute=lambda prices: self._momentum(prices, 3),
                description="3-period price momentum — short-term trend continuation",
            ),
            CandidateSignal(
                name="momentum_10",
                compute=lambda prices: self._momentum(prices, 10),
                description="10-period price momentum — medium-term trend",
            ),
        ]

    def record_cycle(
        self,
        price: float,
        robust_alpha: float,
        confidence: float,
    ) -> None:
        """Feed price and existing signal values each cycle."""
        self._price_history.append(price)
        self._existing_signals["robust_alpha"].append(robust_alpha)
        self._existing_signals["confidence"].append(confidence)
        prices = list(self._price_history)
        if len(prices) >= 2:
            ret = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0.0
            self._forward_returns.append(ret)

    async def run_discovery(self) -> list[CandidateSignal]:
        """Run full discovery pipeline. Returns list of evaluated CandidateSignals."""
        prices = list(self._price_history)
        if len(prices) < _MIN_OBSERVATIONS:
            log.info(f"SignalDiscovery: insufficient data ({len(prices)} < {_MIN_OBSERVATIONS})")
            return []

        fwd_returns = list(self._forward_returns)
        n = min(len(prices), len(fwd_returns))
        if n < _MIN_OBSERVATIONS:
            return []

        existing_vecs: dict[str, list[float]] = {
            k: list(v)[-n:] for k, v in self._existing_signals.items()
        }

        for candidate in self._candidates:
            signal_vals: list[float] = []
            for i in range(n):
                window = prices[max(0, i - 50): i + 1]
                try:
                    val = candidate.compute(window)
                    signal_vals.append(float(val) if math.isfinite(val) else 0.0)
                except Exception:
                    signal_vals.append(0.0)

            if len(signal_vals) < _MIN_OBSERVATIONS:
                candidate.status = "UNTESTED"
                continue

            rets_slice = fwd_returns[-len(signal_vals):]
            candidate.ic = _pearson(signal_vals, rets_slice)
            candidate.observations = len(signal_vals)

            max_corr = max(
                abs(_pearson(signal_vals, existing_vecs.get(k, [])[-len(signal_vals):]))
                for k in existing_vecs
            )
            candidate.max_corr_with_existing = max_corr

            if candidate.ic >= _IC_THRESHOLD and max_corr < _CORR_THRESHOLD:
                candidate.status = "PROMISING"
            elif candidate.ic < 0.01:
                candidate.status = "WEAK"
            elif max_corr >= _CORR_THRESHOLD:
                candidate.status = "REDUNDANT"
            else:
                candidate.status = "MARGINAL"

        self._results = list(self._candidates)
        self._last_run = time.time()
        promising = [c for c in self._results if c.status == "PROMISING"]
        if promising:
            log.info(f"SignalDiscovery: {len(promising)} promising candidates: {[c.name for c in promising]}")
            self._register_promising(promising)
        return self._results

    def _register_promising(self, candidates: list[CandidateSignal]) -> None:
        """Auto-register promising candidates in ExperimentLab as feature toggles."""
        try:
            from wallet_manager.experiment_lab import get_experiment_lab
            lab = get_experiment_lab()
            for c in candidates:
                if c.name not in [v.name for v in lab._feature_toggles.values()]:
                    lab.register_feature_toggle_variant(
                        name=f"candidate_{c.name}",
                        disabled_features=[],
                        enabled_features=[c.name],
                        description=f"AUTO-DISCOVERED: {c.description} IC={c.ic:.4f}",
                    )
                    log.info(f"SignalDiscovery: registered '{c.name}' in ExperimentLab")
        except Exception as exc:
            log.debug(f"SignalDiscovery: ExperimentLab registration failed: {exc}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_run": self._last_run,
            "n_prices": len(self._price_history),
            "candidates": [c.to_dict() for c in (self._results or self._candidates)],
            "promising_count": sum(1 for c in self._results if c.status == "PROMISING"),
        }

    # Signal computations

    @staticmethod
    def _rsi(prices: list[float], period: int) -> float:
        if len(prices) < period + 1:
            return 50.0
        changes = [prices[i] - prices[i - 1] for i in range(-period, 0)]
        gains = [max(c, 0) for c in changes]
        losses = [abs(min(c, 0)) for c in changes]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss < 1e-10:
            return 1.0  # All gains, normalize to +1
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0  # Normalize to [-1, +1]

    @staticmethod
    def _bollinger_position(prices: list[float], period: int) -> float:
        if len(prices) < period:
            return 0.0
        window = prices[-period:]
        mean = sum(window) / period
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
        if std < 1e-10:
            return 0.0
        current = prices[-1]
        return max(-1.0, min(1.0, (current - mean) / (2 * std)))

    @staticmethod
    def _atr_ratio(prices: list[float], period: int) -> float:
        if len(prices) < period * 2 + 1:
            return 1.0
        def atr(p: list[float], n: int) -> float:
            trs = [abs(p[i] - p[i - 1]) for i in range(-n, 0)]
            return sum(trs) / n if trs else 0.0
        short_atr = atr(prices, period)
        long_atr = atr(prices, period * 2)
        if long_atr < 1e-10:
            return 1.0
        ratio = short_atr / long_atr
        return max(-1.0, min(1.0, (ratio - 1.0)))  # 0 = baseline, >0 = high vol

    @staticmethod
    def _vwap_deviation(prices: list[float], period: int) -> float:
        if len(prices) < period:
            return 0.0
        window = prices[-period:]
        vwap = sum(window) / period
        current = prices[-1]
        if vwap < 1e-10:
            return 0.0
        dev = (current - vwap) / vwap
        return max(-1.0, min(1.0, dev * 20))

    @staticmethod
    def _momentum(prices: list[float], period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        ref = prices[-period - 1]
        if ref < 1e-10:
            return 0.0
        mom = (prices[-1] - ref) / ref
        return max(-1.0, min(1.0, mom * 10))


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 5:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (dx * dy) if dx * dy > 1e-12 else 0.0


_pipeline: SignalDiscoveryPipeline | None = None


def get_signal_discovery() -> SignalDiscoveryPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SignalDiscoveryPipeline()
    return _pipeline
