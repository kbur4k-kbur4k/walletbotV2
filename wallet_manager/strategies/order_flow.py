"""
wallet_manager/strategies/order_flow.py

Sprint JS-4.2: Order Flow Imbalance (OFI)
==========================================
OFI = (up_ticks - down_ticks) / total_ticks over last N price updates.
OFI > 0.30: buying pressure => add to long signal weight.
OFI < -0.30: selling pressure => add to short signal weight.
Spread dynamics: sudden spread widening => liquidity withdrawal => bearish.

Requires tick_collector (JS-4.4) for real-time tick data.
Falls back to zero signals when insufficient tick data is available.

RenTech insight: order flow information arrives BEFORE price movement.
This is microstructure alpha — the market tells you intent before it acts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_OFI_WINDOW = 100       # Ticks to look back for OFI calculation
_SPREAD_WINDOW = 20     # Periods for spread baseline
_STRONG_OFI = 0.30      # |OFI| > 0.30 = strong signal
_SPREAD_SPIKE_RATIO = 2.0  # Spread > 2x median = liquidity withdrawal


@dataclass
class TickObservation:
    """A single price tick observation."""
    timestamp: float
    price: float
    direction: int  # +1 = up tick, -1 = down tick, 0 = neutral


class OrderFlowImbalance:
    """
    Computes Order Flow Imbalance from tick-level price data.

    OFI measures the balance between buying and selling pressure
    in the order flow. Positive OFI = more bids being hit (bullish).
    Negative OFI = more offers being hit (bearish).

    Used as an ML feature — normalized to [-1, +1] range.
    Provides directional signal BEFORE price confirms the move.

    Usage:
        ofi = get_order_flow_imbalance()
        ofi.record_tick(price, bid_spread_bps)
        features = ofi.get_features()
    """

    def __init__(self, window: int = _OFI_WINDOW) -> None:
        self._ticks: deque[TickObservation] = deque(maxlen=window)
        self._spread_history: deque[float] = deque(maxlen=_SPREAD_WINDOW)
        self._last_price: float = 0.0
        self._session_up_ticks: int = 0
        self._session_down_ticks: int = 0
        self._session_neutral: int = 0

    def record_tick(self, price: float, spread_bps: float = 0.0) -> None:
        """Record a new price observation and classify the tick direction."""
        if price <= 0:
            return

        direction = 0
        if self._last_price > 0:
            if price > self._last_price:
                direction = 1
                self._session_up_ticks += 1
            elif price < self._last_price:
                direction = -1
                self._session_down_ticks += 1
            else:
                self._session_neutral += 1

        self._ticks.append(TickObservation(
            timestamp=time.time(),
            price=price,
            direction=direction,
        ))
        self._last_price = price

        if spread_bps > 0:
            self._spread_history.append(spread_bps)

    def compute_ofi(self, window: int | None = None) -> float:
        """
        Compute Order Flow Imbalance over the last `window` ticks.
        Returns value in [-1, +1]. Zero when insufficient data.
        """
        ticks = list(self._ticks) if window is None else list(self._ticks)[-window:]
        if len(ticks) < 5:
            return 0.0
        up = sum(1 for t in ticks if t.direction > 0)
        down = sum(1 for t in ticks if t.direction < 0)
        total = up + down
        if total == 0:
            return 0.0
        return round((up - down) / total, 4)

    def compute_ofi_momentum(self) -> float:
        """
        Compute OFI momentum: difference between short-window and long-window OFI.
        Positive = OFI accelerating upward (bullish momentum building).
        Negative = OFI decelerating upward (bullish momentum fading).
        """
        n = len(self._ticks)
        if n < 20:
            return 0.0
        ofi_short = self.compute_ofi(window=min(20, n))
        ofi_long = self.compute_ofi(window=min(100, n))
        return round(ofi_short - ofi_long, 4)

    def compute_spread_signal(self) -> float:
        """
        Detect spread spikes as liquidity withdrawal signals.
        Returns: 0.0 = normal, -1.0 = severe spread spike (bearish microstructure).
        """
        spreads = list(self._spread_history)
        if len(spreads) < 5:
            return 0.0
        median = sorted(spreads)[len(spreads) // 2]
        current = spreads[-1]
        if median <= 0:
            return 0.0
        ratio = current / median
        if ratio >= _SPREAD_SPIKE_RATIO:
            # Normalize: ratio 2 = -0.5, ratio 4 = -1.0
            return max(-1.0, -min(1.0, (ratio - 1.0) / 3.0))
        return 0.0

    def get_signal_strength(self) -> str:
        """Return human-readable signal interpretation."""
        ofi = self.compute_ofi()
        if ofi >= _STRONG_OFI:
            return "STRONG_BUY_PRESSURE"
        elif ofi >= 0.10:
            return "MILD_BUY_PRESSURE"
        elif ofi <= -_STRONG_OFI:
            return "STRONG_SELL_PRESSURE"
        elif ofi <= -0.10:
            return "MILD_SELL_PRESSURE"
        return "BALANCED"

    def get_features(self) -> dict[str, float]:
        """
        Return OFI features as normalized [-1, +1] values suitable for ML input.
        Returns zeros when insufficient tick data.
        """
        ofi = self.compute_ofi()
        ofi_momentum = self.compute_ofi_momentum()
        spread_signal = self.compute_spread_signal()
        return {
            "ofi": ofi,
            "ofi_momentum": ofi_momentum,
            "spread_signal": spread_signal,
            "ofi_composite": round((ofi * 0.6 + ofi_momentum * 0.3 + spread_signal * 0.1), 4),
        }

    def get_snapshot(self) -> dict[str, Any]:
        """Full snapshot for API/dashboard."""
        ofi = self.compute_ofi()
        n = len(self._ticks)
        total_session = self._session_up_ticks + self._session_down_ticks + self._session_neutral
        return {
            "ofi": ofi,
            "ofi_momentum": self.compute_ofi_momentum(),
            "spread_signal": self.compute_spread_signal(),
            "signal_strength": self.get_signal_strength(),
            "tick_window_size": n,
            "features": self.get_features(),
            "session_stats": {
                "up_ticks": self._session_up_ticks,
                "down_ticks": self._session_down_ticks,
                "neutral_ticks": self._session_neutral,
                "total_ticks": total_session,
                "session_ofi": round(
                    (self._session_up_ticks - self._session_down_ticks) /
                    max(self._session_up_ticks + self._session_down_ticks, 1), 4
                ),
            },
            "spread_history_len": len(self._spread_history),
            "last_spread_bps": round(self._spread_history[-1], 3) if self._spread_history else None,
        }


_ofi: OrderFlowImbalance | None = None


def get_order_flow_imbalance() -> OrderFlowImbalance:
    global _ofi
    if _ofi is None:
        _ofi = OrderFlowImbalance()
    return _ofi
