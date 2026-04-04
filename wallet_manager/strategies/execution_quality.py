"""
wallet_manager/strategies/execution_quality.py

Sprint JS-5.1 + JS-5.3: Execution Quality
==========================================
JS-5.1  TWAP Execution Emulator      — compare TWAP vs immediate fill
JS-5.3  Slippage Calibration Loop    — auto-calibrate slippage model

All analysis is done in paper mode — no real transactions affected.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class TradeExecutionRecord:
    timestamp: float
    immediate_fill_price: float
    estimated_slippage_bps: float
    actual_slippage_bps: float = 0.0
    twap_price: float = 0.0
    trade_size_usd: float = 0.0
    execution_quality_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "immediate_fill_price": self.immediate_fill_price,
            "estimated_slippage_bps": round(self.estimated_slippage_bps, 3),
            "actual_slippage_bps": round(self.actual_slippage_bps, 3),
            "twap_price": round(self.twap_price, 4),
            "execution_quality_score": round(self.execution_quality_score, 4),
        }


class TWAPEmulator:
    """
    JS-5.1: Simulates TWAP execution by averaging N price samples over T minutes.
    Compares the TWAP fill price to the immediate fill price.
    If TWAP saves > 0.05%, immediate execution is suboptimal.
    """

    def __init__(self, slices: int = 5, window_minutes: float = 5.0) -> None:
        self._slices = slices
        self._window_s = window_minutes * 60.0
        self._pending_twaPs: list[tuple[float, float, list[float]]] = []  # (start_ts, fill_price, [samples])

    def start_twap(self, immediate_fill_price: float) -> None:
        """Start a new TWAP calculation starting from the immediate fill price."""
        self._pending_twaPs.append((time.time(), immediate_fill_price, [immediate_fill_price]))

    def update_prices(self, current_price: float) -> list[dict[str, Any]]:
        """Feed current price. Returns completed TWAP comparisons."""
        now = time.time()
        completed: list[dict[str, Any]] = []
        still_pending = []
        for start_ts, fill_price, samples in self._pending_twaPs:
            elapsed = now - start_ts
            slice_interval = self._window_s / self._slices if self._slices > 0 else 1.0
            expected_samples = min(self._slices, int(elapsed / slice_interval) + 1) if slice_interval > 0 else self._slices
            if len(samples) < expected_samples:
                samples.append(current_price)
            if elapsed >= self._window_s:
                twap = sum(samples) / len(samples) if samples else fill_price
                saving_bps = (fill_price - twap) / fill_price * 10_000 if fill_price > 0 else 0.0
                completed.append({
                    "immediate_fill": fill_price,
                    "twap_price": round(twap, 4),
                    "saving_bps": round(saving_bps, 3),
                    "suboptimal": saving_bps > 0.5,
                })
            else:
                still_pending.append((start_ts, fill_price, samples))
        self._pending_twaPs = still_pending
        return completed


class SlippageCalibrator:
    """
    JS-5.3: Tracks estimated vs actual slippage and computes a calibration factor.
    Auto-adjusts the slippage model when calibration_factor diverges significantly.
    """

    def __init__(self, window: int = 50) -> None:
        self._records: deque[tuple[float, float]] = deque(maxlen=window)  # (estimated, actual)
        self._calibration_factor: float = 1.0

    def record(self, estimated_bps: float, actual_bps: float) -> None:
        self._records.append((estimated_bps, actual_bps))
        self._update_factor()

    def _update_factor(self) -> None:
        if len(self._records) < 10:
            return
        ratios = [act / est for est, act in self._records if est > 0.1]
        if not ratios:
            return
        self._calibration_factor = round(sum(ratios) / len(ratios), 4)

    def apply_calibration(self, estimated_bps: float) -> float:
        """Return calibration-adjusted slippage estimate."""
        return round(estimated_bps * self._calibration_factor, 4)

    def to_dict(self) -> dict[str, Any]:
        n = len(self._records)
        status = "WELL_CALIBRATED"
        if self._calibration_factor > 1.3:
            status = "UNDERESTIMATING"
        elif self._calibration_factor < 0.7:
            status = "OVERESTIMATING"
        return {
            "n_records": n,
            "calibration_factor": self._calibration_factor,
            "status": status,
            "alert": status != "WELL_CALIBRATED" and n >= 10,
            "recent_avg_actual_bps": round(
                sum(act for _, act in list(self._records)[-10:]) / min(n, 10), 3
            ) if n > 0 else 0.0,
        }


class ExecutionQualityMonitor:
    """Facade owning TWAP emulator + slippage calibrator."""

    def __init__(self) -> None:
        self.twap = TWAPEmulator()
        self.calibrator = SlippageCalibrator()
        self._history: deque[dict[str, Any]] = deque(maxlen=200)

    def on_trade_entry(self, fill_price: float, estimated_slippage_bps: float, trade_size_usd: float) -> None:
        self.twap.start_twap(fill_price)

    def on_price_update(self, current_price: float) -> None:
        completed = self.twap.update_prices(current_price)
        for c in completed:
            self._history.append(c)

    def on_trade_close(self, estimated_bps: float, actual_bps: float) -> None:
        self.calibrator.record(estimated_bps, actual_bps)

    def get_timing_snapshot(self) -> dict[str, Any]:
        """Return timing optimization snapshot (alias for get_optimal_timing().to_dict())."""
        return get_optimal_timing().to_dict()

    def to_dict(self) -> dict[str, Any]:
        recent = list(self._history)[-10:]
        suboptimal_count = sum(1 for r in recent if r.get("suboptimal"))
        return {
            "twap_history_count": len(self._history),
            "suboptimal_executions_last10": suboptimal_count,
            "recent_twap": recent[-3:] if recent else [],
            "slippage_calibration": self.calibrator.to_dict(),
        }


class OptimalExecutionTiming:
    """
    JS-5.2: Tracks actual slippage by hour-of-day bucket (24 buckets).
    Flags low-liquidity windows where slippage > 1.5x median.
    Recommends avoiding execution during flagged windows.
    """

    def __init__(self) -> None:
        self._hourly_slippage: list[list[float]] = [[] for _ in range(24)]
        self._max_per_hour = 200

    def record_execution(self, slippage_bps: float, timestamp: float | None = None) -> None:
        import datetime
        import time as _time
        ts = timestamp or _time.time()
        hour = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).hour
        bucket = self._hourly_slippage[hour]
        bucket.append(slippage_bps)
        if len(bucket) > self._max_per_hour:
            self._hourly_slippage[hour] = bucket[-self._max_per_hour:]

    def get_current_window_quality(self) -> dict[str, Any]:
        import datetime
        import time as _time
        current_hour = datetime.datetime.fromtimestamp(_time.time(), tz=datetime.UTC).hour
        bucket = self._hourly_slippage[current_hour]
        if not bucket:
            return {"hour": current_hour, "status": "NO_DATA", "avg_slippage_bps": None, "avoid": False}
        avg = sum(bucket) / len(bucket)
        all_avgs = [sum(b) / len(b) for b in self._hourly_slippage if b]
        median = sorted(all_avgs)[len(all_avgs) // 2] if all_avgs else avg
        ratio = avg / median if median > 0 else 1.0
        avoid = ratio > 1.5 and len(bucket) >= 10
        return {
            "hour": current_hour,
            "avg_slippage_bps": round(avg, 3),
            "median_bps": round(median, 3),
            "ratio_to_median": round(ratio, 3),
            "status": "AVOID" if avoid else "OK",
            "avoid": avoid,
            "n_observations": len(bucket),
        }

    def get_best_hours(self, n: int = 6) -> list[int]:
        """Return the N hours with lowest average slippage."""
        avgs = [(h, sum(b) / len(b)) for h, b in enumerate(self._hourly_slippage) if len(b) >= 5]
        avgs.sort(key=lambda x: x[1])
        return [h for h, _ in avgs[:n]]

    def to_dict(self) -> dict[str, Any]:
        current = self.get_current_window_quality()
        best = self.get_best_hours(6)
        hourly_summary = []
        for h, bucket in enumerate(self._hourly_slippage):
            if bucket:
                hourly_summary.append({"hour": h, "avg_bps": round(sum(bucket)/len(bucket), 3), "n": len(bucket)})
        return {
            "current_window": current,
            "best_hours_utc": best,
            "avoid_now": current["avoid"],
            "hourly_summary": hourly_summary,
        }


_eq_monitor: ExecutionQualityMonitor | None = None
_timing_monitor: OptimalExecutionTiming | None = None


def get_optimal_timing() -> OptimalExecutionTiming:
    global _timing_monitor
    if _timing_monitor is None:
        _timing_monitor = OptimalExecutionTiming()
    return _timing_monitor


def get_execution_quality() -> ExecutionQualityMonitor:
    global _eq_monitor
    if _eq_monitor is None:
        _eq_monitor = ExecutionQualityMonitor()
    return _eq_monitor
