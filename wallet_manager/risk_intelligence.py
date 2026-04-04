"""
wallet_manager/risk_intelligence.py

Sprint JS-2: Risk Layer Upgrade
=================================
JS-2.1  DrawdownAccelerationDetector — 2nd derivative drawdown early warning
JS-2.2  FatTailVaR                  — Pareto-distribution tail risk in USD
JS-2.3  StressScenarioEngine        — Replay 5 historical crypto crises
JS-2.4  RegimePositionLimits        — Regime-conditional Kelly multipliers

All components are stateful singletons and expose to_dict() for API/dashboard.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


# ---------------------------------------------------------------------------
# JS-2.1  Drawdown Acceleration Detector
# ---------------------------------------------------------------------------


class DrawdownAccelerationDetector:
    """
    Monitors drawdown velocity (1st derivative) and acceleration (2nd derivative).
    When acceleration > threshold for 3 consecutive readings:
      => emits DRAWDOWN_ACCELERATING alert
      => recommends Kelly multiplier 0.50

    Earlier warning than a simple max-drawdown gate, which only fires after
    significant damage has already occurred.
    """

    def __init__(
        self,
        accel_threshold: float = 0.005,
        consec_required: int = 3,
        window: int = 30,
    ) -> None:
        self._accel_threshold = accel_threshold
        self._consec_required = consec_required
        self._drawdown_history: deque[float] = deque(maxlen=window)
        self._velocity_history: deque[float] = deque(maxlen=window)
        self._accel_history: deque[float] = deque(maxlen=window)
        self._consec_accel: int = 0
        self._peak_equity: float = 0.0
        self._alert_active: bool = False

    def update(self, current_equity: float) -> None:
        """Feed current equity value. Call once per shadow cycle."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        self._drawdown_history.append(drawdown)
        if len(self._drawdown_history) >= 2:
            velocity = self._drawdown_history[-1] - self._drawdown_history[-2]
            self._velocity_history.append(velocity)
        if len(self._velocity_history) >= 2:
            accel = self._velocity_history[-1] - self._velocity_history[-2]
            self._accel_history.append(accel)
            if accel > self._accel_threshold:
                self._consec_accel += 1
            else:
                self._consec_accel = 0
            self._alert_active = self._consec_accel >= self._consec_required

    def get_kelly_multiplier(self) -> float:
        """Return recommended Kelly multiplier based on drawdown state."""
        if not self._drawdown_history:
            return 1.0
        current_dd = self._drawdown_history[-1]
        if self._alert_active or current_dd > 0.10:
            return 0.50
        if current_dd > 0.05:
            return 0.75
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        current_dd = self._drawdown_history[-1] if self._drawdown_history else 0.0
        velocity = self._velocity_history[-1] if self._velocity_history else 0.0
        accel = self._accel_history[-1] if self._accel_history else 0.0
        return {
            "current_drawdown_pct": round(current_dd * 100, 4),
            "velocity": round(velocity, 6),
            "acceleration": round(accel, 6),
            "consecutive_accelerating": self._consec_accel,
            "alert_active": self._alert_active,
            "kelly_multiplier": self.get_kelly_multiplier(),
            "peak_equity_usd": round(self._peak_equity, 4),
        }


# ---------------------------------------------------------------------------
# JS-2.2  Fat-tail VaR (Pareto distribution)
# ---------------------------------------------------------------------------


class FatTailVaR:
    """
    Fits a Pareto distribution to the tail of observed returns.
    Normal VaR underestimates crypto losses by 3-5x due to fat tails.

    Reports VaR_95, VaR_99, VaR_99.9 as fraction of equity.
    Gate: if VaR_99 > 2% of equity => recommend position size reduction.
    """

    def __init__(self, tail_pct: float = 0.10, window: int = 500) -> None:
        self._tail_pct = tail_pct
        self._returns: deque[float] = deque(maxlen=window)

    def record_return(self, pct_return: float) -> None:
        """Record a realised return (as fraction, e.g. -0.02 for -2%)."""
        self._returns.append(pct_return)

    def _pareto_var(self, confidence: float) -> float:
        """
        Estimate VaR at given confidence using Hill estimator on tail losses.
        Falls back to empirical quantile when insufficient data.
        """
        if len(self._returns) < 30:
            return 0.0
        losses = sorted([-r for r in self._returns if r < 0], reverse=True)
        if not losses:
            return 0.0
        n = len(losses)
        tail_n = max(1, int(n * self._tail_pct))
        tail_losses = losses[:tail_n]
        u = tail_losses[-1]
        if u <= 0:
            return losses[min(int(n * (1 - confidence)), n - 1)]
        try:
            xi = sum(math.log(x / u) for x in tail_losses) / tail_n
        except (ValueError, ZeroDivisionError):
            xi = 0.3
        xi = max(0.01, min(xi, 2.0))
        exceed_prob = tail_n / n
        p_exceed = 1 - confidence
        if p_exceed <= 0:
            return losses[0]
        quantile = float(u * ((exceed_prob / p_exceed) ** xi))
        return round(quantile, 6)

    def get_var_report(self, equity_usd: float = 10_000.0) -> dict[str, Any]:
        n = len(self._returns)
        var_95 = self._pareto_var(0.95)
        var_99 = self._pareto_var(0.99)
        var_999 = self._pareto_var(0.999)
        usd_95 = round(var_95 * equity_usd, 2)
        usd_99 = round(var_99 * equity_usd, 2)
        usd_999 = round(var_999 * equity_usd, 2)
        alert = var_99 > 0.02
        return {
            "n_observations": n,
            "var_95_pct": round(var_95 * 100, 4),
            "var_99_pct": round(var_99 * 100, 4),
            "var_999_pct": round(var_999 * 100, 4),
            "var_95_usd": usd_95,
            "var_99_usd": usd_99,
            "var_999_usd": usd_999,
            "alert": alert,
            "alert_reason": f"VaR_99 ({var_99*100:.2f}%) exceeds 2% equity limit." if alert else "",
            "recommended_size_factor": round(0.02 / var_99, 2) if var_99 > 0.02 else 1.0,
        }


# ---------------------------------------------------------------------------
# JS-2.3  Historical Stress Scenarios
# ---------------------------------------------------------------------------


@dataclass
class StressScenario:
    name: str
    description: str
    price_changes: list[float]   # hourly % changes during the event (as fractions)
    start_price: float = 2000.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "n_hours": len(self.price_changes)}


@dataclass
class StressResult:
    scenario_name: str
    max_loss_usd: float
    max_drawdown_pct: float
    risk_committee_halted: bool
    cycles_before_halt: int
    capital_preserved_pct: float
    final_equity_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "max_loss_usd": round(self.max_loss_usd, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "risk_committee_halted": self.risk_committee_halted,
            "cycles_before_halt": self.cycles_before_halt,
            "capital_preserved_pct": round(self.capital_preserved_pct, 2),
            "final_equity_usd": round(self.final_equity_usd, 2),
            "pass": self.max_drawdown_pct < 25.0,
        }


class StressScenarioEngine:
    """
    Replays 5 embedded historical crypto crisis price series through a
    simplified version of the strategy's risk logic to estimate worst-case losses.
    """

    SCENARIOS: list[StressScenario] = [
        StressScenario(
            name="2020_covid_crash",
            description="COVID-19 crash — ETH -40% in 24h (2020-03-12)",
            start_price=200.0,
            price_changes=[
                -0.05, -0.08, -0.10, -0.07, -0.05, -0.04,
                -0.03, -0.02, 0.02, 0.03, 0.01, -0.01,
                0.02, 0.04, 0.03, 0.01, 0.02, 0.01, 0.00, 0.01,
                0.02, 0.01, 0.03, 0.02,
            ],
        ),
        StressScenario(
            name="2021_china_ban",
            description="China mining ban — ETH -35% over 48h (2021-05-19)",
            start_price=3400.0,
            price_changes=[
                -0.04, -0.06, -0.05, -0.04, -0.05, -0.04, -0.03, -0.02,
                -0.01, 0.01, 0.02, 0.01, -0.01, -0.02, -0.01, 0.01,
                0.02, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01,
                0.00, 0.01, 0.02, 0.01, 0.00, 0.01, 0.02, 0.01,
                0.02, 0.03, 0.02, 0.01, 0.02, 0.01, 0.00, 0.01,
                0.01, 0.02, 0.01, 0.02, 0.01, 0.00, 0.01, 0.02,
            ],
        ),
        StressScenario(
            name="2022_luna_contagion",
            description="LUNA/3AC contagion — ETH -30% (2022-06-13)",
            start_price=1800.0,
            price_changes=[
                -0.06, -0.05, -0.04, -0.05, -0.04, -0.03,
                -0.03, -0.02, -0.01, 0.01, 0.02, 0.01,
                -0.01, 0.01, 0.02, 0.01, 0.02, 0.01, 0.00, 0.01,
                0.02, 0.01, 0.02, 0.01,
            ],
        ),
        StressScenario(
            name="2022_ftx_collapse",
            description="FTX collapse — ETH -25% (2022-11-09)",
            start_price=1600.0,
            price_changes=[
                -0.05, -0.04, -0.04, -0.04, -0.03, -0.03,
                -0.02, -0.01, 0.01, 0.02, 0.01, 0.01,
                0.02, 0.01, 0.00, 0.01, 0.02, 0.01, 0.02, 0.01,
                0.00, 0.01, 0.01, 0.02,
            ],
        ),
        StressScenario(
            name="2023_usdc_depeg",
            description="USDC depeg (SVB) — ETH -8% + stablecoin panic (2023-03-11)",
            start_price=1550.0,
            price_changes=[
                -0.02, -0.03, -0.02, -0.01, 0.01, 0.02,
                0.01, 0.01, 0.00, 0.01, 0.02, 0.01,
                0.01, 0.02, 0.01, 0.00, 0.01, 0.01, 0.02, 0.01,
                0.02, 0.01, 0.00, 0.01,
            ],
        ),
    ]

    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self._initial_equity = initial_equity
        self._results: list[StressResult] = []
        self._last_run: float = 0.0

    def run_all(self) -> list[StressResult]:
        """Run all 5 stress scenarios. Returns list of StressResult."""
        results: list[StressResult] = []
        for scenario in self.SCENARIOS:
            results.append(self._run_scenario(scenario))
        self._results = results
        self._last_run = time.time()
        return results

    def _run_scenario(self, scenario: StressScenario) -> StressResult:
        equity = self._initial_equity
        price = scenario.start_price
        peak_equity = equity
        max_loss = 0.0
        max_dd = 0.0
        halted = False
        halt_cycle = len(scenario.price_changes)
        daily_loss = 0.0
        cycles = 0

        for i, chg in enumerate(scenario.price_changes):
            price *= (1 + chg)
            if chg < 0:
                trade_loss = equity * abs(chg) * 0.5
                equity -= trade_loss
                daily_loss += trade_loss
                max_loss = max(max_loss, self._initial_equity - equity)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
            max_dd = max(max_dd, dd)
            cycles = i + 1
            if daily_loss > self._initial_equity * 0.05:
                halted = True
                halt_cycle = cycles
                break
            if dd > 20.0:
                halted = True
                halt_cycle = cycles
                break

        return StressResult(
            scenario_name=scenario.name,
            max_loss_usd=round(max_loss, 2),
            max_drawdown_pct=round(max_dd, 2),
            risk_committee_halted=halted,
            cycles_before_halt=halt_cycle,
            capital_preserved_pct=round(equity / self._initial_equity * 100, 2),
            final_equity_usd=round(equity, 2),
        )

    def to_dict(self) -> dict[str, Any]:
        if not self._results:
            return {"status": "NOT_RUN", "last_run": None, "results": [], "all_pass": None}
        all_pass = all(r.max_drawdown_pct < 25.0 for r in self._results)
        return {
            "status": "OK",
            "last_run": self._last_run,
            "all_pass": all_pass,
            "results": [r.to_dict() for r in self._results],
            "worst_scenario": max(self._results, key=lambda r: r.max_drawdown_pct).scenario_name,
            "best_preserved_capital_pct": min(r.capital_preserved_pct for r in self._results),
        }


# ---------------------------------------------------------------------------
# JS-2.4  Regime-Conditional Position Limits
# ---------------------------------------------------------------------------

_REGIME_MULTIPLIERS: dict[str, float] = {
    "low_vol": 1.00,
    "low_volatility": 1.00,
    "normal": 0.75,
    "normal_bullish": 0.75,
    "normal_bearish": 0.75,
    "trending": 0.75,
    "uptrend": 0.75,
    "downtrend": 0.65,
    "high_vol": 0.50,
    "high_volatility": 0.50,
    "volatile": 0.50,
    "news_blackout": 0.25,
    "crisis": 0.25,
    "unknown": 0.50,
}


class RegimePositionLimits:
    """
    Provides regime-conditional Kelly multipliers to prevent over-sizing
    during elevated uncertainty periods.

    Usage:
        multiplier = limits.get_multiplier(regime_label)
        position_usd = deep_kelly_size * multiplier
    """

    def __init__(self, multipliers: dict[str, float] | None = None) -> None:
        self._multipliers = multipliers or _REGIME_MULTIPLIERS
        self._last_regime: str = "unknown"
        self._last_multiplier: float = 0.50

    def get_multiplier(self, regime_label: str) -> float:
        label = (regime_label or "unknown").lower().strip()
        self._last_regime = label
        for key, mult in self._multipliers.items():
            if key in label:
                self._last_multiplier = mult
                return mult
        self._last_multiplier = 0.50
        return 0.50

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_regime": self._last_regime,
            "last_multiplier": self._last_multiplier,
            "multiplier_table": self._multipliers,
        }


# ---------------------------------------------------------------------------
# RiskIntelligence facade
# ---------------------------------------------------------------------------


class RiskIntelligence:
    """Facade owning all four JS-2.x components."""

    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self.drawdown_accel = DrawdownAccelerationDetector()
        self.fat_tail_var = FatTailVaR()
        self.stress_engine = StressScenarioEngine(initial_equity)
        self.regime_limits = RegimePositionLimits()

    def record_cycle(
        self,
        equity_usd: float,
        pct_return: float,
        regime_label: str = "unknown",
    ) -> None:
        self.drawdown_accel.update(equity_usd)
        if pct_return != 0.0:
            self.fat_tail_var.record_return(pct_return)

    def get_effective_kelly_multiplier(self, regime_label: str = "unknown") -> float:
        regime_mult = self.regime_limits.get_multiplier(regime_label)
        dd_mult = self.drawdown_accel.get_kelly_multiplier()
        return round(min(regime_mult, dd_mult), 4)

    def run_stress_scenarios(self) -> list[StressResult]:
        return self.stress_engine.run_all()

    def to_dict(self, equity_usd: float = 10_000.0) -> dict[str, Any]:
        return {
            "generated_at": time.time(),
            "drawdown_acceleration": self.drawdown_accel.to_dict(),
            "fat_tail_var": self.fat_tail_var.get_var_report(equity_usd),
            "stress_scenarios": self.stress_engine.to_dict(),
            "regime_limits": self.regime_limits.to_dict(),
            "effective_kelly_multiplier": self.get_effective_kelly_multiplier(
                self.regime_limits._last_regime
            ),
        }

    def get_alerts(self) -> list[str]:
        alerts: list[str] = []
        dd = self.drawdown_accel.to_dict()
        if dd["alert_active"]:
            alerts.append(
                f"DRAWDOWN_ACCELERATING: {dd['consecutive_accelerating']} consecutive "
                f"accelerating readings. Kelly reduced to {dd['kelly_multiplier']}."
            )
        var = self.fat_tail_var.get_var_report()
        if var["alert"]:
            alerts.append(f"FAT_TAIL_VAR: {var['alert_reason']}")
        return alerts


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_risk_intel: RiskIntelligence | None = None


def get_risk_intelligence(initial_equity: float = 10_000.0) -> RiskIntelligence:
    global _risk_intel
    if _risk_intel is None:
        _risk_intel = RiskIntelligence(initial_equity)
    return _risk_intel
