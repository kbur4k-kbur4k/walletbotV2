"""
wallet_manager/strategies/bayesian_kelly.py

Sprint JS-3.2: Bayesian Kelly Criterion
=========================================
Standard Kelly uses a point estimate of win probability.
Bayesian Kelly uses Beta(wins+1, losses+1) posterior to account for estimation uncertainty.

With 10 trades observed: posterior is wide → conservative sizing.
With 200 trades observed: posterior is narrow → approach full Kelly.

This prevents over-betting when win-rate estimate is based on small samples,
which is the most common Kelly failure mode.
"""

from __future__ import annotations

import math
from typing import Any


class BayesianKelly:
    """
    Computes uncertainty-adjusted Kelly fraction using Bayesian inference.

    The Kelly fraction under uncertainty is:
        f* = E[p] - (1 - E[p]) / (E[payoff] + 1)
    But with a Beta posterior, we use a conservative estimate:
        f_bayes = E[f*] - k * std[f*]  where k is a risk-aversion parameter.

    Gracefully reduces to standard Kelly when n_trades → ∞.
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,   # k * std subtracted from mean Kelly
        min_trades_for_full_kelly: int = 100,
    ) -> None:
        self._risk_aversion = risk_aversion
        self._min_trades = min_trades_for_full_kelly
        self._wins: int = 0
        self._losses: int = 0

    def update(self, win: bool) -> None:
        """Record a closed trade outcome."""
        if win:
            self._wins += 1
        else:
            self._losses += 1

    @property
    def n_trades(self) -> int:
        return self._wins + self._losses

    def _beta_mean(self) -> float:
        """Posterior mean of win probability: E[p] = (wins+1) / (wins+losses+2)"""
        return (self._wins + 1) / (self._wins + self._losses + 2)

    def _beta_variance(self) -> float:
        """Posterior variance of Beta distribution."""
        a = self._wins + 1
        b = self._losses + 1
        n = a + b
        return (a * b) / (n * n * (n + 1))

    def compute_kelly(
        self,
        win_fraction: float | None = None,
        loss_fraction: float | None = None,
        avg_win_usd: float = 0.0,
        avg_loss_usd: float = 0.0,
    ) -> float:
        """
        Compute Bayesian Kelly fraction.
        win_fraction: fraction of equity gained on win (e.g. 0.02 for 2%)
        loss_fraction: fraction of equity lost on loss (e.g. 0.01 for 1%)
        Falls back to avg_win_usd/avg_loss_usd if fractions not provided.
        Returns Kelly fraction clipped to [0, 0.25].
        """
        p = self._beta_mean()
        p_var = self._beta_variance()
        p_std = math.sqrt(p_var)

        if win_fraction is not None and loss_fraction is not None and loss_fraction > 0:
            b = win_fraction / loss_fraction
        elif avg_win_usd > 0 and avg_loss_usd > 0:
            b = avg_win_usd / avg_loss_usd
        else:
            b = 1.5  # default payoff ratio

        q = 1.0 - p
        standard_kelly = (b * p - q) / b if b > 0 else 0.0

        uncertainty_discount = self._risk_aversion * p_std * abs(standard_kelly)
        bayes_kelly = standard_kelly - uncertainty_discount

        data_maturity = min(1.0, self.n_trades / self._min_trades)
        if data_maturity < 1.0:
            bayes_kelly = bayes_kelly * (0.25 + 0.75 * data_maturity)

        return max(0.0, min(0.25, bayes_kelly))

    def get_uncertainty_factor(self) -> float:
        """
        Returns [0, 1] uncertainty factor. 1.0 = certain, 0.0 = maximum uncertainty.
        Can be used to scale position size independently of Kelly.
        """
        n = self.n_trades
        if n >= self._min_trades:
            return 1.0
        return math.sqrt(n / self._min_trades)

    def to_dict(self) -> dict[str, Any]:
        p = self._beta_mean()
        p_std = math.sqrt(self._beta_variance())
        kelly = self.compute_kelly()
        return {
            "wins": self._wins,
            "losses": self._losses,
            "n_trades": self.n_trades,
            "posterior_win_prob": round(p, 4),
            "posterior_std": round(p_std, 4),
            "bayes_kelly_fraction": round(kelly, 4),
            "uncertainty_factor": round(self.get_uncertainty_factor(), 4),
            "confidence_interval_95": [
                round(max(0, p - 2 * p_std), 4),
                round(min(1, p + 2 * p_std), 4),
            ],
            "data_maturity_pct": round(min(100, self.n_trades / self._min_trades * 100), 1),
        }


_bkelly: BayesianKelly | None = None


def get_bayesian_kelly() -> BayesianKelly:
    global _bkelly
    if _bkelly is None:
        _bkelly = BayesianKelly()
    return _bkelly
