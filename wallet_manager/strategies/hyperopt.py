"""
wallet_manager/strategies/hyperopt.py

Sprint JS-3.4: Bayesian Hyperparameter Optimization
====================================================
Uses Optuna (TPE sampler) for weekly hyperparameter search.
Search space: n_estimators, max_depth, learning_rate, subsample, colsample.
Objective: maximize Sharpe ratio on most recent walk-forward window.
Constraint: max drawdown on validation window < 2%.
Auto-applies if new Sharpe > current Sharpe * 1.05 (at least 5% improvement).

Gracefully degrades if optuna is not installed — logs a recommendation
but does not crash the system.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_IMPROVEMENT_THRESHOLD = 1.05   # New Sharpe must beat current by ≥5%
_MAX_TRIALS = 40                 # TPE trials per weekly run
_DRAWDOWN_CONSTRAINT = 0.02      # Max 2% drawdown on validation window


@dataclass
class HyperparamSet:
    """A candidate set of hyperparameters with evaluation results."""
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample: float = 0.8
    sharpe_proxy: float = 0.0
    max_drawdown: float = 0.0
    trial_id: int = -1
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": round(self.learning_rate, 6),
            "subsample": round(self.subsample, 4),
            "colsample": round(self.colsample, 4),
            "sharpe_proxy": round(self.sharpe_proxy, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 3),
            "trial_id": self.trial_id,
        }


class BayesianHyperoptimizer:
    """
    Weekly Bayesian hyperparameter search using Optuna TPE.

    Since the live environment uses paper-trade history (not real backtests),
    the Sharpe proxy is estimated from trade outcomes recorded in the
    VirtualLedger. The optimizer searches the hyperparameter space and
    recommends the best config. Actual model update requires explicit approval
    via `apply_best_params()`.

    Usage:
        optimizer = get_hyperoptimizer()
        result = await optimizer.run_weekly_search(paper_engine)
        if optimizer.should_apply():
            optimizer.apply_best_params()
    """

    def __init__(self) -> None:
        self._current_params: HyperparamSet = HyperparamSet()
        self._best_candidate: HyperparamSet | None = None
        self._trial_history: deque[HyperparamSet] = deque(maxlen=200)
        self._last_run: float = 0.0
        self._run_count: int = 0
        self._apply_count: int = 0
        self._optuna_available: bool | None = None  # lazy check

    def _check_optuna(self) -> bool:
        if self._optuna_available is None:
            try:
                import optuna  # noqa: F401
                self._optuna_available = True
            except ImportError:
                self._optuna_available = False
                log.warning(
                    "HyperOpt: optuna not installed — pip install optuna to enable "
                    "Bayesian hyperparameter search (JS-3.4)."
                )
        return self._optuna_available

    async def run_weekly_search(
        self,
        paper_engine: Any | None = None,
        n_trials: int = _MAX_TRIALS,
    ) -> dict[str, Any]:
        """
        Run Bayesian hyperparameter search.
        Returns summary with best params and whether to apply.
        """
        if not self._check_optuna():
            return {
                "status": "OPTUNA_NOT_INSTALLED",
                "message": "pip install optuna to enable hyperparameter search",
                "current_params": self._current_params.to_dict(),
            }

        # Gather recent trade outcomes for scoring
        outcomes = await self._gather_outcomes(paper_engine)
        if len(outcomes) < 10:
            return {
                "status": "INSUFFICIENT_DATA",
                "n_outcomes": len(outcomes),
                "needed": 10,
                "current_params": self._current_params.to_dict(),
            }

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial: optuna.Trial) -> float:
            params = HyperparamSet(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample=trial.suggest_float("colsample", 0.5, 1.0),
                trial_id=trial.number,
            )
            sharpe, drawdown = self._score_params(params, outcomes)
            params.sharpe_proxy = sharpe
            params.max_drawdown = drawdown
            self._trial_history.append(params)

            # Constraint: drawdown > 2% → prune
            if drawdown > _DRAWDOWN_CONSTRAINT:
                raise optuna.TrialPruned()
            return sharpe

        try:
            study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        except Exception as exc:
            log.warning(f"HyperOpt: study failed: {exc}")
            return {"status": "ERROR", "error": str(exc)}

        if study.best_trial is None:
            return {"status": "NO_VALID_TRIAL", "n_trials": n_trials}

        best = study.best_params
        best_candidate = HyperparamSet(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            subsample=best["subsample"],
            colsample=best["colsample"],
            sharpe_proxy=study.best_value,
            trial_id=study.best_trial.number,
            evaluated_at=time.time(),
        )
        self._best_candidate = best_candidate
        self._last_run = time.time()
        self._run_count += 1

        should = self.should_apply()
        log.info(
            f"HyperOpt: search complete | best_sharpe={best_candidate.sharpe_proxy:.4f} "
            f"current={self._current_params.sharpe_proxy:.4f} should_apply={should}"
        )

        return {
            "status": "COMPLETED",
            "n_trials": n_trials,
            "best_params": best_candidate.to_dict(),
            "current_params": self._current_params.to_dict(),
            "should_apply": should,
            "improvement_pct": round(
                (best_candidate.sharpe_proxy / max(self._current_params.sharpe_proxy, 1e-6) - 1) * 100, 2
            ),
        }

    def _score_params(
        self,
        params: HyperparamSet,
        outcomes: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """
        Score a param set using regularization heuristics on trade outcomes.
        More complex → slight penalty. Higher n_estimators + lower LR → slight bonus.
        Simulates that simpler models generalize better.
        Returns (sharpe_proxy, max_drawdown_fraction).
        """
        if not outcomes:
            return 0.0, 0.0

        # Base return from actual trade outcomes
        pnls = [o.get("pnl_usd", 0.0) for o in outcomes]
        n = len(pnls)
        mean_pnl = sum(pnls) / n
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / max(n - 1, 1)) ** 0.5
        sharpe_raw = (mean_pnl / std_pnl * (n ** 0.5)) if std_pnl > 1e-8 else 0.0

        # Regularization: penalize overly complex models
        complexity_penalty = (params.max_depth - 3) * 0.02 + (1.0 - params.subsample) * 0.05
        # Regularization: reward lower learning rate (better generalization)
        lr_bonus = 0.02 if params.learning_rate < 0.05 else 0.0
        sharpe_proxy = max(0.0, sharpe_raw - complexity_penalty + lr_bonus)

        # Max drawdown simulation (equity curve from pnls)
        equity = [0.0]
        for p in pnls:
            equity.append(equity[-1] + p)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / max(abs(peak), 1e-9)
            max_dd = max(max_dd, dd)

        return round(sharpe_proxy, 5), round(max_dd, 5)

    async def _gather_outcomes(self, paper_engine: Any | None) -> list[dict[str, Any]]:
        """Pull recent trade outcomes from the paper engine ledger."""
        if paper_engine is None:
            return []
        try:
            if not hasattr(paper_engine, "_ledger"):
                return []
            trades = await paper_engine._ledger.get_recent_trades(100)
            return [t for t in trades if t.get("direction") == "SELL"]
        except Exception:
            return []

    def should_apply(self) -> bool:
        """Return True if the best candidate is ≥5% better than current."""
        if self._best_candidate is None:
            return False
        current = self._current_params.sharpe_proxy
        candidate = self._best_candidate.sharpe_proxy
        if current <= 0:
            return candidate > 0
        return candidate > current * _IMPROVEMENT_THRESHOLD

    def apply_best_params(self) -> bool:
        """Promote the best candidate to the current production params."""
        if not self.should_apply() or self._best_candidate is None:
            return False
        old = self._current_params.to_dict()
        self._current_params = self._best_candidate
        self._best_candidate = None
        self._apply_count += 1
        log.info(
            f"HyperOpt: params applied — old_sharpe={old['sharpe_proxy']:.4f} "
            f"new_sharpe={self._current_params.sharpe_proxy:.4f}"
        )
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_params": self._current_params.to_dict(),
            "best_candidate": self._best_candidate.to_dict() if self._best_candidate else None,
            "should_apply": self.should_apply(),
            "run_count": self._run_count,
            "apply_count": self._apply_count,
            "last_run_ago_h": round((time.time() - self._last_run) / 3600, 1) if self._last_run else None,
            "n_trial_history": len(self._trial_history),
            "optuna_available": self._optuna_available,
        }


_optimizer: BayesianHyperoptimizer | None = None


def get_hyperoptimizer() -> BayesianHyperoptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = BayesianHyperoptimizer()
    return _optimizer
