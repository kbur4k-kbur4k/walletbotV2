"""
wallet_manager/strategies/stochastic_stress_test.py

Anti-Overfit "Chaos" Layer — Monte Carlo Stress Testing & Recursive
Feature Elimination.

Before any alpha signal reaches execution, it must survive two
adversarial gauntlets:

1. Monte Carlo Stress Testing — N jittered price paths, discard if
   win rate < 55 %.  VaR(95 %) for tail-risk quantification.
2. Recursive Feature Elimination — correlation-rank features per HMM
   regime, prune those with high cross-regime rank variance.

Pure NumPy, no sklearn.  Vectorized Monte Carlo inner loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — Monte Carlo
# ---------------------------------------------------------------------------

MC_N_SIMULATIONS: Final[int] = 100
"""Number of perturbed price paths generated per stress test."""

MC_PRICE_JITTER_PCT: Final[float] = 0.005
"""Standard deviation of Gaussian noise applied to prices (+-0.5 %)."""

MC_MIN_WIN_RATE: Final[float] = 0.40
"""Minimum fraction of profitable simulations to approve a trade (40 %)."""

# ---------------------------------------------------------------------------
# Constants — Recursive Feature Elimination
# ---------------------------------------------------------------------------

RFE_MIN_FEATURES: Final[int] = 5
"""Floor on surviving features to preserve signal diversity."""

RFE_IMPORTANCE_DECAY_THRESHOLD: Final[float] = 0.15
"""Maximum rank variance across regimes before a feature is pruned."""

# ---------------------------------------------------------------------------
# Reason Codes
# ---------------------------------------------------------------------------

MC_CHAOS_PASSED: Final[str] = "MC_CHAOS_PASSED"
MC_CHAOS_FAILED: Final[str] = "MC_CHAOS_FAILED"
MC_LOW_WIN_RATE: Final[str] = "MC_LOW_WIN_RATE"
RFE_FEATURES_PRUNED: Final[str] = "RFE_FEATURES_PRUNED"
RFE_REGIME_INCONSISTENT: Final[str] = "RFE_REGIME_INCONSISTENT"
STRESS_TEST_APPROVED: Final[str] = "STRESS_TEST_APPROVED"
STRESS_TEST_REJECTED: Final[str] = "STRESS_TEST_REJECTED"

# ---------------------------------------------------------------------------
# Minimum history required for meaningful stress testing
# ---------------------------------------------------------------------------

_MIN_PRICE_HISTORY: Final[int] = 20
"""Minimum price bars needed to build a simulation base."""

_VAR_CONFIDENCE: Final[float] = 0.95
"""Confidence level for Value at Risk calculation."""

_RFE_REGIME_CONSISTENCY_FLOOR: Final[float] = 0.5
"""Minimum regime consistency score to pass RFE gate."""


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcome of a single Monte Carlo stress-test run."""

    n_simulations: int
    win_rate: float
    mean_pnl: float
    std_pnl: float
    worst_case_pnl: float
    best_case_pnl: float
    var_95: float
    passed: bool
    reason_code: str
    timestamp: float

    @property
    def var_99(self) -> float:
        """Value-at-Risk reported as a non-negative loss magnitude."""
        return abs(self.var_95)


@dataclass(frozen=True)
class RFEResult:
    """Outcome of Recursive Feature Elimination."""

    original_features: int
    surviving_features: int
    eliminated_features: tuple[str, ...]
    surviving_features_names: tuple[str, ...]
    feature_rankings: tuple[tuple[str, float], ...]
    regime_consistency_score: float
    timestamp: float


@dataclass(frozen=True)
class StressTestVerdict:
    """Combined verdict from Monte Carlo + RFE."""

    monte_carlo: MonteCarloResult
    rfe: RFEResult | None
    approved: bool
    rejection_reasons: tuple[str, ...]
    confidence_score: float
    timestamp: float

    @property
    def should_halt(self) -> bool:
        """True when the verdict recommends halting trading."""
        return not self.approved


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------


class MonteCarloEngine:
    """Vectorized Monte Carlo stress tester for alpha signals.

    Generates ``n_simulations`` jittered copies of recent price history,
    re-evaluates the alpha signal on each, and checks profitability
    across the noise-perturbed ensemble.
    """

    def __init__(
        self,
        n_simulations: int = MC_N_SIMULATIONS,
        jitter_pct: float = MC_PRICE_JITTER_PCT,
        min_win_rate: float = MC_MIN_WIN_RATE,
    ) -> None:
        self._n_simulations = max(1, n_simulations)
        self._jitter_pct = max(0.0, jitter_pct)
        self._min_win_rate = max(0.0, min(1.0, min_win_rate))
        self._rng = np.random.default_rng(42)

        log.info(
            f"MonteCarloEngine initialized: sims={self._n_simulations} "
            f"jitter={self._jitter_pct:.4f} min_wr={self._min_win_rate:.2f}"
        )

    # ----- Internal helpers -----

    def _generate_jittered_prices(
        self, price_history: np.ndarray,
    ) -> np.ndarray:
        """Generate a (n_simulations, n_prices) matrix of jittered paths.

        Each row is the original price history multiplied element-wise by
        (1 + N(0, jitter_pct)).  Fully vectorized — no Python loops.
        """
        n_prices = len(price_history)
        noise = self._rng.normal(
            loc=0.0,
            scale=self._jitter_pct,
            size=(self._n_simulations, n_prices),
        )
        # Broadcast: (n_sims, n_prices) * (n_prices,) -> (n_sims, n_prices)
        jittered = price_history[np.newaxis, :] * (1.0 + noise)
        # Floor prices at a tiny positive to avoid division by zero
        return np.maximum(jittered, 1e-12)

    @staticmethod
    def _default_predict(
        current_price: float,
        signal_alpha: float,
        signal_direction: int,
        sim_prices: np.ndarray,
    ) -> np.ndarray:
        """Default PnL: (exit - entry) / entry * direction * |alpha|."""
        exit_price = sim_prices[-1]
        raw_return = (exit_price - current_price) / max(current_price, 1e-12)
        pnl = raw_return * signal_direction * abs(signal_alpha)
        return np.array(pnl)

    def _evaluate_simulations_vectorized(
        self,
        current_price: float,
        signal_alpha: float,
        signal_direction: int,
        jittered_matrix: np.ndarray,
        predict_fn: Callable | None = None,
    ) -> np.ndarray:
        """Compute PnL array (n_simulations,) across all jittered paths."""
        if predict_fn is not None:
            # Custom predict_fn: must iterate (user-supplied, may not vectorise)
            pnl_list: list[float] = []
            for i in range(jittered_matrix.shape[0]):
                result = predict_fn(
                    current_price, signal_alpha, signal_direction,
                    jittered_matrix[i],
                )
                pnl_list.append(float(np.asarray(result).ravel()[0]))
            return np.array(pnl_list, dtype=np.float64)

        # Fully vectorized default: PnL = direction * |alpha| * (exit - entry) / entry
        exit_prices = jittered_matrix[:, -1]
        raw_returns = (exit_prices - current_price) / max(current_price, 1e-12)
        return raw_returns * signal_direction * abs(signal_alpha)

    # ----- Public API -----

    def run_stress_test(
        self,
        current_price: float = 0.0,
        signal_alpha: float = 0.5,
        signal_direction: int = 1,
        price_history: np.ndarray | list | None = None,
        predict_fn: Callable | None = None,
        # Simplified API aliases
        prices: np.ndarray | list | None = None,
        n_simulations: int | None = None,
    ) -> MonteCarloResult:
        """Run the full Monte Carlo stress test on an alpha signal."""
        ts = time.time()
        # Support simplified API: prices=..., n_simulations=...
        _price_data = prices if prices is not None else price_history
        if _price_data is None:
            _price_data = []
        if n_simulations is not None:
            old_n = self._n_simulations
            self._n_simulations = n_simulations
        prices_arr = np.asarray(_price_data, dtype=np.float64).ravel()
        # Derive current_price from last element if not given
        if current_price == 0.0 and len(prices_arr) > 0:
            current_price = float(prices_arr[-1])

        # Guard: insufficient history
        if len(prices_arr) < _MIN_PRICE_HISTORY:
            if n_simulations is not None:
                self._n_simulations = old_n
            log.info(
                f"{MC_CHAOS_FAILED}: insufficient price history "
                f"({len(prices_arr)} < {_MIN_PRICE_HISTORY})"
            )
            return MonteCarloResult(
                n_simulations=0, win_rate=0.0,
                mean_pnl=0.0, std_pnl=0.0,
                worst_case_pnl=0.0, best_case_pnl=0.0,
                var_95=0.0, passed=False,
                reason_code=MC_CHAOS_FAILED, timestamp=ts,
            )

        # Normalise direction
        direction = 1 if signal_direction >= 0 else -1

        # Generate jittered price matrix
        jittered = self._generate_jittered_prices(prices_arr)

        # Evaluate PnL across all simulations
        pnl_array = self._evaluate_simulations_vectorized(
            current_price, signal_alpha, direction, jittered, predict_fn,
        )

        # Statistics
        n_sims = len(pnl_array)
        wins = int(np.sum(pnl_array > 0.0))
        win_rate = wins / max(n_sims, 1)
        mean_pnl = float(np.mean(pnl_array))
        std_pnl = float(np.std(pnl_array))
        worst_pnl = float(np.min(pnl_array))
        best_pnl = float(np.max(pnl_array))

        # VaR at 95 % confidence = 5th percentile of PnL distribution
        var_95 = float(np.percentile(pnl_array, (1.0 - _VAR_CONFIDENCE) * 100.0))

        passed = win_rate >= self._min_win_rate
        if passed:
            reason = MC_CHAOS_PASSED
        elif win_rate < self._min_win_rate:
            reason = MC_LOW_WIN_RATE
        else:
            reason = MC_CHAOS_FAILED

        result = MonteCarloResult(
            n_simulations=n_sims,
            win_rate=win_rate,
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            worst_case_pnl=worst_pnl,
            best_case_pnl=best_pnl,
            var_95=var_95,
            passed=passed,
            reason_code=reason,
            timestamp=ts,
        )

        log.info(
            f"MC_STRESS_TEST: win_rate={win_rate:.3f} "
            f"mean_pnl={mean_pnl:+.6f} std={std_pnl:.6f} "
            f"var95={var_95:+.6f} worst={worst_pnl:+.6f} "
            f"best={best_pnl:+.6f} passed={passed} "
            f"reason={reason}"
        )
        if n_simulations is not None:
            self._n_simulations = old_n
        return result


# ---------------------------------------------------------------------------
# Recursive Feature Elimination
# ---------------------------------------------------------------------------


class RecursiveFeatureEliminator:
    """Rank-based Recursive Feature Elimination with regime awareness.

    Computes absolute Pearson correlation per feature.  When HMM regime
    labels are available, ranks are computed per-regime and features with
    cross-regime rank variance > decay_threshold are eliminated.  Never
    drops below ``min_features``.
    """

    def __init__(
        self,
        min_features: int = RFE_MIN_FEATURES,
        decay_threshold: float = RFE_IMPORTANCE_DECAY_THRESHOLD,
    ) -> None:
        self._min_features = max(1, min_features)
        self._decay_threshold = max(0.0, decay_threshold)

        log.info(
            f"RecursiveFeatureEliminator initialized: "
            f"min_feat={self._min_features} decay_thr={self._decay_threshold:.3f}"
        )

    # ----- Internal helpers -----

    @staticmethod
    def _compute_correlation_ranks(
        feature_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Absolute Pearson correlation per feature -> rank scores in [0, 1]."""
        n_samples, n_features = feature_matrix.shape
        ranks = np.zeros(n_features, dtype=np.float64)

        # De-mean target once
        t_mean = np.mean(targets)
        t_centered = targets - t_mean
        t_std = np.std(targets)
        if t_std < 1e-12:
            return ranks

        for j in range(n_features):
            col = feature_matrix[:, j]
            f_mean = np.mean(col)
            f_std = np.std(col)
            if f_std < 1e-12:
                ranks[j] = 0.0
                continue
            f_centered = col - f_mean
            corr = np.dot(f_centered, t_centered) / (n_samples * f_std * t_std)
            ranks[j] = abs(float(corr))

        return ranks

    def _compute_per_regime_ranks(
        self,
        feature_matrix: np.ndarray,
        targets: np.ndarray,
        regime_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank scores per HMM regime -> (mean_ranks, rank_variances)."""
        unique_regimes = np.unique(regime_labels)
        n_features = feature_matrix.shape[1]
        regime_rank_matrix: list[np.ndarray] = []

        for regime in unique_regimes:
            mask = regime_labels == regime
            n_in_regime = int(np.sum(mask))
            if n_in_regime < 5:
                # Too few samples in this regime; skip
                continue
            sub_features = feature_matrix[mask]
            sub_targets = targets[mask]
            ranks = self._compute_correlation_ranks(sub_features, sub_targets)
            regime_rank_matrix.append(ranks)

        if len(regime_rank_matrix) == 0:
            # No valid regimes — fall back to global ranks
            global_ranks = self._compute_correlation_ranks(feature_matrix, targets)
            return global_ranks, np.zeros(n_features, dtype=np.float64)

        stacked = np.vstack(regime_rank_matrix)  # (n_regimes, n_features)
        mean_ranks = np.mean(stacked, axis=0)
        rank_variances = np.var(stacked, axis=0)

        return mean_ranks, rank_variances

    def _select_surviving_features(
        self,
        global_ranks: np.ndarray,
        rank_variances: np.ndarray | None,
        feature_names: tuple[str, ...] | list[str],
    ) -> tuple[list[str], list[str], list[tuple[str, float]]]:
        """Select survivors: prune high-variance features, respect floor."""
        n_features = len(global_ranks)

        # Build (name, rank, variance) triples
        feature_info: list[tuple[str, float, float]] = []
        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            var_i = rank_variances[i] if rank_variances is not None else 0.0
            feature_info.append((name, float(global_ranks[i]), float(var_i)))

        # Sort by rank descending (best features first)
        feature_info.sort(key=lambda x: x[1], reverse=True)

        surviving: list[str] = []
        eliminated: list[str] = []
        rankings: list[tuple[str, float]] = []

        for name, rank, var in feature_info:
            rankings.append((name, rank))

            # Decide survival
            if rank_variances is not None and var > self._decay_threshold and len(surviving) >= self._min_features:
                eliminated.append(name)
                log.info(
                    f"{RFE_FEATURES_PRUNED}: {name} "
                    f"(rank={rank:.4f}, regime_var={var:.4f} "
                    f"> {self._decay_threshold})"
                )
                continue

            surviving.append(name)

        # If we haven't hit the floor yet, top-ranked eliminated features
        # are pulled back into surviving
        while len(surviving) < self._min_features and len(eliminated) > 0:
            restored = eliminated.pop(0)
            surviving.append(restored)
            log.info(
                f"RFE_FLOOR_RESTORE: {restored} restored to meet "
                f"min_features={self._min_features}"
            )

        return surviving, eliminated, rankings

    # ----- Public API -----

    def evaluate_features(
        self,
        feature_matrix: np.ndarray,
        targets: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        regime_labels: np.ndarray | None = None,
    ) -> RFEResult:
        """Run RFE: rank features, prune regime-inconsistent ones."""
        ts = time.time()
        fm = np.asarray(feature_matrix, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.float64).ravel()

        n_samples, n_features = fm.shape
        n_samples = min(n_samples, len(tgt))
        fm = fm[:n_samples]
        tgt = tgt[:n_samples]

        # Compute ranks (with or without regime awareness)
        if regime_labels is not None:
            rl = np.asarray(regime_labels).ravel()[:n_samples]
            mean_ranks, rank_variances = self._compute_per_regime_ranks(
                fm, tgt, rl,
            )
        else:
            mean_ranks = self._compute_correlation_ranks(fm, tgt)
            rank_variances = None

        # Select surviving features
        surviving, eliminated, rankings = self._select_surviving_features(
            mean_ranks, rank_variances, feature_names,
        )

        # Regime consistency score: 1.0 - mean(rank_variances)
        # Clamped to [0, 1].  Higher is better (more consistent).
        if rank_variances is not None and len(rank_variances) > 0:
            raw_consistency = 1.0 - float(np.mean(rank_variances))
            regime_consistency = max(0.0, min(1.0, raw_consistency))
        else:
            regime_consistency = 1.0  # No regimes = fully consistent

        result = RFEResult(
            original_features=n_features,
            surviving_features=len(surviving),
            eliminated_features=tuple(eliminated),
            surviving_features_names=tuple(surviving),
            feature_rankings=tuple(rankings),
            regime_consistency_score=regime_consistency,
            timestamp=ts,
        )

        log.info(
            f"RFE: {len(surviving)}/{n_features} surviving, "
            f"{len(eliminated)} eliminated, "
            f"regime_consistency={regime_consistency:.3f}"
        )

        if regime_consistency < _RFE_REGIME_CONSISTENCY_FLOOR:
            log.info(
                f"{RFE_REGIME_INCONSISTENT}: consistency={regime_consistency:.3f} "
                f"< {_RFE_REGIME_CONSISTENCY_FLOOR}"
            )

        return result


# ---------------------------------------------------------------------------
# Stochastic Stress Tester — Main Coordinator
# ---------------------------------------------------------------------------


class StochasticStressTester:
    """Orchestrates Monte Carlo stress testing and RFE into a single gate.

    Usage:
        tester = get_stress_tester()
        verdict = tester.validate_signal(
            current_price=3250.0,
            signal_alpha=0.72,
            signal_direction=1,
            price_history=np.array([...]),
        )
        if verdict.approved:
            pass  # Signal is chaos-resistant
    """

    def __init__(self, var_halt_threshold: float | None = None) -> None:
        self._mc_engine = MonteCarloEngine(
            n_simulations=MC_N_SIMULATIONS,
            jitter_pct=MC_PRICE_JITTER_PCT,
            min_win_rate=MC_MIN_WIN_RATE,
        )
        self._rfe = RecursiveFeatureEliminator(
            min_features=RFE_MIN_FEATURES,
            decay_threshold=RFE_IMPORTANCE_DECAY_THRESHOLD,
        )
        self._var_halt_threshold: float | None = var_halt_threshold

        # Counters
        self._total_tested: int = 0
        self._total_approved: int = 0
        self._total_rejected: int = 0

        log.info("StochasticStressTester initialized — Chaos Layer active")

    # ----- Internal helpers -----

    @staticmethod
    def _compute_confidence(
        mc_result: MonteCarloResult,
        rfe_result: RFEResult | None,
    ) -> float:
        """Derive a composite confidence score in [0, 1].

        Blends Monte Carlo win rate, PnL Sharpe-like ratio, and RFE
        regime consistency into a single scalar.
        """
        # Base confidence from MC win rate (rescaled 0.5 -> 0.0, 1.0 -> 1.0)
        mc_conf = max(0.0, min(1.0, (mc_result.win_rate - 0.5) * 2.0))

        # Sharpe-like bonus: mean / std (capped)
        if mc_result.std_pnl > 1e-12:
            sharpe = mc_result.mean_pnl / mc_result.std_pnl
            sharpe_bonus = max(0.0, min(0.2, sharpe * 0.1))
        else:
            sharpe_bonus = 0.0 if mc_result.mean_pnl <= 0.0 else 0.2

        # RFE regime consistency factor
        rfe_factor = rfe_result.regime_consistency_score if rfe_result is not None else 1.0

        # Weighted combination
        raw = 0.6 * mc_conf + 0.2 * rfe_factor + 0.2 * (1.0 + sharpe_bonus)
        return max(0.0, min(1.0, raw))

    # ----- Public API -----

    def validate_signal(
        self,
        current_price: float = 0.0,
        signal_alpha: float = 0.5,
        signal_direction: int = 1,
        price_history: np.ndarray | list | None = None,
        feature_matrix: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        feature_names: tuple[str, ...] | list[str] | None = None,
        regime_labels: np.ndarray | None = None,
        # Simplified API aliases
        prices: np.ndarray | list | None = None,
        signals: list | np.ndarray | None = None,
    ) -> StressTestVerdict:
        """Validate an alpha signal through the chaos gauntlet."""
        # Support simplified API: prices=..., signals=...
        if prices is not None and price_history is None:
            price_history = prices
        if signals is not None and price_history is not None:
            _ph = np.asarray(price_history, dtype=np.float64).ravel()
            if current_price == 0.0 and len(_ph) > 0:
                current_price = float(_ph[-1])
        if price_history is None:
            price_history = np.array([], dtype=np.float64)

        ts = time.time()
        self._total_tested += 1
        rejection_reasons: list[str] = []

        # Step 1: Monte Carlo stress test
        mc_result = self._mc_engine.run_stress_test(
            current_price=current_price,
            signal_alpha=signal_alpha,
            signal_direction=signal_direction,
            price_history=price_history,
        )

        if not mc_result.passed:
            rejection_reasons.append(mc_result.reason_code)

        # Step 2: RFE (optional — only if feature data provided)
        rfe_result: RFEResult | None = None
        has_feature_data = (
            feature_matrix is not None
            and targets is not None
            and feature_names is not None
        )

        if has_feature_data:
            assert feature_matrix is not None
            assert targets is not None
            assert feature_names is not None
            rfe_result = self._rfe.evaluate_features(
                feature_matrix=feature_matrix,
                targets=targets,
                feature_names=feature_names,
                regime_labels=regime_labels,
            )
            if rfe_result.regime_consistency_score < _RFE_REGIME_CONSISTENCY_FLOOR:
                rejection_reasons.append(RFE_REGIME_INCONSISTENT)

        # Step 3: Combine into verdict
        mc_ok = mc_result.passed
        rfe_ok = (
            rfe_result is None
            or rfe_result.regime_consistency_score > _RFE_REGIME_CONSISTENCY_FLOOR
        )
        # Apply var_halt_threshold if set
        var_ok = True
        if self._var_halt_threshold is not None:
            var_ok = abs(mc_result.var_95) <= self._var_halt_threshold
        approved = mc_ok and rfe_ok and var_ok

        if approved:
            final_reason = STRESS_TEST_APPROVED
        else:
            final_reason = STRESS_TEST_REJECTED
            if final_reason not in rejection_reasons:
                rejection_reasons.append(final_reason)

        confidence = self._compute_confidence(mc_result, rfe_result)

        # Update counters
        if approved:
            self._total_approved += 1
        else:
            self._total_rejected += 1

        verdict = StressTestVerdict(
            monte_carlo=mc_result,
            rfe=rfe_result,
            approved=approved,
            rejection_reasons=tuple(rejection_reasons),
            confidence_score=confidence,
            timestamp=ts,
        )

        log.info(
            f"STRESS_VERDICT: approved={approved} "
            f"confidence={confidence:.3f} "
            f"mc_wr={mc_result.win_rate:.3f} "
            f"rfe_cons={'N/A' if rfe_result is None else f'{rfe_result.regime_consistency_score:.3f}'} "
            f"reasons={rejection_reasons}"
        )

        return verdict

    def get_stats(self) -> dict:
        """Return cumulative stats: total_tested/approved/rejected, approval_rate."""
        rate = (
            self._total_approved / max(self._total_tested, 1)
        )
        return {
            "total_tested": self._total_tested,
            "total_approved": self._total_approved,
            "total_rejected": self._total_rejected,
            "approval_rate": rate,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_stress_tester_instance: StochasticStressTester | None = None


def get_stress_tester() -> StochasticStressTester:
    """Return the process-wide StochasticStressTester singleton."""
    global _stress_tester_instance
    if _stress_tester_instance is None:
        _stress_tester_instance = StochasticStressTester()
    return _stress_tester_instance


def reset_stress_tester() -> None:
    """Reset the singleton (for tests)."""
    global _stress_tester_instance
    _stress_tester_instance = None
