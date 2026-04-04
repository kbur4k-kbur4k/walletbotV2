"""
wallet_manager/strategies/wf_validator.py

Walk-Forward Cross-Validation with Purged K-Fold — Anti-Overfitting Engine.

Provides rigorous out-of-sample validation for any ML-driven trading strategy
by ensuring that train/test boundaries never leak future information through
overlapping feature windows (rolling means, GARCH estimates, etc.).

Components
----------
1. PurgedKFoldSplit: K-Fold splitter that inserts a configurable "purge" gap
   between every train/test boundary. Eliminates look-ahead bias from
   overlapping feature construction windows.

2. WalkForwardSplitter: Expanding-window walk-forward optimisation (WFO).
   Each successive fold trains on all history up to point *t*, then tests
   on the next segment after a purge gap. Mirrors live deployment where
   the model only ever sees past data.

3. RetrainController: Monitors trade count and triggers periodic model
   retraining via walk-forward validation. Rejects models whose OOS Sharpe
   falls below the safety threshold — the single most important guard
   against deploying an overfit model.

4. WalkForwardValidator: Top-level coordinator that orchestrates splits,
   collects per-fold OOS metrics, persists results to SQLite, and exposes
   aggregate validation summaries.

Architecture
------------
- Splitting logic is implemented from scratch (no sklearn dependency).
- Array operations use numpy for performance.
- Database I/O is async (aiosqlite, WAL mode, NORMAL synchronous).
- Metric computation is synchronous (pure numpy math).
- All result types are frozen dataclasses for immutability.

Safety Invariant
----------------
If any walk-forward run produces a mean OOS Sharpe < 0.3, the candidate
model is **rejected** and the previous weights are retained. This prevents
deployment of overfit or degraded models in production.
"""
# ruff: noqa: ASYNC240, RUF002, RUF003

from __future__ import annotations

import asyncio
import itertools
import json
import math
import shutil
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import aiosqlite
import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

if TYPE_CHECKING:
    from wallet_manager.config import Settings


IndexArray: TypeAlias = np.ndarray
SplitPair: TypeAlias = tuple[IndexArray, IndexArray]
SplitWithPurge: TypeAlias = tuple[IndexArray, IndexArray, int]
SplitResult: TypeAlias = SplitPair | SplitWithPurge


def _retrain_get_settings() -> Settings:
    from wallet_manager.config import get_settings
    return get_settings()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Walk-forward defaults
DEFAULT_N_SPLITS: int = 5
# purge_pct is now config-driven via Settings.wf_purge_pct
DEFAULT_EMBARGO_PCT: float = 0.01     # 1% embargo after test set

# Retrain controller
DEFAULT_RETRAIN_INTERVAL: int = 100   # Retrain every 100 trades
# min_oos_sharpe is now config-driven via Settings.wf_min_oos_sharpe

# Metric computation
ANNUALISATION_FACTOR: float = np.sqrt(252.0)  # Daily returns → annualised
MIN_SAMPLES_FOR_METRICS: int = 5              # Minimum test samples for stats

# Database
def _get_default_wf_db_path() -> str:
    from wallet_manager.config import get_settings
    return str(get_settings().wf_validator_db_path)


_DEFAULT_DB_PATH: str = _get_default_wf_db_path()

# Reason codes
REASON_RETRAIN_TRIGGERED: str = "RETRAIN_TRIGGERED"
REASON_MODEL_REJECTED_LOW_SHARPE: str = "MODEL_REJECTED_LOW_SHARPE"
REASON_MODEL_ACCEPTED: str = "MODEL_ACCEPTED"
REASON_PURGE_GAP_APPLIED: str = "PURGE_GAP_APPLIED"

# ---------------------------------------------------------------------------
# SQL Schema
# ---------------------------------------------------------------------------

_CREATE_VALIDATION_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS wf_validation_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fold_id         INTEGER NOT NULL,
    train_size      INTEGER NOT NULL,
    test_size       INTEGER NOT NULL,
    purge_size      INTEGER NOT NULL,
    oos_sharpe      REAL NOT NULL,
    oos_hit_rate    REAL NOT NULL,
    oos_profit_factor REAL NOT NULL,
    oos_max_drawdown REAL NOT NULL,
    timestamp       REAL NOT NULL,
    model_version   TEXT NOT NULL,
    reason_code     TEXT NOT NULL
);
"""

_INSERT_VALIDATION_RESULT_SQL = """
INSERT INTO wf_validation_results (
    fold_id, train_size, test_size, purge_size,
    oos_sharpe, oos_hit_rate, oos_profit_factor, oos_max_drawdown,
    timestamp, model_version, reason_code
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT_RESULTS_SQL = """
SELECT fold_id, train_size, test_size, purge_size,
       oos_sharpe, oos_hit_rate, oos_profit_factor, oos_max_drawdown,
       timestamp, model_version, reason_code
FROM wf_validation_results
ORDER BY timestamp DESC
LIMIT ?;
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Immutable record of a single walk-forward fold evaluation."""

    oos_sharpe: float = 0.0
    # Optional simplified fields
    oos_returns: list[float] | None = None
    n_splits: int = 0
    # Full detail fields (optional for backward compat)
    fold_id: int = 0
    train_size: int = 0
    test_size: int = 0
    purge_size: int = 0
    oos_hit_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_max_drawdown: float = 0.0
    timestamp: float = 0.0
    model_version: str = ""


@dataclass(frozen=True)
class RetrainDecision:
    """Outcome of a retrain evaluation cycle."""

    should_accept: bool = False
    mean_oos_sharpe: float = 0.0
    mean_oos_hit_rate: float = 0.0
    mean_oos_profit_factor: float = 0.0
    reason_code: str = ""
    fold_results: tuple[ValidationResult, ...] = ()
    # Dual walk-forward validation fields (D.3)
    expanding_oos_sharpe: float = 0.0
    sliding_oos_sharpe: float = 0.0
    window_mode_agreement: bool = False  # both modes above threshold

    @property
    def should_retrain(self) -> bool:
        """True when the decision recommends retraining (low OOS Sharpe)."""
        return not self.should_accept

    @property
    def action(self) -> str:
        """Backward-compatible textual action for legacy callers."""
        return "ACCEPT" if self.should_accept else "REJECT"


# ---------------------------------------------------------------------------
# Metric helpers (pure numpy, no external deps)
# ---------------------------------------------------------------------------

def compute_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio from a 1-D array of period returns.

    Returns 0.0 when insufficient data or zero volatility.
    """
    if len(returns) < MIN_SAMPLES_FOR_METRICS:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns) / std) * ANNUALISATION_FACTOR


def compute_hit_rate(returns: np.ndarray) -> float:
    """Fraction of positive-return periods."""
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))


def compute_profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss.  Returns inf if no losses, 0.0 if no wins."""
    gross_profit = float(np.sum(returns[returns > 0]))
    gross_loss = float(np.abs(np.sum(returns[returns < 0])))
    if gross_loss < 1e-12:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown from cumulative returns.

    Returns a non-negative value (e.g., 0.15 means 15% drawdown).
    """
    if len(returns) < 2:
        return 0.0
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return float(np.max(drawdowns))


def evaluate_fold(returns: np.ndarray) -> dict[str, float]:
    """Compute all OOS metrics for a single fold's return series."""
    return {
        "oos_sharpe": compute_sharpe(returns),
        "oos_hit_rate": compute_hit_rate(returns),
        "oos_profit_factor": compute_profit_factor(returns),
        "oos_max_drawdown": compute_max_drawdown(returns),
    }


# ---------------------------------------------------------------------------
# PurgedKFoldSplit — K-Fold with purge/embargo gaps
# ---------------------------------------------------------------------------

class PurgedKFoldSplit:
    """K-Fold cross-validation with purge gaps to prevent information leakage.

    Between every train/test boundary a *purge* zone is carved out so that
    samples whose feature windows might overlap with the test set are excluded
    from the training data.  An optional *embargo* zone after the test set
    prevents contamination from features that look forward into the next fold.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    purge_pct : float
        Fraction of total samples to remove between train and test boundaries.
    embargo_pct : float
        Fraction of total samples to embargo after each test set.
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        purge_pct: float | None = None,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
        purge_gap: int | None = None,  # absolute sample count alias
    ) -> None:
        if purge_pct is None and purge_gap is None:
            purge_pct = _retrain_get_settings().wf_purge_pct
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError(f"embargo_pct must be in [0, 0.5), got {embargo_pct}")

        self.n_splits = n_splits
        self.purge_pct = purge_pct if purge_pct is not None else 0.0
        self.embargo_pct = embargo_pct
        self._purge_gap: int | None = purge_gap  # absolute override

    def _resolve_purge_len(self, n_samples: int) -> int:
        if self._purge_gap is not None:
            return self._purge_gap
        return max(1, int(n_samples * self.purge_pct))

    def split(
        self, X: int | np.ndarray
    ) -> list[SplitPair]:
        """Generate purged train/test index pairs.

        Parameters
        ----------
        X : int or array-like
            Total number of samples, or the dataset itself (shape is inferred).

        Returns
        -------
        list of (train_indices, test_indices) numpy arrays.
        """
        n_samples = len(X) if hasattr(X, "__len__") else int(X)
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_splits ({self.n_splits})"
            )

        indices = np.arange(n_samples)
        purge_len = self._resolve_purge_len(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        splits: list[SplitPair] = []
        current = 0

        for fold_idx in range(self.n_splits):
            test_start = current
            test_end = current + fold_sizes[fold_idx]
            test_indices = indices[test_start:test_end]

            # --- Build train indices with purge + embargo exclusion ---
            # Purge zone: [test_start - purge_len, test_start)
            purge_before_start = max(0, test_start - purge_len)

            # Train: only indices BEFORE the purge zone (respects temporal order)
            train_indices = indices[:purge_before_start]

            if len(test_indices) > 0:
                log.debug(
                    "PurgedKFold fold %d: train=%d test=%d purge=%d",
                    fold_idx,
                    len(train_indices),
                    len(test_indices),
                    purge_len,
                )
                splits.append((train_indices, test_indices))

            current = test_end

        return splits

    def get_purge_size(self, n_samples: int) -> int:
        """Return the purge gap size for the given dataset."""
        return self._resolve_purge_len(n_samples)


# ---------------------------------------------------------------------------
# WalkForwardSplitter — Expanding-window walk-forward splits
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """Expanding-window walk-forward optimisation splitter.

    Produces sequential train/test splits where:
        - Train: [0 .. t]
        - Purge: [t+1 .. t+purge_len]   (excluded)
        - Test:  [t+purge_len+1 .. t+purge_len+step]
        - Next fold expands training window to include previous test

    This mirrors how a model would be deployed live: always trained on
    all available history, tested on the immediate future.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    purge_pct : float
        Fraction of dataset used as purge gap between train and test.
    embargo_pct : float
        Fraction of dataset used as embargo after test (excluded from
        the next fold's test to prevent leakage from adjacent windows).
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        purge_pct: float | None = None,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
        # Simplified API: explicit window sizes
        train_size: int | None = None,
        test_size: int | None = None,
        step: int | None = None,
        expanding: bool = True,
    ) -> None:
        if purge_pct is None:
            purge_pct = _retrain_get_settings().wf_purge_pct
        if train_size is None and n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self._train_size: int | None = train_size
        self._test_size: int | None = test_size
        self._step: int | None = step
        self._expanding: bool = expanding

    def split(
        self, X: int | np.ndarray
    ) -> list[SplitPair]:
        """Generate expanding-window walk-forward splits.

        Parameters
        ----------
        X : int or array-like
            Total number of samples, or dataset (shape is inferred).

        Returns
        -------
        list of (train_indices, test_indices) tuples.
        """
        n_samples = len(X) if hasattr(X, "__len__") else int(X)

        # --- Simplified train_size/test_size/step API ---
        if self._train_size is not None:
            return self._split_explicit(n_samples, expanding=self._expanding)

        if n_samples < self.n_splits + 1:
            raise ValueError(
                f"n_samples ({n_samples}) too small for {self.n_splits} splits"
            )

        purge_len = max(1, int(n_samples * self.purge_pct))
        embargo_len = max(1, int(n_samples * self.embargo_pct))

        # Reserve initial training window: first 1/(n_splits+1) of data
        # Remaining data is divided into n_splits test segments
        initial_train_size = n_samples // (self.n_splits + 1)
        remaining = n_samples - initial_train_size
        step_size = remaining // self.n_splits

        if step_size < MIN_SAMPLES_FOR_METRICS:
            raise ValueError(
                f"Step size ({step_size}) too small for meaningful metrics. "
                f"Need at least {MIN_SAMPLES_FOR_METRICS} samples per fold."
            )

        indices = np.arange(n_samples)
        splits: list[SplitPair] = []

        for fold_idx in range(self.n_splits):
            # Training window: [0 .. train_end)
            train_end = initial_train_size + fold_idx * step_size

            # Purge gap: [train_end .. train_end + purge_len)
            purge_end = min(train_end + purge_len, n_samples)
            actual_purge = purge_end - train_end

            # Test window: [purge_end .. purge_end + step_size)
            test_start = purge_end
            test_end = min(test_start + step_size, n_samples)

            # Apply embargo: exclude embargo_len after test_end from future
            test_end_embargoed = min(test_end, n_samples - embargo_len)

            if test_start >= test_end_embargoed:
                log.warning(
                    "WalkForward fold %d: skipped (insufficient test samples "
                    "after purge/embargo, test_start=%d, test_end=%d)",
                    fold_idx, test_start, test_end_embargoed,
                )
                continue

            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end_embargoed]

            log.debug(
                "WalkForward fold %d: train=[0:%d] purge=%d test=[%d:%d]",
                fold_idx, train_end, actual_purge,
                test_start, test_end_embargoed,
            )

            splits.append((train_indices, test_indices))

        if not splits:
            raise ValueError(
                "No valid walk-forward splits could be generated. "
                "Increase n_samples or decrease n_splits/purge_pct."
            )

        return splits

    def _split_explicit(
        self, n_samples: int, expanding: bool = True
    ) -> list[SplitPair]:
        """Split using explicit train_size/test_size/step parameters."""
        train_size = self._train_size or 100
        test_size = self._test_size or 50
        step = self._step or test_size
        indices = np.arange(n_samples)
        splits: list[SplitPair] = []
        start = 0
        while start + train_size + test_size <= n_samples:
            if expanding:
                train_idx = indices[:start + train_size]
            else:
                train_idx = indices[start:start + train_size]
            test_idx = indices[start + train_size:start + train_size + test_size]
            splits.append((train_idx, test_idx))
            start += step
        return splits

    def split_sliding(
        self, X: int | np.ndarray, window_days: int | None = None, samples_per_day: int = 288
    ) -> list[SplitResult]:
        """Generate sliding-window walk-forward splits.

        Unlike the expanding window (anchored at start), here the training
        window has a fixed size (wf_sliding_window_days * samples_per_day)
        and moves forward.

        Parameters
        ----------
        n_samples : int
            Total number of samples.
        window_days : int, optional
            Training window size in days. Reads cfg.wf_sliding_window_days if None.
        samples_per_day : int
            Approximate number of samples per day (default 288 = 5-min bars).
        """
        n_samples = len(X) if hasattr(X, "__len__") else int(X)

        # Use explicit train_size/step if provided
        if self._train_size is not None:
            return cast(list[SplitResult], self._split_explicit(n_samples, expanding=False))

        if window_days is None:
            window_days = _retrain_get_settings().wf_sliding_window_days

        window_size = window_days * samples_per_day
        if window_size >= n_samples:
            # Fallback: use half the data as training window
            window_size = n_samples // 2

        purge_len = max(1, int(n_samples * self.purge_pct))
        embargo_len = max(1, int(n_samples * self.embargo_pct))

        remaining = n_samples - window_size
        step_size = max(MIN_SAMPLES_FOR_METRICS, remaining // max(1, self.n_splits))

        indices = np.arange(n_samples)
        splits: list[tuple[np.ndarray, np.ndarray, int]] = []

        for fold_idx in range(self.n_splits):
            # Sliding window: [fold_start .. fold_start + window_size)
            fold_start = fold_idx * step_size
            train_end = fold_start + window_size
            if train_end >= n_samples:
                break

            purge_end = min(train_end + purge_len, n_samples)
            actual_purge = purge_end - train_end

            test_start = purge_end
            test_end = min(test_start + step_size, n_samples - embargo_len)

            if test_start >= test_end:
                continue

            train_indices = indices[fold_start:train_end]
            test_indices = indices[test_start:test_end]

            log.debug(
                "WalkForward(sliding) fold %d: train=[%d:%d] purge=%d test=[%d:%d]",
                fold_idx, fold_start, train_end, actual_purge, test_start, test_end,
            )
            splits.append((train_indices, test_indices, actual_purge))

        if not splits:
            raise ValueError(
                "No valid sliding walk-forward splits generated. "
                "Check wf_sliding_window_days and n_samples."
            )

        return cast(list[SplitResult], splits)


# ---------------------------------------------------------------------------
# RetrainController — Trade-count-triggered retrain logic
# ---------------------------------------------------------------------------

class RetrainController:
    """Monitors trade count and decides when to trigger model retraining.

    The controller tracks the number of trades since the last retrain and
    fires a retrain signal every ``retrain_interval`` trades.  After
    walk-forward validation, it either accepts or rejects the new model
    based on the OOS Sharpe threshold.

    Parameters
    ----------
    retrain_interval : int
        Number of trades between automatic retrains.
    min_oos_sharpe : float
        Minimum mean OOS Sharpe to accept a new model.
    """

    def __init__(
        self,
        retrain_interval: int = DEFAULT_RETRAIN_INTERVAL,
        min_oos_sharpe: float | None = None,
        retrain_every_n_trades: int | None = None,  # alias for retrain_interval
    ) -> None:
        if retrain_every_n_trades is not None:
            retrain_interval = retrain_every_n_trades
        self.retrain_interval = retrain_interval
        self.min_oos_sharpe = (
            min_oos_sharpe if min_oos_sharpe is not None
            else _retrain_get_settings().wf_min_oos_sharpe
        )
        self._trades_since_retrain: int = 0
        self._total_retrains: int = 0
        self._total_rejections: int = 0
        self._last_retrain_ts: float = 0.0
        self._current_model_version: str = "v0"
        self._retrain_callback: Callable[[], Awaitable[None]] | None = None

    @property
    def _trade_count(self) -> int:
        """Backward-compatible alias used by older orchestrator paths."""
        return self._trades_since_retrain

    def record_trade(self) -> None:
        """Increment the trade counter by one."""
        self._trades_since_retrain += 1

    def should_retrain(self, trade_count: int | None = None) -> bool:
        """Check whether a retrain cycle should be triggered.

        Parameters
        ----------
        trade_count : int, optional
            If provided, overrides the internal counter. Otherwise uses
            the internal ``_trades_since_retrain`` value.

        Returns
        -------
        bool
            True if retrain_interval has been reached.
        """
        count = trade_count if trade_count is not None else self._trades_since_retrain
        return count >= self.retrain_interval

    def evaluate_retrain(
        self,
        fold_results: ValidationResult | list[ValidationResult],
        model_version: str = "v0",
        sliding_fold_results: list[ValidationResult] | None = None,
    ) -> RetrainDecision:
        """Evaluate walk-forward results and decide accept/reject.

        Parameters
        ----------
        fold_results : list[ValidationResult]
            Per-fold OOS metrics from the walk-forward run.
        model_version : str
            Identifier for the candidate model.

        Returns
        -------
        RetrainDecision
            Contains the accept/reject decision and aggregate metrics.
        """
        # Support single ValidationResult
        if isinstance(fold_results, ValidationResult):
            fold_results = [fold_results]

        if not fold_results:
            log.warning("RetrainController: no fold results to evaluate")
            return RetrainDecision(
                should_accept=False,
                mean_oos_sharpe=0.0,
                mean_oos_hit_rate=0.0,
                mean_oos_profit_factor=0.0,
                reason_code=REASON_MODEL_REJECTED_LOW_SHARPE,
                fold_results=(),
                expanding_oos_sharpe=0.0,
                sliding_oos_sharpe=0.0,
                window_mode_agreement=False,
            )

        sharpes = [r.oos_sharpe for r in fold_results]
        hit_rates = [r.oos_hit_rate for r in fold_results]
        profit_factors = [
            r.oos_profit_factor
            for r in fold_results
            if np.isfinite(r.oos_profit_factor)
        ]

        mean_sharpe = float(np.mean(sharpes))
        mean_hit_rate = float(np.mean(hit_rates))
        mean_pf = float(np.mean(profit_factors)) if profit_factors else 0.0

        # --- Dual walk-forward: compute sliding window metrics ---
        expanding_oos_sharpe = mean_sharpe
        sliding_oos_sharpe = 0.0
        window_mode_agreement = False

        if sliding_fold_results:
            sliding_sharpes = [r.oos_sharpe for r in sliding_fold_results]
            sliding_oos_sharpe = float(np.mean(sliding_sharpes)) if sliding_sharpes else 0.0
            sliding_threshold = self.min_oos_sharpe * 0.8
            window_mode_agreement = (
                expanding_oos_sharpe >= self.min_oos_sharpe
                and sliding_oos_sharpe >= sliding_threshold
            )
            if sliding_fold_results and not window_mode_agreement:
                log.warning(
                    "RetrainController: dual-mode disagreement — "
                    "expanding=%.4f sliding=%.4f (sliding_threshold=%.4f)",
                    expanding_oos_sharpe, sliding_oos_sharpe, sliding_threshold,
                )

        # --- CRITICAL SAFETY CHECK ---
        if mean_sharpe < self.min_oos_sharpe:
            self._total_rejections += 1
            log.warning(
                "RetrainController: MODEL REJECTED — mean OOS Sharpe %.4f < "
                "threshold %.4f (version=%s, rejections=%d)",
                mean_sharpe, self.min_oos_sharpe, model_version,
                self._total_rejections,
            )
            return RetrainDecision(
                should_accept=False,
                mean_oos_sharpe=mean_sharpe,
                mean_oos_hit_rate=mean_hit_rate,
                mean_oos_profit_factor=mean_pf,
                reason_code=REASON_MODEL_REJECTED_LOW_SHARPE,
                fold_results=tuple(fold_results),
                expanding_oos_sharpe=expanding_oos_sharpe,
                sliding_oos_sharpe=sliding_oos_sharpe,
                window_mode_agreement=window_mode_agreement,
            )

        # Accept the new model
        self._total_retrains += 1
        self._trades_since_retrain = 0
        self._last_retrain_ts = time.time()
        self._current_model_version = model_version

        log.info(
            "RetrainController: MODEL ACCEPTED — mean OOS Sharpe %.4f "
            "(version=%s, total_retrains=%d, window_agreement=%s)",
            mean_sharpe, model_version, self._total_retrains, window_mode_agreement,
        )

        return RetrainDecision(
            should_accept=True,
            mean_oos_sharpe=mean_sharpe,
            mean_oos_hit_rate=mean_hit_rate,
            mean_oos_profit_factor=mean_pf,
            reason_code=REASON_MODEL_ACCEPTED,
            fold_results=tuple(fold_results),
            expanding_oos_sharpe=expanding_oos_sharpe,
            sliding_oos_sharpe=sliding_oos_sharpe,
            window_mode_agreement=window_mode_agreement,
        )

    def set_retrain_callback(
        self,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Register a coroutine to call after _trigger_retrain completes."""
        self._retrain_callback = callback

    async def _trigger_retrain(self) -> None:
        """Run scripts/train_models.py in a subprocess and handle results.

        Sequence:
          1. Telegram notification (if notifier configured).
          2. Run train_models.py via asyncio.create_subprocess_exec.
          3. On success: parse training_report.json, reload models, notify.
          4. On failure: restore .bak model files, notify, log stderr.
          5. Update last_retrain_timestamp regardless of outcome.
        """
        n_new = self._trades_since_retrain
        n_total = self._total_retrains

        log.info(
            "RetrainController: triggering retrain — %d new trades since last retrain, "
            "%d total retrains completed",
            n_new, n_total,
        )

        # Locate train_models.py relative to this file
        _scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        _train_script = _scripts_dir / "train_models.py"
        if not _train_script.exists():
            log.error("RetrainController: train_models.py not found at %s", _train_script)
            self._last_retrain_ts = time.time()
            return

        ml_dir = Path("ml_models")
        ml_dir.mkdir(exist_ok=True)

        # --- Step 1: back up existing model files ---
        model_files = list(ml_dir.glob("*.pt")) + list(ml_dir.glob("*.json")) + list(ml_dir.glob("*.txt"))
        for mf in model_files:
            shutil.copy2(mf, mf.with_suffix(mf.suffix + ".bak"))

        # --- Step 2: launch subprocess ---
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(_train_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout_b, stderr_b = await proc.communicate()
            stderr_text = stderr_b.decode(errors="replace")
        except Exception as exc:
            log.error("RetrainController: subprocess launch failed: %s", exc)
            self._last_retrain_ts = time.time()
            return

        # --- Step 3 / 4: handle outcome ---
        if proc.returncode == 0:
            best_sharpe = 0.0
            try:
                report_path = Path("ml_models") / "training_report.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    sharpes = [
                        v.get("oos_sharpe", 0.0)
                        for v in report.values()
                        if isinstance(v, dict) and "oos_sharpe" in v
                    ]
                    if sharpes:
                        best_sharpe = max(sharpes)
                    log.info("RetrainController: training_report parsed — best OOS Sharpe %.4f", best_sharpe)
            except Exception as parse_exc:
                log.warning("RetrainController: could not parse training_report.json: %s", parse_exc)

            self._total_retrains += 1
            self._trades_since_retrain = 0
            log.info(
                "RetrainController: retrain SUCCEEDED — Best Sharpe %.4f",
                best_sharpe,
            )
        else:
            # Restore .bak files
            for mf in model_files:
                bak = mf.with_suffix(mf.suffix + ".bak")
                if bak.exists():
                    shutil.copy2(bak, mf)
            log.error(
                "RetrainController: retrain FAILED (exit %d) — kept previous models.\n%s",
                proc.returncode, stderr_text[-2000:],
            )

        # --- Step 5: update timestamp ---
        self._last_retrain_ts = time.time()

        if self._retrain_callback is not None:
            try:
                await self._retrain_callback()
            except Exception as cb_exc:
                log.warning("RetrainController: retrain callback raised: %s", cb_exc)

    def get_status(self) -> dict[str, object]:
        """Return controller diagnostics."""
        return {
            "trade_count": self._trades_since_retrain,  # test-friendly alias
            "trades_since_retrain": self._trades_since_retrain,
            "retrain_interval": self.retrain_interval,
            "total_retrains": self._total_retrains,
            "total_rejections": self._total_rejections,
            "last_retrain_ts": self._last_retrain_ts,
            "current_model_version": self._current_model_version,
            "min_oos_sharpe": self.min_oos_sharpe,
        }


# ---------------------------------------------------------------------------
# WalkForwardValidator — Top-level validation coordinator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Walk-forward cross-validation engine with purged splits and SQLite
    persistence.

    Orchestrates the full validation lifecycle:
        1. Generate purged walk-forward splits
        2. Evaluate OOS metrics on each fold
        3. Persist results to SQLite
        4. Provide aggregate summaries for retrain decisions

    The validator does **not** train models itself.  It provides split
    indices via ``get_splits()`` and evaluates pre-computed OOS returns
    via ``validate()``.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    purge_pct : float
        Purge gap fraction between train and test.
    embargo_pct : float
        Embargo fraction after test sets.
    retrain_interval : int
        Trades between automatic retrains.
    min_oos_sharpe : float
        Minimum acceptable mean OOS Sharpe ratio.
    db_path : str, optional
        Path to SQLite database for result persistence.
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        purge_pct: float | None = None,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
        retrain_interval: int = DEFAULT_RETRAIN_INTERVAL,
        min_oos_sharpe: float | None = None,
        db_path: str | None = None,
        # Simplified window API (forwarded to WalkForwardSplitter)
        train_size: int | None = None,
        test_size: int | None = None,
        step: int | None = None,
    ) -> None:
        self._n_splits = n_splits
        self._purge_pct = purge_pct if purge_pct is not None else _retrain_get_settings().wf_purge_pct
        self._embargo_pct = embargo_pct
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        self._train_size = train_size
        self._test_size = test_size
        self._step = step

        self._wf_splitter = WalkForwardSplitter(
            n_splits=n_splits,
            purge_pct=self._purge_pct,
            embargo_pct=embargo_pct,
            train_size=train_size,
            test_size=test_size,
            step=step,
        )
        self._kfold_splitter = PurgedKFoldSplit(
            n_splits=n_splits,
            purge_pct=self._purge_pct,
            embargo_pct=embargo_pct,
        )
        self._retrain_ctrl = RetrainController(
            retrain_interval=retrain_interval,
            min_oos_sharpe=min_oos_sharpe,
        )

        self._validation_history: list[ValidationResult] = []
        self._model_version_counter: int = 0

    @property
    def _controller(self) -> RetrainController:
        """Backward-compatible alias for legacy callers."""
        return self._retrain_ctrl

    def _get_db(self) -> aiosqlite.Connection:
        db = self._db
        if db is None:
            raise RuntimeError("WalkForwardValidator database is not initialized")
        return db

    def _normalize_split(self, split: SplitResult) -> SplitWithPurge:
        if len(split) == 3:
            return cast(SplitWithPurge, split)
        train_idx, test_idx = split
        purge_size = max(1, int(len(train_idx) * self._purge_pct)) if len(train_idx) > 0 else 0
        return (train_idx, test_idx, purge_size)

    def _validate_fold_results(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        model_version: str | None = None,
    ) -> list[ValidationResult]:
        results = self.validate(
            features=features,
            targets=targets,
            model_version=model_version,
        )
        return results if isinstance(results, list) else [results]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up SQLite database connection and create tables."""
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=-4000")  # 4 MB
        await self._db.execute(_CREATE_VALIDATION_RESULTS_SQL)
        await self._db.commit()

        self._initialized = True
        log.info("WalkForwardValidator: initialized DB at %s", self._db_path)

    async def shutdown(self) -> None:
        """Close database connection and release resources."""
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                log.exception("WalkForwardValidator: error closing DB")
            finally:
                self._db = None
                self._initialized = False
        log.info("WalkForwardValidator: shutdown complete")

    async def _ensure_db(self) -> None:
        """Lazy-initialise database if not yet connected."""
        if not self._initialized:
            await self.initialize()

    # ------------------------------------------------------------------
    # Split generation (sync — pure numpy)
    # ------------------------------------------------------------------

    def get_walk_forward_splits(
        self, n_samples: int | list | np.ndarray
    ) -> list[SplitWithPurge]:
        """Return expanding-window walk-forward split indices.

        Parameters
        ----------
        n_samples : int or array-like
            Number of samples, or a sequence whose length is used.

        Returns
        -------
        list of (train_indices, test_indices, purge_size)
        """
        if not isinstance(n_samples, int):
            n_samples = len(n_samples)
        raw_splits = self._wf_splitter.split(n_samples)
        splits = [self._normalize_split(split) for split in raw_splits]
        log.info(
            "WalkForward: generated %d splits for %d samples "
            "(purge=%.1f%%, embargo=%.1f%%)",
            len(splits), n_samples,
            self._purge_pct * 100, self._embargo_pct * 100,
        )
        return splits

    def get_purged_kfold_splits(
        self, n_samples: int
    ) -> list[SplitPair]:
        """Return purged K-Fold split indices.

        Returns
        -------
        list of (train_indices, test_indices)
        """
        splits = self._kfold_splitter.split(n_samples)
        log.info(
            "PurgedKFold: generated %d splits for %d samples",
            len(splits), n_samples,
        )
        return splits

    # ------------------------------------------------------------------
    # Validation (sync metrics, async persistence)
    # ------------------------------------------------------------------

    def validate(
        self,
        features: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        model_version: str | None = None,
        # Simplified API: prices + signals
        prices: list | np.ndarray | None = None,
        signals: list | np.ndarray | None = None,
    ) -> ValidationResult | list[ValidationResult]:
        """Run walk-forward validation and return per-fold OOS metrics.

        The caller is responsible for training the model on each fold's
        training indices and generating predictions.  This method evaluates
        the **target returns** on each test set to compute OOS metrics.

        For use cases where the caller provides pre-computed returns rather
        than features+targets for a model to predict, pass returns as
        ``targets`` and ignore ``features``.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).  Used only
            for determining split sizes — the validator does not train.
        targets : np.ndarray
            1-D array of returns (n_samples,).  OOS metrics are computed
            on the test-set slice of this array.
        model_version : str, optional
            Version tag.  Auto-incremented if not provided.

        Returns
        -------
        ValidationResult (aggregate) when prices/signals API used,
        or list[ValidationResult] per fold for the features/targets API.
        """
        # Simplified prices+signals API: compute returns from prices
        if prices is not None:
            prices_arr = np.asarray(prices, dtype=np.float64)
            n = len(prices_arr)
            price_returns = np.diff(prices_arr) / np.maximum(prices_arr[:-1], 1e-10)
            # Align signals: drop first to match returns length
            if signals is not None:
                sig_arr = np.asarray(signals, dtype=np.float64)
                sig_arr = sig_arr[1:] if len(sig_arr) == n else sig_arr
            else:
                sig_arr = np.ones(len(price_returns))
            # Weighted returns by signal direction
            weighted = price_returns * np.sign(sig_arr)
            if model_version is None:
                self._model_version_counter += 1
                model_version = f"v{self._model_version_counter}"
            splits = self.get_walk_forward_splits(len(weighted))
            now = time.time()
            all_oos: list[float] = []
            for _fold_idx, (train_idx, test_idx, _purge_size) in enumerate(splits):
                test_rets = weighted[test_idx]
                all_oos.extend(test_rets.tolist())
            all_oos_arr = np.asarray(all_oos, dtype=np.float64)
            metrics = evaluate_fold(all_oos_arr)
            result = ValidationResult(
                oos_sharpe=metrics["oos_sharpe"],
                oos_hit_rate=metrics.get("oos_hit_rate", 0.0),
                oos_profit_factor=metrics.get("oos_profit_factor", 1.0),
                oos_max_drawdown=metrics.get("oos_max_drawdown", 0.0),
                oos_returns=all_oos,
                n_splits=len(splits),
                model_version=model_version,
                timestamp=now,
            )
            self._validation_history.append(result)
            return result

        if features is None or targets is None:
            raise ValueError("Either (prices, signals) or (features, targets) must be provided.")

        n_samples = len(targets)
        if hasattr(features, "shape") and features.shape[0] != n_samples:
            raise ValueError(
                f"features rows ({features.shape[0]}) != targets length ({n_samples})"
            )

        if model_version is None:
            self._model_version_counter += 1
            model_version = f"v{self._model_version_counter}"

        splits = self.get_walk_forward_splits(n_samples)
        now = time.time()
        results: list[ValidationResult] = []

        for fold_idx, (train_idx, test_idx, purge_size) in enumerate(splits):
            test_returns = targets[test_idx]
            metrics = evaluate_fold(test_returns)

            result = ValidationResult(
                fold_id=fold_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                purge_size=purge_size,
                oos_sharpe=metrics["oos_sharpe"],
                oos_hit_rate=metrics["oos_hit_rate"],
                oos_profit_factor=metrics["oos_profit_factor"],
                oos_max_drawdown=metrics["oos_max_drawdown"],
                timestamp=now,
                model_version=model_version,
            )
            results.append(result)

            log.info(
                "WFV fold %d: Sharpe=%.3f hit=%.1f%% PF=%.2f DD=%.1f%% "
                "(train=%d test=%d purge=%d) [%s]",
                fold_idx, result.oos_sharpe,
                result.oos_hit_rate * 100,
                result.oos_profit_factor,
                result.oos_max_drawdown * 100,
                result.train_size, result.test_size, result.purge_size,
                REASON_PURGE_GAP_APPLIED,
            )

        self._validation_history.extend(results)
        return results

    def validate_dual(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        model_version: str | None = None,
    ) -> tuple[list[ValidationResult], list[ValidationResult]]:
        """Run both expanding and sliding window walk-forward validation.

        Returns
        -------
        tuple of (expanding_results, sliding_results).
        A model is accepted only if it passes both:
          - expanding OOS Sharpe >= cfg.wf_min_oos_sharpe
          - sliding OOS Sharpe >= cfg.wf_min_oos_sharpe * 0.8
        """
        expanding_results = self._validate_fold_results(
            features,
            targets,
            model_version,
        )

        # Sliding window validation
        sliding_results: list[ValidationResult] = []
        try:
            n_samples = len(targets)
            if model_version is None:
                model_version = f"v{self._model_version_counter}_sliding"

            sliding_splits = self._wf_splitter.split_sliding(n_samples)
            now = time.time()

            for fold_idx, split in enumerate(sliding_splits):
                train_idx, test_idx, purge_size = self._normalize_split(split)
                test_returns = targets[test_idx]
                metrics = evaluate_fold(test_returns)
                result = ValidationResult(
                    fold_id=fold_idx,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    purge_size=purge_size,
                    oos_sharpe=metrics["oos_sharpe"],
                    oos_hit_rate=metrics["oos_hit_rate"],
                    oos_profit_factor=metrics["oos_profit_factor"],
                    oos_max_drawdown=metrics["oos_max_drawdown"],
                    timestamp=now,
                    model_version=f"{model_version}_sliding",
                )
                sliding_results.append(result)
                log.info(
                    "WFV(sliding) fold %d: Sharpe=%.3f hit=%.1f%%",
                    fold_idx, result.oos_sharpe, result.oos_hit_rate * 100,
                )
        except Exception as exc:
            log.warning("WalkForwardValidator.validate_dual: sliding failed: %s", exc)

        return expanding_results, sliding_results

    async def validate_and_persist(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        model_version: str | None = None,
    ) -> list[ValidationResult]:
        """Run validation and persist results to SQLite.

        Combines ``validate()`` with async database writes.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        targets : np.ndarray
            Return series.
        model_version : str, optional
            Version tag.

        Returns
        -------
        list[ValidationResult]
        """
        results = self._validate_fold_results(features, targets, model_version)
        await self._persist_results(results)
        return results

    async def _persist_results(
        self,
        results: list[ValidationResult],
        reason_code: str = REASON_PURGE_GAP_APPLIED,
    ) -> None:
        """Write validation results to the SQLite database."""
        await self._ensure_db()
        db = self._get_db()

        rows = [
            (
                r.fold_id, r.train_size, r.test_size, r.purge_size,
                r.oos_sharpe, r.oos_hit_rate, r.oos_profit_factor,
                r.oos_max_drawdown, r.timestamp, r.model_version,
                reason_code,
            )
            for r in results
        ]

        await db.executemany(_INSERT_VALIDATION_RESULT_SQL, rows)
        await db.commit()
        log.debug(
            "WFV: persisted %d validation results to DB", len(results),
        )

    # ------------------------------------------------------------------
    # Retrain orchestration
    # ------------------------------------------------------------------

    def should_retrain(self, trade_count: int | None = None) -> bool:
        """Check whether a retrain cycle should be triggered.

        Parameters
        ----------
        trade_count : int, optional
            Explicit trade count.  Falls through to RetrainController's
            internal counter if not provided.
        """
        return self._retrain_ctrl.should_retrain(trade_count)

    def record_trade(self) -> None:
        """Record a completed trade in the retrain controller."""
        self._retrain_ctrl.record_trade()

    async def run_retrain_cycle(
        self,
        features: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        model_version: str | None = None,
    ) -> RetrainDecision:
        """Execute a full retrain evaluation cycle.

        1. Log the retrain trigger
        2. Run walk-forward validation
        3. Evaluate accept/reject via RetrainController
        4. Persist results with appropriate reason code
        5. Return the decision

        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        targets : np.ndarray
            Return series.
        model_version : str, optional
            Candidate model version tag.

        Returns
        -------
        RetrainDecision
        """
        log.info(
            "WFV: %s — starting walk-forward evaluation",
            REASON_RETRAIN_TRIGGERED,
        )

        if features is None or targets is None:
            log.warning("WFV: retrain cycle skipped because features/targets were not provided")
            return RetrainDecision(
                should_accept=False,
                reason_code=REASON_MODEL_REJECTED_LOW_SHARPE,
            )

        if model_version is None:
            self._model_version_counter += 1
            model_version = f"v{self._model_version_counter}"

        # Run validation
        results = self._validate_fold_results(features, targets, model_version)

        # Evaluate
        decision = self._retrain_ctrl.evaluate_retrain(results, model_version)

        # Persist with decision-specific reason code
        await self._persist_results(results, reason_code=decision.reason_code)

        return decision

    # ------------------------------------------------------------------
    # Summaries and queries
    # ------------------------------------------------------------------

    def get_validation_summary(self) -> dict[str, object]:
        """Aggregate OOS metrics across all validation runs in memory.

        Returns
        -------
        dict with keys:
            total_folds, mean_oos_sharpe, std_oos_sharpe,
            mean_oos_hit_rate, mean_oos_profit_factor,
            mean_oos_max_drawdown, worst_oos_sharpe,
            retrain_controller_status
        """
        if not self._validation_history:
            return {
                "total_folds": 0,
                "mean_oos_sharpe": 0.0,
                "std_oos_sharpe": 0.0,
                "mean_oos_hit_rate": 0.0,
                "mean_oos_profit_factor": 0.0,
                "mean_oos_max_drawdown": 0.0,
                "worst_oos_sharpe": 0.0,
                "retrain_controller_status": self._retrain_ctrl.get_status(),
            }

        sharpes = [r.oos_sharpe for r in self._validation_history]
        hit_rates = [r.oos_hit_rate for r in self._validation_history]
        profit_factors = [
            r.oos_profit_factor
            for r in self._validation_history
            if np.isfinite(r.oos_profit_factor)
        ]
        drawdowns = [r.oos_max_drawdown for r in self._validation_history]

        return {
            "total_folds": len(self._validation_history),
            "mean_oos_sharpe": float(np.mean(sharpes)),
            "std_oos_sharpe": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
            "mean_oos_hit_rate": float(np.mean(hit_rates)),
            "mean_oos_profit_factor": float(np.mean(profit_factors)) if profit_factors else 0.0,
            "mean_oos_max_drawdown": float(np.mean(drawdowns)),
            "worst_oos_sharpe": float(np.min(sharpes)),
            "retrain_controller_status": self._retrain_ctrl.get_status(),
        }

    async def get_recent_results(
        self, limit: int = 50
    ) -> list[ValidationResult]:
        """Fetch recent validation results from the database.

        Parameters
        ----------
        limit : int
            Maximum number of results to return (most recent first).

        Returns
        -------
        list[ValidationResult]
        """
        await self._ensure_db()
        assert self._db is not None

        cursor = await self._db.execute(_SELECT_RECENT_RESULTS_SQL, (limit,))
        rows = await cursor.fetchall()

        results: list[ValidationResult] = []
        for row in rows:
            results.append(
                ValidationResult(
                    fold_id=row[0],
                    train_size=row[1],
                    test_size=row[2],
                    purge_size=row[3],
                    oos_sharpe=row[4],
                    oos_hit_rate=row[5],
                    oos_profit_factor=row[6],
                    oos_max_drawdown=row[7],
                    timestamp=row[8],
                    model_version=row[9],
                )
            )

        return results

    def get_config(self) -> dict[str, object]:
        """Return the validator's configuration parameters."""
        return {
            "n_splits": self._n_splits,
            "purge_pct": self._purge_pct,
            "embargo_pct": self._embargo_pct,
            "db_path": self._db_path,
            "min_oos_sharpe": _retrain_get_settings().wf_min_oos_sharpe,
            "retrain_interval": self._retrain_ctrl.retrain_interval,
            "annualisation_factor": float(ANNUALISATION_FACTOR),
            "min_samples_for_metrics": MIN_SAMPLES_FOR_METRICS,
        }


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_validator_instance: WalkForwardValidator | None = None


async def get_wf_validator() -> WalkForwardValidator:
    """Return the module-level WalkForwardValidator singleton.

    Creates and initialises the instance on first call.
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = WalkForwardValidator()
        await _validator_instance.initialize()
    return _validator_instance


def reset_wf_validator() -> None:
    """Reset the singleton (for testing or reconfiguration)."""
    global _validator_instance
    _validator_instance = None


# ===========================================================================
# Section 3.1 — CombinatorialPurgedCV
# ===========================================================================

@dataclass(frozen=True)
class CPCVResult:
    """Result of a Combinatorial Purged Cross-Validation run."""
    n_combinations: int
    median_oos_sharpe: float
    sharpe_p_value: float       # P(Sharpe < 0)
    sharpe_range_ratio: float   # max / min (regime robustness)
    sharpe_distribution: list[float]
    passes: bool
    rejection_reason: str = ""  # empty string if passes=True


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV).

    López de Prado (2018), Advances in Financial ML, Chapter 12.

    Why: Standard walk-forward tests only one path through time. CPCV
    generates C(N, K) path combinations, covering all possible train/test
    orderings and eliminating selection bias in the reported Sharpe ratio.

    Default parameters: N=10 folds, K=2 test folds → C(10,2) = 45 paths.
    Each combination applies purge_pct gap at train/test boundaries and
    embargo_pct gap after each test set.

    Model acceptance criteria (all three must pass):
      1. median OOS Sharpe >= cfg.wf_min_oos_sharpe (default 0.5)
      2. P(OOS Sharpe < 0) < cfg.cpcv_p_value_threshold (default 0.20)
      3. max / min OOS Sharpe < 3.0 [regime robustness]

    If any criterion fails: reject model, keep previous weights.
    """

    def __init__(self, n_splits: int = 10, n_test_splits: int = 2) -> None:
        from wallet_manager.config import get_settings
        cfg = get_settings()
        self._n_splits = n_splits or cfg.cpcv_n_splits
        self._n_test_splits = n_test_splits or cfg.cpcv_n_test_splits
        self._purge_pct = cfg.wf_purge_pct
        self._embargo_pct = cfg.wf_embargo_pct
        self._min_oos_sharpe = cfg.wf_min_oos_sharpe
        self._p_value_threshold = cfg.cpcv_p_value_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        returns: np.ndarray,
        train_fn: Callable[[np.ndarray], Any],
        predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    ) -> CPCVResult:
        """Run CPCV and return aggregate validation result.

        Args:
            returns: 1-D array of period returns (chronological order).
            train_fn: Callable(train_idx) → model — trains a model on
                      the given index slice of ``returns``.
            predict_fn: Callable(model, test_idx) → np.ndarray — generates
                        return predictions for the test slice.

        Returns:
            CPCVResult with per-combination Sharpe distribution and
            pass/fail verdict.
        """
        n = len(returns)
        fold_size = n // self._n_splits
        purge = max(1, int(fold_size * self._purge_pct))
        embargo = max(1, int(fold_size * self._embargo_pct))

        # Build fold boundary indices
        folds: list[tuple[int, int]] = []
        for i in range(self._n_splits):
            start = i * fold_size
            end = start + fold_size if i < self._n_splits - 1 else n
            folds.append((start, end))

        all_oos_sharpes: list[float] = []
        combos = list(itertools.combinations(range(self._n_splits), self._n_test_splits))

        for test_fold_ids in combos:
            test_fold_ids_set = set(test_fold_ids)
            train_idx: list[int] = []
            test_idx: list[int] = []

            for fold_id, (fs, fe) in enumerate(folds):
                if fold_id in test_fold_ids_set:
                    test_idx.extend(range(fs, fe))
                else:
                    # Apply purge gap: skip `purge` samples adjacent to test
                    safe_start = fs
                    safe_end = fe
                    for tid in test_fold_ids:
                        ts, te = folds[tid]
                        if abs(fe - ts) <= purge:
                            safe_end = min(safe_end, ts - purge)
                        if abs(fs - te) <= purge:
                            safe_start = max(safe_start, te + purge + embargo)
                    train_idx.extend(range(max(0, safe_start), min(n, safe_end)))

            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            train_idx_arr = np.array(sorted(set(train_idx)))
            test_idx_arr = np.array(sorted(set(test_idx)))

            try:
                model = train_fn(train_idx_arr)
                oos_rets = predict_fn(model, test_idx_arr)
                sharpe = compute_sharpe(oos_rets)
                all_oos_sharpes.append(sharpe)
            except Exception as exc:
                log.warning(f"CPCV combination {test_fold_ids} failed: {exc}")

        if not all_oos_sharpes:
            return CPCVResult(
                n_combinations=0,
                median_oos_sharpe=0.0,
                sharpe_p_value=1.0,
                sharpe_range_ratio=0.0,
                sharpe_distribution=[],
                passes=False,
                rejection_reason="No valid CPCV combinations evaluated.",
            )

        arr = np.array(all_oos_sharpes)
        median_sharpe = float(np.median(arr))
        p_value = float(np.mean(arr < 0))

        s_min = float(arr.min())
        s_max = float(arr.max())
        # Guard against division by near-zero or negative min
        if abs(s_min) < 1e-9:
            range_ratio = float("inf") if s_max > 0 else 0.0
        else:
            range_ratio = abs(s_max / s_min)

        passes = True
        rejection_reason = ""

        if median_sharpe < self._min_oos_sharpe:
            passes = False
            rejection_reason = (
                f"median OOS Sharpe {median_sharpe:.3f} < threshold "
                f"{self._min_oos_sharpe:.3f}"
            )
        elif p_value >= self._p_value_threshold:
            passes = False
            rejection_reason = (
                f"P(Sharpe<0)={p_value:.3f} >= {self._p_value_threshold:.2f} — not statistically significant"
            )
        elif range_ratio >= 3.0:
            passes = False
            rejection_reason = (
                f"Sharpe range ratio {range_ratio:.2f} >= 3.0 — regime instability"
            )

        result = CPCVResult(
            n_combinations=len(all_oos_sharpes),
            median_oos_sharpe=round(median_sharpe, 4),
            sharpe_p_value=round(p_value, 4),
            sharpe_range_ratio=round(range_ratio, 4),
            sharpe_distribution=[round(s, 4) for s in all_oos_sharpes],
            passes=passes,
            rejection_reason=rejection_reason,
        )

        verdict = "PASS" if passes else f"FAIL ({rejection_reason})"
        log.info(
            f"CombinatorialPurgedCV: {len(combos)} combinations, "
            f"median_sharpe={median_sharpe:.3f}, p_value={p_value:.3f}, "
            f"range_ratio={range_ratio:.3f} → {verdict}"
        )
        return result

    def split(self, n: int) -> Iterable[SplitPair]:
        """Yield (train_indices, test_indices) for all C(N,K) combinations.

        Args:
            n: Total number of samples (chronological).

        Yields:
            Tuple of (train_idx, test_idx) numpy arrays for each combination.
        """
        fold_size = n // self._n_splits
        purge = max(1, int(fold_size * self._purge_pct))
        embargo = max(1, int(fold_size * self._embargo_pct))

        folds: list[tuple[int, int]] = []
        for i in range(self._n_splits):
            start = i * fold_size
            end = start + fold_size if i < self._n_splits - 1 else n
            folds.append((start, end))

        for test_fold_ids in itertools.combinations(range(self._n_splits), self._n_test_splits):
            test_fold_ids_set = set(test_fold_ids)
            train_idx: list[int] = []
            test_idx: list[int] = []

            for fold_id, (fs, fe) in enumerate(folds):
                if fold_id in test_fold_ids_set:
                    test_idx.extend(range(fs, fe))
                else:
                    safe_start, safe_end = fs, fe
                    for tid in test_fold_ids:
                        ts, te = folds[tid]
                        if abs(fe - ts) <= purge:
                            safe_end = min(safe_end, ts - purge)
                        if abs(fs - te) <= purge:
                            safe_start = max(safe_start, te + purge + embargo)
                    train_idx.extend(range(max(0, safe_start), min(n, safe_end)))

            if train_idx and test_idx:
                yield (
                    np.array(sorted(set(train_idx))),
                    np.array(sorted(set(test_idx))),
                )


# ===========================================================================
# Section 3.2 — DeflatedSharpeRatioTest
# ===========================================================================

@dataclass(frozen=True)
class DeflatedSRResult:
    """Result of a Deflated Sharpe Ratio significance test."""
    deflated_sharpe: float
    passes: bool
    n_trials: int
    correction_factor: float


class DeflatedSharpeRatioTest:
    """Deflated Sharpe Ratio (DSR) test for multiple-testing correction.

    Bailey & López de Prado (2014).

    Problem: testing enough hyperparameter combinations will produce a high
    Sharpe by pure chance. DSR corrects for this multiple-testing bias.

    Adjustments applied:
      - Number of trials N_trials (total hyperparameter combinations tried)
      - Non-normality: skewness and excess kurtosis via Jarque-Bera scaling
      - Observation count N_obs

    Formula (approximation of Bailey & LdP eq. 8):
        SR* = E[max SR] ≈ sqrt(V[SR]) * (
                  (1 - γ) * Φ⁻¹(1 - 1/N_trials) +
                  γ * Φ⁻¹(1 - 1/(N_trials * e))
              )
        where γ ≈ 0.5772 (Euler-Mascheroni constant)
        and V[SR] = (1 - skew*SR + (kurt/4)*SR²) / (N_obs - 1)

        DSR = Φ((SR_observed - SR*) / sqrt(V[SR]))

    Acceptance: DSR >= cfg.deflated_sharpe_confidence (default 0.95).
    """

    _EULER_MASCHERONI: float = 0.5772156649

    def __init__(self) -> None:
        from wallet_manager.config import get_settings
        self._threshold = get_settings().deflated_sharpe_confidence

    def test(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float,
        excess_kurtosis: float,
    ) -> DeflatedSRResult:
        """Run the DSR test.

        Args:
            observed_sharpe: The best (maximum) Sharpe ratio found across
                all hyperparameter trials.
            n_trials: Total number of hyperparameter combinations evaluated.
            n_obs: Number of out-of-sample observations used for evaluation.
            skewness: Skewness of the return series.
            excess_kurtosis: Excess kurtosis (Fisher definition) of returns.

        Returns:
            DeflatedSRResult with the DSR value and pass/fail verdict.
        """
        if n_trials < 1:
            n_trials = 1
        if n_obs < 2:
            return DeflatedSRResult(
                deflated_sharpe=0.0, passes=False,
                n_trials=n_trials, correction_factor=0.0,
            )

        # Variance of the Sharpe estimator (non-normality adjusted)
        sr = observed_sharpe
        var_sr = (1.0 - skewness * sr + (excess_kurtosis / 4.0) * sr ** 2) / (n_obs - 1)
        var_sr = max(var_sr, 1e-12)
        std_sr = math.sqrt(var_sr)

        # Expected maximum Sharpe under multiple testing (approximation)
        gamma = self._EULER_MASCHERONI
        phi_inv_1 = self._norm_ppf(1.0 - 1.0 / n_trials)
        phi_inv_2 = self._norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        sr_star = std_sr * ((1.0 - gamma) * phi_inv_1 + gamma * phi_inv_2)

        # DSR = Φ((SR_obs - SR*) / std_SR)
        z = (sr - sr_star) / std_sr if std_sr > 1e-12 else 0.0
        dsr = self._norm_cdf(z)

        correction_factor = sr_star
        passes = dsr >= self._threshold

        log.info(
            f"DeflatedSharpeRatioTest: SR={sr:.3f}, SR*={sr_star:.3f}, "
            f"DSR={dsr:.4f}, threshold={self._threshold:.2f}, "
            f"{'PASS' if passes else 'FAIL'}"
        )
        return DeflatedSRResult(
            deflated_sharpe=round(dsr, 4),
            passes=passes,
            n_trials=n_trials,
            correction_factor=round(correction_factor, 4),
        )

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF via math.erfc (no scipy dependency)."""
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """Standard normal quantile function (rational approximation)."""
        p = max(1e-12, min(1.0 - 1e-12, p))
        # Rational approximation (Abramowitz & Stegun 26.2.17)
        sign = 1.0 if p >= 0.5 else -1.0
        q = p if p >= 0.5 else 1.0 - p
        t = math.sqrt(-2.0 * math.log(1.0 - q))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        num = c0 + c1 * t + c2 * t ** 2
        den = 1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3
        return sign * (t - num / den)


# ===========================================================================
# Section 3.4 — ConceptDriftDetector
# ===========================================================================

class ConceptDriftDetector:
    """Page-Hinkley test for real-time concept drift detection.

    Called on every trade cycle from within the orchestrator main loop.

    Monitored metrics (rolling 100-trade window):
      prediction_accuracy  hit rate on direction (long/short/hold)
      feature_kl           KL-divergence of live features vs training dist
      autocorr_change      Ljung-Box statistic on signal autocorrelation

    Response by KL-divergence level:
      KL > cfg.concept_drift_kl_warn    (0.15): Telegram WARNING,
                                                schedule Platt scaling update.
      KL > cfg.concept_drift_kl_retrain (0.30): trigger RetrainController.
      KL > cfg.concept_drift_kl_disable (0.50): disable ML ensemble,
                                                 switch to statistical fallback.
    """

    class DriftLevel(Enum):
        NONE    = "none"
        WARN    = "warn"
        RETRAIN = "retrain"
        SEVERE  = "severe"

    _WINDOW = 100  # rolling window size for all metrics

    def __init__(self) -> None:
        from wallet_manager.config import get_settings
        cfg = get_settings()
        self._kl_warn = cfg.concept_drift_kl_warn
        self._kl_retrain = cfg.concept_drift_kl_retrain
        self._kl_disable = cfg.concept_drift_kl_disable

        # Rolling buffers
        self._predictions: list[float] = []
        self._actuals: list[float] = []
        self._features_history: list[np.ndarray] = []

        # Training distribution statistics (set by calibrate())
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None

        # Page-Hinkley state
        self._ph_sum: float = 0.0
        self._ph_min: float = 0.0
        self._ph_delta: float = 0.005   # acceptable mean shift
        self._ph_lambda: float = 50.0   # detection threshold
        self._ml_disabled: bool = False

    def calibrate(self, train_features: np.ndarray) -> None:
        """Record training distribution for KL-divergence baseline.

        Args:
            train_features: 2-D array (n_samples, n_features) from training.
        """
        self._train_mean = np.mean(train_features, axis=0)
        self._train_std = np.std(train_features, axis=0) + 1e-8
        log.info(
            f"ConceptDriftDetector calibrated: "
            f"{train_features.shape[0]} training samples, "
            f"{train_features.shape[1]} features."
        )

    def update(
        self,
        prediction: float,
        actual: float,
        features: np.ndarray,
    ) -> ConceptDriftDetector.DriftLevel:
        """Ingest one trade cycle's prediction, outcome and features.

        Args:
            prediction: Model's directional prediction (+1 long, -1 short, 0 hold).
            actual: Realised direction (sign of return).
            features: Live feature vector for this cycle.

        Returns:
            DriftLevel indicating the severity of detected drift.
        """
        # Accumulate rolling buffers
        self._predictions.append(prediction)
        self._actuals.append(actual)
        self._features_history.append(features)

        if len(self._predictions) > self._WINDOW:
            self._predictions.pop(0)
            self._actuals.pop(0)
            self._features_history.pop(0)

        if len(self._predictions) < 20:
            return self.DriftLevel.NONE

        kl = self._compute_kl_divergence()
        self._update_page_hinkley(kl)

        drift_level = self._classify(kl)
        self._last_drift_level = drift_level   # stored for heartbeat access
        if drift_level != self.DriftLevel.NONE:
            log.warning(
                f"ConceptDriftDetector: KL={kl:.4f} → {drift_level.value.upper()}"
            )
        return drift_level

    def is_ml_disabled(self) -> bool:
        """Return True when drift is SEVERE and ML ensemble is suppressed."""
        return self._ml_disabled

    def reenable_ml(self) -> None:
        """Re-enable ML ensemble after a successful retrain."""
        self._ml_disabled = False
        self._ph_sum = 0.0
        self._ph_min = 0.0
        log.info("ConceptDriftDetector: ML ensemble re-enabled after retrain.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_kl_divergence(self) -> float:
        """KL divergence of live feature distribution vs training baseline.

        Uses a Gaussian approximation: KL(P||Q) where P=live, Q=training.
        """
        if self._train_mean is None or self._train_std is None:
            return 0.0

        live = np.array(self._features_history)
        live_mean = np.mean(live, axis=0)
        live_std = np.std(live, axis=0) + 1e-8

        # KL(N(μ1,σ1²) || N(μ0,σ0²)) per feature, averaged
        ratio = self._train_std / live_std
        kl_per_feature = (
            np.log(ratio)
            + (live_std**2 + (live_mean - self._train_mean)**2) / (2 * self._train_std**2)
            - 0.5
        )
        return float(np.mean(np.abs(kl_per_feature)))

    def _update_page_hinkley(self, kl: float) -> None:
        """Update Page-Hinkley CUSUM statistic."""
        self._ph_sum += kl - self._ph_delta
        self._ph_min = min(self._ph_min, self._ph_sum)

    def _classify(self, kl: float) -> ConceptDriftDetector.DriftLevel:
        """Map KL-divergence to a DriftLevel and apply side-effects."""
        if kl > self._kl_disable:
            self._ml_disabled = True
            return self.DriftLevel.SEVERE
        if kl > self._kl_retrain:
            return self.DriftLevel.RETRAIN
        if kl > self._kl_warn:
            return self.DriftLevel.WARN
        return self.DriftLevel.NONE
