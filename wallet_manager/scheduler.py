"""
wallet_manager/scheduler.py

Weekly random scheduler for airdrop farming swaps.

Uses SchedulerManager to persist state to disk (schedule_data.json) per wallet.
First run for new wallets: 120-300 seconds (test-friendly). Subsequent: weekly.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# One week in seconds
WEEK_SECONDS = 7 * 24 * 3600

# Monthly rebalance window (30 days)
MONTH_SECONDS = 30 * 24 * 3600

# First-run: random delay 120-300 seconds for testing
FIRST_RUN_MIN_SECONDS = 120
FIRST_RUN_MAX_SECONDS = 300

# Smart gas deferral: when gas too high, retry in 2-4 hours (not 1 week)
GAS_DEFER_MIN_SECONDS = 7200   # 2 hours
GAS_DEFER_MAX_SECONDS = 14400  # 4 hours


def _default_data_path() -> Path:
    """Return default path for schedule_data.json (package dir or CWD)."""
    from wallet_manager.config import get_settings
    p = str(get_settings().schedule_data_path)
    if p:
        return Path(p)
    return Path(__file__).resolve().parent / "schedule_data.json"


class SchedulerManager:
    """Persistent per-wallet scheduler with disk-backed state.

    Stores state in schedule_data.json. Each wallet_id has its own
    last_completed and next_run. First run for new wallets: 120-300s.
    Subsequent runs: weekly random offset.
    """

    def __init__(
        self,
        data_path: Path | None = None,
        schedule_file: str | None = None,  # alias for data_path
    ) -> None:
        if schedule_file is not None and data_path is None:
            data_path = Path(schedule_file)
        self._path = data_path or _default_data_path()
        self._cache: dict[str, dict] = {}
        self._load()  # load immediately so _data is populated

    @property
    def _data(self) -> dict[str, dict]:
        """Alias for _cache — simple {wallet_id: {last_run: epoch}} format."""
        return self._cache

    @_data.setter
    def _data(self, value: dict) -> None:
        self._cache = value

    def _load(self) -> None:
        """Load state from disk."""
        if not self._path.exists():
            self._cache = {}
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                self._cache = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"SchedulerManager: could not load {self._path}: {exc}. Starting fresh.")
            self._cache = {}

    def _save(self) -> None:
        """Persist state to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            log.warning(f"SchedulerManager: could not save {self._path}: {exc}")

    def _ensure_loaded(self) -> None:
        """Load from disk if file exists and we haven't populated cache from it yet."""
        if self._path.exists() and not self._cache:
            self._load()
        elif not self._path.exists():
            self._cache = {}

    # ------------------------------------------------------------------
    # Simple API for tests: should_run_swap_this_cycle / mark_ran / defer
    # ------------------------------------------------------------------

    def should_run_swap_this_cycle(self, wallet_id: str, now: float | None = None) -> bool:
        """Return True if this wallet should run a swap this cycle.

        - New wallets (no record): return True immediately.
        - Wallets with a ``last_run`` timestamp: due after WEEK_SECONDS.
        - Wallets with a ``defer_until`` timestamp: not due until that time.
        - Wallets with a ``next_run_epoch`` (old format): delegates to is_due.
        """
        t = now if now is not None else time.time()
        entry = self._cache.get(wallet_id)
        if entry is None:
            return True  # New wallet: run immediately

        # Check deferral
        defer_until = float(entry.get("defer_until", 0))
        if defer_until > t:
            return False

        # Simple last_run format
        last_run = entry.get("last_run")
        if last_run is not None:
            return (t - float(last_run)) >= WEEK_SECONDS

        # Legacy next_run_epoch format
        next_run = float(entry.get("next_run_epoch", 0))
        if next_run > 0:
            return t >= next_run

        return True

    def mark_ran(self, wallet_id: str, now: float | None = None) -> None:
        """Mark that a swap just ran for this wallet."""
        t = now if now is not None else time.time()
        entry = self._cache.get(wallet_id) or {}
        entry["last_run"] = t
        entry.pop("defer_until", None)
        self._cache[wallet_id] = entry
        self._save()

    def defer_due_to_gas(self, wallet_id: str, defer_hours: float = 2.0, now: float | None = None) -> None:
        """Defer this wallet's next run by defer_hours (gas too high)."""
        t = now if now is not None else time.time()
        entry = self._cache.get(wallet_id) or {}
        entry["defer_until"] = t + defer_hours * 3600
        self._cache[wallet_id] = entry
        self._save()

    def is_due(self, wallet_id: str, now: float | None = None) -> bool:
        """Return True if this wallet is due for a swap."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        entry = self._cache.get(wallet_id)
        if entry is None:
            return False  # New wallet: will be scheduled, but not due yet
        return t >= float(entry.get("next_run_epoch", 0))

    def seconds_until_due(self, wallet_id: str, now: float | None = None) -> float:
        """Return seconds until next scheduled run for this wallet."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        entry = self._cache.get(wallet_id)
        if entry is None:
            return 0.0
        next_run = float(entry.get("next_run_epoch", 0))
        return max(0.0, next_run - t)

    def schedule_first_run(self, wallet_id: str, now: float | None = None) -> None:
        """Schedule first run for a new wallet: 120-300 seconds from now."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        offset = random.uniform(FIRST_RUN_MIN_SECONDS, FIRST_RUN_MAX_SECONDS)
        next_run = t + offset
        self._cache[wallet_id] = {
            "last_completed_epoch": 0.0,
            "next_run_epoch": next_run,
        }
        self._save()
        log.info(
            f"Scheduler: wallet '{wallet_id}' first run in {offset:.0f}s "
            f"(epoch {next_run:.0f}).",
        )

    def schedule_next(self, wallet_id: str, now: float | None = None) -> None:
        """Set next_run to a random time within the next week."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        offset = random.uniform(3600, WEEK_SECONDS)
        next_run = t + offset
        entry = self._cache.get(wallet_id) or {}
        entry["last_completed_epoch"] = t
        entry["next_run_epoch"] = next_run
        self._cache[wallet_id] = entry
        self._save()
        log.info(
            f"Scheduler: wallet '{wallet_id}' next swap in {offset / 3600:.1f}h "
            f"(epoch {next_run:.0f}).",
        )

    def schedule_gas_deferral(self, wallet_id: str, now: float | None = None) -> None:
        """Set next_run to 2-4 hours from now (gas too high, retry later).

        Use when GasPriceTooHighError occurs — do not treat as failure;
        reschedule for a short window instead of 1 week.
        """
        t = now if now is not None else time.time()
        self._ensure_loaded()
        offset = random.uniform(GAS_DEFER_MIN_SECONDS, GAS_DEFER_MAX_SECONDS)
        next_run = t + offset
        entry = self._cache.get(wallet_id) or {}
        entry["next_run_epoch"] = next_run
        self._cache[wallet_id] = entry
        self._save()
        log.info(
            f"Scheduler: wallet '{wallet_id}' gas deferral — retry in {offset / 3600:.1f}h "
            f"(epoch {next_run:.0f}).",
        )

    def has_record(self, wallet_id: str) -> bool:
        """Return True if this wallet has a schedule record."""
        self._ensure_loaded()
        return wallet_id in self._cache

    def get_next_run_epoch(self, wallet_id: str) -> float:
        """Return next_run_epoch for this wallet, or 0 if none."""
        self._ensure_loaded()
        entry = self._cache.get(wallet_id)
        if entry is None:
            return 0.0
        return float(entry.get("next_run_epoch", 0))

    # ------------------------------------------------------------------
    # Monthly rebalance (for $150 injection)
    # ------------------------------------------------------------------

    def is_initial_rebalance(self) -> bool:
        """Return True if this is the first rebalance (initial $250 allocation)."""
        self._ensure_loaded()
        last = float(self._cache.get("_monthly", {}).get("last_rebalance_epoch", 0))
        return last <= 0

    def is_weekly_hyperopt_due(self, now: float | None = None) -> bool:
        """Return True if weekly hyperparameter optimization is due."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        last = float(self._cache.get("_hyperopt", {}).get("last_run_epoch", 0))
        if last <= 0:
            return True  # First run: due immediately
        return (t - last) >= WEEK_SECONDS  # Subsequent: weekly

    def mark_weekly_hyperopt_done(self, now: float | None = None) -> None:
        """Mark the weekly hyperopt as completed."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        entry = self._cache.get("_hyperopt") or {}
        entry["last_run_epoch"] = t
        self._cache["_hyperopt"] = entry
        self._save()

    def is_monthly_rebalance_due(self, now: float | None = None) -> bool:
        """Return True if a monthly rebalance (capital injection) is due."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        last = float(self._cache.get("_monthly", {}).get("last_rebalance_epoch", 0))
        if last <= 0:
            return True  # First run: due immediately ($250 initial)
        return (t - last) >= MONTH_SECONDS  # Subsequent: $150 monthly

    def mark_monthly_rebalance_done(self, now: float | None = None) -> None:
        """Mark the monthly rebalance as completed."""
        t = now if now is not None else time.time()
        self._ensure_loaded()
        entry = self._cache.get("_monthly") or {}
        entry["last_rebalance_epoch"] = t
        self._cache["_monthly"] = entry
        self._save()
        log.info(f"Scheduler: monthly rebalance marked done (epoch {t:.0f}).")


def should_run_swap_this_cycle(
    scheduler: SchedulerManager,
    wallet_id: str,
) -> tuple[bool, float]:
    """
    Check if a swap should run this orchestrator cycle for the given wallet.

    Returns:
        (should_run, seconds_until_due).
        - If wallet has no record: schedules first run (120-300s), returns (False, offset).
        - If not yet due: returns (False, seconds_until_due).
        - If due: returns (True, 0).
    """
    if not scheduler.has_record(wallet_id):
        scheduler.schedule_first_run(wallet_id)
        return False, scheduler.seconds_until_due(wallet_id)

    if scheduler.is_due(wallet_id):
        return True, 0.0

    secs = scheduler.seconds_until_due(wallet_id)
    return False, secs
