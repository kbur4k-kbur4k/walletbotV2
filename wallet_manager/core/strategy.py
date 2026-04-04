"""
wallet_manager/core/strategy.py

Execution Variance Engine.

Responsibility
--------------
Provide a single, injectable ``ExecutionStrategy`` that introduces controlled
statistical variance into three dimensions of every task batch:

1. **Amount variance** — transaction values are derived as a percentage of
   the wallet's current balance, with a Gaussian noise term layered on top.
   This avoids cliff-edge behaviour at fixed round numbers and produces
   amounts that vary naturally between runs.

2. **Temporal variance** — every inter-step delay receives an additive
   Gaussian jitter so that the execution timeline is irregular.  No two runs
   should produce the same rhythm.

3. **Order variance** — wallet order and sub-task order within a wallet
   session are independently shuffled before execution begins.  This prevents
   systematic patterns in on-chain footprint timing.

Design
------
``ExecutionStrategy`` is a pure, stateless service: it performs only
mathematical transformations and carries no I/O, network, or signing logic.
It implements :class:`~wallet_manager.core.interfaces.IStrategy` so it can be
swapped or extended without touching the orchestrator.

All random samples use Python's ``random`` module (PRNG, not CSPRNG).
This is intentional: the randomness here governs execution timing and
amounts, not cryptographic secrets.  Using ``secrets`` would be overkill
and slower.

Note on framing
---------------
This module provides **execution variance** — a practice that benefits
operational resilience, avoids thundering-herd effects at gas-sensitive
moments, and produces more natural-looking on-chain timing.  It is not a
tool for deceiving third-party platforms or circumventing their policies.
Callers are responsible for ensuring their use of this module complies with
the terms of service of any protocol they interact with.
"""

from __future__ import annotations

import random
from typing import Any, TypeVar

from wallet_manager.core.interfaces import IStrategy
from wallet_manager.core.models import StrategyConfig
from wallet_manager.exceptions import StrategyError
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Absolute minimum transaction value (1 wei) — prevents zero-value dust
_ABSOLUTE_MIN_WEI: int = 1

# Gaussian clip bound: jitter is clamped to ±N*sigma to prevent extreme outliers
_JITTER_CLIP_SIGMA: float = 2.5

# Floor for any jittered delay
_MIN_DELAY_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# ExecutionStrategy
# ---------------------------------------------------------------------------


class ExecutionStrategy(IStrategy):
    """Stateless execution variance engine implementing :class:`IStrategy`.

    All methods are pure mathematical transformations — no I/O, no state
    mutations.  A single instance may safely be shared across all coroutines.

    Args:
        config: :class:`~wallet_manager.core.models.StrategyConfig` controlling
            variance parameters.  Defaults to sensible production values.

    Example:
        .. code-block:: python

            strategy = ExecutionStrategy()

            # Compute a jittered amount: 10–15% of a 1 ETH balance
            amount_wei = strategy.calculate_amount(
                balance_wei=10**18,
                min_pct=0.10,
                max_pct=0.15,
            )

            # Jitter a 2-second inter-step delay
            delay = strategy.jittered_delay(base_seconds=2.0)

            # Randomise task order before execution
            shuffled = strategy.shuffle_tasks(["approve", "swap", "bridge"])
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self._cfg = config or StrategyConfig()
        log.debug(
            f"ExecutionStrategy: min_pct={self._cfg.min_pct:.2f}, max_pct={self._cfg.max_pct:.2f}, "
            f"amount_sigma={self._cfg.amount_sigma_pct:.2f}, delay_sigma={self._cfg.delay_sigma_seconds:.2f}s.",
        )

    # ------------------------------------------------------------------
    # IStrategy implementation
    # ------------------------------------------------------------------

    def calculate_amount(
        self,
        balance_wei: int,
        min_pct: float | None = None,
        max_pct: float | None = None,
    ) -> int:
        """Compute a transaction amount as a jittered fraction of *balance_wei*.

        Algorithm
        ---------
        1. Sample a base fraction uniformly from ``[min_pct, max_pct]``.
        2. Compute ``base_amount = balance_wei * base_fraction``.
        3. Add Gaussian noise: ``jitter ~ N(0, (amount * sigma_pct)²)``,
           clipped to ``±CLIP_SIGMA * sigma_pct * base_amount``.
        4. Clamp the result to ``[1 wei, balance_wei]``.

        Args:
            balance_wei: Current wallet balance in wei.
            min_pct: Minimum fraction of balance (0.0–1.0).
                Defaults to ``config.min_pct``.
            max_pct: Maximum fraction of balance (0.0–1.0).
                Defaults to ``config.max_pct``.

        Returns:
            Amount in wei, always in [1, balance_wei].

        Raises:
            StrategyError: If ``min_pct > max_pct`` or balance is zero.
        """
        lo = min_pct if min_pct is not None else self._cfg.min_pct
        hi = max_pct if max_pct is not None else self._cfg.max_pct

        self._validate_pct_range(lo, hi)

        if balance_wei <= 0:
            raise StrategyError(
                f"Cannot calculate amount on a zero or negative balance "
                f"({balance_wei} wei)."
            )

        # Step 1: uniform base fraction
        base_pct = random.uniform(lo, hi)

        # Step 2: nominal amount
        base_amount = int(balance_wei * base_pct)

        # Step 3: Gaussian jitter on the amount
        sigma = base_amount * self._cfg.amount_sigma_pct
        if sigma > 0:
            raw_jitter = random.gauss(0.0, sigma)
            clip_bound = _JITTER_CLIP_SIGMA * sigma
            jitter = int(max(-clip_bound, min(clip_bound, raw_jitter)))
        else:
            jitter = 0

        # Step 4: clamp to valid range
        amount = max(_ABSOLUTE_MIN_WEI, min(balance_wei, base_amount + jitter))

        log.debug(
            f"Strategy.calculate_amount: balance={balance_wei}, pct={base_pct:.4f}, "
            f"base={base_amount}, jitter={jitter}, final={amount} wei.",
        )
        return amount

    def jittered_delay(
        self,
        base_seconds: float,
        sigma_override: float | None = None,
    ) -> float:
        """Return a Gaussian-jittered version of *base_seconds*.

        Jitter is sampled from ``N(0, σ²)`` and clipped to ``±CLIP_SIGMA*σ``
        before being added to the base.  The result is clamped to a
        non-negative floor of ``_MIN_DELAY_FLOOR`` seconds.

        Args:
            base_seconds: Nominal delay in seconds.
            sigma_override: Override the config ``delay_sigma_seconds`` for
                this specific call (e.g. for a longer cool-down step).

        Returns:
            Adjusted delay in seconds, always ≥ ``_MIN_DELAY_FLOOR``.
        """
        sigma = sigma_override if sigma_override is not None else self._cfg.delay_sigma_seconds

        jitter = 0.0
        if sigma > 0:
            raw = random.gauss(0.0, sigma)
            clip = _JITTER_CLIP_SIGMA * sigma
            jitter = max(-clip, min(clip, raw))

        delay = max(_MIN_DELAY_FLOOR, base_seconds + jitter)
        log.debug(
            f"Strategy.jittered_delay: base={base_seconds:.3f}s, sigma={sigma:.3f}, jitter={jitter:.3f} → {delay:.3f}s.",
        )
        return delay

    def shuffle_tasks(self, tasks: list[T]) -> list[T]:
        """Return a shuffled copy of *tasks* without modifying the original.

        Uses Fisher-Yates (via ``random.sample``) for an unbiased permutation.

        Args:
            tasks: List of task descriptors in any order.

        Returns:
            New list containing the same elements in a randomised order.
        """
        if not tasks:
            return []
        shuffled = random.sample(tasks, len(tasks))
        log.debug(f"Strategy.shuffle_tasks: {len(shuffled)} tasks shuffled.")
        return shuffled

    def build_execution_plan(
        self,
        wallet_ids: list[str],
        tasks_per_wallet: dict[str, list[Any]],
    ) -> list[tuple[str, list[Any]]]:
        """Build a complete, variance-applied execution plan.

        Applies wallet-order shuffling (if ``config.shuffle_wallets``) and
        per-wallet task shuffling (if ``config.shuffle_tasks``) in a single
        call, returning a ready-to-execute ordered list.

        Args:
            wallet_ids: Ordered list of wallet identifiers.
            tasks_per_wallet: Mapping of ``wallet_id → task_list``.

        Returns:
            List of ``(wallet_id, shuffled_task_list)`` tuples in execution order.

        Example:
            .. code-block:: python

                plan = strategy.build_execution_plan(
                    wallet_ids=["w1", "w2", "w3"],
                    tasks_per_wallet={
                        "w1": ["swap", "bridge"],
                        "w2": ["swap"],
                        "w3": ["bridge", "stake"],
                    },
                )
                for wallet_id, tasks in plan:
                    await run_wallet(wallet_id, tasks)
        """
        ordered = (
            self.shuffle_tasks(wallet_ids)
            if self._cfg.shuffle_wallets
            else list(wallet_ids)
        )

        plan: list[tuple[str, list[Any]]] = []
        for wid in ordered:
            tasks = tasks_per_wallet.get(wid, [])
            shuffled_tasks = (
                self.shuffle_tasks(tasks)
                if self._cfg.shuffle_tasks
                else list(tasks)
            )
            plan.append((wid, shuffled_tasks))

        log.debug(
            f"Strategy.build_execution_plan: {len(plan)} wallets, shuffle_wallets={self._cfg.shuffle_wallets}, "
            f"shuffle_tasks={self._cfg.shuffle_tasks}.",
        )
        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pct_range(min_pct: float, max_pct: float) -> None:
        """Validate that percentage bounds are sane.

        Args:
            min_pct: Lower bound.
            max_pct: Upper bound.

        Raises:
            StrategyError: If bounds are out of range or inverted.
        """
        if not (0.0 < min_pct <= 1.0):
            raise StrategyError(
                f"min_pct must be in (0, 1]; got {min_pct}."
            )
        if not (0.0 < max_pct <= 1.0):
            raise StrategyError(
                f"max_pct must be in (0, 1]; got {max_pct}."
            )
        if min_pct > max_pct:
            raise StrategyError(
                f"min_pct ({min_pct}) must not exceed max_pct ({max_pct})."
            )


# ---------------------------------------------------------------------------
# Optional type hint that was missing from __future__ import
# ---------------------------------------------------------------------------
