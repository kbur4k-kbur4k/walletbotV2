"""
wallet_manager/portfolio_manager.py

4-Pillar Portfolio Orchestrator — dynamic budget allocation engine.

Manages the 40/35/10/15 split across four strategy pillars:
  - YIELD   (40%): Aave V3 USDC supply on Base
  - ALGO    (35%): ML ensemble mean-reversion trading on Base
  - SHORT   (10%): Hedged delta-neutral synthetic short via Aave V3
  - AIRDROP (15%): Ping-Pong ETH<->USDC farming for protocol activity

Budget model
------------
* All values tracked in USD-equivalent terms
* Rebalancing distributes new injections according to target weights

The PortfolioManager is stateless with respect to on-chain balances.
It computes *target* allocations; the orchestrator is responsible for
executing the resulting instructions via WalletManager.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy type enum (shared across analytics, orchestrator, strategies)
# ---------------------------------------------------------------------------


class StrategyType(Enum):
    """Identifies each portfolio pillar for analytics and routing."""

    YIELD = "YIELD"
    ALGO = "ALGO"
    SHORT = "SHORT"
    AIRDROP = "AIRDROP"


# ---------------------------------------------------------------------------
# Allocation weights — loaded from config (env-overridable)
# ---------------------------------------------------------------------------
# YIELD  (40%): Passive Aave V3 USDC supply — zero lazy capital.
# ALGO   (35%): Mean-reversion long (ML ensemble: XGBoost/LightGBM/LSTM/Transformer).
# SHORT  (10%): Hedged delta-neutral synthetic short via Aave V3 borrow.
# AIRDROP(15%): Ping-pong ETH<->USDC farming for protocol activity.
#
# The 10:1 ratio: for every $10 the ALGO (long) captures in a bull cycle,
# the SHORT captures $1 in the correction. The combined axis ensures
# profitability in both directions.
# ---------------------------------------------------------------------------


def _get_weights() -> dict[StrategyType, float]:
    """Return portfolio weights from config (env-overridable)."""
    try:
        from wallet_manager.config import get_settings
        cfg = get_settings()
        return {
            StrategyType.YIELD: cfg.yield_weight,
            StrategyType.ALGO: cfg.algo_weight,
            StrategyType.SHORT: cfg.short_weight,
            StrategyType.AIRDROP: cfg.airdrop_weight,
        }
    except Exception:
        return {
            StrategyType.YIELD: 0.40,
            StrategyType.ALGO: 0.35,
            StrategyType.SHORT: 0.10,
            StrategyType.AIRDROP: 0.15,
        }


# Module-level constants (kept for backward-compat; prefer config values)
YIELD_WEIGHT: float = 0.40
ALGO_WEIGHT: float = 0.35
SHORT_WEIGHT: float = 0.10
AIRDROP_WEIGHT: float = 0.15

_WEIGHTS: dict[StrategyType, float] = {
    StrategyType.YIELD: YIELD_WEIGHT,
    StrategyType.ALGO: ALGO_WEIGHT,
    StrategyType.SHORT: SHORT_WEIGHT,
    StrategyType.AIRDROP: AIRDROP_WEIGHT,
}

# Budget constants (USD) — prefer config value; kept for backward-compat
MONTHLY_INJECTION_USD: float = 150.0


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PillarAllocation:
    """Target allocation for a single strategy pillar.

    Attributes:
        strategy: Which pillar this allocation belongs to.
        target_usd: Ideal USD value this pillar should hold.
        current_usd: Actual USD value currently held by this pillar.
        delta_usd: Amount to add (positive) or remove (negative) to reach target.
        weight: Target weight as a fraction (0.0-1.0).
    """

    strategy: StrategyType
    target_usd: float
    current_usd: float
    delta_usd: float
    weight: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time snapshot of the full portfolio state.

    Attributes:
        total_value_usd: Sum of all pillar values in USD.
        allocations: Per-pillar allocation details.
        timestamp: Unix epoch when the snapshot was taken.
        needs_rebalance: True if any pillar deviates > threshold from target.
    """

    total_value_usd: float
    allocations: tuple[PillarAllocation, ...]
    timestamp: float
    needs_rebalance: bool


@dataclass
class PillarState:
    """Mutable tracking state for one pillar.

    Attributes:
        strategy: Pillar identifier.
        deposited_usd: Cumulative USD deposited into this pillar.
        current_value_usd: Latest known USD value (updated by strategy).
    """

    strategy: StrategyType
    deposited_usd: float = 0.0
    current_value_usd: float = 0.0


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------


class PortfolioManager:
    """Dynamic budget allocator for the 4-pillar portfolio.

    Computes target allocations, tracks cumulative deposits, and produces
    rebalancing instructions when new capital is injected.

    Args:
        initial_budget_usd: Starting capital.  Defaults to 0.0 (real balance
            fetched via sync_from_chain).
        monthly_injection_usd: Recurring monthly addition.  Defaults to $150.
        rebalance_threshold_pct: Minimum deviation (fraction) from target
            weight before a rebalance is flagged.  Default 5%.

    Example:
        .. code-block:: python

            pm = PortfolioManager()
            pm.initialize_allocations()

            # After fetching on-chain balances:
            pm.update_pillar_value(StrategyType.YIELD, 128.50)
            pm.update_pillar_value(StrategyType.ALGO, 85.00)
            pm.update_pillar_value(StrategyType.AIRDROP, 38.00)

            snapshot = pm.get_snapshot()
            injection = pm.compute_injection_splits(150.0)
    """

    def __init__(
        self,
        initial_budget_usd: float = 0.0,
        monthly_injection_usd: float | None = None,
        rebalance_threshold_pct: float | None = None,
    ) -> None:
        # Load defaults from config if not explicitly provided
        try:
            from wallet_manager.config import get_settings
            cfg = get_settings()
            _default_injection = cfg.monthly_injection_usd
            _default_threshold = cfg.rebalance_drift_threshold
        except Exception:
            _default_injection = MONTHLY_INJECTION_USD
            _default_threshold = 0.05

        self._initial_budget = initial_budget_usd
        self._monthly_injection = monthly_injection_usd if monthly_injection_usd is not None else _default_injection
        self._rebalance_threshold = rebalance_threshold_pct if rebalance_threshold_pct is not None else _default_threshold
        self._total_deposited: float = 0.0
        self._weights = _get_weights()

        self._pillars: dict[StrategyType, PillarState] = {
            st: PillarState(strategy=st) for st in StrategyType
        }

        log.info(
            f"PortfolioManager created: initial=${initial_budget_usd:.0f}, "
            f"monthly=${self._monthly_injection:.0f}, threshold={self._rebalance_threshold:.0%}.",
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_allocations(self) -> dict[StrategyType, float]:
        """Distribute the initial budget across pillars.

        Returns:
            Dict mapping each pillar to its initial USD allocation.
        """
        splits: dict[StrategyType, float] = {}
        for strategy, weight in self._weights.items():
            amount = self._initial_budget * weight
            self._pillars[strategy].deposited_usd = amount
            self._pillars[strategy].current_value_usd = amount
            splits[strategy] = amount

        self._total_deposited = self._initial_budget
        log.info(
            f"Initial allocation: YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        return splits

    # ------------------------------------------------------------------
    # Value tracking
    # ------------------------------------------------------------------

    def update_pillar_value(self, strategy: StrategyType, current_usd: float) -> None:
        """Update the current USD value for a pillar after on-chain queries.

        Args:
            strategy: Which pillar to update.
            current_usd: Latest USD-equivalent value held by the pillar.
        """
        self._pillars[strategy].current_value_usd = current_usd
        log.debug(f"Pillar {strategy.value} value updated: ${current_usd:.2f}.")

    async def sync_from_chain(
        self,
        wallet_address: str,
        rpc_url: str,
        chain_id: int,
        eth_price_usd: float,
    ) -> None:
        """Fetch real on-chain balances and update all pillar values.

        Queries:
          ETH balance      -> USD value -> ALGO pillar
          Spot USDC        -> AIRDROP pillar
          Aave aToken bal  -> YIELD pillar
          Short pos value  -> SHORT pillar

        Calls update_pillar_value() for each in the correct order so
        total_value_usd reflects the real on-chain portfolio.
        """
        from wallet_manager import actions
        from wallet_manager.strategies import yield_strategy

        # --- ETH balance -> ALGO pillar ---
        try:
            eth_wei = await actions.fetch_native_balance(rpc_url, wallet_address)
            eth_usd = (eth_wei / 1e18) * eth_price_usd
        except Exception:
            eth_usd = self._pillars[StrategyType.ALGO].current_value_usd

        # --- Spot USDC -> AIRDROP pillar ---
        try:
            usdc_raw = await actions.fetch_usdc_balance(rpc_url, wallet_address, chain_id)
            usdc_usd = usdc_raw / 1e6
        except Exception:
            usdc_usd = self._pillars[StrategyType.AIRDROP].current_value_usd

        # --- Aave aToken balance -> YIELD pillar ---
        try:
            aave_raw = await yield_strategy.get_aave_usdc_balance(rpc_url, wallet_address)
            aave_usd = aave_raw / 1e6
        except Exception:
            aave_usd = self._pillars[StrategyType.YIELD].current_value_usd

        # --- Short position value -> SHORT pillar ---
        try:
            from wallet_manager.strategies import short_strategy
            short_state = short_strategy.get_short_position_state()
            short_usd = float(getattr(short_state, "position_value_usd", 0.0))
        except Exception:
            short_usd = self._pillars[StrategyType.SHORT].current_value_usd

        self.update_pillar_value(StrategyType.ALGO, eth_usd)
        self.update_pillar_value(StrategyType.AIRDROP, usdc_usd)
        self.update_pillar_value(StrategyType.YIELD, aave_usd)
        self.update_pillar_value(StrategyType.SHORT, short_usd)

        log.info(
            f"sync_from_chain: total=${self.total_value_usd:.2f} "
            f"(ETH=${eth_usd:.2f}, USDC=${usdc_usd:.2f}, "
            f"Aave=${aave_usd:.2f}, Short=${short_usd:.2f})"
        )

    def get_pillar_value(self, strategy: StrategyType) -> float:
        """Return the current USD value for a pillar."""
        return self._pillars[strategy].current_value_usd

    @property
    def total_value_usd(self) -> float:
        """Sum of all pillar current values."""
        return sum(p.current_value_usd for p in self._pillars.values())

    # ------------------------------------------------------------------
    # Allocation queries
    # ------------------------------------------------------------------

    def get_target_allocation(self, strategy: StrategyType) -> float:
        """Return the target USD allocation for a pillar based on total value.

        Args:
            strategy: Pillar to query.

        Returns:
            Target USD value this pillar should ideally hold.
        """
        return self.total_value_usd * self._weights[strategy]

    def get_pillar_budget_usd(self, strategy: StrategyType) -> float:
        """Return the operational budget for a specific pillar.

        This is the current value of the pillar, which the strategy
        should use as its available capital.
        """
        return self._pillars[strategy].current_value_usd

    def needs_rebalancing(self, current_allocations: dict[StrategyType, float] | None = None) -> bool:
        """Return True only when any pillar drifts beyond cfg.rebalance_drift_threshold.

        Args:
            current_allocations: Optional mapping of strategy -> current USD value.
                If None, uses internal tracked pillar values.

        Returns:
            True if any pillar deviates more than the configured threshold.
        """
        total = self.total_value_usd
        if total <= 0:
            return False
        allocations = current_allocations or {
            st: p.current_value_usd for st, p in self._pillars.items()
        }
        for strategy, weight in self._weights.items():
            current = allocations.get(strategy, 0.0)
            actual_weight = current / total
            if abs(actual_weight - weight) > self._rebalance_threshold:
                return True
        return False

    def get_snapshot(self) -> PortfolioSnapshot:
        """Build a point-in-time portfolio snapshot with rebalance detection.

        Returns:
            :class:`PortfolioSnapshot` with per-pillar allocations and
            a ``needs_rebalance`` flag.
        """
        total = self.total_value_usd
        needs_rebalance = False
        allocations: list[PillarAllocation] = []

        for strategy, weight in self._weights.items():
            pillar = self._pillars[strategy]
            target = total * weight
            delta = target - pillar.current_value_usd

            # Check if deviation exceeds threshold
            if total > 0:
                actual_weight = pillar.current_value_usd / total
                if abs(actual_weight - weight) > self._rebalance_threshold:
                    needs_rebalance = True

            allocations.append(
                PillarAllocation(
                    strategy=strategy,
                    target_usd=round(target, 2),
                    current_usd=round(pillar.current_value_usd, 2),
                    delta_usd=round(delta, 2),
                    weight=weight,
                )
            )

        return PortfolioSnapshot(
            total_value_usd=round(total, 2),
            allocations=tuple(allocations),
            timestamp=time.time(),
            needs_rebalance=needs_rebalance,
        )

    # ------------------------------------------------------------------
    # Injection / rebalancing
    # ------------------------------------------------------------------

    def compute_injection_splits(
        self, injection_usd: float | None = None
    ) -> dict[StrategyType, float]:
        """Compute how to distribute a new capital injection.

        The injection is split according to target weights (50/35/15),
        with adjustments to compensate for under-allocated pillars.

        Args:
            injection_usd: USD amount to inject.  Defaults to the
                configured monthly injection ($150).

        Returns:
            Dict mapping each pillar to the USD amount it should receive.
        """
        amount = injection_usd if injection_usd is not None else self._monthly_injection
        total_after = self.total_value_usd + amount

        splits: dict[StrategyType, float] = {}
        for strategy, weight in self._weights.items():
            pillar = self._pillars[strategy]
            target_after = total_after * weight
            deficit = target_after - pillar.current_value_usd
            # Each pillar gets at least its proportional share,
            # but we cap at the injection amount per-pillar
            allocation = max(0.0, min(deficit, amount * weight * 2))
            splits[strategy] = round(allocation, 2)

        # Normalize so splits sum exactly to injection amount
        raw_total = sum(splits.values())
        if raw_total > 0:
            scale = amount / raw_total
            splits = {k: round(v * scale, 2) for k, v in splits.items()}

        # Fix rounding residual
        residual = round(amount - sum(splits.values()), 2)
        if residual != 0:
            # Assign residual to the largest pillar (YIELD)
            splits[StrategyType.YIELD] = round(
                splits[StrategyType.YIELD] + residual, 2
            )

        self._total_deposited += amount
        for strategy, alloc in splits.items():
            self._pillars[strategy].deposited_usd += alloc

        log.info(
            f"Injection ${amount:.2f} split: "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        return splits

    def rebalance_monthly_topup(
        self, injection_usd: float | None = None
    ) -> dict[StrategyType, float]:
        """
        Distribute monthly top-up ($150) according to target weights (50/35/15).

        Called when new capital is added. Returns splits for YIELD, ALGO, AIRDROP
        that sum to the injection amount.

        Args:
            injection_usd: USD to inject. Defaults to monthly_injection_usd ($150).

        Returns:
            Dict mapping each pillar to the USD amount it should receive.
        """
        return self.compute_injection_splits(injection_usd)

    def rebalance_portfolio(
        self, injection_usd: float | None = None
    ) -> dict[StrategyType, float]:
        """
        Handle the monthly $150 injection according to 50/35/15 ratio.

        Coordinates capital distribution across all three strategies.
        Call when SchedulerManager.is_monthly_rebalance_due() is True.

        Args:
            injection_usd: USD to inject. Defaults to monthly_injection_usd ($150).

        Returns:
            Dict mapping each pillar (YIELD, ALGO, AIRDROP) to USD amount to allocate.
        """
        return self.rebalance_monthly_topup(injection_usd)

    def compute_rebalance_transfers(self) -> list[tuple[StrategyType, StrategyType, float]]:
        """Compute transfers needed to rebalance the portfolio.

        Returns:
            List of ``(from_pillar, to_pillar, usd_amount)`` tuples.
            Empty if no rebalance is needed.
        """
        snapshot = self.get_snapshot()
        if not snapshot.needs_rebalance:
            return []

        # Identify over-allocated and under-allocated pillars
        over: list[tuple[StrategyType, float]] = []
        under: list[tuple[StrategyType, float]] = []

        for alloc in snapshot.allocations:
            if alloc.delta_usd > 1.0:  # needs more capital
                under.append((alloc.strategy, alloc.delta_usd))
            elif alloc.delta_usd < -1.0:  # has excess capital
                over.append((alloc.strategy, abs(alloc.delta_usd)))

        transfers: list[tuple[StrategyType, StrategyType, float]] = []
        for over_strat, over_amt in over:
            remaining = over_amt
            for i, (under_strat, under_amt) in enumerate(under):
                if remaining <= 0 or under_amt <= 0:
                    continue
                transfer = min(remaining, under_amt)
                transfers.append((over_strat, under_strat, round(transfer, 2)))
                remaining -= transfer
                under[i] = (under_strat, under_amt - transfer)

        if transfers:
            log.info(f"Rebalance plan: {len(transfers)} transfer(s) computed.")

        return transfers

    # ------------------------------------------------------------------
    # Algo-specific budget helpers
    # ------------------------------------------------------------------

    def get_algo_trade_budget(self) -> float:
        """Return 20% of the ALGO pillar's current value for a single trade.

        The algo strategy uses 20% of its allocated budget per trade entry.
        """
        return self._pillars[StrategyType.ALGO].current_value_usd * 0.20

    def get_short_budget_usd(self) -> float:
        """Return the SHORT pillar's current budget for position sizing.

        The short strategy sizes positions against this budget, capped
        at 15% of total equity for risk management.
        """
        return self._pillars[StrategyType.SHORT].current_value_usd

    def get_short_max_position_usd(self) -> float:
        """Return the maximum allowable short position in USD.

        Capped at the smaller of: SHORT pillar value or 15% of total equity.
        """
        return min(
            self._pillars[StrategyType.SHORT].current_value_usd,
            self.total_value_usd * 0.15,
        ) * 0.20

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_summary(self) -> str:
        """Return a human-readable portfolio summary string."""
        snapshot = self.get_snapshot()
        lines = [
            f"Portfolio Total: ${snapshot.total_value_usd:.2f}",
            f"Rebalance needed: {snapshot.needs_rebalance}",
            "---",
        ]
        for alloc in snapshot.allocations:
            pct = (alloc.current_usd / snapshot.total_value_usd * 100) if snapshot.total_value_usd > 0 else 0
            lines.append(
                f"  {alloc.strategy.value:>7}: ${alloc.current_usd:>8.2f} "
                f"(target ${alloc.target_usd:>8.2f}, {pct:>5.1f}%, "
                f"delta ${alloc.delta_usd:>+8.2f})"
            )
        return "\n".join(lines)
