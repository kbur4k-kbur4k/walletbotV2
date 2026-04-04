"""
wallet_manager/strategies/kelly_criterion.py

QUANT-X Kelly Criterion Position Sizing — The "Fortress" Risk Engine.

Dynamically calculates the optimal fraction of capital for each trade
based on the historical win/loss ratio and Z-Score confidence level.

Kelly Formula
-------------
    f* = (p × b - q) / b

Where:
    f* = Optimal fraction of capital to risk
    p  = Probability of winning (historical win rate)
    q  = Probability of losing (1 - p)
    b  = Win/Loss ratio (average win / average loss)

Safety Modifications
--------------------
1. Half-Kelly: We use f*/2 to reduce variance while capturing ~75% of
   the compounding benefit (industry standard for hedge funds).
2. Z-Score Confidence Scaling: Position size scaled by signal confidence.
   Higher confidence (|z| > 3.0) → closer to full half-Kelly.
   Lower confidence (|z| ≈ 2.0) → reduced fraction.
3. Maximum Position Cap: Never exceed 25% of capital per trade.
4. Minimum Position Floor: Below $5 is not worth the gas.
5. Drawdown Throttle: After consecutive losses, reduce sizing further.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Kelly scaling
KELLY_FRACTION = 0.5             # Half-Kelly (industry standard)
MAX_POSITION_PCT = 0.25          # Never risk more than 25% of capital
MIN_POSITION_USD = 5.0           # Minimum position size
MIN_TRADES_FOR_KELLY = 10        # Minimum trades before using Kelly

# Confidence scaling thresholds
CONFIDENCE_FULL = 3.0            # |z| >= 3.0 → full half-Kelly
CONFIDENCE_MIN = 2.0             # |z| = 2.0 → minimum fraction

# Drawdown throttle
CONSECUTIVE_LOSS_THROTTLE = 3    # After 3 consecutive losses, throttle
THROTTLE_FACTOR = 0.5            # Reduce by 50% during drawdown

# Default win rate (before enough data)
DEFAULT_WIN_RATE = 0.55          # Conservative 55% assumption
DEFAULT_WIN_LOSS_RATIO = 1.5     # Average win is 1.5x average loss

# Trade history persistence
MAX_TRADE_HISTORY = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeResult:
    """Record of a completed trade for Kelly calculations."""

    pnl_usd: float           # Profit/loss in USD (positive = win)
    entry_price: float
    exit_price: float
    position_usd: float      # Position size in USD
    z_score_at_entry: float   # Z-Score when trade was opened
    strategy: str             # "ALGO", "SHORT", etc.
    timestamp: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl_usd > 0

    @property
    def return_pct(self) -> float:
        if self.position_usd <= 0:
            return 0.0
        return self.pnl_usd / self.position_usd

    def to_dict(self) -> dict:
        return {
            "pnl_usd": self.pnl_usd,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_usd": self.position_usd,
            "z_score_at_entry": self.z_score_at_entry,
            "strategy": self.strategy,
            "timestamp": self.timestamp or time.time(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TradeResult:
        return cls(
            pnl_usd=float(d.get("pnl_usd", 0)),
            entry_price=float(d.get("entry_price", 0)),
            exit_price=float(d.get("exit_price", 0)),
            position_usd=float(d.get("position_usd", 0)),
            z_score_at_entry=float(d.get("z_score_at_entry", 0)),
            strategy=str(d.get("strategy", "")),
            timestamp=float(d.get("timestamp", 0)),
        )


@dataclass(frozen=True)
class PositionSizeResult:
    """Output of Kelly Criterion position sizing.

    Attributes:
        position_usd: Recommended position size in USD.
        position_pct: As fraction of total capital.
        kelly_raw: Raw Kelly fraction (before scaling).
        kelly_scaled: Half-Kelly after confidence scaling.
        confidence_multiplier: Z-Score confidence scaling factor.
        drawdown_multiplier: Drawdown throttle factor.
        win_rate: Historical win rate used.
        win_loss_ratio: Historical win/loss ratio used.
        trade_count: Number of historical trades considered.
        reasoning: Human-readable explanation of sizing decision.
    """

    position_usd: float
    position_pct: float
    kelly_raw: float
    kelly_scaled: float
    confidence_multiplier: float
    drawdown_multiplier: float
    win_rate: float
    win_loss_ratio: float
    trade_count: int
    reasoning: str


# ---------------------------------------------------------------------------
# Kelly Criterion Engine
# ---------------------------------------------------------------------------


class KellyCriterion:
    """Institutional-grade position sizing using the Kelly Criterion.

    Maintains a rolling history of trade results to compute win rate
    and win/loss ratio. Uses Half-Kelly with confidence scaling for
    conservative but optimal capital allocation.

    Usage:
        kelly = KellyCriterion()

        # After each trade completes:
        kelly.record_trade(TradeResult(pnl_usd=15.0, ...))

        # Before entering a new trade:
        sizing = kelly.calculate_position_size(
            total_capital_usd=1000.0,
            z_score=-2.8,
            strategy="ALGO",
        )
        print(f"Position: ${sizing.position_usd:.2f}")
    """

    def __init__(self, history_path: Path | None = None) -> None:
        self._trades: deque[TradeResult] = deque(maxlen=MAX_TRADE_HISTORY)
        self._history_path = history_path or (
            Path(__file__).resolve().parent.parent / "kelly_trade_history.json"
        )
        self._load_history()

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade result."""
        if trade.timestamp <= 0:
            trade = TradeResult(
                pnl_usd=trade.pnl_usd,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                position_usd=trade.position_usd,
                z_score_at_entry=trade.z_score_at_entry,
                strategy=trade.strategy,
                timestamp=time.time(),
            )
        self._trades.append(trade)
        self._save_history()

        log.info(
            f"Kelly: recorded trade — PnL: ${trade.pnl_usd:+.2f}, "
            f"Win rate: {self.win_rate:.1%}, W/L ratio: {self.win_loss_ratio:.2f}, "
            f"Total trades: {len(self._trades)}."
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def win_rate(self) -> float:
        """Historical win rate [0, 1]."""
        if not self._trades:
            return DEFAULT_WIN_RATE
        wins = sum(1 for t in self._trades if t.is_win)
        return wins / len(self._trades)

    @property
    def win_loss_ratio(self) -> float:
        """Average win / average loss ratio."""
        wins = [t.pnl_usd for t in self._trades if t.is_win]
        losses = [-t.pnl_usd for t in self._trades if not t.is_win]

        if not wins or not losses:
            return DEFAULT_WIN_LOSS_RATIO

        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss <= 0:
            return DEFAULT_WIN_LOSS_RATIO

        return avg_win / avg_loss

    @property
    def consecutive_losses(self) -> int:
        """Count of current consecutive losing trades."""
        count = 0
        for t in reversed(self._trades):
            if not t.is_win:
                count += 1
            else:
                break
        return count

    @property
    def total_pnl_usd(self) -> float:
        """Lifetime P&L in USD."""
        return sum(t.pnl_usd for t in self._trades)

    @property
    def sharpe_proxy(self) -> float:
        """Simplified Sharpe ratio proxy (mean return / std return)."""
        if len(self._trades) < 5:
            return 0.0
        returns = [t.return_pct for t in self._trades]
        import numpy as np
        arr = np.array(returns, dtype=np.float64)
        std = float(np.std(arr))
        if std <= 0:
            return 0.0
        return float(np.mean(arr)) / std

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        total_capital_usd: float,
        z_score: float,
        strategy: str = "ALGO",
        alpha_score: float = 0.0,
    ) -> PositionSizeResult:
        """Calculate optimal position size using Kelly Criterion.

        Args:
            total_capital_usd: Total available capital in USD.
            z_score: Current Z-Score (used for confidence scaling).
            strategy: Strategy identifier for filtering history.
            alpha_score: Composite alpha score for additional confidence.

        Returns:
            PositionSizeResult with recommended position size.
        """
        # --- Base Kelly ---
        p = self.win_rate
        q = 1.0 - p
        b = self.win_loss_ratio

        kelly_raw = 0.0 if b <= 0 else (p * b - q) / b

        # Clamp to [0, 1] — negative Kelly means don't trade
        kelly_raw = max(0.0, min(1.0, kelly_raw))

        # --- Half-Kelly ---
        kelly_half = kelly_raw * KELLY_FRACTION

        # --- Confidence scaling based on Z-Score ---
        abs_z = abs(z_score)
        if abs_z >= CONFIDENCE_FULL:
            confidence_mult = 1.0
        elif abs_z >= CONFIDENCE_MIN:
            # Linear interpolation between CONFIDENCE_MIN and CONFIDENCE_FULL
            confidence_mult = (abs_z - CONFIDENCE_MIN) / (CONFIDENCE_FULL - CONFIDENCE_MIN)
            confidence_mult = max(0.3, min(1.0, confidence_mult))
        else:
            confidence_mult = 0.0  # Z-Score too weak — don't trade

        # Boost confidence if alpha score confirms direction
        if alpha_score != 0 and abs(alpha_score) > 0.5:
            confidence_mult = min(1.0, confidence_mult * 1.15)

        kelly_scaled = kelly_half * confidence_mult

        # --- Drawdown throttle ---
        drawdown_mult = 1.0
        if self.consecutive_losses >= CONSECUTIVE_LOSS_THROTTLE:
            drawdown_mult = THROTTLE_FACTOR
            # Further reduce for extended losing streaks
            excess = self.consecutive_losses - CONSECUTIVE_LOSS_THROTTLE
            drawdown_mult *= max(0.25, 1.0 - 0.1 * excess)

        kelly_final = kelly_scaled * drawdown_mult

        # --- Apply caps ---
        position_pct = min(kelly_final, MAX_POSITION_PCT)
        position_usd = total_capital_usd * position_pct

        # --- Minimum position check ---
        if position_usd < MIN_POSITION_USD:
            position_usd = 0.0
            position_pct = 0.0
            reasoning = (
                f"Position ${position_usd:.2f} below minimum ${MIN_POSITION_USD}. "
                f"Kelly={kelly_raw:.3f}, scaled={kelly_scaled:.3f}, z={z_score:.2f}."
            )
        elif self.trade_count < MIN_TRADES_FOR_KELLY:
            # Use default sizing before enough data
            position_pct = min(0.10, MAX_POSITION_PCT)  # Conservative 10%
            position_usd = total_capital_usd * position_pct
            reasoning = (
                f"Insufficient history ({self.trade_count}/{MIN_TRADES_FOR_KELLY} trades). "
                f"Using default 10%. Will switch to Kelly after more data."
            )
        else:
            reasoning = (
                f"Kelly: p={p:.2f}, b={b:.2f}, f*={kelly_raw:.3f}, "
                f"half-Kelly={kelly_half:.3f}, confidence={confidence_mult:.2f}, "
                f"drawdown={drawdown_mult:.2f}, final={kelly_final:.3f}."
            )

        log.debug(
            f"Kelly sizing: ${position_usd:.2f} ({position_pct:.1%}) — {reasoning}"
        )

        return PositionSizeResult(
            position_usd=position_usd,
            position_pct=position_pct,
            kelly_raw=kelly_raw,
            kelly_scaled=kelly_scaled,
            confidence_multiplier=confidence_mult,
            drawdown_multiplier=drawdown_mult,
            win_rate=p,
            win_loss_ratio=b,
            trade_count=self.trade_count,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        """Load trade history from disk."""
        if not self._history_path.exists():
            return
        try:
            with open(self._history_path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data.get("trades", [])[-MAX_TRADE_HISTORY:]:
                self._trades.append(TradeResult.from_dict(d))
            log.info(f"Kelly: loaded {len(self._trades)} historical trades.")
        except Exception as exc:
            log.warning(f"Kelly: history load failed: {exc}")

    def _save_history(self) -> None:
        """Persist trade history to disk."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "trades": [t.to_dict() for t in self._trades],
                "stats": {
                    "win_rate": self.win_rate,
                    "win_loss_ratio": self.win_loss_ratio,
                    "total_pnl_usd": self.total_pnl_usd,
                    "trade_count": self.trade_count,
                    "last_updated": time.time(),
                },
            }
            with open(self._history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            log.warning(f"Kelly: history save failed: {exc}")

    def get_stats(self) -> dict:
        """Return comprehensive Kelly statistics."""
        return {
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "win_loss_ratio": self.win_loss_ratio,
            "consecutive_losses": self.consecutive_losses,
            "total_pnl_usd": self.total_pnl_usd,
            "sharpe_proxy": self.sharpe_proxy,
            "min_trades_for_kelly": MIN_TRADES_FOR_KELLY,
            "using_kelly": self.trade_count >= MIN_TRADES_FOR_KELLY,
        }


# ---------------------------------------------------------------------------
# Regime-Aware Kelly
# ---------------------------------------------------------------------------


# Regime multipliers applied to the base Kelly fraction result
_REGIME_MULTIPLIERS: dict[str, float] = {
    "TRENDING_BULL": 1.0,   # bull — unchanged
    "LOW_VOL": 0.8,         # sideways / flat — reduce 20%
    "MEDIUM_VOL": 0.8,      # choppy / sideways — reduce 20%
    "HIGH_VOL": 0.5,        # high vol — halve
    "TRENDING_BEAR": 0.6,   # bear — reduce 40%
    "NEWS_BLACKOUT": 0.2,   # crisis / blackout — near-zero
}


class RegimeAwareKelly:
    """Extends KellyCriterion with regime-conditional sizing.

    Regime multipliers (applied to standard Kelly fraction result):
      bull:      1.0   (unchanged)
      sideways:  0.8   (reduce 20%)
      high_vol:  0.5   (halve — spread eats alpha)
      bear:      0.6   (reduce 40%)
      crisis:    0.2   (near-zero — capital preservation mode)

    Drawdown taper:
      current_drawdown > 5%:  × 0.7
      current_drawdown > 10%: × 0.3
      current_drawdown > 15%: × 0.0 (stop trading)
    """

    def _get_adaptive_kelly_fraction(self, trade_count: int) -> float:
        """Scale Kelly fraction based on trade history depth.

        Prevents aggressive sizing before statistical significance is reached.

        trade_count < 50:   fraction = 0.10 (minimal — nearly flat)
        trade_count < 200:  fraction = 0.15
        trade_count < 500:  fraction = cfg.kelly_fraction * 0.5
        trade_count >= 500: fraction = cfg.kelly_fraction (full configured value)
        """
        from wallet_manager.config import get_settings
        cfg = get_settings()
        if trade_count < 50:
            return 0.10
        elif trade_count < 200:
            return 0.15
        elif trade_count < cfg.min_trades_before_full_kelly:
            return cfg.kelly_fraction * 0.5
        return cfg.kelly_fraction

    def compute_sized_position(
        self,
        base_kelly: float,
        regime: str,           # MarketRegime enum .name
        current_drawdown: float,
        portfolio_value_usd: float,
        max_position_usd: float,
        trade_count: int = 0,          # used for adaptive Kelly scaling
    ) -> float:
        """Final position size in USD after all adjustments.

        Args:
            base_kelly: Raw Kelly fraction result (0.0–1.0).
            regime: MarketRegime enum name (e.g. "TRENDING_BULL").
            current_drawdown: Current portfolio drawdown as a fraction (0.15 = 15%).
            portfolio_value_usd: Total portfolio value in USD.
            max_position_usd: Hard cap on position size in USD.
            trade_count: Number of completed trades (for adaptive Kelly scaling).

        Returns:
            Position size in USD after regime and drawdown adjustments.
        """
        regime_mult = _REGIME_MULTIPLIERS.get(regime, 0.8)

        # Drawdown taper
        abs_dd = abs(current_drawdown)
        if abs_dd > 0.15:
            drawdown_mult = 0.0
        elif abs_dd > 0.10:
            drawdown_mult = 0.3
        elif abs_dd > 0.05:
            drawdown_mult = 0.7
        else:
            drawdown_mult = 1.0

        # Adaptive Kelly: scale fraction by trade history depth
        adaptive_kelly = self._get_adaptive_kelly_fraction(trade_count)
        adjusted_fraction = adaptive_kelly * regime_mult * drawdown_mult
        position_usd = portfolio_value_usd * adjusted_fraction
        position_usd = min(position_usd, max_position_usd)
        position_usd = max(0.0, position_usd)

        log.debug(
            "RegimeAwareKelly: regime=%s regime_mult=%.2f dd=%.1f%% dd_mult=%.2f "
            "adaptive=%.3f adjusted=%.3f position=$%.2f trade_count=%d",
            regime, regime_mult, abs_dd * 100, drawdown_mult,
            adaptive_kelly, adjusted_fraction, position_usd, trade_count,
        )
        return position_usd


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_kelly_instance: KellyCriterion | None = None


def get_kelly_criterion() -> KellyCriterion:
    """Return the process-wide KellyCriterion singleton."""
    global _kelly_instance
    if _kelly_instance is None:
        _kelly_instance = KellyCriterion()
    return _kelly_instance


def reset_kelly_criterion() -> None:
    """Reset the singleton (for tests)."""
    global _kelly_instance
    _kelly_instance = None
