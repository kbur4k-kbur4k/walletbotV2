"""
wallet_manager/strategies/deep_kelly_risk.py

Deep Kelly Risk Engine — PPO Reinforcement Learning Position Sizing.

Replaces static Kelly formulas with a Proximal Policy Optimization (PPO)
agent that learns optimal position sizing from historical trade outcomes,
maximizing for long-term Sharpe Ratio.

Components
----------
1. PPO Agent: Neural network policy that maps (market state, portfolio state)
   → position size fraction. Trained to maximize risk-adjusted returns.

2. Drawdown Controller: Automated circuit breaker that:
   - Reduces sizing by 50% after 5% portfolio drawdown
   - Halts trading completely after 10% drawdown
   - Resumes gradually after recovery

3. Value-at-Risk (VaR): Real-time 99% confidence interval VaR using
   historical simulation. Ensures no single trade can cause catastrophic loss.

4. Regime-Adaptive Sizing: Position size auto-scales based on GARCH
   volatility regime — smaller in high-vol, larger in low-vol.

Architecture
------------
The PPO agent is optional — when not trained, the engine falls back to
an enhanced Kelly Criterion with drawdown + VaR overlays. Training is
done offline via quant_analytics.py.

The state space fed to the PPO agent:
  [alpha_score, garch_sigma, vol_regime, drawdown_pct, win_rate,
   wl_ratio, consecutive_losses, portfolio_value, var_99]

The action space: continuous fraction [0, max_position_pct]
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Position sizing limits
MAX_POSITION_PCT = 0.25          # Never exceed 25% of capital
MIN_POSITION_USD = 5.0           # Floor for gas-efficiency
MIN_TRADES_FOR_RL = 50           # Minimum trades before PPO takes over

# Drawdown control
DRAWDOWN_REDUCE_THRESHOLD = 0.05  # 5% drawdown → reduce sizing by 50%
DRAWDOWN_HALT_THRESHOLD = 0.10    # 10% drawdown → halt all trading
DRAWDOWN_RECOVERY_THRESHOLD = 0.03  # Resume when drawdown < 3%

# VaR configuration
VAR_CONFIDENCE = 0.99            # 99% confidence interval
VAR_LOOKBACK = 100               # Historical returns for VaR
VAR_MAX_LOSS_PCT = 0.05          # Max 5% portfolio VaR per trade

# PPO hyperparameters (for offline training)
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_EPSILON_CLIP = 0.2
PPO_HIDDEN_DIM = 64
PPO_STATE_DIM = 9
PPO_ACTION_DIM = 1

# Kelly fallback
KELLY_FRACTION = 0.5
DEFAULT_WIN_RATE = 0.55
DEFAULT_WL_RATIO = 1.5

# Persistence
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models"
TRADE_HISTORY_PATH = Path(__file__).resolve().parent.parent / "deep_kelly_history.json"
MAX_TRADE_HISTORY = 1000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeOutcome:
    """Record of a completed trade for RL training."""

    pnl_usd: float
    pnl_pct: float
    position_pct: float          # Fraction of capital allocated
    entry_alpha: float           # Alpha score at entry
    entry_garch_sigma: float     # GARCH volatility at entry
    entry_vol_regime: str
    holding_periods: int         # How many 5-min periods held
    timestamp: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl_usd > 0

    def to_dict(self) -> dict:
        return {
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "position_pct": self.position_pct,
            "entry_alpha": self.entry_alpha,
            "entry_garch_sigma": self.entry_garch_sigma,
            "entry_vol_regime": self.entry_vol_regime,
            "holding_periods": self.holding_periods,
            "timestamp": self.timestamp or time.time(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TradeOutcome:
        return cls(
            pnl_usd=float(d.get("pnl_usd", 0)),
            pnl_pct=float(d.get("pnl_pct", 0)),
            position_pct=float(d.get("position_pct", 0)),
            entry_alpha=float(d.get("entry_alpha", 0)),
            entry_garch_sigma=float(d.get("entry_garch_sigma", 0)),
            entry_vol_regime=str(d.get("entry_vol_regime", "normal")),
            holding_periods=int(d.get("holding_periods", 1)),
            timestamp=float(d.get("timestamp", 0)),
        )


@dataclass(frozen=True)
class RiskDecision:
    """Output of the Deep Kelly Risk Engine.

    Attributes:
        position_pct: Recommended position size as fraction of capital.
        position_usd: Recommended position size in USD.
        source: 'ppo', 'kelly_enhanced', or 'halted'.
        var_99_pct: Portfolio VaR at 99% confidence.
        drawdown_pct: Current drawdown from peak equity.
        drawdown_state: 'normal', 'reduced', or 'halted'.
        vol_regime: Volatility regime used for adjustment.
        sharpe_estimate: Rolling Sharpe ratio estimate.
        reasoning: Human-readable explanation.
    """

    position_pct: float
    position_usd: float
    source: str
    var_99_pct: float
    drawdown_pct: float
    drawdown_state: str
    vol_regime: str
    sharpe_estimate: float
    reasoning: str


# ---------------------------------------------------------------------------
# PPO Agent (Simplified for On-Device Inference)
# ---------------------------------------------------------------------------


class PPOAgent:
    """Proximal Policy Optimization agent for position sizing.

    The agent learns to map market/portfolio state to optimal position
    size. Training is done offline; this class handles inference.

    State space (9 dimensions):
        [alpha_score, garch_sigma, vol_regime_encoded, drawdown_pct,
         win_rate, wl_ratio, consecutive_losses, portfolio_value_norm, var_99]

    Action space: continuous [0, MAX_POSITION_PCT]
    """

    def __init__(self) -> None:
        self._policy_net: Any | None = None
        self._is_loaded = False

    def load_model(self, model_path: Path | None = None) -> bool:
        """Load pre-trained PPO policy network."""
        path = model_path or (MODEL_DIR / "ppo_risk_policy.pt")
        try:
            import torch
            self._policy_net = torch.jit.load(str(path), map_location="cpu")
            policy_net = self._policy_net
            if policy_net is None:
                raise RuntimeError("PPO policy failed to load")
            policy_net.eval()
            self._is_loaded = True
            log.info(f"PPO policy loaded from {path}")
            return True
        except Exception as exc:
            log.info(f"PPO model not found ({exc}). Using Kelly fallback.")
            self._is_loaded = False
            return False

    def predict_position_size(self, state: np.ndarray) -> float:
        """Predict optimal position size from state vector.

        Args:
            state: 9-dimensional state vector (normalized).

        Returns:
            Position size as fraction of capital [0, MAX_POSITION_PCT].
        """
        if not self._is_loaded or self._policy_net is None:
            return -1.0  # Signal to use fallback

        try:
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self._policy_net(state_tensor)
                position_pct = torch.sigmoid(action).item() * MAX_POSITION_PCT
            return float(max(0.0, min(MAX_POSITION_PCT, position_pct)))
        except Exception as exc:
            log.debug(f"PPO inference failed: {exc}")
            return -1.0


# ---------------------------------------------------------------------------
# Drawdown Controller
# ---------------------------------------------------------------------------


class DrawdownController:
    """Portfolio drawdown monitoring and position throttling.

    Tracks peak equity and current drawdown. Implements tiered
    response:
      - 0-5% drawdown: Normal operations
      - 5-10% drawdown: Reduce position sizes by 50%
      - >10% drawdown: Complete trading halt

    Recovery: Trading resumes at normal sizing when drawdown
    recovers below DRAWDOWN_RECOVERY_THRESHOLD.
    """

    def __init__(self) -> None:
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._drawdown_pct: float = 0.0
        self._state: str = "normal"  # normal, reduced, halted

    def update(self, equity_usd: float) -> str:
        """Update equity and recalculate drawdown.

        Args:
            equity_usd: Current total portfolio value in USD.

        Returns:
            Drawdown state: 'normal', 'reduced', or 'halted'.
        """
        self._current_equity = equity_usd

        if equity_usd > self._peak_equity:
            self._peak_equity = equity_usd

        if self._peak_equity > 0:
            self._drawdown_pct = (self._peak_equity - equity_usd) / self._peak_equity
        else:
            self._drawdown_pct = 0.0

        # State transitions
        if self._drawdown_pct >= DRAWDOWN_HALT_THRESHOLD:
            if self._state != "halted":
                log.critical(
                    f"RISK: Drawdown {self._drawdown_pct:.1%} >= {DRAWDOWN_HALT_THRESHOLD:.0%}. "
                    f"HALTING all trading."
                )
            self._state = "halted"
        elif self._drawdown_pct >= DRAWDOWN_REDUCE_THRESHOLD:
            if self._state == "normal":
                log.warning(
                    f"RISK: Drawdown {self._drawdown_pct:.1%} >= {DRAWDOWN_REDUCE_THRESHOLD:.0%}. "
                    f"Reducing position sizes by 50%."
                )
            self._state = "reduced"
        elif self._drawdown_pct < DRAWDOWN_RECOVERY_THRESHOLD:
            if self._state != "normal":
                log.info(
                    f"RISK: Drawdown recovered to {self._drawdown_pct:.1%}. "
                    f"Resuming normal operations."
                )
            self._state = "normal"

        return self._state

    @property
    def drawdown_pct(self) -> float:
        return self._drawdown_pct

    @property
    def state(self) -> str:
        return self._state

    @property
    def sizing_multiplier(self) -> float:
        """Multiplier to apply to position sizes."""
        if self._state == "halted":
            return 0.0
        elif self._state == "reduced":
            return 0.5
        return 1.0


# ---------------------------------------------------------------------------
# Value-at-Risk (VaR) Calculator
# ---------------------------------------------------------------------------


class VaRCalculator:
    """Historical simulation VaR at 99% confidence.

    Uses the last N trade returns to estimate the maximum expected
    loss at the given confidence level. Ensures no single position
    can cause catastrophic portfolio damage.
    """

    def __init__(
        self,
        confidence: float = VAR_CONFIDENCE,
        lookback: int = VAR_LOOKBACK,
    ) -> None:
        self._confidence = confidence
        self._returns: deque[float] = deque(maxlen=lookback)

    def add_return(self, pct_return: float) -> None:
        """Add a trade return (as fraction, e.g., 0.02 = 2%)."""
        self._returns.append(pct_return)

    def calculate(self) -> float:
        """Calculate VaR at configured confidence level.

        Returns:
            VaR as positive fraction (e.g., 0.05 = 5% max loss).
            Returns a conservative default if insufficient data.
        """
        if len(self._returns) < 10:
            return VAR_MAX_LOSS_PCT  # Conservative default

        returns_arr = np.array(self._returns, dtype=np.float64)
        # Historical VaR: percentile of losses
        var_percentile = (1.0 - self._confidence) * 100  # e.g., 1st percentile
        var_value = float(np.percentile(returns_arr, var_percentile))

        # VaR is the loss (negative return), express as positive
        return abs(min(var_value, 0.0))

    def max_position_pct(self, target_var: float = VAR_MAX_LOSS_PCT) -> float:
        """Calculate max position size to stay within VaR target.

        Args:
            target_var: Maximum acceptable VaR (e.g., 0.05 = 5%).

        Returns:
            Max position fraction to keep portfolio VaR under target.
        """
        current_var = self.calculate()
        if current_var <= 0:
            return MAX_POSITION_PCT

        # Position size = target_var / trade_var
        max_pct = target_var / current_var
        return min(max_pct, MAX_POSITION_PCT)


# ---------------------------------------------------------------------------
# Deep Kelly Risk Engine (Main Class)
# ---------------------------------------------------------------------------


class DeepKellyRiskEngine:
    """Institutional risk management engine with PPO + drawdown + VaR.

    Hierarchy of position sizing:
    1. PPO agent (if trained, >50 trades) → optimal fraction
    2. Enhanced Kelly Criterion (fallback) → half-Kelly with adjustments
    3. Drawdown controller → multiplier (1.0, 0.5, or 0.0)
    4. VaR constraint → hard cap on max position
    5. Regime filter → vol-adjusted scaling

    The final position size is: min(source_size × drawdown_mult, var_cap)

    Usage:
        engine = DeepKellyRiskEngine()
        engine.update_equity(1500.0)

        decision = engine.calculate_position(
            total_capital_usd=1500.0,
            alpha_score=0.7,
            garch_sigma=0.001,
            vol_regime="normal",
        )
        print(f"Allocate: ${decision.position_usd:.2f}")
    """

    def __init__(self) -> None:
        # --- Legacy components (kept for backward compat) ---
        self.ppo = PPOAgent()
        self.drawdown = DrawdownController()
        self.var = VaRCalculator()

        # --- Section 4.2 upgraded components ---
        self.tail_risk = TailRiskManager()
        self.portfolio_heat_map = PortfolioHeatMap()
        self.adaptive_drawdown = AdaptiveDrawdownController()
        self.ppo_v2 = PPOAgentV2()

        # Trade history for Kelly/PPO
        self._trades: deque[TradeOutcome] = deque(maxlen=MAX_TRADE_HISTORY)
        self._consecutive_losses: int = 0

        # Load models
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.ppo.load_model()
        self.ppo_v2.load_model()
        self._load_history()

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, outcome: TradeOutcome) -> None:
        """Record a completed trade for RL training and statistics."""
        if outcome.timestamp <= 0:
            outcome = TradeOutcome(
                pnl_usd=outcome.pnl_usd,
                pnl_pct=outcome.pnl_pct,
                position_pct=outcome.position_pct,
                entry_alpha=outcome.entry_alpha,
                entry_garch_sigma=outcome.entry_garch_sigma,
                entry_vol_regime=outcome.entry_vol_regime,
                holding_periods=outcome.holding_periods,
                timestamp=time.time(),
            )
        self._trades.append(outcome)
        self.var.add_return(outcome.pnl_pct)
        self.tail_risk.update(outcome.pnl_pct)
        self.portfolio_heat_map.update(outcome.position_pct, outcome.pnl_pct)
        self.ppo_v2.increment_trade_count()

        if outcome.is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self.adaptive_drawdown.record_loss(outcome.pnl_usd)

        self._save_history()

    def update_equity(self, equity_usd: float) -> str:
        """Update portfolio equity for drawdown tracking.

        Returns:
            Drawdown state: 'normal', 'reduced', 'halted'.
        """
        self.adaptive_drawdown.update_equity(equity_usd)
        return self.drawdown.update(equity_usd)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position(
        self,
        total_capital_usd: float,
        alpha_score: float,
        garch_sigma: float,
        vol_regime: str = "normal",
        confidence: float = 1.0,
    ) -> RiskDecision:
        """Calculate optimal position size using PPO or Kelly fallback.

        Args:
            total_capital_usd: Total portfolio value.
            alpha_score: ML alpha signal in [-1, +1].
            garch_sigma: Current GARCH conditional volatility.
            vol_regime: 'low', 'normal', or 'high'.
            confidence: Signal confidence [0, 1].

        Returns:
            RiskDecision with position size and diagnostics.
        """
        # Drawdown check first
        dd_state = self.drawdown.state
        if dd_state == "halted":
            return RiskDecision(
                position_pct=0.0,
                position_usd=0.0,
                source="halted",
                var_99_pct=self.var.calculate(),
                drawdown_pct=self.drawdown.drawdown_pct,
                drawdown_state="halted",
                vol_regime=vol_regime,
                sharpe_estimate=self._sharpe_estimate(),
                reasoning=(
                    f"Trading HALTED: drawdown {self.drawdown.drawdown_pct:.1%} "
                    f">= {DRAWDOWN_HALT_THRESHOLD:.0%}. Wait for recovery."
                ),
            )

        if alpha_score <= 0:
            var_99 = self.var.calculate()
            return RiskDecision(
                position_pct=0.0,
                position_usd=0.0,
                source="kelly_enhanced",
                var_99_pct=var_99,
                drawdown_pct=self.drawdown.drawdown_pct,
                drawdown_state=dd_state,
                vol_regime=vol_regime,
                sharpe_estimate=self._sharpe_estimate(),
                reasoning="Non-positive alpha: no long exposure opened.",
            )

        # Build PPO state vector
        win_rate = self._win_rate()
        wl_ratio = self._wl_ratio()
        var_99 = self.var.calculate()
        portfolio_norm = total_capital_usd / 1000.0  # Normalize to ~1.0 scale

        state = np.array([
            alpha_score,
            garch_sigma * 1000,  # Scale up for network
            {"low": -1, "normal": 0, "high": 1}.get(vol_regime, 0),
            self.drawdown.drawdown_pct,
            win_rate,
            wl_ratio,
            min(self._consecutive_losses / 5.0, 1.0),
            portfolio_norm,
            var_99,
        ], dtype=np.float64)

        # Try PPO first
        source = "kelly_enhanced"
        position_pct = -1.0

        if len(self._trades) >= MIN_TRADES_FOR_RL:
            ppo_result = self.ppo.predict_position_size(state)
            if ppo_result >= 0:
                position_pct = ppo_result
                source = "ppo"

        # Fallback: Enhanced Kelly
        if position_pct < 0:
            position_pct = self._enhanced_kelly(
                alpha_score=alpha_score,
                garch_sigma=garch_sigma,
                vol_regime=vol_regime,
                confidence=confidence,
            )

        # Apply drawdown multiplier
        position_pct *= self.drawdown.sizing_multiplier

        # VaR constraint: cap position to keep VaR under target
        var_cap = self.var.max_position_pct(VAR_MAX_LOSS_PCT)
        position_pct = min(position_pct, var_cap)

        # Volatility regime adjustment
        regime_mult = {"low": 1.2, "normal": 1.0, "high": 0.6}.get(vol_regime, 1.0)
        position_pct *= regime_mult

        # Final clamp
        position_pct = max(0.0, min(MAX_POSITION_PCT, position_pct))
        position_usd = total_capital_usd * position_pct

        if position_usd < MIN_POSITION_USD:
            position_pct = 0.0
            position_usd = 0.0

        reasoning = (
            f"src={source} kelly_wr={win_rate:.2f} wl={wl_ratio:.2f} "
            f"dd={self.drawdown.drawdown_pct:.1%}({dd_state}) "
            f"var99={var_99:.3f} regime={vol_regime}×{regime_mult:.1f} "
            f"consec_loss={self._consecutive_losses}"
        )

        log.debug(f"RISK: ${position_usd:.2f} ({position_pct:.1%}) — {reasoning}")

        return RiskDecision(
            position_pct=position_pct,
            position_usd=position_usd,
            source=source,
            var_99_pct=var_99,
            drawdown_pct=self.drawdown.drawdown_pct,
            drawdown_state=dd_state,
            vol_regime=vol_regime,
            sharpe_estimate=self._sharpe_estimate(),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Enhanced Kelly (fallback)
    # ------------------------------------------------------------------

    def _enhanced_kelly(
        self,
        alpha_score: float,
        garch_sigma: float,
        vol_regime: str,
        confidence: float,
    ) -> float:
        """Enhanced Kelly Criterion with GARCH and alpha adjustments.

        Kelly formula: f* = (p × b - q) / b
        Enhanced with:
        - Alpha-based win probability adjustment
        - GARCH inverse scaling (less when volatile)
        - Confidence multiplier
        - Consecutive loss throttle
        """
        # No position for near-zero signals (direction-agnostic)
        abs_alpha = abs(alpha_score)
        if abs_alpha <= 0:
            return 0.0

        p = self._win_rate()
        b = self._wl_ratio()
        q = 1.0 - p

        if b <= 0:
            return 0.0

        kelly_raw = max(0.0, (p * b - q) / b)
        kelly_half = kelly_raw * KELLY_FRACTION

        # Alpha confidence: scale by signal strength (direction-agnostic)
        alpha_mult = min(1.0, abs_alpha / 0.6)

        # GARCH inverse scaling: higher vol → smaller position
        long_run_sigma = max(self._median_garch_sigma(), 1e-8)
        vol_scale = min(1.5, long_run_sigma / max(garch_sigma, 1e-8))
        vol_scale = max(0.3, vol_scale)

        # Consecutive loss throttle
        loss_mult = 1.0
        if self._consecutive_losses >= 3:
            loss_mult = max(0.2, 1.0 - 0.15 * (self._consecutive_losses - 2))

        position_pct = kelly_half * alpha_mult * vol_scale * confidence * loss_mult
        return max(0.0, min(MAX_POSITION_PCT, position_pct))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _win_rate(self) -> float:
        if not self._trades:
            return DEFAULT_WIN_RATE
        wins = sum(1 for t in self._trades if t.is_win)
        return wins / len(self._trades)

    def _wl_ratio(self) -> float:
        wins = [t.pnl_pct for t in self._trades if t.is_win]
        losses = [-t.pnl_pct for t in self._trades if not t.is_win]
        if not wins or not losses:
            return DEFAULT_WL_RATIO
        avg_w = sum(wins) / len(wins)
        avg_l = sum(losses) / len(losses)
        return avg_w / avg_l if avg_l > 0 else DEFAULT_WL_RATIO

    def _sharpe_estimate(self) -> float:
        if len(self._trades) < 5:
            return 0.0
        returns = np.array([t.pnl_pct for t in self._trades], dtype=np.float64)
        std = float(np.std(returns))
        if std <= 0:
            return 0.0
        return float(np.mean(returns)) / std

    def _median_garch_sigma(self) -> float:
        sigmas = [t.entry_garch_sigma for t in self._trades if t.entry_garch_sigma > 0]
        if not sigmas:
            return 0.001
        return float(np.median(sigmas))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        try:
            with open(TRADE_HISTORY_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as exc:
            log.warning(f"DeepKelly: history load failed: {exc}")
            return
        try:
            for d in data.get("trades", [])[-MAX_TRADE_HISTORY:]:
                outcome = TradeOutcome.from_dict(d)
                self._trades.append(outcome)
                self.var.add_return(outcome.pnl_pct)
            log.info(f"DeepKelly: loaded {len(self._trades)} historical trades.")
        except Exception as exc:
            log.warning(f"DeepKelly: history load failed: {exc}")

    def _save_history(self) -> None:
        try:
            TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "trades": [t.to_dict() for t in self._trades],
                "stats": {
                    "win_rate": self._win_rate(),
                    "wl_ratio": self._wl_ratio(),
                    "sharpe": self._sharpe_estimate(),
                    "trade_count": len(self._trades),
                    "var_99": self.var.calculate(),
                    "last_updated": time.time(),
                },
            }
            with open(TRADE_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            log.warning(f"DeepKelly: history save failed: {exc}")

    def get_stats(self) -> dict:
        """Return comprehensive risk engine statistics."""
        return {
            "trade_count": len(self._trades),
            "win_rate": self._win_rate(),
            "wl_ratio": self._wl_ratio(),
            "sharpe": self._sharpe_estimate(),
            "consecutive_losses": self._consecutive_losses,
            "var_99": self.var.calculate(),
            "drawdown_pct": self.drawdown.drawdown_pct,
            "drawdown_state": self.drawdown.state,
            "peak_equity": self.drawdown._peak_equity,
            "ppo_loaded": self.ppo._is_loaded,
            "using_ppo": self.ppo._is_loaded and len(self._trades) >= MIN_TRADES_FOR_RL,
            # Section 4.2 additions
            "cvar_95": self.tail_risk.cvar,
            "gpd_xi": self.tail_risk.gpd_xi,
            "tail_risk_signal": self.tail_risk.tail_risk_signal,
            "portfolio_heat": self.portfolio_heat_map.heat,
            "heat_overheated": self.portfolio_heat_map.is_overheated,
            "adaptive_dd_state": self.adaptive_drawdown.state,
            "adaptive_dd_pct": self.adaptive_drawdown.drawdown_pct,
            "revenge_cooldown": self.adaptive_drawdown.in_revenge_cooldown,
            "ppo_v2_loaded": self.ppo_v2.is_loaded,
            "ppo_v2_curriculum_stage": self.ppo_v2.curriculum_stage(),
        }


# ---------------------------------------------------------------------------
# PortfolioHeatMap — Markowitz-inspired concentration risk heat map
# ---------------------------------------------------------------------------


class PortfolioHeatMap:
    """Tracks portfolio concentration and downside-amplified risk heat.

    Uses Herfindahl-Hirschman Index (HHI) on rolling position sizes,
    amplified by downside realized volatility. Heat in [0, 1]; ≥0.7 is
    considered "overheated" and triggers conservative sizing.
    """

    OVERHEAT_THRESHOLD = 0.7

    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._position_sizes: deque[float] = deque(maxlen=window)
        self._returns: deque[float] = deque(maxlen=window)
        self._heat: float = 0.0

    def update(self, position_pct: float, portfolio_return: float) -> None:
        self._position_sizes.append(max(0.0, position_pct))
        self._returns.append(portfolio_return)
        self._heat = self._compute()

    def _compute(self) -> float:
        if len(self._position_sizes) < 5:
            return 0.0
        pos = np.array(self._position_sizes, dtype=np.float64)
        ret = np.array(self._returns, dtype=np.float64)

        total = float(np.sum(pos))
        # HHI of normalized position weights
        hhi = float(np.sum(pos ** 2) / max(total ** 2, 1e-10))

        # Amplify by downside volatility (semi-deviation)
        neg_rets = ret[ret < 0]
        downside_vol = float(np.std(neg_rets)) if len(neg_rets) >= 3 else 0.0

        return min(1.0, hhi * (1.0 + downside_vol * 100.0))

    @property
    def heat(self) -> float:
        return self._heat

    @property
    def is_overheated(self) -> bool:
        return self._heat >= self.OVERHEAT_THRESHOLD


# ---------------------------------------------------------------------------
# TailRiskManager — CVaR and GPD tail-fit risk estimation
# ---------------------------------------------------------------------------


class TailRiskManager:
    """Conditional Value at Risk (Expected Shortfall) with GPD tail fit.

    CVaR (α) = E[loss | loss > VaR(α)]

    GPD (Generalized Pareto Distribution) is fit to exceedances beyond the
    VaR threshold using method-of-moments estimation. This improves tail
    quantile accuracy compared to historical simulation alone.
    """

    def __init__(self, window: int = 100, confidence: float = 0.95) -> None:
        self._window = window
        self._confidence = confidence
        self._returns: deque[float] = deque(maxlen=window)
        self._cvar: float = 0.0
        self._gpd_xi: float = 0.0    # GPD shape (ξ > 0 = heavy tail)
        self._gpd_sigma: float = 1.0  # GPD scale (σ)

    def update(self, log_return: float) -> None:
        self._returns.append(log_return)
        if len(self._returns) >= 20:
            self._cvar = self._calculate_cvar()
            self._fit_gpd_tail()

    def _calculate_cvar(self) -> float:
        rets = sorted(self._returns)  # ascending (worst first after slicing)
        n = len(rets)
        cutoff = int(n * (1.0 - self._confidence))
        tail = rets[: max(cutoff, 1)]
        return abs(float(np.mean(tail)))

    def _fit_gpd_tail(self) -> None:
        """Method-of-moments GPD fit to lower-tail exceedances."""
        rets = np.array(self._returns, dtype=np.float64)
        threshold = float(np.quantile(rets, 1.0 - self._confidence))
        excess = -(rets[rets < threshold] - threshold)  # positive exceedances

        if len(excess) < 3:
            return
        m = float(np.mean(excess))
        v = float(np.var(excess))
        if v < 1e-10:
            return
        ratio = m ** 2 / v
        self._gpd_xi = 0.5 * (1.0 - ratio)
        self._gpd_sigma = 0.5 * m * (1.0 + ratio)

    @property
    def cvar(self) -> float:
        return self._cvar

    @property
    def gpd_xi(self) -> float:
        return self._gpd_xi

    @property
    def gpd_sigma(self) -> float:
        return self._gpd_sigma

    @property
    def tail_risk_signal(self) -> float:
        """Normalised tail risk [0, 1]; 1.0 corresponds to CVaR ≥ 5%."""
        return min(1.0, self._cvar / 0.05)


# ---------------------------------------------------------------------------
# AdaptiveDrawdownController — vol-adjusted thresholds + anti-revenge logic
# ---------------------------------------------------------------------------


class AdaptiveDrawdownController:
    """Drawdown controller with two enhancements over the base DrawdownController:

    1. Vol-adjusted thresholds: When GARCH σ is elevated the halt/reduce
       thresholds tighten by up to 40%, preventing riding out larger losses
       in turbulent markets.

    2. Anti-revenge trading: Detects ≥3 losses within 15 minutes and
       imposes a 30-minute cooldown to prevent emotional re-entries.
    """

    # Anti-revenge parameters
    REVENGE_MIN_LOSSES = 3
    REVENGE_WINDOW_S = 900    # 15 min
    REVENGE_COOLDOWN_S = 1800  # 30 min

    def __init__(self) -> None:
        self._peak_equity: float = 0.0
        self._drawdown_pct: float = 0.0
        self._state: str = "normal"
        self._garch_sigma: float = 0.0
        self._loss_times: deque[float] = deque(maxlen=20)
        self._revenge_cooldown_until: float = 0.0

    # --- Public update methods ---

    def update_equity(self, equity: float, garch_sigma: float = 0.0) -> str:
        """Update equity and garch sigma; returns current state."""
        if garch_sigma > 0:
            self._garch_sigma = garch_sigma
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            self._drawdown_pct = (self._peak_equity - equity) / self._peak_equity
        self._update_state()
        return self._state

    def record_loss(self, pnl_usd: float) -> None:
        """Record a loss for anti-revenge detection."""
        if pnl_usd < 0:
            self._loss_times.append(time.time())
            self._check_revenge()

    # --- Internal helpers ---

    def _vol_adjusted_reduce(self) -> float:
        base = DRAWDOWN_REDUCE_THRESHOLD
        if self._garch_sigma > 0.03:
            factor = min(1.4, 1.0 + (self._garch_sigma - 0.03) / 0.03 * 0.4)
            return base / factor
        return base

    def _vol_adjusted_halt(self) -> float:
        base = DRAWDOWN_HALT_THRESHOLD
        if self._garch_sigma > 0.03:
            factor = min(1.3, 1.0 + (self._garch_sigma - 0.03) / 0.03 * 0.3)
            return base / factor
        return base

    def _update_state(self) -> None:
        halt_t = self._vol_adjusted_halt()
        reduce_t = self._vol_adjusted_reduce()

        if self._drawdown_pct >= halt_t:
            self._state = "halted"
        elif self._drawdown_pct >= reduce_t:
            self._state = "reduced"
        elif self._drawdown_pct < DRAWDOWN_RECOVERY_THRESHOLD:
            self._state = "normal"

    def _check_revenge(self) -> None:
        now = time.time()
        recent = [t for t in self._loss_times if now - t <= self.REVENGE_WINDOW_S]
        if len(recent) >= self.REVENGE_MIN_LOSSES:
            self._revenge_cooldown_until = now + self.REVENGE_COOLDOWN_S
            log.warning(
                f"AdaptiveDD: anti-revenge triggered — "
                f"{len(recent)} losses in {self.REVENGE_WINDOW_S}s, "
                f"cooldown {self.REVENGE_COOLDOWN_S}s"
            )

    # --- Query methods ---

    def is_trading_allowed(self) -> bool:
        if self._state == "halted":
            return False
        return time.time() >= self._revenge_cooldown_until

    def get_size_multiplier(self) -> float:
        if not self.is_trading_allowed():
            return 0.0
        if self._state == "reduced":
            return 0.5
        return 1.0

    @property
    def drawdown_pct(self) -> float:
        return self._drawdown_pct

    @property
    def state(self) -> str:
        return self._state

    @property
    def in_revenge_cooldown(self) -> bool:
        return time.time() < self._revenge_cooldown_until


# ---------------------------------------------------------------------------
# PPOAgentV2 — 12-dim state with curriculum learning
# ---------------------------------------------------------------------------


class PPOAgentV2:
    """Extended PPO agent with 12-dimensional state and curriculum learning.

    Curriculum stages gate state complexity to stabilise training:
        Stage 1 (< 500 trades): 9-dim base state (same as PPOAgent).
        Stage 2 (500–1999 trades): 10-dim (+ cvar_pct).
        Stage 3 (≥ 2000 trades): 12-dim full state (+cvar, heat, tail_risk).

    The 3 new state components vs PPOAgent:
        [9]  cvar_pct       — CVaR as fraction of capital
        [10] portfolio_heat — concentration heat [0, 1]
        [11] tail_risk      — GPD-based tail risk signal [0, 1]
    """

    STATE_DIM_V2 = 12
    STAGE2_THRESHOLD = 500
    STAGE3_THRESHOLD = 2000

    def __init__(self) -> None:
        self._policy_net: Any | None = None
        self._is_loaded = False
        self._trade_count: int = 0

    def load_model(self, model_path: Path | None = None) -> bool:
        path = model_path or (MODEL_DIR / "ppo_risk_policy_v2.pt")
        if not path.exists():
            log.debug(f"PPOAgentV2 model not found at {path}. Using Kelly fallback.")
            return False
        try:
            import torch
            self._policy_net = torch.jit.load(str(path), map_location="cpu")
            policy_net = self._policy_net
            if policy_net is None:
                raise RuntimeError("PPOAgentV2 policy failed to load")
            policy_net.eval()
            self._is_loaded = True
            log.info(f"PPOAgentV2 loaded from {path}")
            return True
        except Exception as exc:
            log.warning(f"PPOAgentV2 failed to load ({exc}). Using Kelly fallback.")
            return False

    def curriculum_stage(self) -> int:
        if self._trade_count >= self.STAGE3_THRESHOLD:
            return 3
        if self._trade_count >= self.STAGE2_THRESHOLD:
            return 2
        return 1

    def build_state(
        self,
        alpha_score: float,
        garch_sigma: float,
        vol_regime: str,
        drawdown_pct: float,
        win_rate: float,
        wl_ratio: float,
        consecutive_losses: int,
        portfolio_value_norm: float,
        var_99: float,
        cvar_pct: float = 0.0,
        portfolio_heat: float = 0.0,
        tail_risk_signal: float = 0.0,
    ) -> np.ndarray:
        """Build state array according to current curriculum stage."""
        vol_enc = {"low": -1.0, "normal": 0.0, "high": 1.0}.get(vol_regime, 0.0)
        base = np.array([
            alpha_score,
            garch_sigma * 1000.0,
            vol_enc,
            drawdown_pct,
            win_rate,
            wl_ratio,
            min(consecutive_losses / 5.0, 1.0),
            portfolio_value_norm,
            var_99,
        ], dtype=np.float64)

        stage = self.curriculum_stage()
        if stage == 1:
            return base
        extra = np.array([cvar_pct, portfolio_heat, tail_risk_signal], dtype=np.float64)
        if stage == 2:
            return np.concatenate([base, extra[:1]])
        return np.concatenate([base, extra])

    def predict(self, state: np.ndarray) -> float:
        """Return recommended position fraction or -1.0 (use fallback)."""
        if self._is_loaded and self._policy_net is not None:
            try:
                import torch
                tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = self._policy_net(tensor)
                return float(np.clip(action.item(), 0.0, MAX_POSITION_PCT))
            except Exception as exc:
                log.debug(f"PPOAgentV2 inference failed: {exc}")
        return -1.0

    def increment_trade_count(self) -> None:
        self._trade_count += 1

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: DeepKellyRiskEngine | None = None


def get_deep_kelly_engine() -> DeepKellyRiskEngine:
    """Return the process-wide DeepKellyRiskEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DeepKellyRiskEngine()
    return _engine_instance


def reset_deep_kelly_engine() -> None:
    """Reset the singleton (for tests)."""
    global _engine_instance
    _engine_instance = None
