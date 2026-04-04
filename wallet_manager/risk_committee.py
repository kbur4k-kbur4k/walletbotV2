"""
wallet_manager/risk_committee.py

Phase 6: Risk Committee
========================
A second defensive layer above the execution core.  Rules-first,
score-second design.  The Risk Committee evaluates the current state
and returns a verdict (ALLOW / BLOCK / REDUCE) with a reason code
before each trade cycle.

Brakes implemented:
- DailyLossLimit          : halt if daily realized PnL < -threshold
- ConsecutiveLossBrake    : halt after N consecutive losses
- HighSpreadBrake         : skip if bid-ask spread > threshold
- LowLiquidityBrake       : skip if estimated pool depth is too thin
- HighVolatilityBrake     : reduce/skip if 1h volatility > threshold
- ModelDisagreementBrake  : skip if ensemble models disagree
- ConfidenceDecayBrake    : skip if recent confidence is declining
- SessionKillSwitch       : manual emergency stop flag

Design rules:
- Rules first: each brake is a hard yes/no gate
- Score second: overall risk score 0–100
- AI explanation: optional narrative (Phase 8 hook)
- No brake modifies state — all brakes are read-only observers
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class BrakeVerdict(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REDUCE = "REDUCE"   # Trade allowed but size should be reduced


@dataclass(frozen=True)
class BrakeResult:
    brake_name: str
    verdict: BrakeVerdict
    reason: str
    severity: int       # 0 = info, 1 = warn, 2 = block
    value: float = 0.0
    threshold: float = 0.0

    def is_blocking(self) -> bool:
        return self.verdict == BrakeVerdict.BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "brake": self.brake_name,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "severity": self.severity,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class CommitteeVerdict:
    """Aggregate verdict from all brakes."""
    allowed: bool
    overall_verdict: BrakeVerdict
    risk_score: int                         # 0 = no risk, 100 = maximum risk
    brake_results: list[BrakeResult] = field(default_factory=list)
    blocking_brakes: list[str] = field(default_factory=list)
    narrative: str = ""
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "overall_verdict": self.overall_verdict.value,
            "risk_score": self.risk_score,
            "brake_results": [b.to_dict() for b in self.brake_results],
            "blocking_brakes": self.blocking_brakes,
            "narrative": self.narrative,
            "evaluated_at": self.evaluated_at,
        }


# ---------------------------------------------------------------------------
# Individual Brakes
# ---------------------------------------------------------------------------


class DailyLossLimitBrake:
    """Halt trading if daily realized PnL falls below a threshold."""

    def __init__(self, max_daily_loss_usd: float = 50.0) -> None:
        self.threshold = max_daily_loss_usd

    async def evaluate(self, paper_engine: Any | None) -> BrakeResult:
        if not paper_engine:
            return BrakeResult("DailyLossLimit", BrakeVerdict.ALLOW, "No engine available", 0)

        conn = paper_engine._ledger._get_conn()
        day_start = time.time() - 86400.0
        async with conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0.0) AS daily_pnl FROM paper_trades "
            "WHERE direction = 'SELL' AND timestamp >= ?",
            (day_start,),
        ) as cur:
            row = await cur.fetchone()
        daily_pnl = float(row["daily_pnl"]) if row else 0.0

        if daily_pnl < -self.threshold:
            return BrakeResult(
                "DailyLossLimit", BrakeVerdict.BLOCK,
                f"Daily loss ${daily_pnl:.2f} exceeds limit -${self.threshold:.2f}",
                severity=2, value=daily_pnl, threshold=-self.threshold,
            )
        return BrakeResult(
            "DailyLossLimit", BrakeVerdict.ALLOW,
            f"Daily PnL ${daily_pnl:.2f} within limit", 0,
            value=daily_pnl, threshold=-self.threshold,
        )


class ConsecutiveLossBrake:
    """Halt after N consecutive losing closed trades."""

    def __init__(self, max_consecutive: int = 4) -> None:
        self.max_consecutive = max_consecutive

    async def evaluate(self, paper_engine: Any | None) -> BrakeResult:
        if not paper_engine:
            return BrakeResult("ConsecutiveLoss", BrakeVerdict.ALLOW, "No engine available", 0)

        conn = paper_engine._ledger._get_conn()
        async with conn.execute(
            "SELECT pnl_usd FROM paper_trades WHERE direction = 'SELL' ORDER BY timestamp DESC LIMIT ?",
            (self.max_consecutive + 1,),
        ) as cur:
            rows = await cur.fetchall()

        consecutive = 0
        for row in rows:
            if float(row["pnl_usd"]) <= 0:
                consecutive += 1
            else:
                break

        if consecutive >= self.max_consecutive:
            return BrakeResult(
                "ConsecutiveLoss", BrakeVerdict.BLOCK,
                f"{consecutive} consecutive losing trades (limit: {self.max_consecutive})",
                severity=2, value=float(consecutive), threshold=float(self.max_consecutive),
            )
        return BrakeResult(
            "ConsecutiveLoss", BrakeVerdict.ALLOW,
            f"{consecutive} consecutive losses (limit: {self.max_consecutive})", 0,
            value=float(consecutive), threshold=float(self.max_consecutive),
        )


class HighVolatilityBrake:
    """Reduce position size if recent price volatility is extreme.

    Compares the most recent price move (last N balance-history entries)
    against a configurable threshold.
    """

    def __init__(self, max_volatility_pct: float = 5.0) -> None:
        self.threshold = max_volatility_pct

    async def evaluate(self, paper_engine: Any | None, current_price: float = 0.0) -> BrakeResult:
        if not paper_engine:
            return BrakeResult("HighVolatility", BrakeVerdict.ALLOW, "No engine available", 0)

        conn = paper_engine._ledger._get_conn()
        async with conn.execute(
            "SELECT eth_price FROM paper_balance_history ORDER BY timestamp DESC LIMIT 12"
        ) as cur:
            rows = await cur.fetchall()

        prices = [float(r["eth_price"]) for r in rows if float(r["eth_price"]) > 0]
        if len(prices) < 3:
            return BrakeResult("HighVolatility", BrakeVerdict.ALLOW, "Insufficient price history", 0)

        import math
        returns = [(prices[i] - prices[i + 1]) / prices[i + 1] * 100 for i in range(len(prices) - 1)]
        volatility = math.sqrt(sum(r * r for r in returns) / len(returns))

        if volatility > self.threshold * 2:
            return BrakeResult(
                "HighVolatility", BrakeVerdict.BLOCK,
                f"Extreme volatility {volatility:.2f}% (block threshold: {self.threshold * 2:.2f}%)",
                severity=2, value=round(volatility, 2), threshold=self.threshold * 2,
            )
        if volatility > self.threshold:
            return BrakeResult(
                "HighVolatility", BrakeVerdict.REDUCE,
                f"Elevated volatility {volatility:.2f}% (reduce threshold: {self.threshold:.2f}%)",
                severity=1, value=round(volatility, 2), threshold=self.threshold,
            )
        return BrakeResult(
            "HighVolatility", BrakeVerdict.ALLOW,
            f"Volatility {volatility:.2f}% within limits", 0,
            value=round(volatility, 2), threshold=self.threshold,
        )


class HighSpreadBrake:
    """Skip if the estimated bid-ask spread is too wide."""

    def __init__(self, max_spread_bps: float = 30.0) -> None:
        self.threshold = max_spread_bps

    def evaluate(self, spread_bps: float = 0.0) -> BrakeResult:
        if spread_bps > self.threshold:
            return BrakeResult(
                "HighSpread", BrakeVerdict.BLOCK,
                f"Spread {spread_bps:.1f} bps exceeds limit {self.threshold:.1f} bps",
                severity=2, value=spread_bps, threshold=self.threshold,
            )
        return BrakeResult(
            "HighSpread", BrakeVerdict.ALLOW,
            f"Spread {spread_bps:.1f} bps within limit", 0,
            value=spread_bps, threshold=self.threshold,
        )


class LowLiquidityBrake:
    """Skip or reduce if estimated pool depth is too thin to absorb the trade."""

    def __init__(
        self,
        min_depth_usd: float = 500.0,
        reduce_depth_usd: float = 2000.0,
    ) -> None:
        self.min_depth_usd = min_depth_usd
        self.reduce_depth_usd = reduce_depth_usd

    def evaluate(self, pool_depth_usd: float = 0.0) -> BrakeResult:
        if pool_depth_usd <= 0.0:
            return BrakeResult(
                "LowLiquidity", BrakeVerdict.ALLOW,
                "Pool depth not provided — skipping liquidity check.", 0,
                value=0.0, threshold=self.min_depth_usd,
            )
        if pool_depth_usd < self.min_depth_usd:
            return BrakeResult(
                "LowLiquidity", BrakeVerdict.BLOCK,
                f"Pool depth ${pool_depth_usd:.0f} below minimum ${self.min_depth_usd:.0f}.", 2,
                value=round(pool_depth_usd, 2), threshold=self.min_depth_usd,
            )
        if pool_depth_usd < self.reduce_depth_usd:
            return BrakeResult(
                "LowLiquidity", BrakeVerdict.REDUCE,
                f"Pool depth ${pool_depth_usd:.0f} thin — reduce position size.", 1,
                value=round(pool_depth_usd, 2), threshold=self.reduce_depth_usd,
            )
        return BrakeResult(
            "LowLiquidity", BrakeVerdict.ALLOW,
            f"Pool depth ${pool_depth_usd:.0f} adequate.", 0,
            value=round(pool_depth_usd, 2), threshold=self.reduce_depth_usd,
        )


class ModelDisagreementBrake:
    """Skip if ML model ensemble confidence is below threshold or models disagree."""

    def __init__(self, min_confidence: float = 0.35, max_disagreement: float = 0.3) -> None:
        self.min_confidence = min_confidence
        self.max_disagreement = max_disagreement

    def evaluate(self, confidence: float = 0.0, model_disagreement: float = 0.0) -> BrakeResult:
        if confidence < self.min_confidence:
            return BrakeResult(
                "ModelDisagreement", BrakeVerdict.BLOCK,
                f"Confidence {confidence:.3f} below minimum {self.min_confidence:.3f}",
                severity=2, value=confidence, threshold=self.min_confidence,
            )
        if model_disagreement > self.max_disagreement:
            return BrakeResult(
                "ModelDisagreement", BrakeVerdict.REDUCE,
                f"Model disagreement {model_disagreement:.3f} above threshold {self.max_disagreement:.3f}",
                severity=1, value=model_disagreement, threshold=self.max_disagreement,
            )
        return BrakeResult(
            "ModelDisagreement", BrakeVerdict.ALLOW,
            f"Models agree (confidence={confidence:.3f})", 0,
            value=confidence, threshold=self.min_confidence,
        )


class ConfidenceDecayBrake:
    """Skip if recent average confidence is decaying vs historical baseline."""

    def __init__(self, decay_threshold: float = 0.15) -> None:
        self.decay_threshold = decay_threshold  # Fractional drop that triggers brake

    async def evaluate(self, paper_engine: Any | None) -> BrakeResult:
        if not paper_engine:
            return BrakeResult("ConfidenceDecay", BrakeVerdict.ALLOW, "No engine available", 0)

        conn = paper_engine._ledger._get_conn()
        async with conn.execute(
            "SELECT confidence FROM paper_decisions WHERE confidence > 0 ORDER BY timestamp DESC LIMIT 40"
        ) as cur:
            rows = await cur.fetchall()

        if len(rows) < 10:
            return BrakeResult("ConfidenceDecay", BrakeVerdict.ALLOW, "Insufficient decision history", 0)

        values = [float(r["confidence"]) for r in rows]
        recent_avg = sum(values[:5]) / 5
        baseline_avg = sum(values[10:]) / max(len(values[10:]), 1)

        if baseline_avg > 0:
            decay = (baseline_avg - recent_avg) / baseline_avg
        else:
            decay = 0.0

        if decay > self.decay_threshold:
            return BrakeResult(
                "ConfidenceDecay", BrakeVerdict.REDUCE,
                f"Confidence decaying {decay * 100:.1f}% (recent={recent_avg:.3f}, baseline={baseline_avg:.3f})",
                severity=1, value=round(decay, 4), threshold=self.decay_threshold,
            )
        return BrakeResult(
            "ConfidenceDecay", BrakeVerdict.ALLOW,
            f"Confidence stable (recent={recent_avg:.3f}, baseline={baseline_avg:.3f})", 0,
            value=round(recent_avg, 4), threshold=0.0,
        )


class SessionKillSwitch:
    """Manual emergency stop flag."""

    def __init__(self) -> None:
        self._active: bool = False
        self._reason: str = ""

    def activate(self, reason: str = "Manual emergency stop") -> None:
        self._active = True
        self._reason = reason
        log.warning(f"SessionKillSwitch ACTIVATED: {reason}")

    def deactivate(self) -> None:
        self._active = False
        self._reason = ""
        log.info("SessionKillSwitch deactivated.")

    def evaluate(self) -> BrakeResult:
        if self._active:
            return BrakeResult(
                "SessionKillSwitch", BrakeVerdict.BLOCK,
                f"Kill switch active: {self._reason}", severity=2,
            )
        return BrakeResult("SessionKillSwitch", BrakeVerdict.ALLOW, "Kill switch inactive", 0)


# ---------------------------------------------------------------------------
# Risk Committee — aggregator
# ---------------------------------------------------------------------------


class RiskCommittee:
    """Aggregates all brakes into a single committee verdict.

    Usage::

        committee = get_risk_committee()
        verdict = await committee.evaluate(
            paper_engine=engine,
            confidence=0.72,
            spread_bps=5.0,
        )
        if not verdict.allowed:
            log.info(f"Risk committee blocked trade: {verdict.blocking_brakes}")
    """

    def __init__(
        self,
        max_daily_loss_usd: float = 50.0,
        max_consecutive_losses: int = 4,
        max_volatility_pct: float = 5.0,
        max_spread_bps: float = 30.0,
        min_pool_depth_usd: float = 500.0,
        min_confidence: float = 0.35,
        max_model_disagreement: float = 0.3,
        confidence_decay_threshold: float = 0.15,
    ) -> None:
        self.daily_loss = DailyLossLimitBrake(max_daily_loss_usd)
        self.consec_loss = ConsecutiveLossBrake(max_consecutive_losses)
        self.volatility = HighVolatilityBrake(max_volatility_pct)
        self.spread = HighSpreadBrake(max_spread_bps)
        self.liquidity = LowLiquidityBrake(min_pool_depth_usd)
        self.model_disagreement = ModelDisagreementBrake(min_confidence, max_model_disagreement)
        self.confidence_decay = ConfidenceDecayBrake(confidence_decay_threshold)
        self.kill_switch = SessionKillSwitch()

    async def evaluate(
        self,
        paper_engine: Any | None = None,
        confidence: float = 0.0,
        spread_bps: float = 0.0,
        pool_depth_usd: float = 0.0,
        model_disagreement: float = 0.0,
        current_price: float = 0.0,
    ) -> CommitteeVerdict:
        """Evaluate all brakes and return a combined verdict."""
        results: list[BrakeResult] = []

        # Kill switch first (fastest check)
        results.append(self.kill_switch.evaluate())

        # Async DB-based checks
        results.append(await self.daily_loss.evaluate(paper_engine))
        results.append(await self.consec_loss.evaluate(paper_engine))
        results.append(await self.volatility.evaluate(paper_engine, current_price))
        results.append(await self.confidence_decay.evaluate(paper_engine))

        # Synchronous signal-based checks
        results.append(self.spread.evaluate(spread_bps))
        results.append(self.liquidity.evaluate(pool_depth_usd))
        results.append(self.model_disagreement.evaluate(confidence, model_disagreement))

        # Aggregate
        blocking = [r for r in results if r.is_blocking()]
        reducing = [r for r in results if r.verdict == BrakeVerdict.REDUCE]

        if blocking:
            overall = BrakeVerdict.BLOCK
            allowed = False
        elif reducing:
            overall = BrakeVerdict.REDUCE
            allowed = True
        else:
            overall = BrakeVerdict.ALLOW
            allowed = True

        # Risk score: count severity-weighted contributions
        score = min(100, sum(r.severity * 33 for r in results if r.severity > 0))

        narrative = self._build_narrative(blocking, reducing, score)

        return CommitteeVerdict(
            allowed=allowed,
            overall_verdict=overall,
            risk_score=score,
            brake_results=results,
            blocking_brakes=[r.brake_name for r in blocking],
            narrative=narrative,
        )

    def _build_narrative(
        self,
        blocking: list[BrakeResult],
        reducing: list[BrakeResult],
        score: int,
    ) -> str:
        if not blocking and not reducing:
            return f"All brakes clear. Risk score: {score}/100."
        parts: list[str] = []
        if blocking:
            parts.append(f"BLOCKED by: {', '.join(b.brake_name for b in blocking)}.")
            parts.extend(b.reason for b in blocking)
        if reducing:
            parts.append(f"SIZE REDUCTION advised by: {', '.join(r.brake_name for r in reducing)}.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_committee: RiskCommittee | None = None


def get_risk_committee(**kwargs: Any) -> RiskCommittee:
    """Return the module-level RiskCommittee singleton.

    If not provided in kwargs, defaults are pulled from centralized settings.
    """
    global _committee
    if _committee is None:
        from wallet_manager.config import get_settings
        cfg = get_settings()

        # Merge kwargs with settings-based defaults
        defaults = {
            "max_daily_loss_usd": _float_env("MAX_DAILY_LOSS_USD", 50.0), # fallback if not in Settings
            "max_consecutive_losses": _int_env("MAX_CONSECUTIVE_LOSS_LIMIT", 4),
            "min_confidence": cfg.ml_confidence_threshold,
            "max_model_disagreement": 0.3, # default
        }
        # Note: Some fields might need to be added to Settings dataclass if not there.
        # For now, we prioritize the blocking one: min_confidence.

        # Check if they passed specific overrides
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs[key] = val

        _committee = RiskCommittee(**kwargs)
    return _committee

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default

