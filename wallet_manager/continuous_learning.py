"""
wallet_manager/continuous_learning.py

Phase 10: Continuous Learning
==============================
Makes the system improve safely over time based on evidence.

Components:
- DriftReviewer         : monitors concept drift level trends
- SetupDecayTracker     : tracks which setups are degrading over time
- ThresholdRetuner      : generates evidence-based threshold adjustment suggestions
- PerformanceChangelog  : logs meaningful strategy changes with timestamps
- ContinuousLearner     : orchestrates all components

Design:
- No autonomous parameter changes — all suggestions require human approval
- Changelog documents what was tried and what happened
- Drift detection wraps existing ConceptDriftDetector
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_CHANGELOG_PATH = Path("data/strategy_changelog.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DriftSnapshot:
    timestamp: float
    drift_level: str       # NOMINAL / MILD / MODERATE / SEVERE
    drift_score: float
    features_drifted: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_level": self.drift_level,
            "drift_score": self.drift_score,
            "features_drifted": self.features_drifted,
        }


@dataclass
class SetupDecayRecord:
    setup_tag: str
    window_early_pnl: float     # PnL in the first half of history
    window_recent_pnl: float    # PnL in the second half of history
    early_win_rate: float
    recent_win_rate: float
    decay_detected: bool
    sample_count: int

    @property
    def pnl_change(self) -> float:
        return self.window_recent_pnl - self.window_early_pnl

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_tag": self.setup_tag,
            "window_early_pnl": round(self.window_early_pnl, 4),
            "window_recent_pnl": round(self.window_recent_pnl, 4),
            "pnl_change": round(self.pnl_change, 4),
            "early_win_rate": round(self.early_win_rate, 3),
            "recent_win_rate": round(self.recent_win_rate, 3),
            "decay_detected": self.decay_detected,
            "sample_count": self.sample_count,
        }


@dataclass
class ChangelogEntry:
    timestamp: float
    category: str               # threshold_change | setup_retired | drift_response | experiment
    description: str
    before: dict[str, Any]
    after: dict[str, Any]
    outcome: str = ""           # Filled in later after observing effect

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self.timestamp)),
            "category": self.category,
            "description": self.description,
            "before": self.before,
            "after": self.after,
            "outcome": self.outcome,
        }


# ---------------------------------------------------------------------------
# Drift Reviewer
# ---------------------------------------------------------------------------


class DriftReviewer:
    """Monitors concept drift level trends over time."""

    def __init__(self) -> None:
        self._history: list[DriftSnapshot] = []
        self._max_history = 100

    def record(self, drift_level: str, drift_score: float = 0.0, features: list[str] | None = None) -> None:
        snap = DriftSnapshot(
            timestamp=time.time(),
            drift_level=drift_level,
            drift_score=drift_score,
            features_drifted=features or [],
        )
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def trend_summary(self) -> dict[str, Any]:
        if not self._history:
            return {"status": "no_data", "recent_level": "UNKNOWN", "trend": "stable"}

        recent = self._history[-10:]
        levels = {"NOMINAL": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
        recent_scores = [levels.get(s.drift_level, 0) for s in recent]
        avg_recent = sum(recent_scores) / len(recent_scores)
        current = self._history[-1].drift_level

        # Trend: compare first half vs second half of recent window
        if len(recent_scores) >= 4:
            mid = len(recent_scores) // 2
            early_avg = sum(recent_scores[:mid]) / mid
            late_avg = sum(recent_scores[mid:]) / (len(recent_scores) - mid)
            trend = "increasing" if late_avg > early_avg + 0.5 else "decreasing" if late_avg < early_avg - 0.5 else "stable"
        else:
            trend = "stable"

        severe_count = sum(1 for s in recent if s.drift_level == "SEVERE")
        return {
            "status": "ok",
            "recent_level": current,
            "avg_score_10": round(avg_recent, 2),
            "trend": trend,
            "severe_count_last_10": severe_count,
            "alert": severe_count >= 3 or (trend == "increasing" and avg_recent >= 2.0),
        }


# ---------------------------------------------------------------------------
# Setup Decay Tracker
# ---------------------------------------------------------------------------


class SetupDecayTracker:
    """Splits setup history into early/recent windows and detects decay."""

    async def analyse(self, paper_engine: Any | None) -> list[SetupDecayRecord]:
        if not paper_engine:
            return []

        conn = paper_engine._ledger._get_conn()
        async with conn.execute(
            "SELECT setup_tag, MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts "
            "FROM paper_decisions WHERE setup_tag IS NOT NULL "
            "GROUP BY setup_tag HAVING COUNT(*) >= 6"
        ) as cur:
            setup_rows = await cur.fetchall()

        results: list[SetupDecayRecord] = []
        for row in setup_rows:
            tag = row["setup_tag"]
            min_ts = float(row["min_ts"])
            max_ts = float(row["max_ts"])
            mid_ts = (min_ts + max_ts) / 2.0

            async with conn.execute(
                "SELECT d.action, t.pnl_usd, d.timestamp "
                "FROM paper_decisions d "
                "LEFT JOIN paper_trades t ON (ABS(d.timestamp - t.timestamp) < 5.0 AND t.direction = d.action) "
                "WHERE d.setup_tag = ? AND d.action IN ('BUY','SELL')",
                (tag,),
            ) as cur2:
                trade_rows = await cur2.fetchall()

            early_pnls = [float(r["pnl_usd"] or 0.0) for r in trade_rows if float(r["timestamp"]) < mid_ts]
            recent_pnls = [float(r["pnl_usd"] or 0.0) for r in trade_rows if float(r["timestamp"]) >= mid_ts]

            early_wins = sum(1 for p in early_pnls if p > 0)
            recent_wins = sum(1 for p in recent_pnls if p > 0)
            early_wr = early_wins / len(early_pnls) if early_pnls else 0.0
            recent_wr = recent_wins / len(recent_pnls) if recent_pnls else 0.0

            early_pnl_sum = sum(early_pnls)
            recent_pnl_sum = sum(recent_pnls)
            decay = (recent_pnl_sum < early_pnl_sum - 0.01) and (recent_wr < early_wr - 0.1)

            results.append(SetupDecayRecord(
                setup_tag=tag,
                window_early_pnl=early_pnl_sum,
                window_recent_pnl=recent_pnl_sum,
                early_win_rate=early_wr,
                recent_win_rate=recent_wr,
                decay_detected=decay,
                sample_count=len(trade_rows),
            ))

        return sorted(results, key=lambda r: r.pnl_change)


# ---------------------------------------------------------------------------
# Threshold Retuner
# ---------------------------------------------------------------------------


class ThresholdRetuner:
    """Generates evidence-based threshold adjustment suggestions."""

    async def suggest(self, paper_engine: Any | None) -> list[dict[str, Any]]:
        """Return a list of threshold change suggestions with evidence."""
        if not paper_engine:
            return []

        from wallet_manager.research_brain import ResearchBrain
        brain = ResearchBrain(paper_engine=paper_engine)
        report = await brain.threshold_tuning_suggestions()
        suggestions = []
        for rec in report.recommendations:
            suggestions.append({
                "suggestion": rec,
                "evidence": "Based on paper decision analytics",
                "confidence": "low" if not paper_engine._ledger._closed_trade_count or
                              paper_engine._ledger._closed_trade_count < 20 else "medium",
                "generated_at": time.time(),
            })
        return suggestions


# ---------------------------------------------------------------------------
# Performance Changelog
# ---------------------------------------------------------------------------


class PerformanceChangelog:
    """Persistent log of strategy changes and their outcomes."""

    def __init__(self, path: Path = _CHANGELOG_PATH) -> None:
        self._path = path
        self._entries: list[ChangelogEntry] = []
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                for item in data:
                    self._entries.append(ChangelogEntry(**item))
        except Exception:
            self._entries = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps([e.to_dict() for e in self._entries], indent=2))
        except Exception as exc:
            log.warning(f"PerformanceChangelog save failed: {exc}")

    def record(
        self,
        category: str,
        description: str,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> ChangelogEntry:
        entry = ChangelogEntry(
            timestamp=time.time(),
            category=category,
            description=description,
            before=before,
            after=after,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def update_outcome(self, timestamp: float, outcome: str) -> bool:
        for entry in self._entries:
            if abs(entry.timestamp - timestamp) < 1.0:
                entry.outcome = outcome
                self._save()
                return True
        return False

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return [e.to_dict() for e in sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]]


# ---------------------------------------------------------------------------
# JS-7.4: Feature Importance Drift Monitor
# ---------------------------------------------------------------------------


class FeatureImportanceDrift:
    """
    JS-7.4: Tracks weekly feature importance rankings from XGBoost/LightGBM.
    If a feature drops from top-3 to bottom-3 over 4 consecutive weeks:
      => Log "Feature decay detected: {feature}"
      => Flag for ExperimentLab feature_toggle review.

    Importance snapshots are weekly dicts: {feature_name: importance_score}.
    """

    _CONSECUTIVE_DROPS_THRESHOLD = 4
    _MAX_HISTORY = 52  # 52 weeks

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []   # list of {ts, rankings: {name: rank}}
        self._alerts: list[dict[str, Any]] = []

    def record_importance_snapshot(self, importances: dict[str, float]) -> None:
        """
        Record a new importance snapshot. Keys are feature names, values are scores.
        Automatically computes rankings and checks for decay patterns.
        """
        if not importances:
            return

        # Rank features: rank 1 = most important
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        rankings = {name: rank + 1 for rank, (name, _) in enumerate(sorted_features)}
        n_features = len(rankings)

        self._history.append({
            "ts": time.time(),
            "rankings": rankings,
            "n_features": n_features,
        })
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

        self._check_decay(rankings, n_features)

    def _check_decay(self, current_rankings: dict[str, int], n: int) -> None:
        """Check if any feature has been in bottom-3 for 4+ consecutive snapshots."""
        if len(self._history) < self._CONSECUTIVE_DROPS_THRESHOLD:
            return

        recent = self._history[-self._CONSECUTIVE_DROPS_THRESHOLD:]
        top_cutoff = min(3, n)
        for feature in current_rankings:
            was_top = False
            consecutive_bottom = 0

            for snap in recent:
                rank = snap["rankings"].get(feature, n + 1)
                n_snap = snap["n_features"]
                bottom_threshold = max(n_snap - 2, top_cutoff + 1)
                if rank <= top_cutoff:
                    was_top = True
                if rank >= bottom_threshold:
                    consecutive_bottom += 1

            if was_top and consecutive_bottom >= self._CONSECUTIVE_DROPS_THRESHOLD:
                # Only alert once per feature (deduplicate)
                existing = any(a["feature"] == feature for a in self._alerts[-20:])
                if not existing:
                    alert = {
                        "feature": feature,
                        "current_rank": current_rankings[feature],
                        "n_features": n,
                        "consecutive_bottom_snapshots": consecutive_bottom,
                        "detected_at": time.time(),
                        "recommendation": f"Test disabling '{feature}' via ExperimentLab feature toggle",
                    }
                    self._alerts.append(alert)
                    log.warning(
                        f"FeatureImportanceDrift: decay detected for '{feature}' — "
                        f"was top-3, now rank {current_rankings[feature]}/{n} for "
                        f"{consecutive_bottom} consecutive snapshots"
                    )

    def get_recent_alerts(self, n: int = 10) -> list[dict[str, Any]]:
        return sorted(self._alerts, key=lambda a: a["detected_at"], reverse=True)[:n]

    def to_dict(self) -> dict[str, Any]:
        latest = self._history[-1] if self._history else {}
        return {
            "snapshots_recorded": len(self._history),
            "latest_rankings": latest.get("rankings", {}),
            "decay_alerts": self.get_recent_alerts(5),
            "total_alerts": len(self._alerts),
        }


# ---------------------------------------------------------------------------
# Continuous Learner
# ---------------------------------------------------------------------------


class ContinuousLearner:
    """Orchestrates drift review, setup decay, threshold tuning, changelog, and feature drift."""

    def __init__(self) -> None:
        self.drift_reviewer = DriftReviewer()
        self.setup_decay = SetupDecayTracker()
        self.retuner = ThresholdRetuner()
        self.changelog = PerformanceChangelog()
        self.feature_drift = FeatureImportanceDrift()


    async def generate_learning_report(self, paper_engine: Any | None = None) -> dict[str, Any]:
        """Produce a full continuous-learning status report."""
        drift_summary = self.drift_reviewer.trend_summary()
        decay_records = await self.setup_decay.analyse(paper_engine)
        tuning_suggestions = await self.retuner.suggest(paper_engine)
        recent_changes = self.changelog.recent(5)

        decaying_setups = [r.to_dict() for r in decay_records if r.decay_detected]
        stable_setups = [r.to_dict() for r in decay_records if not r.decay_detected]

        alerts: list[str] = []
        if drift_summary.get("alert"):
            alerts.append(f"Drift alert: {drift_summary['recent_level']} trend={drift_summary['trend']}")
        for s in decaying_setups:
            alerts.append(f"Setup decay: {s['setup_tag']} — PnL change: ${s['pnl_change']:.4f}")
        for fd_alert in self.feature_drift.get_recent_alerts(3):
            alerts.append(f"Feature decay: {fd_alert['feature']} — {fd_alert['recommendation']}")

        return {
            "generated_at": time.time(),
            "drift": drift_summary,
            "setup_decay": {
                "decaying": decaying_setups,
                "stable": stable_setups,
                "total_analysed": len(decay_records),
            },
            "threshold_suggestions": tuning_suggestions,
            "changelog_recent": recent_changes,
            "feature_drift": self.feature_drift.to_dict(),
            "alerts": alerts,
        }



# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_learner: ContinuousLearner | None = None


def get_continuous_learner() -> ContinuousLearner:
    global _learner
    if _learner is None:
        _learner = ContinuousLearner()
    return _learner
