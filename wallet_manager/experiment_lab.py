"""
wallet_manager/experiment_lab.py

Phase 7: Experiment Lab
========================
Shadow strategies and threshold A/B testing framework.
New ideas compete in shadow mode before touching live capital.

Architecture:
- Experiment          : config for one variant
- ExperimentResult    : outcome snapshot for one cycle
- ExperimentLab       : runs registered variants in shadow mode and tracks results
- ThresholdVariant    : convenience wrapper for threshold-only experiments

Usage::

    lab = get_experiment_lab()

    # Register a variant with different alpha/confidence thresholds
    lab.register_threshold_variant(
        name="relaxed_confidence",
        alpha_threshold=0.05,
        confidence_threshold=0.30,
        description="Test whether relaxed confidence generates more trades with acceptable quality",
    )

    # In the shadow cycle, call:
    for variant in lab.active_variants():
        signal_override = variant.would_trade(robust_alpha=alpha, confidence=conf)
        lab.record_cycle(variant.name, action=signal_override, pnl=0.0, alpha=alpha, confidence=conf)

    # Weekly: call lab.get_variant_report() to compare variants
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThresholdVariant:
    """A single threshold configuration to test against the baseline."""

    name: str
    alpha_threshold: float          # Minimum |robust_alpha| to consider entry
    confidence_threshold: float     # Minimum ML confidence to enter
    adf_pvalue_threshold: float = 0.10
    description: str = ""
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def would_trade(self, robust_alpha: float, confidence: float, adf_pvalue: float = 0.0) -> bool:
        """Return True if this variant would enter a trade with the given signals."""
        alpha_ok = abs(robust_alpha) >= self.alpha_threshold
        conf_ok = confidence >= self.confidence_threshold
        adf_ok = adf_pvalue <= self.adf_pvalue_threshold if adf_pvalue > 0 else True
        return alpha_ok and conf_ok and adf_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha_threshold": self.alpha_threshold,
            "confidence_threshold": self.confidence_threshold,
            "adf_pvalue_threshold": self.adf_pvalue_threshold,
            "description": self.description,
            "active": self.active,
        }


@dataclass
class FeatureToggleVariant:
    """Tests the effect of enabling or disabling a named feature flag."""

    name: str
    enabled_features: frozenset[str]    # features that are ON in this variant
    disabled_features: frozenset[str]   # features that are OFF in this variant
    description: str = ""
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def feature_active(self, feature_name: str) -> bool:
        """Return True if this variant enables the given feature."""
        return feature_name not in self.disabled_features

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled_features": sorted(self.enabled_features),
            "disabled_features": sorted(self.disabled_features),
            "description": self.description,
            "active": self.active,
        }


@dataclass
class RegimeVariant:
    """Tests a strategy that only trades in specific market regimes."""

    name: str
    allowed_regimes: frozenset[str]     # only trade when regime is in this set
    alpha_threshold: float = 0.08
    confidence_threshold: float = 0.35
    description: str = ""
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def would_trade(self, robust_alpha: float, confidence: float, regime_label: str = "") -> bool:
        if self.allowed_regimes and regime_label not in self.allowed_regimes:
            return False
        return abs(robust_alpha) >= self.alpha_threshold and confidence >= self.confidence_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed_regimes": sorted(self.allowed_regimes),
            "alpha_threshold": self.alpha_threshold,
            "confidence_threshold": self.confidence_threshold,
            "description": self.description,
            "active": self.active,
        }


@dataclass
class ExperimentCycleRecord:
    """One decision cycle recorded for a variant."""

    timestamp: float
    variant_name: str
    would_trade: bool       # What the variant decided
    baseline_traded: bool   # What the live system actually did
    robust_alpha: float
    confidence: float
    pnl_usd: float = 0.0   # Only populated if the live system traded

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "variant_name": self.variant_name,
            "would_trade": self.would_trade,
            "baseline_traded": self.baseline_traded,
            "robust_alpha": self.robust_alpha,
            "confidence": self.confidence,
            "pnl_usd": self.pnl_usd,
        }


# ---------------------------------------------------------------------------
# Experiment Lab
# ---------------------------------------------------------------------------


class ExperimentLab:
    """Tracks multiple threshold variants in shadow mode.

    Each variant observes every signal cycle and records whether it
    would have traded vs what the live system did.  At the end of the
    period, `get_variant_report()` shows which variant had the best
    hypothetical outcomes.
    """

    def __init__(self) -> None:
        self._variants: dict[str, ThresholdVariant] = {}
        self._feature_toggles: dict[str, FeatureToggleVariant] = {}
        self._regime_variants: dict[str, RegimeVariant] = {}
        self._records: dict[str, list[ExperimentCycleRecord]] = {}
        self._max_records_per_variant: int = 5000

        # Register the baseline variant (mirrors current live thresholds)
        self.register_threshold_variant(
            name="baseline",
            alpha_threshold=0.08,
            confidence_threshold=0.35,
            adf_pvalue_threshold=0.10,
            description="Current production thresholds (baseline for comparison)",
        )

    def register_threshold_variant(
        self,
        name: str,
        alpha_threshold: float,
        confidence_threshold: float,
        adf_pvalue_threshold: float = 0.10,
        description: str = "",
    ) -> ThresholdVariant:
        """Register a new threshold variant for shadow testing."""
        variant = ThresholdVariant(
            name=name,
            alpha_threshold=alpha_threshold,
            confidence_threshold=confidence_threshold,
            adf_pvalue_threshold=adf_pvalue_threshold,
            description=description,
        )
        self._variants[name] = variant
        self._records[name] = []
        log.info(f"ExperimentLab: registered variant '{name}'")
        return variant

    def register_feature_toggle_variant(
        self,
        name: str,
        disabled_features: list[str],
        enabled_features: list[str] | None = None,
        description: str = "",
    ) -> FeatureToggleVariant:
        """Register a feature-toggle experiment variant."""
        variant = FeatureToggleVariant(
            name=name,
            enabled_features=frozenset(enabled_features or []),
            disabled_features=frozenset(disabled_features),
            description=description,
        )
        self._feature_toggles[name] = variant
        log.info(f"ExperimentLab: registered feature-toggle variant '{name}'")
        return variant

    def register_regime_variant(
        self,
        name: str,
        allowed_regimes: list[str],
        alpha_threshold: float = 0.08,
        confidence_threshold: float = 0.35,
        description: str = "",
    ) -> RegimeVariant:
        """Register a regime-specific strategy variant."""
        variant = RegimeVariant(
            name=name,
            allowed_regimes=frozenset(allowed_regimes),
            alpha_threshold=alpha_threshold,
            confidence_threshold=confidence_threshold,
            description=description,
        )
        self._regime_variants[name] = variant
        self._records[f"regime:{name}"] = []
        log.info(f"ExperimentLab: registered regime variant '{name}' for regimes {allowed_regimes}")
        return variant

    def record_regime_cycle(
        self,
        regime_label: str,
        robust_alpha: float,
        confidence: float,
        baseline_traded: bool,
        pnl_usd: float = 0.0,
    ) -> None:
        """Record a cycle against all active regime variants."""
        for variant in self._regime_variants.values():
            if not variant.active:
                continue
            key = f"regime:{variant.name}"
            wt = variant.would_trade(robust_alpha, confidence, regime_label)
            if key not in self._records:
                self._records[key] = []
            self._records[key].append(ExperimentCycleRecord(
                timestamp=time.time(),
                variant_name=key,
                would_trade=wt,
                baseline_traded=baseline_traded,
                robust_alpha=robust_alpha,
                confidence=confidence,
                pnl_usd=pnl_usd if baseline_traded else 0.0,
            ))
            if len(self._records[key]) > self._max_records_per_variant:
                self._records[key] = self._records[key][-self._max_records_per_variant:]

    def get_feature_toggle_report(self) -> list[dict[str, Any]]:
        """Return a summary of all registered feature-toggle variants."""
        return [v.to_dict() for v in self._feature_toggles.values()]

    def get_regime_variant_report(self) -> list[dict[str, Any]]:
        """Return a performance summary for each regime variant."""
        report: list[dict[str, Any]] = []
        for variant in self._regime_variants.values():
            key = f"regime:{variant.name}"
            records = self._records.get(key, [])
            total = len(records)
            if total == 0:
                report.append({**variant.to_dict(), "cycles": 0})
                continue
            would_trade = sum(1 for r in records if r.would_trade)
            agreement = sum(1 for r in records if r.would_trade == r.baseline_traded)
            shared_pnl = sum(r.pnl_usd for r in records if r.would_trade and r.baseline_traded)
            report.append({
                **variant.to_dict(),
                "cycles": total,
                "variant_would_trade": would_trade,
                "agreement_pct": round(agreement / total * 100, 1),
                "shared_realized_pnl": round(shared_pnl, 4),
            })
        return report

    def active_variants(self) -> list[ThresholdVariant]:
        """Return all active threshold variants."""
        return [v for v in self._variants.values() if v.active]

    def record_cycle(
        self,
        variant_name: str,
        would_trade: bool,
        baseline_traded: bool,
        robust_alpha: float,
        confidence: float,
        pnl_usd: float = 0.0,
    ) -> None:
        """Record one decision cycle for a variant."""
        if variant_name not in self._records:
            return
        records = self._records[variant_name]
        records.append(ExperimentCycleRecord(
            timestamp=time.time(),
            variant_name=variant_name,
            would_trade=would_trade,
            baseline_traded=baseline_traded,
            robust_alpha=robust_alpha,
            confidence=confidence,
            pnl_usd=pnl_usd,
        ))
        # Trim to max
        if len(records) > self._max_records_per_variant:
            self._records[variant_name] = records[-self._max_records_per_variant:]

    def record_all_variants(
        self,
        robust_alpha: float,
        confidence: float,
        adf_pvalue: float,
        baseline_traded: bool,
        pnl_usd: float = 0.0,
    ) -> None:
        """Convenience: record one cycle for ALL active variants at once."""
        for variant in self.active_variants():
            wt = variant.would_trade(robust_alpha, confidence, adf_pvalue)
            self.record_cycle(
                variant_name=variant.name,
                would_trade=wt,
                baseline_traded=baseline_traded,
                robust_alpha=robust_alpha,
                confidence=confidence,
                pnl_usd=pnl_usd if baseline_traded else 0.0,
            )

    def get_variant_report(self) -> list[dict[str, Any]]:
        """Return a summary report for all variants."""
        report: list[dict[str, Any]] = []
        for name, variant in self._variants.items():
            records = self._records.get(name, [])
            if not records:
                report.append({"name": name, "cycles": 0})
                continue

            total = len(records)
            would_trade_count = sum(1 for r in records if r.would_trade)
            baseline_trade_count = sum(1 for r in records if r.baseline_traded)
            agreement_count = sum(1 for r in records if r.would_trade == r.baseline_traded)

            # Trades that variant would have taken but baseline skipped
            variant_only = sum(1 for r in records if r.would_trade and not r.baseline_traded)
            # Trades baseline took but variant skipped
            baseline_only = sum(1 for r in records if not r.would_trade and r.baseline_traded)

            # For trades baseline took: compare to variant decision
            shared_pnl = sum(r.pnl_usd for r in records if r.baseline_traded and r.would_trade)
            baseline_pnl = sum(r.pnl_usd for r in records if r.baseline_traded)

            report.append({
                "name": name,
                "description": variant.description,
                "active": variant.active,
                "cycles": total,
                "variant_would_trade": would_trade_count,
                "baseline_traded": baseline_trade_count,
                "agreement_pct": round(agreement_count / total * 100, 1) if total else 0,
                "variant_only_signals": variant_only,
                "baseline_only_signals": baseline_only,
                "shared_realized_pnl": round(shared_pnl, 4),
                "baseline_total_pnl": round(baseline_pnl, 4),
                "thresholds": variant.to_dict(),
            })
        return report

    def compare_variants(self) -> dict[str, Any]:
        """Return a comparison highlighting the best-performing variant."""
        report = self.get_variant_report()
        if not report:
            return {"winner": None, "variants": []}

        # Simple ranking: prefer variant that agrees with baseline high-PnL trades
        ranked = sorted(
            [r for r in report if r.get("cycles", 0) >= 10],
            key=lambda r: r.get("shared_realized_pnl", 0.0),
            reverse=True,
        )
        winner = ranked[0]["name"] if ranked else None
        return {
            "winner": winner,
            "recommendation": (
                f"Variant '{winner}' shows best shared PnL. Consider adopting its thresholds."
                if winner and winner != "baseline"
                else "Baseline thresholds are performing best or insufficient data for comparison."
            ),
            "variants": report,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_variants": len(self._variants),
            "active_variants": len(self.active_variants()),
            "comparison": self.compare_variants(),
            "feature_toggles": self.get_feature_toggle_report(),
            "regime_variants": self.get_regime_variant_report(),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_lab: ExperimentLab | None = None


def get_experiment_lab() -> ExperimentLab:
    """Return the module-level ExperimentLab singleton."""
    global _lab
    if _lab is None:
        _lab = ExperimentLab()
        # Register default variants to compare against baseline
        _lab.register_threshold_variant(
            name="relaxed_confidence",
            alpha_threshold=0.08,
            confidence_threshold=0.28,
            description="Lower confidence floor — more trades but potentially lower quality",
        )
        _lab.register_threshold_variant(
            name="strict_confidence",
            alpha_threshold=0.10,
            confidence_threshold=0.45,
            description="Higher confidence floor — fewer trades but higher selectivity",
        )
        _lab.register_threshold_variant(
            name="wide_alpha",
            alpha_threshold=0.04,
            confidence_threshold=0.35,
            description="Lower alpha threshold — tests whether weaker signals still have edge",
        )
        # Regime-specific variants
        _lab.register_regime_variant(
            name="low_vol_only",
            allowed_regimes=["low_vol", "normal", "normal_bullish", "normal_bearish"],
            alpha_threshold=0.08,
            confidence_threshold=0.35,
            description="Only trade in low-volatility regimes — avoids high-vol chop",
        )
        _lab.register_regime_variant(
            name="trend_only",
            allowed_regimes=["uptrend", "downtrend", "strong_uptrend", "strong_downtrend"],
            alpha_threshold=0.06,
            confidence_threshold=0.32,
            description="Only trade in trending regimes — relaxed thresholds when trend is confirmed",
        )
        # Feature toggle variants
        _lab.register_feature_toggle_variant(
            name="no_adf_filter",
            disabled_features=["adf_stationarity_gate"],
            description="Test whether removing the ADF stationarity gate increases trade frequency with acceptable quality",
        )
        _lab.register_feature_toggle_variant(
            name="no_macro_blackout",
            disabled_features=["macro_calendar_blackout"],
            description="Test PnL impact of ignoring macro event blackouts",
        )
    return _lab
