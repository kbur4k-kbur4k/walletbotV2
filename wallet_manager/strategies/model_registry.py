"""
wallet_manager/strategies/model_registry.py

Sprint JS-7.3: Champion/Challenger Model Framework
====================================================
Prevents regressing to worse models during retraining.
New model runs as "challenger" in shadow mode; only swaps champion
when challenger Sharpe > champion Sharpe * 1.10 over same cycles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_MIN_CHALLENGER_CYCLES = 50
_IMPROVEMENT_THRESHOLD = 1.10   # challenger must beat champion by 10%


@dataclass
class ModelRecord:
    model_id: str
    registered_at: float = field(default_factory=time.time)
    total_cycles: int = 0
    correct_direction: int = 0
    total_pnl: float = 0.0
    is_champion: bool = False
    promoted_at: float | None = None
    notes: str = ""

    @property
    def win_rate(self) -> float:
        return self.correct_direction / self.total_cycles if self.total_cycles > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_cycles if self.total_cycles > 0 else 0.0

    def sharpe_proxy(self) -> float:
        """Simple Sharpe proxy: win_rate * avg_pnl (higher is better)."""
        return self.win_rate * max(self.avg_pnl, 0.0) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "is_champion": self.is_champion,
            "total_cycles": self.total_cycles,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "avg_pnl_usd": round(self.avg_pnl, 4),
            "total_pnl_usd": round(self.total_pnl, 4),
            "sharpe_proxy": round(self.sharpe_proxy(), 6),
            "promoted_at": self.promoted_at,
            "notes": self.notes,
        }


class ChampionChallengerRegistry:
    """
    Manages champion and challenger model records.
    Champion is the current production model.
    Challenger is the candidate replacement in shadow mode.

    Usage:
        registry = get_model_registry()
        registry.register_challenger("model_v2")
        registry.record_cycle("model_v2", predicted_up=True, actual_up=True, pnl=0.05)
        if registry.should_promote():
            registry.promote_challenger()
    """

    def __init__(self) -> None:
        self._champion: ModelRecord | None = None
        self._challenger: ModelRecord | None = None
        self._history: list[dict[str, Any]] = []

    def register_champion(self, model_id: str, notes: str = "") -> ModelRecord:
        rec = ModelRecord(model_id=model_id, is_champion=True, notes=notes)
        rec.promoted_at = time.time()
        self._champion = rec
        log.info(f"ModelRegistry: champion registered: '{model_id}'")
        return rec

    def register_challenger(self, model_id: str, notes: str = "") -> ModelRecord:
        rec = ModelRecord(model_id=model_id, is_champion=False, notes=notes)
        self._challenger = rec
        log.info(f"ModelRegistry: challenger registered: '{model_id}'")
        return rec

    def record_cycle(
        self,
        model_id: str,
        predicted_up: bool,
        actual_up: bool,
        pnl_usd: float,
    ) -> None:
        """Record one prediction cycle for a specific model."""
        target = None
        if self._champion and self._champion.model_id == model_id:
            target = self._champion
        elif self._challenger and self._challenger.model_id == model_id:
            target = self._challenger
        if target is None:
            return
        target.total_cycles += 1
        target.total_pnl += pnl_usd
        if predicted_up == actual_up:
            target.correct_direction += 1

    def should_promote(self) -> bool:
        """Return True if challenger is significantly better than champion."""
        if not self._champion or not self._challenger:
            return False
        if self._challenger.total_cycles < _MIN_CHALLENGER_CYCLES:
            return False
        champ_sharpe = self._champion.sharpe_proxy()
        chal_sharpe = self._challenger.sharpe_proxy()
        return chal_sharpe > champ_sharpe * _IMPROVEMENT_THRESHOLD

    def record_shadow_cycle(
        self,
        predicted_up: bool,
        actual_up: bool,
        pnl_usd: float,
    ) -> None:
        """Record a 2-week shadow cycle for the current challenger (if registered)."""
        if not self._challenger:
            return
        self.record_cycle(self._challenger.model_id, predicted_up, actual_up, pnl_usd)

    def promote_challenger(self) -> bool:
        """Promote challenger to champion if conditions met."""
        if not self.should_promote():
            return False
        old_champ = self._champion
        challenger = self._challenger
        if challenger is None:
            return False
        challenger.is_champion = True
        challenger.promoted_at = time.time()
        if old_champ:
            self._history.append({
                "demoted": old_champ.to_dict(),
                "promoted": challenger.to_dict(),
                "timestamp": time.time(),
            })
            log.info(f"ModelRegistry: '{challenger.model_id}' promoted over '{old_champ.model_id}'")
        self._champion = challenger
        self._challenger = None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": self._champion.to_dict() if self._champion else None,
            "challenger": self._challenger.to_dict() if self._challenger else None,
            "should_promote": self.should_promote(),
            "promotion_history_count": len(self._history),
            "recent_promotions": self._history[-3:],
        }


_registry: ChampionChallengerRegistry | None = None


def get_model_registry() -> ChampionChallengerRegistry:
    global _registry
    if _registry is None:
        _registry = ChampionChallengerRegistry()
        _registry.register_champion("robust_alpha_v1", notes="Initial production model")
    return _registry
