"""Dynamic ensemble weight management for the ML alpha engine.

Weights are persisted to ``ml_models/ensemble_weights.json`` and can be
updated at runtime based on per-model rolling OOS Sharpe ratios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# Path is resolved relative to the project root (where pyproject.toml lives),
# two directories above this file: …/wallet_manager/ml_models/
_DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "ml_models" / "ensemble_weights.json"

_SHARPE_FLOOR = 0.3   # Models below this OOS Sharpe get zeroed
_WEIGHT_FLOOR = 0.05  # Minimum weight for non-zeroed models


class DynamicEnsembleWeighter:
    """Manages ensemble model weights with optional dynamic Sharpe-based updates.

    Weights are loaded from *path* at construction time; if the file is absent
    the hardcoded :attr:`DEFAULT_WEIGHTS` are used instead.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "xgb": 0.40,
        "lgbm": 0.25,
        "lstm": 0.25,
        "transformer": 0.10,
    }
    # FIX: load accepted-model weights from training_report.json at class creation
    @classmethod
    def _weights_from_report(cls) -> dict[str, float]:
        import json as _j
        from pathlib import Path as _P
        report_path = _P(__file__).resolve().parent.parent / "ml_models" / "training_report.json"
        try:
            report = _j.loads(report_path.read_text())
            models = report.get("models", {})
            _keys = ["xgboost", "lightgbm", "lstm", "transformer"]
            _alias = {"xgboost": "xgb", "lightgbm": "lgbm", "lstm": "lstm", "transformer": "transformer"}
            accepted = [_alias[k] for k in _keys if models.get(k, {}).get("accepted", False)]
            if not accepted:
                return {"transformer": 1.0}
            w = 1.0 / len(accepted)
            return {k: (w if k in accepted else 0.0) for k in ["xgb", "lgbm", "lstm", "transformer"]}
        except Exception:
            return {"transformer": 1.0}

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_WEIGHTS_PATH
        self._weights: dict[str, float] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """Return the current weight mapping (copy to prevent mutation)."""
        return dict(self._weights)

    def update_weights(self, sharpe_by_model: dict[str, float]) -> dict[str, float]:
        """Recompute weights based on per-model rolling OOS Sharpe values.

        Algorithm:
        1. Zero out models whose Sharpe < 0.3.
        2. Distribute weight proportionally (by default weight) among the rest.
        3. Apply a floor of 0.05 to every non-zeroed model.
        4. Normalise so weights sum to 1.0.
        5. If *all* models are zeroed fall back to :attr:`DEFAULT_WEIGHTS`.

        Args:
            sharpe_by_model: Mapping of model key → rolling OOS Sharpe ratio.

        Returns:
            Updated weight dict (also stored internally).
        """
        eligible = {
            k: v for k, v in self.DEFAULT_WEIGHTS.items()
            if sharpe_by_model.get(k, 0.0) >= _SHARPE_FLOOR
        }

        if not eligible:
            log.warning(
                "All models have OOS Sharpe below %.2f — falling back to defaults.",
                _SHARPE_FLOOR,
            )
            self._weights = dict(self.DEFAULT_WEIGHTS)
            return self.get_weights()

        # Proportional redistribution using default weights as the base
        base_sum = sum(eligible.values())
        new_weights: dict[str, float] = {}
        for key in self.DEFAULT_WEIGHTS:
            if key in eligible:
                new_weights[key] = eligible[key] / base_sum
            else:
                new_weights[key] = 0.0

        # Apply floor to non-zeroed models
        floored: dict[str, float] = {}
        for key, w in new_weights.items():
            floored[key] = max(w, _WEIGHT_FLOOR) if w > 0.0 else 0.0

        # Normalise
        total = sum(floored.values())
        self._weights = {k: v / total for k, v in floored.items()}

        log.info("Ensemble weights updated: %s", self._weights)
        return self.get_weights()

    def persist(self, path: Optional[str] = None) -> None:
        """Save current weights to *path* (defaults to the configured path).

        Creates parent directories if they don't exist.
        """
        target = Path(path) if path else self._path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self._weights, fh, indent=2)
        log.debug("Ensemble weights persisted to %s", target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, float]:
        """Load weights from JSON, falling back to defaults if file is absent."""
        if not self._path.exists():
            log.debug(
                "Ensemble weights file not found at %s — using defaults.", self._path
            )
            return dict(self.DEFAULT_WEIGHTS)
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data: dict[str, float] = json.load(fh)
            log.info("Loaded ensemble weights from %s: %s", self._path, data)
            return data
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Failed to load ensemble weights from %s (%s) — using defaults.",
                self._path,
                exc,
            )
            return dict(self.DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[DynamicEnsembleWeighter] = None


def get_singleton() -> DynamicEnsembleWeighter:
    """Return (and lazily create) the module-level :class:`DynamicEnsembleWeighter`."""
    global _singleton  # noqa: PLW0603
    if _singleton is None:
        _singleton = DynamicEnsembleWeighter()
    return _singleton
