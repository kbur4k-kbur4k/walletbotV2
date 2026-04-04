"""Unit tests for DynamicEnsembleWeighter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wallet_manager.strategies.ensemble_weighter import DynamicEnsembleWeighter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weighter(tmp_path: Path, weights: dict | None = None) -> DynamicEnsembleWeighter:
    """Create a weighter backed by a temporary JSON file."""
    json_path = tmp_path / "ensemble_weights.json"
    if weights is not None:
        json_path.write_text(json.dumps(weights), encoding="utf-8")
    return DynamicEnsembleWeighter(path=str(json_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_default_weights_sum_to_one(self):
        """Weights from DEFAULT_WEIGHTS must sum to exactly 1.0."""
        total = sum(DynamicEnsembleWeighter.DEFAULT_WEIGHTS.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9), f"DEFAULT_WEIGHTS sum is {total}"

    def test_fallback_when_json_absent(self, tmp_path: Path):
        """When the JSON file does not exist, get_weights() returns the defaults."""
        missing_path = tmp_path / "nonexistent.json"
        weighter = DynamicEnsembleWeighter(path=str(missing_path))
        assert weighter.get_weights() == DynamicEnsembleWeighter.DEFAULT_WEIGHTS


class TestUpdateWeights:
    def test_low_sharpe_model_excluded(self, tmp_path: Path):
        """A model with Sharpe < 0.3 must receive weight 0; the rest must sum to 1."""
        weighter = _make_weighter(tmp_path)
        sharpes = {"xgb": 0.8, "lgbm": 0.9, "lstm": 0.7, "transformer": 0.1}
        updated = weighter.update_weights(sharpes)

        assert updated["transformer"] == 0.0, "transformer Sharpe=0.1 should be zeroed"
        remaining = {k: v for k, v in updated.items() if v > 0}
        assert math.isclose(sum(remaining.values()), 1.0, abs_tol=1e-9)

    def test_all_models_low_sharpe_falls_back_to_defaults(self, tmp_path: Path):
        """When ALL models have Sharpe < 0.3, update_weights() returns defaults."""
        weighter = _make_weighter(tmp_path)
        sharpes = {"xgb": 0.0, "lgbm": 0.0, "lstm": 0.0, "transformer": 0.0}
        updated = weighter.update_weights(sharpes)
        assert updated == DynamicEnsembleWeighter.DEFAULT_WEIGHTS

    def test_floor_applied(self, tmp_path: Path):
        """Non-zeroed models after exclusion must each have weight >= 0.05."""
        weighter = _make_weighter(tmp_path)
        # Give one model a tiny default-weight proxy: make lgbm/lstm low Sharpe
        sharpes = {"xgb": 1.0, "lgbm": 0.1, "lstm": 0.1, "transformer": 0.8}
        updated = weighter.update_weights(sharpes)

        for model, weight in updated.items():
            if weight > 0:
                assert weight >= 0.05, f"{model} weight {weight} is below floor 0.05"

    def test_weights_sum_to_one_after_update(self, tmp_path: Path):
        """Weights must always sum to 1.0 after update_weights()."""
        weighter = _make_weighter(tmp_path)
        sharpes = {"xgb": 0.5, "lgbm": 0.4, "lstm": 0.2, "transformer": 0.6}
        updated = weighter.update_weights(sharpes)
        assert math.isclose(sum(updated.values()), 1.0, abs_tol=1e-9)


class TestPersist:
    def test_persist_round_trip(self, tmp_path: Path):
        """persist() writes weights; a new instance reading that file returns them."""
        weighter = _make_weighter(tmp_path)
        sharpes = {"xgb": 0.9, "lgbm": 0.8, "lstm": 0.7, "transformer": 0.6}
        original = weighter.update_weights(sharpes)

        persist_path = tmp_path / "saved_weights.json"
        weighter.persist(path=str(persist_path))

        loaded = DynamicEnsembleWeighter(path=str(persist_path))
        for key, val in original.items():
            assert math.isclose(loaded.get_weights()[key], val, abs_tol=1e-9), (
                f"Round-trip mismatch for {key}: {loaded.get_weights()[key]} != {val}"
            )
