from __future__ import annotations

from wallet_manager.strategies import model_registry as model_registry_module
from wallet_manager.strategies.model_registry import ChampionChallengerRegistry, get_model_registry


def test_model_recording_and_promotion_flow() -> None:
    registry = ChampionChallengerRegistry()
    champion = registry.register_champion("champion_v1")
    challenger = registry.register_challenger("challenger_v2")

    for _ in range(60):
        registry.record_cycle(champion.model_id, predicted_up=True, actual_up=False, pnl_usd=-0.1)
        registry.record_cycle(challenger.model_id, predicted_up=True, actual_up=True, pnl_usd=0.5)

    assert registry.should_promote() is True
    assert registry.promote_challenger() is True

    payload = registry.to_dict()
    assert payload["champion"]["model_id"] == challenger.model_id
    assert payload["challenger"] is None
    assert payload["promotion_history_count"] == 1


def test_record_shadow_cycle_is_noop_without_challenger() -> None:
    registry = ChampionChallengerRegistry()
    registry.register_champion("champion_v1")

    registry.record_shadow_cycle(predicted_up=True, actual_up=True, pnl_usd=1.0)

    assert registry.to_dict()["champion"]["total_cycles"] == 0
    assert registry.should_promote() is False


def test_get_model_registry_returns_initialized_singleton() -> None:
    model_registry_module._registry = None

    registry = get_model_registry()
    payload = registry.to_dict()

    assert payload["champion"]["model_id"] == "robust_alpha_v1"
    assert payload["champion"]["is_champion"] is True
