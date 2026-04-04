"""tests/integration/test_execution_mode.py — Execution mode integration tests."""
from __future__ import annotations

import inspect

import pytest

from wallet_manager.config import VALID_EXECUTION_MODES, Settings, get_settings
from wallet_manager.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(mode: str) -> Settings:
    return Settings(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode=mode,
        garch_alpha=0.10,
        garch_beta=0.85,
    )


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_all_six_modes_valid():
    modes = ["legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"]
    for mode in modes:
        cfg = _settings(mode)
        assert cfg.execution_mode == mode


def test_invalid_mode_raises_configuration_error():
    with pytest.raises(ConfigurationError) as exc_info:
        _settings("bad_mode")
    msg = str(exc_info.value)
    assert "bad_mode" in msg
    # Error message must list valid modes
    assert "ml_quant" in msg or "legacy" in msg


def test_legacy_mode_settings_are_accessible():
    cfg = _settings("legacy")
    # Must have all required fields
    assert hasattr(cfg, "loop_interval_seconds")
    assert hasattr(cfg, "algo_loop_interval_seconds")
    assert hasattr(cfg, "telegram_bot_token")


def test_ml_quant_mode_settings_are_accessible():
    cfg = _settings("ml_quant")
    assert hasattr(cfg, "xgboost_weight")
    assert hasattr(cfg, "lightgbm_weight")
    assert hasattr(cfg, "lstm_weight")
    assert hasattr(cfg, "transformer_weight")
    total = cfg.xgboost_weight + cfg.lightgbm_weight + cfg.lstm_weight + cfg.transformer_weight
    assert abs(total - 1.0) < 1e-6


def test_garch_stationarity_validated():
    with pytest.raises(ConfigurationError):
        Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="ml_quant",
            garch_alpha=0.5,
            garch_beta=0.6,  # 0.5 + 0.6 >= 1.0
        )


# ---------------------------------------------------------------------------
# New tests — parametrized over all six modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"])
def test_each_mode_produces_valid_settings_object(mode: str):
    """Every recognised execution mode must construct without raising."""
    cfg = _settings(mode)
    assert cfg.execution_mode == mode
    assert isinstance(cfg, Settings)


@pytest.mark.parametrize("mode", ["legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"])
def test_each_mode_has_non_empty_string(mode: str):
    """execution_mode must be a non-empty string for every valid mode."""
    cfg = _settings(mode)
    assert isinstance(cfg.execution_mode, str)
    assert cfg.execution_mode.strip() != ""


def test_valid_execution_modes_set_covers_all_six():
    """VALID_EXECUTION_MODES constant must contain all six documented modes."""
    expected = {"legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"}
    assert expected <= VALID_EXECUTION_MODES, (
        f"Missing from VALID_EXECUTION_MODES: {expected - VALID_EXECUTION_MODES}"
    )


@pytest.mark.parametrize("mode", ["legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"])
def test_each_mode_in_valid_execution_modes(mode: str):
    """Each of the six modes must appear in the VALID_EXECUTION_MODES frozenset."""
    assert mode in VALID_EXECUTION_MODES


# ---------------------------------------------------------------------------
# Task factory import tests — each must be callable without raising on import
# ---------------------------------------------------------------------------


def test_default_task_factory_is_callable():
    """_default_task_factory (legacy mode) must be importable and callable."""
    from wallet_manager.orchestrator import _default_task_factory
    assert callable(_default_task_factory)


def test_robust_paper_task_factory_is_callable():
    """_robust_paper_task_factory (robust_paper mode) must be importable and callable."""
    from wallet_manager.orchestrator import _robust_paper_task_factory
    assert callable(_robust_paper_task_factory)


def test_robust_task_factory_is_callable():
    """_robust_task_factory (robust/robust_live mode) must be importable and callable."""
    from wallet_manager.orchestrator import _robust_task_factory
    assert callable(_robust_task_factory)


def test_ml_quant_task_factory_is_callable():
    """_ml_quant_task_factory (ml_quant mode) must be importable and callable."""
    from wallet_manager.orchestrator import _ml_quant_task_factory
    assert callable(_ml_quant_task_factory)


def test_quantx_task_factory_is_callable():
    """_quantx_task_factory (quantx mode) must be importable and callable."""
    from wallet_manager.orchestrator import _quantx_task_factory
    assert callable(_quantx_task_factory)


def test_three_pillar_task_factory_is_callable():
    """_three_pillar_task_factory (router for non-legacy modes) must be callable."""
    from wallet_manager.orchestrator import _three_pillar_task_factory
    assert callable(_three_pillar_task_factory)


def test_airdrop_task_factory_is_callable():
    """_airdrop_task_factory (shared sub-factory) must be importable and callable."""
    from wallet_manager.orchestrator import _airdrop_task_factory
    assert callable(_airdrop_task_factory)


# ---------------------------------------------------------------------------
# Task factory signature tests — all factories must accept the same protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory_name", [
    "_default_task_factory",
    "_robust_paper_task_factory",
    "_robust_task_factory",
    "_ml_quant_task_factory",
    "_quantx_task_factory",
    "_three_pillar_task_factory",
    "_airdrop_task_factory",
])
def test_task_factory_is_coroutine_function(factory_name: str):
    """Every task factory must be an async (coroutine) function."""
    import wallet_manager.orchestrator as orch
    factory = getattr(orch, factory_name)
    assert inspect.iscoroutinefunction(factory), (
        f"{factory_name} must be an async def, not a plain function"
    )


@pytest.mark.parametrize("factory_name", [
    "_default_task_factory",
    "_robust_paper_task_factory",
    "_robust_task_factory",
    "_ml_quant_task_factory",
    "_quantx_task_factory",
    "_three_pillar_task_factory",
    "_airdrop_task_factory",
])
def test_task_factory_accepts_three_positional_args(factory_name: str):
    """Every task factory must accept (manager, wallet_id, task_context)."""
    import wallet_manager.orchestrator as orch
    factory = getattr(orch, factory_name)
    sig = inspect.signature(factory)
    params = list(sig.parameters.values())
    # Allow *args or exactly 3 named positional params
    positional = [
        p for p in params
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    assert len(positional) >= 3, (
        f"{factory_name} must accept at least 3 positional args "
        f"(manager, wallet_id, task_context); got {len(positional)}"
    )


# ---------------------------------------------------------------------------
# Settings field completeness per mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["robust", "robust_live", "robust_paper"])
def test_robust_modes_have_fracdiff_settings(mode: str):
    """Robust modes rely on fracdiff settings that must be present."""
    cfg = _settings(mode)
    assert hasattr(cfg, "fracdiff_d_min")
    assert hasattr(cfg, "fracdiff_d_max")
    assert hasattr(cfg, "fracdiff_d_step")
    assert cfg.fracdiff_d_min >= 0.0
    assert cfg.fracdiff_d_max <= 1.0
    assert cfg.fracdiff_d_min < cfg.fracdiff_d_max


@pytest.mark.parametrize("mode", ["ml_quant", "robust", "robust_live"])
def test_ml_modes_have_garch_settings(mode: str):
    """ML and robust modes must carry valid GARCH(1,1) parameters."""
    cfg = _settings(mode)
    assert cfg.garch_alpha >= 0.0
    assert cfg.garch_beta >= 0.0
    assert cfg.garch_alpha + cfg.garch_beta < 1.0, (
        "GARCH stationarity violated: alpha + beta must be < 1.0"
    )


def test_quantx_mode_has_engine_mode_field():
    """quantx mode must expose the quantx_engine_mode config field."""
    cfg = _settings("quantx")
    assert hasattr(cfg, "quantx_engine_mode")
    assert isinstance(cfg.quantx_engine_mode, str)


def test_all_modes_have_loop_interval_seconds():
    """loop_interval_seconds must be a positive float for every mode."""
    for mode in VALID_EXECUTION_MODES:
        cfg = _settings(mode)
        assert cfg.loop_interval_seconds > 0, (
            f"loop_interval_seconds must be > 0 for mode {mode!r}"
        )


def test_all_modes_have_positive_max_concurrent_wallets():
    """max_concurrent_wallets must be >= 1 for every mode."""
    for mode in VALID_EXECUTION_MODES:
        cfg = _settings(mode)
        assert cfg.max_concurrent_wallets >= 1
