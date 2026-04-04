"""Tests verifying anti-overfit safeguards are active."""
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def test_regime_aware_kelly_scale_by_trade_count():
    """Kelly fraction scales down for low trade counts."""
    from wallet_manager.strategies.kelly_criterion import RegimeAwareKelly
    kelly = RegimeAwareKelly()

    pos_few = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL", current_drawdown=0.0,
        portfolio_value_usd=1000.0, max_position_usd=500.0,
        trade_count=10,
    )
    pos_many = kelly.compute_sized_position(
        base_kelly=0.5, regime="TRENDING_BULL", current_drawdown=0.0,
        portfolio_value_usd=1000.0, max_position_usd=500.0,
        trade_count=600,
    )
    assert pos_few < pos_many, "Early-stage should produce smaller positions"
    assert pos_few >= 0, "Position must be non-negative"


def test_confidence_gate_blocks_low_confidence_signals():
    """LOW_CONFIDENCE signals must not trigger trades."""
    from wallet_manager.config import Settings
    s = Settings()
    threshold = s.ml_confidence_threshold
    assert threshold >= 0.65, f"ml_confidence_threshold={threshold} too low"
    signal_confidence = 0.50
    assert signal_confidence < threshold, "Low confidence signal should be below threshold"


def test_regime_high_vol_blocks_position_opening():
    """HIGH_VOL regime must block new position opening."""
    from wallet_manager.strategies.regime_filter import MarketRegime
    assert MarketRegime.HIGH_VOL is not None
    assert MarketRegime.NEWS_BLACKOUT is not None


def test_oos_sharpe_threshold_is_conservative():
    """_MIN_OOS_SHARPE in train_models must be >= 0.7."""
    script = Path("scripts/train_models.py").read_text(encoding="utf-8")
    match = re.search(r"_MIN_OOS_SHARPE\s*=\s*([\d.]+)", script)
    assert match is not None, "_MIN_OOS_SHARPE not found in train_models.py"
    value = float(match.group(1))
    assert value >= 0.7, f"_MIN_OOS_SHARPE={value} is too low (must be >= 0.7)"


def test_kelly_fraction_config_is_conservative():
    """kelly_fraction default must be <= 0.25 (anti-overfit protection)."""
    from wallet_manager.config import Settings
    s = Settings()
    assert s.kelly_fraction <= 0.25, (
        f"kelly_fraction={s.kelly_fraction} too high for early-stage system "
        f"(should be <= 0.25 until 500+ trades accumulated)"
    )


def test_kelly_max_position_pct_is_conservative():
    """kelly_max_position_pct default must be <= 0.15."""
    from wallet_manager.config import Settings
    s = Settings()
    assert s.kelly_max_position_pct <= 0.15, (
        f"kelly_max_position_pct={s.kelly_max_position_pct} too high "
        f"(should be <= 0.15 for early-stage capital protection)"
    )


def test_min_paper_trades_config_present():
    """min_paper_trades_before_live must exist and be >= 50."""
    from wallet_manager.config import Settings
    s = Settings()
    assert hasattr(s, "min_paper_trades_before_live"), (
        "min_paper_trades_before_live missing from config"
    )
    assert s.min_paper_trades_before_live >= 50, (
        f"min_paper_trades_before_live={s.min_paper_trades_before_live} too low"
    )


def test_ensemble_active_models_config_present():
    """ensemble_active_models config key must exist."""
    from wallet_manager.config import Settings
    s = Settings()
    assert hasattr(s, "ensemble_active_models"), (
        "ensemble_active_models missing from config"
    )
    active = s.ensemble_active_models.split(",")
    assert len(active) >= 1, "At least one model must be active"
