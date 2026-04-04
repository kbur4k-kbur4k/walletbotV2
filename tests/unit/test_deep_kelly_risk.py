"""tests/unit/test_deep_kelly_risk.py — DeepKellyRiskEngine kapsamlı testler."""
from __future__ import annotations

import numpy as np
import pytest

from wallet_manager.strategies.deep_kelly_risk import (
    VAR_MAX_LOSS_PCT,
    DeepKellyRiskEngine,
    DrawdownController,
    PPOAgent,
    RiskDecision,
    TradeOutcome,
    VaRCalculator,
)


def _make_outcome(pnl_usd=10.0, pnl_pct=0.01, position_pct=0.1,
                  entry_alpha=0.5, entry_garch_sigma=0.001,
                  entry_vol_regime="normal", holding_periods=1):
    return TradeOutcome(
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        position_pct=position_pct,
        entry_alpha=entry_alpha,
        entry_garch_sigma=entry_garch_sigma,
        entry_vol_regime=entry_vol_regime,
        holding_periods=holding_periods,
    )


class TestTradeOutcome:
    def test_win_detection(self):
        outcome = _make_outcome(pnl_usd=50.0)
        assert outcome.is_win is True

    def test_loss_detection(self):
        outcome = _make_outcome(pnl_usd=-30.0, pnl_pct=-0.015)
        assert outcome.is_win is False

    def test_breakeven_not_win(self):
        outcome = _make_outcome(pnl_usd=0.0, pnl_pct=0.0)
        assert outcome.is_win is False

    def test_to_dict_roundtrip(self):
        original = _make_outcome(pnl_usd=75.0, pnl_pct=0.025, position_pct=0.15)
        d = original.to_dict()
        restored = TradeOutcome.from_dict(d)
        assert restored.pnl_usd == original.pnl_usd
        assert restored.position_pct == original.position_pct


class TestDrawdownController:
    def test_no_drawdown_state_normal(self):
        ctrl = DrawdownController()
        ctrl.update(1000.0)  # peak
        ctrl.update(980.0)   # 2% drawdown — normal
        assert ctrl.sizing_multiplier == pytest.approx(1.0)

    def test_reduce_at_5pct_drawdown(self):
        ctrl = DrawdownController()
        ctrl.update(1000.0)
        ctrl.update(940.0)  # 6% drawdown — reduce zone
        assert ctrl.sizing_multiplier < 1.0
        assert ctrl.sizing_multiplier > 0.0

    def test_halt_at_10pct_drawdown(self):
        ctrl = DrawdownController()
        ctrl.update(1000.0)
        ctrl.update(850.0)  # 15% drawdown — halt zone
        assert ctrl.sizing_multiplier == pytest.approx(0.0)

    def test_drawdown_pct_calculation(self):
        ctrl = DrawdownController()
        ctrl.update(1000.0)
        ctrl.update(900.0)
        assert ctrl.drawdown_pct == pytest.approx(0.10, abs=0.001)

    def test_recovery_restores_multiplier(self):
        ctrl = DrawdownController()
        ctrl.update(1000.0)
        ctrl.update(850.0)  # halt
        ctrl.update(1000.0)  # new peak
        assert ctrl.sizing_multiplier > 0.0


class TestVaRCalculator:
    def test_empty_returns_conservative_default(self):
        """When insufficient data, VaRCalculator returns VAR_MAX_LOSS_PCT default."""
        calc = VaRCalculator()
        assert calc.calculate() == pytest.approx(VAR_MAX_LOSS_PCT)

    def test_var_increases_with_volatility(self):
        calc_low = VaRCalculator()
        calc_high = VaRCalculator()
        np.random.seed(42)
        for r in np.random.normal(0, 0.01, 100):
            calc_low.add_return(float(r))
        for r in np.random.normal(0, 0.05, 100):
            calc_high.add_return(float(r))
        assert calc_high.calculate() > calc_low.calculate()

    def test_var_is_positive(self):
        """VaR her zaman pozitif döner (kayıp miktarı)."""
        calc = VaRCalculator()
        np.random.seed(0)
        for r in np.random.normal(-0.001, 0.02, 200):
            calc.add_return(float(r))
        assert calc.calculate() >= 0.0

    def test_max_position_pct_bounded(self):
        """max_position_pct her zaman [0, MAX_POSITION_PCT] aralığında."""
        calc = VaRCalculator()
        np.random.seed(1)
        for r in np.random.normal(0, 0.03, 100):
            calc.add_return(float(r))
        pct = calc.max_position_pct()
        assert 0.0 <= pct <= 0.25  # MAX_POSITION_PCT = 0.25


class TestDeepKellyRiskEngine:
    def _engine(self):
        return DeepKellyRiskEngine()

    def test_calculate_position_below_max(self):
        engine = self._engine()
        decision = engine.calculate_position(
            total_capital_usd=1000.0,
            alpha_score=0.7,
            garch_sigma=0.001,
            vol_regime="normal",
        )
        assert isinstance(decision, RiskDecision)
        assert decision.position_usd <= 1000.0 * 0.25  # MAX_POSITION_PCT

    def test_zero_alpha_minimal_position(self):
        """Alpha sıfırsa pozisyon açılmamalı veya çok küçük olmalı."""
        engine = self._engine()
        decision = engine.calculate_position(
            total_capital_usd=1000.0,
            alpha_score=0.0,
            garch_sigma=0.001,
            vol_regime="normal",
        )
        assert decision.position_usd == pytest.approx(0.0, abs=1.0)

    def test_negative_alpha_no_position(self):
        """Negatif alpha'da pozisyon açılmamalı."""
        engine = self._engine()
        decision = engine.calculate_position(
            total_capital_usd=1000.0,
            alpha_score=-0.5,
            garch_sigma=0.001,
            vol_regime="normal",
        )
        assert decision.position_usd == pytest.approx(0.0, abs=0.01)

    def test_record_trade_updates_history(self):
        engine = self._engine()
        for i in range(10):
            engine.record_trade(_make_outcome(
                pnl_usd=10.0 if i % 2 == 0 else -5.0,
                pnl_pct=0.01 if i % 2 == 0 else -0.005,
            ))
        assert 0.3 <= engine._win_rate() <= 0.7

    def test_drawdown_halt_zero_position(self):
        """%10 drawdown sonrası pozisyon sıfır olmalı."""
        engine = self._engine()
        engine.update_equity(1000.0)
        engine.update_equity(850.0)  # 15% drawdown — halt
        decision = engine.calculate_position(
            total_capital_usd=850.0,
            alpha_score=0.9,
            garch_sigma=0.001,
            vol_regime="normal",
        )
        assert decision.position_usd == pytest.approx(0.0, abs=0.01)

    def test_insufficient_history_uses_kelly_fallback(self):
        """50 trade'den az varsa PPO yerine Kelly kullanılır."""
        engine = self._engine()
        for _ in range(5):
            engine.record_trade(_make_outcome())
        decision = engine.calculate_position(
            total_capital_usd=1000.0,
            alpha_score=0.6,
            garch_sigma=0.001,
            vol_regime="normal",
        )
        assert decision is not None
