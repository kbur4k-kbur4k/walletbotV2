"""tests/integration/test_paper_engine_simulation.py — PaperEngine E2E."""
from __future__ import annotations

import numpy as np
import pytest

from wallet_manager.strategies.paper_engine import PaperTradingEngine


class TestPaperEngineSimulation:
    def _prices(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.001, 0.02, n)
        return (2000.0 * np.cumprod(1 + returns)).tolist()

    @pytest.mark.asyncio
    async def test_100_step_simulation(self):
        engine = PaperTradingEngine(initial_capital_usd=1000.0)
        prices = self._prices(n=100)
        for i, price in enumerate(prices):
            signal = 0.6 if i % 3 == 0 else -0.1
            await engine.step(price=price, signal=signal)
        metrics = engine.get_metrics()
        assert "total_pnl_usd" in metrics
        assert "sharpe_ratio" in metrics
        assert "total_trades" in metrics

    @pytest.mark.asyncio
    async def test_no_trade_on_weak_signal(self):
        engine = PaperTradingEngine(initial_capital_usd=1000.0, min_confidence=0.60)
        prices = self._prices(n=50)
        for price in prices:
            await engine.step(price=price, signal=0.40)
        metrics = engine.get_metrics()
        assert metrics["total_trades"] == 0

    @pytest.mark.asyncio
    async def test_position_sizing_respects_kelly(self):
        engine = PaperTradingEngine(initial_capital_usd=1000.0)
        prices = self._prices(n=30)
        max_exposure = 0.0
        for price in prices:
            await engine.step(price=price, signal=0.90)
            exposure = engine.current_exposure_usd
            max_exposure = max(max_exposure, exposure)
        assert max_exposure <= 1000.0 * 0.25 + 1.0  # 1 USD tolerance
