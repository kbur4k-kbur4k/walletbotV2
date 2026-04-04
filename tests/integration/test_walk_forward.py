"""tests/integration/test_walk_forward.py — WalkForwardValidator E2E testleri."""
from __future__ import annotations

import numpy as np
import pytest

from wallet_manager.strategies.wf_validator import WalkForwardValidator


class TestWalkForwardValidatorE2E:
    def _synthetic_prices(self, n=600, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0005, 0.02, n)
        return (2000.0 * np.cumprod(1 + returns)).tolist()

    def test_get_splits_returns_nonempty(self):
        wfv = WalkForwardValidator(train_size=200, test_size=50, step=50)
        prices = self._synthetic_prices()
        splits = wfv.get_walk_forward_splits(prices)
        assert len(splits) > 0

    @pytest.mark.asyncio
    async def test_validate_returns_result(self, tmp_path):
        wfv = WalkForwardValidator(
            train_size=150, test_size=50, step=50,
            db_path=str(tmp_path / "wfv.db"),
        )
        await wfv.initialize()
        prices = self._synthetic_prices(n=500)
        signals = [0.5] * len(prices)
        result = wfv.validate(prices=prices, signals=signals)
        assert result is not None
        assert hasattr(result, "oos_sharpe")
        await wfv.shutdown()

    def test_oos_sharpe_reasonable_for_noisy_signal(self):
        wfv = WalkForwardValidator(train_size=100, test_size=30, step=30)
        prices = self._synthetic_prices(n=400)
        rng = np.random.default_rng(99)
        signals = rng.standard_normal(len(prices)).tolist()
        result = wfv.validate(prices=prices, signals=signals)
        assert result.oos_sharpe < 1.0
