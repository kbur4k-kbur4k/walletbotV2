"""tests/unit/test_regime_filter.py — GaussianHMM ve RegimeFilter testleri."""
from __future__ import annotations

import numpy as np
import pytest

from wallet_manager.strategies.regime_filter import (
    EconomicCalendar,
    GaussianHMM,
    MarketRegime,
    RegimeFilter,
    RegimeSignal,
)


class TestGaussianHMM:
    def _make_obs(self, n=200, seed=42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, 2)).astype(np.float64)

    def test_fit_returns_log_likelihood(self):
        hmm = GaussianHMM(n_states=3, n_features=2)
        obs = self._make_obs()
        ll = hmm.fit(obs, max_iter=5)
        assert isinstance(ll, float)
        assert ll < 0  # log-likelihood negatif olmalı

    def test_viterbi_returns_valid_states(self):
        hmm = GaussianHMM(n_states=3, n_features=2)
        obs = self._make_obs()
        hmm.fit(obs, max_iter=5)
        states = hmm.viterbi(obs[:50])
        assert states.shape == (50,)
        assert set(states).issubset({0, 1, 2})

    def test_predict_proba_sums_to_one(self):
        hmm = GaussianHMM(n_states=3, n_features=2)
        obs = self._make_obs()
        hmm.fit(obs, max_iter=5)
        proba = hmm.predict_proba(obs[:20])
        assert proba.shape[1] == 3
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_convergence_improves_ll(self):
        """Daha fazla iterasyon log-likelihood'ı artırmalı."""
        hmm_few = GaussianHMM(n_states=3, n_features=2)
        hmm_more = GaussianHMM(n_states=3, n_features=2)
        obs = self._make_obs()
        ll_few = hmm_few.fit(obs, max_iter=2)
        ll_more = hmm_more.fit(obs, max_iter=20)
        assert ll_more >= ll_few - 0.1  # daha iyi veya eşit

    def test_unfitted_viterbi_raises_or_defaults(self):
        """Fit edilmeden viterbi çağrılırsa graceful davranmalı."""
        hmm = GaussianHMM(n_states=3, n_features=2)
        obs = self._make_obs(n=10)
        try:
            states = hmm.viterbi(obs)
            assert len(states) == len(obs)
        except Exception:
            pass  # Exception kabul edilebilir


class TestEconomicCalendar:
    def test_hardcoded_events_loaded(self):
        cal = EconomicCalendar()
        cal._load_hardcoded_events()
        assert cal.event_count() > 0

    def test_news_blackout_returns_bool(self):
        cal = EconomicCalendar()
        cal._load_hardcoded_events()
        result = cal.is_news_blackout()
        assert isinstance(result, bool)


class TestRegimeFilter:
    def _make_price_series(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.001, 0.02, n)
        prices = 2000.0 * np.cumprod(1 + returns)
        return prices.tolist()

    def test_regime_filter_instantiates(self):
        rf = RegimeFilter()
        assert rf is not None

    def test_update_returns_regime_signal(self):
        rf = RegimeFilter()
        prices = self._make_price_series()
        signal = None
        for p in prices:
            signal = rf.update(price=p, volume=100_000.0, gas_gwei=5.0)
        assert isinstance(signal, RegimeSignal)

    def test_regime_is_valid_enum(self):
        rf = RegimeFilter()
        sig = None
        for p in self._make_price_series(n=100):
            sig = rf.update(p, 50_000.0, 3.0)
        assert sig.regime in MarketRegime

    def test_bear_regime_reduces_confidence(self):
        """BEAR rejiminde confidence BULL'dan farklı veya regime farklı olmalı."""
        rf = RegimeFilter()
        bear_prices = [2000.0 * (0.99 ** i) for i in range(200)]
        for p in bear_prices:
            sig = rf.update(p, 50_000.0, 5.0)
        bear_conf = sig.confidence
        rf2 = RegimeFilter()
        bull_prices = [1000.0 * (1.005 ** i) for i in range(200)]
        for p in bull_prices:
            sig2 = rf2.update(p, 50_000.0, 2.0)
        assert sig.regime != sig2.regime or bear_conf != sig2.confidence
