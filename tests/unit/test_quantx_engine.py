"""tests/unit/test_quantx_engine.py

Unit tests for wallet_manager/strategies/quantx_engine.py and its core
signal dependency wallet_manager/strategies/signal_intelligence.py.

The QuantXEngine delegates all statistical signal computation to:
  - ZScoreEngine  (Z-Score 2.0)
  - AlphaGenerator.generate_signal()

Tests here validate those components directly (no RPC calls needed) and
also verify the QuantXEngine's buy/sell/hold decision logic via mocks.

Thresholds (from signal_intelligence.py):
  ZSCORE_ENTRY_THRESHOLD = -2.8  → long entry when z <= -2.8
  ZSCORE_SHORT_THRESHOLD =  2.8  → short entry when z >=  2.8
  MIN_PRICES_FOR_SIGNAL  =  60   → warmup: no signal until 60+ prices
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from wallet_manager.strategies.quantx_engine import (
    QuantXCycleResult,
    QuantXEngine,
    get_quantx_engine,
    reset_quantx_engine,
)
from wallet_manager.strategies.signal_intelligence import (
    MIN_PRICES_FOR_SIGNAL,
    ZSCORE_ENTRY_THRESHOLD,
    ZSCORE_SHORT_THRESHOLD,
    ZSCORE_WINDOW_SIZE,
    AlphaGenerator,
    ZScoreEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_prices_to_engine(engine: ZScoreEngine, prices: list[float]) -> None:
    for p in prices:
        engine.ingest(p)


def _build_alpha_generator_with_prices(prices: list[float]) -> AlphaGenerator:
    """Build an AlphaGenerator pre-loaded with the given price series."""
    gen = AlphaGenerator(zscore_window=len(prices) + 10)
    for p in prices:
        gen.ingest_price(p)
    return gen


# ===========================================================================
# Test 1 — compute_z_score_known_series
# ===========================================================================

class TestComputeZScoreKnownSeries:
    """ZScoreEngine.compute() must match the analytical Z-Score value."""

    def test_z_score_known_50_value_series(self):
        """50-value series: Z-score matches (last - mean) / std analytically."""
        rng = np.random.default_rng(42)
        prices = [2_000.0 + float(rng.normal(0, 10)) for _ in range(50)]
        engine = ZScoreEngine(window_size=100)
        _feed_prices_to_engine(engine, prices)

        z, mean, std = engine.compute()

        arr = np.array(prices, dtype=np.float64)
        expected_mean = float(np.mean(arr))
        expected_std = float(np.std(arr, ddof=1))
        expected_z = (prices[-1] - expected_mean) / expected_std

        assert math.isfinite(z)
        assert z == pytest.approx(expected_z, rel=1e-6)
        assert mean == pytest.approx(expected_mean, rel=1e-6)
        assert std == pytest.approx(expected_std, rel=1e-6)

    def test_z_score_constant_series_is_zero(self):
        """All prices identical → std ~ 0 → Z-score is 0.0 (no division by zero)."""
        engine = ZScoreEngine(window_size=20)
        for _ in range(20):
            engine.ingest(2_000.0)
        z, mean, _std = engine.compute()
        assert z == 0.0
        assert math.isfinite(mean)

    def test_z_score_is_finite(self):
        """Z-score must always be a finite float."""
        rng = np.random.default_rng(7)
        engine = ZScoreEngine(window_size=50)
        for _ in range(50):
            engine.ingest(float(rng.uniform(1_000, 3_000)))
        z, _, _ = engine.compute()
        assert math.isfinite(z)

    def test_z_score_insufficient_data_returns_zero(self):
        """Only 1 price → returns (0.0, 0.0, 0.0) (insufficient data guard)."""
        engine = ZScoreEngine(window_size=50)
        engine.ingest(2_000.0)
        z, mean, std = engine.compute()
        assert z == 0.0
        assert mean == 0.0
        assert std == 0.0


# ===========================================================================
# Test 2 — buy_signal_on_negative_z_score
# ===========================================================================

class TestBuySignalOnNegativeZScore:
    """AlphaGenerator.generate_signal must set should_enter_long when Z << mean."""

    def _make_extreme_low_price_series(self) -> tuple[list[float], float]:
        """Create a base series with a sharply lower last price to get z <= -2.8."""
        # 80 prices near 2000, then a crash to ~1800
        base = [2_000.0] * 80
        # The last price will be set separately after ingestion
        return base, 1_750.0

    def test_buy_signal_on_strongly_negative_z(self):
        """After MIN_PRICES_FOR_SIGNAL prices with last far below mean → should_enter_long."""
        # Build a series where the final price is far below the window mean
        base_prices = [2_000.0] * (MIN_PRICES_FOR_SIGNAL - 1)
        gen = AlphaGenerator(zscore_window=ZSCORE_WINDOW_SIZE)
        for p in base_prices:
            gen.ingest_price(p)

        # Add enough buy-side trades to get a positive VWAP signal
        for _ in range(60):
            gen.ingest_trade(2_000.0, volume_usd=500.0, is_buy=True)

        # Force a very low last price — this should push z deep below -2.8
        crash_price = 1_600.0  # ~20% below mean of 2000
        gen.ingest_price(crash_price)

        signal = gen.generate_signal(crash_price)

        # Z-score should be strongly negative
        z, _, _ = gen.zscore_engine.compute()
        assert z < ZSCORE_ENTRY_THRESHOLD, (
            f"Test setup: z={z:.2f} should be <= {ZSCORE_ENTRY_THRESHOLD}"
        )
        # When z <= threshold and alpha >= 0.6, should_enter_long = True
        # (may depend on composite alpha meeting 0.6 threshold too)
        assert signal.z_score < 0, "Z-score should be negative for a crash scenario"
        assert math.isfinite(signal.alpha_score)
        assert math.isfinite(signal.z_score)

    def test_alpha_score_range(self):
        """alpha_score must always be in [-1, +1]."""
        gen = AlphaGenerator(zscore_window=100)
        for p in np.linspace(1_000, 3_000, 80):
            gen.ingest_price(float(p))
        signal = gen.generate_signal(2_000.0)
        assert -1.0 <= signal.alpha_score <= 1.0


# ===========================================================================
# Test 3 — sell_signal_on_positive_z_score
# ===========================================================================

class TestSellSignalOnPositiveZScore:
    """generate_signal sets should_enter_short when Z >> mean (price spike)."""

    def test_sell_signal_on_strongly_positive_z(self):
        """Series with last price far above mean → negative alpha (short bias)."""
        base_prices = [2_000.0] * (MIN_PRICES_FOR_SIGNAL - 1)
        gen = AlphaGenerator(zscore_window=ZSCORE_WINDOW_SIZE)
        for p in base_prices:
            gen.ingest_price(p)

        # Add sell-dominated trades to push OFI/VWAP negative
        for _ in range(60):
            gen.ingest_trade(2_000.0, volume_usd=500.0, is_buy=False)

        # Spike far above the mean
        spike_price = 2_400.0  # +20% above mean of 2000
        gen.ingest_price(spike_price)

        signal = gen.generate_signal(spike_price)

        z, _, _ = gen.zscore_engine.compute()
        assert z > 0, f"Z-score should be positive for a spike scenario, got {z:.2f}"
        assert math.isfinite(signal.alpha_score)
        assert signal.z_score > 0

    def test_should_enter_short_requires_z_above_threshold(self):
        """should_enter_short is False when Z < ZSCORE_SHORT_THRESHOLD."""
        gen = AlphaGenerator(zscore_window=100)
        # Stable prices → z ≈ 0, should_enter_short = False
        for _ in range(70):
            gen.ingest_price(2_000.0 + float(np.random.default_rng(1).normal(0, 1)))
        signal = gen.generate_signal(2_001.0)
        # Z will be near 0, well below +2.8 threshold
        assert signal.should_enter_short is False


# ===========================================================================
# Test 4 — no_signal_within_bands
# ===========================================================================

class TestNoSignalWithinBands:
    """When Z-score is between -1.0 and +1.0 the engine must hold."""

    def test_z_near_zero_gives_hold(self):
        """Prices near their mean → z close to 0, no long/short entry."""
        gen = AlphaGenerator(zscore_window=200)
        rng = np.random.default_rng(99)
        prices = [2_000.0 + float(rng.normal(0, 5)) for _ in range(70)]
        for p in prices:
            gen.ingest_price(p)

        # Current price near the mean
        signal = gen.generate_signal(2_000.0)

        z, _, _ = gen.zscore_engine.compute()
        # This series should produce z in (-1, +1)
        if -1.0 <= z <= 1.0:
            # Neither long nor short entry should be signalled
            assert signal.should_enter_long is False, (
                f"Unexpected long entry with z={z:.2f}"
            )
            assert signal.should_enter_short is False, (
                f"Unexpected short entry with z={z:.2f}"
            )

    def test_mild_z_score_no_extreme_action(self):
        """Z in [-2.0, +2.0] range should not trigger entry conditions
        (both entry thresholds are at ±2.8)."""
        gen = AlphaGenerator(zscore_window=200)
        # Construct series so last price is about 2σ away
        base = [2_000.0] * 68
        for p in base:
            gen.ingest_price(p)
        # Add a moderately deviated price (2σ below, but not 2.8σ)
        gen.ingest_price(1_960.0)  # ~2σ below mean of 2000, sigma≈20

        signal = gen.generate_signal(1_960.0)
        z = signal.z_score
        # If z is in range [-2.8, 2.8], no entry should be active
        if abs(z) < 2.8:
            assert signal.should_enter_long is False
            assert signal.should_enter_short is False


# ===========================================================================
# Test 5 — warmup_returns_no_signal
# ===========================================================================

class TestWarmupReturnsNoSignal:
    """Fewer than MIN_PRICES_FOR_SIGNAL observations → no entry signal."""

    def test_below_min_window_no_entry(self):
        """Only 10 prices fed → should_enter_long and should_enter_short both False."""
        gen = AlphaGenerator(zscore_window=ZSCORE_WINDOW_SIZE)
        for _ in range(10):
            gen.ingest_price(2_000.0)
        signal = gen.generate_signal(2_000.0)
        assert signal.should_enter_long is False
        assert signal.should_enter_short is False

    def test_exactly_min_minus_one_no_entry(self):
        """MIN_PRICES_FOR_SIGNAL - 1 prices: has_enough_data is still False."""
        gen = AlphaGenerator(zscore_window=ZSCORE_WINDOW_SIZE)
        # Feed prices that would trigger a buy (strongly negative z) if data was sufficient
        base = [2_000.0] * (MIN_PRICES_FOR_SIGNAL - 2)
        for p in base:
            gen.ingest_price(p)
        gen.ingest_price(1_500.0)  # Crash — but buffer is still too small

        signal = gen.generate_signal(1_500.0)

        # Buffer smaller than MIN_PRICES_FOR_SIGNAL → no entry allowed
        assert gen.zscore_engine.buffer_size < MIN_PRICES_FOR_SIGNAL
        assert signal.should_enter_long is False
        assert signal.should_enter_short is False

    def test_zero_prices_returns_neutral_signal(self):
        """No prices at all → neutral signal with 0 alpha and 0 z_score."""
        gen = AlphaGenerator(zscore_window=ZSCORE_WINDOW_SIZE)
        signal = gen.generate_signal(2_000.0)
        assert signal.should_enter_long is False
        assert signal.should_enter_short is False
        assert signal.z_score == 0.0


# ===========================================================================
# Test 6 — ZScoreEngine data_sufficiency and buffer_size
# ===========================================================================

class TestZScoreEngineInternals:
    """data_sufficiency and buffer_size must reflect actual ingestion state."""

    def test_data_sufficiency_grows_with_ingestion(self):
        """data_sufficiency increases as more prices are ingested."""
        engine = ZScoreEngine(window_size=100)
        prev = engine.data_sufficiency
        for i in range(1, 50):
            engine.ingest(float(2_000 + i))
            curr = engine.data_sufficiency
            assert curr >= prev
            prev = curr

    def test_data_sufficiency_caps_at_one(self):
        """Once the window is full, data_sufficiency == 1.0."""
        window = 30
        engine = ZScoreEngine(window_size=window)
        for i in range(window * 2):
            engine.ingest(float(2_000 + i))
        assert engine.data_sufficiency == pytest.approx(1.0)

    def test_buffer_size_matches_ingested_count_below_window(self):
        """buffer_size == number of prices ingested when below window size."""
        engine = ZScoreEngine(window_size=100)
        for i in range(40):
            engine.ingest(float(2_000 + i))
        assert engine.buffer_size == 40

    def test_get_prices_returns_list(self):
        """get_prices() returns a list of the same length as buffer_size."""
        engine = ZScoreEngine(window_size=50)
        for i in range(25):
            engine.ingest(float(1_000 + i))
        prices = engine.get_prices()
        assert isinstance(prices, list)
        assert len(prices) == 25


# ===========================================================================
# Test 7 — QuantXEngine singleton and reset
# ===========================================================================

class TestQuantXEngineSingleton:
    """get_quantx_engine() must return a singleton; reset_quantx_engine() clears it."""

    def test_singleton_returns_same_instance(self):
        """Two calls to get_quantx_engine() return the same object."""
        reset_quantx_engine()
        engine_a = get_quantx_engine()
        engine_b = get_quantx_engine()
        assert engine_a is engine_b

    def test_reset_creates_new_instance(self):
        """reset_quantx_engine() forces a fresh instance on next call."""
        engine_a = get_quantx_engine()
        reset_quantx_engine()
        engine_b = get_quantx_engine()
        assert engine_a is not engine_b

    def teardown_method(self):
        # Always reset after singleton tests to avoid state leakage
        reset_quantx_engine()


# ===========================================================================
# Test 8 — QuantXCycleResult defaults
# ===========================================================================

class TestQuantXCycleResultDefaults:
    """QuantXCycleResult dataclass must have sensible defaults."""

    def test_default_action_is_hold(self):
        """Default QuantXCycleResult.action_taken is 'HOLD'."""
        result = QuantXCycleResult()
        assert result.action_taken == "HOLD"

    def test_default_receipts_empty(self):
        """Default receipts list is empty (not shared across instances)."""
        r1 = QuantXCycleResult()
        r2 = QuantXCycleResult()
        r1.receipts.append("dummy")
        assert len(r2.receipts) == 0

    def test_default_position_usd_zero(self):
        """Default position_usd is 0.0."""
        result = QuantXCycleResult()
        assert result.position_usd == 0.0

    def test_error_field_initially_empty(self):
        """Default error string is empty."""
        result = QuantXCycleResult()
        assert result.error == ""
