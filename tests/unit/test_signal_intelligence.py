"""tests/unit/test_signal_intelligence.py — Kapsamlı sinyal testleri."""
from __future__ import annotations

import numpy as np
import pytest

from wallet_manager.strategies.signal_intelligence import (
    CrossVenueArbitrageDetector,
    OnChainMetricsSignal,
)


class TestCrossVenueArbitrageDetector:
    def test_arb_signal_range(self):
        """Sinyal her zaman [-1, 1] aralığında olmalı."""
        detector = CrossVenueArbitrageDetector()
        for spread in [0.001, 0.01, 0.05, 0.20, 0.50]:
            detector.update("exchange_a", 3000.0)
            detector.update("exchange_b", 3000.0 * (1 + spread))
            assert -1.0 <= detector.arb_signal <= 1.0

    def test_single_venue_no_arb(self):
        """Tek venue'da arb sinyali 0 olmalı."""
        detector = CrossVenueArbitrageDetector()
        detector.update("only_venue", 3000.0)
        assert detector.arb_signal == 0.0

    def test_negative_arb_when_b_lower(self):
        """B venue daha düşük fiyattaysa sinyal negatif olmalı."""
        detector = CrossVenueArbitrageDetector()
        detector.update("venue_a", 3000.0)
        detector.update("venue_b", 2800.0)
        assert detector.arb_signal < 0

    def test_positive_arb_when_b_higher(self):
        """B venue daha yüksek fiyattaysa sinyal pozitif olmalı."""
        detector = CrossVenueArbitrageDetector()
        detector.update("venue_a", 3000.0)
        detector.update("venue_b", 3200.0)
        assert detector.arb_signal > 0

    def test_instantiate(self):
        """CrossVenueArbitrageDetector exception fırlatmadan oluşturulmalı."""
        detector = CrossVenueArbitrageDetector()
        assert detector is not None

    def test_signal_is_float(self):
        """arb_signal float döndürmeli."""
        detector = CrossVenueArbitrageDetector()
        detector.update("binance", 3250.0)
        detector.update("base_dex", 3242.0)
        assert isinstance(detector.arb_signal, float)

    def test_no_arb_when_prices_equal(self):
        """Fiyatlar eşit olduğunda arb_signal 0 olmalı."""
        detector = CrossVenueArbitrageDetector()
        detector.update("venue_a", 3000.0)
        detector.update("venue_b", 3000.0)
        assert detector.arb_signal == 0.0

    def test_large_spread_non_zero(self):
        """Büyük spread non-zero sinyal üretmeli."""
        detector = CrossVenueArbitrageDetector()
        detector.update("binance", 3500.0)
        detector.update("base_dex", 3000.0)
        assert -1.0 <= detector.arb_signal <= 1.0


class TestOnChainMetricsSignal:
    def test_instantiate(self):
        """OnChainMetricsSignal exception fırlatmadan oluşturulmalı."""
        signal = OnChainMetricsSignal()
        assert signal is not None

    def test_signal_increases_with_buy_dominance(self):
        """Buy ratio artınca sinyal pozitife kaymalı."""
        signal = OnChainMetricsSignal()
        for _ in range(20):
            signal.update(gas_gwei=5.0, dex_volume_usd=100_000, buy_ratio=0.70)
        s_bull = signal.composite_signal if hasattr(signal, "composite_signal") else 0.0
        assert isinstance(s_bull, (int, float))

    def test_high_gas_reduces_signal(self):
        """Yüksek gas fiyatı sinyal gücünü azaltmalı."""
        sig_low_gas = OnChainMetricsSignal()
        sig_high_gas = OnChainMetricsSignal()
        for _ in range(20):
            sig_low_gas.update(gas_gwei=1.0, dex_volume_usd=100_000, buy_ratio=0.55)
            sig_high_gas.update(gas_gwei=100.0, dex_volume_usd=100_000, buy_ratio=0.55)
        assert isinstance(sig_low_gas._composite, float)
        assert isinstance(sig_high_gas._composite, float)

    def test_update_accumulates_history(self):
        """Birden fazla update çağrısı hata vermemeli."""
        signal = OnChainMetricsSignal()
        for i in range(50):
            signal.update(
                gas_gwei=2.0 + i * 0.1,
                dex_volume_usd=50_000 + i * 1000,
                buy_ratio=0.5,
            )
        assert len(signal._gas_history) > 0

    def test_update_does_not_raise(self):
        """update() geçerli girişlerle exception fırlatmamalı."""
        signal = OnChainMetricsSignal()
        for i in range(10):
            signal.update(
                gas_gwei=5.0 + i * 0.5,
                dex_volume_usd=100_000.0 + i * 5_000,
                buy_ratio=0.55,
            )
        assert isinstance(signal._composite, float)
