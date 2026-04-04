"""tests/unit/test_flash_crash.py — FlashCrashDetector kapsamlı testler."""
from __future__ import annotations

import time

import pytest

from wallet_manager.strategies.flash_crash_detector import FlashCrashDetector


class TestCrashDetection:
    def test_exact_threshold_triggers(self):
        """Tam eşikte crash algılanır."""
        detector = FlashCrashDetector(crash_threshold=0.08, window_seconds=300.0)
        for _ in range(10):
            detector.ingest_price(2000.0)
        detector._last_check_time = time.time() - 30
        alert = detector.ingest_price(2000.0 * 0.92)
        assert alert is not None

    def test_below_threshold_no_alert(self):
        """Eşiğin altında crash algılanmaz."""
        detector = FlashCrashDetector(crash_threshold=0.08, window_seconds=300.0)
        for _ in range(10):
            detector.ingest_price(2000.0)
        detector._last_check_time = time.time() - 30
        detector.ingest_price(2000.0 * 0.95)  # sadece %5 düşüş
        assert not detector.is_kill_switch_active

    def test_kill_switch_prevents_double_trigger(self):
        """KILL_SWITCH aktifken ikinci alert gönderilmez."""
        detector = FlashCrashDetector(crash_threshold=0.05, window_seconds=300.0)
        for _ in range(10):
            detector.ingest_price(2000.0)
        detector._kill_switch_active = True
        detector._last_check_time = time.time() - 30
        alert = detector.ingest_price(1500.0)
        assert alert is None  # Zaten aktif, ikinci alert yok

    def test_reset_clears_state(self):
        """reset_kill_switch() sonrası kill switch temizlenir."""
        detector = FlashCrashDetector(crash_threshold=0.05, window_seconds=300.0)
        detector._kill_switch_active = True
        detector.reset_kill_switch()
        assert not detector.is_kill_switch_active
        detector.ingest_price(2000.0)

    def test_window_expiry_resets_baseline(self):
        """Pencere süresi geçince eski fiyat baseline güncellenir."""
        detector = FlashCrashDetector(crash_threshold=0.08, window_seconds=1.0)
        detector.ingest_price(2000.0)
        time.sleep(1.1)
        detector.ingest_price(1900.0)
        assert not detector.is_kill_switch_active

    def test_price_recovery_no_false_positive(self):
        """Fiyat düşüp çıkınca false positive olmamalı."""
        detector = FlashCrashDetector(crash_threshold=0.10, window_seconds=300.0)
        prices = [2000.0, 1950.0, 1980.0, 2010.0, 2050.0]
        for p in prices:
            detector.ingest_price(p)
        assert not detector.is_kill_switch_active

    def test_no_crash_on_stable_prices(self):
        """Stabil fiyatlarda kill switch devreye girmez."""
        detector = FlashCrashDetector(crash_threshold=0.08, window_seconds=300.0)
        for price in [2000.0, 2001.0, 2002.0, 2000.5, 2001.5]:
            detector.ingest_price(price)
        assert not detector.is_kill_switch_active

    def test_crash_detected_on_sharp_drop(self):
        """Keskin düşüşte crash algılanır."""
        detector = FlashCrashDetector(crash_threshold=0.05, window_seconds=300.0)
        for _ in range(5):
            detector.ingest_price(2000.0)
        detector._last_check_time = time.time() - 20
        alert = detector.ingest_price(1800.0)  # 10% drop
        assert alert is not None or detector.is_kill_switch_active

    def test_detector_initial_state(self):
        """Başlangıç durumunda kill switch aktif değil."""
        detector = FlashCrashDetector()
        assert not detector.is_kill_switch_active

    def test_ingesting_price_does_not_raise(self):
        """Fiyat ingestion exception fırlatmamalı."""
        detector = FlashCrashDetector(crash_threshold=0.08, window_seconds=60.0)
        for price in [2000.0, 1900.0, 1800.0]:
            detector.ingest_price(price)
        assert len(detector._prices) >= 1
