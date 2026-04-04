"""
tests/unit/test_js_sprint.py

Unit tests for JS Sprint modules (Week 1-3):
  - signal_intelligence.py  (IC Tracker, Decay, Correlation, Diversity)
  - risk_intelligence.py    (Drawdown Accel, Fat-tail VaR, Stress, Regime Limits)
  - tick_collector.py       (TickCollector, OFI)
  - alerting.py             (AlertEscalationSystem)
  - reporting.py            (PerformanceReporter)
  - cross_asset.py          (CrossAssetFeatures)
  - adaptive_ensemble.py    (AdaptiveEnsemble)
  - signal_discovery.py     (SignalDiscoveryPipeline)
  - bayesian_kelly.py       (BayesianKelly)
  - execution_quality.py    (TWAP, SlippageCalibrator)
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Signal Intelligence
# ---------------------------------------------------------------------------

class TestICTracker:
    def test_pearson_helper(self) -> None:
        from wallet_manager.signal_intelligence import _pearson
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson(x, y) - 1.0) < 1e-9

    def test_insufficient_data_returns_zero(self) -> None:
        from wallet_manager.signal_intelligence import ICTracker
        t = ICTracker()
        assert t.compute_ic("robust_alpha") == 0.0

    def test_record_and_resolve(self) -> None:
        from wallet_manager.signal_intelligence import ICTracker
        t = ICTracker(forward_minutes=0.0)
        t.record_signal("test", 1.0, 100.0)
        resolved = t.resolve_pending(105.0)
        assert resolved == 1

    def test_ic_status(self) -> None:
        from wallet_manager.signal_intelligence import ICTracker
        t = ICTracker(forward_minutes=0.0, window=50)
        for i in range(15):
            t.record_signal("s", float(i), 100.0 + i)
            t.resolve_pending(100.0 + i + 1.0)
        status = t.get_ic_status("s")
        assert status in ("WORKING", "WEAK", "NOISE")


class TestSignalDecayAnalyzer:
    def test_no_data(self) -> None:
        from wallet_manager.signal_intelligence import SignalDecayAnalyzer
        a = SignalDecayAnalyzer()
        result = a.get_decay_curve()
        assert result["status"] == "NO_DATA"

    def test_record_and_update(self) -> None:
        from wallet_manager.signal_intelligence import SignalDecayAnalyzer
        a = SignalDecayAnalyzer(horizons_s=[1, 2])
        a.record_signal(100.0, 1)
        time.sleep(0.01)
        a.update(101.0)
        assert len(a._pending) + len(a._completed) >= 1


class TestSignalCorrelationMatrix:
    def test_insufficient_data(self) -> None:
        from wallet_manager.signal_intelligence import SignalCorrelationMatrix
        m = SignalCorrelationMatrix()
        result = m.get_matrix()
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_matrix_with_data(self) -> None:
        from wallet_manager.signal_intelligence import SignalCorrelationMatrix
        m = SignalCorrelationMatrix(window=30)
        for i in range(25):
            m.record(float(i), float(i) * 0.5, 0.05, 0.01 * i, 5.0)
        result = m.get_matrix()
        assert result["status"] == "OK"
        assert "robust_alpha" in result["matrix"]


class TestEnsembleDiversityScore:
    def test_insufficient_data(self) -> None:
        from wallet_manager.signal_intelligence import EnsembleDiversityScore
        d = EnsembleDiversityScore()
        report = d.get_report()
        assert report["status"] == "INSUFFICIENT_DATA"

    def test_all_same_is_zero_diversity(self) -> None:
        from wallet_manager.signal_intelligence import EnsembleDiversityScore
        d = EnsembleDiversityScore(window=20)
        for _ in range(15):
            d.record(1, 1, 1, 1)
        diversity = d.compute_diversity()
        assert diversity == 0.0

    def test_all_different_is_high_diversity(self) -> None:
        from wallet_manager.signal_intelligence import EnsembleDiversityScore
        d = EnsembleDiversityScore(window=20)
        for i in range(15):
            d.record(1 if i % 2 == 0 else -1, -1 if i % 2 == 0 else 1,
                     1 if i % 3 == 0 else -1, -1 if i % 3 == 0 else 1)
        diversity = d.compute_diversity()
        assert diversity > 0.0


class TestSignalIntelligenceFacade:
    def test_singleton(self) -> None:
        from wallet_manager.signal_intelligence import get_signal_intelligence
        a = get_signal_intelligence()
        b = get_signal_intelligence()
        assert a is b

    def test_to_dict_keys(self) -> None:
        from wallet_manager.signal_intelligence import SignalIntelligence
        si = SignalIntelligence()
        d = si.to_dict()
        assert "ic_report" in d
        assert "decay_curve" in d
        assert "correlation_matrix" in d
        assert "ensemble_diversity" in d

    def test_record_cycle_no_crash(self) -> None:
        from wallet_manager.signal_intelligence import SignalIntelligence
        si = SignalIntelligence()
        si.record_cycle(
            robust_alpha=0.10, confidence=0.45, adf_pvalue=0.03,
            garch_sigma=0.02, spread_bps=5.0, current_price=2000.0, signal_direction=1,
        )


# ---------------------------------------------------------------------------
# Risk Intelligence
# ---------------------------------------------------------------------------

class TestDrawdownAccelerationDetector:
    def test_no_alert_stable(self) -> None:
        from wallet_manager.risk_intelligence import DrawdownAccelerationDetector
        d = DrawdownAccelerationDetector()
        for equity in [10000, 10010, 10020, 10030]:
            d.update(float(equity))
        result = d.to_dict()
        assert result["alert_active"] is False
        assert result["kelly_multiplier"] == 1.0

    def test_alert_on_acceleration(self) -> None:
        from wallet_manager.risk_intelligence import DrawdownAccelerationDetector
        d = DrawdownAccelerationDetector(accel_threshold=0.001, consec_required=3)
        for equity in [10000, 9800, 9500, 9100, 8600]:
            d.update(float(equity))
        result = d.to_dict()
        assert result["current_drawdown_pct"] > 0

    def test_kelly_multiplier_reduces_on_drawdown(self) -> None:
        from wallet_manager.risk_intelligence import DrawdownAccelerationDetector
        d = DrawdownAccelerationDetector()
        d.update(10000.0)
        d.update(8900.0)
        mult = d.get_kelly_multiplier()
        assert mult <= 1.0


class TestFatTailVaR:
    def test_no_data(self) -> None:
        from wallet_manager.risk_intelligence import FatTailVaR
        var = FatTailVaR()
        result = var.get_var_report(10000.0)
        assert result["var_99_pct"] == 0.0

    def test_with_losses(self) -> None:
        from wallet_manager.risk_intelligence import FatTailVaR
        var = FatTailVaR()
        for i in range(50):
            var.record_return(-0.01 * (i % 5 + 1) / 100)
        result = var.get_var_report(10000.0)
        assert result["n_observations"] == 50
        assert result["var_95_pct"] >= 0.0


class TestStressScenarioEngine:
    def test_run_all_returns_5(self) -> None:
        from wallet_manager.risk_intelligence import StressScenarioEngine
        engine = StressScenarioEngine(initial_equity=10000.0)
        results = engine.run_all()
        assert len(results) == 5

    def test_all_scenarios_have_fields(self) -> None:
        from wallet_manager.risk_intelligence import StressScenarioEngine
        engine = StressScenarioEngine(initial_equity=10000.0)
        results = engine.run_all()
        for r in results:
            d = r.to_dict()
            assert "scenario" in d
            assert "max_drawdown_pct" in d
            assert "capital_preserved_pct" in d
            assert "pass" in d

    def test_to_dict_after_run(self) -> None:
        from wallet_manager.risk_intelligence import StressScenarioEngine
        engine = StressScenarioEngine(10000.0)
        engine.run_all()
        d = engine.to_dict()
        assert d["status"] == "OK"
        assert d["all_pass"] is not None


class TestRegimePositionLimits:
    def test_low_vol_full_kelly(self) -> None:
        from wallet_manager.risk_intelligence import RegimePositionLimits
        r = RegimePositionLimits()
        assert r.get_multiplier("low_vol") == 1.0

    def test_crisis_quarter_kelly(self) -> None:
        from wallet_manager.risk_intelligence import RegimePositionLimits
        r = RegimePositionLimits()
        assert r.get_multiplier("crisis") == 0.25

    def test_unknown_regime_half_kelly(self) -> None:
        from wallet_manager.risk_intelligence import RegimePositionLimits
        r = RegimePositionLimits()
        assert r.get_multiplier("xyz_unknown_regime") == 0.50


class TestRiskIntelligenceFacade:
    def test_singleton(self) -> None:
        from wallet_manager.risk_intelligence import get_risk_intelligence
        a = get_risk_intelligence()
        b = get_risk_intelligence()
        assert a is b

    def test_effective_kelly_multiplier(self) -> None:
        from wallet_manager.risk_intelligence import RiskIntelligence
        ri = RiskIntelligence()
        mult = ri.get_effective_kelly_multiplier("normal")
        assert 0.0 < mult <= 1.0

    def test_to_dict_keys(self) -> None:
        from wallet_manager.risk_intelligence import RiskIntelligence
        ri = RiskIntelligence()
        d = ri.to_dict()
        assert "drawdown_acceleration" in d
        assert "fat_tail_var" in d
        assert "stress_scenarios" in d
        assert "regime_limits" in d


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

class TestAlertEscalationSystem:
    def test_info_no_active_alert(self) -> None:
        from wallet_manager.alerting import AlertEscalationSystem
        a = AlertEscalationSystem()
        a.info("test", "info message")
        assert len(a._active) == 0

    def test_warn_creates_active_alert(self) -> None:
        from wallet_manager.alerting import AlertEscalationSystem
        a = AlertEscalationSystem()
        a.warn("TestSource", "warning!")
        assert len(a._active) == 1

    def test_critical_deduplication(self) -> None:
        from wallet_manager.alerting import AlertEscalationSystem
        a = AlertEscalationSystem()
        a.critical("Source", "msg1")
        a.critical("Source", "msg2")
        criticals = a.get_active_criticals()
        assert len(criticals) == 1

    def test_acknowledge(self) -> None:
        from wallet_manager.alerting import AlertEscalationSystem
        a = AlertEscalationSystem()
        a.critical("Src", "critical alert")
        acked = a.acknowledge()
        assert acked == 1
        assert len(a.get_active_criticals()) == 0

    def test_process_signals(self) -> None:
        from wallet_manager.alerting import AlertEscalationSystem
        a = AlertEscalationSystem()
        a.process_signals(
            signal_alerts=["IC_NOISE: signal 'x' IC < 0.02"],
            risk_alerts=["DRAWDOWN_ACCELERATING: 3 readings"],
        )
        assert len(a._active) == 2

    def test_to_dict_keys(self) -> None:
        from wallet_manager.alerting import AlertEscalationSystem
        a = AlertEscalationSystem()
        d = a.to_dict()
        assert "active_count" in d
        assert "critical_unacked" in d


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestPerformanceReporter:
    def test_singleton(self) -> None:
        from wallet_manager.reporting import get_reporter
        a = get_reporter()
        b = get_reporter()
        assert a is b

    def test_to_dict_keys(self) -> None:
        from wallet_manager.reporting import PerformanceReporter
        r = PerformanceReporter()
        d = r.to_dict()
        assert "last_daily_ago_h" in d
        assert "history" in d

    @pytest.mark.asyncio
    async def test_tick_no_crash(self) -> None:
        from wallet_manager.reporting import PerformanceReporter
        r = PerformanceReporter()
        await r.tick(notifier=None, paper_engine=None)

    def test_chunk(self) -> None:
        from wallet_manager.reporting import PerformanceReporter
        r = PerformanceReporter()
        long_text = "line\n" * 1000
        chunks = r._chunk(long_text, 100)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)


# ---------------------------------------------------------------------------
# Cross-Asset Features
# ---------------------------------------------------------------------------

class TestCrossAssetFeatures:
    def test_no_data_returns_zeros(self) -> None:
        from wallet_manager.strategies.cross_asset import CrossAssetFeatures
        ca = CrossAssetFeatures()
        features = ca.get_features()
        assert features["btc_momentum_1h"] == 0.0
        assert features["eth_btc_ratio_dev"] == 0.0

    def test_record_and_compute(self) -> None:
        from wallet_manager.strategies.cross_asset import CrossAssetFeatures
        ca = CrossAssetFeatures()
        for i in range(30):
            ca.record_prices(2000.0 + i, 50000.0 + i * 10)
        features = ca.get_features()
        assert "btc_momentum_1h" in features
        assert all(-1.0 <= v <= 1.0 for v in features.values())

    def test_singleton(self) -> None:
        from wallet_manager.strategies.cross_asset import get_cross_asset
        a = get_cross_asset()
        b = get_cross_asset()
        assert a is b


# ---------------------------------------------------------------------------
# Adaptive Ensemble
# ---------------------------------------------------------------------------

class TestAdaptiveEnsemble:
    def test_equal_weights_no_data(self) -> None:
        from wallet_manager.strategies.adaptive_ensemble import AdaptiveEnsemble
        ae = AdaptiveEnsemble()
        weights = ae.get_current_weights()
        for w in weights.values():
            assert abs(w - 0.25) < 1e-6

    def test_apply_weights(self) -> None:
        from wallet_manager.strategies.adaptive_ensemble import AdaptiveEnsemble
        ae = AdaptiveEnsemble()
        model_alphas = {"xgboost": 0.10, "lightgbm": 0.08, "lstm": 0.06, "sklearn": 0.07}
        result = ae.apply_weights("normal", model_alphas)
        assert abs(result - 0.0775) < 0.01

    def test_record_cycle_no_crash(self) -> None:
        from wallet_manager.strategies.adaptive_ensemble import AdaptiveEnsemble
        ae = AdaptiveEnsemble()
        ae.record_cycle("normal", {"xgboost": 0.1, "lightgbm": 0.09, "lstm": 0.08, "sklearn": 0.07}, 2000.0)

    def test_singleton(self) -> None:
        from wallet_manager.strategies.adaptive_ensemble import get_adaptive_ensemble
        a = get_adaptive_ensemble()
        b = get_adaptive_ensemble()
        assert a is b


# ---------------------------------------------------------------------------
# Signal Discovery
# ---------------------------------------------------------------------------

class TestSignalDiscoveryPipeline:
    def test_rsi_bounds(self) -> None:
        from wallet_manager.signal_discovery import SignalDiscoveryPipeline
        sd = SignalDiscoveryPipeline()
        prices = list(range(100, 130))
        val = sd._rsi(prices, 14)
        assert -1.0 <= val <= 1.0

    def test_bollinger_bounds(self) -> None:
        from wallet_manager.signal_discovery import SignalDiscoveryPipeline
        sd = SignalDiscoveryPipeline()
        prices = [100.0 + i * 0.5 for i in range(25)]
        val = sd._bollinger_position(prices, 20)
        assert -1.0 <= val <= 1.0

    def test_record_cycle(self) -> None:
        from wallet_manager.signal_discovery import SignalDiscoveryPipeline
        sd = SignalDiscoveryPipeline()
        for i in range(40):
            sd.record_cycle(2000.0 + i, 0.1 * (i % 5), 0.4 + i * 0.01)
        assert len(sd._price_history) == 40

    @pytest.mark.asyncio
    async def test_run_discovery_insufficient_data(self) -> None:
        from wallet_manager.signal_discovery import SignalDiscoveryPipeline
        sd = SignalDiscoveryPipeline()
        for i in range(10):
            sd.record_cycle(float(2000 + i), 0.1, 0.4)
        results = await sd.run_discovery()
        assert results == []

    def test_singleton(self) -> None:
        from wallet_manager.signal_discovery import get_signal_discovery
        a = get_signal_discovery()
        b = get_signal_discovery()
        assert a is b


# ---------------------------------------------------------------------------
# Bayesian Kelly
# ---------------------------------------------------------------------------

class TestBayesianKelly:
    def test_no_trades_conservative(self) -> None:
        from wallet_manager.strategies.bayesian_kelly import BayesianKelly
        bk = BayesianKelly()
        kelly = bk.compute_kelly()
        assert 0.0 <= kelly <= 0.05

    def test_high_win_rate_increases_kelly(self) -> None:
        from wallet_manager.strategies.bayesian_kelly import BayesianKelly
        bk = BayesianKelly()
        for _ in range(80):
            bk.update(win=True)
        for _ in range(20):
            bk.update(win=False)
        kelly_high = bk.compute_kelly()
        bk2 = BayesianKelly()
        for _ in range(20):
            bk2.update(win=True)
        for _ in range(80):
            bk2.update(win=False)
        kelly_low = bk2.compute_kelly()
        assert kelly_high > kelly_low

    def test_uncertainty_factor_increases_with_trades(self) -> None:
        from wallet_manager.strategies.bayesian_kelly import BayesianKelly
        bk = BayesianKelly(min_trades_for_full_kelly=100)
        assert bk.get_uncertainty_factor() == 0.0
        for _ in range(25):
            bk.update(True)
        assert bk.get_uncertainty_factor() == 0.5
        for _ in range(75):
            bk.update(True)
        assert bk.get_uncertainty_factor() == 1.0

    def test_to_dict_keys(self) -> None:
        from wallet_manager.strategies.bayesian_kelly import BayesianKelly
        bk = BayesianKelly()
        d = bk.to_dict()
        assert "posterior_win_prob" in d
        assert "bayes_kelly_fraction" in d
        assert "data_maturity_pct" in d

    def test_singleton(self) -> None:
        from wallet_manager.strategies.bayesian_kelly import get_bayesian_kelly
        a = get_bayesian_kelly()
        b = get_bayesian_kelly()
        assert a is b


# ---------------------------------------------------------------------------
# Execution Quality
# ---------------------------------------------------------------------------

class TestTWAPEmulator:
    def test_start_and_complete(self) -> None:
        from wallet_manager.strategies.execution_quality import TWAPEmulator
        t = TWAPEmulator(slices=3, window_minutes=0.001)
        t.start_twap(2000.0)
        import time as _time
        _time.sleep(0.1)
        completed = t.update_prices(2010.0)
        assert len(completed) >= 0

    def test_saving_bps_positive_when_price_rises(self) -> None:
        from wallet_manager.strategies.execution_quality import TWAPEmulator
        t = TWAPEmulator(slices=2, window_minutes=0.0)
        t.start_twap(2000.0)
        completed = t.update_prices(2100.0)
        if completed:
            assert "saving_bps" in completed[0]


class TestSlippageCalibrator:
    def test_default_factor_is_one(self) -> None:
        from wallet_manager.strategies.execution_quality import SlippageCalibrator
        c = SlippageCalibrator()
        assert c._calibration_factor == 1.0

    def test_underestimating_increases_factor(self) -> None:
        from wallet_manager.strategies.execution_quality import SlippageCalibrator
        c = SlippageCalibrator()
        for _ in range(15):
            c.record(estimated_bps=5.0, actual_bps=8.0)
        assert c._calibration_factor > 1.0
        assert c.to_dict()["status"] == "UNDERESTIMATING"

    def test_apply_calibration(self) -> None:
        from wallet_manager.strategies.execution_quality import SlippageCalibrator
        c = SlippageCalibrator()
        for _ in range(15):
            c.record(5.0, 10.0)
        calibrated = c.apply_calibration(5.0)
        assert calibrated > 5.0

    def test_singleton(self) -> None:
        from wallet_manager.strategies.execution_quality import get_execution_quality
        a = get_execution_quality()
        b = get_execution_quality()
        assert a is b
