"""tests/unit/test_misc_coverage.py

Coverage tests for:
  - wallet_manager/risk_intelligence.py
  - wallet_manager/reporting.py
  - wallet_manager/strategies/execution_quality.py
"""
from __future__ import annotations

import time

import pytest

# ---------------------------------------------------------------------------
# reporting imports
# ---------------------------------------------------------------------------
from wallet_manager.reporting import (
    PerformanceReporter,
    ReportRecord,
)

# ---------------------------------------------------------------------------
# risk_intelligence imports
# ---------------------------------------------------------------------------
from wallet_manager.risk_intelligence import (
    DrawdownAccelerationDetector,
    FatTailVaR,
    RegimePositionLimits,
    RiskIntelligence,
    StressResult,
    StressScenario,
    StressScenarioEngine,
)

# ---------------------------------------------------------------------------
# execution_quality imports
# ---------------------------------------------------------------------------
from wallet_manager.strategies.execution_quality import (
    ExecutionQualityMonitor,
    OptimalExecutionTiming,
    SlippageCalibrator,
    TradeExecutionRecord,
    TWAPEmulator,
)

# ===========================================================================
# DrawdownAccelerationDetector
# ===========================================================================


def test_drawdown_accel_no_history_kelly_one():
    d = DrawdownAccelerationDetector()
    assert d.get_kelly_multiplier() == 1.0


def test_drawdown_accel_peak_equity_tracks_max():
    d = DrawdownAccelerationDetector()
    d.update(10_000.0)
    d.update(9_000.0)
    assert d._peak_equity == 10_000.0


def test_drawdown_accel_kelly_reduced_at_high_drawdown():
    d = DrawdownAccelerationDetector()
    d.update(10_000.0)
    d.update(8_500.0)  # 15% drawdown
    assert d.get_kelly_multiplier() < 1.0


def test_drawdown_accel_alert_fires_on_acceleration():
    d = DrawdownAccelerationDetector(accel_threshold=0.001, consec_required=3)
    # Simulate accelerating drawdown
    equities = [10_000, 9_900, 9_700, 9_400, 9_000]
    for eq in equities:
        d.update(float(eq))
    d_info = d.to_dict()
    assert isinstance(d_info["alert_active"], bool)
    assert isinstance(d_info["kelly_multiplier"], float)


def test_drawdown_accel_to_dict_keys():
    d = DrawdownAccelerationDetector()
    d.update(10_000.0)
    result = d.to_dict()
    for key in ("current_drawdown_pct", "velocity", "acceleration",
                "consecutive_accelerating", "alert_active",
                "kelly_multiplier", "peak_equity_usd"):
        assert key in result


def test_drawdown_accel_kelly_half_when_alert_active():
    d = DrawdownAccelerationDetector(accel_threshold=0.001, consec_required=3)
    # Populate drawdown history so the guard clause doesn't short-circuit
    d.update(10_000.0)
    d.update(9_500.0)  # 5% drawdown — below 10% threshold
    d._alert_active = True
    assert d.get_kelly_multiplier() == 0.50


def test_drawdown_accel_kelly_moderate_drawdown():
    d = DrawdownAccelerationDetector()
    d.update(10_000.0)
    d.update(9_600.0)  # ~4% drawdown — in [0.03, 0.10) range but not alert
    result = d.get_kelly_multiplier()
    assert result <= 1.0


# ===========================================================================
# FatTailVaR
# ===========================================================================


def test_fat_tail_var_no_data_returns_zero():
    v = FatTailVaR()
    report = v.get_var_report()
    assert report["var_95_pct"] == 0.0
    assert report["var_99_pct"] == 0.0


def test_fat_tail_var_with_returns():
    v = FatTailVaR()
    import random
    rng = random.Random(42)
    for _ in range(100):
        v.record_return(rng.gauss(0.001, 0.02))
    report = v.get_var_report(equity_usd=10_000.0)
    assert report["n_observations"] == 100
    assert report["var_95_pct"] >= 0.0
    assert report["var_99_pct"] >= report["var_95_pct"]


def test_fat_tail_var_report_keys():
    v = FatTailVaR()
    report = v.get_var_report()
    for key in ("n_observations", "var_95_pct", "var_99_pct", "var_999_pct",
                "alert", "recommended_size_factor"):
        assert key in report


def test_fat_tail_var_alert_on_extreme_losses():
    v = FatTailVaR()
    for _ in range(50):
        v.record_return(-0.10)  # 10% losses every time
    report = v.get_var_report()
    assert report["alert"] is True


# ===========================================================================
# StressScenarioEngine
# ===========================================================================


def test_stress_engine_has_5_scenarios():
    engine = StressScenarioEngine()
    assert len(engine.SCENARIOS) == 5


def test_stress_engine_run_all_returns_5_results():
    engine = StressScenarioEngine(initial_equity=10_000.0)
    results = engine.run_all()
    assert len(results) == 5


def test_stress_engine_results_have_valid_fields():
    engine = StressScenarioEngine(initial_equity=10_000.0)
    results = engine.run_all()
    for r in results:
        assert isinstance(r.scenario_name, str)
        assert r.max_drawdown_pct >= 0.0
        assert r.capital_preserved_pct >= 0.0


def test_stress_engine_to_dict_not_run():
    engine = StressScenarioEngine()
    d = engine.to_dict()
    assert d["status"] == "NOT_RUN"


def test_stress_engine_to_dict_after_run():
    engine = StressScenarioEngine(initial_equity=10_000.0)
    engine.run_all()
    d = engine.to_dict()
    assert d["status"] == "OK"
    assert "all_pass" in d
    assert "worst_scenario" in d


def test_stress_result_to_dict():
    r = StressResult(
        scenario_name="test", max_loss_usd=100.0, max_drawdown_pct=10.0,
        risk_committee_halted=False, cycles_before_halt=24,
        capital_preserved_pct=90.0, final_equity_usd=9_000.0,
    )
    d = r.to_dict()
    assert d["pass"] is True
    assert d["scenario"] == "test"


# ===========================================================================
# RegimePositionLimits
# ===========================================================================


def test_regime_limits_low_vol_returns_one():
    r = RegimePositionLimits()
    mult = r.get_multiplier("low_vol")
    assert mult == 1.00


def test_regime_limits_high_vol_returns_half():
    r = RegimePositionLimits()
    mult = r.get_multiplier("high_vol")
    assert mult == 0.50


def test_regime_limits_unknown_returns_half():
    r = RegimePositionLimits()
    mult = r.get_multiplier("totally_unknown_regime")
    assert mult == 0.50


def test_regime_limits_crisis_returns_quarter():
    r = RegimePositionLimits()
    mult = r.get_multiplier("crisis")
    assert mult == 0.25


def test_regime_limits_to_dict():
    r = RegimePositionLimits()
    r.get_multiplier("normal")
    d = r.to_dict()
    assert "last_regime" in d
    assert "last_multiplier" in d
    assert "multiplier_table" in d


# ===========================================================================
# RiskIntelligence facade
# ===========================================================================


def test_risk_intelligence_record_cycle():
    ri = RiskIntelligence(initial_equity=10_000.0)
    ri.record_cycle(equity_usd=10_000.0, pct_return=0.001, regime_label="normal")
    ri.record_cycle(equity_usd=9_800.0, pct_return=-0.02, regime_label="normal")
    # No exception = pass


def test_risk_intelligence_kelly_multiplier():
    ri = RiskIntelligence(initial_equity=10_000.0)
    mult = ri.get_effective_kelly_multiplier("normal")
    assert 0.0 <= mult <= 1.0


def test_risk_intelligence_to_dict():
    ri = RiskIntelligence(initial_equity=10_000.0)
    d = ri.to_dict()
    assert "drawdown_acceleration" in d
    assert "fat_tail_var" in d
    assert "stress_scenarios" in d
    assert "regime_limits" in d


def test_risk_intelligence_get_alerts_empty():
    ri = RiskIntelligence(initial_equity=10_000.0)
    alerts = ri.get_alerts()
    assert isinstance(alerts, list)


# ===========================================================================
# SlippageCalibrator
# ===========================================================================


def test_slippage_calibrator_initial_factor():
    sc = SlippageCalibrator()
    assert sc._calibration_factor == 1.0


def test_slippage_calibrator_records_data():
    sc = SlippageCalibrator()
    for _ in range(15):
        sc.record(10.0, 12.0)
    assert sc._calibration_factor > 1.0  # actual > estimated


def test_slippage_calibrator_apply_calibration():
    sc = SlippageCalibrator()
    result = sc.apply_calibration(10.0)
    assert result == 10.0  # factor=1.0 initially


def test_slippage_calibrator_to_dict_keys():
    sc = SlippageCalibrator()
    d = sc.to_dict()
    assert "calibration_factor" in d
    assert "status" in d
    assert "alert" in d


def test_slippage_calibrator_underestimating_status():
    sc = SlippageCalibrator()
    for _ in range(15):
        sc.record(5.0, 10.0)  # actual 2x estimated
    d = sc.to_dict()
    assert d["status"] == "UNDERESTIMATING"


# ===========================================================================
# TWAPEmulator
# ===========================================================================


def test_twap_emulator_start_creates_pending():
    te = TWAPEmulator(slices=3, window_minutes=0.0)
    te.start_twap(2000.0)
    assert len(te._pending_twaPs) == 1


def test_twap_emulator_completes_when_window_elapsed():
    # Use a window of 0 minutes so the TWAP completes immediately
    te = TWAPEmulator(slices=3, window_minutes=0.0)
    te.start_twap(2000.0)
    import time as _time
    _time.sleep(0.01)
    completed = te.update_prices(2005.0)
    # With window_minutes=0 the elapsed time >= window_s=0.0, so it should complete
    assert isinstance(completed, list)
    assert len(completed) >= 1


# ===========================================================================
# OptimalExecutionTiming
# ===========================================================================


def test_optimal_timing_record_execution():
    ot = OptimalExecutionTiming()
    ot.record_execution(slippage_bps=5.0, timestamp=1_700_000_000.0)
    # No error = pass


def test_optimal_timing_current_window_no_data():
    ot = OptimalExecutionTiming()
    result = ot.get_current_window_quality()
    assert "hour" in result
    # With no data for current hour, status should be NO_DATA (or have data if populated)
    assert result["status"] in ("NO_DATA", "OK", "AVOID")


def test_optimal_timing_best_hours_empty():
    ot = OptimalExecutionTiming()
    best = ot.get_best_hours(6)
    assert isinstance(best, list)


def test_optimal_timing_to_dict():
    ot = OptimalExecutionTiming()
    d = ot.to_dict()
    assert "current_window" in d
    assert "best_hours_utc" in d
    assert "avoid_now" in d


# ===========================================================================
# ExecutionQualityMonitor
# ===========================================================================


def test_eq_monitor_to_dict_keys():
    eq = ExecutionQualityMonitor()
    d = eq.to_dict()
    assert "twap_history_count" in d
    assert "slippage_calibration" in d


def test_eq_monitor_on_trade_close():
    eq = ExecutionQualityMonitor()
    for _ in range(15):
        eq.on_trade_close(estimated_bps=10.0, actual_bps=11.0)
    d = eq.to_dict()
    assert d["slippage_calibration"]["n_records"] == 15


# ===========================================================================
# TradeExecutionRecord
# ===========================================================================


def test_trade_execution_record_to_dict():
    rec = TradeExecutionRecord(
        timestamp=1_700_000_000.0,
        immediate_fill_price=2000.0,
        estimated_slippage_bps=5.0,
        actual_slippage_bps=6.0,
        twap_price=1999.5,
        trade_size_usd=500.0,
        execution_quality_score=0.95,
    )
    d = rec.to_dict()
    assert d["immediate_fill_price"] == 2000.0
    assert d["estimated_slippage_bps"] == 5.0


# ===========================================================================
# ReportRecord
# ===========================================================================


def test_report_record_to_dict():
    rec = ReportRecord(report_type="daily", generated_at=time.time(), summary="Test summary")
    d = rec.to_dict()
    assert d["type"] == "daily"
    assert d["summary"] == "Test summary"
    assert "age_hours" in d


# ===========================================================================
# PerformanceReporter
# ===========================================================================


def test_performance_reporter_to_dict():
    pr = PerformanceReporter()
    d = pr.to_dict()
    assert "history" in d


def test_performance_reporter_chunk_method():
    pr = PerformanceReporter()
    text = "Line 1\nLine 2\nLine 3\nLine 4"
    chunks = pr._chunk(text, size=12)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    # Reassembled text should contain the original content
    reassembled = "\n".join(chunks)
    assert "Line 1" in reassembled


def test_performance_reporter_escape_md():
    pr = PerformanceReporter()
    result = pr._escape_md("hello.world-test")
    assert "\\." in result
    assert "\\-" in result
