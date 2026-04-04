"""
Tests for FeatureDriftMonitor.

Covers: instantiation, update(), report structure, accumulation,
reset/window behaviour, and correct detection of obvious distribution shift.
"""
import numpy as np
import pytest

from wallet_manager.strategies.robust_alpha import FeatureDriftMonitor

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def monitor() -> FeatureDriftMonitor:
    return FeatureDriftMonitor()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_instantiation(monitor):
    assert monitor is not None


def test_update_does_not_raise(monitor, sample_feature_vector, feature_names):
    """update() accepts a 30-dim vector without raising."""
    monitor.update(sample_feature_vector, feature_names)
    assert monitor._update_count >= 1


def test_get_drift_report_returns_required_keys(monitor, feature_names):
    """get_drift_report() always returns dict with the three required keys."""
    for _ in range(5):
        monitor.update(np.random.randn(30), feature_names)
    report = monitor.get_drift_report()
    assert isinstance(report, dict)
    for key in ("drifted_features", "worst_feature", "worst_kl"):
        assert key in report, f"Missing key: {key}"


def test_drifted_features_is_list(monitor, feature_names):
    """drifted_features is always a list, even before enough data."""
    monitor.update(np.zeros(30), feature_names)
    report = monitor.get_drift_report()
    assert isinstance(report["drifted_features"], list)


def test_worst_kl_non_negative_after_accumulation(monitor, feature_names):
    """worst_kl is a finite non-negative float after many updates."""
    for _ in range(50):
        monitor.update(np.random.randn(30), feature_names)
    report = monitor.get_drift_report()
    kl = report["worst_kl"]
    assert isinstance(kl, float)
    assert kl >= 0.0
    assert np.isfinite(kl)


def test_singleton_returns_same_instance():
    """get_feature_drift_monitor() always returns the same object."""
    from wallet_manager.strategies.robust_alpha import get_feature_drift_monitor
    a = get_feature_drift_monitor()
    b = get_feature_drift_monitor()
    assert a is b


def test_extreme_distribution_shift_is_detected(feature_names):
    """
    Features trained on N(0,1) and then fed N(100,1) must produce
    a non-zero worst_kl — the monitor must detect the obvious shift.
    """
    from wallet_manager.strategies.robust_alpha import FeatureDriftMonitor
    mon = FeatureDriftMonitor()
    rng = np.random.default_rng(seed=7)

    # Reference window: N(0, 1)
    for _ in range(300):
        mon.update(rng.standard_normal(30), feature_names)

    # Live window: N(100, 1) — extreme, unambiguous drift
    for _ in range(100):
        mon.update(rng.standard_normal(30) + 100.0, feature_names)

    report = mon.get_drift_report()
    # A competent KL estimator MUST flag this; 0.0 would be a bug.
    assert report["worst_kl"] > 0.0, (
        "worst_kl is 0.0 after feeding N(100,1) to an N(0,1)-trained monitor. "
        "KL divergence detection is broken."
    )
    # At least one feature should appear in the drifted list
    assert len(report["drifted_features"]) > 0
