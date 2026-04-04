"""tests/unit/test_circuit_breaker.py — CircuitBreaker FSM tests."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from wallet_manager.resilience.circuit_breaker import CircuitBreaker, CircuitState


def _breaker(threshold: int = 3, timeout: float = 30.0) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, recovery_timeout=timeout)


def test_initial_state_closed():
    cb = _breaker()
    assert cb.state is CircuitState.CLOSED
    assert cb.allow_request() is True


def test_closed_to_open_on_threshold():
    cb = _breaker(threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_open_blocks_requests():
    cb = _breaker(threshold=1)
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.allow_request() is False


def test_open_to_half_open_after_timeout():
    cb = _breaker(threshold=1, timeout=0.01)
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    time.sleep(0.05)
    result = cb.allow_request()
    assert cb.state is CircuitState.HALF_OPEN
    assert result is True


def test_half_open_to_closed_on_success():
    cb = _breaker(threshold=1, timeout=0.01)
    cb.record_failure()
    time.sleep(0.05)
    cb.allow_request()  # transitions to HALF_OPEN
    assert cb.state is CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_half_open_to_open_on_failure():
    cb = _breaker(threshold=1, timeout=0.01)
    cb.record_failure()
    time.sleep(0.05)
    cb.allow_request()  # transitions to HALF_OPEN
    assert cb.state is CircuitState.HALF_OPEN
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_success_resets_failure_count():
    cb = _breaker(threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.state is CircuitState.CLOSED


def test_does_not_open_below_threshold():
    cb = _breaker(threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert cb.state is CircuitState.CLOSED
