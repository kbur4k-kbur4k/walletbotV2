"""
wallet_manager/resilience/circuit_breaker.py

Circuit Breaker finite-state machine.

Responsibility
--------------
Track failure counts for a single resource and gate access to it when
repeated failures indicate systemic unavailability.  This module contains
**no** network code, no cryptography, and no wallet logic.

State machine
-------------
::

    ┌─────────┐   failure_count >= threshold   ┌──────┐
    │ CLOSED  │ ──────────────────────────────► │ OPEN │
    └─────────┘                                 └──────┘
         ▲                                          │
         │  record_success()                        │  recovery_timeout elapsed
         │                                          ▼
         │                                    ┌───────────┐
         └────────────────────────────────────│ HALF_OPEN │
                                              └───────────┘

* **CLOSED** — normal operation; all requests pass through.
* **OPEN** — all requests are blocked immediately to protect the downstream
  system from a cascade of failing calls.
* **HALF_OPEN** — one probe request is allowed through.  A success closes the
  circuit; a failure re-opens it and resets the recovery timer.

Thread / coroutine safety
-------------------------
``CircuitBreaker`` is **not** thread-safe by design.  In an asyncio context,
coroutines run on a single OS thread and the event loop guarantees that state
transitions are atomic between ``await`` points.  If you need thread safety,
wrap state mutations in ``asyncio.Lock``.
"""

from __future__ import annotations

import time
from enum import Enum, auto

from wallet_manager.config import Settings, get_settings
from wallet_manager.core.interfaces import ICircuitBreaker
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


def _cfg() -> Settings:
    return get_settings()


class CircuitState(Enum):
    """Enumeration of circuit breaker states."""

    CLOSED = auto()     # Normal — requests flow through
    OPEN = auto()       # Tripped — requests blocked
    HALF_OPEN = auto()  # Recovery probe — one request allowed


class CircuitBreaker(ICircuitBreaker):
    """Concrete circuit breaker implementing :class:`~wallet_manager.core.interfaces.ICircuitBreaker`.

    One instance should be created **per wallet** so that a failing RPC
    endpoint for one wallet does not block others.

    Args:
        failure_threshold: Number of consecutive failures before tripping.
        recovery_timeout: Seconds to remain OPEN before probing for recovery.

    Example:
        .. code-block:: python

            breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=15.0)

            if not breaker.allow_request():
                raise RuntimeError("Circuit is OPEN.")
            try:
                result = await risky_call()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 0,
        recovery_timeout: float = 0.0,
    ) -> None:
        cfg = _cfg()
        self._failure_threshold = failure_threshold or cfg.circuit_failure_threshold
        self._recovery_timeout = recovery_timeout or cfg.circuit_recovery_timeout

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0

    # ------------------------------------------------------------------
    # ICircuitBreaker implementation
    # ------------------------------------------------------------------

    def allow_request(self) -> bool:
        """Indicate whether the next request should be attempted.

        Transitions OPEN → HALF_OPEN automatically once *recovery_timeout*
        has elapsed, enabling a recovery probe.

        Returns:
            ``True`` if the request should proceed; ``False`` if blocked.
        """
        if self._state is CircuitState.CLOSED:
            return True

        if self._state is CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
                return True  # Allow the single recovery probe
            log.debug(
                f"Circuit OPEN — {self._recovery_timeout - elapsed:.1f}s remaining before probe.",
            )
            return False

        # HALF_OPEN: allow exactly one probe through
        return True

    def record_success(self) -> None:
        """Reset the circuit to CLOSED and clear the failure counter.

        Should be called immediately after a successful operation completes.
        """
        if self._state is not CircuitState.CLOSED:
            log.info("Circuit recovered — transitioning to CLOSED.")
        self._failure_count = 0
        self._transition(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """Increment the failure counter; trip the circuit at the threshold.

        Should be called immediately after a failed operation.  When
        HALF_OPEN, a single failure immediately re-opens the circuit and
        resets the recovery timer.
        """
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state is CircuitState.HALF_OPEN:
            # Probe failed — return to OPEN and restart the recovery clock
            log.warning("Probe failed in HALF_OPEN state — circuit re-opened.")
            self._transition(CircuitState.OPEN)
            return

        if self._failure_count >= self._failure_threshold:
            log.warning(
                f"Failure threshold ({self._failure_threshold}) reached — circuit OPEN.",
            )
            self._transition(CircuitState.OPEN)

    # ------------------------------------------------------------------
    # Properties (read-only external visibility)
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (read-only).

        Returns:
            The active :class:`CircuitState` enum member.
        """
        return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures recorded since last success.

        Returns:
            Non-negative integer failure counter.
        """
        return self._failure_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: CircuitState) -> None:
        """Change internal state and emit a debug log entry.

        Args:
            new_state: The :class:`CircuitState` to transition to.
        """
        if new_state is not self._state:
            log.debug(
                f"CircuitBreaker: {self._state.name} → {new_state.name}.",
            )
        self._state = new_state
