"""
wallet_manager/resilience/retry.py

``@async_retry`` — a composable exponential back-off decorator for async code.

Responsibility
--------------
This module has exactly one job: wrap an async function so that transient
failures are retried with a randomised delay strategy.  It has **no**
knowledge of wallets, keys, HTTP sessions, or circuit breakers.

Back-off strategy
-----------------
**Full Jitter** (recommended by the AWS Architecture Blog for distributed
systems under load):

    delay = random.uniform(0, base_delay * 2 ** (attempt - 1))

This produces delays that cover the entire range from zero to the exponential
cap, which empirically outperforms "Equal Jitter" and "Decorrelated Jitter"
under realistic load patterns.  Unlike fixed or multiplicative delays, Full
Jitter prevents correlated retries (the thundering-herd problem) without
sacrificing average retry latency.

Usage
-----
Apply the decorator with defaults::

    @async_retry()
    async def call_rpc(...):
        ...

Override defaults for a specific call site::

    @async_retry(max_attempts=6, base_delay=2.0)
    async def call_rpc(...):
        ...
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from functools import wraps
from typing import Any

import aiohttp

from wallet_manager.config import Settings, get_settings
from wallet_manager.exceptions import MaxRetriesExceededError

log = logging.getLogger(__name__)


def _cfg() -> Settings:
    return get_settings()


# Exception types that indicate a transient failure worth retrying
_DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)


def async_retry(
    max_attempts: int = 0,
    base_delay: float = 0.0,
    retryable_exceptions: tuple[type[Exception], ...] = _DEFAULT_RETRYABLE,
) -> Callable[..., Any]:
    """Decorator factory: retry an async function with Full Jitter back-off.

    Wraps the decorated coroutine so that on a retryable exception the call is
    delayed and repeated up to *max_attempts* times.  Non-retryable exceptions
    propagate immediately without waiting.

    Args:
        max_attempts: Total number of attempts (1 = no retry).
        base_delay: Base delay in seconds for the exponential calculation.
            On attempt *k* the maximum possible delay is
            ``base_delay * 2 ** (k - 1)``; the actual delay is a uniform
            sample from ``[0, cap]``.
        retryable_exceptions: Tuple of exception classes that trigger a retry.
            All other exceptions propagate immediately.

    Returns:
        Async decorator that wraps the target coroutine.

    Example:
        .. code-block:: python

            @async_retry(max_attempts=5, base_delay=2.0)
            async def fetch_data(url: str) -> dict:
                async with aiohttp.ClientSession() as s:
                    async with s.get(url) as r:
                        return await r.json()
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            last_exc: Exception | None = None
            cfg = _cfg()
            _max = max_attempts if max_attempts > 0 else cfg.retry_attempts
            _delay = base_delay if base_delay > 0 else cfg.retry_base_delay

            for attempt in range(1, _max + 1):
                try:
                    return await func(*args, **kwargs)

                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == _max:
                        # Exhausted — break to raise below
                        break

                    # Full Jitter: uniform sample over [0, exponential cap]
                    cap: float = _delay * (2 ** (attempt - 1))
                    delay: float = random.uniform(0.0, cap)

                    log.warning(
                        f"{func.__qualname__} | attempt {attempt}/{_max} failed ({type(exc).__name__}: {exc}). "
                        f"Retrying in {delay:.3f}s.",
                    )
                    await asyncio.sleep(delay)

                except Exception:
                    # Non-retryable — propagate immediately
                    raise

            raise MaxRetriesExceededError(
                f"{func.__qualname__}: all {_max} attempt(s) exhausted.",
                attempts=_max,
                original_exception=last_exc,
            ) from last_exc

        return wrapper
    return decorator
