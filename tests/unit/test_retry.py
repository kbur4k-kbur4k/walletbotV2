"""tests/unit/test_retry.py — async_retry decorator tests."""
from __future__ import annotations

import asyncio

import pytest

from wallet_manager.exceptions import MaxRetriesExceededError
from wallet_manager.resilience.retry import async_retry


@pytest.mark.asyncio
async def test_succeeds_on_first_attempt():
    call_count = 0

    @async_retry(max_attempts=3, base_delay=0.0)
    async def fn():
        nonlocal call_count
        call_count += 1
        return 42

    result = await fn()
    assert result == 42
    assert call_count == 1


@pytest.mark.asyncio
async def test_retries_on_transient_error():
    call_count = 0

    @async_retry(max_attempts=3, base_delay=0.0)
    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await fn()
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_raises_after_max_attempts():
    call_count = 0

    @async_retry(max_attempts=3, base_delay=0.0)
    async def fn():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("always fails")

    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await fn()
    assert call_count == 3
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.original_exception, ConnectionError)


@pytest.mark.asyncio
async def test_non_retryable_propagates_immediately():
    call_count = 0

    @async_retry(max_attempts=5, base_delay=0.0)
    async def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        await fn()
    assert call_count == 1


@pytest.mark.asyncio
async def test_timeout_error_is_retried():
    call_count = 0

    @async_retry(max_attempts=3, base_delay=0.0)
    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise TimeoutError()
        return "done"

    result = await fn()
    assert result == "done"
    assert call_count == 2
