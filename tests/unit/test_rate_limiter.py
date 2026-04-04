"""tests/unit/test_rate_limiter.py — Unit tests for wallet_manager.utils.rate_limiter."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from wallet_manager.utils.rate_limiter import TokenBucket


class TestTokenBucketWithinCapacity:
    @pytest.mark.asyncio
    async def test_acquire_within_capacity_immediate(self):
        """rate=10, capacity=10 — acquiring 10 tokens one-by-one should complete fast.

        No sleep should be needed because the bucket starts full.
        """
        bucket = TokenBucket(rate=10.0, capacity=10.0)

        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            for _ in range(10):
                await bucket.acquire()

        # With capacity=10 and 10 acquires of 1 token each, no sleep needed.
        assert len(sleep_calls) == 0

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens_within_capacity(self):
        """acquire(tokens=3.0) with capacity=10 and a full bucket succeeds immediately."""
        bucket = TokenBucket(rate=5.0, capacity=10.0)

        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await bucket.acquire(tokens=3.0)

        assert len(sleep_calls) == 0


class TestTokenBucketExceedsCapacity:
    @pytest.mark.asyncio
    async def test_acquire_beyond_capacity_sleeps(self):
        """rate=2, capacity=1 — acquiring twice should require at least one sleep."""
        bucket = TokenBucket(rate=2.0, capacity=1.0)

        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            # Simulate passage of time so the second acquire can proceed
            sleep_calls.append(duration)
            # Refill the bucket as if time has passed for the slept duration
            bucket._tokens = min(bucket.capacity, bucket._tokens + duration * bucket.rate)
            bucket._last = time.monotonic()

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await bucket.acquire()  # consumes the 1 available token
            await bucket.acquire()  # bucket empty → should sleep

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] > 0.0

    @pytest.mark.asyncio
    async def test_sleep_duration_is_proportional_to_rate(self):
        """Slow rate means proportionally longer sleep when bucket is empty."""
        bucket_fast = TokenBucket(rate=10.0, capacity=1.0)
        bucket_slow = TokenBucket(rate=1.0, capacity=1.0)

        fast_sleeps: list[float] = []
        slow_sleeps: list[float] = []

        async def fake_sleep_fast(duration: float) -> None:
            fast_sleeps.append(duration)
            bucket_fast._tokens = bucket_fast.capacity
            bucket_fast._last = time.monotonic()

        async def fake_sleep_slow(duration: float) -> None:
            slow_sleeps.append(duration)
            bucket_slow._tokens = bucket_slow.capacity
            bucket_slow._last = time.monotonic()

        with patch("asyncio.sleep", side_effect=fake_sleep_fast):
            await bucket_fast.acquire()
            await bucket_fast.acquire()

        with patch("asyncio.sleep", side_effect=fake_sleep_slow):
            await bucket_slow.acquire()
            await bucket_slow.acquire()

        # Slow bucket should require a longer sleep than fast bucket
        assert slow_sleeps[0] > fast_sleeps[0]


class TestTokenBucketCapacityLimit:
    @pytest.mark.asyncio
    async def test_capacity_is_never_exceeded(self):
        """After a very long simulated wait, token count must not exceed capacity."""
        capacity = 5.0
        bucket = TokenBucket(rate=10.0, capacity=capacity)

        # Simulate that a huge amount of time has passed by backdating _last
        bucket._last = time.monotonic() - 1000.0  # 1000 seconds ago

        # Calling acquire will refill, but must be clamped to capacity
        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await bucket.acquire(tokens=1.0)

        # No sleep should have occurred (bucket was overflowing before the cap)
        assert len(sleep_calls) == 0
        # Remaining tokens must not exceed capacity
        assert bucket._tokens <= capacity

    def test_initial_tokens_equal_capacity(self):
        """Newly constructed bucket starts with tokens == capacity."""
        bucket = TokenBucket(rate=3.0, capacity=7.0)
        assert bucket._tokens == 7.0

    @pytest.mark.asyncio
    async def test_tokens_decrease_after_acquire_sync(self):
        """Token count decreases by the requested amount after a successful acquire."""
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        await bucket.acquire(tokens=4.0)
        # Allow for slight floating-point drift from tiny elapsed time
        assert bucket._tokens <= 6.0 + 0.1

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens_triggers_sleep_when_insufficient(self):
        """Requesting tokens=5.0 from a bucket with only 2.0 tokens must sleep."""
        bucket = TokenBucket(rate=1.0, capacity=10.0)
        bucket._tokens = 2.0  # Force low token count

        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)
            # Replenish so the loop can exit
            bucket._tokens = 10.0
            bucket._last = time.monotonic()

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await bucket.acquire(tokens=5.0)

        assert len(sleep_calls) >= 1
