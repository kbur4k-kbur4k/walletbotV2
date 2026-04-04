"""Token bucket rate limiter for Telegram and RPC calls."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Thread-safe async token bucket rate limiter.

    Args:
        rate: Tokens replenished per second.
        capacity: Maximum token capacity (burst size).
    """

    rate: float
    capacity: float
    _tokens: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last = time.monotonic()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire *tokens* from the bucket, sleeping if necessary.

        Args:
            tokens: Number of tokens to consume (default 1.0).
        """
        while True:
            now = time.monotonic()
            self._tokens = min(
                self.capacity,
                self._tokens + (now - self._last) * self.rate,
            )
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            await asyncio.sleep((tokens - self._tokens) / self.rate)
