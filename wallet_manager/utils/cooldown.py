"""
wallet_manager/utils/cooldown.py

Traffic regulation via randomised inter-request delays.

Responsibility
--------------
Implement :class:`~wallet_manager.core.interfaces.ICooldown` using a
Poisson-process model with an optional Gaussian jitter overlay.  This
module performs **only** delay arithmetic and scheduling — no network, no
crypto, no wallet logic.

Mathematical model
------------------
In a Poisson process with rate λ (requests per second), the time between
consecutive events follows an Exponential distribution with mean 1/λ.  We
sample from this distribution using the inverse-CDF method::

    base_delay = -ln(U) / λ  ≡  random.expovariate(λ)

A zero-mean Gaussian term *ε ~ N(0, σ²)*, clipped to ±2σ, is added on top::

    total_delay = max(min_delay, base_delay + clip(ε, -2σ, +2σ))

This produces delays that:

* Have a realistic heavy-tailed distribution (Exponential) rather than a
  uniform band, which is harder to fingerprint as synthetic traffic.
* Are never negative (clamped to *min_delay*).
* Exhibit short bursts and occasional longer pauses — closely mimicking
  human interaction cadence.

Choosing λ
----------
λ = 2.0 → mean delay ≈ 0.5 s (active trading cadence)
λ = 0.5 → mean delay ≈ 2.0 s (cautious / monitoring cadence)
"""

from __future__ import annotations

import asyncio
import random

from wallet_manager.config import Settings, get_settings
from wallet_manager.core.interfaces import ICooldown
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


def _cfg() -> Settings:
    return get_settings()


# Ghost Protocol: all tuneable constants are now loaded via _cfg()


class CooldownManager(ICooldown):
    """Concrete traffic regulator implementing :class:`~wallet_manager.core.interfaces.ICooldown`.

    Applies Poisson-distributed delays with bounded Gaussian jitter to make
    outbound traffic patterns resemble human-driven interaction rather than
    scripted polling.

    Args:
        lam: Poisson rate parameter λ (requests / second).  Larger values
            produce shorter mean delays.
        gaussian_sigma: Standard deviation of the Gaussian jitter in seconds.
            Set to ``0`` to disable jitter entirely.

    Example:
        .. code-block:: python

            cooldown = CooldownManager(lam=1.0, gaussian_sigma=0.2)
            await cooldown.wait()   # suspends for a Poisson-sampled duration
    """

    def __init__(
        self,
        lam: float = 0.0,
        gaussian_sigma: float = -1.0,
    ) -> None:
        self._cooldowns: dict[str, float] = {}
        cfg = _cfg()
        lam = lam if lam > 0 else cfg.poisson_lambda
        gaussian_sigma = gaussian_sigma if gaussian_sigma >= 0 else cfg.gaussian_jitter_sigma
        if lam <= 0:
            raise ValueError(f"Poisson rate λ must be positive; got {lam}.")
        if gaussian_sigma < 0:
            raise ValueError(f"Gaussian sigma must be non-negative; got {gaussian_sigma}.")

        self._lam = lam
        self._sigma = gaussian_sigma

    # ------------------------------------------------------------------
    # ICooldown implementation
    # ------------------------------------------------------------------

    async def wait(self) -> None:
        """Suspend execution for a Poisson-sampled delay with Gaussian jitter.

        Sampling procedure:

        1. Draw ``base`` from ``Exponential(λ)``.
        2. Draw ``jitter`` from ``N(0, σ²)``; clip to ``[-2σ, +2σ]``.
        3. Compute ``delay = max(MIN_DELAY, base + jitter)``.
        4. ``await asyncio.sleep(delay)``.

        Returns:
            None — control resumes after *delay* seconds.
        """
        base: float = random.expovariate(self._lam)

        jitter: float = 0.0
        if self._sigma > 0:
            raw_jitter = random.gauss(0.0, self._sigma)
            # Clip jitter to ±3σ for Ghost: higher variance, no identical offsets
            clip_bound = 3.0 * self._sigma
            jitter = max(-clip_bound, min(clip_bound, raw_jitter))

        # Unique micro-offset: ensure no two waits are identical (clock + random)
        micro = random.uniform(0.001, 0.5)
        delay: float = max(_cfg().cooldown_min_delay, base + jitter + micro)
        log.debug(
            f"CooldownManager: sleeping {delay:.3f}s (base={base:.3f}, jitter={jitter:+.3f}).",
        )
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Per-wallet cooldown management
    # ------------------------------------------------------------------

    def trigger_cooldown(self, wallet_id: str, hours: float) -> None:
        """Put *wallet_id* into cooldown for *hours* hours.

        Args:
            wallet_id: Unique wallet identifier string.
            hours: Duration of the cooldown in hours.  Zero or negative
                values expire immediately (wallet is never considered in
                cooldown after the call returns).
        """
        import time
        expiry = time.monotonic() + hours * 3600.0
        self._cooldowns[wallet_id] = expiry
        log.debug("CooldownManager: wallet %s in cooldown for %.2fh (expires %s)", wallet_id, hours, expiry)

    def is_in_cooldown(self, wallet_id: str) -> bool:
        """Return ``True`` if *wallet_id* is currently in cooldown.

        Args:
            wallet_id: Unique wallet identifier string.

        Returns:
            ``True`` if a non-expired cooldown exists for *wallet_id*.
        """
        import time
        expiry = self._cooldowns.get(wallet_id)
        if expiry is None:
            return False
        if time.monotonic() < expiry:
            return True
        # Lazily remove expired entry
        del self._cooldowns[wallet_id]
        return False

    def clear(self, wallet_id: str) -> None:
        """Remove the cooldown record for *wallet_id*, if any.

        Args:
            wallet_id: Unique wallet identifier string.
        """
        self._cooldowns.pop(wallet_id, None)
        log.debug("CooldownManager: cleared cooldown for wallet %s", wallet_id)

    def clear_all(self) -> None:
        """Remove all active cooldown records."""
        self._cooldowns.clear()
        log.debug("CooldownManager: all cooldowns cleared")
