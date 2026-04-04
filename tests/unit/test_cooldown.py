"""tests/unit/test_cooldown.py — Unit tests for wallet_manager.utils.cooldown."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from wallet_manager.utils.cooldown import CooldownManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(lam: float = 100.0, sigma: float = 0.0) -> CooldownManager:
    """Return a CooldownManager with fast settings suitable for tests."""
    return CooldownManager(lam=lam, gaussian_sigma=sigma)


# ---------------------------------------------------------------------------
# Per-wallet cooldown tests
# ---------------------------------------------------------------------------


class TestIsInCooldown:
    def test_not_in_cooldown_initially(self):
        """A brand-new wallet_id is never in cooldown."""
        mgr = _make_manager()
        assert mgr.is_in_cooldown("w1") is False

    def test_in_cooldown_after_trigger(self):
        """trigger_cooldown('w1', hours=1.0) → is_in_cooldown('w1') is True."""
        mgr = _make_manager()
        mgr.trigger_cooldown("w1", hours=1.0)
        assert mgr.is_in_cooldown("w1") is True

    def test_cooldown_expires_immediately_at_zero_duration(self):
        """trigger_cooldown with hours=0.0 → cooldown expires at once.

        Using a tiny negative time offset ensures the expiry is in the past.
        """
        mgr = _make_manager()
        # Trigger with zero hours, then back-date the expiry so it is in the past.
        mgr.trigger_cooldown("w1", hours=0.0)
        # Force expiry to be safely in the past
        mgr._cooldowns["w1"] = time.monotonic() - 0.001
        assert mgr.is_in_cooldown("w1") is False

    def test_multiple_wallets_independent(self):
        """Triggering cooldown on w1 must not affect w2."""
        mgr = _make_manager()
        mgr.trigger_cooldown("w1", hours=1.0)
        assert mgr.is_in_cooldown("w1") is True
        assert mgr.is_in_cooldown("w2") is False

    def test_clear_removes_cooldown(self):
        """clear('w1') after trigger → is_in_cooldown('w1') is False."""
        mgr = _make_manager()
        mgr.trigger_cooldown("w1", hours=2.0)
        assert mgr.is_in_cooldown("w1") is True
        mgr.clear("w1")
        assert mgr.is_in_cooldown("w1") is False

    def test_clear_nonexistent_wallet_does_not_raise(self):
        """clear() on an unknown wallet_id must not raise."""
        mgr = _make_manager()
        mgr.clear("nonexistent_wallet")
        assert not mgr.is_in_cooldown("nonexistent_wallet")

    def test_clear_all_removes_all_cooldowns(self):
        """clear_all() removes every active cooldown."""
        mgr = _make_manager()
        mgr.trigger_cooldown("w1", hours=1.0)
        mgr.trigger_cooldown("w2", hours=2.0)
        mgr.clear_all()
        assert mgr.is_in_cooldown("w1") is False
        assert mgr.is_in_cooldown("w2") is False

    def test_trigger_overwrites_previous_cooldown(self):
        """Calling trigger_cooldown twice on the same wallet updates the expiry."""
        mgr = _make_manager()
        mgr.trigger_cooldown("w1", hours=0.0)
        mgr._cooldowns["w1"] = time.monotonic() - 1.0  # expired
        # Re-trigger with a longer duration
        mgr.trigger_cooldown("w1", hours=2.0)
        assert mgr.is_in_cooldown("w1") is True

    def test_expired_entry_lazily_cleaned_up(self):
        """After expiry is_in_cooldown returns False and removes the entry."""
        mgr = _make_manager()
        mgr.trigger_cooldown("w1", hours=1.0)
        # Manually expire
        mgr._cooldowns["w1"] = time.monotonic() - 1.0
        assert mgr.is_in_cooldown("w1") is False
        assert "w1" not in mgr._cooldowns


# ---------------------------------------------------------------------------
# wait() method tests
# ---------------------------------------------------------------------------


class TestWaitMethod:
    @pytest.mark.asyncio
    async def test_wait_does_not_raise(self):
        """CooldownManager.wait() completes without raising (sleep is mocked)."""
        mgr = CooldownManager(lam=100.0, gaussian_sigma=0.0)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await mgr.wait()

        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        assert delay >= 0.0

    @pytest.mark.asyncio
    async def test_wait_calls_asyncio_sleep_once(self):
        """wait() calls asyncio.sleep exactly once per invocation."""
        mgr = CooldownManager(lam=100.0, gaussian_sigma=0.0)

        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await mgr.wait()

        assert len(sleep_calls) == 1

    @pytest.mark.asyncio
    async def test_wait_sleep_duration_is_positive(self):
        """The sleep duration produced by wait() must always be positive."""
        mgr = CooldownManager(lam=10.0, gaussian_sigma=0.0)

        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await mgr.wait()

        assert sleep_calls[0] > 0.0

    def test_invalid_lambda_raises_value_error(self):
        """Constructing CooldownManager when both lam and config default are non-positive raises ValueError.

        lam=0.0 causes a fallback to cfg.poisson_lambda.  To force a ValueError
        we patch get_settings so the config default is also 0.
        """
        from types import SimpleNamespace
        from unittest.mock import patch
        fake_cfg = SimpleNamespace(poisson_lambda=0.0, gaussian_jitter_sigma=0.1, cooldown_min_delay=0.0)
        with patch("wallet_manager.utils.cooldown._cfg", return_value=fake_cfg), pytest.raises(
            ValueError,
            match="Poisson rate",
        ):
            CooldownManager(lam=0.0, gaussian_sigma=0.0)

    def test_invalid_sigma_raises_value_error(self):
        """Constructing CooldownManager with a negative sigma raises ValueError.

        sigma=-1.0 triggers a fallback to cfg.gaussian_jitter_sigma.  We patch
        _cfg so the fallback value is also negative, forcing the ValueError.
        """
        from types import SimpleNamespace
        from unittest.mock import patch
        fake_cfg = SimpleNamespace(poisson_lambda=1.0, gaussian_jitter_sigma=-0.5, cooldown_min_delay=0.0)
        with patch("wallet_manager.utils.cooldown._cfg", return_value=fake_cfg), pytest.raises(
            ValueError,
            match="Gaussian sigma",
        ):
            CooldownManager(lam=1.0, gaussian_sigma=-1.0)
