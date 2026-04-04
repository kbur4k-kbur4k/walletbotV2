"""
wallet_manager/orchestrator_gas.py

GasGuard — gas monitoring, per-wallet gas gating, and gas event notifications.

Extracted from orchestrator.py (Task 4.1).  The GasGuard class owns the
``_wait_for_safe_gas`` polling loop and all GAS_PAUSE / GAS_RESUME /
GAS_DEFERRED notification logic so that WalletOrchestrator can stay focused
on coordination rather than gas mechanics.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from wallet_manager.core.interfaces import INotifier
from wallet_manager.core.models import (
    EventKind,
    NotificationEvent,
    WalletConfig,
)
from wallet_manager.exceptions import GasPriceTooHighError
from wallet_manager.network.gas_monitor import GasMonitor
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


class GasGuard:
    """Encapsulates gas-price monitoring and wallet suspension/resumption.

    Used by :class:`~wallet_manager.orchestrator.WalletOrchestrator` to gate
    each wallet coroutine until the current gas price drops below the wallet's
    configured ceiling.

    Args:
        gas_monitor: A :class:`~wallet_manager.network.gas_monitor.GasMonitor`
            instance (typically constructed from the first wallet's RPC URL).
        gas_poll_interval: Seconds between re-checks while gas is elevated.
        gas_wait_timeout: Maximum seconds to wait before raising
            :exc:`~wallet_manager.exceptions.GasPriceTooHighError`.

    Example::

        guard = GasGuard(
            gas_monitor=GasMonitor(rpc_url="https://mainnet.base.org"),
            gas_poll_interval=15.0,
            gas_wait_timeout=600.0,
        )
        await guard.wait_for_safe_gas(wallet_config, notifier, safe_notify)
    """

    def __init__(
        self,
        gas_monitor: GasMonitor,
        gas_poll_interval: float,
        gas_wait_timeout: float,
    ) -> None:
        self._gas_monitor = gas_monitor
        self._gas_poll_interval = gas_poll_interval
        self._gas_wait_timeout = gas_wait_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def wait_for_safe_gas(
        self,
        config: WalletConfig,
        notifier: INotifier,
        safe_notify: Callable[[INotifier, NotificationEvent], Awaitable[None]],
    ) -> None:
        """Suspend the calling coroutine until gas is within the wallet's limit.

        Sends a ``GAS_PAUSE`` notification on first suspension and a
        ``GAS_RESUME`` notification when execution proceeds.

        Args:
            config: Wallet config carrying the ``max_gas_price_gwei`` ceiling.
            notifier: Active notification channel.
            safe_notify: Async callable ``(notifier, event) → None`` that
                swallows errors (e.g. ``WalletOrchestrator._safe_notify``).

        Raises:
            GasPriceTooHighError: If gas has not normalised within
                ``gas_wait_timeout`` seconds.
        """
        try:
            safe, current = await self._gas_monitor.is_safe(float(config.max_gas_price_gwei))
        except Exception as exc:
            log.debug("Gas monitor unavailable (%s); proceeding as safe.", exc)
            return  # default to safe when monitor is unreachable

        if safe:
            return

        # First check failed: enter polling loop
        log.info(
            "Gas too high for wallet '%s': %.2f Gwei (max %.2f). Pausing.",
            config.wallet_id,
            current,
            config.max_gas_price_gwei,
        )
        await safe_notify(
            notifier,
            NotificationEvent(
                kind=EventKind.GAS_PAUSE,
                wallet_id=config.wallet_id,
                gas_price_gwei=float(current),
                detail=(
                    f"Gas {float(current):.2f} Gwei exceeds ceiling "
                    f"{float(config.max_gas_price_gwei):.2f} Gwei."
                ),
            ),
        )

        deadline = time.monotonic() + self._gas_wait_timeout

        while time.monotonic() < deadline:
            await asyncio.sleep(self._gas_poll_interval)
            safe, current = await self._gas_monitor.is_safe(float(config.max_gas_price_gwei))

            if safe:
                log.info(
                    "Gas normalised for wallet '%s': %.2f Gwei. Resuming.",
                    config.wallet_id,
                    current,
                )
                await safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.GAS_RESUME,
                        wallet_id=config.wallet_id,
                        gas_price_gwei=float(current),
                    ),
                )
                return

            log.debug(
                "Gas still high for wallet '%s': %.2f Gwei. Retrying in %.0fs.",
                config.wallet_id,
                current,
                self._gas_poll_interval,
            )

        # Timeout exceeded
        raise GasPriceTooHighError(
            f"Gas wait timeout ({self._gas_wait_timeout:.0f}s) exceeded "
            f"for wallet '{config.wallet_id}'.",
            wallet_id=config.wallet_id,
            requested_gwei=float(current),
            max_gwei=float(config.max_gas_price_gwei),
        )

    async def current_gwei(self) -> float:
        """Return the current gas price in Gwei (0.0 on error)."""
        try:
            return await self._gas_monitor.current_gwei()
        except Exception:
            return 0.0
