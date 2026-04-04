"""
wallet_manager/core/manager.py

WalletManager — single public orchestration entry point.

Responsibility: wire dependencies, coordinate lifecycle, delegate every
operation to a specialist. Zero crypto, zero HTTP, zero retry loops here.
"""

from __future__ import annotations

import asyncio

import aiohttp

from wallet_manager.config import Settings, get_settings
from wallet_manager.core.interfaces import (
    IBalanceProvider,
    ICircuitBreaker,
    ICooldown,
    ITxBroadcaster,
)
from wallet_manager.core.models import TxParams, TxReceipt, WalletConfig
from wallet_manager.exceptions import (
    CircuitOpenError,
    GasPriceTooHighError,
    WalletNotFoundError,
)
from wallet_manager.network.session_factory import SessionFactory
from wallet_manager.resilience.circuit_breaker import CircuitBreaker
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


# ---------------------------------------------------------------------------
# Internal per-wallet state container
# ---------------------------------------------------------------------------

class _WalletContext:
    """Groups all mutable runtime state for one wallet.

    Attributes:
        config: Immutable wallet configuration record.
        session: Dedicated, isolated aiohttp session.
        breaker: Independent circuit breaker instance.
        nonce_lock: Per-wallet lock that serialises concurrent nonce fetches.
    """

    __slots__ = ("breaker", "config", "nonce_lock", "session")

    def __init__(
        self,
        config: WalletConfig,
        session: aiohttp.ClientSession,
        breaker: ICircuitBreaker,
    ) -> None:
        self.config = config
        self.session = session
        self.breaker = breaker
        self.nonce_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# WalletManager
# ---------------------------------------------------------------------------

class WalletManager:
    """Centralised async manager for a fleet of cryptocurrency wallets.

    Uses plaintext private keys from wallets.json (vault disabled).
    All dependencies are injected; production defaults are resolved lazily
    from the process-wide :class:`~wallet_manager.config.Settings` singleton.

    Args:
        wallet_configs: List of :class:`~wallet_manager.core.models.WalletConfig`.
        settings: Optional settings override (useful in tests).
        cooldown: Traffic regulator (default: :class:`~wallet_manager.utils.cooldown.CooldownManager`).
        balance_provider: Balance query service (default: ``Web3BalanceProvider``).
        tx_broadcaster: Signing & broadcast service (default: ``Web3TxBroadcaster``).

    Example:
        .. code-block:: python

            async with WalletManager(configs) as mgr:
                balance = await mgr.get_safe_balance("hot_wallet_01")
                receipt = await mgr.sign_and_broadcast_tx("hot_wallet_01", tx)
    """

    def __init__(
        self,
        wallet_configs: list[WalletConfig],
        *,
        settings: Settings | None = None,
        cooldown: ICooldown | None = None,
        balance_provider: IBalanceProvider | None = None,
        tx_broadcaster: ITxBroadcaster | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()

        if cooldown is None:
            from wallet_manager.utils.cooldown import CooldownManager
            cooldown = CooldownManager(
                lam=self._settings.poisson_lambda,
                gaussian_sigma=self._settings.gaussian_jitter_sigma,
            )
        if balance_provider is None:
            from wallet_manager.network.balance_provider import Web3BalanceProvider
            balance_provider = Web3BalanceProvider()
        if tx_broadcaster is None:
            from wallet_manager.network.tx_broadcaster import Web3TxBroadcaster
            tx_broadcaster = Web3TxBroadcaster()

        self._cooldown: ICooldown = cooldown
        self._balance_provider: IBalanceProvider = balance_provider
        self._tx_broadcaster: ITxBroadcaster = tx_broadcaster
        self._configs: list[WalletConfig] = wallet_configs
        self._contexts: dict[str, _WalletContext] = {}

        log.debug(f"WalletManager initialised with {len(wallet_configs)} wallet config(s).")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> WalletManager:
        """Open one isolated aiohttp session per configured wallet."""
        for cfg in self._configs:
            session = SessionFactory.build(
                wallet_id=cfg.wallet_id,
                total_timeout=self._settings.session_total_timeout,
                connect_timeout=self._settings.session_connect_timeout,
            )
            self._contexts[cfg.wallet_id] = _WalletContext(
                config=cfg,
                session=session,
                breaker=CircuitBreaker(
                    failure_threshold=self._settings.circuit_failure_threshold,
                    recovery_timeout=self._settings.circuit_recovery_timeout,
                ),
            )
        log.debug(f"{len(self._contexts)} wallet session(s) opened.")
        return self

    async def __aexit__(self, *_: object) -> None:
        """Gracefully close every aiohttp session and clear internal state."""
        close_tasks = [ctx.session.close() for ctx in self._contexts.values()]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        self._contexts.clear()
        log.debug("All wallet sessions closed.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self, wallet_id: str) -> WalletConfig:
        """Return the configuration for the given wallet.

        Args:
            wallet_id: Identifier matching a loaded :class:`WalletConfig`.

        Returns:
            The wallet's configuration record.

        Raises:
            WalletNotFoundError: If *wallet_id* is unknown.
        """
        return self._resolve_context(wallet_id).config

    async def get_safe_balance(self, wallet_id: str) -> int:
        """Return the on-chain native-currency balance for *wallet_id* in wei.

        Args:
            wallet_id: Identifier matching a loaded :class:`WalletConfig`.

        Returns:
            Balance in wei as a non-negative integer.

        Raises:
            WalletNotFoundError: If *wallet_id* is unknown.
            CircuitOpenError: If the circuit breaker is blocking requests.
            BalanceFetchError: If all retry attempts fail.
        """
        ctx = self._resolve_context(wallet_id)
        self._assert_circuit_closed(ctx)
        await self._cooldown.wait()

        address = self._derive_address(ctx)
        try:
            balance: int = await self._balance_provider.get_balance(
                ctx.config.rpc_url, address
            )
            ctx.breaker.record_success()
            log.info(f"Balance fetched for wallet '{wallet_id}'.")
            return balance
        except Exception as exc:
            ctx.breaker.record_failure()
            log.error(f"Balance fetch failed for wallet '{wallet_id}': {exc}")
            raise

    async def sign_and_broadcast_tx(
        self,
        wallet_id: str,
        tx_params: TxParams,
    ) -> TxReceipt:
        """Validate, sign, and broadcast a transaction for *wallet_id*.

        Workflow: gas check → cooldown → nonce lock → nonce fetch →
        sign & send → receipt.

        Args:
            wallet_id: Identifier of the signing wallet.
            tx_params: Transaction parameters (nonce is overwritten by on-chain value).

        Returns:
            :class:`~wallet_manager.core.models.TxReceipt` with tx hash,
            wallet ID, and nonce consumed.

        Raises:
            WalletNotFoundError: If *wallet_id* is unknown.
            CircuitOpenError: If the circuit breaker is blocking requests.
            GasPriceTooHighError: If gas price exceeds the configured maximum.
            NonceFetchError: If the nonce cannot be retrieved after retries.
            BroadcastError: If broadcast fails after retries.
            RuntimeError: If called in paper trade mode.
        """
        _exec_mode = self._settings.execution_mode
        if _exec_mode == "robust_paper" or self._settings.paper_trade:
            raise RuntimeError(
                f"BLOCKED: sign_and_broadcast_tx called for wallet '{wallet_id}' "
                f"while EXECUTION_MODE={_exec_mode!r} (paper trade). "
                f"Real transactions are NEVER allowed in paper trade mode."
            )
        ctx = self._resolve_context(wallet_id)
        self._assert_circuit_closed(ctx)

        # Early, cheap gas price rejection
        if tx_params.gas_price_gwei > ctx.config.max_gas_price_gwei:
            raise GasPriceTooHighError(
                f"Gas price {tx_params.gas_price_gwei:.2f} Gwei exceeds the "
                f"{ctx.config.max_gas_price_gwei:.2f} Gwei ceiling.",
                wallet_id=wallet_id,
                requested_gwei=tx_params.gas_price_gwei,
                max_gwei=ctx.config.max_gas_price_gwei,
            )

        await self._cooldown.wait()

        async with ctx.nonce_lock:
            # Nonce fetch + sign happen inside the lock; concurrent callers
            # cannot race on the same nonce value.
            address = self._derive_address(ctx)
            nonce = await self._tx_broadcaster.get_nonce(ctx.config.rpc_url, address)

            # Inject resolved nonce into a new frozen TxParams
            tx_with_nonce = TxParams(
                from_address=tx_params.from_address,
                to=tx_params.to,
                value=tx_params.value,
                gas=tx_params.gas,
                gas_price_wei=tx_params.gas_price_wei,
                data=tx_params.data,
                nonce=nonce,
            )

            # ── Sign & broadcast (plaintext key from config) ───────────────
            receipt = await self._tx_broadcaster.sign_and_send(
                ctx.config.rpc_url, tx_with_nonce, ctx.config.private_key
            )

        # Stamp wallet_id onto the receipt (broadcaster is wallet-agnostic)
        stamped = TxReceipt(
            tx_hash=receipt.tx_hash,
            wallet_id=wallet_id,
            nonce_used=receipt.nonce_used,
        )

        ctx.breaker.record_success()
        log.info(
            f"TX broadcast for wallet '{wallet_id}' | hash: {stamped.tx_hash} | nonce: {stamped.nonce_used}.",
        )
        return stamped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_context(self, wallet_id: str) -> _WalletContext:
        """Return the runtime context for *wallet_id*.

        Raises:
            WalletNotFoundError: If *wallet_id* is not registered.
        """
        ctx = self._contexts.get(wallet_id)
        if ctx is None:
            raise WalletNotFoundError(
                f"No session found for wallet '{wallet_id}'. "
                f"Registered wallets: {list(self._contexts.keys())}. "
                "Ensure the manager is used inside an 'async with' block.",
                wallet_id=wallet_id,
            )
        return ctx

    def _derive_address(self, ctx: _WalletContext) -> str:
        """Return the Ethereum address for the wallet.

        Returns the pre-computed address if available; otherwise derives
        from the plaintext private key.
        """
        if ctx.config.address:
            return ctx.config.address

        from eth_account import Account

        return str(Account.from_key(ctx.config.private_key).address)

    @staticmethod
    def _assert_circuit_closed(ctx: _WalletContext) -> None:
        """Raise ``CircuitOpenError`` if the wallet's circuit is blocking requests.

        Raises:
            CircuitOpenError: If :meth:`ICircuitBreaker.allow_request` returns False.
        """
        if not ctx.breaker.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker is OPEN for wallet '{ctx.config.wallet_id}'. "
                "Wait for the recovery timeout before retrying.",
                wallet_id=ctx.config.wallet_id,
            )
