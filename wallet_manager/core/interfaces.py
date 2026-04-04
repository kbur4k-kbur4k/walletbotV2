"""
wallet_manager/core/interfaces.py

Abstract Base Classes (ABCs) that define the contracts between every layer.

``WalletManager`` depends **only** on these interfaces, never on concrete
implementations.  This satisfies the Dependency Inversion Principle: high-
level policy (the manager) is decoupled from low-level details (HTTP, crypto).

Adding a new encryption backend, a different HTTP library, or a custom delay
strategy requires only a new class that satisfies the relevant ABC — the
manager's source code is never touched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from wallet_manager.core.models import TxParams, TxReceipt

if TYPE_CHECKING:  # pragma: no cover
    from wallet_manager.core.models import ChainConfig, NotificationEvent


# ---------------------------------------------------------------------------
# Security interfaces
# ---------------------------------------------------------------------------


class IVault(ABC):
    """Contract for an AES-256 (or equivalent) key vault.

    Concrete implementations must guarantee that plaintext private keys are
    never persisted and are held in memory only for the minimum duration
    required to perform a signing operation.
    """

    @abstractmethod
    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt a plaintext secret and return an opaque ciphertext token.

        Args:
            plaintext: Sensitive string (e.g. raw private key hex).

        Returns:
            Encrypted bytes that can be safely persisted.
        """

    @abstractmethod
    def decrypt(self, token: bytes) -> str:
        """Decrypt a ciphertext token and return the original plaintext.

        Callers **must not** cache the returned value.

        Args:
            token: Ciphertext bytes previously produced by :meth:`encrypt`.

        Returns:
            Decrypted plaintext string.
        """


# ---------------------------------------------------------------------------
# Traffic regulation interface
# ---------------------------------------------------------------------------


class ICooldown(ABC):
    """Contract for inter-request delay strategies.

    Implementations may use fixed delays, Poisson processes, token buckets,
    or any other strategy.  ``WalletManager`` calls :meth:`wait` before every
    outbound network operation.
    """

    @abstractmethod
    async def wait(self) -> None:
        """Suspend the current coroutine for a strategy-defined duration.

        Returns:
            None — control resumes after the delay has elapsed.
        """


# ---------------------------------------------------------------------------
# Network operation interfaces (Interface Segregation Principle)
# ---------------------------------------------------------------------------


class IBalanceProvider(ABC):
    """Narrow interface for on-chain balance queries.

    Kept separate from :class:`ITxBroadcaster` so that read-only consumers
    do not depend on transaction-signing capabilities.
    """

    @abstractmethod
    async def get_balance(self, rpc_url: str, address: str) -> int:
        """Return the native-currency balance of *address* in wei.

        Args:
            rpc_url: JSON-RPC endpoint to query.
            address: Checksummed Ethereum address.

        Returns:
            Balance in wei.
        """


class ITxBroadcaster(ABC):
    """Narrow interface for transaction signing and broadcasting.

    Separated from :class:`IBalanceProvider` to honour the Interface
    Segregation Principle — components that only read state do not pull in
    the signing dependency.
    """

    @abstractmethod
    async def get_nonce(self, rpc_url: str, address: str) -> int:
        """Fetch the pending nonce for *address*.

        Args:
            rpc_url: JSON-RPC endpoint to query.
            address: Sender checksummed address.

        Returns:
            Next available nonce integer.
        """

    @abstractmethod
    async def sign_and_send(
        self,
        rpc_url: str,
        tx_params: TxParams,
        plaintext_private_key: str,
    ) -> TxReceipt:
        """Sign *tx_params* with *plaintext_private_key* and broadcast.

        The ``plaintext_private_key`` argument must be treated as ephemeral:
        implementations must not persist or log it.

        Args:
            rpc_url: JSON-RPC endpoint to broadcast through.
            tx_params: Fully populated transaction parameters.
            plaintext_private_key: Hex-encoded private key (no ``0x`` prefix
                required; implementations normalise internally).

        Returns:
            :class:`~wallet_manager.core.models.TxReceipt` with the tx hash.
        """


# ---------------------------------------------------------------------------
# Notification interface
# ---------------------------------------------------------------------------


class INotifier(ABC):
    """Contract for async event notification channels (Telegram, Slack, etc.).

    ``WalletOrchestrator`` depends only on this interface; the concrete
    ``TelegramNotifier`` is injected at construction time.  Keeping the
    interface narrow (three lifecycle methods) means the orchestrator never
    needs to know *how* a notification is delivered.
    """

    @abstractmethod
    async def send(self, event: NotificationEvent) -> None:
        """Enqueue *event* for delivery to the configured channel.

        Must be non-blocking from the caller's perspective — implementations
        use an internal queue so the calling coroutine is never delayed by
        outbound network I/O.

        Args:
            event: Typed notification event to deliver.
        """

    @abstractmethod
    async def start(self) -> None:
        """Start the background delivery worker.

        Call once before any :meth:`send` invocations — typically inside an
        ``async with`` block or at application boot.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Drain the queue and shut down the background worker gracefully.

        Await during application shutdown to guarantee every enqueued event
        is delivered before the process exits.
        """


# ---------------------------------------------------------------------------
# Resilience interfaces
# ---------------------------------------------------------------------------


class ICircuitBreaker(ABC):
    """Contract for a circuit breaker protecting a single resource.

    Implementations track failure counts and control whether new calls are
    permitted, following the CLOSED → OPEN → HALF_OPEN finite-state machine.
    """

    @abstractmethod
    def allow_request(self) -> bool:
        """Indicate whether the next request should be attempted.

        Returns:
            ``True`` if the circuit is CLOSED or HALF_OPEN; ``False`` if OPEN.
        """

    @abstractmethod
    def record_success(self) -> None:
        """Signal that the most recent operation completed successfully."""

    @abstractmethod
    def record_failure(self) -> None:
        """Signal that the most recent operation failed."""


# ---------------------------------------------------------------------------
# Nonce management interface
# ---------------------------------------------------------------------------


class INonceManager(ABC):
    """Contract for a per-wallet async nonce tracker.

    Implementations maintain an in-memory nonce cache that increments
    optimistically and re-syncs with the chain on conflicts.
    """

    @abstractmethod
    async def get_and_increment(self, wallet_id: str, rpc_url: str, address: str) -> int:
        """Return the next usable nonce and increment the internal counter.

        Args:
            wallet_id: Wallet identifier (used as cache key).
            rpc_url: RPC endpoint for chain sync when cache is cold.
            address: Sender address for on-chain nonce lookup.

        Returns:
            Integer nonce to stamp on the next transaction.
        """

    @abstractmethod
    async def rewind(self, wallet_id: str) -> None:
        """Rewind the cached nonce by one after a failed broadcast.

        Call this when a signed transaction is never broadcast so the nonce
        slot is not permanently skipped.

        Args:
            wallet_id: Wallet whose counter should be decremented.
        """

    @abstractmethod
    async def invalidate(self, wallet_id: str) -> None:
        """Force a full re-sync with the chain on the next call.

        Call this after a ``nonce too low`` error to realign the cache.

        Args:
            wallet_id: Wallet whose cache entry should be cleared.
        """


# ---------------------------------------------------------------------------
# Multi-chain router interface
# ---------------------------------------------------------------------------


class IRouter(ABC):
    """Contract for a multi-chain RPC provider router.

    Implementations manage a pool of RPC URLs per chain and transparently
    fail over to backup providers when the active one is unavailable.
    """

    @abstractmethod
    def get_rpc(self, chain_id: int) -> str:
        """Return the currently active RPC URL for *chain_id*.

        Args:
            chain_id: EVM chain identifier.

        Returns:
            Active RPC URL string.

        Raises:
            NoAvailableRpcError: If all providers for the chain are exhausted.
        """

    @abstractmethod
    def mark_failed(self, chain_id: int, rpc_url: str) -> None:
        """Record a failure for *rpc_url* and rotate to the next provider.

        Args:
            chain_id: EVM chain identifier.
            rpc_url: The URL that produced the failure.
        """

    @abstractmethod
    def get_chain_config(self, chain_id: int) -> ChainConfig:
        """Return the full :class:`~wallet_manager.core.models.ChainConfig`
        for *chain_id*.

        Args:
            chain_id: EVM chain identifier.

        Raises:
            KeyError: If *chain_id* is not registered.
        """


# ---------------------------------------------------------------------------
# Execution strategy interface
# ---------------------------------------------------------------------------


class IStrategy(ABC):
    """Contract for execution variance calculators.

    Separating strategy from the orchestrator lets the variance logic be
    unit-tested independently and swapped without touching orchestration code.
    """

    @abstractmethod
    def calculate_amount(
        self,
        balance_wei: int,
        min_pct: float,
        max_pct: float,
    ) -> int:
        """Compute a transaction amount as a jittered percentage of *balance_wei*.

        Args:
            balance_wei: Current wallet balance in wei.
            min_pct: Minimum fraction to use (0.0–1.0).
            max_pct: Maximum fraction to use (0.0–1.0).

        Returns:
            Amount in wei, clamped to [min_pct, max_pct] of balance.
        """

    @abstractmethod
    def jittered_delay(self, base_seconds: float) -> float:
        """Return a Gaussian-jittered version of *base_seconds*.

        Args:
            base_seconds: Nominal delay in seconds.

        Returns:
            Adjusted delay, always non-negative.
        """

    @abstractmethod
    def shuffle_tasks(self, tasks: list[Any]) -> list[Any]:
        """Return a shuffled copy of *tasks* without modifying the original.

        Args:
            tasks: List of task descriptors in any order.

        Returns:
            New list with the same elements in a randomised order.
        """
