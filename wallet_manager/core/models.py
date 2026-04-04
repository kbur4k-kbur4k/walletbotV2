"""
wallet_manager/core/models.py

Immutable value objects shared across the entire package.

All dataclasses are frozen (hashable, safe to use as dict keys or set members)
and carry no business logic — they are pure data carriers following the
Single Responsibility Principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# ---------------------------------------------------------------------------
# Wallet configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalletConfig:
    """Immutable configuration record for a single wallet.

    This object is constructed once at startup and never mutated.
    With vault disabled, ``private_key`` holds the plaintext hex private key
    (with or without 0x prefix) read directly from wallets.json.

    Attributes:
        wallet_id: Unique human-readable identifier (e.g. ``"hot_wallet_01"``).
        private_key: Plaintext hex private key (0x-prefixed or not).
        rpc_url: JSON-RPC HTTP endpoint for the target EVM chain.
        chain_id: EVM chain identifier (1 = Ethereum mainnet, 137 = Polygon…).
        max_gas_price_gwei: Hard ceiling on gas price; transactions above this
            value are rejected before signing.
        address: Optional pre-computed checksummed Ethereum address.  When
            supplied, address lookups skip the derive cycle.
    """

    wallet_id: str
    private_key: str
    rpc_url: str
    chain_id: int
    max_gas_price_gwei: float = 100.0
    address: str | None = None


# ---------------------------------------------------------------------------
# Transaction primitives
# ---------------------------------------------------------------------------


class TxStatus(Enum):
    """Outcome states for transactions (used by TxRepairService and receipts)."""

    PENDING   = auto()   # Broadcast but not yet mined
    MINED     = auto()   # Included in a block
    REPLACED  = auto()   # Original dropped; replacement mined
    DROPPED   = auto()   # Evicted from mempool, no replacement
    SUCCESS   = auto()   # Confirmed successfully (used by tests and receipt fixtures)


@dataclass(frozen=True)
class TxParams:
    """Validated, normalised EVM transaction parameter set.

    All monetary values are in **wei**; gas price is in **wei** as well so
    that callers never need to convert at the call site.

    Attributes:
        from_address: Sender checksummed address.
        to: Recipient checksummed address.
        value: Amount of native currency to transfer (wei).
        gas: Gas limit for the transaction.
        gas_price_wei: Gas price in wei.
        data: Optional ABI-encoded call data (for contract interactions).
        nonce: Transaction sequence number; populated by the manager
            immediately before signing.
    """

    from_address: str
    to: str
    value: int
    gas: int
    gas_price_wei: int
    data: bytes = field(default=b"")
    nonce: int = field(default=0)

    @property
    def gas_price_gwei(self) -> float:
        """Return gas price expressed in Gwei for human-readable comparisons.

        Returns:
            Gas price in Gwei as a float.
        """
        return self.gas_price_wei / 1e9

    def as_web3_dict(self) -> dict[str, Any]:
        """Serialise to the dict format expected by web3.py signing methods.

        Returns:
            Dictionary with keys matching the web3.py transaction spec.
        """
        return {
            "from": self.from_address,
            "to": self.to,
            "value": self.value,
            "gas": self.gas,
            "gasPrice": self.gas_price_wei,
            "data": self.data,
            "nonce": self.nonce,
        }


@dataclass(frozen=True)
class TxReceipt:
    """Lightweight receipt returned after a successful broadcast.

    Attributes:
        tx_hash: Hex-encoded transaction hash (``0x…``).
        wallet_id: Originating wallet identifier.
        nonce_used: The nonce consumed by this transaction.
        status: Final transaction status.
        gas_used: Actual gas consumed by the transaction.
        block_number: Block number in which the transaction was mined.
        chain_id: EVM chain identifier.
    """

    tx_hash: str
    wallet_id: str
    nonce_used: int
    status: TxStatus = TxStatus.SUCCESS
    gas_used: int = 21_000
    block_number: int = 0
    chain_id: int = 8453


# ---------------------------------------------------------------------------
# Notification primitives
# ---------------------------------------------------------------------------


class EventKind(Enum):
    """Enumeration of all observable system events.

    Each member maps to a distinct notification trigger in the orchestrator.
    Using a typed enum (instead of raw strings) prevents typos at call sites
    and makes exhaustive matching possible in the notifier's formatter.
    """

    # Lifecycle
    SYSTEM_BOOT = auto()       # Orchestrator started successfully
    SYSTEM_SHUTDOWN = auto()   # Orchestrator completed all wallets cleanly
    SYSTEM_ALERT = auto()      # Generic system-level alert (e.g. vault key rotation reminder)

    # Gas monitor
    GAS_PAUSE = auto()         # Gas price exceeded ceiling; system paused
    GAS_RESUME = auto()        # Gas price dropped; system resumed
    GAS_DEFERRED = auto()      # Gas too high; rescheduled for 2-4h, not failed

    # Per-wallet outcomes
    WALLET_SUCCESS = auto()    # A wallet completed its task batch without error
    WALLET_FAILURE = auto()    # A wallet's task batch raised an unrecoverable error

    # Configuration changes
    EXECUTION_MODE_CHANGED = auto()  # Operator changed execution mode via /setmode


@dataclass(frozen=True)
class NotificationEvent:
    """Immutable payload carried from the orchestrator to the notifier.

    Attributes:
        kind: Semantic event type (see :class:`EventKind`).
        wallet_id: Wallet identifier for wallet-scoped events; ``None`` for
            system-level events.
        detail: Optional freeform detail string (e.g. exception message,
            gas price reading).  Must never contain raw private keys or
            un-masked addresses — the notifier applies address masking, but
            callers should not pass raw secrets here.
        gas_price_gwei: Current gas price snapshot for gas-related events.
        tx_hash: Transaction hash for ``WALLET_SUCCESS`` events.
    """

    kind: EventKind
    wallet_id: str | None = None
    detail: str | None = None
    gas_price_gwei: float | None = None
    tx_hash: str | None = None


# ---------------------------------------------------------------------------
# Multi-chain router primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainConfig:
    """Immutable configuration for a single EVM-compatible chain.

    Attributes:
        chain_id: EVM chain identifier (e.g. 1=Ethereum, 8453=Base).
        name: Human-readable chain name for logging and notifications.
        native_symbol: Native currency symbol (e.g. "ETH", "MATIC").
        primary_rpc: Preferred JSON-RPC HTTP endpoint.
        fallback_rpcs: Ordered list of backup endpoints tried on primary failure.
        max_gas_price_gwei: Chain-level gas ceiling.
        block_time_seconds: Approximate block time used for stuck-tx deadlines.
    """

    chain_id: int
    name: str
    native_symbol: str
    primary_rpc: str
    fallback_rpcs: tuple[str, ...] = field(default_factory=tuple)
    max_gas_price_gwei: float = 100.0
    block_time_seconds: float = 12.0


# ---------------------------------------------------------------------------
# Transaction repair primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TxRecord:
    """Full lifecycle record for one tracked transaction.

    Attributes:
        tx_hash: Original transaction hash.
        wallet_id: Originating wallet identifier.
        chain_id: Chain where the transaction was submitted.
        nonce: EVM nonce used.
        gas_price_gwei: Gas price at submission time.
        gas_limit: Gas limit set on the transaction.
        value_wei: Native currency transferred.
        status: Current lifecycle state.
        submitted_at: Unix epoch timestamp of first broadcast.
        gas_used: Actual gas consumed (None if pending).
        mined_at: Unix epoch timestamp of mining confirmation.
        replacement_hash: Hash of the speed-up replacement tx, if any.
        strategy_type: Portfolio pillar that originated this tx (YIELD, ALGO, AIRDROP).
        strategy: Strategy name alias (e.g. ALGO, YIELD, AIRDROP).
        direction: Trade direction alias (e.g. BUY, SELL, SUPPLY).
        amount_wei: Amount transferred in wei (alias for value_wei).
    """

    tx_hash: str
    wallet_id: str
    nonce: int
    status: TxStatus
    chain_id: int = 0
    gas_price_gwei: float = 0.0
    gas_limit: int = 0
    value_wei: int = 0
    submitted_at: float = 0.0
    gas_used: int | None = None
    mined_at: float | None = None
    replacement_hash: str | None = None
    strategy_type: str | None = None
    strategy_id: str | None = None
    strategy: str | None = None
    direction: str | None = None
    amount_wei: int | None = None


# ---------------------------------------------------------------------------
# Strategy primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyConfig:
    """Execution variance parameters for a single task batch.

    All ``*_pct`` fields are fractions of wallet balance (0.0-1.0).

    Attributes:
        min_pct: Minimum fraction of balance per transaction.
        max_pct: Maximum fraction of balance per transaction.
        amount_sigma_pct: Gaussian sigma as fraction of chosen amount.
        delay_sigma_seconds: Std-deviation of Gaussian delay jitter in seconds.
        shuffle_tasks: Whether to randomise sub-task order within a session.
        shuffle_wallets: Whether to randomise wallet execution order.
    """

    min_pct: float = 0.10
    max_pct: float = 0.15
    amount_sigma_pct: float = 0.05
    delay_sigma_seconds: float = 0.5
    shuffle_tasks: bool = True
    shuffle_wallets: bool = True
