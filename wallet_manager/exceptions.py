"""
wallet_manager/exceptions.py

Domain-specific exception hierarchy for the wallet_manager package.
"""

from __future__ import annotations


class WalletManagerError(Exception):
    """Root exception for all wallet_manager failures."""

    def __init__(self, message: str, wallet_id: str | None = None) -> None:
        self.message = message
        self.wallet_id = wallet_id
        suffix = f" [wallet_id={wallet_id!r}]" if wallet_id else ""
        super().__init__(f"{message}{suffix}")


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(WalletManagerError):
    """Raised when the package cannot be initialised due to a config problem."""


class WalletNotFoundError(WalletManagerError):
    """Raised when a wallet_id is referenced but not registered."""


# ---------------------------------------------------------------------------
# Security / vault errors
# ---------------------------------------------------------------------------

class VaultError(WalletManagerError):
    """Base class for encryption vault failures."""


class DecryptionError(VaultError):
    """Raised when Fernet decryption fails."""


class EncryptionError(VaultError):
    """Raised when Fernet encryption fails unexpectedly."""


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------

class NetworkError(WalletManagerError):
    """Base class for all RPC / HTTP failures."""


class BalanceFetchError(NetworkError):
    """Raised when a balance query fails after all retry attempts."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        address: str | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.address = address


class NonceFetchError(NetworkError):
    """Raised when the on-chain nonce cannot be retrieved after all retries."""


class BroadcastError(NetworkError):
    """Raised when a signed transaction cannot be broadcast to the network."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        tx_hash: str | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.tx_hash = tx_hash


# ---------------------------------------------------------------------------
# Transaction validation errors
# ---------------------------------------------------------------------------

class TransactionValidationError(WalletManagerError):
    """Raised when a transaction is rejected before it reaches the network."""


class GasPriceTooHighError(TransactionValidationError):
    """Raised when the requested gas price exceeds the configured maximum."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        requested_gwei: float = 0.0,
        max_gwei: float = 0.0,
    ) -> None:
        super().__init__(message, wallet_id)
        self.requested_gwei = requested_gwei
        self.max_gwei = max_gwei


class InvalidNonceError(TransactionValidationError):
    """Raised when a nonce conflict or underpriced replacement is detected."""


# ---------------------------------------------------------------------------
# Resilience errors
# ---------------------------------------------------------------------------

class ResilienceError(WalletManagerError):
    """Base class for circuit breaker and retry-policy failures."""


class CircuitOpenError(ResilienceError):
    """Raised when a request is blocked because the circuit breaker is OPEN."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        seconds_until_probe: float | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.seconds_until_probe = seconds_until_probe


class MaxRetriesExceededError(ResilienceError):
    """Raised when all retry attempts for an operation have been exhausted."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        attempts: int = 0,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.attempts = attempts
        self.original_exception = original_exception


# ---------------------------------------------------------------------------
# Notification errors
# ---------------------------------------------------------------------------

class NotifierError(WalletManagerError):
    """Base class for notification delivery failures.

    These errors are non-fatal: a failed Telegram delivery must never
    interrupt wallet execution.  The orchestrator logs them and continues.
    """


class NotifierConfigurationError(NotifierError):
    """Raised when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing."""


class NotifierDeliveryError(NotifierError):
    """Raised when the Telegram API returns a non-2xx response after retries."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.http_status = http_status


# ---------------------------------------------------------------------------
# Transaction repair errors
# ---------------------------------------------------------------------------

class TxRepairError(WalletManagerError):
    """Base class for transaction repair failures."""


class TxStuckError(TxRepairError):
    """Raised when a transaction is not mined within the configured deadline."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        tx_hash: str | None = None,
        deadline_seconds: float = 0.0,
    ) -> None:
        super().__init__(message, wallet_id)
        self.tx_hash = tx_hash
        self.deadline_seconds = deadline_seconds


class TxReplacedError(TxRepairError):
    """Raised (informational) when a stuck tx is successfully replaced via RBF.

    This is not a failure — callers should treat it as a success with a new hash.
    """

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        original_hash: str | None = None,
        replacement_hash: str | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.original_hash = original_hash
        self.replacement_hash = replacement_hash


# ---------------------------------------------------------------------------
# Router errors
# ---------------------------------------------------------------------------

class RouterError(WalletManagerError):
    """Base class for multi-chain router failures."""


class NoAvailableRpcError(RouterError):
    """Raised when all RPC providers for a chain are exhausted or unavailable."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        chain_id: int | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.chain_id = chain_id


class ChainNotConfiguredError(RouterError):
    """Raised when a requested chain_id has no registered configuration."""


# ---------------------------------------------------------------------------
# Strategy errors
# ---------------------------------------------------------------------------

class StrategyError(WalletManagerError):
    """Raised when a strategy calculation produces an invalid result."""


# ---------------------------------------------------------------------------
# Analytics errors
# ---------------------------------------------------------------------------

class AnalyticsError(WalletManagerError):
    """Base class for local analytics database failures."""


class AnalyticsWriteError(AnalyticsError):
    """Raised when writing a transaction record to SQLite fails."""


class AnalyticsQueryError(AnalyticsError):
    """Raised when querying the analytics database fails."""


# ---------------------------------------------------------------------------
# DeFi strategy errors
# ---------------------------------------------------------------------------

class DeFiStrategyError(WalletManagerError):
    """Base class for DeFi-specific strategy failures."""


class InsufficientCollateralError(DeFiStrategyError):
    """Raised when collateral is insufficient for a borrow operation."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        required_usd: float = 0.0,
        available_usd: float = 0.0,
    ) -> None:
        super().__init__(message, wallet_id)
        self.required_usd = required_usd
        self.available_usd = available_usd


class BorrowCapExceededError(DeFiStrategyError):
    """Raised when the borrow amount exceeds Aave V3 caps."""


class FlashCrashDetectedError(DeFiStrategyError):
    """Raised when a flash crash exceeding the threshold is detected."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        drop_pct: float = 0.0,
        window_seconds: float = 0.0,
    ) -> None:
        super().__init__(message, wallet_id)
        self.drop_pct = drop_pct
        self.window_seconds = window_seconds


class KillSwitchActivatedError(DeFiStrategyError):
    """Raised when the emergency KILL_SWITCH is triggered."""


class ShortPositionError(DeFiStrategyError):
    """Raised on failures during synthetic short lifecycle."""

    def __init__(
        self,
        message: str,
        wallet_id: str | None = None,
        position_id: str | None = None,
    ) -> None:
        super().__init__(message, wallet_id)
        self.position_id = position_id


class IdleSweepError(DeFiStrategyError):
    """Raised when idle capital sweep to Aave fails."""
