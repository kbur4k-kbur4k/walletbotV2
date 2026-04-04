"""
wallet_manager/network/balance_provider.py

Async on-chain balance query service.

Responsibility
--------------
Implement :class:`~wallet_manager.core.interfaces.IBalanceProvider` using
``web3.py``.  This module handles **only** read queries — it contains no
signing, no nonce management, and no retry logic (retry is a cross-cutting
concern applied via the ``@async_retry`` decorator from the resilience layer).

Separation of concerns
-----------------------
``IBalanceProvider`` is intentionally narrow (one method).  A component that
only reads balances must not be coupled to transaction-signing capabilities.
This follows the Interface Segregation Principle and keeps test doubles small.
"""

from __future__ import annotations

from web3 import AsyncWeb3

from wallet_manager.core.interfaces import IBalanceProvider
from wallet_manager.resilience.retry import async_retry
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.utils.web3_utils import close_w3_provider

log = build_logger(__name__)


class Web3BalanceProvider(IBalanceProvider):
    """Production implementation of :class:`IBalanceProvider` using web3.py.

    Each call creates a short-lived ``AsyncWeb3`` instance bound to the
    supplied *rpc_url*.  This is intentional: the provider is stateless so
    that it can be shared across wallets without any session coupling.

    Retry behaviour is applied by the ``@async_retry`` decorator from the
    resilience layer — this class contains no retry loops itself.

    Example:
        .. code-block:: python

            provider = Web3BalanceProvider()
            balance = await provider.get_balance(
                "https://mainnet.infura.io/v3/TOKEN",
                "0xAbCd...",
            )
    """

    @async_retry()
    async def get_balance(self, rpc_url: str, address: str) -> int:
        """Fetch the native-currency balance for *address* from *rpc_url*.

        Args:
            rpc_url: JSON-RPC HTTP(S) endpoint for the target chain.
            address: Checksummed or un-checksummed Ethereum address.

        Returns:
            Balance in wei as a non-negative integer.

        Raises:
            web3.exceptions.BadResponseFormat: On malformed RPC responses.
            aiohttp.ClientError: On network-level failures (triggers retry).
            asyncio.TimeoutError: If the RPC call times out (triggers retry).
        """
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        checksum_addr = AsyncWeb3.to_checksum_address(address)
        try:
            log.debug(f"Fetching balance for address {address} via {rpc_url}.")
            balance: int = await w3.eth.get_balance(checksum_addr)
            log.debug(f"Balance for {address}: {balance} wei.")
            return balance
        finally:
            await close_w3_provider(w3)
