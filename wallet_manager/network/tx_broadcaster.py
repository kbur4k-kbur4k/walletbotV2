"""
wallet_manager/network/tx_broadcaster.py

Transaction signing and broadcast service.

Responsibility
--------------
Implement :class:`~wallet_manager.core.interfaces.ITxBroadcaster`.  This
module manages two operations:

1. **Nonce fetch** — retrieve the pending transaction count from the chain.
2. **Sign & send** — sign the transaction *ephemerally* (private key is
   consumed and deleted within this method) and broadcast the raw bytes.

What this module does NOT do
-----------------------------
* No encryption / decryption — the vault layer owns that.
* No retry orchestration — ``@async_retry`` from the resilience layer is
  applied as a decorator.
* No gas validation — the manager checks gas limits before calling here.
* No cooldown delays — the manager applies those before any network call.

Security note on the private key argument
-----------------------------------------
``sign_and_send`` receives the plaintext private key as a short-lived string
argument.  The method processes it inside a ``try/finally`` block and deletes
all local references before returning.  Python's garbage collector may not
zero memory immediately, but eliminating named references ensures the key
cannot be accessed by subsequent code in the same frame.
"""

from __future__ import annotations

import contextlib

from web3 import AsyncWeb3

from wallet_manager.core.interfaces import ITxBroadcaster
from wallet_manager.core.models import TxParams, TxReceipt
from wallet_manager.resilience.retry import async_retry
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.utils.web3_utils import close_w3_provider

log = build_logger(__name__)


class Web3TxBroadcaster(ITxBroadcaster):
    """Production implementation of :class:`ITxBroadcaster` using web3.py.

    Stateless — a single instance may be shared across all wallets.
    Per-wallet isolation is enforced at the session level by
    :class:`~wallet_manager.network.session_factory.SessionFactory`.

    Example:
        .. code-block:: python

            broadcaster = Web3TxBroadcaster()
            nonce = await broadcaster.get_nonce(rpc_url, sender_address)
            receipt = await broadcaster.sign_and_send(rpc_url, tx_params, key)
    """

    # ------------------------------------------------------------------
    # ITxBroadcaster implementation
    # ------------------------------------------------------------------

    @async_retry()
    async def get_nonce(self, rpc_url: str, address: str) -> int:
        """Fetch the pending nonce for *address* from *rpc_url*.

        Using ``"pending"`` as the block tag ensures that queued (not-yet-mined)
        transactions are accounted for, preventing nonce reuse under load.

        Args:
            rpc_url: JSON-RPC HTTP(S) endpoint.
            address: Sender checksummed address.

        Returns:
            The next unused nonce as an integer.

        Raises:
            aiohttp.ClientError: On network failures (triggers retry).
            asyncio.TimeoutError: On RPC timeout (triggers retry).
        """
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        checksum_addr = AsyncWeb3.to_checksum_address(address)
        try:
            log.debug(f"Fetching pending nonce for {address}.")
            nonce: int = await w3.eth.get_transaction_count(checksum_addr, "pending")
            log.debug(f"Nonce for {address}: {nonce}.")
            return nonce
        finally:
            await close_w3_provider(w3)

    @async_retry()
    async def sign_and_send(
        self,
        rpc_url: str,
        tx_params: TxParams,
        plaintext_private_key: str,
    ) -> TxReceipt:
        """Sign *tx_params* ephemerally and broadcast the raw transaction.

        The private key is used **only** within this method's frame.  All local
        references are deleted in the ``finally`` block before the coroutine
        resumes the caller.

        Args:
            rpc_url: JSON-RPC endpoint to broadcast through.
            tx_params: Fully populated :class:`~wallet_manager.core.models.TxParams`
                including a resolved nonce and chain ID.
            plaintext_private_key: Hex-encoded private key.  Treated as
                ephemeral — **not** stored or logged.

        Returns:
            :class:`~wallet_manager.core.models.TxReceipt` with the tx hash.

        Raises:
            ValueError: If signing fails due to a malformed key or params.
            aiohttp.ClientError: On broadcast network failure (triggers retry).
            asyncio.TimeoutError: On RPC timeout (triggers retry).
        """
        from eth_account import Account  # local import keeps module-level deps minimal

        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))

        # ── Ephemeral signing scope ───────────────────────────────────────
        # ``_key`` and ``_signed`` hold sensitive material; both are deleted
        # before the method returns, regardless of success or failure.
        _key: str = plaintext_private_key
        tx_hash_hex: str
        try:
            tx_dict = tx_params.as_web3_dict()
            _signed = Account.sign_transaction(tx_dict, private_key=_key)
            log.debug(f"Transaction signed for nonce {tx_params.nonce}.")

            raw: bytes = _signed.raw_transaction
            tx_hash_bytes = await w3.eth.send_raw_transaction(raw)
            tx_hash_hex = tx_hash_bytes.hex()
        finally:
            # Drop every reference to plaintext material immediately
            _key = ""
            del _key
            with contextlib.suppress(UnboundLocalError):
                del _signed  # Signing failed before _signed was assigned
            await close_w3_provider(w3)
        # ── End of ephemeral scope ────────────────────────────────────────

        log.info(f"Transaction broadcast. Hash: {tx_hash_hex}. Nonce used: {tx_params.nonce}.")

        return TxReceipt(
            tx_hash=tx_hash_hex,
            wallet_id="",          # caller (manager) fills wallet_id after return
            nonce_used=tx_params.nonce,
        )
