"""
wallet_manager/network/tx_repair.py

Advanced Nonce Manager and Transaction Repair Service.

Responsibility
--------------
Two tightly-coupled concerns that share the same domain (nonce arithmetic
and transaction lifecycle) live here together — separating them further
would require passing nonce state across module boundaries unnecessarily.

1.  **NonceManager** — per-wallet async nonce tracker.
    Maintains an in-memory cache of each wallet's next usable nonce so that
    concurrent signing coroutines never collide.  Syncs lazily with the chain
    on first use and forcibly on ``nonce too low`` errors.

2.  **TxRepairService** — stuck transaction detector and RBF replacement engine.
    Polls ``eth_getTransactionReceipt`` after broadcast.  If a transaction is
    not mined within a configurable deadline it triggers a Replace-By-Fee (RBF)
    resubmission: same nonce, same calldata, gas price raised by a random factor
    between ``RBF_GAS_BUMP_MIN`` (15%) and ``RBF_GAS_BUMP_MAX`` (25%) percent.

RBF design notes
----------------
Ethereum (and most EVM chains) will accept a replacement transaction for an
already-pending nonce if and only if the new gas price is ≥ 110% of the
original (i.e. a minimum 10% bump).  We use 15–25% to ensure nodes that
enforce a stricter floor still accept the replacement, while avoiding an
unnecessarily large gas overspend.

The replacement is signed ephemerally using the same Vault pattern as the
primary broadcaster — the private key reference is deleted inside a
``try/finally`` block immediately after signing.

Concurrency model
-----------------
Each wallet has its own ``asyncio.Lock`` inside ``NonceManager``.  This is the
same lock that guards nonce operations in ``WalletManager``; callers should
use either the manager's lock OR the NonceManager directly — not both — to
avoid deadlock.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

from eth_typing import HexStr
from web3 import AsyncWeb3

from wallet_manager.config import Settings, get_settings
from wallet_manager.core.interfaces import INonceManager
from wallet_manager.core.models import TxParams, TxRecord, TxStatus
from wallet_manager.exceptions import TxStuckError
from wallet_manager.resilience.retry import async_retry
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.utils.web3_utils import close_w3_provider

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants (loaded from centralized config)
# ---------------------------------------------------------------------------


def _cfg() -> Settings:
    """Shorthand for centralized settings singleton."""
    return get_settings()


# ---------------------------------------------------------------------------
# NonceManager
# ---------------------------------------------------------------------------


@dataclass
class _NonceEntry:
    """Mutable nonce cache entry for one wallet.

    Attributes:
        value: Next nonce to use (optimistic, incremented before broadcast).
        synced: Whether the value has been confirmed against the chain.
        lock: Per-wallet asyncio lock for serialising concurrent increments.
    """

    value: int = -1          # -1 = uninitialized; will sync on first use
    synced: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class NonceManager(INonceManager):
    """Async per-wallet nonce cache implementing :class:`INonceManager`.

    Each wallet's nonce is incremented optimistically after every ``get_and_increment``
    call so that concurrent task coroutines queued behind the same lock each
    receive a distinct, sequentially increasing nonce without querying the
    chain more than once per session.

    On a ``nonce too low`` error (detected by the caller), ``invalidate()``
    forces a full chain re-sync on the next access.

    Example:
        .. code-block:: python

            nonce_mgr = NonceManager()
            nonce = await nonce_mgr.get_and_increment(
                "hot_wallet_01", rpc_url, sender_address
            )
            # Build and sign tx with `nonce`, then broadcast.
            # If broadcast raises "nonce too low":
            await nonce_mgr.invalidate("hot_wallet_01")
    """

    def __init__(self) -> None:
        # wallet_id → _NonceEntry; defaultdict creates entries on first access
        self._cache: dict[str, _NonceEntry] = defaultdict(_NonceEntry)

    # ------------------------------------------------------------------
    # INonceManager implementation
    # ------------------------------------------------------------------

    async def get_and_increment(
        self,
        wallet_id: str,
        rpc_url: str,
        address: str,
    ) -> int:
        """Return the next usable nonce and advance the internal counter.

        If the cache entry is cold (first call or after ``invalidate``), the
        method fetches the on-chain pending nonce before returning.

        Args:
            wallet_id: Cache key; typically the wallet's human-readable ID.
            rpc_url: JSON-RPC endpoint used for chain sync when cache is cold.
            address: Sender checksummed address for on-chain nonce lookup.

        Returns:
            Integer nonce to stamp on the next outgoing transaction.
        """
        entry = self._cache[wallet_id]

        async with entry.lock:
            if not entry.synced or entry.value < 0:
                # Cold cache — fetch pending nonce from chain
                entry.value = await self._fetch_pending_nonce(rpc_url, address)
                entry.synced = True
                log.debug(
                    f"NonceManager: cold-sync for wallet '{wallet_id}' → nonce={entry.value}.",
                )

            nonce = entry.value
            entry.value += 1   # Optimistic increment before broadcast
            log.debug(
                f"NonceManager: issued nonce {nonce} to wallet '{wallet_id}' (next={entry.value}).",
            )
            return nonce

    async def rewind(self, wallet_id: str) -> None:
        """Decrement the cached nonce by one after a pre-broadcast failure.

        Use this when a transaction is signed but never submitted, so the
        slot is not permanently lost.

        Args:
            wallet_id: Wallet whose counter should be decremented.
        """
        entry = self._cache[wallet_id]
        async with entry.lock:
            if entry.synced and entry.value > 0:
                entry.value -= 1
                log.debug(
                    f"NonceManager: rewound nonce for wallet '{wallet_id}' → {entry.value}.",
                )

    async def invalidate(self, wallet_id: str) -> None:
        """Mark the cache as stale; force a chain re-sync on the next call.

        Call this after receiving a ``nonce too low`` error from the RPC node
        to realign the in-memory counter with the canonical chain state.

        Args:
            wallet_id: Wallet whose cache entry should be cleared.
        """
        entry = self._cache[wallet_id]
        async with entry.lock:
            entry.synced = False
            log.info(
                f"NonceManager: cache invalidated for wallet '{wallet_id}'. Will re-sync on next call.",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @async_retry()
    async def _fetch_pending_nonce(self, rpc_url: str, address: str) -> int:
        """Fetch the pending transaction count from the chain.

        ``"pending"`` as the block tag includes queued-but-unmined transactions,
        which is essential to avoid nonce collisions under concurrent load.

        Args:
            rpc_url: JSON-RPC endpoint.
            address: Sender checksummed address.

        Returns:
            Current pending nonce as an integer.
        """
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        checksum = AsyncWeb3.to_checksum_address(address)
        try:
            return await w3.eth.get_transaction_count(checksum, "pending")
        finally:
            await close_w3_provider(w3)


# ---------------------------------------------------------------------------
# TxRepairService
# ---------------------------------------------------------------------------


class TxRepairService:
    """Stuck-transaction detector and Replace-By-Fee (RBF) replacement engine.

    After a transaction is broadcast, call :meth:`watch_and_repair` with the
    original ``TxParams`` and a private-key decrypt callback.  The service
    polls for a receipt and, if none arrives before *deadline_seconds*, builds
    a replacement transaction with a bumped gas price and re-broadcasts it.

    Args:
        nonce_manager: Shared ``NonceManager`` instance so that replacements
            do not consume a new nonce slot.
        deadline_seconds: Seconds to wait for mining before triggering RBF.
        poll_interval: Seconds between receipt-polling attempts.
        max_rbf_attempts: Maximum number of consecutive speed-up attempts.

    Example:
        .. code-block:: python

            repair = TxRepairService(nonce_manager=nonce_mgr)

            receipt = await repair.watch_and_repair(
                rpc_url="https://...",
                tx_hash="0x...",
                original_params=tx_params,
                chain_id=1,
                wallet_id="hot_wallet_01",
                decrypt_key_fn=lambda: vault.decrypt(encrypted_key, wallet_id),
            )
    """

    def __init__(
        self,
        nonce_manager: INonceManager,
        deadline_seconds: float = 0.0,
        poll_interval: float = 0.0,
        max_rbf_attempts: int = 0,
    ) -> None:
        cfg = _cfg()
        self._nonce_mgr = nonce_manager
        self._deadline = deadline_seconds or cfg.tx_stuck_deadline
        self._poll = poll_interval or cfg.tx_poll_interval
        self._max_rbf = max_rbf_attempts or cfg.max_rbf_attempts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def watch_and_repair(  # type: ignore[no-untyped-def]
        self,
        rpc_url: str,
        tx_hash: str,
        original_params: TxParams,
        chain_id: int,
        wallet_id: str,
        decrypt_key_fn,  # Callable[[], str]  — returns ephemeral plaintext key
    ) -> TxRecord:
        """Watch *tx_hash* for mining; trigger RBF if stuck.

        Polls ``eth_getTransactionReceipt`` every ``poll_interval`` seconds.
        If no receipt arrives within ``deadline_seconds``, executes a
        Replace-By-Fee loop up to ``max_rbf_attempts`` times.

        Args:
            rpc_url: JSON-RPC endpoint for polling and replacement broadcast.
            tx_hash: Hash of the already-broadcast transaction to watch.
            original_params: The :class:`~wallet_manager.core.models.TxParams`
                used for the original transaction (provides nonce, gas, calldata).
            chain_id: EVM chain identifier; stamped onto the returned record.
            wallet_id: Wallet identifier for logging and records.
            decrypt_key_fn: Zero-argument callable that returns the plaintext
                private key as an ephemeral string.  Called only when a
                replacement tx must be signed.  The caller owns key lifecycle.

        Returns:
            :class:`~wallet_manager.core.models.TxRecord` describing the final
            outcome (``MINED`` on success, ``REPLACED`` if RBF succeeded).

        Raises:
            TxStuckError: If all RBF attempts are exhausted with no confirmation.
        """
        submitted_at = time.time()
        current_hash = tx_hash
        current_params = original_params

        for rbf_attempt in range(self._max_rbf + 1):
            deadline = time.monotonic() + self._deadline
            log.info(
                f"TxRepair: watching {current_hash} for wallet '{wallet_id}' (attempt {rbf_attempt + 1}, deadline {self._deadline:.0f}s).",
            )

            # --- Poll loop ---
            receipt = await self._poll_until_mined_or_deadline(
                rpc_url, current_hash, deadline
            )

            if receipt is not None:
                # Transaction confirmed
                gas_used: int = receipt.get("gasUsed", 0)
                mined_status = TxStatus.MINED if rbf_attempt == 0 else TxStatus.REPLACED
                log.info(
                    f"TxRepair: {current_hash} mined (gas used: {gas_used}). Status: {mined_status.name}.",
                )
                return TxRecord(
                    tx_hash=tx_hash,              # always the original hash
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=original_params.nonce,
                    gas_price_gwei=original_params.gas_price_gwei,
                    gas_limit=original_params.gas,
                    value_wei=original_params.value,
                    status=mined_status,
                    submitted_at=submitted_at,
                    gas_used=gas_used,
                    mined_at=time.time(),
                    replacement_hash=current_hash if rbf_attempt > 0 else None,
                )

            # --- Deadline reached without mining: attempt RBF ---
            if rbf_attempt >= self._max_rbf:
                break

            log.warning(
                f"TxRepair: {current_hash} stuck after {self._deadline:.0f}s. Attempting RBF (attempt {rbf_attempt + 1}/{self._max_rbf}).",
            )

            try:
                replacement_hash, replacement_params = await self._build_and_send_rbf(
                    rpc_url=rpc_url,
                    original_params=current_params,
                    chain_id=chain_id,
                    wallet_id=wallet_id,
                    decrypt_key_fn=decrypt_key_fn,
                )
                current_hash = replacement_hash
                current_params = replacement_params
                log.info(
                    f"TxRepair: replacement broadcast → {replacement_hash} (gas: {replacement_params.gas_price_gwei:.2f} Gwei).",
                )
            except Exception as exc:
                log.error(
                    f"TxRepair: RBF attempt {rbf_attempt + 1} failed for wallet '{wallet_id}': {exc}",
                )
                # Continue outer loop — maybe original gets mined during retry

        # All attempts exhausted
        raise TxStuckError(
            f"Transaction {tx_hash} was not mined after {self._max_rbf} "
            f"RBF attempts (total wait ≈ "
            f"{self._max_rbf * self._deadline:.0f}s).",
            wallet_id=wallet_id,
            tx_hash=tx_hash,
            deadline_seconds=self._deadline * self._max_rbf,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _poll_until_mined_or_deadline(
        self,
        rpc_url: str,
        tx_hash: str,
        deadline: float,
    ) -> dict[str, Any] | None:
        """Poll ``eth_getTransactionReceipt`` until mined or deadline passes.

        Args:
            rpc_url: JSON-RPC endpoint.
            tx_hash: Transaction hash to poll.
            deadline: ``time.monotonic()`` deadline timestamp.

        Returns:
            Receipt dict if mined; ``None`` if deadline elapsed first.
        """
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        try:
            while time.monotonic() < deadline:
                try:
                    receipt = await w3.eth.get_transaction_receipt(cast(HexStr, tx_hash))
                    if receipt is not None:
                        return dict(receipt)
                except Exception as exc:
                    # Transient RPC errors during polling are non-fatal
                    log.debug(f"TxRepair: receipt poll error ({exc}). Will retry.")

                await asyncio.sleep(self._poll)
        finally:
            await close_w3_provider(w3)

        return None

    async def _build_and_send_rbf(  # type: ignore[no-untyped-def]
        self,
        rpc_url: str,
        original_params: TxParams,
        chain_id: int,
        wallet_id: str,
        decrypt_key_fn,
    ) -> tuple[str, TxParams]:
        """Build a replacement transaction with a bumped gas price and broadcast.

        The gas price bump is sampled uniformly from
        [``RBF_GAS_BUMP_MIN``, ``RBF_GAS_BUMP_MAX``] to introduce variance
        that avoids a predictable escalation pattern.

        Args:
            rpc_url: JSON-RPC endpoint.
            original_params: Parameters of the stuck transaction.
            chain_id: Chain identifier for the replacement.
            wallet_id: Wallet identifier for logging.
            decrypt_key_fn: Callable returning the ephemeral plaintext key.

        Returns:
            Tuple of ``(new_tx_hash, new_TxParams)`` after successful broadcast.

        Raises:
            Exception: Propagates any signing or broadcast failures.
        """
        from eth_account import Account

        # Randomised gas bump: 15–25% above original
        bump_pct = random.uniform(_cfg().rbf_gas_bump_min, _cfg().rbf_gas_bump_max)
        new_gas_price_wei = int(original_params.gas_price_wei * (1.0 + bump_pct))

        replacement = TxParams(
            from_address=original_params.from_address,
            to=original_params.to,
            value=original_params.value,
            gas=original_params.gas,
            gas_price_wei=new_gas_price_wei,
            data=original_params.data,
            nonce=original_params.nonce,    # SAME nonce — this is the RBF mechanism
        )

        tx_dict = replacement.as_web3_dict()
        tx_dict["chainId"] = chain_id

        # ── Ephemeral signing scope ───────────────────────────────────────
        _key: str = ""
        raw_tx: bytes
        try:
            _key = decrypt_key_fn()
            signed = Account.sign_transaction(tx_dict, private_key=_key)
            raw_tx = signed.raw_transaction
        finally:
            _key = ""
            del _key
            with contextlib.suppress(UnboundLocalError):
                del signed
        # ── End of ephemeral scope ────────────────────────────────────────

        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        try:
            tx_hash_bytes = await w3.eth.send_raw_transaction(raw_tx)
        finally:
            await close_w3_provider(w3)
        return tx_hash_bytes.hex(), replacement
