"""tests/unit/test_tx_broadcaster.py — tests for network/tx_broadcaster.py Web3TxBroadcaster."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.core.models import TxParams, TxReceipt, TxStatus
from wallet_manager.network.tx_broadcaster import Web3TxBroadcaster

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DUMMY_RPC = "https://rpc.example.com"
DUMMY_ADDRESS = "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"
DUMMY_PRIVATE_KEY = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
KNOWN_TX_HASH_BYTES = bytes.fromhex("deadbeef" + "00" * 28)
KNOWN_TX_HASH_HEX = KNOWN_TX_HASH_BYTES.hex()


def _make_tx_params(nonce: int = 5, gas_price_gwei: float = 10.0) -> TxParams:
    return TxParams(
        from_address=DUMMY_ADDRESS,
        to="0x1111111111111111111111111111111111111111",
        value=1_000_000_000_000_000,
        gas=21_000,
        gas_price_wei=int(gas_price_gwei * 1e9),
        nonce=nonce,
    )


# ---------------------------------------------------------------------------
# test_broadcast_builds_signed_tx
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_builds_signed_tx():
    """sign_and_send calls Account.sign_transaction with the correct tx dict
    and then broadcasts the raw transaction bytes."""
    tx_params = _make_tx_params(nonce=3)
    tx_dict = tx_params.as_web3_dict()

    mock_signed = MagicMock()
    mock_signed.raw_transaction = b"\x01\x02\x03"

    mock_w3 = MagicMock()
    mock_w3.eth.send_raw_transaction = AsyncMock(return_value=KNOWN_TX_HASH_BYTES)

    with (
        patch("wallet_manager.network.tx_broadcaster.AsyncWeb3") as mock_async_web3_cls,
        patch("eth_account.Account") as mock_account,
    ):
        mock_async_web3_cls.return_value = mock_w3
        mock_async_web3_cls.AsyncHTTPProvider = MagicMock()
        mock_account.sign_transaction.return_value = mock_signed

        broadcaster = Web3TxBroadcaster()
        receipt = await broadcaster.sign_and_send(DUMMY_RPC, tx_params, DUMMY_PRIVATE_KEY)

    mock_account.sign_transaction.assert_called_once_with(tx_dict, private_key=DUMMY_PRIVATE_KEY)
    mock_w3.eth.send_raw_transaction.assert_awaited_once_with(b"\x01\x02\x03")
    assert isinstance(receipt, TxReceipt)


# ---------------------------------------------------------------------------
# test_broadcast_returns_tx_hash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_returns_tx_hash():
    """sign_and_send returns a TxReceipt whose tx_hash matches the hash returned
    by send_raw_transaction, and wallet_id is left as empty string."""
    tx_params = _make_tx_params(nonce=7)

    mock_signed = MagicMock()
    mock_signed.raw_transaction = b"\xab\xcd"

    mock_w3 = MagicMock()
    mock_w3.eth.send_raw_transaction = AsyncMock(return_value=KNOWN_TX_HASH_BYTES)

    with (
        patch("wallet_manager.network.tx_broadcaster.AsyncWeb3") as mock_async_web3_cls,
        patch("eth_account.Account") as mock_account,
    ):
        mock_async_web3_cls.return_value = mock_w3
        mock_async_web3_cls.AsyncHTTPProvider = MagicMock()
        mock_account.sign_transaction.return_value = mock_signed

        broadcaster = Web3TxBroadcaster()
        receipt = await broadcaster.sign_and_send(DUMMY_RPC, tx_params, DUMMY_PRIVATE_KEY)

    assert receipt.tx_hash == KNOWN_TX_HASH_HEX
    assert receipt.wallet_id == ""   # manager stamps wallet_id after the call
    assert receipt.nonce_used == 7


# ---------------------------------------------------------------------------
# test_broadcast_rpc_failure_raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_rpc_failure_raises():
    """If send_raw_transaction raises Exception('nonce too low'), the exception
    propagates to the caller (after retry exhaustion)."""
    tx_params = _make_tx_params(nonce=0)

    mock_signed = MagicMock()
    mock_signed.raw_transaction = b"\xff"

    mock_w3 = MagicMock()
    mock_w3.eth.send_raw_transaction = AsyncMock(
        side_effect=Exception("nonce too low")
    )

    with (
        patch("wallet_manager.network.tx_broadcaster.AsyncWeb3") as mock_async_web3_cls,
        patch("eth_account.Account") as mock_account,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_async_web3_cls.return_value = mock_w3
        mock_async_web3_cls.AsyncHTTPProvider = MagicMock()
        mock_account.sign_transaction.return_value = mock_signed

        broadcaster = Web3TxBroadcaster()
        with pytest.raises(Exception, match="nonce too low"):
            await broadcaster.sign_and_send(DUMMY_RPC, tx_params, DUMMY_PRIVATE_KEY)


# ---------------------------------------------------------------------------
# test_eip1559_tx_params_used  (legacy gasPrice, not EIP-1559)
# ---------------------------------------------------------------------------

def test_legacy_gas_price_in_web3_dict():
    """TxParams.as_web3_dict() includes the 'gasPrice' key (legacy format),
    not EIP-1559 'maxFeePerGas'/'maxPriorityFeePerGas'."""
    tx_params = _make_tx_params(gas_price_gwei=25.0)
    tx_dict = tx_params.as_web3_dict()

    assert "gasPrice" in tx_dict, "Expected legacy 'gasPrice' key in web3 dict"
    assert tx_dict["gasPrice"] == int(25.0 * 1e9)
    assert "maxFeePerGas" not in tx_dict
    assert "maxPriorityFeePerGas" not in tx_dict


# ---------------------------------------------------------------------------
# test_get_nonce_calls_get_transaction_count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_nonce_calls_get_transaction_count():
    """get_nonce calls w3.eth.get_transaction_count with the checksummed address
    and the 'pending' block tag, returning the nonce integer."""
    expected_nonce = 17

    mock_w3 = MagicMock()
    mock_w3.eth.get_transaction_count = AsyncMock(return_value=expected_nonce)

    checksummed = "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"

    with (
        patch("wallet_manager.network.tx_broadcaster.AsyncWeb3") as mock_async_web3_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_async_web3_cls.return_value = mock_w3
        mock_async_web3_cls.AsyncHTTPProvider = MagicMock()
        mock_async_web3_cls.to_checksum_address = MagicMock(return_value=checksummed)

        broadcaster = Web3TxBroadcaster()
        nonce = await broadcaster.get_nonce(DUMMY_RPC, DUMMY_ADDRESS)

    assert nonce == expected_nonce
    mock_w3.eth.get_transaction_count.assert_awaited_once_with(checksummed, "pending")


# ---------------------------------------------------------------------------
# test_sign_and_send_nonce_used_matches_tx_params
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sign_and_send_nonce_used_matches_tx_params():
    """TxReceipt.nonce_used reflects the nonce set in TxParams, not a default."""
    expected_nonce = 99
    tx_params = _make_tx_params(nonce=expected_nonce)

    mock_signed = MagicMock()
    mock_signed.raw_transaction = b"\x00"

    mock_w3 = MagicMock()
    mock_w3.eth.send_raw_transaction = AsyncMock(return_value=b"\x00" * 32)

    with (
        patch("wallet_manager.network.tx_broadcaster.AsyncWeb3") as mock_async_web3_cls,
        patch("eth_account.Account") as mock_account,
    ):
        mock_async_web3_cls.return_value = mock_w3
        mock_async_web3_cls.AsyncHTTPProvider = MagicMock()
        mock_account.sign_transaction.return_value = mock_signed

        broadcaster = Web3TxBroadcaster()
        receipt = await broadcaster.sign_and_send(DUMMY_RPC, tx_params, DUMMY_PRIVATE_KEY)

    assert receipt.nonce_used == expected_nonce
