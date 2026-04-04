"""tests/unit/test_balance_provider.py — tests for network/balance_provider.py Web3BalanceProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.network.balance_provider import Web3BalanceProvider

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DUMMY_RPC = "https://rpc.example.com"
RAW_ADDRESS = "0xabcdef0123456789abcdef0123456789abcdef01"
CHECKSUM_ADDRESS = "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"
ONE_ETH_WEI = 1_000_000_000_000_000_000


# ---------------------------------------------------------------------------
# test_fetch_native_balance_returns_wei
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_native_balance_returns_wei():
    """get_balance returns the raw wei integer provided by w3.eth.get_balance."""
    mock_w3 = MagicMock()
    mock_w3.eth.get_balance = AsyncMock(return_value=ONE_ETH_WEI)

    with (
        patch("wallet_manager.network.balance_provider.AsyncWeb3") as mock_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_cls.return_value = mock_w3
        mock_cls.AsyncHTTPProvider = MagicMock()
        mock_cls.to_checksum_address = MagicMock(return_value=CHECKSUM_ADDRESS)

        provider = Web3BalanceProvider()
        result = await provider.get_balance(DUMMY_RPC, RAW_ADDRESS)

    assert result == ONE_ETH_WEI


# ---------------------------------------------------------------------------
# test_fetch_native_balance_rpc_error_propagates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_native_balance_rpc_error_propagates():
    """If w3.eth.get_balance raises an Exception, the exception propagates
    to the caller after retries are exhausted (retry is bypassed in this test)."""
    mock_w3 = MagicMock()
    mock_w3.eth.get_balance = AsyncMock(side_effect=Exception("RPC connection refused"))

    with (
        patch("wallet_manager.network.balance_provider.AsyncWeb3") as mock_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_cls.return_value = mock_w3
        mock_cls.AsyncHTTPProvider = MagicMock()
        mock_cls.to_checksum_address = MagicMock(return_value=CHECKSUM_ADDRESS)

        provider = Web3BalanceProvider()
        with pytest.raises(Exception, match="RPC connection refused"):
            await provider.get_balance(DUMMY_RPC, RAW_ADDRESS)


# ---------------------------------------------------------------------------
# test_fetch_balance_checksums_address
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_balance_checksums_address():
    """get_balance calls AsyncWeb3.to_checksum_address with the raw input address
    and passes the checksummed result to w3.eth.get_balance."""
    mock_w3 = MagicMock()
    mock_w3.eth.get_balance = AsyncMock(return_value=0)

    with (
        patch("wallet_manager.network.balance_provider.AsyncWeb3") as mock_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_cls.return_value = mock_w3
        mock_cls.AsyncHTTPProvider = MagicMock()
        mock_cls.to_checksum_address = MagicMock(return_value=CHECKSUM_ADDRESS)

        provider = Web3BalanceProvider()
        await provider.get_balance(DUMMY_RPC, RAW_ADDRESS)

    mock_cls.to_checksum_address.assert_called_once_with(RAW_ADDRESS)
    mock_w3.eth.get_balance.assert_awaited_once_with(CHECKSUM_ADDRESS)


# ---------------------------------------------------------------------------
# test_fetch_balance_zero_returns_zero
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_balance_zero_returns_zero():
    """get_balance returns 0 when the on-chain balance is zero (empty wallet)."""
    mock_w3 = MagicMock()
    mock_w3.eth.get_balance = AsyncMock(return_value=0)

    with (
        patch("wallet_manager.network.balance_provider.AsyncWeb3") as mock_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_cls.return_value = mock_w3
        mock_cls.AsyncHTTPProvider = MagicMock()
        mock_cls.to_checksum_address = MagicMock(return_value=CHECKSUM_ADDRESS)

        provider = Web3BalanceProvider()
        result = await provider.get_balance(DUMMY_RPC, RAW_ADDRESS)

    assert result == 0


# ---------------------------------------------------------------------------
# test_fetch_balance_creates_provider_with_rpc_url
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_balance_creates_provider_with_rpc_url():
    """get_balance creates an AsyncHTTPProvider with the supplied rpc_url."""
    mock_w3 = MagicMock()
    mock_w3.eth.get_balance = AsyncMock(return_value=100)

    with (
        patch("wallet_manager.network.balance_provider.AsyncWeb3") as mock_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_cls.return_value = mock_w3
        mock_cls.AsyncHTTPProvider = MagicMock()
        mock_cls.to_checksum_address = MagicMock(return_value=CHECKSUM_ADDRESS)

        provider = Web3BalanceProvider()
        await provider.get_balance(DUMMY_RPC, RAW_ADDRESS)

    mock_cls.AsyncHTTPProvider.assert_called_once_with(DUMMY_RPC)


# ---------------------------------------------------------------------------
# test_fetch_balance_large_value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_balance_large_value():
    """get_balance can handle very large wei values (whale wallet scenario)."""
    large_balance = 10_000 * ONE_ETH_WEI  # 10,000 ETH

    mock_w3 = MagicMock()
    mock_w3.eth.get_balance = AsyncMock(return_value=large_balance)

    with (
        patch("wallet_manager.network.balance_provider.AsyncWeb3") as mock_cls,
        patch("wallet_manager.resilience.retry.async_retry", lambda *a, **kw: lambda fn: fn),
    ):
        mock_cls.return_value = mock_w3
        mock_cls.AsyncHTTPProvider = MagicMock()
        mock_cls.to_checksum_address = MagicMock(return_value=CHECKSUM_ADDRESS)

        provider = Web3BalanceProvider()
        result = await provider.get_balance(DUMMY_RPC, RAW_ADDRESS)

    assert result == large_balance
    assert isinstance(result, int)
