"""tests/unit/test_health_check.py — Unit tests for wallet_manager.utils.health_check."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.utils.health_check import (
    check_aave_contract_accessibility,
    check_rpc_connectivity,
)

RPC_URL = "http://localhost:8545"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(status: int = 200, json_data: dict | None = None):
    """Build a fake aiohttp response context-manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    # Support `async with session.post(...) as resp:`
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_mock_session(response_cm):
    """Build a fake aiohttp.ClientSession context-manager."""
    session = MagicMock()
    session.post = MagicMock(return_value=response_cm)
    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return session_cm


# ---------------------------------------------------------------------------
# check_rpc_connectivity
# ---------------------------------------------------------------------------


class TestCheckRpcConnectivity:
    @pytest.mark.asyncio
    async def test_check_rpc_connectivity_returns_true_on_success(self):
        """Valid eth_blockNumber hex response → True."""
        json_data = {"jsonrpc": "2.0", "id": 1, "result": "0x1234abc"}
        resp_cm = _make_mock_response(status=200, json_data=json_data)
        session_cm = _make_mock_session(resp_cm)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await check_rpc_connectivity(RPC_URL)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_rpc_connectivity_returns_false_on_timeout(self):
        """asyncio.TimeoutError raised by aiohttp → False (no exception propagated)."""
        with patch("aiohttp.ClientSession") as mock_cls:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
            session_cm.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = session_cm

            result = await check_rpc_connectivity(RPC_URL, timeout=1.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_rpc_connectivity_returns_false_on_invalid_response(self):
        """Response that lacks a valid 0x-prefixed result → False."""
        json_data = {"jsonrpc": "2.0", "id": 1, "result": None}
        resp_cm = _make_mock_response(status=200, json_data=json_data)
        session_cm = _make_mock_session(resp_cm)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await check_rpc_connectivity(RPC_URL)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_rpc_connectivity_returns_false_on_http_error(self):
        """HTTP 500 from RPC endpoint → False."""
        resp_cm = _make_mock_response(status=500, json_data={})
        session_cm = _make_mock_session(resp_cm)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await check_rpc_connectivity(RPC_URL)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_rpc_connectivity_returns_false_on_connection_error(self):
        """Generic exception (e.g. connection refused) → False."""
        with patch("aiohttp.ClientSession") as mock_cls:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
            session_cm.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = session_cm

            result = await check_rpc_connectivity(RPC_URL)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_rpc_connectivity_result_without_0x_prefix(self):
        """Result string that does not start with '0x' → False."""
        json_data = {"jsonrpc": "2.0", "id": 1, "result": "12345"}
        resp_cm = _make_mock_response(status=200, json_data=json_data)
        session_cm = _make_mock_session(resp_cm)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await check_rpc_connectivity(RPC_URL)

        assert result is False


# ---------------------------------------------------------------------------
# check_aave_contract_accessibility
# ---------------------------------------------------------------------------


class TestCheckAaveContractAccessibility:
    @pytest.mark.asyncio
    async def test_check_aave_accessible(self):
        """eth_call to Aave V3 Pool returns non-empty hex → True."""
        json_data = {"jsonrpc": "2.0", "id": 1, "result": "0xabcdef1234"}
        resp_cm = _make_mock_response(status=200, json_data=json_data)
        session_cm = _make_mock_session(resp_cm)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await check_aave_contract_accessibility(RPC_URL, chain_id=8453)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_aave_skipped_on_non_base_chain(self):
        """chain_id != 8453 → True (skipped, no network call)."""
        result = await check_aave_contract_accessibility(RPC_URL, chain_id=1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_aave_returns_false_on_empty_result(self):
        """eth_call returns '0x' (empty data) → False."""
        json_data = {"jsonrpc": "2.0", "id": 1, "result": "0x"}
        resp_cm = _make_mock_response(status=200, json_data=json_data)
        session_cm = _make_mock_session(resp_cm)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await check_aave_contract_accessibility(RPC_URL, chain_id=8453)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_aave_returns_false_on_timeout(self):
        """Timeout during Aave eth_call → False (no exception propagated)."""
        with patch("aiohttp.ClientSession") as mock_cls:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
            session_cm.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = session_cm

            result = await check_aave_contract_accessibility(RPC_URL, chain_id=8453)

        assert result is False
