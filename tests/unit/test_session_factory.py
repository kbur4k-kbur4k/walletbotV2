"""tests/unit/test_session_factory.py — tests for network/session_factory.py SessionFactory."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from wallet_manager.network.session_factory import SessionFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_agent(session: aiohttp.ClientSession) -> str:
    """Extract the User-Agent from a session's default headers."""
    # aiohttp stores default headers in session._default_headers
    headers = session.headers
    return headers.get("User-Agent", "")


# ---------------------------------------------------------------------------
# test_build_returns_aiohttp_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_returns_aiohttp_session():
    """SessionFactory.build() returns an aiohttp.ClientSession instance."""
    session = SessionFactory.build("w1")
    try:
        assert isinstance(session, aiohttp.ClientSession)
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# test_different_wallets_produce_different_user_agents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_wallets_produce_different_user_agents():
    """Sessions built for different wallet IDs have different User-Agent headers
    (the hash suffix embedded in the UA differs between IDs)."""
    session_w1 = SessionFactory.build("wallet_alpha")
    session_w2 = SessionFactory.build("wallet_beta")
    try:
        ua_w1 = _get_user_agent(session_w1)
        ua_w2 = _get_user_agent(session_w2)
        assert ua_w1 != ua_w2, (
            f"Expected different UAs for different wallet IDs.\n"
            f"  wallet_alpha: {ua_w1}\n"
            f"  wallet_beta:  {ua_w2}"
        )
    finally:
        await session_w1.close()
        await session_w2.close()


# ---------------------------------------------------------------------------
# test_same_wallet_consistent_user_agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_wallet_consistent_user_agent():
    """Two sessions built for the same wallet ID have matching AssetManager
    hash suffixes in their User-Agent (the hash component is deterministic)."""
    session_a = SessionFactory.build("deterministic_wallet")
    session_b = SessionFactory.build("deterministic_wallet")
    try:
        ua_a = _get_user_agent(session_a)
        ua_b = _get_user_agent(session_b)

        # Extract the AssetManager/<hash> suffix which is fully deterministic
        def _extract_hash_suffix(ua: str) -> str:
            for part in ua.split():
                if part.startswith("AssetManager/"):
                    return part
            return ""

        assert _extract_hash_suffix(ua_a) == _extract_hash_suffix(ua_b), (
            f"Hash suffix should be identical for same wallet ID.\n"
            f"  Session A: {ua_a}\n"
            f"  Session B: {ua_b}"
        )
        assert _extract_hash_suffix(ua_a) != "", "AssetManager hash suffix must be present"
    finally:
        await session_a.close()
        await session_b.close()


# ---------------------------------------------------------------------------
# test_session_has_timeout_configured
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_has_timeout_configured():
    """Build with specific timeout values. Session connector is set (not None)
    and session timeout matches the supplied values."""
    total_timeout = 45.0
    connect_timeout = 8.0

    session = SessionFactory.build(
        "timeout_wallet",
        total_timeout=total_timeout,
        connect_timeout=connect_timeout,
    )
    try:
        # Connector must be set (each session owns its own TCPConnector)
        assert session.connector is not None
        assert isinstance(session.connector, aiohttp.TCPConnector)

        # Timeout object is accessible via session.timeout
        assert session.timeout.total == total_timeout
        assert session.timeout.connect == connect_timeout
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# test_session_cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_cleanup():
    """After calling session.close(), session.closed is True."""
    session = SessionFactory.build("cleanup_wallet")
    assert not session.closed, "Session should be open immediately after build"

    await session.close()

    assert session.closed, "Session should be closed after calling close()"


# ---------------------------------------------------------------------------
# test_user_agent_contains_asset_manager_hash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_agent_contains_asset_manager_hash():
    """The User-Agent string contains the 'AssetManager/<hash>' component
    derived from the wallet ID."""
    wallet_id = "hash_check_wallet"
    session = SessionFactory.build(wallet_id)
    try:
        ua = _get_user_agent(session)
        assert "AssetManager/" in ua, (
            f"Expected 'AssetManager/<hash>' in User-Agent, got: {ua}"
        )
        # Verify the embedded hash is 8 hex characters (first 8 of sha256)
        import hashlib
        expected_hash = hashlib.sha256(wallet_id.encode("utf-8")).hexdigest()[:8]
        assert f"AssetManager/{expected_hash}" in ua, (
            f"Expected 'AssetManager/{expected_hash}' in User-Agent, got: {ua}"
        )
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# test_session_default_headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_default_headers():
    """Built session has Accept and Content-Type headers set to application/json."""
    session = SessionFactory.build("headers_wallet")
    try:
        headers = session.headers
        assert headers.get("Accept") == "application/json"
        assert headers.get("Content-Type") == "application/json"
    finally:
        await session.close()
