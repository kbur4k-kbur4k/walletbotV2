"""
wallet_manager/network/session_factory.py

Factory for isolated aiohttp.ClientSession instances — one per wallet.
Each session has its own TCPConnector (no shared pools) and a unique
User-Agent derived via one-way hash from the wallet ID.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

import aiohttp


class SessionFactory:
    """Static factory for isolated :class:`aiohttp.ClientSession` instances."""

    _UA_VERSION_MIN: int = 90
    _UA_VERSION_MAX: int = 115

    # Per-wallet session registry (class-level, shared across all callers)
    _sessions: dict[str, aiohttp.ClientSession] = {}
    # Lock is created lazily inside a running event loop
    _lock: asyncio.Lock | None = None

    @staticmethod
    def build(
        wallet_id: str,
        total_timeout: float = 30.0,
        connect_timeout: float = 10.0,
    ) -> aiohttp.ClientSession:
        """Create a fully isolated aiohttp session for *wallet_id*.

        Args:
            wallet_id: Wallet identifier used to seed the User-Agent hash.
            total_timeout: Total request timeout in seconds.
            connect_timeout: TCP connect timeout in seconds.

        Returns:
            A new :class:`aiohttp.ClientSession` with a private connector.
        """
        user_agent = SessionFactory._derive_user_agent(wallet_id)

        # Each session owns its connector — no cross-wallet connection pools
        connector = aiohttp.TCPConnector(
            ssl=True,
            limit=10,
            limit_per_host=5,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
        )

        return aiohttp.ClientSession(
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            connector=connector,
            timeout=timeout,
            raise_for_status=False,
        )

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Return the class-level lock, creating it lazily inside a running loop."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def get_session(
        cls,
        wallet_id: str,
        total_timeout: float = 30.0,
        connect_timeout: float = 10.0,
    ) -> aiohttp.ClientSession:
        """Return an existing session for *wallet_id*, or create a new one.

        Sessions are cached per wallet_id.  A new
        :class:`aiohttp.TCPConnector` with connection pooling is created for
        each wallet to guarantee per-wallet isolation.

        Args:
            wallet_id: Wallet identifier used as the session cache key.
            total_timeout: Total request timeout in seconds.
            connect_timeout: TCP connect timeout in seconds.

        Returns:
            An open :class:`aiohttp.ClientSession` for *wallet_id*.
        """
        lock = cls._get_lock()
        async with lock:
            existing = cls._sessions.get(wallet_id)
            if existing is not None and not existing.closed:
                return existing

            connector = aiohttp.TCPConnector(
                ssl=True,
                limit=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=total_timeout,
                connect=connect_timeout,
            )
            user_agent = cls._derive_user_agent(wallet_id)
            session = aiohttp.ClientSession(
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                connector=connector,
                timeout=timeout,
                raise_for_status=False,
            )
            cls._sessions[wallet_id] = session
            return session

    @classmethod
    async def close_all(cls) -> None:
        """Close all managed sessions gracefully and clear the session registry."""
        lock = cls._get_lock()
        async with lock:
            for wallet_id, session in list(cls._sessions.items()):
                if not session.closed:
                    await session.close()
            cls._sessions.clear()

    @staticmethod
    def _derive_user_agent(wallet_id: str) -> str:
        """Build a pseudo-random but deterministic browser-like User-Agent.

        Args:
            wallet_id: Source string for the deterministic hash component.

        Returns:
            User-Agent string in a common browser format.
        """
        uid_hash = hashlib.sha256(wallet_id.encode("utf-8")).hexdigest()[:8]
        uid_int = int(uid_hash, 16)
        version_range = SessionFactory._UA_VERSION_MAX - SessionFactory._UA_VERSION_MIN + 1
        version = SessionFactory._UA_VERSION_MIN + (uid_int % version_range)
        return (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{version}.0.0.0 Safari/537.36 "
            f"AssetManager/{uid_hash}"
        )
