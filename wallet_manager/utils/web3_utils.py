"""
wallet_manager/utils/web3_utils.py

Shared web3.py helpers that don't belong to any specific strategy module.
"""
from __future__ import annotations


async def close_w3_provider(w3: object) -> None:
    """Best-effort cleanup of the aiohttp session inside an AsyncHTTPProvider.

    web3.py's AsyncHTTPProvider creates an aiohttp.ClientSession internally.
    Without explicit cleanup, Python's garbage collector emits
    "Unclosed client session" ResourceWarnings.

    Tries ``provider.disconnect()`` first (web3.py >= 6), then falls back to
    directly closing the ``_session`` attribute.
    """
    try:
        provider = getattr(w3, "provider", None)
        if provider is None:
            return
        disconnect = getattr(provider, "disconnect", None)
        if callable(disconnect):
            await disconnect()
            return
        session = getattr(provider, "_session", None)
        if session is not None and not getattr(session, "closed", True):
            await session.close()
    except Exception:
        pass
