"""
wallet_manager/network/gas_monitor.py

Async gas-price monitor for the Base (or any EVM-compatible) network.

GasMonitor polls ``eth_gasPrice`` via a direct JSON-RPC call and compares
the result against a configurable ceiling expressed in Gwei.  Each call
uses a short-lived aiohttp.ClientSession (async with) so no sessions are
left open.

Usage
-----
::

    monitor = GasMonitor(rpc_url="https://mainnet.base.org")
    safe, current_gwei = await monitor.is_safe(max_gwei=30.0)
    if not safe:
        print(f"Gas too high: {current_gwei:.1f} Gwei")
"""

from __future__ import annotations

import aiohttp


class GasMonitor:
    """Async gas price monitor using the RPC endpoint of a reference wallet.

    Polls ``eth_gasPrice`` via a direct JSON-RPC call and compares against
    a configurable ceiling.  Stateless between calls — each
    :meth:`current_gwei` invocation uses a self-closing aiohttp session.

    Args:
        rpc_url: JSON-RPC endpoint used for gas price queries.
    """

    def __init__(self, rpc_url: str) -> None:
        self._rpc_url = rpc_url

    async def current_gwei(self) -> float:
        """Fetch the current network gas price in Gwei.

        Returns:
            Gas price as a float in Gwei.

        Raises:
            Exception: Propagates network errors to the caller.
        """
        payload = {"jsonrpc": "2.0", "method": "eth_gasPrice", "params": [], "id": 1}
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._rpc_url, json=payload) as resp:
                data = await resp.json(content_type=None)
        hex_price: str = data.get("result", "0x0")
        return int(hex_price, 16) / 1e9

    async def is_safe(self, max_gwei: float) -> tuple[bool, float]:
        """Check whether the current gas price is within the safe ceiling.

        Args:
            max_gwei: Maximum acceptable gas price in Gwei.

        Returns:
            A ``(safe, current_gwei)`` tuple where ``safe`` is ``True``
            if the current price is ≤ *max_gwei*.
        """
        current = await self.current_gwei()
        return current <= max_gwei, current
