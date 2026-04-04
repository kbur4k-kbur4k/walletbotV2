"""
wallet_manager/utils/health_check.py

Startup health checks for RPC connectivity and contract accessibility.
Integrated into orchestrator._validate_startup_config().
"""

# ruff: noqa: ASYNC109

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# Aave V3 Pool contract on Base (chain ID 8453)
_AAVE_V3_POOL_BASE = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"


def _load_wallet_count(wallets_path: str) -> int:
    path = Path(wallets_path)
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else len(data.get("wallets", []))


async def check_rpc_connectivity(rpc_url: str, timeout: float = 5.0) -> bool:
    """Check RPC connectivity via eth_blockNumber.

    Args:
        rpc_url: JSON-RPC HTTP endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        True if the endpoint responds with a valid block number.
    """
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session,
            session.post(
                rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp,
        ):
            if resp.status != 200:
                log.warning("health_check: RPC %s returned HTTP %s", rpc_url[:40], resp.status)
                return False
            data = await resp.json(content_type=None)
            result = data.get("result")
            if result and isinstance(result, str) and result.startswith("0x"):
                log.debug("health_check: RPC %s OK (block %s)", rpc_url[:40], int(result, 16))
                return True
            log.warning("health_check: RPC %s unexpected result: %s", rpc_url[:40], result)
            return False
    except Exception as exc:
        log.warning("health_check: RPC %s unreachable: %s", rpc_url[:40], type(exc).__name__)
        return False


async def check_aave_contract_accessibility(rpc_url: str, chain_id: int) -> bool:
    """Check that the Aave V3 Pool contract responds on the given RPC.

    Uses eth_call to query the pool's getReservesList() method signature.

    Args:
        rpc_url: JSON-RPC HTTP endpoint URL.
        chain_id: Chain ID (must be 8453 for Base).

    Returns:
        True if the contract is accessible.
    """
    if chain_id != 8453:
        log.debug("health_check: Aave check skipped (chain_id=%s, expected 8453)", chain_id)
        return True

    # getReservesList() selector: 0xd1946dbc
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": _AAVE_V3_POOL_BASE, "data": "0xd1946dbc"}, "latest"],
        "id": 1,
    }
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session,
            session.post(
                rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp,
        ):
            if resp.status != 200:
                return False
            data = await resp.json(content_type=None)
            result = data.get("result", "")
            # A non-empty hex response means the contract responded
            if result and result != "0x":
                log.debug("health_check: Aave V3 Pool accessible on %s", rpc_url[:40])
                return True
            return False
    except Exception as exc:
        log.warning(
            "health_check: Aave contract check failed on %s: %s",
            rpc_url[:40], type(exc).__name__,
        )
        return False


async def run_startup_health_checks(settings: object) -> list[str]:
    """Run all startup health checks and return a list of problem descriptions.

    Args:
        settings: A :class:`~wallet_manager.config.Settings` instance.

    Returns:
        List of problem strings (empty = all checks passed).
    """
    problems: list[str] = []

    rpc_candidates: list[str] = []
    for attr in ("base_rpc_1", "base_rpc_2", "base_rpc_3", "private_rpc_1"):
        val = getattr(settings, attr, "")
        if val:
            rpc_candidates.append(val)

    if not rpc_candidates:
        problems.append("No Base RPC endpoints configured (BASE_RPC_1..3 or PRIVATE_RPC_1)")
        return problems

    rpc_ok = False
    for rpc_url in rpc_candidates:
        if await check_rpc_connectivity(rpc_url):
            rpc_ok = True
            # Also check Aave contract accessibility on the first working RPC
            chain_id = getattr(settings, "chain_id", 8453) if hasattr(settings, "chain_id") else 8453
            aave_ok = await check_aave_contract_accessibility(rpc_url, chain_id)
            if not aave_ok:
                problems.append(f"Aave V3 Pool contract not accessible on {rpc_url[:40]}")
            break

    if not rpc_ok:
        problems.append(
            f"All Base RPC endpoints unreachable: {', '.join(u[:30] for u in rpc_candidates)}"
        )

    return problems


# ---------------------------------------------------------------------------
# HTTP health-check handler classes (Task 6.2)
# ---------------------------------------------------------------------------


@dataclass
class _Response:
    status: int
    body: str
    content_type: str = "application/json"


class HealthCheck:
    """Simple HTTP health-check handlers for /live, /ready, /metrics/summary."""

    async def handle_live(self, request: object) -> _Response:
        """/live — always 200 while the process is running."""
        return _Response(status=200, body=json.dumps({"status": "ok"}))

    async def handle_ready(self, request: object) -> _Response:
        """/ready — 200 only when basic config looks sane, 503 otherwise."""
        rpc_url = os.environ.get("RPC_URL_PRIMARY", "")
        try:
            from wallet_manager.config import get_settings
            settings = get_settings()
            wallets_path = os.environ.get(
                "WALLETS_FILE",
                str(getattr(settings, "wallet_config_path", "wallets.json")),
            )
        except Exception:
            wallets_path = "wallets.json"

        try:
            wallet_count = await asyncio.to_thread(_load_wallet_count, wallets_path)
        except Exception:
            wallet_count = 0

        if rpc_url and wallet_count > 0:
            return _Response(status=200, body=json.dumps({"status": "ready", "wallets": wallet_count}))
        reasons = []
        if not rpc_url:
            reasons.append("RPC_URL_PRIMARY not set")
        if wallet_count == 0:
            reasons.append("no wallets configured")
        return _Response(status=503, body=json.dumps({"status": "not_ready", "reasons": reasons}))

    async def handle_metrics_summary(self, request: object) -> _Response:
        """/metrics/summary — returns a lightweight portfolio snapshot."""
        # Import shared state where available; fall back to safe defaults.
        portfolio_value = 0.0
        active_wallets = 0
        last_cycle_ts = None
        ml_confidence = None
        execution_mode = "robust_paper"

        try:
            from wallet_manager import orchestrator as _orch
            portfolio_value = float(getattr(_orch, "_portfolio_value_usd", 0.0))
            active_wallets = int(getattr(_orch, "_active_wallet_count", 0))
            last_cycle_ts = getattr(_orch, "_last_cycle_timestamp", None)
            ml_confidence = getattr(_orch, "_ml_confidence_last", None)
            execution_mode = getattr(_orch, "_execution_mode", "robust_paper")
        except Exception:
            pass

        payload = {
            "portfolio_value_usd": portfolio_value,
            "active_wallets": active_wallets,
            "last_cycle_timestamp": last_cycle_ts,
            "ml_confidence_last": ml_confidence,
            "current_execution_mode": execution_mode,
        }
        return _Response(status=200, body=json.dumps(payload))
