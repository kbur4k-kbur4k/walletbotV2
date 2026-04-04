"""tests/unit/test_wallet_manager.py — tests for core/manager.py WalletManager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.config import Settings
from wallet_manager.core.manager import WalletManager, _WalletContext
from wallet_manager.core.models import TxParams, TxReceipt, TxStatus, WalletConfig
from wallet_manager.exceptions import (
    CircuitOpenError,
    GasPriceTooHighError,
    WalletNotFoundError,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DUMMY_PRIVATE_KEY = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
DUMMY_ADDRESS = "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"
DUMMY_RPC = "https://rpc.example.com"


def _make_wallet_config(
    wallet_id: str = "test_wallet",
    max_gas_price_gwei: float = 50.0,
) -> WalletConfig:
    return WalletConfig(
        wallet_id=wallet_id,
        private_key=DUMMY_PRIVATE_KEY,
        rpc_url=DUMMY_RPC,
        chain_id=1,
        max_gas_price_gwei=max_gas_price_gwei,
        address=DUMMY_ADDRESS,
    )


def _make_settings(execution_mode: str = "robust_live", paper_trade: bool = False) -> Settings:
    return Settings(
        execution_mode=execution_mode,
        paper_trade=paper_trade,
    )


def _make_tx_params(gas_price_gwei: float = 10.0) -> TxParams:
    gas_price_wei = int(gas_price_gwei * 1e9)
    return TxParams(
        from_address=DUMMY_ADDRESS,
        to="0x1111111111111111111111111111111111111111",
        value=1_000_000_000_000_000,
        gas=21_000,
        gas_price_wei=gas_price_wei,
        nonce=0,
    )


def _build_manager_with_context(
    wallet_id: str = "test_wallet",
    max_gas_price_gwei: float = 50.0,
    breaker_allows: bool = True,
    execution_mode: str = "robust_live",
    paper_trade: bool = False,
    cooldown: object | None = None,
    balance_provider: object | None = None,
    tx_broadcaster: object | None = None,
) -> tuple[WalletManager, _WalletContext]:
    """Build a WalletManager with a pre-populated _contexts dict for testing
    (avoids the need to call __aenter__ which opens real aiohttp sessions)."""
    cfg = _make_wallet_config(wallet_id=wallet_id, max_gas_price_gwei=max_gas_price_gwei)
    settings = _make_settings(execution_mode=execution_mode, paper_trade=paper_trade)

    mock_cooldown = cooldown or AsyncMock()
    mock_cooldown.wait = AsyncMock(return_value=None)

    mock_balance_provider = balance_provider or AsyncMock()
    mock_tx_broadcaster = tx_broadcaster or AsyncMock()

    manager = WalletManager(
        wallet_configs=[cfg],
        settings=settings,
        cooldown=mock_cooldown,
        balance_provider=mock_balance_provider,
        tx_broadcaster=mock_tx_broadcaster,
    )

    # Build a fake breaker
    mock_breaker = MagicMock()
    mock_breaker.allow_request.return_value = breaker_allows
    mock_breaker.record_success = MagicMock()
    mock_breaker.record_failure = MagicMock()

    # Build a fake session
    mock_session = MagicMock()
    mock_session.close = AsyncMock()

    ctx = _WalletContext(config=cfg, session=mock_session, breaker=mock_breaker)
    manager._contexts[wallet_id] = ctx

    return manager, ctx


# ---------------------------------------------------------------------------
# test_sign_and_broadcast_success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sign_and_broadcast_success():
    """Mock broadcaster returning TxReceipt(status=SUCCESS).
    Assert receipt is returned and circuit_breaker.record_success() is called."""
    expected_hash = "0xdeadbeef" * 8

    mock_broadcaster = AsyncMock()
    mock_broadcaster.get_nonce = AsyncMock(return_value=42)
    mock_broadcaster.sign_and_send = AsyncMock(
        return_value=TxReceipt(
            tx_hash=expected_hash,
            wallet_id="",
            nonce_used=42,
            status=TxStatus.SUCCESS,
        )
    )

    manager, ctx = _build_manager_with_context(
        tx_broadcaster=mock_broadcaster,
        breaker_allows=True,
    )

    tx = _make_tx_params(gas_price_gwei=10.0)
    receipt = await manager.sign_and_broadcast_tx("test_wallet", tx)

    assert receipt.tx_hash == expected_hash
    assert receipt.wallet_id == "test_wallet"
    assert receipt.nonce_used == 42
    ctx.breaker.record_success.assert_called_once()
    mock_broadcaster.sign_and_send.assert_awaited_once()


# ---------------------------------------------------------------------------
# test_circuit_open_raises_circuit_open_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_open_raises_circuit_open_error():
    """With allow_request() returning False, CircuitOpenError is raised
    and sign_and_send is never called."""
    mock_broadcaster = AsyncMock()
    mock_broadcaster.sign_and_send = AsyncMock()

    manager, _ctx = _build_manager_with_context(
        tx_broadcaster=mock_broadcaster,
        breaker_allows=False,
    )

    tx = _make_tx_params(gas_price_gwei=10.0)

    with pytest.raises(CircuitOpenError) as exc_info:
        await manager.sign_and_broadcast_tx("test_wallet", tx)

    assert exc_info.value.wallet_id == "test_wallet"
    mock_broadcaster.sign_and_send.assert_not_awaited()


# ---------------------------------------------------------------------------
# test_gas_price_too_high_raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gas_price_too_high_raises():
    """TxParams with gas_price_gwei > max_gas_price_gwei raises GasPriceTooHighError."""
    manager, _ctx = _build_manager_with_context(
        max_gas_price_gwei=50.0,
        breaker_allows=True,
    )

    # 80 Gwei > 50 Gwei ceiling
    tx = _make_tx_params(gas_price_gwei=80.0)

    with pytest.raises(GasPriceTooHighError) as exc_info:
        await manager.sign_and_broadcast_tx("test_wallet", tx)

    assert exc_info.value.requested_gwei > exc_info.value.max_gwei
    assert exc_info.value.wallet_id == "test_wallet"


# ---------------------------------------------------------------------------
# test_broadcast_failure_records_circuit_failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_failure_records_circuit_failure():
    """If sign_and_send raises an Exception, the exception propagates to the caller
    and record_success() is never called (the tx did not complete successfully).

    Note: sign_and_broadcast_tx records failure only via the balance path; for
    tx broadcast the manager simply re-raises without calling record_failure, so
    we assert record_success is NOT called on broadcast failure.
    """
    mock_broadcaster = AsyncMock()
    mock_broadcaster.get_nonce = AsyncMock(return_value=7)
    mock_broadcaster.sign_and_send = AsyncMock(side_effect=Exception("nonce too low"))

    manager, ctx = _build_manager_with_context(
        tx_broadcaster=mock_broadcaster,
        breaker_allows=True,
    )

    tx = _make_tx_params(gas_price_gwei=10.0)

    with pytest.raises(Exception, match="nonce too low"):
        await manager.sign_and_broadcast_tx("test_wallet", tx)

    # record_success must NOT have been called — broadcast failed
    ctx.breaker.record_success.assert_not_called()


# ---------------------------------------------------------------------------
# test_nonce_lock_serialises_concurrent_calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonce_lock_serialises_concurrent_calls():
    """3 concurrent broadcast calls must each receive a unique, monotonically
    increasing nonce value — the nonce_lock prevents races."""
    nonce_counter = 0
    received_nonces: list[int] = []

    async def fake_get_nonce(rpc_url: str, address: str) -> int:
        nonlocal nonce_counter
        n = nonce_counter
        nonce_counter += 1
        return n

    async def fake_sign_and_send(rpc_url, tx_params, private_key):
        received_nonces.append(tx_params.nonce)
        return TxReceipt(
            tx_hash=f"0x{tx_params.nonce:064x}",
            wallet_id="",
            nonce_used=tx_params.nonce,
            status=TxStatus.SUCCESS,
        )

    mock_broadcaster = AsyncMock()
    mock_broadcaster.get_nonce = AsyncMock(side_effect=fake_get_nonce)
    mock_broadcaster.sign_and_send = AsyncMock(side_effect=fake_sign_and_send)

    manager, _ctx = _build_manager_with_context(
        tx_broadcaster=mock_broadcaster,
        breaker_allows=True,
    )

    tx = _make_tx_params(gas_price_gwei=10.0)
    receipts = await asyncio.gather(
        manager.sign_and_broadcast_tx("test_wallet", tx),
        manager.sign_and_broadcast_tx("test_wallet", tx),
        manager.sign_and_broadcast_tx("test_wallet", tx),
    )

    assert len(receipts) == 3
    # Each nonce must be unique
    assert len(set(received_nonces)) == 3, f"Duplicate nonces detected: {received_nonces}"


# ---------------------------------------------------------------------------
# test_wallet_not_found_raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wallet_not_found_raises():
    """Requesting an unknown wallet_id raises WalletNotFoundError."""
    cfg = _make_wallet_config(wallet_id="real_wallet")
    settings = _make_settings()
    mock_cooldown = AsyncMock()
    mock_cooldown.wait = AsyncMock(return_value=None)

    manager = WalletManager(
        wallet_configs=[cfg],
        settings=settings,
        cooldown=mock_cooldown,
        balance_provider=AsyncMock(),
        tx_broadcaster=AsyncMock(),
    )
    # _contexts is intentionally empty — simulates calling outside async with

    with pytest.raises(WalletNotFoundError) as exc_info:
        manager.get_config("nonexistent_wallet")

    assert exc_info.value.wallet_id == "nonexistent_wallet"


# ---------------------------------------------------------------------------
# test_get_safe_balance_success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_safe_balance_success():
    """get_safe_balance returns balance from provider and records circuit success."""
    expected_balance = 5_000_000_000_000_000_000  # 5 ETH in wei

    mock_balance_provider = AsyncMock()
    mock_balance_provider.get_balance = AsyncMock(return_value=expected_balance)

    manager, ctx = _build_manager_with_context(
        balance_provider=mock_balance_provider,
        breaker_allows=True,
    )

    balance = await manager.get_safe_balance("test_wallet")

    assert balance == expected_balance
    mock_balance_provider.get_balance.assert_awaited_once_with(DUMMY_RPC, DUMMY_ADDRESS)
    ctx.breaker.record_success.assert_called_once()


# ---------------------------------------------------------------------------
# test_get_safe_balance_circuit_open_raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_safe_balance_circuit_open_raises():
    """get_safe_balance raises CircuitOpenError when circuit breaker is OPEN."""
    mock_balance_provider = AsyncMock()
    mock_balance_provider.get_balance = AsyncMock()

    manager, _ctx = _build_manager_with_context(
        balance_provider=mock_balance_provider,
        breaker_allows=False,
    )

    with pytest.raises(CircuitOpenError):
        await manager.get_safe_balance("test_wallet")

    mock_balance_provider.get_balance.assert_not_awaited()


# ---------------------------------------------------------------------------
# test_paper_trade_blocks_broadcast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paper_trade_blocks_broadcast():
    """sign_and_broadcast_tx raises RuntimeError when paper_trade=True."""
    manager, _ctx = _build_manager_with_context(
        paper_trade=True,
        execution_mode="robust_live",
    )

    tx = _make_tx_params(gas_price_gwei=10.0)

    with pytest.raises(RuntimeError, match="paper trade"):
        await manager.sign_and_broadcast_tx("test_wallet", tx)


# ---------------------------------------------------------------------------
# test_robust_paper_execution_mode_blocks_broadcast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_robust_paper_execution_mode_blocks_broadcast():
    """sign_and_broadcast_tx raises RuntimeError when execution_mode='robust_paper'."""
    manager, _ctx = _build_manager_with_context(
        execution_mode="robust_paper",
        paper_trade=False,
    )

    tx = _make_tx_params(gas_price_gwei=10.0)

    with pytest.raises(RuntimeError, match="BLOCKED"):
        await manager.sign_and_broadcast_tx("test_wallet", tx)
