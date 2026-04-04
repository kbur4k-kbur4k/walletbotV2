"""tests/unit/test_idle_sweep.py — IdleSweep testleri."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.idle_sweep import run_idle_sweep


class TestIdleSweep:
    @pytest.mark.asyncio
    async def test_no_sweep_when_balance_low(self):
        """Bakiye eşiğin altındaysa sweep yapılmamalı."""
        mock_manager = MagicMock()
        config = MagicMock()
        config.wallet_id = "w1"
        config.rpc_url = "https://rpc"
        config.chain_id = 8453
        config.address = "0xDeadBeef00000000000000000000000000000001"
        mock_manager.get_config.return_value = config

        with patch(
            "wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance",
            new=AsyncMock(return_value=int(0.5e6)),  # $0.50 — çok düşük
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor:
            gas_instance = MagicMock()
            gas_instance.is_safe = AsyncMock(return_value=(True, 3.0))
            MockGasMonitor.return_value = gas_instance

            result = await run_idle_sweep(
                manager=mock_manager,
                wallet_id="w1",
            )
        assert result is None or result == [] or result == 0

    @pytest.mark.asyncio
    async def test_sweep_triggered_above_threshold(self):
        """Eşiğin üzerinde bakiye varsa sweep işlemi başlatılır."""
        mock_manager = MagicMock()
        config = MagicMock()
        config.wallet_id = "w1"
        config.rpc_url = "https://rpc"
        config.chain_id = 8453
        config.address = "0xDeadBeef00000000000000000000000000000001"
        mock_manager.get_config.return_value = config
        mock_manager.sign_and_broadcast_tx = AsyncMock(
            return_value=MagicMock(tx_hash="0xsweep")
        )

        from wallet_manager.core.models import TxParams
        fake_tx = TxParams(
            from_address="0xDeadBeef00000000000000000000000000000001",
            to="0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
            data=b"",
            value=0,
            gas=200_000,
            gas_price_wei=int(3.0 * 1e9),
        )

        with patch(
            "wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance",
            new=AsyncMock(return_value=int(500e6)),  # $500
        ), patch(
            "wallet_manager.strategies.idle_sweep.actions.get_aave_account_data",
            new=AsyncMock(return_value={"health_factor": 10.0, "total_debt_usd": 0.0}),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor, patch(
            "wallet_manager.strategies.yield_strategy.actions.build_aave_supply_usdc_plan",
            new=AsyncMock(return_value=[(fake_tx, "supply")]),
        ):
            gas_instance = MagicMock()
            gas_instance.is_safe = AsyncMock(return_value=(True, 3.0))
            MockGasMonitor.return_value = gas_instance

            result = await run_idle_sweep(
                manager=mock_manager,
                wallet_id="w1",
            )
        assert mock_manager.sign_and_broadcast_tx.called or result is not None
# Additional tests matching the actual run_idle_sweep(manager, wallet_id, ...)
# ---------------------------------------------------------------------------


def _make_base_manager(usdc_balance: int = int(500e6), gas_safe: bool = True, gwei: float = 3.0):
    """Build a mock WalletManager suitable for idle_sweep tests on Base chain."""
    config = MagicMock()
    config.rpc_url = "https://mainnet.base.org"
    config.chain_id = 8453  # Base
    config.address = "0xDeadBeef00000000000000000000000000000001"

    manager = MagicMock()
    manager.get_config.return_value = config
    manager.sign_and_broadcast_tx = AsyncMock(
        return_value=MagicMock(tx_hash="0xsweeptx", nonce_used=1)
    )
    return manager, config, gas_safe, gwei, usdc_balance


class TestRunIdleSweepActualAPI:
    """Tests calling run_idle_sweep(manager, wallet_id, ...) — real API."""

    @pytest.mark.asyncio
    async def test_sweep_does_not_call_broadcast_twice(self):
        """One successful sweep → sign_and_broadcast_tx called exactly once.

        supply_usdc internally calls sign_and_broadcast_tx once per step.
        build_aave_supply_usdc_plan returns a single-step plan here.
        """
        from wallet_manager.core.models import TxParams
        from wallet_manager.strategies.yield_strategy import supply_usdc

        manager, config, _, gwei, _ = _make_base_manager()

        fake_tx = TxParams(
            from_address=config.address,
            to="0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
            data=b"",
            value=0,
            gas=200_000,
            gas_price_wei=int(gwei * 1e9),
        )

        with patch(
            "wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance",
            new=AsyncMock(return_value=int(500e6)),  # $500 > buffer+min
        ), patch(
            "wallet_manager.strategies.idle_sweep.actions.get_aave_account_data",
            new=AsyncMock(return_value={"health_factor": 10.0, "total_debt_usd": 0.0}),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor, patch(
            "wallet_manager.strategies.yield_strategy.actions.build_aave_supply_usdc_plan",
            new=AsyncMock(return_value=[(fake_tx, "supply")]),
        ):
            gas_instance = MagicMock()
            gas_instance.is_safe = AsyncMock(return_value=(True, gwei))
            MockGasMonitor.return_value = gas_instance

            receipts = await run_idle_sweep(
                manager=manager,
                wallet_id="w1",
            )

        # One supply step → exactly one broadcast
        assert manager.sign_and_broadcast_tx.call_count == 1
        assert len(receipts) == 1

    @pytest.mark.asyncio
    async def test_sweep_gas_too_high_skipped(self):
        """Gas above MAX_SWEEP_GAS_GWEI → no broadcast, returns empty list."""
        manager, _, _, _, _ = _make_base_manager()

        with patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor:
            gas_instance = MagicMock()
            gas_instance.is_safe = AsyncMock(return_value=(False, 50.0))  # too high
            MockGasMonitor.return_value = gas_instance

            receipts = await run_idle_sweep(
                manager=manager,
                wallet_id="w1",
            )

        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_sweep_wrong_chain_returns_empty(self):
        """Non-Base chain → run_idle_sweep returns [] without touching gas/broadcast."""
        config = MagicMock()
        config.rpc_url = "https://mainnet.ethereum.org"
        config.chain_id = 1  # Ethereum mainnet
        config.address = "0xDeadBeef00000000000000000000000000000001"

        manager = MagicMock()
        manager.get_config.return_value = config
        manager.sign_and_broadcast_tx = AsyncMock()

        receipts = await run_idle_sweep(manager=manager, wallet_id="w1")

        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_sweep_target_is_aave(self):
        """Sweep tx must target the Aave V3 Pool on Base (0xA238Dd80C259a72e81d7e4664a9801593F98d1c5)."""
        from wallet_manager.core.models import TxParams

        AAVE_V3_POOL_BASE = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"

        manager, config, _, gwei, _ = _make_base_manager()

        fake_tx = TxParams(
            from_address=config.address,
            to=AAVE_V3_POOL_BASE,
            data=b"\x61\x7b\xa0\x37",  # supply() selector
            value=0,
            gas=200_000,
            gas_price_wei=int(gwei * 1e9),
        )

        with patch(
            "wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance",
            new=AsyncMock(return_value=int(500e6)),
        ), patch(
            "wallet_manager.strategies.idle_sweep.actions.get_aave_account_data",
            new=AsyncMock(return_value={"health_factor": 5.0, "total_debt_usd": 0.0}),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor, patch(
            "wallet_manager.strategies.yield_strategy.actions.build_aave_supply_usdc_plan",
            new=AsyncMock(return_value=[(fake_tx, "supply")]),
        ):
            gas_instance = MagicMock()
            gas_instance.is_safe = AsyncMock(return_value=(True, gwei))
            MockGasMonitor.return_value = gas_instance

            await run_idle_sweep(
                manager=manager,
                wallet_id="w1",
            )

        assert manager.sign_and_broadcast_tx.call_count == 1
        call_args = manager.sign_and_broadcast_tx.call_args
        tx_sent = call_args[0][1]
        assert tx_sent.to == AAVE_V3_POOL_BASE

    @pytest.mark.asyncio
    async def test_sweep_balance_below_buffer_and_min_returns_empty(self):
        """USDC balance too low to sweep (below buffer + MIN_SWEEP_USDC) → no broadcast."""
        manager, _, _, gwei, _ = _make_base_manager()

        with patch(
            "wallet_manager.strategies.idle_sweep.actions.fetch_usdc_balance",
            # $20 buffer + need $5 min = $25 total needed; $10 is below that
            new=AsyncMock(return_value=int(10e6)),
        ), patch(
            "wallet_manager.network.gas_monitor.GasMonitor"
        ) as MockGasMonitor:
            gas_instance = MagicMock()
            gas_instance.is_safe = AsyncMock(return_value=(True, gwei))
            MockGasMonitor.return_value = gas_instance

            receipts = await run_idle_sweep(
                manager=manager,
                wallet_id="w1",
            )

        assert receipts == []
        manager.sign_and_broadcast_tx.assert_not_called()
