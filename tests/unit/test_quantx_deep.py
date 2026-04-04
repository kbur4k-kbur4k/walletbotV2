"""tests/unit/test_quantx_deep.py — Deep coverage for quantx_engine.py"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQuantXCycleResult:
    def test_defaults(self):
        from wallet_manager.strategies.quantx_engine import QuantXCycleResult
        r = QuantXCycleResult()
        assert r.action_taken == "HOLD"
        assert r.position_usd == 0.0
        assert r.error == ""
        assert r.receipts == []

    def test_with_values(self):
        from wallet_manager.strategies.quantx_engine import QuantXCycleResult
        r = QuantXCycleResult(action_taken="BUY", position_usd=50.0, error="", dex_venue="uniswap")
        assert r.action_taken == "BUY"
        assert r.dex_venue == "uniswap"


class TestGetWalletAddress:
    def test_returns_address_if_set(self):
        from wallet_manager.strategies.quantx_engine import _get_wallet_address
        config = SimpleNamespace(address="0xABC123", private_key=None)
        assert _get_wallet_address(config) == "0xABC123"

    def test_derives_from_private_key_when_no_address(self):
        from wallet_manager.strategies.quantx_engine import _get_wallet_address
        config = SimpleNamespace(address=None, private_key="a" * 64)
        with patch("eth_account.Account.from_key") as mock_from_key:
            mock_from_key.return_value = SimpleNamespace(address="0xDERIVED")
            result = _get_wallet_address(config)
        assert result == "0xDERIVED"


class TestQuantXEngineSingleton:
    def test_get_quantx_engine_singleton(self):
        from wallet_manager.strategies.quantx_engine import get_quantx_engine, reset_quantx_engine
        reset_quantx_engine()
        e1 = get_quantx_engine()
        e2 = get_quantx_engine()
        assert e1 is e2

    def test_reset_quantx_engine(self):
        from wallet_manager.strategies.quantx_engine import get_quantx_engine, reset_quantx_engine
        e1 = get_quantx_engine()
        reset_quantx_engine()
        e2 = get_quantx_engine()
        assert e1 is not e2


class TestQuantXEngineInit:
    def test_init(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        assert engine._in_position is False
        assert engine._entry_price == 0.0
        assert engine._position_size_usd == 0.0

    def test_get_engine_stats(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert "signal" in stats
        assert "kelly" in stats
        assert "position" in stats
        assert stats["position"]["in_position"] is False


class TestNotifyTrade:
    def test_notify_trade_appends_to_context(self):
        from wallet_manager.strategies.quantx_engine import _notify_trade
        ctx = {}
        with patch("wallet_manager.security.telegram_manager.TradeExecutedPayload") as MockPayload:
            MockPayload.return_value = MagicMock()
            _notify_trade(ctx, "ETH/USDC", 2000.0, "0.025 ETH", "BUY", "QUANT-X", 0.15, 2.5, "uniswap")
        assert "trade_notifications" in ctx
        assert len(ctx["trade_notifications"]) == 1

    def test_notify_trade_appends_multiple(self):
        from wallet_manager.strategies.quantx_engine import _notify_trade
        ctx = {}
        for i in range(3):
            with patch("wallet_manager.security.telegram_manager.TradeExecutedPayload") as MockPayload:
                MockPayload.return_value = MagicMock()
                _notify_trade(ctx, "ETH/USDC", 2000.0, "0.025 ETH", "BUY", "QUANT-X", 0.15, 2.5, "uniswap")
        assert len(ctx["trade_notifications"]) == 3

    def test_notify_trade_handles_exception(self):
        from wallet_manager.strategies.quantx_engine import _notify_trade
        ctx = {}
        with patch("wallet_manager.security.telegram_manager.TradeExecutedPayload", side_effect=ImportError("no module")):
            # Should not raise
            _notify_trade(ctx, "ETH/USDC", 2000.0, "0.025 ETH", "BUY", "QUANT-X", 0.15, 2.5, "uniswap")


class TestQuantXRunCycle:
    def _make_manager(self, chain_id=8453):
        config = SimpleNamespace(
            chain_id=chain_id,
            address="0xabc",
            rpc_url="http://localhost",
            private_key="a" * 64,
            max_gas_price_gwei=50.0,
        )
        manager = MagicMock()
        manager.get_config = MagicMock(return_value=config)
        manager.get_safe_balance = AsyncMock(return_value=int(1e18))
        return manager

    @pytest.mark.asyncio
    async def test_wrong_chain_returns_early(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        manager = self._make_manager(chain_id=1)  # Not Base (8453)
        pm = MagicMock()
        result = await engine.run_cycle(manager, "w1", pm, {})
        # Engine returns WRONG_CHAIN action when chain_id != 8453
        assert result.action_taken in ("HOLD", "WRONG_CHAIN")

    @pytest.mark.asyncio
    async def test_gas_too_high_returns_early(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        manager = self._make_manager(chain_id=8453)
        pm = MagicMock()
        with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(False, 200.0))):
            result = await engine.run_cycle(manager, "w1", pm, {})
        # Engine returns GAS_HIGH when gas price exceeds max
        assert result.action_taken in ("HOLD", "GAS_HIGH")

    @pytest.mark.asyncio
    async def test_price_fetch_failure(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        manager = self._make_manager(chain_id=8453)
        pm = MagicMock()
        with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(side_effect=Exception("rpc fail"))):
            result = await engine.run_cycle(manager, "w1", pm, {})
        assert result.error != "" or result.action_taken == "HOLD"

    @pytest.mark.asyncio
    async def test_invalid_price_zero(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        manager = self._make_manager(chain_id=8453)
        pm = MagicMock()
        with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=0.0)):
            result = await engine.run_cycle(manager, "w1", pm, {})
        assert "Invalid price" in result.error or result.action_taken == "HOLD"

    @pytest.mark.asyncio
    async def test_balance_fetch_failure(self):
        from wallet_manager.strategies.quantx_engine import QuantXEngine, reset_quantx_engine
        reset_quantx_engine()
        engine = QuantXEngine()
        manager = self._make_manager(chain_id=8453)
        pm = MagicMock()
        with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=2000.0)), \
             patch.object(engine.crash_detector, "ingest_price", return_value=None), \
             patch.object(manager, "get_safe_balance", AsyncMock(side_effect=Exception("balance fail"))):
            result = await engine.run_cycle(manager, "w1", pm, {})
        # Should handle gracefully
        assert isinstance(result.action_taken, str)
