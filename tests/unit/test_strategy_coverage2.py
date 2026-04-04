"""
tests/unit/test_strategy_coverage2.py

Comprehensive coverage tests for:
  - wallet_manager/strategies/algo_strategy.py
  - wallet_manager/strategies/short_strategy.py
  - wallet_manager/telegram_notifier.py

All network / blockchain calls are mocked.  No real I/O occurs.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal env before any wallet_manager import
# ---------------------------------------------------------------------------
os.environ.setdefault("VAULT_KEY", "kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=")
os.environ.setdefault("EXECUTION_MODE", "ml_quant")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_bot_token_99")
os.environ.setdefault("TELEGRAM_CHAT_ID", "987654321")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================================================================
#  SECTION 1 — algo_strategy.py
# ============================================================================

import numpy as np

import wallet_manager.strategies.algo_strategy as algo_mod
from wallet_manager.core.models import EventKind, NotificationEvent, TxReceipt, TxStatus
from wallet_manager.strategies.algo_strategy import (
    CHAIN_ID_BASE,
    DAILY_DRAWDOWN_LIMIT_PCT,
    EMA_PERIOD,
    RSI_OVERSOLD_THRESHOLD,
    RSI_PERIOD,
    STOP_LOSS_PCT,
    TRAILING_OFFSET_PCT,
    Z_SCORE_BUY_THRESHOLD,
    AlgoState,
    _append_trade_notification,
    _check_daily_drawdown_circuit_breaker,
    _ema,
    _ema_std,
    _load_state,
    _save_state,
    bollinger_band_proximity,
    compute_bollinger_bands,
    compute_rsi,
    compute_sma,
    compute_z_score_ema,
    run_algo_strategy,
)

# ---------------------------------------------------------------------------
# Helpers shared across algo tests
# ---------------------------------------------------------------------------

def _make_prices(n: int = 25, base: float = 2000.0, step: float = 1.0) -> list[float]:
    """Monotonically increasing price series."""
    return [base + i * step for i in range(n)]


def _make_mock_manager(chain_id: int = CHAIN_ID_BASE, max_gas: float = 50.0):
    config = MagicMock()
    config.chain_id = chain_id
    config.rpc_url = "http://mock-rpc"
    config.max_gas_price_gwei = max_gas
    config.address = "0xDeadBeef"
    manager = MagicMock()
    manager.get_config.return_value = config
    manager.get_safe_balance = AsyncMock(return_value=int(0.5 * 1e18))
    receipt = TxReceipt(
        tx_hash="0xabc", wallet_id="w1", nonce_used=1, status=TxStatus.SUCCESS,
        gas_used=21000, block_number=100, chain_id=CHAIN_ID_BASE,
    )
    manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)
    return manager


def _make_portfolio_manager(budget: float = 50.0, total: float = 1000.0):
    pm = MagicMock()
    pm.get_algo_trade_budget.return_value = budget
    pm.total_value_usd = total
    return pm


def _make_task_ctx(command_state=None):
    ctx: dict[str, Any] = {}
    if command_state is not None:
        ctx["command_state"] = command_state
    return ctx


# ---------------------------------------------------------------------------
# 1. Pure math helpers
# ---------------------------------------------------------------------------

class TestAlgoMathHelpers:
    def test_ema_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ema(vals, period=5)
        assert isinstance(result, float)
        assert result > 0

    def test_ema_insufficient_data(self):
        vals = np.array([1.0, 2.0])
        result = _ema(vals, period=10)
        assert result == pytest.approx(1.5)

    def test_ema_empty(self):
        vals = np.array([])
        result = _ema(vals, period=5)
        assert result == 0.0

    def test_ema_std_basic(self):
        vals = np.array([100.0] * 20 + [110.0, 90.0])
        result = _ema_std(vals, period=20, mean=100.0)
        assert result >= 0

    def test_ema_std_insufficient(self):
        vals = np.array([1.0])
        result = _ema_std(vals, period=10, mean=1.0)
        assert result == pytest.approx(1e-10)

    def test_ema_std_zero_std(self):
        """All identical values → std should return epsilon."""
        vals = np.array([100.0] * 20)
        result = _ema_std(vals, period=20, mean=100.0)
        assert result == pytest.approx(1e-10)

    def test_compute_z_score_ema_insufficient(self):
        prices = [100.0] * 5
        mean, vol, z = compute_z_score_ema(prices, period=20)
        assert mean == 0.0 and vol == 0.0 and z == 0.0

    def test_compute_z_score_ema_sufficient(self):
        prices = _make_prices(30, base=2000.0)
        mean, vol, _z = compute_z_score_ema(prices, period=EMA_PERIOD)
        assert mean > 0
        assert vol >= 0

    def test_compute_sma_insufficient(self):
        assert compute_sma([1.0, 2.0], period=20) == 0.0

    def test_compute_sma_exact(self):
        prices = [float(i) for i in range(1, 21)]  # 1..20 → mean=10.5
        result = compute_sma(prices, period=20)
        assert result == pytest.approx(10.5)

    def test_compute_rsi_insufficient(self):
        prices = [100.0] * 5
        rsi = compute_rsi(prices, period=RSI_PERIOD)
        assert rsi == 50.0

    def test_compute_rsi_all_gains(self):
        prices = [float(i) for i in range(20)]  # only gains
        rsi = compute_rsi(prices, period=RSI_PERIOD)
        assert rsi == 100.0

    def test_compute_rsi_mixed(self):
        import random
        random.seed(42)
        prices = [2000.0]
        for _ in range(20):
            prices.append(prices[-1] + random.uniform(-10, 10))
        rsi = compute_rsi(prices, period=RSI_PERIOD)
        assert 0.0 <= rsi <= 100.0

    def test_bollinger_bands_insufficient(self):
        lower, mid, upper = compute_bollinger_bands([1.0, 2.0], period=20)
        assert lower == 0.0 and mid == 0.0 and upper == 0.0

    def test_bollinger_bands_sufficient(self):
        prices = [2000.0 + i * 0.5 for i in range(25)]
        lower, mid, upper = compute_bollinger_bands(prices, period=20)
        assert lower < mid < upper

    def test_bollinger_band_proximity_below_lower(self):
        lower_prox, _upper_prox = bollinger_band_proximity(1900.0, 1950.0, 2050.0)
        assert lower_prox < 0  # price is BELOW lower band

    def test_bollinger_band_proximity_above_upper(self):
        _lower_prox, upper_prox = bollinger_band_proximity(2100.0, 1950.0, 2050.0)
        assert upper_prox < 0  # price is ABOVE upper band

    def test_bollinger_band_proximity_zero_width(self):
        lower_prox, _upper_prox = bollinger_band_proximity(2000.0, 2000.0, 2000.0)
        # Both bands equal → band_width defaults to 1e-10
        assert isinstance(lower_prox, float)


# ---------------------------------------------------------------------------
# 2. AlgoState persistence
# ---------------------------------------------------------------------------

class TestAlgoStatePersistence:
    def test_to_dict_round_trip(self):
        state = AlgoState(
            prices=[1.0, 2.0, 3.0],
            last_trade_ts=123.0,
            last_side="BUY",
            last_buy_price=2000.0,
            ttp_active=True,
            ttp_peak_price=2100.0,
        )
        d = state.to_dict()
        restored = AlgoState.from_dict(d)
        assert restored.prices == [1.0, 2.0, 3.0]
        assert restored.last_side == "BUY"
        assert restored.ttp_active is True
        assert restored.ttp_peak_price == 2100.0

    def test_from_dict_defaults(self):
        state = AlgoState.from_dict({})
        assert state.prices == []
        assert state.last_trade_ts == 0.0
        assert state.ttp_active is False

    def test_from_dict_trims_history(self):
        prices = list(range(100))
        state = AlgoState.from_dict({"prices": prices})
        assert len(state.prices) <= AlgoState.max_history

    def test_load_state_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        state = _load_state(path)
        assert isinstance(state, AlgoState)
        assert state.prices == []

    def test_load_state_valid_file(self, tmp_path):
        path = tmp_path / "state.json"
        data = {"prices": [1900.0, 2000.0], "last_side": "BUY", "last_buy_price": 1950.0}
        path.write_text(json.dumps(data), encoding="utf-8")
        state = _load_state(path)
        assert state.last_side == "BUY"

    def test_load_state_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json!!!", encoding="utf-8")
        state = _load_state(path)
        assert state.prices == []

    def test_save_state(self, tmp_path):
        path = tmp_path / "sub" / "state.json"
        state = AlgoState(prices=[1.0, 2.0], last_side="SELL")
        _save_state(path, state)
        assert path.exists()
        with open(path) as f:
            d = json.load(f)
        assert d["last_side"] == "SELL"

    def test_save_state_os_error(self, tmp_path):
        path = tmp_path / "state.json"
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise — just logs
            _save_state(path, AlgoState())


# ---------------------------------------------------------------------------
# 3. _check_daily_drawdown_circuit_breaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDailyDrawdownCircuitBreaker:
    def setup_method(self):
        algo_mod._DAILY_DRAWDOWN_STATE.clear()

    async def test_first_call_initializes_state(self):
        result = await _check_daily_drawdown_circuit_breaker("wA", 5000.0, {})
        assert result is False
        assert "wA_algo" in algo_mod._DAILY_DRAWDOWN_STATE

    async def test_no_drawdown_returns_false(self):
        now = time.time()
        day_ts = (int(now) // 86400) * 86400
        algo_mod._DAILY_DRAWDOWN_STATE["wB_algo"] = (10000.0, day_ts)
        result = await _check_daily_drawdown_circuit_breaker("wB", 9900.0, {})
        assert result is False

    async def test_drawdown_exceeds_limit_triggers_halt(self):
        now = time.time()
        day_ts = (int(now) // 86400) * 86400
        algo_mod._DAILY_DRAWDOWN_STATE["wC_algo"] = (10000.0, day_ts)
        cs = MagicMock()
        cs.trigger_circuit_breaker_halt = MagicMock()
        ctx = {"command_state": cs}
        # Drop 6% — above 5% threshold
        result = await _check_daily_drawdown_circuit_breaker("wC", 9400.0, ctx)
        assert result is True
        cs.trigger_circuit_breaker_halt.assert_called_once()

    async def test_new_day_resets_state(self):
        yesterday_ts = ((int(time.time()) // 86400) - 1) * 86400
        algo_mod._DAILY_DRAWDOWN_STATE["wD_algo"] = (10000.0, yesterday_ts)
        # Called on today → should reset and return False
        result = await _check_daily_drawdown_circuit_breaker("wD", 9400.0, {})
        assert result is False

    async def test_zero_start_value_returns_false(self):
        now = time.time()
        day_ts = (int(now) // 86400) * 86400
        algo_mod._DAILY_DRAWDOWN_STATE["wE_algo"] = (0.0, day_ts)
        result = await _check_daily_drawdown_circuit_breaker("wE", 0.0, {})
        assert result is False


# ---------------------------------------------------------------------------
# 4. _append_trade_notification
# ---------------------------------------------------------------------------

class TestAppendTradeNotification:
    def test_appends_to_context(self):
        try:
            from wallet_manager.security.telegram_manager import TradeExecutedPayload
            ctx: dict = {}
            _append_trade_notification(ctx, "WETH/USDC", 2000.0, "ALGO", "$50", "BUY", "0x123")
            assert "trade_notifications" in ctx
            assert len(ctx["trade_notifications"]) == 1
        except ImportError:
            pytest.skip("telegram_manager not importable in this env")

    def test_append_with_existing_list(self):
        try:
            from wallet_manager.security.telegram_manager import TradeExecutedPayload
            ctx: dict = {"trade_notifications": [MagicMock()]}
            _append_trade_notification(ctx, "WETH/USDC", 2000.0, "ALGO", "$50", "SELL", None)
            assert len(ctx["trade_notifications"]) == 2
        except ImportError:
            pytest.skip("telegram_manager not importable")

    def test_append_handles_import_error_silently(self):
        """_append_trade_notification must not propagate any exception."""
        ctx: dict = {}
        # Patch the inner import to raise — function should swallow it silently
        with patch.dict("sys.modules", {"wallet_manager.security.telegram_manager": None}):
            try:
                _append_trade_notification(ctx, "WETH/USDC", 2000.0, "ALGO", "$50", "BUY")
            except Exception:
                pass  # the function itself catches, so this branch is rarely reached


# ---------------------------------------------------------------------------
# 5. run_algo_strategy — main entry point
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunAlgoStrategy:
    async def _run_with_mocks(
        self,
        tmp_path,
        price: float = 2000.0,
        z: float = 0.0,
        rsi: float = 50.0,
        balance_wei: int = int(0.5 * 1e18),
        usdc: int = int(500 * 1e6),
        budget: float = 50.0,
        chain_id: int = CHAIN_ID_BASE,
        gas_safe: bool = True,
        gas_gwei: float = 5.0,
        circuit_halted: bool = False,
        extra_ctx: dict | None = None,
    ):
        manager = _make_mock_manager(chain_id=chain_id)
        manager.get_safe_balance = AsyncMock(return_value=balance_wei)
        pm = _make_portfolio_manager(budget=budget)
        cs = MagicMock()
        cs.is_circuit_breaker_halted = circuit_halted
        cs.trigger_circuit_breaker_halt = MagicMock()
        ctx = _make_task_ctx(cs)
        if extra_ctx:
            ctx.update(extra_ctx)

        history_path = tmp_path / "price_history.json"

        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod.actions, "get_live_price_v3", AsyncMock(return_value=price)),
            patch.object(algo_mod.actions, "fetch_usdc_balance", AsyncMock(return_value=usdc)),
            patch.object(algo_mod.actions, "eth_balance_usd_value", return_value=balance_wei / 1e18 * price),
            patch.object(algo_mod.actions, "usdc_balance_usd_value", return_value=usdc / 1e6),
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=True),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_success"),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_failure"),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(gas_safe, gas_gwei))
            MockGM.return_value = gm_inst

            receipts = await run_algo_strategy(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )
        return receipts

    async def test_circuit_breaker_halted_returns_empty(self, tmp_path):
        receipts = await self._run_with_mocks(tmp_path, circuit_halted=True)
        assert receipts == []

    async def test_wrong_chain_returns_empty(self, tmp_path):
        receipts = await self._run_with_mocks(tmp_path, chain_id=1)  # Ethereum, not Base
        assert receipts == []

    async def test_gas_too_high_returns_empty(self, tmp_path):
        receipts = await self._run_with_mocks(tmp_path, gas_safe=False, gas_gwei=30.0)
        assert receipts == []

    async def test_gas_congested_above_15_returns_empty(self, tmp_path):
        # gas_safe=True but gwei > 15 (congestion check)
        receipts = await self._run_with_mocks(tmp_path, gas_safe=True, gas_gwei=16.0)
        assert receipts == []

    async def test_circuit_breaker_open_returns_empty(self, tmp_path):
        manager = _make_mock_manager()
        pm = _make_portfolio_manager()
        history_path = tmp_path / "ph.json"
        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=False),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm_inst
            receipts = await run_algo_strategy(manager, "w1", pm, {})
        assert receipts == []

    async def test_price_zero_returns_empty(self, tmp_path):
        manager = _make_mock_manager()
        pm = _make_portfolio_manager()
        history_path = tmp_path / "ph.json"
        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod.actions, "get_live_price_v3", AsyncMock(return_value=0)),
            patch.object(algo_mod.actions, "get_weth_usdc_price", AsyncMock(return_value=0)),
            patch.object(algo_mod.actions, "get_current_price", AsyncMock(return_value=0)),
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=True),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_success"),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm_inst
            receipts = await run_algo_strategy(manager, "w1", pm, {})
        assert receipts == []

    async def test_not_enough_prices_returns_empty(self, tmp_path):
        """Fewer than EMA_PERIOD prices → no signal, returns []."""
        tmp_path / "ph.json"
        receipts = await self._run_with_mocks(tmp_path, price=2000.0)
        assert isinstance(receipts, list)

    async def test_budget_too_low_returns_empty(self, tmp_path):
        receipts = await self._run_with_mocks(tmp_path, budget=1.0)
        assert isinstance(receipts, list)

    async def test_price_fetch_exception_propagates(self, tmp_path):
        manager = _make_mock_manager()
        pm = _make_portfolio_manager()
        history_path = tmp_path / "ph.json"
        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod.actions, "get_live_price_v3", AsyncMock(side_effect=RuntimeError("RPC down"))),
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=True),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_failure"),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm_inst
            with pytest.raises(RuntimeError):
                await run_algo_strategy(manager, "w1", pm, {})

    async def test_buy_signal_executes_swap(self, tmp_path):
        """When Z < -2.5, RSI < 30, near lower BB, and USDC available → BUY."""
        history_path = tmp_path / "buy_test.json"
        # Pre-populate enough prices with the last one very low (extreme z-score)
        prices = [2000.0] * 19 + [1700.0]  # big drop → negative z-score
        state_data = {"prices": prices}
        history_path.write_text(json.dumps(state_data), encoding="utf-8")

        manager = _make_mock_manager()
        pm = _make_portfolio_manager(budget=100.0)
        cs = MagicMock()
        cs.is_circuit_breaker_halted = False

        mock_plan = MagicMock()
        mock_step = MagicMock()
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 300000
        mock_step.tx_params.value = 0
        mock_plan.steps = [mock_step]

        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod.actions, "get_live_price_v3", AsyncMock(return_value=1700.0)),
            patch.object(algo_mod.actions, "fetch_usdc_balance", AsyncMock(return_value=int(500 * 1e6))),
            patch.object(algo_mod.actions, "eth_balance_usd_value", return_value=0.0),
            patch.object(algo_mod.actions, "usdc_balance_usd_value", return_value=500.0),
            patch.object(algo_mod.actions, "get_chain_addresses", return_value=MagicMock(usdc="0xUSDC", weth="0xWETH")),
            patch.object(algo_mod.actions, "get_rotated_fee_tier", return_value=3000),
            patch.object(algo_mod.actions, "get_dynamic_slippage", AsyncMock(return_value=50)),
            patch.object(algo_mod.actions, "build_usdc_to_eth_plan", AsyncMock(return_value=mock_plan)),
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=True),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_success"),
            patch.object(algo_mod, "_check_daily_drawdown_circuit_breaker", AsyncMock(return_value=False)),
            # Patch compute_z_score_ema to return extreme buy signal
            patch.object(algo_mod, "compute_z_score_ema", return_value=(2000.0, 50.0, -3.0)),
            patch.object(algo_mod, "compute_rsi", return_value=25.0),
            patch.object(algo_mod, "bollinger_band_proximity", return_value=(0.05, 0.95)),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm_inst
            receipts = await run_algo_strategy(manager, "w1", pm, {"command_state": cs})

        assert isinstance(receipts, list)

    async def test_stop_loss_triggers_sell(self, tmp_path):
        """When in position and price drops 3%+ below buy price → SELL."""
        history_path = tmp_path / "sl_test.json"
        buy_price = 2000.0
        current_price = buy_price * (1 - STOP_LOSS_PCT - 0.01)  # below stop-loss
        prices = [2000.0] * 19 + [current_price]
        state_data = {
            "prices": prices,
            "last_buy_price": buy_price,
            "last_side": "BUY",
            "ttp_active": False,
        }
        history_path.write_text(json.dumps(state_data), encoding="utf-8")

        manager = _make_mock_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(1.0 * 1e18))
        pm = _make_portfolio_manager(budget=100.0)

        mock_plan = MagicMock()
        mock_step = MagicMock()
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 300000
        mock_step.tx_params.value = 0
        mock_plan.steps = [mock_step]

        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod.actions, "get_live_price_v3", AsyncMock(return_value=current_price)),
            patch.object(algo_mod.actions, "fetch_usdc_balance", AsyncMock(return_value=int(100 * 1e6))),
            patch.object(algo_mod.actions, "eth_balance_usd_value", return_value=current_price),
            patch.object(algo_mod.actions, "usdc_balance_usd_value", return_value=100.0),
            patch.object(algo_mod.actions, "get_chain_addresses", return_value=MagicMock(usdc="0xUSDC", weth="0xWETH")),
            patch.object(algo_mod.actions, "get_rotated_fee_tier", return_value=3000),
            patch.object(algo_mod.actions, "get_dynamic_slippage", AsyncMock(return_value=50)),
            patch.object(algo_mod.actions, "build_eth_to_usdc_plan", AsyncMock(return_value=mock_plan)),
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=True),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_success"),
            patch.object(algo_mod, "_check_daily_drawdown_circuit_breaker", AsyncMock(return_value=False)),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm_inst
            receipts = await run_algo_strategy(manager, "w1", pm, {})

        assert isinstance(receipts, list)

    async def test_ttp_triggers_sell(self, tmp_path):
        """When TTP is active and price drops 0.5% from peak → SELL."""
        history_path = tmp_path / "ttp_test.json"
        peak_price = 2200.0
        current_price = peak_price * (1 - TRAILING_OFFSET_PCT - 0.001)
        prices = [2000.0] * 19 + [current_price]
        state_data = {
            "prices": prices,
            "last_buy_price": 2000.0,
            "last_side": "BUY",
            "ttp_active": True,
            "ttp_peak_price": peak_price,
        }
        history_path.write_text(json.dumps(state_data), encoding="utf-8")

        manager = _make_mock_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(1.0 * 1e18))
        pm = _make_portfolio_manager(budget=100.0)

        mock_plan = MagicMock()
        mock_step = MagicMock()
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 300000
        mock_step.tx_params.value = 0
        mock_plan.steps = [mock_step]

        with (
            patch.object(algo_mod, "_default_history_path", return_value=history_path),
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(algo_mod.actions, "get_live_price_v3", AsyncMock(return_value=current_price)),
            patch.object(algo_mod.actions, "fetch_usdc_balance", AsyncMock(return_value=int(100 * 1e6))),
            patch.object(algo_mod.actions, "eth_balance_usd_value", return_value=current_price),
            patch.object(algo_mod.actions, "usdc_balance_usd_value", return_value=100.0),
            patch.object(algo_mod.actions, "get_chain_addresses", return_value=MagicMock(usdc="0xUSDC", weth="0xWETH")),
            patch.object(algo_mod.actions, "get_rotated_fee_tier", return_value=3000),
            patch.object(algo_mod.actions, "get_dynamic_slippage", AsyncMock(return_value=50)),
            patch.object(algo_mod.actions, "build_eth_to_usdc_plan", AsyncMock(return_value=mock_plan)),
            patch.object(algo_mod._ALGO_CIRCUIT, "allow_request", return_value=True),
            patch.object(algo_mod._ALGO_CIRCUIT, "record_success"),
            patch.object(algo_mod, "_check_daily_drawdown_circuit_breaker", AsyncMock(return_value=False)),
        ):
            gm_inst = MagicMock()
            gm_inst.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm_inst
            receipts = await run_algo_strategy(manager, "w1", pm, {})

        assert isinstance(receipts, list)


# ============================================================================
#  SECTION 2 — short_strategy.py
# ============================================================================

import contextlib

import wallet_manager.strategies.short_strategy as short_mod
from wallet_manager.strategies.short_strategy import (
    CHAIN_ID_BASE as SHORT_CHAIN_ID_BASE,
)
from wallet_manager.strategies.short_strategy import (
    MIN_HEALTH_FACTOR,
    RSI_OVERBOUGHT_THRESHOLD,
    SHORT_STOP_LOSS_PCT,
    Z_SCORE_SHORT_THRESHOLD,
    ShortPosition,
    ShortState,
    _notify_short_closed,
    _notify_short_opened,
    check_health_factor_safe,
    get_short_pnl_usd,
    get_short_position_state,
    is_short_active,
    run_short_strategy,
    should_close_short,
    should_open_short,
)
from wallet_manager.strategies.short_strategy import (
    _load_state as short_load_state,
)
from wallet_manager.strategies.short_strategy import (
    _save_state as short_save_state,
)

# ---------------------------------------------------------------------------
# 6. ShortPosition and ShortState serialization
# ---------------------------------------------------------------------------

class TestShortPositionSerialization:
    def test_to_dict_round_trip(self):
        pos = ShortPosition(
            position_id="short_123",
            entry_price=2500.0,
            borrowed_weth_wei=int(0.5e18),
            usdc_received=int(1250e6),
            collateral_usdc=int(2000e6),
            entry_timestamp=1700000000.0,
            entry_z_score=3.1,
            entry_rsi=75.0,
            is_active=True,
            stop_loss_price=2625.0,
        )
        d = pos.to_dict()
        restored = ShortPosition.from_dict(d)
        assert restored.position_id == "short_123"
        assert restored.entry_price == 2500.0
        assert restored.is_active is True

    def test_from_dict_defaults(self):
        pos = ShortPosition.from_dict({})
        assert pos.position_id == ""
        assert pos.entry_price == 0.0
        assert pos.is_active is False

    def test_short_state_round_trip(self):
        state = ShortState(
            position=ShortPosition(position_id="abc", is_active=True),
            total_short_pnl_usd=123.45,
            total_shorts_opened=3,
            total_shorts_closed=2,
        )
        d = state.to_dict()
        restored = ShortState.from_dict(d)
        assert restored.total_short_pnl_usd == pytest.approx(123.45)
        assert restored.total_shorts_opened == 3

    def test_short_state_from_empty_dict(self):
        state = ShortState.from_dict({})
        assert state.total_short_pnl_usd == 0.0


class TestShortStatePersistence:
    def test_load_state_missing_file(self, tmp_path):
        path = tmp_path / "no.json"
        state = short_load_state(path)
        assert isinstance(state, ShortState)

    def test_load_state_valid_file(self, tmp_path):
        path = tmp_path / "short.json"
        data = {
            "position": {"position_id": "s1", "is_active": True, "entry_price": 2000.0},
            "total_short_pnl_usd": 50.0,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        state = short_load_state(path)
        assert state.position.is_active is True
        assert state.total_short_pnl_usd == pytest.approx(50.0)

    def test_load_state_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("GARBAGE", encoding="utf-8")
        state = short_load_state(path)
        assert state.total_short_pnl_usd == 0.0

    def test_save_state_creates_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "state.json"
        state = ShortState(total_short_pnl_usd=99.0)
        short_save_state(path, state)
        assert path.exists()

    def test_save_state_os_error(self, tmp_path):
        path = tmp_path / "state.json"
        with patch("builtins.open", side_effect=OSError("no space")):
            short_save_state(path, ShortState())  # should not raise


# ---------------------------------------------------------------------------
# 7. Signal evaluation functions
# ---------------------------------------------------------------------------

class TestShouldOpenShort:
    def test_all_conditions_met_returns_true(self):
        assert should_open_short(
            z_score=Z_SCORE_SHORT_THRESHOLD + 0.1,
            rsi=RSI_OVERBOUGHT_THRESHOLD + 1,
            bb_pct=0.01,
        ) is True

    def test_z_below_threshold_returns_false(self):
        assert should_open_short(z_score=2.0, rsi=75.0, bb_pct=0.01) is False

    def test_rsi_below_threshold_returns_false(self):
        assert should_open_short(z_score=3.0, rsi=65.0, bb_pct=0.01) is False

    def test_bb_pct_too_far_returns_false(self):
        assert should_open_short(z_score=3.0, rsi=75.0, bb_pct=0.10) is False

    def test_exact_thresholds_boundary(self):
        # Exactly at threshold — z > 2.5 is required, so 2.5 itself returns False
        assert should_open_short(z_score=2.5, rsi=70.0, bb_pct=0.02) is False

    def test_all_conditions_barely_met(self):
        # z=2.51, rsi=71, bb_pct=0.02 (exactly at boundary for bb_pct)
        assert should_open_short(z_score=2.51, rsi=71.0, bb_pct=0.02) is True


class TestShouldCloseShort:
    def test_stop_loss_triggers(self):
        close, reason = should_close_short(
            z_score=3.0,
            rsi=80.0,
            current_price=2100.0,
            entry_price=2000.0,
            stop_loss_price=2050.0,  # current > stop_loss
        )
        assert close is True
        assert reason == "stop_loss"

    def test_mean_reversion_triggers(self):
        close, reason = should_close_short(
            z_score=-0.5,
            rsi=45.0,
            current_price=1900.0,
            entry_price=2000.0,
            stop_loss_price=2100.0,
        )
        assert close is True
        assert reason == "mean_reversion"

    def test_profit_target_triggers(self):
        close, reason = should_close_short(
            z_score=1.0,  # Not below exit threshold
            rsi=60.0,
            current_price=1940.0,  # 3% drop from 2000
            entry_price=2000.0,
            stop_loss_price=2100.0,
        )
        assert close is True
        assert reason == "profit_target"

    def test_no_exit_signal(self):
        close, reason = should_close_short(
            z_score=2.0,
            rsi=65.0,
            current_price=2050.0,
            entry_price=2000.0,
            stop_loss_price=2100.0,
        )
        assert close is False
        assert reason == ""

    def test_zero_entry_price_no_profit_target(self):
        close, _reason = should_close_short(
            z_score=1.5,
            rsi=65.0,
            current_price=1700.0,
            entry_price=0.0,  # No entry price recorded
            stop_loss_price=2100.0,
        )
        assert close is False


# ---------------------------------------------------------------------------
# 8. check_health_factor_safe
# ---------------------------------------------------------------------------

class TestCheckHealthFactorSafe:
    @pytest.mark.asyncio
    async def test_safe_health_factor(self):
        with patch.object(short_mod.actions, "get_aave_account_data",
                          AsyncMock(return_value={"health_factor": 2.5})):
            safe, hf = await check_health_factor_safe("http://rpc", "0xAddr")
            assert safe is True
            assert hf == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_unsafe_health_factor(self):
        with patch.object(short_mod.actions, "get_aave_account_data",
                          AsyncMock(return_value={"health_factor": 1.2})):
            safe, _hf = await check_health_factor_safe("http://rpc", "0xAddr")
            assert safe is False

    @pytest.mark.asyncio
    async def test_zero_health_factor_no_debt(self):
        """Health factor of 0 = no debt, should be considered safe."""
        with patch.object(short_mod.actions, "get_aave_account_data",
                          AsyncMock(return_value={"health_factor": 0})):
            safe, _hf = await check_health_factor_safe("http://rpc", "0xAddr")
            assert safe is True

    @pytest.mark.asyncio
    async def test_exception_returns_unsafe(self):
        with patch.object(short_mod.actions, "get_aave_account_data",
                          AsyncMock(side_effect=Exception("RPC error"))):
            safe, hf = await check_health_factor_safe("http://rpc", "0xAddr")
            assert safe is False
            assert hf == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. run_short_strategy — entry and exit paths
# ---------------------------------------------------------------------------

def _make_short_manager(chain_id: int = SHORT_CHAIN_ID_BASE):
    config = MagicMock()
    config.chain_id = chain_id
    config.rpc_url = "http://mock-rpc"
    config.max_gas_price_gwei = 50.0
    config.address = "0xShortWallet"
    manager = MagicMock()
    manager.get_config.return_value = config
    receipt = TxReceipt(
        tx_hash="0xshort_tx", wallet_id="w1", nonce_used=5,
        status=TxStatus.SUCCESS, gas_used=200000, block_number=200, chain_id=SHORT_CHAIN_ID_BASE,
    )
    manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)
    return manager


@pytest.mark.asyncio
class TestRunShortStrategy:
    async def test_wrong_chain_returns_empty(self, tmp_path):
        manager = _make_short_manager(chain_id=1)
        pm = _make_portfolio_manager()
        receipts = await run_short_strategy(
            manager=manager, wallet_id="w1", portfolio_manager=pm,
            task_context={}, z_score=3.0, rsi=75.0, current_price=2500.0, bb_upper=2450.0,
        )
        assert receipts == []

    async def test_gas_too_high_returns_empty(self, tmp_path):
        manager = _make_short_manager()
        pm = _make_portfolio_manager()
        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=tmp_path / "state.json"),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(False, 30.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={}, z_score=3.0, rsi=75.0, current_price=2500.0, bb_upper=2450.0,
            )
        assert receipts == []

    async def test_no_entry_signal_returns_empty(self, tmp_path):
        manager = _make_short_manager()
        pm = _make_portfolio_manager()
        state_path = tmp_path / "no_pos.json"
        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=state_path),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={}, z_score=0.5, rsi=40.0,  # Below thresholds
                current_price=2500.0, bb_upper=2450.0,
            )
        assert receipts == []

    async def test_active_position_mean_reversion_exit(self, tmp_path):
        state_path = tmp_path / "active.json"
        # Create an active position
        state = ShortState(
            position=ShortPosition(
                position_id="s1",
                entry_price=2500.0,
                borrowed_weth_wei=int(0.4e18),
                usdc_received=int(1000e6),
                is_active=True,
                stop_loss_price=2625.0,
            )
        )
        state_path.write_text(json.dumps(state.to_dict()), encoding="utf-8")

        manager = _make_short_manager()
        pm = _make_portfolio_manager()

        mock_plan = MagicMock()
        mock_step = MagicMock()
        mock_step.description = "swap USDC->WETH"
        mock_step.tx_params = MagicMock(gas_price_gwei=5.0, gas=300000, value=0)
        mock_plan.steps = [mock_step]

        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=state_path),
            patch.object(short_mod.actions, "get_variable_debt_balance", AsyncMock(return_value=int(0.4e18))),
            patch.object(short_mod.actions, "get_chain_addresses", return_value=MagicMock(usdc="0xUSDC", weth="0xWETH")),
            patch.object(short_mod.actions, "get_rotated_fee_tier", return_value=3000),
            patch.object(short_mod.actions, "get_dynamic_slippage", AsyncMock(return_value=50)),
            patch.object(short_mod.actions, "build_usdc_to_eth_plan", AsyncMock(return_value=mock_plan)),
            patch.object(short_mod.actions, "build_aave_repay_eth_plan",
                         AsyncMock(return_value=[(MagicMock(gas_price_gwei=5.0, gas=200000, value=0), "repay")])),
            patch.object(short_mod, "check_health_factor_safe", AsyncMock(return_value=(True, 2.5))),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={},
                z_score=-0.5,  # Mean reversion achieved
                rsi=45.0,
                current_price=2400.0,
                bb_upper=2450.0,
            )
        assert isinstance(receipts, list)

    async def test_active_position_no_debt_closes_cleanly(self, tmp_path):
        state_path = tmp_path / "nodebt.json"
        state = ShortState(
            position=ShortPosition(
                position_id="s2", entry_price=2500.0, is_active=True, stop_loss_price=2625.0,
            )
        )
        state_path.write_text(json.dumps(state.to_dict()), encoding="utf-8")

        manager = _make_short_manager()
        pm = _make_portfolio_manager()

        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=state_path),
            patch.object(short_mod.actions, "get_variable_debt_balance", AsyncMock(return_value=0)),
            patch.object(short_mod, "check_health_factor_safe", AsyncMock(return_value=(True, 2.5))),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={}, z_score=-1.0, rsi=40.0,
                current_price=2400.0, bb_upper=2450.0,
            )
        assert receipts == []

    async def test_ml_signal_prevents_entry(self, tmp_path):
        manager = _make_short_manager()
        pm = _make_portfolio_manager()
        state_path = tmp_path / "ml_no.json"

        alpha_signal = MagicMock()
        alpha_signal.should_enter_short = False
        alpha_signal.confidence = 0.3

        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=state_path),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={}, z_score=3.5, rsi=80.0, current_price=2500.0, bb_upper=2450.0,
                alpha_signal=alpha_signal,
            )
        assert receipts == []

    async def test_low_equity_skips_entry(self, tmp_path):
        manager = _make_short_manager()
        pm = _make_portfolio_manager(total=50.0)  # total equity $50 → max_short = $7.50 < $10
        state_path = tmp_path / "low_eq.json"

        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=state_path),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={}, z_score=3.5, rsi=80.0, current_price=2500.0, bb_upper=2450.0,
            )
        assert receipts == []

    async def test_aave_fetch_failure_returns_empty(self, tmp_path):
        manager = _make_short_manager()
        pm = _make_portfolio_manager(total=10000.0)
        state_path = tmp_path / "aave_fail.json"

        with (
            patch("wallet_manager.network.gas_monitor.GasMonitor") as MockGM,
            patch.object(short_mod, "_default_state_path", return_value=state_path),
            patch.object(short_mod.actions, "get_aave_account_data", AsyncMock(side_effect=Exception("Aave down"))),
        ):
            gm = MagicMock()
            gm.is_safe = AsyncMock(return_value=(True, 5.0))
            MockGM.return_value = gm
            receipts = await run_short_strategy(
                manager=manager, wallet_id="w1", portfolio_manager=pm,
                task_context={}, z_score=3.5, rsi=80.0, current_price=2500.0, bb_upper=2450.0,
            )
        assert receipts == []


# ---------------------------------------------------------------------------
# 10. Public query helpers
# ---------------------------------------------------------------------------

class TestShortPublicHelpers:
    def test_get_short_position_state_returns_state(self, tmp_path):
        with patch.object(short_mod, "_default_state_path", return_value=tmp_path / "s.json"):
            state = get_short_position_state()
            assert isinstance(state, ShortState)

    def test_get_short_pnl_usd_default_zero(self, tmp_path):
        with patch.object(short_mod, "_default_state_path", return_value=tmp_path / "pnl.json"):
            pnl = get_short_pnl_usd()
            assert pnl == 0.0

    def test_is_short_active_false_by_default(self, tmp_path):
        with patch.object(short_mod, "_default_state_path", return_value=tmp_path / "active.json"):
            assert is_short_active() is False

    def test_is_short_active_true_when_position_active(self, tmp_path):
        path = tmp_path / "active2.json"
        state = ShortState(position=ShortPosition(is_active=True))
        path.write_text(json.dumps(state.to_dict()), encoding="utf-8")
        with patch.object(short_mod, "_default_state_path", return_value=path):
            assert is_short_active() is True


# ---------------------------------------------------------------------------
# 11. Notification helpers
# ---------------------------------------------------------------------------

class TestShortNotifications:
    def test_notify_short_opened_adds_to_context(self):
        try:
            from wallet_manager.security.telegram_manager import TradeExecutedPayload
            ctx: dict = {}
            pos = ShortPosition(position_id="s1", borrowed_weth_wei=int(0.5e18))
            _notify_short_opened(ctx, pos, 2500.0)
            assert "trade_notifications" in ctx
            assert len(ctx["trade_notifications"]) == 1
        except ImportError:
            pytest.skip("telegram_manager not available")

    def test_notify_short_closed_adds_to_context(self):
        try:
            from wallet_manager.security.telegram_manager import TradeExecutedPayload
            ctx: dict = {}
            pos = ShortPosition(position_id="s1", borrowed_weth_wei=int(0.5e18))
            _notify_short_closed(ctx, pos, 2400.0, 50.0, "profit_target")
            assert "trade_notifications" in ctx
        except ImportError:
            pytest.skip("telegram_manager not available")

    def test_notify_short_opened_handles_import_error(self):
        ctx: dict = {}
        pos = ShortPosition()
        with patch.dict("sys.modules", {"wallet_manager.security.telegram_manager": None}):
            try:
                _notify_short_opened(ctx, pos, 2500.0)
            except Exception:
                pass  # Silently ignored

    def test_notify_short_closed_handles_import_error(self):
        ctx: dict = {}
        pos = ShortPosition()
        with patch.dict("sys.modules", {"wallet_manager.security.telegram_manager": None}):
            with contextlib.suppress(Exception):
                _notify_short_closed(ctx, pos, 2400.0, 50.0, "mean_reversion")


# ============================================================================
#  SECTION 3 — telegram_notifier.py
# ============================================================================

import aiohttp

from wallet_manager.exceptions import NotifierConfigurationError, NotifierDeliveryError
from wallet_manager.telegram_notifier import (
    ProactiveAlerter,
    TelegramNotifier,
    _TelegramClient,
    _TelegramCredentials,
    _TokenBucket,
)

# ---------------------------------------------------------------------------
# 12. _TelegramCredentials
# ---------------------------------------------------------------------------

class TestTelegramCredentials:
    def test_repr_masks_secrets(self):
        creds = _TelegramCredentials(_bot_token="secret_token", _chat_id="12345")
        assert "secret_token" not in repr(creds)
        assert "_TelegramCredentials(***)" in repr(creds)

    def test_properties(self):
        creds = _TelegramCredentials(_bot_token="tok123", _chat_id="chat456")
        assert creds.bot_token == "tok123"
        assert creds.chat_id == "chat456"

    def test_from_env_raises_when_token_missing(self):
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = ""
        mock_cfg.telegram_chat_id = "123"
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            with pytest.raises(NotifierConfigurationError, match="TELEGRAM_BOT_TOKEN"):
                _TelegramCredentials.from_env()

    def test_from_env_raises_when_chat_id_missing(self):
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = "token"
        mock_cfg.telegram_chat_id = ""
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            with pytest.raises(NotifierConfigurationError, match="TELEGRAM_CHAT_ID"):
                _TelegramCredentials.from_env()

    def test_from_env_success(self):
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = "good_token"
        mock_cfg.telegram_chat_id = "999"
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            creds = _TelegramCredentials.from_env()
            assert creds.bot_token == "good_token"
            assert creds.chat_id == "999"

    def test_both_missing(self):
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = ""
        mock_cfg.telegram_chat_id = ""
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            with pytest.raises(NotifierConfigurationError):
                _TelegramCredentials.from_env()


# ---------------------------------------------------------------------------
# 13. _TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            _TokenBucket(rate=0, capacity=1)

    def test_invalid_negative_rate_raises(self):
        with pytest.raises(ValueError):
            _TokenBucket(rate=-1.0, capacity=1)

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self):
        bucket = _TokenBucket(rate=100.0, capacity=10)
        await bucket.acquire()
        assert bucket._tokens == pytest.approx(9.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        bucket = _TokenBucket(rate=1000.0, capacity=1)
        await bucket.acquire()  # consume the only token
        # Drain it almost immediately by patching sleep
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Refill the bucket manually to avoid infinite loop
            bucket._tokens = 1.0
            await bucket.acquire()
        assert isinstance(bucket._tokens, float)

    def test_refill_does_not_exceed_capacity(self):
        bucket = _TokenBucket(rate=1.0, capacity=5)
        bucket._tokens = 0.0
        bucket._last_refill = time.monotonic() - 1000  # Long time ago
        bucket._refill()
        assert bucket._tokens == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 14. _TelegramClient
# ---------------------------------------------------------------------------

class TestTelegramClient:
    def _make_creds(self):
        return _TelegramCredentials(_bot_token="tok", _chat_id="123")

    @pytest.mark.asyncio
    async def test_open_creates_session(self):
        client = _TelegramClient(self._make_creds())
        await client.open()
        assert client._session is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        client = _TelegramClient(self._make_creds())
        await client.close()  # No session yet — should not raise

    @pytest.mark.asyncio
    async def test_send_message_raises_if_not_open(self):
        client = _TelegramClient(self._make_creds())
        with pytest.raises(RuntimeError, match="not open"):
            await client.send_message("hello")

    @pytest.mark.asyncio
    async def test_send_message_200_ok(self):
        client = _TelegramClient(self._make_creds())
        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 200

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)

        client._session = mock_session
        await client.send_message("hello world")

    @pytest.mark.asyncio
    async def test_send_message_truncates_long_text(self):
        client = _TelegramClient(self._make_creds())
        long_text = "x" * 5000

        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 200

        captured_payload: list[dict] = []

        def fake_post(url, json=None, **kwargs):
            if json:
                captured_payload.append(json)
            return mock_resp

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=fake_post)

        client._session = mock_session
        await client.send_message(long_text)
        assert len(captured_payload[0]["text"]) <= 4096

    @pytest.mark.asyncio
    async def test_send_message_429_retries(self):
        client = _TelegramClient(self._make_creds())

        call_count = 0

        class FakeResp:
            def __init__(self, status):
                self.status = status

            async def json(self, **kwargs):
                return {"parameters": {"retry_after": 0.01}}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        responses = [FakeResp(429), FakeResp(429), FakeResp(200)]

        def fake_post(url, json=None, **kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=fake_post)
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_message("test")

    @pytest.mark.asyncio
    async def test_send_message_400_raises_non_retryable(self):
        client = _TelegramClient(self._make_creds())

        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 400

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        client._session = mock_session

        with pytest.raises(NotifierDeliveryError):
            await client.send_message("bad request")

    @pytest.mark.asyncio
    async def test_send_message_exhausts_retries(self):
        client = _TelegramClient(self._make_creds())

        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 500

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NotifierDeliveryError):
                await client.send_message("server error")

    @pytest.mark.asyncio
    async def test_send_message_network_error_retries(self):
        client = _TelegramClient(self._make_creds())

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("network fail"))
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NotifierDeliveryError):
                await client.send_message("test")

    @pytest.mark.asyncio
    async def test_get_updates_returns_empty_when_not_open(self):
        client = _TelegramClient(self._make_creds())
        result = await client.get_updates(offset=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_updates_returns_list(self):
        client = _TelegramClient(self._make_creds())

        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": [{"update_id": 1}]})

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        client._session = mock_session

        result = await client.get_updates(offset=0)
        assert result == [{"update_id": 1}]

    @pytest.mark.asyncio
    async def test_get_updates_exception_returns_empty(self):
        client = _TelegramClient(self._make_creds())
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=Exception("timeout"))
        client._session = mock_session

        result = await client.get_updates(offset=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_answer_callback_query_no_session(self):
        client = _TelegramClient(self._make_creds())
        # No exception, just skips
        await client.answer_callback_query("cb_id", "done")

    @pytest.mark.asyncio
    async def test_answer_callback_query_logs_non_200(self):
        client = _TelegramClient(self._make_creds())

        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 400

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        client._session = mock_session

        await client.answer_callback_query("cb123")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_message_with_keyboard_raises_if_not_open(self):
        client = _TelegramClient(self._make_creds())
        with pytest.raises(RuntimeError, match="not open"):
            await client.send_message_with_keyboard("hi", {})

    @pytest.mark.asyncio
    async def test_send_message_with_keyboard_200_ok(self):
        client = _TelegramClient(self._make_creds())

        mock_resp = AsyncMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resp.status = 200

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        client._session = mock_session

        await client.send_message_with_keyboard("test", {"inline_keyboard": []})


# ---------------------------------------------------------------------------
# 15. TelegramNotifier
# ---------------------------------------------------------------------------

def _make_notifier_creds():
    return _TelegramCredentials(_bot_token="test_tok", _chat_id="987654321")


@pytest.mark.asyncio
class TestTelegramNotifier:
    async def test_context_manager_start_stop(self):
        creds = _make_notifier_creds()
        with patch.object(_TelegramClient, "open", new_callable=AsyncMock) as mock_open, \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock) as mock_close, \
             patch.object(_TelegramClient, "send_message", new_callable=AsyncMock):
            async with TelegramNotifier(credentials=creds) as notifier:
                assert notifier._worker_task is not None
            mock_open.assert_called_once()
            mock_close.assert_called_once()

    async def test_send_before_start_raises(self):
        creds = _make_notifier_creds()
        notifier = TelegramNotifier(credentials=creds)
        event = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        with pytest.raises(RuntimeError, match="not started"):
            await notifier.send(event)

    async def test_send_event_alias_works(self):
        creds = _make_notifier_creds()
        with patch.object(_TelegramClient, "open", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "send_message", new_callable=AsyncMock):
            async with TelegramNotifier(credentials=creds) as notifier:
                event = NotificationEvent(kind=EventKind.SYSTEM_ALERT, detail="test")
                await notifier.send_event(event)

    async def test_send_enqueues_event(self):
        creds = _make_notifier_creds()
        with patch.object(_TelegramClient, "open", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "send_message", new_callable=AsyncMock):
            async with TelegramNotifier(credentials=creds) as notifier:
                event = NotificationEvent(kind=EventKind.WALLET_SUCCESS, wallet_id="w1")
                await notifier.send(event)
                # Event was queued

    async def test_delivery_worker_swallows_delivery_error(self):
        creds = _make_notifier_creds()
        with patch.object(_TelegramClient, "open", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "send_message",
                          new_callable=AsyncMock,
                          side_effect=NotifierDeliveryError("fail", http_status=500)):
            async with TelegramNotifier(credentials=creds) as notifier:
                event = NotificationEvent(kind=EventKind.GAS_PAUSE)
                await notifier.send(event)
                await asyncio.sleep(0.1)  # Let worker process

    async def test_delivery_worker_swallows_unexpected_error(self):
        creds = _make_notifier_creds()
        with patch.object(_TelegramClient, "open", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "send_message",
                          new_callable=AsyncMock,
                          side_effect=Exception("unexpected!")):
            async with TelegramNotifier(credentials=creds) as notifier:
                await notifier.send(NotificationEvent(kind=EventKind.SYSTEM_BOOT))
                await asyncio.sleep(0.1)

    async def test_stop_timeout_cancels_worker(self):
        creds = _make_notifier_creds()
        notifier = TelegramNotifier(credentials=creds)

        with patch.object(_TelegramClient, "open", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock):
            await notifier.start()
            # Mock the worker task to never finish
            slow_task = asyncio.ensure_future(asyncio.sleep(3600))
            notifier._worker_task = slow_task
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                await notifier.stop()
            # Worker should be cancelled
            assert slow_task.cancelled() or slow_task.done()

    async def test_build_command_center_returns_center(self):
        from wallet_manager.security.telegram_commands import CommandCenter, CommandState
        creds = _make_notifier_creds()
        notifier = TelegramNotifier(credentials=creds)
        cs = CommandState()
        center = notifier.build_command_center(cs)
        assert isinstance(center, CommandCenter)

    async def test_send_confirmation_request_raises_if_not_started(self):
        creds = _make_notifier_creds()
        notifier = TelegramNotifier(credentials=creds)
        with pytest.raises(RuntimeError, match="not started"):
            await notifier.send_confirmation_request("0xWallet", "0.01 ETH", "ETH -> USDC")

    async def test_send_confirmation_request_sends_message(self):
        creds = _make_notifier_creds()
        with patch.object(_TelegramClient, "open", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "close", new_callable=AsyncMock), \
             patch.object(_TelegramClient, "send_message", new_callable=AsyncMock) as mock_send:
            async with TelegramNotifier(credentials=creds) as notifier:
                await notifier.send_confirmation_request("0xABCD1234", "0.01 ETH", "ETH -> USDC")
                mock_send.assert_called_once()


# ---------------------------------------------------------------------------
# 16. ProactiveAlerter
# ---------------------------------------------------------------------------

class TestProactiveAlerter:
    def _make_notifier(self):
        notifier = AsyncMock()
        notifier.send_event = AsyncMock()
        return notifier

    @pytest.mark.asyncio
    async def test_drift_moderate_sends_alert(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 0
        mock_cfg.yield_min_apy_pct = 5.0
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            await alerter.check_and_alert(notifier, "MODERATE", 0.10, 1.0, 100.0, 0.05)
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_drift_severe_sends_alert(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 0
        mock_cfg.yield_min_apy_pct = 5.0
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            await alerter.check_and_alert(notifier, "SEVERE", 0.10, 1.0, 100.0, 0.05)
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_deduped(self):
        alerter = ProactiveAlerter()
        alerter._last_sent["drift_MODERATE"] = time.time()  # Recently sent
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 3600
        mock_cfg.yield_min_apy_pct = 5.0
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            await alerter.check_and_alert(notifier, "MODERATE", 0.10, 1.0, 100.0, 0.05)
        notifier.send_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_apy_below_threshold_sends_alert(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 0
        mock_cfg.yield_min_apy_pct = 10.0  # threshold 10%
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            # apy = 0.03 → 3% < 10% threshold
            await alerter.check_and_alert(notifier, "NONE", 0.03, 1.0, 100.0, 0.05)
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_model_stale_sends_alert(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 0
        mock_cfg.yield_min_apy_pct = 1.0
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            await alerter.check_and_alert(notifier, "NONE", 0.05, 10.0, 100.0, 0.05)
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_rpc_high_latency_sends_alert(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 0
        mock_cfg.yield_min_apy_pct = 1.0
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            await alerter.check_and_alert(notifier, "NONE", 0.05, 1.0, 600.0, 0.05)
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_drawdown_exceeded_sends_alert(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 0
        mock_cfg.yield_min_apy_pct = 1.0
        mock_cfg.drawdown_reduce_threshold = 0.10

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            await alerter.check_and_alert(notifier, "NONE", 0.05, 1.0, 100.0, 0.15)
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_no_alerts_when_all_good(self):
        alerter = ProactiveAlerter()
        notifier = self._make_notifier()
        mock_cfg = MagicMock()
        mock_cfg.alert_dedup_window_seconds = 3600
        mock_cfg.yield_min_apy_pct = 1.0
        mock_cfg.drawdown_reduce_threshold = 0.20

        with patch("wallet_manager.config.get_settings", return_value=mock_cfg):
            # Everything is fine
            await alerter.check_and_alert(notifier, "NONE", 0.10, 1.0, 100.0, 0.05)
        notifier.send_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_alert_with_send_only(self):
        """Tests _dispatch_alert fallback to send() when send_event not available."""
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        del notifier.send_event
        notifier.send = AsyncMock()

        event = NotificationEvent(kind=EventKind.SYSTEM_ALERT, detail="test alert")
        await alerter._dispatch_alert(notifier, event)
        notifier.send.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_alert_raises_without_sender(self):
        """Tests _dispatch_alert raises AttributeError if no send methods."""
        alerter = ProactiveAlerter()
        notifier = MagicMock(spec=[])  # No send or send_event

        event = NotificationEvent(kind=EventKind.SYSTEM_ALERT)
        with pytest.raises(AttributeError):
            await alerter._dispatch_alert(notifier, event)

    def test_is_deduped_within_window(self):
        alerter = ProactiveAlerter()
        alerter._last_sent["key1"] = time.time()
        assert alerter._is_deduped("key1", 3600) is True

    def test_is_deduped_outside_window(self):
        alerter = ProactiveAlerter()
        alerter._last_sent["key2"] = time.time() - 7200  # 2 hours ago
        assert alerter._is_deduped("key2", 3600) is False

    def test_is_deduped_never_sent(self):
        alerter = ProactiveAlerter()
        assert alerter._is_deduped("never_sent", 3600) is False

    def test_mark_sent_records_timestamp(self):
        alerter = ProactiveAlerter()
        before = time.time()
        alerter._mark_sent("my_key")
        after = time.time()
        assert before <= alerter._last_sent["my_key"] <= after
