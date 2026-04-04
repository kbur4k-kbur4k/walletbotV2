"""tests/unit/test_execution_logic.py — Unit tests for wallet_manager.strategies.execution_logic."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import execution_logic — orchestrator is only imported at runtime inside run_cycle,
# so the module-level import here is safe.
from wallet_manager.strategies.execution_logic import (
    ExecutionCycleResult,
    MLQuantExecutionEngine,
    reset_ml_execution_engine,
)


class _FakeGasMonitor:
    """Stand-in GasMonitor for tests that need gas to be 'safe'."""

    def __init__(self, rpc_url: str = "") -> None:
        self._rpc_url = rpc_url

    async def is_safe(self, max_gwei: float) -> tuple[bool, float]:
        return True, 5.0


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_signal(
    alpha_score: float = 0.5,
    prob_up: float = 0.65,
    prob_down: float = 0.35,
    confidence: float = 0.75,
    garch_sigma: float = 0.002,
    vol_regime: str = "normal",
    should_enter_long: bool = True,
    should_enter_short: bool = False,
    should_exit: bool = False,
):
    """Build a minimal AlphaMLSignal-like namespace for tests."""
    return SimpleNamespace(
        alpha_score=alpha_score,
        prob_up=prob_up,
        prob_down=prob_down,
        confidence=confidence,
        garch_sigma=garch_sigma,
        volatility_regime=vol_regime,
        should_enter_long=should_enter_long,
        should_enter_short=should_enter_short,
        should_exit=should_exit,
        xgb_prediction=0.5,
        lgbm_prediction=0.5,
        lstm_prediction=0.5,
        features=None,
    )


def _make_wallet_config(chain_id: int = 8453, max_gas: float = 25.0):
    cfg = MagicMock()
    cfg.chain_id = chain_id
    cfg.rpc_url = "http://localhost:8545"
    cfg.address = "0xDeadBeefDeadBeefDeadBeefDeadBeefDeadBeef"
    cfg.private_key = "0x" + "a" * 64
    cfg.max_gas_price_gwei = max_gas
    return cfg


def _make_engine(signal=None, price: float = 2000.0) -> MLQuantExecutionEngine:
    """Return an MLQuantExecutionEngine with all heavy deps replaced by mocks."""
    reset_ml_execution_engine()
    engine = MLQuantExecutionEngine.__new__(MLQuantExecutionEngine)

    if signal is None:
        signal = _make_signal()

    alpha = MagicMock()
    alpha.process_tick = MagicMock(return_value=signal)
    alpha.last_signal = signal
    alpha.get_engine_stats = MagicMock(return_value={})
    alpha.set_microstructure = MagicMock()
    alpha.feature_engineer = MagicMock()
    alpha.feature_engineer._prices = [float(price)] * 30

    risk = MagicMock()
    risk.update_equity = MagicMock(return_value="normal")
    risk.calculate_position = MagicMock(
        return_value=SimpleNamespace(
            position_usd=100.0,
            position_pct=0.05,
            kelly_fraction=0.05,
            drawdown_state="normal",
            source="kelly",
            var_99_pct=0.01,
        )
    )
    risk.drawdown = MagicMock()
    risk.drawdown._peak_equity = 10_000.0
    risk.drawdown.current_drawdown = 0.0
    risk.record_trade = MagicMock()
    risk.get_stats = MagicMock(return_value={})

    sor = MagicMock()
    sor.find_best_route = AsyncMock(return_value=None)

    mev = MagicMock()
    mev.apply_jitter = AsyncMock(return_value=50.0)
    mev.get_stats = MagicMock(return_value={})

    crash = MagicMock()
    crash.is_kill_switch_active = False
    crash.ingest_price = MagicMock(return_value=None)
    crash.get_buffer_stats = MagicMock(return_value={})
    crash._threshold = 0.08
    crash._window = 300.0

    regime_kelly = MagicMock()
    regime_kelly.compute_sized_position = MagicMock(return_value=100.0)

    engine.alpha = alpha
    engine.risk = risk
    engine.sor = sor
    engine.mev = mev
    engine.crash_detector = crash
    engine.regime_kelly = regime_kelly
    engine._cycle_count = 0
    engine._total_trade_count = 0
    engine._last_retrain_trigger_ts = 0.0
    engine._retrain_controller = None
    engine._in_position = False
    engine._entry_price = 0.0
    engine._entry_mid_price = 0.0
    engine._position_size_usd = 0.0
    engine._entry_alpha = 0.0
    engine._entry_garch_sigma = 0.0
    engine._entry_vol_regime = "normal"
    engine._entry_timestamp = 0.0

    return engine


def _make_manager(chain_id: int = 8453, balance_wei: int = 0):
    manager = MagicMock()
    manager.get_config = MagicMock(return_value=_make_wallet_config(chain_id=chain_id))
    manager.get_safe_balance = AsyncMock(return_value=balance_wei)
    manager.sign_and_broadcast_tx = AsyncMock(
        return_value=SimpleNamespace(tx_hash="0xabc123", nonce_used=1, status="SUCCESS")
    )
    return manager


def _make_pm(budget: float = 200.0):
    pm = MagicMock()
    pm.get_algo_trade_budget = MagicMock(return_value=budget)
    return pm


def _make_ctx():
    return {"command_state": None, "trade_notifications": []}


def _low_vol_regime():
    return MagicMock(
        get_current_regime=MagicMock(
            return_value=SimpleNamespace(
                regime=SimpleNamespace(name="LOW_VOL", value="LOW_VOL")
            )
        )
    )


# ---------------------------------------------------------------------------
# Context manager: mock GasMonitor + price/balance side-effects
# ---------------------------------------------------------------------------

def _gas_safe_patches(price: float = 2000.0, gas_gwei: float = 5.0, safe: bool = True):
    """Return a list of (target, mock) pairs for patching the common run_cycle deps."""

    class _SafeGasMonitor:
        def __init__(self, rpc_url=""):
            pass

        async def is_safe(self, _max: float):
            return safe, gas_gwei

    return [
        patch("wallet_manager.network.gas_monitor.GasMonitor", _SafeGasMonitor),
        patch("wallet_manager.strategies.execution_logic.actions.get_live_price_v3",
              AsyncMock(return_value=price)),
        patch("wallet_manager.strategies.execution_logic.actions.get_weth_usdc_price",
              AsyncMock(return_value=price)),
        patch("wallet_manager.strategies.execution_logic.actions.get_current_price",
              AsyncMock(return_value=price)),
        patch("wallet_manager.strategies.execution_logic.actions.fetch_usdc_balance",
              AsyncMock(return_value=int(500 * 1e6))),
        patch("wallet_manager.strategies.execution_logic.actions.eth_balance_usd_value",
              MagicMock(return_value=0.0)),
        patch("wallet_manager.strategies.execution_logic.actions.usdc_balance_usd_value",
              MagicMock(return_value=500.0)),
        patch("wallet_manager.strategies.execution_logic.run_idle_sweep",
              AsyncMock(return_value=[])),
        patch("wallet_manager.strategies.execution_logic.get_regime_filter",
              MagicMock(return_value=_low_vol_regime())),
        patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance",
              AsyncMock(return_value=100.0)),
        patch("wallet_manager.strategies.signal_intelligence.get_alpha_generator",
              MagicMock(return_value=MagicMock(read_ofi=AsyncMock(return_value=None)))),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecuteReturnsTxReceiptOrNone:
    @pytest.mark.asyncio
    async def test_execute_returns_execution_cycle_result(self):
        """run_cycle() must return an ExecutionCycleResult regardless of outcome."""
        signal = _make_signal(
            alpha_score=0.0,
            should_enter_long=False,
            should_enter_short=False,
            should_exit=False,
        )
        engine = _make_engine(signal=signal)
        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert isinstance(result, ExecutionCycleResult)

    @pytest.mark.asyncio
    async def test_hold_result_has_no_receipts(self):
        """A HOLD cycle must produce an empty receipts list."""
        signal = _make_signal(
            alpha_score=0.0,
            should_enter_long=False,
            should_enter_short=False,
            should_exit=False,
        )
        engine = _make_engine(signal=signal)
        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert result.receipts == []
        assert result.action_taken == "HOLD"


class TestFlashCrashSkipsExecution:
    @pytest.mark.asyncio
    async def test_flash_crash_active_skips_execution(self):
        """When kill_switch_active=True, run_cycle returns KILL_SWITCH immediately."""
        engine = _make_engine()
        engine.crash_detector.is_kill_switch_active = True

        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        # kill_switch check runs before GasMonitor import, no patches needed
        result = await engine.run_cycle(
            manager=manager,
            wallet_id="w1",
            portfolio_manager=pm,
            task_context=ctx,
        )

        assert result.action_taken == "KILL_SWITCH"
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_halt_skips_execution(self):
        """circuit_breaker_halted=True → HALTED, no broadcast."""
        engine = _make_engine()
        command_state = MagicMock()
        command_state.is_circuit_breaker_halted = True
        ctx = {"command_state": command_state, "trade_notifications": []}

        manager = _make_manager()

        result = await engine.run_cycle(
            manager=manager,
            wallet_id="w1",
            portfolio_manager=_make_pm(),
            task_context=ctx,
        )

        assert result.action_taken == "HALTED"
        manager.sign_and_broadcast_tx.assert_not_called()


class TestZeroAlphaSkipsExecution:
    @pytest.mark.asyncio
    async def test_zero_alpha_skips_execution(self):
        """When alpha_score=0.0 and no signals, action_taken is HOLD and no broadcast."""
        signal = _make_signal(
            alpha_score=0.0,
            should_enter_long=False,
            should_enter_short=False,
            should_exit=False,
        )
        engine = _make_engine(signal=signal)
        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert result.action_taken == "HOLD"
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrong_chain_skips_execution(self):
        """chain_id != 8453 → WRONG_CHAIN, no price fetch, no broadcast."""
        engine = _make_engine()
        manager = _make_manager(chain_id=1)  # Ethereum mainnet

        result = await engine.run_cycle(
            manager=manager,
            wallet_id="w1",
            portfolio_manager=_make_pm(),
            task_context=_make_ctx(),
        )

        assert result.action_taken == "WRONG_CHAIN"
        manager.sign_and_broadcast_tx.assert_not_called()


class TestExecutionRecordsTrade:
    @pytest.mark.asyncio
    async def test_execution_records_trade_to_analytics(self):
        """run_cycle() calls quant_analytics.record_cycle() when signal is present."""
        signal = _make_signal(
            alpha_score=0.8,
            confidence=0.90,
            should_enter_long=False,   # keep it a HOLD to avoid deep BUY path
            should_enter_short=False,
            should_exit=False,
        )
        engine = _make_engine(signal=signal)
        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        quant_analytics = MagicMock()
        quant_analytics.record_cycle = AsyncMock()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
                quant_analytics=quant_analytics,
            )

        # record_cycle is invoked at step 8 of run_cycle when signal is present
        quant_analytics.record_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_analytics_not_called_when_no_signal(self):
        """When alpha.process_tick returns None (warmup), record_cycle is not called."""
        engine = _make_engine()
        engine.alpha.process_tick = MagicMock(return_value=None)  # warmup
        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        quant_analytics = MagicMock()
        quant_analytics.record_cycle = AsyncMock()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
                quant_analytics=quant_analytics,
            )

        assert result.action_taken == "WARMUP"
        quant_analytics.record_cycle.assert_not_called()


class TestHaltOnDrawdown:
    @pytest.mark.asyncio
    async def test_halt_on_drawdown_no_broadcast(self):
        """When risk engine returns position_usd=0 (drawdown halt), no tx is broadcast."""
        signal = _make_signal(alpha_score=0.9, should_enter_long=True)
        engine = _make_engine(signal=signal)

        # Force the risk engine to block the trade
        engine.risk.calculate_position = MagicMock(
            return_value=SimpleNamespace(
                position_usd=0.0,
                position_pct=0.0,
                kelly_fraction=0.0,
                drawdown_state="HALTED",
                source="drawdown_halt",
                var_99_pct=0.0,
            )
        )
        engine.regime_kelly.compute_sized_position = MagicMock(return_value=0.0)

        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert "RISK_BLOCKED" in result.action_taken
        manager.sign_and_broadcast_tx.assert_not_called()


class TestSellExecution:
    @pytest.mark.asyncio
    async def test_exit_signal_sells_full_balance_and_clears_position(self):
        signal = _make_signal(
            alpha_score=-0.4,
            should_enter_long=False,
            should_enter_short=False,
            should_exit=True,
        )
        engine = _make_engine(signal=signal)
        engine._in_position = True
        engine._entry_price = 2000.0
        engine._position_size_usd = 100.0
        engine._entry_alpha = 0.2
        engine._entry_garch_sigma = 0.002
        engine._entry_vol_regime = "normal"
        engine._entry_timestamp = 1.0

        balance_wei = int(0.5 * 1e18)
        manager = _make_manager(balance_wei=balance_wei)
        pm = _make_pm()
        ctx = _make_ctx()

        captured_amounts: list[int] = []

        async def _fake_build_plan(**kwargs):
            captured_amounts.append(kwargs["amount_wei"])
            return SimpleNamespace(
                steps=[
                    SimpleNamespace(
                        tx_params=SimpleNamespace(
                            gas_price_gwei=5.0,
                            gas=21000,
                            value=0,
                        )
                    )
                ]
            )

        patches = _gas_safe_patches(price=2100.0)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "wallet_manager.strategies.execution_logic.actions.fetch_usdc_balance",
                AsyncMock(return_value=0),
            ),
            patch(
                "wallet_manager.strategies.execution_logic.actions.eth_balance_usd_value",
                MagicMock(return_value=1050.0),
            ),
            patch(
                "wallet_manager.strategies.execution_logic.actions.usdc_balance_usd_value",
                MagicMock(return_value=0.0),
            ),
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            patch(
                "wallet_manager.strategies.execution_logic.actions.get_dynamic_slippage",
                AsyncMock(return_value=10.0),
            ),
            patch(
                "wallet_manager.strategies.execution_logic.actions.build_eth_to_usdc_plan",
                AsyncMock(side_effect=_fake_build_plan),
            ),
            patch(
                "wallet_manager.strategies.execution_logic.actions.get_chain_addresses",
                MagicMock(return_value=SimpleNamespace(weth="weth", usdc="usdc")),
            ),
            patch(
                "wallet_manager.strategies.execution_logic.actions.get_rotated_fee_tier",
                MagicMock(return_value=3000),
            ),
        ):
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert result.action_taken == "SELL"
        assert captured_amounts == [balance_wei]
        assert engine._in_position is False
        assert engine._position_size_usd == 0.0

    @pytest.mark.asyncio
    async def test_gas_high_returns_early(self):
        """Gas above the wallet's max_gas limit → GAS_HIGH, no broadcast."""
        engine = _make_engine()

        class _HighGasMonitor:
            def __init__(self, rpc_url=""):
                pass

            async def is_safe(self, _max: float):
                return False, 30.0

        manager = _make_manager()
        pm = _make_pm()
        ctx = _make_ctx()

        with patch("wallet_manager.network.gas_monitor.GasMonitor", _HighGasMonitor):
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert result.action_taken == "GAS_HIGH"
        manager.sign_and_broadcast_tx.assert_not_called()

    @pytest.mark.asyncio
    async def test_sizing_multiplier_zero_no_broadcast(self):
        """get_algo_trade_budget() returns 0 → budget ≤ 0, no trade, RISK_BLOCKED."""
        signal = _make_signal(alpha_score=0.9, should_enter_long=True)
        engine = _make_engine(signal=signal)

        # Risk says 100 USD but budget gate says 0
        engine.risk.calculate_position = MagicMock(
            return_value=SimpleNamespace(
                position_usd=0.0,   # zero position blocks the trade
                position_pct=0.0,
                kelly_fraction=0.0,
                drawdown_state="HALTED",
                source="drawdown_halt",
                var_99_pct=0.0,
            )
        )

        manager = _make_manager()
        pm = _make_pm(budget=0.0)
        ctx = _make_ctx()

        patches = _gas_safe_patches()
        with contextlib.ExitStack() as stack:

            for p in patches:
                stack.enter_context(p)
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=pm,
                task_context=ctx,
            )

        assert "RISK_BLOCKED" in result.action_taken
        manager.sign_and_broadcast_tx.assert_not_called()
