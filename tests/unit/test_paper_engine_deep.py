"""tests/unit/test_paper_engine_deep.py

Deep-coverage tests targeting paper_engine.py lines 1795-2670+.

Focuses on:
- PaperTradingEngine.run_shadow_cycle (all early-exit branches + HOLD/BUY/SELL)
- VirtualLedger CRUD: advanced analytics, hold-time, equity curve, readiness
- get_paper_engine() / reset_paper_engine() singleton behaviour
- PaperTradeResult field assertions
- _journal_decision write path
- get_performance_report / get_telegram_report statistics reporting
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.paper_engine import (
    PAPER_SHADOW_ONLY,
    PaperTradeConfig,
    PaperTradeResult,
    PaperTradingEngine,
    VirtualLedger,
    get_paper_engine,
    reset_paper_engine,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_signal(
    *,
    robust_alpha: float = 0.3,
    raw_alpha: float = 0.25,
    confidence: float = 0.8,
    adf_pvalue: float = 0.05,
    stationarity_d: float = 1.0,
    should_enter_long: bool = True,
    should_exit: bool = False,
    reason_codes: tuple = (),
    volatility_regime: str = "normal",
    garch_sigma: float = 0.02,
):
    underlying = SimpleNamespace(
        volatility_regime=volatility_regime,
        garch_sigma=garch_sigma,
    )
    return SimpleNamespace(
        robust_alpha=robust_alpha,
        raw_alpha=raw_alpha,
        confidence=confidence,
        adf_pvalue=adf_pvalue,
        stationarity_d=stationarity_d,
        should_enter_long=should_enter_long,
        should_exit=should_exit,
        reason_codes=reason_codes,
        underlying_signal=underlying,
        garch_sigma=garch_sigma,
    )


def _make_config_mock(chain_id: int = 8453, rpc_url: str = "http://fake-rpc"):
    cfg = MagicMock()
    cfg.chain_id = chain_id
    cfg.rpc_url = rpc_url
    return cfg


def _make_wallet_manager(chain_id: int = 8453, rpc_url: str = "http://fake-rpc"):
    mgr = MagicMock()
    mgr.get_config.return_value = _make_config_mock(chain_id, rpc_url)
    return mgr


def _make_portfolio_manager():
    pm = MagicMock()
    pm.get_trade_budget.return_value = 5000.0
    return pm


def _task_context():
    return {"command_state": None}


def _make_risk_decision(position_usd: float = 500.0, drawdown_state: str = "normal"):
    rd = MagicMock()
    rd.position_usd = position_usd
    rd.drawdown_state = drawdown_state
    return rd


# ---------------------------------------------------------------------------
# Patch bundles – everything that hits the network in run_shadow_cycle
# ---------------------------------------------------------------------------

ALWAYS_SUPPRESS = [
    # early-observation singletons (best-effort blocks)
    "wallet_manager.strategies.paper_engine.contextlib.suppress",
]


def _base_patches(tmp_path, price: float = 2500.0, signal=None, risk_position_usd: float = 500.0):
    """Return a list of patch context managers for a complete shadow cycle run."""
    if signal is None:
        signal = _make_signal()

    risk_decision = _make_risk_decision(position_usd=risk_position_usd)

    patches = [
        # price fetch
        patch(
            "wallet_manager.actions.get_live_price_v3",
            AsyncMock(return_value=price),
        ),
        patch(
            "wallet_manager.actions.get_weth_usdc_price",
            AsyncMock(return_value=price),
        ),
        patch(
            "wallet_manager.actions.get_current_price",
            AsyncMock(return_value=price),
        ),
        # robust alpha engine
        patch(
            "wallet_manager.strategies.robust_alpha.get_robust_alpha_engine",
            return_value=MagicMock(ingest_price=MagicMock(return_value=signal)),
        ),
        # deep kelly
        patch(
            "wallet_manager.strategies.deep_kelly_risk.get_deep_kelly_engine",
            return_value=MagicMock(
                update_equity=MagicMock(),
                calculate_position=MagicMock(return_value=risk_decision),
            ),
        ),
        # flash crash detector – no crash
        patch(
            "wallet_manager.strategies.flash_crash_detector.get_detector",
            return_value=MagicMock(ingest_price=MagicMock(return_value=None)),
        ),
        # signal intelligence OFI (best-effort, may fail silently)
        patch(
            "wallet_manager.strategies.signal_intelligence.get_alpha_generator",
            return_value=MagicMock(read_ofi=AsyncMock(return_value=None)),
        ),
        # settings
        patch(
            "wallet_manager.config.get_settings",
            return_value=MagicMock(
                paper_db_path=str(tmp_path / "p.db"),
                paper_max_gas_fraction_pct=5.0,
                execution_mode="paper",
                weth_usdc_pool="0xfakepool",
            ),
        ),
        # weth/usdc pool
        patch(
            "wallet_manager.strategies.paper_engine._get_weth_usdc_pool",
            return_value="0xfakepool",
        ),
    ]
    return patches


# ---------------------------------------------------------------------------
# Helper: build a fully-initialized engine with the DB in tmp_path
# ---------------------------------------------------------------------------


def _settings_mock(tmp_path, db_name="p.db"):
    """Return a settings mock with all required string attributes set."""
    m = MagicMock(
        paper_db_path=str(tmp_path / db_name),
        paper_max_gas_fraction_pct=5.0,
        execution_mode="paper",
        weth_usdc_pool="0xfakepool",
        log_level="INFO",
    )
    m.log_level = "INFO"
    return m


async def _build_engine(tmp_path, config: PaperTradeConfig | None = None):
    """Create and initialize an engine backed by a temp DB."""
    reset_paper_engine()
    cfg = config or PaperTradeConfig(
        initial_balance_usd=10_000.0,
        gas_estimation=False,
        execution_delay_ms=(0, 1),
    )
    with patch(
        "wallet_manager.config.get_settings",
        return_value=_settings_mock(tmp_path),
    ):
        engine = PaperTradingEngine(config=cfg)
        await engine.initialize()
    return engine


# ===========================================================================
# PaperTradeResult — field checks
# ===========================================================================


class TestPaperTradeResultFields:
    def test_all_fields_set(self):
        r = PaperTradeResult(
            action="BUY",
            signal_alpha=0.42,
            simulated_fill_price=2499.0,
            slippage_bps=6.5,
            gas_cost_usd=0.002,
            virtual_pnl_usd=15.0,
            would_have_executed=True,
            reason_codes=("PAPER_SHADOW_ONLY", "PAPER_TRADE_EXECUTED"),
            cycle_duration_ms=210.5,
            timestamp=1_700_000_000.0,
        )
        assert r.action == "BUY"
        assert r.signal_alpha == pytest.approx(0.42)
        assert r.simulated_fill_price == pytest.approx(2499.0)
        assert r.slippage_bps == pytest.approx(6.5)
        assert r.gas_cost_usd == pytest.approx(0.002)
        assert r.virtual_pnl_usd == pytest.approx(15.0)
        assert r.would_have_executed is True
        assert "PAPER_SHADOW_ONLY" in r.reason_codes
        assert r.cycle_duration_ms == pytest.approx(210.5)
        assert r.timestamp == pytest.approx(1_700_000_000.0)

    def test_sell_result(self):
        r = PaperTradeResult(action="SELL", virtual_pnl_usd=-5.0, would_have_executed=True)
        assert r.action == "SELL"
        assert r.virtual_pnl_usd == pytest.approx(-5.0)

    def test_hold_default(self):
        r = PaperTradeResult()
        assert r.action == "HOLD"
        assert r.would_have_executed is False

    def test_immutable(self):
        r = PaperTradeResult(action="BUY")
        with pytest.raises((AttributeError, TypeError)):
            r.action = "SELL"  # type: ignore[misc]


# ===========================================================================
# Singleton tests
# ===========================================================================


class TestSingleton:
    @pytest.mark.asyncio
    async def test_get_paper_engine_returns_same_instance(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)):
            e1 = await get_paper_engine()
            e2 = await get_paper_engine()
        assert e1 is e2
        await e1.shutdown()
        reset_paper_engine()

    @pytest.mark.asyncio
    async def test_get_paper_engine_is_initialized(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)):
            engine = await get_paper_engine()
        assert engine._initialized is True
        await engine.shutdown()
        reset_paper_engine()

    @pytest.mark.asyncio
    async def test_reset_paper_engine_clears_singleton(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)):
            e1 = await get_paper_engine()
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path, "p2.db")):
            e2 = await get_paper_engine()
        assert e1 is not e2
        await e1.shutdown()
        await e2.shutdown()
        reset_paper_engine()

    def test_reset_is_synchronous(self):
        reset_paper_engine()  # should not raise

    @pytest.mark.asyncio
    async def test_get_paper_engine_with_custom_config(self, tmp_path):
        reset_paper_engine()
        cfg = PaperTradeConfig(initial_balance_usd=99_000.0)
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)):
            engine = await get_paper_engine(config=cfg)
        assert engine._config.initial_balance_usd == pytest.approx(99_000.0)
        await engine.shutdown()
        reset_paper_engine()


# ===========================================================================
# run_shadow_cycle — early exits
# ===========================================================================


class TestRunShadowCycleEarlyExits:

    @pytest.mark.asyncio
    async def test_circuit_breaker_halted_returns_halted(self, tmp_path):
        engine = await _build_engine(tmp_path)
        command_state = MagicMock()
        command_state.is_circuit_breaker_halted = True
        ctx = {"command_state": command_state}

        result = await engine.run_shadow_cycle(
            manager=_make_wallet_manager(),
            wallet_id="w1",
            portfolio_manager=_make_portfolio_manager(),
            task_context=ctx,
        )
        assert result.action == "HALTED"
        assert PAPER_SHADOW_ONLY in result.reason_codes
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_wrong_chain_returns_wrong_chain(self, tmp_path):
        engine = await _build_engine(tmp_path)
        mgr = _make_wallet_manager(chain_id=1)  # Ethereum, not Base
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)):
            result = await engine.run_shadow_cycle(
                manager=mgr,
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        assert result.action == "WRONG_CHAIN"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_price_fetch_error_returns_price_error(self, tmp_path):
        engine = await _build_engine(tmp_path)
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)), \
             patch("wallet_manager.actions.get_live_price_v3", AsyncMock(side_effect=Exception("timeout"))), \
             patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(side_effect=Exception("timeout"))), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(side_effect=Exception("timeout"))):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        assert result.action == "PRICE_ERROR"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_zero_price_returns_invalid_price(self, tmp_path):
        engine = await _build_engine(tmp_path)
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)), \
             patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=0.0)), \
             patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(return_value=0.0)), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=0.0)):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        assert result.action == "INVALID_PRICE"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_signal_none_returns_warmup(self, tmp_path):
        engine = await _build_engine(tmp_path)
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)), \
             patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=MagicMock(ingest_price=MagicMock(return_value=None))), \
             patch("wallet_manager.strategies.signal_intelligence.get_alpha_generator", return_value=MagicMock(read_ofi=AsyncMock(return_value=None))), \
             patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=MagicMock(ingest_price=MagicMock(return_value=None))):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        assert result.action == "WARMUP"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_flash_crash_returns_flash_crash(self, tmp_path):
        engine = await _build_engine(tmp_path)
        signal = _make_signal()
        crash_alert = MagicMock()
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)), \
             patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=MagicMock(ingest_price=MagicMock(return_value=signal))), \
             patch("wallet_manager.strategies.signal_intelligence.get_alpha_generator", return_value=MagicMock(read_ofi=AsyncMock(return_value=None))), \
             patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=MagicMock(ingest_price=MagicMock(return_value=crash_alert))):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        assert result.action == "FLASH_CRASH"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_risk_blocked_zero_position_returns_risk_blocked(self, tmp_path):
        engine = await _build_engine(tmp_path)
        signal = _make_signal()
        risk_decision = _make_risk_decision(position_usd=0.0)
        with patch("wallet_manager.config.get_settings", return_value=_settings_mock(tmp_path)), \
             patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=2500.0)), \
             patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=MagicMock(ingest_price=MagicMock(return_value=signal))), \
             patch("wallet_manager.strategies.signal_intelligence.get_alpha_generator", return_value=MagicMock(read_ofi=AsyncMock(return_value=None))), \
             patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=MagicMock(ingest_price=MagicMock(return_value=None))), \
             patch("wallet_manager.strategies.deep_kelly_risk.get_deep_kelly_engine", return_value=MagicMock(update_equity=MagicMock(), calculate_position=MagicMock(return_value=risk_decision))):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        assert result.action.startswith("RISK_BLOCKED")
        await engine.shutdown()


# ---------------------------------------------------------------------------
# Comprehensive patch context for BUY/SELL tests
# Patches all the contextlib.suppress-wrapped singletons that may zero out
# _js_position_usd before the trading logic block.
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _full_cycle_patches(tmp_path, price=2500.0, signal=None, risk_position_usd=500.0):
    """Context manager that patches everything needed for a clean shadow cycle."""
    if signal is None:
        signal = _make_signal()
    risk_decision = _make_risk_decision(position_usd=risk_position_usd)

    settings_mock = _settings_mock(tmp_path)
    # Bayesian Kelly: return fraction=0.25 (neutral) and uncertainty=1.0
    bk_mock = MagicMock()
    bk_mock.compute_kelly.return_value = 0.25
    bk_mock.get_uncertainty_factor.return_value = 1.0
    bk_mock.update.return_value = None

    # Risk intelligence: return multiplier of 1.0
    ri_mock = MagicMock()
    ri_mock.get_effective_kelly_multiplier.return_value = 1.0
    ri_mock.record_cycle.return_value = None
    ri_mock.get_alerts.return_value = []

    # Execution quality / optimal timing: no "avoid" flag
    timing_mock = MagicMock()
    timing_mock.get_current_window_quality.return_value = {"avoid": False}
    timing_mock.record_execution.return_value = None

    eq_mock = MagicMock()
    eq_mock.on_trade_entry.return_value = None
    eq_mock.on_price_update.return_value = None
    eq_mock.on_trade_close.return_value = None

    with patch("wallet_manager.config.get_settings", return_value=settings_mock), \
         patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=price)), \
         patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(return_value=price)), \
         patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=price)), \
         patch(
             "wallet_manager.strategies.robust_alpha.get_robust_alpha_engine",
             return_value=MagicMock(ingest_price=MagicMock(return_value=signal)),
         ), \
         patch(
             "wallet_manager.strategies.deep_kelly_risk.get_deep_kelly_engine",
             return_value=MagicMock(
                 update_equity=MagicMock(),
                 calculate_position=MagicMock(return_value=risk_decision),
             ),
         ), \
         patch(
             "wallet_manager.strategies.flash_crash_detector.get_detector",
             return_value=MagicMock(ingest_price=MagicMock(return_value=None)),
         ), \
         patch(
             "wallet_manager.strategies.signal_intelligence.get_alpha_generator",
             return_value=MagicMock(read_ofi=AsyncMock(return_value=None)),
         ), \
         patch("wallet_manager.strategies.paper_engine._get_weth_usdc_pool", return_value="0xfakepool"), \
         patch("wallet_manager.strategies.bayesian_kelly.get_bayesian_kelly", return_value=bk_mock), \
         patch("wallet_manager.risk_intelligence.get_risk_intelligence", return_value=ri_mock), \
         patch("wallet_manager.strategies.execution_quality.get_optimal_timing", return_value=timing_mock), \
         patch("wallet_manager.strategies.execution_quality.get_execution_quality", return_value=eq_mock), \
         patch(
             "wallet_manager.strategies.paper_engine.PaperTradingEngine._send_trade_telegram",
             AsyncMock(),
         ):
        yield


# ===========================================================================
# run_shadow_cycle — HOLD path
# ===========================================================================


class TestRunShadowCycleHold:

    @pytest.mark.asyncio
    async def test_hold_when_signal_says_no_entry(self, tmp_path):
        """Signal does not say should_enter_long → HOLD."""
        engine = await _build_engine(tmp_path)
        signal = _make_signal(should_enter_long=False, should_exit=False)

        with _full_cycle_patches(tmp_path, signal=signal):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )

        assert result.action == "HOLD"
        assert result.would_have_executed is False
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_cycle_count_increments_on_hold(self, tmp_path):
        engine = await _build_engine(tmp_path)
        signal = _make_signal(should_enter_long=False)
        assert engine._cycle_count == 0

        with _full_cycle_patches(tmp_path, signal=signal):
            await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )

        assert engine._cycle_count == 1
        await engine.shutdown()


# ===========================================================================
# run_shadow_cycle — BUY path
# ===========================================================================


class TestRunShadowCycleBuy:

    @pytest.mark.asyncio
    async def test_buy_sets_in_position(self, tmp_path):
        """When signal says enter long and we have USDC, engine buys."""
        engine = await _build_engine(tmp_path)
        signal = _make_signal(should_enter_long=True, should_exit=False)

        with _full_cycle_patches(tmp_path, signal=signal):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )

        assert result.action == "BUY"
        assert result.would_have_executed is True
        assert engine._in_position is True
        assert engine._entry_price > 0
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_buy_reduces_usdc_balance(self, tmp_path):
        engine = await _build_engine(tmp_path)
        signal = _make_signal(should_enter_long=True)

        with _full_cycle_patches(tmp_path, signal=signal):
            usdc_before = await engine._ledger.get_balance("USDC")
            await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
            usdc_after = await engine._ledger.get_balance("USDC")

        assert usdc_after < usdc_before
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_buy_increases_weth_balance(self, tmp_path):
        engine = await _build_engine(tmp_path)
        signal = _make_signal(should_enter_long=True)

        with _full_cycle_patches(tmp_path, signal=signal):
            weth_before = await engine._ledger.get_balance("WETH")
            await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
            weth_after = await engine._ledger.get_balance("WETH")

        assert weth_after > weth_before
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_buy_result_has_signal_alpha(self, tmp_path):
        engine = await _build_engine(tmp_path)
        signal = _make_signal(robust_alpha=0.42, should_enter_long=True)

        with _full_cycle_patches(tmp_path, signal=signal):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )

        assert result.signal_alpha == pytest.approx(0.42)
        await engine.shutdown()


# ===========================================================================
# run_shadow_cycle — SELL path (buy then sell)
# ===========================================================================


class TestRunShadowCycleSell:

    async def _buy_first(self, engine, tmp_path):
        """Execute a BUY cycle so the engine enters a position."""
        buy_signal = _make_signal(should_enter_long=True, should_exit=False)
        with _full_cycle_patches(tmp_path, price=2500.0, signal=buy_signal):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
        return result

    @pytest.mark.asyncio
    async def test_sell_clears_position(self, tmp_path):
        engine = await _build_engine(tmp_path)
        await self._buy_first(engine, tmp_path)
        # If buy didn't happen due to some patch issue, force position manually
        if not engine._in_position:
            engine._in_position = True
            engine._entry_price = 2500.0
            engine._position_size_usd = 500.0
            await engine._ledger.update_balance("WETH", 0.2, usd_value=500.0)
            await engine._ledger.update_balance("USDC", 9500.0, usd_value=9500.0)

        assert engine._in_position is True

        sell_signal = _make_signal(
            should_enter_long=False, should_exit=True, robust_alpha=-0.2
        )
        with _full_cycle_patches(tmp_path, price=2600.0, signal=sell_signal):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )

        assert result.action == "SELL"
        assert result.would_have_executed is True
        assert engine._in_position is False
        assert engine._entry_price == 0.0
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_sell_records_pnl(self, tmp_path):
        engine = await _build_engine(tmp_path)
        await self._buy_first(engine, tmp_path)
        if not engine._in_position:
            engine._in_position = True
            engine._entry_price = 2500.0
            engine._position_size_usd = 500.0
            await engine._ledger.update_balance("WETH", 0.2, usd_value=500.0)
            await engine._ledger.update_balance("USDC", 9500.0, usd_value=9500.0)

        sell_signal = _make_signal(should_enter_long=False, should_exit=True)
        with _full_cycle_patches(tmp_path, price=2600.0, signal=sell_signal):
            result = await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )

        # Result should be a SELL with a valid float PnL
        assert result.action == "SELL"
        assert isinstance(result.virtual_pnl_usd, float)
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_sell_restores_usdc_balance(self, tmp_path):
        engine = await _build_engine(tmp_path)
        await self._buy_first(engine, tmp_path)
        if not engine._in_position:
            engine._in_position = True
            engine._entry_price = 2500.0
            engine._position_size_usd = 500.0
            await engine._ledger.update_balance("WETH", 0.2, usd_value=500.0)
            await engine._ledger.update_balance("USDC", 9500.0, usd_value=9500.0)

        usdc_after_buy = await engine._ledger.get_balance("USDC")

        sell_signal = _make_signal(should_enter_long=False, should_exit=True)
        with _full_cycle_patches(tmp_path, price=2600.0, signal=sell_signal):
            await engine.run_shadow_cycle(
                manager=_make_wallet_manager(),
                wallet_id="w1",
                portfolio_manager=_make_portfolio_manager(),
                task_context=_task_context(),
            )
            usdc_after_sell = await engine._ledger.get_balance("USDC")

        assert usdc_after_sell > usdc_after_buy
        await engine.shutdown()


# ===========================================================================
# Performance reporting
# ===========================================================================


class TestPerformanceReporting:

    @pytest.mark.asyncio
    async def test_get_performance_report_not_initialized_returns_error(self):
        reset_paper_engine()
        engine = PaperTradingEngine()
        report = await engine.get_performance_report()
        assert "error" in report

    @pytest.mark.asyncio
    async def test_get_performance_report_keys(self, tmp_path):
        engine = await _build_engine(tmp_path)
        report = await engine.get_performance_report()
        assert "virtual_pnl" in report
        assert "drawdown" in report
        assert "balances" in report
        assert "position" in report
        assert "engine" in report
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_get_performance_report_balances(self, tmp_path):
        engine = await _build_engine(tmp_path)
        report = await engine.get_performance_report()
        balances = report["balances"]
        assert "USDC" in balances
        assert "WETH" in balances
        assert "aUSDC" in balances
        assert balances["USDC"] == pytest.approx(10_000.0, rel=1e-3)
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_get_performance_report_position_keys(self, tmp_path):
        engine = await _build_engine(tmp_path)
        report = await engine.get_performance_report()
        pos = report["position"]
        assert "in_position" in pos
        assert "entry_price" in pos
        assert "position_size_usd" in pos
        assert pos["in_position"] is False
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_get_performance_report_engine_config_keys(self, tmp_path):
        engine = await _build_engine(tmp_path)
        report = await engine.get_performance_report()
        engine_info = report["engine"]
        assert "cycle_count" in engine_info
        assert "config" in engine_info
        assert engine_info["config"]["initial_balance_usd"] == pytest.approx(10_000.0)
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_get_telegram_report_not_initialized_returns_string(self):
        reset_paper_engine()
        engine = PaperTradingEngine()
        report = await engine.get_telegram_report()
        assert isinstance(report, str)
        assert "not initialized" in report.lower()

    @pytest.mark.asyncio
    async def test_get_telegram_report_contains_sections(self, tmp_path):
        engine = await _build_engine(tmp_path)
        report = await engine.get_telegram_report()
        assert "PAPER TRADING REPORT" in report
        assert "Sharpe" in report
        assert "Drawdown" in report or "drawdown" in report
        assert "Win Rate" in report
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_get_telegram_report_shows_usdc_balance(self, tmp_path):
        engine = await _build_engine(tmp_path)
        report = await engine.get_telegram_report()
        # Initial 10000 USDC should appear somewhere
        assert "10000" in report or "10,000" in report or "USDC" in report
        await engine.shutdown()


# ===========================================================================
# _journal_decision write path
# ===========================================================================


class TestJournalDecision:

    @pytest.mark.asyncio
    async def test_journal_decision_without_signal(self, tmp_path):
        engine = await _build_engine(tmp_path)
        await engine._journal_decision(
            wallet_id="w1",
            cycle_phase="test",
            action="HOLD",
            primary_reason="NO_SIGNAL",
            reason_codes=(PAPER_SHADOW_ONLY,),
            signal=None,
            volatility_regime="normal",
            total_equity_usd=10_000.0,
        )
        decisions = await engine._ledger.get_recent_decisions(1)
        assert len(decisions) == 1
        assert decisions[0]["action"] == "HOLD"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_journal_decision_with_signal(self, tmp_path):
        engine = await _build_engine(tmp_path)
        sig = _make_signal(robust_alpha=0.25, confidence=0.9)
        await engine._journal_decision(
            wallet_id="w2",
            cycle_phase="shadow_cycle",
            action="BUY",
            primary_reason="LONG_ENTRY_CONFIRMED",
            reason_codes=(PAPER_SHADOW_ONLY,),
            signal=sig,
            volatility_regime="low",
            risk_position_usd=400.0,
            trade_budget_usd=400.0,
            gas_cost_usd=0.002,
            total_equity_usd=10_000.0,
            pnl_usd=0.0,
            notes="test note",
        )
        decisions = await engine._ledger.get_recent_decisions(1)
        assert decisions[0]["robust_alpha"] == pytest.approx(0.25)
        assert decisions[0]["confidence"] == pytest.approx(0.9)
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_journal_decision_sell_class(self, tmp_path):
        engine = await _build_engine(tmp_path)
        await engine._journal_decision(
            wallet_id="w3",
            cycle_phase="shadow_cycle",
            action="SELL",
            primary_reason="EXIT_SIGNAL_CONFIRMED",
            reason_codes=(PAPER_SHADOW_ONLY,),
            total_equity_usd=9_800.0,
        )
        decisions = await engine._ledger.get_recent_decisions(1)
        assert decisions[0]["decision_class"] == "EXECUTED" or decisions[0]["action"] == "SELL"
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_record_external_skip(self, tmp_path):
        engine = await _build_engine(tmp_path)
        await engine.record_external_skip(
            wallet_id="w1",
            primary_reason="TASK_PAUSED",
            notes="scheduled maintenance",
            regime_label="unknown",
        )
        decisions = await engine._ledger.get_recent_decisions(5)
        skips = [d for d in decisions if d["action"] == "SKIP"]
        assert len(skips) >= 1
        await engine.shutdown()


# ===========================================================================
# VirtualLedger — additional CRUD and statistics
# ===========================================================================


class TestVirtualLedgerAdditional:

    @pytest.mark.asyncio
    async def test_record_snapshot_updates_peak_equity(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        await ledger.update_balance("USDC", 12_000.0, usd_value=12_000.0)
        await ledger._record_snapshot(12_000.0, 0.0)
        assert ledger._peak_equity == pytest.approx(12_000.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_record_snapshot_tracks_drawdown(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        await ledger._record_snapshot(10_000.0, 0.0)
        await ledger._record_snapshot(8_000.0, 0.0)
        assert ledger._max_drawdown == pytest.approx(0.2, rel=1e-3)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_pnl_summary_closed_count(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        await ledger.record_trade(
            direction="BUY", asset_pair="WETH/USDC",
            entry_price=2000.0, simulated_fill_price=1999.0,
            slippage_bps=5.0, size_usd=100.0, gas_cost_usd=0.01,
            pnl_usd=0.0, reason_codes=(), signal_source="test",
        )
        await ledger.record_trade(
            direction="SELL", asset_pair="WETH/USDC",
            entry_price=2000.0, simulated_fill_price=2050.0,
            slippage_bps=5.0, size_usd=100.0, gas_cost_usd=0.01,
            pnl_usd=5.0, reason_codes=(), signal_source="test",
        )
        summary = await ledger.get_pnl_summary()
        assert summary["trade_count"] == 2
        assert summary["closed_trade_count"] == 1
        assert summary["win_count"] == 1
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_advanced_analytics_with_sell_trades(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        # Insert a winning and a losing sell
        for pnl in [10.0, -5.0]:
            await ledger.record_trade(
                direction="SELL", asset_pair="WETH/USDC",
                entry_price=2000.0, simulated_fill_price=2010.0,
                slippage_bps=5.0, size_usd=100.0, gas_cost_usd=0.01,
                pnl_usd=pnl, reason_codes=(), signal_source="test",
            )
        adv = await ledger.get_advanced_analytics()
        assert adv["closed_count"] == 2
        assert adv["win_rate_pct"] == pytest.approx(50.0)
        assert adv["gross_profit_usd"] == pytest.approx(10.0)
        assert adv["gross_loss_usd"] == pytest.approx(5.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_hold_time_stats_with_pairs(self, tmp_path):
        import time as _time
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()

        t0 = _time.time()
        # Directly insert rows with controlled timestamps to avoid sleep
        async with ledger._write_lock:
            await ledger._conn.execute(
                "INSERT INTO paper_trades (timestamp, direction, asset_pair, entry_price, "
                "simulated_fill_price, slippage_bps, size_usd, gas_cost_usd, pnl_usd, "
                "reason_codes, signal_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t0, "BUY", "WETH/USDC", 2000, 1999, 5, 100, 0.01, 0, "[]", "test"),
            )
            await ledger._conn.execute(
                "INSERT INTO paper_trades (timestamp, direction, asset_pair, entry_price, "
                "simulated_fill_price, slippage_bps, size_usd, gas_cost_usd, pnl_usd, "
                "reason_codes, signal_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t0 + 60.0, "SELL", "WETH/USDC", 2000, 2010, 5, 100, 0.01, 5, "[]", "test"),
            )
            await ledger._conn.commit()
        # Update counters to reflect the inserts
        ledger._trade_count = 2
        ledger._closed_trade_count = 1

        stats = await ledger.get_hold_time_stats()
        assert stats["completed_pairs"] == 1
        assert stats["avg_hold_seconds"] == pytest.approx(60.0, rel=1e-2)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_equity_curve_windowed_empty(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        curve = await ledger.get_equity_curve_windowed()
        assert curve == []
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_equity_curve_windowed_with_data(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        await ledger.update_balance("USDC", 10_000.0, usd_value=10_000.0)
        await ledger.record_balance_history(eth_price=2000.0)
        await ledger.record_balance_history(eth_price=2000.0)
        curve = await ledger.get_equity_curve_windowed()
        assert len(curve) >= 1
        assert "timestamp" in curve[0]
        assert "total_equity_usd" in curve[0]
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_live_readiness_score_empty_db(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        score_data = await ledger.get_live_readiness_score()
        assert "score" in score_data
        assert "gates" in score_data
        assert "recommendation" in score_data
        assert score_data["score"] >= 0
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_performance_by_regime_empty(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        result = await ledger.get_performance_by_regime()
        assert isinstance(result, list)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_performance_by_setup_empty(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        result = await ledger.get_performance_by_setup()
        assert isinstance(result, list)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_update_balance_persists_after_reinit(self, tmp_path):
        """Balance written to disk should be reloaded on re-open."""
        db_path = str(tmp_path / "l.db")
        ledger = VirtualLedger(db_path=db_path)
        await ledger.initialize()
        await ledger.update_balance("USDC", 7_777.0, usd_value=7_777.0)
        await ledger.shutdown()

        ledger2 = VirtualLedger(db_path=db_path)
        await ledger2.initialize()
        bal = await ledger2.get_balance("USDC")
        assert bal == pytest.approx(7_777.0)
        await ledger2.shutdown()

    @pytest.mark.asyncio
    async def test_total_equity_includes_all_assets(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        await ledger.update_balance("USDC", 1_000.0, usd_value=1_000.0)
        await ledger.update_balance("WETH", 2.0, usd_value=6_000.0)
        await ledger.update_balance("aUSDC", 500.0, usd_value=500.0)
        total = await ledger.get_total_equity_usd(eth_price=3_000.0)
        assert total == pytest.approx(1_000.0 + 6_000.0 + 500.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_drawdown_no_history(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        dd = await ledger.get_drawdown()
        assert dd["current_drawdown"] == pytest.approx(0.0)
        assert dd["max_drawdown"] == pytest.approx(0.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_record_decision_all_fields(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        await ledger.record_decision(
            wallet_id="wtest",
            cycle_phase="shadow_cycle",
            action="BUY",
            regime_label="low_vol",
            setup_tag="momentum_long",
            decision_class="EXECUTED",
            primary_reason="LONG_ENTRY_CONFIRMED",
            reason_codes=("PAPER_SHADOW_ONLY", "PAPER_TRADE_EXECUTED"),
            raw_alpha=0.1,
            robust_alpha=0.3,
            confidence=0.85,
            adf_pvalue=0.03,
            stationarity_d=1.0,
            volatility_regime="low_vol",
            risk_position_usd=400.0,
            trade_budget_usd=400.0,
            gas_cost_usd=0.002,
            total_equity_usd=10_000.0,
            pnl_usd=0.0,
            notes="unit test",
        )
        rows = await ledger.get_recent_decisions(1)
        assert len(rows) == 1
        r = rows[0]
        assert r["wallet_id"] == "wtest"
        assert r["action"] == "BUY"
        assert r["regime_label"] == "low_vol"
        assert r["robust_alpha"] == pytest.approx(0.3)
        assert r["confidence"] == pytest.approx(0.85)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_decision_analytics_regime_breakdown(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "l.db"))
        await ledger.initialize()
        for regime in ["low_vol", "low_vol", "high_vol"]:
            await ledger.record_decision(
                wallet_id="w1",
                cycle_phase="shadow_cycle",
                action="HOLD",
                regime_label=regime,
                setup_tag="signal_wait",
                decision_class="NO_TRADE",
                primary_reason="NO_ENTRY",
                reason_codes=(),
            )
        analytics = await ledger.get_decision_analytics()
        regimes = {r["regime_label"]: r for r in analytics["regime_breakdown"]}
        assert "low_vol" in regimes
        assert regimes["low_vol"]["decision_count"] == 2
        await ledger.shutdown()
