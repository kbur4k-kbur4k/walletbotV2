"""tests/unit/test_paper_engine_coverage.py — Comprehensive unit tests for paper_engine.py."""

from __future__ import annotations

import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from wallet_manager.strategies.paper_engine import (
    GAS_UNITS,
    PAPER_GAS_ESTIMATED,
    PAPER_SHADOW_ONLY,
    PAPER_SIGNAL_REJECTED,
    PAPER_SLIPPAGE_APPLIED,
    PAPER_TRADE_EXECUTED,
    GasEstimator,
    PaperTradeConfig,
    PaperTradeResult,
    PaperTradingEngine,
    SlippageSimulator,
    VirtualLedger,
    reset_paper_engine,
)

# ---------------------------------------------------------------------------
# PaperTradeConfig tests
# ---------------------------------------------------------------------------


class TestPaperTradeConfig:
    def test_default_values(self):
        cfg = PaperTradeConfig()
        assert cfg.initial_balance_usd == 10_000.0
        assert cfg.slippage_model == "gaussian"
        assert cfg.slippage_mean_bps == 5.0
        assert cfg.slippage_std_bps == 3.0
        assert cfg.gas_estimation is True
        assert cfg.execution_delay_ms == (200, 800)

    def test_custom_values(self):
        cfg = PaperTradeConfig(
            initial_balance_usd=50_000.0,
            slippage_model="fixed",
            slippage_mean_bps=10.0,
            slippage_std_bps=1.0,
            gas_estimation=False,
            execution_delay_ms=(100, 300),
        )
        assert cfg.initial_balance_usd == 50_000.0
        assert cfg.slippage_model == "fixed"
        assert cfg.slippage_mean_bps == 10.0
        assert cfg.slippage_std_bps == 1.0
        assert cfg.gas_estimation is False
        assert cfg.execution_delay_ms == (100, 300)

    def test_frozen_immutability(self):
        cfg = PaperTradeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.initial_balance_usd = 999.0  # type: ignore[misc]

    def test_equality(self):
        cfg1 = PaperTradeConfig(initial_balance_usd=5000.0)
        cfg2 = PaperTradeConfig(initial_balance_usd=5000.0)
        assert cfg1 == cfg2

    def test_inequality(self):
        cfg1 = PaperTradeConfig(initial_balance_usd=5000.0)
        cfg2 = PaperTradeConfig(initial_balance_usd=6000.0)
        assert cfg1 != cfg2


# ---------------------------------------------------------------------------
# PaperTradeResult tests
# ---------------------------------------------------------------------------


class TestPaperTradeResult:
    def test_default_values(self):
        result = PaperTradeResult()
        assert result.action == "HOLD"
        assert result.signal_alpha == 0.0
        assert result.simulated_fill_price == 0.0
        assert result.slippage_bps == 0.0
        assert result.gas_cost_usd == 0.0
        assert result.virtual_pnl_usd == 0.0
        assert result.would_have_executed is False
        assert result.reason_codes == ()
        assert result.cycle_duration_ms == 0.0
        assert result.timestamp == 0.0

    def test_custom_values(self):
        result = PaperTradeResult(
            action="BUY",
            signal_alpha=0.75,
            simulated_fill_price=2001.50,
            slippage_bps=7.5,
            gas_cost_usd=0.0033,
            virtual_pnl_usd=12.5,
            would_have_executed=True,
            reason_codes=("signal_buy", "paper_trade_executed"),
            cycle_duration_ms=350.0,
            timestamp=1_700_000_000.0,
        )
        assert result.action == "BUY"
        assert result.signal_alpha == 0.75
        assert result.simulated_fill_price == pytest.approx(2001.50)
        assert result.slippage_bps == pytest.approx(7.5)
        assert result.would_have_executed is True
        assert "signal_buy" in result.reason_codes

    def test_frozen_immutability(self):
        result = PaperTradeResult(action="BUY")
        with pytest.raises((AttributeError, TypeError)):
            result.action = "SELL"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SlippageSimulator tests
# ---------------------------------------------------------------------------


class TestSlippageSimulator:
    def test_initialization(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=2.0)
        assert sim._mean_bps == 5.0
        assert sim._std_bps == 2.0

    def test_gaussian_returns_nonnegative_slippage(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=2.0)
        for _ in range(50):
            slip_bps, _fill_price = sim.simulate_slippage(
                size_usd=100.0, pool_depth_usd=1_000_000.0,
                model="gaussian", entry_price=2000.0
            )
            assert slip_bps >= 0.0

    def test_gaussian_fill_price_lower_than_entry(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=1.0)
        for _ in range(20):
            _slip_bps, fill_price = sim.simulate_slippage(
                size_usd=100.0, pool_depth_usd=1_000_000.0,
                model="gaussian", entry_price=2000.0
            )
            assert fill_price <= 2000.0

    def test_fixed_model_deterministic(self):
        sim = SlippageSimulator(mean_bps=10.0, std_bps=0.0)
        slip1, fill1 = sim.simulate_slippage(
            size_usd=100.0, pool_depth_usd=1_000_000.0,
            model="fixed", entry_price=2000.0
        )
        slip2, fill2 = sim.simulate_slippage(
            size_usd=100.0, pool_depth_usd=1_000_000.0,
            model="fixed", entry_price=2000.0
        )
        assert slip1 == slip2
        assert fill1 == fill2

    def test_zero_entry_price_returns_zero_fill(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=2.0)
        _slip_bps, fill_price = sim.simulate_slippage(
            size_usd=100.0, pool_depth_usd=1_000_000.0,
            model="gaussian", entry_price=0.0
        )
        assert fill_price == 0.0

    def test_large_order_gets_more_slippage(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=0.0)
        # Small order
        small_slip, _ = sim.simulate_slippage(
            size_usd=1_000.0, pool_depth_usd=1_000_000.0,
            model="fixed", entry_price=2000.0
        )
        # Large order (10% of pool)
        large_slip, _ = sim.simulate_slippage(
            size_usd=100_000.0, pool_depth_usd=1_000_000.0,
            model="fixed", entry_price=2000.0
        )
        assert large_slip > small_slip

    def test_slippage_capped_at_500_bps(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=0.0)
        # Enormous order (50% of pool) - should be capped
        slip_bps, _ = sim.simulate_slippage(
            size_usd=500_000.0, pool_depth_usd=1_000_000.0,
            model="fixed", entry_price=2000.0
        )
        assert slip_bps <= 500.0

    def test_unknown_model_falls_back_to_gaussian(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=2.0)
        # Should not raise an exception, just fall back
        slip_bps, _fill_price = sim.simulate_slippage(
            size_usd=100.0, pool_depth_usd=1_000_000.0,
            model="invalid_model", entry_price=2000.0
        )
        assert slip_bps >= 0.0

    def test_fill_price_calculation_accuracy(self):
        sim = SlippageSimulator(mean_bps=10.0, std_bps=0.0)
        entry_price = 2000.0
        slip_bps, fill_price = sim.simulate_slippage(
            size_usd=100.0, pool_depth_usd=1_000_000.0,
            model="fixed", entry_price=entry_price
        )
        # fill_price = entry_price * (1 - slip_bps / 10_000)
        expected_fill = entry_price * (1.0 - slip_bps / 10_000.0)
        assert fill_price == pytest.approx(expected_fill, rel=1e-4)

    def test_zero_pool_depth_safe(self):
        sim = SlippageSimulator(mean_bps=5.0, std_bps=0.0)
        # pool_depth_usd=0 should be handled (floor at 1.0)
        slip_bps, _fill_price = sim.simulate_slippage(
            size_usd=100.0, pool_depth_usd=0.0,
            model="fixed", entry_price=2000.0
        )
        # Should not raise; slippage gets capped at 500
        assert slip_bps <= 500.0


# ---------------------------------------------------------------------------
# GasEstimator tests
# ---------------------------------------------------------------------------


class TestGasEstimator:
    @pytest.mark.asyncio
    async def test_estimate_with_no_rpc_uses_fallback(self):
        est = GasEstimator()
        # No rpc_url, cached_gas_gwei = 0 -> uses fallback 0.01
        cost = await est.estimate_gas_cost_usd(tx_type="swap", rpc_url="", eth_price=3000.0)
        expected = GAS_UNITS["swap"] * 0.01 * 1e-9 * 3000.0
        assert cost == pytest.approx(expected, rel=1e-4)

    @pytest.mark.asyncio
    async def test_estimate_swap_vs_borrow_cost_difference(self):
        est = GasEstimator()
        est._cached_gas_gwei = 5.0
        est._cached_eth_price = 3000.0
        est._cache_timestamp = time.time()

        swap_cost = await est.estimate_gas_cost_usd(tx_type="swap", rpc_url="", eth_price=3000.0)
        borrow_cost = await est.estimate_gas_cost_usd(tx_type="borrow", rpc_url="", eth_price=3000.0)

        # borrow uses more gas units than swap
        assert borrow_cost > swap_cost

    @pytest.mark.asyncio
    async def test_unknown_tx_type_falls_back_to_swap(self):
        est = GasEstimator()
        est._cached_gas_gwei = 5.0
        est._cached_eth_price = 3000.0
        est._cache_timestamp = time.time()

        cost = await est.estimate_gas_cost_usd(tx_type="unknown_tx", rpc_url="", eth_price=3000.0)
        swap_cost = await est.estimate_gas_cost_usd(tx_type="swap", rpc_url="", eth_price=3000.0)
        assert cost == pytest.approx(swap_cost, rel=1e-4)

    @pytest.mark.asyncio
    async def test_rpc_failure_uses_conservative_fallback(self):
        est = GasEstimator()
        # Simulate RPC failure
        with patch("web3.AsyncWeb3") as mock_w3_cls:
            mock_w3 = AsyncMock()
            mock_w3.eth.gas_price = AsyncMock(side_effect=Exception("RPC Down"))
            mock_w3_cls.return_value = mock_w3
            with patch("wallet_manager.utils.web3_utils.close_w3_provider", AsyncMock()):
                cost = await est.estimate_gas_cost_usd(
                    tx_type="swap", rpc_url="http://fake-rpc", eth_price=2000.0
                )
        # Should return a valid cost (using fallback 0.01 gwei)
        assert cost > 0.0

    def test_get_cached_gas_gwei_initial(self):
        est = GasEstimator()
        assert est.get_cached_gas_gwei() == 0.0

    def test_get_cached_gas_gwei_after_set(self):
        est = GasEstimator()
        est._cached_gas_gwei = 7.5
        assert est.get_cached_gas_gwei() == pytest.approx(7.5)

    @pytest.mark.asyncio
    async def test_cost_scales_with_eth_price(self):
        est = GasEstimator()
        est._cached_gas_gwei = 1.0
        est._cache_timestamp = time.time()

        # Use two different eth prices
        est._cached_eth_price = 1000.0
        cost_low = await est.estimate_gas_cost_usd(tx_type="swap", rpc_url="", eth_price=1000.0)

        est._cached_eth_price = 4000.0
        cost_high = await est.estimate_gas_cost_usd(tx_type="swap", rpc_url="", eth_price=4000.0)

        assert cost_high == pytest.approx(cost_low * 4.0, rel=1e-3)


# ---------------------------------------------------------------------------
# VirtualLedger tests
# ---------------------------------------------------------------------------


class TestVirtualLedger:
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        # Should not raise — we can read balance
        balance = await ledger.get_balance("USDC")
        assert balance == 0.0
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_update_and_get_balance(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.update_balance("USDC", 5000.0, usd_value=5000.0)
        assert await ledger.get_balance("USDC") == pytest.approx(5000.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_balance_missing_asset_returns_zero(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        balance = await ledger.get_balance("NONEXISTENT_TOKEN")
        assert balance == 0.0
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_total_equity_usd_calculation(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.update_balance("USDC", 5000.0, usd_value=5000.0)
        await ledger.update_balance("WETH", 1.0, usd_value=2000.0)
        await ledger.update_balance("aUSDC", 500.0, usd_value=500.0)
        total = await ledger.get_total_equity_usd(eth_price=2000.0)
        assert total == pytest.approx(7500.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_record_trade_updates_in_memory_state(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.record_trade(
            direction="BUY",
            asset_pair="WETH/USDC",
            entry_price=2000.0,
            simulated_fill_price=1999.0,
            slippage_bps=5.0,
            size_usd=100.0,
            gas_cost_usd=0.01,
            pnl_usd=0.0,
            reason_codes=("test",),
            signal_source="unit_test",
        )
        assert ledger._trade_count == 1
        assert ledger._realized_pnl == pytest.approx(0.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_record_sell_trade_updates_closed_count(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.record_trade(
            direction="SELL",
            asset_pair="WETH/USDC",
            entry_price=2000.0,
            simulated_fill_price=2010.0,
            slippage_bps=5.0,
            size_usd=100.0,
            gas_cost_usd=0.01,
            pnl_usd=10.0,
            reason_codes=("test",),
            signal_source="unit_test",
        )
        assert ledger._closed_trade_count == 1
        assert ledger._win_count == 1
        assert ledger._realized_pnl == pytest.approx(10.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_pnl_summary_zero_trades(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        summary = await ledger.get_pnl_summary()
        assert summary["trade_count"] == 0
        assert summary["win_rate"] == 0.0
        assert summary["avg_slippage_bps"] == 0.0
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_pnl_summary_after_win(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.record_trade(
            direction="SELL",
            asset_pair="WETH/USDC",
            entry_price=2000.0,
            simulated_fill_price=2100.0,
            slippage_bps=5.0,
            size_usd=1000.0,
            gas_cost_usd=0.01,
            pnl_usd=50.0,
            reason_codes=(),
            signal_source="test",
        )
        summary = await ledger.get_pnl_summary()
        assert summary["win_count"] == 1
        assert summary["win_rate"] == pytest.approx(100.0)
        assert summary["realized_pnl"] == pytest.approx(50.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_drawdown_tracking(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        # Manually set equity history to simulate drawdown
        ledger._peak_equity = 10000.0
        ledger._equity_history = [10000.0, 9500.0, 9000.0]
        dd = await ledger.get_drawdown()
        # Current equity from history = 9000, peak = 10000 → drawdown = 10%
        assert dd["current_drawdown"] == pytest.approx(0.1, rel=1e-3)
        assert dd["peak_equity"] == pytest.approx(10000.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_record_decision_and_retrieve(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.record_decision(
            wallet_id="wallet_1",
            cycle_phase="shadow_cycle",
            action="HOLD",
            regime_label="normal",
            setup_tag="signal_wait",
            decision_class="NO_TRADE",
            primary_reason="BELOW_THRESHOLD",
            reason_codes=("paper_shadow_only",),
            raw_alpha=0.02,
            robust_alpha=0.01,
        )
        decisions = await ledger.get_recent_decisions(5)
        assert len(decisions) == 1
        assert decisions[0]["action"] == "HOLD"
        assert decisions[0]["primary_reason"] == "BELOW_THRESHOLD"
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_decision_analytics_summary(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        for action, decision_class in [("BUY", "EXECUTED"), ("HOLD", "NO_TRADE"), ("SKIP", "SKIP")]:
            await ledger.record_decision(
                wallet_id="w1",
                cycle_phase="shadow_cycle",
                action=action,
                regime_label="normal",
                setup_tag="test_tag",
                decision_class=decision_class,
                primary_reason="TEST_REASON",
                reason_codes=(),
            )
        analytics = await ledger.get_decision_analytics()
        assert analytics["summary"]["total_decisions"] == 3
        assert analytics["summary"]["executed_decisions"] == 1
        assert analytics["summary"]["no_trade_decisions"] == 1
        assert analytics["summary"]["skip_decisions"] == 1
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_record_balance_history(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.update_balance("USDC", 8000.0)
        await ledger.update_balance("WETH", 1.0)
        await ledger.record_balance_history(eth_price=2000.0)
        curve = await ledger.get_equity_curve_windowed()
        assert len(curve) == 1
        assert curve[0]["total_equity_usd"] == pytest.approx(10000.0)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_best_worst_last_trade(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        await ledger.record_trade(
            direction="SELL", asset_pair="WETH/USDC",
            entry_price=2000.0, simulated_fill_price=2100.0,
            slippage_bps=5.0, size_usd=200.0, gas_cost_usd=0.01,
            pnl_usd=20.0, reason_codes=(), signal_source="test",
        )
        await ledger.record_trade(
            direction="SELL", asset_pair="WETH/USDC",
            entry_price=2200.0, simulated_fill_price=2100.0,
            slippage_bps=5.0, size_usd=200.0, gas_cost_usd=0.01,
            pnl_usd=-15.0, reason_codes=(), signal_source="test",
        )
        result = await ledger.get_best_worst_last_trade()
        assert result["best_trade"]["pnl_usd"] == pytest.approx(20.0)
        assert result["worst_trade"]["pnl_usd"] == pytest.approx(-15.0)
        assert result["last_trade"] is not None
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_get_recent_trades_returns_correct_count(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        for i in range(5):
            await ledger.record_trade(
                direction="BUY", asset_pair="WETH/USDC",
                entry_price=2000.0 + i, simulated_fill_price=1999.0 + i,
                slippage_bps=5.0, size_usd=100.0, gas_cost_usd=0.01,
                pnl_usd=0.0, reason_codes=(), signal_source="test",
            )
        trades = await ledger.get_recent_trades(n=3)
        assert len(trades) == 3
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_assert_open_raises_when_not_initialized(self):
        ledger = VirtualLedger(db_path=":memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            ledger._assert_open()

    @pytest.mark.asyncio
    async def test_estimate_sharpe_returns_zero_with_insufficient_data(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        ledger._equity_history = [1000.0, 1010.0]  # less than 10 points
        sharpe = ledger._estimate_sharpe()
        assert sharpe == 0.0
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_estimate_sharpe_with_sufficient_data(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        # Give 10+ equity values with variance
        ledger._equity_history = [
            10000, 10010, 10025, 10015, 10030,
            10020, 10040, 10035, 10050, 10045, 10060
        ]
        sharpe = ledger._estimate_sharpe()
        # Should be non-zero since there's variance and positive drift
        assert isinstance(sharpe, float)
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_hold_time_stats_empty(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        stats = await ledger.get_hold_time_stats()
        assert stats["completed_pairs"] == 0
        assert stats["avg_hold_seconds"] == 0.0
        await ledger.shutdown()

    @pytest.mark.asyncio
    async def test_advanced_analytics_empty_returns_zeros(self, tmp_path):
        ledger = VirtualLedger(db_path=str(tmp_path / "test.db"))
        await ledger.initialize()
        result = await ledger.get_advanced_analytics()
        assert result["expectancy_usd"] == 0.0
        assert result["closed_count"] == 0
        assert result["sample_sufficient"] is False
        await ledger.shutdown()


# ---------------------------------------------------------------------------
# PaperTradingEngine initialization tests
# ---------------------------------------------------------------------------


class TestPaperTradingEngineInit:
    def test_init_default_config(self):
        reset_paper_engine()
        engine = PaperTradingEngine()
        assert engine._config.initial_balance_usd == 10_000.0
        assert engine._initialized is False
        assert engine._in_position is False
        assert engine._cycle_count == 0

    def test_init_with_custom_config(self):
        reset_paper_engine()
        cfg = PaperTradeConfig(initial_balance_usd=25_000.0)
        engine = PaperTradingEngine(config=cfg)
        assert engine._config.initial_balance_usd == 25_000.0

    def test_init_with_initial_capital_usd(self):
        reset_paper_engine()
        engine = PaperTradingEngine(initial_capital_usd=15_000.0)
        assert engine._config.initial_balance_usd == 15_000.0

    def test_init_creates_correct_subcomponents(self):
        reset_paper_engine()
        engine = PaperTradingEngine()
        assert isinstance(engine._ledger, VirtualLedger)
        assert isinstance(engine._slippage, SlippageSimulator)
        assert isinstance(engine._gas_estimator, GasEstimator)

    def test_slippage_params_from_config(self):
        reset_paper_engine()
        cfg = PaperTradeConfig(slippage_mean_bps=12.0, slippage_std_bps=4.0)
        engine = PaperTradingEngine(config=cfg)
        assert engine._slippage._mean_bps == 12.0
        assert engine._slippage._std_bps == 4.0


# ---------------------------------------------------------------------------
# _classify_setup_tag helper tests
# ---------------------------------------------------------------------------


class TestClassifySetupTag:
    def test_news_event(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="SKIP", primary_reason="NEWS_EVENT_IMMINENT: NFP",
            in_position=False, robust_alpha=0.0,
        )
        assert tag == "macro_event_override"

    def test_flash_crash(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="FLASH_CRASH", primary_reason="FLASH_CRASH",
            in_position=False, robust_alpha=0.0,
        )
        assert tag == "flash_crash_block"

    def test_risk_blocked(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="RISK_BLOCKED", primary_reason="RISK_BLOCKED: VOL",
            in_position=False, robust_alpha=0.0,
        )
        assert tag == "risk_blocked"

    def test_gas_too_expensive(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="HOLD", primary_reason="GAS_TOO_EXPENSIVE",
            in_position=False, robust_alpha=0.0,
        )
        assert tag == "gas_filter_block"

    def test_buy_positive_alpha(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="BUY", primary_reason="LONG_ENTRY",
            in_position=False, robust_alpha=0.3,
        )
        assert tag == "momentum_long"

    def test_buy_negative_alpha(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="BUY", primary_reason="LONG_ENTRY",
            in_position=False, robust_alpha=-0.2,
        )
        assert tag == "countertrend_long"

    def test_sell(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="SELL", primary_reason="EXIT",
            in_position=True, robust_alpha=0.0,
        )
        assert tag == "exit_long"

    def test_hold_in_position(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="HOLD", primary_reason="NO_SIGNAL",
            in_position=True, robust_alpha=0.0,
        )
        assert tag == "position_hold"

    def test_warmup(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="WARMUP", primary_reason="WARMUP",
            in_position=False, robust_alpha=0.0,
        )
        assert tag == "warmup"

    def test_signal_wait_default(self):
        tag = PaperTradingEngine._classify_setup_tag(
            action="HOLD", primary_reason="BELOW_THRESHOLD",
            in_position=False, robust_alpha=0.0,
        )
        assert tag == "signal_wait"


# ---------------------------------------------------------------------------
# _classify_decision_class tests
# ---------------------------------------------------------------------------


class TestClassifyDecisionClass:
    def test_buy_is_executed(self):
        assert PaperTradingEngine._classify_decision_class("BUY") == "EXECUTED"

    def test_sell_is_executed(self):
        assert PaperTradingEngine._classify_decision_class("SELL") == "EXECUTED"

    def test_hold_is_no_trade(self):
        assert PaperTradingEngine._classify_decision_class("HOLD") == "NO_TRADE"

    def test_warmup_is_no_trade(self):
        assert PaperTradingEngine._classify_decision_class("WARMUP") == "NO_TRADE"

    def test_price_error_is_error(self):
        assert PaperTradingEngine._classify_decision_class("PRICE_ERROR") == "ERROR"

    def test_flash_crash_is_error(self):
        assert PaperTradingEngine._classify_decision_class("FLASH_CRASH") == "ERROR"

    def test_invalid_price_is_error(self):
        assert PaperTradingEngine._classify_decision_class("INVALID_PRICE") == "ERROR"

    def test_unknown_is_skip(self):
        assert PaperTradingEngine._classify_decision_class("CUSTOM_SKIP") == "SKIP"


# ---------------------------------------------------------------------------
# PaperTradingEngine initialize / shutdown tests
# ---------------------------------------------------------------------------


class TestPaperTradingEngineLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_sets_initial_balance(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(paper_db_path=str(tmp_path / "p.db"))
            engine = PaperTradingEngine(config=PaperTradeConfig(initial_balance_usd=7500.0))
            await engine.initialize()
            balance = await engine._ledger.get_balance("USDC")
            assert balance == pytest.approx(7500.0)
            assert engine._initialized is True
            await engine.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(paper_db_path=str(tmp_path / "p.db"))
            engine = PaperTradingEngine()
            await engine.initialize()
            await engine.initialize()  # should not raise or reset state
            assert engine._initialized is True
            await engine.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_clears_initialized_flag(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(paper_db_path=str(tmp_path / "p.db"))
            engine = PaperTradingEngine()
            await engine.initialize()
            await engine.shutdown()
            assert engine._initialized is False

    @pytest.mark.asyncio
    async def test_get_performance_report_after_initialize(self, tmp_path):
        reset_paper_engine()
        with patch("wallet_manager.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(paper_db_path=str(tmp_path / "p.db"))
            engine = PaperTradingEngine()
            await engine.initialize()
            report = await engine.get_performance_report()
            assert isinstance(report, dict)
            await engine.shutdown()


# ---------------------------------------------------------------------------
# Step-based API tests
# ---------------------------------------------------------------------------


class TestPaperTradingEngineStepAPI:
    @pytest.mark.asyncio
    async def test_step_opens_position_on_strong_signal(self):
        engine = PaperTradingEngine(min_confidence=0.5)
        await engine.step(price=2000.0, signal=0.8)
        assert engine.current_exposure_usd > 0.0

    @pytest.mark.asyncio
    async def test_step_does_not_open_on_weak_signal(self):
        engine = PaperTradingEngine(min_confidence=0.5)
        await engine.step(price=2000.0, signal=0.3)
        assert engine.current_exposure_usd == 0.0

    @pytest.mark.asyncio
    async def test_step_closes_position_on_weak_signal(self):
        engine = PaperTradingEngine(min_confidence=0.5)
        await engine.step(price=2000.0, signal=0.8)
        assert engine.current_exposure_usd > 0.0
        await engine.step(price=2100.0, signal=0.2)
        assert engine.current_exposure_usd == 0.0

    @pytest.mark.asyncio
    async def test_step_pnl_profitable_trade(self):
        engine = PaperTradingEngine(min_confidence=0.5)
        await engine.step(price=2000.0, signal=0.8)
        await engine.step(price=2200.0, signal=0.2)  # close at profit
        metrics = engine.get_metrics()
        assert metrics["total_pnl_usd"] > 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_returns_expected_keys(self):
        engine = PaperTradingEngine(min_confidence=0.5)
        metrics = engine.get_metrics()
        assert "total_pnl_usd" in metrics
        assert "total_trades" in metrics
        assert "sharpe_ratio" in metrics
        assert "final_capital_usd" in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_zero_trades(self):
        engine = PaperTradingEngine()
        metrics = engine.get_metrics()
        assert metrics["total_trades"] == 0
        assert metrics["sharpe_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Constant / module-level tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_gas_units_keys_present(self):
        assert "swap" in GAS_UNITS
        assert "supply" in GAS_UNITS
        assert "borrow" in GAS_UNITS

    def test_gas_units_positive(self):
        for tx_type, units in GAS_UNITS.items():
            assert units > 0, f"{tx_type} gas units should be positive"

    def test_reason_code_strings(self):
        assert isinstance(PAPER_TRADE_EXECUTED, str)
        assert isinstance(PAPER_SLIPPAGE_APPLIED, str)
        assert isinstance(PAPER_GAS_ESTIMATED, str)
        assert isinstance(PAPER_SIGNAL_REJECTED, str)
        assert isinstance(PAPER_SHADOW_ONLY, str)
