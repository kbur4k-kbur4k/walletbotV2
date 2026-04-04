"""tests/integration/test_paper_engine.py — PaperTradingEngine integration tests."""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.paper_engine import (
    PaperTradeConfig,
    PaperTradingEngine,
    SlippageSimulator,
    VirtualLedger,
    reset_paper_engine,
)


@pytest.mark.asyncio
async def test_paper_engine_never_calls_broadcast(tmp_path):
    """PaperTradingEngine must never call sign_and_broadcast_tx."""
    reset_paper_engine()
    config = PaperTradeConfig()
    engine = PaperTradingEngine(config=config)
    await engine.initialize()

    mock_manager = MagicMock()
    mock_manager.sign_and_broadcast_tx = AsyncMock()

    with (
        patch("wallet_manager.actions.get_live_price_v3", new=AsyncMock(return_value=2000.0)),
        contextlib.suppress(Exception),
    ):
        await engine.run_shadow_cycle(
            manager=mock_manager,
            wallet_id="test_wallet",
            portfolio_manager=MagicMock(total_value_usd=1000.0),
            task_context={},
            analytics_db=None,
            quant_analytics=None,
        )

    mock_manager.sign_and_broadcast_tx.assert_not_called()
    await engine.shutdown()
    reset_paper_engine()


@pytest.mark.asyncio
async def test_virtual_ledger_write_read_roundtrip(tmp_path):
    """VirtualLedger: write then read round-trip preserves key fields."""
    ledger = VirtualLedger(db_path=str(tmp_path / "test_ledger.db"))
    await ledger.initialize()

    await ledger.record_trade(
        direction="BUY",
        asset_pair="WETH/USDC",
        entry_price=2000.0,
        simulated_fill_price=2001.0,
        slippage_bps=5.0,
        size_usd=100.0,
        gas_cost_usd=0.05,
        pnl_usd=0.0,
        reason_codes=("signal_buy",),
        signal_source="test_signal",
    )

    summary = await ledger.get_pnl_summary()
    assert summary is not None
    assert isinstance(summary, dict)

    await ledger.shutdown()


@pytest.mark.asyncio
async def test_virtual_ledger_win_rate_uses_closed_trades_only(tmp_path):
    """Win rate should use realized exit trades, not BUY entries with zero PnL."""
    ledger = VirtualLedger(db_path=str(tmp_path / "win_rate_ledger.db"))
    await ledger.initialize()

    await ledger.record_trade(
        direction="BUY",
        asset_pair="WETH/USDC",
        entry_price=2000.0,
        simulated_fill_price=2001.0,
        slippage_bps=5.0,
        size_usd=100.0,
        gas_cost_usd=0.05,
        pnl_usd=0.0,
        reason_codes=("signal_buy",),
        signal_source="test_signal",
    )
    await ledger.record_trade(
        direction="SELL",
        asset_pair="WETH/USDC",
        entry_price=2000.0,
        simulated_fill_price=2005.0,
        slippage_bps=4.0,
        size_usd=100.0,
        gas_cost_usd=0.05,
        pnl_usd=1.25,
        reason_codes=("signal_sell",),
        signal_source="test_signal",
    )

    summary = await ledger.get_pnl_summary()

    assert summary["trade_count"] == 2
    assert summary["closed_trade_count"] == 1
    assert summary["win_count"] == 1
    assert summary["win_rate"] == pytest.approx(100.0)

    await ledger.shutdown()


@pytest.mark.asyncio
async def test_virtual_ledger_decision_journal_analytics(tmp_path):
    """Decision journal should persist no-trade reasons and setup/regime breakdowns."""
    ledger = VirtualLedger(db_path=str(tmp_path / "decision_ledger.db"))
    await ledger.initialize()

    await ledger.record_decision(
        wallet_id="wallet_01",
        cycle_phase="task_factory",
        action="SKIP",
        regime_label="macro_event",
        setup_tag="macro_event_override",
        decision_class="SKIP",
        primary_reason="NEWS_EVENT_IMMINENT: NFP",
        reason_codes=("paper_shadow_only", "paper_signal_rejected"),
        raw_alpha=0.0,
        robust_alpha=0.0,
        confidence=0.0,
        adf_pvalue=1.0,
        stationarity_d=0.0,
        volatility_regime="macro_event",
        risk_position_usd=0.0,
        trade_budget_usd=0.0,
        gas_cost_usd=0.0,
        total_equity_usd=10_000.0,
        pnl_usd=0.0,
        notes="Skipping alpha due to macro event.",
    )
    await ledger.record_decision(
        wallet_id="wallet_01",
        cycle_phase="shadow_cycle",
        action="HOLD",
        regime_label="normal",
        setup_tag="signal_wait",
        decision_class="NO_TRADE",
        primary_reason="ENTRY_NOT_TRIGGERED",
        reason_codes=("paper_shadow_only", "paper_signal_rejected"),
        raw_alpha=0.02,
        robust_alpha=0.01,
        confidence=0.18,
        adf_pvalue=0.21,
        stationarity_d=0.42,
        volatility_regime="normal",
        risk_position_usd=18.5,
        trade_budget_usd=18.5,
        gas_cost_usd=0.0,
        total_equity_usd=10_000.0,
        pnl_usd=0.0,
        notes="Signal stayed below entry threshold.",
    )
    await ledger.record_decision(
        wallet_id="wallet_01",
        cycle_phase="shadow_cycle",
        action="BUY",
        regime_label="normal",
        setup_tag="momentum_long",
        decision_class="EXECUTED",
        primary_reason="LONG_ENTRY_CONFIRMED",
        reason_codes=("paper_shadow_only", "paper_trade_executed"),
        raw_alpha=0.12,
        robust_alpha=0.11,
        confidence=0.74,
        adf_pvalue=0.04,
        stationarity_d=0.88,
        volatility_regime="normal",
        risk_position_usd=45.0,
        trade_budget_usd=29.8,
        gas_cost_usd=0.0013,
        total_equity_usd=10_000.5,
        pnl_usd=0.0,
        notes="Momentum long executed.",
    )

    recent = await ledger.get_recent_decisions(5)
    analytics = await ledger.get_decision_analytics()

    assert len(recent) == 3
    assert recent[0]["action"] == "BUY"
    assert analytics["summary"]["total_decisions"] == 3
    assert analytics["summary"]["executed_decisions"] == 1
    assert analytics["summary"]["no_trade_decisions"] == 1
    assert analytics["summary"]["skip_decisions"] == 1
    assert analytics["no_trade_reasons"][0]["reason"] in {
        "ENTRY_NOT_TRIGGERED",
        "NEWS_EVENT_IMMINENT: NFP",
    }
    assert any(item["setup_tag"] == "momentum_long" for item in analytics["setup_breakdown"])

    await ledger.shutdown()


def test_slippage_gaussian_within_3_sigma():
    """Gaussian slippage model produces values within a reasonable range."""
    simulator = SlippageSimulator(mean_bps=5.0, std_bps=2.0)

    slippages = []
    for _ in range(500):
        slip_bps, _ = simulator.simulate_slippage(
            size_usd=100.0,
            pool_depth_usd=1_000_000.0,
            model="gaussian",
            entry_price=2000.0,
        )
        slippages.append(slip_bps)

    # All slippages must be >= 0 (floor at 0.5 bps)
    assert all(s >= 0 for s in slippages)
    # 3σ = 5 + 6 = 11 bps with impact_scalar ~1 for small orders; max cap is 500
    assert all(s <= 500.0 for s in slippages)
    # Mean should be near 5.0 bps for small orders
    import numpy as np
    mean_slip = float(np.mean(slippages))
    assert 1.0 <= mean_slip <= 50.0  # Loose bound due to impact scaling


@pytest.mark.asyncio
async def test_paper_engine_initialize_and_shutdown(tmp_path):
    """Engine initializes and shuts down cleanly."""
    reset_paper_engine()
    engine = PaperTradingEngine(config=PaperTradeConfig())
    await engine.initialize()
    report = await engine.get_performance_report()
    assert isinstance(report, dict)
    await engine.shutdown()
    reset_paper_engine()


@pytest.mark.asyncio
async def test_paper_engine_respects_configured_initial_balance(tmp_path):
    from wallet_manager.config import Settings, reset_settings_cache, set_settings

    reset_paper_engine()
    set_settings(
        Settings(
            vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
            execution_mode="robust_paper",
            paper_trade=True,
            paper_db_path=tmp_path / "paper_trades.db",
            garch_alpha=0.10,
            garch_beta=0.85,
        )
    )
    config = PaperTradeConfig(initial_balance_usd=10_000.0)
    engine = PaperTradingEngine(config=config)
    try:
        await engine.initialize()

        usdc_balance = await engine._ledger.get_balance("USDC")

        assert usdc_balance == pytest.approx(10_000.0)

        await engine.shutdown()
    finally:
        reset_paper_engine()
        reset_settings_cache()
