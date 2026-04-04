"""tests/unit/test_task_factories_missing2.py — Extra coverage for task_factories.py"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.task_factories import (
    _robust_paper_task_factory,
    _robust_task_factory,
)


@pytest.mark.asyncio
async def test_robust_paper_task_factory_macro_suppress():
    """_robust_paper_task_factory macro calendar returns early."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_shield = MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), None)))
    mock_macro = MagicMock(should_suppress_alpha=MagicMock(return_value=(True, "Fed Meeting")))

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=mock_shield), \
         patch("wallet_manager.market_intelligence.get_macro_calendar", return_value=mock_macro), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.paper_engine.get_paper_engine", AsyncMock(return_value=MagicMock(record_external_skip=AsyncMock()))):
        res = await _robust_paper_task_factory(manager, "w1", {"portfolio_manager": MagicMock()})
        assert res == []


@pytest.mark.asyncio
async def test_robust_paper_task_factory_regime_override():
    """_robust_paper_task_factory regime filter safety override returns early."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_shield = MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), None)))
    mock_macro = MagicMock(should_suppress_alpha=MagicMock(return_value=(False, "OK")))
    mock_regime = MagicMock(should_override_to_safety=MagicMock(return_value=(True, "DANGER")), get_current_regime=MagicMock(return_value=SimpleNamespace(regime=SimpleNamespace(value="HIGH_VOL"))))

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=mock_shield), \
         patch("wallet_manager.market_intelligence.get_macro_calendar", return_value=mock_macro), \
         patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=mock_regime), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.paper_engine.get_paper_engine", AsyncMock(return_value=MagicMock(record_external_skip=AsyncMock()))):
        res = await _robust_paper_task_factory(manager, "w1", {"portfolio_manager": MagicMock()})
        assert res == []


@pytest.mark.asyncio
async def test_robust_paper_task_factory_stress_test_reject():
    """_robust_paper_task_factory stress test reject."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_shield = MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), None)))
    mock_macro = MagicMock(should_suppress_alpha=MagicMock(return_value=(False, "OK")))
    mock_regime = MagicMock(should_override_to_safety=MagicMock(return_value=(False, "OK")), get_current_regime=MagicMock(return_value=SimpleNamespace(regime=SimpleNamespace(value="LOW_VOL"))))

    mock_robust = MagicMock(_n_ticks=1, ingest_price=MagicMock(return_value=SimpleNamespace(should_enter_long=True, should_enter_short=False, robust_alpha=0.5)), _price_history=[])
    mock_stress = MagicMock(validate_signal=MagicMock(return_value=SimpleNamespace(approved=False, monte_carlo=SimpleNamespace(win_rate=0.4), rejection_reasons=["MC"])))

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=mock_shield), \
         patch("wallet_manager.market_intelligence.get_macro_calendar", return_value=mock_macro), \
         patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=mock_regime), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.stochastic_stress_test.get_stress_tester", return_value=mock_stress), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.paper_engine.get_paper_engine", AsyncMock(return_value=MagicMock(record_external_skip=AsyncMock()))):
        res = await _robust_paper_task_factory(manager, "w1", {"portfolio_manager": MagicMock()})
        assert res == []


@pytest.mark.asyncio
async def test_robust_paper_task_factory_full_success():
    """_robust_paper_task_factory stress test pass and paper engine executes."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_shield = MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), SimpleNamespace(total_latency_ms=10.0))))
    mock_macro = MagicMock(should_suppress_alpha=MagicMock(return_value=(False, "OK")))
    mock_regime = MagicMock(should_override_to_safety=MagicMock(return_value=(False, "OK")), get_current_regime=MagicMock(return_value=SimpleNamespace(regime=SimpleNamespace(value="LOW_VOL"))))

    mock_robust = MagicMock(_n_ticks=1, ingest_price=MagicMock(return_value=SimpleNamespace(should_enter_long=True, should_enter_short=False, robust_alpha=0.5)), _price_history=[])
    mock_stress = MagicMock(validate_signal=MagicMock(return_value=SimpleNamespace(approved=True, monte_carlo=SimpleNamespace(win_rate=0.8), confidence_score=0.9)))

    mock_paper = MagicMock(
        run_shadow_cycle=AsyncMock(return_value=SimpleNamespace(action="BUY", signal_alpha=0.5, simulated_fill_price=2000, slippage_bps=5.0, gas_cost_usd=1.0, virtual_pnl_usd=0.0, cycle_duration_ms=100.0)),
        get_performance_report=AsyncMock(),
        get_telegram_report=AsyncMock(return_value="report")
    )

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=mock_shield), \
         patch("wallet_manager.market_intelligence.get_macro_calendar", return_value=mock_macro), \
         patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=mock_regime), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.stochastic_stress_test.get_stress_tester", return_value=mock_stress), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.paper_engine.get_paper_engine", AsyncMock(return_value=mock_paper)), \
         patch("wallet_manager.orchestrator.send_heartbeat_telemetry", AsyncMock()), \
         patch("wallet_manager.task_factories._maybe_run_agents", AsyncMock()), \
         patch("wallet_manager.reporting.get_reporter", MagicMock(return_value=MagicMock(tick=AsyncMock()))), \
         patch("wallet_manager.alerting.get_alert_system", MagicMock(return_value=MagicMock(tick=AsyncMock()))):

        ctx = {"portfolio_manager": MagicMock(), "notifier": MagicMock()}
        res = await _robust_paper_task_factory(manager, "w1", ctx)
        assert res == []
        mock_paper.run_shadow_cycle.assert_called_once()


@pytest.mark.asyncio
async def test_robust_task_factory_full_flow():
    """_robust_task_factory all checks pass to execute ML Quant Engine."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))

    mock_pm = MagicMock(get_pillar_budget_usd=MagicMock(return_value=5.0)) # low airdrop budget
    MagicMock(get_yield_pillar_value_usd=AsyncMock(return_value=100.0))
    manager.get_safe_balance = AsyncMock(return_value=1000)

    mock_shield = MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), None)))
    mock_regime = MagicMock(should_override_to_safety=MagicMock(return_value=(False, "OK")), get_current_regime=MagicMock(return_value=SimpleNamespace(regime=SimpleNamespace(value="LOW_VOL"))))

    mock_robust = MagicMock(_n_ticks=1, ingest_price=MagicMock(return_value=SimpleNamespace(should_enter_long=True, should_enter_short=False, robust_alpha=0.5)), _price_history=[], _current_d=0.2, _is_stationary=True)
    mock_stress = MagicMock(validate_signal=MagicMock(return_value=SimpleNamespace(approved=True, monte_carlo=SimpleNamespace(win_rate=0.8), confidence_score=0.9)))
    mock_wf = MagicMock(should_retrain=MagicMock(return_value=False), _controller=SimpleNamespace(_trade_count=10))

    mock_ml = MagicMock(run_cycle=AsyncMock(return_value=SimpleNamespace(receipts=[MagicMock(tx_hash="1")], sweep_receipts=[], action_taken="BUY", risk_source="model", drawdown_state="normal", var_99=0.01, pnl_usd=1.0, cycle_duration_ms=100.0)))

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=1000)), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.actions.eth_balance_usd_value", return_value=100.0), \
         patch("wallet_manager.actions.usdc_balance_usd_value", return_value=100.0), \
         patch("wallet_manager.strategies.yield_strategy.get_yield_pillar_value_usd", AsyncMock(return_value=100.0)), \
         patch("wallet_manager.task_factories.get_settings", MagicMock(return_value=SimpleNamespace(algo_weight=50, airdrop_weight=20))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=mock_shield), \
         patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=mock_regime), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.stochastic_stress_test.get_stress_tester", return_value=mock_stress), \
         patch("wallet_manager.strategies.wf_validator.get_wf_validator", AsyncMock(return_value=mock_wf)), \
         patch("wallet_manager.strategies.liquidity_alpha.get_liquidity_alpha", return_value=MagicMock()), \
         patch("wallet_manager.strategies.quant_analytics.get_quant_analytics", AsyncMock(return_value=MagicMock())), \
         patch("wallet_manager.strategies.execution_logic.get_ml_execution_engine", return_value=mock_ml), \
         patch("wallet_manager.orchestrator.send_heartbeat_telemetry", AsyncMock()):

        ctx = {"portfolio_manager": mock_pm}
        res = await _robust_task_factory(manager, "w1", ctx)
        assert len(res) == 1
        assert res[0].tx_hash == "1"


@pytest.mark.asyncio
async def test_robust_task_factory_wf_retrain():
    """_robust_task_factory triggers wf_validator retrain if should_retrain=True."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_pm = MagicMock(get_pillar_budget_usd=MagicMock(return_value=0.0))
    mock_wf = MagicMock(should_retrain=MagicMock(return_value=True), run_retrain_cycle=AsyncMock(return_value=SimpleNamespace(action="RETRAIN", mean_oos_sharpe=2.0)), _controller=SimpleNamespace(_trade_count=10))
    mock_regime = MagicMock(should_override_to_safety=MagicMock(return_value=(False, "OK")), get_current_regime=MagicMock(return_value=SimpleNamespace(regime=SimpleNamespace(value="LOW_VOL"))))
    mock_ml = MagicMock(run_cycle=AsyncMock(return_value=SimpleNamespace(receipts=[], sweep_receipts=[], action_taken="HOLD", risk_source="model", drawdown_state="normal", var_99=0.01, pnl_usd=0.0, cycle_duration_ms=10.0)))

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.actions.fetch_usdc_balance", AsyncMock(return_value=1000)), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.actions.eth_balance_usd_value", return_value=100.0), \
         patch("wallet_manager.actions.usdc_balance_usd_value", return_value=100.0), \
         patch("wallet_manager.strategies.yield_strategy.get_yield_pillar_value_usd", AsyncMock(return_value=100.0)), \
         patch("wallet_manager.task_factories.get_settings", MagicMock(return_value=SimpleNamespace(algo_weight=50, airdrop_weight=20))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", MagicMock(return_value=MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), None))))), \
         patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=mock_regime), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=MagicMock(_n_ticks=1, ingest_price=MagicMock(return_value=None), _current_d=0.2, _is_stationary=True)), \
         patch("wallet_manager.strategies.stochastic_stress_test.get_stress_tester", return_value=MagicMock()), \
         patch("wallet_manager.strategies.wf_validator.get_wf_validator", AsyncMock(return_value=mock_wf)), \
         patch("wallet_manager.strategies.liquidity_alpha.get_liquidity_alpha", return_value=MagicMock()), \
         patch("wallet_manager.strategies.quant_analytics.get_quant_analytics", AsyncMock(return_value=MagicMock())), \
         patch("wallet_manager.strategies.execution_logic.get_ml_execution_engine", return_value=mock_ml), \
         patch("wallet_manager.orchestrator.send_heartbeat_telemetry", AsyncMock()):

        await _robust_task_factory(manager, "w1", {"portfolio_manager": mock_pm})
        mock_wf.run_retrain_cycle.assert_called_once()
