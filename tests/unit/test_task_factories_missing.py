"""tests/unit/test_task_factories_missing.py — Extra coverage for task_factories.py"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.task_factories import (
    _collect_alerter_inputs,
    _default_task_factory,
    _maybe_run_agents,
    _preload_robust_engine,
    _robust_paper_task_factory,
    _robust_task_factory,
)


@pytest.mark.asyncio
async def test_maybe_run_agents_time_elapsed():
    """Test _maybe_run_agents when time interval elapses and reports are delivered."""
    mock_engine = MagicMock()
    mock_report = SimpleNamespace(used_llm=True, title="Test", body="Body")
    mock_registry = MagicMock(run_all=AsyncMock(return_value=[mock_report]))
    mock_notifier = MagicMock(send_message=AsyncMock())

    with patch("wallet_manager.task_factories._last_agent_run_ts", 0.0), \
         patch("wallet_manager.ai_agents.get_agent_registry", return_value=mock_registry), \
         patch("wallet_manager.strategies.paper_engine._engine_instance", mock_engine):

        await _maybe_run_agents({"notifier": mock_notifier})
        mock_registry.run_all.assert_called_once()
        mock_notifier.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_run_agents_long_report():
    """Test _maybe_run_agents truncation for long reports."""
    mock_registry = MagicMock(run_all=AsyncMock(return_value=[SimpleNamespace(used_llm=False, title="Test", body="A"*500)]))
    mock_notifier = MagicMock(send_message=AsyncMock())

    with patch("wallet_manager.task_factories._last_agent_run_ts", 0.0), \
         patch("wallet_manager.ai_agents.get_agent_registry", return_value=mock_registry), \
         patch("wallet_manager.strategies.paper_engine._engine_instance", MagicMock()):
        await _maybe_run_agents({"notifier": mock_notifier})
        mock_notifier.send_message.assert_called_once()
        args, _kwargs = mock_notifier.send_message.call_args
        assert "truncated" in args[0]


@pytest.mark.asyncio
async def test_collect_alerter_inputs_all_exceptions_caught():
    """Test _collect_alerter_inputs when sub-components fail safely."""
    with patch("wallet_manager.task_factories._drift_detector", None), \
         patch("wallet_manager.strategies.yield_strategy.get_apy_monitor", side_effect=RuntimeError("boom")):
        res = await _collect_alerter_inputs(None, MagicMock(ppo_model_path="invalid_path"))
        assert res["drift_level"] == "NOMINAL"
        assert res["apy"] == 0.0


@pytest.mark.asyncio
async def test_collect_alerter_inputs_success_paths():
    """Test _collect_alerter_inputs with valid mocks."""
    mock_snapshot = SimpleNamespace(
        total_value_usd=1000.0,
        allocations=[SimpleNamespace(current_usd=1200.0)]
    )
    with patch("wallet_manager.task_factories._drift_detector", SimpleNamespace(_last_drift_level=SimpleNamespace(name="HIGH"))), \
         patch("wallet_manager.strategies.yield_strategy.get_apy_monitor", return_value=SimpleNamespace(last_apy=5.0)), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=0.0), \
         patch("wallet_manager.orchestrator._get_rpc_router", return_value=SimpleNamespace(_latency_tracker={"a": [10.0, 20.0]})):

        res = await _collect_alerter_inputs(mock_snapshot, MagicMock(ppo_model_path="test"))
        assert res["drift_level"] == "HIGH"
        assert res["apy"] == 5.0
        assert res["rpc_latency_ms"] == 20.0
        assert res["drawdown"] > 0


@pytest.mark.asyncio
async def test_default_task_factory_unsafe_gas():
    """Gas monitor not safe causes wait."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", max_gas_price_gwei=50.0)))
    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(False, 100.0))), \
         patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1), \
         patch("asyncio.sleep", AsyncMock()):
        res = await _default_task_factory(manager, "w1", {})
        assert res == []


@pytest.mark.asyncio
async def test_default_task_factory_safe_gas_success():
    """Gas monitor safe fetches balance."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", max_gas_price_gwei=50.0)))
    manager.get_safe_balance = AsyncMock()
    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 10.0))), \
         patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 2), \
         patch("asyncio.sleep", AsyncMock()):
        await _default_task_factory(manager, "w1", {})
        manager.get_safe_balance.assert_called()


@pytest.mark.asyncio
async def test_preload_robust_engine_coingecko_unavailable():
    """Pre-seed falls back to synthetic prices if CoinGecko is unavailable."""
    engine = MagicMock(_n_ticks=0, ingest_price=MagicMock())
    with patch("aiohttp.ClientSession", side_effect=Exception("Failed")):
        await _preload_robust_engine(engine, 2000.0)
        assert engine.ingest_price.call_count > 100


@pytest.mark.asyncio
async def test_preload_robust_engine_coingecko_success():
    """Pre-seed parses CoinGecko prices."""
    engine = MagicMock(_n_ticks=0, ingest_price=MagicMock())
    mock_response = AsyncMock(status=200, json=AsyncMock(return_value={"prices": [[1, 2000.0], [2, 2050.0]]}))
    mock_session = MagicMock()
    mock_session.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
    with patch("aiohttp.ClientSession", return_value=mock_session):
        await _preload_robust_engine(engine, 2000.0)
        assert engine.ingest_price.call_count > 0


@pytest.mark.asyncio
async def test_robust_paper_task_factory_skips_wrong_chain():
    """_robust_paper_task_factory skips non-Base."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=1, address="0x", rpc_url="url", private_key="a"*64)))
    res = await _robust_paper_task_factory(manager, "w1", {"portfolio_manager": MagicMock()})
    assert res == []


@pytest.mark.asyncio
async def test_robust_paper_task_factory_latency_invalidates():
    """_robust_paper_task_factory latency invalidation skips cycle."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_shield = MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=True, quote_age_s=5, reason_code="old"), None)))
    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=mock_shield), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.paper_engine.get_paper_engine", AsyncMock(return_value=MagicMock(record_external_skip=AsyncMock()))):
        res = await _robust_paper_task_factory(manager, "w1", {"portfolio_manager": MagicMock()})
        assert res == []


@pytest.mark.asyncio
async def test_robust_task_factory_rebalance_trigger():
    """Scheduler rebalance is executed if due."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    from wallet_manager.portfolio_manager import StrategyType
    mock_pm = MagicMock(rebalance_portfolio=MagicMock(return_value={StrategyType.YIELD: 1, StrategyType.ALGO: 1, StrategyType.AIRDROP: 1}), _initial_budget=100)
    mock_sched = MagicMock(is_monthly_rebalance_due=MagicMock(return_value=True), is_initial_rebalance=MagicMock(return_value=True))
    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(False, 50.0))):
        await _robust_task_factory(manager, "w1", {"portfolio_manager": mock_pm, "scheduler": mock_sched})
        mock_pm.rebalance_portfolio.assert_called_once()
        mock_sched.mark_monthly_rebalance_done.assert_called_once()


@pytest.mark.asyncio
async def test_robust_task_factory_regime_override():
    """Regime HMM safety override forces Aave supply."""
    manager = MagicMock(get_config=MagicMock(return_value=SimpleNamespace(chain_id=8453, address="0x", rpc_url="url", private_key="a"*64, max_gas_price_gwei=50.0)))
    mock_regime = MagicMock(should_override_to_safety=MagicMock(return_value=(True, "DANGER")))

    with patch("wallet_manager.network.gas_monitor.GasMonitor.is_safe", AsyncMock(return_value=(True, 5.0))), \
         patch("wallet_manager.actions.fetch_eth_price_usd", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.latency_shield.get_latency_shield", MagicMock(return_value=MagicMock(apply_latency_shield=AsyncMock(return_value=(SimpleNamespace(should_invalidate_signals=False), None))))), \
         patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=mock_regime), \
         patch("wallet_manager.strategies.yield_strategy.get_usdc_balance", AsyncMock(return_value=10000000)), \
         patch("wallet_manager.strategies.yield_strategy.execute_supply_usdc", AsyncMock(return_value=[1, 2])), \
         patch("wallet_manager.orchestrator.send_heartbeat_telemetry", AsyncMock()):

        res = await _robust_task_factory(manager, "w1", {"portfolio_manager": MagicMock()})
        assert res == [1, 2]
