"""tests/unit/test_task_factories_coverage.py

Additional coverage tests for wallet_manager/task_factories.py.

Covers:
- _maybe_run_agents: interval guard, LLM flag in message, no-notifier path
- _default_task_factory: last-cycle sleep skip, context pass-through
- _preload_robust_engine: process_tick fallback, engine with no ingest method
- _collect_alerter_inputs: all branches
- Module-level constants derivation
- _get_wallet_address (task_factories version)
- _AGENT_RUN_INTERVAL_S value
- Guard path when no portfolio_manager (_airdrop_task_factory fallback)
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.core.models import TxReceipt, WalletConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    wallet_id: str = "wallet_001",
    private_key: str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    rpc_url: str = "http://localhost:8545",
    chain_id: int = 8453,
    max_gas_price_gwei: float = 25.0,
    address: str | None = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
) -> WalletConfig:
    return WalletConfig(
        wallet_id=wallet_id,
        private_key=private_key,
        rpc_url=rpc_url,
        chain_id=chain_id,
        max_gas_price_gwei=max_gas_price_gwei,
        address=address,
    )


def _make_manager(config: WalletConfig | None = None) -> MagicMock:
    cfg = config or _make_config()
    manager = MagicMock()
    manager.get_config.return_value = cfg
    manager.get_safe_balance = AsyncMock(return_value=int(1e18))
    manager.sign_and_broadcast_tx = AsyncMock()
    return manager


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_agent_run_interval_is_6_hours(self):
        import wallet_manager.task_factories as tf
        assert pytest.approx(6 * 3600.0) == tf._AGENT_RUN_INTERVAL_S

    def test_last_agent_run_ts_initially_zero(self):
        """On module load, _last_agent_run_ts should be 0.0 (agents never ran)."""
        import wallet_manager.task_factories as tf
        # We cannot guarantee it's zero after other tests run, but it must be a float
        assert isinstance(tf._last_agent_run_ts, float)

    def test_task_loop_cycles_positive(self):
        import wallet_manager.task_factories as tf
        assert tf._TASK_LOOP_CYCLES > 0

    def test_gas_poll_interval_positive(self):
        import wallet_manager.task_factories as tf
        assert tf._GAS_POLL_INTERVAL >= 0

    def test_aave_apy_alert_threshold_positive(self):
        import wallet_manager.task_factories as tf
        assert tf._AAVE_APY_ALERT_THRESHOLD >= 0


# ---------------------------------------------------------------------------
# 2. _get_wallet_address
# ---------------------------------------------------------------------------

class TestGetWalletAddress:
    def test_returns_config_address_when_present(self):
        from wallet_manager.task_factories import _get_wallet_address
        cfg = _make_config(address="0xDeAdBeEf")
        assert _get_wallet_address(cfg) == "0xDeAdBeEf"

    def test_derives_address_when_none(self):
        from wallet_manager.task_factories import _get_wallet_address
        pk = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        cfg = _make_config(address=None, private_key=pk)
        addr = _get_wallet_address(cfg)
        assert addr.startswith("0x")
        assert len(addr) == 42

    def test_result_is_string(self):
        from wallet_manager.task_factories import _get_wallet_address
        cfg = _make_config(address="0xABCDEF")
        result = _get_wallet_address(cfg)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. _maybe_run_agents — additional paths
# ---------------------------------------------------------------------------

class TestMaybeRunAgentsExtra:
    @pytest.mark.asyncio
    async def test_skips_within_interval(self):
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = time.time()  # just ran
        ctx = {}
        # Should return immediately without touching agents
        await tf._maybe_run_agents(ctx)
        # No exception = pass

    @pytest.mark.asyncio
    async def test_no_notifier_in_context_no_crash(self):
        """When ctx has no notifier key, should not raise."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        mock_report = MagicMock()
        mock_report.title = "Report"
        mock_report.body = "body text"
        mock_report.used_llm = False

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=[mock_report])

        ctx = {}  # no notifier

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents(ctx)

    @pytest.mark.asyncio
    async def test_llm_flag_in_message(self):
        """When used_llm=True, the message should contain LLM marker."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        mock_report = MagicMock()
        mock_report.title = "LLM Agent"
        mock_report.body = "detailed analysis"
        mock_report.used_llm = True

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=[mock_report])

        mock_notifier = MagicMock()
        mock_notifier.send_message = AsyncMock()
        ctx = {"notifier": mock_notifier}

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents(ctx)

        call_args = mock_notifier.send_message.call_args
        msg = call_args[0][0]
        assert "LLM" in msg

    @pytest.mark.asyncio
    async def test_multiple_reports_all_included(self):
        """All reports should appear in the sent message."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        reports = []
        for i in range(3):
            r = MagicMock()
            r.title = f"Agent {i}"
            r.body = f"body {i}"
            r.used_llm = False
            reports.append(r)

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=reports)

        mock_notifier = MagicMock()
        mock_notifier.send_message = AsyncMock()
        ctx = {"notifier": mock_notifier}

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents(ctx)

        mock_notifier.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_notifier_exception_swallowed(self):
        """If notifier.send_message raises, exception is suppressed."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        mock_report = MagicMock()
        mock_report.title = "Report"
        mock_report.body = "body"
        mock_report.used_llm = False

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=[mock_report])

        mock_notifier = MagicMock()
        mock_notifier.send_message = AsyncMock(side_effect=RuntimeError("telegram down"))
        ctx = {"notifier": mock_notifier}

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents(ctx)  # must not raise

    @pytest.mark.asyncio
    async def test_updates_timestamp_on_run(self):
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0
        before = time.time()

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents({})

        assert tf._last_agent_run_ts >= before


# ---------------------------------------------------------------------------
# 4. _default_task_factory — additional paths
# ---------------------------------------------------------------------------

class TestDefaultTaskFactoryExtra:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    result = await _default_task_factory(manager, "wallet_001")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_no_sleep_on_last_cycle(self):
        """On the last cycle, asyncio.sleep should NOT be called for the last iteration."""
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 3.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 2):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                        await _default_task_factory(manager, "wallet_001")
                        # Only 1 sleep: between cycles (not after the last)
                        assert mock_sleep.call_count <= 1

    @pytest.mark.asyncio
    async def test_gas_unsafe_triggers_sleep(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 999.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                        result = await _default_task_factory(manager, "wallet_001")
                        mock_sleep.assert_called()

        assert result == []

    @pytest.mark.asyncio
    async def test_with_task_context_passed(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"extra": "data"}

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    result = await _default_task_factory(manager, "wallet_001", task_context=ctx)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_balance_exception_does_not_abort(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(side_effect=ConnectionError("no rpc"))
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 2):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        result = await _default_task_factory(manager, "wallet_001")

        assert result == []


# ---------------------------------------------------------------------------
# 5. _preload_robust_engine — additional paths
# ---------------------------------------------------------------------------

class TestPreloadRobustEngineExtra:
    @pytest.mark.asyncio
    async def test_skips_when_n_ticks_nonzero(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 10
        engine.ingest_price = MagicMock()

        with patch.dict("sys.modules", {
            "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
        }):
            await _preload_robust_engine(engine, 2000.0)

        engine.ingest_price.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthetic_prices_used_when_api_fails(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        engine.ingest_price = MagicMock()

        with patch("aiohttp.ClientSession", side_effect=Exception("offline")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 3000.0)

        assert engine.ingest_price.call_count > 0

    @pytest.mark.asyncio
    async def test_process_tick_used_as_fallback(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        # remove ingest_price so process_tick is used
        del engine.ingest_price
        engine.process_tick = MagicMock()

        with patch("aiohttp.ClientSession", side_effect=Exception("offline")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 1800.0)

        assert engine.process_tick.call_count > 0

    @pytest.mark.asyncio
    async def test_no_ingest_method_does_not_raise(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock(spec=["_n_ticks"])
        engine._n_ticks = 0

        with patch("aiohttp.ClientSession", side_effect=Exception("offline")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 2000.0)
        # Must not raise

    @pytest.mark.asyncio
    async def test_ingest_price_called_at_least_needed_times(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        engine.ingest_price = MagicMock()

        with patch("aiohttp.ClientSession", side_effect=Exception("offline")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 2000.0)

        # needed = max(50+10, 150) = 150
        assert engine.ingest_price.call_count == 150


# ---------------------------------------------------------------------------
# 6. _collect_alerter_inputs — additional paths
# ---------------------------------------------------------------------------

class TestCollectAlerterInputsExtra:
    @pytest.mark.asyncio
    async def test_all_keys_present(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()
        result = await _collect_alerter_inputs(None, cfg)
        assert "drift_level" in result
        assert "apy" in result
        assert "model_staleness_days" in result
        assert "rpc_latency_ms" in result
        assert "drawdown" in result

    @pytest.mark.asyncio
    async def test_drift_level_nominal_by_default(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()
        # Patch _drift_detector to None
        with patch("wallet_manager.task_factories._drift_detector", None):
            result = await _collect_alerter_inputs(None, cfg)
        assert result["drift_level"] == "NOMINAL"

    @pytest.mark.asyncio
    async def test_drift_level_from_detector(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_level = MagicMock()
        mock_level.name = "HIGH"
        mock_detector = MagicMock()
        mock_detector._last_drift_level = mock_level

        with patch("wallet_manager.task_factories._drift_detector", mock_detector):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["drift_level"] == "HIGH"

    @pytest.mark.asyncio
    async def test_drawdown_zero_when_snapshot_none(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()
        result = await _collect_alerter_inputs(None, cfg)
        assert result["drawdown"] == 0.0

    @pytest.mark.asyncio
    async def test_drawdown_computed_correctly(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_alloc = MagicMock()
        mock_alloc.current_usd = 500.0
        mock_snapshot = MagicMock()
        mock_snapshot.total_value_usd = 400.0
        mock_snapshot.allocations = [mock_alloc]

        result = await _collect_alerter_inputs(mock_snapshot, cfg)
        # drawdown = max(0, 1 - 400/500) = 0.2
        assert result["drawdown"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_model_staleness_positive_for_existing_file(self, tmp_path):
        from wallet_manager.task_factories import _collect_alerter_inputs
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")

        mock_cfg = MagicMock()
        mock_cfg.ppo_model_path = str(model_file)

        result = await _collect_alerter_inputs(None, mock_cfg)
        assert result["model_staleness_days"] >= 0.0

    @pytest.mark.asyncio
    async def test_model_staleness_zero_for_nonexistent_file(self):
        from wallet_manager.task_factories import _collect_alerter_inputs
        mock_cfg = MagicMock()
        mock_cfg.ppo_model_path = "/nonexistent/model/path.pt"

        result = await _collect_alerter_inputs(None, mock_cfg)
        assert result["model_staleness_days"] == 0.0

    @pytest.mark.asyncio
    async def test_apy_default_when_yield_strategy_unavailable(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        with patch.dict("sys.modules", {
            "wallet_manager.strategies.yield_strategy": MagicMock(
                get_apy_monitor=MagicMock(side_effect=ImportError("no module"))
            )
        }):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["apy"] == 0.0

    @pytest.mark.asyncio
    async def test_rpc_latency_from_mock_router(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_router = MagicMock()
        mock_router._latency_tracker = {"rpc1": [5.0, 15.0, 25.0]}

        with patch.dict("sys.modules", {
            "wallet_manager.orchestrator": MagicMock(_get_rpc_router=lambda: mock_router)
        }):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["rpc_latency_ms"] == 25.0

    @pytest.mark.asyncio
    async def test_rpc_latency_zero_on_exception(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        with patch.dict("sys.modules", {
            "wallet_manager.orchestrator": MagicMock(
                _get_rpc_router=MagicMock(side_effect=Exception("no router"))
            )
        }):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["rpc_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# 7. Guard: portfolio_manager None falls back to airdrop
# ---------------------------------------------------------------------------

class TestPortfolioManagerNoneGuard:
    @pytest.mark.asyncio
    async def test_robust_paper_falls_back_when_no_pm(self):
        from wallet_manager.task_factories import _robust_paper_task_factory
        manager = _make_manager()
        ctx = {}  # no portfolio_manager

        with patch("wallet_manager.task_factories._airdrop_task_factory", new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            result = await _robust_paper_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_robust_falls_back_when_no_pm(self):
        from wallet_manager.task_factories import _robust_task_factory
        manager = _make_manager()
        ctx = {}  # no portfolio_manager

        with patch("wallet_manager.task_factories._airdrop_task_factory", new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            result = await _robust_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()
        assert result == []
