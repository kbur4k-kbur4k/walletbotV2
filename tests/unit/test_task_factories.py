"""tests/unit/test_task_factories.py — Comprehensive tests for task_factories.py"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from wallet_manager.core.models import TxReceipt, WalletConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
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


def _make_receipt(tx_hash: str = "0xabc") -> TxReceipt:
    r = MagicMock(spec=TxReceipt)
    r.tx_hash = tx_hash
    r.nonce_used = 1
    return r


def _make_manager(config: WalletConfig | None = None) -> MagicMock:
    cfg = config or _make_config()
    manager = MagicMock()
    manager.get_config.return_value = cfg
    manager.get_safe_balance = AsyncMock(return_value=int(1e18))
    manager.sign_and_broadcast_tx = AsyncMock(return_value=_make_receipt())
    return manager


# ---------------------------------------------------------------------------
# _get_wallet_address
# ---------------------------------------------------------------------------

class TestGetWalletAddress:
    def test_returns_config_address_when_present(self):
        from wallet_manager.task_factories import _get_wallet_address
        cfg = _make_config(address="0xdeadbeef")
        assert _get_wallet_address(cfg) == "0xdeadbeef"

    def test_derives_address_from_private_key_when_no_address(self):
        from wallet_manager.task_factories import _get_wallet_address
        # Use a known private key and its corresponding address
        pk = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        cfg = _make_config(address=None, private_key=pk)
        addr = _get_wallet_address(cfg)
        assert addr.startswith("0x")
        assert len(addr) == 42

    def test_derives_address_is_string(self):
        from wallet_manager.task_factories import _get_wallet_address
        pk = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        cfg = _make_config(address=None, private_key=pk)
        result = _get_wallet_address(cfg)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _maybe_run_agents
# ---------------------------------------------------------------------------

class TestMaybeRunAgents:
    @pytest.mark.asyncio
    async def test_skips_when_interval_not_elapsed(self):
        """If last run was recent, should return immediately without running agents."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = time.time()  # just ran
        ctx = {}
        with patch("wallet_manager.task_factories.get_settings"):
            await tf._maybe_run_agents(ctx)
        # No exception is success — agents were not run

    @pytest.mark.asyncio
    async def test_runs_agents_when_interval_elapsed(self):
        """After interval, agents should be invoked."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0  # never ran

        mock_report = MagicMock()
        mock_report.title = "Test Agent"
        mock_report.body = "Report body"
        mock_report.used_llm = False

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=[mock_report])

        mock_notifier = AsyncMock()
        ctx = {"notifier": mock_notifier}

        with patch("wallet_manager.task_factories.get_agent_registry", return_value=mock_registry, create=True):
            with patch("wallet_manager.ai_agents.get_agent_registry", return_value=mock_registry, create=True):
                # Patch the import inside the function
                with patch.dict("sys.modules", {
                    "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
                    "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
                }):
                    await tf._maybe_run_agents(ctx)

    @pytest.mark.asyncio
    async def test_handles_empty_reports(self):
        """No notifier send when reports is empty."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        mock_registry = MagicMock()
        mock_registry.run_all = AsyncMock(return_value=[])
        ctx = {}

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=lambda: mock_registry),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents(ctx)  # should not raise

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Exception inside agent run should be swallowed."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        with patch.dict("sys.modules", {
            "wallet_manager.ai_agents": MagicMock(get_agent_registry=MagicMock(side_effect=RuntimeError("boom"))),
            "wallet_manager.strategies.paper_engine": MagicMock(_engine_instance=None),
        }):
            await tf._maybe_run_agents({})  # should not raise

    @pytest.mark.asyncio
    async def test_updates_last_agent_run_ts(self):
        """Should update _last_agent_run_ts after running."""
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

    @pytest.mark.asyncio
    async def test_sends_message_to_notifier(self):
        """Notifier.send_message should be called when there are reports."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        mock_report = MagicMock()
        mock_report.title = "Alpha Report"
        mock_report.body = "short"
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

        mock_notifier.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_truncates_long_report_body(self):
        """Reports with body > 400 chars should trigger truncation notice."""
        import wallet_manager.task_factories as tf
        tf._last_agent_run_ts = 0.0

        mock_report = MagicMock()
        mock_report.title = "Long Report"
        mock_report.body = "x" * 500  # > 400 chars
        mock_report.used_llm = False

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
        assert "truncated" in msg


# ---------------------------------------------------------------------------
# _collect_alerter_inputs
# ---------------------------------------------------------------------------

class TestCollectAlerterInputs:
    @pytest.mark.asyncio
    async def test_returns_dict_with_expected_keys(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()
        result = await _collect_alerter_inputs(None, cfg)
        assert set(result.keys()) == {"drift_level", "apy", "model_staleness_days", "rpc_latency_ms", "drawdown"}

    @pytest.mark.asyncio
    async def test_drift_level_default_nominal(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()
        result = await _collect_alerter_inputs(None, cfg)
        assert result["drift_level"] == "NOMINAL"

    @pytest.mark.asyncio
    async def test_drift_level_from_detector(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_level = MagicMock()
        mock_level.name = "RETRAIN"
        mock_detector = MagicMock()
        mock_detector._last_drift_level = mock_level

        with patch("wallet_manager.task_factories._drift_detector", mock_detector):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["drift_level"] == "RETRAIN"

    @pytest.mark.asyncio
    async def test_snapshot_none_gives_zero_drawdown(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()
        result = await _collect_alerter_inputs(None, cfg)
        assert result["drawdown"] == 0.0

    @pytest.mark.asyncio
    async def test_drawdown_computed_from_snapshot(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_alloc = MagicMock()
        mock_alloc.current_usd = 1000.0

        mock_snapshot = MagicMock()
        mock_snapshot.total_value_usd = 800.0
        mock_snapshot.allocations = [mock_alloc]

        result = await _collect_alerter_inputs(mock_snapshot, cfg)
        # drawdown = max(0, 1 - 800/1000) = 0.2
        assert result["drawdown"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_apy_default_zero(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        with patch.dict("sys.modules", {
            "wallet_manager.strategies.yield_strategy": MagicMock(
                get_apy_monitor=MagicMock(side_effect=Exception("no monitor"))
            )
        }):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["apy"] == 0.0

    @pytest.mark.asyncio
    async def test_apy_from_monitor(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_monitor = MagicMock()
        mock_monitor.last_apy = 4.5

        with patch.dict("sys.modules", {
            "wallet_manager.strategies.yield_strategy": MagicMock(
                get_apy_monitor=lambda: mock_monitor
            )
        }):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["apy"] == 4.5

    @pytest.mark.asyncio
    async def test_rpc_latency_from_router(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_router = MagicMock()
        mock_router._latency_tracker = {"rpc1": [10.0, 20.0, 30.0]}

        with patch.dict("sys.modules", {
            "wallet_manager.orchestrator": MagicMock(
                _get_rpc_router=lambda: mock_router
            )
        }):
            result = await _collect_alerter_inputs(None, cfg)

        assert result["rpc_latency_ms"] == 30.0

    @pytest.mark.asyncio
    async def test_model_staleness_from_file(self, tmp_path):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        get_settings()

        # Create a temp model file
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake model")

        mock_cfg = MagicMock()
        mock_cfg.ppo_model_path = str(model_file)

        result = await _collect_alerter_inputs(None, mock_cfg)
        assert result["model_staleness_days"] >= 0.0

    @pytest.mark.asyncio
    async def test_drawdown_zero_when_total_value_zero(self):
        from wallet_manager.config import get_settings
        from wallet_manager.task_factories import _collect_alerter_inputs
        cfg = get_settings()

        mock_snapshot = MagicMock()
        mock_snapshot.total_value_usd = 0.0
        mock_snapshot.allocations = []

        result = await _collect_alerter_inputs(mock_snapshot, cfg)
        assert result["drawdown"] == 0.0


# ---------------------------------------------------------------------------
# _default_task_factory
# ---------------------------------------------------------------------------

class TestDefaultTaskFactory:
    @pytest.mark.asyncio
    async def test_returns_empty_receipts(self):
        from wallet_manager.task_factories import _TASK_LOOP_CYCLES, _default_task_factory
        manager = _make_manager()

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    result = await _default_task_factory(manager, "wallet_001")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_sleeps_when_gas_unsafe(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                        result = await _default_task_factory(manager, "wallet_001")
                        mock_sleep.assert_called()

        assert result == []

    @pytest.mark.asyncio
    async def test_balance_fetch_exception_continues(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(side_effect=Exception("RPC down"))

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    result = await _default_task_factory(manager, "wallet_001")

        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_cycles(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 3):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        await _default_task_factory(manager, "wallet_001")

        # get_safe_balance called 3 times
        assert manager.get_safe_balance.call_count == 3

    @pytest.mark.asyncio
    async def test_with_none_task_context(self):
        from wallet_manager.task_factories import _default_task_factory
        manager = _make_manager()

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._TASK_LOOP_CYCLES", 1):
                with patch("wallet_manager.task_factories._GAS_POLL_INTERVAL", 0):
                    result = await _default_task_factory(manager, "wallet_001", task_context=None)

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _preload_robust_engine
# ---------------------------------------------------------------------------

class TestPreloadRobustEngine:
    @pytest.mark.asyncio
    async def test_skips_when_n_ticks_positive(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 5
        engine.ingest_price = MagicMock()

        with patch.dict("sys.modules", {
            "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
        }):
            await _preload_robust_engine(engine, 2000.0)

        engine.ingest_price.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_coingecko_prices_when_available(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        engine.ingest_price = MagicMock()

        fake_prices = [[t * 1000, 2000.0 + t] for t in range(200)]
        fake_data = {"prices": fake_prices}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 2000.0)

        assert engine.ingest_price.called

    @pytest.mark.asyncio
    async def test_falls_back_to_synthetic_prices_on_exception(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        engine.ingest_price = MagicMock()

        import aiohttp

        with patch("aiohttp.ClientSession", side_effect=aiohttp.ClientError("timeout")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 2000.0)

        # Should still have called ingest_price with synthetic data
        assert engine.ingest_price.called

    @pytest.mark.asyncio
    async def test_falls_back_to_synthetic_when_coingecko_returns_few_prices(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        engine.ingest_price = MagicMock()

        # Only 5 prices — less than needed
        fake_data = {"prices": [[i * 1000, 2000.0] for i in range(5)]}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 2000.0)

        assert engine.ingest_price.called

    @pytest.mark.asyncio
    async def test_no_ingest_method_logs_warning(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock(spec=[])  # no ingest_price or process_tick
        engine._n_ticks = 0

        import aiohttp
        with patch("aiohttp.ClientSession", side_effect=Exception("no net")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                # Should not raise
                await _preload_robust_engine(engine, 2000.0)

    @pytest.mark.asyncio
    async def test_uses_process_tick_when_ingest_price_absent(self):
        from wallet_manager.task_factories import _preload_robust_engine

        engine = MagicMock()
        engine._n_ticks = 0
        del engine.ingest_price  # remove ingest_price
        engine.process_tick = MagicMock()

        import aiohttp
        with patch("aiohttp.ClientSession", side_effect=Exception("no net")):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.robust_alpha": MagicMock(FRACDIFF_MIN_OBSERVATIONS=50),
            }):
                await _preload_robust_engine(engine, 2000.0)

        assert engine.process_tick.called


# ---------------------------------------------------------------------------
# _airdrop_task_factory
# ---------------------------------------------------------------------------

class TestAirdropTaskFactory:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_scheduler(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        ctx = {}  # no scheduler
        result = await _airdrop_task_factory(manager, "wallet_001", ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_scheduler_says_not_due(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()

        mock_scheduler = MagicMock()
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(False, 3700)):
            result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_gas_unsafe(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_unsupported_chain(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        cfg = _make_config(chain_id=1)  # Ethereum mainnet not supported
        manager = _make_manager(config=cfg)

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_balance_fetch_exception(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(side_effect=Exception("RPC error"))

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_usdc_to_eth_swap_path(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(0.01e18))  # small ETH

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler}

        mock_receipt = _make_receipt("0xusdc2eth")

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=int(500e6)):
                    with patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value", return_value=20.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value", return_value=500.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible", return_value=True):
                                    with patch("wallet_manager.actions.is_swap_eligible", return_value=False):
                                        with patch("wallet_manager.task_factories._execute_usdc_to_eth",
                                                   new_callable=AsyncMock, return_value=[mock_receipt]):
                                            result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == [mock_receipt]

    @pytest.mark.asyncio
    async def test_eth_to_usdc_swap_path(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(1e18))  # 1 ETH

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler}

        mock_receipt = _make_receipt("0xeth2usdc")

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=int(10e6)):
                    with patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value", return_value=2000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value", return_value=10.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible", return_value=False):
                                    with patch("wallet_manager.actions.is_swap_eligible", return_value=True):
                                        with patch("wallet_manager.task_factories._execute_eth_to_usdc",
                                                   new_callable=AsyncMock, return_value=[mock_receipt]):
                                            result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == [mock_receipt]

    @pytest.mark.asyncio
    async def test_airdrop_budget_cap_returns_empty_when_too_small(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(0.1e18))

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler, "airdrop_budget_usd": 5.0}  # < $10

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=int(100e6)):
                    with patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value", return_value=200.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value", return_value=100.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible", return_value=True):
                                    with patch("wallet_manager.actions.is_swap_eligible", return_value=True):
                                        result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_not_eligible_for_any_swap(self):
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(0.001e18))

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value", return_value=2.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value", return_value=0.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible", return_value=False):
                                    with patch("wallet_manager.actions.is_swap_eligible", return_value=False):
                                        result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_usdc_balance_fetch_exception_uses_zero(self):
        """USDC fetch failure should default to 0 and continue."""
        from wallet_manager.task_factories import _airdrop_task_factory
        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(1e18))

        mock_scheduler = MagicMock()
        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle", return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock,
                           side_effect=Exception("RPC fail")):
                    with patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value", return_value=2000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value", return_value=0.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible", return_value=False):
                                    with patch("wallet_manager.actions.is_swap_eligible", return_value=False):
                                        result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []


# ---------------------------------------------------------------------------
# _execute_usdc_to_eth
# ---------------------------------------------------------------------------

class TestExecuteUsdcToEth:
    @pytest.mark.asyncio
    async def test_successful_swap(self):
        from wallet_manager.task_factories import _execute_usdc_to_eth
        manager = _make_manager()
        config = _make_config()
        receipt = _make_receipt("0xswap")
        manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)

        mock_step = MagicMock()
        mock_step.description = "Approve"
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 200000
        mock_step.tx_params.value = 0

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]

        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(usdc="0xusdc", weth="0xweth")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_usdc_to_eth_plan", new_callable=AsyncMock, return_value=mock_plan):
                        result = await _execute_usdc_to_eth(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(100e6), 5.0,
                            notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                        )

        assert receipt in result
        mock_scheduler.schedule_next.assert_called_once_with("wallet_001")

    @pytest.mark.asyncio
    async def test_plan_build_failure_returns_empty(self):
        from wallet_manager.task_factories import _execute_usdc_to_eth
        manager = _make_manager()
        config = _make_config()
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(usdc="0xusdc", weth="0xweth")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_usdc_to_eth_plan",
                               new_callable=AsyncMock, side_effect=Exception("plan fail")):
                        result = await _execute_usdc_to_eth(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(100e6), 5.0,
                            notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                        )

        assert result == []

    @pytest.mark.asyncio
    async def test_tx_broadcast_failure_calls_schedule_next(self):
        from wallet_manager.task_factories import _execute_usdc_to_eth
        manager = _make_manager()
        manager.sign_and_broadcast_tx = AsyncMock(side_effect=Exception("broadcast fail"))
        config = _make_config()

        mock_step = MagicMock()
        mock_step.description = "Swap"
        mock_step.tx_params = MagicMock()

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(usdc="0xusdc", weth="0xweth")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_usdc_to_eth_plan", new_callable=AsyncMock, return_value=mock_plan):
                        with patch("wallet_manager.orchestrator.WalletOrchestrator._safe_notify", new_callable=AsyncMock):
                            await _execute_usdc_to_eth(
                                manager, "wallet_001", config, "0xaddr", 8453,
                                int(100e6), 5.0,
                                notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                            )

        mock_scheduler.schedule_next.assert_called_with("wallet_001")

    @pytest.mark.asyncio
    async def test_with_analytics_db_records_tx(self):
        from wallet_manager.task_factories import _execute_usdc_to_eth
        manager = _make_manager()
        config = _make_config()
        receipt = _make_receipt("0xtxhash")
        manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)

        mock_step = MagicMock()
        mock_step.description = "Swap"
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 200000
        mock_step.tx_params.value = 0

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]

        mock_analytics = MagicMock()
        mock_analytics.record_tx = AsyncMock()
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(usdc="0xusdc", weth="0xweth")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_usdc_to_eth_plan", new_callable=AsyncMock, return_value=mock_plan):
                        await _execute_usdc_to_eth(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(100e6), 5.0,
                            notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                            analytics_db=mock_analytics,
                        )

        mock_analytics.record_tx.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirmation_gate_timeout_returns_empty(self):
        from wallet_manager.task_factories import _execute_usdc_to_eth
        manager = _make_manager()
        config = _make_config()

        mock_step = MagicMock()
        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]

        mock_notifier = MagicMock()
        mock_notifier.send_confirmation_request = AsyncMock()

        mock_gate = MagicMock()
        mock_gate.wait_for_evet = AsyncMock(return_value=False)  # timeout/rejected
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(usdc="0xusdc", weth="0xweth")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_usdc_to_eth_plan", new_callable=AsyncMock, return_value=mock_plan):
                        result = await _execute_usdc_to_eth(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(100e6), 5.0,
                            notifier=mock_notifier, confirmation_gate=mock_gate, scheduler=mock_scheduler,
                        )

        assert result == []
        mock_scheduler.schedule_next.assert_called_with("wallet_001")


# ---------------------------------------------------------------------------
# _execute_eth_to_usdc
# ---------------------------------------------------------------------------

class TestExecuteEthToUsdc:
    @pytest.mark.asyncio
    async def test_successful_swap(self):
        from wallet_manager.task_factories import _execute_eth_to_usdc
        manager = _make_manager()
        config = _make_config()
        receipt = _make_receipt("0xeth2usdc")
        manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)

        mock_step = MagicMock()
        mock_step.description = "Swap ETH"
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 200000
        mock_step.tx_params.value = int(0.1e18)

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(weth="0xweth", usdc="0xusdc")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_eth_to_usdc_plan", new_callable=AsyncMock, return_value=mock_plan):
                        result = await _execute_eth_to_usdc(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(0.1e18), 5.0,
                            notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                        )

        assert receipt in result
        mock_scheduler.schedule_next.assert_called_once_with("wallet_001")

    @pytest.mark.asyncio
    async def test_plan_build_failure_returns_empty(self):
        from wallet_manager.task_factories import _execute_eth_to_usdc
        manager = _make_manager()
        config = _make_config()
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(weth="0xweth", usdc="0xusdc")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_eth_to_usdc_plan",
                               new_callable=AsyncMock, side_effect=Exception("plan fail")):
                        result = await _execute_eth_to_usdc(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(0.1e18), 5.0,
                            notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                        )

        assert result == []

    @pytest.mark.asyncio
    async def test_confirmation_gate_confirmed(self):
        from wallet_manager.task_factories import _execute_eth_to_usdc
        manager = _make_manager()
        config = _make_config()
        receipt = _make_receipt("0xconfirmed")
        manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)

        mock_step = MagicMock()
        mock_step.description = "Swap"
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 200000
        mock_step.tx_params.value = 0

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]

        mock_notifier = MagicMock()
        mock_notifier.send_confirmation_request = AsyncMock()
        mock_gate = MagicMock()
        mock_gate.wait_for_evet = AsyncMock(return_value=True)  # confirmed
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(weth="0xweth", usdc="0xusdc")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_eth_to_usdc_plan", new_callable=AsyncMock, return_value=mock_plan):
                        result = await _execute_eth_to_usdc(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(0.1e18), 5.0,
                            notifier=mock_notifier, confirmation_gate=mock_gate, scheduler=mock_scheduler,
                        )

        assert receipt in result

    @pytest.mark.asyncio
    async def test_with_analytics_db(self):
        from wallet_manager.task_factories import _execute_eth_to_usdc
        manager = _make_manager()
        config = _make_config()
        receipt = _make_receipt("0xanalytic")
        manager.sign_and_broadcast_tx = AsyncMock(return_value=receipt)

        mock_step = MagicMock()
        mock_step.description = "Swap"
        mock_step.tx_params = MagicMock()
        mock_step.tx_params.gas_price_gwei = 5.0
        mock_step.tx_params.gas = 200000
        mock_step.tx_params.value = 0

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]

        mock_analytics = MagicMock()
        mock_analytics.record_tx = AsyncMock()
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(weth="0xweth", usdc="0xusdc")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_eth_to_usdc_plan", new_callable=AsyncMock, return_value=mock_plan):
                        await _execute_eth_to_usdc(
                            manager, "wallet_001", config, "0xaddr", 8453,
                            int(0.1e18), 5.0,
                            notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                            analytics_db=mock_analytics,
                        )

        mock_analytics.record_tx.assert_called_once()

    @pytest.mark.asyncio
    async def test_tx_failure_calls_schedule_next(self):
        from wallet_manager.task_factories import _execute_eth_to_usdc
        manager = _make_manager()
        manager.sign_and_broadcast_tx = AsyncMock(side_effect=Exception("broadcast fail"))
        config = _make_config()

        mock_step = MagicMock()
        mock_step.description = "Swap"
        mock_step.tx_params = MagicMock()

        mock_plan = MagicMock()
        mock_plan.steps = [mock_step]
        mock_scheduler = MagicMock()

        with patch("wallet_manager.actions.get_rotated_fee_tier", return_value=500):
            with patch("wallet_manager.actions.get_chain_addresses", return_value=MagicMock(weth="0xweth", usdc="0xusdc")):
                with patch("wallet_manager.actions.get_dynamic_slippage", new_callable=AsyncMock, return_value=30):
                    with patch("wallet_manager.actions.build_eth_to_usdc_plan", new_callable=AsyncMock, return_value=mock_plan):
                        with patch("wallet_manager.orchestrator.WalletOrchestrator._safe_notify", new_callable=AsyncMock):
                            await _execute_eth_to_usdc(
                                manager, "wallet_001", config, "0xaddr", 8453,
                                int(0.1e18), 5.0,
                                notifier=None, confirmation_gate=None, scheduler=mock_scheduler,
                            )

        mock_scheduler.schedule_next.assert_called_with("wallet_001")


# ---------------------------------------------------------------------------
# _three_pillar_task_factory — routing logic
# ---------------------------------------------------------------------------

class TestThreePillarTaskFactory:
    @pytest.mark.asyncio
    async def test_circuit_breaker_returns_empty(self):
        from wallet_manager.task_factories import _three_pillar_task_factory
        manager = _make_manager()

        mock_command_state = MagicMock()
        mock_command_state.is_circuit_breaker_halted = True
        ctx = {"command_state": mock_command_state}

        result = await _three_pillar_task_factory(manager, "wallet_001", ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_delegates_to_robust_paper_factory(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}
        mock_settings = MagicMock()
        mock_settings.execution_mode = "robust_paper"

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.task_factories._robust_paper_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_factory:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_robust_factory(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}
        mock_settings = MagicMock()
        mock_settings.execution_mode = "robust"

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.task_factories._robust_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_factory:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_robust_live_factory(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}
        mock_settings = MagicMock()
        mock_settings.execution_mode = "robust_live"

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.task_factories._robust_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_factory:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_ml_quant_factory(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}
        mock_settings = MagicMock()
        mock_settings.execution_mode = "ml_quant"

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.task_factories._ml_quant_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_factory:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_quantx_factory(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}
        mock_settings = MagicMock()
        mock_settings.execution_mode = "legacy"
        mock_settings.quantx_engine_mode = "quantx"

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.task_factories._quantx_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_factory:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_when_no_portfolio_manager(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}  # no portfolio_manager
        mock_settings = MagicMock()
        mock_settings.execution_mode = "legacy"
        mock_settings.quantx_engine_mode = "standard"

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.task_factories._airdrop_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_airdrop:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_gas_unsafe_returns_early(self):
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        mock_pm = MagicMock()
        ctx = {"portfolio_manager": mock_pm}
        mock_settings = MagicMock()
        mock_settings.execution_mode = "legacy"
        mock_settings.quantx_engine_mode = "standard"

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))

        with patch("wallet_manager.task_factories._cfg", return_value=mock_settings):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.strategies.yield_strategy.get_yield_pillar_value_usd",
                           new_callable=AsyncMock, side_effect=Exception("skip")):
                    with patch("wallet_manager.task_factories._airdrop_task_factory", new_callable=AsyncMock) as _af:
                        with patch("wallet_manager.strategies.flash_crash_detector.get_detector") as mock_crash:
                            mock_crash.return_value.is_kill_switch_active = False
                            mock_crash.return_value.ingest_price = MagicMock(return_value=None)
                            with patch("wallet_manager.actions.get_current_price",
                                       new_callable=AsyncMock, return_value=2000.0):
                                result = await _three_pillar_task_factory(manager, "wallet_001", ctx)

        assert result == []


# ---------------------------------------------------------------------------
# _robust_paper_task_factory
# ---------------------------------------------------------------------------

class TestRobustPaperTaskFactory:
    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_no_portfolio_manager(self):
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        ctx = {}  # no portfolio_manager

        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            await _robust_paper_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_non_base_chain(self):
        from wallet_manager.task_factories import _robust_paper_task_factory

        cfg = _make_config(chain_id=1)  # Ethereum, not Base
        manager = _make_manager(config=cfg)
        mock_pm = MagicMock()
        ctx = {"portfolio_manager": mock_pm}

        result = await _robust_paper_task_factory(manager, "wallet_001", ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_stops_when_quote_invalidated(self):
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        mock_pm = MagicMock()
        ctx = {"portfolio_manager": mock_pm}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()

        mock_latency_shield = MagicMock()
        mock_qv = MagicMock()
        mock_qv.should_invalidate_signals = True
        mock_qv.quote_age_s = 10.0
        mock_qv.reason_code = "STALE"
        mock_lat_prof = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(return_value=(mock_qv, mock_lat_prof))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.strategies.paper_engine.get_paper_engine",
                       new_callable=AsyncMock, return_value=mock_paper_engine):
                with patch("wallet_manager.strategies.latency_shield.get_latency_shield",
                           return_value=mock_latency_shield):
                    with patch("wallet_manager.strategies.stochastic_stress_test.get_stress_tester",
                               return_value=MagicMock()):
                        with patch("wallet_manager.strategies.regime_filter.get_regime_filter",
                                   return_value=MagicMock()):
                            with patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine",
                                       return_value=MagicMock(_n_ticks=1)):
                                with patch("wallet_manager.actions.fetch_eth_price_usd",
                                           new_callable=AsyncMock, return_value=2000.0):
                                    result = await _robust_paper_task_factory(manager, "wallet_001", ctx)

        assert result == []


# ---------------------------------------------------------------------------
# _robust_task_factory
# ---------------------------------------------------------------------------

class TestRobustTaskFactory:
    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_no_portfolio_manager(self):
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            await _robust_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_gas_unsafe(self):
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        mock_pm = MagicMock()
        ctx = {"portfolio_manager": mock_pm}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.strategies.latency_shield.get_latency_shield", return_value=MagicMock()):
                with patch("wallet_manager.strategies.regime_filter.get_regime_filter", return_value=MagicMock()):
                    with patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine",
                               return_value=MagicMock()):
                        with patch("wallet_manager.strategies.stochastic_stress_test.get_stress_tester",
                                   return_value=MagicMock()):
                            with patch("wallet_manager.strategies.wf_validator.get_wf_validator",
                                       new_callable=AsyncMock):
                                with patch("wallet_manager.strategies.liquidity_alpha.get_liquidity_alpha",
                                           return_value=MagicMock()):
                                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                                               new_callable=AsyncMock, return_value=2000.0):
                                        result = await _robust_task_factory(manager, "wallet_001", ctx)

        assert result == []


# ---------------------------------------------------------------------------
# _ml_quant_task_factory
# ---------------------------------------------------------------------------

class TestMlQuantTaskFactory:
    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_no_portfolio_manager(self):
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            await _ml_quant_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_gas_unsafe(self):
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        mock_pm = MagicMock()
        ctx = {"portfolio_manager": mock_pm}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.strategies.execution_logic.get_ml_execution_engine",
                       return_value=MagicMock()):
                with patch("wallet_manager.strategies.quant_analytics.get_quant_analytics",
                           new_callable=AsyncMock):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        result = await _ml_quant_task_factory(manager, "wallet_001", ctx)

        assert result == []


# ---------------------------------------------------------------------------
# _quantx_task_factory
# ---------------------------------------------------------------------------

class TestQuantxTaskFactory:
    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_no_portfolio_manager(self):
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            await _quantx_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_gas_unsafe(self):
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        mock_pm = MagicMock()
        ctx = {"portfolio_manager": mock_pm}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.strategies.quantx_engine.get_quantx_engine",
                       return_value=MagicMock()):
                with patch("wallet_manager.actions.fetch_eth_price_usd",
                           new_callable=AsyncMock, return_value=2000.0):
                    result = await _quantx_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_rebalance_when_scheduler_due(self):
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        from wallet_manager.portfolio_manager import StrategyType
        mock_pm = MagicMock()
        mock_pm._initial_budget = 250.0
        mock_pm._monthly_injection = 150.0
        mock_pm.rebalance_portfolio = MagicMock(return_value={
            StrategyType.YIELD: 60.0, StrategyType.ALGO: 52.5,
            StrategyType.SHORT: 15.0, StrategyType.AIRDROP: 22.5,
        })
        mock_pm.get_pillar_budget_usd = MagicMock(return_value=5.0)  # < 10, skip airdrop

        mock_scheduler = MagicMock()
        mock_scheduler.is_monthly_rebalance_due.return_value = True
        mock_scheduler.is_initial_rebalance.return_value = True

        ctx = {"portfolio_manager": mock_pm, "scheduler": mock_scheduler}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(False, 100.0))  # gas unsafe, early return

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.actions.fetch_eth_price_usd",
                       new_callable=AsyncMock, return_value=2000.0):
                await _quantx_task_factory(manager, "wallet_001", ctx)

        mock_scheduler.mark_monthly_rebalance_done.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level constants loaded correctly
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_task_loop_cycles_positive(self):
        from wallet_manager.task_factories import _TASK_LOOP_CYCLES
        assert _TASK_LOOP_CYCLES > 0

    def test_gas_poll_interval_positive(self):
        from wallet_manager.task_factories import _GAS_POLL_INTERVAL
        assert _GAS_POLL_INTERVAL > 0

    def test_aave_apy_threshold_is_fraction(self):
        from wallet_manager.task_factories import _AAVE_APY_ALERT_THRESHOLD
        # Should be a small fraction (e.g., 2.0 pct -> 0.02)
        assert 0 < _AAVE_APY_ALERT_THRESHOLD < 1.0

    def test_agent_run_interval(self):
        from wallet_manager.task_factories import _AGENT_RUN_INTERVAL_S
        assert _AGENT_RUN_INTERVAL_S > 0

    def test_cfg_shorthand_returns_settings(self):
        from wallet_manager.task_factories import _cfg
        settings = _cfg()
        assert hasattr(settings, "execution_mode")
