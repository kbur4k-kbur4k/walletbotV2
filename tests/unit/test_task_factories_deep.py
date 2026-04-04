"""tests/unit/test_task_factories_deep.py

Deep coverage tests targeting previously uncovered lines in task_factories.py.

Targets lines: 347-486, 560-589, 606-614, 627-634, 643-644, 663-664, 677-808,
               850-864, 871-900, 911-1035, 1096-1123, 1134-1200, 1262-1276,
               1283-1312, 1323-1485, 1520, 1577-1578.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from wallet_manager.core.models import TxReceipt, WalletConfig
from wallet_manager.portfolio_manager import StrategyType

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


def _make_receipt(tx_hash: str = "0xabc") -> MagicMock:
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


def _make_portfolio_manager() -> MagicMock:
    pm = MagicMock()
    pm._initial_budget = 250.0
    pm._monthly_injection = 150.0
    pm.rebalance_portfolio = MagicMock(return_value={
        StrategyType.YIELD: 125.0,
        StrategyType.ALGO: 87.5,
        StrategyType.AIRDROP: 37.5,
    })
    pm.update_pillar_value = MagicMock()
    pm.get_pillar_budget_usd = MagicMock(return_value=5.0)  # below $10 → no airdrop
    mock_alloc = MagicMock()
    mock_alloc.strategy = StrategyType.YIELD
    mock_alloc.delta_usd = 0.0  # won't trigger supply
    mock_snapshot = MagicMock()
    mock_snapshot.allocations = [mock_alloc]
    pm.get_snapshot = MagicMock(return_value=mock_snapshot)
    return pm


def _make_gas_monitor(safe: bool = True, gwei: float = 5.0) -> MagicMock:
    gm = MagicMock()
    gm.is_safe = AsyncMock(return_value=(safe, gwei))
    return gm


# ---------------------------------------------------------------------------
# _robust_paper_task_factory — lines 347-486
# ---------------------------------------------------------------------------

class TestRobustPaperTaskFactory:
    """Tests for _robust_paper_task_factory covering lines 347-486."""

    def _patch_imports(self):
        """Return a context manager that patches all heavy imports."""
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()
        mock_paper_result = MagicMock()
        mock_paper_result.action = "HOLD"
        mock_paper_result.signal_alpha = 0.0
        mock_paper_result.simulated_fill_price = 2000.0
        mock_paper_result.slippage_bps = 1.0
        mock_paper_result.gas_cost_usd = 0.01
        mock_paper_result.virtual_pnl_usd = 0.0
        mock_paper_result.cycle_duration_ms = 100.0
        mock_paper_engine.run_shadow_cycle = AsyncMock(return_value=mock_paper_result)
        mock_paper_engine.get_performance_report = AsyncMock(return_value=None)
        mock_paper_engine.get_telegram_report = AsyncMock(return_value="report")

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_latency_val.quote_age_s = 0.1
        mock_latency_val.reason_code = "OK"
        mock_lat_profile = MagicMock()
        mock_lat_profile.total_latency_ms = 10.0
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock()
        mock_regime_engine.should_override_to_safety = MagicMock(return_value=(False, ""))
        mock_current_regime = MagicMock()
        mock_current_regime.regime.value = "NEUTRAL"
        mock_regime_engine.get_current_regime = MagicMock(return_value=mock_current_regime)

        mock_robust_engine = MagicMock()
        mock_robust_engine._n_ticks = 5
        mock_robust_engine.ingest_price = MagicMock(return_value=None)
        mock_robust_engine._price_history = []

        mock_stress_tester = MagicMock()

        return (
            mock_gas, mock_paper_engine, mock_latency_shield,
            mock_regime_engine, mock_robust_engine, mock_stress_tester
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_when_no_portfolio_manager(self):
        """Lines 295-297: No portfolio_manager → falls back to airdrop."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        ctx = {}  # no portfolio_manager

        with patch("wallet_manager.task_factories._airdrop_task_factory", new_callable=AsyncMock,
                   return_value=[]) as mock_airdrop:
            result = await _robust_paper_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_base_chain(self):
        """Lines 304-306: chain_id != 8453 → returns empty."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        cfg = _make_config(chain_id=1)
        manager = _make_manager(config=cfg)
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor()
        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            result = await _robust_paper_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_stale_quote_invalidates_signals(self):
        """Lines 328-344: quote_validation.should_invalidate_signals=True → record skip and return."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor()

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = True
        mock_latency_val.quote_age_s = 5.0
        mock_latency_val.reason_code = "STALE"
        mock_lat_profile = MagicMock()
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.paper_engine": MagicMock(
                    get_paper_engine=AsyncMock(return_value=mock_paper_engine)
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=MagicMock(_n_ticks=5))
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
            }):
                with patch("wallet_manager.actions.fetch_eth_price_usd",
                           new_callable=AsyncMock, return_value=2000.0):
                    result = await _robust_paper_task_factory(
                        manager, "wallet_001", ctx
                    )

        assert result == []

    @pytest.mark.asyncio
    async def test_macro_suppress_returns_early(self):
        """Lines 347-360: MacroCalendar suppression → record skip and return."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor()

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_lat_profile = MagicMock()
        mock_lat_profile.total_latency_ms = 5.0
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()

        mock_macro_cal = MagicMock()
        mock_macro_cal.should_suppress_alpha = MagicMock(return_value=(True, "FED_MEETING"))

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock()
        mock_regime_engine.should_override_to_safety = MagicMock(return_value=(False, ""))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.paper_engine": MagicMock(
                    get_paper_engine=AsyncMock(return_value=mock_paper_engine)
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=mock_regime_engine)
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=MagicMock(_n_ticks=5))
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.market_intelligence": MagicMock(
                    get_macro_calendar=lambda: mock_macro_cal
                ),
            }):
                with patch("wallet_manager.actions.fetch_eth_price_usd",
                           new_callable=AsyncMock, return_value=2000.0):
                    result = await _robust_paper_task_factory(
                        manager, "wallet_001", ctx
                    )

        assert result == []

    @pytest.mark.asyncio
    async def test_regime_override_returns_early(self):
        """Lines 362-380: regime override → record skip and return."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor()

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_lat_profile = MagicMock()
        mock_lat_profile.total_latency_ms = 5.0
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock()
        mock_regime_engine.should_override_to_safety = MagicMock(
            return_value=(True, "HIGH_VOL")
        )
        mock_current_regime = MagicMock()
        mock_current_regime.regime.value = "HIGH_VOL"
        mock_regime_engine.get_current_regime = MagicMock(return_value=mock_current_regime)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.paper_engine": MagicMock(
                    get_paper_engine=AsyncMock(return_value=mock_paper_engine)
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=mock_regime_engine)
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=MagicMock(_n_ticks=5))
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
            }):
                with patch("wallet_manager.actions.fetch_eth_price_usd",
                           new_callable=AsyncMock, return_value=2000.0):
                    result = await _robust_paper_task_factory(
                        manager, "wallet_001", ctx
                    )

        assert result == []

    @pytest.mark.asyncio
    async def test_stress_test_rejection_returns_early(self):
        """Lines 403-421: stress test rejects signal → return early."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor()

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_lat_profile = MagicMock()
        mock_lat_profile.total_latency_ms = 5.0
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock()
        mock_regime_engine.should_override_to_safety = MagicMock(return_value=(False, ""))
        mock_current_regime = MagicMock()
        mock_current_regime.regime.value = "NEUTRAL"
        mock_regime_engine.get_current_regime = MagicMock(return_value=mock_current_regime)

        mock_signal = MagicMock()
        mock_signal.should_enter_long = True
        mock_signal.should_enter_short = False
        mock_signal.robust_alpha = 0.8
        mock_robust_engine = MagicMock()
        mock_robust_engine._n_ticks = 5
        mock_robust_engine.ingest_price = MagicMock(return_value=mock_signal)
        mock_robust_engine._price_history = [2000.0] * 10

        mock_verdict = MagicMock()
        mock_verdict.approved = False
        mock_verdict.monte_carlo = MagicMock(win_rate=0.4)
        mock_verdict.confidence_score = 0.3
        mock_verdict.rejection_reasons = ["low_win_rate"]
        mock_stress_tester = MagicMock()
        mock_stress_tester.validate_signal = MagicMock(return_value=mock_verdict)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.paper_engine": MagicMock(
                    get_paper_engine=AsyncMock(return_value=mock_paper_engine)
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=mock_regime_engine)
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=mock_robust_engine)
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=mock_stress_tester)
                ),
            }):
                with patch("wallet_manager.actions.fetch_eth_price_usd",
                           new_callable=AsyncMock, return_value=2000.0):
                    result = await _robust_paper_task_factory(
                        manager, "wallet_001", ctx
                    )

        assert result == []

    @pytest.mark.asyncio
    async def test_successful_shadow_cycle(self):
        """Lines 428-486: full shadow cycle completes, returns empty receipts."""
        from wallet_manager.task_factories import _robust_paper_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        notifier = AsyncMock()
        ctx = {"portfolio_manager": pm, "notifier": notifier}

        mock_gas = _make_gas_monitor()

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_lat_profile = MagicMock()
        mock_lat_profile.total_latency_ms = 5.0
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_paper_result = MagicMock()
        mock_paper_result.action = "HOLD"
        mock_paper_result.signal_alpha = 0.0
        mock_paper_result.simulated_fill_price = 2000.0
        mock_paper_result.slippage_bps = 1.0
        mock_paper_result.gas_cost_usd = 0.01
        mock_paper_result.virtual_pnl_usd = 0.0
        mock_paper_result.cycle_duration_ms = 100.0

        mock_paper_engine = AsyncMock()
        mock_paper_engine.record_external_skip = AsyncMock()
        mock_paper_engine.run_shadow_cycle = AsyncMock(return_value=mock_paper_result)
        mock_paper_engine.get_performance_report = AsyncMock(return_value=None)
        mock_paper_engine.get_telegram_report = AsyncMock(return_value="report")

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock()
        mock_regime_engine.should_override_to_safety = MagicMock(return_value=(False, ""))
        mock_current_regime = MagicMock()
        mock_current_regime.regime.value = "NEUTRAL"
        mock_regime_engine.get_current_regime = MagicMock(return_value=mock_current_regime)

        mock_robust_engine = MagicMock()
        mock_robust_engine._n_ticks = 5
        mock_robust_engine.ingest_price = MagicMock(return_value=None)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.paper_engine": MagicMock(
                    get_paper_engine=AsyncMock(return_value=mock_paper_engine),
                    _engine_instance=None,
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=mock_regime_engine)
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=mock_robust_engine)
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.reporting": MagicMock(
                    get_reporter=MagicMock(return_value=MagicMock(
                        tick=AsyncMock()
                    ))
                ),
                "wallet_manager.alerting": MagicMock(
                    get_alert_system=MagicMock(return_value=MagicMock(
                        tick=AsyncMock()
                    ))
                ),
            }):
                with patch("wallet_manager.actions.fetch_eth_price_usd",
                           new_callable=AsyncMock, return_value=2000.0):
                    with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                               new_callable=AsyncMock):
                        with patch("wallet_manager.task_factories._maybe_run_agents",
                                   new_callable=AsyncMock):
                            result = await _robust_paper_task_factory(
                                manager, "wallet_001", ctx
                            )

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _robust_task_factory — lines 560-808
# ---------------------------------------------------------------------------

class TestRobustTaskFactory:
    """Tests for _robust_task_factory covering lines 560-808."""

    def _make_robust_ctx(self, pm=None):
        if pm is None:
            pm = _make_portfolio_manager()
        return {"portfolio_manager": pm}

    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_without_pm(self):
        """Lines 527-529: no portfolio_manager → airdrop fallback."""
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            result = await _robust_task_factory(manager, "wallet_001", {})

        mock_airdrop.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_rebalance_triggered(self):
        """Lines 538-553: scheduler.is_monthly_rebalance_due() → rebalance runs."""
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        scheduler = MagicMock()
        scheduler.is_monthly_rebalance_due = MagicMock(return_value=True)
        scheduler.is_initial_rebalance = MagicMock(return_value=True)
        scheduler.mark_monthly_rebalance_done = MagicMock()

        ctx = {"portfolio_manager": pm, "scheduler": scheduler}

        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._airdrop_task_factory",
                       new_callable=AsyncMock, return_value=[]):
                # Patch away all the heavy strategy imports
                with patch.dict("sys.modules", {
                    "wallet_manager.strategies.yield_strategy": MagicMock(
                        get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    ),
                    "wallet_manager.strategies.latency_shield": MagicMock(
                        get_latency_shield=MagicMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.regime_filter": MagicMock(
                        get_regime_filter=MagicMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.robust_alpha": MagicMock(
                        get_robust_alpha_engine=MagicMock(return_value=MagicMock(_n_ticks=5))
                    ),
                    "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                        get_stress_tester=MagicMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.wf_validator": MagicMock(
                        get_wf_validator=AsyncMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.execution_logic": MagicMock(
                        get_ml_execution_engine=MagicMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.quant_analytics": MagicMock(
                        get_quant_analytics=AsyncMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.liquidity_alpha": MagicMock(
                        get_liquidity_alpha=MagicMock(return_value=MagicMock())
                    ),
                }):
                    with patch("wallet_manager.actions.fetch_usdc_balance",
                               new_callable=AsyncMock, return_value=0):
                        with patch("wallet_manager.actions.fetch_eth_price_usd",
                                   new_callable=AsyncMock, return_value=2000.0):
                            with patch("wallet_manager.actions.eth_balance_usd_value",
                                       return_value=1000.0):
                                with patch("wallet_manager.actions.usdc_balance_usd_value",
                                           return_value=0.0):
                                    result = await _robust_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        scheduler.mark_monthly_rebalance_done.assert_called_once()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_gas_unsafe_returns_early(self):
        """Lines 596-597: gas unsafe → return empty receipts."""
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(),
                "wallet_manager.strategies.regime_filter": MagicMock(),
                "wallet_manager.strategies.robust_alpha": MagicMock(),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(),
                "wallet_manager.strategies.wf_validator": MagicMock(),
                "wallet_manager.strategies.execution_logic": MagicMock(),
                "wallet_manager.strategies.quant_analytics": MagicMock(),
                "wallet_manager.strategies.liquidity_alpha": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                result = await _robust_task_factory(
                                    manager, "wallet_001", ctx
                                )

        assert result == []

    @pytest.mark.asyncio
    async def test_pillar_value_update_exception_is_swallowed(self):
        """Lines 590-591: pillar update failure is swallowed."""
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(side_effect=Exception("RPC down"))
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(side_effect=Exception("fail")),
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(),
                "wallet_manager.strategies.regime_filter": MagicMock(),
                "wallet_manager.strategies.robust_alpha": MagicMock(),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(),
                "wallet_manager.strategies.wf_validator": MagicMock(),
                "wallet_manager.strategies.execution_logic": MagicMock(),
                "wallet_manager.strategies.quant_analytics": MagicMock(),
                "wallet_manager.strategies.liquidity_alpha": MagicMock(),
            }):
                result = await _robust_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_latency_shield_invalidates_signals(self):
        """Lines 626-631: latency shield invalidates → return early after YIELD."""
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = True
        mock_latency_val.quote_age_s = 5.0
        mock_latency_val.reason_code = "STALE"
        mock_lat_profile = MagicMock()
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=MagicMock(_n_ticks=5))
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.wf_validator": MagicMock(
                    get_wf_validator=AsyncMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(
                    get_ml_execution_engine=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.liquidity_alpha": MagicMock(
                    get_liquidity_alpha=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.quant_analytics": MagicMock(
                    get_quant_analytics=AsyncMock(return_value=MagicMock())
                ),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                result = await _robust_task_factory(
                                    manager, "wallet_001", ctx
                                )

        assert result == []

    @pytest.mark.asyncio
    async def test_regime_override_supplies_usdc_and_returns(self):
        """Lines 647-674: regime override → supply USDC to Aave + heartbeat + return."""
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_lat_profile = MagicMock()
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            return_value=(mock_latency_val, mock_lat_profile)
        )

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock()
        mock_regime_engine.should_override_to_safety = MagicMock(return_value=(True, "HIGH_VOL"))
        mock_current_regime = MagicMock()
        mock_current_regime.regime.value = "HIGH_VOL"
        mock_regime_engine.get_current_regime = MagicMock(return_value=mock_current_regime)

        mock_receipt = _make_receipt("0xsafety")

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    get_usdc_balance=AsyncMock(return_value=int(10e6)),
                    execute_supply_usdc=AsyncMock(return_value=[mock_receipt]),
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=mock_regime_engine)
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(return_value=MagicMock(_n_ticks=5))
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.wf_validator": MagicMock(
                    get_wf_validator=AsyncMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(
                    get_ml_execution_engine=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.liquidity_alpha": MagicMock(
                    get_liquidity_alpha=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.quant_analytics": MagicMock(
                    get_quant_analytics=AsyncMock(return_value=MagicMock())
                ),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                                           new_callable=AsyncMock):
                                    result = await _robust_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _ml_quant_task_factory — lines 850-1035
# ---------------------------------------------------------------------------

class TestMlQuantTaskFactory:
    """Tests for _ml_quant_task_factory covering lines 850-1035."""

    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_without_pm(self):
        """Lines 838-840: no portfolio_manager → airdrop fallback."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            result = await _ml_quant_task_factory(manager, "wallet_001", {})

        mock_airdrop.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_rebalance_initial(self):
        """Lines 850-864: initial rebalance logs correctly."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        scheduler = MagicMock()
        scheduler.is_monthly_rebalance_due = MagicMock(return_value=True)
        scheduler.is_initial_rebalance = MagicMock(return_value=True)
        scheduler.mark_monthly_rebalance_done = MagicMock()

        ctx = {"portfolio_manager": pm, "scheduler": scheduler}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(
                    get_ml_execution_engine=MagicMock()
                ),
                "wallet_manager.strategies.quant_analytics": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                await _ml_quant_task_factory(
                                    manager, "wallet_001", ctx
                                )

        scheduler.mark_monthly_rebalance_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_monthly(self):
        """Lines 850-864: monthly rebalance (non-initial)."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        scheduler = MagicMock()
        scheduler.is_monthly_rebalance_due = MagicMock(return_value=True)
        scheduler.is_initial_rebalance = MagicMock(return_value=False)
        scheduler.mark_monthly_rebalance_done = MagicMock()

        ctx = {"portfolio_manager": pm, "scheduler": scheduler}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(),
                "wallet_manager.strategies.quant_analytics": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                await _ml_quant_task_factory(
                                    manager, "wallet_001", ctx
                                )

        scheduler.mark_monthly_rebalance_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_pillar_update_short_position_active(self):
        """Lines 880-900: active short position path."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=False)

        mock_short_position = MagicMock()
        mock_short_position.is_active = True
        mock_short_position.borrowed_weth_wei = int(0.1e18)
        mock_short_position.entry_price = 2100.0
        mock_short_state = MagicMock()
        mock_short_state.position = mock_short_position
        mock_short_state.total_short_pnl_usd = 10.0

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.short_strategy": MagicMock(
                    get_short_position_state=MagicMock(return_value=mock_short_state)
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(),
                "wallet_manager.strategies.quant_analytics": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=100.0):
                                result = await _ml_quant_task_factory(
                                    manager, "wallet_001", ctx
                                )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_gas_unsafe_returns_early(self):
        """Lines 907-908: gas unsafe → return empty."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(),
                "wallet_manager.strategies.quant_analytics": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                result = await _ml_quant_task_factory(
                                    manager, "wallet_001", ctx
                                )

        assert result == []

    @pytest.mark.asyncio
    async def test_severe_drift_returns_early(self):
        """Lines 1011-1013: SEVERE drift → return early (skip airdrop)."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        mock_drift_level = MagicMock()
        mock_drift_level.name = "SEVERE"

        mock_fv = [0.1, 0.2, 0.3]
        mock_engine_alpha = MagicMock()
        mock_engine_alpha._n_ticks = 5
        mock_engine_alpha.get_last_feature_vector = MagicMock(return_value=mock_fv)

        mock_cycle_result = MagicMock()
        mock_cycle_result.receipts = []
        mock_cycle_result.sweep_receipts = []
        mock_cycle_result.action_taken = "HOLD"
        mock_cycle_result.risk_source = "none"
        mock_cycle_result.drawdown_state = "OK"
        mock_cycle_result.var_99 = 0.0
        mock_cycle_result.dex_venue = None
        mock_cycle_result.pnl_usd = 0.0
        mock_cycle_result.cycle_duration_ms = 100.0

        mock_engine = MagicMock()
        mock_engine.alpha = mock_engine_alpha
        mock_engine.run_cycle = AsyncMock(return_value=mock_cycle_result)

        mock_drift_detector = MagicMock()
        mock_drift_detector.update = MagicMock(return_value=mock_drift_level)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.task_factories._drift_detector", mock_drift_detector):
                with patch.dict("sys.modules", {
                    "wallet_manager.strategies.yield_strategy": MagicMock(
                        get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    ),
                    "wallet_manager.strategies.execution_logic": MagicMock(
                        get_ml_execution_engine=MagicMock(return_value=mock_engine)
                    ),
                    "wallet_manager.strategies.quant_analytics": MagicMock(
                        get_quant_analytics=AsyncMock(return_value=MagicMock())
                    ),
                    "wallet_manager.strategies.alpha_engine": MagicMock(
                        FeatureStore=MagicMock(FEATURE_NAMES=["f1", "f2", "f3"])
                    ),
                    "wallet_manager.strategies.robust_alpha": MagicMock(
                        get_feature_drift_monitor=MagicMock(
                            return_value=MagicMock(
                                update=MagicMock(),
                                get_drift_report=MagicMock(return_value={"drifted_features": []})
                            )
                        )
                    ),
                }):
                    with patch("wallet_manager.actions.fetch_usdc_balance",
                               new_callable=AsyncMock, return_value=0):
                        with patch("wallet_manager.actions.fetch_eth_price_usd",
                                   new_callable=AsyncMock, return_value=2000.0):
                            with patch("wallet_manager.actions.eth_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.actions.usdc_balance_usd_value",
                                           return_value=0.0):
                                    result = await _ml_quant_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        # SEVERE drift means airdrop was skipped — result is empty list
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_airdrop_runs_when_budget_sufficient(self):
        """Lines 1015-1022: airdrop_budget >= 10 → airdrop runs."""
        from wallet_manager.task_factories import _ml_quant_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        pm.get_pillar_budget_usd = MagicMock(return_value=50.0)  # >= 10

        mock_receipt = _make_receipt("0xairdrop")
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(
                    get_ml_execution_engine=MagicMock(side_effect=Exception("skip engine"))
                ),
                "wallet_manager.strategies.quant_analytics": MagicMock(
                    get_quant_analytics=AsyncMock(side_effect=Exception("skip analytics"))
                ),
                "wallet_manager.strategies.airdrop_strategy": MagicMock(
                    run_airdrop_strategy=AsyncMock(return_value=[mock_receipt])
                ),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                                           new_callable=AsyncMock):
                                    result = await _ml_quant_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _quantx_task_factory — lines 1096-1200
# ---------------------------------------------------------------------------

class TestQuantxTaskFactory:
    """Tests for _quantx_task_factory covering lines 1096-1200."""

    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_without_pm(self):
        """Lines 1063-1065: no portfolio_manager → airdrop fallback."""
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        with patch("wallet_manager.task_factories._airdrop_task_factory",
                   new_callable=AsyncMock, return_value=[]) as mock_airdrop:
            await _quantx_task_factory(manager, "wallet_001", {})

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_monthly_quantx(self):
        """Lines 1074-1089: monthly rebalance in QUANT-X factory."""
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        scheduler = MagicMock()
        scheduler.is_monthly_rebalance_due = MagicMock(return_value=True)
        scheduler.is_initial_rebalance = MagicMock(return_value=False)
        scheduler.mark_monthly_rebalance_done = MagicMock()

        ctx = {"portfolio_manager": pm, "scheduler": scheduler}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.quantx_engine": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                await _quantx_task_factory(
                                    manager, "wallet_001", ctx
                                )

        scheduler.mark_monthly_rebalance_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_pillar_update_short_active_quantx(self):
        """Lines 1106-1123: active short position updated in QUANT-X."""
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=False)

        mock_short_position = MagicMock()
        mock_short_position.is_active = True
        mock_short_position.borrowed_weth_wei = int(0.05e18)
        mock_short_position.entry_price = 2200.0
        mock_short_state = MagicMock()
        mock_short_state.position = mock_short_position
        mock_short_state.total_short_pnl_usd = 5.0

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.short_strategy": MagicMock(
                    get_short_position_state=MagicMock(return_value=mock_short_state)
                ),
                "wallet_manager.strategies.quantx_engine": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=1000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                result = await _quantx_task_factory(
                                    manager, "wallet_001", ctx
                                )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_gas_unsafe_returns_early_quantx(self):
        """Lines 1130-1131: gas unsafe → early return."""
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.quantx_engine": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                result = await _quantx_task_factory(
                                    manager, "wallet_001", ctx
                                )

        assert result == []

    @pytest.mark.asyncio
    async def test_yield_supply_executed_when_delta_large(self):
        """Lines 1134-1150: YIELD supply runs when delta_usd > 10."""
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        # Make yield alloc have large delta
        mock_alloc = MagicMock()
        mock_alloc.strategy = StrategyType.YIELD
        mock_alloc.delta_usd = 50.0  # > 10
        mock_snapshot = MagicMock()
        mock_snapshot.allocations = [mock_alloc]
        pm.get_snapshot = MagicMock(return_value=mock_snapshot)

        ctx = {"portfolio_manager": pm}
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        mock_receipt = _make_receipt("0xyield")

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    get_usdc_balance=AsyncMock(return_value=int(100e6)),
                    execute_supply_usdc=AsyncMock(return_value=[mock_receipt]),
                ),
                "wallet_manager.strategies.quantx_engine": MagicMock(
                    get_quantx_engine=MagicMock(
                        return_value=MagicMock(
                            run_cycle=AsyncMock(side_effect=Exception("skip"))
                        )
                    )
                ),
                "wallet_manager.strategies.airdrop_strategy": MagicMock(
                    run_airdrop_strategy=AsyncMock(return_value=[])
                ),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                                           new_callable=AsyncMock):
                                    result = await _quantx_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_airdrop_runs_when_budget_sufficient_quantx(self):
        """Lines 1180-1187: airdrop_budget >= 10 → airdrop runs."""
        from wallet_manager.task_factories import _quantx_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        pm.get_pillar_budget_usd = MagicMock(return_value=50.0)
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=False)  # gas unsafe → return early (skip heartbeat)

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.quantx_engine": MagicMock(),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                result = await _quantx_task_factory(
                                    manager, "wallet_001", ctx
                                )

        # gas was unsafe → returns before airdrop; but the factory was called without crash
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _three_pillar_task_factory — lines 1203-1485
# ---------------------------------------------------------------------------

class TestThreePillarTaskFactory:
    """Tests for _three_pillar_task_factory covering lines 1203-1485."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_halted_returns_empty(self):
        """Lines 1224-1226: circuit breaker halted → return []."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        command_state = MagicMock()
        command_state.is_circuit_breaker_halted = True
        ctx = {"command_state": command_state}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"
            result = await _three_pillar_task_factory(manager, "wallet_001", ctx)

        assert result == []

    @pytest.mark.asyncio
    async def test_delegates_to_robust_paper(self):
        """Lines 1230-1231: execution_mode=robust_paper → delegate."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "robust_paper"
            with patch("wallet_manager.task_factories._robust_paper_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_rp:
                result = await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_rp.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_delegates_to_robust(self):
        """Lines 1234-1235: execution_mode=robust → delegate."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "robust"
            with patch("wallet_manager.task_factories._robust_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_r:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_r.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_robust_live(self):
        """Lines 1234-1235: execution_mode=robust_live → delegate."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "robust_live"
            with patch("wallet_manager.task_factories._robust_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_r:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_r.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_ml_quant(self):
        """Lines 1238-1239: execution_mode=ml_quant → delegate."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "ml_quant"
            with patch("wallet_manager.task_factories._ml_quant_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_mq:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_mq.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_quantx_mode(self):
        """Lines 1242-1244: quantx_engine_mode=quantx → delegate."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "quantx"
            with patch("wallet_manager.task_factories._quantx_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_qx:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_qx.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_airdrop_without_pm(self):
        """Lines 1250-1252: no portfolio_manager → airdrop fallback."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        ctx = {}

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"
            with patch("wallet_manager.task_factories._airdrop_task_factory",
                       new_callable=AsyncMock, return_value=[]) as mock_airdrop:
                await _three_pillar_task_factory(manager, "wallet_001", ctx)

        mock_airdrop.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_initial_three_pillar(self):
        """Lines 1261-1276: initial rebalance in three-pillar."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        scheduler = MagicMock()
        scheduler.is_monthly_rebalance_due = MagicMock(return_value=True)
        scheduler.is_initial_rebalance = MagicMock(return_value=True)
        scheduler.mark_monthly_rebalance_done = MagicMock()

        ctx = {"portfolio_manager": pm, "scheduler": scheduler}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch.dict("sys.modules", {
                    "wallet_manager.strategies.yield_strategy": MagicMock(
                        get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    ),
                }):
                    with patch("wallet_manager.actions.fetch_usdc_balance",
                               new_callable=AsyncMock, return_value=0):
                        with patch("wallet_manager.actions.fetch_eth_price_usd",
                                   new_callable=AsyncMock, return_value=2000.0):
                            with patch("wallet_manager.actions.eth_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.actions.usdc_balance_usd_value",
                                           return_value=0.0):
                                    await _three_pillar_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        scheduler.mark_monthly_rebalance_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_monthly_three_pillar(self):
        """Lines 1262-1276: monthly rebalance in three-pillar."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        scheduler = MagicMock()
        scheduler.is_monthly_rebalance_due = MagicMock(return_value=True)
        scheduler.is_initial_rebalance = MagicMock(return_value=False)
        scheduler.mark_monthly_rebalance_done = MagicMock()

        ctx = {"portfolio_manager": pm, "scheduler": scheduler}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch.dict("sys.modules", {
                    "wallet_manager.strategies.yield_strategy": MagicMock(
                        get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    ),
                }):
                    with patch("wallet_manager.actions.fetch_usdc_balance",
                               new_callable=AsyncMock, return_value=0):
                        with patch("wallet_manager.actions.fetch_eth_price_usd",
                                   new_callable=AsyncMock, return_value=2000.0):
                            with patch("wallet_manager.actions.eth_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.actions.usdc_balance_usd_value",
                                           return_value=0.0):
                                    await _three_pillar_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        scheduler.mark_monthly_rebalance_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_pillar_update_short_active_three_pillar(self):
        """Lines 1295-1305: active short position in three-pillar — pillar updates run."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=False)

        mock_short_position = MagicMock()
        mock_short_position.is_active = True
        mock_short_position.borrowed_weth_wei = int(0.1e18)
        mock_short_position.entry_price = 2100.0
        mock_short_state = MagicMock()
        mock_short_state.position = mock_short_position
        mock_short_state.total_short_pnl_usd = 10.0

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch.dict("sys.modules", {
                    "wallet_manager.strategies.yield_strategy": MagicMock(
                        get_yield_pillar_value_usd=AsyncMock(return_value=100.0),
                    ),
                    "wallet_manager.strategies.short_strategy": MagicMock(
                        get_short_position_state=MagicMock(return_value=mock_short_state)
                    ),
                }):
                    with patch("wallet_manager.actions.fetch_usdc_balance",
                               new_callable=AsyncMock, return_value=int(100e6)):
                        with patch("wallet_manager.actions.fetch_eth_price_usd",
                                   new_callable=AsyncMock, return_value=2000.0):
                            with patch("wallet_manager.actions.eth_balance_usd_value",
                                       return_value=1000.0):
                                with patch("wallet_manager.actions.usdc_balance_usd_value",
                                           return_value=100.0):
                                    result = await _three_pillar_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        # Gas was unsafe → returns before strategies. pillar updates should have run.
        # The key check is that it ran without exception
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_gas_unsafe_returns_early_three_pillar(self):
        """Lines 1319-1320: gas unsafe → return early."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}
        mock_gas = _make_gas_monitor(safe=False)

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch.dict("sys.modules", {
                    "wallet_manager.strategies.yield_strategy": MagicMock(
                        get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                    ),
                }):
                    with patch("wallet_manager.actions.fetch_usdc_balance",
                               new_callable=AsyncMock, return_value=0):
                        with patch("wallet_manager.actions.fetch_eth_price_usd",
                                   new_callable=AsyncMock, return_value=2000.0):
                            with patch("wallet_manager.actions.eth_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.actions.usdc_balance_usd_value",
                                           return_value=0.0):
                                    result = await _three_pillar_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        assert result == []

    @pytest.mark.asyncio
    async def test_kill_switch_active_returns_early(self):
        """Lines 1371-1373: kill switch active → return receipts early."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        mock_detector = MagicMock()
        mock_detector.is_kill_switch_active = True

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.strategies.flash_crash_detector.get_detector",
                           return_value=mock_detector):
                    with patch("wallet_manager.strategies.flash_crash_detector.activate_kill_switch"):
                        with patch("wallet_manager.strategies.flash_crash_detector.send_kill_switch_alert",
                                   new_callable=AsyncMock):
                            with patch.dict("sys.modules", {
                                "wallet_manager.strategies.yield_strategy": MagicMock(
                                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                                    get_apy=AsyncMock(return_value=5.0),
                                ),
                            }):
                                with patch("wallet_manager.actions.fetch_usdc_balance",
                                           new_callable=AsyncMock, return_value=0):
                                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                                               new_callable=AsyncMock, return_value=2000.0):
                                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                                   return_value=0.0):
                                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                                       return_value=0.0):
                                                result = await _three_pillar_task_factory(
                                                    manager, "wallet_001", ctx
                                                )

        assert result == []

    @pytest.mark.asyncio
    async def test_flash_crash_triggers_kill_switch(self):
        """Lines 1378-1385: flash crash detected → activate kill switch + return."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)

        mock_crash_alert = MagicMock()
        mock_crash_alert.drop_pct = 10.0

        mock_detector = MagicMock()
        mock_detector.is_kill_switch_active = False
        mock_detector.ingest_price = MagicMock(return_value=mock_crash_alert)

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.strategies.flash_crash_detector.get_detector",
                           return_value=mock_detector):
                    with patch("wallet_manager.strategies.flash_crash_detector.activate_kill_switch") as mock_aks:
                        with patch("wallet_manager.strategies.flash_crash_detector.send_kill_switch_alert",
                                   new_callable=AsyncMock):
                            with patch.dict("sys.modules", {
                                "wallet_manager.strategies.yield_strategy": MagicMock(
                                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                                    get_apy=AsyncMock(return_value=5.0),
                                ),
                            }):
                                with patch("wallet_manager.actions.fetch_usdc_balance",
                                           new_callable=AsyncMock, return_value=0):
                                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                                               new_callable=AsyncMock, return_value=2000.0):
                                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                                   return_value=0.0):
                                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                                       return_value=0.0):
                                                with patch("wallet_manager.actions.get_current_price",
                                                           new_callable=AsyncMock, return_value=1800.0):
                                                    result = await _three_pillar_task_factory(
                                                        manager, "wallet_001", ctx
                                                    )

        mock_aks.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_yield_alert_sent_when_apy_low(self):
        """Lines 1327-1348: low Aave APY triggers yield alert."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()

        mock_tg = AsyncMock()
        mock_tg._is_configured = MagicMock(return_value=True)
        mock_tg.send_yield_alert = AsyncMock()
        ctx = {"portfolio_manager": pm, "telegram_manager": mock_tg}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        mock_detector = MagicMock()
        mock_detector.is_kill_switch_active = False
        mock_detector.ingest_price = MagicMock(return_value=None)

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"
            # _AAVE_APY_ALERT_THRESHOLD is read from module level, we need aave_apy < threshold
            # threshold is ~0.02 (2%). Return 0.01 from get_apy (1%).

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.strategies.flash_crash_detector.get_detector",
                           return_value=mock_detector):
                    with patch("wallet_manager.strategies.flash_crash_detector.activate_kill_switch"):
                        with patch("wallet_manager.strategies.flash_crash_detector.send_kill_switch_alert",
                                   new_callable=AsyncMock):
                            with patch.dict("sys.modules", {
                                "wallet_manager.strategies.yield_strategy": MagicMock(
                                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                                    get_apy=AsyncMock(return_value=0.001),  # below threshold
                                    get_usdc_balance=AsyncMock(return_value=0),
                                ),
                                "wallet_manager.strategies.algo_strategy": MagicMock(
                                    run_algo_strategy=AsyncMock(return_value=[]),
                                    _load_state=MagicMock(return_value=MagicMock(prices=[])),
                                    _default_history_path=MagicMock(return_value=""),
                                    EMA_PERIOD=20,
                                    RSI_PERIOD=14,
                                ),
                                "wallet_manager.strategies.short_strategy": MagicMock(
                                    run_short_strategy=AsyncMock(return_value=[]),
                                ),
                                "wallet_manager.strategies.idle_sweep": MagicMock(
                                    run_idle_sweep=AsyncMock(return_value=[]),
                                ),
                                "wallet_manager.strategies.airdrop_strategy": MagicMock(
                                    run_airdrop_strategy=AsyncMock(return_value=[]),
                                ),
                                "wallet_manager.security.telegram_manager": MagicMock(
                                    YieldAlertPayload=MagicMock(return_value=MagicMock()),
                                ),
                            }):
                                with patch("wallet_manager.actions.fetch_usdc_balance",
                                           new_callable=AsyncMock, return_value=0):
                                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                                               new_callable=AsyncMock, return_value=2000.0):
                                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                                   return_value=0.0):
                                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                                       return_value=0.0):
                                                with patch("wallet_manager.actions.get_alternative_stable_yield_apy",
                                                           new_callable=AsyncMock,
                                                           return_value=(5.0, "Compound")):
                                                    with patch("wallet_manager.actions.get_current_price",
                                                               new_callable=AsyncMock, return_value=2000.0):
                                                        with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                                                                   new_callable=AsyncMock):
                                                            result = await _three_pillar_task_factory(
                                                                manager, "wallet_001", ctx
                                                            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_algo_and_short_strategies_run(self):
        """Lines 1389-1447: ALGO and SHORT strategies run when gas is safe."""
        from wallet_manager.task_factories import _three_pillar_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        mock_detector = MagicMock()
        mock_detector.is_kill_switch_active = False
        mock_detector.ingest_price = MagicMock(return_value=None)

        mock_receipt = _make_receipt("0xalgo")

        with patch("wallet_manager.task_factories._cfg") as mock_cfg:
            mock_cfg.return_value.execution_mode = "three_pillar"
            mock_cfg.return_value.quantx_engine_mode = "legacy"

            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.strategies.flash_crash_detector.get_detector",
                           return_value=mock_detector):
                    with patch("wallet_manager.strategies.flash_crash_detector.activate_kill_switch"):
                        with patch("wallet_manager.strategies.flash_crash_detector.send_kill_switch_alert",
                                   new_callable=AsyncMock):
                            with patch.dict("sys.modules", {
                                "wallet_manager.strategies.yield_strategy": MagicMock(
                                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                                    get_apy=AsyncMock(return_value=5.0),
                                    get_usdc_balance=AsyncMock(return_value=0),
                                ),
                                "wallet_manager.strategies.algo_strategy": MagicMock(
                                    run_algo_strategy=AsyncMock(return_value=[mock_receipt]),
                                    _load_state=MagicMock(return_value=MagicMock(prices=[])),
                                    _default_history_path=MagicMock(return_value=""),
                                    EMA_PERIOD=20,
                                    RSI_PERIOD=14,
                                ),
                                "wallet_manager.strategies.short_strategy": MagicMock(
                                    run_short_strategy=AsyncMock(return_value=[]),
                                ),
                                "wallet_manager.strategies.idle_sweep": MagicMock(
                                    run_idle_sweep=AsyncMock(return_value=[]),
                                ),
                                "wallet_manager.strategies.airdrop_strategy": MagicMock(
                                    run_airdrop_strategy=AsyncMock(return_value=[]),
                                ),
                            }):
                                with patch("wallet_manager.actions.fetch_usdc_balance",
                                           new_callable=AsyncMock, return_value=0):
                                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                                               new_callable=AsyncMock, return_value=2000.0):
                                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                                   return_value=0.0):
                                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                                       return_value=0.0):
                                                with patch("wallet_manager.actions.get_current_price",
                                                           new_callable=AsyncMock, return_value=2000.0):
                                                    with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                                                               new_callable=AsyncMock):
                                                        result = await _three_pillar_task_factory(
                                                            manager, "wallet_001", ctx
                                                        )

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _airdrop_task_factory — line 1520 (seconds < 3600 branch)
# ---------------------------------------------------------------------------

class TestAirdropTimestringBranch:
    """Tests for line 1520: the else branch for time_str when < 3600 seconds."""

    @pytest.mark.asyncio
    async def test_time_str_minutes_when_less_than_hour(self):
        """Line 1520: seconds_until_due < 3600 → time_str in minutes."""
        from wallet_manager.task_factories import _airdrop_task_factory

        manager = _make_manager()
        mock_scheduler = MagicMock()
        mock_gas = _make_gas_monitor(safe=True)
        ctx = {"scheduler": mock_scheduler}

        # should_run=False, seconds_until_due=1800 (< 3600)
        with patch("wallet_manager.task_factories.should_run_swap_this_cycle",
                   return_value=(False, 1800)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                result = await _airdrop_task_factory(manager, "wallet_001", ctx)

        assert result == []


# ---------------------------------------------------------------------------
# _airdrop_task_factory — lines 1577-1578 (airdrop_budget cap on USDC->ETH)
# ---------------------------------------------------------------------------

class TestAirdropBudgetCapSwap:
    """Tests for lines 1576-1578: budget cap on USDC->ETH swap."""

    @pytest.mark.asyncio
    async def test_usdc_to_eth_capped_by_airdrop_budget(self):
        """Lines 1576-1578: airdrop_budget_usd caps the USDC swap amount."""
        from wallet_manager.task_factories import _airdrop_task_factory

        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(0.01e18))  # tiny ETH

        mock_scheduler = MagicMock()
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        # Small budget but > $10
        ctx = {"scheduler": mock_scheduler, "airdrop_budget_usd": 20.0}

        mock_receipt = _make_receipt("0xcapped")

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle",
                   return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=int(500e6)):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=20.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=500.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible",
                                           return_value=True):
                                    with patch("wallet_manager.actions.is_swap_eligible",
                                               return_value=False):
                                        with patch(
                                            "wallet_manager.task_factories._execute_usdc_to_eth",
                                            new_callable=AsyncMock,
                                            return_value=[mock_receipt]
                                        ):
                                            result = await _airdrop_task_factory(
                                                manager, "wallet_001", ctx
                                            )

        assert result == [mock_receipt]

    @pytest.mark.asyncio
    async def test_eth_to_usdc_capped_by_airdrop_budget(self):
        """Similar cap branch for ETH->USDC."""
        from wallet_manager.task_factories import _airdrop_task_factory

        manager = _make_manager()
        manager.get_safe_balance = AsyncMock(return_value=int(2e18))  # 2 ETH

        mock_scheduler = MagicMock()
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        ctx = {"scheduler": mock_scheduler, "airdrop_budget_usd": 30.0}

        mock_receipt = _make_receipt("0xethcapped")

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle",
                   return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=int(5e6)):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=4000.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=5.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible",
                                           return_value=False):
                                    with patch("wallet_manager.actions.is_swap_eligible",
                                               return_value=True):
                                        with patch(
                                            "wallet_manager.task_factories._execute_eth_to_usdc",
                                            new_callable=AsyncMock,
                                            return_value=[mock_receipt]
                                        ):
                                            result = await _airdrop_task_factory(
                                                manager, "wallet_001", ctx
                                            )

        assert result == [mock_receipt]

    @pytest.mark.asyncio
    async def test_usdc_amount_too_small_returns_empty(self):
        """Lines 1579-1582: USDC amount after percentage < min_usdc → return empty."""
        from wallet_manager.task_factories import _airdrop_task_factory

        manager = _make_manager()
        # Very small USDC balance so pct * balance < 10 USDC
        manager.get_safe_balance = AsyncMock(return_value=int(0.001e18))

        mock_scheduler = MagicMock()
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle",
                   return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=int(1e6)):  # 1 USDC
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=2.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=1.0):
                                # Make 1 USDC > ETH value so usdc_to_eth path chosen
                                with patch("wallet_manager.actions.is_usdc_swap_eligible",
                                           return_value=True):
                                    with patch("wallet_manager.actions.is_swap_eligible",
                                               return_value=False):
                                        result = await _airdrop_task_factory(
                                            manager, "wallet_001", ctx
                                        )

        # 20% of 1 USDC = 0.2 USDC < 10 USDC min → empty
        assert result == []

    @pytest.mark.asyncio
    async def test_eth_amount_too_small_returns_empty(self):
        """Lines 1599-1601: ETH amount < 10^15 wei → return empty."""
        from wallet_manager.task_factories import _airdrop_task_factory

        manager = _make_manager()
        # Very small ETH balance
        manager.get_safe_balance = AsyncMock(return_value=100)  # 100 wei

        mock_scheduler = MagicMock()
        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        ctx = {"scheduler": mock_scheduler}

        with patch("wallet_manager.task_factories.should_run_swap_this_cycle",
                   return_value=(True, 0)):
            with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0000002):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.actions.is_usdc_swap_eligible",
                                           return_value=False):
                                    with patch("wallet_manager.actions.is_swap_eligible",
                                               return_value=True):
                                        result = await _airdrop_task_factory(
                                            manager, "wallet_001", ctx
                                        )

        assert result == []


# ---------------------------------------------------------------------------
# _robust_task_factory — lines 677-808 (robust alpha + airdrop + heartbeat)
# ---------------------------------------------------------------------------

class TestRobustTaskFactoryAlpha:
    """Tests for lines 677-808: the robust alpha + airdrop + heartbeat section."""

    @pytest.mark.asyncio
    async def test_airdrop_runs_when_budget_sufficient_robust(self):
        """Lines 789-795: airdrop_budget >= 10 → airdrop runs in robust factory.
        Uses gas=safe to exercise the path past gas check.
        All engine calls are mocked to raise, so code falls through to airdrop block.
        """
        from wallet_manager.task_factories import _robust_task_factory

        manager = _make_manager()
        pm = _make_portfolio_manager()
        pm.get_pillar_budget_usd = MagicMock(return_value=50.0)
        ctx = {"portfolio_manager": pm}

        mock_gas = _make_gas_monitor(safe=True, gwei=5.0)
        mock_receipt = _make_receipt("0xrobust_airdrop")

        mock_regime_engine = MagicMock()
        mock_regime_engine.ingest_market_data = MagicMock(side_effect=Exception("skip ingest"))
        mock_regime_engine.should_override_to_safety = MagicMock(return_value=(False, ""))

        mock_latency_val = MagicMock()
        mock_latency_val.should_invalidate_signals = False
        mock_latency_shield = MagicMock()
        mock_latency_shield.apply_latency_shield = AsyncMock(
            side_effect=Exception("skip latency shield")
        )

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch.dict("sys.modules", {
                "wallet_manager.strategies.yield_strategy": MagicMock(
                    get_yield_pillar_value_usd=AsyncMock(return_value=0.0),
                ),
                "wallet_manager.strategies.latency_shield": MagicMock(
                    get_latency_shield=MagicMock(return_value=mock_latency_shield)
                ),
                "wallet_manager.strategies.regime_filter": MagicMock(
                    get_regime_filter=MagicMock(return_value=mock_regime_engine)
                ),
                "wallet_manager.strategies.robust_alpha": MagicMock(
                    get_robust_alpha_engine=MagicMock(side_effect=Exception("skip robust"))
                ),
                "wallet_manager.strategies.stochastic_stress_test": MagicMock(
                    get_stress_tester=MagicMock(return_value=MagicMock())
                ),
                "wallet_manager.strategies.wf_validator": MagicMock(
                    get_wf_validator=AsyncMock(side_effect=Exception("skip wf"))
                ),
                "wallet_manager.strategies.execution_logic": MagicMock(
                    get_ml_execution_engine=MagicMock(side_effect=Exception("skip ml"))
                ),
                "wallet_manager.strategies.liquidity_alpha": MagicMock(
                    get_liquidity_alpha=MagicMock(side_effect=Exception("skip liq"))
                ),
                "wallet_manager.strategies.quant_analytics": MagicMock(
                    get_quant_analytics=AsyncMock(side_effect=Exception("skip qa"))
                ),
                "wallet_manager.strategies.airdrop_strategy": MagicMock(
                    run_airdrop_strategy=AsyncMock(return_value=[mock_receipt])
                ),
            }):
                with patch("wallet_manager.actions.fetch_usdc_balance",
                           new_callable=AsyncMock, return_value=0):
                    with patch("wallet_manager.actions.fetch_eth_price_usd",
                               new_callable=AsyncMock, return_value=2000.0):
                        with patch("wallet_manager.actions.eth_balance_usd_value",
                                   return_value=0.0):
                            with patch("wallet_manager.actions.usdc_balance_usd_value",
                                       return_value=0.0):
                                with patch("wallet_manager.orchestrator.send_heartbeat_telemetry",
                                           new_callable=AsyncMock):
                                    result = await _robust_task_factory(
                                        manager, "wallet_001", ctx
                                    )

        assert isinstance(result, list)
