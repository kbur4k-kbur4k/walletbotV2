"""tests/unit/test_paper_engine_missing.py — Extra coverage for paper_engine.py shadow cycle."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.paper_engine import (
    PaperTradeConfig,
    PaperTradeResult,
    PaperTradingEngine,
)


@pytest.fixture
def base_config():
    return PaperTradeConfig()

@pytest.fixture
def engine(base_config):
    eng = PaperTradingEngine(base_config)
    eng._ledger = AsyncMock()
    eng._ledger.get_total_equity_usd = AsyncMock(return_value=1000.0)
    eng._ledger.get_balance = AsyncMock(return_value=1000.0)
    eng._ledger.get_drawdown = AsyncMock(return_value={"current_drawdown": 0.0, "max_drawdown": 0.0})
    eng._ledger._max_drawdown = 0.0
    eng._ledger._last_equity_usd = 1000.0
    eng._gas_estimator = AsyncMock()
    eng._gas_estimator.estimate_gas_cost_usd = AsyncMock(return_value=1.0)
    eng._slippage = MagicMock()
    eng._slippage.simulate_slippage = MagicMock(return_value=(5.0, 2005.0))
    eng._journal_decision = AsyncMock()
    return eng

@pytest.mark.asyncio
async def test_run_shadow_cycle_flash_crash(engine):
    """Test flash crash circuit breaker."""
    mock_crash = MagicMock(ingest_price=MagicMock(return_value="CRASH_ALERT"))

    mock_signal = SimpleNamespace(robust_alpha=0.5, raw_alpha=0.5, confidence=0.5, stationarity_d=0.5, adf_pvalue=0.5)
    mock_robust = MagicMock(ingest_price=MagicMock(return_value=mock_signal))

    with patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=mock_crash):
        res = await engine.run_shadow_cycle(MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", chain_id=8453))), "w1", portfolio_manager=MagicMock(), task_context={})
        assert res.action == "FLASH_CRASH"
        engine._journal_decision.assert_called_once()

@pytest.mark.asyncio
async def test_run_shadow_cycle_risk_committee_reject(engine):
    """Test risk committee reject."""
    mock_robust = MagicMock(ingest_price=MagicMock(return_value=SimpleNamespace(robust_alpha=0.5, raw_alpha=0.6, confidence=0.8, stationarity_d=0.2, underlying_signal=None, adf_pvalue=0.01)))
    mock_rc = MagicMock(evaluate=AsyncMock(return_value=SimpleNamespace(allowed=False, blocking_brakes=["VOL"], narrative="High Vol")))

    with patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=MagicMock(ingest_price=MagicMock(return_value=None))), \
         patch("wallet_manager.risk_committee.get_risk_committee", return_value=mock_rc):
        res = await engine.run_shadow_cycle(MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", chain_id=8453))), "w1", portfolio_manager=MagicMock(), task_context={})
        assert res.action == "RISK_BLOCKED"

@pytest.mark.asyncio
async def test_run_shadow_cycle_buy_success(engine):
    """Test standard BUY path through all integrations."""
    # Ensure BUY condition is met
    engine._in_position = False

    mock_signal = SimpleNamespace(
        robust_alpha=0.5, raw_alpha=0.6, confidence=0.8, stationarity_d=0.2,
        underlying_signal=SimpleNamespace(volatility_regime="low_vol", garch_sigma=0.01),
        adf_pvalue=0.01, should_enter_long=True, should_enter_short=False, should_exit=False, reason_codes=[]
    )
    mock_robust = MagicMock(ingest_price=MagicMock(return_value=mock_signal))
    mock_rc = MagicMock(evaluate=AsyncMock(return_value=SimpleNamespace(allowed=True, blocking_brakes=[], narrative="OK")))
    mock_dk = MagicMock(calculate_position=MagicMock(return_value=SimpleNamespace(position_usd=500.0, drawdown_state="normal")))

    with patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=MagicMock(ingest_price=MagicMock(return_value=None))), \
         patch("wallet_manager.risk_committee.get_risk_committee", return_value=mock_rc), \
         patch("wallet_manager.strategies.deep_kelly_risk.get_deep_kelly_engine", return_value=mock_dk), \
         patch("wallet_manager.risk_intelligence.get_risk_intelligence", side_effect=Exception("Mocked out")), \
         patch("wallet_manager.strategies.bayesian_kelly.get_bayesian_kelly", side_effect=Exception("Mocked out")), \
         patch("wallet_manager.strategies.execution_quality.get_optimal_timing", side_effect=Exception("Mocked")), \
         patch("wallet_manager.config.get_settings", MagicMock(return_value=SimpleNamespace(paper_max_gas_fraction_pct=5.0))):

        # Disable shadow mode to avoid extra external checks
        res = await engine.run_shadow_cycle(MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", chain_id=8453))), "w1", portfolio_manager=MagicMock(), task_context={})
        assert res.action == "BUY", f"res: {res}"
        assert engine._in_position is True

@pytest.mark.asyncio
async def test_run_shadow_cycle_sell_success(engine):
    """Test standard SELL path through all integrations."""
    # Ensure SELL condition is met
    engine._in_position = True
    engine._entry_price = 1900.0
    engine._position_size_usd = 1000.0
    engine._ledger.get_balance = AsyncMock(side_effect=lambda a: 0.0 if a == "USDC" else 0.5) # 0.5 WETH

    mock_signal = SimpleNamespace(
        robust_alpha=-0.5, raw_alpha=-0.6, confidence=0.8, stationarity_d=0.2,
        underlying_signal=SimpleNamespace(volatility_regime="low_vol", garch_sigma=0.01),
        adf_pvalue=0.01, should_enter_long=False, should_enter_short=True, should_exit=True, reason_codes=[]
    )
    mock_robust = MagicMock(ingest_price=MagicMock(return_value=mock_signal))
    mock_rc = MagicMock(evaluate=AsyncMock(return_value=SimpleNamespace(allowed=True, blocking_brakes=[], narrative="OK")))
    mock_dk = MagicMock(calculate_position=MagicMock(return_value=SimpleNamespace(position_usd=500.0, drawdown_state="normal")))

    with patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=mock_robust), \
         patch("wallet_manager.strategies.flash_crash_detector.get_detector", return_value=MagicMock(ingest_price=MagicMock(return_value=None))), \
         patch("wallet_manager.risk_committee.get_risk_committee", return_value=mock_rc), \
         patch("wallet_manager.strategies.deep_kelly_risk.get_deep_kelly_engine", return_value=mock_dk), \
         patch("wallet_manager.config.get_settings", MagicMock(return_value=SimpleNamespace(paper_max_gas_fraction_pct=5.0))):

        res = await engine.run_shadow_cycle(MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", chain_id=8453))), "w1", portfolio_manager=MagicMock(), task_context={})
        assert res.action == "SELL"
        assert engine._in_position is False

@pytest.mark.asyncio
async def test_run_shadow_cycle_price_error(engine):
    """Test behavior if price fetch completely fails."""
    with patch("wallet_manager.actions.get_live_price_v3", AsyncMock(side_effect=Exception("API Down"))), \
         patch("wallet_manager.actions.get_weth_usdc_price", AsyncMock(return_value=0.0)), \
         patch("wallet_manager.actions.get_current_price", AsyncMock(return_value=0.0)):
        res = await engine.run_shadow_cycle(MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", chain_id=8453))), "w1", portfolio_manager=MagicMock(), task_context={})
        assert res.action == "PRICE_ERROR"

@pytest.mark.asyncio
async def test_run_shadow_cycle_warmup(engine):
    """Test behavior if robust engine returns None (warmup)."""
    with patch("wallet_manager.actions.get_live_price_v3", AsyncMock(return_value=2000.0)), \
         patch("wallet_manager.strategies.robust_alpha.get_robust_alpha_engine", return_value=MagicMock(ingest_price=MagicMock(return_value=None))):
        res = await engine.run_shadow_cycle(MagicMock(get_config=MagicMock(return_value=SimpleNamespace(rpc_url="url", chain_id=8453))), "w1", portfolio_manager=MagicMock(), task_context={})
        assert res.action == "WARMUP"
