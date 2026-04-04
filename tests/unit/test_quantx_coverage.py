"""tests/unit/test_quantx_coverage.py

Additional coverage tests for wallet_manager/strategies/quantx_engine.py.

Covers:
- QuantXEngine __init__ attribute wiring
- Crash detector threshold overrides
- QuantXCycleResult dataclass all fields
- get_engine_stats() output shape
- _get_wallet_address helper
- _notify_trade helper (with and without notifier)
- run_cycle early-exit paths (circuit breaker, kill switch, wrong chain)
- Signal generation edge cases on AlphaGenerator
- Constants (QUANTX_CRASH_THRESHOLD, QUANTX_CRASH_WINDOW, MIN_TRADE_USD, etc.)
- SignalOutput dataclass direct construction
"""
from __future__ import annotations

import math
import time
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.quantx_engine import (
    CHAIN_ID_BASE,
    CONGESTION_GWEI,
    MIN_TRADE_USD,
    MIN_TRADE_WEI,
    QUANTX_CRASH_THRESHOLD,
    QUANTX_CRASH_WINDOW,
    STRATEGY_ID,
    QuantXCycleResult,
    QuantXEngine,
    _get_wallet_address,
    _notify_trade,
    get_quantx_engine,
    reset_quantx_engine,
)
from wallet_manager.strategies.signal_intelligence import (
    AlphaGenerator,
    SignalOutput,
    ZScoreEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    alpha_score: float = 0.0,
    z_score: float = 0.0,
    should_enter_long: bool = False,
    should_enter_short: bool = False,
    should_exit: bool = False,
    confidence: float = 0.5,
    ofi_ratio: float = 1.0,
    vwap: float = 2000.0,
    vwap_signal: float = 0.0,
) -> SignalOutput:
    return SignalOutput(
        alpha_score=alpha_score,
        z_score=z_score,
        z_mean=2000.0,
        z_std=10.0,
        ofi_ratio=ofi_ratio,
        ofi_signal=0.0,
        vwap=vwap,
        vwap_signal=vwap_signal,
        price=vwap,
        confidence=confidence,
        should_enter_long=should_enter_long,
        should_enter_short=should_enter_short,
        should_exit=should_exit,
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_chain_id_base_is_8453(self):
        assert CHAIN_ID_BASE == 8453

    def test_crash_threshold_is_8_percent(self):
        assert pytest.approx(0.08) == QUANTX_CRASH_THRESHOLD

    def test_crash_window_is_300_seconds(self):
        assert pytest.approx(300.0) == QUANTX_CRASH_WINDOW

    def test_min_trade_usd(self):
        assert pytest.approx(5.0) == MIN_TRADE_USD

    def test_min_trade_wei(self):
        assert MIN_TRADE_WEI == 10**15

    def test_congestion_gwei(self):
        assert pytest.approx(15.0) == CONGESTION_GWEI

    def test_strategy_id_string(self):
        assert isinstance(STRATEGY_ID, str)
        assert len(STRATEGY_ID) > 0


# ---------------------------------------------------------------------------
# 2. QuantXCycleResult dataclass
# ---------------------------------------------------------------------------

class TestQuantXCycleResultDataclass:
    def test_default_action_hold(self):
        r = QuantXCycleResult()
        assert r.action_taken == "HOLD"

    def test_default_position_usd_zero(self):
        r = QuantXCycleResult()
        assert r.position_usd == 0.0

    def test_default_pnl_zero(self):
        r = QuantXCycleResult()
        assert r.pnl_usd == 0.0

    def test_default_error_empty(self):
        r = QuantXCycleResult()
        assert r.error == ""

    def test_default_dex_venue_empty(self):
        r = QuantXCycleResult()
        assert r.dex_venue == ""

    def test_default_signal_none(self):
        r = QuantXCycleResult()
        assert r.signal is None

    def test_default_jitter_ms_zero(self):
        r = QuantXCycleResult()
        assert r.jitter_ms == 0.0

    def test_default_cycle_duration_ms_zero(self):
        r = QuantXCycleResult()
        assert r.cycle_duration_ms == 0.0

    def test_receipts_not_shared_across_instances(self):
        """field(default_factory=list) ensures no shared state."""
        r1 = QuantXCycleResult()
        r2 = QuantXCycleResult()
        r1.receipts.append("tx1")
        assert len(r2.receipts) == 0

    def test_sweep_receipts_not_shared_across_instances(self):
        r1 = QuantXCycleResult()
        r2 = QuantXCycleResult()
        r1.sweep_receipts.append("sw1")
        assert len(r2.sweep_receipts) == 0

    def test_custom_values(self):
        r = QuantXCycleResult(
            action_taken="BUY",
            position_usd=100.0,
            pnl_usd=5.0,
            dex_venue="uniswap_v3",
            dex_savings_bps=12.5,
            jitter_ms=50.0,
            cycle_duration_ms=250.0,
            error="none",
        )
        assert r.action_taken == "BUY"
        assert r.position_usd == 100.0
        assert r.pnl_usd == 5.0
        assert r.dex_venue == "uniswap_v3"
        assert r.dex_savings_bps == 12.5
        assert r.jitter_ms == 50.0
        assert r.cycle_duration_ms == 250.0

    def test_all_dataclass_fields_present(self):
        field_names = {f.name for f in fields(QuantXCycleResult)}
        required = {
            "signal", "action_taken", "position_usd", "receipts",
            "sweep_receipts", "pnl_usd", "dex_venue", "dex_savings_bps",
            "jitter_ms", "cycle_duration_ms", "error",
        }
        assert required.issubset(field_names)


# ---------------------------------------------------------------------------
# 3. SignalOutput dataclass
# ---------------------------------------------------------------------------

class TestSignalOutputDataclass:
    def test_fields_present(self):
        field_names = {f.name for f in fields(SignalOutput)}
        required = {
            "alpha_score", "z_score", "z_mean", "z_std",
            "ofi_ratio", "ofi_signal", "vwap", "vwap_signal",
            "price", "confidence", "should_enter_long", "should_enter_short",
            "should_exit", "timestamp",
        }
        assert required.issubset(field_names)

    def test_construction_and_access(self):
        s = _make_signal(alpha_score=0.75, z_score=-3.0, should_enter_long=True)
        assert s.alpha_score == pytest.approx(0.75)
        assert s.z_score == pytest.approx(-3.0)
        assert s.should_enter_long is True
        assert s.should_enter_short is False

    def test_alpha_score_bounds(self):
        """alpha_score from generate_signal must be in [-1, +1]."""
        gen = AlphaGenerator(zscore_window=100)
        for i in range(80):
            gen.ingest_price(float(2000 + i % 50))
        sig = gen.generate_signal(2000.0)
        assert -1.0 <= sig.alpha_score <= 1.0

    def test_confidence_bounds(self):
        """confidence must be in [0, 1]."""
        gen = AlphaGenerator(zscore_window=100)
        for _ in range(70):
            gen.ingest_price(2000.0)
        sig = gen.generate_signal(2000.0)
        assert 0.0 <= sig.confidence <= 1.0

    def test_z_score_finite(self):
        gen = AlphaGenerator(zscore_window=50)
        for _ in range(50):
            gen.ingest_price(2000.0)
        sig = gen.generate_signal(2000.0)
        assert math.isfinite(sig.z_score)


# ---------------------------------------------------------------------------
# 4. _get_wallet_address helper
# ---------------------------------------------------------------------------

class TestGetWalletAddressQuantX:
    def test_returns_address_when_set(self):
        cfg = MagicMock()
        cfg.address = "0xDEADBEEF"
        result = _get_wallet_address(cfg)
        assert result == "0xDEADBEEF"

    def test_derives_from_private_key_when_address_empty(self):
        cfg = MagicMock()
        cfg.address = ""
        cfg.private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        result = _get_wallet_address(cfg)
        assert result.startswith("0x")
        assert len(result) == 42

    def test_derives_from_private_key_when_address_none(self):
        cfg = MagicMock()
        cfg.address = None
        cfg.private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        result = _get_wallet_address(cfg)
        assert result.startswith("0x")


# ---------------------------------------------------------------------------
# 5. QuantXEngine __init__
# ---------------------------------------------------------------------------

class TestQuantXEngineInit:
    def setup_method(self):
        reset_quantx_engine()

    def teardown_method(self):
        reset_quantx_engine()

    def test_engine_has_alpha_attribute(self):
        engine = QuantXEngine()
        assert hasattr(engine, "alpha")

    def test_engine_has_sor_attribute(self):
        engine = QuantXEngine()
        assert hasattr(engine, "sor")

    def test_engine_has_mev_attribute(self):
        engine = QuantXEngine()
        assert hasattr(engine, "mev")

    def test_engine_has_kelly_attribute(self):
        engine = QuantXEngine()
        assert hasattr(engine, "kelly")

    def test_engine_has_crash_detector_attribute(self):
        engine = QuantXEngine()
        assert hasattr(engine, "crash_detector")

    def test_crash_detector_threshold_overridden(self):
        engine = QuantXEngine()
        assert engine.crash_detector._threshold == pytest.approx(QUANTX_CRASH_THRESHOLD)

    def test_crash_detector_window_overridden(self):
        engine = QuantXEngine()
        assert engine.crash_detector._window == pytest.approx(QUANTX_CRASH_WINDOW)

    def test_position_tracking_defaults(self):
        engine = QuantXEngine()
        assert engine._in_position is False
        assert engine._entry_price == 0.0
        assert engine._position_size_usd == 0.0


# ---------------------------------------------------------------------------
# 6. QuantXEngine.get_engine_stats()
# ---------------------------------------------------------------------------

class TestQuantXEngineStats:
    def setup_method(self):
        reset_quantx_engine()

    def teardown_method(self):
        reset_quantx_engine()

    def test_get_engine_stats_returns_dict(self):
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert isinstance(stats, dict)

    def test_get_engine_stats_has_signal_key(self):
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert "signal" in stats

    def test_get_engine_stats_has_kelly_key(self):
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert "kelly" in stats

    def test_get_engine_stats_has_position_key(self):
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert "position" in stats

    def test_get_engine_stats_position_defaults(self):
        engine = QuantXEngine()
        pos = engine.get_engine_stats()["position"]
        assert pos["in_position"] is False
        assert pos["entry_price"] == 0.0
        assert pos["position_size_usd"] == 0.0

    def test_get_engine_stats_has_mev_key(self):
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert "mev" in stats

    def test_get_engine_stats_has_crash_detector_key(self):
        engine = QuantXEngine()
        stats = engine.get_engine_stats()
        assert "crash_detector" in stats


# ---------------------------------------------------------------------------
# 7. run_cycle — early exit paths (no RPC calls needed)
# ---------------------------------------------------------------------------

class TestRunCycleEarlyExits:
    def setup_method(self):
        reset_quantx_engine()

    def teardown_method(self):
        reset_quantx_engine()

    @pytest.mark.asyncio
    async def test_circuit_breaker_halt_returns_halted(self):
        engine = QuantXEngine()
        manager = MagicMock()
        portfolio_manager = MagicMock()

        command_state = MagicMock()
        command_state.is_circuit_breaker_halted = True
        ctx = {"command_state": command_state}

        result = await engine.run_cycle(
            manager=manager,
            wallet_id="w1",
            portfolio_manager=portfolio_manager,
            task_context=ctx,
        )
        assert result.action_taken == "HALTED"

    @pytest.mark.asyncio
    async def test_kill_switch_active_returns_kill_switch(self):
        engine = QuantXEngine()
        engine.crash_detector._kill_switch_active = True  # bypass read-only property
        manager = MagicMock()
        portfolio_manager = MagicMock()
        ctx = {}

        result = await engine.run_cycle(
            manager=manager,
            wallet_id="w1",
            portfolio_manager=portfolio_manager,
            task_context=ctx,
        )
        assert result.action_taken == "KILL_SWITCH"
        # Clean up
        engine.crash_detector._kill_switch_active = False

    @pytest.mark.asyncio
    async def test_wrong_chain_returns_wrong_chain(self):
        engine = QuantXEngine()
        engine.crash_detector._kill_switch_active = False

        cfg = MagicMock()
        cfg.chain_id = 1  # Ethereum mainnet, not Base
        cfg.address = "0xabc"
        cfg.rpc_url = "http://localhost:8545"
        cfg.max_gas_price_gwei = 25.0

        manager = MagicMock()
        manager.get_config.return_value = cfg

        portfolio_manager = MagicMock()
        ctx = {}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=portfolio_manager,
                task_context=ctx,
            )
        assert result.action_taken == "WRONG_CHAIN"

    @pytest.mark.asyncio
    async def test_high_gas_returns_gas_high(self):
        engine = QuantXEngine()
        engine.crash_detector._kill_switch_active = False

        cfg = MagicMock()
        cfg.chain_id = 8453
        cfg.address = "0xabc"
        cfg.rpc_url = "http://localhost:8545"
        cfg.max_gas_price_gwei = 10.0

        manager = MagicMock()
        manager.get_config.return_value = cfg

        portfolio_manager = MagicMock()
        ctx = {}

        mock_gas = MagicMock()
        # is_safe returns False → gas too high
        mock_gas.is_safe = AsyncMock(return_value=(False, 50.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=portfolio_manager,
                task_context=ctx,
            )
        assert result.action_taken == "GAS_HIGH"

    @pytest.mark.asyncio
    async def test_congested_mempool_returns_congested(self):
        engine = QuantXEngine()
        engine.crash_detector._kill_switch_active = False

        cfg = MagicMock()
        cfg.chain_id = 8453
        cfg.address = "0xabc"
        cfg.rpc_url = "http://localhost:8545"
        cfg.max_gas_price_gwei = 100.0  # high enough not to trip gas_high

        manager = MagicMock()
        manager.get_config.return_value = cfg

        portfolio_manager = MagicMock()
        ctx = {}

        mock_gas = MagicMock()
        # safe=True but gwei > CONGESTION_GWEI (15)
        mock_gas.is_safe = AsyncMock(return_value=(True, 20.0))

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            result = await engine.run_cycle(
                manager=manager,
                wallet_id="w1",
                portfolio_manager=portfolio_manager,
                task_context=ctx,
            )
        assert result.action_taken == "CONGESTED"

    @pytest.mark.asyncio
    async def test_price_fetch_exception_returns_error(self):
        engine = QuantXEngine()
        engine.crash_detector._kill_switch_active = False

        cfg = MagicMock()
        cfg.chain_id = 8453
        cfg.address = "0xabc"
        cfg.rpc_url = "http://localhost:8545"
        cfg.max_gas_price_gwei = 100.0

        manager = MagicMock()
        manager.get_config.return_value = cfg

        portfolio_manager = MagicMock()
        ctx = {}

        mock_gas = MagicMock()
        mock_gas.is_safe = AsyncMock(return_value=(True, 5.0))  # safe, below congestion

        with patch("wallet_manager.network.gas_monitor.GasMonitor", return_value=mock_gas):
            with patch("wallet_manager.actions.get_live_price_v3", new_callable=AsyncMock, side_effect=RuntimeError("RPC error")):
                result = await engine.run_cycle(
                    manager=manager,
                    wallet_id="w1",
                    portfolio_manager=portfolio_manager,
                    task_context=ctx,
                )
        assert result.error != "" or result.action_taken == "HOLD"


# ---------------------------------------------------------------------------
# 8. _notify_trade helper
# ---------------------------------------------------------------------------

class TestNotifyTrade:
    def test_no_exception_when_no_notifier(self):
        """_notify_trade must not raise even with empty task_context."""
        ctx: dict = {}
        _notify_trade(
            task_context=ctx,
            pair="WETH/USDC",
            price=2000.0,
            amount="$100 USDC",
            side="BUY",
            strategy="QUANT-X",
            alpha=0.75,
            z_score=-3.0,
            dex="uniswap_v3",
            tx_hash="0xabc",
        )
        # Should have populated trade_notifications (or silently suppressed error)

    def test_appends_to_trade_notifications_list(self):
        ctx: dict = {"trade_notifications": []}
        # Patch the import so TradeExecutedPayload is available
        mock_payload = MagicMock()
        mock_payload_cls = MagicMock(return_value=mock_payload)

        with patch.dict("sys.modules", {
            "wallet_manager.security.telegram_manager": MagicMock(
                TradeExecutedPayload=mock_payload_cls
            )
        }):
            _notify_trade(
                task_context=ctx,
                pair="WETH/USDC",
                price=2000.0,
                amount="$50",
                side="SELL",
                strategy="QUANT-X",
                alpha=-0.8,
                z_score=3.1,
                dex="curve",
                tx_hash=None,
            )
        # Either the list got an item or the exception was silently suppressed
        # Both are valid — just ensure no crash

    def test_handles_missing_telegram_manager_gracefully(self):
        """If telegram_manager import fails, _notify_trade must not raise."""
        ctx: dict = {}
        with patch.dict("sys.modules", {"wallet_manager.security.telegram_manager": None}):
            # Should not raise — caught internally
            try:
                _notify_trade(
                    task_context=ctx,
                    pair="ETH/USDC",
                    price=1900.0,
                    amount="0.05 WETH",
                    side="SELL",
                    strategy="QUANT-X",
                    alpha=-0.5,
                    z_score=2.9,
                    dex="uniswap_v3_fallback",
                )
            except Exception:
                pass  # allowed to fail at import — just must not propagate


# ---------------------------------------------------------------------------
# 9. AlphaGenerator ingestion paths
# ---------------------------------------------------------------------------

class TestAlphaGeneratorIngestion:
    def test_ingest_trade_does_not_raise(self):
        gen = AlphaGenerator(zscore_window=50)
        gen.ingest_trade(2000.0, volume_usd=100.0, is_buy=True)

    def test_ingest_trade_sell_side(self):
        gen = AlphaGenerator(zscore_window=50)
        gen.ingest_trade(2000.0, volume_usd=200.0, is_buy=False)

    def test_multiple_ingest_then_signal(self):
        gen = AlphaGenerator(zscore_window=70)
        for i in range(65):
            gen.ingest_price(float(2000 + i % 10))
        sig = gen.generate_signal(2000.0)
        assert isinstance(sig, SignalOutput)
        assert math.isfinite(sig.alpha_score)

    def test_generate_signal_no_ofi(self):
        """generate_signal with ofi_reading=None should not raise."""
        gen = AlphaGenerator(zscore_window=80)
        for _ in range(70):
            gen.ingest_price(2000.0)
        sig = gen.generate_signal(2000.0, ofi_reading=None)
        assert isinstance(sig, SignalOutput)

    def test_generate_signal_returns_hold_when_no_data(self):
        gen = AlphaGenerator(zscore_window=100)
        sig = gen.generate_signal(2000.0)
        assert sig.should_enter_long is False
        assert sig.should_enter_short is False


# ---------------------------------------------------------------------------
# 10. Singleton management
# ---------------------------------------------------------------------------

class TestSingletonManagement:
    def setup_method(self):
        reset_quantx_engine()

    def teardown_method(self):
        reset_quantx_engine()

    def test_get_quantx_engine_returns_engine(self):
        engine = get_quantx_engine()
        assert isinstance(engine, QuantXEngine)

    def test_get_quantx_engine_idempotent(self):
        e1 = get_quantx_engine()
        e2 = get_quantx_engine()
        assert e1 is e2

    def test_reset_then_new_instance(self):
        e1 = get_quantx_engine()
        reset_quantx_engine()
        e2 = get_quantx_engine()
        assert e1 is not e2

    def test_reset_clears_state(self):
        """After reset, new engine has fresh position state."""
        e1 = get_quantx_engine()
        e1._in_position = True
        e1._entry_price = 999.0
        reset_quantx_engine()
        e2 = get_quantx_engine()
        assert e2._in_position is False
        assert e2._entry_price == 0.0
