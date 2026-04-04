"""
wallet_manager/strategies/__init__.py

Portfolio strategy modules — QUANT-X Institutional Grade DeFi Engine.

Pillars:
  - YIELD   (40%): Passive Aave V3 USDC supply — zero lazy capital.
  - ALGO    (35%): QUANT-X Alpha Engine (Z-Score 2.0, OFI, VWAP, Kelly Criterion).
  - SHORT   (10%): Hedged delta-neutral synthetic short via Aave V3 borrow.
  - AIRDROP (15%): Ping-pong ETH<->USDC farming for protocol activity.

QUANT-X modules:
  - signal_intelligence: Z-Score 2.0 + OFI + VWAP composite alpha generator.
  - smart_order_router: Multi-DEX aggregation (Uniswap V3, SushiSwap, Aerodrome).
  - mev_shield: Private RPC routing + Gaussian Jitter + Dynamic RBF.
  - kelly_criterion: Optimal position sizing based on historical win/loss.
  - quantx_engine: Unified execution engine integrating all QUANT-X modules.

Support modules:
  - idle_sweep: Automatically sweeps idle USDC into Aave V3.
  - flash_crash_detector: 8% / 5min flash crash detection + KILL_SWITCH.

Robustness Shield (Phase 4 — Anti-Overfit):
  - robust_alpha: Fractional Differentiation + Permutation Feature Importance.
  - wf_validator: Purged Walk-Forward Cross-Validation engine.
  - regime_filter: Hidden Markov Model regime detection + Economic Calendar.
  - liquidity_alpha: V3 tick-level liquidity profiling + split-order execution.

Final Optimization (Phase 5 — Maximum Potency):
  - paper_engine: High-Fidelity Paper Trading with slippage sim + virtual ledger.
  - stochastic_stress_test: Monte Carlo chaos layer + Recursive Feature Elimination.
  - latency_shield: Network latency emulation + stale quote protection.
  - oci_deployer: OCI watchdog + health-check API + systemd template.
"""
from __future__ import annotations

import contextlib
import importlib
import logging

log = logging.getLogger(__name__)

# Defensive imports — missing optional ML dependencies must not crash startup
_STRATEGY_MODULES = [
    "alpha_engine",
    "deep_kelly_risk",
    "execution_logic",
    "regime_filter",
    "signal_intelligence",
    "smart_order_router",
    "market_microstructure",
    "alpha_decay_monitor",
    "wf_validator",
    "robust_alpha",
    "paper_engine",
    "liquidity_alpha",
    "mev_shield",
    "flash_crash_detector",
    "latency_shield",
    "stochastic_stress_test",
]

for _module_name in _STRATEGY_MODULES:
    try:
        importlib.import_module(f".{_module_name}", package=__name__)
    except ImportError as _exc:
        log.warning("Strategy import failed: %s — %s", _module_name, _exc)

# Always-required strategy modules — noqa: F401 because these are re-exported for
# callers that do ``from wallet_manager.strategies import yield_strategy`` etc.
from . import (  # noqa: F401
    airdrop_strategy,
    algo_strategy,
    idle_sweep,
    kelly_criterion,
    oci_deployer,
    quant_analytics,
    quantx_engine,
    short_strategy,
    yield_strategy,
)

# Conditional re-exports (graceful if ML libs absent)
with contextlib.suppress(ImportError):
    from wallet_manager.strategies.yield_strategy import (
        UserPosition,
        get_apy,
        get_user_position,
        get_yield_pillar_value_usd,
        supply_usdc,
        withdraw_usdc,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.algo_strategy import (
        bollinger_band_proximity,
        compute_bollinger_bands,
        run_algo_strategy,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.short_strategy import run_short_strategy

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.airdrop_strategy import run_airdrop_strategy

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.idle_sweep import run_idle_sweep

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.flash_crash_detector import (
        CrashAlert,
        FlashCrashDetector,
        get_detector,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.signal_intelligence import (
        AlphaGenerator,
        SignalOutput,
        get_alpha_generator,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.smart_order_router import (
        SmartOrderRouter,
        get_smart_order_router,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.mev_shield import (
        MEVShield,
        get_mev_shield,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.kelly_criterion import (
        KellyCriterion,
        get_kelly_criterion,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.quantx_engine import (
        QuantXEngine,
        get_quantx_engine,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.alpha_engine import (
        StochasticMLAlphaEngine,
        get_alpha_engine,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.deep_kelly_risk import (
        DeepKellyRiskEngine,
        get_deep_kelly_engine,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.execution_logic import (
        MLQuantExecutionEngine,
        get_ml_execution_engine,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.quant_analytics import (
        QuantAnalytics,
        get_quant_analytics,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.liquidity_alpha import (
        LiquidityAlpha,
        get_liquidity_alpha,
        reset_liquidity_alpha,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.robust_alpha import (
        RobustAlphaEngine,
        get_robust_alpha_engine,
        reset_robust_alpha_engine,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.wf_validator import (
        WalkForwardValidator,
        get_wf_validator,
        reset_wf_validator,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.regime_filter import (
        RegimeFilter,
        get_regime_filter,
        reset_regime_filter,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.paper_engine import (
        PaperExecutionBridge,
        PaperTradingEngine,
        get_paper_engine,
        reset_paper_engine,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.stochastic_stress_test import (
        StochasticStressTester,
        get_stress_tester,
        reset_stress_tester,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.latency_shield import (
        LatencyShield,
        get_latency_shield,
        reset_latency_shield,
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.oci_deployer import (
        OCIDeployer,
        get_oci_deployer,
        reset_oci_deployer,
    )
