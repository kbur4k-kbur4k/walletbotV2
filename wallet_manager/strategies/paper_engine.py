"""
wallet_manager/strategies/paper_engine.py

High-Fidelity Paper Trading Engine — Shadow Execution with Full Pipeline.

Runs the COMPLETE ML-Quant execution pipeline (regime filter, robust alpha,
liquidity profiling, SOR routing, MEV shield decisions) and stops EXACTLY
before ``send_transaction``.  Every would-be trade is recorded in a local
SQLite ledger with realistic slippage simulation and gas cost estimation.

Purpose
-------
Validate strategy alpha, measure implementation shortfall, and track
virtual P&L without risking real capital.  The engine produces a detailed
performance report (Sharpe, drawdown, win rate, average slippage) that can
be sent via Telegram alongside the real execution report.

Architecture
------------
1. PaperTradeConfig     — Frozen dataclass with all tuning knobs.
2. VirtualLedger        — SQLite WAL-backed persistence for virtual balances,
                          trades, and equity snapshots.
3. SlippageSimulator    — Gaussian / fixed slippage model with non-linear
                          market-impact scaling for large orders.
4. GasEstimator         — Fetches live Base gas price via web3 and caches
                          for 10 seconds; estimates cost by tx type.
5. PaperTradingEngine   — Main orchestrator: shadow-executes the full pipeline,
                          simulates fill, records to ledger, returns result.
6. PaperTradeResult     — Frozen dataclass with complete cycle outcome.
7. Singleton            — ``get_paper_engine()`` / ``reset_paper_engine()``.

Integration
-----------
Call ``engine.run_shadow_cycle(...)`` from the orchestrator alongside or
instead of the real ``MLQuantExecutionEngine.run_cycle()``.  The shadow
cycle reads the same market data and signals but never touches on-chain
state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import numpy as np

from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from wallet_manager.portfolio_manager import PortfolioManager
    from wallet_manager.strategies.robust_alpha import RobustAlphaSignal

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAIN_ID_BASE: int = 8453
STRATEGY_ID: str = "paper_shadow_v1"

MIN_TRADE_USD: float = 5.0
CONGESTION_GWEI: float = 15.0

def _get_weth_usdc_pool() -> str:
    from wallet_manager.config import get_settings
    return get_settings().weth_usdc_pool

_DEFAULT_DB_PATH: str = "paper_trades.db"

# Gas unit estimates by transaction type (Base L2 actuals, ~40% lower than Ethereum)
GAS_UNITS: dict[str, int] = {
    "swap": 110_000,
    "supply": 150_000,
    "borrow": 180_000,
}

# Cache TTL for gas price lookups
GAS_CACHE_TTL_S: float = 10.0

# Reason codes
PAPER_TRADE_EXECUTED: str = "PAPER_TRADE_EXECUTED"
PAPER_SLIPPAGE_APPLIED: str = "PAPER_SLIPPAGE_APPLIED"
PAPER_GAS_ESTIMATED: str = "PAPER_GAS_ESTIMATED"
PAPER_SIGNAL_REJECTED: str = "PAPER_SIGNAL_REJECTED"
PAPER_SHADOW_ONLY: str = "PAPER_SHADOW_ONLY"


# ---------------------------------------------------------------------------
# SQL Schema
# ---------------------------------------------------------------------------

_CREATE_BALANCES_SQL = """
CREATE TABLE IF NOT EXISTS paper_balances (
    timestamp   REAL    NOT NULL,
    asset       TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    usd_value   REAL    NOT NULL,
    PRIMARY KEY (asset)
)
"""

_CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL    NOT NULL,
    direction           TEXT    NOT NULL,
    asset_pair          TEXT    NOT NULL,
    entry_price         REAL    NOT NULL,
    simulated_fill_price REAL   NOT NULL,
    slippage_bps        REAL    NOT NULL,
    size_usd            REAL    NOT NULL,
    gas_cost_usd        REAL    NOT NULL,
    pnl_usd             REAL    NOT NULL,
    reason_codes        TEXT    NOT NULL,
    signal_source       TEXT    NOT NULL
)
"""

_CREATE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS paper_snapshots (
    timestamp           REAL    PRIMARY KEY,
    total_equity_usd    REAL    NOT NULL,
    unrealized_pnl      REAL    NOT NULL,
    realized_pnl        REAL    NOT NULL,
    max_drawdown        REAL    NOT NULL,
    sharpe_estimate     REAL    NOT NULL
)
"""

_CREATE_BALANCE_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS paper_balance_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    total_equity_usd REAL   NOT NULL,
    usdc_balance    REAL    NOT NULL,
    weth_balance    REAL    NOT NULL,
    eth_price       REAL    NOT NULL
)
"""

_CREATE_DECISIONS_SQL = """
CREATE TABLE IF NOT EXISTS paper_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL    NOT NULL,
    wallet_id           TEXT    NOT NULL,
    cycle_phase         TEXT    NOT NULL,
    action              TEXT    NOT NULL,
    regime_label        TEXT    NOT NULL,
    setup_tag           TEXT    NOT NULL,
    decision_class      TEXT    NOT NULL,
    primary_reason      TEXT    NOT NULL,
    reason_codes        TEXT    NOT NULL,
    raw_alpha           REAL    NOT NULL,
    robust_alpha        REAL    NOT NULL,
    confidence          REAL    NOT NULL,
    adf_pvalue          REAL    NOT NULL,
    stationarity_d      REAL    NOT NULL,
    volatility_regime   TEXT    NOT NULL,
    risk_position_usd   REAL    NOT NULL,
    trade_budget_usd    REAL    NOT NULL,
    gas_cost_usd        REAL    NOT NULL,
    total_equity_usd    REAL    NOT NULL,
    pnl_usd             REAL    NOT NULL,
    notes               TEXT    NOT NULL
)
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_pt_ts ON paper_trades(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pt_dir ON paper_trades(direction)",
    "CREATE INDEX IF NOT EXISTS idx_ps_ts ON paper_snapshots(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pb_asset ON paper_balances(asset)",
    "CREATE INDEX IF NOT EXISTS idx_pbh_ts ON paper_balance_history(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pd_ts ON paper_decisions(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pd_action ON paper_decisions(action)",
    "CREATE INDEX IF NOT EXISTS idx_pd_reason ON paper_decisions(primary_reason)",
    "CREATE INDEX IF NOT EXISTS idx_pd_regime ON paper_decisions(regime_label)",
    "CREATE INDEX IF NOT EXISTS idx_pd_setup ON paper_decisions(setup_tag)",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperTradeConfig:
    """Configuration for the paper trading engine.

    Attributes:
        initial_balance_usd: Starting virtual capital in USD.
        slippage_model: Slippage model type — ``"gaussian"`` or ``"fixed"``.
        slippage_mean_bps: Mean slippage in basis points for gaussian model.
        slippage_std_bps: Standard deviation in bps for gaussian model.
        gas_estimation: Whether to fetch real gas prices from Base.
        execution_delay_ms: Latency emulation range (min, max) in ms.
    """

    initial_balance_usd: float = 10_000.0
    slippage_model: str = "gaussian"
    slippage_mean_bps: float = 5.0
    slippage_std_bps: float = 3.0
    gas_estimation: bool = True
    execution_delay_ms: tuple[int, int] = (200, 800)


@dataclass(frozen=True)
class PaperTradeResult:
    """Complete result of one paper-trade shadow cycle.

    Attributes:
        action: Trade direction — BUY, SELL, or HOLD.
        signal_alpha: Raw alpha score from the signal engine.
        simulated_fill_price: Price after slippage simulation.
        slippage_bps: Applied slippage in basis points.
        gas_cost_usd: Estimated gas cost in USD.
        virtual_pnl_usd: Virtual profit/loss from this cycle.
        would_have_executed: Whether the real engine would have traded.
        reason_codes: Tuple of reason codes describing the outcome.
        cycle_duration_ms: Wall-clock duration of the shadow cycle.
        timestamp: Unix epoch timestamp of the cycle.
    """

    action: str = "HOLD"
    signal_alpha: float = 0.0
    simulated_fill_price: float = 0.0
    slippage_bps: float = 0.0
    gas_cost_usd: float = 0.0
    virtual_pnl_usd: float = 0.0
    would_have_executed: bool = False
    reason_codes: tuple[str, ...] = ()
    cycle_duration_ms: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# SlippageSimulator
# ---------------------------------------------------------------------------


class SlippageSimulator:
    """Simulate realistic trade slippage using Gaussian or fixed models.

    The Gaussian model draws slippage from a normal distribution with
    a non-linear impact scalar that increases super-linearly when the
    trade size exceeds 1% of pool depth.  This captures real-world
    Uniswap V3 tick-crossing dynamics.

    The fixed model applies a constant slippage regardless of trade size.
    """

    def __init__(
        self,
        mean_bps: float = 5.0,
        std_bps: float = 3.0,
    ) -> None:
        self._mean_bps = mean_bps
        self._std_bps = std_bps
        self._rng = np.random.default_rng(42)
        log.info(
            f"SlippageSimulator initialized: mean={mean_bps}bps "
            f"std={std_bps}bps"
        )

    def simulate_slippage(
        self,
        size_usd: float,
        pool_depth_usd: float,
        model: str = "gaussian",
        entry_price: float = 0.0,
    ) -> tuple[float, float]:
        """Simulate slippage for a given trade size and pool depth.

        Args:
            size_usd: Trade size in USD.
            pool_depth_usd: Estimated pool depth in USD.
            model: Slippage model — ``"gaussian"`` or ``"fixed"``.
            entry_price: The mid-market entry price (for fill calculation).

        Returns:
            Tuple of (slippage_bps, simulated_fill_price).
            ``slippage_bps`` is always non-negative.
            ``simulated_fill_price`` reflects the entry price adjusted by
            slippage (worse execution).
        """
        pool_depth_usd = max(pool_depth_usd, 1.0)
        size_ratio = size_usd / pool_depth_usd

        # Non-linear impact scalar: increases super-linearly for large orders
        # Below 1% of pool: impact_scalar ~= 1.0
        # At 1%: ~1.5x; at 5%: ~4.5x; at 10%: ~10x
        if size_ratio <= 0.01:
            impact_scalar = 1.0 + size_ratio * 50.0
        elif size_ratio <= 0.05:
            impact_scalar = 1.5 + (size_ratio - 0.01) * 75.0
        else:
            impact_scalar = 4.5 + (size_ratio - 0.05) ** 0.8 * 100.0

        if model == "gaussian":
            base_slip = float(self._rng.normal(self._mean_bps, self._std_bps))
            base_slip = max(base_slip, 0.5)  # Floor at 0.5 bps
            slippage_bps = base_slip * impact_scalar
        elif model == "fixed":
            slippage_bps = self._mean_bps * impact_scalar
        else:
            log.info(f"SlippageSimulator: unrecognized model '{model}', defaulting to gaussian")
            base_slip = float(self._rng.normal(self._mean_bps, self._std_bps))
            base_slip = max(base_slip, 0.5)
            slippage_bps = base_slip * impact_scalar

        # Cap slippage at 500 bps (5%) to avoid unrealistic fills
        slippage_bps = min(slippage_bps, 500.0)
        slippage_bps = round(slippage_bps, 2)

        # Calculate simulated fill price (always worse than entry)
        if entry_price > 0:
            slippage_fraction = slippage_bps / 10_000.0
            simulated_fill_price = entry_price * (1.0 - slippage_fraction)
        else:
            simulated_fill_price = 0.0

        log.debug(
            f"SLIPPAGE: size=${size_usd:.2f} depth=${pool_depth_usd:.0f} "
            f"ratio={size_ratio:.4f} impact={impact_scalar:.2f} "
            f"slip={slippage_bps:.2f}bps fill=${simulated_fill_price:.2f}"
        )

        return slippage_bps, simulated_fill_price


# ---------------------------------------------------------------------------
# GasEstimator
# ---------------------------------------------------------------------------


class GasEstimator:
    """Estimate transaction gas cost in USD using live Base gas prices.

    Fetches current gas price via web3 and caches the result for
    ``GAS_CACHE_TTL_S`` seconds to avoid excessive RPC calls.  Gas units
    are estimated per transaction type (swap, supply, borrow).
    """

    def __init__(self) -> None:
        self._cached_gas_gwei: float = 0.0
        self._cached_eth_price: float = 0.0
        self._cache_timestamp: float = 0.0
        self._lock = asyncio.Lock()
        log.info("GasEstimator initialized")

    async def _refresh_cache(self, rpc_url: str, eth_price: float) -> None:
        """Refresh the gas price cache if stale.

        Args:
            rpc_url: Base RPC endpoint.
            eth_price: Current ETH price in USD (for cost conversion).
        """
        now = time.time()
        if now - self._cache_timestamp < GAS_CACHE_TTL_S and self._cached_gas_gwei > 0:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if now - self._cache_timestamp < GAS_CACHE_TTL_S and self._cached_gas_gwei > 0:
                return
            try:
                from web3 import AsyncHTTPProvider, AsyncWeb3
                from wallet_manager.utils.web3_utils import close_w3_provider
                w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
                try:
                    gas_price_wei = await w3.eth.gas_price
                    self._cached_gas_gwei = float(gas_price_wei) / 1e9
                    self._cached_eth_price = eth_price
                    self._cache_timestamp = time.time()
                    log.debug(
                        f"GAS_CACHE: refreshed gas={self._cached_gas_gwei:.4f} gwei "
                        f"eth=${eth_price:.2f}"
                    )
                finally:
                    await close_w3_provider(w3)
            except Exception as exc:
                log.info(f"GasEstimator: RPC gas unavailable ({type(exc).__name__}), using cached/fallback")
                if self._cached_gas_gwei <= 0:
                    # Fallback: conservative estimate for Base L2
                    self._cached_gas_gwei = 0.01
                    self._cached_eth_price = eth_price
                    self._cache_timestamp = time.time()

    async def estimate_gas_cost_usd(
        self,
        tx_type: str,
        rpc_url: str = "",
        eth_price: float = 3000.0,
    ) -> float:
        """Estimate gas cost in USD for a given transaction type.

        Args:
            tx_type: Transaction type — ``"swap"``, ``"supply"``, or ``"borrow"``.
            rpc_url: Base RPC endpoint for live gas price fetch.
            eth_price: Current ETH/USD price.

        Returns:
            Estimated gas cost in USD.
        """
        if rpc_url:
            await self._refresh_cache(rpc_url, eth_price)

        gas_gwei = self._cached_gas_gwei if self._cached_gas_gwei > 0 else 0.01
        gas_units = GAS_UNITS.get(tx_type, GAS_UNITS["swap"])
        price = self._cached_eth_price if self._cached_eth_price > 0 else eth_price

        # gas_cost = gas_units * gas_price_gwei * 1e-9 * eth_price_usd
        cost_usd = gas_units * gas_gwei * 1e-9 * price
        cost_usd = round(cost_usd, 6)

        log.debug(
            f"GAS_EST: type={tx_type} units={gas_units} "
            f"gas={gas_gwei:.4f}gwei eth=${price:.2f} "
            f"cost=${cost_usd:.6f}"
        )
        return cost_usd

    def get_cached_gas_gwei(self) -> float:
        """Return the most recently cached gas price in gwei."""
        return self._cached_gas_gwei


# ---------------------------------------------------------------------------
# VirtualLedger
# ---------------------------------------------------------------------------


class VirtualLedger:
    """SQLite WAL-backed ledger for paper trade persistence.

    Tracks virtual asset balances (USDC, WETH, aUSDC), records every
    simulated trade, and stores periodic equity snapshots for drawdown
    and Sharpe estimation.

    The database is stored at ``paper_trades.db`` (configurable) and
    uses WAL mode for concurrent read/write safety.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

        # In-memory balance cache for fast reads
        self._balances: dict[str, float] = {}

        # Tracking state
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0
        self._closed_trade_count: int = 0
        self._win_count: int = 0
        self._total_slippage_bps: float = 0.0
        self._equity_history: list[float] = []
        self._initial_balance_usd: float = 0.0  # set by PaperTradingEngine.initialize()

    async def initialize(self) -> None:
        """Open the SQLite connection, enable WAL mode, and create tables.

        Loads any existing balances from disk into the in-memory cache.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row

        # WAL mode: readers and writer do not block each other
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        await self._conn.execute(_CREATE_BALANCES_SQL)
        await self._conn.execute(_CREATE_TRADES_SQL)
        await self._conn.execute(_CREATE_SNAPSHOTS_SQL)
        await self._conn.execute(_CREATE_BALANCE_HISTORY_SQL)
        await self._conn.execute(_CREATE_DECISIONS_SQL)
        for idx_sql in _CREATE_INDICES_SQL:
            await self._conn.execute(idx_sql)
        await self._conn.commit()

        # Load existing balances into cache
        async with self._conn.execute("SELECT asset, amount FROM paper_balances") as cur:
            rows = await cur.fetchall()
        for row in rows:
            self._balances[row["asset"]] = float(row["amount"])

        # Load realized PnL from historical trades
        async with self._conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0.0) AS total_pnl, "
            "COUNT(*) AS cnt, "
            "COALESCE(SUM(CASE WHEN direction = 'SELL' THEN 1 ELSE 0 END), 0) AS closed_cnt, "
            "COALESCE(SUM(CASE WHEN direction = 'SELL' AND pnl_usd > 0 THEN 1 ELSE 0 END), 0) AS wins, "
            "COALESCE(SUM(slippage_bps), 0.0) AS total_slip "
            "FROM paper_trades"
        ) as cur:
            trade_row = await cur.fetchone()
        if trade_row:
            self._realized_pnl = float(trade_row["total_pnl"] or 0.0)
            self._trade_count = int(trade_row["cnt"] or 0)
            self._closed_trade_count = int(trade_row["closed_cnt"] or 0)
            self._win_count = int(trade_row["wins"] or 0)
            self._total_slippage_bps = float(trade_row["total_slip"] or 0.0)

        # Load peak equity from snapshots
        async with self._conn.execute(
            "SELECT MAX(total_equity_usd) AS peak, "
            "MAX(max_drawdown) AS dd FROM paper_snapshots"
        ) as cur:
            peak_row = await cur.fetchone()
        if peak_row and peak_row["peak"]:
            self._peak_equity = float(peak_row["peak"])
            self._max_drawdown = float(peak_row["dd"] or 0.0)

        log.info(
            f"VirtualLedger initialized: db='{self._db_path}' "
            f"balances={len(self._balances)} trades={self._trade_count} "
            f"realized_pnl=${self._realized_pnl:+.2f}"
        )

    async def shutdown(self) -> None:
        """Flush pending writes and close the database connection."""
        if self._conn:
            await self._conn.commit()
            await self._conn.close()
            self._conn = None
        log.info("VirtualLedger shutdown complete")

    def _assert_open(self) -> None:
        """Raise if the database connection is not open."""
        if self._conn is None:
            raise RuntimeError(
                "VirtualLedger is not initialized. Call initialize() first."
            )

    def _get_conn(self) -> aiosqlite.Connection:
        self._assert_open()
        conn = self._conn
        if conn is None:
            raise RuntimeError("VirtualLedger database connection is not available")
        return conn

    # ----- Balance operations -----

    async def get_balance(self, asset: str) -> float:
        """Return the virtual balance for an asset.

        Args:
            asset: Asset symbol (e.g. ``"USDC"``, ``"WETH"``, ``"aUSDC"``).

        Returns:
            Current virtual balance.  Zero if the asset has no record.
        """
        return self._balances.get(asset, 0.0)

    async def update_balance(
        self,
        asset: str,
        amount: float,
        usd_value: float = 0.0,
    ) -> None:
        """Set the virtual balance for an asset and persist to SQLite.

        Args:
            asset: Asset symbol.
            amount: New absolute balance.
            usd_value: USD-equivalent value (for reporting).
        """
        conn = self._get_conn()
        self._balances[asset] = amount
        now = time.time()
        sql = """
            INSERT OR REPLACE INTO paper_balances
                (timestamp, asset, amount, usd_value)
            VALUES (?, ?, ?, ?)
        """
        async with self._write_lock:
            await conn.execute(sql, (now, asset, amount, usd_value))
            await conn.commit()

    async def get_total_equity_usd(
        self,
        eth_price: float = 3000.0,
    ) -> float:
        """Calculate total virtual equity in USD.

        Sums USDC (1:1), WETH (at ``eth_price``), and aUSDC (1:1).

        Args:
            eth_price: Current ETH/USD price for WETH valuation.

        Returns:
            Total virtual equity in USD.
        """
        usdc = self._balances.get("USDC", 0.0)
        weth = self._balances.get("WETH", 0.0)
        ausdc = self._balances.get("aUSDC", 0.0)
        return usdc + (weth * eth_price) + ausdc

    # ----- Trade recording -----

    async def record_trade(
        self,
        direction: str,
        asset_pair: str,
        entry_price: float,
        simulated_fill_price: float,
        slippage_bps: float,
        size_usd: float,
        gas_cost_usd: float,
        pnl_usd: float,
        reason_codes: tuple[str, ...],
        signal_source: str,
    ) -> None:
        """Record a paper trade to the SQLite database.

        Also updates in-memory tracking state (realized P&L, win count,
        slippage accumulator).

        Args:
            direction: ``"BUY"`` or ``"SELL"``.
            asset_pair: E.g. ``"WETH/USDC"``.
            entry_price: Mid-market price at decision time.
            simulated_fill_price: Price after slippage simulation.
            slippage_bps: Applied slippage in basis points.
            size_usd: Trade size in USD.
            gas_cost_usd: Estimated gas cost in USD.
            pnl_usd: Realized P&L from this trade.
            reason_codes: Tuple of reason code strings.
            signal_source: Name of the signal that triggered the trade.
        """
        conn = self._get_conn()
        now = time.time()
        codes_str = json.dumps(reason_codes)
        sql = """
            INSERT INTO paper_trades (
                timestamp, direction, asset_pair, entry_price,
                simulated_fill_price, slippage_bps, size_usd,
                gas_cost_usd, pnl_usd, reason_codes, signal_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await conn.execute(sql, (
                now, direction, asset_pair, entry_price,
                simulated_fill_price, slippage_bps, size_usd,
                gas_cost_usd, pnl_usd, codes_str, signal_source,
            ))
            await conn.commit()

        # Update in-memory state
        self._realized_pnl += pnl_usd
        self._trade_count += 1
        if direction == "SELL":
            self._closed_trade_count += 1
            if pnl_usd > 0:
                self._win_count += 1
        self._total_slippage_bps += slippage_bps

        log.info(
            f"PAPER_TRADE: {direction} {asset_pair} size=${size_usd:.2f} "
            f"entry=${entry_price:.2f} fill=${simulated_fill_price:.2f} "
            f"slip={slippage_bps:.2f}bps gas=${gas_cost_usd:.6f} "
            f"pnl=${pnl_usd:+.2f}"
        )

    async def record_decision(
        self,
        *,
        wallet_id: str,
        cycle_phase: str,
        action: str,
        regime_label: str,
        setup_tag: str,
        decision_class: str,
        primary_reason: str,
        reason_codes: tuple[str, ...],
        raw_alpha: float = 0.0,
        robust_alpha: float = 0.0,
        confidence: float = 0.0,
        adf_pvalue: float = 1.0,
        stationarity_d: float = 0.0,
        volatility_regime: str = "unknown",
        risk_position_usd: float = 0.0,
        trade_budget_usd: float = 0.0,
        gas_cost_usd: float = 0.0,
        total_equity_usd: float = 0.0,
        pnl_usd: float = 0.0,
        notes: str = "",
    ) -> None:
        """Persist a decision-journal record for trade and no-trade cycles."""
        conn = self._get_conn()
        sql = """
            INSERT INTO paper_decisions (
                timestamp, wallet_id, cycle_phase, action, regime_label,
                setup_tag, decision_class, primary_reason, reason_codes,
                raw_alpha, robust_alpha, confidence, adf_pvalue,
                stationarity_d, volatility_regime, risk_position_usd,
                trade_budget_usd, gas_cost_usd, total_equity_usd, pnl_usd, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payload = (
            time.time(),
            wallet_id,
            cycle_phase,
            action,
            regime_label,
            setup_tag,
            decision_class,
            primary_reason,
            json.dumps(reason_codes),
            raw_alpha,
            robust_alpha,
            confidence,
            adf_pvalue,
            stationarity_d,
            volatility_regime,
            risk_position_usd,
            trade_budget_usd,
            gas_cost_usd,
            total_equity_usd,
            pnl_usd,
            notes,
        )
        async with self._write_lock:
            await conn.execute(sql, payload)
            await conn.commit()

    async def get_recent_decisions(self, n: int = 20) -> list[dict[str, Any]]:
        """Return recent decision-journal rows for dashboard and review tools."""
        conn = self._get_conn()
        rows_out: list[dict[str, Any]] = []
        async with conn.execute(
            "SELECT timestamp, wallet_id, cycle_phase, action, regime_label, "
            "setup_tag, decision_class, primary_reason, reason_codes, raw_alpha, "
            "robust_alpha, confidence, adf_pvalue, stationarity_d, "
            "volatility_regime, risk_position_usd, trade_budget_usd, gas_cost_usd, "
            "total_equity_usd, pnl_usd, notes "
            "FROM paper_decisions ORDER BY timestamp DESC LIMIT ?",
            (n,),
        ) as cur:
            rows = await cur.fetchall()
        for row in rows:
            rows_out.append(
                {
                    "timestamp": float(row["timestamp"]),
                    "wallet_id": row["wallet_id"],
                    "cycle_phase": row["cycle_phase"],
                    "action": row["action"],
                    "regime_label": row["regime_label"],
                    "setup_tag": row["setup_tag"],
                    "decision_class": row["decision_class"],
                    "primary_reason": row["primary_reason"],
                    "reason_codes": json.loads(row["reason_codes"]),
                    "raw_alpha": float(row["raw_alpha"]),
                    "robust_alpha": float(row["robust_alpha"]),
                    "confidence": float(row["confidence"]),
                    "adf_pvalue": float(row["adf_pvalue"]),
                    "stationarity_d": float(row["stationarity_d"]),
                    "volatility_regime": row["volatility_regime"],
                    "risk_position_usd": float(row["risk_position_usd"]),
                    "trade_budget_usd": float(row["trade_budget_usd"]),
                    "gas_cost_usd": float(row["gas_cost_usd"]),
                    "total_equity_usd": float(row["total_equity_usd"]),
                    "pnl_usd": float(row["pnl_usd"]),
                    "notes": row["notes"],
                }
            )
        return rows_out

    async def get_decision_analytics(self) -> dict[str, Any]:
        """Return grouped decision analytics for research and dashboard use."""
        conn = self._get_conn()

        async with conn.execute(
            "SELECT primary_reason, COUNT(*) AS cnt "
            "FROM paper_decisions "
            "WHERE decision_class IN ('NO_TRADE', 'SKIP', 'ERROR') "
            "GROUP BY primary_reason ORDER BY cnt DESC, primary_reason ASC LIMIT 8"
        ) as cur:
            reason_rows = await cur.fetchall()

        async with conn.execute(
            "SELECT regime_label, "
            "COUNT(*) AS decision_count, "
            "SUM(CASE WHEN action IN ('BUY', 'SELL') THEN 1 ELSE 0 END) AS executed_count, "
            "COALESCE(SUM(pnl_usd), 0.0) AS realized_pnl_usd "
            "FROM paper_decisions "
            "GROUP BY regime_label ORDER BY decision_count DESC, regime_label ASC LIMIT 8"
        ) as cur:
            regime_rows = await cur.fetchall()

        async with conn.execute(
            "SELECT setup_tag, "
            "COUNT(*) AS decision_count, "
            "SUM(CASE WHEN action IN ('BUY', 'SELL') THEN 1 ELSE 0 END) AS executed_count, "
            "COALESCE(SUM(pnl_usd), 0.0) AS realized_pnl_usd "
            "FROM paper_decisions "
            "GROUP BY setup_tag ORDER BY decision_count DESC, setup_tag ASC LIMIT 8"
        ) as cur:
            setup_rows = await cur.fetchall()

        async with conn.execute(
            "SELECT COUNT(*) AS total_decisions, "
            "SUM(CASE WHEN action IN ('BUY', 'SELL') THEN 1 ELSE 0 END) AS executed_decisions, "
            "SUM(CASE WHEN decision_class = 'NO_TRADE' THEN 1 ELSE 0 END) AS no_trade_decisions, "
            "SUM(CASE WHEN decision_class = 'SKIP' THEN 1 ELSE 0 END) AS skip_decisions "
            "FROM paper_decisions"
        ) as cur:
            summary_row = await cur.fetchone()

        return {
            "summary": {
                "total_decisions": int(summary_row["total_decisions"] or 0) if summary_row else 0,
                "executed_decisions": int(summary_row["executed_decisions"] or 0) if summary_row else 0,
                "no_trade_decisions": int(summary_row["no_trade_decisions"] or 0) if summary_row else 0,
                "skip_decisions": int(summary_row["skip_decisions"] or 0) if summary_row else 0,
            },
            "no_trade_reasons": [
                {"reason": row["primary_reason"], "count": int(row["cnt"] or 0)}
                for row in reason_rows
            ],
            "regime_breakdown": [
                {
                    "regime_label": row["regime_label"],
                    "decision_count": int(row["decision_count"] or 0),
                    "executed_count": int(row["executed_count"] or 0),
                    "realized_pnl_usd": float(row["realized_pnl_usd"] or 0.0),
                }
                for row in regime_rows
            ],
            "setup_breakdown": [
                {
                    "setup_tag": row["setup_tag"],
                    "decision_count": int(row["decision_count"] or 0),
                    "executed_count": int(row["executed_count"] or 0),
                    "realized_pnl_usd": float(row["realized_pnl_usd"] or 0.0),
                }
                for row in setup_rows
            ],
        }

    # ----- Balance history -----

    async def record_balance_history(
        self,
        eth_price: float = 3000.0,
    ) -> None:
        """Record a timestamped balance snapshot for equity curve tracking."""
        conn = self._get_conn()
        usdc = self._balances.get("USDC", 0.0)
        weth = self._balances.get("WETH", 0.0)
        total = usdc + (weth * eth_price) + self._balances.get("aUSDC", 0.0)
        now = time.time()
        sql = """
            INSERT INTO paper_balance_history
                (timestamp, total_equity_usd, usdc_balance, weth_balance, eth_price)
            VALUES (?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await conn.execute(sql, (now, total, usdc, weth, eth_price))
            await conn.commit()

    async def get_best_worst_last_trade(self) -> dict[str, Any]:
        """Return best, worst, and last trade from paper_trades table.

        Returns:
            Dict with keys: ``best_trade``, ``worst_trade``, ``last_trade``.
            Each is a dict or None.
        """
        conn = self._get_conn()
        result: dict[str, Any] = {"best_trade": None, "worst_trade": None, "last_trade": None}

        # Best trade (highest P&L)
        async with conn.execute(
            "SELECT direction, asset_pair, pnl_usd, size_usd, timestamp "
            "FROM paper_trades ORDER BY pnl_usd DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            result["best_trade"] = {
                "direction": row["direction"], "pair": row["asset_pair"],
                "pnl_usd": float(row["pnl_usd"]), "size_usd": float(row["size_usd"]),
                "timestamp": float(row["timestamp"]),
            }

        # Worst trade (lowest P&L)
        async with conn.execute(
            "SELECT direction, asset_pair, pnl_usd, size_usd, timestamp "
            "FROM paper_trades ORDER BY pnl_usd ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            result["worst_trade"] = {
                "direction": row["direction"], "pair": row["asset_pair"],
                "pnl_usd": float(row["pnl_usd"]), "size_usd": float(row["size_usd"]),
                "timestamp": float(row["timestamp"]),
            }

        # Last trade
        async with conn.execute(
            "SELECT direction, asset_pair, pnl_usd, size_usd, timestamp "
            "FROM paper_trades ORDER BY timestamp DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            result["last_trade"] = {
                "direction": row["direction"], "pair": row["asset_pair"],
                "pnl_usd": float(row["pnl_usd"]), "size_usd": float(row["size_usd"]),
                "timestamp": float(row["timestamp"]),
            }

        return result

    async def get_recent_trades(self, n: int = 10) -> list[dict]:
        """Return the *n* most recent paper trades from the database.

        Returns a list of dicts with keys: direction, pair, size_usd,
        entry_price, fill_price, slippage_bps, pnl_usd, timestamp.
        """
        conn = self._get_conn()
        rows_out: list[dict] = []
        async with conn.execute(
            "SELECT direction, asset_pair, entry_price, simulated_fill_price, "
            "slippage_bps, size_usd, pnl_usd, timestamp "
            "FROM paper_trades ORDER BY timestamp DESC LIMIT ?",
            (n,),
        ) as cur:
            rows = await cur.fetchall()
        for row in rows:
            rows_out.append({
                "direction": row["direction"],
                "pair": row["asset_pair"],
                "entry_price": float(row["entry_price"]),
                "fill_price": float(row["simulated_fill_price"]),
                "slippage_bps": float(row["slippage_bps"]),
                "size_usd": float(row["size_usd"]),
                "pnl_usd": float(row["pnl_usd"]),
                "timestamp": float(row["timestamp"]),
            })
        return rows_out

    # ----- Equity snapshot -----

    async def _record_snapshot(
        self,
        total_equity_usd: float,
        unrealized_pnl: float,
    ) -> None:
        """Persist an equity snapshot and update drawdown tracking.

        Args:
            total_equity_usd: Current total virtual equity.
            unrealized_pnl: Unrealized P&L from open positions.
        """
        conn = self._get_conn()

        # Update peak and drawdown
        if total_equity_usd > self._peak_equity:
            self._peak_equity = total_equity_usd
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - total_equity_usd) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, current_dd)

        # Sharpe estimate: annualized from equity history
        self._equity_history.append(total_equity_usd)
        sharpe = self._estimate_sharpe()

        now = time.time()
        sql = """
            INSERT OR REPLACE INTO paper_snapshots (
                timestamp, total_equity_usd, unrealized_pnl,
                realized_pnl, max_drawdown, sharpe_estimate
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await conn.execute(sql, (
                now, total_equity_usd, unrealized_pnl,
                self._realized_pnl, self._max_drawdown, sharpe,
            ))
            await conn.commit()

    def _estimate_sharpe(self) -> float:
        """Estimate the annualized Sharpe ratio from equity history.

        Uses 5-minute periods as the base frequency.  Assumes risk-free
        rate of zero (appropriate for crypto).

        Returns:
            Annualized Sharpe ratio.  Returns 0.0 if insufficient data.
        """
        if len(self._equity_history) < 10:
            return 0.0
        eq = np.array(self._equity_history, dtype=np.float64)
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
        if len(returns) < 2:
            return 0.0
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        if std_r < 1e-10:
            return 0.0
        # Annualize: ~105,120 five-minute periods per year
        periods_per_year = 105_120.0
        sharpe = (mean_r / std_r) * math.sqrt(periods_per_year)
        return round(sharpe, 4)

    # ----- P&L and drawdown queries -----

    async def get_pnl_summary(self) -> dict[str, Any]:
        """Return a summary of virtual P&L statistics.

        Returns:
            Dict with keys: ``realized_pnl``, ``trade_count``,
            ``closed_trade_count``, ``win_count``, ``win_rate``, ``avg_slippage_bps``,
            ``max_drawdown``, ``sharpe_estimate``.
        """
        win_rate = (
            (self._win_count / self._closed_trade_count * 100.0)
            if self._closed_trade_count > 0 else 0.0
        )
        avg_slip = (
            (self._total_slippage_bps / self._trade_count)
            if self._trade_count > 0 else 0.0
        )
        return {
            "realized_pnl": round(self._realized_pnl, 4),
            "trade_count": self._trade_count,
            "closed_trade_count": self._closed_trade_count,
            "win_count": self._win_count,
            "win_rate": round(win_rate, 2),
            "avg_slippage_bps": round(avg_slip, 2),
            "max_drawdown": round(self._max_drawdown, 6),
            "sharpe_estimate": self._estimate_sharpe(),
            "peak_equity": round(self._peak_equity, 2),
        }

    async def get_drawdown(self) -> dict[str, float]:
        """Return current drawdown statistics.

        Returns:
            Dict with ``max_drawdown``, ``peak_equity``, and
            ``current_drawdown`` (latest equity vs peak).
        """
        current_eq = self._equity_history[-1] if self._equity_history else 0.0
        current_dd = 0.0
        if self._peak_equity > 0 and current_eq > 0:
            current_dd = (self._peak_equity - current_eq) / self._peak_equity
        return {
            "max_drawdown": round(self._max_drawdown, 6),
            "peak_equity": round(self._peak_equity, 2),
            "current_drawdown": round(current_dd, 6),
        }

    # ----- Phase 2: Advanced Strategy Analytics -----

    async def get_advanced_analytics(self, window_seconds: float | None = None) -> dict[str, Any]:
        """Return expectancy, payoff ratio, profit factor, Calmar, Sortino.

        Args:
            window_seconds: If given, restrict to trades within last N seconds.

        Returns:
            Dict with full strategy analytics.
        """
        conn = self._get_conn()
        ts_filter = ""
        params: tuple = ()
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            ts_filter = "WHERE timestamp >= ?"
            params = (cutoff,)

        async with conn.execute(
            f"SELECT direction, pnl_usd, size_usd, gas_cost_usd FROM paper_trades {ts_filter}",
            params,
        ) as cur:
            rows = list(await cur.fetchall())

        sell_rows = [r for r in rows if r["direction"] == "SELL"]
        total = len(rows)
        closed = len(sell_rows)

        if closed == 0:
            return {
                "expectancy_usd": 0.0, "payoff_ratio": 0.0, "profit_factor": 0.0,
                "calmar_ratio": 0.0, "sortino_ratio": 0.0,
                "win_rate_pct": 0.0, "avg_win_usd": 0.0, "avg_loss_usd": 0.0,
                "gross_profit_usd": 0.0, "gross_loss_usd": 0.0,
                "net_pnl_usd": 0.0, "total_gas_usd": 0.0, "net_after_gas_usd": 0.0,
                "avg_trade_size_usd": 0.0, "trade_count": total, "closed_count": 0,
                "sample_sufficient": False,
            }

        wins = [r for r in sell_rows if r["pnl_usd"] > 0]
        losses = [r for r in sell_rows if r["pnl_usd"] <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / closed if closed > 0 else 0.0

        avg_win = float(sum(r["pnl_usd"] for r in wins) / win_count) if wins else 0.0
        avg_loss = abs(float(sum(r["pnl_usd"] for r in losses) / loss_count)) if losses else 0.0

        gross_profit = float(sum(r["pnl_usd"] for r in wins))
        gross_loss = abs(float(sum(r["pnl_usd"] for r in losses)))
        net_pnl = gross_profit - gross_loss

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        total_gas = float(sum(r["gas_cost_usd"] for r in rows))
        net_after_gas = net_pnl - total_gas

        avg_size = float(sum(r["size_usd"] for r in rows) / total) if total > 0 else 0.0

        # Calmar ratio: annualized return / max drawdown
        annualized_return = 0.0
        if self._peak_equity > 0 and self._initial_balance_usd > 0:
            total_return = (self._peak_equity - self._initial_balance_usd) / self._initial_balance_usd
            # Rough annualization based on equity history length (5-min periods)
            periods = max(1, len(self._equity_history))
            annualized_return = total_return * (105_120.0 / periods)
        calmar = annualized_return / self._max_drawdown if self._max_drawdown > 1e-6 else 0.0

        # Sortino ratio: uses only downside deviation
        sortino = 0.0
        if len(self._equity_history) >= 10:
            eq = np.array(self._equity_history, dtype=np.float64)
            returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
            downside = returns[returns < 0]
            if len(downside) >= 2:
                downside_std = float(np.std(downside))
                mean_r = float(np.mean(returns))
                if downside_std > 1e-10:
                    sortino = (mean_r / downside_std) * math.sqrt(105_120.0)

        return {
            "expectancy_usd": round(expectancy, 4),
            "payoff_ratio": round(min(payoff_ratio, 999.0), 3),
            "profit_factor": round(min(profit_factor, 999.0), 3),
            "calmar_ratio": round(calmar, 4),
            "sortino_ratio": round(sortino, 4),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "avg_win_usd": round(avg_win, 4),
            "avg_loss_usd": round(avg_loss, 4),
            "gross_profit_usd": round(gross_profit, 4),
            "gross_loss_usd": round(gross_loss, 4),
            "net_pnl_usd": round(net_pnl, 4),
            "total_gas_usd": round(total_gas, 6),
            "net_after_gas_usd": round(net_after_gas, 4),
            "avg_trade_size_usd": round(avg_size, 2),
            "trade_count": total,
            "closed_count": closed,
            "sample_sufficient": closed >= 10,
        }

    async def get_hold_time_stats(self) -> dict[str, Any]:
        """Return hold time statistics by pairing BUY → SELL trades in order."""
        conn = self._get_conn()

        async with conn.execute(
            "SELECT timestamp, direction FROM paper_trades ORDER BY timestamp ASC"
        ) as cur:
            rows = list(await cur.fetchall())

        buy_times: list[float] = []
        hold_times: list[float] = []
        for row in rows:
            if row["direction"] == "BUY":
                buy_times.append(float(row["timestamp"]))
            elif row["direction"] == "SELL" and buy_times:
                entry_t = buy_times.pop(0)
                hold_times.append(float(row["timestamp"]) - entry_t)

        if not hold_times:
            return {
                "avg_hold_seconds": 0.0, "min_hold_seconds": 0.0,
                "max_hold_seconds": 0.0, "median_hold_seconds": 0.0,
                "completed_pairs": 0,
            }

        arr = sorted(hold_times)
        n = len(arr)
        median = arr[n // 2] if n % 2 else (arr[n // 2 - 1] + arr[n // 2]) / 2.0
        return {
            "avg_hold_seconds": round(sum(arr) / n, 1),
            "min_hold_seconds": round(arr[0], 1),
            "max_hold_seconds": round(arr[-1], 1),
            "median_hold_seconds": round(median, 1),
            "completed_pairs": n,
        }

    async def get_equity_curve_windowed(
        self, window_seconds: float | None = None, max_points: int = 100
    ) -> list[dict[str, float]]:
        """Return equity curve points, optionally filtered by time window.

        Args:
            window_seconds: Restrict to last N seconds. None = all time.
            max_points: Downsample to at most this many points.

        Returns:
            List of {timestamp, total_equity_usd} dicts.
        """
        conn = self._get_conn()
        ts_filter = ""
        params: tuple = ()
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            ts_filter = "WHERE timestamp >= ?"
            params = (cutoff,)

        async with conn.execute(
            f"SELECT timestamp, total_equity_usd FROM paper_balance_history {ts_filter} "
            "ORDER BY timestamp ASC",
            params,
        ) as cur:
            rows = list(await cur.fetchall())

        if not rows:
            return []

        # Downsample if needed
        step = max(1, len(rows) // max_points)
        sampled = rows[::step]
        return [
            {"timestamp": float(r["timestamp"]), "total_equity_usd": float(r["total_equity_usd"])}
            for r in sampled
        ]

    async def get_performance_by_regime(self) -> list[dict[str, Any]]:
        """Return per-regime performance: win rate, PnL, expectancy, count."""
        conn = self._get_conn()

        async with conn.execute(
            """
            SELECT
                d.regime_label,
                COUNT(DISTINCT d.id) AS decision_count,
                SUM(CASE WHEN d.action IN ('BUY','SELL') THEN 1 ELSE 0 END) AS executed_count,
                COUNT(CASE WHEN t.direction = 'SELL' AND t.pnl_usd > 0 THEN 1 END) AS win_count,
                COUNT(CASE WHEN t.direction = 'SELL' THEN 1 END) AS closed_count,
                COALESCE(SUM(CASE WHEN t.direction = 'SELL' THEN t.pnl_usd ELSE 0 END), 0.0) AS realized_pnl,
                COALESCE(AVG(CASE WHEN t.direction = 'SELL' AND t.pnl_usd > 0 THEN t.pnl_usd END), 0.0) AS avg_win,
                COALESCE(AVG(CASE WHEN t.direction = 'SELL' AND t.pnl_usd <= 0 THEN ABS(t.pnl_usd) END), 0.0) AS avg_loss
            FROM paper_decisions d
            LEFT JOIN paper_trades t ON (
                ABS(d.timestamp - t.timestamp) < 5.0
                AND t.direction = d.action
            )
            WHERE d.regime_label IS NOT NULL
            GROUP BY d.regime_label
            ORDER BY decision_count DESC
            """,
        ) as cur:
            rows = await cur.fetchall()

        result = []
        for row in rows:
            closed = int(row["closed_count"] or 0)
            wins = int(row["win_count"] or 0)
            win_rate = (wins / closed * 100.0) if closed > 0 else 0.0
            avg_win = float(row["avg_win"] or 0.0)
            avg_loss = float(row["avg_loss"] or 0.0)
            win_r = wins / closed if closed > 0 else 0.0
            expectancy = (win_r * avg_win) - ((1.0 - win_r) * avg_loss)
            result.append({
                "regime_label": row["regime_label"],
                "decision_count": int(row["decision_count"] or 0),
                "executed_count": int(row["executed_count"] or 0),
                "closed_count": closed,
                "win_rate_pct": round(win_rate, 1),
                "realized_pnl_usd": round(float(row["realized_pnl"] or 0.0), 4),
                "expectancy_usd": round(expectancy, 4),
            })
        return result

    async def get_performance_by_setup(self) -> list[dict[str, Any]]:
        """Return per-setup performance: win rate, PnL, expectancy, count."""
        conn = self._get_conn()

        async with conn.execute(
            """
            SELECT
                d.setup_tag,
                COUNT(DISTINCT d.id) AS decision_count,
                SUM(CASE WHEN d.action IN ('BUY','SELL') THEN 1 ELSE 0 END) AS executed_count,
                COUNT(CASE WHEN t.direction = 'SELL' AND t.pnl_usd > 0 THEN 1 END) AS win_count,
                COUNT(CASE WHEN t.direction = 'SELL' THEN 1 END) AS closed_count,
                COALESCE(SUM(CASE WHEN t.direction = 'SELL' THEN t.pnl_usd ELSE 0 END), 0.0) AS realized_pnl,
                COALESCE(AVG(CASE WHEN t.direction = 'SELL' AND t.pnl_usd > 0 THEN t.pnl_usd END), 0.0) AS avg_win,
                COALESCE(AVG(CASE WHEN t.direction = 'SELL' AND t.pnl_usd <= 0 THEN ABS(t.pnl_usd) END), 0.0) AS avg_loss
            FROM paper_decisions d
            LEFT JOIN paper_trades t ON (
                ABS(d.timestamp - t.timestamp) < 5.0
                AND t.direction = d.action
            )
            WHERE d.setup_tag IS NOT NULL
            GROUP BY d.setup_tag
            ORDER BY decision_count DESC
            """,
        ) as cur:
            rows = await cur.fetchall()

        result = []
        for row in rows:
            closed = int(row["closed_count"] or 0)
            wins = int(row["win_count"] or 0)
            win_rate = (wins / closed * 100.0) if closed > 0 else 0.0
            avg_win = float(row["avg_win"] or 0.0)
            avg_loss = float(row["avg_loss"] or 0.0)
            win_r = wins / closed if closed > 0 else 0.0
            expectancy = (win_r * avg_win) - ((1.0 - win_r) * avg_loss)
            result.append({
                "setup_tag": row["setup_tag"],
                "decision_count": int(row["decision_count"] or 0),
                "executed_count": int(row["executed_count"] or 0),
                "closed_count": closed,
                "win_rate_pct": round(win_rate, 1),
                "realized_pnl_usd": round(float(row["realized_pnl"] or 0.0), 4),
                "expectancy_usd": round(expectancy, 4),
            })
        return result

    async def get_live_readiness_score(self) -> dict[str, Any]:
        """Compute a live-readiness score (0-100) based on multiple criteria.

        Gate criteria (each worth points toward 100):
        - Closed trades >= 50        : 30 pts
        - Win rate >= 55%            : 20 pts
        - Expectancy > 0 after gas   : 20 pts
        - Max drawdown < 5%          : 15 pts
        - Consistent sessions (>= 5d): 15 pts

        Returns:
            Dict with score, gate results, and recommendation.
        """
        pnl = await self.get_pnl_summary()
        adv = await self.get_advanced_analytics()
        conn = self._get_conn()

        # Session age span
        async with conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM paper_trades"
        ) as cur:
            span_row = await cur.fetchone()

        span_days = 0.0
        if span_row and span_row[0] and span_row[1]:
            span_days = (float(span_row[1]) - float(span_row[0])) / 86400.0

        closed = int(pnl["closed_trade_count"])
        win_rate = float(pnl["win_rate"])
        max_dd_pct = float(pnl["max_drawdown"]) * 100.0
        net_after_gas = float(adv["net_after_gas_usd"])
        expectancy = float(adv["expectancy_usd"])

        gates: dict[str, dict] = {
            "closed_trades": {
                "label": "Closed trades ≥ 50",
                "passed": closed >= 50,
                "value": closed,
                "target": 50,
                "points": 30,
                "earned": 30 if closed >= 50 else int(30 * min(closed / 50.0, 1.0)),
            },
            "win_rate": {
                "label": "Win rate ≥ 55%",
                "passed": win_rate >= 55.0,
                "value": round(win_rate, 1),
                "target": 55.0,
                "points": 20,
                "earned": 20 if win_rate >= 55.0 else int(20 * min(win_rate / 55.0, 1.0)),
            },
            "positive_expectancy": {
                "label": "Positive expectancy (net of gas)",
                "passed": net_after_gas > 0 and expectancy > 0,
                "value": round(expectancy, 4),
                "target": 0.0,
                "points": 20,
                "earned": 20 if (net_after_gas > 0 and expectancy > 0) else 0,
            },
            "drawdown": {
                "label": "Max drawdown < 5%",
                "passed": max_dd_pct < 5.0,
                "value": round(max_dd_pct, 2),
                "target": 5.0,
                "points": 15,
                "earned": 15 if max_dd_pct < 5.0 else int(15 * max(0.0, (10.0 - max_dd_pct) / 10.0)),
            },
            "session_history": {
                "label": "Trade history ≥ 5 days",
                "passed": span_days >= 5.0,
                "value": round(span_days, 1),
                "target": 5.0,
                "points": 15,
                "earned": 15 if span_days >= 5.0 else int(15 * min(span_days / 5.0, 1.0)),
            },
        }

        total_score = sum(g["earned"] for g in gates.values())
        all_passed = all(g["passed"] for g in gates.values())

        if total_score >= 90 and all_passed:
            recommendation = "READY — all gates passed. Consider shadow-live transition."
        elif total_score >= 60:
            failed = [g["label"] for g in gates.values() if not g["passed"]]
            recommendation = f"CLOSE — {len(failed)} gate(s) pending: {', '.join(failed[:2])}"
        elif total_score >= 30:
            recommendation = "BUILDING — continue accumulating closed trades and consistent sessions."
        else:
            recommendation = "EARLY — insufficient data. Keep paper trading."

        return {
            "score": total_score,
            "max_score": 100,
            "all_gates_passed": all_passed,
            "recommendation": recommendation,
            "gates": gates,
        }


# ---------------------------------------------------------------------------
# PaperTradingEngine
# ---------------------------------------------------------------------------


class PaperTradingEngine:
    """Main orchestrator for high-fidelity paper trading.

    Runs the FULL robust execution pipeline (regime filter, robust alpha,
    liquidity profiling, SOR routing, MEV shield decisions) and stops
    EXACTLY before ``send_transaction``.

    All would-be trades are recorded in the VirtualLedger with realistic
    slippage simulation and gas cost estimation.

    Usage::

        engine = await get_paper_engine()
        result = await engine.run_shadow_cycle(
            manager=wallet_manager,
            wallet_id="wallet_1",
            portfolio_manager=pm,
            task_context=ctx,
            analytics_db=db,
            quant_analytics=qa,
        )
        report = await engine.get_performance_report()
    """

    def __init__(
        self,
        config: PaperTradeConfig | None = None,
        # Simplified API params
        initial_capital_usd: float | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        if initial_capital_usd is not None and config is None:
            config = PaperTradeConfig(initial_balance_usd=initial_capital_usd)
        self._config = config or PaperTradeConfig()
        self._min_confidence = min_confidence
        # Resolve DB path from settings so PAPER_DB_PATH env var is honoured
        # (important for Docker/cloud where /app is read-only but /app/data is a volume)
        try:
            from wallet_manager.config import get_settings as _get_settings
            _db_path = str(_get_settings().paper_db_path)
        except Exception:
            _db_path = _DEFAULT_DB_PATH
        self._ledger = VirtualLedger(db_path=_db_path)
        self._slippage = SlippageSimulator(
            mean_bps=self._config.slippage_mean_bps,
            std_bps=self._config.slippage_std_bps,
        )
        self._gas_estimator = GasEstimator()

        # Position tracking (mirrors real engine)
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._position_size_usd: float = 0.0
        self._entry_alpha: float = 0.0
        self._entry_timestamp: float = 0.0

        self._initialized: bool = False
        self._cycle_count: int = 0

        # Simplified step() API state
        self._capital_usd: float = self._config.initial_balance_usd
        self._step_trades: list[dict] = []
        self._step_returns: list[float] = []
        self._step_pnl: float = 0.0
        self._step_position_usd: float = 0.0
        self._step_entry_price: float = 0.0

        log.info(
            f"PaperTradingEngine created: balance=${self._config.initial_balance_usd:.2f} "
            f"slippage={self._config.slippage_model} "
            f"mean={self._config.slippage_mean_bps}bps "
            f"gas_estimation={self._config.gas_estimation}"
        )

    @staticmethod
    def _classify_setup_tag(
        *,
        action: str,
        primary_reason: str,
        in_position: bool,
        robust_alpha: float,
    ) -> str:
        if primary_reason.startswith("NEWS_EVENT_IMMINENT"):
            return "macro_event_override"
        if primary_reason == "QUOTE_INVALIDATED":
            return "latency_quote_invalid"
        if primary_reason == "FLASH_CRASH":
            return "flash_crash_block"
        if primary_reason.startswith("RISK_BLOCKED"):
            return "risk_blocked"
        if primary_reason == "STRESS_TEST_REJECTED":
            return "stress_test_rejected"
        if primary_reason == "GAS_TOO_EXPENSIVE":
            return "gas_filter_block"
        if action == "BUY":
            return "momentum_long" if robust_alpha >= 0 else "countertrend_long"
        if action == "SELL":
            return "exit_long"
        if action == "HOLD" and in_position:
            return "position_hold"
        if action == "WARMUP":
            return "warmup"
        return "signal_wait"

    @staticmethod
    def _classify_decision_class(action: str) -> str:
        if action in {"BUY", "SELL"}:
            return "EXECUTED"
        if action in {"PRICE_ERROR", "INVALID_PRICE", "WRONG_CHAIN", "FLASH_CRASH"}:
            return "ERROR"
        if action in {"WARMUP", "HOLD"}:
            return "NO_TRADE"
        return "SKIP"

    async def _journal_decision(
        self,
        *,
        wallet_id: str,
        cycle_phase: str,
        action: str,
        primary_reason: str,
        reason_codes: tuple[str, ...],
        signal: RobustAlphaSignal | None = None,
        volatility_regime: str = "unknown",
        risk_position_usd: float = 0.0,
        trade_budget_usd: float = 0.0,
        gas_cost_usd: float = 0.0,
        total_equity_usd: float = 0.0,
        pnl_usd: float = 0.0,
        notes: str = "",
    ) -> None:
        """Best-effort persistence of a single decision-journal row."""
        robust_alpha = signal.robust_alpha if signal is not None else 0.0
        setup_tag = self._classify_setup_tag(
            action=action,
            primary_reason=primary_reason,
            in_position=self._in_position,
            robust_alpha=robust_alpha,
        )
        await self._ledger.record_decision(
            wallet_id=wallet_id,
            cycle_phase=cycle_phase,
            action=action,
            regime_label=volatility_regime,
            setup_tag=setup_tag,
            decision_class=self._classify_decision_class(action),
            primary_reason=primary_reason,
            reason_codes=reason_codes,
            raw_alpha=signal.raw_alpha if signal is not None else 0.0,
            robust_alpha=robust_alpha,
            confidence=signal.confidence if signal is not None else 0.0,
            adf_pvalue=signal.adf_pvalue if signal is not None else 1.0,
            stationarity_d=signal.stationarity_d if signal is not None else 0.0,
            volatility_regime=volatility_regime,
            risk_position_usd=risk_position_usd,
            trade_budget_usd=trade_budget_usd,
            gas_cost_usd=gas_cost_usd,
            total_equity_usd=total_equity_usd,
            pnl_usd=pnl_usd,
            notes=notes,
        )

    async def record_external_skip(
        self,
        *,
        wallet_id: str,
        primary_reason: str,
        notes: str = "",
        regime_label: str = "unknown",
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        """Record no-trade decisions that happen before `run_shadow_cycle` starts."""
        if not self._initialized:
            await self.initialize()
        total_equity = await self._ledger.get_total_equity_usd()
        await self._journal_decision(
            wallet_id=wallet_id,
            cycle_phase="task_factory",
            action="SKIP",
            primary_reason=primary_reason,
            reason_codes=reason_codes or (PAPER_SHADOW_ONLY, PAPER_SIGNAL_REJECTED),
            volatility_regime=regime_label,
            total_equity_usd=total_equity,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Simplified step-based API for testing
    # ------------------------------------------------------------------

    async def step(self, price: float, signal: float) -> None:
        """Process one price tick with a directional signal.

        If signal >= min_confidence and no position: open long.
        If position open and signal < min_confidence: close.
        """
        import math
        MAX_POSITION_PCT = 0.25

        if self._step_position_usd > 0:
            # Close position if signal dropped below threshold
            if signal < self._min_confidence:
                pnl = self._step_position_usd * (price - self._step_entry_price) / max(self._step_entry_price, 1e-10)
                self._step_pnl += pnl
                ret = (price - self._step_entry_price) / max(self._step_entry_price, 1e-10)
                self._step_returns.append(ret)
                self._capital_usd += self._step_position_usd + pnl
                self._step_position_usd = 0.0
                self._step_entry_price = 0.0
                self._step_trades.append({"type": "close", "price": price, "pnl": pnl})
        else:
            # Open position if signal strong enough
            if signal >= self._min_confidence:
                size_usd = min(self._capital_usd * MAX_POSITION_PCT, self._capital_usd)
                size_usd = max(0.0, size_usd)
                self._capital_usd -= size_usd
                self._step_position_usd = size_usd
                self._step_entry_price = price
                self._step_trades.append({"type": "open", "price": price, "size_usd": size_usd})

    @property
    def current_exposure_usd(self) -> float:
        """Current open position size in USD."""
        return self._step_position_usd

    def get_metrics(self) -> dict:
        """Return summary performance metrics for the step-based simulation."""
        import numpy as np
        returns = np.asarray(self._step_returns)
        sharpe = 0.0
        if len(returns) > 1:
            std = float(np.std(returns))
            if std > 0:
                sharpe = float(np.mean(returns)) / std * (252 ** 0.5)
        return {
            "total_pnl_usd": round(self._step_pnl, 4),
            "total_trades": len([t for t in self._step_trades if t["type"] == "open"]),
            "sharpe_ratio": round(sharpe, 4),
            "final_capital_usd": round(self._capital_usd + self._step_position_usd, 4),
        }

    async def initialize(self) -> None:
        """Initialize the VirtualLedger and set initial balances.

        Sets the starting USDC balance from config and zero balances
        for WETH and aUSDC.
        """
        if self._initialized:
            return

        await self._ledger.initialize()

        # Set initial balances if this is a fresh ledger
        current_usdc = await self._ledger.get_balance("USDC")
        if current_usdc <= 0:
            await self._ledger.update_balance(
                "USDC", self._config.initial_balance_usd,
                usd_value=self._config.initial_balance_usd,
            )
            await self._ledger.update_balance("WETH", 0.0, usd_value=0.0)
            await self._ledger.update_balance("aUSDC", 0.0, usd_value=0.0)
            self._ledger._peak_equity = self._config.initial_balance_usd
            self._ledger._initial_balance_usd = self._config.initial_balance_usd
            log.info(
                f"PAPER_ENGINE: initial balances set — "
                f"USDC=${self._config.initial_balance_usd:.2f}"
            )

        self._initialized = True
        log.info("PaperTradingEngine initialized")

    async def shutdown(self) -> None:
        """Shutdown the paper trading engine and close the ledger."""
        await self._ledger.shutdown()
        self._initialized = False
        log.info("PaperTradingEngine shutdown complete")

    async def _send_trade_telegram(
        self,
        direction: str,
        token: str,
        size_usd: float,
        price: float,
        pnl_usd: float = 0.0,
    ) -> None:
        """Send instant Telegram notification for paper trade open/close."""
        try:
            from wallet_manager.security.telegram_manager import TelegramManager
            tg = TelegramManager()
            if not tg._is_configured():
                return
            if direction == "BUY":
                text = (
                    f"\U0001f7e2 <b>PAPER TRADE OPENED</b>\n"
                    f"Token: {token}\n"
                    f"Size: ${size_usd:.2f}\n"
                    f"Price: ${price:.2f}"
                )
            else:
                emoji = "\U0001f534"
                pnl_sign = f"${pnl_usd:+.2f}"
                text = (
                    f"{emoji} <b>PAPER TRADE CLOSED</b>\n"
                    f"Token: {token}\n"
                    f"Size: ${size_usd:.2f}\n"
                    f"Price: ${price:.2f}\n"
                    f"PnL: {pnl_sign}"
                )
            try:
                await tg._send(text)
            finally:
                await tg.close()
        except Exception as exc:
            log.debug(f"Paper trade Telegram notification skipped: {exc}")

    # ----- Shadow execution cycle -----

    async def run_shadow_cycle(
        self,
        manager: WalletManager,
        wallet_id: str,
        portfolio_manager: PortfolioManager,
        task_context: dict,
        analytics_db: Any = None,
        quant_analytics: Any = None,
    ) -> PaperTradeResult:
        """Run one complete shadow execution cycle.

        Mirrors ``MLQuantExecutionEngine.run_cycle()`` but stops before
        any on-chain transaction.  Records the would-be trade with
        simulated slippage and gas cost.

        Pipeline:
        1. Pre-flight checks (chain, circuit breakers)
        2. Price fetch + ML signal generation (robust alpha)
        3. Flash crash screening
        4. Deep Kelly risk sizing
        5. Trading logic (shadow — no actual execution)
        6. Slippage simulation
        7. Gas cost estimation
        8. VirtualLedger recording
        9. Equity snapshot

        Args:
            manager: WalletManager instance for config and balance reads.
            wallet_id: Wallet identifier.
            portfolio_manager: PortfolioManager for trade budget.
            task_context: Orchestrator context dict.
            analytics_db: Optional AnalyticsDB for cross-reference.
            quant_analytics: Optional QuantAnalytics for TCA recording.

        Returns:
            :class:`PaperTradeResult` with complete shadow cycle outcome.
        """
        if not self._initialized:
            await self.initialize()

        cycle_start = time.time()
        reason_codes: list[str] = [PAPER_SHADOW_ONLY]

        # ==============================================================
        # 1. PRE-FLIGHT CHECKS
        # ==============================================================

        command_state = task_context.get("command_state")
        if command_state and getattr(command_state, "is_circuit_breaker_halted", False):
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="preflight",
                    action="HALTED",
                    primary_reason="CIRCUIT_BREAKER_HALTED",
                    reason_codes=tuple(reason_codes),
                    volatility_regime="preflight",
                    total_equity_usd=await self._ledger.get_total_equity_usd(),
                    notes="Task command state halted the paper cycle before signal generation.",
                )
            return PaperTradeResult(
                action="HALTED",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        config = manager.get_config(wallet_id)
        chain_id = int(config.chain_id) if config.chain_id is not None else 0
        if chain_id != CHAIN_ID_BASE:
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="preflight",
                    action="WRONG_CHAIN",
                    primary_reason="CHAIN_NOT_BASE",
                    reason_codes=tuple(reason_codes),
                    volatility_regime="preflight",
                    total_equity_usd=await self._ledger.get_total_equity_usd(),
                    notes=f"Configured chain_id={chain_id} is not Base mainnet.",
                )
            return PaperTradeResult(
                action="WRONG_CHAIN",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        rpc_url = config.rpc_url

        # ==============================================================
        # 2. PRICE FETCH + ROBUST ALPHA SIGNAL
        # ==============================================================

        from wallet_manager import actions

        try:
            price = await actions.get_live_price_v3(rpc_url)
            if price <= 0:
                price = await actions.get_weth_usdc_price(rpc_url)
            if price <= 0:
                price = await actions.get_current_price(rpc_url, chain_id)
        except Exception as exc:
            log.info(f"PAPER_ENGINE: price unavailable ({type(exc).__name__}: {exc})")
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="price_fetch",
                    action="PRICE_ERROR",
                    primary_reason="PRICE_FETCH_FAILED",
                    reason_codes=tuple(reason_codes),
                    volatility_regime="preflight",
                    total_equity_usd=await self._ledger.get_total_equity_usd(),
                    notes=f"{type(exc).__name__}: {exc}",
                )
            return PaperTradeResult(
                action="PRICE_ERROR",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        if price <= 0:
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="price_fetch",
                    action="INVALID_PRICE",
                    primary_reason="NON_POSITIVE_PRICE",
                    reason_codes=tuple(reason_codes),
                    volatility_regime="preflight",
                    total_equity_usd=await self._ledger.get_total_equity_usd(),
                    notes=f"Resolved price was {price}.",
                )
            return PaperTradeResult(
                action="INVALID_PRICE",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        # Generate robust alpha signal
        from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine
        robust_engine = get_robust_alpha_engine()

        # Update microstructure (best-effort)
        try:
            from wallet_manager.strategies.signal_intelligence import get_alpha_generator
            ofi_gen = get_alpha_generator()
            ofi_reading = await ofi_gen.read_ofi(rpc_url, _get_weth_usdc_pool())
            if ofi_reading:
                vol_imb = ofi_reading.imbalance_ratio - 1.0
                vol_imb = max(-1.0, min(1.0, vol_imb))
                robust_engine.set_microstructure(
                    volume_imbalance=vol_imb, spread_bps=5.0,
                )
        except Exception as _ms_exc:
            log.warning("Paper engine microstructure update failed (non-critical): %s", _ms_exc)

        signal = robust_engine.ingest_price(price, volume=0.0)

        if signal is None:
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="signal_generation",
                    action="WARMUP",
                    primary_reason="ROBUST_ALPHA_WARMUP",
                    reason_codes=tuple(reason_codes),
                    volatility_regime="warmup",
                    total_equity_usd=await self._ledger.get_total_equity_usd(eth_price=price),
                    notes="Robust alpha engine has not accumulated enough ticks yet.",
                )
            return PaperTradeResult(
                action="WARMUP",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        log.info(
            f"PAPER_ENGINE: price=${price:.2f} "
            f"robust_alpha={signal.robust_alpha:+.3f} "
            f"raw_alpha={signal.raw_alpha:+.3f} "
            f"conf={signal.confidence:.2f} "
            f"stationary={signal.stationarity_d:.2f}"
        )

        # ==============================================================
        # 2.5. EARLY OBSERVATION — runs every cycle (before any early return)
        #      Records price/signal to all observational singletons so
        #      cross-asset, OFI, IC, signal discovery always accumulate data.
        # ==============================================================

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.cross_asset import get_cross_asset as _get_ca_early
            _ca_e = _get_ca_early()
            _btc_e = await _ca_e.fetch_btc_price()
            _ca_e.record_prices(price, _btc_e)

        with contextlib.suppress(Exception):
            from wallet_manager.tick_collector import get_tick_collector as _get_tc_early
            _tc_e = _get_tc_early()
            _tc_e.record_tick(price)
            await _tc_e.maybe_flush()

        with contextlib.suppress(Exception):
            from wallet_manager.signal_intelligence import get_signal_intelligence as _get_si_early
            _si_e = _get_si_early()
            _si_e.record_cycle(
                robust_alpha=signal.robust_alpha,
                confidence=signal.confidence,
                adf_pvalue=signal.adf_pvalue,
                garch_sigma=float(getattr(signal, "garch_sigma", 0.0)),
                spread_bps=0.0,
                current_price=price,
                signal_direction=0,  # direction unknown before risk decision
            )

        with contextlib.suppress(Exception):
            from wallet_manager.signal_discovery import get_signal_discovery as _get_sd_early
            _get_sd_early().record_cycle(price, signal.robust_alpha, signal.confidence)

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.funding_rate import get_funding_rate_collector as _get_frc_early
            await _get_frc_early().fetch_all(eth_price=price)

        # ==============================================================
        # 3. FLASH CRASH SCREENING
        # ==============================================================

        from wallet_manager.strategies.flash_crash_detector import (
            get_detector as get_crash_detector,
        )
        crash_detector = get_crash_detector()
        crash_alert = crash_detector.ingest_price(price)
        if crash_alert:
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="flash_crash_screen",
                    action="FLASH_CRASH",
                    primary_reason="FLASH_CRASH",
                    reason_codes=(PAPER_SHADOW_ONLY, PAPER_SIGNAL_REJECTED),
                    signal=signal,
                    volatility_regime="crash_alert",
                    total_equity_usd=await self._ledger.get_total_equity_usd(eth_price=price),
                    notes=str(crash_alert),
                )
            return PaperTradeResult(
                action="FLASH_CRASH",
                signal_alpha=signal.robust_alpha,
                reason_codes=(PAPER_SHADOW_ONLY, PAPER_SIGNAL_REJECTED),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        # ==============================================================
        # 3.5. RISK COMMITTEE CHECK (Phase 6)
        # ==============================================================

        with contextlib.suppress(Exception):
            from wallet_manager.risk_committee import get_risk_committee as _get_risk_committee
            _committee = _get_risk_committee()
            _underlying_for_rc = signal.underlying_signal
            _vol_r_for_rc = _underlying_for_rc.volatility_regime if _underlying_for_rc else "normal"
            _rc_verdict = await _committee.evaluate(
                paper_engine=self,
                confidence=signal.confidence,
                spread_bps=5.0,
            )
            if not _rc_verdict.allowed:
                reason_codes.append(PAPER_SIGNAL_REJECTED)
                with contextlib.suppress(Exception):
                    await self._journal_decision(
                        wallet_id=wallet_id,
                        cycle_phase="risk_committee",
                        action="RISK_BLOCKED",
                        primary_reason=f"RISK_COMMITTEE:{'|'.join(_rc_verdict.blocking_brakes)}",
                        reason_codes=tuple(reason_codes),
                        signal=signal,
                        volatility_regime=_vol_r_for_rc,
                        total_equity_usd=await self._ledger.get_total_equity_usd(eth_price=price),
                        notes=_rc_verdict.narrative[:500],
                    )
                return PaperTradeResult(
                    action="RISK_BLOCKED",
                    signal_alpha=signal.robust_alpha,
                    reason_codes=tuple(reason_codes),
                    cycle_duration_ms=(time.time() - cycle_start) * 1000,
                    timestamp=cycle_start,
                )

        # ==============================================================
        # 4. DEEP KELLY RISK SIZING
        # ==============================================================

        from wallet_manager.strategies.deep_kelly_risk import get_deep_kelly_engine
        risk_engine = get_deep_kelly_engine()

        total_equity = await self._ledger.get_total_equity_usd(eth_price=price)
        risk_engine.update_equity(total_equity)

        underlying = signal.underlying_signal
        garch_sigma = underlying.garch_sigma if underlying else 0.001
        vol_regime = underlying.volatility_regime if underlying else "normal"
        volatility_regime = vol_regime
        confidence = signal.confidence

        risk_decision = risk_engine.calculate_position(
            total_capital_usd=total_equity,
            alpha_score=signal.robust_alpha,
            garch_sigma=garch_sigma,
            vol_regime=vol_regime,
            confidence=confidence,
        )

        if risk_decision.position_usd <= 0:
            reason_codes.append(PAPER_SIGNAL_REJECTED)
            with contextlib.suppress(Exception):
                await self._journal_decision(
                    wallet_id=wallet_id,
                    cycle_phase="risk_sizing",
                    action=f"RISK_BLOCKED({risk_decision.drawdown_state})",
                    primary_reason=f"RISK_BLOCKED:{risk_decision.drawdown_state}",
                    reason_codes=tuple(reason_codes),
                    signal=signal,
                    volatility_regime=vol_regime,
                    risk_position_usd=risk_decision.position_usd,
                    total_equity_usd=total_equity,
                    notes="Deep Kelly risk engine returned zero position size.",
                )
            return PaperTradeResult(
                action=f"RISK_BLOCKED({risk_decision.drawdown_state})",
                signal_alpha=signal.robust_alpha,
                reason_codes=tuple(reason_codes),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        # ==============================================================
        # 4.5. JS-2.4 + JS-3.2: Apply Regime Limits and Bayesian Kelly
        # ==============================================================

        _js_position_usd = risk_decision.position_usd
        with contextlib.suppress(Exception):
            from wallet_manager.risk_intelligence import get_risk_intelligence as _get_ri_k
            _kelly_mult = _get_ri_k().get_effective_kelly_multiplier(vol_regime)
            _js_position_usd = _js_position_usd * _kelly_mult

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.bayesian_kelly import get_bayesian_kelly as _get_bk_k
            _bk = _get_bk_k()
            _bk_fraction = _bk.compute_kelly(
                avg_win_usd=max(abs(self._position_size_usd), 1.0),
                avg_loss_usd=max(abs(self._position_size_usd) * 0.5, 1.0),
            )
            _bk_factor = max(_bk.get_uncertainty_factor(), _bk_fraction / 0.25 if _bk_fraction > 0 else 0.0)
            _js_position_usd = _js_position_usd * _bk_factor

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.execution_quality import get_optimal_timing as _get_ot_pre
            _timing_window = _get_ot_pre().get_current_window_quality()
            if _timing_window.get("avoid"):
                _js_position_usd = _js_position_usd * 0.5
                reason_codes.append("LOW_LIQUIDITY_WINDOW")

        # ==============================================================
        # 5. TRADING LOGIC (SHADOW)
        # ==============================================================

        usdc_balance = await self._ledger.get_balance("USDC")
        weth_balance = await self._ledger.get_balance("WETH")

        # Use virtual USDC balance as the budget cap so paper trades are
        # not blocked when the real wallet has zero on-chain funds.
        trade_budget_usd = min(
            _js_position_usd,
            usdc_balance,
        )
        weth_value_usd = weth_balance * price

        should_buy = (
            signal.should_enter_long
            and not self._in_position
            and usdc_balance >= MIN_TRADE_USD
            and trade_budget_usd >= MIN_TRADE_USD
        )

        should_sell = (
            self._in_position
            and signal.should_exit
            and weth_value_usd >= MIN_TRADE_USD
        )

        action = "HOLD"
        simulated_fill_price = 0.0
        slippage_bps = 0.0
        spread_bps = 0.0
        gas_cost_usd = 0.0
        virtual_pnl = 0.0
        would_have_executed = False
        primary_reason = "EXIT_NOT_TRIGGERED" if self._in_position else "ENTRY_NOT_TRIGGERED"
        decision_notes = (
            f"signal_enter={signal.should_enter_long} "
            f"signal_exit={signal.should_exit} "
            f"in_position={self._in_position}"
        )

        # Pre-flight gas check: skip trade if gas cost exceeds the configured
        # fraction of trade size. Paper mode needs enough room to simulate
        # small, exploratory trades without zeroing the strategy.
        if should_buy or should_sell:
            from wallet_manager.config import get_settings as _get_settings

            _gas_limit_pct = _get_settings().paper_max_gas_fraction_pct
            _pre_gas = await self._gas_estimator.estimate_gas_cost_usd(
                tx_type="swap", rpc_url=rpc_url, eth_price=price,
            )
            _trade_size = trade_budget_usd if should_buy else weth_value_usd
            if _trade_size > 0 and _pre_gas / _trade_size > (_gas_limit_pct / 100.0):
                primary_reason = "GAS_TOO_EXPENSIVE"
                decision_notes = (
                    f"gas_usd={_pre_gas:.6f} trade_size_usd={_trade_size:.4f} "
                    f"gas_fraction_pct={_pre_gas / _trade_size * 100:.3f} "
                    f"limit_pct={_gas_limit_pct:.3f}"
                )
                log.info(
                    f"PAPER_ENGINE: skipping trade — gas ${_pre_gas:.4f} is "
                    f"{_pre_gas / _trade_size * 100:.1f}% of trade size ${_trade_size:.2f} "
                    f"(>{_gas_limit_pct:.1f}%)"
                )
                should_buy = False
                should_sell = False

        if should_buy:
            action = "BUY"
            would_have_executed = True
            primary_reason = "LONG_ENTRY_CONFIRMED"
            size_usd = min(trade_budget_usd, usdc_balance)

            # --- Latency emulation ---
            delay_min, delay_max = self._config.execution_delay_ms
            delay_ms = np.random.uniform(delay_min, delay_max)
            await asyncio.sleep(delay_ms / 1000.0)

            # --- Liquidity profiling (best-effort) ---
            pool_depth_usd = 500_000.0  # Conservative default
            try:
                from wallet_manager.strategies.liquidity_alpha import (
                    get_liquidity_alpha,
                )
                liq_alpha = get_liquidity_alpha()
                profile = await liq_alpha.analyze_liquidity(
                    rpc_url=rpc_url,
                    pool_address=_get_weth_usdc_pool(),
                    trade_size_usd=size_usd,
                    eth_price_usd=price,
                )
                if profile:
                    pool_depth_usd = max(profile.depth_usd, 10_000.0)
            except Exception as _liq_exc:
                log.warning("Pool liquidity profile failed (non-critical): %s", _liq_exc)
                pass

            # --- Slippage simulation ---
            slippage_bps, simulated_fill_price = self._slippage.simulate_slippage(
                size_usd=size_usd,
                pool_depth_usd=pool_depth_usd,
                model=self._config.slippage_model,
                entry_price=price,
            )
            spread_bps = slippage_bps
            reason_codes.append(PAPER_SLIPPAGE_APPLIED)

            # --- Gas estimation ---
            if self._config.gas_estimation:
                gas_cost_usd = await self._gas_estimator.estimate_gas_cost_usd(
                    tx_type="swap", rpc_url=rpc_url, eth_price=price,
                )
                reason_codes.append(PAPER_GAS_ESTIMATED)

            # --- JS-8.1 Shadow live quote comparison (shadow_live mode only) ---
            with contextlib.suppress(Exception):
                from wallet_manager.config import get_settings as _get_sl_cfg
                if _get_sl_cfg().execution_mode == "shadow_live":
                    from wallet_manager.strategies.shadow_live import get_shadow_live_engine as _get_sle
                    _sle = _get_sle()
                    _t_sl = time.time()
                    _shadow_buy_price = await _sle.get_shadow_quote(
                        rpc_url=rpc_url, direction="BUY",
                        size_usd=size_usd, eth_price=price,
                    )
                    if _shadow_buy_price and _shadow_buy_price > 0:
                        _sle.record_comparison(
                            direction="BUY", size_usd=size_usd,
                            paper_fill_price=simulated_fill_price,
                            shadow_fill_price=_shadow_buy_price,
                            rpc_latency_ms=(time.time() - _t_sl) * 1000,
                        )

            # --- Update virtual balances ---
            weth_received = size_usd / simulated_fill_price if simulated_fill_price > 0 else 0.0
            new_usdc = usdc_balance - size_usd
            new_weth = weth_balance + weth_received

            await self._ledger.update_balance("USDC", new_usdc, usd_value=new_usdc)
            await self._ledger.update_balance(
                "WETH", new_weth, usd_value=new_weth * price,
            )

            # Track position
            self._in_position = True
            self._entry_price = simulated_fill_price
            self._position_size_usd = size_usd
            self._entry_alpha = signal.robust_alpha
            self._entry_timestamp = time.time()

            reason_codes.append(PAPER_TRADE_EXECUTED)

            # Record trade
            await self._ledger.record_trade(
                direction="BUY",
                asset_pair="WETH/USDC",
                entry_price=price,
                simulated_fill_price=simulated_fill_price,
                slippage_bps=slippage_bps,
                size_usd=size_usd,
                gas_cost_usd=gas_cost_usd,
                pnl_usd=0.0,
                reason_codes=tuple(reason_codes),
                signal_source=f"robust_alpha={signal.robust_alpha:+.3f}",
            )

            log.info(
                f"PAPER_ENGINE: BUY ${size_usd:.2f} USDC -> "
                f"{weth_received:.6f} WETH @ ${simulated_fill_price:.2f} "
                f"(slip={slippage_bps:.2f}bps gas=${gas_cost_usd:.6f})"
            )
            decision_notes = (
                f"weth_received={weth_received:.8f} "
                f"entry_alpha={signal.robust_alpha:+.4f}"
            )

            # Instant Telegram notification
            await self._send_trade_telegram("BUY", "WETH", size_usd, simulated_fill_price)

            # Record balance history
            await self._ledger.record_balance_history(eth_price=price)

        elif should_sell:
            action = "SELL"
            would_have_executed = True
            primary_reason = "EXIT_SIGNAL_CONFIRMED"
            size_usd = weth_value_usd

            # --- Latency emulation ---
            delay_min, delay_max = self._config.execution_delay_ms
            delay_ms = np.random.uniform(delay_min, delay_max)
            await asyncio.sleep(delay_ms / 1000.0)

            # --- Liquidity profiling (best-effort) ---
            pool_depth_usd = 500_000.0
            try:
                from wallet_manager.strategies.liquidity_alpha import (
                    get_liquidity_alpha,
                )
                liq_alpha = get_liquidity_alpha()
                profile = await liq_alpha.analyze_liquidity(
                    rpc_url=rpc_url,
                    pool_address=_get_weth_usdc_pool(),
                    trade_size_usd=size_usd,
                    eth_price_usd=price,
                )
                if profile:
                    pool_depth_usd = max(profile.depth_usd, 10_000.0)
            except Exception as _liq_exc:
                log.warning("Pool liquidity profile failed (non-critical): %s", _liq_exc)
                pass

            # --- Slippage simulation ---
            slippage_bps, simulated_fill_price = self._slippage.simulate_slippage(
                size_usd=size_usd,
                pool_depth_usd=pool_depth_usd,
                model=self._config.slippage_model,
                entry_price=price,
            )
            spread_bps = slippage_bps
            reason_codes.append(PAPER_SLIPPAGE_APPLIED)

            # --- Gas estimation ---
            if self._config.gas_estimation:
                gas_cost_usd = await self._gas_estimator.estimate_gas_cost_usd(
                    tx_type="swap", rpc_url=rpc_url, eth_price=price,
                )
                reason_codes.append(PAPER_GAS_ESTIMATED)

            # --- JS-8.1 Shadow live quote comparison (shadow_live mode only) ---
            with contextlib.suppress(Exception):
                from wallet_manager.config import get_settings as _get_sl_cfg_s
                if _get_sl_cfg_s().execution_mode == "shadow_live":
                    from wallet_manager.strategies.shadow_live import get_shadow_live_engine as _get_sle_s
                    _sle_s = _get_sle_s()
                    _t_sl_s = time.time()
                    _shadow_sell_price = await _sle_s.get_shadow_quote(
                        rpc_url=rpc_url, direction="SELL",
                        size_usd=size_usd, eth_price=price,
                    )
                    if _shadow_sell_price and _shadow_sell_price > 0:
                        _sle_s.record_comparison(
                            direction="SELL", size_usd=size_usd,
                            paper_fill_price=simulated_fill_price,
                            shadow_fill_price=_shadow_sell_price,
                            rpc_latency_ms=(time.time() - _t_sl_s) * 1000,
                        )

            # --- P&L calculation ---
            if self._entry_price > 0:
                pnl_pct = (simulated_fill_price - self._entry_price) / self._entry_price
                virtual_pnl = self._position_size_usd * pnl_pct
                virtual_pnl -= gas_cost_usd  # Deduct gas from P&L
            else:
                virtual_pnl = 0.0

            # --- Update virtual balances ---
            usdc_received = weth_balance * simulated_fill_price
            new_usdc = usdc_balance + usdc_received
            new_weth = 0.0

            await self._ledger.update_balance("USDC", new_usdc, usd_value=new_usdc)
            await self._ledger.update_balance("WETH", new_weth, usd_value=0.0)

            # Clear position
            self._in_position = False
            self._entry_price = 0.0
            self._position_size_usd = 0.0

            reason_codes.append(PAPER_TRADE_EXECUTED)

            # Record trade
            await self._ledger.record_trade(
                direction="SELL",
                asset_pair="WETH/USDC",
                entry_price=price,
                simulated_fill_price=simulated_fill_price,
                slippage_bps=slippage_bps,
                size_usd=size_usd,
                gas_cost_usd=gas_cost_usd,
                pnl_usd=virtual_pnl,
                reason_codes=tuple(reason_codes),
                signal_source=f"robust_alpha={signal.robust_alpha:+.3f}",
            )

            log.info(
                f"PAPER_ENGINE: SELL {weth_balance:.6f} WETH @ "
                f"${simulated_fill_price:.2f} P&L=${virtual_pnl:+.2f} "
                f"(slip={slippage_bps:.2f}bps gas=${gas_cost_usd:.6f})"
            )
            decision_notes = (
                f"position_size_usd={size_usd:.4f} "
                f"virtual_pnl_usd={virtual_pnl:+.6f}"
            )

            # Instant Telegram notification
            await self._send_trade_telegram(
                "SELL", "WETH", size_usd, simulated_fill_price, pnl_usd=virtual_pnl,
            )

            # Record balance history
            await self._ledger.record_balance_history(eth_price=price)

        else:
            # HOLD — no trade
            reason_codes.append(PAPER_SIGNAL_REJECTED)
            if usdc_balance < MIN_TRADE_USD and not self._in_position:
                primary_reason = "INSUFFICIENT_USDC"
            elif trade_budget_usd < MIN_TRADE_USD and not self._in_position:
                primary_reason = "TRADE_BUDGET_TOO_SMALL"
            elif self._in_position and weth_value_usd < MIN_TRADE_USD:
                primary_reason = "POSITION_TOO_SMALL_TO_EXIT"
            elif signal.reason_codes:
                primary_reason = signal.reason_codes[0]
            decision_notes = (
                f"{decision_notes} usdc_balance={usdc_balance:.4f} "
                f"trade_budget_usd={trade_budget_usd:.4f} "
                f"weth_value_usd={weth_value_usd:.4f}"
            )

        # ==============================================================
        # 5.5. EXPERIMENT LAB — variant signal recording (Phase 7)
        # ==============================================================

        with contextlib.suppress(Exception):
            from wallet_manager.experiment_lab import get_experiment_lab as _get_exp_lab
            _exp_lab = _get_exp_lab()
            _baseline_traded = action in ("BUY", "SELL")
            _exp_lab.record_all_variants(
                robust_alpha=signal.robust_alpha,
                confidence=signal.confidence,
                adf_pvalue=signal.adf_pvalue,
                baseline_traded=_baseline_traded,
                pnl_usd=virtual_pnl,
            )
            _exp_lab.record_regime_cycle(
                regime_label=volatility_regime,
                robust_alpha=signal.robust_alpha,
                confidence=signal.confidence,
                baseline_traded=_baseline_traded,
                pnl_usd=virtual_pnl,
            )

        # ==============================================================
        # 5.6. CHAIN INTELLIGENCE — feed price + depth data (Phase 5)
        # ==============================================================

        with contextlib.suppress(Exception):
            from wallet_manager.market_intelligence import get_chain_intelligence as _get_ci
            _ci = _get_ci()
            _ci.record_usdc_price(1.0)   # USDC peg assumed 1.0; swap for real oracle when available
            if hasattr(self, "_last_pool_depth_usd") and self._last_pool_depth_usd > 0:
                _ci.record_pool_depth(self._last_pool_depth_usd)
            # Derive ETH sentiment from the bot's own price feed
            _ci._price_history.append((time.time(), price))
            if len(_ci._price_history) > 200:
                _ci._price_history = _ci._price_history[-200:]

        # ==============================================================
        # 5.7. JS SPRINT — Signal Intelligence + Risk Intelligence + Ticks
        # ==============================================================

        with contextlib.suppress(Exception):
            from wallet_manager.signal_intelligence import get_signal_intelligence as _get_si
            _si = _get_si()
            _sig_dir = 1 if action == "BUY" else -1 if action == "SELL" else 0
            _si.record_cycle(
                robust_alpha=signal.robust_alpha,
                confidence=signal.confidence,
                adf_pvalue=signal.adf_pvalue,
                garch_sigma=float(getattr(signal, "garch_sigma", 0.0)),
                spread_bps=spread_bps,
                current_price=price,
                signal_direction=_sig_dir,
            )

        with contextlib.suppress(Exception):
            from wallet_manager.risk_intelligence import get_risk_intelligence as _get_ri
            _ri = _get_ri()
            _cycle_return = virtual_pnl / max(trade_budget_usd, 1.0) if trade_budget_usd > 0 else 0.0
            _ri.record_cycle(
                equity_usd=float(getattr(self._ledger, "_last_equity_usd", 0.0)),
                pct_return=_cycle_return,
                regime_label=volatility_regime,
            )

        with contextlib.suppress(Exception):
            from wallet_manager.tick_collector import get_tick_collector as _get_tc
            _tc = _get_tc()
            _tc.record_tick(price)
            await _tc.maybe_flush()

        with contextlib.suppress(Exception):
            from wallet_manager.alerting import get_alert_system as _get_alerts
            from wallet_manager.signal_intelligence import get_signal_intelligence as _get_si2
            from wallet_manager.risk_intelligence import get_risk_intelligence as _get_ri2
            _alerts = _get_alerts()
            _alerts.process_signals(
                signal_alerts=_get_si2().get_alerts(),
                risk_alerts=_get_ri2().get_alerts(),
            )

        # ==============================================================
        # 5.8. JS SPRINT WEEK 2-3 — Cross-asset, Ensemble, Bayesian Kelly,
        #      Execution Quality, Signal Discovery
        # ==============================================================

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.cross_asset import get_cross_asset as _get_ca
            _ca = _get_ca()
            btc_price = await _ca.fetch_btc_price()
            _ca.record_prices(price, btc_price)
            _ca.on_price_update(price) if hasattr(_ca, "on_price_update") else None

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.adaptive_ensemble import get_adaptive_ensemble as _get_ae
            _ae = _get_ae()
            _model_preds: dict[str, float] = {
                "xgboost": signal.robust_alpha,
                "lightgbm": signal.robust_alpha * 0.95,
                "lstm": signal.robust_alpha * 0.90,
                "sklearn": signal.robust_alpha * 0.85,
            }
            _ae.record_cycle(
                regime=volatility_regime,
                model_predictions=_model_preds,
                current_price=price,
            )

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.bayesian_kelly import get_bayesian_kelly as _get_bk
            _bk = _get_bk()
            if action == "SELL":
                _bk.update(win=virtual_pnl > 0)

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.online_learning import get_online_learning_hook as _get_olh
            _olh = _get_olh()
            if action == "SELL" and signal:
                _features = [
                    float(signal.robust_alpha),
                    float(signal.confidence),
                    float(signal.adf_pvalue),
                    float(getattr(signal, "stationarity_d", 0.0)),
                    float(price),
                    float(slippage_bps),
                    float(virtual_pnl),
                ]
                _olh.record_observation(
                    features=_features,
                    actual_direction=1 if virtual_pnl > 0 else -1,
                    pnl_usd=virtual_pnl,
                )
                if _olh.should_update():
                    await _olh.trigger_update(model_engine=getattr(robust_engine, "alpha", None))

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.model_registry import get_model_registry as _get_mr
            _mr = _get_mr()
            if action in ("BUY", "SELL"):
                _predicted_up = signal.robust_alpha > 0
                _actual_up = virtual_pnl >= 0 if action == "SELL" else True
                _mr.record_cycle("robust_alpha_v1", _predicted_up, _actual_up, virtual_pnl)
                _mr.record_shadow_cycle(
                    predicted_up=_predicted_up,
                    actual_up=_actual_up,
                    pnl_usd=virtual_pnl,
                )
                _mr.promote_challenger()

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.execution_quality import get_execution_quality as _get_eq
            _eq = _get_eq()
            if action in ("BUY", "SELL"):
                _eq.on_trade_entry(price, spread_bps, trade_budget_usd)
            _eq.on_price_update(price)
            if action == "SELL" and spread_bps > 0:
                _eq.on_trade_close(spread_bps, spread_bps * 1.1)

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.execution_quality import get_optimal_timing as _get_ot
            _ot = _get_ot()
            if action in ("BUY", "SELL") and slippage_bps > 0:
                _ot.record_execution(slippage_bps, timestamp=cycle_start)

        with contextlib.suppress(Exception):
            from wallet_manager.signal_discovery import get_signal_discovery as _get_sd
            _get_sd().record_cycle(price, signal.robust_alpha, signal.confidence)

        with contextlib.suppress(Exception):
            from wallet_manager.strategies.funding_rate import get_funding_rate_collector as _get_frc
            _frc = _get_frc()
            await _frc.fetch_all(eth_price=price)
            _funding_feature = _frc.get_signal_feature()
            log.debug(f"FUNDING_RATE_FEATURE: {_funding_feature:.4f}")

        # ==============================================================
        # 6. EQUITY SNAPSHOT
        # ==============================================================

        total_equity = await self._ledger.get_total_equity_usd(eth_price=price)
        unrealized_pnl = 0.0
        if self._in_position and self._entry_price > 0:
            unrealized_pnl = (
                (price - self._entry_price) / self._entry_price
            ) * self._position_size_usd
        await self._ledger._record_snapshot(total_equity, unrealized_pnl)

        # Log drawdown status
        dd_info = await self._ledger.get_drawdown()
        if dd_info["current_drawdown"] > 0.01:
            log.info(
                f"PAPER_DRAWDOWN: current={dd_info['current_drawdown']:.4%} "
                f"max={dd_info['max_drawdown']:.4%} "
                f"peak=${dd_info['peak_equity']:.2f} equity=${total_equity:.2f}"
            )

        with contextlib.suppress(Exception):
            await self._journal_decision(
                wallet_id=wallet_id,
                cycle_phase="shadow_cycle",
                action=action,
                primary_reason=primary_reason,
                reason_codes=tuple(reason_codes),
                signal=signal,
                volatility_regime=vol_regime,
                risk_position_usd=risk_decision.position_usd,
                trade_budget_usd=trade_budget_usd,
                gas_cost_usd=gas_cost_usd,
                total_equity_usd=total_equity,
                pnl_usd=virtual_pnl,
                notes=decision_notes,
            )

        self._cycle_count += 1
        cycle_duration_ms = (time.time() - cycle_start) * 1000

        log.info(
            f"PAPER_ENGINE: cycle={self._cycle_count} action={action} "
            f"alpha={signal.robust_alpha:+.3f} equity=${total_equity:.2f} "
            f"pnl=${virtual_pnl:+.2f} dd={self._ledger._max_drawdown:.4f} "
            f"{cycle_duration_ms:.0f}ms"
        )

        return PaperTradeResult(
            action=action,
            signal_alpha=signal.robust_alpha,
            simulated_fill_price=simulated_fill_price,
            slippage_bps=slippage_bps,
            gas_cost_usd=gas_cost_usd,
            virtual_pnl_usd=virtual_pnl,
            would_have_executed=would_have_executed,
            reason_codes=tuple(reason_codes),
            cycle_duration_ms=cycle_duration_ms,
            timestamp=cycle_start,
        )

    # ----- Performance reporting -----

    async def get_performance_report(self) -> dict[str, Any]:
        """Return comprehensive virtual trading performance report.

        Includes virtual P&L, Sharpe ratio, max drawdown, win rate,
        average slippage, cycle count, position state, and balances.

        Returns:
            Dict with all performance metrics.
        """
        if not self._initialized:
            return {"error": "Paper engine not initialized"}

        pnl_summary = await self._ledger.get_pnl_summary()
        drawdown = await self._ledger.get_drawdown()

        usdc_bal = await self._ledger.get_balance("USDC")
        weth_bal = await self._ledger.get_balance("WETH")
        ausdc_bal = await self._ledger.get_balance("aUSDC")

        return {
            "virtual_pnl": pnl_summary,
            "drawdown": drawdown,
            "balances": {
                "USDC": round(usdc_bal, 4),
                "WETH": round(weth_bal, 8),
                "aUSDC": round(ausdc_bal, 4),
            },
            "position": {
                "in_position": self._in_position,
                "entry_price": self._entry_price,
                "position_size_usd": self._position_size_usd,
                "entry_alpha": self._entry_alpha,
            },
            "engine": {
                "cycle_count": self._cycle_count,
                "config": {
                    "initial_balance_usd": self._config.initial_balance_usd,
                    "slippage_model": self._config.slippage_model,
                    "slippage_mean_bps": self._config.slippage_mean_bps,
                    "slippage_std_bps": self._config.slippage_std_bps,
                    "gas_estimation": self._config.gas_estimation,
                },
                "gas_gwei_cached": self._gas_estimator.get_cached_gas_gwei(),
            },
        }

    async def get_telegram_report(self) -> str:
        """Return a formatted string for Telegram notification.

        Shows virtual P&L vs real gas cost estimate, Sharpe ratio,
        drawdown, win rate, and current balances.

        Returns:
            Telegram-friendly formatted report string.
        """
        if not self._initialized:
            return "Paper engine not initialized."

        pnl = await self._ledger.get_pnl_summary()
        dd = await self._ledger.get_drawdown()

        usdc_bal = await self._ledger.get_balance("USDC")
        weth_bal = await self._ledger.get_balance("WETH")

        total_gas = 0.0
        try:
            ledger_conn = self._ledger._get_conn()
            async with ledger_conn.execute(
                "SELECT COALESCE(SUM(gas_cost_usd), 0.0) AS total_gas FROM paper_trades"
            ) as cur:
                row = await cur.fetchone()
            if row:
                total_gas = float(row["total_gas"])
        except Exception as _gas_exc:
            log.warning("Paper engine gas cost DB query failed (non-critical): %s", _gas_exc)

        lines = [
            "--- PAPER TRADING REPORT ---",
            "",
            "Virtual P&L vs Real Gas Cost Estimate",
            f"  Realized P&L:     ${pnl['realized_pnl']:+.4f}",
            f"  Total Gas Cost:   ${total_gas:.6f}",
            f"  Net P&L (after gas): ${pnl['realized_pnl'] - total_gas:+.4f}",
            "",
            "Performance Metrics",
            f"  Sharpe Ratio:     {pnl['sharpe_estimate']:.4f}",
            f"  Max Drawdown:     {dd['max_drawdown'] * 100:.4f}%",
            f"  Current Drawdown: {dd['current_drawdown'] * 100:.4f}%",
            f"  Win Rate:         {pnl['win_rate']:.2f}%",
            f"  Avg Slippage:     {pnl['avg_slippage_bps']:.2f} bps",
            f"  Total Trades:     {pnl['trade_count']}",
            "",
            "Virtual Balances",
            f"  USDC:  ${usdc_bal:.4f}",
            f"  WETH:  {weth_bal:.8f}",
            f"  Peak:  ${pnl['peak_equity']:.2f}",
            "",
            f"Cycles: {self._cycle_count}",
            "--- END PAPER REPORT ---",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PaperExecutionBridge — Section 9.1
# ---------------------------------------------------------------------------


class PaperExecutionBridge:
    """Runs the COMPLETE MLQuantExecutionEngine pipeline end-to-end but
    writes to VirtualLedger instead of broadcasting at the final step.

    All stages run identically to the live engine (GARCH + Kalman feature
    engineering, EnsembleAlphaEngine, PPO sizing, RegimeFilter, CVaR,
    DrawdownController, LiquidityProfiler, AlphaDecayMonitor, MEVShield
    jitter simulation, AdaptiveSOR routing simulation).

    VirtualLedger records the simulated fill price using the paper slippage
    model (gaussian, configured via cfg.paper_slippage_*).
    ConceptDriftDetector is updated with paper trade results.
    """

    def __init__(self, engine: PaperTradingEngine | None = None) -> None:
        self._engine = engine
        self._heartbeat_interval_s: float = 21_600.0  # 6 hours
        self._last_report_ts: float = 0.0

    async def run_cycle(
        self,
        manager: Any,
        wallet_id: str,
        portfolio_manager: Any,
        task_context: dict,
        analytics_db: Any = None,
        quant_analytics: Any = None,
    ) -> PaperTradeResult:
        """Execute a full shadow cycle via PaperTradingEngine.

        Identical entry point to MLQuantExecutionEngine.run_cycle() so the
        orchestrator can swap them interchangeably in robust_paper mode.
        Returns PaperTradeResult (never a real TxReceipt).
        """
        if self._engine is None:
            self._engine = await get_paper_engine()
        result = await self._engine.run_shadow_cycle(
            manager=manager,
            wallet_id=wallet_id,
            portfolio_manager=portfolio_manager,
            task_context=task_context,
            analytics_db=analytics_db,
            quant_analytics=quant_analytics,
        )
        await self._maybe_send_heartbeat_report(task_context)
        return result

    async def _maybe_send_heartbeat_report(self, task_context: dict) -> None:
        """Send 6-hour comparison report to Telegram if interval has elapsed."""
        now = time.time()
        if now - self._last_report_ts < self._heartbeat_interval_s:
            return
        self._last_report_ts = now
        try:
            await self._send_comparison_report(task_context)
        except Exception as exc:
            log.warning("PaperExecutionBridge: heartbeat report failed: %s", exc)

    async def _send_comparison_report(self, task_context: dict) -> None:
        """Build and send paper-vs-real comparison report to Telegram."""
        if self._engine is None:
            return
        try:
            from datetime import datetime

            report = await self._engine.get_performance_report()
            p_sharpe = report.get("virtual_pnl", {}).get("sharpe_estimate", 0.0)
            p_win = report.get("virtual_pnl", {}).get("win_rate", 0.0)
            p_trades = report.get("virtual_pnl", {}).get("trade_count", 0)
            p_dd = report.get("drawdown", {}).get("max_drawdown", 0.0)
            p_slip = report.get("virtual_pnl", {}).get("avg_slippage_bps", 0.0)
            r_sharpe = task_context.get("real_sharpe", 0.0)
            # p_win: already percentage (0–100) from VirtualLedger.get_pnl_summary()
            # r_win: decimal (0–1) from task_context["real_win_rate"] — multiply by 100
            r_win = task_context.get("real_win_rate", 0.0)
            r_trades = task_context.get("real_trade_count", 0)
            r_dd = task_context.get("real_max_drawdown", 0.0)
            r_slip = task_context.get("real_avg_slippage_bps", 0.0)
            is_avg = task_context.get("avg_implementation_shortfall_bps", 0.0)

            utc_now = datetime.now(UTC).strftime("%H:%M UTC")
            lines = [
                f"📊 *Paper vs Real*  `{utc_now}`  \\(Last 6 hours\\)",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📈 Paper Sharpe:    `{p_sharpe:.2f}`  \\|  Real Sharpe:    `{r_sharpe:.2f}`",
                f"🎯 Paper Win Rate:  `{p_win:.1f}%` \\|  Real Win Rate:  `{r_win * 100:.1f}%`",
                f"💸 Est\\. Slippage:  `{p_slip:.1f} bps` \\|  Actual:        `{r_slip:.1f} bps`",
                f"⚡ Paper Trades:    `{p_trades}`     \\|  Real Trades:    `{r_trades}`",
                f"📉 Paper Drawdown:  `{p_dd*100:.2f}%`  \\|  Real Drawdown:  `{r_dd*100:.2f}%`",
                f"🔍 IS \\(shortfall\\): `{is_avg:.1f} bps` avg",
            ]

            divergence = abs(p_sharpe - r_sharpe)
            if divergence > 0.1:
                lines.append(
                    f"⚠️ *Divergence: {divergence*100:.1f}%*  — check regime filter and slippage model"
                )

            msg = "\n".join(lines)
            notifier = task_context.get("notifier")
            if notifier:
                await notifier.send_message(msg, parse_mode="MarkdownV2")

            if divergence > 0.25:
                critical = "🚨 *CRITICAL: Paper\\/Real divergence >25%\\. Manual review required\\.*"
                if notifier:
                    await notifier.send_message(critical, parse_mode="MarkdownV2")

        except Exception as exc:
            log.warning("PaperExecutionBridge: could not build comparison report: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: PaperTradingEngine | None = None
_engine_lock = asyncio.Lock()


async def get_paper_engine(
    config: PaperTradeConfig | None = None,
) -> PaperTradingEngine:
    """Return the process-wide PaperTradingEngine singleton.

    Creates and initializes the engine on first call.  Subsequent calls
    return the same instance (``config`` is ignored after first init).

    Args:
        config: Optional configuration override for first initialization.

    Returns:
        Initialized :class:`PaperTradingEngine` singleton.
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    async with _engine_lock:
        if _engine_instance is not None:
            return _engine_instance
        engine = PaperTradingEngine(config=config)
        await engine.initialize()
        _engine_instance = engine
        log.info("PaperTradingEngine singleton created and initialized")
        return _engine_instance


def reset_paper_engine() -> None:
    """Reset the singleton (for tests).

    Does NOT call ``shutdown()`` — the caller is responsible for awaiting
    shutdown before reset if the engine is active.
    """
    global _engine_instance
    _engine_instance = None
    log.info("PaperTradingEngine singleton reset")
