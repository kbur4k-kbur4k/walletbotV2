"""
wallet_manager/strategies/quant_analytics.py

Quant-Lab Analytics Engine — Post-Trade Analysis & ML Retraining Pipeline.

Components
----------
1. Transaction Cost Analysis (TCA)
   - Tracks execution price vs mid-price at entry time
   - Calculates Implementation Shortfall per trade
   - Measures slippage, gas costs, and total execution cost

2. SQLite Database (WAL Mode)
   - Logs all market features, ML predictions, trade decisions, outcomes
   - Schema designed for continuous model retraining
   - Separate tables: market_features, ml_predictions, trade_executions

3. Performance Analytics
   - Rolling Sharpe Ratio, Sortino Ratio, Calmar Ratio
   - Win rate, profit factor, average trade duration
   - Strategy attribution (which signals drive PnL)

4. Model Retraining Data Export
   - Export labeled training data (features → outcome) for XGBoost/LSTM
   - Automatic feature-label alignment for supervised learning
   - Generates training CSVs on demand

Architecture
------------
All writes are async (aiosqlite) with WAL mode for zero read-blocking.
Tables use parameterised queries for SQL injection prevention.
Timestamps stored as Unix epoch (float) for cross-platform compatibility.
"""
# ruff: noqa: RUF003

from __future__ import annotations

import asyncio
import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiosqlite
import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_default_db_path() -> str:
    from wallet_manager.config import get_settings
    return str(get_settings().quant_analytics_db_path)


_DEFAULT_DB_PATH = _get_default_db_path()

# TCA configuration
SLIPPAGE_WARNING_BPS = 50.0     # Warn if slippage > 50 bps
GAS_COST_WARNING_USD = 5.0      # Warn if gas > $5 per trade


class FeatureSnapshotLike(Protocol):
    price: float


class MarketFeaturesLike(Protocol):
    price: float
    log_return: float
    kalman_price: float
    kalman_trend: float
    garch_sigma: float
    realized_vol_5: float
    realized_vol_30: float
    vol_ratio: float
    momentum_5: float
    momentum_15: float
    momentum_30: float
    momentum_60: float
    momentum_120: float
    volume_imbalance: float
    spread_bps: float
    trade_intensity: float
    skewness: float
    kurtosis: float


class SignalLike(Protocol):
    features: MarketFeaturesLike | None
    alpha_score: float
    prob_up: float
    prob_down: float
    xgb_prediction: float
    lstm_prediction: float
    garch_sigma: float
    volatility_regime: str
    confidence: float
    should_enter_long: bool
    should_enter_short: bool
    should_exit: bool


# ---------------------------------------------------------------------------
# SQL Schema
# ---------------------------------------------------------------------------

_CREATE_MARKET_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS market_features (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    price           REAL NOT NULL,
    log_return      REAL,
    kalman_price    REAL,
    kalman_trend    REAL,
    garch_sigma     REAL,
    realized_vol_5  REAL,
    realized_vol_30 REAL,
    vol_ratio       REAL,
    momentum_5      REAL,
    momentum_15     REAL,
    momentum_30     REAL,
    momentum_60     REAL,
    momentum_120    REAL,
    volume_imbalance REAL,
    spread_bps      REAL,
    trade_intensity REAL,
    skewness        REAL,
    kurtosis        REAL,
    vol_regime      TEXT
);
"""

_CREATE_ML_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS ml_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    price           REAL NOT NULL,
    alpha_score     REAL NOT NULL,
    prob_up         REAL NOT NULL,
    prob_down       REAL NOT NULL,
    xgb_prediction  REAL,
    lstm_prediction REAL,
    garch_sigma     REAL,
    vol_regime      TEXT,
    confidence      REAL,
    should_long     INTEGER,
    should_short    INTEGER,
    should_exit     INTEGER,
    action_taken    TEXT
);
"""

_CREATE_TRADE_EXECUTIONS_SQL = """
CREATE TABLE IF NOT EXISTS trade_executions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               REAL NOT NULL,
    side                    TEXT NOT NULL,
    entry_price             REAL,
    exit_price              REAL,
    mid_price_at_entry      REAL,
    execution_price         REAL,
    position_usd            REAL,
    pnl_usd                 REAL,
    pnl_pct                 REAL,
    slippage_bps            REAL,
    gas_cost_gwei           REAL,
    gas_cost_usd            REAL,
    implementation_shortfall_bps REAL,
    dex_venue               TEXT,
    dex_savings_bps         REAL,
    risk_source             TEXT,
    var_99                  REAL,
    drawdown_state          TEXT,
    alpha_at_entry          REAL,
    garch_at_entry          REAL,
    vol_regime              TEXT,
    cycle_duration_ms       REAL,
    jitter_ms               REAL
);
"""

_CREATE_PERFORMANCE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    total_tvl_usd   REAL,
    total_pnl_usd   REAL,
    trade_count     INTEGER,
    win_rate        REAL,
    sharpe_ratio    REAL,
    sortino_ratio   REAL,
    max_drawdown    REAL,
    profit_factor   REAL,
    avg_trade_pnl   REAL,
    avg_hold_time   REAL
);
"""


# ---------------------------------------------------------------------------
# TCA Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TCAReport:
    """Transaction Cost Analysis for a single trade.

    Implementation Shortfall = (Execution Price - Decision Price) / Decision Price
    Total Cost = Slippage + Gas + Opportunity Cost
    """

    trade_id: int
    side: str
    decision_price: float        # Mid-price when trade was decided
    execution_price: float       # Actual fill price
    slippage_bps: float          # (exec - decision) / decision × 10000
    gas_cost_usd: float
    implementation_shortfall_bps: float
    total_cost_bps: float        # slippage + gas (in bps of trade value)
    dex_venue: str
    timestamp: float


@dataclass
class PerformanceMetrics:
    """Rolling performance analytics."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_hold_time_min: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0
    avg_slippage_bps: float = 0.0
    avg_implementation_shortfall_bps: float = 0.0


# ---------------------------------------------------------------------------
# Quant Analytics Engine
# ---------------------------------------------------------------------------


class QuantAnalytics:
    """Post-trade analytics and ML data pipeline.

    Handles:
    - Recording market features, ML predictions, and trade outcomes
    - Computing TCA (Transaction Cost Analysis)
    - Calculating performance metrics (Sharpe, Sortino, etc.)
    - Exporting training data for model retraining

    Usage:
        analytics = QuantAnalytics()
        await analytics.initialize()

        # After each cycle:
        await analytics.record_cycle(price, signal, action, ...)

        # Get performance:
        metrics = await analytics.get_performance_metrics()

        # Export training data:
        await analytics.export_training_data("training_data.csv")
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        import aiosqlite

        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=-8000")  # 8MB cache

        await self._db.execute(_CREATE_MARKET_FEATURES_SQL)
        await self._db.execute(_CREATE_ML_PREDICTIONS_SQL)
        await self._db.execute(_CREATE_TRADE_EXECUTIONS_SQL)
        await self._db.execute(_CREATE_PERFORMANCE_SNAPSHOTS_SQL)
        await self._db.commit()

        self._initialized = True
        log.info(f"QuantAnalytics: initialized DB at {self._db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def _ensure_db(self) -> None:
        """Ensure database is initialized."""
        if not self._initialized:
            await self.initialize()

    def _get_db(self) -> aiosqlite.Connection:
        db = self._db
        if db is None:
            raise RuntimeError("QuantAnalytics database is not initialized")
        return db

    @staticmethod
    def _write_training_csv(
        output_path: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> int:
        path = Path(output_path)
        count = 0
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    async def record_market_features(
        self,
        price: float,
        features: MarketFeaturesLike | None = None,
        vol_regime: str = "normal",
    ) -> None:
        """Record a market feature snapshot."""
        await self._ensure_db()
        db = self._get_db()

        if features is not None:
            fv = features
            await db.execute(
                """INSERT INTO market_features
                   (timestamp, price, log_return, kalman_price, kalman_trend,
                    garch_sigma, realized_vol_5, realized_vol_30, vol_ratio,
                    momentum_5, momentum_15, momentum_30, momentum_60, momentum_120,
                    volume_imbalance, spread_bps, trade_intensity, skewness, kurtosis,
                    vol_regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), price, fv.log_return, fv.kalman_price,
                    fv.kalman_trend, fv.garch_sigma, fv.realized_vol_5,
                    fv.realized_vol_30, fv.vol_ratio, fv.momentum_5,
                    fv.momentum_15, fv.momentum_30, fv.momentum_60,
                    fv.momentum_120, fv.volume_imbalance, fv.spread_bps,
                    fv.trade_intensity, fv.skewness, fv.kurtosis, vol_regime,
                ),
            )
        else:
            await db.execute(
                """INSERT INTO market_features (timestamp, price, vol_regime)
                   VALUES (?, ?, ?)""",
                (time.time(), price, vol_regime),
            )
        await db.commit()

    async def record_ml_prediction(
        self,
        signal: SignalLike | None,
        action_taken: str = "HOLD",
    ) -> None:
        """Record an ML prediction for audit and retraining."""
        await self._ensure_db()
        db = self._get_db()

        if signal is None:
            return

        signal_price = signal.features.price if signal.features is not None else 0.0
        await db.execute(
            """INSERT INTO ml_predictions
               (timestamp, price, alpha_score, prob_up, prob_down,
                xgb_prediction, lstm_prediction, garch_sigma, vol_regime,
                confidence, should_long, should_short, should_exit, action_taken)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                signal_price,
                signal.alpha_score,
                signal.prob_up,
                signal.prob_down,
                signal.xgb_prediction,
                signal.lstm_prediction,
                signal.garch_sigma,
                signal.volatility_regime,
                signal.confidence,
                int(signal.should_enter_long),
                int(signal.should_enter_short),
                int(signal.should_exit),
                action_taken,
            ),
        )
        await db.commit()

    async def record_trade_execution(
        self,
        side: str,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        mid_price_at_entry: float = 0.0,
        execution_price: float = 0.0,
        position_usd: float = 0.0,
        pnl_usd: float = 0.0,
        pnl_pct: float = 0.0,
        slippage_bps: float = 0.0,
        gas_cost_gwei: float = 0.0,
        gas_cost_usd: float = 0.0,
        implementation_shortfall_bps: float = 0.0,
        dex_venue: str = "",
        dex_savings_bps: float = 0.0,
        risk_source: str = "",
        var_99: float = 0.0,
        drawdown_state: str = "normal",
        alpha_at_entry: float = 0.0,
        garch_at_entry: float = 0.0,
        vol_regime: str = "normal",
        cycle_duration_ms: float = 0.0,
        jitter_ms: float = 0.0,
    ) -> None:
        """Record a trade execution with full TCA data."""
        await self._ensure_db()
        db = self._get_db()

        await db.execute(
            """INSERT INTO trade_executions
               (timestamp, side, entry_price, exit_price, mid_price_at_entry,
                execution_price, position_usd, pnl_usd, pnl_pct,
                slippage_bps, gas_cost_gwei, gas_cost_usd,
                implementation_shortfall_bps, dex_venue, dex_savings_bps,
                risk_source, var_99, drawdown_state, alpha_at_entry,
                garch_at_entry, vol_regime, cycle_duration_ms, jitter_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), side, entry_price, exit_price,
                mid_price_at_entry, execution_price, position_usd,
                pnl_usd, pnl_pct, slippage_bps, gas_cost_gwei,
                gas_cost_usd, implementation_shortfall_bps,
                dex_venue, dex_savings_bps, risk_source, var_99,
                drawdown_state, alpha_at_entry, garch_at_entry,
                vol_regime, cycle_duration_ms, jitter_ms,
            ),
        )
        await db.commit()

    async def record_cycle(
        self,
        price: float,
        signal: SignalLike | None,
        action: str,
        position_usd: float = 0.0,
        pnl_usd: float = 0.0,
        dex_venue: str = "",
        gas_gwei: float = 0.0,
        mid_price: float = 0.0,
        execution_price: float = 0.0,
        implementation_shortfall_bps: float = 0.0,
    ) -> None:
        """Convenience: record all analytics for one execution cycle."""
        await self._ensure_db()

        # Record market features
        if signal is not None and signal.features is not None:
            await self.record_market_features(
                price, signal.features, signal.volatility_regime
            )

        # Record ML prediction
        await self.record_ml_prediction(signal, action)

        # Record trade if executed
        if action in ("BUY", "SELL", "SHORT_OPEN", "SHORT_CLOSE"):
            await self.record_trade_execution(
                side=action,
                execution_price=execution_price,
                mid_price_at_entry=mid_price,
                position_usd=position_usd,
                pnl_usd=pnl_usd,
                implementation_shortfall_bps=implementation_shortfall_bps,
                dex_venue=dex_venue,
                gas_cost_gwei=gas_gwei,
                alpha_at_entry=signal.alpha_score if signal else 0.0,
                garch_at_entry=signal.garch_sigma if signal else 0.0,
                vol_regime=signal.volatility_regime if signal else "normal",
            )

    # ------------------------------------------------------------------
    # TCA (Transaction Cost Analysis)
    # ------------------------------------------------------------------

    async def get_tca_report(self, last_n: int = 50) -> list[TCAReport]:
        """Generate TCA reports for the last N trades."""
        await self._ensure_db()
        db = self._get_db()

        cursor = await db.execute(
            """SELECT id, side, mid_price_at_entry, execution_price,
                      slippage_bps, gas_cost_usd,
                      implementation_shortfall_bps, dex_venue, timestamp
               FROM trade_executions
               ORDER BY timestamp DESC LIMIT ?""",
            (last_n,),
        )
        rows = list(await cursor.fetchall())

        reports = []
        for row in rows:
            (tid, side, mid_p, exec_p, slip, gas, impl_short, dex, ts) = row

            if mid_p and exec_p and mid_p > 0:
                actual_slip = abs(exec_p - mid_p) / mid_p * 10000
            else:
                actual_slip = slip or 0.0

            total_cost = actual_slip
            if gas and exec_p and exec_p > 0:
                gas_bps = (gas or 0) / max(exec_p, 1) * 10000
                total_cost += gas_bps

            reports.append(TCAReport(
                trade_id=tid,
                side=side,
                decision_price=mid_p or 0,
                execution_price=exec_p or 0,
                slippage_bps=actual_slip,
                gas_cost_usd=gas or 0,
                implementation_shortfall_bps=impl_short or 0,
                total_cost_bps=total_cost,
                dex_venue=dex or "",
                timestamp=ts,
            ))

        return reports

    async def get_avg_implementation_shortfall(self, days: int = 7) -> float:
        """Get average implementation shortfall over the last N days."""
        await self._ensure_db()
        db = self._get_db()

        cutoff = time.time() - days * 86400
        cursor = await db.execute(
            """SELECT AVG(implementation_shortfall_bps)
               FROM trade_executions WHERE timestamp > ?""",
            (cutoff,),
        )
        row = await cursor.fetchone()
        return float(row[0]) if row and row[0] else 0.0

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------

    async def get_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """Calculate rolling performance metrics over the last N days."""
        await self._ensure_db()
        db = self._get_db()

        cutoff = time.time() - days * 86400
        cursor = await db.execute(
            """SELECT pnl_usd, pnl_pct, slippage_bps,
                      implementation_shortfall_bps, position_usd
               FROM trade_executions
               WHERE timestamp > ? AND side IN ('BUY', 'SELL')
               ORDER BY timestamp""",
            (cutoff,),
        )
        rows = list(await cursor.fetchall())

        if not rows:
            return PerformanceMetrics()

        pnl_usd = [r[0] or 0 for r in rows]
        pnl_pct = [r[1] or 0 for r in rows]
        slippages = [r[2] or 0 for r in rows]
        impl_shorts = [r[3] or 0 for r in rows]

        returns = np.array(pnl_pct, dtype=np.float64)
        n = len(returns)

        # Sharpe Ratio (annualized, assuming 5-min bars)
        periods_per_year = 365 * 24 * 12  # 5-min bars
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

        # Sortino Ratio (downside deviation only)
        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else std_ret
        sortino = (mean_ret / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

        # Win rate + profit factor
        wins = sum(1 for p in pnl_usd if p > 0)
        win_rate = wins / n if n > 0 else 0.0
        gross_profit = sum(p for p in pnl_usd if p > 0)
        gross_loss = abs(sum(p for p in pnl_usd if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown from PnL series
        cumulative = np.cumsum(pnl_usd)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        calmar = (float(np.sum(returns)) / max_dd) if max_dd > 0 else 0.0

        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=float(np.mean(pnl_usd)),
            avg_hold_time_min=0.0,
            max_drawdown=max_dd,
            total_pnl=float(np.sum(pnl_usd)),
            trade_count=n,
            avg_slippage_bps=float(np.mean(slippages)) if slippages else 0.0,
            avg_implementation_shortfall_bps=float(np.mean(impl_shorts)) if impl_shorts else 0.0,
        )

    async def save_performance_snapshot(self) -> None:
        """Save current performance metrics as a snapshot."""
        metrics = await self.get_performance_metrics()
        await self._ensure_db()
        db = self._get_db()

        await db.execute(
            """INSERT INTO performance_snapshots
               (timestamp, total_tvl_usd, total_pnl_usd, trade_count,
                win_rate, sharpe_ratio, sortino_ratio, max_drawdown,
                profit_factor, avg_trade_pnl, avg_hold_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), 0, metrics.total_pnl, metrics.trade_count,
                metrics.win_rate, metrics.sharpe_ratio, metrics.sortino_ratio,
                metrics.max_drawdown, metrics.profit_factor,
                metrics.avg_trade_pnl, metrics.avg_hold_time_min,
            ),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Training Data Export
    # ------------------------------------------------------------------

    async def export_training_data(
        self,
        output_path: str = "ml_training_data.csv",
        lookforward_periods: int = 1,
    ) -> int:
        """Export labeled training data for model retraining.

        Joins market_features with future price movement labels.

        Args:
            output_path: CSV file path for output.
            lookforward_periods: How many periods ahead to label.

        Returns:
            Number of rows exported.
        """
        await self._ensure_db()
        db = self._get_db()

        cursor = await db.execute(
            """SELECT f.timestamp, f.price, f.log_return, f.kalman_price,
                      f.kalman_trend, f.garch_sigma, f.realized_vol_5,
                      f.realized_vol_30, f.vol_ratio, f.momentum_5,
                      f.momentum_15, f.momentum_30, f.momentum_60,
                      f.momentum_120, f.volume_imbalance, f.spread_bps,
                      f.trade_intensity, f.skewness, f.kurtosis, f.vol_regime
               FROM market_features f
               ORDER BY f.timestamp"""
        )
        rows = list(await cursor.fetchall())

        if len(rows) < lookforward_periods + 1:
            log.warning("Not enough data for training export.")
            return 0

        headers = [
            "timestamp", "price", "log_return", "kalman_price", "kalman_trend",
            "garch_sigma", "rv5", "rv30", "vol_ratio",
            "m5", "m15", "m30", "m60", "m120",
            "vol_imbalance", "spread_bps", "trade_intensity",
            "skewness", "kurtosis", "vol_regime",
            "future_return", "label",
        ]

        export_rows: list[list[object]] = []
        for i in range(len(rows) - lookforward_periods):
            current_price = rows[i][1]
            future_price = rows[i + lookforward_periods][1]

            if current_price and future_price and current_price > 0:
                future_return = (future_price - current_price) / current_price
                label = 1 if future_return > 0 else 0
            else:
                future_return = 0.0
                label = 0

            export_rows.append([*list(rows[i]), future_return, label])

        count = await asyncio.to_thread(
            self._write_training_csv,
            output_path,
            headers,
            export_rows,
        )

        log.info(f"QuantAnalytics: exported {count} training rows to {output_path}")
        return count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def get_table_counts(self) -> dict[str, int]:
        """Return row counts for all tables."""
        await self._ensure_db()
        db = self._get_db()

        counts: dict[str, int] = {}
        for table in ["market_features", "ml_predictions",
                       "trade_executions", "performance_snapshots"]:
            cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            counts[table] = int(row[0]) if row else 0

        return counts


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analytics_instance: QuantAnalytics | None = None


async def get_quant_analytics() -> QuantAnalytics:
    """Return the process-wide QuantAnalytics singleton."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = QuantAnalytics()
        await _analytics_instance.initialize()
    return _analytics_instance


def reset_quant_analytics() -> None:
    """Reset the singleton (for tests)."""
    global _analytics_instance
    _analytics_instance = None
