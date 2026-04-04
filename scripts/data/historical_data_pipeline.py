"""
scripts/data/historical_data_pipeline.py

Multi-source historical data pipeline for wallet_manager ML training.

Components:
  - DataQualityReport   : Structured quality metrics for a candle dataset.
  - HistoricalDataPipeline : Orchestrates fetch → preprocess → label → save.
  - RegimeBalancedSampler  : Weighted oversampling for regime-balanced training.
  - IncrementalDataUpdater : Nightly async updater that runs inside orchestrator.

Data sources (priority order):
  1. Uniswap V3 subgraph   (on-chain tick data, most accurate)
  2. DeFiLlama Coins API   (hourly OHLC, no key needed)
  3. Binance REST           (5-min CEX proxy)
  4. eth_getLogs RPC scan  (last resort)

Regime labels (rolling 30-day window):
  bull     30d return > +20% AND garch_sigma < 0.03
  bear     30d return < -20% OR rolling max_drawdown > 25%
  sideways |30d return| < 5% AND garch_sigma < 0.02
  high_vol garch_sigma > 0.04 (regardless of direction)
  crisis   30d return < -40% OR daily_vol > 0.08
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_5MIN_SECONDS = 300
_CANDLES_PER_DAY = 288          # 24h / 5min
_CANDLES_PER_30D = 8_640        # 30 days of 5-min candles
_GARCH_WINDOW = 288             # 1 day of 5-min candles for GARCH sigma

# Regime thresholds
_BULL_RETURN = 0.20
_BEAR_RETURN = -0.20
_CRISIS_RETURN = -0.40
_SIDEWAYS_ABS = 0.05
_BULL_SIGMA = 0.03
_BEAR_DRAWDOWN = 0.25
_SIDEWAYS_SIGMA = 0.02
_HIGH_VOL_SIGMA = 0.04
_CRISIS_DAILY_VOL = 0.08

# Gap-fill thresholds (in number of missing 5-min candles)
_GAP_FFILL_MAX = 6
_GAP_INTERP_MAX = 12

# Source priority for deduplication
_SOURCE_PRIORITY = {"uniswap_subgraph": 0, "defillama": 1, "binance": 2, "rpc_scan": 3}

# DB schema
_CREATE_CANDLES_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    timestamp           INTEGER PRIMARY KEY,
    open                REAL NOT NULL,
    high                REAL NOT NULL,
    low                 REAL NOT NULL,
    close               REAL NOT NULL,
    volume_usdc         REAL,
    trade_count         REAL,
    buy_volume_usdc     REAL,
    sell_volume_usdc    REAL,
    gas_price_mean_gwei REAL,
    source              TEXT,
    cex_only            INTEGER DEFAULT 0,
    price_diverged      INTEGER DEFAULT 0,
    sample_weight       REAL DEFAULT 1.0,
    purge_zone          INTEGER DEFAULT 0,
    stale               INTEGER DEFAULT 0,
    regime_label        TEXT,
    return_5m           REAL,
    garch_sigma         REAL
);
"""
_CREATE_META_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    total_candles: int
    date_range_days: float
    missing_pct: float              # WARNING if > 2%, FAIL if > 10%
    spike_count: int                # candles where |5m return| > 15%
    stale_count: int                # 3+ consecutive candles with 0% change
    regime_distribution: dict      # {"bull": 0.35, "bear": 0.22, ...}
    sources_used: list[str]
    passes_minimum: bool            # total_candles >= 105_120

    def log_summary(self) -> None:
        status = "PASS" if self.passes_minimum else "FAIL"
        log.info(
            f"DataQualityReport [{status}]: {self.total_candles:,} candles over "
            f"{self.date_range_days:.1f} days | missing={self.missing_pct:.2%} | "
            f"spikes={self.spike_count} | stale={self.stale_count} | "
            f"regimes={self.regime_distribution} | sources={self.sources_used}"
        )
        if self.missing_pct > 0.10:
            log.error("DataQualityReport: missing_pct > 10% — FAIL.")
        elif self.missing_pct > 0.02:
            log.warning("DataQualityReport: missing_pct > 2% — WARNING.")


# ---------------------------------------------------------------------------
# GARCH(1,1) sigma estimator (pure Python, no arch dependency)
# ---------------------------------------------------------------------------

def _garch_sigma_series(returns: pd.Series, omega: float, alpha: float, beta: float) -> pd.Series:
    """Compute rolling GARCH(1,1) conditional volatility."""
    n = len(returns)
    sigma2 = np.full(n, float(returns.var() or 1e-6))
    r = returns.to_numpy()
    for i in range(1, n):
        sigma2[i] = omega + alpha * r[i - 1] ** 2 + beta * sigma2[i - 1]
    return pd.Series(np.sqrt(sigma2), index=returns.index)


# ---------------------------------------------------------------------------
# HistoricalDataPipeline
# ---------------------------------------------------------------------------

class HistoricalDataPipeline:
    """Orchestrates multi-source historical data collection and preprocessing.

    Fetch priority: Uniswap subgraph → DeFiLlama → Binance → RPC scan.
    Preprocessing: dedup → gap fill → spike flag → stale flag → winsorise
                   → regime label → upsert to SQLite.
    """

    MIN_REQUIRED_CANDLES: int = 105_120   # 2 years at 5-min
    TARGET_CANDLES: int = 157_680         # 3 years target

    def __init__(
        self,
        db_path: Optional[Path] = None,
        start_ts: Optional[int] = None,
    ) -> None:
        from wallet_manager.config import get_settings
        cfg = get_settings()
        self._cfg = cfg
        self._db_path = db_path or cfg.historical_data_db_path
        self._start_ts = start_ts or (int(time.time()) - 3 * 365 * 24 * 3600)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, force_refresh: bool = False) -> DataQualityReport:
        """Run the full pipeline: fetch → preprocess → label → save.

        Args:
            force_refresh: If True, re-fetch all data even if DB is populated.

        Returns:
            DataQualityReport describing the resulting dataset.
        """
        log.info(
            f"HistoricalDataPipeline.run(): db={self._db_path}, "
            f"force_refresh={force_refresh}"
        )
        await self._ensure_schema()

        if not force_refresh:
            existing = await self._count_candles()
            if existing >= self.MIN_REQUIRED_CANDLES:
                log.info(
                    f"DB already has {existing:,} candles (>= {self.MIN_REQUIRED_CANDLES:,}). "
                    "Skipping full fetch. Use force_refresh=True to override."
                )
                df = await self.load_for_training()
                return await self._validate_quality(df)

        df = await self._fetch_all_sources()
        if df.empty:
            log.error("HistoricalDataPipeline: all sources returned empty data.")
            return DataQualityReport(
                total_candles=0, date_range_days=0, missing_pct=1.0,
                spike_count=0, stale_count=0, regime_distribution={},
                sources_used=[], passes_minimum=False,
            )

        df = await self._normalize_to_5m(df)
        df = await self._fill_gaps(df)
        df = self._flag_spikes(df)
        df = self._flag_stale(df)
        df = self._winsorise(df)
        df = await self._label_regimes(df)

        saved = await self._save_incremental(df)
        log.info(f"HistoricalDataPipeline: {saved:,} candles upserted to {self._db_path}.")

        report = await self._validate_quality(df)
        report.log_summary()
        return report

    async def load_for_training(self) -> pd.DataFrame:
        """Load candles suitable for ML training.

        Excludes purge_zone=True and stale=True rows.
        Includes sample_weight column.

        Returns:
            DataFrame ready for RegimeBalancedSampler.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM candles WHERE purge_zone=0 AND stale=0 ORDER BY timestamp"
            )
            rows = await cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["price_diverged"] = df["price_diverged"].astype(bool)
        df["cex_only"] = df["cex_only"].astype(bool)
        df["purge_zone"] = df["purge_zone"].astype(bool)
        df["stale"] = df["stale"].astype(bool)
        log.info(f"load_for_training: {len(df):,} rows loaded.")
        return df

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def _fetch_all_sources(self) -> pd.DataFrame:
        """Fetch from all sources in priority order, merging results."""
        from scripts.data.fetch_uniswap_subgraph import fetch_swap_history, swaps_to_ohlcv
        from scripts.data.fetch_defillama import fetch_defillama_prices
        from scripts.data.fetch_binance import fetch_binance_ohlcv

        frames: list[pd.DataFrame] = []
        sources_fetched: list[str] = []

        # Priority 1: Uniswap subgraph
        if self._cfg.subgraph_url:
            try:
                log.info("Fetching from Uniswap subgraph...")
                swaps = await fetch_swap_history(start_ts=self._start_ts)
                if not swaps.empty:
                    ohlcv = swaps_to_ohlcv(swaps)
                    ohlcv["source"] = "uniswap_subgraph"
                    ohlcv["cex_only"] = False
                    ohlcv["price_diverged"] = False
                    ohlcv["sample_weight"] = 1.0
                    frames.append(ohlcv)
                    sources_fetched.append("uniswap_subgraph")
                    log.info(f"Subgraph: {len(ohlcv):,} candles.")
            except Exception as exc:
                log.warning(f"Subgraph fetch failed: {exc}")

        # Priority 2: DeFiLlama
        if self._cfg.defillama_enabled:
            try:
                log.info("Fetching from DeFiLlama...")
                df_llama = await fetch_defillama_prices(start_ts=self._start_ts)
                if not df_llama.empty:
                    df_llama["cex_only"] = False
                    df_llama["price_diverged"] = False
                    df_llama["sample_weight"] = 1.0
                    frames.append(df_llama)
                    sources_fetched.append("defillama")
                    log.info(f"DeFiLlama: {len(df_llama):,} candles.")
            except Exception as exc:
                log.warning(f"DeFiLlama fetch failed: {exc}")

        # Priority 3: Binance
        if self._cfg.binance_fallback_enabled:
            onchain_df = frames[0] if frames else None
            try:
                log.info("Fetching from Binance...")
                df_binance = await fetch_binance_ohlcv(
                    start_ms=self._start_ts * 1000,
                    onchain_df=onchain_df,
                )
                if not df_binance.empty:
                    frames.append(df_binance)
                    sources_fetched.append("binance")
                    log.info(f"Binance: {len(df_binance):,} candles.")
            except Exception as exc:
                log.warning(f"Binance fetch failed: {exc}")

        if not frames:
            return pd.DataFrame()

        # Merge: keep highest-priority source per timestamp
        combined = pd.concat(frames, ignore_index=True)
        combined["_priority"] = combined["source"].map(
            lambda s: _SOURCE_PRIORITY.get(s, 99)
        )
        combined = combined.sort_values(["timestamp", "_priority"])
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
        combined = combined.drop(columns=["_priority"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        log.info(
            f"_fetch_all_sources: {len(combined):,} unique candles from "
            f"{sources_fetched}."
        )
        return combined

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    async def _normalize_to_5m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure index is a complete 5-min grid; no gaps introduced here."""
        df = df.copy()
        df["timestamp"] = df["timestamp"].astype(int)

        # Align timestamps to 5-min boundaries
        df["timestamp"] = (df["timestamp"] // _5MIN_SECONDS) * _5MIN_SECONDS
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return df

    async def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing 5-min candles according to gap-size rules.

        Rules:
          1–6  missing candles → forward fill
          7–12 missing candles → linear interpolation on close
          13+  missing candles → mark purge_zone=True
        """
        if df.empty:
            return df

        df = df.copy()
        df["purge_zone"] = df.get("purge_zone", pd.Series(False, index=df.index))

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        full_grid = pd.RangeIndex(
            start=ts_min, stop=ts_max + _5MIN_SECONDS, step=_5MIN_SECONDS
        )
        full_ts = pd.DataFrame({"timestamp": full_grid})

        df = full_ts.merge(df, on="timestamp", how="left")

        # Identify gap runs
        missing_mask = df["close"].isna()
        gap_start = None
        gap_groups: list[tuple[int, int]] = []  # (start_idx, end_idx inclusive)

        for i, is_missing in enumerate(missing_mask):
            if is_missing and gap_start is None:
                gap_start = i
            elif not is_missing and gap_start is not None:
                gap_groups.append((gap_start, i - 1))
                gap_start = None
        if gap_start is not None:
            gap_groups.append((gap_start, len(df) - 1))

        for start_i, end_i in gap_groups:
            gap_len = end_i - start_i + 1
            if gap_len <= _GAP_FFILL_MAX:
                # Forward fill: carry last known value
                if start_i > 0:
                    fill_val = df.iloc[start_i - 1]
                    for col in ["open", "high", "low", "close"]:
                        df.loc[start_i:end_i, col] = fill_val.get(col, np.nan)
                    df.loc[start_i:end_i, "source"] = fill_val.get("source", "ffill")
                    df.loc[start_i:end_i, "volume_usdc"] = 0.0
                    df.loc[start_i:end_i, "trade_count"] = 0.0
                    df.loc[start_i:end_i, "sample_weight"] = fill_val.get("sample_weight", 1.0)
            elif gap_len <= _GAP_INTERP_MAX:
                # Linear interpolation on close
                before = df.iloc[start_i - 1]["close"] if start_i > 0 else np.nan
                after_idx = end_i + 1
                after = df.iloc[after_idx]["close"] if after_idx < len(df) else np.nan
                if pd.notna(before) and pd.notna(after):
                    interp_closes = np.linspace(before, after, gap_len + 2)[1:-1]
                    for j, idx in enumerate(range(start_i, end_i + 1)):
                        c = interp_closes[j]
                        df.loc[idx, "open"] = c
                        df.loc[idx, "high"] = c
                        df.loc[idx, "low"] = c
                        df.loc[idx, "close"] = c
                    df.loc[start_i:end_i, "source"] = "interpolated"
                    df.loc[start_i:end_i, "volume_usdc"] = 0.0
                    df.loc[start_i:end_i, "trade_count"] = 0.0
                    df.loc[start_i:end_i, "sample_weight"] = 0.5
            else:
                # Mark as purge_zone
                df.loc[start_i:end_i, "purge_zone"] = True
                if start_i > 0:
                    fill_val = df.iloc[start_i - 1]
                    for col in ["open", "high", "low", "close"]:
                        df.loc[start_i:end_i, col] = fill_val.get(col, np.nan)
                    df.loc[start_i:end_i, "source"] = "purge_zone"
                    df.loc[start_i:end_i, "volume_usdc"] = 0.0

        # Fill any remaining NaN columns with defaults
        for col in ["cex_only", "price_diverged", "purge_zone", "stale"]:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        df["sample_weight"] = df["sample_weight"].fillna(1.0)
        df["source"] = df["source"].fillna("unknown")

        n_gaps = sum(e - s + 1 for s, e in gap_groups)
        n_purge = df["purge_zone"].sum()
        log.info(
            f"_fill_gaps: {n_gaps} missing candles processed, "
            f"{n_purge} marked purge_zone."
        )
        return df.reset_index(drop=True)

    def _flag_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark candles where |5m return| > 15% with sample_weight=0.1."""
        df = df.copy()
        df["return_5m"] = df["close"].pct_change()
        spike_mask = df["return_5m"].abs() > 0.15
        df.loc[spike_mask, "sample_weight"] = df.loc[spike_mask, "sample_weight"].clip(upper=0.1)
        n = int(spike_mask.sum())
        if n:
            log.info(f"_flag_spikes: {n} spike candles (|ret|>15%) downweighted.")
        return df

    def _flag_stale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark 3+ consecutive candles with 0% return as stale=True."""
        df = df.copy()
        df["stale"] = False
        zero_ret = (df["return_5m"].fillna(0) == 0)
        run_len = 0
        run_indices: list[int] = []

        for i, is_zero in enumerate(zero_ret):
            if is_zero:
                run_len += 1
                run_indices.append(i)
            else:
                if run_len >= 3:
                    df.loc[run_indices, "stale"] = True
                run_len = 0
                run_indices = []

        if run_len >= 3:
            df.loc[run_indices, "stale"] = True

        n = int(df["stale"].sum())
        if n:
            log.info(f"_flag_stale: {n} candles marked stale.")
        return df

    def _winsorise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip returns to [-4σ, +4σ] using a rolling 288-period σ."""
        df = df.copy()
        ret = df["return_5m"]
        rolling_std = ret.rolling(_GARCH_WINDOW, min_periods=10).std()
        upper = 4 * rolling_std
        lower = -4 * rolling_std
        df["return_5m"] = ret.clip(lower=lower, upper=upper)
        return df

    # ------------------------------------------------------------------
    # Regime labeling
    # ------------------------------------------------------------------

    async def _label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling GARCH sigma and label each candle's market regime.

        Regime criteria (applied to rolling 30-day windows):
          bull     30d return > +20% AND garch_sigma < 0.03
          bear     30d return < -20% OR rolling max_drawdown > 25%
          sideways |30d return| < 5% AND garch_sigma < 0.02
          high_vol garch_sigma > 0.04 (regardless of direction)
          crisis   30d return < -40% OR daily_vol > 0.08
        """
        df = df.copy()
        cfg = self._cfg

        # GARCH(1,1) conditional volatility
        ret = df["return_5m"].fillna(0)
        df["garch_sigma"] = _garch_sigma_series(
            ret, omega=cfg.garch_omega, alpha=cfg.garch_alpha, beta=cfg.garch_beta
        )

        # Rolling 30-day return
        df["return_30d"] = df["close"].pct_change(_CANDLES_PER_30D)

        # Rolling max drawdown over 30 days
        roll_close = df["close"].rolling(_CANDLES_PER_30D, min_periods=1)
        roll_max = roll_close.max()
        df["max_drawdown_30d"] = (df["close"] - roll_max) / roll_max.replace(0, np.nan)

        # Daily volatility: rolling std of 5-min returns over 1 day
        df["daily_vol"] = ret.rolling(_CANDLES_PER_DAY, min_periods=12).std()

        def _assign_regime(row: pd.Series) -> str:
            r30 = row.get("return_30d", 0.0) or 0.0
            gs = row.get("garch_sigma", 0.0) or 0.0
            dd = abs(row.get("max_drawdown_30d", 0.0) or 0.0)
            dv = row.get("daily_vol", 0.0) or 0.0

            # Crisis first (most severe)
            if r30 < _CRISIS_RETURN or dv > _CRISIS_DAILY_VOL:
                return "crisis"
            # High vol (direction-agnostic)
            if gs > _HIGH_VOL_SIGMA:
                return "high_vol"
            # Bear
            if r30 < _BEAR_RETURN or dd > _BEAR_DRAWDOWN:
                return "bear"
            # Bull
            if r30 > _BULL_RETURN and gs < _BULL_SIGMA:
                return "bull"
            # Sideways
            if abs(r30) < _SIDEWAYS_ABS and gs < _SIDEWAYS_SIGMA:
                return "sideways"
            # Default: high_vol catch-all
            return "high_vol"

        df["regime_label"] = df.apply(_assign_regime, axis=1)

        dist = df["regime_label"].value_counts(normalize=True).to_dict()
        log.info(f"_label_regimes: distribution = {dist}")

        # Drop helper columns not stored in DB
        df = df.drop(columns=["return_30d", "max_drawdown_30d", "daily_vol"], errors="ignore")
        return df

    # ------------------------------------------------------------------
    # Quality validation
    # ------------------------------------------------------------------

    async def _validate_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """Compute DataQualityReport from a processed DataFrame."""
        if df.empty:
            return DataQualityReport(
                total_candles=0, date_range_days=0, missing_pct=1.0,
                spike_count=0, stale_count=0, regime_distribution={},
                sources_used=[], passes_minimum=False,
            )

        total = len(df)
        ts_range = df["timestamp"].max() - df["timestamp"].min()
        date_range_days = ts_range / 86400

        expected = max(1, int(ts_range / _5MIN_SECONDS) + 1)
        missing_pct = max(0.0, 1.0 - total / expected)

        spike_count = int((df.get("return_5m", pd.Series(dtype=float)).abs() > 0.15).sum())
        stale_count = int(df.get("stale", pd.Series(False)).sum())

        regime_dist = {}
        if "regime_label" in df.columns:
            regime_dist = df["regime_label"].value_counts(normalize=True).round(4).to_dict()

        sources_used = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []

        return DataQualityReport(
            total_candles=total,
            date_range_days=round(date_range_days, 1),
            missing_pct=round(missing_pct, 4),
            spike_count=spike_count,
            stale_count=stale_count,
            regime_distribution=regime_dist,
            sources_used=sources_used,
            passes_minimum=(total >= self.MIN_REQUIRED_CANDLES),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_CREATE_CANDLES_SQL)
            await db.execute(_CREATE_META_SQL)
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            await db.commit()

    async def _count_candles(self) -> int:
        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM candles")
                row = await cursor.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    async def _save_incremental(self, df: pd.DataFrame) -> int:
        """Upsert candles into SQLite. Returns number of rows written."""
        if df.empty:
            return 0

        await self._ensure_schema()

        # Normalise boolean columns to int for SQLite
        for col in ["cex_only", "price_diverged", "purge_zone", "stale"]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        cols = [
            "timestamp", "open", "high", "low", "close",
            "volume_usdc", "trade_count", "buy_volume_usdc",
            "sell_volume_usdc", "gas_price_mean_gwei",
            "source", "cex_only", "price_diverged",
            "sample_weight", "purge_zone", "stale",
            "regime_label", "return_5m", "garch_sigma",
        ]
        # Only keep columns that exist in the DataFrame
        cols = [c for c in cols if c in df.columns]
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(cols)
        upsert_sql = (
            f"INSERT OR REPLACE INTO candles ({col_list}) VALUES ({placeholders})"
        )

        rows_data = [
            tuple(row[c] if pd.notna(row.get(c)) else None for c in cols)
            for _, row in df.iterrows()
        ]

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.executemany(upsert_sql, rows_data)
            # Update last_update meta
            await db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_update", str(int(time.time()))),
            )
            await db.commit()

        return len(rows_data)


# ---------------------------------------------------------------------------
# RegimeBalancedSampler (Section 2.5)
# ---------------------------------------------------------------------------

class RegimeBalancedSampler:
    """Weighted oversampling for regime-balanced ML training batches.

    Target distribution:
        bull:      25%
        bear:      25%
        sideways:  25%
        high_vol:  15%
        crisis:    10%

    Strategy: oversample underrepresented regimes with replacement.
    Never undersample — all data points are preserved.
    Spike rows (sample_weight=0.1) and CEX-diverged rows (sample_weight=0.5)
    are included but drawn proportionally less often.
    """

    TARGET_DISTRIBUTION: dict[str, float] = {
        "bull": 0.25,
        "bear": 0.25,
        "sideways": 0.25,
        "high_vol": 0.15,
        "crisis": 0.10,
    }

    def prepare(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: str = "regime_label",
        regime_col: str = "regime_label",
        weight_col: str = "sample_weight",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y, sample_weights) arrays for model training.

        Args:
            df: Processed candle DataFrame from load_for_training().
            feature_cols: Columns to use as features. If None, uses all
                numeric columns except label/regime/weight/timestamp.
            label_col: Column containing the prediction target.
            regime_col: Column containing regime labels for reweighting.
            weight_col: Column containing base sample weights.

        Returns:
            (X, y, sample_weights) as numpy arrays.
        """
        if feature_cols is None:
            exclude = {label_col, regime_col, weight_col, "timestamp",
                       "purge_zone", "stale", "source", "cex_only",
                       "price_diverged"}
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        df = df.dropna(subset=feature_cols + [label_col]).copy()

        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy()
        base_weights = df[weight_col].fillna(1.0).to_numpy(dtype=np.float64)

        # Compute regime-rebalancing multiplier
        regime_multiplier = np.ones(len(df), dtype=np.float64)
        if regime_col in df.columns:
            regime_counts = df[regime_col].value_counts()
            total = len(df)
            for regime, target_frac in self.TARGET_DISTRIBUTION.items():
                if regime not in regime_counts:
                    continue
                actual_frac = regime_counts[regime] / total
                if actual_frac > 0:
                    mult = target_frac / actual_frac
                    mask = df[regime_col] == regime
                    regime_multiplier[mask.to_numpy()] = mult

        sample_weights = base_weights * regime_multiplier

        # Normalise so weights sum to len(df)
        total_w = sample_weights.sum()
        if total_w > 0:
            sample_weights = sample_weights / total_w * len(df)

        log.info(
            f"RegimeBalancedSampler.prepare(): X={X.shape}, "
            f"features={feature_cols[:5]}..., "
            f"regime_dist={df[regime_col].value_counts(normalize=True).to_dict()}"
        )
        return X, y, sample_weights


# ---------------------------------------------------------------------------
# IncrementalDataUpdater (Section 2.6)
# ---------------------------------------------------------------------------

class IncrementalDataUpdater:
    """Nightly async updater — runs inside the orchestrator process.

    Wakes at cfg.incremental_update_hour_utc each night. Fetches only
    candles newer than max(timestamp) in the DB, applies the full
    preprocessing pipeline, and triggers retraining when enough new
    data has accumulated.
    """

    def __init__(self, pipeline: Optional[HistoricalDataPipeline] = None) -> None:
        from wallet_manager.config import get_settings
        self._cfg = get_settings()
        self._pipeline = pipeline or HistoricalDataPipeline()
        self._last_retrain_ts: float = 0.0

    async def run_forever(self) -> None:
        """Infinite async loop — cancelled cleanly on orchestrator shutdown."""
        log.info(
            f"IncrementalDataUpdater started. "
            f"Nightly update at {self._cfg.incremental_update_hour_utc:02d}:00 UTC."
        )
        while True:
            try:
                await self._wait_until_update_hour()
                await self._run_once()
            except asyncio.CancelledError:
                log.info("IncrementalDataUpdater cancelled — shutting down.")
                return
            except Exception as exc:
                log.exception(f"IncrementalDataUpdater error: {exc}. Retrying tomorrow.")
                await asyncio.sleep(3600)

    async def _wait_until_update_hour(self) -> None:
        """Sleep until the configured UTC hour."""
        now = datetime.now(timezone.utc)
        target_hour = self._cfg.incremental_update_hour_utc
        next_run = now.replace(
            hour=target_hour, minute=0, second=0, microsecond=0
        )
        if now.hour >= target_hour:
            # Already past today's window — wait until tomorrow
            next_run = next_run.replace(day=now.day + 1)

        wait_s = (next_run - now).total_seconds()
        log.info(f"IncrementalDataUpdater: sleeping {wait_s / 3600:.1f}h until next update.")
        await asyncio.sleep(max(0, wait_s))

    async def _run_once(self) -> None:
        """Fetch incremental candles, preprocess, and upsert."""
        cfg = self._cfg
        db_path = self._pipeline._db_path

        # Step 1: Query max(timestamp) from DB
        max_ts = await self._get_max_timestamp(db_path)
        if max_ts == 0:
            log.warning("IncrementalDataUpdater: DB empty, running full pipeline.")
            await self._pipeline.run()
            self._last_retrain_ts = time.time()
            return

        log.info(
            f"IncrementalDataUpdater: fetching candles after "
            f"{datetime.utcfromtimestamp(max_ts).isoformat()}Z"
        )

        # Step 2: Fetch only newer candles
        from scripts.data.fetch_uniswap_subgraph import fetch_swap_history, swaps_to_ohlcv
        try:
            swaps = await fetch_swap_history(start_ts=max_ts + 1)
            if swaps.empty:
                log.info("IncrementalDataUpdater: no new swaps found.")
                return
            new_df = swaps_to_ohlcv(swaps)
            new_df["source"] = "uniswap_subgraph"
            new_df["cex_only"] = False
            new_df["price_diverged"] = False
            new_df["sample_weight"] = 1.0
        except Exception as exc:
            log.warning(f"IncrementalDataUpdater: subgraph fetch failed: {exc}")
            return

        # Step 3: Apply preprocessing
        new_df = await self._pipeline._normalize_to_5m(new_df)
        new_df = await self._pipeline._fill_gaps(new_df)
        new_df = self._pipeline._flag_spikes(new_df)
        new_df = self._pipeline._flag_stale(new_df)
        new_df = self._pipeline._winsorise(new_df)
        new_df = await self._pipeline._label_regimes(new_df)

        # Step 4: Upsert
        saved = await self._pipeline._save_incremental(new_df)
        log.info(f"IncrementalDataUpdater: {saved} new candles upserted.")

        # Step 5: Refresh quality report
        full_df = await self._pipeline.load_for_training()
        report = await self._pipeline._validate_quality(full_df)
        report.log_summary()

        # Step 6: Trigger retraining if enough new data
        days_since_retrain = (time.time() - self._last_retrain_ts) / 86400
        if (
            saved >= cfg.min_new_candles_for_retrain
            and days_since_retrain >= cfg.min_days_between_retrains
        ):
            log.info(
                f"IncrementalDataUpdater: {saved} new candles AND "
                f"{days_since_retrain:.1f}d since last retrain — triggering RetrainController."
            )
            await self._trigger_retrain(full_df)
            self._last_retrain_ts = time.time()

    async def _get_max_timestamp(self, db_path: Path) -> int:
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT MAX(timestamp) FROM candles")
                row = await cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    async def _trigger_retrain(self, df: pd.DataFrame) -> None:
        """Placeholder — RetrainController is wired in Section 5."""
        log.info(
            f"RetrainController trigger: {len(df):,} training candles available."
        )
        # Section 5 will replace this with the actual RetrainController call.
