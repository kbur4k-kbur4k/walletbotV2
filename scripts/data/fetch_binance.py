"""
scripts/data/fetch_binance.py

Fetches ETHUSDT 5-min OHLCV from Binance public REST API.
No API key required. Marks rows that diverge >0.5% from on-chain data.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_5MIN_MS = 5 * 60 * 1000  # 5 minutes in milliseconds


async def fetch_binance_ohlcv(
    start_ms: Optional[int] = None,
    symbol: str = "ETHUSDT",
    interval: str = "5m",
    limit: int = 1000,
    onchain_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Paginate Binance klines until present and return 5-min OHLCV.

    Args:
        start_ms: Start time in milliseconds. Defaults to 3 years ago.
        symbol: Binance trading pair (default "ETHUSDT").
        interval: Kline interval (default "5m").
        limit: Candles per request (max 1000).
        onchain_df: Optional on-chain OHLCV DataFrame for divergence check.
            Must have 'timestamp' (unix s) and 'close' columns.

    Returns:
        DataFrame with columns:
            timestamp (unix int), open, high, low, close, volume_usdc,
            source="binance", cex_only=True,
            price_diverged (bool), sample_weight (float)
    """
    if start_ms is None:
        start_ms = int(time.time() - 3 * 365 * 24 * 3600) * 1000

    now_ms = int(time.time() * 1000)
    all_rows: list[dict] = []
    current_ms = start_ms
    backoff = 2.0

    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while current_ms < now_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_ms,
                "limit": limit,
            }

            for attempt in range(6):
                try:
                    async with session.get(_BINANCE_KLINES_URL, params=params) as resp:
                        if resp.status in (429, 418):
                            wait = backoff * (2**attempt)
                            log.warning(
                                f"Binance rate-limited (HTTP {resp.status}) — "
                                f"backing off {wait:.1f}s"
                            )
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        klines = await resp.json()
                    break
                except aiohttp.ClientError as exc:
                    wait = backoff * (2**attempt)
                    log.warning(f"Binance request error: {exc} — retry in {wait:.1f}s")
                    await asyncio.sleep(wait)
            else:
                log.error("fetch_binance_ohlcv: max retries exhausted, stopping.")
                break

            if not klines:
                break

            for k in klines:
                # Binance kline: [open_time, open, high, low, close, volume, ...]
                open_time_ms = int(k[0])
                all_rows.append({
                    "timestamp": open_time_ms // 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume_usdc": float(k[7]),  # quote asset volume = USDT ≈ USDC
                    "source": "binance",
                    "cex_only": True,
                    "price_diverged": False,
                    "sample_weight": 1.0,
                })

            last_open_ms = int(klines[-1][0])
            current_ms = last_open_ms + _5MIN_MS

            if len(klines) < limit:
                break

            log.debug(
                f"Binance fetch: {len(all_rows)} candles so far, "
                f"up to {pd.Timestamp(last_open_ms, unit='ms', tz='UTC')}"
            )

            # Throttle slightly to respect rate limits
            await asyncio.sleep(0.25)

    if not all_rows:
        log.warning("fetch_binance_ohlcv: returned 0 rows.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    log.info(f"fetch_binance_ohlcv: {len(df)} candles fetched for {symbol}.")

    # Divergence check against on-chain data
    if onchain_df is not None and not onchain_df.empty:
        df = _mark_price_divergence(df, onchain_df)

    return df


def _mark_price_divergence(
    binance_df: pd.DataFrame,
    onchain_df: pd.DataFrame,
    threshold_pct: float = 0.005,
) -> pd.DataFrame:
    """Mark Binance rows that diverge >0.5% from on-chain close prices.

    Diverged rows are kept but get sample_weight=0.5 for training.

    Args:
        binance_df: Binance OHLCV DataFrame with 'timestamp' and 'close'.
        onchain_df: On-chain OHLCV DataFrame with 'timestamp' and 'close'.
        threshold_pct: Fractional divergence threshold (default 0.005 = 0.5%).

    Returns:
        binance_df with 'price_diverged' and 'sample_weight' columns updated.
    """
    onchain_lookup = onchain_df.set_index("timestamp")["close"]

    def _check_row(row: pd.Series) -> tuple[bool, float]:
        ts = row["timestamp"]
        if ts not in onchain_lookup.index:
            return False, 1.0
        onchain_close = onchain_lookup[ts]
        if onchain_close == 0:
            return False, 1.0
        divergence = abs(row["close"] - onchain_close) / onchain_close
        if divergence > threshold_pct:
            return True, 0.5
        return False, 1.0

    results = binance_df.apply(_check_row, axis=1, result_type="expand")
    results.columns = ["price_diverged", "sample_weight"]
    binance_df = binance_df.copy()
    binance_df["price_diverged"] = results["price_diverged"]
    binance_df["sample_weight"] = results["sample_weight"]

    n_diverged = binance_df["price_diverged"].sum()
    log.info(
        f"Divergence check: {n_diverged}/{len(binance_df)} candles "
        f"diverged > {threshold_pct:.1%} from on-chain."
    )
    return binance_df
