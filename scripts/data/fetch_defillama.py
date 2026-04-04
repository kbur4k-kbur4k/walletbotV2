"""
scripts/data/fetch_defillama.py

Fetches ETH/USDC price history from the DeFiLlama Coins API.
No API key required. Returns 5-min OHLCV (upsampled from 1h).
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

# WETH on Base
_DEFILLAMA_TOKEN = "base:0x4200000000000000000000000000000000000006"
_DEFILLAMA_BASE_URL = "https://coins.llama.fi/chart/{token}"


async def fetch_defillama_prices(
    start_ts: Optional[int] = None,
    span_hours: int = 26_280,   # 3 years in hours
    period: str = "1h",
) -> pd.DataFrame:
    """Fetch hourly OHLC from DeFiLlama and upsample to 5-min candles.

    Args:
        start_ts: Unix timestamp to start from. Defaults to 3 years ago.
        span_hours: Number of hours of data to request.
        period: DeFiLlama period string (default "1h").

    Returns:
        DataFrame with columns:
            timestamp (unix int), open, high, low, close,
            volume_usdc (NaN), trade_count (NaN), source="defillama"
    """
    if start_ts is None:
        start_ts = int(time.time()) - 3 * 365 * 24 * 3600

    url = _DEFILLAMA_BASE_URL.format(token=_DEFILLAMA_TOKEN)
    params = {
        "start": start_ts,
        "span": span_hours,
        "period": period,
    }

    timeout = aiohttp.ClientTimeout(total=120)
    backoff = 2.0
    body = None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(6):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status in (429, 503):
                        wait = backoff * (2**attempt)
                        log.warning(
                            f"DeFiLlama HTTP {resp.status} — backing off {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    body = await resp.json(content_type=None)
                break
            except aiohttp.ClientError as exc:
                wait = backoff * (2**attempt)
                log.warning(f"DeFiLlama request error: {exc} — retry in {wait:.1f}s")
                await asyncio.sleep(wait)

    if body is None:
        log.error("fetch_defillama_prices: all retries exhausted.")
        return pd.DataFrame()

    prices = body.get("coins", {}).get(_DEFILLAMA_TOKEN, {}).get("prices", [])
    if not prices:
        log.warning("fetch_defillama_prices: empty prices array in response.")
        return pd.DataFrame()

    # Build hourly DataFrame
    records = [{"timestamp": int(p["timestamp"]), "close": float(p["price"])} for p in prices]
    df = pd.DataFrame(records).drop_duplicates("timestamp").sort_values("timestamp")

    # DeFiLlama returns only close price — synthesise OHLC from consecutive closes
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume_usdc"] = np.nan
    df["trade_count"] = np.nan
    df["source"] = "defillama"

    # Upsample 1h -> 5min by forward-filling within each hour
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("dt")

    # Reindex to 5-min grid
    freq_5m = pd.tseries.frequencies.to_offset("5min")
    start_dt = df.index.min().floor("1h")
    end_dt = df.index.max().ceil("1h")
    idx_5m = pd.date_range(start=start_dt, end=end_dt, freq=freq_5m, tz="UTC")

    df_5m = df[["open", "high", "low", "close", "volume_usdc", "trade_count", "source"]].reindex(idx_5m)
    df_5m = df_5m.ffill()

    df_5m.index.name = "dt"
    df_5m = df_5m.reset_index()
    df_5m["timestamp"] = df_5m["dt"].astype("int64") // 10**9

    cols = [
        "timestamp", "open", "high", "low", "close",
        "volume_usdc", "trade_count", "source",
    ]
    result = df_5m[cols].dropna(subset=["close"]).reset_index(drop=True)
    log.info(f"fetch_defillama_prices: {len(result)} 5-min candles returned.")
    return result
