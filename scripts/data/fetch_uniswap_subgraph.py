"""
scripts/data/fetch_uniswap_subgraph.py

Fetches WETH/USDC swap history from the Uniswap V3 subgraph on Base.
Produces raw swap-level data and resamples it to OHLCV candles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd

log = logging.getLogger(__name__)

# Pool address for WETH/USDC 0.05% on Base
WETH_USDC_POOL_BASE = "0x4C36388bE6F416A29C8d8Eee81C771cE6bE14B5"

_GRAPHQL_QUERY = """
query GetSwaps($pool: String!, $first: Int!, $startTime: Int!, $skip: Int!) {
  swaps(
    where: { pool: $pool, timestamp_gte: $startTime }
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
  ) {
    timestamp
    amount0
    amount1
    sqrtPriceX96
    tick
    logIndex
    transaction { gasUsed gasPrice }
  }
}
"""

_CHECKPOINT_PATH = Path("historical_data_checkpoint.json")


def _sqrtprice_to_usd(sqrtPriceX96: str) -> float:
    """Convert sqrtPriceX96 (string) to WETH price in USD.

    price = (sqrtPriceX96 / 2**96)**2 * (10**12)
    The 10**12 factor corrects for WETH (18 decimals) vs USDC (6 decimals).
    """
    val = int(sqrtPriceX96)
    price = (val / (2**96)) ** 2 * (10**12)
    return price


def _load_checkpoint() -> dict:
    if _CHECKPOINT_PATH.exists():
        try:
            return json.loads(_CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(data: dict) -> None:
    _CHECKPOINT_PATH.write_text(json.dumps(data))


async def fetch_swap_history(
    start_ts: Optional[int] = None,
    batch_size: int = 1000,
    max_batches: int = 10_000,
    subgraph_url: Optional[str] = None,
    pool: str = WETH_USDC_POOL_BASE,
) -> pd.DataFrame:
    """Paginated fetch of swap history from the Uniswap V3 subgraph.

    Args:
        start_ts: Unix timestamp to start from. Defaults to 3 years ago.
        batch_size: Swaps per GraphQL page (max 1000).
        max_batches: Hard ceiling on total pages to prevent runaway loops.
        subgraph_url: Override endpoint; reads cfg.subgraph_url if None.
        pool: Pool address to query.

    Returns:
        DataFrame with columns:
            timestamp, price_usd, amount0, amount1,
            gas_used, gas_price_gwei, log_index, source
    """
    from wallet_manager.config import get_settings
    cfg = get_settings()

    if subgraph_url is None:
        subgraph_url = cfg.subgraph_url
    if start_ts is None:
        start_ts = int(time.time()) - 3 * 365 * 24 * 3600

    checkpoint = _load_checkpoint()
    resume_ts: int = checkpoint.get("last_timestamp", start_ts)
    rows: list[dict] = checkpoint.get("rows", [])
    batch_num = 0

    log.info(
        f"fetch_swap_history: start_ts={resume_ts}, url={subgraph_url}, "
        f"pool={pool[:10]}..."
    )

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        skip = 0
        current_start = resume_ts

        while batch_num < max_batches:
            variables = {
                "pool": pool.lower(),
                "first": batch_size,
                "startTime": current_start,
                "skip": skip,
            }
            payload = {"query": _GRAPHQL_QUERY, "variables": variables}

            # Exponential backoff on rate-limit / server errors
            backoff = 2.0
            for attempt in range(6):
                try:
                    async with session.post(subgraph_url, json=payload) as resp:
                        if resp.status in (429, 503):
                            wait = backoff * (2**attempt)
                            log.warning(
                                f"Subgraph HTTP {resp.status} — backing off {wait:.1f}s"
                            )
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        body = await resp.json()
                    break
                except aiohttp.ClientError as exc:
                    wait = backoff * (2**attempt)
                    log.warning(f"Subgraph request error: {exc} — retry in {wait:.1f}s")
                    await asyncio.sleep(wait)
            else:
                log.error("fetch_swap_history: max retries exhausted, stopping.")
                break

            swaps = (body.get("data") or {}).get("swaps", [])
            if not swaps:
                log.info(f"fetch_swap_history: no more swaps after batch {batch_num}.")
                break

            for s in swaps:
                try:
                    price = _sqrtprice_to_usd(s["sqrtPriceX96"])
                    tx = s.get("transaction") or {}
                    gas_used = int(tx.get("gasUsed") or 0)
                    gas_price_wei = int(tx.get("gasPrice") or 0)
                    rows.append({
                        "timestamp": int(s["timestamp"]),
                        "price_usd": price,
                        "amount0": float(s["amount0"]),
                        "amount1": float(s["amount1"]),
                        "gas_used": gas_used,
                        "gas_price_gwei": gas_price_wei / 1e9,
                        "log_index": int(s.get("logIndex") or 0),
                        "source": "uniswap_subgraph",
                    })
                except (KeyError, ValueError, TypeError) as exc:
                    log.debug(f"Skipping malformed swap: {exc}")

            batch_num += 1
            log.debug(
                f"Batch {batch_num}: fetched {len(swaps)} swaps "
                f"(total rows so far: {len(rows)})"
            )

            # Checkpoint every 100 batches
            if batch_num % 100 == 0:
                last_ts = int(swaps[-1]["timestamp"]) if swaps else current_start
                _save_checkpoint({"last_timestamp": last_ts, "rows": rows})
                log.info(f"Checkpoint saved at batch {batch_num}, ts={last_ts}.")

            # Advance pagination: if fewer than batch_size returned, we're done
            if len(swaps) < batch_size:
                break

            # Move skip forward; reset when it hits 5000 (subgraph limit)
            skip += batch_size
            if skip >= 5000:
                # Jump start_ts to last seen timestamp to avoid skip ceiling
                current_start = int(swaps[-1]["timestamp"])
                skip = 0

    # Clear checkpoint on successful completion
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()

    if not rows:
        log.warning("fetch_swap_history: returned 0 rows.")
        return pd.DataFrame(columns=[
            "timestamp", "price_usd", "amount0", "amount1",
            "gas_used", "gas_price_gwei", "log_index", "source",
        ])

    df = pd.DataFrame(rows)
    # Deduplicate by (timestamp, log_index)
    df = df.drop_duplicates(subset=["timestamp", "log_index"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"fetch_swap_history: {len(df)} unique swaps fetched.")
    return df


def swaps_to_ohlcv(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Resample swap-level data into OHLCV candles.

    Args:
        df: Raw swap DataFrame from fetch_swap_history().
        freq: Pandas offset alias for candle width (default "5min").

    Returns:
        DataFrame with columns:
            timestamp (unix int), open, high, low, close (price_usd),
            volume_usdc, trade_count, buy_volume_usdc, sell_volume_usdc,
            gas_price_mean_gwei
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("dt").sort_index()

    # Volume = |amount1| (USDC); sign determines direction
    df["abs_amount1"] = df["amount1"].abs()
    df["buy_usdc"] = df["amount1"].where(df["amount1"] < 0, 0).abs()
    df["sell_usdc"] = df["amount1"].where(df["amount1"] > 0, 0)

    ohlcv = df["price_usd"].resample(freq).ohlc()
    ohlcv.columns = ["open", "high", "low", "close"]

    volume = df["abs_amount1"].resample(freq).sum().rename("volume_usdc")
    count = df["price_usd"].resample(freq).count().rename("trade_count")
    buy_vol = df["buy_usdc"].resample(freq).sum().rename("buy_volume_usdc")
    sell_vol = df["sell_usdc"].resample(freq).sum().rename("sell_volume_usdc")
    gas_mean = df["gas_price_gwei"].resample(freq).mean().rename("gas_price_mean_gwei")

    result = pd.concat([ohlcv, volume, count, buy_vol, sell_vol, gas_mean], axis=1)
    result = result.dropna(subset=["close"])
    result.index.name = "dt"
    result = result.reset_index()
    result["timestamp"] = result["dt"].astype("int64") // 10**9
    result = result.drop(columns=["dt"])

    cols = [
        "timestamp", "open", "high", "low", "close",
        "volume_usdc", "trade_count", "buy_volume_usdc",
        "sell_volume_usdc", "gas_price_mean_gwei",
    ]
    return result[cols].reset_index(drop=True)
