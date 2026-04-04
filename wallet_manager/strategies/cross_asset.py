"""
wallet_manager/strategies/cross_asset.py

Sprint JS-4.1: Cross-Asset Signal Integration
===============================================
BTC is a leading indicator for ETH ~80% of the time.
Adding cross-asset features to the ML feature vector improves signal quality.

Features provided:
  - btc_momentum_1h     : BTC 1h price change (leading signal for ETH)
  - eth_btc_ratio_dev   : ETH/BTC ratio deviation from 20-period MA (mean reversion)
  - eth_btc_corr_30     : Rolling 30-period ETH/BTC correlation (signal reliability)
  - btc_dominance_proxy : BTC price momentum sign agrees with ETH direction?

Data source: free CoinGecko API (no API key required) with local caching.
Gracefully returns zeros when data is unavailable.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

_COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
_FETCH_INTERVAL_S = 60.0       # Fetch BTC price at most once per minute
_HISTORY_MAXLEN = 200


class CrossAssetFeatures:
    """
    Fetches BTC price and computes cross-asset features for the ML model.
    All features are dimensionless scalars in [-1, +1] range for model compatibility.
    """

    def __init__(self) -> None:
        self._btc_history: deque[float] = deque(maxlen=_HISTORY_MAXLEN)
        self._eth_history: deque[float] = deque(maxlen=_HISTORY_MAXLEN)
        self._last_fetch: float = 0.0
        self._last_btc_price: float = 0.0
        self._fetch_lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._fetch_lock is None:
            self._fetch_lock = asyncio.Lock()
        return self._fetch_lock

    async def fetch_btc_price(self) -> float:
        """Fetch BTC price from CoinGecko. Returns cached value if recent."""
        now = time.time()
        if now - self._last_fetch < _FETCH_INTERVAL_S and self._last_btc_price > 0:
            return self._last_btc_price
        async with self._get_lock():
            if now - self._last_fetch < _FETCH_INTERVAL_S and self._last_btc_price > 0:
                return self._last_btc_price
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session, session.get(
                    _COINGECKO_URL,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                btc = float(data["bitcoin"]["usd"])
                self._last_btc_price = btc
                self._last_fetch = time.time()
                log.debug(f"CrossAsset: BTC price = ${btc:.2f}")
                return btc
            except Exception as exc:
                log.debug(f"CrossAsset: BTC fetch failed: {exc}")
                return self._last_btc_price or 0.0

    def record_prices(self, eth_price: float, btc_price: float) -> None:
        """Record current ETH and BTC prices for feature computation."""
        if eth_price > 0:
            self._eth_history.append(eth_price)
        if btc_price > 0:
            self._btc_history.append(btc_price)

    def get_features(self) -> dict[str, float]:
        """
        Return dict of cross-asset features, all clipped to [-1, +1].
        Returns zeros when insufficient data.
        """
        features = {
            "btc_momentum_1h": 0.0,
            "eth_btc_ratio_dev": 0.0,
            "eth_btc_corr_30": 0.0,
            "btc_dominance_signal": 0.0,
        }
        n_btc = len(self._btc_history)
        n_eth = len(self._eth_history)
        if n_btc < 2 or n_eth < 2:
            return features

        # 1. BTC 1h momentum (proxy: last vs 60-period ago, clipped)
        lookback = min(60, n_btc - 1)
        btc_list = list(self._btc_history)
        btc_mom = (btc_list[-1] - btc_list[-lookback - 1]) / btc_list[-lookback - 1] if btc_list[-lookback - 1] > 0 else 0.0
        features["btc_momentum_1h"] = max(-1.0, min(1.0, btc_mom * 10))

        # 2. ETH/BTC ratio deviation from 20-period MA
        n = min(n_btc, n_eth, 20)
        eth_list = list(self._eth_history)
        ratios = [eth_list[-i - 1] / btc_list[-i - 1] for i in range(n) if btc_list[-i - 1] > 0]
        if len(ratios) >= 5:
            ma = sum(ratios) / len(ratios)
            current_ratio = eth_list[-1] / btc_list[-1] if btc_list[-1] > 0 else ma
            ratio_dev = (current_ratio - ma) / ma if ma > 0 else 0.0
            features["eth_btc_ratio_dev"] = max(-1.0, min(1.0, ratio_dev * 20))

        # 3. Rolling 30-period correlation between ETH and BTC returns
        n_corr = min(n_btc - 1, n_eth - 1, 30)
        if n_corr >= 10:
            btc_rets = [(btc_list[-i - 1] - btc_list[-i - 2]) / btc_list[-i - 2]
                        for i in range(n_corr) if btc_list[-i - 2] > 0]
            eth_rets = [(eth_list[-i - 1] - eth_list[-i - 2]) / eth_list[-i - 2]
                        for i in range(n_corr) if eth_list[-i - 2] > 0]
            n_min = min(len(btc_rets), len(eth_rets))
            if n_min >= 10:
                features["eth_btc_corr_30"] = _pearson(btc_rets[:n_min], eth_rets[:n_min])

        # 4. BTC dominance signal: does BTC momentum agree with ETH direction?
        btc_dir = 1 if btc_mom > 0 else -1 if btc_mom < 0 else 0
        eth_mom = (eth_list[-1] - eth_list[-min(60, n_eth - 1) - 1]) / eth_list[-min(60, n_eth - 1) - 1] if n_eth > 1 else 0.0
        eth_dir = 1 if eth_mom > 0 else -1 if eth_mom < 0 else 0
        features["btc_dominance_signal"] = float(btc_dir * eth_dir)  # +1 if agree, -1 if diverge

        return features

    def get_snapshot(self) -> dict[str, Any]:
        features = self.get_features()
        return {
            "last_btc_price": self._last_btc_price,
            "n_btc_history": len(self._btc_history),
            "n_eth_history": len(self._eth_history),
            "features": features,
            "last_fetch_ago_s": round(time.time() - self._last_fetch, 0) if self._last_fetch else None,
        }


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (dx * dy) if dx * dy > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# JS-4.3  Funding Rate Signal
# ---------------------------------------------------------------------------

_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT&limit=8"


class FundingRateSignal:
    """
    Fetches ETH perpetuals funding rate from Binance (free, no API key).
    Positive funding = longs paying shorts = over-leveraged longs → bearish.
    Negative funding = shorts paying longs = over-leveraged shorts → bullish.
    Returns normalized signal in [-1, +1].
    """

    def __init__(self) -> None:
        self._history: deque[float] = deque(maxlen=24)   # last 24 funding periods
        self._last_fetch: float = 0.0
        self._last_rate: float = 0.0
        self._fetch_interval_s = 480.0   # 8h funding periods, fetch every 8 min for freshness

    async def fetch(self) -> float:
        """Fetch latest funding rate. Returns normalized [-1, +1] signal."""
        now = time.time()
        if now - self._last_fetch < self._fetch_interval_s and self._last_rate != 0.0:
            return self._normalize(self._last_rate)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session, session.get(
                _BINANCE_FUNDING_URL,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
            if data and isinstance(data, list):
                latest = float(data[-1]["fundingRate"])
                self._last_rate = latest
                self._history.append(latest)
                self._last_fetch = now
                return self._normalize(latest)
        except Exception as exc:
            log.debug(f"FundingRate: fetch failed: {exc}")
        return self._normalize(self._last_rate)

    @staticmethod
    def _normalize(rate: float) -> float:
        """Normalize funding rate to [-1, +1]. ±0.05%/8h is typical extreme."""
        return max(-1.0, min(1.0, rate / 0.0005))

    def get_signal(self) -> dict[str, Any]:
        """Return current funding rate snapshot."""
        rate = self._last_rate
        norm = self._normalize(rate)
        sentiment = "BEARISH" if norm > 0.3 else "BULLISH" if norm < -0.3 else "NEUTRAL"
        return {
            "raw_rate": round(rate * 100, 6),    # as % e.g. 0.01 = 0.01%
            "normalized_signal": round(norm, 4),
            "sentiment": sentiment,
            "last_fetch_ago_s": round(time.time() - self._last_fetch) if self._last_fetch else None,
            "avg_24h_rate_pct": round(sum(self._history) / len(self._history) * 100, 6) if self._history else 0.0,
        }


_funding: FundingRateSignal | None = None


def get_funding_rate_signal() -> FundingRateSignal:
    global _funding
    if _funding is None:
        _funding = FundingRateSignal()
    return _funding


_cross_asset: CrossAssetFeatures | None = None


def get_cross_asset() -> CrossAssetFeatures:
    global _cross_asset
    if _cross_asset is None:
        _cross_asset = CrossAssetFeatures()
    return _cross_asset
