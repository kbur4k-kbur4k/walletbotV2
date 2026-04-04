"""
wallet_manager/strategies/signal_intelligence.py

QUANT-X Signal Intelligence — The "Alpha" Generator.

Institutional-grade signal engine based on Statistical Mean Reversion
and Market Microstructure Analysis. No lagging indicators (RSI, MACD).

Signal Components
-----------------
1. Z-Score 2.0 (σ-Pricing): Real-time price deviation relative to a
   24-hour rolling window. Entry at σ < -2.8, Exit at σ > 0.

2. Order Flow Imbalance (OFI): Monitor Uniswap V3 pool tick depth.
   If ask-side liquidity is significantly thinner than bid-side at
   the current tick, flag "Bullish Momentum".

3. VWAP (Volume Weighted Average Price): Computed from on-chain swap
   events. Only enter longs when price is significantly below VWAP
   to ensure institutional-level entry prices.

4. Composite Alpha Score: Weighted combination of all three signals
   produces a single [-1, +1] score. -1 = max short, +1 = max long.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Z-Score 2.0 parameters
ZSCORE_WINDOW_SIZE = 288          # 24h at 5-min intervals
ZSCORE_ENTRY_THRESHOLD = -2.8    # σ below mean → strong buy
ZSCORE_EXIT_THRESHOLD = 0.0      # Reverted to mean → exit
ZSCORE_SHORT_THRESHOLD = 2.8     # σ above mean → short signal

# OFI parameters
OFI_TICK_RANGE = 100              # Ticks around current price to scan
OFI_BULLISH_RATIO = 1.5          # Bid depth must be 1.5x Ask for bullish flag
OFI_BEARISH_RATIO = 0.67         # Inverse for bearish

# VWAP parameters
VWAP_WINDOW_TRADES = 500         # Rolling window of recent trades for VWAP
VWAP_ENTRY_DISCOUNT = 0.005     # 0.5% below VWAP for entry qualification

# Composite weights
WEIGHT_ZSCORE = 0.45
WEIGHT_OFI = 0.25
WEIGHT_VWAP = 0.30

# Minimum data requirements
MIN_PRICES_FOR_SIGNAL = 60       # At least 5 hours of data


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeObservation:
    """Single on-chain trade observation."""

    price: float
    volume_usd: float
    timestamp: float
    is_buy: bool


@dataclass(frozen=True)
class OFIReading:
    """Order Flow Imbalance snapshot from DEX pool."""

    bid_depth_usd: float       # Total USD liquidity on bid side
    ask_depth_usd: float       # Total USD liquidity on ask side
    imbalance_ratio: float     # bid / ask (>1 = bullish)
    current_tick: int
    timestamp: float


@dataclass(frozen=True)
class SignalOutput:
    """Composite signal output from the Alpha Generator.

    Attributes:
        alpha_score: Composite score in [-1, +1]. Positive = long, negative = short.
        z_score: Current Z-Score (σ-deviation from 24h rolling mean).
        z_mean: Rolling window mean price.
        z_std: Rolling window standard deviation.
        ofi_ratio: Order flow imbalance ratio (bid/ask depth).
        ofi_signal: Normalised OFI signal in [-1, +1].
        vwap: Current Volume Weighted Average Price.
        vwap_signal: Normalised VWAP signal in [-1, +1].
        price: Current price at signal generation time.
        confidence: Signal confidence [0, 1] based on data availability.
        should_enter_long: True if composite conditions favour a long entry.
        should_enter_short: True if composite conditions favour a short entry.
        should_exit: True if position should be closed (mean reversion).
        timestamp: Signal generation timestamp.
    """

    alpha_score: float
    z_score: float
    z_mean: float
    z_std: float
    ofi_ratio: float
    ofi_signal: float
    vwap: float
    vwap_signal: float
    price: float
    confidence: float
    should_enter_long: bool
    should_enter_short: bool
    should_exit: bool
    timestamp: float


# ---------------------------------------------------------------------------
# Z-Score 2.0 Engine — Standard Deviation Pricing
# ---------------------------------------------------------------------------


class ZScoreEngine:
    """Real-time Z-Score calculator using a 24h rolling window.

    Unlike EMA-based Z-Score (which smooths away extremes), this uses
    the raw standard deviation of the full window for unbiased σ measurement.

    Entry: σ < -2.8 (price is 2.8 standard deviations below 24h mean).
    Exit:  σ > 0 (price has reverted to or above the mean).
    """

    def __init__(self, window_size: int = ZSCORE_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._prices: deque[float] = deque(maxlen=window_size)

    def ingest(self, price: float) -> None:
        """Add a new price observation to the rolling window."""
        self._prices.append(price)

    def compute(self) -> tuple[float, float, float]:
        """Compute Z-Score from the rolling window.

        Returns:
            (z_score, mean, std). Returns (0.0, 0.0, 0.0) if insufficient data.
        """
        n = len(self._prices)
        if n < 2:
            return 0.0, 0.0, 0.0

        arr = np.array(self._prices, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))  # Sample std for unbiased estimate

        if std < 1e-10:
            return 0.0, mean, 0.0

        current = float(self._prices[-1])
        z = (current - mean) / std
        return float(z), mean, std

    @property
    def data_sufficiency(self) -> float:
        """Fraction of window filled [0, 1]."""
        return min(1.0, len(self._prices) / self._window_size)

    @property
    def buffer_size(self) -> int:
        return len(self._prices)

    def get_prices(self) -> list[float]:
        """Return a copy of the price buffer."""
        return list(self._prices)


# ---------------------------------------------------------------------------
# Order Flow Imbalance (OFI) Analyser
# ---------------------------------------------------------------------------


class OFIAnalyser:
    """Uniswap V3 pool depth analyser for order flow imbalance.

    Reads tick-level liquidity from the Uniswap V3 pool to determine
    whether the bid side (buying pressure) or ask side (selling pressure)
    is dominant.

    A high bid/ask ratio suggests bullish momentum — more capital is
    positioned to buy at current levels than to sell.
    """

    def __init__(self, tick_range: int = OFI_TICK_RANGE) -> None:
        self._tick_range = tick_range
        self._readings: deque[OFIReading] = deque(maxlen=60)  # Last 60 readings

    async def read_pool_depth(
        self,
        rpc_url: str,
        pool_address: str,
    ) -> OFIReading:
        """Read bid/ask depth from Uniswap V3 pool tick data.

        Scans ticks around the current price to measure concentrated
        liquidity on each side.

        Args:
            rpc_url: RPC endpoint for on-chain reads.
            pool_address: Uniswap V3 pool contract address.

        Returns:
            OFIReading with bid/ask depth and imbalance ratio.
        """
        from web3 import AsyncWeb3, Web3
        from wallet_manager.actions import _make_w3

        w3 = _make_w3(rpc_url)

        # Use synchronous Web3.to_checksum_address for address normalization.
        # AsyncWeb3's instance .to_checksum_address raises "Unknown format" for
        # mixed-case checksummed addresses on some web3.py versions.
        checksummed_pool = Web3.to_checksum_address(pool_address)

        # Minimal pool ABI for slot0 and liquidity reads
        pool_abi = [
            {
                "inputs": [],
                "name": "slot0",
                "outputs": [
                    {"name": "sqrtPriceX96", "type": "uint160"},
                    {"name": "tick", "type": "int24"},
                    {"name": "observationIndex", "type": "uint16"},
                    {"name": "observationCardinality", "type": "uint16"},
                    {"name": "observationCardinalityNext", "type": "uint16"},
                    {"name": "feeProtocol", "type": "uint8"},
                    {"name": "unlocked", "type": "bool"},
                ],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [],
                "name": "liquidity",
                "outputs": [{"name": "", "type": "uint128"}],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [{"name": "tick", "type": "int24"}],
                "name": "ticks",
                "outputs": [
                    {"name": "liquidityGross", "type": "uint128"},
                    {"name": "liquidityNet", "type": "int128"},
                    {"name": "feeGrowthOutside0X128", "type": "uint256"},
                    {"name": "feeGrowthOutside1X128", "type": "uint256"},
                    {"name": "tickCumulativeOutside", "type": "int56"},
                    {"name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
                    {"name": "secondsOutside", "type": "uint32"},
                    {"name": "initialized", "type": "bool"},
                ],
                "stateMutability": "view",
                "type": "function",
            },
        ]

        pool = w3.eth.contract(
            address=checksummed_pool,
            abi=pool_abi,
        )

        try:
            slot0 = await pool.functions.slot0().call()
            current_tick = int(slot0[1])
            current_liquidity = await pool.functions.liquidity().call()

            # Scan ticks: below current = bid side, above current = ask side
            tick_spacing = 60  # Base WETH/USDC 0.3% pool tick spacing
            bid_liquidity = 0
            ask_liquidity = 0

            # Sample key ticks around current price
            num_ticks = min(self._tick_range // tick_spacing, 10)

            for i in range(1, num_ticks + 1):
                try:
                    # Bid side (below current tick)
                    bid_tick = current_tick - (i * tick_spacing)
                    bid_data = await pool.functions.ticks(bid_tick).call()
                    bid_liquidity += int(bid_data[0])  # liquidityGross

                    # Ask side (above current tick)
                    ask_tick = current_tick + (i * tick_spacing)
                    ask_data = await pool.functions.ticks(ask_tick).call()
                    ask_liquidity += int(ask_data[0])  # liquidityGross
                except Exception:
                    continue

            # Add current in-range liquidity proportionally
            bid_liquidity += int(current_liquidity) // 2
            ask_liquidity += int(current_liquidity) // 2

            # Normalise to USD-equivalent (rough: liquidity is in sqrt(P) units)
            bid_depth_usd = float(bid_liquidity) / 1e18
            ask_depth_usd = float(ask_liquidity) / 1e18

            ratio = bid_depth_usd / ask_depth_usd if ask_depth_usd > 0 else 1.0

            reading = OFIReading(
                bid_depth_usd=bid_depth_usd,
                ask_depth_usd=ask_depth_usd,
                imbalance_ratio=ratio,
                current_tick=current_tick,
                timestamp=time.time(),
            )
            self._readings.append(reading)
            return reading

        except Exception as exc:
            log.warning(f"OFI: pool depth read failed: {exc}")
            return OFIReading(
                bid_depth_usd=0.0,
                ask_depth_usd=0.0,
                imbalance_ratio=1.0,
                current_tick=0,
                timestamp=time.time(),
            )

    def normalise_signal(self, reading: OFIReading) -> float:
        """Convert OFI ratio to normalised signal [-1, +1].

        Mapping:
            ratio > 1.5 → +1.0 (strong bullish)
            ratio = 1.0 → 0.0  (neutral)
            ratio < 0.67 → -1.0 (strong bearish)
        """
        if reading.imbalance_ratio <= 0:
            return 0.0

        # Log-transform for symmetric scaling
        log_ratio = math.log(reading.imbalance_ratio)
        # Scale: log(1.5) ≈ 0.405, log(0.67) ≈ -0.405
        signal = log_ratio / math.log(OFI_BULLISH_RATIO)
        return max(-1.0, min(1.0, signal))

    @property
    def last_reading(self) -> OFIReading | None:
        return self._readings[-1] if self._readings else None


# ---------------------------------------------------------------------------
# VWAP Calculator
# ---------------------------------------------------------------------------


class VWAPCalculator:
    """Volume Weighted Average Price from on-chain swap events.

    VWAP = Σ(price_i × volume_i) / Σ(volume_i)

    Used as "True North" for price. Only enter longs when price is
    significantly below VWAP to ensure institutional-level entries.
    """

    def __init__(self, max_trades: int = VWAP_WINDOW_TRADES) -> None:
        self._trades: deque[TradeObservation] = deque(maxlen=max_trades)

    def ingest_trade(self, trade: TradeObservation) -> None:
        """Add a trade observation to the rolling VWAP window."""
        self._trades.append(trade)

    def ingest_from_price_volume(
        self,
        price: float,
        volume_usd: float,
        is_buy: bool = True,
    ) -> None:
        """Convenience: create and ingest a TradeObservation."""
        self._trades.append(
            TradeObservation(
                price=price,
                volume_usd=volume_usd,
                timestamp=time.time(),
                is_buy=is_buy,
            )
        )

    def compute(self) -> float:
        """Compute current VWAP.

        Returns:
            VWAP price. Returns 0.0 if no trades available.
        """
        if not self._trades:
            return 0.0

        total_pv = 0.0
        total_v = 0.0
        for t in self._trades:
            total_pv += t.price * t.volume_usd
            total_v += t.volume_usd

        if total_v <= 0:
            return 0.0

        return total_pv / total_v

    def normalise_signal(self, current_price: float) -> float:
        """Convert price-vs-VWAP to normalised signal [-1, +1].

        Mapping:
            price < VWAP × (1 - discount) → +1.0 (strong buy: below VWAP)
            price = VWAP → 0.0 (neutral)
            price > VWAP × (1 + discount) → -1.0 (strong sell: above VWAP)

        Returns:
            Signal value. 0.0 if VWAP unavailable.
        """
        vwap = self.compute()
        if vwap <= 0:
            return 0.0

        # Percentage deviation from VWAP
        deviation = (current_price - vwap) / vwap

        # Scale: ±0.5% discount maps to ±1.0
        signal = -deviation / VWAP_ENTRY_DISCOUNT
        return max(-1.0, min(1.0, signal))

    @property
    def trade_count(self) -> int:
        return len(self._trades)


# ---------------------------------------------------------------------------
# Composite Alpha Generator
# ---------------------------------------------------------------------------


class AlphaGenerator:
    """Institutional-grade composite signal generator.

    Combines Z-Score 2.0, OFI, and VWAP into a single alpha score
    that drives all trading decisions. No lagging indicators.

    The alpha score ranges from -1 (max short) to +1 (max long).
    Entry/exit decisions are made by comparing the composite score
    against threshold levels.

    Usage:
        generator = AlphaGenerator()

        # Each price tick:
        generator.ingest_price(current_price)
        generator.ingest_trade(price, volume, is_buy)

        # Each cycle:
        ofi = await generator.read_ofi(rpc_url, pool_address)
        signal = generator.generate_signal(current_price)

        if signal.should_enter_long:
            # Execute buy
        elif signal.should_enter_short:
            # Execute short
        elif signal.should_exit:
            # Close position
    """

    def __init__(
        self,
        zscore_window: int = ZSCORE_WINDOW_SIZE,
        ofi_tick_range: int = OFI_TICK_RANGE,
        vwap_max_trades: int = VWAP_WINDOW_TRADES,
    ) -> None:
        self.zscore_engine = ZScoreEngine(window_size=zscore_window)
        self.ofi_analyser = OFIAnalyser(tick_range=ofi_tick_range)
        self.vwap_calculator = VWAPCalculator(max_trades=vwap_max_trades)

    def ingest_price(self, price: float) -> None:
        """Ingest a price observation into the Z-Score engine."""
        self.zscore_engine.ingest(price)

    def ingest_trade(
        self,
        price: float,
        volume_usd: float,
        is_buy: bool = True,
    ) -> None:
        """Ingest a trade observation into the VWAP calculator."""
        self.vwap_calculator.ingest_from_price_volume(price, volume_usd, is_buy)

    async def read_ofi(self, rpc_url: str, pool_address: str) -> OFIReading:
        """Read current Order Flow Imbalance from the DEX pool."""
        return await self.ofi_analyser.read_pool_depth(rpc_url, pool_address)

    def generate_signal(
        self,
        current_price: float,
        ofi_reading: OFIReading | None = None,
    ) -> SignalOutput:
        """Generate composite alpha signal.

        Args:
            current_price: Current market price.
            ofi_reading: Optional pre-fetched OFI reading. If None, uses
                         last cached reading.

        Returns:
            SignalOutput with composite alpha score and individual signals.
        """
        now = time.time()

        # --- Z-Score 2.0 ---
        z_score, z_mean, z_std = self.zscore_engine.compute()
        z_data_pct = self.zscore_engine.data_sufficiency

        # Normalise Z-Score to [-1, +1]
        # z < -2.8 → +1 (buy), z > +2.8 → -1 (sell)
        z_signal = -z_score / abs(ZSCORE_ENTRY_THRESHOLD)
        z_signal = max(-1.0, min(1.0, z_signal))

        # --- OFI ---
        ofi = ofi_reading or self.ofi_analyser.last_reading
        if ofi is not None:
            ofi_signal = self.ofi_analyser.normalise_signal(ofi)
            ofi_ratio = ofi.imbalance_ratio
        else:
            ofi_signal = 0.0
            ofi_ratio = 1.0

        # --- VWAP ---
        vwap = self.vwap_calculator.compute()
        vwap_signal = self.vwap_calculator.normalise_signal(current_price)

        # --- Confidence ---
        # Based on data availability across all three signals
        z_conf = z_data_pct
        ofi_conf = 1.0 if ofi is not None else 0.0
        vwap_conf = min(1.0, self.vwap_calculator.trade_count / 50)
        confidence = (z_conf * WEIGHT_ZSCORE + ofi_conf * WEIGHT_OFI + vwap_conf * WEIGHT_VWAP)

        # --- Composite Alpha Score ---
        # Weighted sum, adjusted by individual confidence
        raw_alpha = (
            z_signal * WEIGHT_ZSCORE * z_conf
            + ofi_signal * WEIGHT_OFI * ofi_conf
            + vwap_signal * WEIGHT_VWAP * vwap_conf
        )
        # Normalise by total effective weight
        effective_weight = (
            WEIGHT_ZSCORE * z_conf
            + WEIGHT_OFI * ofi_conf
            + WEIGHT_VWAP * vwap_conf
        )
        alpha_score = raw_alpha / effective_weight if effective_weight > 0 else 0.0
        alpha_score = max(-1.0, min(1.0, alpha_score))

        # --- Entry/Exit decisions ---
        has_enough_data = self.zscore_engine.buffer_size >= MIN_PRICES_FOR_SIGNAL

        should_enter_long = (
            has_enough_data
            and alpha_score >= 0.6
            and z_score <= ZSCORE_ENTRY_THRESHOLD
            and (vwap <= 0 or current_price < vwap * (1 - VWAP_ENTRY_DISCOUNT))
        )

        should_enter_short = (
            has_enough_data
            and alpha_score <= -0.6
            and z_score >= ZSCORE_SHORT_THRESHOLD
        )

        should_exit = (
            has_enough_data
            and abs(z_score) <= 0.3
            and abs(alpha_score) < 0.2
        )

        return SignalOutput(
            alpha_score=alpha_score,
            z_score=z_score,
            z_mean=z_mean,
            z_std=z_std,
            ofi_ratio=ofi_ratio,
            ofi_signal=ofi_signal,
            vwap=vwap,
            vwap_signal=vwap_signal,
            price=current_price,
            confidence=confidence,
            should_enter_long=should_enter_long,
            should_enter_short=should_enter_short,
            should_exit=should_exit,
            timestamp=now,
        )


# ---------------------------------------------------------------------------
# CrossVenueArbitrageDetector — multi-venue spread monitoring
# ---------------------------------------------------------------------------


class CrossVenueArbitrageDetector:
    """Detects cross-venue price arbitrage opportunities.

    Maintains per-venue price histories and flags when the best/worst
    spread exceeds a configurable threshold. The signal reflects which
    direction the lagging venue is expected to move to close the gap:

        CEX price > DEX price → DEX expected to catch up → +ve signal (bullish DEX)
        DEX price > CEX price → DEX premium → -ve signal (mean-reversion pressure)

    Usage::

        arb = CrossVenueArbitrageDetector()
        arb.update("binance", 3250.0)
        arb.update("base_dex", 3242.0)
        if arb.is_arbitrage_opportunity:
            print(f"Arb: {arb.spread_pct:.3%} lead={arb.lead_venue}")
    """

    ARB_THRESHOLD_PCT: float = 0.003  # 0.3% minimum spread to flag
    PRICE_WINDOW: int = 10             # rolling window size per venue

    def __init__(self, threshold_pct: float = ARB_THRESHOLD_PCT) -> None:
        self._threshold = threshold_pct
        self._price_buffers: dict[str, deque] = {}
        self._latest: dict[str, float] = {}
        self._arb_signal: float = 0.0
        self._spread_pct: float = 0.0
        self._lead_venue: str = ""
        self._lag_venue: str = ""

    def update(self, venue: str, price: float) -> None:
        """Record a new price observation for a venue.

        Args:
            venue: Venue identifier (e.g. 'binance', 'base_dex', 'coinbase').
            price: Current mid-price for that venue.
        """
        if venue not in self._price_buffers:
            self._price_buffers[venue] = deque(maxlen=self.PRICE_WINDOW)
        self._price_buffers[venue].append(price)
        self._latest[venue] = price
        self._recompute()

    def _recompute(self) -> None:
        if len(self._latest) < 2:
            self._arb_signal = 0.0
            self._spread_pct = 0.0
            return

        prices = list(self._latest.values())
        venues = list(self._latest.keys())
        median_p = float(np.median(prices))
        if median_p <= 0:
            return

        max_idx = int(np.argmax(prices))
        min_idx = int(np.argmin(prices))
        self._spread_pct = (prices[max_idx] - prices[min_idx]) / median_p

        if self._spread_pct < self._threshold:
            self._arb_signal = 0.0
            return

        self._lead_venue = venues[max_idx]
        self._lag_venue = venues[min_idx]

        # Positive signal = lead venue was inserted after lag venue (newer venue leads)
        # Negative signal = earlier-inserted venue leads (mean-reversion pressure)
        lead_is_dex = any(k in self._lead_venue.lower() for k in ("dex", "base", "uniswap"))
        lag_is_dex = any(k in self._lag_venue.lower() for k in ("dex", "base", "uniswap"))
        if lag_is_dex and not lead_is_dex:
            # CEX leads, DEX expected to catch up → bullish for DEX
            sign = 1.0
        elif lead_is_dex and not lag_is_dex:
            # DEX premium → mean-reversion pressure → negative
            sign = -1.0
        else:
            # Generic: positive if lead venue has higher insertion index (newer venue leads)
            sign = 1.0 if max_idx > min_idx else -1.0
        # Scale: 0 at threshold, 1 at 10×threshold
        magnitude = min(1.0, (self._spread_pct - self._threshold) / max(self._threshold * 9, 1e-10))
        self._arb_signal = sign * magnitude

    @property
    def arb_signal(self) -> float:
        """Directional alpha signal in [-1, +1]."""
        return self._arb_signal

    @property
    def spread_pct(self) -> float:
        """Current best-worst spread as a fraction."""
        return self._spread_pct

    @property
    def is_arbitrage_opportunity(self) -> bool:
        return self._spread_pct >= self._threshold

    @property
    def lead_venue(self) -> str:
        return self._lead_venue

    @property
    def lag_venue(self) -> str:
        return self._lag_venue


# ---------------------------------------------------------------------------
# OnChainMetricsSignal — gas trend, whale movement, DEX volume surge
# ---------------------------------------------------------------------------


class OnChainMetricsSignal:
    """Aggregates three on-chain behavioral metrics into a composite signal.

    Gas trend:
        Moderate rise → more activity → mild bullish.
        Extreme spike → congestion / panic → mildly cautionary.

    Whale movement:
        Volume ≥ WHALE_MULT × rolling median AND directional bias → signal.
        Large buy surge → bullish; large sell surge → bearish.

    DEX volume surge:
        Volume ≥ VOLUME_SURGE_MULT × 24h average → institutional-level demand.
        Pure surge (no directional info) → positive signal.

    Composite = 0.30 × gas + 0.40 × whale + 0.30 × volume_surge ∈ [-1, +1]
    """

    GAS_WINDOW: int = 60        # bars of gas history
    GAS_SURGE_MULT: float = 2.5  # ratio above which gas is "spiking"
    WHALE_MULT: float = 10.0    # volume ratio for whale detection
    VOLUME_SURGE_MULT: float = 3.0  # volume ratio for DEX surge

    def __init__(self) -> None:
        self._gas_history: deque[float] = deque(maxlen=self.GAS_WINDOW)
        self._vol_history: deque[float] = deque(maxlen=288)  # 24h at 5-min bars
        self._gas_trend: float = 0.0
        self._whale_signal: float = 0.0
        self._vol_surge: float = 0.0
        self._composite: float = 0.0

    def update(
        self,
        gas_gwei: float,
        dex_volume_usd: float,
        buy_ratio: float = 0.5,
    ) -> None:
        """Ingest one bar of on-chain metrics.

        Args:
            gas_gwei: Current gas price in gwei (Base uses EIP-1559 base fee).
            dex_volume_usd: DEX volume for this bar in USD.
            buy_ratio: Fraction of volume attributed to buys [0, 1].
        """
        self._gas_history.append(gas_gwei)
        self._vol_history.append(dex_volume_usd)

        self._compute_gas_trend()
        self._compute_whale_signal(dex_volume_usd, buy_ratio)
        self._compute_vol_surge(dex_volume_usd)

        self._composite = max(-1.0, min(1.0,
            0.30 * self._gas_trend
            + 0.40 * self._whale_signal
            + 0.30 * self._vol_surge
        ))

    def _compute_gas_trend(self) -> None:
        if len(self._gas_history) < 5:
            return
        gas = np.array(self._gas_history, dtype=np.float64)
        median_gas = float(np.median(gas))
        if median_gas <= 0:
            return
        ratio = gas[-1] / median_gas
        if ratio >= self.GAS_SURGE_MULT:
            self._gas_trend = -0.3  # congestion / panic caution
        elif ratio >= 1.5:
            self._gas_trend = 0.5   # healthy activity surge — mild bullish
        else:
            self._gas_trend = max(-1.0, min(1.0, (ratio - 1.0) * 2.0))

    def _compute_whale_signal(self, volume: float, buy_ratio: float) -> None:
        if len(self._vol_history) < 5:
            return
        median_vol = float(np.median(self._vol_history))
        if median_vol <= 0:
            return
        ratio = volume / median_vol
        if ratio < self.WHALE_MULT:
            self._whale_signal = 0.0
            return
        direction = 1.0 if buy_ratio > 0.55 else (-1.0 if buy_ratio < 0.45 else 0.0)
        magnitude = min(1.0, (ratio - self.WHALE_MULT) / self.WHALE_MULT)
        self._whale_signal = direction * magnitude

    def _compute_vol_surge(self, volume: float) -> None:
        if len(self._vol_history) < 12:
            return
        avg_24h = float(np.mean(self._vol_history))
        if avg_24h <= 0:
            return
        ratio = volume / avg_24h
        if ratio >= self.VOLUME_SURGE_MULT:
            self._vol_surge = min(1.0, (ratio - self.VOLUME_SURGE_MULT) / self.VOLUME_SURGE_MULT)
        else:
            self._vol_surge = 0.0

    @property
    def gas_trend(self) -> float:
        return self._gas_trend

    @property
    def whale_signal(self) -> float:
        return self._whale_signal

    @property
    def volume_surge(self) -> float:
        return self._vol_surge

    @property
    def composite(self) -> float:
        """Composite on-chain signal in [-1, +1]."""
        return self._composite


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_alpha_generator: AlphaGenerator | None = None


def get_alpha_generator() -> AlphaGenerator:
    """Return the process-wide AlphaGenerator singleton."""
    global _alpha_generator
    if _alpha_generator is None:
        _alpha_generator = AlphaGenerator()
    return _alpha_generator


def reset_alpha_generator() -> None:
    """Reset the singleton (for tests)."""
    global _alpha_generator
    _alpha_generator = None
