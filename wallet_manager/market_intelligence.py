"""
wallet_manager/market_intelligence.py

Phase 5: Market Intelligence
=============================
Structured macro calendar, event impact classification, and event-aware
regime override.  Replaces the one-off NFP string-match with a proper
calendar-driven system.

Design:
- MacroEvent dataclass with name, date, expected_impact, category
- MacroCalendar: in-process registry + check "is there an imminent event?"
- EventImpactClassifier: maps event names to impact levels
- SeasonalCalendar: pre-populated US macro events for the current quarter

No external API calls.  Calendar is updated here each quarter or when
the user pushes a new event list.  The task_factory can call
`calendar.get_imminent_events(window_hours=N)` to decide whether to skip alpha.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ImpactLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventCategory(StrEnum):
    EMPLOYMENT = "EMPLOYMENT"
    INFLATION = "INFLATION"
    GROWTH = "GROWTH"
    MONETARY_POLICY = "MONETARY_POLICY"
    LIQUIDITY = "LIQUIDITY"
    GEOPOLITICAL = "GEOPOLITICAL"
    CRYPTO_SPECIFIC = "CRYPTO_SPECIFIC"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MacroEvent:
    name: str
    timestamp_utc: float               # Unix timestamp of event
    impact: ImpactLevel
    category: EventCategory
    description: str = ""
    skip_alpha_window_hours: float = 2.0    # How many hours before/after to suppress alpha
    avoid_window_hours_before: float = 1.0  # Extra lead time to go flat

    @property
    def dt_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_utc, tz=UTC)

    def is_imminent(self, now_ts: float | None = None, window_hours: float = 2.0) -> bool:
        """Return True if event is within ±window_hours of now."""
        now = now_ts or time.time()
        delta = abs(self.timestamp_utc - now)
        return delta <= window_hours * 3600

    def is_pre_event(self, now_ts: float | None = None) -> bool:
        """Return True if we are inside the lead-avoidance window before event."""
        now = now_ts or time.time()
        time_to_event = self.timestamp_utc - now
        return 0 < time_to_event <= self.avoid_window_hours_before * 3600

    def is_active(self, now_ts: float | None = None) -> bool:
        """Return True if we are inside the skip_alpha window around the event."""
        now = now_ts or time.time()
        return abs(self.timestamp_utc - now) <= self.skip_alpha_window_hours * 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp_utc": self.timestamp_utc,
            "datetime_utc": self.dt_utc.strftime("%Y-%m-%d %H:%M UTC"),
            "impact": self.impact.value,
            "category": self.category.value,
            "description": self.description,
            "skip_alpha_window_hours": self.skip_alpha_window_hours,
        }


# ---------------------------------------------------------------------------
# Event Impact Classifier
# ---------------------------------------------------------------------------


class EventImpactClassifier:
    """Maps event names (case-insensitive keyword match) to impact levels."""

    # (keyword, impact, category, skip_window_hours, lead_hours)
    _RULES: list[tuple[str, ImpactLevel, EventCategory, float, float]] = [
        # CRITICAL events — go fully flat
        ("nfp", ImpactLevel.CRITICAL, EventCategory.EMPLOYMENT, 3.0, 1.0),
        ("non-farm payroll", ImpactLevel.CRITICAL, EventCategory.EMPLOYMENT, 3.0, 1.0),
        ("fomc", ImpactLevel.CRITICAL, EventCategory.MONETARY_POLICY, 4.0, 2.0),
        ("fed decision", ImpactLevel.CRITICAL, EventCategory.MONETARY_POLICY, 4.0, 2.0),
        ("interest rate decision", ImpactLevel.CRITICAL, EventCategory.MONETARY_POLICY, 4.0, 2.0),
        ("cpi", ImpactLevel.CRITICAL, EventCategory.INFLATION, 2.0, 1.0),
        ("consumer price index", ImpactLevel.CRITICAL, EventCategory.INFLATION, 2.0, 1.0),
        ("pce", ImpactLevel.CRITICAL, EventCategory.INFLATION, 2.0, 0.5),
        ("gdp", ImpactLevel.HIGH, EventCategory.GROWTH, 2.0, 0.5),
        ("etf approval", ImpactLevel.CRITICAL, EventCategory.CRYPTO_SPECIFIC, 6.0, 2.0),
        ("sec ruling", ImpactLevel.CRITICAL, EventCategory.CRYPTO_SPECIFIC, 6.0, 2.0),
        ("exchange hack", ImpactLevel.CRITICAL, EventCategory.CRYPTO_SPECIFIC, 12.0, 0.0),
        # HIGH events
        ("pmi", ImpactLevel.HIGH, EventCategory.GROWTH, 1.5, 0.5),
        ("retail sales", ImpactLevel.HIGH, EventCategory.GROWTH, 1.5, 0.5),
        ("jobs claims", ImpactLevel.HIGH, EventCategory.EMPLOYMENT, 1.5, 0.5),
        ("unemployment", ImpactLevel.HIGH, EventCategory.EMPLOYMENT, 2.0, 0.5),
        ("halving", ImpactLevel.HIGH, EventCategory.CRYPTO_SPECIFIC, 24.0, 6.0),
        # MEDIUM events
        ("housing", ImpactLevel.MEDIUM, EventCategory.GROWTH, 1.0, 0.5),
        ("durables", ImpactLevel.MEDIUM, EventCategory.GROWTH, 1.0, 0.0),
        ("ppi", ImpactLevel.MEDIUM, EventCategory.INFLATION, 1.0, 0.5),
    ]

    @classmethod
    def classify(cls, name: str) -> tuple[ImpactLevel, EventCategory, float, float]:
        """Return (impact, category, skip_window_hours, lead_hours) for event name."""
        name_lower = name.lower()
        for keyword, impact, category, skip, lead in cls._RULES:
            if keyword in name_lower:
                return impact, category, skip, lead
        return ImpactLevel.LOW, EventCategory.OTHER, 0.5, 0.0


# ---------------------------------------------------------------------------
# Macro Calendar
# ---------------------------------------------------------------------------


class MacroCalendar:
    """Registry of scheduled macro events with imminent-event detection.

    Pre-populated with key 2026 Q2 US macro events.  Additional events
    can be added programmatically via `add_event()`.
    """

    def __init__(self) -> None:
        self._events: list[MacroEvent] = []
        self._populate_2026_q2()

    def _populate_2026_q2(self) -> None:
        """Pre-populate 2026 Q2 macro schedule (approximate dates)."""
        events_raw: list[tuple[str, str, str]] = [
            # (name, datetime_utc_str, impact_override_or_empty)
            # April 2026
            ("NFP 2026-04-03", "2026-04-03 12:30", ""),
            ("CPI 2026-04-10", "2026-04-10 12:30", ""),
            ("PPI 2026-04-11", "2026-04-11 12:30", ""),
            ("Retail Sales 2026-04-15", "2026-04-15 12:30", ""),
            ("Housing Starts 2026-04-16", "2026-04-16 12:30", ""),
            ("FOMC Minutes 2026-04-08", "2026-04-08 18:00", ""),
            # May 2026
            ("NFP 2026-05-01", "2026-05-01 12:30", ""),
            ("CPI 2026-05-12", "2026-05-12 12:30", ""),
            ("FOMC Decision 2026-05-06", "2026-05-06 18:00", ""),
            ("GDP Q1 Advance 2026-04-30", "2026-04-30 12:30", ""),
            # June 2026
            ("NFP 2026-06-05", "2026-06-05 12:30", ""),
            ("CPI 2026-06-11", "2026-06-11 12:30", ""),
            ("FOMC Decision 2026-06-17", "2026-06-17 18:00", ""),
            ("PCE 2026-06-26", "2026-06-26 12:30", ""),
        ]
        for name, dt_str, _ in events_raw:
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                impact, category, skip_w, lead_w = EventImpactClassifier.classify(name)
                self._events.append(MacroEvent(
                    name=name,
                    timestamp_utc=dt.timestamp(),
                    impact=impact,
                    category=category,
                    skip_alpha_window_hours=skip_w,
                    avoid_window_hours_before=lead_w,
                ))
            except ValueError:
                pass

    def add_event(
        self,
        name: str,
        timestamp_utc: float,
        impact: ImpactLevel | None = None,
        category: EventCategory | None = None,
        skip_alpha_window_hours: float | None = None,
        avoid_window_hours_before: float | None = None,
        description: str = "",
    ) -> MacroEvent:
        """Add a custom event to the calendar."""
        classified_impact, classified_cat, skip_w, lead_w = EventImpactClassifier.classify(name)
        event = MacroEvent(
            name=name,
            timestamp_utc=timestamp_utc,
            impact=impact or classified_impact,
            category=category or classified_cat,
            skip_alpha_window_hours=skip_alpha_window_hours if skip_alpha_window_hours is not None else skip_w,
            avoid_window_hours_before=avoid_window_hours_before if avoid_window_hours_before is not None else lead_w,
            description=description,
        )
        self._events.append(event)
        return event

    def get_imminent_events(
        self,
        now_ts: float | None = None,
        window_hours: float = 2.0,
        min_impact: ImpactLevel = ImpactLevel.HIGH,
    ) -> list[MacroEvent]:
        """Return all events within ±window_hours of now at or above min_impact."""
        now = now_ts or time.time()
        impact_order = {ImpactLevel.LOW: 0, ImpactLevel.MEDIUM: 1, ImpactLevel.HIGH: 2, ImpactLevel.CRITICAL: 3}
        min_order = impact_order[min_impact]
        return [
            e for e in self._events
            if e.is_imminent(now, window_hours)
            and impact_order.get(e.impact, 0) >= min_order
        ]

    def should_suppress_alpha(
        self,
        now_ts: float | None = None,
        window_hours: float = 2.0,
    ) -> tuple[bool, str]:
        """Return (should_suppress, reason_string).

        Used by task_factory to decide whether to skip alpha execution.
        Checks for any HIGH or CRITICAL imminent events.
        """
        events = self.get_imminent_events(now_ts=now_ts, window_hours=window_hours)
        if not events:
            return False, ""
        event = events[0]  # Most urgent
        reason = (
            f"NEWS_EVENT_IMMINENT: {event.name} "
            f"(impact={event.impact.value}, cat={event.category.value})"
        )
        return True, reason

    def upcoming_events(self, hours_ahead: float = 48.0) -> list[MacroEvent]:
        """Return events in the next N hours, sorted by time."""
        now = time.time()
        cutoff = now + hours_ahead * 3600
        upcoming = [e for e in self._events if now < e.timestamp_utc <= cutoff]
        return sorted(upcoming, key=lambda e: e.timestamp_utc)

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {
            "total_events": len(self._events),
            "imminent_high_plus": [e.to_dict() for e in self.get_imminent_events(now, window_hours=4.0)],
            "next_48h": [e.to_dict() for e in self.upcoming_events(48.0)],
        }


# ---------------------------------------------------------------------------
# Chain Intelligence (Phase 5 — on-chain monitoring, no external API)
# ---------------------------------------------------------------------------


class ChainIntelligence:
    """
    On-chain monitoring layer using the bot's existing RPC connection.

    Covers:
    - Stablecoin depeg watcher  (USDC price vs $1.00)
    - Liquidity change watcher  (pool depth snapshots)
    - On-chain flow summaries   (recent ETH/USDC block activity)
    - Whale movement watcher    (large USDC/WETH transfers)
    - Momentum sentiment        (rule-based from price history)

    All methods are async and gracefully degrade to empty results when
    the RPC is unavailable.
    """

    # USDC ERC-20 on Base / Ethereum Mainnet
    _USDC_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"   # Base USDC
    # Whale threshold: transfers above this USD value are flagged
    _WHALE_THRESHOLD_USD = 500_000.0

    def __init__(self) -> None:
        self._price_history: list[tuple[float, float]] = []   # (timestamp, price_usd)
        self._pool_depth_history: list[tuple[float, float]] = []  # (timestamp, depth_usd)
        self._whale_alerts: list[dict[str, Any]] = []
        self._max_history: int = 200

    # ------------------------------------------------------------------
    # 1. Stablecoin depeg watcher
    # ------------------------------------------------------------------

    def record_usdc_price(self, usdc_price_usd: float) -> None:
        """Call this each cycle with the current USDC price (should be ~1.0)."""
        self._price_history.append((time.time(), usdc_price_usd))
        if len(self._price_history) > self._max_history:
            self._price_history = self._price_history[-self._max_history:]

    def get_depeg_warning(self) -> dict[str, Any]:
        """Return depeg status based on recent USDC price readings."""
        if not self._price_history:
            return {"status": "NO_DATA", "current_price": None, "deviation_pct": 0.0, "alert": False}
        _, latest = self._price_history[-1]
        deviation_pct = abs(latest - 1.0) * 100.0
        if deviation_pct >= 1.0:
            status = "CRITICAL_DEPEG"
            alert = True
        elif deviation_pct >= 0.3:
            status = "MINOR_DEPEG"
            alert = True
        else:
            status = "PEG_STABLE"
            alert = False
        return {
            "status": status,
            "current_price": round(latest, 6),
            "deviation_pct": round(deviation_pct, 4),
            "alert": alert,
            "checked_at": self._price_history[-1][0],
        }

    # ------------------------------------------------------------------
    # 2. Liquidity change watcher
    # ------------------------------------------------------------------

    def record_pool_depth(self, depth_usd: float) -> None:
        """Record a pool depth snapshot (USD value of ETH/USDC pool depth)."""
        self._pool_depth_history.append((time.time(), depth_usd))
        if len(self._pool_depth_history) > self._max_history:
            self._pool_depth_history = self._pool_depth_history[-self._max_history:]

    def get_liquidity_change(self) -> dict[str, Any]:
        """Compare recent pool depth to earlier baseline. Returns change % and alert."""
        if len(self._pool_depth_history) < 2:
            return {"status": "NO_DATA", "current_depth_usd": None, "change_pct": 0.0, "alert": False}
        _, current = self._pool_depth_history[-1]
        # Baseline: median of earlier 20 readings
        baseline_readings = [d for _, d in self._pool_depth_history[:-1]][-20:]
        baseline = sorted(baseline_readings)[len(baseline_readings) // 2]
        change_pct = (current - baseline) / baseline * 100.0 if baseline > 0 else 0.0
        alert = change_pct < -20.0  # >20% liquidity drop is a warning
        return {
            "status": "LOW_LIQUIDITY" if alert else "NORMAL",
            "current_depth_usd": round(current, 2),
            "baseline_depth_usd": round(baseline, 2),
            "change_pct": round(change_pct, 2),
            "alert": alert,
        }

    # ------------------------------------------------------------------
    # 3. On-chain flow summaries
    # ------------------------------------------------------------------

    async def get_flow_summary(self, rpc_url: str = "") -> dict[str, Any]:
        """Fetch recent block gas/activity as a proxy for on-chain flow intensity."""
        if not rpc_url:
            return {"status": "NO_RPC", "block_gas_usage_pct": None, "activity": "unknown"}
        try:
            import aiohttp
            payload = {"jsonrpc": "2.0", "method": "eth_getBlockByNumber",
                       "params": ["latest", False], "id": 1}
            async with aiohttp.ClientSession() as session, session.post(
                rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
            block = data.get("result", {})
            gas_used = int(block.get("gasUsed", "0x0"), 16)
            gas_limit = int(block.get("gasLimit", "0x1"), 16)
            block_number = int(block.get("number", "0x0"), 16)
            usage_pct = gas_used / gas_limit * 100.0 if gas_limit > 0 else 0.0
            activity = "HIGH" if usage_pct > 80 else "MEDIUM" if usage_pct > 50 else "LOW"
            return {
                "status": "OK",
                "block_number": block_number,
                "gas_used": gas_used,
                "gas_limit": gas_limit,
                "block_gas_usage_pct": round(usage_pct, 2),
                "activity": activity,
                "alert": usage_pct > 90,
            }
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc), "activity": "unknown"}

    # ------------------------------------------------------------------
    # 4. Whale movement watcher
    # ------------------------------------------------------------------

    def record_whale_transfer(
        self,
        tx_hash: str,
        from_addr: str,
        to_addr: str,
        amount_usd: float,
        token: str = "USDC",
    ) -> None:
        """Record a detected whale transfer."""
        if amount_usd >= self._WHALE_THRESHOLD_USD:
            self._whale_alerts.append({
                "timestamp": time.time(),
                "tx_hash": tx_hash,
                "from": from_addr[:10] + "…",
                "to": to_addr[:10] + "…",
                "amount_usd": round(amount_usd, 2),
                "token": token,
            })
            # Keep only last 50 whale alerts
            self._whale_alerts = self._whale_alerts[-50:]

    def get_recent_whale_alerts(self, lookback_hours: float = 6.0) -> list[dict[str, Any]]:
        """Return whale alerts from the past N hours."""
        cutoff = time.time() - lookback_hours * 3600
        return [a for a in self._whale_alerts if a["timestamp"] >= cutoff]

    def get_whale_summary(self) -> dict[str, Any]:
        recent = self.get_recent_whale_alerts(6.0)
        total_volume = sum(a["amount_usd"] for a in recent)
        return {
            "alerts_6h": len(recent),
            "total_volume_usd": round(total_volume, 2),
            "alert": len(recent) >= 3 or total_volume >= 5_000_000,
            "recent": recent[-5:],
        }

    # ------------------------------------------------------------------
    # 5. Momentum sentiment (rule-based, no external API)
    # ------------------------------------------------------------------

    def get_momentum_sentiment(self) -> dict[str, Any]:
        """
        Derive market sentiment from price history.
        - BULLISH : last price > 5-period MA, positive momentum
        - BEARISH : last price < 5-period MA, negative momentum
        - NEUTRAL : near MA, weak momentum
        """
        if len(self._price_history) < 5:
            return {"sentiment": "NEUTRAL", "score": 0, "momentum_pct": 0.0, "data_points": len(self._price_history)}

        prices = [p for _, p in self._price_history[-20:]]
        latest = prices[-1]
        ma5 = sum(prices[-5:]) / 5
        oldest = prices[0]
        momentum_pct = (latest - oldest) / oldest * 100.0 if oldest > 0 else 0.0

        if latest > ma5 * 1.005 and momentum_pct > 0.5:
            sentiment = "BULLISH"
            score = min(100, int(momentum_pct * 10))
        elif latest < ma5 * 0.995 and momentum_pct < -0.5:
            sentiment = "BEARISH"
            score = max(-100, int(momentum_pct * 10))
        else:
            sentiment = "NEUTRAL"
            score = 0

        return {
            "sentiment": sentiment,
            "score": score,
            "momentum_pct": round(momentum_pct, 4),
            "latest_price": round(latest, 4),
            "ma5": round(ma5, 4),
            "data_points": len(self._price_history),
        }

    # ------------------------------------------------------------------
    # Aggregate snapshot
    # ------------------------------------------------------------------

    async def get_snapshot(self, rpc_url: str = "") -> dict[str, Any]:
        """Return a combined on-chain intelligence snapshot."""
        flow = await self.get_flow_summary(rpc_url)
        return {
            "generated_at": time.time(),
            "stablecoin_depeg": self.get_depeg_warning(),
            "liquidity": self.get_liquidity_change(),
            "on_chain_flow": flow,
            "whale": self.get_whale_summary(),
            "sentiment": self.get_momentum_sentiment(),
        }


# ---------------------------------------------------------------------------
# ChainIntelligence singleton
# ---------------------------------------------------------------------------

_chain_intel: ChainIntelligence | None = None


def get_chain_intelligence() -> ChainIntelligence:
    """Return the module-level ChainIntelligence singleton."""
    global _chain_intel
    if _chain_intel is None:
        _chain_intel = ChainIntelligence()
    return _chain_intel


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_calendar: MacroCalendar | None = None


def get_macro_calendar() -> MacroCalendar:
    """Return the module-level MacroCalendar singleton."""
    global _calendar
    if _calendar is None:
        _calendar = MacroCalendar()
    return _calendar
