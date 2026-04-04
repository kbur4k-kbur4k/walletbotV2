"""
wallet_manager/security/telegram_manager.py

Professional Telegram Command & Control — trade notifications, daily summary, system alerts.

Notifications:
  - Trade Executed: Pair, Price, Strategy Type, Amount
  - Daily Summary: Total TVL, Aave Yield accrued, Algo PnL
  - System Alerts: High Gas, Tx Failures, Circuit Breaker triggers

Security: Tokens read from .env only. NEVER logged or exposed.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Final

import aiohttp

from wallet_manager.env_loader import load_project_env
from wallet_manager.security.logging_filter import build_logger

load_project_env()

log = build_logger(__name__)

_TELEGRAM_API_BASE: Final[str] = "https://api.telegram.org"
_SEND_PATH: Final[str] = "/bot{token}/sendMessage"
_API_TIMEOUT: Final[float] = 10.0

_HEX_ADDR_RE: Final[re.Pattern[str]] = re.compile(
    r"(0x[0-9a-fA-F]{4})[0-9a-fA-F]{32,58}([0-9a-fA-F]{4})"
)


def _mask_address(text: str) -> str:
    """Mask hex addresses for safe logging/display."""
    return _HEX_ADDR_RE.sub(r"\1****\2", text)


@dataclass(frozen=True)
class _TelegramCreds:
    """Credentials from env. __repr__ never exposes token."""

    _token: str = field(repr=False)
    _chat_id: str = field(repr=False)

    def __repr__(self) -> str:
        return "_TelegramCreds(***)"

    @property
    def token(self) -> str:
        return self._token

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @classmethod
    def from_env(cls) -> _TelegramCreds:
        from wallet_manager.config import get_settings
        cfg = get_settings()
        token = cfg.telegram_bot_token
        chat_id = cfg.telegram_chat_id
        if not token or not chat_id:
            return cls(_token="", _chat_id="")
        return cls(_token=token, _chat_id=chat_id)


@dataclass
class TradeExecutedPayload:
    """Payload for trade executed notification."""

    pair: str
    price: float
    strategy_type: str
    amount: str
    side: str
    tx_hash: str | None = None


@dataclass
class DailySummaryPayload:
    """Payload for daily summary notification."""

    total_tvl_usd: float
    aave_yield_usd: float
    aave_apy_pct: float
    algo_pnl_usd: float


@dataclass
class SystemAlertPayload:
    """Payload for system alert notification."""

    alert_type: str
    detail: str
    severity: str = "warning"


@dataclass
class CircuitBreakerPayload:
    """Payload for circuit breaker (daily drawdown) alert."""

    alert_type: str
    detail: str
    severity: str = "critical"
    start_value_usd: float = 0.0
    current_value_usd: float = 0.0
    drawdown_pct: float = 0.0


@dataclass
class StopLossPayload:
    """Payload for stop-loss hit alert — triggers circuit breaker halt."""

    wallet_id: str
    pair: str
    price: float
    last_buy_price: float
    pct_loss: float


@dataclass
class HeartbeatPayload:
    """Payload for 6-hour heartbeat: TVL, positions, unrealized PnL, Aave yield, gas."""

    total_tvl_usd: float
    open_positions_count: int
    unrealized_pnl_usd: float
    aave_yield_earned_usd: float
    aave_apy_pct: float
    current_gas_gwei: float
    strategy_status: str = ""  # e.g. "ALGO: active, YIELD: supplied, AIRDROP: ready"
    # Paper trading fields
    paper_balance_usd: float = 0.0
    paper_starting_balance_usd: float = 10_000.0
    total_trades: int = 0
    win_rate_pct: float = 0.0
    best_trade_pnl: float = 0.0
    best_trade_pair: str = ""
    worst_trade_pnl: float = 0.0
    worst_trade_pair: str = ""
    last_trade_time: str = ""
    last_trade_token: str = ""
    last_trade_pnl: float = 0.0
    signal_status: str = ""  # Why idle — which condition is not met
    gas_suitable: bool = True
    next_run_seconds: float = 0.0


@dataclass
class YieldAlertPayload:
    """Payload when Aave APY drops below 3% and another protocol offers more."""

    aave_apy_pct: float
    alternative_protocol: str
    alternative_apy_pct: float
    detail: str


class TelegramManager:
    """
    Professional Telegram notification manager for trading system.

    All methods are fire-and-forget. Token is read from .env and NEVER logged.
    """

    def __init__(
        self,
        credentials: _TelegramCreds | None = None,
        settings: object = None,
    ) -> None:
        if settings is not None:
            # Accept a settings-like object with telegram_bot_token / telegram_chat_id
            token = getattr(settings, "telegram_bot_token", "") or ""
            chat_id = str(getattr(settings, "telegram_chat_id", "") or "")
            self._creds = _TelegramCreds(_token=token, _chat_id=chat_id)
        else:
            self._creds = credentials or _TelegramCreds.from_env()
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()
        # Confirm-attempt rate limiting (3 per 60s)
        import time as _time
        self._confirm_attempts: dict[str, list[float]] = {}
        self._confirm_rate_limit = 3
        self._confirm_window_s = 60.0

    def is_authorized(self, update: object) -> bool:
        """Return True when the update's chat_id matches the configured chat."""
        message = getattr(update, "message", None)
        chat_id = getattr(message, "chat_id", None)
        if chat_id is None:
            return False
        return str(chat_id) == str(self._creds.chat_id)

    def _record_confirm_attempt(self, chat_id: str) -> None:
        """Record a confirm attempt for rate-limiting."""
        import time as _time
        now = _time.time()
        bucket = self._confirm_attempts.setdefault(chat_id, [])
        bucket.append(now)
        # Prune old entries
        self._confirm_attempts[chat_id] = [
            t for t in bucket if now - t < self._confirm_window_s
        ]

    def _is_confirm_rate_limited(self, chat_id: str) -> bool:
        """Return True when chat_id has exceeded the confirm attempt limit."""
        import time as _time
        now = _time.time()
        bucket = self._confirm_attempts.get(chat_id, [])
        recent = [t for t in bucket if now - t < self._confirm_window_s]
        return len(recent) >= self._confirm_rate_limit

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
                headers={"Content-Type": "application/json"},
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _is_configured(self) -> bool:
        return bool(self._creds.token and self._creds.chat_id)

    async def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self._is_configured():
            log.debug("TelegramManager: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skip.")
            return False
        await self._ensure_session()
        assert self._session is not None  # set by _ensure_session
        url = _TELEGRAM_API_BASE + _SEND_PATH.format(token=self._creds.token)
        payload = {
            "chat_id": self._creds.chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            async with self._lock, self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                log.info(
                    f"TelegramManager: HTTP {resp.status} from Telegram API. "
                    "Check bot token and chat_id configuration."
                )
        except Exception as exc:
            log.info(f"TelegramManager: send unavailable ({type(exc).__name__}). Will retry next cycle.")
            if isinstance(exc, aiohttp.ServerDisconnectedError):
                await self.close()
        return False

    async def send_trade_executed(self, payload: TradeExecutedPayload) -> bool:
        """
        Send instant notification: Trade Executed (Pair, Price, Strategy, Amount).
        """
        tx_str = f"\n🔗 Tx: <code>{_mask_address(payload.tx_hash or '')}</code>" if payload.tx_hash else ""
        text = (
            f"✅ <b>Trade Executed</b>\n"
            f"Pair: {payload.pair}\n"
            f"Price: ${payload.price:.2f}\n"
            f"Strategy: {payload.strategy_type}\n"
            f"Side: {payload.side}\n"
            f"Amount: {payload.amount}{tx_str}"
        )
        return await self._send(text)

    async def send_daily_summary(self, payload: DailySummaryPayload) -> bool:
        """
        Send daily summary: Total TVL, Aave Yield accrued, Algo PnL.
        """
        text = (
            f"📊 <b>Daily Summary</b>\n"
            f"Total TVL: ${payload.total_tvl_usd:.2f}\n"
            f"Aave Yield: ${payload.aave_yield_usd:.2f} ({payload.aave_apy_pct:.2f}% APY)\n"
            f"Algo PnL: ${payload.algo_pnl_usd:+.2f}"
        )
        return await self._send(text)

    async def send_system_alert(self, payload: SystemAlertPayload) -> bool:
        """
        Send system alert: High Gas, Tx Failures, Circuit Breaker.
        """
        emoji = "🔴" if payload.severity == "critical" else "⚠️"
        safe_detail = _mask_address(payload.detail)
        text = (
            f"{emoji} <b>System Alert</b>\n"
            f"Type: {payload.alert_type}\n"
            f"Detail: {safe_detail}"
        )
        return await self._send(text)

    async def send_circuit_breaker_alert(self, payload: CircuitBreakerPayload) -> bool:
        """Send circuit breaker (daily drawdown/stop-loss) alert — halting all trades."""
        text = (
            f"🛑 <b>Circuit Breaker Triggered</b>\n"
            f"Type: {payload.alert_type}\n"
            f"Detail: {payload.detail}\n"
            f"Start: ${payload.start_value_usd:.2f} → Now: ${payload.current_value_usd:.2f}\n"
            f"Drawdown: {payload.drawdown_pct:.2f}%\n\n"
            f"<i>Reply /resume to override and resume trading.</i>"
        )
        return await self._send(text)

    async def send_stop_loss_alert(self, payload: StopLossPayload) -> bool:
        """Send high-priority alert when 3% stop-loss hit — circuit breaker active."""
        text = (
            f"🔴 <b>Stop-Loss Triggered</b>\n"
            f"Wallet: {_mask_address(payload.wallet_id)}\n"
            f"Pair: {payload.pair}\n"
            f"Entry: ${payload.last_buy_price:.2f} → Exit: ${payload.price:.2f}\n"
            f"Loss: {payload.pct_loss:.2f}%\n\n"
            f"<i>Reply /resume to override 24h halt and resume trading.</i>"
        )
        return await self._send(text)

    async def send_heartbeat(self, payload: HeartbeatPayload) -> bool:
        """Send 6-hour heartbeat with rich paper trading + portfolio data."""
        # Paper balance change
        paper_change = payload.paper_balance_usd - payload.paper_starting_balance_usd
        paper_pct = (paper_change / payload.paper_starting_balance_usd * 100) if payload.paper_starting_balance_usd > 0 else 0

        # Gas status
        gas_emoji = "\u2705" if payload.gas_suitable else "\u274c"
        gas_status = f"{gas_emoji} {payload.current_gas_gwei:.2f} Gwei ({'suitable' if payload.gas_suitable else 'too high'})"

        # Next run
        if payload.next_run_seconds >= 3600:
            next_run_str = f"{payload.next_run_seconds / 3600:.1f}h"
        elif payload.next_run_seconds >= 60:
            next_run_str = f"{payload.next_run_seconds / 60:.0f}m"
        else:
            next_run_str = f"{payload.next_run_seconds:.0f}s"

        # Last trade info
        last_trade_str = "None yet"
        if payload.last_trade_time:
            last_trade_str = f"{payload.last_trade_time} | {payload.last_trade_token} | ${payload.last_trade_pnl:+.2f}"

        lines = [
            "\U0001f493 <b>Portfolio Heartbeat</b>",
            f"\U0001f4b0 Paper Balance: ${payload.paper_balance_usd:.2f} ({paper_pct:+.1f}% vs ${payload.paper_starting_balance_usd:.0f} start)",
            f"\U0001f4ca Total TVL: ${payload.total_tvl_usd:.2f}",
            f"\U0001f4c8 Net PnL: ${payload.unrealized_pnl_usd:+.2f}",
            "",
            f"\U0001f4c8 Total Trades: {payload.total_trades}",
            f"\U0001f3af Win Rate: {payload.win_rate_pct:.1f}%",
            f"\U0001f4ca Best Trade: ${payload.best_trade_pnl:+.2f} ({payload.best_trade_pair})" if payload.best_trade_pair else "\U0001f4ca Best Trade: N/A",
            f"\U0001f4c9 Worst Trade: ${payload.worst_trade_pnl:+.2f} ({payload.worst_trade_pair})" if payload.worst_trade_pair else "\U0001f4c9 Worst Trade: N/A",
            f"\u23f1\ufe0f Last Trade: {last_trade_str}",
            "",
            f"\U0001f3e6 Aave: ${payload.aave_yield_earned_usd:.2f} yield ({payload.aave_apy_pct:.2f}% APY)",
            f"\u26fd Gas: {gas_status}",
            f"\U0001f50d Signal: {payload.signal_status}" if payload.signal_status else "\U0001f50d Signal: Monitoring...",
            f"\U0001f504 Next Run: {next_run_str}",
        ]
        if payload.strategy_status:
            lines.append(f"\U0001f4cb Strategy: {payload.strategy_status}")

        text = "\n".join(lines)
        return await self._send(text)

    async def send_yield_alert(self, payload: YieldAlertPayload) -> bool:
        """Alert when Aave APY < 2% — manual reallocation or auto-compound rewards to Algo."""
        text = (
            f"📉 <b>Yield Alert</b>\n"
            f"Aave APY: {payload.aave_apy_pct:.2f}% (below 2%)\n"
            f"Better option: {payload.alternative_protocol} @ {payload.alternative_apy_pct:.2f}%\n"
            f"Detail: {payload.detail}\n\n"
            f"<i>Consider manual reallocation or auto-compound rewards to Algo budget.</i>"
        )
        return await self._send(text)


async def notify_trade_executed(
    pair: str,
    price: float,
    strategy_type: str,
    amount: str,
    side: str,
    tx_hash: str | None = None,
) -> bool:
    """Convenience: send trade executed notification."""
    mgr = TelegramManager()
    payload = TradeExecutedPayload(
        pair=pair, price=price, strategy_type=strategy_type,
        amount=amount, side=side, tx_hash=tx_hash,
    )
    return await mgr.send_trade_executed(payload)


async def notify_daily_summary(
    total_tvl_usd: float,
    aave_yield_usd: float,
    aave_apy_pct: float,
    algo_pnl_usd: float,
) -> bool:
    """Convenience: send daily summary notification."""
    mgr = TelegramManager()
    payload = DailySummaryPayload(
        total_tvl_usd=total_tvl_usd,
        aave_yield_usd=aave_yield_usd,
        aave_apy_pct=aave_apy_pct,
        algo_pnl_usd=algo_pnl_usd,
    )
    return await mgr.send_daily_summary(payload)


async def notify_system_alert(
    alert_type: str,
    detail: str,
    severity: str = "warning",
) -> bool:
    """Convenience: send system alert (high gas, tx failure, circuit breaker)."""
    mgr = TelegramManager()
    payload = SystemAlertPayload(
        alert_type=alert_type,
        detail=detail,
        severity=severity,
    )
    return await mgr.send_system_alert(payload)
