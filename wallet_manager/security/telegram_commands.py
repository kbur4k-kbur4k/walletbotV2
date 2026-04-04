"""
wallet_manager/security/telegram_commands.py

Telegram command dispatch — CommandState, CommandCenter, and supporting types.

Extracted from telegram_notifier.py to keep that module under 600 lines.
All public names are re-exported from telegram_notifier for backward compat.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

from wallet_manager.config import get_settings
from wallet_manager.core.models import EventKind, NotificationEvent
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Address masker (duplicated here so this module is self-contained)
# ---------------------------------------------------------------------------

_HEX_ADDR_RE: Final[re.Pattern[str]] = re.compile(
    r"(0x[0-9a-fA-F]{4})[0-9a-fA-F]{32,58}([0-9a-fA-F]{4})"
)


def _mask_address(text: str) -> str:
    """Replace hex addresses/keys with a partially-visible masked form."""
    return _HEX_ADDR_RE.sub(r"\1****\2", text)


# ---------------------------------------------------------------------------
# Message formatter (moved here; re-exported from telegram_notifier)
# ---------------------------------------------------------------------------

_MAX_MSG_LEN: Final[int] = 4096


def _split_message(text: str, max_len: int = _MAX_MSG_LEN) -> list[str]:
    """Split text > max_len at logical section boundaries (Section 5.4).

    Tries ━━━ section separators, then double newlines, then single newlines.
    Falls back to a hard split at max_len if no boundary is found.
    """
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    while len(text) > max_len:
        split_at = text.rfind("\n\u2501", 0, max_len)   # ━ line
        if split_at == -1:
            split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        parts.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip("\n")
    if text.strip():
        parts.append(text)
    return parts


class _MessageFormatter:
    """Renders :class:`~wallet_manager.core.models.NotificationEvent` into
    Telegram MarkdownV2 strings with address masking applied."""

    _MD_RESERVED: Final[re.Pattern[str]] = re.compile(
        r"([_\*\[\]\(\)\~\`\>\#\+\-\=\|\{\}\.\!])"
    )

    @staticmethod
    def format(event: NotificationEvent) -> str:
        """Render *event* as a MarkdownV2-safe Telegram message.

        Args:
            event: Notification event to render.

        Returns:
            Escaped MarkdownV2 string.
        """
        dispatch = {
            EventKind.SYSTEM_BOOT:     _MessageFormatter._fmt_boot,
            EventKind.SYSTEM_SHUTDOWN: _MessageFormatter._fmt_shutdown,
            EventKind.GAS_PAUSE:       _MessageFormatter._fmt_gas_pause,
            EventKind.GAS_RESUME:      _MessageFormatter._fmt_gas_resume,
            EventKind.GAS_DEFERRED:    _MessageFormatter._fmt_gas_deferred,
            EventKind.WALLET_SUCCESS:  _MessageFormatter._fmt_wallet_success,
            EventKind.WALLET_FAILURE:  _MessageFormatter._fmt_wallet_failure,
        }
        raw = dispatch.get(event.kind, _MessageFormatter._fmt_generic)(event)
        return _MessageFormatter._escape_md(raw)

    @staticmethod
    def _fmt_boot(e: NotificationEvent) -> str:
        return "🚀 *Wallet Orchestrator — STARTED*\nSystem initialised. All sessions open."

    @staticmethod
    def _fmt_shutdown(e: NotificationEvent) -> str:
        detail = f"\n{e.detail}" if e.detail else ""
        return f"🏁 *Wallet Orchestrator — COMPLETED*{detail}"

    @staticmethod
    def _fmt_gas_pause(e: NotificationEvent) -> str:
        gwei = f"{float(e.gas_price_gwei):.2f} Gwei" if e.gas_price_gwei is not None else "unknown"
        detail = f"\n{e.detail}" if e.detail else ""
        return f"⛽ *GAS ALERT — Paused*\nGas: `{gwei}`{detail}"

    @staticmethod
    def _fmt_gas_resume(e: NotificationEvent) -> str:
        gwei = f"{float(e.gas_price_gwei):.2f} Gwei" if e.gas_price_gwei is not None else "unknown"
        return f"✅ *GAS NORMAL — Resumed*\nGas: `{gwei}`"

    @staticmethod
    def _fmt_gas_deferred(e: NotificationEvent) -> str:
        gwei = f"{float(e.gas_price_gwei):.2f} Gwei" if e.gas_price_gwei is not None else "unknown"
        return (
            f"⛽ Gas çok yüksek ({gwei}). İşlem iptal edilmedi, "
            f"fırsat için 2 saat sonra uyanacağım."
        )

    @staticmethod
    def _fmt_wallet_success(e: NotificationEvent) -> str:
        wid = _mask_address(e.wallet_id or "unknown")
        tx = f"\n🔗 `{_mask_address(e.tx_hash)}`" if e.tx_hash else ""
        return f"✅ *Wallet {wid}* — Success{tx}"

    @staticmethod
    def _fmt_wallet_failure(e: NotificationEvent) -> str:
        wid = _mask_address(e.wallet_id or "unknown")
        reason = f"\n💬 {_mask_address(e.detail)}" if e.detail else ""
        return f"❌ *Wallet {wid}* — Failed{reason}"

    @staticmethod
    def _fmt_generic(e: NotificationEvent) -> str:
        return f"ℹ️ *{e.kind.name}*\n{_mask_address(e.detail or '')}"

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape MarkdownV2 reserved characters outside code spans.

        Args:
            text: Raw text possibly containing MarkdownV2 specials.

        Returns:
            Escaped string safe for the Telegram API.
        """
        parts = text.split("`")
        escaped = []
        for i, part in enumerate(parts):
            escaped.append(
                _MessageFormatter._MD_RESERVED.sub(r"\\\1", part)
                if i % 2 == 0
                else part
            )
        return "`".join(escaped)


# ---------------------------------------------------------------------------
# Command routing constants
# ---------------------------------------------------------------------------

# Telegram recommends long-poll timeout ≤ 60 s
_POLL_TIMEOUT_SECONDS: Final[int] = 30

_CMD_STATUS  = "/status"
_CMD_PAUSE   = "/pause"
_CMD_RESUME  = "/resume"
_CMD_STOP    = "/stop"
_CMD_HELP    = "/help"
_CMD_PNL     = "/pnl"
_CMD_TRADES  = "/trades"
_CMD_REGIME  = "/regime"
_CMD_PAPER   = "/paper"
_CMD_RISK    = "/risk"
_CMD_RETRAIN = "/retrain"
_CMD_MODE    = "/mode"
_CMD_HEALTH  = "/health"
_CMD_ROTATE_VAULT = "/rotate_vault_key"
# New commands (2.4)
_CMD_SETMODE       = "/setmode"
_CMD_WALLETS       = "/wallets"
_CMD_ALLOCATION    = "/allocation"
_CMD_RESET_CIRCUIT = "/reset_circuit"
_CMD_REPORT        = "/report"
_CMD_GAS           = "/gas"
# Commands requiring confirmation before destructive ops
_HEAVY_CMDS: frozenset[str] = frozenset({_CMD_RETRAIN})
# Per-user rate limit: max 10 commands/minute
_RATE_LIMIT_WINDOW: Final[float] = 60.0
_RATE_LIMIT_MAX: Final[int] = 10
_HEAVY_CMD_WINDOW: Final[float] = 3600.0
_HEAVY_CMD_MAX: Final[int] = 1

# Inline keyboard for /status (Section 5.5)
_STATUS_KEYBOARD: Final[dict] = {
    "inline_keyboard": [
        [
            {"text": "\U0001f504 Refresh", "callback_data": "cb_status"},
            {"text": "\U0001f4ca P&L",     "callback_data": "cb_pnl"},
            {"text": "\u26a0\ufe0f Risk",  "callback_data": "cb_risk"},
        ],
        [
            {"text": "\u23f8 Pause",        "callback_data": "cb_pause"},
            {"text": "\U0001f4cb Trades",   "callback_data": "cb_trades"},
            {"text": "\U0001f4f0 Regime",   "callback_data": "cb_regime"},
        ],
    ]
}

# ---------------------------------------------------------------------------
# CommandState — shared control-plane between CommandCenter and Orchestrator
# ---------------------------------------------------------------------------


class CommandState:
    """Shared asyncio control primitives between the command center and orchestrator.

    The orchestrator holds a reference to this object and checks its state
    before processing each wallet.  :class:`CommandCenter` sets/clears
    events in response to bot commands.

    Circuit Breaker: When daily drawdown or stop-loss hits, trading is halted
    for 24h. Operator can override via /resume.

    Example:
        .. code-block:: python

            state = CommandState()

            # In the orchestrator run loop:
            for wallet_id in plan:
                if state.is_stopped:
                    break
                await state.wait_for_resume()   # suspends while paused
                await run_wallet(wallet_id)
    """

    def __init__(self) -> None:
        # Inverted semantics: event SET = "not paused" (ready to run)
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._pause_event.set()   # Start unpaused
        # Circuit breaker: halt all trading until this timestamp (0 = not halted)
        self._circuit_breaker_halt_until: float = 0.0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """True if the orchestrator should suspend before the next wallet."""
        return not self._pause_event.is_set()

    @property
    def is_stopped(self) -> bool:
        """True if the orchestrator should drain and exit after the current TX."""
        return self._stop_event.is_set()

    @property
    def is_circuit_breaker_halted(self) -> bool:
        """True if trading is halted (daily drawdown or stop-loss). Clear via /resume."""
        import time
        if self._circuit_breaker_halt_until <= 0:
            return False
        if time.time() >= self._circuit_breaker_halt_until:
            self._circuit_breaker_halt_until = 0.0
            return False
        return True

    # ------------------------------------------------------------------
    # Orchestrator-facing awaitable
    # ------------------------------------------------------------------

    async def wait_for_resume(self) -> None:
        """Suspend the caller until /resume is received (or if not paused, return immediately).

        Returns:
            None — control resumes when :meth:`resume` is called.
        """
        await self._pause_event.wait()

    # ------------------------------------------------------------------
    # CommandCenter-facing setters
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Apply a global pause (invoked by /pause command)."""
        self._pause_event.clear()
        log.warning("CommandState: PAUSED by operator command.")

    def resume(self) -> None:
        """Clear the pause and circuit breaker halt (invoked by /resume command)."""
        self._pause_event.set()
        if self._circuit_breaker_halt_until > 0:
            self._circuit_breaker_halt_until = 0.0
            log.info("CommandState: Circuit breaker halt CLEARED by operator.")
        log.info("CommandState: RESUMED by operator command.")

    def trigger_circuit_breaker_halt(self, duration_seconds: float = 86400.0) -> None:
        """Halt all trading for duration_seconds (default 24h). Clear via /resume."""
        import time
        self._circuit_breaker_halt_until = time.time() + duration_seconds
        log.warning(
            f"CommandState: Circuit breaker HALT — all trading suspended for {duration_seconds / 3600:.1f}h. "
            "Send /resume to override.",
        )

    def stop(self) -> None:
        """Signal a graceful stop (invoked by /stop command).

        Also clears any active pause so the orchestrator can drain and exit
        rather than blocking indefinitely on :meth:`wait_for_resume`.
        """
        self._stop_event.set()
        self._pause_event.set()   # Unblock waiting coroutines so they see is_stopped
        log.warning("CommandState: STOP requested by operator command.")

    def request_retrain(self) -> None:
        """Flag that a manual model retrain has been requested by the operator."""
        self._retrain_requested = True
        log.info("CommandState: manual retrain requested by operator.")

    def clear_retrain(self) -> None:
        """Clear the retrain flag (call after retraining completes)."""
        self._retrain_requested = False

    @property
    def retrain_requested(self) -> bool:
        return getattr(self, "_retrain_requested", False)

    def request_mode_change(self, mode: str) -> None:
        """Request an execution mode change (invoked by /mode command)."""
        self._requested_mode: str = mode
        log.info(f"CommandState: mode change to '{mode}' requested by operator.")

    def pop_mode_change(self) -> str | None:
        """Return and clear any pending mode-change request."""
        mode = getattr(self, "_requested_mode", None)
        self._requested_mode = None  # type: ignore[assignment]
        return mode


# ---------------------------------------------------------------------------
# ConfirmationGate — wait for "EVET" before proceeding with a transaction
# ---------------------------------------------------------------------------

_CMD_EVET = "evet"


class ConfirmationGate:
    """Shared gate for Telegram transaction confirmation (EVET = Yes in Turkish).

    Before sending a swap transaction, the orchestrator sends a confirmation
    message and awaits :meth:`wait_for_evet`.  The :class:`CommandCenter`
    signals this gate when it receives "EVET" from the authorised chat.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._confirmed: bool = False
        self._lock = asyncio.Lock()

    def request_confirmation(self) -> None:
        """Reset the gate for a new confirmation request."""
        self._event.clear()
        self._confirmed = False

    def signal_evet(self) -> None:
        """Called by CommandCenter when "EVET" is received."""
        self._confirmed = True
        self._event.set()

    async def wait_for_evet(self, timeout_seconds: float = 300.0) -> bool:
        """Block until EVET is received or timeout.

        Returns:
            True if EVET was received, False if timeout.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout_seconds)
            return self._confirmed
        except TimeoutError:
            return False


# ---------------------------------------------------------------------------
# StatusProvider — injectable live-status reporter for /status command
# ---------------------------------------------------------------------------


class StatusProvider:
    """Base class for providing a live /status reply string.

    Subclass this and pass an instance to :class:`CommandCenter` to surface
    real-time wallet states.  The default implementation returns a placeholder.

    Example:
        .. code-block:: python

            class MyStatus(StatusProvider):
                def __init__(self, orchestrator):
                    self._orch = orchestrator

                async def build_status_message(self) -> str:
                    lines = []
                    for wid, result in self._orch.live_results.items():
                        lines.append(f"{'✅' if result.success else '❌'} {wid}")
                    return "\\n".join(lines) or "No wallets processed yet."
    """

    async def build_status_message(self) -> str:
        """Build and return the /status reply string.

        Returns:
            Plain text status message (address masking applied by caller).
        """
        return "\u2139\ufe0f Status provider not configured."


class LiveStatusProvider(StatusProvider):
    """Full portfolio dashboard pulling live on-chain data (Section 5.2).

    Accepts any orchestrator object via duck typing; all attributes fall back
    gracefully when unavailable so the bot never crashes on missing data.
    """

    _VALID_MODES: tuple[str, ...] = (
        "legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"
    )

    def __init__(self, orchestrator: object = None) -> None:
        self._orch = orchestrator

    def _attr(self, name: str, default: Any = None) -> Any:
        return getattr(self._orch, name, default)

    # ------------------------------------------------------------------
    # Section 5.2 — /status full dashboard
    # ------------------------------------------------------------------

    async def build_status_message(self) -> str:
        return self._build_dashboard()

    async def build_risk_message(self) -> str:
        """Section 5.3 — /risk full metrics."""
        return self._build_risk()

    async def build_pnl_message(self) -> str:
        return self._build_pnl()

    async def build_trades_message(self) -> str:
        import time as _t
        ts = _t.strftime("%H:%M UTC", _t.gmtime())
        header = f"Last Trades                  {ts}\n" + "\u2501" * 40
        try:
            from wallet_manager.config import get_settings as _gs
            _cfg = _gs()
            if _cfg.paper_trade or _cfg.execution_mode == "robust_paper":
                from wallet_manager.strategies.paper_engine import _engine_instance
                if _engine_instance is not None and _engine_instance._initialized:
                    trades = await _engine_instance._ledger.get_recent_trades(10)
                    if trades:
                        lines = [header]
                        for t in trades:
                            pnl = t["pnl_usd"]
                            sign = "+" if pnl >= 0 else ""
                            lines.append(
                                f"{t['direction']:4s} | {t['pair']:9s} | "
                                f"${t['size_usd']:.2f} @ ${t['fill_price']:.2f} | "
                                f"slip={t['slippage_bps']:.1f}bps | "
                                f"PnL: {sign}${pnl:.4f}"
                            )
                        return "\n".join(lines)
                    return header + "\nNo paper trades recorded yet."
        except Exception:
            pass
        return self._build_trades()

    async def build_regime_message(self) -> str:
        return self._build_regime()

    async def build_paper_message(self) -> str:
        return self._build_paper()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    @staticmethod
    def _fmtu(v: float) -> str:
        return f"{v:,.2f}"

    def _build_dashboard(self) -> str:
        import time as _t
        ts = _t.strftime("%H:%M UTC", _t.gmtime())
        total = float(self._attr("portfolio_total_usd") or 0.0)
        chg_usd = float(self._attr("portfolio_24h_change_usd") or 0.0)
        chg_pct = float(self._attr("portfolio_24h_change_pct") or 0.0)
        sign = "+" if chg_usd >= 0 else ""

        pillars = self._attr("pillars") or {}
        y  = pillars.get("YIELD",   {})
        a  = pillars.get("ALGO",    {})
        s  = pillars.get("SHORT",   {})
        ai = pillars.get("AIRDROP", {})

        wallet  = _mask_address(str(self._attr("wallet_address") or "0x????...????"))
        eth_b   = float(self._attr("eth_balance")   or 0.0)
        eth_p   = float(self._attr("eth_price_usd") or 3000.0)
        usdc_b  = float(self._attr("usdc_balance")  or 0.0)
        aave_b  = float(self._attr("aave_balance")  or 0.0)
        gas     = float(self._attr("gas_gwei")       or 0.0)
        mode    = str(self._attr("execution_mode")   or "unknown")
        regime  = str(self._attr("current_regime")   or "UNKNOWN")
        up_s    = float(self._attr("uptime_seconds") or 0.0)
        uh, um  = int(up_s // 3600), int((up_s % 3600) // 60)
        hb_ago  = float(self._attr("last_heartbeat_ago_minutes") or 0.0)
        ps      = float(self._attr("paper_sharpe") or 0.0)
        rs      = float(self._attr("real_sharpe")  or 0.0)
        last_tx = _mask_address(str(self._attr("last_tx_hash") or "0x????...????"))
        tx_ago  = float(self._attr("last_tx_ago_minutes") or 0.0)
        rebal   = "Yes" if self._attr("rebalance_needed") else "No"
        f = self._fmtu
        return (
            f"Portfolio Dashboard          {ts}\n"
            "\u2501" * 40 + "\n"
            f"Total Portfolio:  ${f(total)}\n"
            f"24h Change:       {sign}${f(chg_usd)}  ({sign}{chg_pct:.2f}%)\n"
            "\n"
            "Pillar Breakdown:\n"
            f"YIELD   (40%): ${f(y.get('value',0))} / target ${f(y.get('target',0))}  [Aave APY: {y.get('apy',0.0):.2f}%]\n"
            f"ALGO    (35%): ${f(a.get('value',0))} / target ${f(a.get('target',0))}  [Open: {a.get('open_positions',0)} positions]\n"
            f"SHORT   (10%): ${f(s.get('value',0))} / target ${f(s.get('target',0))}  [Leverage: {s.get('leverage',0.0):.1f}x]\n"
            f"AIRDROP (15%): ${f(ai.get('value',0))} / target ${f(ai.get('target',0))}  [Next swap: {ai.get('next_swap_minutes',0):.0f}m]\n"
            "\n"
            f"Wallet: {wallet}\n"
            f"   ETH:  {eth_b:.4f} ETH (~${f(eth_b * eth_p)})\n"
            f"   USDC: ${f(usdc_b)}\n"
            f"   Aave: ${f(aave_b)}\n"
            "\n"
            f"Gas: {gas:.0f} Gwei\n"
            f"Mode: {mode} | Regime: {regime}\n"
            f"Uptime: {uh}h {um}m | Heartbeat: {hb_ago:.0f}m ago\n"
            f"Paper Sharpe: {ps:.2f} | Real Sharpe: {rs:.2f}\n"
            f"Last TX: {last_tx} ({tx_ago:.0f} min ago)\n"
            f"Rebalance needed: {rebal}"
        )

    def _build_risk(self) -> str:
        import time as _t
        ts = _t.strftime("%H:%M UTC", _t.gmtime())
        dd      = float(self._attr("drawdown_pct")        or 0.0)
        var99   = float(self._attr("var_99_usd")          or 0.0)
        cvar99  = float(self._attr("cvar_99_usd")         or 0.0)
        kelly   = float(self._attr("kelly_fraction")      or 0.0)
        wr      = float(self._attr("win_rate_50")         or 0.0)
        sh7     = float(self._attr("sharpe_7d")           or 0.0)
        so7     = float(self._attr("sortino_7d")          or 0.0)
        sigma   = float(self._attr("garch_sigma")         or 0.0)
        cb_open = bool(self._attr("circuit_breaker_open") or False)
        cb_fail = int(self._attr("circuit_failures")      or 0)
        ir_z    = float(self._attr("ir_zscore")           or 0.0)
        ir_ofi  = float(self._attr("ir_ofi")              or 0.0)
        ir_ml   = float(self._attr("ir_ml")               or 0.0)
        hl_min  = float(self._attr("signal_half_life_minutes") or 0.0)
        exec_s  = float(self._attr("signal_exec_seconds")     or 0.0)
        regime  = str(self._attr("current_regime")            or "UNKNOWN")
        cb_str  = "OPEN" if cb_open else "CLOSED"
        f = self._fmtu
        return (
            f"Risk Metrics                 {ts}\n"
            "\u2501" * 40 + "\n"
            f"Drawdown:    {dd:.2f}%  (limit: 10%)\n"
            f"VaR  (99%):  ${f(var99)}\n"
            f"CVaR (99%):  ${f(cvar99)}\n"
            f"Kelly:       {kelly:.2f}   (half-Kelly: {kelly/2:.2f})\n"
            f"Win Rate:    {wr*100:.1f}%  (last 50 trades)\n"
            f"Sharpe (7d): {sh7:.2f} | Sortino: {so7:.2f}\n"
            f"Vol Regime:  {regime} (sigma: {sigma:.4f})\n"
            f"Circuit Breaker: {cb_str} ({cb_fail} failures)\n"
            f"IR Tracker:  zscore {ir_z:.2f} | ofi {ir_ofi:.2f} | ml {ir_ml:.2f}\n"
            f"Alpha Decay: last signal half-life {hl_min:.0f}m (executed in {exec_s:.0f}s)"
        )

    def _build_pnl(self) -> str:
        import time as _t
        ts   = _t.strftime("%H:%M UTC", _t.gmtime())
        pnl  = self._attr("pnl_summary") or {}
        d24  = float(pnl.get("24h", 0.0))
        d7   = float(pnl.get("7d",  0.0))
        d30  = float(pnl.get("30d", 0.0))
        s24  = "+" if d24 >= 0 else ""
        s7   = "+" if d7  >= 0 else ""
        s30  = "+" if d30 >= 0 else ""
        f = self._fmtu
        return (
            f"P&L Summary                  {ts}\n"
            "\u2501" * 40 + "\n"
            f"24h:   {s24}${f(d24)}\n"
            f" 7d:   {s7}${f(d7)}\n"
            f"30d:   {s30}${f(d30)}"
        )

    def _build_trades(self) -> str:
        import time as _t
        ts     = _t.strftime("%H:%M UTC", _t.gmtime())
        trades = self._attr("last_trades") or []
        header = f"Last Trades                  {ts}\n" + "\u2501" * 40
        if not trades:
            return header + "\nNo trades recorded yet."
        lines = [header]
        for t in trades[:10]:
            pnl = float(t.get("pnl", 0.0))
            s   = "+" if pnl >= 0 else ""
            lines.append(
                f"{t.get('strategy','?'):8s} | {t.get('venue','?'):8s} | "
                f"{t.get('entry',0.0):.4f} -> {t.get('exit',0.0):.4f} | "
                f"PnL: {s}${pnl:,.2f}"
            )
        return "\n".join(lines)

    def _build_regime(self) -> str:
        import time as _t
        ts      = _t.strftime("%H:%M UTC", _t.gmtime())
        regime  = str(self._attr("current_regime")  or "UNKNOWN")
        nxt_ev  = str(self._attr("next_calendar_event") or "No upcoming events")
        nxt_h   = self._attr("next_calendar_event_hours")
        ev_str  = nxt_ev + (f" (in {nxt_h:.1f}h)" if nxt_h is not None else "")
        vel     = float(self._attr("velocity_ewm") or 0.0)
        probs   = self._attr("hmm_state_probs") or {}
        prob_s  = " | ".join(f"{k}: {v:.0%}" for k, v in list(probs.items())[:3]) or "N/A"
        return (
            f"Regime Monitor               {ts}\n"
            "\u2501" * 40 + "\n"
            f"Current regime:  {regime}\n"
            f"EWM velocity:    {vel:.6f}\n"
            f"HMM posteriors:  {prob_s}\n"
            f"Next event:      {ev_str}"
        )

    def _build_paper(self) -> str:
        import time as _t
        ts  = _t.strftime("%H:%M UTC", _t.gmtime())
        pvr = self._attr("paper_vs_real") or {}
        ps  = float(pvr.get("paper_sharpe",  self._attr("paper_sharpe")  or 0.0))
        rs  = float(pvr.get("real_sharpe",   self._attr("real_sharpe")   or 0.0))
        pp  = float(pvr.get("paper_pnl_6h",  0.0))
        rp  = float(pvr.get("real_pnl_6h",   0.0))
        dr  = float(pvr.get("drift_pct",      0.0))
        f   = self._fmtu
        return (
            f"Paper vs Real (last 6h)      {ts}\n"
            "\u2501" * 40 + "\n"
            f"Paper  — PnL: ${f(pp)} | Sharpe: {ps:.2f}\n"
            f"Real   — PnL: ${f(rp)} | Sharpe: {rs:.2f}\n"
            f"Drift: {dr:.2f}%"
        )


# ---------------------------------------------------------------------------
# CommandCenter — inbound bot command processor
# ---------------------------------------------------------------------------


class CommandCenter:
    """Long-polling inbound command processor for the Telegram Bot API.

    Runs as a background asyncio task alongside :class:`TelegramNotifier`.
    Shares the same :class:`_TelegramClient` session to avoid duplicate
    TCP connections.

    Args:
        credentials: Shared Telegram credentials.
        command_state: Shared :class:`CommandState` for the orchestrator.
        client: Opened :class:`_TelegramClient` instance.
        status_provider: Optional :class:`StatusProvider` for /status replies.

    Example:
        .. code-block:: python

            state = CommandState()
            async with TelegramNotifier() as notifier:
                center = notifier.build_command_center(state)
                await center.start()
                ...
                await center.stop()
    """

    def __init__(
        self,
        credentials: _TelegramCredentials,
        command_state: CommandState,
        client: _TelegramClient,
        status_provider: StatusProvider | None = None,
        confirmation_gate: ConfirmationGate | None = None,
    ) -> None:
        self._creds = credentials
        self._state = command_state
        self._client = client
        self._status_provider = status_provider or StatusProvider()
        self._confirmation_gate = confirmation_gate
        self._worker_task: asyncio.Task[None] | None = None
        self._last_update_id: int = 0
        self._pending_retrain: bool = False
        self._pending_mode: str = ""
        self._pending_reset_circuit: str = ""
        # Per-user rate limiting: {chat_id: [(timestamp, cmd), ...]}
        self._cmd_history: dict[str, list[float]] = {}
        self._heavy_cmd_history: dict[str, list[float]] = {}

    async def start(self) -> None:
        """Launch the background polling worker."""
        self._worker_task = asyncio.ensure_future(self._poll_loop())
        log.debug("CommandCenter polling worker started.")

    async def stop(self) -> None:
        """Cancel the polling worker gracefully."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        log.debug("CommandCenter stopped.")

    # ------------------------------------------------------------------
    # Background polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Continuously long-poll Telegram and dispatch incoming commands.

        A single non-fatal error never kills the loop — errors are logged
        and the loop resumes after a short back-off.
        """
        while True:
            try:
                updates = await self._client.get_updates(
                    offset=self._last_update_id + 1,
                    timeout=_POLL_TIMEOUT_SECONDS,
                )
                for update in updates:
                    self._last_update_id = max(
                        self._last_update_id,
                        update.get("update_id", 0),
                    )
                    await self._dispatch(update)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning(f"CommandCenter: poll error ({exc}). Retrying in 5s.")
                await asyncio.sleep(5.0)

    async def _dispatch(self, update: dict) -> None:
        """Route a single Update to the appropriate command handler.

        Only processes messages from the authorised ``chat_id``.

        Args:
            update: Raw Telegram Update object as a dict.
        """
        # Handle callback_query from inline keyboard buttons (Section 5.5)
        callback = update.get("callback_query", {})
        if callback:
            await self._dispatch_callback(callback)
            return

        message = update.get("message", {})
        if not message:
            return

        from_chat_id = str(message.get("chat", {}).get("id", ""))

        # Security gate: silently ignore any message not from the authorised chat
        if from_chat_id != self._creds.chat_id:
            return

        text: str = (message.get("text") or "").strip()
        text_lower = text.lower()
        log.info(f"CommandCenter: received command '{text_lower}'.")

        # Per-user rate limiting
        if not self._check_rate_limit(from_chat_id, text_lower):
            return

        if text_lower.startswith(_CMD_STATUS):
            await self._handle_status(from_chat_id)
        elif text_lower.startswith(_CMD_PAUSE):
            await self._handle_pause(from_chat_id)
        elif text_lower.startswith(_CMD_RESUME):
            await self._handle_resume(from_chat_id)
        elif text_lower.startswith(_CMD_STOP):
            await self._handle_stop(from_chat_id)
        elif text_lower.startswith(_CMD_HELP):
            await self._handle_help(from_chat_id)
        elif text_lower.startswith(_CMD_PNL):
            await self._handle_pnl(from_chat_id)
        elif text_lower.startswith(_CMD_TRADES):
            await self._handle_trades(from_chat_id)
        elif text_lower.startswith(_CMD_REGIME):
            await self._handle_regime(from_chat_id)
        elif text_lower.startswith(_CMD_PAPER):
            await self._handle_paper(from_chat_id)
        elif text_lower.startswith(_CMD_RISK):
            await self._handle_risk(from_chat_id)
        elif text_lower.startswith(_CMD_RETRAIN):
            confirmed = "confirm" in text_lower
            await self._handle_retrain(from_chat_id, confirmed)
        elif text_lower.startswith(_CMD_MODE):
            arg = text_lower[len(_CMD_MODE):].strip()
            await self._handle_mode(from_chat_id, arg)
        elif text_lower.startswith(_CMD_HEALTH):
            await self._handle_health(from_chat_id)
        elif text_lower.startswith(_CMD_ROTATE_VAULT):
            arg = text[len(_CMD_ROTATE_VAULT):].strip()
            await self._handle_rotate_vault(from_chat_id, arg)
        elif text_lower.startswith(_CMD_SETMODE):
            arg = text[len(_CMD_SETMODE):].strip()
            await self._handle_setmode(from_chat_id, arg)
        elif text_lower.startswith(_CMD_WALLETS):
            await self._handle_wallets(from_chat_id)
        elif text_lower.startswith(_CMD_ALLOCATION):
            await self._handle_allocation(from_chat_id)
        elif text_lower.startswith(_CMD_RESET_CIRCUIT):
            arg = text[len(_CMD_RESET_CIRCUIT):].strip()
            await self._handle_reset_circuit(from_chat_id, arg)
        elif text_lower.startswith(_CMD_REPORT):
            arg = text_lower[len(_CMD_REPORT):].strip()
            await self._handle_report(from_chat_id, arg)
        elif text_lower.startswith(_CMD_GAS):
            await self._handle_gas(from_chat_id)
        elif text.strip().upper() == _CMD_EVET.upper() and self._confirmation_gate:
            self._confirmation_gate.signal_evet()
            await self._reply(from_chat_id, "\u2705 *Onay al\u0131nd\u0131* \\— i\u015flem devam ediyor\\.")
        else:
            await self._reply(
                from_chat_id,
                "\u2753 Unknown command\\. Send /help for available commands\\.",
            )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _check_rate_limit(self, chat_id: str, cmd: str) -> bool:
        """Return True if the command is within rate limits; False and send warning otherwise."""
        import asyncio as _asyncio
        now = time.monotonic()

        # Heavy command rate limit (1/hour)
        is_heavy = any(cmd.startswith(h) for h in _HEAVY_CMDS)
        if is_heavy:
            history = self._heavy_cmd_history.setdefault(chat_id, [])
            history[:] = [t for t in history if now - t < _HEAVY_CMD_WINDOW]
            if len(history) >= _HEAVY_CMD_MAX:
                wait = int(_HEAVY_CMD_WINDOW - (now - history[0]))
                _asyncio.ensure_future(
                    self._reply(chat_id, f"⏸ Çok fazla ağır komut\\. {wait} saniye bekleyin\\.")
                )
                return False
            history.append(now)
            return True

        # Normal rate limit (10/minute)
        history = self._cmd_history.setdefault(chat_id, [])
        history[:] = [t for t in history if now - t < _RATE_LIMIT_WINDOW]
        if len(history) >= _RATE_LIMIT_MAX:
            wait = int(_RATE_LIMIT_WINDOW - (now - history[0]))
            try:
                _asyncio.ensure_future(
                    self._reply(chat_id, f"⏸ Çok fazla komut\\. {wait} saniye bekleyin\\.")
                )
            except RuntimeError:
                pass  # No event loop in sync test context — skip fire-and-forget
            return False
        history.append(now)
        return True

    async def _handle_setmode(self, chat_id: str, arg: str) -> None:
        """Change execution mode without restart. Requires 'confirm' for dangerous modes."""
        valid = ["legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"]
        dangerous = {"ml_quant", "robust", "robust_live"}

        parts = arg.split()
        mode = parts[0].lower() if parts else ""
        confirmed = len(parts) > 1 and parts[-1].lower() == "confirm"

        if not mode:
            current = "unknown"
            try:
                from wallet_manager.config import get_settings
                current = get_settings().execution_mode
            except Exception:
                pass
            mode_list = "\n".join(f"  \\• {m}" for m in valid)
            await self._reply(
                chat_id,
                f"Current mode: *{_MessageFormatter._escape_md(current)}*\n"
                f"Valid modes:\n{mode_list}\n"
                f"Usage: /setmode \\<mode\\>\n"
                f"Dangerous modes require: /setmode \\<mode\\> confirm",
            )
            return

        if mode not in valid:
            await self._reply(
                chat_id,
                f"\u274c Unknown mode '{_MessageFormatter._escape_md(mode)}'\n"
                f"Valid: {_MessageFormatter._escape_md(', '.join(valid))}",
            )
            return

        if mode in dangerous and not confirmed:
            await self._reply(
                chat_id,
                f"\u26a0\ufe0f Mode *{_MessageFormatter._escape_md(mode)}* is a production mode\\.\n"
                f"To confirm: `/setmode {_MessageFormatter._escape_md(mode)} confirm`",
            )
            return

        self._state.request_mode_change(mode)
        await self._reply(
            chat_id,
            f"\u2705 Execution mode changed to *{_MessageFormatter._escape_md(mode)}*\\.\n"
            f"Orchestrator will apply on next loop iteration\\.",
        )

    async def _handle_wallets(self, chat_id: str) -> None:
        """Show summary of all active wallets."""
        try:
            from wallet_manager.orchestrator import _get_circuit_breakers, _get_wallet_configs
            wallets = _get_wallet_configs() or []
            cbs = _get_circuit_breakers() or {}

            if not wallets:
                await self._reply(chat_id, "\U0001f4bc No active wallets configured\\.")
                return

            lines = [f"\U0001f4bc *Active Wallets \\({len(wallets)}\\)*\n"]
            for w in wallets:
                addr = w.address or "unknown"
                masked = f"{addr[:6]}\\.\\.\\..{addr[-4:]}" if len(addr) >= 10 else addr
                cb = cbs.get(w.wallet_id)
                cb_state = "CLOSED"
                if cb is not None:
                    try:
                        cb_state = str(cast(Any, cb).state.name)
                    except Exception:
                        cb_state = "UNKNOWN"
                cb_emoji = "\u2705" if cb_state == "CLOSED" else "\u26a0\ufe0f"
                lines.append(
                    f"{_MessageFormatter._escape_md(w.wallet_id)} \\| "
                    f"`{masked}` \\| "
                    f"{cb_emoji} {_MessageFormatter._escape_md(cb_state)}"
                )
            await self._reply(chat_id, "\n".join(lines))
        except Exception as exc:
            await self._reply(chat_id, f"\u26a0\ufe0f Wallet info unavailable: {_MessageFormatter._escape_md(str(exc)[:80])}")

    async def _handle_allocation(self, chat_id: str) -> None:
        """Show current portfolio allocation vs targets."""
        try:
            from wallet_manager.orchestrator import _get_portfolio_manager
            pm = _get_portfolio_manager()
            if pm is None:
                await self._reply(chat_id, "\u26a0\ufe0f Portfolio manager not available\\.")
                return
            snap = pm.get_snapshot()
            total = snap.total_value_usd
            drift_threshold = 0.05
            try:
                from wallet_manager.config import get_settings
                drift_threshold = get_settings().rebalance_drift_threshold
            except Exception:
                pass

            needs_rb = pm.needs_rebalancing()
            rb_emoji = "\u26a0\ufe0f" if needs_rb else "\u2705"
            lines = [
                f"\U0001f4ca *Portfolio Allocation* \\(Total: ${_MessageFormatter._escape_md(f'{total:.2f}')}\\)",
                f"Rebalance threshold: {_MessageFormatter._escape_md(f'{drift_threshold*100:.1f}')}%  {rb_emoji}",
                "",
            ]
            for alloc in snap.allocations:
                current_pct = (alloc.current_usd / total * 100) if total > 0 else 0
                target_pct = alloc.weight * 100
                delta_pct = current_pct - target_pct
                drift_emoji = "\u26a0\ufe0f" if abs(delta_pct) > drift_threshold * 100 else "\u2705"
                lines.append(
                    f"{drift_emoji} *{alloc.strategy.value}*: "
                    f"\\${_MessageFormatter._escape_md(f'{alloc.current_usd:.2f}')} "
                    f"\\({_MessageFormatter._escape_md(f'{current_pct:.1f}')}% vs "
                    f"{_MessageFormatter._escape_md(f'{target_pct:.0f}')}% target, "
                    f"delta {_MessageFormatter._escape_md(f'{delta_pct:+.1f}')}%\\)"
                )
            await self._reply(chat_id, "\n".join(lines))
        except Exception as exc:
            await self._reply(chat_id, f"\u26a0\ufe0f Allocation data unavailable: {_MessageFormatter._escape_md(str(exc)[:80])}")

    async def _handle_reset_circuit(self, chat_id: str, arg: str) -> None:
        """Force reset a wallet's circuit breaker. Requires 'confirm'."""
        parts = arg.split()
        wallet_id = parts[0] if parts else ""
        confirmed = len(parts) > 1 and parts[-1].lower() == "confirm"

        if not wallet_id:
            await self._reply(
                chat_id,
                "\u26a0\ufe0f Usage: `/reset_circuit <wallet_id> confirm`",
            )
            return

        if not confirmed:
            self._pending_reset_circuit = wallet_id
            await self._reply(
                chat_id,
                f"\u26a0\ufe0f Reset circuit breaker for *{_MessageFormatter._escape_md(wallet_id)}*?\n"
                f"Send: `/reset_circuit {_MessageFormatter._escape_md(wallet_id)} confirm`",
            )
            return

        try:
            from wallet_manager.orchestrator import _get_circuit_breakers
            cbs = _get_circuit_breakers() or {}
            cb = cbs.get(wallet_id)
            if cb is None:
                await self._reply(chat_id, f"\u274c Wallet `{_MessageFormatter._escape_md(wallet_id)}` not found\\.")
                return
            cast(Any, cb).force_close()
            self._pending_reset_circuit = ""
            await self._reply(
                chat_id,
                f"\u2705 Circuit breaker for *{_MessageFormatter._escape_md(wallet_id)}* reset to CLOSED\\.",
            )
        except Exception as exc:
            await self._reply(chat_id, f"\u274c Reset failed: {_MessageFormatter._escape_md(str(exc)[:80])}")

    async def _handle_report(self, chat_id: str, period: str) -> None:
        """Send P&L report for the specified period."""
        valid_periods = {"daily", "weekly", "monthly"}
        if period not in valid_periods:
            await self._reply(
                chat_id,
                "\u26a0\ufe0f Usage: `/report <period>`\nValid periods: daily, weekly, monthly",
            )
            return
        try:
            import datetime

            from wallet_manager.config import get_settings
            from wallet_manager.resilience.analytics import AnalyticsDB
            cfg = get_settings()
            db = AnalyticsDB(str(cfg.quant_analytics_db_path))

            days_map = {"daily": 1, "weekly": 7, "monthly": 30}
            days = days_map[period]
            date_str = datetime.date.today().isoformat()

            summary = await db.get_daily_summary(date_str)
            lines = [
                f"\U0001f4ca *{_MessageFormatter._escape_md(period.capitalize())} Report*",
                f"Period: last {days} day\\(s\\)",
                f"Trades: {_MessageFormatter._escape_md(str(summary.get('trade_count', 0)))}",
                "P&L: \\$" + _MessageFormatter._escape_md(f"{summary.get('total_pnl_usd', 0.0):.2f}"),
                "Avg Gas: " + _MessageFormatter._escape_md(f"{summary.get('avg_gas_gwei', 0.0):.2f}") + " Gwei",
            ]
            await self._reply(chat_id, "\n".join(lines))
        except Exception as exc:
            await self._reply(chat_id, f"\u26a0\ufe0f Report unavailable: {_MessageFormatter._escape_md(str(exc)[:80])}")

    async def _handle_gas(self, chat_id: str) -> None:
        """Show current gas price and wallets affected."""
        try:
            from wallet_manager.orchestrator import _get_gas_price_gwei, _get_wallet_configs

            gas = _get_gas_price_gwei()
            wallets = _get_wallet_configs() or []

            gas_str = f"{gas:.2f}" if gas is not None else "N/A"
            lines = [f"\u26fd *Current Gas Price*: {_MessageFormatter._escape_md(gas_str)} Gwei\n"]

            if gas is not None and wallets:
                for w in wallets:
                    if gas > w.max_gas_price_gwei:
                        lines.append(
                            f"\u274c {_MessageFormatter._escape_md(w.wallet_id)}: suspended "
                            f"\\(max {_MessageFormatter._escape_md(f'{w.max_gas_price_gwei:.1f}')} Gwei\\)"
                        )
                    else:
                        lines.append(
                            f"\u2705 {_MessageFormatter._escape_md(w.wallet_id)}: active"
                        )
            await self._reply(chat_id, "\n".join(lines))
        except Exception as exc:
            await self._reply(chat_id, f"\u26a0\ufe0f Gas data unavailable: {_MessageFormatter._escape_md(str(exc)[:80])}")

    async def _handle_health(self, chat_id: str) -> None:
        """Reply with system health report."""
        lines = [
            "\U0001f7e2 *System Health Report*",
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
        ]
        # RPC endpoints
        try:
            from wallet_manager.orchestrator import _get_rpc_router
            router = _get_rpc_router()
            if router is not None:
                status = router.pool_status()
                for chain_id, providers in status.items():
                    healthy = sum(1 for p in providers if p.get("available"))
                    total = len(providers)
                    latencies = []
                    for p in providers:
                        url = p.get("url", "")
                        lat = router.latency_score(chain_id, url)
                        if lat < 1.0:
                            latencies.append(f"{int(1.0/lat - 1)}ms")
                    lat_str = " / ".join(latencies) if latencies else "N/A"
                    lines.append(f"\U0001f517 RPC endpoints: {healthy}/{total} healthy  \\(latencies: {_MessageFormatter._escape_md(lat_str)}\\)")
            else:
                lines.append("\U0001f517 RPC: unavailable")
        except Exception as exc:
            lines.append(f"\U0001f517 RPC: error \\({_MessageFormatter._escape_md(str(exc)[:40])}\\)")

        # Drift detector
        try:
            from wallet_manager.task_factories import _drift_detector

            if _drift_detector is not None:
                drift_level = getattr(_drift_detector, "_drift_level", None)
                drift_name = drift_level.name if drift_level is not None else "NOMINAL"
            else:
                drift_name = "N/A"
            lines.append(f"\u26a0\ufe0f Drift detector: {_MessageFormatter._escape_md(drift_name)}")
        except Exception:
            lines.append("\u26a0\ufe0f Drift detector: N/A")

        # Aave APY
        try:
            from wallet_manager.config import get_settings
            from wallet_manager.strategies.yield_strategy import get_apy_monitor

            cfg = get_settings()
            min_apy = cfg.yield_min_apy_pct
            configs = getattr(self._status_provider, "_configs", [])
            _rpc = configs[0].rpc_url if configs else ""
            if _rpc:
                current_apy = await get_apy_monitor().get_current_apy(_rpc)
                apy_pct = current_apy * 100
                ok_emoji = "\u2705" if apy_pct >= min_apy else "\u274c"
                lines.append(
                    f"\U0001f4ca Aave APY: {apy_pct:.2f}%  {ok_emoji} "
                    f"\\(threshold: {min_apy:.1f}%\\)"
                )
        except Exception:
            lines.append("\U0001f4ca Aave APY: N/A")

        # FeatureDriftMonitor
        try:
            from wallet_manager.strategies.robust_alpha import get_feature_drift_monitor
            _fdm_report = get_feature_drift_monitor().get_drift_report()
            _fdm_status = (
                f"FDM: {len(_fdm_report.get('drifted_features', []))} drifted | "
                f"worst={_fdm_report.get('worst_feature', 'none')} "
                f"KL={_fdm_report.get('worst_kl', 0.0):.4f}"
            )
        except Exception:
            _fdm_status = "FDM: unavailable"
        lines.append(f"\U0001f4ca {_MessageFormatter._escape_md(_fdm_status)}")

        # Disk
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            lines.append(f"\U0001f4be Disk: {free / 1e9:.1f} GB free")
        except Exception:
            lines.append("\U0001f4be Disk: N/A")

        msg = "\n".join(lines)
        await self._reply(chat_id, msg)

    async def _handle_rotate_vault(self, chat_id: str, new_key: str) -> None:
        """Handle /rotate_vault_key <new_key> — two-step confirmation required."""
        if not new_key:
            await self._reply(
                chat_id,
                "\u26a0\ufe0f Usage: `/rotate_vault_key <new_fernet_key>`\n"
                "Add `confirm` at end to confirm: `/rotate_vault_key <key> confirm`",
            )
            return

        parts = new_key.split()
        confirmed = len(parts) > 1 and parts[-1].lower() == "confirm"
        key_value = parts[0]

        if not confirmed:
            await self._reply(
                chat_id,
                "\u26a0\ufe0f *Vault key rotation — confirmation required*\n"
                f"To confirm, send:\n`/rotate_vault_key {key_value[:8]}\\.\\.\\. confirm`",
            )
            return

        try:
            import pathlib

            # Validate new key by constructing a Vault instance with it temporarily
            from cryptography.fernet import Fernet

            from wallet_manager.config import get_settings
            from wallet_manager.security.vault_manager import get_vault_manager
            Fernet(key_value.encode("utf-8"))  # raises if invalid

            # Update the environment / settings (in-memory only)
            import os
            os.environ["VAULT_KEY"] = key_value
            log.info("VaultKeyManager: vault key rotated by operator.")

            # Record rotation timestamp
            cfg = get_settings()
            wallets_path = pathlib.Path(
                getattr(cfg, "wallet_config_path", "wallets.json")
            )
            get_vault_manager().record_rotation(wallets_path)

            await self._reply(chat_id, "\u2705 Vault key rotated successfully\\.")
        except Exception as exc:
            err = _MessageFormatter._escape_md(str(exc)[:100])
            await self._reply(chat_id, f"\u274c Vault key rotation failed: {err}")

    async def _handle_status(self, chat_id: str) -> None:
        """Reply with live portfolio dashboard + inline keyboard (Section 5.2 + 5.5)."""
        try:
            raw_status = await self._status_provider.build_status_message()
            safe_status = _mask_address(raw_status)
            escaped = _MessageFormatter._escape_md(safe_status)
            parts = _split_message(escaped)
            # Send inline keyboard only on the last part
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    try:
                        await self._client.send_message_with_keyboard(
                            part, _STATUS_KEYBOARD, chat_id=chat_id
                        )
                    except Exception:
                        await self._reply(chat_id, part)
                else:
                    await self._reply(chat_id, part)
        except Exception as exc:
            err = _MessageFormatter._escape_md(str(exc))
            await self._reply(chat_id, f"\u26a0\ufe0f Status unavailable: {err}")

    async def _dispatch_callback(self, callback: dict) -> None:
        """Route inline keyboard button presses (Section 5.5)."""
        from_chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
        cb_id = callback.get("id", "")

        # Always acknowledge the callback to remove the spinner
        await self._client.answer_callback_query(cb_id)

        if from_chat_id != self._creds.chat_id:
            return

        data = callback.get("data", "")
        if data == "cb_status":
            await self._handle_status(from_chat_id)
        elif data == "cb_pnl":
            await self._handle_pnl(from_chat_id)
        elif data == "cb_risk":
            await self._handle_risk(from_chat_id)
        elif data == "cb_pause":
            await self._handle_pause(from_chat_id)
        elif data == "cb_trades":
            await self._handle_trades(from_chat_id)
        elif data == "cb_regime":
            await self._handle_regime(from_chat_id)

    async def _handle_pause(self, chat_id: str) -> None:
        """Apply global pause and confirm to the operator.

        Args:
            chat_id: Destination chat to reply to.
        """
        if self._state.is_paused:
            await self._reply(chat_id, "⏸ System is *already paused*\\.")
            return
        self._state.pause()
        await self._reply(
            chat_id,
            "⏸ *System PAUSED*\n"
            "The current transaction will complete, then execution suspends\\.\n"
            "Send /resume to continue or /stop for graceful shutdown\\.",
        )

    async def _handle_resume(self, chat_id: str) -> None:
        """Clear global pause and/or circuit breaker halt. Confirm to the operator.

        Args:
            chat_id: Destination chat to reply to.
        """
        if self._state.is_stopped:
            await self._reply(chat_id, "🛑 Stop has been requested; cannot resume\\.")
            return
        cleared_cb = (
            hasattr(self._state, "is_circuit_breaker_halted")
            and self._state.is_circuit_breaker_halted
        )
        was_paused = self._state.is_paused
        self._state.resume()
        if cleared_cb:
            await self._reply(
                chat_id,
                "▶️ *Circuit breaker CLEARED* — trading resumed\\. "
                "Send /resume again if also paused\\.",
            )
        elif was_paused:
            await self._reply(chat_id, "▶️ *System RESUMED* — execution continues\\.")
        else:
            await self._reply(chat_id, "▶️ System is *already running*\\.")

    async def _handle_stop(self, chat_id: str) -> None:
        """Request graceful stop and confirm to the operator.

        The orchestrator will finish the current transaction, then exit cleanly.

        Args:
            chat_id: Destination chat to reply to.
        """
        if self._state.is_stopped:
            await self._reply(chat_id, "🛑 Stop already requested\\. Waiting for drain\\.")
            return
        self._state.stop()
        await self._reply(
            chat_id,
            "🛑 *GRACEFUL STOP requested*\n"
            "Current transaction will complete, then the orchestrator exits\\.\n"
            "⚠️ This action cannot be undone in this session\\.",
        )

    async def _handle_help(self, chat_id: str) -> None:
        """Send the full command reference (Section 5.1)."""
        help_text = (
            "\U0001f916 *Available Commands*\n\n"
            "/status \\— Full portfolio dashboard\n"
            "/pnl \\— Last 24h / 7d / 30d P&L summary\n"
            "/trades \\— Last 10 trades \\(entry, exit, pnl, strategy, venue\\)\n"
            "/regime \\— Current HMM state \\+ next economic calendar event\n"
            "/paper \\— Paper engine vs real execution \\(last 6h\\)\n"
            "/risk \\— Real\\-time risk metrics\n"
            "/retrain \\— Trigger manual model retraining \\(requires /retrain confirm\\)\n"
            "/mode \\[m\\] \\— Change execution mode \\(shows valid list; requires confirmation\\)\n"
            "/setmode \\<mode\\> \\— Change execution mode without restart\n"
            "/wallets \\— List active wallets with balances and circuit breaker status\n"
            "/allocation \\— Show portfolio allocation vs targets\n"
            "/reset\\_circuit \\<wallet\\_id\\> confirm \\— Force reset a circuit breaker\n"
            "/report \\<daily\\|weekly\\|monthly\\> \\— P&L and performance report\n"
            "/gas \\— Current gas price and wallet suspension status\n"
            "/pause \\— Suspend execution after current TX\n"
            "/resume \\— Resume paused execution\n"
            "/stop \\— Graceful shutdown \\(finishes current TX\\)\n"
            "/help \\— This message\n\n"
            "_Rate limit: 10 commands/min per user\\. Heavy commands: 1/hour\\._"
        )
        await self._reply(chat_id, help_text)

    # ------------------------------------------------------------------
    # New command handlers (Section 5.1)
    # ------------------------------------------------------------------

    async def _handle_pnl(self, chat_id: str) -> None:
        try:
            if isinstance(self._status_provider, LiveStatusProvider):
                msg = await self._status_provider.build_pnl_message()
            else:
                msg = "P&L data not available. Configure a LiveStatusProvider."
            await self._reply(chat_id, _MessageFormatter._escape_md(msg))
        except Exception as exc:
            await self._reply(chat_id, _MessageFormatter._escape_md(f"\u26a0\ufe0f P&L unavailable: {exc}"))

    async def _handle_trades(self, chat_id: str) -> None:
        try:
            if isinstance(self._status_provider, LiveStatusProvider):
                msg = await self._status_provider.build_trades_message()
            else:
                msg = "Trade history not available. Configure a LiveStatusProvider."
            await self._reply(chat_id, _MessageFormatter._escape_md(msg))
        except Exception as exc:
            await self._reply(chat_id, _MessageFormatter._escape_md(f"\u26a0\ufe0f Trades unavailable: {exc}"))

    async def _handle_regime(self, chat_id: str) -> None:
        try:
            if isinstance(self._status_provider, LiveStatusProvider):
                msg = await self._status_provider.build_regime_message()
            else:
                msg = "Regime data not available. Configure a LiveStatusProvider."
            await self._reply(chat_id, _MessageFormatter._escape_md(msg))
        except Exception as exc:
            await self._reply(chat_id, _MessageFormatter._escape_md(f"\u26a0\ufe0f Regime unavailable: {exc}"))

    async def _handle_paper(self, chat_id: str) -> None:
        try:
            if isinstance(self._status_provider, LiveStatusProvider):
                msg = await self._status_provider.build_paper_message()
            else:
                msg = "Paper engine data not available. Configure a LiveStatusProvider."
            await self._reply(chat_id, _MessageFormatter._escape_md(msg))
        except Exception as exc:
            await self._reply(chat_id, _MessageFormatter._escape_md(f"\u26a0\ufe0f Paper data unavailable: {exc}"))

    async def _handle_risk(self, chat_id: str) -> None:
        """Section 5.3 — full risk metrics dashboard."""
        try:
            if isinstance(self._status_provider, LiveStatusProvider):
                msg = await self._status_provider.build_risk_message()
            else:
                msg = "Risk data not available. Configure a LiveStatusProvider."
            await self._reply(chat_id, _MessageFormatter._escape_md(msg))
        except Exception as exc:
            await self._reply(chat_id, _MessageFormatter._escape_md(f"\u26a0\ufe0f Risk data unavailable: {exc}"))

    async def _handle_retrain(self, chat_id: str, confirmed: bool) -> None:
        """Two-step confirmation for manual model retrain (Section 5.1)."""
        if confirmed:
            if self._pending_retrain:
                self._pending_retrain = False
                self._state.request_retrain()
                await self._reply(
                    chat_id,
                    "\U0001f501 *Retrain triggered*\\. Monitor logs for progress\\.",
                )
            else:
                await self._reply(
                    chat_id,
                    "\u26a0\ufe0f No pending retrain request\\. Send /retrain first\\.",
                )
        else:
            self._pending_retrain = True
            _model_version = "unknown"
            _trained_at = "unknown"
            try:
                import json as _json
                from pathlib import Path as _Path
                _report_path = (
                    _Path(__file__).resolve().parent.parent
                    / "ml_models"
                    / "training_report.json"
                )
                if _report_path.exists():
                    _rdata = _json.loads(_report_path.read_text(encoding="utf-8"))
                    _model_version = _rdata.get("version", "unknown")
                    _trained_at = _rdata.get("trained_at", "unknown")
            except Exception:
                pass
            await self._reply(
                chat_id,
                "\u26a0\ufe0f *Manual retrain requested*\n"
                f"Current model version: {_MessageFormatter._escape_md(_model_version)}\n"
                f"Last trained: {_MessageFormatter._escape_md(_trained_at)}\n"
                "This will halt trading during retraining\\.\n"
                "Send /retrain confirm to proceed, or any other command to cancel\\.",
            )

    async def _handle_mode(self, chat_id: str, arg: str) -> None:
        """Change execution mode with confirmation (Section 5.1)."""
        valid = ["legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live"]
        if not arg:
            current = getattr(self._status_provider._orch, "execution_mode", "unknown") \
                      if hasattr(self._status_provider, "_orch") else "unknown"
            mode_list = "\n".join(f"  \\• {m}" for m in valid)
            await self._reply(
                chat_id,
                f"Current mode: {_MessageFormatter._escape_md(current)}\n"
                f"Valid modes:\n{mode_list}\n"
                f"Usage: /mode \\<name\\>",
            )
            return
        if arg not in valid:
            await self._reply(
                chat_id,
                f"\u274c Unknown mode '{_MessageFormatter._escape_md(arg)}'\n"
                f"Valid: {_MessageFormatter._escape_md(', '.join(valid))}",
            )
            return
        if self._pending_mode == arg:
            # Second call = confirmation
            self._pending_mode = ""
            self._state.request_mode_change(arg)
            await self._reply(
                chat_id,
                f"\u2705 Mode change to *{_MessageFormatter._escape_md(arg)}* confirmed\\.",
            )
        else:
            self._pending_mode = arg
            await self._reply(
                chat_id,
                f"\u26a0\ufe0f Change execution mode to *{_MessageFormatter._escape_md(arg)}*?\n"
                f"Send /mode {_MessageFormatter._escape_md(arg)} again to confirm\\.",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _reply(self, chat_id: str, text: str) -> None:
        """Split-and-send reply with retry (Section 5.4).

        Messages > 4096 chars are split at logical section boundaries.
        Each part is sent with up to 3 attempts and 5-second back-off.

        Args:
            chat_id: Destination chat identifier.
            text: MarkdownV2-formatted reply text.
        """
        for part in _split_message(text):
            for attempt in range(1, 4):
                try:
                    await self._client.send_message(part, chat_id=chat_id)
                    break
                except Exception as exc:
                    if attempt == 3:
                        log.warning(
                            f"CommandCenter: failed to send reply after 3 attempts "
                            f"to {chat_id}: {exc}"
                        )
                    else:
                        await asyncio.sleep(5.0)
