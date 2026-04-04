"""
wallet_manager/telegram_notifier.py

Real-time async Telegram notification system AND interactive command center.

This module upgrades the original one-way notifier into a bidirectional
system.  It contains two cooperating top-level classes:

1.  **TelegramNotifier** — outbound alert pipeline (unchanged from v1):
    fire-and-forget queue → token-bucket rate limiter → aiohttp POST.

2.  **CommandCenter** — inbound long-polling listener that calls
    ``getUpdates`` and dispatches operator commands to the orchestrator
    via shared :class:`CommandState` asyncio primitives.

Both classes share a single :class:`_TelegramCredentials` instance and a
single :class:`_TelegramClient` HTTP session, avoiding duplicate connections.

Command protocol
----------------
Recognised bot commands (only from the authorised ``TELEGRAM_CHAT_ID``)::

    /status  — reply with per-wallet balance + task summary
    /pause   — set the global pause event; orchestrator suspends after current TX
    /resume  — clear the pause event; orchestrator continues
    /stop    — set the stop event; orchestrator drains current TX then exits
    /help    — command reference

Control flow integration
------------------------
The orchestrator holds a :class:`CommandState` reference and checks it
before processing each wallet::

    state = CommandState()
    ...
    for wallet_id in plan:
        if state.is_stopped:
            break
        await state.wait_for_resume()   # blocks while paused
        await run_wallet(wallet_id)

Security
--------
* Only messages from the configured ``TELEGRAM_CHAT_ID`` are processed;
  all others are silently ignored to prevent command injection.
* Bot token is never logged, repr'd, or included in exception messages.
* ``update_id`` offset is tracked to prevent double-processing.

Rate limiting
-------------
Token-bucket at 1 msg/s default (``TELEGRAM_RATE_LIMIT`` env var), burst
configurable via ``TELEGRAM_RATE_BURST``.
"""

# ruff: noqa: RUF001, ASYNC109

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Final

import aiohttp

from wallet_manager.config import Settings
from wallet_manager.core.interfaces import INotifier
from wallet_manager.core.models import EventKind, NotificationEvent
from wallet_manager.env_loader import load_project_env
from wallet_manager.exceptions import (
    NotifierConfigurationError,
    NotifierDeliveryError,
)
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.security.telegram_commands import (
    CommandCenter,
    CommandState,
    ConfirmationGate,
    LiveStatusProvider,
    StatusProvider,
    _mask_address,
    _MessageFormatter,
)

load_project_env()

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TELEGRAM_API_BASE: Final[str] = "https://api.telegram.org"
_SEND_PATH: Final[str] = "/bot{token}/sendMessage"
_GET_UPDATES_PATH: Final[str] = "/bot{token}/getUpdates"

def _cfg() -> Settings:
    from wallet_manager.config import get_settings
    return get_settings()


_DEFAULT_RATE_LIMIT: Final[float] = 1.0  # overridden at runtime via _cfg()
_DEFAULT_BURST: Final[int] = 1           # overridden at runtime via _cfg()

# Telegram recommends long-poll timeout ≤ 60 s
_POLL_TIMEOUT_SECONDS: Final[int] = 30

_API_MAX_RETRIES: Final[int] = 3
_API_RETRY_BASE_DELAY: Final[float] = 2.0
_API_TIMEOUT: Final[float] = 10.0

_STOP_SENTINEL: Final[object] = object()

_ANSWER_CB_PATH: Final[str] = "/bot{token}/answerCallbackQuery"
_MAX_MSG_LEN: Final[int] = 4096


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TelegramCredentials:
    """Validated Telegram credentials. ``__repr__`` never exposes the token."""

    _bot_token: str = field(repr=False)
    _chat_id: str = field(repr=False)

    def __repr__(self) -> str:
        return "_TelegramCredentials(***)"

    @property
    def bot_token(self) -> str:
        """Bot API token (read-only)."""
        return self._bot_token

    @property
    def chat_id(self) -> str:
        """Authorised chat / channel ID (read-only)."""
        return self._chat_id

    @classmethod
    def from_env(cls) -> _TelegramCredentials:
        """Load and validate credentials from environment variables.

        Returns:
            Validated credentials instance.

        Raises:
            NotifierConfigurationError: If any variable is absent.
        """
        cfg = _cfg()
        token = cfg.telegram_bot_token
        chat_id = cfg.telegram_chat_id
        missing = [
            name for name, val in (
                ("TELEGRAM_BOT_TOKEN", token),
                ("TELEGRAM_CHAT_ID", chat_id),
            )
            if not val
        ]
        if missing:
            raise NotifierConfigurationError(
                f"Missing required env variable(s): {', '.join(missing)}. "
                "See .env.example."
            )
        return cls(_bot_token=token, _chat_id=chat_id)


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Asyncio token-bucket rate limiter for outbound message pacing."""

    def __init__(self, rate: float, capacity: int) -> None:
        if rate <= 0:
            raise ValueError(f"Rate must be positive; got {rate}.")
        self._rate = rate
        self._capacity = float(capacity)
        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available and consume one."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(
            self._capacity,
            self._tokens + (now - self._last_refill) * self._rate,
        )
        self._last_refill = now


# ---------------------------------------------------------------------------
# Low-level Telegram HTTP client (shared by notifier + command center)
# ---------------------------------------------------------------------------


class _TelegramClient:
    """Minimal aiohttp wrapper for the Telegram Bot API.

    Shared between :class:`TelegramNotifier` (outbound) and
    :class:`CommandCenter` (inbound) to avoid duplicate TCP connections.

    Args:
        credentials: Validated credentials object.
        timeout: Per-request total timeout in seconds.
    """

    def __init__(
        self,
        credentials: _TelegramCredentials,
        timeout: float = _API_TIMEOUT,
    ) -> None:
        self._creds = credentials
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        """Create the underlying aiohttp session."""
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_message(
        self,
        text: str,
        chat_id: str | None = None,
    ) -> None:
        """POST a MarkdownV2 message to *chat_id* (or the default chat).

        Args:
            text: MarkdownV2-formatted message (≤ 4096 chars).
            chat_id: Override destination (defaults to credentials chat_id).

        Raises:
            NotifierDeliveryError: After all retries are exhausted.
            RuntimeError: If called before :meth:`open`.
        """
        if self._session is None or self._session.closed:
            raise RuntimeError("_TelegramClient not open. Call open() first.")

        if len(text) > 4096:
            text = text[:4090] + "\n…"

        url = _TELEGRAM_API_BASE + _SEND_PATH.format(token=self._creds.bot_token)
        payload = {
            "chat_id": chat_id or self._creds.chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }

        last_exc: Exception | None = None
        for attempt in range(1, _API_MAX_RETRIES + 1):
            try:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return
                    if resp.status == 429:
                        body = await resp.json(content_type=None)
                        wait = float(
                            body.get("parameters", {}).get(
                                "retry_after", _API_RETRY_BASE_DELAY
                            )
                        )
                        log.warning(f"Telegram 429 — waiting {float(wait):.1f}s.")
                        await asyncio.sleep(wait)
                        continue
                    if 400 <= resp.status < 500:
                        raise NotifierDeliveryError(
                            f"Non-retryable HTTP {resp.status} from Telegram API.",
                            http_status=resp.status,
                        )
                    last_exc = NotifierDeliveryError(
                        f"Server error {resp.status}.",
                        http_status=resp.status,
                    )
            except (TimeoutError, aiohttp.ClientError) as exc:
                last_exc = exc
                log.warning(
                    f"Telegram network error (attempt {attempt}/{_API_MAX_RETRIES}): {type(exc).__name__}",
                )
                if isinstance(exc, aiohttp.ServerDisconnectedError):
                    await self.close()
                    await self.open()
            if attempt < _API_MAX_RETRIES:
                await asyncio.sleep(_API_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        raise NotifierDeliveryError(
            f"Message delivery failed after {_API_MAX_RETRIES} attempts."
        ) from last_exc

    async def get_updates(
        self,
        offset: int,
        timeout: int = _POLL_TIMEOUT_SECONDS,
    ) -> list[dict]:
        """Long-poll ``getUpdates`` and return new Update objects.

        Args:
            offset: Earliest update_id to request (exclusive of already-seen).
            timeout: Long-polling wait in seconds (Telegram server holds the
                connection open until a message arrives or *timeout* elapses).

        Returns:
            List of Telegram Update dicts; empty on error or no updates.
        """
        if self._session is None or self._session.closed:
            return []

        url = _TELEGRAM_API_BASE + _GET_UPDATES_PATH.format(
            token=self._creds.bot_token
        )
        params: dict[str, str] = {
            "offset": str(offset),
            "timeout": str(timeout),
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        try:
            # Slightly longer aiohttp timeout than the Telegram poll timeout
            poll_timeout = aiohttp.ClientTimeout(total=float(timeout) + 5.0)
            async with self._session.get(
                url, params=params, timeout=poll_timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", [])
                    return result if isinstance(result, list) else []
        except Exception as exc:
            log.debug(f"getUpdates non-fatal error: {exc}")
        return []

    async def send_message_with_keyboard(
        self,
        text: str,
        reply_markup: dict,
        chat_id: str | None = None,
    ) -> None:
        """POST a MarkdownV2 message with an inline keyboard.

        Args:
            text: MarkdownV2-formatted message (≤ 4096 chars).
            reply_markup: Telegram InlineKeyboardMarkup dict.
            chat_id: Override destination (defaults to credentials chat_id).
        """
        if self._session is None or self._session.closed:
            raise RuntimeError("_TelegramClient not open.")

        if len(text) > _MAX_MSG_LEN:
            text = text[:_MAX_MSG_LEN - 6] + "\n\u2026"

        url = _TELEGRAM_API_BASE + _SEND_PATH.format(token=self._creds.bot_token)
        payload = {
            "chat_id": chat_id or self._creds.chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
            "reply_markup": reply_markup,
        }
        last_exc: Exception | None = None
        for attempt in range(1, _API_MAX_RETRIES + 1):
            try:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return
                    if resp.status == 429:
                        body = await resp.json(content_type=None)
                        wait = float(body.get("parameters", {}).get("retry_after", 5.0))
                        await asyncio.sleep(wait)
                        continue
                    if 400 <= resp.status < 500:
                        raise NotifierDeliveryError(
                            f"Non-retryable HTTP {resp.status}.", http_status=resp.status
                        )
                    last_exc = NotifierDeliveryError(f"Server error {resp.status}.", http_status=resp.status)
            except (TimeoutError, aiohttp.ClientError) as exc:
                last_exc = exc
                if isinstance(exc, aiohttp.ServerDisconnectedError):
                    await self.close()
                    await self.open()
            if attempt < _API_MAX_RETRIES:
                await asyncio.sleep(_API_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        raise NotifierDeliveryError(f"Keyboard message delivery failed after {_API_MAX_RETRIES} attempts.") from last_exc

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        """Acknowledge a callback query so Telegram removes the loading spinner.

        Args:
            callback_query_id: The ``id`` field from the callback_query update.
            text: Optional short notification to show the user.
        """
        if self._session is None or self._session.closed:
            return
        url = _TELEGRAM_API_BASE + _ANSWER_CB_PATH.format(token=self._creds.bot_token)
        try:
            async with self._session.post(
                url, json={"callback_query_id": callback_query_id, "text": text}
            ) as resp:
                if resp.status != 200:
                    log.debug(f"answerCallbackQuery returned {resp.status}")
        except Exception as exc:
            log.debug(f"answerCallbackQuery error: {exc}")


# ---------------------------------------------------------------------------
# TelegramNotifier — public INotifier (outbound)
# ---------------------------------------------------------------------------


class TelegramNotifier(INotifier):
    """Async outbound Telegram notifier implementing :class:`~wallet_manager.core.interfaces.INotifier`.

    Pair with :class:`CommandCenter` (via :meth:`build_command_center`) for
    bidirectional operation.

    Args:
        rate_limit: Token-bucket refill rate in messages/second.
        burst: Token-bucket burst capacity.
        queue_max_size: Internal queue capacity (0 = unbounded).
        credentials: Optional pre-constructed credentials (for testing).

    Example:
        .. code-block:: python

            state = CommandState()
            async with TelegramNotifier() as notifier:
                center = notifier.build_command_center(state)
                await center.start()

                await notifier.send(NotificationEvent(kind=EventKind.SYSTEM_BOOT))
                # ... orchestrator runs, checking state.is_paused / is_stopped ...
                await notifier.send(NotificationEvent(kind=EventKind.SYSTEM_SHUTDOWN))

                await center.stop()
    """

    def __init__(
        self,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        burst: int = _DEFAULT_BURST,
        queue_max_size: int = 100,
        credentials: _TelegramCredentials | None = None,
    ) -> None:
        self._credentials = credentials or _TelegramCredentials.from_env()
        self._rate_limiter = _TokenBucket(rate=rate_limit, capacity=burst)
        self._client = _TelegramClient(self._credentials)
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=queue_max_size)
        self._worker_task: asyncio.Task[None] | None = None
        log.debug(
            f"TelegramNotifier initialised (rate={rate_limit:.1f} msg/s, burst={burst}).",
        )

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> TelegramNotifier:
        """Open the HTTP session and start the delivery worker."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Drain the queue, stop the worker, and close the session."""
        await self.stop()

    # ------------------------------------------------------------------
    # INotifier implementation
    # ------------------------------------------------------------------

    async def send(self, event: NotificationEvent) -> None:
        """Enqueue *event* for non-blocking async delivery.

        Args:
            event: Notification event to deliver.

        Raises:
            RuntimeError: If called before :meth:`start`.
        """
        if self._worker_task is None or self._worker_task.done():
            raise RuntimeError(
                "TelegramNotifier not started. Use 'async with' or call start() first."
            )
        await self._queue.put(event)

    async def send_event(self, event: NotificationEvent) -> None:
        """Backward-compatible alias for older notifier callers."""
        await self.send(event)

    async def send_confirmation_request(
        self, wallet_id: str, amount_str: str, direction: str
    ) -> None:
        """Send transaction confirmation request and wait for rate limiter.

        Call this before a swap; then await confirmation_gate.wait_for_evet().

        Args:
            wallet_id: Wallet identifier.
            amount_str: Human-readable amount (e.g. "0.01 ETH").
            direction: e.g. "ETH -> USDC".
        """
        if self._worker_task is None or self._worker_task.done():
            raise RuntimeError("TelegramNotifier not started.")
        raw = (
            f"⚠️ İŞLEM ONAYI: {_mask_address(wallet_id)} için {amount_str} "
            f"{direction} takası yapılsın mı? (Onay için 'EVET' yazın)"
        )
        escaped = _MessageFormatter._escape_md(raw)
        await self._rate_limiter.acquire()
        await self._client.send_message(escaped)

    async def start(self) -> None:
        """Open the HTTP session and launch the background delivery worker."""
        await self._client.open()
        self._worker_task = asyncio.ensure_future(self._delivery_worker())
        log.debug("TelegramNotifier delivery worker started.")

    async def stop(self) -> None:
        """Drain the queue, stop the worker, and close the HTTP session.

        Waits up to 30 seconds for the queue to drain before force-cancelling.
        """
        log.debug(
            f"TelegramNotifier stopping — draining {self._queue.qsize()} item(s).",
        )
        await self._queue.put(_STOP_SENTINEL)
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(self._worker_task, timeout=30.0)
            except TimeoutError:
                log.warning("Drain timed out; cancelling worker.")
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task
        await self._client.close()
        log.debug("TelegramNotifier stopped.")

    # ------------------------------------------------------------------
    # Factory method for the command center
    # ------------------------------------------------------------------

    def build_command_center(
        self,
        command_state: CommandState,
        status_provider: StatusProvider | None = None,
        live_status_provider: LiveStatusProvider | None = None,
        confirmation_gate: ConfirmationGate | None = None,
    ) -> CommandCenter:
        """Construct a :class:`CommandCenter` that shares this notifier's HTTP session.

        Using a shared :class:`_TelegramClient` avoids duplicate TCP connections
        and ensures the bot token is held in only one place.

        Args:
            command_state: Shared :class:`CommandState` for the orchestrator.
            status_provider: Optional custom status reporter.
            confirmation_gate: Optional :class:`ConfirmationGate` for EVET confirmation.

        Returns:
            Configured :class:`CommandCenter` ready for :meth:`~CommandCenter.start`.
        """
        return CommandCenter(
            credentials=self._credentials,
            command_state=command_state,
            client=self._client,
            status_provider=live_status_provider or status_provider,
            confirmation_gate=confirmation_gate,
        )

    # ------------------------------------------------------------------
    # Background delivery worker
    # ------------------------------------------------------------------

    async def _delivery_worker(self) -> None:
        """Consume events from the queue and deliver them to Telegram.

        Applies rate limiting before each delivery.  Delivery errors are
        logged and swallowed — a Telegram outage must not stop wallet execution.
        Exits cleanly when the ``_STOP_SENTINEL`` is dequeued.
        """
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP_SENTINEL:
                    return

                event: NotificationEvent = item  # type: ignore[assignment]
                await self._rate_limiter.acquire()

                try:
                    text = _MessageFormatter.format(event)
                    await self._client.send_message(text)
                    log.debug(f"Notification delivered: {event.kind.name}.")
                except NotifierDeliveryError as exc:
                    log.error(
                        f"Delivery failed for {event.kind.name} (HTTP {exc.http_status if exc.http_status is not None else '?'}): {exc.message}",
                    )
                except Exception as exc:
                    log.error(
                        f"Unexpected delivery error for {event.kind.name}: {exc}",
                    )
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# ProactiveAlerter
# ---------------------------------------------------------------------------


class ProactiveAlerter:
    """Checks alert conditions every cfg.proactive_alert_interval_seconds
    (default 300). Deduplication: same alert type is sent at most once
    within cfg.alert_dedup_window_seconds (default 3600).
    """

    def __init__(self) -> None:
        self._last_sent: dict[str, float] = {}

    def _is_deduped(self, key: str, dedup_window: float) -> bool:
        import time
        last = self._last_sent.get(key, 0.0)
        return time.time() - last < dedup_window

    def _mark_sent(self, key: str) -> None:
        import time
        self._last_sent[key] = time.time()

    @staticmethod
    async def _dispatch_alert(notifier: INotifier | Any, event: NotificationEvent) -> None:
        """Send alerts through send_event when available, otherwise fall back to send."""
        sender = getattr(notifier, "send_event", None)
        if sender is None:
            sender = getattr(notifier, "send", None)
        if sender is None:
            raise AttributeError("Notifier has neither send_event nor send.")
        result = sender(event)
        if hasattr(result, "__await__"):
            await result

    async def check_and_alert(
        self,
        notifier: INotifier,
        drift_level: str,
        apy: float,
        model_staleness_days: float,
        rpc_latency_ms: float,
        drawdown: float,
    ) -> None:
        """Check alert conditions and send Telegram notifications for triggered ones."""
        from wallet_manager.config import get_settings

        cfg = get_settings()
        dedup = cfg.alert_dedup_window_seconds

        # 1. ConceptDrift is MODERATE or SEVERE
        if drift_level in ("MODERATE", "SEVERE"):
            key = f"drift_{drift_level}"
            if not self._is_deduped(key, dedup):
                await self._dispatch_alert(notifier, NotificationEvent(
                    kind=EventKind.SYSTEM_ALERT,
                    detail=f"ConceptDrift: {drift_level} — model may need retraining.",
                ))
                self._mark_sent(key)

        # 2. Aave APY below threshold
        if apy * 100 < cfg.yield_min_apy_pct:
            key = "apy_below_threshold"
            if not self._is_deduped(key, dedup):
                await self._dispatch_alert(notifier, NotificationEvent(
                    kind=EventKind.SYSTEM_ALERT,
                    detail=f"Aave APY {apy * 100:.2f}% < threshold {cfg.yield_min_apy_pct:.1f}%",
                ))
                self._mark_sent(key)

        # 3. ML model stale (>7 days)
        if model_staleness_days > 7:
            key = "model_stale"
            if not self._is_deduped(key, dedup):
                await self._dispatch_alert(notifier, NotificationEvent(
                    kind=EventKind.SYSTEM_ALERT,
                    detail=f"ML model stale: {model_staleness_days:.1f} days since last retrain.",
                ))
                self._mark_sent(key)

        # 4. RPC endpoint high latency (>500ms)
        if rpc_latency_ms > 500:
            key = "rpc_high_latency"
            if not self._is_deduped(key, dedup):
                await self._dispatch_alert(notifier, NotificationEvent(
                    kind=EventKind.SYSTEM_ALERT,
                    detail=f"RPC latency {rpc_latency_ms:.0f}ms > 500ms threshold.",
                ))
                self._mark_sent(key)

        # 5. Portfolio drawdown exceeds threshold
        if drawdown > cfg.drawdown_reduce_threshold:
            key = "drawdown_exceeded"
            if not self._is_deduped(key, dedup):
                await self._dispatch_alert(notifier, NotificationEvent(
                    kind=EventKind.SYSTEM_ALERT,
                    detail=f"Portfolio drawdown {drawdown * 100:.1f}% > threshold {cfg.drawdown_reduce_threshold * 100:.1f}%",
                ))
                self._mark_sent(key)
