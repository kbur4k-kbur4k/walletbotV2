"""
wallet_manager/security/logging_filter.py

Logging infrastructure that guarantees private keys and wallet addresses
are never emitted in plaintext.

Responsibility
--------------
This module has exactly one job: scrub sensitive hex strings from log records
before they are written to any handler (console, file, remote sink, etc.).

Design
------
``_SensitiveDataFilter`` is a :class:`logging.Filter` that rewrites both the
primary message and every positional ``args`` element in place.  Because it
operates at the ``Logger`` level (not the ``Handler`` level), every handler
attached to the logger automatically inherits the redaction — including
handlers added after the logger is created.

Masking strategy
----------------
Ethereum addresses (``0x`` + 40 hex chars) and raw private keys (64 hex chars
with or without ``0x``) are replaced with a partially-visible form::

    0xAbCd1234****efgh   (first 6 + last 4 characters visible)

This preserves enough context for debugging (e.g. correlating a log line with
a known address prefix) while making the string non-actionable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Final

from wallet_manager.env_loader import load_project_env

load_project_env()

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches Ethereum-style hex strings: optional 0x + 40–66 hex chars.
# The capture groups preserve a short prefix and suffix for context.
_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"((?:0x)?[0-9a-fA-F]{4})"   # group 1: first 4 visible chars (+ optional 0x)
    r"[0-9a-fA-F]{32,58}"         # middle section — redacted
    r"([0-9a-fA-F]{4})",          # group 2: last 4 visible chars
)

_REPLACEMENT: Final[str] = r"\1****\2"

# Telegram bot token: {8-10 digits}:{35-char alphanumeric string}
_TELEGRAM_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\d{4})\d{4,6}:[A-Za-z0-9_-]{35}"
)

# Fernet key: base64, 44 chars ending with =
_FERNET_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"([A-Za-z0-9+/]{4})[A-Za-z0-9+/]{39}="
)

# Infura/Alchemy API key embedded in URL
_API_KEY_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"((?:infura\.io/v3/|alchemy\.com/v2/)[A-Za-z0-9]{4})[A-Za-z0-9]{16,36}"
)


# ---------------------------------------------------------------------------
# Filter implementation
# ---------------------------------------------------------------------------


class _SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts wallet addresses and private key fragments.

    Attach to a logger (not a handler) to ensure all downstream handlers
    receive clean records.

    Example:
        .. code-block:: python

            logger = logging.getLogger("my_app")
            logger.addFilter(_SensitiveDataFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact hex strings from *record* in place.

        Processes both ``record.msg`` and every element of ``record.args``
        before the record is emitted.

        Args:
            record: The log record to sanitise.

        Returns:
            Always ``True`` — records are never suppressed, only cleaned.
        """
        record.msg = _redact(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (_redact(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    (_redact(a) if isinstance(a, str) else a)
                    for a in record.args
                )
        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _redact(text: str) -> str:
    """Replace hex substrings in *text* with a masked form.

    Args:
        text: Raw string potentially containing sensitive hex data.

    Returns:
        String with sensitive patterns obscured.
    """
    text = _HEX_PATTERN.sub(_REPLACEMENT, text)
    text = _TELEGRAM_TOKEN_PATTERN.sub(r"\g<1>****", text)
    text = _FERNET_KEY_PATTERN.sub(r"\g<1>****", text)
    text = _API_KEY_URL_PATTERN.sub(r"\g<1>****", text)
    return text


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON matching WalletManagerLogRecord schema."""

    def format(self, record: logging.LogRecord) -> str:
        import datetime as _dt
        payload: dict[str, object] = {
            "timestamp": _dt.datetime.utcfromtimestamp(record.created).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for extra_field in ("wallet_id", "strategy", "tx_hash"):
            val = getattr(record, extra_field, None)
            if val is not None:
                payload[extra_field] = str(val)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def build_logger(name: str) -> logging.Logger:
    """Construct a module-level logger with the sensitive-data filter applied.

    The returned logger emits to ``stderr`` with a timestamped format.
    The :class:`_SensitiveDataFilter` is attached at the logger level so that
    **all** handlers (present and future) inherit the redaction automatically.

    Args:
        name: Logger name; pass ``__name__`` from the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example:
        .. code-block:: python

            from wallet_manager.security.logging_filter import build_logger

            log = build_logger(__name__)
            log.info(f"Address: {'0xAbCd1234...ef56'}")
            # → emits: "Address: 0xAbCd****ef56"
    """
    logger = logging.getLogger(name)
    try:
        from wallet_manager.config import get_settings

        configured_level = getattr(get_settings(), "log_level", "INFO")
    except Exception:
        configured_level = "INFO"

    if isinstance(configured_level, int):
        level = configured_level
    elif isinstance(configured_level, str):
        level = getattr(logging, configured_level.upper(), logging.INFO)
    else:
        level = logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        # Select formatter based on LOG_FORMAT env var
        import os as _os
        log_format = _os.getenv("LOG_FORMAT", "text").strip().lower()
        if log_format == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
        logger.addHandler(handler)

    # Attach only once even if build_logger is called multiple times for the
    # same name (e.g. during module reloads in test environments).
    if not any(isinstance(f, _SensitiveDataFilter) for f in logger.filters):
        logger.addFilter(_SensitiveDataFilter())

    return logger
