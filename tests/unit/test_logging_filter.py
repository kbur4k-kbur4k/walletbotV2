"""
tests/unit/test_logging_filter.py

Unit tests for wallet_manager/security/logging_filter.py

Covers:
- _redact() hex redaction (bare and 0x-prefixed)
- _redact() Fernet key redaction
- _redact() passthrough for non-sensitive messages
- build_logger() returns a logging.Logger
- _SensitiveDataFilter is attached to the logger returned by build_logger()
"""

from __future__ import annotations

import logging

import pytest

from wallet_manager.security.logging_filter import (
    _redact,
    _SensitiveDataFilter,
    build_logger,
)

# ---------------------------------------------------------------------------
# _redact() — hex strings
# ---------------------------------------------------------------------------


class TestRedactHex:
    """Tests for raw hex-string redaction."""

    def test_private_key_hex_redacted(self):
        """A 64-char hex string (bare private key) must be partially masked."""
        raw_key = "a" * 64  # 64 lowercase hex chars — typical private key
        result = _redact(raw_key)

        # Must not contain the 32-58 middle hex chars verbatim
        assert "****" in result, "Middle section should be replaced with ****"
        # Prefix and suffix must still be visible
        assert result.startswith("aaaa"), "First 4 chars should remain visible"
        assert result.endswith("aaaa"), "Last 4 chars should remain visible"
        # Full original key must not survive intact
        assert result != raw_key

    def test_0x_prefixed_private_key_redacted(self):
        """A 0x-prefixed 64-char hex string must also be redacted."""
        raw_key = "0x" + "b" * 64
        result = _redact(raw_key)

        assert "****" in result
        # The 0x plus the first 4 hex chars form group 1
        assert "bbbb" in result, "Last 4 visible chars should appear after ****"
        assert result != raw_key

    def test_hex_in_longer_message_redacted(self):
        """Hex key embedded in a longer log message is still redacted."""
        key = "deadbeef" + "c" * 48 + "cafe"  # 4 + 48 + 4 = 56 chars — within range
        msg = f"Processing transaction for key {key} on chain"
        result = _redact(msg)

        assert "****" in result
        assert key not in result


# ---------------------------------------------------------------------------
# _redact() — Fernet keys
# ---------------------------------------------------------------------------


class TestRedactFernet:
    """Tests for Fernet key redaction."""

    def test_fernet_token_redacted(self):
        """A 44-char base64 Fernet key (ending with =) must be redacted."""
        # Fernet key format: 43 base64 chars + trailing '='
        fernet_key = "AAAA" + "B" * 39 + "="
        result = _redact(fernet_key)

        assert "****" in result, "Fernet key body should be replaced with ****"
        assert fernet_key not in result, "Original Fernet key must not appear"
        # First 4 chars of the base64 key remain visible per pattern
        assert result.startswith("AAAA")

    def test_fernet_key_in_context_redacted(self):
        """Fernet key inside a longer string is still caught."""
        fernet_key = "ZZZz" + "a" * 39 + "="
        msg = f"vault_key={fernet_key}; reloading"
        result = _redact(msg)

        assert fernet_key not in result
        assert "****" in result


# ---------------------------------------------------------------------------
# _redact() — passthrough
# ---------------------------------------------------------------------------


class TestRedactPassthrough:
    def test_non_sensitive_message_unchanged(self):
        """A plain operational log message must not be altered."""
        msg = "cycle completed for wallet_01"
        assert _redact(msg) == msg

    def test_short_hex_not_redacted(self):
        """Hex strings shorter than 40 chars (below pattern minimum) pass through."""
        short_hex = "deadbeef1234"  # only 12 chars
        assert _redact(short_hex) == short_hex

    def test_empty_string_unchanged(self):
        assert _redact("") == ""

    def test_numeric_string_unchanged(self):
        assert _redact("123456789") == "123456789"


# ---------------------------------------------------------------------------
# build_logger()
# ---------------------------------------------------------------------------


class TestBuildLogger:
    def test_build_logger_returns_logger(self):
        """build_logger() must return a standard logging.Logger instance."""
        logger = build_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_build_logger_uses_given_name(self):
        """The returned logger's name must match the argument."""
        logger = build_logger("wallet_manager.test_unique_name")
        assert logger.name == "wallet_manager.test_unique_name"

    def test_filter_attached_to_logger(self):
        """The returned logger must have _SensitiveDataFilter in logger.filters."""
        logger = build_logger("test_filter_check")
        filter_types = [type(f) for f in logger.filters]
        assert _SensitiveDataFilter in filter_types, (
            "_SensitiveDataFilter must be present in logger.filters"
        )

    def test_build_logger_idempotent_filter(self):
        """Calling build_logger twice for the same name must not duplicate filters."""
        name = "test_idempotent_filter"
        build_logger(name)
        build_logger(name)
        logger = logging.getLogger(name)
        sensitive_filters = [
            f for f in logger.filters if isinstance(f, _SensitiveDataFilter)
        ]
        assert len(sensitive_filters) == 1, (
            "Exactly one _SensitiveDataFilter should be present, not duplicated"
        )


# ---------------------------------------------------------------------------
# Integration: filter applied at record level
# ---------------------------------------------------------------------------


class TestFilterIntegration:
    def test_filter_redacts_record_msg_in_place(self):
        """_SensitiveDataFilter.filter() must rewrite record.msg."""
        sensitive_hex = "f" * 64
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=sensitive_hex,
            args=(),
            exc_info=None,
        )
        f = _SensitiveDataFilter()
        result = f.filter(record)

        assert result is True, "filter() must return True (never suppress records)"
        assert "****" in record.msg
        assert record.msg != sensitive_hex

    def test_filter_redacts_args_tuple(self):
        """_SensitiveDataFilter.filter() must also redact positional args."""
        sensitive_hex = "1234" + "a" * 56 + "5678"  # 64 hex chars
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="key=%s",
            args=(sensitive_hex,),
            exc_info=None,
        )
        f = _SensitiveDataFilter()
        f.filter(record)

        assert isinstance(record.args, tuple)
        assert "****" in record.args[0]

    def test_filter_allows_non_sensitive_record_unchanged(self):
        """Non-sensitive log records pass through unmodified."""
        msg = "cycle completed for wallet_01"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        f = _SensitiveDataFilter()
        f.filter(record)

        assert record.msg == msg
