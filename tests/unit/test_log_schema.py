"""Tests for structured JSON log schema (Task 6.1)."""
from __future__ import annotations

import json
import logging

import pytest


def test_json_formatter_required_fields():
    """JSON formatter produces records with all required fields."""
    from wallet_manager.security.logging_filter import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert "timestamp" in data
    assert "level" in data
    assert "logger" in data
    assert "message" in data
    assert data["level"] == "INFO"
    assert data["logger"] == "test.logger"
    assert data["message"] == "test message"
    # timestamp must be ISO 8601 with Z suffix
    assert data["timestamp"].endswith("Z")


def test_json_formatter_optional_fields():
    """JSON formatter includes optional fields when set on the record."""
    from wallet_manager.security.logging_filter import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="tx sent",
        args=(),
        exc_info=None,
    )
    record.wallet_id = "wallet_0"  # type: ignore[attr-defined]
    record.tx_hash = "0xdeadbeef"  # type: ignore[attr-defined]

    output = formatter.format(record)
    data = json.loads(output)
    assert data.get("wallet_id") == "wallet_0"
    assert data.get("tx_hash") == "0xdeadbeef"


def test_log_schema_typed_dict_fields():
    """WalletManagerLogRecord TypedDict has the expected keys."""
    import typing

    from wallet_manager.utils.log_schema import WalletManagerLogRecord

    hints = typing.get_type_hints(WalletManagerLogRecord)
    required_keys = {"timestamp", "level", "logger", "message"}
    optional_keys = {"wallet_id", "strategy", "tx_hash"}
    for k in required_keys | optional_keys:
        assert k in hints, f"Missing key: {k}"
