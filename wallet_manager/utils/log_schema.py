"""
wallet_manager/utils/log_schema.py

Typed schema definition for structured JSON log records.
"""
from __future__ import annotations
from typing import TypedDict


class WalletManagerLogRecord(TypedDict, total=False):
    timestamp: str        # ISO 8601, e.g. "2024-01-01T12:00:00.000Z"
    level: str            # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    logger: str           # logger name, e.g. "wallet_manager.orchestrator"
    message: str          # log message
    wallet_id: str        # optional: wallet identifier
    strategy: str         # optional: e.g. "ALGO", "YIELD", "SHORT", "AIRDROP"
    tx_hash: str          # optional: transaction hash
