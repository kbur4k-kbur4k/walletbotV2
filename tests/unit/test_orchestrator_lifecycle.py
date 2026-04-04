from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from wallet_manager.config import Settings
from wallet_manager.exceptions import ConfigurationError
from wallet_manager.orchestrator_lifecycle import LifecycleManager


def _wallet_file(tmp_path, count: int = 1, wallets: list | None = None):
    if wallets is None:
        wallets = [
            {"wallet_id": f"wallet_{idx}", "address": f"0x{idx:040x}"}
            for idx in range(count)
        ]
    path = tmp_path / "wallets.json"
    path.write_text(json.dumps(wallets), encoding="utf-8")
    return path


def _paper_engine_with_trades(trade_count: int) -> MagicMock:
    engine = MagicMock()
    engine._ledger.get_pnl_summary = AsyncMock(return_value={"trade_count": trade_count})
    return engine


@pytest.mark.asyncio
async def test_validate_startup_blocks_live_mode_without_enable(tmp_path):
    cfg = Settings(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode="robust_live",
        wallet_config_path=_wallet_file(tmp_path, count=1),
        live_trading_enabled=False,
    )

    with (
        patch("wallet_manager.utils.health_check.run_startup_health_checks", new=AsyncMock(return_value=[])),
        patch(
            "wallet_manager.strategies.paper_engine.get_paper_engine",
            new=AsyncMock(return_value=_paper_engine_with_trades(100)),
        ),
        pytest.raises(ConfigurationError, match="LIVE_TRADING_ENABLED"),
    ):
        await LifecycleManager.validate_startup_config(cfg)


@pytest.mark.asyncio
async def test_validate_startup_blocks_live_mode_with_too_many_wallets(tmp_path):
    cfg = Settings(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode="robust_live",
        wallet_config_path=_wallet_file(tmp_path, count=2),
        live_trading_enabled=True,
        live_max_wallets=1,
    )

    with (
        patch("wallet_manager.utils.health_check.run_startup_health_checks", new=AsyncMock(return_value=[])),
        patch(
            "wallet_manager.strategies.paper_engine.get_paper_engine",
            new=AsyncMock(return_value=_paper_engine_with_trades(100)),
        ),
        pytest.raises(ConfigurationError, match="at most 1 wallet"),
    ):
        await LifecycleManager.validate_startup_config(cfg)


@pytest.mark.asyncio
async def test_validate_startup_rejects_undecryptable_encrypted_wallet(tmp_path):
    wrong_key = Fernet.generate_key().decode("utf-8")
    token_key = Fernet.generate_key().decode("utf-8")
    token = Fernet(token_key.encode("utf-8")).encrypt(("d" * 64).encode("utf-8")).decode("utf-8")

    wallets_path = tmp_path / "wallets.json"
    wallets_path.write_text(
        __import__("json").dumps(
            [
                {
                    "wallet_id": "wallet_01",
                    "address": "0x4200000000000000000000000000000000000006",
                    "private_key": token,
                    "vault_encrypted": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    cfg = Settings(
        vault_key=wrong_key,
        execution_mode="robust_paper",
        wallet_config_path=wallets_path,
    )

    with patch(
        "wallet_manager.utils.health_check.run_startup_health_checks",
        new=AsyncMock(return_value=[]),
    ), pytest.raises(ConfigurationError, match="could not be decrypted"):
        await LifecycleManager.validate_startup_config(cfg)
