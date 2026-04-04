from __future__ import annotations

import json

from cryptography.fernet import Fernet

from scripts.setup_wizard import bootstrap_workspace, validate_workspace


def test_bootstrap_workspace_creates_and_syncs_files(tmp_path):
    (tmp_path / "wallet_manager").mkdir()
    (tmp_path / ".env.example").write_text("VAULT_KEY=GENERATE_WITH_FERNET\n", encoding="utf-8")
    (tmp_path / "wallets.json.example").write_text("[]\n", encoding="utf-8")

    actions = bootstrap_workspace(tmp_path)

    assert (tmp_path / ".env").exists()
    assert (tmp_path / "wallet_manager" / ".env").exists()
    assert (tmp_path / "wallets.json").exists()
    assert any("created .env" in action for action in actions)
    assert any("synced wallet_manager/.env" in action for action in actions)


def test_validate_workspace_accepts_complete_bootstrap(tmp_path):
    (tmp_path / "wallet_manager").mkdir()
    key = Fernet.generate_key().decode("utf-8")
    env_text = "\n".join(
        [
            f"VAULT_KEY={key}",
            "TELEGRAM_BOT_TOKEN=test_token",
            "TELEGRAM_CHAT_ID=123456",
            "EXECUTION_MODE=legacy",
            "BASE_RPC_1=https://mainnet.base.org",
            "GARCH_ALPHA=0.10",
            "GARCH_BETA=0.85",
        ]
    )
    (tmp_path / ".env").write_text(env_text, encoding="utf-8")
    (tmp_path / "wallet_manager" / ".env").write_text(env_text, encoding="utf-8")
    (tmp_path / "wallets.json").write_text(
        json.dumps(
            [
                {
                    "wallet_id": "wallet_01",
                    "address": "0x4200000000000000000000000000000000000006",
                    "private_key": "a" * 64,
                    "rpc_url": "https://mainnet.base.org",
                    "chain_id": 8453,
                    "enabled": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    errors, warnings = validate_workspace(tmp_path)

    assert errors == []
    assert warnings == []


def test_validate_workspace_rejects_mismatched_vault_key(tmp_path):
    (tmp_path / "wallet_manager").mkdir()
    good_key = Fernet.generate_key().decode("utf-8")
    wrong_key = Fernet.generate_key().decode("utf-8")
    token = Fernet(good_key.encode("utf-8")).encrypt(("e" * 64).encode("utf-8")).decode("utf-8")

    env_text = "\n".join(
        [
            f"VAULT_KEY={wrong_key}",
            "TELEGRAM_BOT_TOKEN=test_token",
            "TELEGRAM_CHAT_ID=123456",
            "EXECUTION_MODE=legacy",
            "BASE_RPC_1=https://mainnet.base.org",
            "GARCH_ALPHA=0.10",
            "GARCH_BETA=0.85",
        ]
    )
    (tmp_path / ".env").write_text(env_text, encoding="utf-8")
    (tmp_path / "wallet_manager" / ".env").write_text(env_text, encoding="utf-8")
    (tmp_path / "wallets.json").write_text(
        json.dumps(
            [
                {
                    "wallet_id": "wallet_01",
                    "address": "0x4200000000000000000000000000000000000006",
                    "private_key": token,
                    "vault_encrypted": True,
                    "rpc_url": "https://mainnet.base.org",
                    "chain_id": 8453,
                    "enabled": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    errors, _warnings = validate_workspace(tmp_path)

    assert any("could not be decrypted" in error for error in errors)
