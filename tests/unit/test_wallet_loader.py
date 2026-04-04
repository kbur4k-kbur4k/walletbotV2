from __future__ import annotations

import json

import pytest
from cryptography.fernet import Fernet


def test_load_wallet_configs_applies_base_defaults(tmp_path, monkeypatch):
    import wallet_manager.config as cfg_module
    from wallet_manager.orchestrator import _load_wallet_configs

    wallets_path = tmp_path / "wallets.json"
    wallets_path.write_text(
        json.dumps(
            [
                {
                    "wallet_id": "wallet_01",
                    "address": "0x4200000000000000000000000000000000000006",
                    "private_key": "a" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WALLET_CONFIG_PATH", str(wallets_path))
    monkeypatch.setenv("BASE_RPC_1", "https://mainnet.base.org")
    monkeypatch.setenv("EXECUTION_MODE", "legacy")
    monkeypatch.setenv("GARCH_ALPHA", "0.10")
    monkeypatch.setenv("GARCH_BETA", "0.85")
    cfg_module._settings_cache = None

    configs = _load_wallet_configs()

    assert configs[0].rpc_url == "https://mainnet.base.org"
    assert configs[0].chain_id == 8453


def test_load_wallet_configs_decrypts_encrypted_private_keys(tmp_path, monkeypatch):
    import wallet_manager.config as cfg_module
    from wallet_manager.orchestrator import _load_wallet_configs

    plaintext_key = "b" * 64
    vault_key = Fernet.generate_key().decode("utf-8")
    token = Fernet(vault_key.encode("utf-8")).encrypt(plaintext_key.encode("utf-8")).decode("utf-8")

    wallets_path = tmp_path / "wallets.json"
    wallets_path.write_text(
        json.dumps(
            [
                {
                    "wallet_id": "wallet_01",
                    "address": "0x4200000000000000000000000000000000000006",
                    "encrypted_private_key": token,
                    "rpc_url": "https://mainnet.base.org",
                    "chain_id": 8453,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VAULT_KEY", vault_key)
    monkeypatch.setenv("WALLET_CONFIG_PATH", str(wallets_path))
    monkeypatch.setenv("EXECUTION_MODE", "legacy")
    monkeypatch.setenv("GARCH_ALPHA", "0.10")
    monkeypatch.setenv("GARCH_BETA", "0.85")
    cfg_module._settings_cache = None

    configs = _load_wallet_configs()

    assert configs[0].private_key == plaintext_key


def test_load_wallet_configs_accepts_double_base64_tokens_without_warning(
    tmp_path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
):
    import base64

    import wallet_manager.config as cfg_module
    from wallet_manager.orchestrator import _load_wallet_configs

    plaintext_key = "c" * 64
    vault_key = Fernet.generate_key().decode("utf-8")
    raw_token = Fernet(vault_key.encode("utf-8")).encrypt(plaintext_key.encode("utf-8"))
    double_b64_token = base64.b64encode(raw_token).decode("utf-8")

    wallets_path = tmp_path / "wallets.json"
    wallets_path.write_text(
        json.dumps(
            [
                {
                    "wallet_id": "wallet_01",
                    "address": "0x4200000000000000000000000000000000000006",
                    "private_key": double_b64_token,
                    "vault_encrypted": True,
                    "rpc_url": "https://mainnet.base.org",
                    "chain_id": 8453,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VAULT_KEY", vault_key)
    monkeypatch.setenv("WALLET_CONFIG_PATH", str(wallets_path))
    monkeypatch.setenv("EXECUTION_MODE", "legacy")
    monkeypatch.setenv("GARCH_ALPHA", "0.10")
    monkeypatch.setenv("GARCH_BETA", "0.85")
    cfg_module._settings_cache = None

    with caplog.at_level("WARNING"):
        configs = _load_wallet_configs()

    assert configs[0].private_key == plaintext_key
    assert "decryption failed" not in caplog.text.lower()
