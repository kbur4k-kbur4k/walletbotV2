"""tests/unit/test_vault_key_rotation.py — Vault key rotation tests."""
from __future__ import annotations

import base64
import json

import pytest
from cryptography.fernet import Fernet

from wallet_manager.config import Settings
from wallet_manager.exceptions import ConfigurationError, DecryptionError, EncryptionError
from wallet_manager.security.vault import Vault


def _vault(key: str | None = None) -> Vault:
    k = key or Fernet.generate_key().decode()
    return Vault(settings=Settings(
        vault_key=k,
        execution_mode="ml_quant",
        garch_alpha=0.10,
        garch_beta=0.85,
    ))


class TestKeyRotation:

    def test_generate_key_is_valid_fernet(self):
        key = Vault.generate_key()
        fernet = Fernet(key.encode())
        assert fernet is not None

    def test_generate_key_unique_each_call(self):
        k1 = Vault.generate_key()
        k2 = Vault.generate_key()
        assert k1 != k2

    @pytest.mark.asyncio
    async def test_rotate_key_absent_file_updates_fernet(self, tmp_path):
        """When wallets.json is absent rotate_key must still switch the active key."""
        vault = _vault()
        old_token = vault.encrypt("secret_data")
        new_key = Vault.generate_key()

        count = await vault.rotate_key(
            new_key,
            wallets_path=tmp_path / "no_wallets.json",
        )
        assert count == 0

        # New key is now active — old token must be unreadable
        with pytest.raises(DecryptionError):
            vault.decrypt(old_token)

    @pytest.mark.asyncio
    async def test_rotate_key_new_tokens_work(self, tmp_path):
        """After rotation new encrypt/decrypt round-trips must succeed."""
        vault = _vault()
        await vault.rotate_key(
            Vault.generate_key(),
            wallets_path=tmp_path / "no_wallets.json",
        )
        token = vault.encrypt("new_secret_after_rotation")
        assert vault.decrypt(token) == "new_secret_after_rotation"

    @pytest.mark.asyncio
    async def test_rotate_key_invalid_key_raises(self, tmp_path):
        vault = _vault()
        with pytest.raises((ValueError, ConfigurationError)):
            await vault.rotate_key(
                "not_a_valid_fernet_key",
                wallets_path=tmp_path / "no_wallets.json",
            )

    @pytest.mark.asyncio
    async def test_rotate_key_re_encrypts_wallet_file(self, tmp_path):
        """rotate_key must re-encrypt vault_encrypted entries in wallets.json."""
        vault = _vault()
        plaintext = "0xdeadbeefdeadbeef" * 4

        old_token = vault.encrypt(plaintext)
        wallets_file = tmp_path / "wallets.json"
        wallets_file.write_text(
            json.dumps([{
                "wallet_id": "w1",
                "vault_encrypted": True,
                "private_key": base64.b64encode(old_token).decode(),
            }]),
            encoding="utf-8",
        )

        new_key = Vault.generate_key()
        count = await vault.rotate_key(new_key, wallets_path=wallets_file)
        assert count == 1

        # Old token must now fail
        with pytest.raises(DecryptionError):
            vault.decrypt(old_token)

        # Re-read wallets.json and verify new token decrypts correctly
        data = json.loads(wallets_file.read_text())
        new_token = data[0]["private_key"].encode()
        assert vault.decrypt(new_token) == plaintext

    @pytest.mark.asyncio
    async def test_rotate_key_read_failure_keeps_old_key_active(self, tmp_path):
        vault = _vault()
        old_token = vault.encrypt("secret_data")

        wallets_file = tmp_path / "wallets.json"
        wallets_file.write_text("{not-valid-json", encoding="utf-8")

        with pytest.raises(EncryptionError):
            await vault.rotate_key(Vault.generate_key(), wallets_path=wallets_file)

        assert vault.decrypt(old_token) == "secret_data"
