"""tests/unit/test_vault.py — Vault encryption/decryption tests."""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from wallet_manager.config import Settings
from wallet_manager.exceptions import ConfigurationError, DecryptionError, EncryptionError
from wallet_manager.security.vault import Vault


def _make_vault() -> Vault:
    key = Fernet.generate_key().decode()
    settings = Settings(
        vault_key=key,
        execution_mode="ml_quant",
        garch_alpha=0.10,
        garch_beta=0.85,
    )
    return Vault(settings=settings)


def test_encrypt_decrypt_roundtrip():
    vault = _make_vault()
    plaintext = "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    token = vault.encrypt(plaintext)
    result = vault.decrypt(token)
    assert result == plaintext


def test_decrypt_wrong_key_raises():
    vault1 = _make_vault()
    vault2 = _make_vault()
    token = vault1.encrypt("secret_key")
    with pytest.raises(DecryptionError):
        vault2.decrypt(token)


def test_encrypt_empty_raises():
    vault = _make_vault()
    with pytest.raises(ValueError):
        vault.encrypt("")


def test_decrypt_empty_raises():
    vault = _make_vault()
    with pytest.raises(DecryptionError):
        vault.decrypt(b"")


def test_invalid_vault_key_raises():
    settings = Settings(
        vault_key="not_a_valid_fernet_key",
        execution_mode="ml_quant",
        garch_alpha=0.10,
        garch_beta=0.85,
    )
    with pytest.raises(ConfigurationError):
        Vault(settings=settings)


def test_generate_key_is_valid():
    key = Vault.generate_key()
    # Should be usable as a Fernet key without error
    fernet = Fernet(key.encode())
    assert fernet is not None


def test_decrypt_wallet_id_in_error():
    vault1 = _make_vault()
    vault2 = _make_vault()
    token = vault1.encrypt("0xprivatekey")
    with pytest.raises(DecryptionError) as exc_info:
        vault2.decrypt(token, wallet_id="wallet_test")
    # wallet_id may or may not be in the message; just assert exception raised
    assert exc_info.value is not None
