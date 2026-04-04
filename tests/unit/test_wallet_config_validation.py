"""tests/unit/test_wallet_config_validation.py

Tests for the wallet config JSON schema validator (ADIM 4.4).
"""
from __future__ import annotations

import pytest

from wallet_manager.exceptions import ConfigurationError
from wallet_manager.orchestrator import validate_wallets_schema

_VALID_WALLET = {
    "wallet_id": "wallet_01",
    "address": "0x4200000000000000000000000000000000000006",
    "private_key": "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
    "chain_id": 8453,
    "rpc_url": "https://mainnet.base.org",
    "max_gas_price_gwei": 50.0,
    "enabled": True,
}


def test_valid_wallet_config_passes():
    """A correctly formed wallet config must not raise."""
    validate_wallets_schema([_VALID_WALLET])
    assert True  # no exception raised — validation passed


def test_invalid_address_raises_configuration_error():
    """An invalid address format must raise ConfigurationError."""
    bad = {**_VALID_WALLET, "address": "not_an_address"}
    with pytest.raises(ConfigurationError):
        validate_wallets_schema([bad])


def test_missing_wallet_id_raises():
    """Missing wallet_id must raise ConfigurationError."""
    bad = {k: v for k, v in _VALID_WALLET.items() if k != "wallet_id"}
    with pytest.raises(ConfigurationError, match="wallet_id"):
        validate_wallets_schema([bad])


def test_invalid_wallet_id_pattern_raises():
    """wallet_id with invalid characters must raise ConfigurationError."""
    bad = {**_VALID_WALLET, "wallet_id": "wallet 01!"}
    with pytest.raises(ConfigurationError, match="wallet_id"):
        validate_wallets_schema([bad])


def test_missing_private_key_raises():
    """Missing private_key (and encrypted_private_key) must raise."""
    bad = {k: v for k, v in _VALID_WALLET.items() if k not in ("private_key", "encrypted_private_key")}
    with pytest.raises(ConfigurationError):
        validate_wallets_schema([bad])


def test_max_gas_price_above_limit_raises():
    """max_gas_price_gwei >= 10000 must raise ConfigurationError."""
    bad = {**_VALID_WALLET, "max_gas_price_gwei": 15000.0}
    with pytest.raises(ConfigurationError, match="max_gas_price_gwei"):
        validate_wallets_schema([bad])


def test_all_errors_collected_before_raising():
    """Multiple invalid wallets must report ALL errors in a single exception."""
    bad1 = {k: v for k, v in _VALID_WALLET.items() if k != "wallet_id"}
    bad2 = {**_VALID_WALLET, "wallet_id": "wallet_02", "max_gas_price_gwei": 99999.0}

    with pytest.raises(ConfigurationError) as exc_info:
        validate_wallets_schema([bad1, bad2])

    msg = str(exc_info.value)
    # Both errors should be present in the same exception
    assert "wallet_id" in msg
    assert "max_gas_price_gwei" in msg


def test_vault_meta_entry_is_skipped():
    """_vault_meta dict entries must be silently skipped."""
    data = [{"_vault_meta": {"version": 1}}, _VALID_WALLET]
    validate_wallets_schema(data)  # must not raise
    assert True  # no exception raised


def test_enabled_field_accepts_bool():
    """enabled=False is a valid wallet config."""
    entry = {**_VALID_WALLET, "enabled": False}
    validate_wallets_schema([entry])
    assert True  # no exception raised


def test_enabled_field_rejects_string():
    """enabled='true' (string) must raise ConfigurationError."""
    bad = {**_VALID_WALLET, "enabled": "true"}
    with pytest.raises(ConfigurationError, match="enabled"):
        validate_wallets_schema([bad])
