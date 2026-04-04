"""
tests/unit/test_compound_integration.py

Unit tests for Compound V3 APY integration (Task 1.2).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.actions import (
    DEFAULT_ALTERNATIVE_APY_PCT,
    _get_compound_v3_apy,
    get_alternative_stable_yield_apy,
)

RPC_URL = "https://base-rpc.example.com"

# Compound V3 rate_per_second that gives ~4.2% APY:
# apy = (rate / 1e18) * 31536000 * 100 => 4.2
# rate = 4.2 / 100 / 31536000 * 1e18 ~= 1_331_502_720
SECONDS_PER_YEAR = 365 * 24 * 3600
TARGET_APY_PCT = 4.2
TARGET_RATE = int(TARGET_APY_PCT / 100 / SECONDS_PER_YEAR * 1e18)


def _make_comet_mock(utilization: int, rate_per_second: int) -> MagicMock:
    """Build a mock web3 contract that returns given utilization and supply rate."""
    comet = MagicMock()
    comet.functions.getUtilization.return_value.call = AsyncMock(return_value=utilization)
    comet.functions.getSupplyRate.return_value.call = AsyncMock(return_value=rate_per_second)
    return comet


class TestGetCompoundV3Apy:
    async def test_valid_utilization_returns_apy(self) -> None:
        """Mock RPC returning valid utilization -> APY returned as percentage."""
        utilization = 800_000_000_000_000_000  # 0.8 (80%)
        comet_mock = _make_comet_mock(utilization, TARGET_RATE)

        mock_w3_cls = MagicMock()
        w3_instance = MagicMock()
        w3_instance.eth.contract.return_value = comet_mock
        mock_w3_cls.return_value = w3_instance
        mock_w3_cls.to_checksum_address = lambda addr: addr

        with patch("web3.AsyncWeb3", mock_w3_cls), patch("web3.providers.AsyncHTTPProvider"):
            result = await _get_compound_v3_apy(RPC_URL)

        assert result is not None
        assert 0.0 < result < 50.0, f"APY {result} out of expected 0-50% range"
        assert abs(result - TARGET_APY_PCT) < 0.01

    async def test_rpc_exception_returns_none(self) -> None:
        """Mock RPC raising an exception -> returns None (fallback test)."""
        mock_w3_cls = MagicMock()
        mock_w3_cls.side_effect = Exception("RPC error")

        with patch("web3.AsyncWeb3", mock_w3_cls):
            result = await _get_compound_v3_apy(RPC_URL)

        assert result is None

    async def test_apy_within_0_50_range(self) -> None:
        """APY value must be in a plausible 0-50% range."""
        utilization = 500_000_000_000_000_000  # 50%
        comet_mock = _make_comet_mock(utilization, TARGET_RATE)

        mock_w3_cls = MagicMock()
        w3_instance = MagicMock()
        w3_instance.eth.contract.return_value = comet_mock
        mock_w3_cls.return_value = w3_instance
        mock_w3_cls.to_checksum_address = lambda addr: addr

        with patch("web3.AsyncWeb3", mock_w3_cls), patch("web3.providers.AsyncHTTPProvider"):
            result = await _get_compound_v3_apy(RPC_URL)

        assert result is not None
        assert 0.0 <= result <= 50.0


class TestGetAlternativeStableYieldApy:
    async def test_compound_v3_success_returns_compound(self) -> None:
        """When Compound V3 succeeds, returns Compound V3 APY and name."""
        with patch("wallet_manager.actions._get_compound_v3_apy", new=AsyncMock(return_value=4.2)):
            apy, name = await get_alternative_stable_yield_apy(RPC_URL)

        assert apy == 4.2
        assert name == "Compound V3"

    async def test_compound_v3_failure_falls_back_to_constant(self) -> None:
        """When Compound V3 fails (None), falls back to DEFAULT_ALTERNATIVE_APY_PCT."""
        with patch("wallet_manager.actions._get_compound_v3_apy", new=AsyncMock(return_value=None)):
            apy, name = await get_alternative_stable_yield_apy(RPC_URL)

        assert apy == DEFAULT_ALTERNATIVE_APY_PCT
        assert "Compound" in name or "Stable" in name
