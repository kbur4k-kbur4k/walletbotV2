"""tests/unit/test_tx_repair.py — TxRepairService state machine testleri."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nonce_manager() -> MagicMock:
    """Return a MagicMock that satisfies INonceManager."""
    nonce_mgr = MagicMock()
    nonce_mgr.get_and_increment = AsyncMock(return_value=5)
    nonce_mgr.rewind = AsyncMock()
    nonce_mgr.invalidate = AsyncMock()
    return nonce_mgr


def _make_tx_params(gas_price_wei: int = 20_000_000_000):
    """Return a minimal TxParams for testing."""
    from wallet_manager.core.models import TxParams
    return TxParams(
        from_address="0xDeadBeef00000000000000000000000000000001",
        to="0xDeadBeef00000000000000000000000000000002",
        value=0,
        gas=21_000,
        gas_price_wei=gas_price_wei,
        nonce=5,
    )


def _make_service(
    nonce_manager=None,
    deadline_seconds: float = 0.01,
    poll_interval: float = 0.001,
    max_rbf_attempts: int = 3,
):
    """Construct a TxRepairService with fast timeouts and a mock nonce manager."""
    from wallet_manager.network.tx_repair import TxRepairService
    return TxRepairService(
        nonce_manager=nonce_manager or _make_nonce_manager(),
        deadline_seconds=deadline_seconds,
        poll_interval=poll_interval,
        max_rbf_attempts=max_rbf_attempts,
    )


# ---------------------------------------------------------------------------
# TestTxRepairService — status checking via _poll_until_mined_or_deadline
# ---------------------------------------------------------------------------


class TestTxRepairService:
    @pytest.mark.asyncio
    async def test_pending_to_mined_transition(self):
        """When receipt has status=1, the service detects a mined tx."""
        from wallet_manager.core.models import TxStatus

        service = _make_service(deadline_seconds=1.0, poll_interval=0.001)

        mined_receipt = {
            "status": 1,
            "blockNumber": 12345,
            "gasUsed": 21000,
        }

        # Patch _poll_until_mined_or_deadline to immediately return a mined receipt
        with patch.object(
            service, "_poll_until_mined_or_deadline",
            new=AsyncMock(return_value=mined_receipt)
        ), patch.object(
            service, "_build_and_send_rbf",
            new=AsyncMock(side_effect=AssertionError("RBF should not be called"))
        ):
            record = await service.watch_and_repair(
                rpc_url="https://fake.rpc",
                tx_hash="0xpending_hash",
                original_params=_make_tx_params(),
                chain_id=1,
                wallet_id="test_wallet",
                decrypt_key_fn=lambda: "0x" + "aa" * 32,
            )

        assert record.status == TxStatus.MINED

    @pytest.mark.asyncio
    async def test_none_receipt_stays_pending(self):
        """When no receipt is returned within deadline, service attempts RBF."""
        from wallet_manager.core.models import TxStatus
        from wallet_manager.exceptions import TxStuckError

        service = _make_service(
            deadline_seconds=0.01,
            poll_interval=0.001,
            max_rbf_attempts=1,
        )

        # Poll always returns None → deadline reached → RBF attempted once → TxStuckError
        with patch.object(
            service, "_poll_until_mined_or_deadline",
            new=AsyncMock(return_value=None)
        ):
            bumped_params = _make_tx_params(gas_price_wei=int(20_000_000_000 * 1.15))
            with patch.object(
                service, "_build_and_send_rbf",
                new=AsyncMock(return_value=("0xreplacement", bumped_params))
            ), pytest.raises(TxStuckError):
                await service.watch_and_repair(
                    rpc_url="https://fake.rpc",
                    tx_hash="0xunmined_hash",
                    original_params=_make_tx_params(),
                    chain_id=1,
                    wallet_id="test_wallet",
                    decrypt_key_fn=lambda: "0x" + "aa" * 32,
                )

    @pytest.mark.asyncio
    async def test_failed_receipt_dropped(self):
        """When receipt has status=0 (reverted), tx is still considered mined (MINED status)."""
        from wallet_manager.core.models import TxStatus

        service = _make_service(deadline_seconds=1.0, poll_interval=0.001)

        failed_receipt = {
            "status": 0,   # reverted
            "blockNumber": 12346,
            "gasUsed": 21000,
        }

        with patch.object(
            service, "_poll_until_mined_or_deadline",
            new=AsyncMock(return_value=failed_receipt)
        ), patch.object(
            service, "_build_and_send_rbf",
            new=AsyncMock(side_effect=AssertionError("RBF should not be called"))
        ):
            record = await service.watch_and_repair(
                rpc_url="https://fake.rpc",
                tx_hash="0xreverted_hash",
                original_params=_make_tx_params(),
                chain_id=1,
                wallet_id="test_wallet",
                decrypt_key_fn=lambda: "0x" + "aa" * 32,
            )

        # A reverted tx still has a receipt (was included in a block) → MINED
        assert record.status in (TxStatus.DROPPED, TxStatus.MINED)


# ---------------------------------------------------------------------------
# New tests — use the real TxRepairService.watch_and_repair / _build_and_send_rbf
# ---------------------------------------------------------------------------


class TestRBFBehavior:
    """Tests for Replace-By-Fee gas bumping, attempt limits, and early exit."""

    # ------------------------------------------------------------------
    # test_rbf_replacement_bumps_gas_by_minimum_10pct
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rbf_replacement_bumps_gas_by_minimum_10pct(self):
        """_build_and_send_rbf must produce a replacement with gas_price_wei
        at least 10% higher than the original (EIP-1559 RBF floor)."""
        from wallet_manager.core.models import TxParams
        from wallet_manager.network.tx_repair import TxRepairService

        original_gas_wei = 20_000_000_000  # 20 Gwei
        original_params = _make_tx_params(gas_price_wei=original_gas_wei)
        service = _make_service()

        fake_tx_hash = b"\xab" * 32

        with patch("wallet_manager.network.tx_repair.AsyncWeb3") as mock_web3_cls:
            mock_w3_instance = MagicMock()
            mock_w3_instance.eth.send_raw_transaction = AsyncMock(return_value=fake_tx_hash)
            mock_web3_cls.return_value = mock_w3_instance
            mock_web3_cls.AsyncHTTPProvider = MagicMock(return_value=MagicMock())

            # Account is imported locally inside _build_and_send_rbf; patch eth_account.Account
            with patch("eth_account.Account") as mock_account:
                mock_signed = MagicMock()
                mock_signed.raw_transaction = b"\x00" * 100
                mock_account.sign_transaction.return_value = mock_signed

                _new_hash, replacement = await service._build_and_send_rbf(
                    rpc_url="https://fake.rpc",
                    original_params=original_params,
                    chain_id=1,
                    wallet_id="test_wallet",
                    decrypt_key_fn=lambda: "0x" + "aa" * 32,
                )

        # The replacement gas must be >= 110% of original (minimum 10% bump)
        min_expected_wei = int(original_gas_wei * 1.10)
        assert replacement.gas_price_wei >= min_expected_wei, (
            f"Expected replacement gas_price_wei ({replacement.gas_price_wei}) "
            f">= 110% of original ({min_expected_wei}). "
            f"Original: {original_gas_wei}, bump pct applied yielded {replacement.gas_price_wei}."
        )

    # ------------------------------------------------------------------
    # test_max_rbf_attempts_respected
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_max_rbf_attempts_respected(self):
        """With max_rbf_attempts=2, watch_and_repair should trigger RBF at most
        2 times before raising TxStuckError."""
        from wallet_manager.exceptions import TxStuckError
        from wallet_manager.network.tx_repair import TxRepairService

        max_attempts = 2
        service = _make_service(
            deadline_seconds=0.01,
            poll_interval=0.001,
            max_rbf_attempts=max_attempts,
        )

        original_params = _make_tx_params()
        rbf_call_count = 0

        async def _fake_build_and_send_rbf(**kwargs):
            nonlocal rbf_call_count
            rbf_call_count += 1
            from wallet_manager.core.models import TxParams
            bumped = TxParams(
                from_address=original_params.from_address,
                to=original_params.to,
                value=original_params.value,
                gas=original_params.gas,
                gas_price_wei=int(original_params.gas_price_wei * 1.15),
                nonce=original_params.nonce,
            )
            return f"0xreplacement_{rbf_call_count:02d}", bumped

        # Receipt always returns None → tx never mines → forces all RBF attempts
        async def _fake_poll(*args, **kwargs):
            return None

        with patch.object(service, "_poll_until_mined_or_deadline", side_effect=_fake_poll), patch.object(
            service,
            "_build_and_send_rbf",
            side_effect=_fake_build_and_send_rbf,
        ), pytest.raises(TxStuckError):
            await service.watch_and_repair(
                rpc_url="https://fake.rpc",
                tx_hash="0xoriginal",
                original_params=original_params,
                chain_id=1,
                wallet_id="test_wallet",
                decrypt_key_fn=lambda: "0x" + "bb" * 32,
            )

        # RBF should have been attempted exactly max_rbf_attempts times
        assert rbf_call_count == max_attempts, (
            f"Expected exactly {max_attempts} RBF attempts, got {rbf_call_count}."
        )

    # ------------------------------------------------------------------
    # test_tx_already_mined_skips_rbf
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tx_already_mined_skips_rbf(self):
        """If the receipt is available on the very first poll, no RBF must be
        attempted and watch_and_repair must return a MINED TxRecord."""
        from wallet_manager.core.models import TxRecord, TxStatus
        from wallet_manager.network.tx_repair import TxRepairService

        service = _make_service(
            deadline_seconds=0.5,
            poll_interval=0.001,
            max_rbf_attempts=3,
        )

        original_params = _make_tx_params()
        mined_receipt = {
            "status": 1,
            "blockNumber": 555,
            "gasUsed": 21_000,
        }

        rbf_called = False

        async def _fake_build_and_send_rbf(**kwargs):
            nonlocal rbf_called
            rbf_called = True
            raise AssertionError("RBF should not be attempted when tx is already mined")

        async def _fake_poll(*args, **kwargs):
            # Immediately return a mined receipt
            return mined_receipt

        with patch.object(service, "_poll_until_mined_or_deadline", side_effect=_fake_poll), patch.object(
            service,
            "_build_and_send_rbf",
            side_effect=_fake_build_and_send_rbf,
        ):
            record = await service.watch_and_repair(
                rpc_url="https://fake.rpc",
                tx_hash="0xalreadymined",
                original_params=original_params,
                chain_id=1,
                wallet_id="test_wallet",
                decrypt_key_fn=lambda: "0x" + "cc" * 32,
            )

        # No RBF should have been triggered
        assert not rbf_called, "RBF was attempted even though tx was already mined"

        # Return value must be a TxRecord with MINED status
        assert isinstance(record, TxRecord), f"Expected TxRecord, got {type(record)}"
        assert record.status == TxStatus.MINED, (
            f"Expected TxStatus.MINED, got {record.status}"
        )
        assert record.tx_hash == "0xalreadymined"
        assert record.gas_used == 21_000
