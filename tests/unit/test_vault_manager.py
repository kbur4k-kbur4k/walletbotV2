"""
tests/unit/test_vault_manager.py

Unit tests for wallet_manager/security/vault_manager.py

Covers VaultKeyManager:
- days_since_rotation() returns float('inf') for missing file
- days_since_rotation() returns float('inf') when no _vault_meta key
- days_since_rotation() correctly computes elapsed days
- record_rotation() updates last_rotated_at in-place
- check_and_remind() calls notifier.send_event when rotation is overdue
- check_and_remind() does NOT call notifier.send_event when rotation is fresh
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.security.vault_manager import VaultKeyManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wallets(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# days_since_rotation()
# ---------------------------------------------------------------------------


class TestDaysSinceRotation:
    def test_returns_inf_when_no_file(self, tmp_path: Path):
        """Non-existent wallets.json -> float('inf')."""
        manager = VaultKeyManager()
        missing = tmp_path / "does_not_exist.json"
        result = manager.days_since_rotation(missing)
        assert result == float("inf")

    def test_returns_inf_when_no_meta_dict(self, tmp_path: Path):
        """JSON dict without '_vault_meta' key -> float('inf')."""
        wallets = tmp_path / "wallets.json"
        _write_wallets(wallets, {"wallet_01": "some_data"})

        manager = VaultKeyManager()
        result = manager.days_since_rotation(wallets)
        assert result == float("inf")

    def test_returns_inf_when_no_meta_list(self, tmp_path: Path):
        """JSON list with no item containing '_vault_meta' -> float('inf')."""
        wallets = tmp_path / "wallets.json"
        _write_wallets(wallets, [{"address": "0xabc"}])

        manager = VaultKeyManager()
        result = manager.days_since_rotation(wallets)
        assert result == float("inf")

    def test_returns_inf_when_meta_has_no_timestamps(self, tmp_path: Path):
        """_vault_meta present but no last_rotated_at / key_created_at -> inf."""
        wallets = tmp_path / "wallets.json"
        _write_wallets(wallets, {"_vault_meta": {}})

        manager = VaultKeyManager()
        result = manager.days_since_rotation(wallets)
        assert result == float("inf")

    def test_calculates_correctly_one_day_ago(self, tmp_path: Path):
        """last_rotated_at set 1 day ago -> result is approximately 1.0."""
        one_day_ago = time.time() - 86400
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            {"_vault_meta": {"last_rotated_at": one_day_ago}},
        )

        manager = VaultKeyManager()
        result = manager.days_since_rotation(wallets)

        # Allow ±0.01 day (~15 seconds) for execution overhead
        assert pytest.approx(result, abs=0.01) == 1.0

    def test_calculates_from_key_created_at_when_no_rotated_at(
        self, tmp_path: Path
    ):
        """Falls back to key_created_at when last_rotated_at is absent."""
        two_days_ago = time.time() - 2 * 86400
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            {"_vault_meta": {"key_created_at": two_days_ago}},
        )

        manager = VaultKeyManager()
        result = manager.days_since_rotation(wallets)

        assert pytest.approx(result, abs=0.02) == 2.0

    def test_calculates_correctly_from_list_format(self, tmp_path: Path):
        """Works when wallets.json is a list and _vault_meta is an item."""
        one_day_ago = time.time() - 86400
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            [
                {"address": "0x1234"},
                {"_vault_meta": {"last_rotated_at": one_day_ago}},
            ],
        )

        manager = VaultKeyManager()
        result = manager.days_since_rotation(wallets)

        assert pytest.approx(result, abs=0.01) == 1.0


# ---------------------------------------------------------------------------
# record_rotation()
# ---------------------------------------------------------------------------


class TestRecordRotation:
    def test_record_rotation_updates_timestamp_dict(self, tmp_path: Path):
        """record_rotation() writes a fresh last_rotated_at into a dict JSON."""
        old_ts = time.time() - 7 * 86400  # 7 days ago
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            {"_vault_meta": {"key_created_at": old_ts, "last_rotated_at": old_ts}},
        )

        before = time.time()
        VaultKeyManager.record_rotation(wallets)
        after = time.time()

        data = json.loads(wallets.read_text())
        new_ts = data["_vault_meta"]["last_rotated_at"]

        assert before <= new_ts <= after, (
            "last_rotated_at should be updated to approximately now"
        )
        assert new_ts > old_ts

    def test_record_rotation_creates_meta_if_missing(self, tmp_path: Path):
        """record_rotation() creates _vault_meta when not present in dict."""
        wallets = tmp_path / "wallets.json"
        _write_wallets(wallets, {"wallet_01": "data"})

        before = time.time()
        VaultKeyManager.record_rotation(wallets)
        after = time.time()

        data = json.loads(wallets.read_text())
        assert "_vault_meta" in data
        assert before <= data["_vault_meta"]["last_rotated_at"] <= after

    def test_record_rotation_updates_timestamp_list_format(self, tmp_path: Path):
        """record_rotation() updates last_rotated_at in list-format wallets.json."""
        old_ts = time.time() - 14 * 86400
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            [
                {"address": "0xabc"},
                {"_vault_meta": {"last_rotated_at": old_ts}},
            ],
        )

        VaultKeyManager.record_rotation(wallets)

        data = json.loads(wallets.read_text())
        meta_item = next(
            item for item in data if isinstance(item, dict) and "_vault_meta" in item
        )
        assert meta_item["_vault_meta"]["last_rotated_at"] > old_ts

    def test_record_rotation_noop_when_file_missing(self, tmp_path: Path):
        """record_rotation() silently does nothing when the file does not exist."""
        missing = tmp_path / "no_such_file.json"
        # Should not raise
        VaultKeyManager.record_rotation(missing)
        assert not missing.exists()


# ---------------------------------------------------------------------------
# check_and_remind()
# ---------------------------------------------------------------------------


class TestCheckAndRemind:
    @pytest.mark.asyncio
    async def test_sends_alert_when_overdue(self, tmp_path: Path):
        """Notifier.send_event is called when rotation is overdue."""
        # Set last rotation to 200 days ago (well beyond any threshold)
        old_ts = time.time() - 200 * 86400
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            {"_vault_meta": {"last_rotated_at": old_ts}},
        )

        notifier = MagicMock()
        notifier.send_event = AsyncMock()

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_cfg:
            mock_cfg.return_value.vault_key_rotation_days = 90

            manager = VaultKeyManager()
            await manager.check_and_remind(notifier, wallets)

        notifier.send_event.assert_awaited_once()
        assert notifier.send_event.await_count == 1
        assert notifier.send_event.await_count == 1

    @pytest.mark.asyncio
    async def test_no_alert_when_fresh(self, tmp_path: Path):
        """Notifier.send_event is NOT called when rotation is recent."""
        recent_ts = time.time() - 5 * 86400  # only 5 days ago
        wallets = tmp_path / "wallets.json"
        _write_wallets(
            wallets,
            {"_vault_meta": {"last_rotated_at": recent_ts}},
        )

        notifier = MagicMock()
        notifier.send_event = AsyncMock()

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_cfg:
            mock_cfg.return_value.vault_key_rotation_days = 90

            manager = VaultKeyManager()
            await manager.check_and_remind(notifier, wallets)

        notifier.send_event.assert_not_awaited()
        assert notifier.send_event.await_count == 0

    @pytest.mark.asyncio
    async def test_sends_alert_when_no_meta_inf(self, tmp_path: Path):
        """float('inf') days also triggers the reminder (inf > threshold)."""
        wallets = tmp_path / "wallets.json"
        _write_wallets(wallets, {"wallet_01": "no_meta_here"})

        notifier = MagicMock()
        notifier.send_event = AsyncMock()

        with patch(
            "wallet_manager.config.get_settings"
        ) as mock_cfg:
            mock_cfg.return_value.vault_key_rotation_days = 90

            manager = VaultKeyManager()
            await manager.check_and_remind(notifier, wallets)

        notifier.send_event.assert_awaited_once()
        assert notifier.send_event.await_count == 1
