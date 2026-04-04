"""
wallet_manager/security/vault_manager.py

Vault Key Age Tracking and Rotation Reminder.

Tracks vault key age and sends a Telegram reminder when rotation
is overdue. Rotation threshold is cfg.vault_key_rotation_days (default 90).

NO automatic rotation — only a reminder. The operator must call
Vault.rotate_key() via the /rotate_vault_key Telegram command.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    pass

log = build_logger(__name__)

_VAULT_META_KEY = "_vault_meta"


class VaultKeyManager:
    """Tracks vault key age and sends a Telegram reminder when rotation
    is overdue.

    Save key creation time in wallets.json:
      "_vault_meta": {"key_created_at": float, "last_rotated_at": float}

    Rotation reminder condition:
      days_since_rotation > cfg.vault_key_rotation_days (default 90)

    NO automatic rotation — only a reminder. The operator must call
    Vault.rotate_key() via the /rotate_vault_key Telegram command.
    """

    def days_since_rotation(self, wallets_path: Path) -> float:
        """Return number of days since the vault key was last rotated.

        Reads metadata from wallets_path. Returns float('inf') if no
        metadata is found (key has never been tracked).
        """
        try:
            if not wallets_path.exists():
                return float("inf")
            with open(wallets_path, encoding="utf-8") as f:
                data = json.load(f)

            # data may be a list of wallet dicts or a dict with a meta key
            if isinstance(data, list):
                meta = {}
                for item in data:
                    if isinstance(item, dict) and _VAULT_META_KEY in item:
                        meta = item[_VAULT_META_KEY]
                        break
            elif isinstance(data, dict):
                meta = data.get(_VAULT_META_KEY, {})
            else:
                return float("inf")

            last_rotated = meta.get("last_rotated_at") or meta.get("key_created_at")
            if last_rotated is None:
                return float("inf")

            elapsed_seconds = time.time() - float(last_rotated)
            return elapsed_seconds / 86400.0

        except Exception as exc:
            log.warning("VaultKeyManager: could not read vault meta: %s", exc)
            return float("inf")

    async def check_and_remind(self, notifier: Any, wallets_path: Path) -> None:
        """Send a Telegram reminder if vault key rotation is overdue."""
        from wallet_manager.config import get_settings
        from wallet_manager.core.models import EventKind, NotificationEvent

        cfg = get_settings()
        threshold_days = cfg.vault_key_rotation_days
        days = self.days_since_rotation(wallets_path)

        if days > threshold_days:
            log.warning(
                "VaultKeyManager: vault key is %.1f days old (threshold=%d days). "
                "Use /rotate_vault_key to rotate.",
                days, threshold_days,
            )
            try:
                await notifier.send_event(NotificationEvent(
                    kind=EventKind.SYSTEM_ALERT,
                    detail=(
                        f"Vault key rotation overdue: {days:.0f} days old "
                        f"(threshold: {threshold_days} days). "
                        "Use /rotate_vault_key <new_key> to rotate."
                    ),
                ))
            except Exception as exc:
                log.warning("VaultKeyManager: Telegram alert failed: %s", exc)
        else:
            log.debug(
                "VaultKeyManager: vault key age=%.1f days (threshold=%d days) — OK.",
                days, threshold_days,
            )

    @staticmethod
    def record_rotation(wallets_path: Path) -> None:
        """Update the last_rotated_at timestamp in wallets.json."""
        try:
            if not wallets_path.exists():
                return
            with open(wallets_path, encoding="utf-8") as f:
                data = json.load(f)

            now = time.time()
            if isinstance(data, list):
                found = False
                for item in data:
                    if isinstance(item, dict) and _VAULT_META_KEY in item:
                        item[_VAULT_META_KEY]["last_rotated_at"] = now
                        found = True
                        break
                if not found:
                    data.append({_VAULT_META_KEY: {"key_created_at": now, "last_rotated_at": now}})
            elif isinstance(data, dict):
                if _VAULT_META_KEY not in data:
                    data[_VAULT_META_KEY] = {}
                data[_VAULT_META_KEY]["last_rotated_at"] = now
            else:
                return

            with open(wallets_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            log.info("VaultKeyManager: recorded key rotation timestamp.")
        except Exception as exc:
            log.warning("VaultKeyManager: could not update rotation timestamp: %s", exc)


_vault_manager: VaultKeyManager | None = None


def get_vault_manager() -> VaultKeyManager:
    """Return the process-wide VaultKeyManager singleton."""
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = VaultKeyManager()
    return _vault_manager
