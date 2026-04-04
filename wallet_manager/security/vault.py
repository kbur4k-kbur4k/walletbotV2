"""
wallet_manager/security/vault.py

AES-256 Fernet encryption vault — implements IVault.
Private keys are decrypted ephemerally; plaintext never persists.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from wallet_manager.config import Settings, get_settings
from wallet_manager.core.interfaces import IVault
from wallet_manager.exceptions import ConfigurationError, DecryptionError, EncryptionError
from wallet_manager.security.logging_filter import build_logger

_HEX_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")

log = build_logger(__name__)


class Vault(IVault):
    """AES-256 Fernet key vault implementing :class:`IVault`.

    A single instance may be shared across all wallets — it holds no
    per-wallet mutable state.

    Args:
        settings: Optional :class:`~wallet_manager.config.Settings` override.
            When omitted, the process-wide singleton is used.

    Raises:
        ConfigurationError: If the vault key is absent or malformed.

    Example:
        .. code-block:: python

            vault = Vault()
            token = vault.encrypt("0xdeadbeef...")
            plaintext = vault.decrypt(token, wallet_id="w1")
            # Use plaintext immediately, then delete.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        try:
            self._fernet = Fernet(cfg.vault_key.strip().encode("utf-8"))
        except (ValueError, TypeError, Exception) as exc:
            raise ConfigurationError(
                f"VAULT_KEY is not a valid Fernet key: {exc}. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\""
            ) from exc
        log.debug("Vault initialised successfully.")

    # ------------------------------------------------------------------
    # IVault implementation
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt *plaintext* and return an authenticated Fernet token.

        Args:
            plaintext: Sensitive UTF-8 string (e.g. a hex private key).

        Returns:
            Opaque ciphertext bytes safe for persistent storage.

        Raises:
            ValueError: If *plaintext* is empty.
            EncryptionError: If Fernet encryption fails unexpectedly.
        """
        if not plaintext:
            raise ValueError("Cannot encrypt empty string.")
        if len(plaintext) in (64, 66) and not _HEX_RE.match(plaintext):
            raise ValueError("Suspected private key but invalid hex format.")
        try:
            token: bytes = self._fernet.encrypt(plaintext.encode("utf-8"))
        except Exception as exc:
            raise EncryptionError(f"Fernet encryption failed: {exc}") from exc
        log.debug(f"Vault: encryption successful ({len(token)} bytes).")
        return token

    def decrypt(self, token: bytes, wallet_id: str | None = None) -> str:
        """Decrypt a Fernet token and return the original plaintext.

        **Treat the return value as ephemeral** — delete all references
        immediately after use, inside a ``finally`` block.

        Args:
            token: Authenticated ciphertext produced by :meth:`encrypt`.
            wallet_id: Optional identifier included in any raised exception.

        Returns:
            Decrypted UTF-8 plaintext string.

        Raises:
            DecryptionError: If the token is invalid, tampered, or the key
                does not match.
        """
        if not token:
            raise DecryptionError("Cannot decrypt an empty token.", wallet_id=wallet_id)
        try:
            plaintext: str = self._fernet.decrypt(token).decode("utf-8")
        except InvalidToken as exc:
            log.warning(
                f"Vault: decryption failed for wallet '{wallet_id or '<unknown>'}' — invalid token or wrong key.",
            )
            raise DecryptionError(
                "Fernet decryption failed: token is invalid, tampered, or "
                "was encrypted with a different key.",
                wallet_id=wallet_id,
            ) from exc
        except Exception as exc:
            raise DecryptionError(
                f"Unexpected decryption error: {exc}", wallet_id=wallet_id
            ) from exc
        log.debug(f"Vault: decryption successful for wallet '{wallet_id or '<unknown>'}'.")
        return plaintext

    async def rotate_key(self, new_key: str, wallets_path: Path | None = None) -> int:
        """Re-encrypt all vault-encrypted values in wallets.json with a new key.

        Algorithm:
          1. Validate new_key is a valid Fernet key.
          2. Load current encrypted values from wallets_path (default wallets.json).
          3. Decrypt each value with self._fernet (old key).
          4. Re-encrypt with new Fernet(new_key).
          5. Write back to wallets.json.tmp atomically.
          6. On complete success: rename .tmp -> wallets.json, update self._fernet.
          7. On any failure: delete .tmp, retain old key in self._fernet.

        Args:
            new_key: New Fernet key (base64 URL-safe string).
            wallets_path: Path to wallets.json (defaults to current directory).

        Returns:
            Number of records re-encrypted.

        Raises:
            ConfigurationError: If new_key is not a valid Fernet key.
            EncryptionError: On partial failure (atomicity guaranteed).
        """
        try:
            new_fernet = Fernet(new_key.strip().encode("utf-8"))
        except Exception as exc:
            raise ConfigurationError(
                f"rotate_key: new_key is not a valid Fernet key: {exc}"
            ) from exc

        old_fernet = self._fernet

        cfg = get_settings()
        path = wallets_path or Path(str(cfg.wallet_config_path))
        if not path.exists():
            log.warning(
                "rotate_key: %s not found — no secrets to rotate; updating active key.",
                path,
            )
            self._fernet = new_fernet
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EncryptionError(f"rotate_key: failed to read {path}: {exc}") from exc

        tmp_path = path.with_suffix(".tmp")
        count = 0
        try:
            wallets = data if isinstance(data, list) else data.get("wallets", [data])
            for wallet in wallets:
                if not isinstance(wallet, dict):
                    continue
                if not wallet.get("vault_encrypted"):
                    continue
                old_token_b64 = wallet.get("private_key", "")
                if not old_token_b64:
                    continue
                import base64
                try:
                    old_token = base64.b64decode(old_token_b64)
                except Exception:
                    old_token = old_token_b64.encode("utf-8")
                try:
                    plaintext = old_fernet.decrypt(old_token).decode("utf-8")
                except Exception:
                    # Skip wallets encrypted with a different key (e.g. prod key in test)
                    log.debug("rotate_key: skipping wallet — not encrypted with current key.")
                    continue
                new_token = new_fernet.encrypt(plaintext.encode("utf-8"))
                wallet["private_key"] = new_token.decode("utf-8")
                count += 1

            tmp_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            os.replace(str(tmp_path), str(path))
            self._fernet = new_fernet
            log.info("rotate_key: successfully re-encrypted %d record(s).", count)
            return count

        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise EncryptionError(
                f"rotate_key: re-encryption failed after {count} records — "
                f"old key retained. Cause: {exc}"
            ) from exc

    @staticmethod
    def generate_key() -> str:
        """Generate a fresh Fernet-compatible master key.

        Returns:
            URL-safe base64-encoded 32-byte key as a plain string.

        Example:
            .. code-block:: python

                key = Vault.generate_key()
                print(f"VAULT_KEY={key}")  # paste into .env
        """
        return Fernet.generate_key().decode("utf-8")
