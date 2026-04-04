"""
wallet_manager/orchestrator_lifecycle.py

LifecycleManager — boot sequence, signal handlers, startup validation,
graceful shutdown, and vault migration.

Extracted from orchestrator.py (Task 4.1).  The LifecycleManager class owns
SIGTERM/SIGINT registration and the async startup validation so that the main
orchestrator module stays focused on wallet coordination.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Protocol

from wallet_manager.config import Settings, get_settings
from wallet_manager.core.models import EventKind, NotificationEvent
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)


class _SupportsStop(Protocol):
    def stop(self) -> None: ...


class _SupportsEncrypt(Protocol):
    def encrypt(self, plaintext: str) -> bytes: ...


class LifecycleManager:
    """Owns the process lifecycle: signal handling, startup validation, and shutdown.

    Typically used once per process — the orchestrator creates a single
    :class:`LifecycleManager`, calls :meth:`setup_signal_handlers` after the
    event loop is running, and awaits :meth:`validate_startup_config` before
    any wallet work begins.

    All methods are safe to call as standalone functions via the class — no
    instance state is required — but grouping them in a class makes injection
    and testing easier.
    """

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    @staticmethod
    def setup_signal_handlers(
        loop: asyncio.AbstractEventLoop,
        command_state: _SupportsStop | object,
    ) -> None:
        """Register SIGTERM and SIGINT handlers for graceful shutdown.

        Translates OS signals into a :meth:`CommandState.stop` call so the
        orchestrator drains in-flight transactions before exiting.  Docker
        ``stop`` and systemd ``stop`` both use SIGTERM.

        Args:
            loop: The running event loop.
            command_state: Shared
                :class:`~wallet_manager.telegram_notifier.CommandState`
                (or any object with a ``stop()`` method).
        """
        import signal as _signal

        def _handle_signal(sig: _signal.Signals) -> None:
            log.warning("Received signal %s — requesting graceful shutdown.", sig.name)
            if hasattr(command_state, "stop"):
                command_state.stop()

        for sig in (_signal.SIGTERM, _signal.SIGINT):
            try:
                def _register(signal_value: _signal.Signals) -> None:
                    _handle_signal(signal_value)

                loop.add_signal_handler(sig, _register, sig)
            except (NotImplementedError, RuntimeError):
                # Windows does not support loop.add_signal_handler; fall back
                # to the default signal handler (KeyboardInterrupt still works).
                pass

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    @staticmethod
    async def validate_startup_config(cfg: Settings | None = None) -> None:
        """Run before the orchestrator begins any wallet work.

        Checklist:
        - Is VAULT_KEY set? (if vault is enabled)
        - Are TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID set?
        - Is at least one of INFURA_KEY or ALCHEMY_KEY set?
        - Does WALLET_CONFIG_PATH exist and parse successfully?
        - Is EXECUTION_MODE a valid value?
        - Does the ml_models/ directory exist? (in ml_quant mode)
        - Are model files present in ml_models/? (warning, not error)

        Raises:
            ConfigurationError: On critical failures.
        """
        from wallet_manager.exceptions import ConfigurationError

        if cfg is None:
            cfg = get_settings()

        warnings: list[str] = []

        # Telegram credentials
        if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
            warnings.append(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — notifications disabled."
            )

        # RPC keys
        if not cfg.infura_key and not cfg.alchemy_key:
            warnings.append(
                "INFURA_KEY and ALCHEMY_KEY not set — using public BASE_RPC endpoints only."
            )

        # Wallet config file
        wallet_path = cfg.wallet_config_path
        if not wallet_path.is_file():
            raise ConfigurationError(
                f"WALLET_CONFIG_PATH={wallet_path} does not exist or is not a file."
            )
        try:
            wallet_data = json.loads(wallet_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(
                f"WALLET_CONFIG_PATH={wallet_path} could not be parsed as JSON: {exc}"
            ) from exc

        wallet_count = len(wallet_data) if isinstance(wallet_data, list) else len(wallet_data.get("wallets", []))
        if wallet_count <= 0:
            raise ConfigurationError("wallets.json contains no configured wallets.")

        encrypted_wallets = [
            item
            for item in (wallet_data if isinstance(wallet_data, list) else wallet_data.get("wallets", []))
            if isinstance(item, dict)
            and "_vault_meta" not in item
            and (item.get("vault_encrypted") or item.get("encrypted_private_key"))
        ]
        if encrypted_wallets:
            from wallet_manager.orchestrator import _encrypted_key_candidates
            from wallet_manager.security.vault import Vault

            if not cfg.vault_key:
                raise ConfigurationError(
                    "Encrypted wallet keys detected but VAULT_KEY is missing."
                )

            vault = Vault(cfg)
            for wallet in encrypted_wallets:
                wallet_id = str(wallet.get("wallet_id", "<unknown>"))
                token_value = wallet.get("encrypted_private_key") or wallet.get("private_key")
                token_str = str(token_value or "").strip()
                if not token_str:
                    raise ConfigurationError(
                        f"Wallet '{wallet_id}' is marked encrypted but has no token value."
                    )

                last_error: Exception | None = None
                for candidate in _encrypted_key_candidates(token_str):
                    try:
                        vault.decrypt(candidate, wallet_id=wallet_id)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc

                if last_error is not None:
                    raise ConfigurationError(
                        f"Wallet '{wallet_id}' encrypted key could not be decrypted with the active VAULT_KEY."
                    ) from last_error

        if cfg.is_live_execution and not cfg.live_trading_enabled:
            raise ConfigurationError(
                f"EXECUTION_MODE={cfg.execution_mode} is a live-capable mode, but "
                "LIVE_TRADING_ENABLED is false. Keep robust_paper or explicitly enable live trading."
            )
        if cfg.is_live_execution and wallet_count > cfg.live_max_wallets:
            raise ConfigurationError(
                f"Live guard allows at most {cfg.live_max_wallets} wallet(s), "
                f"but {wallet_count} wallet(s) are configured."
            )

        # ML models check (warning only)
        if cfg.execution_mode == "ml_quant":
            ml_dir = Path(__file__).resolve().parent / "ml_models"
            if not ml_dir.is_dir():
                warnings.append(
                    "ml_models/ directory not found. ML engine will use statistical fallback."
                )
            else:
                model_files = list(ml_dir.glob("*.json")) + list(ml_dir.glob("*.pt"))
                if not model_files:
                    warnings.append(
                        "No model files in ml_models/. Run scripts/train_models.py first. "
                        "ML engine will use statistical fallback."
                    )

        # Paper trading gate: require minimum paper trades before going live
        _exec_mode = cfg.execution_mode
        if _exec_mode in ("ml_quant", "robust", "robust_live"):
            try:
                from wallet_manager.exceptions import ConfigurationError as _CE
                from wallet_manager.strategies.paper_engine import get_paper_engine

                _pe = await get_paper_engine()
                _pnl = await _pe._ledger.get_pnl_summary()
                _paper_trades = _pnl.get("trade_count", 0)
                _required = cfg.min_paper_trades_before_live
                if _paper_trades < _required:
                    raise _CE(
                        f"EXECUTION_MODE={_exec_mode} requires at least {_required} "
                        f"completed paper trades (found: {_paper_trades}). "
                        f"Run with EXECUTION_MODE=robust_paper first."
                    )
            except ConfigurationError:
                raise
            except Exception as _pe_exc:
                log.warning(
                    "Paper trade gate check failed (%s) — proceeding. "
                    "Ensure paper engine has sufficient trade history before going live.",
                    _pe_exc,
                )

        # RPC + Aave connectivity checks
        try:
            from wallet_manager.utils.health_check import run_startup_health_checks

            health_problems = await run_startup_health_checks(cfg)
            for problem in health_problems:
                warnings.append(f"Health check: {problem}")
        except Exception as _hc_exc:
            log.warning("Startup health checks could not run: %s", _hc_exc)

        for w in warnings:
            log.warning("Startup check: %s", w)

        # Send warnings via Telegram if possible
        if warnings and cfg.telegram_bot_token and cfg.telegram_chat_id:
            try:
                from wallet_manager.telegram_notifier import TelegramNotifier

                async with TelegramNotifier() as notifier:
                    await notifier.send(
                        NotificationEvent(
                            kind=EventKind.SYSTEM_BOOT,
                            detail="⚠️ Startup warnings:\n"
                            + "\n".join(f"• {w}" for w in warnings),
                        )
                    )
            except Exception as _boot_notif_exc:
                log.warning(
                    "Boot notification failed (best-effort): %s", _boot_notif_exc
                )

    # ------------------------------------------------------------------
    # Vault migration
    # ------------------------------------------------------------------

    @staticmethod
    async def migrate_wallets_to_vault(
        vault: _SupportsEncrypt,
        config_path: Path,
    ) -> None:
        """Migrate plaintext private keys in wallets.json to Vault encryption.

        For every private_key field that is a plaintext hex string:
          1. Write backup: wallets.json.bak (before modifying anything)
          2. Encrypt the key with vault.encrypt()
          3. Replace the field value with the encrypted token (base64 str)
          4. Add field: "vault_encrypted": true
          5. Write the modified wallets.json atomically (write to .tmp, rename)
        Idempotent: skips any wallet where vault_encrypted is already true.
        Logs each migrated wallet_id (masked, never the key itself).
        """
        if not config_path.is_file():
            log.warning(
                "migrate_wallets_to_vault: %s not found, skipping.", config_path
            )
            return

        raw = config_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error(
                "migrate_wallets_to_vault: failed to parse %s: %s", config_path, exc
            )
            return

        wallets = data if isinstance(data, list) else [data]
        migrated = 0

        for wallet in wallets:
            if wallet.get("vault_encrypted"):
                continue
            key = wallet.get("private_key", "")
            if not key:
                continue
            # Write backup before first migration
            if migrated == 0:
                bak = config_path.with_suffix(".json.bak")
                bak.write_text(raw, encoding="utf-8")
                log.info("Vault migration: backup written to %s.", bak)

            token_bytes = vault.encrypt(key)
            wallet["private_key"] = token_bytes.decode("utf-8")
            wallet["vault_encrypted"] = True

            wallet_id = wallet.get("wallet_id", "<unknown>")
            log.info("Vault migration: encrypted key for wallet '%s'.", wallet_id)
            migrated += 1

        if migrated == 0:
            log.debug("Vault migration: all wallets already encrypted, nothing to do.")
            return

        # Atomic write: .tmp then rename
        updated = json.dumps(
            data if not isinstance(data, list) else wallets, indent=2
        )
        tmp_path = config_path.with_suffix(".json.tmp")
        tmp_path.write_text(updated, encoding="utf-8")
        tmp_path.replace(config_path)
        log.info(
            "Vault migration: %d wallet(s) migrated and written atomically.", migrated
        )
