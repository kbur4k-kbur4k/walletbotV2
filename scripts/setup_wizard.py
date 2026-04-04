#!/usr/bin/env python3
"""Bootstrap and validate a local WalletBot workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


PLACEHOLDER_PREFIXES = (
    "YOUR_",
    "GENERATE_WITH_",
    "REPLACE_WITH_",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_env_pairs(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    if not path.exists():
        return pairs
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if stripped in {"changeme", "none", "null"}:
        return True
    return any(stripped.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def bootstrap_workspace(root: Path, *, force: bool = False) -> list[str]:
    actions: list[str] = []
    env_example = root / ".env.example"
    root_env = root / ".env"
    package_env = root / "wallet_manager" / ".env"
    wallet_example = root / "wallets.json.example"
    wallets_file = root / "wallets.json"

    if env_example.exists() and (force or not root_env.exists()):
        shutil.copyfile(env_example, root_env)
        actions.append("created .env from .env.example")

    if root_env.exists():
        root_env_text = root_env.read_text(encoding="utf-8")
        if force or not package_env.exists() or package_env.read_text(encoding="utf-8") != root_env_text:
            package_env.write_text(root_env_text, encoding="utf-8")
            actions.append("synced wallet_manager/.env from root .env")

    if wallet_example.exists() and (force or not wallets_file.exists()):
        shutil.copyfile(wallet_example, wallets_file)
        actions.append("created wallets.json from wallets.json.example")

    if not actions:
        actions.append("workspace files already present; nothing overwritten")
    return actions


def validate_workspace(root: Path) -> tuple[list[str], list[str]]:
    import wallet_manager.config as cfg_module
    from wallet_manager.config import Settings
    from wallet_manager.orchestrator import _load_wallet_configs, validate_wallets_schema

    errors: list[str] = []
    warnings: list[str] = []

    env_path = root / ".env"
    package_env_path = root / "wallet_manager" / ".env"
    wallets_path = root / "wallets.json"

    if not env_path.exists():
        errors.append(".env is missing")
    if not wallets_path.exists():
        errors.append("wallets.json is missing")
    if errors:
        return errors, warnings

    env_pairs = _read_env_pairs(env_path)
    for key in ("VAULT_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        value = env_pairs.get(key, "")
        if not value:
            errors.append(f"{key} is missing in .env")
        elif _is_placeholder(value):
            warnings.append(f"{key} still uses a placeholder value")

    if package_env_path.exists():
        root_env_text = env_path.read_text(encoding="utf-8")
        package_env_text = package_env_path.read_text(encoding="utf-8")
        if root_env_text != package_env_text:
            warnings.append("wallet_manager/.env is out of sync with root .env")
    else:
        warnings.append("wallet_manager/.env is missing; Docker startup may fail")

    try:
        wallet_data = json.loads(wallets_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"wallets.json is not valid JSON: {exc}")
        return errors, warnings

    try:
        validate_wallets_schema(wallet_data)
    except Exception as exc:
        errors.append(str(exc))

    for wallet in wallet_data if isinstance(wallet_data, list) else []:
        if not isinstance(wallet, dict):
            continue
        if _is_placeholder(str(wallet.get("encrypted_private_key", "") or wallet.get("private_key", ""))):
            warnings.append(f"wallet '{wallet.get('wallet_id', '?')}' still uses a placeholder private key")

    original_values = {key: os.environ.get(key) for key in env_pairs}
    try:
        for key, value in env_pairs.items():
            os.environ[key] = value
        os.environ["WALLET_CONFIG_PATH"] = str(wallets_path)
        cfg_module._settings_cache = None
        Settings.from_env()
        _load_wallet_configs()
    except Exception as exc:
        errors.append(f"environment validation failed: {exc}")
    finally:
        cfg_module._settings_cache = None
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite generated files")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="skip bootstrap and only validate the current workspace",
    )
    args = parser.parse_args()

    root = repo_root()
    if not args.validate_only:
        for action in bootstrap_workspace(root, force=args.force):
            print(f"[setup] {action}")

    errors, warnings = validate_workspace(root)

    for warning in warnings:
        print(f"[warn] {warning}")
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    print("[ok] workspace bootstrap and validation completed")
    print("[next] python -m wallet_manager.orchestrator")
    print("[next] open http://localhost:9090/dashboard after startup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
