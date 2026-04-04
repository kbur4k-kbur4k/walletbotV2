"""
wallet_manager/env_loader.py

Shared .env discovery so every module resolves configuration the same way.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _candidate_env_paths() -> list[Path]:
    """Return env files in precedence order from lowest to highest priority."""
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent
    return [
        package_dir / ".env",
        repo_root / ".env",
    ]


def load_project_env() -> list[Path]:
    """Load package and repo .env files, letting repo-root values win."""
    loaded: list[Path] = []
    for idx, path in enumerate(_candidate_env_paths()):
        if path.exists():
            # The package-local .env provides low-priority defaults. The repo-root
            # .env should override it so Docker/local runs see one canonical config.
            load_dotenv(path, override=idx > 0)
            loaded.append(path)
    return loaded
