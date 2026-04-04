from __future__ import annotations

import os


def test_repo_env_overrides_package_env(tmp_path, monkeypatch):
    from wallet_manager import env_loader

    package_env = tmp_path / "package.env"
    repo_env = tmp_path / "repo.env"
    package_env.write_text("PAPER_INITIAL_BALANCE_USD=1000\n", encoding="utf-8")
    repo_env.write_text("PAPER_INITIAL_BALANCE_USD=10000\n", encoding="utf-8")

    monkeypatch.delenv("PAPER_INITIAL_BALANCE_USD", raising=False)
    monkeypatch.setattr(
        env_loader,
        "_candidate_env_paths",
        lambda: [package_env, repo_env],
    )

    loaded = env_loader.load_project_env()

    assert loaded == [package_env, repo_env]
    assert os.getenv("PAPER_INITIAL_BALANCE_USD") == "10000"
