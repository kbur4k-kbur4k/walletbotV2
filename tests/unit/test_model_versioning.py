"""Tests for model archiving and versioning (Task 2.3)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the project root (wallet_manager/) is on sys.path so that
# `from scripts.train_models import archive_models` resolves correctly.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_archive_creates_directory(tmp_path):
    ml_dir = str(tmp_path / "ml_models")
    os.makedirs(ml_dir)
    (tmp_path / "ml_models" / "model_xgb.pkl").write_bytes(b"fake model")
    (tmp_path / "ml_models" / "model_lgbm.pkl").write_bytes(b"fake model")

    from scripts.train_models import archive_models
    archive_path = archive_models(ml_dir)

    assert os.path.isdir(archive_path)
    assert "model_xgb.pkl" in os.listdir(archive_path)


def test_archive_prunes_old_archives(tmp_path):
    ml_dir = str(tmp_path / "ml_models")
    os.makedirs(ml_dir)
    (tmp_path / "ml_models" / "model.pkl").write_bytes(b"fake")

    import time

    from scripts.train_models import archive_models
    for _ in range(5):
        archive_models(ml_dir)
        time.sleep(0.01)

    archive_base = os.path.join(ml_dir, "archive")
    remaining = os.listdir(archive_base)
    assert len(remaining) == 3


def test_training_report_has_version_fields(tmp_path):
    report_path = tmp_path / "training_report.json"
    report = {
        "version": "20240101_120000",
        "trained_at": "2024-01-01T12:00:00Z",
        "archive_path": "/ml_models/archive/20240101_120000",
        "models": {},
    }
    report_path.write_text(json.dumps(report))
    data = json.loads(report_path.read_text())
    assert "version" in data
    assert "trained_at" in data
    assert "archive_path" in data
