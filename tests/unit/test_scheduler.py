"""tests/unit/test_scheduler.py — SchedulerManager testleri."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from wallet_manager.scheduler import SchedulerManager


class TestSchedulerManager:
    def _sched(self, tmp_path):
        return SchedulerManager(schedule_file=str(tmp_path / "schedule.json"))

    def test_new_wallet_is_due(self, tmp_path):
        sched = self._sched(tmp_path)
        assert sched.should_run_swap_this_cycle("new_wallet_001") is True

    def test_just_ran_not_due(self, tmp_path):
        sched = self._sched(tmp_path)
        sched.mark_ran("wallet_001")
        assert sched.should_run_swap_this_cycle("wallet_001") is False

    def test_old_run_is_due_again(self, tmp_path):
        sched = self._sched(tmp_path)
        sched._data["wallet_002"] = {"last_run": time.time() - 86400 * 8}  # 8 gün önce
        sched._save()
        sched2 = SchedulerManager(schedule_file=str(tmp_path / "schedule.json"))
        assert sched2.should_run_swap_this_cycle("wallet_002") is True

    def test_corrupted_json_graceful_recovery(self, tmp_path):
        schedule_file = tmp_path / "schedule.json"
        schedule_file.write_text("{not valid json[")
        sched = SchedulerManager(schedule_file=str(schedule_file))
        assert sched.should_run_swap_this_cycle("any_wallet") is True

    def test_gas_deferral_postpones_run(self, tmp_path):
        sched = self._sched(tmp_path)
        sched.defer_due_to_gas("wallet_003", defer_hours=2.0)
        assert sched.should_run_swap_this_cycle("wallet_003") is False
