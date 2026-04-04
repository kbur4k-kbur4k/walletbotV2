"""tests/unit/test_core_strategy.py

Comprehensive tests for wallet_manager/core/strategy.py (ExecutionStrategy).
"""
from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from wallet_manager.core.models import StrategyConfig
from wallet_manager.core.strategy import _ABSOLUTE_MIN_WEI, _MIN_DELAY_FLOOR, ExecutionStrategy
from wallet_manager.exceptions import StrategyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(**kwargs) -> ExecutionStrategy:
    cfg = StrategyConfig(**kwargs)
    return ExecutionStrategy(config=cfg)


# ---------------------------------------------------------------------------
# _validate_pct_range
# ---------------------------------------------------------------------------

class TestValidatePctRange:
    def test_valid_range(self):
        # Should not raise
        ExecutionStrategy._validate_pct_range(0.10, 0.20)

    def test_valid_equal(self):
        ExecutionStrategy._validate_pct_range(0.15, 0.15)

    def test_invalid_min_zero(self):
        with pytest.raises(StrategyError, match="min_pct"):
            ExecutionStrategy._validate_pct_range(0.0, 0.20)

    def test_invalid_min_negative(self):
        with pytest.raises(StrategyError, match="min_pct"):
            ExecutionStrategy._validate_pct_range(-0.1, 0.20)

    def test_invalid_min_above_one(self):
        with pytest.raises(StrategyError, match="min_pct"):
            ExecutionStrategy._validate_pct_range(1.1, 1.5)

    def test_invalid_max_zero(self):
        with pytest.raises(StrategyError, match="max_pct"):
            ExecutionStrategy._validate_pct_range(0.10, 0.0)

    def test_invalid_max_negative(self):
        with pytest.raises(StrategyError, match="max_pct"):
            ExecutionStrategy._validate_pct_range(0.10, -0.5)

    def test_invalid_max_above_one(self):
        with pytest.raises(StrategyError, match="max_pct"):
            ExecutionStrategy._validate_pct_range(0.10, 1.5)

    def test_inverted_min_gt_max(self):
        with pytest.raises(StrategyError, match="must not exceed"):
            ExecutionStrategy._validate_pct_range(0.20, 0.10)


# ---------------------------------------------------------------------------
# calculate_amount
# ---------------------------------------------------------------------------

class TestCalculateAmount:
    def test_normal_returns_in_valid_range(self):
        strategy = _make_strategy(min_pct=0.10, max_pct=0.20, amount_sigma_pct=0.0)
        balance = 10 ** 18  # 1 ETH in wei
        amount = strategy.calculate_amount(balance)
        assert _ABSOLUTE_MIN_WEI <= amount <= balance

    def test_zero_balance_raises(self):
        strategy = _make_strategy()
        with pytest.raises(StrategyError, match="zero or negative"):
            strategy.calculate_amount(0)

    def test_negative_balance_raises(self):
        strategy = _make_strategy()
        with pytest.raises(StrategyError, match="zero or negative"):
            strategy.calculate_amount(-100)

    def test_min_gt_max_raises(self):
        strategy = _make_strategy(min_pct=0.10, max_pct=0.20)
        with pytest.raises(StrategyError, match="must not exceed"):
            strategy.calculate_amount(10 ** 18, min_pct=0.30, max_pct=0.10)

    def test_override_pct_used(self):
        """Explicit min_pct/max_pct override config values."""
        strategy = _make_strategy(min_pct=0.01, max_pct=0.02, amount_sigma_pct=0.0)
        balance = 10 ** 18
        amount = strategy.calculate_amount(balance, min_pct=0.50, max_pct=0.50)
        # With 0 sigma, should be exactly 50% of balance
        assert amount == balance // 2

    def test_zero_sigma_no_jitter(self):
        """When sigma=0, jitter=0 and result equals base_amount clamped."""
        strategy = _make_strategy(min_pct=0.10, max_pct=0.10, amount_sigma_pct=0.0)
        balance = 1000
        amount = strategy.calculate_amount(balance)
        assert amount == 100  # exactly 10%

    def test_result_clamped_to_min_wei(self):
        """Very small balance should return at least _ABSOLUTE_MIN_WEI."""
        strategy = _make_strategy(min_pct=0.10, max_pct=0.10, amount_sigma_pct=0.0)
        # balance=5 → 10% = 0 (int), but should be clamped to 1
        amount = strategy.calculate_amount(5)
        assert amount >= _ABSOLUTE_MIN_WEI

    def test_result_never_exceeds_balance(self):
        """Jitter must not push amount above balance."""
        strategy = _make_strategy(min_pct=0.90, max_pct=0.95, amount_sigma_pct=0.50)
        balance = 10 ** 18
        for _ in range(50):
            amount = strategy.calculate_amount(balance)
            assert amount <= balance

    def test_sigma_positive_path(self):
        """When sigma > 0, the gauss path should execute without error."""
        strategy = _make_strategy(min_pct=0.10, max_pct=0.20, amount_sigma_pct=0.10)
        balance = 10 ** 18
        # Just call several times — no assertion needed beyond no exception
        for _ in range(20):
            result = strategy.calculate_amount(balance)
            assert _ABSOLUTE_MIN_WEI <= result <= balance


# ---------------------------------------------------------------------------
# jittered_delay
# ---------------------------------------------------------------------------

class TestJitteredDelay:
    def test_normal_delay_positive(self):
        strategy = _make_strategy(delay_sigma_seconds=0.5)
        for _ in range(20):
            d = strategy.jittered_delay(2.0)
            assert d >= _MIN_DELAY_FLOOR

    def test_sigma_zero_returns_base(self):
        strategy = _make_strategy(delay_sigma_seconds=0.0)
        d = strategy.jittered_delay(3.0)
        assert d == 3.0  # no jitter, so returns base directly

    def test_sigma_zero_small_base_clamped(self):
        strategy = _make_strategy(delay_sigma_seconds=0.0)
        d = strategy.jittered_delay(0.0)
        assert d == _MIN_DELAY_FLOOR

    def test_sigma_override_used(self):
        """sigma_override=0 should produce no jitter regardless of config."""
        strategy = _make_strategy(delay_sigma_seconds=10.0)
        d = strategy.jittered_delay(5.0, sigma_override=0.0)
        assert d == 5.0

    def test_sigma_override_positive(self):
        strategy = _make_strategy(delay_sigma_seconds=0.0)
        results = [strategy.jittered_delay(5.0, sigma_override=1.0) for _ in range(30)]
        assert all(r >= _MIN_DELAY_FLOOR for r in results)

    def test_floor_applied_on_high_negative_jitter(self):
        """Even with extreme negative jitter (mocked), floor should apply."""
        strategy = _make_strategy(delay_sigma_seconds=1.0)
        # Force gauss to return large negative via mock
        with patch("wallet_manager.core.strategy.random.gauss", return_value=-1e9):
            d = strategy.jittered_delay(1.0)
        assert d == _MIN_DELAY_FLOOR


# ---------------------------------------------------------------------------
# shuffle_tasks
# ---------------------------------------------------------------------------

class TestShuffleTasks:
    def test_empty_list_returns_empty(self):
        strategy = _make_strategy()
        assert strategy.shuffle_tasks([]) == []

    def test_single_item_returns_same(self):
        strategy = _make_strategy()
        result = strategy.shuffle_tasks(["only"])
        assert result == ["only"]

    def test_same_elements_returned(self):
        strategy = _make_strategy()
        tasks = ["a", "b", "c", "d"]
        result = strategy.shuffle_tasks(tasks)
        assert sorted(result) == sorted(tasks)

    def test_original_not_mutated(self):
        strategy = _make_strategy()
        tasks = ["x", "y", "z"]
        original = tasks[:]
        strategy.shuffle_tasks(tasks)
        assert tasks == original

    def test_returns_new_list(self):
        strategy = _make_strategy()
        tasks = [1, 2, 3]
        assert strategy.shuffle_tasks(tasks) is not tasks

    def test_multiple_items_shuffled_sometimes(self):
        """Run many times and confirm at least one different ordering."""
        strategy = _make_strategy()
        tasks = list(range(10))
        orders = {tuple(strategy.shuffle_tasks(tasks)) for _ in range(50)}
        # Very unlikely all 50 produce same order
        assert len(orders) > 1


# ---------------------------------------------------------------------------
# build_execution_plan
# ---------------------------------------------------------------------------

class TestBuildExecutionPlan:
    def _plan(self, shuffle_wallets: bool, shuffle_tasks: bool):
        cfg = StrategyConfig(shuffle_wallets=shuffle_wallets, shuffle_tasks=shuffle_tasks)
        return ExecutionStrategy(config=cfg)

    def test_no_shuffle_preserves_order(self):
        strategy = self._plan(shuffle_wallets=False, shuffle_tasks=False)
        wallet_ids = ["w1", "w2", "w3"]
        tasks = {"w1": ["a", "b"], "w2": ["c"], "w3": ["d", "e"]}
        plan = strategy.build_execution_plan(wallet_ids, tasks)
        assert [wid for wid, _ in plan] == wallet_ids
        for wid, t in plan:
            assert t == tasks[wid]

    def test_shuffle_wallets_true(self):
        strategy = self._plan(shuffle_wallets=True, shuffle_tasks=False)
        wallet_ids = list(range(20))
        tasks = {w: [w] for w in wallet_ids}
        plan = strategy.build_execution_plan(wallet_ids, tasks)
        assert sorted(wid for wid, _ in plan) == sorted(wallet_ids)

    def test_shuffle_tasks_true(self):
        strategy = self._plan(shuffle_wallets=False, shuffle_tasks=True)
        wallet_ids = ["w1"]
        tasks = {"w1": list(range(10))}
        results = [
            tuple(t for _, ts in strategy.build_execution_plan(wallet_ids, tasks) for t in ts)
            for _ in range(20)
        ]
        assert len(set(results)) > 1  # at least some variation

    def test_missing_wallet_in_tasks_uses_empty(self):
        strategy = self._plan(shuffle_wallets=False, shuffle_tasks=False)
        wallet_ids = ["w1", "ghost"]
        tasks = {"w1": ["x"]}
        plan = strategy.build_execution_plan(wallet_ids, tasks)
        wids = dict(plan)
        assert wids["ghost"] == []

    def test_both_shuffle_true(self):
        strategy = self._plan(shuffle_wallets=True, shuffle_tasks=True)
        wallet_ids = ["a", "b", "c"]
        tasks_map = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7, 8, 9]}
        plan = strategy.build_execution_plan(wallet_ids, tasks_map)
        assert len(plan) == 3
        for wid, t in plan:
            assert sorted(t) == sorted(tasks_map[wid])

    def test_empty_wallet_ids(self):
        strategy = self._plan(shuffle_wallets=False, shuffle_tasks=False)
        plan = strategy.build_execution_plan([], {})
        assert plan == []

    def test_default_config(self):
        """ExecutionStrategy() with no config should use StrategyConfig defaults."""
        strategy = ExecutionStrategy()
        plan = strategy.build_execution_plan(["w1"], {"w1": ["t1"]})
        assert len(plan) == 1
