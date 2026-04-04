"""Tests for backtesting engine (Task 3.1)."""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta

import pytest

# Add scripts dir to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import backtest


def test_compute_sharpe_positive():
    returns = [0.001 + (i % 5) * 0.001 for i in range(252)]
    sharpe = backtest.compute_sharpe(returns)
    assert sharpe > 0.0


def test_compute_sharpe_zero_std():
    returns = [0.0] * 50
    assert backtest.compute_sharpe(returns) == 0.0


def test_compute_max_drawdown():
    equity = [100.0, 110.0, 90.0, 95.0, 100.0]
    dd = backtest.compute_max_drawdown(equity)
    expected = 20.0 / 110.0
    assert abs(dd - expected) < 0.001


def test_compute_win_rate():
    trades = [{"pnl_usd": 5.0}, {"pnl_usd": -2.0}, {"pnl_usd": 3.0}, {"pnl_usd": -1.0}]
    assert backtest.compute_win_rate(trades) == 0.5


def test_simulate_trades_returns_correct_types():
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    trades, equity = backtest.simulate_trades("ml_quant", start, end, 1000.0)
    assert isinstance(trades, list)
    assert isinstance(equity, list)
    assert len(equity) >= 30


def test_output_json_schema(tmp_path):
    end = datetime.utcnow()
    start = end - timedelta(days=10)
    trades, equity_curve = backtest.simulate_trades("legacy", start, end, 500.0)

    daily_returns = [
        (equity_curve[i + 1] - equity_curve[i]) / equity_curve[i]
        for i in range(len(equity_curve) - 1)
        if equity_curve[i] > 0
    ]

    output = {
        "metrics": {
            "mode": "legacy",
            "annualised_sharpe": backtest.compute_sharpe(daily_returns),
            "max_drawdown_pct": backtest.compute_max_drawdown(equity_curve) * 100,
        },
        "trades": trades,
        "equity_curve": equity_curve,
    }
    out_path = str(tmp_path / "test_result.json")
    with open(out_path, "w") as f:
        json.dump(output, f)

    with open(out_path) as f:
        data = json.load(f)
    assert "metrics" in data
    assert "trades" in data
    assert "equity_curve" in data
