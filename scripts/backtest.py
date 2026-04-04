#!/usr/bin/env python3
"""
scripts/backtest.py — Walletbot backtesting engine.

Usage:
    python scripts/backtest.py [OPTIONS]
      --mode          legacy|quantx|ml_quant  (default: ml_quant)
      --start-date    YYYY-MM-DD
      --end-date      YYYY-MM-DD
      --initial-usd   float                   (default: 1000.0)
      --output        path
      --db-path       path to analytics SQLite DB (default: from Settings)
      --plot          flag

Data source priority
--------------------
1. Real trade data from the SQLite analytics DB (``AnalyticsDB``).
2. Synthetic Gaussian simulation (fallback when DB is empty or unavailable).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any

# Ensure the package is importable when run from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walletbot backtesting engine")
    p.add_argument("--mode", default="ml_quant", choices=["legacy", "quantx", "ml_quant"])
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--initial-usd", type=float, default=1000.0)
    p.add_argument("--output", default=None)
    p.add_argument("--db-path", default=None, help="Path to analytics SQLite DB")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def compute_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from a list of daily returns."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0.0:
        return 0.0
    daily_sharpe = (mean_r - risk_free) / std_r
    return daily_sharpe * math.sqrt(252)


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown as a fraction [0, 1]."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_win_rate(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
    return wins / len(trades)


def compute_profit_factor(trades: list[dict[str, Any]]) -> float:
    gross_profit = sum(t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) > 0)
    gross_loss = abs(sum(t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


async def _load_from_analytics_db(
    db_path: str | None,
    mode: str,
    start_date: datetime,
    end_date: datetime,
    initial_usd: float,
) -> tuple[list[dict[str, Any]], list[float]] | None:
    """Load real trade data from the SQLite analytics DB.

    Returns ``(trades, equity_curve)`` built from on-chain transaction records,
    or ``None`` if the DB is unavailable or contains no matching rows.
    """
    try:
        from wallet_manager.resilience.analytics import AnalyticsDB
    except ImportError:
        return None

    db = AnalyticsDB(db_path=db_path) if db_path else AnalyticsDB()
    try:
        await db.open()

        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()

        # Fetch all mined transactions in the date range from the DB
        sql = """
            SELECT strategy_type,
                   submitted_at,
                   mined_at,
                   gas_used,
                   gas_price_gwei,
                   CAST(value_wei AS REAL) AS value_wei
            FROM transactions
            WHERE status = 'MINED'
              AND submitted_at >= ?
              AND submitted_at <= ?
            ORDER BY submitted_at ASC
        """
        async with db._db.execute(sql, (start_ts, end_ts)) as cur:
            rows = await cur.fetchall()

        if not rows:
            return None

        # Build synthetic equity curve from gas cost + value movements.
        # PnL proxy: value_wei converted to approximate USD at ~3000 USD/ETH.
        # Gas cost is subtracted as a direct USD expense.
        ETH_PRICE_USD = 3000.0
        WEI_TO_ETH = 1e-18
        GWEI_TO_ETH = 1e-9

        trades: list[dict[str, Any]] = []
        equity = initial_usd
        equity_curve = [equity]

        # Group rows into daily buckets for the equity curve
        day_index: dict[str, list[dict[str, Any]]] = {}
        days = (end_date - start_date).days or 1
        for i in range(days):
            day = (start_date + timedelta(days=i)).date().isoformat()
            day_index[day] = []

        for row in rows:
            submitted_at = row["submitted_at"]
            day_str = datetime.utcfromtimestamp(submitted_at).date().isoformat()
            gas_cost_usd = (
                (row["gas_used"] or 0) * (row["gas_price_gwei"] or 0) * GWEI_TO_ETH * ETH_PRICE_USD
            )
            value_usd = (row["value_wei"] or 0) * WEI_TO_ETH * ETH_PRICE_USD
            # Treat value_wei outflows as costs; gas is always a cost
            pnl_usd = round(-gas_cost_usd, 4)  # net per-tx PnL approximation
            trade = {
                "date": day_str,
                "mode": mode,
                "pillar": str(row["strategy_type"] or "UNKNOWN").upper(),
                "pnl_usd": pnl_usd,
                "value_usd": round(value_usd, 4),
                "gas_cost_usd": round(gas_cost_usd, 4),
                "duration_hours": 0.0,
                "source": "analytics_db",
            }
            if mined := row["mined_at"]:
                duration_h = (mined - submitted_at) / 3600.0
                trade["duration_hours"] = round(duration_h, 4)
            trades.append(trade)
            if day_str in day_index:
                day_index[day_str].append(trade)

        # Rebuild equity curve day-by-day
        equity_curve = [initial_usd]
        for i in range(days):
            day = (start_date + timedelta(days=i)).date().isoformat()
            day_trades = day_index.get(day, [])
            for t in day_trades:
                equity += t["pnl_usd"]
            equity_curve.append(max(equity, 0.01))

        return trades, equity_curve

    except Exception:
        return None
    finally:
        try:
            await db.close()
        except Exception:
            pass


def _synthetic_trades(
    mode: str,
    start_date: datetime,
    end_date: datetime,
    initial_usd: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Gaussian synthetic trade simulation — used only when DB has no data."""
    import random
    random.seed(42)

    days = (end_date - start_date).days
    equity = initial_usd
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []

    for i in range(max(days, 1)):
        day = start_date + timedelta(days=i)
        n_trades = random.randint(0, 3)
        for _ in range(n_trades):
            pnl = random.gauss(0.5, 8.0)
            equity += pnl
            trades.append({
                "date": day.isoformat(),
                "mode": mode,
                "pillar": random.choice(["YIELD", "ALGO", "SHORT", "AIRDROP"]),
                "pnl_usd": round(pnl, 4),
                "duration_hours": round(abs(random.gauss(4.0, 2.0)), 2),
                "source": "synthetic",
            })
        equity_curve.append(max(equity, 0.01))

    return trades, equity_curve


def load_trades(
    mode: str,
    start_date: datetime,
    end_date: datetime,
    initial_usd: float,
    db_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[float], str]:
    """Load trade data from the analytics DB, falling back to synthetic data.

    Returns:
        ``(trades, equity_curve, source)`` where source is ``"analytics_db"``
        or ``"synthetic"``.
    """
    try:
        result = asyncio.run(
            _load_from_analytics_db(db_path, mode, start_date, end_date, initial_usd)
        )
        if result is not None:
            trades, equity_curve = result
            if trades:
                return trades, equity_curve, "analytics_db"
    except Exception:
        pass

    trades, equity_curve = _synthetic_trades(mode, start_date, end_date, initial_usd)
    return trades, equity_curve, "synthetic"


def simulate_trades(
    mode: str,
    start_date: datetime,
    end_date: datetime,
    initial_usd: float,
    db_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Compatibility wrapper used by tests and older callers."""
    trades, equity_curve, _ = load_trades(
        mode,
        start_date,
        end_date,
        initial_usd,
        db_path=db_path,
    )
    return trades, equity_curve


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 52)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 52)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<32} {v:>10.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:<28} {vv:>10.4f}")
        else:
            print(f"  {k:<32} {str(v):>10}")
    print("=" * 52 + "\n")


def main() -> None:
    args = parse_args()

    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.utcnow()
    )
    start_date = (
        datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.start_date
        else end_date - timedelta(days=90)
    )

    trades, equity_curve, data_source = load_trades(
        args.mode, start_date, end_date, args.initial_usd,
        db_path=getattr(args, "db_path", None),
    )
    print(f"Data source: {data_source} ({len(trades)} trades)")

    daily_returns = [
        (equity_curve[i + 1] - equity_curve[i]) / equity_curve[i]
        for i in range(len(equity_curve) - 1)
        if equity_curve[i] > 0
    ]

    final_equity = equity_curve[-1] if equity_curve else args.initial_usd
    total_return_pct = (final_equity - args.initial_usd) / args.initial_usd * 100

    pillar_pnl: dict[str, float] = {}
    for t in trades:
        p = t.get("pillar", "UNKNOWN")
        pillar_pnl[p] = round(pillar_pnl.get(p, 0.0) + t.get("pnl_usd", 0.0), 4)

    avg_duration = (
        round(sum(t.get("duration_hours", 0) for t in trades) / len(trades), 4) if trades else 0.0
    )

    metrics: dict[str, Any] = {
        "mode": args.mode,
        "data_source": data_source,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "initial_usd": args.initial_usd,
        "final_usd": round(final_equity, 4),
        "total_return_pct": round(total_return_pct, 4),
        "annualised_sharpe": round(compute_sharpe(daily_returns), 4),
        "max_drawdown_pct": round(compute_max_drawdown(equity_curve) * 100, 4),
        "win_rate_pct": round(compute_win_rate(trades) * 100, 4),
        "profit_factor": round(compute_profit_factor(trades), 4),
        "total_trades": len(trades),
        "avg_trade_duration_h": avg_duration,
        "pillar_pnl": pillar_pnl,
    }

    print_summary(metrics)

    output_path = args.output or (
        "results/backtest_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "trades": trades, "equity_curve": equity_curve}, f, indent=2)
    print(f"Results saved to {output_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve)
            plt.title(f"Equity Curve — {args.mode} mode")
            plt.xlabel("Days")
            plt.ylabel("Equity (USD)")
            plot_path = output_path.replace(".json", ".png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Equity curve saved to {plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
