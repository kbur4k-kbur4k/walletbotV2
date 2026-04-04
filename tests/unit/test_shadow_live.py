from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from wallet_manager.strategies.shadow_live import (
    SHADOW_GATE_MIN_SAMPLES,
    ShadowLiveEngine,
)


@pytest.mark.asyncio
async def test_get_shadow_quote_buy_returns_effective_price() -> None:
    engine = ShadowLiveEngine()

    with patch(
        "wallet_manager.actions.quote_amount_out",
        new=AsyncMock(return_value=5 * 10**15),
    ):
        price = await engine.get_shadow_quote(
            rpc_url="https://base.example",
            direction="BUY",
            size_usd=10.0,
            eth_price=2000.0,
        )

    assert price == pytest.approx(2000.0)
    assert engine.to_dict()["n_queries_total"] == 1


@pytest.mark.asyncio
async def test_get_shadow_quote_sell_returns_effective_price() -> None:
    engine = ShadowLiveEngine()

    with patch(
        "wallet_manager.actions.quote_amount_out",
        new=AsyncMock(return_value=10_000_000),
    ):
        price = await engine.get_shadow_quote(
            rpc_url="https://base.example",
            direction="SELL",
            size_usd=20.0,
            eth_price=2000.0,
        )

    assert price == pytest.approx(1000.0)
    assert engine.to_dict()["n_queries_total"] == 1


def test_readiness_gate_and_snapshot() -> None:
    engine = ShadowLiveEngine()

    for _ in range(SHADOW_GATE_MIN_SAMPLES):
        engine.record_comparison(
            direction="BUY",
            size_usd=25.0,
            paper_fill_price=2000.0,
            shadow_fill_price=2001.0,
            rpc_latency_ms=12.0,
        )

    readiness = engine.get_readiness_gate()
    snapshot = engine.to_dict()

    assert readiness["ready"] is True
    assert readiness["status"] == "READY"
    assert len(snapshot["recent_comparisons"]) == 10
