"""tests/unit/test_smart_order_router.py - SmartOrderRouter tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.smart_order_router import RoutePlan, SmartOrderRouter


class TestSmartOrderRouter:
    def _router(self) -> SmartOrderRouter:
        return SmartOrderRouter()

    @pytest.mark.asyncio
    async def test_get_best_route_returns_plan(self) -> None:
        router = self._router()
        with patch.object(router, "_quote_uniswap_v3", new=AsyncMock(return_value=int(1800e6))), patch.object(
            router,
            "_quote_sushiswap",
            new=AsyncMock(return_value=int(1790e6)),
        ):
            plan = await router.get_best_route(
                from_token="ETH",
                to_token="USDC",
                amount_in_wei=int(1e18),
                rpc_url="https://mainnet.base.org",
            )
        assert isinstance(plan, RoutePlan)
        assert plan.expected_out > 0

    @pytest.mark.asyncio
    async def test_router_selects_highest_output(self) -> None:
        """Router should select the DEX with the highest output quote."""
        router = self._router()
        with patch.object(router, "_quote_uniswap_v3", new=AsyncMock(return_value=int(1850e6))), patch.object(
            router,
            "_quote_sushiswap",
            new=AsyncMock(return_value=int(1700e6)),
        ):
            plan = await router.get_best_route("ETH", "USDC", int(1e18), "https://rpc")
        assert plan.dex == "uniswap_v3"

    @pytest.mark.asyncio
    async def test_all_quotes_fail_raises(self) -> None:
        router = self._router()
        with patch.object(
            router,
            "_quote_uniswap_v3",
            new=AsyncMock(side_effect=RuntimeError("RPC fail")),
        ), patch.object(

            router,
            "_quote_sushiswap",
            new=AsyncMock(side_effect=RuntimeError("RPC fail")),
        ), pytest.raises(RuntimeError, match="All DEX quotes failed"):
            await router.get_best_route("ETH", "USDC", int(1e18), "https://rpc")

    @pytest.mark.asyncio
    async def test_find_best_route_caching(self):
        router = self._router()
        router._route_cache["ETH:USDC:1000"] = MagicMock(best_quote=MagicMock(timestamp=9999999999))
        route = await router.find_best_route("http", "ETH", "USDC", 1000)
        assert route is not None

        router.clear_cache()
        assert len(router._route_cache) == 0

    @pytest.mark.asyncio
    async def test_find_best_route_parallel_quotes(self):
        from wallet_manager.strategies.smart_order_router import DEXVenue, RouteQuote
        router = self._router()
        mock_aerodrome = RouteQuote(DEXVenue.AERODROME, "ETH", "USDC", 1000, 2000, 30, 200000, 30.0, 999)
        mock_uni = RouteQuote(DEXVenue.UNISWAP_V3, "ETH", "USDC", 1000, 1900, 500, 200000, 5.0, 999)
        with patch("wallet_manager.strategies.smart_order_router.DEX_REGISTRY"), \
             patch("wallet_manager.strategies.smart_order_router._quote_aerodrome", return_value=mock_aerodrome), \
             patch("wallet_manager.strategies.smart_order_router._quote_uniswap_v3", return_value=mock_uni), \
             patch("wallet_manager.strategies.smart_order_router._quote_sushiswap_v3", return_value=None):
            route = await router.find_best_route("http", "ETH", "USDC", 1000)
            assert route is not None
            assert len(route.all_quotes) > 0

class TestTWAPExecutionScheduler:
    def test_schedule_generation(self):
        from wallet_manager.strategies.smart_order_router import TWAPExecutionScheduler
        scheduler = TWAPExecutionScheduler(n_slices=5, interval_secs=60, jitter_pct=0.1)
        slices = scheduler.schedule(1005)

        assert len(slices) == 5
        assert sum(s["amount_in"] for s in slices) == 1005
        assert slices[0]["amount_in"] == 201
        assert slices[4]["amount_in"] == 201  # Exact division

    def test_get_due_slice(self):
        import time

        from wallet_manager.strategies.smart_order_router import TWAPExecutionScheduler
        scheduler = TWAPExecutionScheduler()
        slices = scheduler.schedule(1000, start_time=time.time() - 100)

        # At least one should be due
        due = scheduler.get_due_slice(slices)
        assert due is not None

        scheduler.mark_executed(due)
        assert due["executed"] is True

class TestAdaptiveSplitRatio:
    def test_adaptive_split(self):
        from wallet_manager.strategies.smart_order_router import AdaptiveSplitRatio
        adapt = AdaptiveSplitRatio()
        adapt.update_venue("v1", 1000, 1000, 2.0)
        adapt.update_venue("v2", 500, 500, 1.0)

        split = adapt.get_split()
        assert "v1" in split
        assert "v2" in split
        # v1 score: 2000 / 2 = 1000
        # v2 score: 1000 / 1 = 1000
        # total = 2000, weights should be 0.5 and 0.5
        assert split["v1"] == 0.5
        assert split["v2"] == 0.5

class TestExecutionShortfallTracker:
    def test_record_shortfall(self):
        from wallet_manager.strategies.smart_order_router import ExecutionShortfallTracker
        tracker = ExecutionShortfallTracker()

        # Buy: 2000 -> 2005 (paid more, adverse = 5/2000 = 0.0025)
        res = tracker.record(2000, 2005, side="buy")
        assert res == 0.0025

        # Sell: 2000 -> 1995 (received less, adverse = 5/2000 = 0.0025)
        res2 = tracker.record(2000, 1995, side="sell")
        assert res2 == 0.0025

        assert tracker.trade_count == 2
        assert tracker.mean_is == 0.0025
        assert tracker.median_is == 0.0025
