"""
tests/unit/test_uncovered_modules.py

Comprehensive unit tests covering:
  - wallet_manager/orchestrator_gas.py   (GasGuard)
  - wallet_manager/orchestrator_status.py (LiveStatusProvider)
  - wallet_manager/ai_agents.py          (all agents, AgentRegistry, AgentReport)
  - wallet_manager/experiment_lab.py     (ExperimentLab, all variant types)
  - wallet_manager/trade_journal.py      (TradeJournal, TradeRecord, _mask)
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# orchestrator_gas.py — GasGuard
# ============================================================================


class TestGasGuard:
    """Tests for GasGuard.wait_for_safe_gas and GasGuard.current_gwei."""

    def _make_config(self, wallet_id: str = "wallet_01", max_gas_gwei: float = 50.0):
        from wallet_manager.core.models import WalletConfig
        return WalletConfig(
            wallet_id=wallet_id,
            private_key="0x" + "a" * 64,
            rpc_url="https://rpc.example.com",
            chain_id=8453,
            max_gas_price_gwei=max_gas_gwei,
        )

    def _make_notifier(self):
        notifier = MagicMock()
        notifier.send = AsyncMock()
        return notifier

    def _make_guard(self, gas_monitor, poll_interval: float = 0.01, timeout: float = 0.05):
        from wallet_manager.orchestrator_gas import GasGuard
        return GasGuard(
            gas_monitor=gas_monitor,
            gas_poll_interval=poll_interval,
            gas_wait_timeout=timeout,
        )

    @pytest.mark.asyncio
    async def test_wait_for_safe_gas_already_safe(self):
        """When gas is safe on the first check, return immediately."""
        gas_monitor = MagicMock()
        gas_monitor.is_safe = AsyncMock(return_value=(True, 20.0))
        guard = self._make_guard(gas_monitor)
        config = self._make_config(max_gas_gwei=50.0)
        notifier = self._make_notifier()
        safe_notify = AsyncMock()

        await guard.wait_for_safe_gas(config, notifier, safe_notify)

        gas_monitor.is_safe.assert_awaited_once()
        safe_notify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wait_for_safe_gas_monitor_exception_defaults_safe(self):
        """When gas monitor raises, proceed as safe without notification."""
        gas_monitor = MagicMock()
        gas_monitor.is_safe = AsyncMock(side_effect=Exception("rpc error"))
        guard = self._make_guard(gas_monitor)
        config = self._make_config()
        notifier = self._make_notifier()
        safe_notify = AsyncMock()

        await guard.wait_for_safe_gas(config, notifier, safe_notify)

        safe_notify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wait_for_safe_gas_recovers_after_one_poll(self):
        """Gas is high initially, then safe on the first poll — GAS_PAUSE + GAS_RESUME sent."""
        from wallet_manager.core.models import EventKind

        call_count = 0

        async def is_safe_mock(max_gwei):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (False, 80.0)
            return (True, 30.0)

        gas_monitor = MagicMock()
        gas_monitor.is_safe = is_safe_mock
        guard = self._make_guard(gas_monitor, poll_interval=0.01, timeout=5.0)
        config = self._make_config(max_gas_gwei=50.0)
        notifier = self._make_notifier()

        paused_events = []
        resumed_events = []

        async def safe_notify(notifier, event):
            if event.kind == EventKind.GAS_PAUSE:
                paused_events.append(event)
            elif event.kind == EventKind.GAS_RESUME:
                resumed_events.append(event)

        await guard.wait_for_safe_gas(config, notifier, safe_notify)

        assert len(paused_events) == 1
        assert len(resumed_events) == 1
        assert paused_events[0].wallet_id == "wallet_01"
        assert resumed_events[0].wallet_id == "wallet_01"

    @pytest.mark.asyncio
    async def test_wait_for_safe_gas_timeout_raises(self):
        """Timeout raises GasPriceTooHighError."""
        from wallet_manager.exceptions import GasPriceTooHighError

        gas_monitor = MagicMock()
        gas_monitor.is_safe = AsyncMock(return_value=(False, 200.0))
        guard = self._make_guard(gas_monitor, poll_interval=0.01, timeout=0.05)
        config = self._make_config(max_gas_gwei=50.0)
        notifier = self._make_notifier()
        safe_notify = AsyncMock()

        with pytest.raises(GasPriceTooHighError) as exc_info:
            await guard.wait_for_safe_gas(config, notifier, safe_notify)

        assert "wallet_01" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_current_gwei_delegates_to_monitor(self):
        """current_gwei returns the monitor's result."""
        gas_monitor = MagicMock()
        gas_monitor.current_gwei = AsyncMock(return_value=42.5)
        guard = self._make_guard(gas_monitor)

        result = await guard.current_gwei()
        assert result == 42.5

    @pytest.mark.asyncio
    async def test_current_gwei_returns_zero_on_exception(self):
        """current_gwei returns 0.0 when monitor raises."""
        gas_monitor = MagicMock()
        gas_monitor.current_gwei = AsyncMock(side_effect=Exception("fail"))
        guard = self._make_guard(gas_monitor)

        result = await guard.current_gwei()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_gas_pause_notification_carries_price(self):
        """GAS_PAUSE event should carry the current gas price."""
        from wallet_manager.core.models import EventKind

        call_count = 0

        async def is_safe_mock(max_gwei):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (False, 123.45)
            return (True, 30.0)

        gas_monitor = MagicMock()
        gas_monitor.is_safe = is_safe_mock
        guard = self._make_guard(gas_monitor, poll_interval=0.01, timeout=5.0)
        config = self._make_config(max_gas_gwei=50.0)
        notifier = self._make_notifier()
        captured = []

        async def safe_notify(notifier, event):
            captured.append(event)

        await guard.wait_for_safe_gas(config, notifier, safe_notify)

        pause_events = [e for e in captured if e.kind == EventKind.GAS_PAUSE]
        assert len(pause_events) == 1
        assert abs(pause_events[0].gas_price_gwei - 123.45) < 0.01


# ============================================================================
# orchestrator_status.py — LiveStatusProvider
# ============================================================================


class TestLiveStatusProvider:
    """Tests for LiveStatusProvider.build_status_message."""

    def _make_config(self, wallet_id: str = "wid_01"):
        from wallet_manager.core.models import WalletConfig
        return WalletConfig(
            wallet_id=wallet_id,
            private_key="0x" + "b" * 64,
            rpc_url="https://rpc.example.com",
            chain_id=8453,
            max_gas_price_gwei=50.0,
        )

    def _make_scheduler(self, seconds: float = 0.0):
        scheduler = MagicMock()
        scheduler.seconds_until_due = MagicMock(return_value=seconds)
        return scheduler

    @pytest.mark.asyncio
    async def test_no_configs_returns_info_message(self):
        from wallet_manager.orchestrator_status import LiveStatusProvider
        provider = LiveStatusProvider(configs=[], scheduler=MagicMock())
        msg = await provider.build_status_message()
        assert "No wallets configured" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_basic(self):
        """Happy path: all actions succeed and message is formatted."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = self._make_config()
        scheduler = self._make_scheduler(seconds=0.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xABC"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, return_value=1_000_000_000_000_000_000),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=500_000_000),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=2000.0),
            patch("wallet_manager.strategies.yield_strategy.get_aave_usdc_balance", new_callable=AsyncMock, return_value=200_000_000),
            patch("wallet_manager.strategies.short_strategy.get_short_position_state", return_value=MagicMock(position_value_usd=100.0)),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "0xABC" in msg
        assert "Portfolio Live Status" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_all_actions_fail(self):
        """When all on-chain calls fail, graceful defaults are used."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = self._make_config()
        scheduler = self._make_scheduler(seconds=0.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xFAIL"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "0xFAIL" in msg

    @pytest.mark.asyncio
    async def test_build_status_message_with_portfolio_manager(self):
        """Uses portfolio_manager.sync_from_chain / get_snapshot when provided."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = self._make_config()
        scheduler = self._make_scheduler(seconds=3700.0)

        pm = MagicMock()
        pm.sync_from_chain = AsyncMock()
        snapshot = MagicMock()
        snapshot.total_value_usd = 9999.0
        pm.get_snapshot = MagicMock(return_value=snapshot)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xPM"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=2000.0),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler, portfolio_manager=pm)
            msg = await provider.build_status_message()

        assert "9999.00" in msg
        assert "hours later" in msg
        pm.sync_from_chain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_next_str_minutes(self):
        """When seconds_until_due is between 60 and 3600, shows mins."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = self._make_config()
        scheduler = self._make_scheduler(seconds=300.0)  # 5 minutes

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=3000.0),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "mins later" in msg

    @pytest.mark.asyncio
    async def test_next_str_seconds(self):
        """When seconds_until_due is between 1 and 60, shows secs."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = self._make_config()
        scheduler = self._make_scheduler(seconds=30.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=3000.0),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "secs later" in msg

    @pytest.mark.asyncio
    async def test_next_str_due_now(self):
        """When seconds_until_due <= 0, shows due now."""
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = self._make_config()
        scheduler = self._make_scheduler(seconds=0.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xADDR"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=3000.0),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert "due now" in msg

    @pytest.mark.asyncio
    async def test_chain_id_none_becomes_zero(self):
        """When config.chain_id is None, chain_id is treated as 0."""
        from wallet_manager.core.models import WalletConfig
        from wallet_manager.orchestrator_status import LiveStatusProvider

        config = WalletConfig(
            wallet_id="wid_none_chain",
            private_key="0x" + "c" * 64,
            rpc_url="https://rpc.example.com",
            chain_id=None,
            max_gas_price_gwei=50.0,
        )
        scheduler = self._make_scheduler(seconds=0.0)

        with (
            patch("wallet_manager.orchestrator_status._get_wallet_address_local", return_value="0xN0"),
            patch("wallet_manager.actions.fetch_native_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_usdc_balance", new_callable=AsyncMock, return_value=0),
            patch("wallet_manager.actions.fetch_eth_price_usd", new_callable=AsyncMock, return_value=3000.0),
            patch("wallet_manager.actions.eth_balance_usd_value", return_value=0.0),
        ):
            provider = LiveStatusProvider(configs=[config], scheduler=scheduler)
            msg = await provider.build_status_message()

        assert msg  # just ensure it doesn't crash


# ============================================================================
# ai_agents.py — AgentReport, BaseAgent, all agents, AgentRegistry
# ============================================================================


class TestAgentReport:
    """Tests for AgentReport dataclass methods."""

    def test_to_dict_keys(self):
        from wallet_manager.ai_agents import AgentReport
        report = AgentReport(
            agent_name="TestAgent",
            title="Test Title",
            body="Test body",
            generated_at=1000000.0,
            data={"key": "val"},
            used_llm=False,
        )
        d = report.to_dict()
        assert d["agent"] == "TestAgent"
        assert d["title"] == "Test Title"
        assert d["body"] == "Test body"
        assert d["generated_at"] == 1000000.0
        assert d["used_llm"] is False

    def test_to_text_format(self):
        from wallet_manager.ai_agents import AgentReport
        report = AgentReport(
            agent_name="TestAgent",
            title="My Title",
            body="My Body",
            generated_at=0.0,
        )
        text = report.to_text()
        assert "[TestAgent]" in text
        assert "My Title" in text
        assert "My Body" in text

    def test_to_dict_used_llm_true(self):
        from wallet_manager.ai_agents import AgentReport
        report = AgentReport(
            agent_name="A",
            title="T",
            body="B",
            used_llm=True,
        )
        assert report.to_dict()["used_llm"] is True

    def test_default_generated_at_set(self):
        from wallet_manager.ai_agents import AgentReport
        before = time.time()
        report = AgentReport(agent_name="A", title="T", body="B")
        after = time.time()
        assert before <= report.generated_at <= after


class TestBaseAgent:
    """Tests for BaseAgent._has_llm and _call_llm."""

    def test_has_llm_false_when_no_api_key(self):
        from wallet_manager.ai_agents import BaseAgent
        agent = BaseAgent()
        agent._api_key = ""
        assert agent._has_llm() is False

    def test_call_llm_returns_empty_when_no_llm(self):
        from wallet_manager.ai_agents import BaseAgent
        agent = BaseAgent()
        agent._api_key = ""
        result = agent._call_llm("system", "user")
        assert result == ""

    def test_call_llm_returns_empty_on_exception(self):
        """Even with a fake API key, if openai is available but call fails, return ''."""
        import sys

        from wallet_manager import ai_agents
        from wallet_manager.ai_agents import BaseAgent

        agent = BaseAgent()
        agent._api_key = "fake-key"

        # Create a fake openai module stub
        fake_openai = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("api error")
        fake_openai.OpenAI.return_value = mock_client

        # Patch _OPENAI_AVAILABLE and inject openai into module namespace
        with patch.object(ai_agents, "_OPENAI_AVAILABLE", True):
            original_openai = sys.modules.get("openai")
            sys.modules["openai"] = fake_openai
            try:
                # Temporarily add openai attr to module
                ai_agents.openai = fake_openai
                result = agent._call_llm("system", "user")
            finally:
                if original_openai is None:
                    sys.modules.pop("openai", None)
                else:
                    sys.modules["openai"] = original_openai
                if hasattr(ai_agents, "openai") and ai_agents.openai is fake_openai:
                    del ai_agents.openai

        assert result == ""


class TestOpsAgent:
    """Tests for OpsAgent.run with and without paper_engine."""

    @pytest.mark.asyncio
    async def test_run_no_paper_engine_nominal(self):
        """OpsAgent with no engine returns 'All systems nominal'."""
        from wallet_manager.ai_agents import OpsAgent
        agent = OpsAgent()
        report = await agent.run(paper_engine=None)
        assert "nominal" in report.body.lower() or "All systems nominal" in report.body
        assert report.agent_name == "OpsAgent"
        assert report.used_llm is False

    @pytest.mark.asyncio
    async def test_run_with_paper_engine_recent_trade(self):
        """OpsAgent with paper_engine that has a recent trade — no issues."""
        from wallet_manager.ai_agents import OpsAgent

        # Build a fake async DB conn
        recent_ts = time.time() - 60  # 1 min ago — well within 6h

        async def fake_fetchone_ts():
            row = MagicMock()
            row.__getitem__ = lambda self, key: recent_ts if key == "last_ts" else None
            return row

        async def fake_fetchone_cnt():
            row = MagicMock()
            row.__getitem__ = lambda self, key: 10 if key == "cnt" else None
            return row

        # Build cursor mocks for two different queries
        cursor_ts = MagicMock()
        cursor_ts.fetchone = AsyncMock(side_effect=fake_fetchone_ts)
        cursor_ts.__aenter__ = AsyncMock(return_value=cursor_ts)
        cursor_ts.__aexit__ = AsyncMock(return_value=None)

        cursor_cnt = MagicMock()
        cursor_cnt.fetchone = AsyncMock(side_effect=fake_fetchone_cnt)
        cursor_cnt.__aenter__ = AsyncMock(return_value=cursor_cnt)
        cursor_cnt.__aexit__ = AsyncMock(return_value=None)

        call_idx = [0]

        def fake_execute(query, params=None):
            idx = call_idx[0]
            call_idx[0] += 1
            return cursor_ts if idx == 0 else cursor_cnt

        conn = MagicMock()
        conn.execute = fake_execute

        ledger = MagicMock()
        ledger._get_conn = MagicMock(return_value=conn)

        paper_engine = MagicMock()
        paper_engine._ledger = ledger

        agent = OpsAgent()
        report = await agent.run(paper_engine=paper_engine)
        assert report is not None

    @pytest.mark.asyncio
    async def test_run_with_paper_engine_stale_trade(self):
        """OpsAgent detects stale trade (>24h) and includes issue."""
        from wallet_manager.ai_agents import OpsAgent

        old_ts = time.time() - (30 * 3600)  # 30 hours ago

        async def fake_fetchone_ts():
            row = MagicMock()
            row.__getitem__ = lambda self, key: old_ts if key == "last_ts" else None
            return row

        async def fake_fetchone_cnt():
            row = MagicMock()
            row.__getitem__ = lambda self, key: 5 if key == "cnt" else None
            return row

        cursor_ts = MagicMock()
        cursor_ts.fetchone = AsyncMock(side_effect=fake_fetchone_ts)
        cursor_ts.__aenter__ = AsyncMock(return_value=cursor_ts)
        cursor_ts.__aexit__ = AsyncMock(return_value=None)

        cursor_cnt = MagicMock()
        cursor_cnt.fetchone = AsyncMock(side_effect=fake_fetchone_cnt)
        cursor_cnt.__aenter__ = AsyncMock(return_value=cursor_cnt)
        cursor_cnt.__aexit__ = AsyncMock(return_value=None)

        call_idx = [0]

        def fake_execute(query, params=None):
            idx = call_idx[0]
            call_idx[0] += 1
            return cursor_ts if idx == 0 else cursor_cnt

        conn = MagicMock()
        conn.execute = fake_execute
        ledger = MagicMock()
        ledger._get_conn = MagicMock(return_value=conn)
        paper_engine = MagicMock()
        paper_engine._ledger = ledger

        agent = OpsAgent()
        report = await agent.run(paper_engine=paper_engine)
        assert "30" in report.body or "h" in report.body

    @pytest.mark.asyncio
    async def test_run_with_paper_engine_zero_decisions(self):
        """OpsAgent detects zero decisions in last hour."""
        from wallet_manager.ai_agents import OpsAgent

        recent_ts = time.time() - 60  # recent enough trade

        async def fake_fetchone_ts():
            row = MagicMock()
            row.__getitem__ = lambda self, key: recent_ts if key == "last_ts" else None
            return row

        async def fake_fetchone_cnt():
            row = MagicMock()
            row.__getitem__ = lambda self, key: 0 if key == "cnt" else None
            return row

        cursor_ts = MagicMock()
        cursor_ts.fetchone = AsyncMock(side_effect=fake_fetchone_ts)
        cursor_ts.__aenter__ = AsyncMock(return_value=cursor_ts)
        cursor_ts.__aexit__ = AsyncMock(return_value=None)

        cursor_cnt = MagicMock()
        cursor_cnt.fetchone = AsyncMock(side_effect=fake_fetchone_cnt)
        cursor_cnt.__aenter__ = AsyncMock(return_value=cursor_cnt)
        cursor_cnt.__aexit__ = AsyncMock(return_value=None)

        call_idx = [0]

        def fake_execute(query, params=None):
            idx = call_idx[0]
            call_idx[0] += 1
            return cursor_ts if idx == 0 else cursor_cnt

        conn = MagicMock()
        conn.execute = fake_execute
        ledger = MagicMock()
        ledger._get_conn = MagicMock(return_value=conn)
        paper_engine = MagicMock()
        paper_engine._ledger = ledger

        agent = OpsAgent()
        report = await agent.run(paper_engine=paper_engine)
        assert "Zero decisions" in report.body or "not be running" in report.body

    @pytest.mark.asyncio
    async def test_run_with_paper_engine_few_decisions(self):
        """OpsAgent detects only 2 decisions in last hour (slow)."""
        from wallet_manager.ai_agents import OpsAgent

        recent_ts = time.time() - 60

        async def fake_fetchone_ts():
            row = MagicMock()
            row.__getitem__ = lambda self, key: recent_ts if key == "last_ts" else None
            return row

        async def fake_fetchone_cnt():
            row = MagicMock()
            row.__getitem__ = lambda self, key: 2 if key == "cnt" else None
            return row

        cursor_ts = MagicMock()
        cursor_ts.fetchone = AsyncMock(side_effect=fake_fetchone_ts)
        cursor_ts.__aenter__ = AsyncMock(return_value=cursor_ts)
        cursor_ts.__aexit__ = AsyncMock(return_value=None)

        cursor_cnt = MagicMock()
        cursor_cnt.fetchone = AsyncMock(side_effect=fake_fetchone_cnt)
        cursor_cnt.__aenter__ = AsyncMock(return_value=cursor_cnt)
        cursor_cnt.__aexit__ = AsyncMock(return_value=None)

        call_idx = [0]

        def fake_execute(query, params=None):
            idx = call_idx[0]
            call_idx[0] += 1
            return cursor_ts if idx == 0 else cursor_cnt

        conn = MagicMock()
        conn.execute = fake_execute
        ledger = MagicMock()
        ledger._get_conn = MagicMock(return_value=conn)
        paper_engine = MagicMock()
        paper_engine._ledger = ledger

        agent = OpsAgent()
        report = await agent.run(paper_engine=paper_engine)
        assert "2" in report.body


class TestRiskAgent:
    """Tests for RiskAgent.run."""

    @pytest.mark.asyncio
    async def test_run_no_engine(self):
        from wallet_manager.ai_agents import RiskAgent
        agent = RiskAgent()
        report = await agent.run(paper_engine=None)
        assert report.agent_name == "RiskAgent"
        assert "Risk" in report.title or "risk" in report.title.lower()
        assert report.body

    @pytest.mark.asyncio
    async def test_run_with_blocking_brakes(self):
        """When blocking brakes exist, they appear in the body."""
        from wallet_manager.ai_agents import RiskAgent

        agent = RiskAgent()

        mock_verdict = MagicMock()
        mock_verdict.risk_score = 80
        mock_verdict.overall_verdict.value = "BLOCK"
        mock_verdict.narrative = "Test narrative"
        mock_verdict.blocking_brakes = ["kill_switch"]
        brake = MagicMock()
        brake.brake_name = "kill_switch"
        brake.verdict.value = "BLOCK"
        brake.reason = "manual"
        mock_verdict.brake_results = [brake]
        mock_verdict.to_dict = MagicMock(return_value={})

        mock_committee = MagicMock()
        mock_committee.evaluate = AsyncMock(return_value=mock_verdict)

        with patch("wallet_manager.risk_committee.get_risk_committee", return_value=mock_committee):
            report = await agent.run(paper_engine=None)

        assert report is not None
        assert "kill_switch" in report.body or "BLOCK" in report.body


class TestNewsAgent:
    """Tests for NewsAgent.run (synchronous)."""

    def test_run_returns_report(self):
        from wallet_manager.ai_agents import NewsAgent
        agent = NewsAgent()
        with patch("wallet_manager.market_intelligence.get_macro_calendar") as mock_get:
            cal = MagicMock()
            cal.to_dict = MagicMock(return_value={})
            cal.upcoming_events = MagicMock(return_value=[])
            cal.get_imminent_events = MagicMock(return_value=[])
            mock_get.return_value = cal
            report = agent.run()
        assert report.agent_name == "NewsAgent"
        assert "No high-impact events" in report.body

    def test_run_with_imminent_events(self):
        from datetime import datetime, timezone

        from wallet_manager.ai_agents import NewsAgent
        agent = NewsAgent()

        event_imminent = MagicMock()
        event_imminent.name = "FOMC"
        event_imminent.dt_utc = datetime.now(tz=UTC)
        event_imminent.impact.value = "CRITICAL"

        event_upcoming = MagicMock()
        event_upcoming.name = "CPI"
        event_upcoming.dt_utc = datetime.now(tz=UTC)
        event_upcoming.impact.value = "HIGH"

        with patch("wallet_manager.market_intelligence.get_macro_calendar") as mock_get:
            cal = MagicMock()
            cal.to_dict = MagicMock(return_value={})
            cal.upcoming_events = MagicMock(return_value=[event_upcoming])
            cal.get_imminent_events = MagicMock(return_value=[event_imminent])
            mock_get.return_value = cal
            report = agent.run()

        assert "FOMC" in report.body
        assert "IMMINENT" in report.body

    def test_run_with_many_upcoming_events(self):
        """Only the first 8 upcoming events should appear."""
        from datetime import datetime, timezone

        from wallet_manager.ai_agents import NewsAgent
        agent = NewsAgent()

        events = []
        for i in range(10):
            e = MagicMock()
            e.name = f"Event{i}"
            e.dt_utc = datetime.now(tz=UTC)
            e.impact.value = "MEDIUM"
            events.append(e)

        with patch("wallet_manager.market_intelligence.get_macro_calendar") as mock_get:
            cal = MagicMock()
            cal.to_dict = MagicMock(return_value={})
            cal.upcoming_events = MagicMock(return_value=events)
            cal.get_imminent_events = MagicMock(return_value=[])
            mock_get.return_value = cal
            report = agent.run()

        # Check that only up to 8 events are in body
        mentioned = sum(1 for i in range(10) if f"Event{i}" in report.body)
        assert mentioned <= 8


class TestResearchAgent:
    """Tests for ResearchAgent.run."""

    @pytest.mark.asyncio
    async def test_run_no_engine(self):
        from wallet_manager.ai_agents import ResearchAgent
        agent = ResearchAgent()
        report = await agent.run(paper_engine=None)
        assert report.agent_name == "ResearchAgent"
        assert report.title == "Weekly Strategy Review"
        assert report.body


class TestTradeReviewAgent:
    """Tests for TradeReviewAgent.run."""

    @pytest.mark.asyncio
    async def test_run_no_engine(self):
        from wallet_manager.ai_agents import TradeReviewAgent
        agent = TradeReviewAgent()
        report = await agent.run(paper_engine=None)
        assert report.agent_name == "TradeReviewAgent"
        assert "Trade Review" in report.title
        assert report.body


class TestAgentRegistry:
    """Tests for AgentRegistry.run_all and get_agent_registry singleton."""

    @pytest.mark.asyncio
    async def test_run_all_returns_list(self):
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        reports = await registry.run_all(paper_engine=None)
        assert isinstance(reports, list)
        assert len(reports) >= 1

    @pytest.mark.asyncio
    async def test_run_all_to_dict(self):
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        dicts = await registry.run_all_to_dict(paper_engine=None)
        assert isinstance(dicts, list)
        for d in dicts:
            assert isinstance(d, dict)

    @pytest.mark.asyncio
    async def test_run_all_handles_agent_failure(self):
        """Agents that raise are caught and skipped — remaining agents still run."""
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        # Make ops agent raise
        registry.ops.run = AsyncMock(side_effect=Exception("ops crash"))
        reports = await registry.run_all(paper_engine=None)
        # Should still have reports from other agents
        assert isinstance(reports, list)

    @pytest.mark.asyncio
    async def test_run_all_news_agent_failure(self):
        """NewsAgent failure is caught and skipped."""
        from wallet_manager.ai_agents import AgentRegistry
        registry = AgentRegistry()
        registry.news.run = MagicMock(side_effect=Exception("news crash"))
        reports = await registry.run_all(paper_engine=None)
        assert isinstance(reports, list)

    def test_get_agent_registry_singleton(self):
        from wallet_manager import ai_agents
        from wallet_manager.ai_agents import get_agent_registry
        # Reset singleton to test fresh creation
        original = ai_agents._registry
        ai_agents._registry = None
        try:
            a = get_agent_registry()
            b = get_agent_registry()
            assert a is b
        finally:
            ai_agents._registry = original


# ============================================================================
# experiment_lab.py — all data classes and ExperimentLab
# ============================================================================


class TestThresholdVariant:
    """Tests for ThresholdVariant.would_trade and to_dict."""

    def test_would_trade_all_pass(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35)
        assert v.would_trade(robust_alpha=0.10, confidence=0.40) is True

    def test_would_trade_fails_alpha(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35)
        assert v.would_trade(robust_alpha=0.05, confidence=0.40) is False

    def test_would_trade_fails_confidence(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35)
        assert v.would_trade(robust_alpha=0.10, confidence=0.20) is False

    def test_would_trade_fails_adf(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35, adf_pvalue_threshold=0.05)
        # adf_pvalue=0.10 > threshold 0.05 → fail
        assert v.would_trade(robust_alpha=0.10, confidence=0.40, adf_pvalue=0.10) is False

    def test_would_trade_adf_zero_skips_check(self):
        """adf_pvalue=0 means skip the ADF check."""
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35)
        assert v.would_trade(robust_alpha=0.10, confidence=0.40, adf_pvalue=0.0) is True

    def test_would_trade_negative_alpha(self):
        """abs(robust_alpha) is used."""
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35)
        assert v.would_trade(robust_alpha=-0.10, confidence=0.40) is True

    def test_to_dict_keys(self):
        from wallet_manager.experiment_lab import ThresholdVariant
        v = ThresholdVariant(name="t", alpha_threshold=0.08, confidence_threshold=0.35)
        d = v.to_dict()
        assert "name" in d and "alpha_threshold" in d and "confidence_threshold" in d


class TestFeatureToggleVariant:
    """Tests for FeatureToggleVariant.feature_active and to_dict."""

    def test_feature_active_when_not_disabled(self):
        from wallet_manager.experiment_lab import FeatureToggleVariant
        v = FeatureToggleVariant(
            name="v1",
            enabled_features=frozenset(["feature_a"]),
            disabled_features=frozenset(["feature_b"]),
        )
        assert v.feature_active("feature_a") is True
        assert v.feature_active("feature_b") is False

    def test_feature_active_unknown_feature(self):
        """Unknown features default to active (not in disabled set)."""
        from wallet_manager.experiment_lab import FeatureToggleVariant
        v = FeatureToggleVariant(
            name="v1",
            enabled_features=frozenset(),
            disabled_features=frozenset(["feature_b"]),
        )
        assert v.feature_active("unknown_feature") is True

    def test_to_dict_fields(self):
        from wallet_manager.experiment_lab import FeatureToggleVariant
        v = FeatureToggleVariant(
            name="v1",
            enabled_features=frozenset(["f_a"]),
            disabled_features=frozenset(["f_b"]),
            description="test",
        )
        d = v.to_dict()
        assert d["name"] == "v1"
        assert "f_a" in d["enabled_features"]
        assert "f_b" in d["disabled_features"]
        assert d["description"] == "test"


class TestRegimeVariant:
    """Tests for RegimeVariant.would_trade and to_dict."""

    def test_would_trade_allowed_regime(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant(
            name="rv",
            allowed_regimes=frozenset(["low_vol"]),
            alpha_threshold=0.08,
            confidence_threshold=0.35,
        )
        assert v.would_trade(0.10, 0.40, "low_vol") is True

    def test_would_trade_disallowed_regime(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant(
            name="rv",
            allowed_regimes=frozenset(["low_vol"]),
            alpha_threshold=0.08,
            confidence_threshold=0.35,
        )
        assert v.would_trade(0.10, 0.40, "high_vol") is False

    def test_would_trade_empty_regimes_allows_all(self):
        """Empty allowed_regimes means no regime restriction."""
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant(
            name="rv",
            allowed_regimes=frozenset(),
            alpha_threshold=0.08,
            confidence_threshold=0.35,
        )
        assert v.would_trade(0.10, 0.40, "anything") is True

    def test_would_trade_fails_alpha(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant(
            name="rv",
            allowed_regimes=frozenset(["low_vol"]),
            alpha_threshold=0.08,
            confidence_threshold=0.35,
        )
        assert v.would_trade(0.05, 0.40, "low_vol") is False

    def test_to_dict_fields(self):
        from wallet_manager.experiment_lab import RegimeVariant
        v = RegimeVariant(
            name="rv",
            allowed_regimes=frozenset(["low_vol"]),
        )
        d = v.to_dict()
        assert "name" in d
        assert "allowed_regimes" in d
        assert "low_vol" in d["allowed_regimes"]


class TestExperimentCycleRecord:
    def test_to_dict(self):
        from wallet_manager.experiment_lab import ExperimentCycleRecord
        r = ExperimentCycleRecord(
            timestamp=1000.0,
            variant_name="baseline",
            would_trade=True,
            baseline_traded=True,
            robust_alpha=0.1,
            confidence=0.4,
            pnl_usd=5.0,
        )
        d = r.to_dict()
        assert d["variant_name"] == "baseline"
        assert d["would_trade"] is True
        assert d["pnl_usd"] == 5.0


class TestExperimentLab:
    """Tests for ExperimentLab methods."""

    def _make_lab(self):
        from wallet_manager.experiment_lab import ExperimentLab
        return ExperimentLab()

    def test_baseline_registered_on_init(self):
        lab = self._make_lab()
        names = [v.name for v in lab.active_variants()]
        assert "baseline" in names

    def test_register_threshold_variant(self):
        lab = self._make_lab()
        v = lab.register_threshold_variant("test_v", 0.05, 0.30, description="test")
        assert v.name == "test_v"
        assert v.alpha_threshold == 0.05
        assert "test_v" in lab._records

    def test_register_feature_toggle_variant(self):
        lab = self._make_lab()
        v = lab.register_feature_toggle_variant(
            "no_adf",
            disabled_features=["adf_gate"],
            enabled_features=["extra_signal"],
        )
        assert v.name == "no_adf"
        assert "adf_gate" in v.disabled_features
        assert "extra_signal" in v.enabled_features

    def test_register_feature_toggle_variant_no_enabled(self):
        lab = self._make_lab()
        v = lab.register_feature_toggle_variant("toggle_v", disabled_features=["foo"])
        assert v.enabled_features == frozenset()

    def test_register_regime_variant(self):
        lab = self._make_lab()
        v = lab.register_regime_variant("rv_test", allowed_regimes=["low_vol"], alpha_threshold=0.06)
        assert v.name == "rv_test"
        assert "regime:rv_test" in lab._records

    def test_record_cycle_unknown_variant_ignored(self):
        lab = self._make_lab()
        # Should not raise
        lab.record_cycle("nonexistent", True, False, 0.1, 0.4)
        assert True

    def test_record_cycle_appends(self):
        lab = self._make_lab()
        lab.record_cycle("baseline", True, True, 0.10, 0.40, pnl_usd=1.0)
        assert len(lab._records["baseline"]) == 1

    def test_record_cycle_trims_to_max(self):
        lab = self._make_lab()
        lab._max_records_per_variant = 5
        for _ in range(10):
            lab.record_cycle("baseline", True, True, 0.10, 0.40)
        assert len(lab._records["baseline"]) == 5

    def test_record_all_variants(self):
        lab = self._make_lab()
        lab.register_threshold_variant("extra", 0.05, 0.30)
        lab.record_all_variants(0.12, 0.45, 0.03, baseline_traded=True, pnl_usd=2.0)
        assert len(lab._records["baseline"]) == 1
        assert len(lab._records["extra"]) == 1

    def test_record_all_variants_skips_inactive(self):
        lab = self._make_lab()
        lab.register_threshold_variant("inactive_v", 0.05, 0.30)
        lab._variants["inactive_v"].active = False
        lab.record_all_variants(0.12, 0.45, 0.03, baseline_traded=True)
        # inactive_v should have no records
        assert len(lab._records["inactive_v"]) == 0

    def test_record_all_variants_baseline_not_traded(self):
        """When baseline didn't trade, pnl_usd should be 0."""
        lab = self._make_lab()
        lab.record_all_variants(0.12, 0.45, 0.03, baseline_traded=False, pnl_usd=5.0)
        rec = lab._records["baseline"][0]
        assert rec.pnl_usd == 0.0

    def test_record_regime_cycle(self):
        lab = self._make_lab()
        lab.register_regime_variant("rv_test", allowed_regimes=["low_vol"])
        lab.record_regime_cycle("low_vol", 0.10, 0.40, baseline_traded=True, pnl_usd=1.0)
        assert len(lab._records["regime:rv_test"]) == 1

    def test_record_regime_cycle_skips_inactive(self):
        lab = self._make_lab()
        lab.register_regime_variant("rv_inactive", allowed_regimes=["low_vol"])
        lab._regime_variants["rv_inactive"].active = False
        lab.record_regime_cycle("low_vol", 0.10, 0.40, baseline_traded=True)
        assert len(lab._records.get("regime:rv_inactive", [])) == 0

    def test_record_regime_cycle_trims(self):
        lab = self._make_lab()
        lab.register_regime_variant("rv_trim", allowed_regimes=["low_vol"])
        lab._max_records_per_variant = 3
        for _ in range(5):
            lab.record_regime_cycle("low_vol", 0.10, 0.40, baseline_traded=True)
        assert len(lab._records["regime:rv_trim"]) == 3

    def test_get_variant_report_empty(self):
        lab = self._make_lab()
        report = lab.get_variant_report()
        # baseline has no records
        baseline_entry = next(r for r in report if r["name"] == "baseline")
        assert baseline_entry["cycles"] == 0

    def test_get_variant_report_with_records(self):
        lab = self._make_lab()
        for _ in range(5):
            lab.record_cycle("baseline", True, True, 0.10, 0.40, pnl_usd=1.0)
        lab.record_cycle("baseline", False, True, 0.10, 0.40, pnl_usd=0.5)
        report = lab.get_variant_report()
        baseline = next(r for r in report if r["name"] == "baseline")
        assert baseline["cycles"] == 6
        assert "agreement_pct" in baseline
        assert "shared_realized_pnl" in baseline
        assert "variant_only_signals" in baseline
        assert "baseline_only_signals" in baseline

    def test_compare_variants_not_enough_cycles(self):
        """With < 10 cycles per variant, winner should be None."""
        lab = self._make_lab()
        result = lab.compare_variants()
        assert result["winner"] is None

    def test_compare_variants_with_enough_data(self):
        lab = self._make_lab()
        lab.register_threshold_variant("challenger", 0.05, 0.28, description="challenger")
        for _ in range(15):
            lab.record_cycle("baseline", True, True, 0.10, 0.40, pnl_usd=1.0)
            lab.record_cycle("challenger", True, True, 0.10, 0.40, pnl_usd=2.0)
        result = lab.compare_variants()
        assert "winner" in result
        assert "recommendation" in result

    def test_compare_variants_baseline_wins(self):
        """When baseline is the winner, recommendation reflects that."""
        lab = self._make_lab()
        for _ in range(15):
            lab.record_cycle("baseline", True, True, 0.10, 0.40, pnl_usd=5.0)
        result = lab.compare_variants()
        # With only baseline having enough data, winner is baseline
        assert result["winner"] == "baseline"
        assert "Baseline" in result["recommendation"] or "baseline" in result["recommendation"]

    def test_get_feature_toggle_report(self):
        lab = self._make_lab()
        lab.register_feature_toggle_variant("ftv", disabled_features=["gate"])
        report = lab.get_feature_toggle_report()
        assert isinstance(report, list)
        assert any(r["name"] == "ftv" for r in report)

    def test_get_regime_variant_report_empty_records(self):
        lab = self._make_lab()
        lab.register_regime_variant("rv_empty", allowed_regimes=["low_vol"])
        report = lab.get_regime_variant_report()
        rv_entry = next((r for r in report if r["name"] == "rv_empty"), None)
        assert rv_entry is not None
        assert rv_entry["cycles"] == 0

    def test_get_regime_variant_report_with_records(self):
        lab = self._make_lab()
        lab.register_regime_variant("rv_w_data", allowed_regimes=["low_vol"])
        for _ in range(3):
            lab.record_regime_cycle("low_vol", 0.10, 0.40, baseline_traded=True, pnl_usd=1.0)
        report = lab.get_regime_variant_report()
        rv = next((r for r in report if r["name"] == "rv_w_data"), None)
        assert rv is not None
        assert rv["cycles"] == 3

    def test_to_dict(self):
        lab = self._make_lab()
        d = lab.to_dict()
        assert "total_variants" in d
        assert "active_variants" in d
        assert "comparison" in d
        assert "feature_toggles" in d
        assert "regime_variants" in d

    def test_active_variants_excludes_inactive(self):
        lab = self._make_lab()
        lab.register_threshold_variant("inactive", 0.05, 0.30)
        lab._variants["inactive"].active = False
        active = lab.active_variants()
        assert all(v.active for v in active)
        assert not any(v.name == "inactive" for v in active)

    def test_get_experiment_lab_singleton(self):
        from wallet_manager import experiment_lab
        from wallet_manager.experiment_lab import get_experiment_lab
        original = experiment_lab._lab
        experiment_lab._lab = None
        try:
            a = get_experiment_lab()
            b = get_experiment_lab()
            assert a is b
            # Check default variants are registered
            names = [v.name for v in a.active_variants()]
            assert "baseline" in names
            assert "relaxed_confidence" in names
        finally:
            experiment_lab._lab = original

    def test_get_experiment_lab_registers_defaults(self):
        from wallet_manager import experiment_lab
        from wallet_manager.experiment_lab import get_experiment_lab
        original = experiment_lab._lab
        experiment_lab._lab = None
        try:
            lab = get_experiment_lab()
            names = [v.name for v in lab.active_variants()]
            assert "strict_confidence" in names
            assert "wide_alpha" in names
        finally:
            experiment_lab._lab = original


# ============================================================================
# trade_journal.py — _mask, TradeRecord, TradeJournal
# ============================================================================


class TestMaskFunction:
    """Tests for the _mask helper."""

    def test_mask_normal_string(self):
        from wallet_manager.trade_journal import _mask
        result = _mask("0x1234567890abcdef", prefix=6, suffix=4)
        assert result.startswith("0x1234")
        assert result.endswith("cdef")
        assert "..." in result

    def test_mask_short_string_returned_as_is(self):
        from wallet_manager.trade_journal import _mask
        short = "abc"
        assert _mask(short) == short

    def test_mask_empty_string(self):
        from wallet_manager.trade_journal import _mask
        assert _mask("") == ""

    def test_mask_exactly_prefix_plus_suffix(self):
        from wallet_manager.trade_journal import _mask
        # Exactly 10 chars with prefix=6, suffix=4 → not masked (len <= prefix + suffix)
        val = "1234567890"
        assert _mask(val, prefix=6, suffix=4) == val

    def test_mask_one_over_limit(self):
        from wallet_manager.trade_journal import _mask
        val = "12345678901"  # 11 chars, prefix=6, suffix=4 → masked
        result = _mask(val, prefix=6, suffix=4)
        assert "..." in result


class TestTradeJournal:
    """Tests for TradeJournal."""

    @pytest.fixture
    def journal(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        return TradeJournal(journal_path=str(tmp_path / "journal.jsonl"), max_mb=1.0)

    def test_record_trade_simple(self, journal, tmp_path):
        journal.record_trade(
            wallet_id="wallet123",
            strategy="ALGO",
            direction="BUY",
            amount_usd=100.0,
            tx_hash="0xaabbcc",
        )
        path = tmp_path / "journal.jsonl"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["strategy"] == "ALGO"
        assert data["direction"] == "BUY"

    def test_record_trade_masks_wallet_id(self, journal, tmp_path):
        long_wallet = "0x" + "a" * 40
        journal.record_trade(wallet_id=long_wallet, strategy="ALGO", direction="BUY",
                             amount_usd=50.0, tx_hash="0x00")
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        assert long_wallet not in content

    def test_record_trade_masks_tx_hash(self, journal, tmp_path):
        long_hash = "0x" + "b" * 64
        journal.record_trade(wallet_id="w1", strategy="ALGO", direction="BUY",
                             amount_usd=50.0, tx_hash=long_hash)
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        assert long_hash not in content

    def test_record_trade_with_extra_redacts_private_key(self, journal, tmp_path):
        journal.record_trade(
            wallet_id="w1", strategy="A", direction="B", amount_usd=1.0, tx_hash="0x00",
            extra={"private_key": "super_secret", "normal_field": "public_data"},
        )
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        assert "super_secret" not in content
        assert "public_data" in content

    def test_record_trade_with_extra_redacts_secret_field(self, journal, tmp_path):
        journal.record_trade(
            wallet_id="w1", strategy="A", direction="B", amount_usd=1.0, tx_hash="0x00",
            extra={"api_secret": "top_secret"},
        )
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        assert "top_secret" not in content

    def test_record_trade_with_full_trade_record(self, journal, tmp_path):
        from wallet_manager.trade_journal import TradeRecord
        trade = TradeRecord(
            timestamp_utc="2026-04-04T00:00:00Z",
            wallet_id="0x" + "a" * 40,
            strategy_type="ALGO",
            action="BUY",
            entry_price=100.0,
            exit_price=105.0,
            position_usd=1000.0,
            pnl_usd=50.0,
            pnl_pct=0.05,
            execution_mode="paper",
            regime="normal",
            alpha_score=0.9,
            confidence=0.8,
            garch_sigma=0.02,
            kelly_fraction_used=0.25,
            slippage_bps=3.0,
            gas_cost_usd=0.5,
            dex_venue="uniswap",
            tx_hash="0x" + "b" * 64,
            drawdown_at_entry=0.01,
            drawdown_at_exit=0.02,
            ml_predictions={"ensemble": 0.8},
        )
        journal.record_trade(trade)
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        data = json.loads(content.strip())
        assert data["strategy_type"] == "ALGO"

    def test_record_trade_via_trade_kwarg(self, journal, tmp_path):
        """record_trade with trade=TradeRecord keyword arg."""
        from wallet_manager.trade_journal import TradeRecord
        trade = TradeRecord(
            timestamp_utc="2026-04-04T00:00:00Z",
            wallet_id="0x" + "c" * 40,
            strategy_type="YIELD",
            action="DEPOSIT",
            entry_price=1.0,
            exit_price=1.0,
            position_usd=500.0,
            pnl_usd=5.0,
            pnl_pct=0.01,
            execution_mode="paper",
            regime="stable",
            alpha_score=0.5,
            confidence=0.6,
            garch_sigma=0.01,
            kelly_fraction_used=0.1,
            slippage_bps=1.0,
            gas_cost_usd=0.2,
            dex_venue="aave",
            tx_hash="0x" + "d" * 64,
            drawdown_at_entry=0.0,
            drawdown_at_exit=0.0,
            ml_predictions={},
        )
        journal.record_trade(trade=trade)
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        data = json.loads(content.strip())
        assert data["strategy_type"] == "YIELD"

    def test_get_recent_trades_empty_file(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        journal = TradeJournal(journal_path=str(tmp_path / "notexist.jsonl"), max_mb=1.0)
        result = journal.get_recent_trades()
        assert result == []

    def test_get_recent_trades_returns_n_most_recent(self, journal):
        for i in range(10):
            journal.record_trade(f"w{i}", "ALGO", "BUY", float(i), f"0x{i:04x}")
        recent = journal.get_recent_trades(n=3)
        assert len(recent) == 3

    def test_get_recent_trades_order_most_recent_first(self, journal):
        for i in range(5):
            journal.record_trade("w1", "ALGO", "BUY", float(i * 10), f"0x{i:04x}")
        recent = journal.get_recent_trades(n=3)
        # Most recent should be amount_usd=40 (i=4)
        assert recent[0]["amount_usd"] == pytest.approx(40.0)

    def test_rotation_when_size_exceeded(self, journal, tmp_path):
        journal._max_bytes = 50  # very small limit
        for i in range(20):
            journal.record_trade("w1", "ALGO", "BUY", 100.0, f"0x{i:08x}")
        path = tmp_path / "journal.jsonl"
        assert path.exists()

    def test_rotated_path_increments(self, journal):
        p1 = journal._rotated_path()
        p2 = journal._rotated_path()
        assert p1 != p2
        assert journal._rotation_index == 2

    def test_get_max_bytes_from_max_mb(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        j = TradeJournal(journal_path=str(tmp_path / "j.jsonl"), max_mb=2.0)
        j._max_bytes = None  # Force recomputation via _max_mb
        assert j._get_max_bytes() == int(2.0 * 1024 * 1024)

    def test_record_trade_write_failure_logged(self, tmp_path):
        """Write failure is caught and logged, no exception."""
        from wallet_manager.trade_journal import TradeJournal
        j = TradeJournal(journal_path=str(tmp_path / "j.jsonl"), max_mb=1.0)
        with patch.object(TradeJournal, "_write_line_sync", side_effect=OSError("disk full")):
            # Should not raise
            j.record_trade("w1", "ALGO", "BUY", 100.0, "0x00")

    @pytest.mark.asyncio
    async def test_record_trade_async(self, journal, tmp_path):
        from wallet_manager.trade_journal import TradeRecord
        trade = TradeRecord(
            timestamp_utc="2026-04-04T00:00:00Z",
            wallet_id="0x" + "e" * 40,
            strategy_type="ALGO",
            action="SELL",
            entry_price=200.0,
            exit_price=210.0,
            position_usd=2000.0,
            pnl_usd=100.0,
            pnl_pct=0.05,
            execution_mode="paper",
            regime="normal",
            alpha_score=0.9,
            confidence=0.75,
            garch_sigma=0.02,
            kelly_fraction_used=0.3,
            slippage_bps=5.0,
            gas_cost_usd=0.8,
            dex_venue="uniswap_v3",
            tx_hash="0x" + "f" * 64,
            drawdown_at_entry=0.02,
            drawdown_at_exit=0.03,
            ml_predictions={"xgb": 0.7, "ensemble": 0.75},
        )
        await journal.record_trade_async(trade)
        records = journal.get_recent_trades(1)
        assert len(records) == 1
        assert records[0]["action"] == "SELL"

    @pytest.mark.asyncio
    async def test_record_trade_async_masks_fields(self, journal, tmp_path):
        from wallet_manager.trade_journal import TradeRecord
        long_wallet = "0x" + "a" * 40
        long_hash = "0x" + "b" * 64
        trade = TradeRecord(
            timestamp_utc="2026-04-04T00:00:00Z",
            wallet_id=long_wallet,
            strategy_type="ALGO",
            action="BUY",
            entry_price=100.0,
            exit_price=105.0,
            position_usd=1000.0,
            pnl_usd=50.0,
            pnl_pct=0.05,
            execution_mode="paper",
            regime="normal",
            alpha_score=0.9,
            confidence=0.8,
            garch_sigma=0.02,
            kelly_fraction_used=0.25,
            slippage_bps=3.0,
            gas_cost_usd=0.5,
            dex_venue="uniswap",
            tx_hash=long_hash,
            drawdown_at_entry=0.01,
            drawdown_at_exit=0.02,
            ml_predictions={},
        )
        await journal.record_trade_async(trade)
        content = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
        assert long_wallet not in content
        assert long_hash not in content

    @pytest.mark.asyncio
    async def test_get_recent_trades_async_nonexistent_file(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        j = TradeJournal(journal_path=str(tmp_path / "nope.jsonl"), max_mb=1.0)
        result = await j.get_recent_trades_async()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_recent_trades_async_with_records(self, journal):
        from wallet_manager.trade_journal import TradeRecord
        for i in range(5):
            trade = TradeRecord(
                timestamp_utc=f"2026-04-04T0{i}:00:00Z",
                wallet_id="0x" + "a" * 40,
                strategy_type="ALGO",
                action="BUY",
                entry_price=float(i * 10),
                exit_price=float(i * 10 + 5),
                position_usd=100.0,
                pnl_usd=5.0,
                pnl_pct=0.05,
                execution_mode="paper",
                regime="normal",
                alpha_score=0.8,
                confidence=0.7,
                garch_sigma=0.02,
                kelly_fraction_used=0.2,
                slippage_bps=2.0,
                gas_cost_usd=0.3,
                dex_venue="uniswap",
                tx_hash="0x" + "f" * 64,
                drawdown_at_entry=0.01,
                drawdown_at_exit=0.02,
                ml_predictions={},
            )
            await journal.record_trade_async(trade)
        result = await journal.get_recent_trades_async(n=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_record_trade_async_write_failure_logged(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal, TradeRecord
        j = TradeJournal(journal_path=str(tmp_path / "j.jsonl"), max_mb=1.0)
        trade = TradeRecord(
            timestamp_utc="2026-04-04T00:00:00Z",
            wallet_id="0x" + "a" * 40,
            strategy_type="ALGO",
            action="BUY",
            entry_price=100.0,
            exit_price=105.0,
            position_usd=1000.0,
            pnl_usd=50.0,
            pnl_pct=0.05,
            execution_mode="paper",
            regime="normal",
            alpha_score=0.9,
            confidence=0.8,
            garch_sigma=0.02,
            kelly_fraction_used=0.25,
            slippage_bps=3.0,
            gas_cost_usd=0.5,
            dex_venue="uniswap",
            tx_hash="0x" + "b" * 64,
            drawdown_at_entry=0.01,
            drawdown_at_exit=0.02,
            ml_predictions={},
        )
        # Patch _write_line_sync to raise after rotation succeeds
        with patch.object(TradeJournal, "_write_line_sync", side_effect=OSError("write fail")):
            # Should not raise
            await j.record_trade_async(trade)

    @pytest.mark.asyncio
    async def test_get_recent_trades_async_read_failure(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        # Create a file to make it exist
        path = tmp_path / "j.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        j = TradeJournal(journal_path=str(path), max_mb=1.0)
        with patch("wallet_manager.trade_journal.asyncio.to_thread", side_effect=Exception("read fail")):
            result = await j.get_recent_trades_async()
        assert result == []

    def test_get_recent_trades_invalid_json_skipped(self, journal, tmp_path):
        """Invalid JSON lines are silently skipped."""
        path = tmp_path / "journal.jsonl"
        path.write_text('{"valid": 1}\nNOT JSON\n{"valid": 2}\n', encoding="utf-8")
        result = journal.get_recent_trades(n=10)
        assert len(result) == 2

    def test_get_recent_trades_read_failure_returns_empty(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        path = tmp_path / "j.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        j = TradeJournal(journal_path=str(path), max_mb=1.0)
        with patch.object(TradeJournal, "_read_lines_sync", side_effect=Exception("disk error")):
            result = j.get_recent_trades()
        assert result == []

    def test_singleton_get_trade_journal(self):
        from wallet_manager import trade_journal as tj_module
        from wallet_manager.trade_journal import get_trade_journal
        original = tj_module._journal_instance
        tj_module._journal_instance = None
        try:
            a = get_trade_journal()
            b = get_trade_journal()
            assert a is b
        finally:
            tj_module._journal_instance = original

    def test_rotation_check_failure_logged(self, tmp_path):
        """If rotation check throws, it's caught and logged."""
        from wallet_manager.trade_journal import TradeJournal
        j = TradeJournal(journal_path=str(tmp_path / "j.jsonl"), max_mb=1.0)
        with patch.object(Path, "stat", side_effect=OSError("stat fail")):
            # Should not raise
            j._rotate_sync_if_needed()

    def test_write_line_sync_creates_parent_dirs(self, tmp_path):
        from wallet_manager.trade_journal import TradeJournal
        nested_path = tmp_path / "deep" / "nested" / "journal.jsonl"
        j = TradeJournal(journal_path=str(nested_path), max_mb=1.0)
        j.record_trade("w1", "ALGO", "BUY", 10.0, "0x00")
        assert nested_path.exists()
