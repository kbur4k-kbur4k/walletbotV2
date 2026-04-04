"""
tests/unit/test_factories_and_notifier.py

Comprehensive tests covering:
  - task_factories_airdrop.py  (re-exports)
  - task_factories_algo.py     (re-exports)
  - task_factories_short.py    (re-exports)
  - task_factories_yield.py    (re-exports)
  - telegram_notifier.py       (all classes/methods)
  - tick_collector.py          (all methods)

At least 60 tests.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Return a MagicMock Settings with safe defaults."""
    s = MagicMock()
    s.telegram_bot_token = "test_token_123"
    s.telegram_chat_id = "123456789"
    s.telegram_rate_limit = 1.0
    s.telegram_rate_burst = 1
    s.yield_min_apy_pct = 1.0
    s.drawdown_reduce_threshold = 0.05
    s.alert_dedup_window_seconds = 3600.0
    s.model_staleness_alert_days = 7.0
    s.rpc_latency_alert_ms = 500.0
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ===========================================================================
# Section 1 — task_factories_airdrop re-export tests
# ===========================================================================

class TestTaskFactoriesAirdropReExports:
    """Verify that task_factories_airdrop.py correctly re-exports the expected symbols."""

    def test_airdrop_task_factory_importable(self):
        from wallet_manager.task_factories_airdrop import _airdrop_task_factory
        assert callable(_airdrop_task_factory)

    def test_execute_eth_to_usdc_importable(self):
        from wallet_manager.task_factories_airdrop import _execute_eth_to_usdc
        assert callable(_execute_eth_to_usdc)

    def test_execute_usdc_to_eth_importable(self):
        from wallet_manager.task_factories_airdrop import _execute_usdc_to_eth
        assert callable(_execute_usdc_to_eth)

    def test_all_list_contents(self):
        import wallet_manager.task_factories_airdrop as mod
        assert "_airdrop_task_factory" in mod.__all__
        assert "_execute_eth_to_usdc" in mod.__all__
        assert "_execute_usdc_to_eth" in mod.__all__

    def test_all_list_length(self):
        import wallet_manager.task_factories_airdrop as mod
        assert len(mod.__all__) == 3

    def test_airdrop_factory_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _airdrop_task_factory as b
        from wallet_manager.task_factories_airdrop import _airdrop_task_factory as a
        assert a is b

    def test_eth_to_usdc_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _execute_eth_to_usdc as b
        from wallet_manager.task_factories_airdrop import _execute_eth_to_usdc as a
        assert a is b

    def test_usdc_to_eth_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _execute_usdc_to_eth as b
        from wallet_manager.task_factories_airdrop import _execute_usdc_to_eth as a
        assert a is b


# ===========================================================================
# Section 2 — task_factories_algo re-export tests
# ===========================================================================

class TestTaskFactoriesAlgoReExports:
    """Verify that task_factories_algo.py correctly re-exports the expected symbols."""

    def test_default_task_factory_importable(self):
        from wallet_manager.task_factories_algo import _default_task_factory
        assert callable(_default_task_factory)

    def test_ml_quant_task_factory_importable(self):
        from wallet_manager.task_factories_algo import _ml_quant_task_factory
        assert callable(_ml_quant_task_factory)

    def test_quantx_task_factory_importable(self):
        from wallet_manager.task_factories_algo import _quantx_task_factory
        assert callable(_quantx_task_factory)

    def test_all_list_contents(self):
        import wallet_manager.task_factories_algo as mod
        assert "_default_task_factory" in mod.__all__
        assert "_ml_quant_task_factory" in mod.__all__
        assert "_quantx_task_factory" in mod.__all__

    def test_all_list_length(self):
        import wallet_manager.task_factories_algo as mod
        assert len(mod.__all__) == 3

    def test_default_factory_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _default_task_factory as b
        from wallet_manager.task_factories_algo import _default_task_factory as a
        assert a is b

    def test_ml_quant_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _ml_quant_task_factory as b
        from wallet_manager.task_factories_algo import _ml_quant_task_factory as a
        assert a is b

    def test_quantx_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _quantx_task_factory as b
        from wallet_manager.task_factories_algo import _quantx_task_factory as a
        assert a is b


# ===========================================================================
# Section 3 — task_factories_short re-export tests
# ===========================================================================

class TestTaskFactoriesShortReExports:
    """Verify that task_factories_short.py correctly re-exports the expected symbols."""

    def test_three_pillar_importable(self):
        from wallet_manager.task_factories_short import _three_pillar_task_factory
        assert callable(_three_pillar_task_factory)

    def test_all_list_contents(self):
        import wallet_manager.task_factories_short as mod
        assert "_three_pillar_task_factory" in mod.__all__

    def test_all_list_length(self):
        import wallet_manager.task_factories_short as mod
        assert len(mod.__all__) == 1

    def test_three_pillar_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _three_pillar_task_factory as b
        from wallet_manager.task_factories_short import _three_pillar_task_factory as a
        assert a is b


# ===========================================================================
# Section 4 — task_factories_yield re-export tests
# ===========================================================================

class TestTaskFactoriesYieldReExports:
    """Verify that task_factories_yield.py correctly re-exports the expected symbols."""

    def test_robust_paper_importable(self):
        from wallet_manager.task_factories_yield import _robust_paper_task_factory
        assert callable(_robust_paper_task_factory)

    def test_robust_importable(self):
        from wallet_manager.task_factories_yield import _robust_task_factory
        assert callable(_robust_task_factory)

    def test_all_list_contents(self):
        import wallet_manager.task_factories_yield as mod
        assert "_robust_paper_task_factory" in mod.__all__
        assert "_robust_task_factory" in mod.__all__

    def test_all_list_length(self):
        import wallet_manager.task_factories_yield as mod
        assert len(mod.__all__) == 2

    def test_robust_paper_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _robust_paper_task_factory as b
        from wallet_manager.task_factories_yield import _robust_paper_task_factory as a
        assert a is b

    def test_robust_same_object_as_task_factories(self):
        from wallet_manager.task_factories import _robust_task_factory as b
        from wallet_manager.task_factories_yield import _robust_task_factory as a
        assert a is b


# ===========================================================================
# Section 5 — telegram_notifier.py
# ===========================================================================

class TestTelegramCredentials:
    """Tests for _TelegramCredentials dataclass."""

    def test_repr_hides_token(self):
        from wallet_manager.telegram_notifier import _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="secret123", _chat_id="999")
        assert "secret123" not in repr(creds)
        assert repr(creds) == "_TelegramCredentials(***)"

    def test_bot_token_property(self):
        from wallet_manager.telegram_notifier import _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="mytoken", _chat_id="42")
        assert creds.bot_token == "mytoken"

    def test_chat_id_property(self):
        from wallet_manager.telegram_notifier import _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="chatid_99")
        assert creds.chat_id == "chatid_99"

    def test_from_env_raises_when_missing(self):
        from wallet_manager.exceptions import NotifierConfigurationError
        from wallet_manager.telegram_notifier import _TelegramCredentials
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = ""
        mock_cfg.telegram_chat_id = ""
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            with pytest.raises(NotifierConfigurationError) as exc_info:
                _TelegramCredentials.from_env()
        assert "TELEGRAM_BOT_TOKEN" in str(exc_info.value)

    def test_from_env_raises_missing_chat_id_only(self):
        from wallet_manager.exceptions import NotifierConfigurationError
        from wallet_manager.telegram_notifier import _TelegramCredentials
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = "sometoken"
        mock_cfg.telegram_chat_id = ""
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            with pytest.raises(NotifierConfigurationError) as exc_info:
                _TelegramCredentials.from_env()
        assert "TELEGRAM_CHAT_ID" in str(exc_info.value)

    def test_from_env_success(self):
        from wallet_manager.telegram_notifier import _TelegramCredentials
        mock_cfg = MagicMock()
        mock_cfg.telegram_bot_token = "valid_token"
        mock_cfg.telegram_chat_id = "12345"
        with patch("wallet_manager.telegram_notifier._cfg", return_value=mock_cfg):
            creds = _TelegramCredentials.from_env()
        assert creds.bot_token == "valid_token"
        assert creds.chat_id == "12345"


class TestTokenBucket:
    """Tests for _TokenBucket rate limiter."""

    def test_raises_on_non_positive_rate(self):
        from wallet_manager.telegram_notifier import _TokenBucket
        with pytest.raises(ValueError):
            _TokenBucket(rate=0.0, capacity=5)

    def test_raises_on_negative_rate(self):
        from wallet_manager.telegram_notifier import _TokenBucket
        with pytest.raises(ValueError):
            _TokenBucket(rate=-1.0, capacity=5)

    @pytest.mark.asyncio
    async def test_acquire_returns_when_token_available(self):
        from wallet_manager.telegram_notifier import _TokenBucket
        bucket = _TokenBucket(rate=100.0, capacity=10)
        # Should complete immediately since we have 10 tokens
        await asyncio.wait_for(bucket.acquire(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_multiple_acquires_drain_tokens(self):
        from wallet_manager.telegram_notifier import _TokenBucket
        bucket = _TokenBucket(rate=100.0, capacity=5)
        # Drain 5 tokens (all available) — should not block
        for _ in range(5):
            await asyncio.wait_for(bucket.acquire(), timeout=1.0)

    def test_refill_adds_tokens_over_time(self):
        from wallet_manager.telegram_notifier import _TokenBucket
        bucket = _TokenBucket(rate=10.0, capacity=10)
        # Manually drain
        bucket._tokens = 0.0
        bucket._last_refill = time.monotonic() - 1.0  # 1 second ago
        bucket._refill()
        assert bucket._tokens > 0.0

    def test_refill_does_not_exceed_capacity(self):
        from wallet_manager.telegram_notifier import _TokenBucket
        bucket = _TokenBucket(rate=10.0, capacity=5)
        # simulate a long time passing
        bucket._last_refill = time.monotonic() - 1000.0
        bucket._refill()
        assert bucket._tokens <= 5.0


class TestTelegramClientInit:
    """Tests for _TelegramClient basic behavior (no network calls)."""

    def test_client_not_open_before_open_called(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="c1")
        client = _TelegramClient(creds)
        assert client._session is None

    @pytest.mark.asyncio
    async def test_send_message_raises_if_not_open(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="c1")
        client = _TelegramClient(creds)
        with pytest.raises(RuntimeError, match="not open"):
            await client.send_message("hello")

    @pytest.mark.asyncio
    async def test_send_message_with_keyboard_raises_if_not_open(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="c1")
        client = _TelegramClient(creds)
        with pytest.raises(RuntimeError, match="not open"):
            await client.send_message_with_keyboard("hello", {})

    @pytest.mark.asyncio
    async def test_get_updates_returns_empty_when_not_open(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="c1")
        client = _TelegramClient(creds)
        result = await client.get_updates(offset=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_answer_callback_returns_silently_if_not_open(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="c1")
        client = _TelegramClient(creds)
        # Should not raise, just return silently
        await client.answer_callback_query("cb_id_123")

    @pytest.mark.asyncio
    async def test_close_no_op_when_not_open(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok", _chat_id="c1")
        client = _TelegramClient(creds)
        # Should not raise
        await client.close()


class TestTelegramClientSendMessage:
    """Tests for _TelegramClient.send_message with mocked aiohttp."""

    def _make_client(self):
        from wallet_manager.telegram_notifier import _TelegramClient, _TelegramCredentials
        creds = _TelegramCredentials(_bot_token="tok123", _chat_id="chat99")
        client = _TelegramClient(creds)
        return client

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        client = self._make_client()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session
        await client.send_message("hello world")
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_truncates_long_text(self):
        client = self._make_client()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session
        long_text = "x" * 5000
        await client.send_message(long_text)
        call_args = mock_session.post.call_args
        sent_text = call_args[1]["json"]["text"] if "json" in call_args[1] else call_args[0][1]["text"]
        assert len(sent_text) <= 4096

    @pytest.mark.asyncio
    async def test_send_message_raises_on_4xx(self):
        from wallet_manager.exceptions import NotifierDeliveryError
        client = self._make_client()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session
        with pytest.raises(NotifierDeliveryError):
            await client.send_message("test msg")

    @pytest.mark.asyncio
    async def test_send_message_exhausts_retries_on_5xx(self):
        from wallet_manager.exceptions import NotifierDeliveryError
        client = self._make_client()
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NotifierDeliveryError):
                await client.send_message("test 5xx")


class TestTelegramNotifier:
    """Tests for TelegramNotifier (with mocked credentials and HTTP)."""

    def _make_creds(self):
        from wallet_manager.telegram_notifier import _TelegramCredentials
        return _TelegramCredentials(_bot_token="tok", _chat_id="chat1")

    def test_init_with_explicit_credentials(self):
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        assert n._credentials is creds

    def test_init_default_rate_limit(self):
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        assert n._rate_limiter._rate == 1.0

    def test_init_custom_rate_limit(self):
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(rate_limit=5.0, burst=3, credentials=creds)
        assert n._rate_limiter._rate == 5.0
        assert n._rate_limiter._capacity == 3.0

    @pytest.mark.asyncio
    async def test_send_raises_when_not_started(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        with pytest.raises(RuntimeError, match="not started"):
            await n.send(NotificationEvent(kind=EventKind.SYSTEM_BOOT))

    @pytest.mark.asyncio
    async def test_send_event_alias(self):
        """send_event should alias send."""
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        n.send = AsyncMock()
        event = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
        await n.send_event(event)
        n.send.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        # Mock the HTTP client so no real network is needed
        n._client.open = AsyncMock()
        n._client.close = AsyncMock()
        n._client.send_message = AsyncMock()

        async with n:
            event = NotificationEvent(kind=EventKind.SYSTEM_BOOT)
            await n.send(event)

        n._client.open.assert_awaited_once()
        n._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_build_command_center_returns_command_center(self):
        from wallet_manager.security.telegram_commands import CommandCenter, CommandState
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        state = CommandState()
        center = n.build_command_center(state)
        assert isinstance(center, CommandCenter)

    @pytest.mark.asyncio
    async def test_send_confirmation_request_raises_when_not_started(self):
        from wallet_manager.telegram_notifier import TelegramNotifier
        creds = self._make_creds()
        n = TelegramNotifier(credentials=creds)
        with pytest.raises(RuntimeError, match="not started"):
            await n.send_confirmation_request("wallet1", "0.01 ETH", "ETH -> USDC")


class TestProactiveAlerter:
    """Tests for ProactiveAlerter class."""

    def _settings_patch(self, **overrides):
        s = MagicMock()
        s.yield_min_apy_pct = 1.0
        s.drawdown_reduce_threshold = 0.05
        s.alert_dedup_window_seconds = 3600.0
        for k, v in overrides.items():
            setattr(s, k, v)
        return patch("wallet_manager.config.get_settings", return_value=s)

    @pytest.mark.asyncio
    async def test_no_alerts_nominal(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch():
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="NOMINAL",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.01,
            )
        notifier.send_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_severe_drift_triggers_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch():
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="SEVERE",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.0,
            )
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_moderate_drift_triggers_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch():
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="MODERATE",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.0,
            )
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_low_apy_triggers_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        # yield_min_apy_pct = 3.0, apy = 0.01 (= 1%) — below threshold
        with self._settings_patch(yield_min_apy_pct=3.0):
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="NOMINAL",
                apy=0.01,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.0,
            )
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_stale_model_triggers_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch():
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="NOMINAL",
                apy=0.05,
                model_staleness_days=10.0,
                rpc_latency_ms=100.0,
                drawdown=0.0,
            )
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_high_rpc_latency_triggers_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch():
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="NOMINAL",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=600.0,
                drawdown=0.0,
            )
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_drawdown_exceeded_triggers_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch(drawdown_reduce_threshold=0.05):
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="NOMINAL",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.10,
            )
        notifier.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_deduplication_prevents_second_alert(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        with self._settings_patch():
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="SEVERE",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.0,
            )
            # call again within dedup window — should not fire again
            await alerter.check_and_alert(
                notifier=notifier,
                drift_level="SEVERE",
                apy=0.05,
                model_staleness_days=1.0,
                rpc_latency_ms=100.0,
                drawdown=0.0,
            )
        assert notifier.send_event.call_count == 1

    def test_is_deduped_false_when_key_not_seen(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        assert alerter._is_deduped("new_key", 3600.0) is False

    def test_is_deduped_true_after_mark_sent(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        alerter._mark_sent("test_key")
        assert alerter._is_deduped("test_key", 3600.0) is True

    def test_is_deduped_false_after_window_expired(self):
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        # Mark sent 2 hours ago
        alerter._last_sent["old_key"] = time.time() - 7201.0
        assert alerter._is_deduped("old_key", 3600.0) is False

    @pytest.mark.asyncio
    async def test_dispatch_alert_uses_send_event(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock()
        notifier.send_event = AsyncMock()
        event = NotificationEvent(kind=EventKind.SYSTEM_ALERT, detail="test")
        await alerter._dispatch_alert(notifier, event)
        notifier.send_event.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_alert_falls_back_to_send(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = MagicMock(spec=[])  # no send_event attribute
        notifier.send = AsyncMock()
        event = NotificationEvent(kind=EventKind.SYSTEM_ALERT, detail="test")
        await alerter._dispatch_alert(notifier, event)
        notifier.send.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_alert_raises_if_no_sender(self):
        from wallet_manager.core.models import EventKind, NotificationEvent
        from wallet_manager.telegram_notifier import ProactiveAlerter
        alerter = ProactiveAlerter()
        notifier = object()  # no send_event or send
        event = NotificationEvent(kind=EventKind.SYSTEM_ALERT, detail="test")
        with pytest.raises(AttributeError):
            await alerter._dispatch_alert(notifier, event)


# ===========================================================================
# Section 6 — tick_collector.py
# ===========================================================================

class TestTickCollectorInit:
    """Tests for TickCollector initialisation."""

    def test_default_db_path(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        assert tc._db_path == tmp_path / "ticks.db"

    def test_initial_state(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        assert tc._last_price == 0.0
        assert tc._tick_count == 0
        assert tc._initialized is False
        assert tc._pending == []

    def test_flush_interval_default(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        assert tc._flush_interval == 50


class TestTickCollectorRecordTick:
    """Tests for TickCollector.record_tick."""

    def test_ignores_non_positive_price(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(0.0)
        assert len(tc._pending) == 0
        tc.record_tick(-1.0)
        assert len(tc._pending) == 0

    def test_records_valid_price(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(1000.0)
        assert len(tc._pending) == 1
        assert tc._tick_count == 1

    def test_uptick_direction(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(1000.0)
        tc.record_tick(1100.0)
        assert tc._pending[-1][2] == 1  # direction +1

    def test_downtick_direction(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(1000.0)
        tc.record_tick(900.0)
        assert tc._pending[-1][2] == -1  # direction -1

    def test_unchanged_direction(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(1000.0)
        tc.record_tick(1000.0)
        assert tc._pending[-1][2] == 0  # unchanged

    def test_first_tick_direction_zero(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(500.0)
        assert tc._pending[0][2] == 0  # no prior price

    def test_updates_last_price(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        tc.record_tick(2500.0)
        assert tc._last_price == 2500.0

    def test_tick_count_increments(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        for price in [100, 200, 300]:
            tc.record_tick(float(price))
        assert tc._tick_count == 3


class TestTickCollectorFlushAndInit:
    """Tests for TickCollector flush/initialize."""

    @pytest.mark.asyncio
    async def test_flush_empty_returns_zero(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        count = await tc.flush()
        assert count == 0

    @pytest.mark.asyncio
    async def test_initialize_creates_db(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        db_path = tmp_path / "test_ticks.db"
        tc = TickCollector(db_path=db_path)
        await tc.initialize()
        assert tc._initialized is True
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_flush_writes_pending_ticks(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        tc.record_tick(1000.0)
        tc.record_tick(1100.0)
        count = await tc.flush()
        assert count == 2
        assert tc._pending == []

    @pytest.mark.asyncio
    async def test_flush_auto_initializes(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "auto_init.db")
        tc.record_tick(1234.0)
        count = await tc.flush()
        assert count == 1
        assert tc._initialized is True

    @pytest.mark.asyncio
    async def test_maybe_flush_below_threshold(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        tc.record_tick(100.0)
        await tc.maybe_flush()
        # Below threshold of 50, pending should still have 1 item
        assert len(tc._pending) == 1

    @pytest.mark.asyncio
    async def test_maybe_flush_at_threshold(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        for i in range(50):
            tc.record_tick(float(100 + i))
        await tc.maybe_flush()
        assert tc._pending == []


class TestTickCollectorQueryMethods:
    """Tests for TickCollector query and analytics methods."""

    @pytest.mark.asyncio
    async def test_get_recent_ticks_not_initialized(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        result = await tc.get_recent_ticks(60.0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_ofi_insufficient_data(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        ofi = await tc.get_ofi(5.0)
        assert ofi == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_not_initialized(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        stats = await tc.get_stats()
        assert stats["initialized"] is False
        assert stats["total_ticks"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_initialized(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        stats = await tc.get_stats()
        assert stats["initialized"] is True
        assert "total_ticks_db" in stats
        assert "ofi_ready" in stats

    @pytest.mark.asyncio
    async def test_prune_old_ticks_not_initialized_returns_zero(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        deleted = await tc.prune_old_ticks()
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_prune_old_ticks_initialized(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        deleted = await tc.prune_old_ticks()
        assert deleted == 0  # nothing to prune in a fresh DB

    @pytest.mark.asyncio
    async def test_get_ofi_features_returns_dict(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        features = tc.get_ofi_features()
        assert isinstance(features, dict)

    @pytest.mark.asyncio
    async def test_get_stats_ofi_signal_neutral_empty_db(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        stats = await tc.get_stats()
        assert stats["ofi_signal"] == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_get_recent_ticks_returns_list(self, tmp_path):
        from wallet_manager.tick_collector import TickCollector
        tc = TickCollector(db_path=tmp_path / "ticks.db")
        await tc.initialize()
        # Insert a tick and flush
        tc.record_tick(1500.0)
        await tc.flush()
        result = await tc.get_recent_ticks(60.0)
        assert isinstance(result, list)


class TestGetTickCollectorSingleton:
    """Tests for get_tick_collector() singleton factory."""

    def test_singleton_returns_same_instance(self):
        import wallet_manager.tick_collector as mod
        # Reset singleton so we get a fresh one for the test
        original = mod._collector
        mod._collector = None
        try:
            a = mod.get_tick_collector()
            b = mod.get_tick_collector()
            assert a is b
        finally:
            mod._collector = original

    def test_singleton_respects_env_var(self, tmp_path, monkeypatch):
        import wallet_manager.tick_collector as mod
        original = mod._collector
        mod._collector = None
        monkeypatch.setenv("TICK_DB_PATH", str(tmp_path / "custom.db"))
        try:
            tc = mod.get_tick_collector()
            assert "custom.db" in str(tc._db_path)
        finally:
            mod._collector = original
