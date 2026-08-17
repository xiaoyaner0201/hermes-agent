"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    sender_scoped_message_event_key,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_event(
    text: str,
    platform: Platform,
    chat_id: str = "12345",
    msg_type: MessageType = MessageType.TEXT,
    user_id: str | None = None,
    chat_type: str = "dm",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
        ),
    )


# =====================================================================
# Shared sender boundary
# =====================================================================


def test_sender_scoped_message_event_key_separates_participants():
    alice = _make_event(
        "Alice text", Platform.DISCORD, user_id="alice", chat_type="group"
    )
    bob = _make_event(
        "Bob text", Platform.DISCORD, user_id="bob", chat_type="group"
    )

    assert sender_scoped_message_event_key("shared-session", alice) != (
        sender_scoped_message_event_key("shared-session", bob)
    )


def test_sender_scoped_message_event_key_fails_closed_without_sender():
    first = _make_event("one", Platform.DISCORD, chat_type="group")
    second = _make_event("two", Platform.DISCORD, chat_type="group")

    assert sender_scoped_message_event_key("shared-session", first) != (
        sender_scoped_message_event_key("shared-session", second)
    )


def test_all_preingress_text_batch_keys_are_sender_scoped():
    """Every adapter batch key must restore the sender omitted by shared sessions."""
    import inspect

    targets = [
        "gateway.platforms.weixin.WeixinAdapter",
        "plugins.platforms.discord.adapter.DiscordAdapter",
        "plugins.platforms.feishu.adapter.FeishuAdapter",
        "plugins.platforms.matrix.adapter.MatrixAdapter",
        "plugins.platforms.simplex.adapter.SimplexAdapter",
        "plugins.platforms.telegram.adapter.TelegramAdapter",
        "plugins.platforms.wecom.adapter.WeComAdapter",
        "plugins.platforms.whatsapp.adapter.WhatsAppAdapter",
    ]
    for dotted in targets:
        module_name, class_name = dotted.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        source = inspect.getsource(cls._text_batch_key)
        assert "sender_scoped_message_event_key" in source, dotted


def test_preingress_batch_keys_separate_shared_session_senders():
    """Exercise every platform key with the same session and two senders."""
    targets = [
        ("gateway.platforms.weixin.WeixinAdapter", Platform.WEIXIN),
        ("plugins.platforms.discord.adapter.DiscordAdapter", Platform.DISCORD),
        ("plugins.platforms.feishu.adapter.FeishuAdapter", Platform.FEISHU),
        ("plugins.platforms.matrix.adapter.MatrixAdapter", Platform.MATRIX),
        ("plugins.platforms.simplex.adapter.SimplexAdapter", Platform.LOCAL),
        ("plugins.platforms.telegram.adapter.TelegramAdapter", Platform.TELEGRAM),
        ("plugins.platforms.wecom.adapter.WeComAdapter", Platform.WECOM),
        ("plugins.platforms.whatsapp.adapter.WhatsAppAdapter", Platform.WHATSAPP),
    ]
    for dotted, platform in targets:
        module_name, class_name = dotted.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        adapter = object.__new__(cls)
        adapter.config = PlatformConfig(
            enabled=True,
            token="fake",
            extra={
                "group_sessions_per_user": False,
                "thread_sessions_per_user": False,
            },
        )
        if class_name == "TelegramAdapter":
            adapter._apply_topic_recovery = lambda event: None
        alice = _make_event(
            "Alice text", platform, user_id="alice", chat_type="group"
        )
        bob = _make_event(
            "Bob text", platform, user_id="bob", chat_type="group"
        )
        assert adapter._text_batch_key(alice) != adapter._text_batch_key(bob), dotted


# =====================================================================
# Discord text batching
# =====================================================================

def _make_discord_adapter():
    """Create a minimal DiscordAdapter for testing text batching."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._text_batch_split_delay_seconds = 0.3  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestDiscordTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_discord_adapter()
        event = _make_event("hello world", Platform.DISCORD)

        adapter._enqueue_text_event(event)

        # Not dispatched yet
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        """Two rapid messages from the same chat should be merged."""
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("Part one of a long", Platform.DISCORD))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("message that was split.", Platform.DISCORD))

        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "Part one" in text
        assert "split" in text

    @pytest.mark.asyncio
    async def test_shared_session_messages_from_different_senders_are_not_merged(self):
        adapter = _make_discord_adapter()
        mock_handler = AsyncMock()
        adapter.handle_message = mock_handler
        alice = _make_event(
            "Alice text", Platform.DISCORD, user_id="alice", chat_type="group"
        )
        bob = _make_event(
            "Bob text", Platform.DISCORD, user_id="bob", chat_type="group"
        )

        assert adapter._text_batch_key(alice) != adapter._text_batch_key(bob)
        adapter._enqueue_text_event(alice)
        adapter._enqueue_text_event(bob)
        await asyncio.sleep(0.2)

        assert mock_handler.call_count == 2
        dispatched = [call.args[0] for call in mock_handler.call_args_list]
        assert {(event.source.user_id, event.text) for event in dispatched} == {
            ("alice", "Alice text"),
            ("bob", "Bob text"),
        }


# =====================================================================
# Matrix text batching
# =====================================================================

def _make_matrix_adapter():
    """Create a minimal MatrixAdapter for testing text batching."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(MatrixAdapter)
    adapter._platform = Platform.MATRIX
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestMatrixTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_matrix_adapter()
        event = _make_event("hello world", Platform.MATRIX)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_matrix_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.MATRIX))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.MATRIX))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


# =====================================================================
# WeCom text batching
# =====================================================================

def _make_wecom_adapter():
    """Create a minimal WeComAdapter for testing text batching."""
    from plugins.platforms.wecom.adapter import WeComAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(WeComAdapter)
    adapter._platform = Platform.WECOM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestWeComTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_wecom_adapter()
        event = _make_event("hello world", Platform.WECOM)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_wecom_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.WECOM))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.WECOM))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


# =====================================================================
# Telegram adaptive delay (PR #6891)
# =====================================================================

def _make_telegram_adapter():
    """Create a minimal TelegramAdapter for testing adaptive delay."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestTelegramAdaptiveDelay:
    @pytest.mark.asyncio
    async def test_short_chunk_uses_normal_delay(self):
        adapter = _make_telegram_adapter()
        adapter._enqueue_text_event(_make_event("short msg", Platform.TELEGRAM))

        # Should flush after the normal 0.1s delay
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_called_once()


# =====================================================================
# Feishu adaptive delay
# =====================================================================

def _make_feishu_adapter():
    """Create a minimal FeishuAdapter for testing adaptive delay."""
    from plugins.platforms.feishu.adapter import FeishuAdapter, FeishuBatchState

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(FeishuAdapter)
    adapter._platform = Platform.FEISHU
    adapter.config = config
    batch_state = FeishuBatchState()
    adapter._pending_text_batches = batch_state.events
    adapter._pending_text_batch_tasks = batch_state.tasks
    adapter._pending_text_batch_counts = batch_state.counts
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._text_batch_max_messages = 20
    adapter._text_batch_max_chars = 50000
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter._handle_message_with_guards = AsyncMock()
    return adapter


class TestFeishuAdaptiveDelay:
    @pytest.mark.asyncio
    async def test_short_chunk_uses_normal_delay(self):
        adapter = _make_feishu_adapter()
        event = _make_event("short msg", Platform.FEISHU)
        await adapter._enqueue_text_event(event)

        await asyncio.sleep(0.15)
        adapter._handle_message_with_guards.assert_called_once()


