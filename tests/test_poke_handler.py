from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nonebot.adapters.onebot.v11 import MessageSegment, PokeNotifyEvent

from nekro_plugin_semantic_sticker import plugin
from nekro_plugin_semantic_sticker.models import ReplyMode, StickerSendResult, UsageContext


def load_poke_handler():
    return importlib.import_module("nekro_plugin_semantic_sticker.poke_handler")


def poke_event(*, target_id: int = 123456, group_id: int | None = 100) -> PokeNotifyEvent:
    return PokeNotifyEvent(
        time=1_786_000_000,
        self_id=123456,
        post_type="notice",
        notice_type="notify",
        sub_type="poke",
        user_id=42,
        target_id=target_id,
        group_id=group_id,
        raw_info=[{"type": "poke"}],
    )


class FakeDirectService:
    def __init__(self, asset_path: Path | None = None, *, result: StickerSendResult | None = None) -> None:
        self.asset_path = asset_path
        self.result = result or {
            "sent": True,
            "sticker_id": "11111111-1111-4111-8111-111111111111",
            "reason": "sent",
            "reply_mode": ReplyMode.IMAGE_ONLY.value,
            "score": 0.91,
        }
        self.calls: list[dict[str, object]] = []

    async def execute_send_direct(
        self,
        usage_context: UsageContext,
        reservation_owner: object,
        intent: str,
        reply_mode: ReplyMode,
        deliver,
    ) -> StickerSendResult:
        self.calls.append({
            "usage_context": usage_context,
            "reservation_owner": reservation_owner,
            "intent": intent,
            "reply_mode": reply_mode,
            "deliver": deliver,
        })
        if self.asset_path is not None:
            await deliver(self.asset_path)
        return self.result


@pytest.mark.asyncio
async def test_matcher_only_blocks_pokes_targeting_the_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    poke_handler = load_poke_handler()
    service = FakeDirectService(result={
        "sent": False,
        "sticker_id": None,
        "reason": "no eligible sticker",
        "reply_mode": ReplyMode.IMAGE_ONLY.value,
        "score": None,
    })
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)
    matcher = poke_handler.bot_poke_matcher
    bot = SimpleNamespace(send=AsyncMock())

    blocked = await matcher.run(bot=bot, event=poke_event())
    not_blocked = await matcher.run(bot=bot, event=poke_event(target_id=99))

    assert matcher.event_type is PokeNotifyEvent
    assert matcher.priority == 1
    assert matcher.block is True
    assert list(inspect.signature(poke_handler._is_bot_target).parameters) == ["event"]
    assert blocked is True
    assert not_blocked is False
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_group_poke_directly_sends_image_bytes_with_user_scoped_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poke_handler = load_poke_handler()
    asset_path = tmp_path / "poke.webp"
    asset_path.write_bytes(b"poke-image")
    service = FakeDirectService(asset_path)
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)
    bot = SimpleNamespace(send=AsyncMock())
    event = poke_event(group_id=100)

    blocked = await poke_handler.bot_poke_matcher.run(bot=bot, event=event)

    assert blocked is True
    call = service.calls[0]
    usage_context = call["usage_context"]
    assert usage_context.logical_chat_key == "onebot_v11-group_100-user_42"
    assert usage_context.physical_channel_key == "onebot_v11-group_100"
    assert usage_context.agent_turn_key.startswith("onebot_v11-poke-group_100-user_42-at_1786000000-")
    assert call["reservation_owner"] is event
    assert "戳一戳" in str(call["intent"])
    assert call["reply_mode"] is ReplyMode.IMAGE_ONLY
    bot.send.assert_awaited_once()
    sent_event, segment = bot.send.await_args.args
    assert sent_event is event
    assert isinstance(segment, MessageSegment)
    assert segment.type == "image"
    assert segment.data == {"file": "base64://cG9rZS1pbWFnZQ=="}


@pytest.mark.asyncio
async def test_private_poke_uses_sender_scoped_private_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    poke_handler = load_poke_handler()
    service = FakeDirectService(result={
        "sent": False,
        "sticker_id": None,
        "reason": "physical channel is busy or cooling down",
        "reply_mode": ReplyMode.IMAGE_ONLY.value,
        "score": None,
    })
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)

    await poke_handler.handle_bot_poke(SimpleNamespace(send=AsyncMock()), poke_event(group_id=None))

    usage_context = service.calls[0]["usage_context"]
    assert usage_context.logical_chat_key == "onebot_v11-private_42"
    assert usage_context.physical_channel_key == "onebot_v11-private_42"
    assert usage_context.agent_turn_key.startswith("onebot_v11-poke-private_42-at_1786000000-")


def test_same_sender_same_second_gets_unique_turn_keys() -> None:
    poke_handler = load_poke_handler()
    event = poke_event()

    first = poke_handler._usage_context(event)
    second = poke_handler._usage_context(event)

    assert first.logical_chat_key == second.logical_chat_key
    assert first.physical_channel_key == second.physical_channel_key
    assert first.agent_turn_key != second.agent_turn_key


def test_poke_event_stub_matches_notice_shape_and_keeps_extra_fields() -> None:
    event = poke_event()

    assert (event.post_type, event.notice_type, event.sub_type) == ("notice", "notify", "poke")
    assert event.raw_info == [{"type": "poke"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["no eligible sticker", "physical channel is busy or cooling down"])
async def test_no_candidate_or_cooldown_stays_silent_without_fallback(
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poke_handler = load_poke_handler()
    service = FakeDirectService(result={
        "sent": False,
        "sticker_id": None,
        "reason": reason,
        "reply_mode": ReplyMode.IMAGE_ONLY.value,
        "score": None,
    })
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)
    bot = SimpleNamespace(send=AsyncMock())

    await poke_handler.handle_bot_poke(bot, poke_event())

    bot.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_failure_is_logged_and_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    poke_handler = load_poke_handler()
    service = FakeDirectService(result={
        "sent": False,
        "sticker_id": None,
        "reason": "image send failed",
        "reply_mode": ReplyMode.IMAGE_ONLY.value,
        "score": None,
    })
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)
    bot = SimpleNamespace(send=AsyncMock())

    with caplog.at_level(logging.WARNING):
        await poke_handler.handle_bot_poke(bot, poke_event())

    bot.send.assert_not_awaited()
    assert "image send failed" in caplog.text


@pytest.mark.asyncio
async def test_real_delivery_exception_is_caught_and_matcher_still_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    poke_handler = load_poke_handler()
    asset_path = tmp_path / "poke.png"
    asset_path.write_bytes(b"poke-image")
    service = FakeDirectService(asset_path)
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)
    bot = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("onebot send failed")))

    with caplog.at_level(logging.ERROR):
        blocked = await poke_handler.bot_poke_matcher.run(bot=bot, event=poke_event())

    assert blocked is True
    bot.send.assert_awaited_once()
    assert "处理 Bot 戳一戳表情回复失败" in caplog.text


@pytest.mark.asyncio
async def test_handler_logs_unexpected_errors_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    poke_handler = load_poke_handler()
    service = SimpleNamespace(execute_send_direct=AsyncMock(side_effect=RuntimeError("service unavailable")))
    monkeypatch.setattr(poke_handler, "get_service", lambda: service)

    with caplog.at_level(logging.ERROR):
        await poke_handler.handle_bot_poke(SimpleNamespace(send=AsyncMock()), poke_event())

    assert "处理 Bot 戳一戳表情回复失败" in caplog.text


def test_poke_handler_imports_no_agent_llm_or_message_paths() -> None:
    poke_handler = load_poke_handler()
    source = inspect.getsource(poke_handler)
    forbidden = {
        "AgentCtx",
        "ChatMessage",
        "message_service",
        "push_human_message",
        "schedule_agent_task",
        "send_agent_request",
        "quota",
        "LLM",
    }

    assert forbidden.isdisjoint(source.split())


def test_plugin_version_and_runtime_import_order_include_poke_handler() -> None:
    package = importlib.import_module("nekro_plugin_semantic_sticker")
    source = inspect.getsource(package)

    assert plugin.version == "1.2.4"
    assert source.index('agent_tools = import_module') < source.index('poke_handler = import_module')
    assert package.poke_handler is load_poke_handler()
