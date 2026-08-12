from __future__ import annotations

import asyncio
import gc
import inspect
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, Mock

import pytest

from nekro_plugin_semantic_sticker.models import ReplyMode, StickerCandidate, UsageContext


class FakeRepository:
    def __init__(self) -> None:
        self.sent_turns: set[str] = set()
        self.cooldown_channels: set[str] = set()
        self.fail_usage = False
        self.record_usage = AsyncMock(side_effect=self._record_usage)

    async def agent_turn_has_send(self, turn_key: str) -> bool:
        return turn_key in self.sent_turns

    async def physical_channel_in_cooldown(self, physical_channel_key: str, *, cooldown_seconds: int) -> bool:
        return physical_channel_key in self.cooldown_channels

    async def _record_usage(
        self,
        sticker_id: str,
        logical_chat_key: str,
        physical_channel_key: str,
        agent_turn_key: str,
        score: float,
    ) -> None:
        if self.fail_usage:
            raise RuntimeError("usage database unavailable")
        self.sent_turns.add(agent_turn_key)
        self.cooldown_channels.add(physical_channel_key)

class FakeRetriever:
    def __init__(self, candidate: StickerCandidate | None) -> None:
        self.candidate = candidate
        self.calls: list[tuple[str, UsageContext]] = []
        self.error: BaseException | None = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def find(self, intent: str, usage_context: UsageContext) -> StickerCandidate | None:
        self.calls.append((intent, usage_context))
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.candidate


class FakeMonotonic:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

class FakeResolver:
    def resolve(self, ctx: object) -> UsageContext:
        return UsageContext(
            logical_chat_key=ctx.chat_key,
            physical_channel_key="onebot_v11-group_100",
            agent_turn_key=ctx.turn_key,
        )


def candidate(path: Path) -> StickerCandidate:
    return StickerCandidate(
        sticker_id="11111111-1111-4111-8111-111111111111",
        vector_score=0.91,
        asset_path=str(path),
        primary_category="confusion",
        emotion_tags=["confused"],
        scene_tags=["asking why"],
        ocr_text="?",
    )


def fake_ctx(tmp_path: Path, *, turn_key: str = "turn-1", fail_send: bool = False):
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    forwarded: list[Path] = []

    def forward_file(path: Path) -> str:
        forwarded.append(Path(path))
        assert Path(path).is_file()
        assert Path(path).parent == shared
        return f"sandbox:{Path(path).name}"

    async def send_image(path: str, *, record: bool = True) -> None:
        assert path.startswith("sandbox:")
        assert record is True
        assert forwarded[-1].is_file()
        if fail_send:
            raise RuntimeError("send failed")

    return SimpleNamespace(
        chat_key="onebot_v11-group_100-user_1",
        turn_key=turn_key,
        fs=SimpleNamespace(shared_path=shared, forward_file=Mock(side_effect=forward_file)),
        send_image=AsyncMock(side_effect=send_image),
        forwarded=forwarded,
    )


def test_tool_signature_allows_only_three_modes() -> None:
    from nekro_plugin_semantic_sticker import agent_tools

    parameter = inspect.signature(agent_tools.send_matching_sticker).parameters["reply_mode"]
    assert set(get_args(parameter.annotation)) == {"image_only", "text_then_image", "auto"}
    assert parameter.default == "auto"


def test_agent_tools_have_prompt_visible_docstrings() -> None:
    from nekro_plugin_semantic_sticker import agent_tools

    send_doc = inspect.getdoc(agent_tools.send_matching_sticker)
    save_doc = inspect.getdoc(agent_tools.save_sticker_from_message)

    assert send_doc and "reply_mode" in send_doc
    assert save_doc and "image_scope" in save_doc
    assert "user_request" in save_doc
    assert "automatic" in save_doc


@pytest.mark.asyncio
async def test_prompt_exposes_explicit_save_and_respects_automatic_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    from nekro_plugin_semantic_sticker import agent_tools

    monkeypatch.setattr(agent_tools.config, "AUTO_COLLECT_ENABLED", False)
    monkeypatch.setattr(agent_tools.config, "STRICT_EMOTION_COLLECT", True)
    prompt = (await agent_tools.semantic_sticker_prompt(SimpleNamespace())).lower()

    assert "optional" in prompt
    assert "text only" in prompt
    assert "image_only" in prompt and "text_then_image" in prompt
    assert "explicitly asks" in prompt
    assert "automatic sticker collection is disabled" in prompt
    assert "strict" in prompt and "screenshot" in prompt


def test_save_tool_signature_has_bounded_image_selection_only() -> None:
    from nekro_plugin_semantic_sticker import agent_tools

    parameters = inspect.signature(agent_tools.save_sticker_from_message).parameters
    assert list(parameters) == [
        "_ctx",
        "image_scope",
        "image_index",
        "save_reason",
        "vision_and_sticker_confirmed",
    ]
    forbidden = {"source_path", "url", "image", "segment", "attachment", "file_path"}
    assert forbidden.isdisjoint(parameters)
    assert set(get_args(parameters["image_scope"].annotation)) == {"auto", "current", "reply"}
    assert set(get_args(parameters["save_reason"].annotation)) == {"user_request", "automatic"}


@pytest.mark.asyncio
async def test_user_requested_save_uses_current_message_image_when_automatic_collection_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nekro_plugin_semantic_sticker import agent_tools
    from nekro_plugin_semantic_sticker.message_images import message_image_registry

    source = tmp_path / "current.png"
    source.write_bytes(b"current-image")
    ctx = SimpleNamespace(chat_key="onebot_v11-group_100-user_42", from_platform_userid="42")
    message = SimpleNamespace(
        adapter_key="onebot_v11",
        chat_key=ctx.chat_key,
        content_data=[SimpleNamespace(type="image", local_path=str(source), file_name=source.name, remote_url=None)],
        ext_data={},
    )
    message_image_registry.clear()
    await message_image_registry.remember(ctx, message)
    outcome = SimpleNamespace(duplicate=False, record=SimpleNamespace(id="sticker-1"))
    service = SimpleNamespace(upload=AsyncMock(return_value=outcome))
    agent_tools.set_service_for_testing(service)
    monkeypatch.setattr(agent_tools.config, "AUTO_COLLECT_ENABLED", False)
    monkeypatch.setattr(agent_tools.config, "STRICT_EMOTION_COLLECT", True)

    try:
        result = await agent_tools.save_sticker_from_message(
            ctx,
            image_scope="current",
            image_index=0,
            save_reason="user_request",
            vision_and_sticker_confirmed=True,
        )
    finally:
        agent_tools.set_service_for_testing(None)
        message_image_registry.clear()

    assert "已提交" in result
    payload = service.upload.await_args.args[0]
    assert payload.content == b"current-image"
    assert payload.filename == "current.png"
    service.upload.assert_awaited_once_with(payload, actor="chat:42")


@pytest.mark.asyncio
async def test_save_tool_enforces_automatic_toggle_and_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nekro_plugin_semantic_sticker import agent_tools
    from nekro_plugin_semantic_sticker.message_images import message_image_registry

    source = tmp_path / "reply.gif"
    source.write_bytes(b"reply-image")
    ctx = SimpleNamespace(chat_key="onebot_v11-private_42", from_platform_userid="42")

    async def load_reference(_adapter_key: str, chat_key: str, _message_id: str) -> object:
        assert chat_key == ctx.chat_key
        return SimpleNamespace(
            content_data=[SimpleNamespace(type="image", local_path=str(source), file_name=source.name, remote_url=None)],
        )

    monkeypatch.setattr(message_image_registry, "_reference_loader", load_reference)
    message_image_registry.clear()
    await message_image_registry.remember(
        ctx,
        SimpleNamespace(
            adapter_key="onebot_v11",
            chat_key=ctx.chat_key,
            content_data=[],
            ext_data={"ref_msg_id": "quoted-1"},
        ),
    )
    service = SimpleNamespace(
        upload=AsyncMock(return_value=SimpleNamespace(duplicate=True, record=SimpleNamespace(id="sticker-2"))),
    )
    agent_tools.set_service_for_testing(service)
    monkeypatch.setattr(agent_tools.config, "AUTO_COLLECT_ENABLED", False)
    monkeypatch.setattr(agent_tools.config, "STRICT_EMOTION_COLLECT", True)

    try:
        automatic = await agent_tools.save_sticker_from_message(
            ctx,
            image_scope="reply",
            save_reason="automatic",
            vision_and_sticker_confirmed=True,
        )
        strict = await agent_tools.save_sticker_from_message(
            ctx,
            image_scope="reply",
            save_reason="user_request",
            vision_and_sticker_confirmed=False,
        )
        monkeypatch.setattr(agent_tools.config, "STRICT_EMOTION_COLLECT", False)
        accepted = await agent_tools.save_sticker_from_message(
            ctx,
            image_scope="reply",
            save_reason="user_request",
            vision_and_sticker_confirmed=False,
        )
    finally:
        agent_tools.set_service_for_testing(None)
        message_image_registry.clear()

    assert "自动保存已关闭" in automatic
    assert "严格模式" in strict
    assert "已经保存过" in accepted
    service.upload.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_tool_reports_missing_current_or_reply_image() -> None:
    from nekro_plugin_semantic_sticker import agent_tools
    from nekro_plugin_semantic_sticker.message_images import message_image_registry

    ctx = SimpleNamespace(chat_key="onebot_v11-private_9", from_platform_userid="9")
    message_image_registry.clear()

    result = await agent_tools.save_sticker_from_message(
        ctx,
        image_scope="auto",
        save_reason="user_request",
        vision_and_sticker_confirmed=True,
    )

    assert "没有找到" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [ReplyMode.IMAGE_ONLY, ReplyMode.TEXT_THEN_IMAGE, ReplyMode.AUTO])
async def test_sender_copies_forwards_sends_and_records_after_success(tmp_path: Path, mode: ReplyMode) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.gif"
    source.write_bytes(b"gif-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    ctx = fake_ctx(tmp_path, turn_key=f"turn-{mode.value}")
    executor = StickerSendExecutor(retriever, repository, FakeResolver())

    result = await executor(ctx, " confusion ", mode)

    assert result == {
        "sent": True,
        "sticker_id": "11111111-1111-4111-8111-111111111111",
        "reason": "sent",
        "reply_mode": mode.value,
        "score": 0.91,
    }
    assert retriever.calls[0][0] == "confusion"
    ctx.fs.forward_file.assert_called_once()
    ctx.send_image.assert_awaited_once()
    repository.record_usage.assert_awaited_once()
    assert ctx.forwarded[0].suffix == ".gif"
    assert ctx.forwarded[0].exists() is False


@pytest.mark.asyncio
async def test_direct_sender_uses_prebuilt_identity_and_delivery_callback(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.webp"
    source.write_bytes(b"webp-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    executor = StickerSendExecutor(retriever, repository, FakeResolver())
    usage_context = UsageContext(
        logical_chat_key="onebot_v11-group_100-user_42",
        physical_channel_key="onebot_v11-group_100",
        agent_turn_key="poke-turn-1",
    )
    reservation_owner = object()
    deliver = AsyncMock()

    result = await executor.execute_direct(
        usage_context,
        reservation_owner,
        " confusion ",
        ReplyMode.IMAGE_ONLY,
        deliver,
    )

    assert result == {
        "sent": True,
        "sticker_id": "11111111-1111-4111-8111-111111111111",
        "reason": "sent",
        "reply_mode": ReplyMode.IMAGE_ONLY.value,
        "score": 0.91,
    }
    assert retriever.calls == [("confusion", usage_context)]
    deliver.assert_awaited_once_with(source.resolve())
    repository.record_usage.assert_awaited_once_with(
        "11111111-1111-4111-8111-111111111111",
        usage_context.logical_chat_key,
        usage_context.physical_channel_key,
        usage_context.agent_turn_key,
        0.91,
    )


@pytest.mark.asyncio
async def test_direct_delivery_failure_releases_shared_reservation(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    executor = StickerSendExecutor(retriever, repository, FakeResolver())
    first_context = UsageContext(
        logical_chat_key="onebot_v11-group_100-user_42",
        physical_channel_key="onebot_v11-group_100",
        agent_turn_key="poke-turn-1",
    )
    retry_context = first_context.model_copy(update={"agent_turn_key": "poke-turn-2"})
    failing_delivery = AsyncMock(side_effect=RuntimeError("send failed"))
    successful_delivery = AsyncMock()

    failed = await executor.execute_direct(
        first_context,
        object(),
        "confusion",
        ReplyMode.IMAGE_ONLY,
        failing_delivery,
    )
    retried = await executor.execute_direct(
        retry_context,
        object(),
        "confusion",
        ReplyMode.IMAGE_ONLY,
        successful_delivery,
    )

    assert failed["sent"] is False
    assert failed["reason"] == "image send failed"
    assert retried["sent"] is True
    successful_delivery.assert_awaited_once_with(source.resolve())
    assert repository.record_usage.await_count == 1


@pytest.mark.asyncio
async def test_direct_and_agent_paths_share_physical_channel_cooldown(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    executor = StickerSendExecutor(retriever, repository, FakeResolver(), cooldown_seconds=20)
    usage_context = UsageContext(
        logical_chat_key="onebot_v11-group_100-user_42",
        physical_channel_key="onebot_v11-group_100",
        agent_turn_key="poke-turn-1",
    )
    ctx = fake_ctx(tmp_path / "agent", turn_key="agent-turn-2")

    direct = await executor.execute_direct(
        usage_context,
        object(),
        "confusion",
        ReplyMode.IMAGE_ONLY,
        AsyncMock(),
    )
    agent = await executor(ctx, "confusion", ReplyMode.AUTO)

    assert direct["sent"] is True
    assert agent["sent"] is False
    assert agent["reason"] == "physical channel is busy or cooling down"
    ctx.send_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_owner_latch_keeps_identity_and_ignores_reused_key_across_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nekro_plugin_semantic_sticker.agent_tools as agent_tools

    class ReservationOwner:
        pass

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    executor = agent_tools.StickerSendExecutor(
        retriever,
        repository,
        FakeResolver(),
        cooldown_seconds=0,
    )
    monkeypatch.setattr(agent_tools, "id", lambda _owner: 7, raising=False)
    first_context = UsageContext(
        logical_chat_key="onebot_v11-group_100-user_42",
        physical_channel_key="onebot_v11-group_100",
        agent_turn_key="poke-turn-1",
    )
    second_context = UsageContext(
        logical_chat_key="onebot_v11-group_200-user_42",
        physical_channel_key="onebot_v11-group_200",
        agent_turn_key="poke-turn-2",
    )
    first_owner = ReservationOwner()
    first_owner_ref = weakref.ref(first_owner)

    first = await executor.execute_direct(
        first_context,
        first_owner,
        "confusion",
        ReplyMode.IMAGE_ONLY,
        AsyncMock(),
    )
    del first_owner
    gc.collect()
    retained_owner = first_owner_ref()
    second_owner = ReservationOwner()
    second = await executor.execute_direct(
        second_context,
        second_owner,
        "confusion",
        ReplyMode.IMAGE_ONLY,
        AsyncMock(),
    )

    assert first["sent"] is True
    assert retained_owner is not None
    assert second["sent"] is True
    assert executor._ctx_latches[7][0] is second_owner


@pytest.mark.asyncio
async def test_service_direct_entry_starts_and_delegates() -> None:
    from nekro_plugin_semantic_sticker.service import StickerService

    expected = {
        "sent": False,
        "sticker_id": None,
        "reason": "no eligible sticker",
        "reply_mode": ReplyMode.IMAGE_ONLY.value,
        "score": None,
    }
    service = object.__new__(StickerService)
    service.ensure_started = AsyncMock()
    service.send_executor = SimpleNamespace(execute_direct=AsyncMock(return_value=expected))
    usage_context = UsageContext(
        logical_chat_key="onebot_v11-private_42",
        physical_channel_key="onebot_v11-private_42",
        agent_turn_key="poke-turn-1",
    )
    owner = object()
    deliver = AsyncMock()

    result = await service.execute_send_direct(
        usage_context,
        owner,
        "poke response",
        ReplyMode.IMAGE_ONLY,
        deliver,
    )

    assert result == expected
    service.ensure_started.assert_awaited_once_with()
    service.send_executor.execute_direct.assert_awaited_once_with(
        usage_context,
        owner,
        "poke response",
        ReplyMode.IMAGE_ONLY,
        deliver,
    )


@pytest.mark.asyncio
async def test_sender_sends_at_most_one_sticker_per_turn(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    ctx = fake_ctx(tmp_path)
    executor = StickerSendExecutor(retriever, repository, FakeResolver())

    first = await executor(ctx, "疑惑", ReplyMode.IMAGE_ONLY)
    second = await executor(ctx, "还是疑惑", ReplyMode.AUTO)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "physical channel is busy or cooling down"
    assert ctx.send_image.await_count == 1


@pytest.mark.asyncio
async def test_low_score_or_cooldown_result_sends_nothing_and_releases_turn(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    repository = FakeRepository()
    retriever = FakeRetriever(None)
    ctx = fake_ctx(tmp_path)
    executor = StickerSendExecutor(retriever, repository, FakeResolver())

    first = await executor(ctx, "unrelated", ReplyMode.AUTO)
    second = await executor(ctx, "still unrelated", ReplyMode.AUTO)

    assert first["sent"] is False and second["sent"] is False
    assert first["reason"] == "no eligible sticker"
    assert ctx.send_image.await_count == 0
    assert repository.record_usage.await_count == 0


@pytest.mark.asyncio
async def test_send_failure_records_nothing_and_releases_reservation(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.webp"
    source.write_bytes(b"webp-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    failing = fake_ctx(tmp_path, fail_send=True)
    executor = StickerSendExecutor(retriever, repository, FakeResolver())

    failed = await executor(failing, "confusion", ReplyMode.AUTO)
    retry_ctx = fake_ctx(tmp_path / "retry", turn_key="turn-1")
    retried = await executor(retry_ctx, "confusion", ReplyMode.AUTO)

    assert failed["sent"] is False
    assert failed["reason"] == "image send failed"
    assert repository.record_usage.await_count == 1
    assert retried["sent"] is True
    assert failing.forwarded[0].exists() is False


@pytest.mark.asyncio
async def test_decorated_tool_delegates_to_service_without_extra_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    import nekro_plugin_semantic_sticker.agent_tools as agent_tools

    service = SimpleNamespace(execute_send=AsyncMock(return_value={
        "sent": False,
        "sticker_id": None,
        "reason": "no eligible sticker",
        "reply_mode": "auto",
        "score": None,
    }))
    monkeypatch.setattr(agent_tools, "get_service", lambda: service)
    quota_spy = SimpleNamespace(send_agent_request=AsyncMock(), reserve=AsyncMock())
    ctx = SimpleNamespace()

    result = await agent_tools.send_matching_sticker(ctx, " confusion ", "auto")

    assert result["sent"] is False
    service.execute_send.assert_awaited_once_with(ctx, "confusion", ReplyMode.AUTO)
    assert quota_spy.send_agent_request.await_count == 0
    assert quota_spy.reserve.await_count == 0
    source = inspect.getsource(agent_tools)
    assert "send_agent_request" not in source
@pytest.mark.asyncio
async def test_physical_channel_reservation_is_atomic_before_retrieval(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    retriever.release.clear()
    first_ctx = fake_ctx(tmp_path / "first", turn_key="turn-1")
    second_ctx = fake_ctx(tmp_path / "second", turn_key="turn-2")
    executor = StickerSendExecutor(retriever, repository, FakeResolver())

    first_task = asyncio.create_task(executor(first_ctx, "confusion", ReplyMode.AUTO))
    await asyncio.wait_for(retriever.started.wait(), timeout=1)
    second = await executor(second_ctx, "confusion", ReplyMode.AUTO)
    retriever.release.set()
    first = await first_task

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "physical channel is busy or cooling down"
    assert len(retriever.calls) == 1
    assert second_ctx.send_image.await_count == 0


@pytest.mark.asyncio
async def test_retrieval_and_shared_path_errors_release_physical_reservation(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    executor = StickerSendExecutor(retriever, repository, FakeResolver())
    broken_retrieval_ctx = fake_ctx(tmp_path / "retrieval", turn_key="turn-1")
    retriever.error = RuntimeError("embedding failed")

    retrieval_failure = await executor(broken_retrieval_ctx, "confusion", ReplyMode.AUTO)
    retriever.error = None
    broken_file_ctx = fake_ctx(tmp_path / "broken-file", turn_key="turn-2")
    del broken_file_ctx.fs.shared_path
    file_failure = await executor(broken_file_ctx, "confusion", ReplyMode.AUTO)
    retry_ctx = fake_ctx(tmp_path / "retry-ok", turn_key="turn-3")
    retried = await executor(retry_ctx, "confusion", ReplyMode.AUTO)

    assert retrieval_failure["reason"] == "sticker retrieval failed"
    assert file_failure["reason"] == "image send failed"
    assert retried["sent"] is True
    assert retry_ctx.send_image.await_count == 1


@pytest.mark.asyncio
async def test_sent_usage_failure_keeps_finite_ctx_and_channel_latches(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    repository.fail_usage = True
    retriever = FakeRetriever(candidate(source))
    monotonic = FakeMonotonic()
    first_ctx = fake_ctx(tmp_path / "first", turn_key="turn-1")
    second_ctx = fake_ctx(tmp_path / "second", turn_key="turn-2")
    executor = StickerSendExecutor(
        retriever,
        repository,
        FakeResolver(),
        cooldown_seconds=20,
        monotonic=monotonic,
    )

    first = await executor(first_ctx, "confusion", ReplyMode.AUTO)
    same_ctx = await executor(first_ctx, "confusion again", ReplyMode.AUTO)
    same_channel = await executor(second_ctx, "confusion again", ReplyMode.AUTO)
    monotonic.advance(21)
    repository.fail_usage = False
    after_expiry = await executor(second_ctx, "confusion later", ReplyMode.AUTO)

    assert first["sent"] is True
    assert first["reason"] == "sent; usage history unavailable"
    assert same_ctx["sent"] is False
    assert same_channel["sent"] is False
    assert after_expiry["sent"] is True
    assert first_ctx.send_image.await_count == 1
    assert second_ctx.send_image.await_count == 1
@pytest.mark.asyncio
async def test_destination_cleanup_error_never_leaks_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nekro_plugin_semantic_sticker.agent_tools import StickerSendExecutor

    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    repository = FakeRepository()
    retriever = FakeRetriever(candidate(source))
    failing_ctx = fake_ctx(tmp_path / "first", turn_key="turn-1", fail_send=True)
    retry_ctx = fake_ctx(tmp_path / "retry", turn_key="turn-2")
    executor = StickerSendExecutor(retriever, repository, FakeResolver())
    original_unlink = Path.unlink
    fail_cleanup = True

    def unlink(path: Path, *args, **kwargs) -> None:
        if fail_cleanup and path.name.startswith("semantic-sticker-"):
            raise PermissionError("cleanup failed")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    first = await executor(failing_ctx, "confusion", ReplyMode.AUTO)
    fail_cleanup = False
    second = await executor(retry_ctx, "confusion", ReplyMode.AUTO)

    assert first["sent"] is False and first["reason"] == "image send failed"
    assert second["sent"] is True
