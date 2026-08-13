import asyncio
import inspect
import mimetypes
import shutil
import time
import uuid
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Literal

from nekro_agent.api import schemas
from nekro_agent.api.plugin import SandboxMethodType

from . import config, plugin
from .message_images import message_image_registry
from .models import ReplyMode, StickerSendResult, UploadPayload, UsageContext
from .retrieval import DefaultContextIdentityResolver, StickerRetriever


class StickerSendExecutor:
    def __init__(
        self,
        retriever: object,
        repository: object,
        identity_resolver: object,
        *,
        cooldown_seconds: int | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.retriever = retriever
        self.repository = repository
        self.identity_resolver = identity_resolver
        configured_cooldown = getattr(
            getattr(retriever, "config", None),
            "PHYSICAL_CHANNEL_COOLDOWN_SECONDS",
            20,
        )
        self._cooldown_seconds = max(
            0,
            int(configured_cooldown if cooldown_seconds is None else cooldown_seconds),
        )
        self._ctx_latch_seconds = max(300, self._cooldown_seconds)
        self._monotonic = monotonic or time.monotonic
        self._reservation_lock = asyncio.Lock()
        self._reserved_channels: set[str] = set()
        self._channel_latches: dict[str, float] = {}
        self._ctx_latches: dict[int, tuple[object, float]] = {}

    async def __call__(
        self,
        ctx: object,
        intent: str,
        reply_mode: ReplyMode,
    ) -> StickerSendResult:
        usage_context: UsageContext = self.identity_resolver.resolve(ctx)

        async def deliver(source: Path) -> None:
            destination: Path | None = None
            try:
                shared_root = Path(ctx.fs.shared_path).resolve()
                shared_root.mkdir(parents=True, exist_ok=True)
                destination = shared_root / f"semantic-sticker-{uuid.uuid4().hex}{source.suffix.casefold()}"
                await asyncio.to_thread(shutil.copy2, source, destination)
                send_path = ctx.fs.forward_file(destination)
                if inspect.isawaitable(send_path):
                    send_path = await send_path
                await ctx.send_image(str(send_path), record=True)
            finally:
                try:
                    if destination is not None:
                        destination.unlink(missing_ok=True)
                except OSError:
                    pass

        return await self.execute_direct(usage_context, ctx, intent, reply_mode, deliver)

    async def execute_direct(
        self,
        usage_context: UsageContext,
        reservation_owner: object,
        intent: str,
        reply_mode: ReplyMode,
        deliver: Callable[[Path], Awaitable[None]],
    ) -> StickerSendResult:
        try:
            reserved = await self._reserve(reservation_owner, usage_context)
        except Exception:
            return self._result(False, None, "sticker reservation unavailable", reply_mode, None)
        if not reserved:
            return self._result(
                False,
                None,
                "physical channel is busy or cooling down",
                reply_mode,
                None,
            )

        try:
            try:
                candidate = await self.retriever.find(intent.strip(), usage_context)
            except Exception:
                return self._result(False, None, "sticker retrieval failed", reply_mode, None)
            if candidate is None:
                return self._result(False, None, "no eligible sticker", reply_mode, None)

            try:
                source = Path(candidate.asset_path).resolve()
                await deliver(source)
            except Exception:
                return self._result(False, None, "image send failed", reply_mode, None)

            await self._latch_sent(reservation_owner, usage_context)
            try:
                await self.repository.record_usage(
                    candidate.sticker_id,
                    usage_context.logical_chat_key,
                    usage_context.physical_channel_key,
                    usage_context.agent_turn_key,
                    candidate.vector_score,
                )
            except Exception:
                return self._result(
                    True,
                    candidate.sticker_id,
                    "sent; usage history unavailable",
                    reply_mode,
                    candidate.vector_score,
                )
            return self._result(True, candidate.sticker_id, "sent", reply_mode, candidate.vector_score)
        finally:
            await self._release(usage_context.physical_channel_key)

    async def _reserve(self, reservation_owner: object, usage_context: UsageContext) -> bool:
        async with self._reservation_lock:
            now = self._monotonic()
            self._prune_latches(now)
            owner_key = id(reservation_owner)
            physical_key = usage_context.physical_channel_key
            if physical_key in self._reserved_channels:
                return False
            if self._channel_latches.get(physical_key, 0.0) > now:
                return False
            owner_latch = self._ctx_latches.get(owner_key)
            if owner_latch is not None:
                latched_owner, expires_at = owner_latch
                if latched_owner is reservation_owner and expires_at > now:
                    return False
            if await self.repository.agent_turn_has_send(usage_context.agent_turn_key):
                return False
            if await self.repository.physical_channel_in_cooldown(
                physical_key,
                cooldown_seconds=self._cooldown_seconds,
            ):
                return False
            self._reserved_channels.add(physical_key)
            return True

    async def _latch_sent(self, reservation_owner: object, usage_context: UsageContext) -> None:
        async with self._reservation_lock:
            now = self._monotonic()
            self._ctx_latches[id(reservation_owner)] = (
                reservation_owner,
                now + self._ctx_latch_seconds,
            )
            if self._cooldown_seconds > 0:
                self._channel_latches[usage_context.physical_channel_key] = now + self._cooldown_seconds

    async def _release(self, physical_channel_key: str) -> None:
        async with self._reservation_lock:
            self._reserved_channels.discard(physical_channel_key)
            self._prune_latches(self._monotonic())

    def _prune_latches(self, now: float) -> None:
        self._channel_latches = {
            key: expires_at
            for key, expires_at in self._channel_latches.items()
            if expires_at > now
        }
        self._ctx_latches = {
            key: (owner, expires_at)
            for key, (owner, expires_at) in self._ctx_latches.items()
            if expires_at > now
        }

    @staticmethod
    def _result(
        sent: bool,
        sticker_id: str | None,
        reason: str,
        reply_mode: ReplyMode,
        score: float | None,
    ) -> StickerSendResult:
        return {
            "sent": sent,
            "sticker_id": sticker_id,
            "reason": reason,
            "reply_mode": reply_mode.value,
            "score": score,
        }
_service = None


def _build_service():
    from nekro_agent.api.core import config as core_config
    from nekro_agent.core.os_env import OsEnv

    from .analysis import VisionAnalyzer
    from .database import StickerRepository
    from .files import ImageStore
    from .service import StickerService
    from .vector_store import EmbeddingProvider, StickerVectorStore

    data_root = Path(OsEnv.DATA_DIR) / "plugin_data" / plugin.key
    repository = StickerRepository(data_root / "stickers.db")
    image_store = ImageStore(data_root, config)
    analyzer = VisionAnalyzer(config, core_config)
    embedding = EmbeddingProvider(
        core_config,
        model_group_name=config.EMBEDDING_MODEL_GROUP,
        dimension=config.VECTOR_DIMENSION,
        timeout=config.ANALYSIS_TIMEOUT_SECONDS,
    )
    vector_store = StickerVectorStore(expected_dimension=config.VECTOR_DIMENSION)
    retriever = StickerRetriever(config, repository, embedding, vector_store)
    sender = StickerSendExecutor(retriever, repository, DefaultContextIdentityResolver())
    return StickerService(
        config,
        repository,
        image_store,
        analyzer,
        embedding,
        vector_store,
        send_executor=sender,
    )


def get_service():
    global _service
    if _service is None:
        _service = _build_service()
    return _service


def set_service_for_testing(service: object | None) -> None:
    global _service
    _service = service

@plugin.mount_prompt_inject_method("semantic_sticker_prompt")
async def semantic_sticker_prompt(_ctx: schemas.AgentCtx) -> str:
    automatic_rule = (
        "Automatic sticker collection is enabled. You may call save_sticker_from_message with "
        "save_reason='automatic' only for a clearly visible reaction sticker that is useful to the shared library. "
        if config.AUTO_COLLECT_ENABLED
        else "Automatic sticker collection is disabled. Never save an image unless the user explicitly asks you to save it. "
    )
    strict_rule = (
        "Strict sticker collection is enabled: screenshots, photos, ordinary images, and images not confirmed by vision "
        "must use vision_and_sticker_confirmed=false and must not be saved. "
        if config.STRICT_EMOTION_COLLECT
        else "Strict sticker collection is disabled, but you must still select only the current or explicitly replied image. "
    )
    return (
        "Stickers are optional. Choose text only by not calling the tool, sticker only with "
        "reply_mode='image_only', or text then sticker with reply_mode='text_then_image'. "
        "Use reply_mode='auto' when either image behavior is suitable. Call send_matching_sticker "
        "only when it improves the reply and pass a concise semantic intent, never raw chat history. "
        "When the user explicitly asks to save a sticker from the current message or an image they replied to, call "
        "save_sticker_from_message with save_reason='user_request'; explicit user requests remain allowed even when "
        "automatic collection is disabled. The save tool can access only current-message or explicitly replied images. "
        + automatic_rule
        + strict_rule
    )


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="send_matching_sticker",
    description="按当前回复意图检索并发送至多一张合适表情包；该工具只负责发送，不执行保存。",
)
async def send_matching_sticker(
    _ctx: schemas.AgentCtx,
    intent: str,
    reply_mode: Literal["image_only", "text_then_image", "auto"] = "auto",
) -> StickerSendResult:
    """Send at most one sticker matching the current reply intent.

    Args:
        intent: Concise semantic intent for the desired reaction; never pass raw chat history.
        reply_mode: Use image_only, text_then_image, or auto to choose how the sticker accompanies the reply.

    Returns:
        A structured result describing whether a sticker was sent and why.
    """
    return await get_service().execute_send(_ctx, intent.strip(), ReplyMode(reply_mode))


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="save_sticker_from_message",
    description="保存当前消息或当前消息明确引用的一张图片，并复用管理面板的自动分析、分类、标签和向量索引流程。",
)
async def save_sticker_from_message(
    _ctx: schemas.AgentCtx,
    image_scope: Literal["auto", "current", "reply"] = "auto",
    image_index: int = 0,
    save_reason: Literal["user_request", "automatic"] = "user_request",
    vision_and_sticker_confirmed: bool = False,
) -> str:
    """Save one bounded current-message or explicitly replied image as a sticker.

    Args:
        image_scope: Select auto, current, or reply; auto prefers the current message image.
        image_index: Zero-based image position within the selected scope.
        save_reason: Use user_request for an explicit user request or automatic for Agent-initiated collection.
        vision_and_sticker_confirmed: Whether vision clearly confirmed a sticker or reaction GIF.

    Returns:
        A concise Chinese result explaining whether the image was saved, duplicated, or rejected.
    """
    if save_reason == "automatic" and not config.AUTO_COLLECT_ENABLED:
        return "自动保存已关闭；只有用户明确要求保存时才能加入表情包库。"
    if config.STRICT_EMOTION_COLLECT and not vision_and_sticker_confirmed:
        return "严格模式已拒绝保存：请先通过视觉确认该图片确实是表情包或 reaction GIF，截图、照片和普通图片不能保存。"
    if image_index < 0:
        return "图片序号不能小于 0。"

    try:
        chat_key = _ctx.chat_key
    except (AttributeError, ValueError):
        return "没有可用的聊天上下文，无法定位要保存的图片。"
    image_ref = message_image_registry.resolve(chat_key, image_scope, image_index)
    if image_ref is None:
        return "当前消息或当前引用消息中没有找到可保存的图片。"
    if not image_ref.local_path:
        return "目标图片没有可用的本地文件，暂时无法保存。"

    source = Path(image_ref.local_path).resolve()
    if not source.is_file():
        return "目标图片的本地文件已不存在，请重新发送图片后再保存。"
    try:
        content = await asyncio.to_thread(source.read_bytes)
        filename = image_ref.file_name or source.name
        content_type = mimetypes.guess_type(filename)[0]
        actor_id = getattr(_ctx, "from_platform_userid", None) or getattr(_ctx, "from_user_id", None)
        actor = f"chat:{actor_id}" if actor_id else f"chat-key:{chat_key}"
        outcome = await get_service().upload(
            UploadPayload(content=content, filename=filename, content_type=content_type),
            actor=actor,
        )
    except Exception:
        return "表情包保存失败，图片未写入表情包库。"

    if outcome.duplicate:
        return "这个表情包已经保存过了，已沿用现有记录。"
    return "表情包已提交，后台正在自动分析内容、分类并生成标签。"


__all__ = [
    "StickerSendExecutor",
    "get_service",
    "semantic_sticker_prompt",
    "save_sticker_from_message",
    "send_matching_sticker",
    "set_service_for_testing",
]
