from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from nonebot import on_type
from nonebot.adapters.onebot.v11 import Bot, MessageSegment, PokeNotifyEvent

from .agent_tools import get_service
from .models import ReplyMode, UsageContext


logger = logging.getLogger(__name__)
POKE_INTENT = "被戳一戳后的互动回应，表达惊讶、疑惑、调皮或轻微不满"
_LOGGABLE_FAILURES = {
    "image send failed",
    "sticker reservation unavailable",
    "sticker retrieval failed",
    "sticker sender is not configured",
}


def _is_bot_target(event: PokeNotifyEvent) -> bool:
    return event.is_tome()


def _usage_context(event: PokeNotifyEvent) -> UsageContext:
    if event.group_id is not None:
        logical_chat_key = f"onebot_v11-group_{event.group_id}-user_{event.user_id}"
        physical_channel_key = f"onebot_v11-group_{event.group_id}"
        turn_scope = f"group_{event.group_id}-user_{event.user_id}"
    else:
        logical_chat_key = f"onebot_v11-private_{event.user_id}"
        physical_channel_key = logical_chat_key
        turn_scope = f"private_{event.user_id}"
    return UsageContext(
        logical_chat_key=logical_chat_key,
        physical_channel_key=physical_channel_key,
        agent_turn_key=f"onebot_v11-poke-{turn_scope}-at_{event.time}-{uuid.uuid4().hex}",
    )


bot_poke_matcher = on_type(
    PokeNotifyEvent,
    rule=_is_bot_target,
    priority=1,
    block=True,
)


@bot_poke_matcher.handle()
async def handle_bot_poke(bot: Bot, event: PokeNotifyEvent) -> None:
    async def deliver(asset_path: Path) -> None:
        image_bytes = await asyncio.to_thread(asset_path.read_bytes)
        await bot.send(event, MessageSegment.image(image_bytes))

    try:
        result = await get_service().execute_send_direct(
            _usage_context(event),
            event,
            POKE_INTENT,
            ReplyMode.IMAGE_ONLY,
            deliver,
        )
        if not result["sent"] and result["reason"] in _LOGGABLE_FAILURES:
            logger.warning("Bot 戳一戳表情回复未发送: %s", result["reason"])
    except Exception:
        logger.exception("处理 Bot 戳一戳表情回复失败")


__all__ = ["POKE_INTENT", "bot_poke_matcher", "handle_bot_poke"]
