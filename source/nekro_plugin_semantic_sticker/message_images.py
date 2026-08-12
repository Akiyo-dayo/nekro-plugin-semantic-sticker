from __future__ import annotations

import inspect
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from . import plugin


ImageScope = Literal["auto", "current", "reply"]
ReferenceLoader = Callable[[str, str, str], Awaitable[object | None]]
_GROUP_USER_KEY = re.compile(r"^(onebot_v11-group_[^-]+)-user_[^-]+$")


@dataclass(frozen=True)
class MessageImageRef:
    file_name: str
    local_path: str | None
    remote_url: str | None


@dataclass(frozen=True)
class _MessageImageEntry:
    current: tuple[MessageImageRef, ...]
    reply: tuple[MessageImageRef, ...]
    expires_at: float


class MessageImageRegistry:
    def __init__(
        self,
        *,
        reference_loader: ReferenceLoader | None = None,
        ttl_seconds: int = 600,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._reference_loader = reference_loader or self._default_reference_loader
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._monotonic = monotonic or time.monotonic
        self._entries: dict[str, _MessageImageEntry] = {}

    async def remember(self, ctx: object, message: object) -> None:
        chat_key = self._chat_key(ctx, message)
        if not chat_key:
            return
        current = tuple(self._extract_images(getattr(message, "content_data", ())))
        reply: tuple[MessageImageRef, ...] = ()
        ref_msg_id = self._reference_message_id(getattr(message, "ext_data", None))
        adapter_key = str(getattr(message, "adapter_key", "") or "")
        if ref_msg_id and adapter_key:
            referenced = await self._reference_loader(adapter_key, chat_key, ref_msg_id)
            if referenced is not None:
                segments = getattr(referenced, "content_data", ())
                parser = getattr(referenced, "parse_content_data", None)
                if callable(parser):
                    segments = parser()
                    if inspect.isawaitable(segments):
                        segments = await segments
                reply = tuple(self._extract_images(segments))
        self._entries[chat_key] = _MessageImageEntry(
            current=current,
            reply=reply,
            expires_at=self._monotonic() + self._ttl_seconds,
        )
        self._prune()

    def resolve(self, chat_key: str, scope: ImageScope, index: int = 0) -> MessageImageRef | None:
        if index < 0:
            return None
        self._prune()
        entry = self._entries.get(chat_key)
        if entry is None:
            return None
        if scope == "current":
            candidates = entry.current
        elif scope == "reply":
            candidates = entry.reply
        elif scope == "auto":
            candidates = entry.current if entry.current else entry.reply
        else:
            return None
        return candidates[index] if index < len(candidates) else None

    def snapshot(self, chat_key: str) -> tuple[MessageImageRef, ...]:
        self._prune()
        entry = self._entries.get(chat_key)
        if entry is None:
            return ()
        return entry.current + entry.reply

    def clear(self) -> None:
        self._entries.clear()

    def _prune(self) -> None:
        now = self._monotonic()
        self._entries = {key: entry for key, entry in self._entries.items() if entry.expires_at > now}

    @staticmethod
    def _chat_key(ctx: object, message: object) -> str:
        try:
            value = getattr(ctx, "chat_key", "")
        except (AttributeError, ValueError):
            value = ""
        return str(value or getattr(message, "chat_key", "") or "")

    @staticmethod
    def _reference_message_id(ext_data: object) -> str:
        if isinstance(ext_data, dict):
            return str(ext_data.get("ref_msg_id") or "")
        return str(getattr(ext_data, "ref_msg_id", "") or "")

    @staticmethod
    def _extract_images(segments: Iterable[object] | object) -> list[MessageImageRef]:
        if not isinstance(segments, Iterable) or isinstance(segments, (str, bytes, dict)):
            return []
        images: list[MessageImageRef] = []
        for segment in segments:
            segment_type = getattr(segment, "type", "")
            if isinstance(segment_type, Enum):
                segment_type = segment_type.value
            if str(segment_type).casefold() != "image":
                continue
            local_path = getattr(segment, "local_path", None)
            remote_url = getattr(segment, "remote_url", None)
            file_name = str(getattr(segment, "file_name", "") or "")
            if not local_path and not remote_url:
                continue
            images.append(
                MessageImageRef(
                    file_name=file_name,
                    local_path=str(local_path) if local_path else None,
                    remote_url=str(remote_url) if remote_url else None,
                ),
            )
        return images

    @staticmethod
    async def _default_reference_loader(adapter_key: str, chat_key: str, message_id: str) -> object | None:
        from nekro_agent.models.db_chat_message import DBChatMessage

        query = DBChatMessage.filter(
            adapter_key=adapter_key,
            message_id=message_id,
        )
        group_match = _GROUP_USER_KEY.fullmatch(chat_key)
        if group_match:
            query = query.filter(chat_key__startswith=f"{group_match.group(1)}-user_")
        else:
            query = query.filter(chat_key=chat_key)
        return await query.order_by("-id").first()


message_image_registry = MessageImageRegistry()


@plugin.mount_on_user_message()
async def remember_message_images(_ctx: object, message: object) -> None:
    await message_image_registry.remember(_ctx, message)


__all__ = [
    "ImageScope",
    "MessageImageRef",
    "MessageImageRegistry",
    "message_image_registry",
    "remember_message_images",
]
