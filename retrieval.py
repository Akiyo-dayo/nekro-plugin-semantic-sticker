from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from .config import SemanticStickerConfig
from .database import StickerRepository
from .models import SafetyState, StickerCandidate, StickerState, UsageContext


_GROUP_USER_KEY = re.compile(r"^(onebot_v11-group_[^-]+)-user_[^-]+$")


class DefaultContextIdentityResolver:
    def __init__(self, *, process_nonce: str | None = None) -> None:
        self._process_nonce = process_nonce or uuid.uuid4().hex

    def logical_chat_key(self, ctx: object) -> str:
        chat_key = str(getattr(ctx, "chat_key", "")).strip()
        if not chat_key:
            raise ValueError("Agent context chat_key is required")
        return chat_key

    def physical_channel_key(self, ctx: object) -> str:
        logical = self.logical_chat_key(ctx)
        match = _GROUP_USER_KEY.fullmatch(logical)
        return match.group(1) if match else logical

    def agent_turn_key(self, ctx: object) -> str:
        explicit = getattr(ctx, "agent_turn_key", None)
        if explicit:
            return str(explicit)
        return f"process:{self._process_nonce}:ctx:{id(ctx)}"
    def resolve(self, ctx: object) -> UsageContext:
        return UsageContext(
            logical_chat_key=self.logical_chat_key(ctx),
            physical_channel_key=self.physical_channel_key(ctx),
            agent_turn_key=self.agent_turn_key(ctx),
        )


def hybrid_score(intent: str, candidate: StickerCandidate) -> float:
    normalized = intent.casefold()
    exact = 0.0
    if candidate.primary_category.casefold() in normalized:
        exact += 0.12
    if any(tag.casefold() in normalized for tag in candidate.emotion_tags + candidate.scene_tags):
        exact += 0.08
    if candidate.ocr_text and candidate.ocr_text.casefold() in normalized:
        exact += 0.04
    return min(1.0, candidate.vector_score + exact)


class StickerRetriever:
    def __init__(
        self,
        config: SemanticStickerConfig,
        repository: StickerRepository,
        embedding: object,
        vector_store: object,
        *,
        candidate_limit: int = 24,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedding = embedding
        self.vector_store = vector_store
        self.candidate_limit = max(1, candidate_limit)

    async def find(self, intent: str, usage_context: UsageContext) -> StickerCandidate | None:
        normalized_intent = " ".join(intent.split()).strip()
        if not normalized_intent:
            return None
        if await self.repository.agent_turn_has_send(usage_context.agent_turn_key):
            return None
        if await self.repository.physical_channel_in_cooldown(
            usage_context.physical_channel_key,
            cooldown_seconds=self.config.PHYSICAL_CHANNEL_COOLDOWN_SECONDS,
        ):
            return None

        vector = await self.embedding.embed(normalized_intent)
        hits = await self.vector_store.search(
            vector,
            limit=self.candidate_limit,
            score_threshold=0.0,
        )
        if not hits:
            return None
        recent = set(
            await self.repository.recent_sticker_ids(
                usage_context.logical_chat_key,
                limit=self.config.RECENT_SELECTION_WINDOW,
            )
        )
        ordered_ids = [hit.sticker_id for hit in hits if hit.sticker_id not in recent]
        records = await self.repository.get_many(ordered_ids)
        last_usage = await self.repository.last_usage_times(ordered_ids)
        candidates: list[StickerCandidate] = []
        for hit in hits:
            if hit.sticker_id in recent:
                continue
            record = records.get(hit.sticker_id)
            if record is None:
                continue
            if record.state is not StickerState.ACTIVE or record.safety is not SafetyState.SAFE:
                continue
            candidate = StickerCandidate(
                sticker_id=record.id,
                vector_score=float(hit.score),
                asset_path=record.asset_path,
                primary_category=record.primary_category,
                emotion_tags=record.emotion_tags,
                scene_tags=record.scene_tags,
                ocr_text=record.ocr_text,
                last_used_at=last_usage.get(record.id),
            )
            score = hybrid_score(normalized_intent, candidate)
            if score < self.config.SEMANTIC_SCORE_THRESHOLD:
                continue
            candidates.append(candidate.model_copy(update={"vector_score": score}))
        if not candidates:
            return None
        oldest = datetime.min.replace(tzinfo=UTC)
        candidates.sort(
            key=lambda candidate: (
                -candidate.vector_score,
                candidate.last_used_at or oldest,
                candidate.sticker_id,
            )
        )
        return candidates[0]


__all__ = [
    "DefaultContextIdentityResolver",
    "StickerRetriever",
    "hybrid_score",
]