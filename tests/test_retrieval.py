from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from nekro_plugin_semantic_sticker.config import SemanticStickerConfig
from nekro_plugin_semantic_sticker.database import StickerRepository
from nekro_plugin_semantic_sticker.models import SafetyState, StickerState, StoredAsset, UsageContext, VisionMetadata
from nekro_plugin_semantic_sticker.vector_store import VectorSearchHit


class FrozenClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self) -> None:
        self.hits: list[VectorSearchHit] = []
        self.calls: list[dict[str, object]] = []

    async def search(self, vector: list[float], *, limit: int, score_threshold: float) -> list[VectorSearchHit]:
        self.calls.append({"vector": vector, "limit": limit, "score_threshold": score_threshold})
        return list(self.hits)


def asset(digest: str) -> StoredAsset:
    return StoredAsset(
        sha256=digest,
        asset_path=f"assets/{digest}.png",
        thumbnail_path=f"thumbnails/{digest}.webp",
        detected_format="PNG",
        detected_extension="png",
        mime_type="image/png",
        byte_size=100,
        width=8,
        height=8,
        frame_count=1,
        animated=False,
    )


def metadata(
    category: str,
    *,
    emotion_tags: list[str] | None = None,
    scene_tags: list[str] | None = None,
    ocr_text: str = "",
    safety: SafetyState = SafetyState.SAFE,
) -> VisionMetadata:
    return VisionMetadata(
        description=f"{category} reaction",
        primary_category=category,
        emotion_tags=emotion_tags or [category],
        scene_tags=scene_tags or [],
        ocr_text=ocr_text,
        suitable_scenarios=[category],
        unsuitable_scenarios=[],
        safety=safety,
    )


async def create_active(repository: StickerRepository, digest: str, data: VisionMetadata):
    record, _ = await repository.create_pending(asset(digest), analysis_version="v1")
    await repository.transition(record.id, StickerState.ANALYZING)
    await repository.transition(record.id, StickerState.INDEXING, metadata=data)
    return await repository.transition(record.id, StickerState.ACTIVE, vector_version=1)


@pytest_asyncio.fixture
async def retrieval_harness(tmp_path: Path):
    from nekro_plugin_semantic_sticker.retrieval import StickerRetriever

    clock = FrozenClock()
    repository = StickerRepository(tmp_path / "stickers.db", clock=clock)
    await repository.initialize()
    embedding = FakeEmbedding()
    vector_store = FakeVectorStore()
    config = SemanticStickerConfig(
        VECTOR_DIMENSION=3,
        SEMANTIC_SCORE_THRESHOLD=0.72,
        RECENT_SELECTION_WINDOW=10,
        PHYSICAL_CHANNEL_COOLDOWN_SECONDS=20,
    )
    retriever = StickerRetriever(config, repository, embedding, vector_store, candidate_limit=24)
    yield retriever, repository, embedding, vector_store, clock
    await repository.close()


def context(user: int = 1, turn: str = "turn-1") -> UsageContext:
    return UsageContext(
        logical_chat_key=f"onebot_v11-group_100-user_{user}",
        physical_channel_key="onebot_v11-group_100",
        agent_turn_key=turn,
    )


@pytest.mark.asyncio
async def test_confusion_intent_prefers_confusion_sticker(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, _clock = retrieval_harness
    confusion = await create_active(repository, "a" * 64, metadata("confusion"))
    happy = await create_active(repository, "b" * 64, metadata("happiness"))
    vector_store.hits = [
        VectorSearchHit(confusion.id, 0.91, {}),
        VectorSearchHit(happy.id, 0.80, {}),
    ]

    result = await retriever.find("我没看懂，为什么会这样？", context())

    assert result is not None and result.sticker_id == confusion.id


@pytest.mark.asyncio
async def test_category_tag_and_ocr_boosts_apply_after_vector_search(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, _clock = retrieval_harness
    boosted = await create_active(
        repository,
        "c" * 64,
        metadata("confusion", emotion_tags=["questioning"], scene_tags=["asking why"], ocr_text="?"),
    )
    plain = await create_active(repository, "d" * 64, metadata("happiness"))
    vector_store.hits = [
        VectorSearchHit(boosted.id, 0.60, {}),
        VectorSearchHit(plain.id, 0.76, {}),
    ]

    result = await retriever.find("confusion questioning asking why ?", context())

    assert result is not None and result.sticker_id == boosted.id
    assert result.vector_score == 0.84
    assert vector_store.calls[-1]["score_threshold"] == 0.0


@pytest.mark.asyncio
async def test_low_score_and_no_eligible_candidate_return_none(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, _clock = retrieval_harness
    sticker = await create_active(repository, "e" * 64, metadata("neutral reaction"))
    vector_store.hits = [VectorSearchHit(sticker.id, 0.71, {})]

    assert await retriever.find("unrelated intent", context()) is None
    vector_store.hits = [VectorSearchHit("missing", 0.99, {})]
    assert await retriever.find("anything", context(turn="turn-2")) is None


@pytest.mark.asyncio
async def test_recent_repeat_is_excluded_for_logical_chat(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, clock = retrieval_harness
    repeated = await create_active(repository, "f" * 64, metadata("confusion"))
    fallback = await create_active(repository, "1" * 64, metadata("happiness"))
    await repository.record_usage(repeated.id, context().logical_chat_key, context().physical_channel_key, "old-turn", 0.9)
    clock.advance(21)
    vector_store.hits = [
        VectorSearchHit(repeated.id, 0.99, {}),
        VectorSearchHit(fallback.id, 0.80, {}),
    ]

    result = await retriever.find("reaction", context(turn="new-turn"))

    assert result is not None and result.sticker_id == fallback.id


@pytest.mark.asyncio
async def test_physical_group_cooldown_collapses_user_scoped_contexts(retrieval_harness) -> None:
    retriever, repository, embedding, vector_store, _clock = retrieval_harness
    sticker = await create_active(repository, "2" * 64, metadata("confusion"))
    await repository.record_usage(
        sticker.id,
        "onebot_v11-group_100-user_1",
        "onebot_v11-group_100",
        "turn-user-1",
        0.9,
    )
    vector_store.hits = [VectorSearchHit(sticker.id, 0.95, {})]

    result = await retriever.find("confusion", context(user=2, turn="turn-user-2"))

    assert result is None
    assert embedding.calls == []


@pytest.mark.asyncio
async def test_private_chat_cooldowns_are_independent(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, _clock = retrieval_harness
    sticker = await create_active(repository, "3" * 64, metadata("confusion"))
    await repository.record_usage(sticker.id, "onebot_v11-private_1", "onebot_v11-private_1", "private-turn-1", 0.9)
    vector_store.hits = [VectorSearchHit(sticker.id, 0.95, {})]
    other_private = UsageContext(
        logical_chat_key="onebot_v11-private_2",
        physical_channel_key="onebot_v11-private_2",
        agent_turn_key="private-turn-2",
    )

    result = await retriever.find("confusion", other_private)

    assert result is not None and result.sticker_id == sticker.id


@pytest.mark.asyncio
async def test_authoritative_sqlite_excludes_inactive_and_unsafe_hits(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, _clock = retrieval_harness
    safe = await create_active(repository, "4" * 64, metadata("confusion"))
    pending, _ = await repository.create_pending(asset("5" * 64), analysis_version="v1")
    unsafe, _ = await repository.create_pending(asset("6" * 64), analysis_version="v1")
    await repository.transition(unsafe.id, StickerState.ANALYZING)
    await repository.transition(unsafe.id, StickerState.FAILED, metadata=metadata("anger", safety=SafetyState.UNSAFE))
    vector_store.hits = [
        VectorSearchHit(pending.id, 0.99, {}),
        VectorSearchHit(unsafe.id, 0.98, {}),
        VectorSearchHit(safe.id, 0.80, {}),
    ]

    result = await retriever.find("reaction", context())

    assert result is not None and result.sticker_id == safe.id


@pytest.mark.asyncio
async def test_ties_prefer_oldest_usage_then_stable_uuid(retrieval_harness) -> None:
    retriever, repository, _embedding, vector_store, clock = retrieval_harness
    first = await create_active(repository, "7" * 64, metadata("confusion"))
    second = await create_active(repository, "8" * 64, metadata("confusion"))
    await repository.record_usage(first.id, "other-chat", "other-channel", "old-a", 0.8)
    clock.advance(5)
    await repository.record_usage(second.id, "other-chat", "other-channel", "old-b", 0.8)
    clock.advance(21)
    vector_store.hits = [VectorSearchHit(second.id, 0.8, {}), VectorSearchHit(first.id, 0.8, {})]

    result = await retriever.find("reaction", context())

    assert result is not None and result.sticker_id == first.id


@pytest.mark.asyncio
async def test_agent_turn_with_existing_send_is_rejected_before_embedding(retrieval_harness) -> None:
    retriever, repository, embedding, vector_store, _clock = retrieval_harness
    sticker = await create_active(repository, "9" * 64, metadata("confusion"))
    await repository.record_usage(sticker.id, "other-chat", "other-channel", "same-turn", 0.9)
    vector_store.hits = [VectorSearchHit(sticker.id, 0.95, {})]

    assert await retriever.find("confusion", context(turn="same-turn")) is None
    assert embedding.calls == []


def test_context_identity_resolver_uses_only_explicit_turn_key_or_process_local_ctx_latch() -> None:
    from nekro_plugin_semantic_sticker.retrieval import DefaultContextIdentityResolver

    resolver = DefaultContextIdentityResolver()
    first = SimpleNamespace(
        chat_key="onebot_v11-group_100-user_1",
        message_id="message-1",
        request_id="request-1",
        turn_id="turn-1",
        container_key="chat-container",
    )
    second = SimpleNamespace(
        chat_key="onebot_v11-group_100-user_2",
        message_id="message-2",
        container_key="chat-container",
    )
    private = SimpleNamespace(chat_key="onebot_v11-private_9")
    explicit = SimpleNamespace(chat_key="onebot_v11-private_9", agent_turn_key="future-turn-key")

    first_key = resolver.agent_turn_key(first)
    assert resolver.logical_chat_key(first) == first.chat_key
    assert resolver.physical_channel_key(first) == "onebot_v11-group_100"
    assert resolver.physical_channel_key(second) == "onebot_v11-group_100"
    assert resolver.physical_channel_key(private) == private.chat_key
    assert first_key == resolver.agent_turn_key(first)
    assert first_key != resolver.agent_turn_key(second)
    assert first_key.startswith("process:")
    assert "message-1" not in first_key and "turn-1" not in first_key and "chat-container" not in first_key
    assert resolver.agent_turn_key(explicit) == "future-turn-key"