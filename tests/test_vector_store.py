from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nekro_plugin_semantic_sticker.models import SafetyState, StickerRecord, StickerState, VisionMetadata


class FakeQdrant:
    def __init__(self, collection_name: str = "Akiyo.semantic_sticker") -> None:
        self.collection_name = collection_name
        self.collection_dimension: int | None = None
        self.points: dict[str, dict[str, object]] = {}
        self.get_collections = AsyncMock(side_effect=self._get_collections)
        self.get_collection = AsyncMock(side_effect=self._get_collection)
        self.create_collection = AsyncMock(side_effect=self._create_collection)
        self.delete_collection = AsyncMock(side_effect=self._delete_collection)
        self.upsert = AsyncMock(side_effect=self._upsert)
        self.search = AsyncMock(side_effect=self._search)
        self.delete = AsyncMock(side_effect=self._delete)
        self.count = AsyncMock(side_effect=self._count)
        self.scroll = AsyncMock(side_effect=self._scroll)
        self.search_results: list[object] = []

    async def _get_collections(self) -> object:
        collections = [] if self.collection_dimension is None else [SimpleNamespace(name=self.collection_name)]
        return SimpleNamespace(collections=collections)

    async def _get_collection(self, collection_name: str) -> object:
        assert collection_name == self.collection_name
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.collection_dimension))
            )
        )

    async def _create_collection(self, *, collection_name: str, vectors_config: dict[str, object]) -> None:
        assert collection_name == self.collection_name
        self.collection_dimension = int(vectors_config["size"])

    async def _delete_collection(self, *, collection_name: str) -> None:
        assert collection_name == self.collection_name
        self.collection_dimension = None
        self.points.clear()
    async def _upsert(self, *, collection_name: str, points: list[dict[str, object]], wait: bool = True) -> None:
        assert collection_name == self.collection_name
        assert wait is True
        for point in points:
            self.points[str(point["id"])] = point

    async def _search(self, **_kwargs: object) -> list[object]:
        return list(self.search_results)

    async def _delete(self, *, collection_name: str, points_selector: dict[str, object], wait: bool = True) -> None:
        assert collection_name == self.collection_name
        assert wait is True
        for point_id in points_selector["points"]:
            self.points.pop(str(point_id), None)

    async def _count(self, *, collection_name: str, exact: bool = True) -> object:
        assert collection_name == self.collection_name
        assert exact is True
        return SimpleNamespace(count=len(self.points))

    async def _scroll(self, **_kwargs: object) -> tuple[list[object], None]:
        points = [SimpleNamespace(id=point_id, payload=point["payload"]) for point_id, point in self.points.items()]
        return points, None


def metadata() -> VisionMetadata:
    return VisionMetadata(
        description="character tilts head with a question mark",
        primary_category="confusion",
        emotion_tags=["confused", "questioning"],
        scene_tags=["did not understand", "asking why"],
        ocr_text="?",
        suitable_scenarios=["expressing confusion", "asking for clarification"],
        unsuitable_scenarios=["formal apology"],
        safety=SafetyState.SAFE,
    )


def record(sticker_id: str = "11111111-1111-4111-8111-111111111111") -> StickerRecord:
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    data = metadata()
    return StickerRecord(
        id=sticker_id,
        sha256="a" * 64,
        asset_path="assets/a.png",
        thumbnail_path="thumbnails/a.webp",
        state=StickerState.ACTIVE,
        safety=SafetyState.SAFE,
        description=data.description,
        primary_category=data.primary_category,
        emotion_tags=data.emotion_tags,
        scene_tags=data.scene_tags,
        ocr_text=data.ocr_text,
        suitable_scenarios=data.suitable_scenarios,
        unsuitable_scenarios=data.unsuitable_scenarios,
        mime_type="image/png",
        width=8,
        height=8,
        frame_count=1,
        animated=False,
        byte_size=100,
        analysis_version="v1",
        vector_version=1,
        created_at=now,
        updated_at=now,
    )


def test_embedding_text_contains_every_labeled_field() -> None:
    from nekro_plugin_semantic_sticker.vector_store import build_embedding_text

    text = build_embedding_text(metadata())

    for label in (
        "description:", "category:", "emotion_tags:", "scene_tags:", "ocr:",
        "suitable_scenarios:", "unsuitable_scenarios:",
    ):
        assert label in text
    assert "character tilts head" in text
    assert "confused | questioning" in text


@pytest.mark.asyncio
async def test_embedding_provider_uses_named_group_and_validates_dimension() -> None:
    from nekro_plugin_semantic_sticker.vector_store import EmbeddingProvider, VectorDimensionMismatch

    group = SimpleNamespace(CHAT_MODEL="embedding-model", API_KEY="key", BASE_URL="https://embed.invalid/v1")
    core = SimpleNamespace(get_model_group_info=lambda name: group if name == "text-embedding" else None)
    embedding_function = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider = EmbeddingProvider(
        core,
        model_group_name="text-embedding",
        dimension=3,
        timeout=20,
        embedding_function=embedding_function,
    )

    assert await provider.embed("confusion") == [0.1, 0.2, 0.3]
    embedding_function.assert_awaited_once_with(
        model="embedding-model",
        input="confusion",
        dimensions=3,
        api_key="key",
        base_url="https://embed.invalid/v1",
        timeout=20,
    )
    embedding_function.return_value = [0.1]
    with pytest.raises(VectorDimensionMismatch):
        await provider.embed("wrong")


@pytest.mark.asyncio
async def test_absent_collection_is_created_once_with_cosine_distance() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    store = StickerVectorStore(fake, expected_dimension=3)

    await store.ensure_collection()
    await store.ensure_collection()

    fake.create_collection.assert_awaited_once_with(
        collection_name="Akiyo.semantic_sticker",
        vectors_config={"size": 3, "distance": "Cosine"},
    )
    assert fake.delete_collection.await_count == 0


@pytest.mark.asyncio
async def test_dimension_mismatch_does_not_recreate_collection() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore, VectorDimensionMismatch

    fake = FakeQdrant()
    fake.collection_dimension = 1024
    store = StickerVectorStore(fake, expected_dimension=1536)

    with pytest.raises(VectorDimensionMismatch, match="1024"):
        await store.ensure_collection()

    assert fake.create_collection.await_count == 0
    assert fake.delete_collection.await_count == 0


@pytest.mark.asyncio
async def test_upsert_is_idempotent_and_uses_exact_payload() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    fake.collection_dimension = 3
    store = StickerVectorStore(fake, expected_dimension=3)
    sticker = record()

    await store.upsert(sticker, [0.1, 0.2, 0.3])
    await store.upsert(sticker, [0.3, 0.2, 0.1])

    assert set(fake.points) == {sticker.id}
    point = fake.points[sticker.id]
    assert point["id"] == sticker.id
    assert point["vector"] == [0.3, 0.2, 0.1]
    assert point["payload"] == {
        "sticker_id": sticker.id,
        "state": "active",
        "safety": "safe",
        "primary_category": "confusion",
        "emotion_tags": ["confused", "questioning"],
        "scene_tags": ["did not understand", "asking why"],
        "ocr_text": "?",
    }


@pytest.mark.asyncio
async def test_upsert_rejects_wrong_vector_length_before_qdrant_call() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore, VectorDimensionMismatch

    fake = FakeQdrant()
    store = StickerVectorStore(fake, expected_dimension=3)

    with pytest.raises(VectorDimensionMismatch):
        await store.upsert(record(), [0.1])

    assert fake.upsert.await_count == 0


@pytest.mark.asyncio
async def test_search_uses_active_safe_filter_and_maps_hits() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    fake.collection_dimension = 3
    sticker = record()
    fake.search_results = [SimpleNamespace(id=sticker.id, score=0.91, payload={"sticker_id": sticker.id})]
    store = StickerVectorStore(fake, expected_dimension=3)

    hits = await store.search([0.1, 0.2, 0.3], limit=5, score_threshold=0.72)

    assert [(hit.sticker_id, hit.score) for hit in hits] == [(sticker.id, 0.91)]
    kwargs = fake.search.await_args.kwargs
    assert kwargs == {
        "collection_name": "Akiyo.semantic_sticker",
        "query_vector": [0.1, 0.2, 0.3],
        "query_filter": {
            "must": [
                {"key": "state", "match": {"value": "active"}},
                {"key": "safety", "match": {"value": "safe"}},
            ]
        },
        "limit": 5,
        "score_threshold": 0.72,
        "with_payload": True,
    }


@pytest.mark.asyncio
async def test_delete_is_idempotent_and_count_inventory_are_complete() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    fake.collection_dimension = 3
    store = StickerVectorStore(fake, expected_dimension=3)
    first = record()
    second = record("22222222-2222-4222-8222-222222222222")
    await store.upsert(first, [0.1, 0.2, 0.3])
    await store.upsert(second, [0.3, 0.2, 0.1])

    assert await store.count() == 2
    assert set(await store.inventory()) == {first.id, second.id}
    await store.delete(first.id)
    await store.delete(first.id)

    assert await store.count() == 1
    assert set(await store.inventory()) == {second.id}


@pytest.mark.asyncio
async def test_async_client_provider_is_resolved_once() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    fake.collection_dimension = 3
    provider = AsyncMock(return_value=fake)
    store = StickerVectorStore(client_provider=provider, expected_dimension=3)

    await store.ensure_collection()
    await store.count()

    provider.assert_awaited_once_with()
@pytest.mark.asyncio
async def test_dimension_validation_can_be_skipped_without_recreating() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    fake.collection_dimension = 2
    store = StickerVectorStore(fake, expected_dimension=3)

    await store.ensure_collection(validate_dimension=False)

    assert fake.delete_collection.await_count == 0
    assert fake.create_collection.await_count == 0


@pytest.mark.asyncio
async def test_recreate_collection_replaces_dimension_and_clears_all_points() -> None:
    from nekro_plugin_semantic_sticker.vector_store import StickerVectorStore

    fake = FakeQdrant()
    fake.collection_dimension = 2
    fake.points["orphan"] = {"id": "orphan", "vector": [1.0, 2.0], "payload": {"state": "active", "safety": "safe"}}
    store = StickerVectorStore(fake, expected_dimension=3)

    await store.recreate_collection()

    assert fake.collection_dimension == 3
    assert fake.points == {}
    fake.delete_collection.assert_awaited_once_with(collection_name="Akiyo.semantic_sticker")
    fake.create_collection.assert_awaited_once_with(
        collection_name="Akiyo.semantic_sticker",
        vectors_config={"size": 3, "distance": "Cosine"},
    )
