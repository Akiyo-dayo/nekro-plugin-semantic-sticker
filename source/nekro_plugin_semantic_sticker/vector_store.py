from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import StickerRecord, VisionMetadata


COLLECTION_NAME = "Akiyo.semantic_sticker"


class VectorDimensionMismatch(ValueError):
    pass


@dataclass(frozen=True)
class VectorSearchHit:
    sticker_id: str
    score: float
    payload: dict[str, Any]


def build_embedding_text(metadata: VisionMetadata | StickerRecord) -> str:
    return "\n".join(
        (
            f"description: {metadata.description}",
            f"category: {metadata.primary_category}",
            f"emotion_tags: {' | '.join(metadata.emotion_tags)}",
            f"scene_tags: {' | '.join(metadata.scene_tags)}",
            f"ocr: {metadata.ocr_text}",
            f"suitable_scenarios: {' | '.join(metadata.suitable_scenarios)}",
            f"unsuitable_scenarios: {' | '.join(metadata.unsuitable_scenarios)}",
        )
    )


async def _default_embedding_function(**kwargs: object) -> list[float]:
    from nekro_agent.services.agent.openai import gen_openai_embeddings

    return await gen_openai_embeddings(**kwargs)


class EmbeddingProvider:
    def __init__(
        self,
        core_config: object,
        *,
        model_group_name: str,
        dimension: int,
        timeout: int = 60,
        embedding_function: Callable[..., Awaitable[Sequence[float]]] | None = None,
    ) -> None:
        self.core_config = core_config
        self.model_group_name = model_group_name
        self.dimension = int(dimension)
        self.timeout = timeout
        self.embedding_function = embedding_function or _default_embedding_function

    async def embed(self, text: str) -> list[float]:
        getter = getattr(self.core_config, "get_model_group_info", None)
        if getter is None:
            raise LookupError("embedding model group resolver is unavailable")
        group = getter(self.model_group_name)
        if group is None:
            raise LookupError(f"embedding model group '{self.model_group_name}' was not found")
        vector = await self.embedding_function(
            model=str(getattr(group, "CHAT_MODEL")),
            input=text,
            dimensions=self.dimension,
            api_key=str(getattr(group, "API_KEY")),
            base_url=str(getattr(group, "BASE_URL")),
            timeout=self.timeout,
        )
        normalized = [float(value) for value in vector]
        _validate_dimension(normalized, self.dimension)
        return normalized


async def _default_client_provider() -> object:
    from nekro_agent.api.core import get_qdrant_client

    client = await get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client is unavailable")
    return client


def _validate_dimension(vector: Sequence[float], expected: int) -> None:
    actual = len(vector)
    if actual != expected:
        raise VectorDimensionMismatch(f"vector dimension mismatch: expected {expected}, got {actual}")


def _qdrant_models() -> object | None:
    try:
        from qdrant_client import models
    except ModuleNotFoundError:
        return None
    return models


def _vector_params(dimension: int) -> object:
    models = _qdrant_models()
    if models is None:
        return {"size": dimension, "distance": "Cosine"}
    return models.VectorParams(size=dimension, distance=models.Distance.COSINE)


def _point_struct(point_id: str, vector: Sequence[float], payload: dict[str, Any]) -> object:
    models = _qdrant_models()
    if models is None:
        return {"id": point_id, "vector": list(vector), "payload": payload}
    return models.PointStruct(id=point_id, vector=list(vector), payload=payload)


def _active_safe_filter() -> object:
    models = _qdrant_models()
    if models is None:
        return {
            "must": [
                {"key": "state", "match": {"value": "active"}},
                {"key": "safety", "match": {"value": "safe"}},
            ]
        }
    return models.Filter(
        must=[
            models.FieldCondition(key="state", match=models.MatchValue(value="active")),
            models.FieldCondition(key="safety", match=models.MatchValue(value="safe")),
        ]
    )


def _point_ids_selector(point_id: str) -> object:
    models = _qdrant_models()
    if models is None:
        return {"points": [point_id]}
    return models.PointIdsList(points=[point_id])


def _extract_collection_dimension(info: object) -> int:
    try:
        vectors = info.config.params.vectors
    except AttributeError as error:
        raise VectorDimensionMismatch("existing collection dimension could not be determined") from error
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, Mapping):
        if "size" in vectors:
            return int(vectors["size"])
        dimensions = [int(value.size) for value in vectors.values() if hasattr(value, "size")]
        if len(dimensions) == 1:
            return dimensions[0]
    raise VectorDimensionMismatch("existing collection uses an unsupported vector configuration")


def _collection_names(response: object) -> set[str]:
    collections = getattr(response, "collections", [])
    return {str(collection.name) for collection in collections}


class StickerVectorStore:
    def __init__(
        self,
        client: object | None = None,
        *,
        client_provider: Callable[[], Awaitable[object]] | None = None,
        expected_dimension: int,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        if client is not None and client_provider is not None:
            raise ValueError("provide either a Qdrant client or client provider, not both")
        self._client_instance = client
        self._client_provider = client_provider or _default_client_provider
        self._client_lock = asyncio.Lock()
        self._collection_lock = asyncio.Lock()
        self.expected_dimension = int(expected_dimension)
        self.collection_name = collection_name

    async def _client(self) -> object:
        if self._client_instance is not None:
            return self._client_instance
        async with self._client_lock:
            if self._client_instance is None:
                self._client_instance = await self._client_provider()
        return self._client_instance

    async def ensure_collection(
        self,
        expected_dimension: int | None = None,
        *,
        validate_dimension: bool = True,
    ) -> None:
        dimension = self.expected_dimension if expected_dimension is None else int(expected_dimension)
        client = await self._client()
        async with self._collection_lock:
            collections = await client.get_collections()
            if self.collection_name not in _collection_names(collections):
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=_vector_params(dimension),
                )
                return
            if not validate_dimension:
                return
            info = await client.get_collection(self.collection_name)
            actual = _extract_collection_dimension(info)
            if actual != dimension:
                raise VectorDimensionMismatch(
                    f"collection dimension mismatch: expected {dimension}, got {actual}"
                )

    async def recreate_collection(self, expected_dimension: int | None = None) -> None:
        dimension = self.expected_dimension if expected_dimension is None else int(expected_dimension)
        client = await self._client()
        async with self._collection_lock:
            collections = await client.get_collections()
            if self.collection_name in _collection_names(collections):
                await client.delete_collection(collection_name=self.collection_name)
            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=_vector_params(dimension),
            )
    async def upsert(self, record: StickerRecord, vector: Sequence[float]) -> None:
        normalized = [float(value) for value in vector]
        _validate_dimension(normalized, self.expected_dimension)
        payload = {
            "sticker_id": record.id,
            "state": record.state.value,
            "safety": record.safety.value,
            "primary_category": record.primary_category,
            "emotion_tags": list(record.emotion_tags),
            "scene_tags": list(record.scene_tags),
            "ocr_text": record.ocr_text,
        }
        client = await self._client()
        await client.upsert(
            collection_name=self.collection_name,
            points=[_point_struct(record.id, normalized, payload)],
            wait=True,
        )

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        score_threshold: float,
    ) -> list[VectorSearchHit]:
        normalized = [float(value) for value in vector]
        _validate_dimension(normalized, self.expected_dimension)
        client = await self._client()
        results = await client.search(
            collection_name=self.collection_name,
            query_vector=normalized,
            query_filter=_active_safe_filter(),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        hits: list[VectorSearchHit] = []
        for result in results:
            payload = dict(getattr(result, "payload", None) or {})
            sticker_id = str(payload.get("sticker_id") or getattr(result, "id"))
            hits.append(
                VectorSearchHit(
                    sticker_id=sticker_id,
                    score=float(getattr(result, "score")),
                    payload=payload,
                )
            )
        return hits

    async def delete(self, sticker_id: str) -> None:
        client = await self._client()
        await client.delete(
            collection_name=self.collection_name,
            points_selector=_point_ids_selector(sticker_id),
            wait=True,
        )

    async def count(self) -> int:
        client = await self._client()
        response = await client.count(collection_name=self.collection_name, exact=True)
        return int(response.count)

    async def inventory(self) -> dict[str, dict[str, Any]]:
        client = await self._client()
        inventory: dict[str, dict[str, Any]] = {}
        offset: object | None = None
        while True:
            points, next_offset = await client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                inventory[str(point.id)] = dict(getattr(point, "payload", None) or {})
            if next_offset is None:
                break
            offset = next_offset
        return inventory


__all__ = [
    "COLLECTION_NAME",
    "EmbeddingProvider",
    "StickerVectorStore",
    "VectorDimensionMismatch",
    "VectorSearchHit",
    "build_embedding_text",
]