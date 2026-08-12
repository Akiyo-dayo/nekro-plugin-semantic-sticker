from __future__ import annotations

import asyncio
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from PIL import Image

from nekro_plugin_semantic_sticker.config import SemanticStickerConfig
from nekro_plugin_semantic_sticker.database import StickerRepository
from nekro_plugin_semantic_sticker.files import ImageStore
from nekro_plugin_semantic_sticker.models import (
    MetadataPatch,
    SafetyState,
    StickerFilters,
    StickerState,
    UploadPayload,
    VisionMetadata,
)


def png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    from io import BytesIO

    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def safe_metadata(description: str = "confused question mark") -> VisionMetadata:
    return VisionMetadata(
        description=description,
        primary_category="confusion",
        emotion_tags=["confused"],
        scene_tags=["asking why"],
        ocr_text="?",
        suitable_scenarios=["asking for clarification"],
        unsuitable_scenarios=["formal apology"],
        safety=SafetyState.SAFE,
    )


def unsafe_metadata() -> VisionMetadata:
    return safe_metadata("unsafe image").model_copy(update={"safety": SafetyState.UNSAFE})


class FakeAnalyzer:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes: deque[object] = deque(outcomes or [safe_metadata()])
        self.calls: list[dict[str, object]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.concurrent = 0
        self.max_concurrent = 0

    async def analyze(self, asset_path: Path, *, mime_type: str, prompt_version: str) -> VisionMetadata:
        self.calls.append({"asset_path": Path(asset_path), "mime_type": mime_type, "prompt_version": prompt_version})
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.set()
        try:
            await self.release.wait()
            outcome = self.outcomes.popleft() if self.outcomes else safe_metadata()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.concurrent -= 1


class FakeEmbedding:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * self.dimension


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, object]] = {}
        self.ensure_calls: list[bool] = []
        self.dimension_mismatch = False
        self.fail_upsert = False
        self.fail_delete_attempts = 0
        self.upsert_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.recreate_calls = 0
        self.upsert_started = asyncio.Event()
        self.upsert_release = asyncio.Event()
        self.upsert_release.set()

    async def ensure_collection(self, *, validate_dimension: bool = True) -> None:
        self.ensure_calls.append(validate_dimension)
        if self.dimension_mismatch and validate_dimension:
            from nekro_plugin_semantic_sticker.vector_store import VectorDimensionMismatch

            raise VectorDimensionMismatch("collection dimension mismatch: expected 3, got 2")

    async def recreate_collection(self) -> None:
        self.recreate_calls += 1
        self.dimension_mismatch = False
        self.points.clear()

    async def upsert(self, record: object, vector: list[float]) -> None:
        self.upsert_calls.append(record.id)
        self.upsert_started.set()
        await self.upsert_release.wait()
        if self.fail_upsert:
            raise RuntimeError("qdrant unavailable secret")
        self.points[record.id] = {"record": record, "vector": list(vector)}

    async def delete(self, sticker_id: str) -> None:
        self.delete_calls.append(sticker_id)
        if self.fail_delete_attempts:
            self.fail_delete_attempts -= 1
            raise RuntimeError("qdrant delete unavailable")
        self.points.pop(sticker_id, None)

    async def count(self) -> int:
        return len(self.points)

@pytest_asyncio.fixture
async def harness(tmp_path: Path):
    from nekro_plugin_semantic_sticker.service import StickerService

    config = SemanticStickerConfig(
        ANALYSIS_CONCURRENCY=1,
        VECTOR_DIMENSION=3,
        ANALYSIS_PROMPT_VERSION="v1",
    )
    repository = StickerRepository(tmp_path / "stickers.db")
    image_store = ImageStore(tmp_path / "plugin_data", config)
    analyzer = FakeAnalyzer()
    embedding = FakeEmbedding()
    vector_store = FakeVectorStore()
    service = StickerService(config, repository, image_store, analyzer, embedding, vector_store)
    yield service, repository, image_store, analyzer, embedding, vector_store
    await service.close()


async def upload(service: object, color: tuple[int, int, int], actor: str = "admin"):
    return await service.upload(
        UploadPayload(content=png_bytes(color), filename="sticker.png", content_type="image/png"),
        actor=actor,
    )


@pytest.mark.asyncio
async def test_upload_returns_persisted_job_before_analysis_finishes(harness) -> None:
    service, repository, _store, analyzer, _embedding, _vector = harness
    analyzer.release.clear()

    outcome = await upload(service, (255, 0, 0))

    assert outcome.record.state is StickerState.PENDING
    assert outcome.job is not None and outcome.job.state == "pending"
    assert analyzer.started.is_set() is False
    assert await repository.scalar("SELECT COUNT(*) FROM jobs") == 1
    analyzer.release.set()
    await service.wait_for_idle()
    assert (await service.get_sticker(outcome.record.id)).state is StickerState.ACTIVE


@pytest.mark.asyncio
async def test_duplicate_upload_creates_no_new_row_file_point_or_job(harness) -> None:
    service, repository, image_store, analyzer, _embedding, vector_store = harness
    first = await upload(service, (0, 255, 0))
    await service.wait_for_idle()
    before_files = image_store.snapshot()
    before_calls = len(analyzer.calls)

    duplicate = await upload(service, (0, 255, 0))
    await service.wait_for_idle()

    assert duplicate.duplicate is True
    assert duplicate.record.id == first.record.id
    assert duplicate.job is None
    assert await repository.scalar("SELECT COUNT(*) FROM stickers") == 1
    assert await repository.scalar("SELECT COUNT(*) FROM jobs") == 1
    assert image_store.snapshot() == before_files
    assert len(analyzer.calls) == before_calls
    assert len(vector_store.points) == 1


@pytest.mark.asyncio
async def test_queue_concurrency_is_bounded_to_configured_one(harness) -> None:
    service, _repository, _store, analyzer, _embedding, _vector = harness
    analyzer.outcomes = deque([safe_metadata("one"), safe_metadata("two"), safe_metadata("three")])
    analyzer.release.clear()
    outcomes = [await upload(service, color) for color in ((1, 0, 0), (2, 0, 0), (3, 0, 0))]
    await asyncio.wait_for(analyzer.started.wait(), timeout=1)

    assert analyzer.max_concurrent == 1
    analyzer.release.set()
    await service.wait_for_idle()
    assert analyzer.max_concurrent == 1
    states = [(await service.get_sticker(item.record.id)).state for item in outcomes]
    assert all(state is StickerState.ACTIVE for state in states)


@pytest.mark.asyncio
async def test_unsafe_result_is_retained_but_never_indexed(harness) -> None:
    service, repository, _store, analyzer, _embedding, vector_store = harness
    analyzer.outcomes = deque([unsafe_metadata()])

    outcome = await upload(service, (10, 10, 10))
    await service.wait_for_idle()
    record = await service.get_sticker(outcome.record.id)

    assert record.state is StickerState.FAILED
    assert record.safety is SafetyState.UNSAFE
    assert await repository.list_searchable_ids() == set()
    assert vector_store.points == {}
    assert (await repository.get_job(outcome.job.id)).state == "completed"


@pytest.mark.asyncio
async def test_analysis_failure_records_sanitized_failure_and_job(harness) -> None:
    service, repository, _store, analyzer, _embedding, vector_store = harness
    analyzer.outcomes = deque([RuntimeError("provider secret-token")])

    outcome = await upload(service, (20, 20, 20))
    await service.wait_for_idle()
    record = await service.get_sticker(outcome.record.id)
    job = await repository.get_job(outcome.job.id)

    assert record.state is StickerState.FAILED
    assert "secret-token" not in (record.error_summary or "")
    assert job.state == "failed" and job.attempt_count == 1
    assert "secret-token" not in (job.error_summary or "")
    assert vector_store.points == {}


@pytest.mark.asyncio
async def test_manual_patch_and_reindex_do_not_call_vision(harness) -> None:
    service, repository, _store, analyzer, embedding, vector_store = harness
    outcome = await upload(service, (30, 30, 30))
    await service.wait_for_idle()
    analysis_calls = len(analyzer.calls)
    before_embedding_calls = len(embedding.calls)

    updated = await service.patch_metadata(
        outcome.record.id,
        MetadataPatch(description="confused and asking why", reason="improve retrieval"),
        actor="admin-user",
    )
    reindexed = await service.reindex(updated.id, actor="admin-user")

    assert reindexed.description == "confused and asking why"
    assert reindexed.vector_version == 2
    assert len(analyzer.calls) == analysis_calls
    assert len(embedding.calls) == before_embedding_calls + 1
    assert vector_store.points[updated.id]["record"].state is StickerState.ACTIVE
    revisions = await repository.revisions(updated.id)
    assert revisions[-1]["actor"] == "admin-user"
    assert revisions[-1]["reason"] == "improve retrieval"


@pytest.mark.asyncio
async def test_vector_failure_never_marks_record_active(harness) -> None:
    service, repository, _store, _analyzer, _embedding, vector_store = harness
    vector_store.fail_upsert = True

    outcome = await upload(service, (40, 40, 40))
    await service.wait_for_idle()

    assert (await service.get_sticker(outcome.record.id)).state is StickerState.FAILED
    assert await repository.list_searchable_ids() == set()
    assert outcome.record.id not in vector_store.points


@pytest.mark.asyncio
async def test_failed_or_rejected_record_can_be_safely_reanalyzed(harness) -> None:
    service, _repository, _store, analyzer, _embedding, vector_store = harness
    analyzer.outcomes = deque([unsafe_metadata(), safe_metadata("now safe")])
    outcome = await upload(service, (50, 50, 50))
    await service.wait_for_idle()
    assert (await service.get_sticker(outcome.record.id)).safety is SafetyState.UNSAFE

    job = await service.reanalyze(outcome.record.id, actor="admin")
    await service.wait_for_idle()
    final = await service.get_sticker(outcome.record.id)

    assert job.job_type == "analysis"
    assert final.state is StickerState.ACTIVE
    assert final.safety is SafetyState.SAFE
    assert final.description == "now safe"
    assert outcome.record.id in vector_store.points


@pytest.mark.asyncio
async def test_single_and_batch_delete_remove_vectors_and_managed_files(harness) -> None:
    service, _repository, _store, _analyzer, _embedding, vector_store = harness
    first = await upload(service, (60, 0, 0))
    second = await upload(service, (70, 0, 0))
    third = await upload(service, (80, 0, 0))
    await service.wait_for_idle()
    first_record = await service.get_sticker(first.record.id)
    assert Path(first_record.asset_path).exists()

    await service.delete(first.record.id, actor="admin")
    result = await service.batch_delete([second.record.id, third.record.id, "missing"], actor="admin")

    assert (await service.get_sticker(first.record.id)).state is StickerState.DELETED
    assert not Path(first_record.asset_path).exists()
    assert first.record.id not in vector_store.points
    assert result.requested == 3 and result.deleted == 2 and result.failed_ids == ["missing"]


@pytest.mark.asyncio
async def test_list_filters_stats_and_full_reindex(harness) -> None:
    service, _repository, _store, _analyzer, embedding, _vector = harness
    first = await upload(service, (90, 0, 0))
    second = await upload(service, (100, 0, 0))
    await service.wait_for_idle()
    await service.patch_metadata(
        second.record.id,
        MetadataPatch(primary_category="happiness", emotion_tags=["happy"], reason="correct category"),
        actor="admin",
    )

    page = await service.list_stickers(StickerFilters(category="happiness", tags=["happy"], limit=10))
    stats = await service.stats()
    before = len(embedding.calls)
    result = await service.full_reindex(actor="admin")

    assert [item.id for item in page.items] == [second.record.id]
    assert page.total == 1
    assert stats.total == 2 and stats.indexed_count == 2 and stats.failure_count == 0
    assert stats.by_category["happiness"] == 1
    assert result.requested == 2 and result.indexed == 2 and result.failed_ids == []
    assert len(embedding.calls) == before + 2
@pytest.mark.asyncio
async def test_ensure_started_retries_after_recovery_failure_and_sets_started_last(harness) -> None:
    service, repository, image_store, _analyzer, _embedding, vector_store = harness
    await repository.initialize()
    validated = image_store.validate(
        UploadPayload(content=png_bytes((110, 0, 0)), filename="recover.png", content_type="image/png")
    )
    asset = image_store.install(validated)
    record, _ = await repository.create_pending(asset, analysis_version="v1")
    await repository.transition(record.id, StickerState.ANALYZING)
    await repository.transition(record.id, StickerState.INDEXING, metadata=safe_metadata())
    active = await repository.transition(record.id, StickerState.ACTIVE, vector_version=1)
    await repository.transition(active.id, StickerState.DELETING)
    vector_store.points[active.id] = {"record": active, "vector": [0.1, 0.1, 0.1]}
    vector_store.fail_delete_attempts = 1

    with pytest.raises(RuntimeError, match="delete unavailable"):
        await service.ensure_started()
    assert service._started is False

    await service.ensure_started()

    assert service._started is True
    assert (await repository.get(active.id)).state is StickerState.DELETED
    assert vector_store.delete_calls.count(active.id) == 2


@pytest.mark.asyncio
async def test_startup_requeues_existing_running_job_without_creating_duplicate(harness) -> None:
    service, repository, image_store, _analyzer, _embedding, _vector_store = harness
    await repository.initialize()
    validated = image_store.validate(
        UploadPayload(content=png_bytes((120, 0, 0)), filename="resume.png", content_type="image/png")
    )
    asset = image_store.install(validated)
    record, _ = await repository.create_pending(asset, analysis_version="v1")
    job_id = await repository.create_job(record.id, "analysis")
    await repository.update_job(job_id, state="running", increment_attempt=True)

    await service.ensure_started()
    await service.wait_for_idle()

    assert (await repository.get(record.id)).state is StickerState.ACTIVE
    assert (await repository.get_job(job_id)).state == "completed"
    assert await repository.scalar("SELECT COUNT(*) FROM jobs WHERE sticker_id = ?", (record.id,)) == 1


@pytest.mark.asyncio
async def test_delete_during_analysis_cannot_resurrect_qdrant_point(harness) -> None:
    service, _repository, _store, analyzer, _embedding, vector_store = harness
    analyzer.release.clear()
    outcome = await upload(service, (130, 0, 0))
    await asyncio.wait_for(analyzer.started.wait(), timeout=1)

    await service.delete(outcome.record.id, actor="admin")
    analyzer.release.set()
    await service.wait_for_idle()

    assert (await service.get_sticker(outcome.record.id)).state is StickerState.DELETED
    assert outcome.record.id not in vector_store.points


@pytest.mark.asyncio
async def test_delete_during_reindex_cannot_resurrect_qdrant_point(harness) -> None:
    service, _repository, _store, _analyzer, _embedding, vector_store = harness
    outcome = await upload(service, (140, 0, 0))
    await service.wait_for_idle()
    vector_store.upsert_started.clear()
    vector_store.upsert_release.clear()

    reindex_task = asyncio.create_task(service.reindex(outcome.record.id, actor="admin"))
    await asyncio.wait_for(vector_store.upsert_started.wait(), timeout=1)
    await service.delete(outcome.record.id, actor="admin")
    vector_store.upsert_release.set()
    with pytest.raises(Exception):
        await reindex_task

    assert (await service.get_sticker(outcome.record.id)).state is StickerState.DELETED
    assert outcome.record.id not in vector_store.points


@pytest.mark.asyncio
async def test_leaving_active_for_reanalysis_removes_searchable_point_immediately(harness) -> None:
    service, _repository, _store, analyzer, _embedding, vector_store = harness
    outcome = await upload(service, (150, 0, 0))
    await service.wait_for_idle()
    assert outcome.record.id in vector_store.points
    analyzer.started.clear()
    analyzer.release.clear()

    await service.reanalyze(outcome.record.id, actor="admin")
    await asyncio.wait_for(analyzer.started.wait(), timeout=1)

    assert outcome.record.id not in vector_store.points
    analyzer.release.set()
    await service.wait_for_idle()


@pytest.mark.asyncio
async def test_full_reindex_recreates_mismatched_collection_and_clears_orphans(harness) -> None:
    service, _repository, _store, _analyzer, _embedding, vector_store = harness
    vector_store.dimension_mismatch = True
    first = await upload(service, (160, 0, 0))
    second = await upload(service, (170, 0, 0))
    await service.wait_for_idle()
    vector_store.points["orphan"] = {"record": None, "vector": [9.0, 9.0]}

    result = await service.full_reindex(actor="admin")

    assert result.indexed == 2 and result.failed_ids == []
    assert vector_store.recreate_calls == 1
    assert set(vector_store.points) == {first.record.id, second.record.id}
    assert vector_store.ensure_calls[0] is False


@pytest.mark.asyncio
async def test_upload_persists_intent_before_install(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    service, repository, image_store, _analyzer, _embedding, _vector_store = harness
    original_install = image_store.install

    def install_after_intent(validated):
        with sqlite3.connect(repository.database_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM stickers WHERE sha256 = ? AND state != 'deleted'",
                (validated.sha256,),
            ).fetchone()[0]
        assert count == 1
        return original_install(validated)

    monkeypatch.setattr(image_store, "install", install_after_intent)

    outcome = await upload(service, (180, 0, 0))
    await service.wait_for_idle()

    assert outcome.duplicate is False


@pytest.mark.asyncio
async def test_failed_install_leaves_cleanup_intent_and_same_hash_can_retry(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    service, repository, image_store, _analyzer, _embedding, _vector_store = harness
    original_install = image_store.install
    attempts = 0

    def fail_once(validated):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("disk write failed")
        return original_install(validated)

    monkeypatch.setattr(image_store, "install", fail_once)
    payload = UploadPayload(content=png_bytes((190, 0, 0)), filename="retry.png", content_type="image/png")

    with pytest.raises(RuntimeError, match="disk write failed"):
        await service.upload(payload, actor="admin")
    assert await repository.scalar("SELECT COUNT(*) FROM stickers WHERE state = 'deleted'") == 1

    outcome = await service.upload(payload, actor="admin")
    await service.wait_for_idle()

    assert outcome.duplicate is False
    assert await repository.scalar("SELECT COUNT(*) FROM stickers WHERE state != 'deleted'") == 1


@pytest.mark.asyncio
async def test_concurrent_same_hash_upload_keeps_one_live_row_asset_and_job(harness) -> None:
    service, repository, image_store, _analyzer, _embedding, _vector_store = harness
    payload = UploadPayload(content=png_bytes((200, 0, 0)), filename="same.png", content_type="image/png")

    first, second = await asyncio.gather(
        service.upload(payload, actor="admin-a"),
        service.upload(payload, actor="admin-b"),
    )
    await service.wait_for_idle()

    assert sorted([first.duplicate, second.duplicate]) == [False, True]
    assert first.record.id == second.record.id
    assert await repository.scalar("SELECT COUNT(*) FROM stickers WHERE state != 'deleted'") == 1
    assert await repository.scalar("SELECT COUNT(*) FROM jobs") == 1
    assert len(image_store.snapshot().assets) == 1


@pytest.mark.asyncio
async def test_reanalyze_updates_analysis_version(harness) -> None:
    service, _repository, _store, _analyzer, _embedding, _vector_store = harness
    outcome = await upload(service, (210, 0, 0))
    await service.wait_for_idle()
    service.config.ANALYSIS_PROMPT_VERSION = "v2"

    await service.reanalyze(outcome.record.id, actor="admin")
    await service.wait_for_idle()

    assert (await service.get_sticker(outcome.record.id)).analysis_version == "v2"
@pytest.mark.asyncio
async def test_concurrent_reanalyze_creates_only_one_open_analysis_job(harness) -> None:
    from nekro_plugin_semantic_sticker.service import StickerBusyError

    service, repository, _store, analyzer, _embedding, _vector_store = harness
    outcome = await upload(service, (220, 0, 0))
    await service.wait_for_idle()
    analyzer.started.clear()
    analyzer.release.clear()

    results = await asyncio.gather(
        service.reanalyze(outcome.record.id, actor="admin-a"),
        service.reanalyze(outcome.record.id, actor="admin-b"),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, StickerBusyError) for result in results) == 1
    assert await repository.scalar(
        "SELECT COUNT(*) FROM jobs WHERE sticker_id = ? AND job_type = 'analysis' AND state IN ('pending', 'running')",
        (outcome.record.id,),
    ) == 1
    analyzer.release.set()
    await service.wait_for_idle()
@pytest.mark.asyncio
async def test_startup_discards_durable_intent_when_files_were_never_installed(harness) -> None:
    service, repository, image_store, _analyzer, _embedding, vector_store = harness
    await repository.initialize()
    validated = image_store.validate(
        UploadPayload(content=png_bytes((230, 0, 0)), filename="crash.png", content_type="image/png")
    )
    planned = service._planned_asset(validated)
    record, _ = await repository.create_pending(planned, analysis_version="v1")
    job_id = await repository.create_job(record.id, "analysis")
    vector_store.points[record.id] = {"record": record, "vector": [0.1, 0.1, 0.1]}

    await service.ensure_started()
    await service.wait_for_idle()

    assert (await repository.get(record.id)).state is StickerState.DELETED
    assert (await repository.get_job(job_id)).state == "cancelled"
    assert record.id not in vector_store.points
    assert not Path(planned.asset_path).exists()
    assert not Path(planned.thumbnail_path).exists()
