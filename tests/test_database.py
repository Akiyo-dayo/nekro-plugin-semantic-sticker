from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from nekro_plugin_semantic_sticker.models import (
    SafetyState,
    StickerRecord,
    StickerState,
    StoredAsset,
    VisionMetadata,
)


pytestmark = pytest.mark.asyncio


class FrozenClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


@pytest_asyncio.fixture
async def repository(tmp_path: Path):
    from nekro_plugin_semantic_sticker.database import StickerRepository

    clock = FrozenClock()
    repo = StickerRepository(tmp_path / "stickers.db", clock=clock)
    await repo.initialize()
    try:
        yield repo, clock
    finally:
        await repo.close()


def stored_asset(digest: str = "a" * 64) -> StoredAsset:
    return StoredAsset(
        sha256=digest, asset_path=f"/data/assets/{digest}.png",
        thumbnail_path=f"/data/thumbnails/{digest}.webp", detected_format="PNG",
        detected_extension="png", mime_type="image/png", byte_size=100, width=8, height=8,
        frame_count=1, animated=False,
    )


def safe_metadata() -> VisionMetadata:
    return VisionMetadata(
        description="character tilts head with a question mark", primary_category="confusion",
        emotion_tags=["confused"], scene_tags=["asking why"], ocr_text="?",
        suitable_scenarios=["asking for clarification"], unsuitable_scenarios=["formal apology"],
        safety=SafetyState.SAFE,
    )


async def create_active(repo, digest: str = "a" * 64) -> StickerRecord:
    record, duplicate = await repo.create_pending(stored_asset(digest), analysis_version="v1")
    assert duplicate is False
    await repo.transition(record.id, StickerState.ANALYZING)
    await repo.transition(record.id, StickerState.INDEXING, metadata=safe_metadata())
    return await repo.transition(record.id, StickerState.ACTIVE, vector_version=1)


async def test_database_enables_wal_foreign_keys_and_required_tables(repository) -> None:
    repo, _clock = repository
    assert (await repo.scalar("PRAGMA journal_mode")).lower() == "wal"
    assert await repo.scalar("PRAGMA foreign_keys") == 1
    tables = set(await repo.table_names())
    assert {"stickers", "jobs", "metadata_revisions", "usage_history"} <= tables


async def test_live_sha_is_unique_but_deleted_hash_can_be_uploaded_again(repository) -> None:
    repo, _clock = repository
    first, duplicate = await repo.create_pending(stored_asset(), analysis_version="v1")
    second, duplicate_again = await repo.create_pending(stored_asset(), analysis_version="v1")
    assert duplicate is False
    assert duplicate_again is True
    assert second.id == first.id
    await repo.transition(first.id, StickerState.ANALYZING)
    await repo.transition(first.id, StickerState.INDEXING, metadata=safe_metadata())
    await repo.transition(first.id, StickerState.ACTIVE, vector_version=1)
    await repo.transition(first.id, StickerState.DELETING)
    await repo.transition(first.id, StickerState.DELETED)
    replacement, replacement_duplicate = await repo.create_pending(stored_asset(), analysis_version="v1")
    assert replacement_duplicate is False
    assert replacement.id != first.id


async def test_state_machine_rejects_illegal_transitions_and_only_active_is_searchable(repository) -> None:
    from nekro_plugin_semantic_sticker.database import InvalidStateTransition

    repo, _clock = repository
    pending, _ = await repo.create_pending(stored_asset(), analysis_version="v1")
    assert await repo.list_searchable_ids() == set()
    with pytest.raises(InvalidStateTransition):
        await repo.transition(pending.id, StickerState.ACTIVE)
    active = await create_active(repo, "b" * 64)
    assert await repo.list_searchable_ids() == {active.id}
    assert active.primary_category == "confusion"
    assert active.ocr_text == "?"


async def test_recovery_is_idempotent_for_interrupted_stickers_and_open_jobs(repository) -> None:
    repo, _clock = repository
    pending, _ = await repo.create_pending(stored_asset("1" * 64), analysis_version="v1")
    pending_job_id = await repo.create_job(pending.id, "analysis")
    analyzing, _ = await repo.create_pending(stored_asset("c" * 64), analysis_version="v1")
    await repo.transition(analyzing.id, StickerState.ANALYZING)
    analyzing_job_id = await repo.create_job(analyzing.id, "analysis")
    await repo.update_job(analyzing_job_id, state="running", increment_attempt=True)
    indexing, _ = await repo.create_pending(stored_asset("d" * 64), analysis_version="v1")
    await repo.transition(indexing.id, StickerState.ANALYZING)
    await repo.transition(indexing.id, StickerState.INDEXING, metadata=safe_metadata())
    deleting = await create_active(repo, "e" * 64)
    await repo.transition(deleting.id, StickerState.DELETING)
    deleting_job_id = await repo.create_job(deleting.id, "analysis")

    first = await repo.recover_interrupted()
    second = await repo.recover_interrupted()
    assert set(first.retry_ids) == {pending.id, analyzing.id, indexing.id}
    assert first.deleting_ids == [deleting.id]
    assert set(second.retry_ids) == set(first.retry_ids)
    assert second.deleting_ids == [deleting.id]
    assert set(second.job_ids) == set(first.job_ids)
    assert pending_job_id in first.job_ids
    assert analyzing_job_id in first.job_ids
    assert deleting_job_id not in first.job_ids
    assert (await repo.get(pending.id)).state is StickerState.RETRY_PENDING
    assert (await repo.get(analyzing.id)).state is StickerState.RETRY_PENDING
    assert (await repo.get(indexing.id)).state is StickerState.RETRY_PENDING
    assert (await repo.get_job(analyzing_job_id)).state == "pending"
    assert (await repo.get_job(deleting_job_id)).state == "cancelled"
    assert await repo.scalar(
        "SELECT COUNT(*) FROM jobs WHERE sticker_id = ? AND job_type = 'analysis' AND state IN ('pending', 'running')",
        (indexing.id,),
    ) == 1


async def test_open_job_claim_is_atomic_and_dirty_duplicates_are_migrated(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.database import StickerRepository

    database_path = tmp_path / "legacy.db"
    repo = StickerRepository(database_path)
    await repo.initialize()
    active = await create_active(repo)
    connection = repo._require_connection()
    await connection.execute("DROP INDEX IF EXISTS uq_jobs_open")
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC).isoformat()
    await connection.executemany(
        "INSERT INTO jobs (id, sticker_id, job_type, state, attempt_count, created_at, updated_at) VALUES (?, ?, 'analysis', ?, 0, ?, ?)",
        [
            ("legacy-a", active.id, "pending", now, now),
            ("legacy-b", active.id, "running", now, now),
        ],
    )
    await connection.commit()
    await repo.close()

    migrated = StickerRepository(database_path)
    await migrated.initialize()
    try:
        assert await migrated.scalar(
            "SELECT COUNT(*) FROM jobs WHERE sticker_id = ? AND job_type = 'analysis' AND state IN ('pending', 'running')",
            (active.id,),
        ) == 1
        assert await migrated.scalar("SELECT COUNT(*) FROM jobs WHERE state = 'superseded'") == 1
        claims = await asyncio.gather(
            migrated.claim_open_job(active.id, "analysis"),
            migrated.claim_open_job(active.id, "analysis"),
        )
        assert claims[0][0].id == claims[1][0].id
        assert [created for _job, created in claims] == [False, False]
    finally:
        await migrated.close()


async def test_index_activation_compare_and_swap_rejects_deleted_work(repository) -> None:
    repo, _clock = repository
    record, _ = await repo.create_pending(stored_asset("f" * 64), analysis_version="v1")
    await repo.transition(record.id, StickerState.ANALYZING)
    indexing = await repo.transition(record.id, StickerState.INDEXING, metadata=safe_metadata())
    snapshot = await repo.indexing_snapshot(indexing.id)
    assert snapshot is not None
    _current, row_version = snapshot

    await repo.transition(indexing.id, StickerState.FAILED, error_summary="delete won")
    await repo.transition(indexing.id, StickerState.DELETING)

    assert await repo.activate_indexed(indexing.id, row_version=row_version, vector_version=1) is None


async def test_metadata_update_and_revision_are_one_transaction(repository) -> None:
    import aiosqlite

    repo, _clock = repository
    active = await create_active(repo)
    before = active.model_dump(mode="json")
    updated = active.model_copy(update={"description": "must roll back"})
    connection = repo._require_connection()
    await connection.execute(
        """
        CREATE TRIGGER reject_revision BEFORE INSERT ON metadata_revisions
        BEGIN
            SELECT RAISE(ABORT, 'revision rejected');
        END
        """
    )
    await connection.commit()

    with pytest.raises(aiosqlite.IntegrityError, match="revision rejected"):
        await repo.update_metadata_with_revision(
            updated,
            actor="admin",
            before=before,
            reason="atomic audit",
        )

    assert (await repo.get(active.id)).description == active.description
    assert await repo.revisions(active.id) == []


async def test_revision_records_actor_before_after_and_reason(repository) -> None:
    repo, _clock = repository
    active = await create_active(repo)
    before = active.model_dump(mode="json")
    updated = active.model_copy(update={"description": "updated confusion"})
    await repo.update_metadata(updated)
    await repo.record_revision(active.id, actor="admin", before=before, after=updated.model_dump(mode="json"), reason="improve retrieval")
    revisions = await repo.revisions(active.id)
    assert revisions == [{
        "actor": "admin", "before": before, "after": updated.model_dump(mode="json"),
        "reason": "improve retrieval", "created_at": "2026-08-03T12:00:00+00:00",
    }]


async def test_usage_history_tracks_logical_physical_turn_and_cooldown(repository) -> None:
    repo, clock = repository
    active = await create_active(repo)
    await repo.record_usage(
        active.id, logical_chat_key="onebot_v11-group_100-user_1",
        physical_channel_key="onebot_v11-group_100", agent_turn_key="turn-1", score=0.91,
    )
    assert await repo.recent_sticker_ids("onebot_v11-group_100-user_1", limit=10) == [active.id]
    assert await repo.physical_channel_in_cooldown("onebot_v11-group_100", cooldown_seconds=20) is True
    assert await repo.agent_turn_has_send("turn-1") is True
    clock.advance(21)
    assert await repo.physical_channel_in_cooldown("onebot_v11-group_100", cooldown_seconds=20) is False


async def test_snapshot_counts_all_plugin_rows(repository) -> None:
    repo, _clock = repository
    active = await create_active(repo)
    await repo.create_job(active.id, "reindex")
    await repo.record_revision(active.id, "admin", {}, {"x": 1}, "test")
    await repo.record_usage(active.id, "chat", "channel", "turn", 0.8)
    snapshot = await repo.snapshot()
    assert snapshot == {"stickers": 1, "jobs": 1, "metadata_revisions": 1, "usage_history": 1}