from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import aiosqlite

from .models import (
    JobRecord,
    SafetyState,
    StickerFilters,
    StickerPage,
    StickerRecord,
    StickerState,
    StickerStats,
    StoredAsset,
    VisionMetadata,
)


class InvalidStateTransition(ValueError):
    pass


class StickerNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class RecoveryPlan:
    retry_ids: list[str]
    deleting_ids: list[str]
    job_ids: list[str]


ALLOWED_TRANSITIONS: dict[StickerState, set[StickerState]] = {
    StickerState.PENDING: {StickerState.ANALYZING, StickerState.RETRY_PENDING, StickerState.FAILED},
    StickerState.ANALYZING: {StickerState.INDEXING, StickerState.RETRY_PENDING, StickerState.FAILED},
    StickerState.INDEXING: {StickerState.ACTIVE, StickerState.RETRY_PENDING, StickerState.FAILED},
    StickerState.ACTIVE: {StickerState.INDEXING, StickerState.DELETING},
    StickerState.FAILED: {StickerState.RETRY_PENDING, StickerState.DELETING},
    StickerState.RETRY_PENDING: {StickerState.ANALYZING, StickerState.DELETING},
    StickerState.DELETING: {StickerState.DELETED},
    StickerState.DELETED: set(),
}


class StickerRepository:
    def __init__(self, database_path: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self.database_path = Path(database_path)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._connection is not None:
            return
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.database_path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.executescript(_SCHEMA)
        await self._migrate_schema(connection)
        await connection.commit()
        self._connection = connection

    async def _migrate_schema(self, connection: aiosqlite.Connection) -> None:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            async with connection.execute("PRAGMA table_info(stickers)") as cursor:
                sticker_columns = {str(row[1]) for row in await cursor.fetchall()}
            if "row_version" not in sticker_columns:
                await connection.execute(
                    "ALTER TABLE stickers ADD COLUMN row_version INTEGER NOT NULL DEFAULT 0"
                )

            async with connection.execute(
                """
                SELECT sticker_id, job_type
                FROM jobs
                WHERE state IN ('pending', 'running')
                GROUP BY sticker_id, job_type
                HAVING COUNT(*) > 1
                """
            ) as cursor:
                duplicate_groups = await cursor.fetchall()
            now = self._now_text()
            for group in duplicate_groups:
                async with connection.execute(
                    """
                    SELECT id
                    FROM jobs
                    WHERE sticker_id = ? AND job_type = ? AND state IN ('pending', 'running')
                    ORDER BY created_at, id
                    """,
                    (group["sticker_id"], group["job_type"]),
                ) as cursor:
                    duplicate_rows = await cursor.fetchall()
                duplicate_ids = [str(row["id"]) for row in duplicate_rows[1:]]
                if duplicate_ids:
                    placeholders = ",".join("?" for _ in duplicate_ids)
                    await connection.execute(
                        f"""
                        UPDATE jobs
                        SET state = 'superseded', error_summary = ?, updated_at = ?
                        WHERE id IN ({placeholders})
                        """,
                        ("duplicate open job reconciled", now, *duplicate_ids),
                    )
            await connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_open
                ON jobs(sticker_id, job_type)
                WHERE state IN ('pending', 'running')
                """
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def scalar(self, query: str, parameters: tuple[Any, ...] = ()) -> Any:
        connection = self._require_connection()
        async with connection.execute(query, parameters) as cursor:
            row = await cursor.fetchone()
        return None if row is None else row[0]

    async def table_names(self) -> list[str]:
        connection = self._require_connection()
        async with connection.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def create_pending(self, asset: StoredAsset, *, analysis_version: str) -> tuple[StickerRecord, bool]:
        connection = self._require_connection()
        async with self._lock:
            existing = await self._get_live_by_sha(asset.sha256)
            if existing is not None:
                return existing, True
            now = self._now_text()
            sticker_id = str(uuid.uuid4())
            try:
                await connection.execute(
                    """
                    INSERT INTO stickers (
                        id, sha256, asset_path, thumbnail_path, mime_type, width, height,
                        frame_count, animated, byte_size, state, safety, description,
                        primary_category, emotion_tags, scene_tags, ocr_text,
                        suitable_scenarios, unsuitable_scenarios, analysis_version,
                        vector_version, error_summary, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 'other', '[]', '[]', '', '[]', '[]', ?, 0, NULL, ?, ?)
                    """,
                    (
                        sticker_id, asset.sha256, asset.asset_path, asset.thumbnail_path,
                        asset.mime_type, asset.width, asset.height, asset.frame_count,
                        int(asset.animated), asset.byte_size, StickerState.PENDING.value,
                        SafetyState.UNSAFE.value, analysis_version, now, now,
                    ),
                )
                await connection.commit()
            except aiosqlite.IntegrityError:
                await connection.rollback()
                existing = await self._get_live_by_sha(asset.sha256)
                if existing is None:
                    raise
                return existing, True
        return await self.get(sticker_id), False

    async def get(self, sticker_id: str) -> StickerRecord:
        connection = self._require_connection()
        async with connection.execute("SELECT * FROM stickers WHERE id = ?", (sticker_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise StickerNotFoundError(sticker_id)
        return self._row_to_record(row)

    async def transition(
        self,
        sticker_id: str,
        new_state: StickerState,
        *,
        metadata: VisionMetadata | None = None,
        vector_version: int | None = None,
        analysis_version: str | None = None,
        error_summary: str | None = None,
    ) -> StickerRecord:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self.get(sticker_id)
                if new_state not in ALLOWED_TRANSITIONS[current.state]:
                    raise InvalidStateTransition(f"{current.state.value} -> {new_state.value}")
                assignments = [
                    "state = ?",
                    "updated_at = ?",
                    "error_summary = ?",
                    "row_version = row_version + 1",
                ]
                values: list[Any] = [new_state.value, self._now_text(), error_summary]
                if metadata is not None:
                    assignments.extend(
                        [
                            "description = ?", "primary_category = ?", "emotion_tags = ?",
                            "scene_tags = ?", "ocr_text = ?", "suitable_scenarios = ?",
                            "unsuitable_scenarios = ?", "safety = ?",
                        ]
                    )
                    values.extend(
                        [
                            metadata.description, metadata.primary_category,
                            self._json(metadata.emotion_tags), self._json(metadata.scene_tags),
                            metadata.ocr_text, self._json(metadata.suitable_scenarios),
                            self._json(metadata.unsuitable_scenarios), metadata.safety.value,
                        ]
                    )
                if vector_version is not None:
                    assignments.append("vector_version = ?")
                    values.append(vector_version)
                if analysis_version is not None:
                    assignments.append("analysis_version = ?")
                    values.append(analysis_version)
                values.append(sticker_id)
                await connection.execute(
                    f"UPDATE stickers SET {', '.join(assignments)} WHERE id = ?",
                    values,
                )
                if new_state in {StickerState.DELETING, StickerState.DELETED}:
                    await connection.execute(
                        """
                        UPDATE jobs
                        SET state = 'cancelled', error_summary = ?, updated_at = ?
                        WHERE sticker_id = ? AND state IN ('pending', 'running')
                        """,
                        ("sticker deletion cancelled job", self._now_text(), sticker_id),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return await self.get(sticker_id)
    async def update_metadata(self, record: StickerRecord) -> StickerRecord:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute(
                """
                UPDATE stickers SET description = ?, primary_category = ?, emotion_tags = ?,
                    scene_tags = ?, ocr_text = ?, suitable_scenarios = ?, unsuitable_scenarios = ?,
                    safety = ?, analysis_version = ?, updated_at = ?,
                    row_version = row_version + 1 WHERE id = ?
                """,
                (
                    record.description, record.primary_category, self._json(record.emotion_tags),
                    self._json(record.scene_tags), record.ocr_text,
                    self._json(record.suitable_scenarios), self._json(record.unsuitable_scenarios),
                    record.safety.value, record.analysis_version, self._now_text(), record.id,
                ),
            )
            await connection.commit()
        return await self.get(record.id)

    async def update_metadata_with_revision(
        self,
        record: StickerRecord,
        *,
        actor: str,
        before: dict[str, Any],
        reason: str,
    ) -> StickerRecord:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                now = self._now_text()
                await connection.execute(
                    """
                    UPDATE stickers SET description = ?, primary_category = ?, emotion_tags = ?,
                        scene_tags = ?, ocr_text = ?, suitable_scenarios = ?, unsuitable_scenarios = ?,
                        safety = ?, analysis_version = ?, updated_at = ?,
                        row_version = row_version + 1 WHERE id = ?
                    """,
                    (
                        record.description, record.primary_category, self._json(record.emotion_tags),
                        self._json(record.scene_tags), record.ocr_text,
                        self._json(record.suitable_scenarios), self._json(record.unsuitable_scenarios),
                        record.safety.value, record.analysis_version, now, record.id,
                    ),
                )
                async with connection.execute(
                    "SELECT * FROM stickers WHERE id = ?",
                    (record.id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise StickerNotFoundError(record.id)
                updated = self._row_to_record(row)
                await connection.execute(
                    """
                    INSERT INTO metadata_revisions (
                        sticker_id, actor, before_json, after_json, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        actor,
                        self._json(before),
                        self._json(updated.model_dump(mode="json")),
                        reason,
                        now,
                    ),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return updated

    async def indexing_snapshot(self, sticker_id: str) -> tuple[StickerRecord, int] | None:
        connection = self._require_connection()
        async with connection.execute(
            """
            SELECT * FROM stickers
            WHERE id = ? AND state = ? AND safety = ?
            """,
            (sticker_id, StickerState.INDEXING.value, SafetyState.SAFE.value),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row), int(row["row_version"])

    async def activate_indexed(
        self,
        sticker_id: str,
        *,
        row_version: int,
        vector_version: int,
    ) -> StickerRecord | None:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    UPDATE stickers
                    SET state = ?, vector_version = ?, error_summary = NULL,
                        updated_at = ?, row_version = row_version + 1
                    WHERE id = ? AND state = ? AND safety = ? AND row_version = ?
                    """,
                    (
                        StickerState.ACTIVE.value,
                        vector_version,
                        self._now_text(),
                        sticker_id,
                        StickerState.INDEXING.value,
                        SafetyState.SAFE.value,
                        row_version,
                    ),
                )
                changed = cursor.rowcount == 1
                await cursor.close()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return await self.get(sticker_id) if changed else None
    async def list_searchable_ids(self) -> set[str]:
        connection = self._require_connection()
        async with connection.execute(
            "SELECT id FROM stickers WHERE state = ? AND safety = ?",
            (StickerState.ACTIVE.value, SafetyState.SAFE.value),
        ) as cursor:
            rows = await cursor.fetchall()
        return {str(row[0]) for row in rows}

    async def recover_interrupted(self) -> RecoveryPlan:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                now = self._now_text()
                await connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'pending', updated_at = ?
                    WHERE state = 'running'
                    """,
                    (now,),
                )
                async with connection.execute(
                    "SELECT id FROM stickers WHERE state = ? ORDER BY created_at, id",
                    (StickerState.DELETING.value,),
                ) as cursor:
                    deleting_ids = [str(row[0]) for row in await cursor.fetchall()]
                await connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'cancelled', error_summary = ?, updated_at = ?
                    WHERE state IN ('pending', 'running')
                      AND sticker_id IN (
                          SELECT id FROM stickers WHERE state IN (?, ?)
                      )
                    """,
                    (
                        "sticker deletion cancelled job",
                        now,
                        StickerState.DELETING.value,
                        StickerState.DELETED.value,
                    ),
                )
                interrupted_states = (
                    StickerState.PENDING.value,
                    StickerState.ANALYZING.value,
                    StickerState.INDEXING.value,
                    StickerState.RETRY_PENDING.value,
                )
                async with connection.execute(
                    """
                    SELECT id
                    FROM stickers
                    WHERE state IN (?, ?, ?, ?)
                    ORDER BY created_at, id
                    """,
                    interrupted_states,
                ) as cursor:
                    retry_ids = [str(row[0]) for row in await cursor.fetchall()]
                await connection.execute(
                    """
                    UPDATE stickers
                    SET state = ?, updated_at = ?, row_version = row_version + 1
                    WHERE state IN (?, ?, ?)
                    """,
                    (
                        StickerState.RETRY_PENDING.value,
                        now,
                        StickerState.PENDING.value,
                        StickerState.ANALYZING.value,
                        StickerState.INDEXING.value,
                    ),
                )
                for sticker_id in retry_ids:
                    async with connection.execute(
                        """
                        SELECT id FROM jobs
                        WHERE sticker_id = ? AND job_type = 'analysis'
                          AND state IN ('pending', 'running')
                        ORDER BY created_at, id
                        LIMIT 1
                        """,
                        (sticker_id,),
                    ) as cursor:
                        existing = await cursor.fetchone()
                    if existing is None:
                        job_id = str(uuid.uuid4())
                        await connection.execute(
                            """
                            INSERT INTO jobs (
                                id, sticker_id, job_type, state, attempt_count, created_at, updated_at
                            ) VALUES (?, ?, 'analysis', 'pending', 0, ?, ?)
                            """,
                            (job_id, sticker_id, now, now),
                        )
                async with connection.execute(
                    """
                    SELECT jobs.id
                    FROM jobs
                    JOIN stickers ON stickers.id = jobs.sticker_id
                    WHERE jobs.state = 'pending'
                      AND stickers.state NOT IN (?, ?)
                    ORDER BY jobs.created_at, jobs.id
                    """,
                    (StickerState.DELETING.value, StickerState.DELETED.value),
                ) as cursor:
                    job_ids = [str(row[0]) for row in await cursor.fetchall()]
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return RecoveryPlan(retry_ids=retry_ids, deleting_ids=deleting_ids, job_ids=job_ids)

    async def discard_uninstalled(self, sticker_id: str) -> bool:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    UPDATE stickers
                    SET state = ?, error_summary = ?, updated_at = ?, row_version = row_version + 1
                    WHERE id = ? AND state IN (?, ?, ?, ?, ?)
                    """,
                    (
                        StickerState.DELETED.value,
                        "upload intent asset missing",
                        self._now_text(),
                        sticker_id,
                        StickerState.PENDING.value,
                        StickerState.ANALYZING.value,
                        StickerState.INDEXING.value,
                        StickerState.RETRY_PENDING.value,
                        StickerState.FAILED.value,
                    ),
                )
                changed = cursor.rowcount == 1
                await cursor.close()
                if changed:
                    await connection.execute(
                        """
                        UPDATE jobs
                        SET state = 'cancelled', error_summary = ?, updated_at = ?
                        WHERE sticker_id = ? AND state IN ('pending', 'running')
                        """,
                        ("upload intent asset missing", self._now_text(), sticker_id),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return changed

    async def claim_open_job(self, sticker_id: str, job_type: str) -> tuple[JobRecord, bool]:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE sticker_id = ? AND job_type = ? AND state IN ('pending', 'running')
                    ORDER BY created_at, id
                    LIMIT 1
                    """,
                    (sticker_id, job_type),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    await connection.commit()
                    return self._row_to_job(existing), False
                job_id = str(uuid.uuid4())
                now = self._now_text()
                await connection.execute(
                    """
                    INSERT INTO jobs (
                        id, sticker_id, job_type, state, attempt_count, created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
                    """,
                    (job_id, sticker_id, job_type, now, now),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return await self.get_job(job_id), True

    async def create_job(self, sticker_id: str, job_type: str) -> str:
        job, _created = await self.claim_open_job(sticker_id, job_type)
        return job.id
    async def record_revision(
        self,
        sticker_id: str,
        actor: str,
        before: dict[str, Any],
        after: dict[str, Any],
        reason: str,
    ) -> None:
        connection = self._require_connection()
        await connection.execute(
            "INSERT INTO metadata_revisions (sticker_id, actor, before_json, after_json, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sticker_id, actor, self._json(before), self._json(after), reason, self._now_text()),
        )
        await connection.commit()

    async def revisions(self, sticker_id: str) -> list[dict[str, Any]]:
        connection = self._require_connection()
        async with connection.execute(
            "SELECT actor, before_json, after_json, reason, created_at FROM metadata_revisions WHERE sticker_id = ? ORDER BY id",
            (sticker_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "actor": row["actor"], "before": json.loads(row["before_json"]),
                "after": json.loads(row["after_json"]), "reason": row["reason"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def record_usage(
        self,
        sticker_id: str,
        logical_chat_key: str,
        physical_channel_key: str,
        agent_turn_key: str,
        score: float,
    ) -> None:
        connection = self._require_connection()
        await connection.execute(
            """
            INSERT INTO usage_history (
                sticker_id, logical_chat_key, physical_channel_key, agent_turn_key, score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sticker_id, logical_chat_key, physical_channel_key, agent_turn_key, score, self._now_text()),
        )
        await connection.commit()

    async def recent_sticker_ids(self, logical_chat_key: str, *, limit: int) -> list[str]:
        connection = self._require_connection()
        async with connection.execute(
            "SELECT sticker_id FROM usage_history WHERE logical_chat_key = ? ORDER BY id DESC LIMIT ?",
            (logical_chat_key, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def physical_channel_in_cooldown(self, physical_channel_key: str, *, cooldown_seconds: int) -> bool:
        connection = self._require_connection()
        threshold = (self.clock() - timedelta(seconds=cooldown_seconds)).isoformat()
        async with connection.execute(
            "SELECT 1 FROM usage_history WHERE physical_channel_key = ? AND created_at > ? LIMIT 1",
            (physical_channel_key, threshold),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def agent_turn_has_send(self, agent_turn_key: str) -> bool:
        connection = self._require_connection()
        async with connection.execute(
            "SELECT 1 FROM usage_history WHERE agent_turn_key = ? LIMIT 1", (agent_turn_key,)
        ) as cursor:
            return await cursor.fetchone() is not None

    async def get_job(self, job_id: str) -> JobRecord:
        connection = self._require_connection()
        async with connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise LookupError(job_id)
        return self._row_to_job(row)

    async def update_job(
        self,
        job_id: str,
        *,
        state: str,
        increment_attempt: bool = False,
        error_summary: str | None = None,
    ) -> JobRecord:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute(
                """
                UPDATE jobs SET state = ?, attempt_count = attempt_count + ?,
                    error_summary = ?, updated_at = ? WHERE id = ?
                """,
                (state, int(increment_attempt), error_summary, self._now_text(), job_id),
            )
            await connection.commit()
        return await self.get_job(job_id)

    async def start_job(self, job_id: str) -> JobRecord | None:
        connection = self._require_connection()
        async with self._lock:
            cursor = await connection.execute(
                """
                UPDATE jobs
                SET state = 'running', attempt_count = attempt_count + 1,
                    error_summary = NULL, updated_at = ?
                WHERE id = ? AND state = 'pending'
                """,
                (self._now_text(), job_id),
            )
            changed = cursor.rowcount == 1
            await cursor.close()
            await connection.commit()
        return await self.get_job(job_id) if changed else None

    async def finish_job(
        self,
        job_id: str,
        *,
        state: str,
        error_summary: str | None = None,
    ) -> JobRecord:
        connection = self._require_connection()
        async with self._lock:
            await connection.execute(
                """
                UPDATE jobs
                SET state = ?, error_summary = ?, updated_at = ?
                WHERE id = ? AND state = 'running'
                """,
                (state, error_summary, self._now_text(), job_id),
            )
            await connection.commit()
        return await self.get_job(job_id)
    async def has_open_job(self, sticker_id: str, job_type: str) -> bool:
        connection = self._require_connection()
        async with connection.execute(
            "SELECT 1 FROM jobs WHERE sticker_id = ? AND job_type = ? AND state IN ('pending', 'running') LIMIT 1",
            (sticker_id, job_type),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def list_stickers(self, filters: StickerFilters) -> StickerPage:
        connection = self._require_connection()
        clauses: list[str] = []
        values: list[Any] = []
        if filters.state is None:
            clauses.append("state != ?")
            values.append(StickerState.DELETED.value)
        else:
            clauses.append("state = ?")
            values.append(filters.state.value)
        if filters.category:
            clauses.append("primary_category = ?")
            values.append(filters.category)
        if filters.created_from:
            clauses.append("created_at >= ?")
            values.append(filters.created_from.isoformat())
        if filters.created_to:
            clauses.append("created_at <= ?")
            values.append(filters.created_to.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with connection.execute(
            f"SELECT * FROM stickers {where} ORDER BY created_at DESC, id",
            tuple(values),
        ) as cursor:
            records = [self._row_to_record(row) for row in await cursor.fetchall()]

        tag_keys = {tag.strip().casefold() for tag in filters.tags if tag.strip()}
        query = filters.query.strip().casefold() if filters.query else ""
        filtered: list[StickerRecord] = []
        for record in records:
            available_tags = {tag.casefold() for tag in record.emotion_tags + record.scene_tags}
            if tag_keys and not tag_keys.issubset(available_tags):
                continue
            if query:
                haystack = " ".join(
                    [
                        record.description,
                        record.primary_category,
                        *record.emotion_tags,
                        *record.scene_tags,
                        record.ocr_text,
                        *record.suitable_scenarios,
                        *record.unsuitable_scenarios,
                    ]
                ).casefold()
                if query not in haystack:
                    continue
            filtered.append(record)
        total = len(filtered)
        offset = max(0, filters.offset)
        limit = max(1, filters.limit)
        return StickerPage(items=filtered[offset:offset + limit], total=total, offset=offset, limit=limit)

    async def all_records(self, *, include_deleted: bool = False) -> list[StickerRecord]:
        if include_deleted:
            connection = self._require_connection()
            async with connection.execute("SELECT * FROM stickers ORDER BY created_at, id") as cursor:
                return [self._row_to_record(row) for row in await cursor.fetchall()]
        return (await self.list_stickers(StickerFilters(offset=0, limit=1_000_000))).items

    async def get_many(self, sticker_ids: list[str]) -> dict[str, StickerRecord]:
        if not sticker_ids:
            return {}
        connection = self._require_connection()
        placeholders = ",".join("?" for _ in sticker_ids)
        async with connection.execute(
            f"SELECT * FROM stickers WHERE id IN ({placeholders})",
            tuple(sticker_ids),
        ) as cursor:
            records = [self._row_to_record(row) for row in await cursor.fetchall()]
        return {record.id: record for record in records}

    async def last_usage_times(self, sticker_ids: list[str]) -> dict[str, datetime | None]:
        if not sticker_ids:
            return {}
        connection = self._require_connection()
        placeholders = ",".join("?" for _ in sticker_ids)
        async with connection.execute(
            f"""
            SELECT sticker_id, MAX(created_at) AS last_used_at
            FROM usage_history
            WHERE sticker_id IN ({placeholders})
            GROUP BY sticker_id
            """,
            tuple(sticker_ids),
        ) as cursor:
            rows = await cursor.fetchall()
        result: dict[str, datetime | None] = {sticker_id: None for sticker_id in sticker_ids}
        for row in rows:
            result[str(row["sticker_id"])] = datetime.fromisoformat(row["last_used_at"])
        return result
    async def sticker_stats(self, *, storage_bytes: int) -> StickerStats:
        records = await self.all_records()
        by_state: dict[str, int] = {}
        by_category: dict[str, int] = {}
        indexed_count = 0
        failure_count = 0
        for record in records:
            by_state[record.state.value] = by_state.get(record.state.value, 0) + 1
            by_category[record.primary_category] = by_category.get(record.primary_category, 0) + 1
            if record.state is StickerState.ACTIVE and record.safety is SafetyState.SAFE:
                indexed_count += 1
            if record.state in {StickerState.FAILED, StickerState.RETRY_PENDING}:
                failure_count += 1
        return StickerStats(
            total=len(records),
            storage_bytes=storage_bytes,
            indexed_count=indexed_count,
            failure_count=failure_count,
            by_state=by_state,
            by_category=by_category,
        )
    async def snapshot(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for table in ("stickers", "jobs", "metadata_revisions", "usage_history"):
            result[table] = int(await self.scalar(f"SELECT COUNT(*) FROM {table}"))
        return result

    async def _get_live_by_sha(self, digest: str) -> StickerRecord | None:
        connection = self._require_connection()
        async with connection.execute(
            "SELECT * FROM stickers WHERE sha256 = ? AND state != ? ORDER BY created_at LIMIT 1",
            (digest, StickerState.DELETED.value),
        ) as cursor:
            row = await cursor.fetchone()
        return None if row is None else self._row_to_record(row)

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("repository is not initialized")
        return self._connection

    def _now_text(self) -> str:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            sticker_id=row["sticker_id"],
            job_type=row["job_type"],
            state=row["state"],
            attempt_count=row["attempt_count"],
            error_summary=row["error_summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> StickerRecord:
        return StickerRecord(
            id=row["id"], sha256=row["sha256"], asset_path=row["asset_path"],
            thumbnail_path=row["thumbnail_path"], state=row["state"], safety=row["safety"],
            description=row["description"], primary_category=row["primary_category"],
            emotion_tags=json.loads(row["emotion_tags"]), scene_tags=json.loads(row["scene_tags"]),
            ocr_text=row["ocr_text"], suitable_scenarios=json.loads(row["suitable_scenarios"]),
            unsuitable_scenarios=json.loads(row["unsuitable_scenarios"]), mime_type=row["mime_type"],
            width=row["width"], height=row["height"], frame_count=row["frame_count"],
            animated=bool(row["animated"]), byte_size=row["byte_size"],
            analysis_version=row["analysis_version"], vector_version=row["vector_version"],
            error_summary=row["error_summary"], created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stickers (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    asset_path TEXT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    animated INTEGER NOT NULL,
    byte_size INTEGER NOT NULL,
    state TEXT NOT NULL,
    safety TEXT NOT NULL,
    description TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    emotion_tags TEXT NOT NULL,
    scene_tags TEXT NOT NULL,
    ocr_text TEXT NOT NULL,
    suitable_scenarios TEXT NOT NULL,
    unsuitable_scenarios TEXT NOT NULL,
    analysis_version TEXT NOT NULL,
    vector_version INTEGER NOT NULL,
    row_version INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_stickers_live_sha256 ON stickers(sha256) WHERE state != 'deleted';
CREATE INDEX IF NOT EXISTS idx_stickers_searchable ON stickers(state, safety, primary_category);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    sticker_id TEXT NOT NULL REFERENCES stickers(id),
    job_type TEXT NOT NULL,
    state TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    error_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sticker_id TEXT NOT NULL REFERENCES stickers(id),
    actor TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sticker_id TEXT NOT NULL REFERENCES stickers(id),
    logical_chat_key TEXT NOT NULL,
    physical_channel_key TEXT NOT NULL,
    agent_turn_key TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_logical ON usage_history(logical_chat_key, id DESC);
CREATE INDEX IF NOT EXISTS idx_usage_physical ON usage_history(physical_channel_key, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_usage_turn ON usage_history(agent_turn_key);
"""