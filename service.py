from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from .config import SemanticStickerConfig
from .database import InvalidStateTransition, StickerNotFoundError, StickerRepository
from .files import ImageStore
from .models import (
    BatchDeleteResult,
    JobRecord,
    MetadataPatch,
    ReindexResult,
    ReplyMode,
    SafetyState,
    StickerFilters,
    StickerPage,
    StickerRecord,
    StickerSendResult,
    StickerState,
    StickerStats,
    StoredAsset,
    UploadOutcome,
    UploadPayload,
    UsageContext,
    ValidatedImage,
    VisionMetadata,
)
from .vector_store import build_embedding_text


class StickerServiceError(RuntimeError):
    pass


class StickerBusyError(StickerServiceError):
    pass


class StickerPolicyError(StickerServiceError):
    pass


class StickerSendExecutorProtocol(Protocol):
    async def __call__(
        self,
        ctx: object,
        intent: str,
        reply_mode: ReplyMode,
    ) -> StickerSendResult: ...

    async def execute_direct(
        self,
        usage_context: UsageContext,
        reservation_owner: object,
        intent: str,
        reply_mode: ReplyMode,
        deliver: Callable[[Path], Awaitable[None]],
    ) -> StickerSendResult: ...


def _clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _clean_list(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _safe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: operation failed"


class StickerService:
    def __init__(
        self,
        config: SemanticStickerConfig,
        repository: StickerRepository,
        image_store: ImageStore,
        analyzer: object,
        embedding: object,
        vector_store: object,
        *,
        send_executor: StickerSendExecutorProtocol | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.image_store = image_store
        self.analyzer = analyzer
        self.embedding = embedding
        self.vector_store = vector_store
        self.send_executor = send_executor
        self._start_lock = asyncio.Lock()
        self._upload_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._deferred_tasks: set[asyncio.Task[None]] = set()

    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("service is closed")
            new_workers: list[asyncio.Task[None]] = []
            try:
                await self.repository.initialize()
                await self.vector_store.ensure_collection(validate_dimension=False)
                recovery = await self.repository.recover_interrupted()
                for sticker_id in recovery.deleting_ids:
                    await self._finish_delete(sticker_id)
                for sticker_id in recovery.retry_ids:
                    record = await self.repository.get(sticker_id)
                    await self.vector_store.delete(sticker_id)
                    if not self._asset_files_present(record):
                        try:
                            self.image_store.delete(self._stored_asset(record))
                        finally:
                            await self.repository.discard_uninstalled(sticker_id)
                resumable_jobs: list[str] = []
                for job_id in recovery.job_ids:
                    job = await self.repository.get_job(job_id)
                    if job.state == "pending":
                        resumable_jobs.append(job_id)
                concurrency = max(1, int(self.config.ANALYSIS_CONCURRENCY))
                new_workers = [
                    asyncio.create_task(self._worker(index), name=f"semantic-sticker-worker-{index}")
                    for index in range(concurrency)
                ]
                for job_id in resumable_jobs:
                    await self._queue.put(job_id)
                self._workers.extend(new_workers)
                self._started = True
            except BaseException:
                for worker in new_workers:
                    worker.cancel()
                if new_workers:
                    await asyncio.gather(*new_workers, return_exceptions=True)
                self._started = False
                raise
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._deferred_tasks):
            task.cancel()
        if self._deferred_tasks:
            await asyncio.gather(*self._deferred_tasks, return_exceptions=True)
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        await self.repository.close()
        self._workers.clear()
        self._started = False
    async def wait_for_idle(self) -> None:
        while self._deferred_tasks:
            await asyncio.gather(*list(self._deferred_tasks), return_exceptions=False)
        await self._queue.join()

    async def upload(self, upload: UploadPayload, actor: str) -> UploadOutcome:
        self._require_actor(actor)
        await self.ensure_started()
        validated = self.image_store.validate(upload)
        async with self._upload_lock:
            planned_asset = self._planned_asset(validated)
            record, duplicate = await self.repository.create_pending(
                planned_asset,
                analysis_version=self.config.ANALYSIS_PROMPT_VERSION,
            )
            try:
                self.image_store.install(validated)
            except BaseException:
                if not duplicate:
                    try:
                        self.image_store.delete(planned_asset)
                    finally:
                        await self.repository.discard_uninstalled(record.id)
                raise

            if duplicate:
                if record.state in {StickerState.PENDING, StickerState.RETRY_PENDING}:
                    job, _created = await self.repository.claim_open_job(record.id, "analysis")
                    self._defer_job(job.id)
                return UploadOutcome(record=record, job=None, duplicate=True)

            job, _created = await self.repository.claim_open_job(record.id, "analysis")
            self._defer_job(job.id)
            return UploadOutcome(record=record, job=job, duplicate=False)
    async def list_stickers(self, filters: StickerFilters) -> StickerPage:
        await self.ensure_started()
        return await self.repository.list_stickers(filters)

    async def get_sticker(self, sticker_id: str) -> StickerRecord:
        await self.ensure_started()
        return await self.repository.get(sticker_id)

    async def patch_metadata(self, sticker_id: str, patch: MetadataPatch, actor: str) -> StickerRecord:
        self._require_actor(actor)
        await self.ensure_started()
        current = await self.repository.get(sticker_id)
        if current.state in {StickerState.DELETING, StickerState.DELETED}:
            raise StickerPolicyError("deleted stickers cannot be edited")
        updates: dict[str, Any] = {}
        for name in (
            "description",
            "primary_category",
            "emotion_tags",
            "scene_tags",
            "ocr_text",
            "suitable_scenarios",
            "unsuitable_scenarios",
        ):
            value = getattr(patch, name)
            if value is None:
                continue
            if isinstance(value, str):
                cleaned: Any = _clean_text(value)
            else:
                cleaned = _clean_list(value)
            updates[name] = cleaned
        if "primary_category" in updates:
            allowed = {item.casefold(): item for item in self.config.CATEGORY_VOCABULARY}
            category = updates["primary_category"].casefold()
            updates["primary_category"] = allowed.get(category, "other")
        before = current.model_dump(mode="json")
        return await self.repository.update_metadata_with_revision(
            current.model_copy(update=updates),
            actor=actor,
            before=before,
            reason=_clean_text(patch.reason),
        )
    async def reanalyze(self, sticker_id: str, actor: str) -> JobRecord:
        self._require_actor(actor)
        await self.ensure_started()
        current = await self.repository.get(sticker_id)
        if current.state in {StickerState.DELETING, StickerState.DELETED}:
            raise StickerPolicyError("deleted stickers cannot be analyzed")
        if current.state not in {StickerState.ACTIVE, StickerState.FAILED, StickerState.RETRY_PENDING}:
            raise StickerBusyError(f"cannot reanalyze sticker in {current.state.value} state")
        job, created = await self.repository.claim_open_job(sticker_id, "analysis")
        if not created:
            raise StickerBusyError("an analysis job is already pending")
        try:
            if current.state is StickerState.FAILED:
                await self.repository.transition(sticker_id, StickerState.RETRY_PENDING)
        except BaseException:
            await self.repository.update_job(
                job.id,
                state="cancelled",
                error_summary="reanalyze state claim failed",
            )
            raise
        self._defer_job(job.id)
        return job
    async def reindex(self, sticker_id: str, actor: str) -> StickerRecord:
        self._require_actor(actor)
        await self.ensure_started()
        record = await self.repository.get(sticker_id)
        if record.safety is not SafetyState.SAFE:
            raise StickerPolicyError("unsafe or disallowed stickers require fresh safe analysis")
        try:
            indexing = await self._move_to_indexing(record)
            vector = await self.embedding.embed(build_embedding_text(indexing))
            return await self._upsert_indexed(indexing.id, vector)
        except Exception as error:
            await self._mark_failed_if_possible(sticker_id, _safe_error(error))
            raise
    async def delete(self, sticker_id: str, actor: str) -> None:
        self._require_actor(actor)
        await self.ensure_started()
        record = await self.repository.get(sticker_id)
        if record.state is StickerState.DELETED:
            return
        if record.state is not StickerState.DELETING:
            if record.state in {StickerState.PENDING, StickerState.ANALYZING, StickerState.INDEXING}:
                record = await self.repository.transition(
                    sticker_id,
                    StickerState.FAILED,
                    error_summary="administrative deletion",
                )
            if record.state in {StickerState.ACTIVE, StickerState.FAILED, StickerState.RETRY_PENDING}:
                await self.repository.transition(sticker_id, StickerState.DELETING)
            else:
                raise StickerPolicyError(f"cannot delete sticker in {record.state.value} state")
        await self._finish_delete(sticker_id)

    async def batch_delete(self, sticker_ids: list[str], actor: str) -> BatchDeleteResult:
        self._require_actor(actor)
        failed: list[str] = []
        deleted = 0
        for sticker_id in sticker_ids:
            try:
                await self.delete(sticker_id, actor)
                deleted += 1
            except Exception:
                failed.append(sticker_id)
        return BatchDeleteResult(requested=len(sticker_ids), deleted=deleted, failed_ids=failed)

    async def full_reindex(self, actor: str) -> ReindexResult:
        self._require_actor(actor)
        await self.ensure_started()
        records = [
            record
            for record in await self.repository.all_records()
            if record.state is StickerState.ACTIVE and record.safety is SafetyState.SAFE
        ]
        await self.vector_store.recreate_collection()
        failed: list[str] = []
        indexed = 0
        for record in records:
            try:
                await self.reindex(record.id, actor)
                indexed += 1
            except Exception:
                failed.append(record.id)
        return ReindexResult(requested=len(records), indexed=indexed, failed_ids=failed)
    async def stats(self) -> StickerStats:
        await self.ensure_started()
        return await self.repository.sticker_stats(storage_bytes=self.image_store.snapshot().total_bytes)

    async def execute_send(self, ctx: object, intent: str, reply_mode: ReplyMode) -> StickerSendResult:
        await self.ensure_started()
        if self.send_executor is None:
            return self._sender_unavailable(reply_mode)
        return await self.send_executor(ctx, intent, reply_mode)

    async def execute_send_direct(
        self,
        usage_context: UsageContext,
        reservation_owner: object,
        intent: str,
        reply_mode: ReplyMode,
        deliver: Callable[[Path], Awaitable[None]],
    ) -> StickerSendResult:
        await self.ensure_started()
        if self.send_executor is None:
            return self._sender_unavailable(reply_mode)
        return await self.send_executor.execute_direct(
            usage_context,
            reservation_owner,
            intent,
            reply_mode,
            deliver,
        )

    @staticmethod
    def _sender_unavailable(reply_mode: ReplyMode) -> StickerSendResult:
        return {
            "sent": False,
            "sticker_id": None,
            "reason": "sticker sender is not configured",
            "reply_mode": reply_mode.value,
            "score": None,
        }

    def _defer_job(self, job_id: str) -> None:
        async def enqueue() -> None:
            await asyncio.sleep(0)
            await self._queue.put(job_id)

        task = asyncio.create_task(enqueue())
        self._deferred_tasks.add(task)
        task.add_done_callback(self._deferred_tasks.discard)

    async def _worker(self, _index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id)
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = await self.repository.start_job(job_id)
        if job is None:
            return
        try:
            if job.job_type != "analysis":
                raise StickerServiceError(f"unsupported job type: {job.job_type}")
            record = await self.repository.get(job.sticker_id)
            if record.state in {StickerState.DELETING, StickerState.DELETED}:
                await self.repository.finish_job(job_id, state="completed")
                return
            await self._process_analysis_job(job.sticker_id)
        except Exception as error:
            summary = _safe_error(error)
            await self._mark_failed_if_possible(job.sticker_id, summary)
            await self.repository.finish_job(job_id, state="failed", error_summary=summary)
            return
        await self.repository.finish_job(job_id, state="completed")
    async def _process_analysis_job(self, sticker_id: str) -> None:
        record = await self.repository.get(sticker_id)
        if record.state is StickerState.FAILED:
            record = await self.repository.transition(sticker_id, StickerState.RETRY_PENDING)
        if record.state in {StickerState.PENDING, StickerState.RETRY_PENDING}:
            record = await self.repository.transition(
                sticker_id,
                StickerState.ANALYZING,
                analysis_version=self.config.ANALYSIS_PROMPT_VERSION,
            )
        elif record.state is StickerState.ACTIVE:
            record = await self._move_to_indexing(
                record,
                analysis_version=self.config.ANALYSIS_PROMPT_VERSION,
            )
        elif record.state not in {StickerState.ANALYZING, StickerState.INDEXING}:
            raise StickerBusyError(f"cannot analyze sticker in {record.state.value} state")

        metadata: VisionMetadata = await self.analyzer.analyze(
            Path(record.asset_path),
            mime_type=record.mime_type,
            prompt_version=self.config.ANALYSIS_PROMPT_VERSION,
        )
        current = await self.repository.get(sticker_id)
        if current.state in {StickerState.DELETING, StickerState.DELETED}:
            await self.vector_store.delete(sticker_id)
            return
        if metadata.safety is not SafetyState.SAFE:
            if current.state in {StickerState.ANALYZING, StickerState.INDEXING}:
                await self.repository.transition(
                    sticker_id,
                    StickerState.FAILED,
                    metadata=metadata,
                    analysis_version=self.config.ANALYSIS_PROMPT_VERSION,
                    error_summary="analysis rejected by safety policy",
                )
            await self.vector_store.delete(sticker_id)
            return

        if current.state is StickerState.ANALYZING:
            indexing = await self.repository.transition(
                sticker_id,
                StickerState.INDEXING,
                metadata=metadata,
                analysis_version=self.config.ANALYSIS_PROMPT_VERSION,
            )
        elif current.state is StickerState.INDEXING:
            indexing = await self.repository.update_metadata(
                current.model_copy(
                    update={
                        "description": metadata.description,
                        "primary_category": metadata.primary_category,
                        "emotion_tags": metadata.emotion_tags,
                        "scene_tags": metadata.scene_tags,
                        "ocr_text": metadata.ocr_text,
                        "suitable_scenarios": metadata.suitable_scenarios,
                        "unsuitable_scenarios": metadata.unsuitable_scenarios,
                        "safety": metadata.safety,
                        "analysis_version": self.config.ANALYSIS_PROMPT_VERSION,
                    }
                )
            )
        else:
            await self.vector_store.delete(sticker_id)
            raise StickerBusyError(f"analysis lost ownership in {current.state.value} state")
        vector = await self.embedding.embed(build_embedding_text(indexing))
        await self._upsert_indexed(sticker_id, vector)
    async def _move_to_indexing(
        self,
        record: StickerRecord,
        *,
        analysis_version: str | None = None,
    ) -> StickerRecord:
        if record.state is StickerState.ACTIVE:
            record = await self.repository.transition(
                record.id,
                StickerState.INDEXING,
                analysis_version=analysis_version,
            )
        else:
            if record.state is StickerState.FAILED:
                record = await self.repository.transition(record.id, StickerState.RETRY_PENDING)
            if record.state is StickerState.RETRY_PENDING:
                record = await self.repository.transition(
                    record.id,
                    StickerState.ANALYZING,
                    analysis_version=analysis_version,
                )
            if record.state is StickerState.PENDING:
                record = await self.repository.transition(
                    record.id,
                    StickerState.ANALYZING,
                    analysis_version=analysis_version,
                )
            if record.state is StickerState.ANALYZING:
                record = await self.repository.transition(
                    record.id,
                    StickerState.INDEXING,
                    analysis_version=analysis_version,
                )
            elif record.state is not StickerState.INDEXING:
                raise StickerPolicyError(f"cannot index sticker in {record.state.value} state")
        await self.vector_store.delete(record.id)
        return record

    async def _upsert_indexed(self, sticker_id: str, vector: list[float]) -> StickerRecord:
        snapshot = await self.repository.indexing_snapshot(sticker_id)
        if snapshot is None:
            await self.vector_store.delete(sticker_id)
            raise StickerBusyError("sticker is no longer authoritative for indexing")
        indexing, row_version = snapshot
        next_version = indexing.vector_version + 1
        vector_record = indexing.model_copy(
            update={
                "state": StickerState.ACTIVE,
                "safety": SafetyState.SAFE,
                "vector_version": next_version,
            }
        )
        await self.vector_store.upsert(vector_record, vector)
        activated = await self.repository.activate_indexed(
            sticker_id,
            row_version=row_version,
            vector_version=next_version,
        )
        if activated is None:
            await self.vector_store.delete(sticker_id)
            raise StickerBusyError("sticker indexing compare-and-swap failed")
        authoritative = await self.repository.get(sticker_id)
        if (
            authoritative.state is not StickerState.ACTIVE
            or authoritative.safety is not SafetyState.SAFE
            or authoritative.vector_version != next_version
        ):
            await self.vector_store.delete(sticker_id)
            raise StickerBusyError("sticker lost active-safe authority after indexing")
        return authoritative

    async def _mark_failed_if_possible(self, sticker_id: str, summary: str) -> None:
        try:
            await self.vector_store.delete(sticker_id)
        except Exception:
            pass
        try:
            record = await self.repository.get(sticker_id)
        except StickerNotFoundError:
            return
        try:
            if record.state in {StickerState.PENDING, StickerState.ANALYZING, StickerState.INDEXING}:
                await self.repository.transition(sticker_id, StickerState.FAILED, error_summary=summary)
            elif record.state is StickerState.RETRY_PENDING:
                await self.repository.transition(sticker_id, StickerState.ANALYZING)
                await self.repository.transition(sticker_id, StickerState.FAILED, error_summary=summary)
        except InvalidStateTransition:
            return
        finally:
            try:
                await self.vector_store.delete(sticker_id)
            except Exception:
                pass

    async def _finish_delete(self, sticker_id: str) -> None:
        record = await self.repository.get(sticker_id)
        if record.state is StickerState.DELETED:
            return
        if record.state is not StickerState.DELETING:
            raise StickerPolicyError("deletion recovery requires deleting state")
        await self.vector_store.delete(sticker_id)
        self.image_store.delete(self._stored_asset(record))
        await self.repository.transition(sticker_id, StickerState.DELETED)

    def _planned_asset(self, image: ValidatedImage) -> StoredAsset:
        return StoredAsset(
            sha256=image.sha256,
            asset_path=str(self.image_store.assets_dir / image.asset_name),
            thumbnail_path=str(self.image_store.thumbnails_dir / f"{image.sha256}.webp"),
            detected_format=image.detected_format,
            detected_extension=image.detected_extension,
            mime_type=image.mime_type,
            byte_size=image.byte_size,
            width=image.width,
            height=image.height,
            frame_count=image.frame_count,
            animated=image.animated,
        )

    @staticmethod
    def _asset_files_present(record: StickerRecord) -> bool:
        return Path(record.asset_path).is_file() and Path(record.thumbnail_path).is_file()
    @staticmethod
    def _stored_asset(record: StickerRecord) -> StoredAsset:
        extension = Path(record.asset_path).suffix.lstrip(".").casefold()
        format_name = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "gif": "GIF", "webp": "WEBP"}.get(
            extension,
            extension.upper(),
        )
        return StoredAsset(
            sha256=record.sha256,
            asset_path=record.asset_path,
            thumbnail_path=record.thumbnail_path,
            detected_format=format_name,
            detected_extension="jpg" if extension == "jpeg" else extension,
            mime_type=record.mime_type,
            byte_size=record.byte_size,
            width=record.width,
            height=record.height,
            frame_count=record.frame_count,
            animated=record.animated,
        )

    @staticmethod
    def _require_actor(actor: str) -> None:
        if not actor or not actor.strip():
            raise ValueError("actor is required")


__all__ = [
    "StickerBusyError",
    "StickerPolicyError",
    "StickerService",
    "StickerServiceError",
]
