from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field


class StickerState(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    INDEXING = "indexing"
    ACTIVE = "active"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"
    DELETING = "deleting"
    DELETED = "deleted"


class SafetyState(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    DISALLOWED = "disallowed"


class ReplyMode(str, Enum):
    IMAGE_ONLY = "image_only"
    TEXT_THEN_IMAGE = "text_then_image"
    AUTO = "auto"


class StickerSendResult(TypedDict):
    sent: bool
    sticker_id: str | None
    reason: str
    reply_mode: str
    score: float | None


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")


class UploadPayload(JsonModel):
    content: bytes
    filename: str | None = None
    content_type: str | None = None


class ValidatedImage(JsonModel):
    original_bytes: bytes
    sha256: str
    detected_format: str
    detected_extension: str
    mime_type: str
    byte_size: int
    width: int
    height: int
    frame_count: int
    animated: bool
    asset_name: str


class StoredAsset(JsonModel):
    sha256: str
    asset_path: str
    thumbnail_path: str
    detected_format: str
    detected_extension: str
    mime_type: str
    byte_size: int
    width: int
    height: int
    frame_count: int
    animated: bool


class FileSnapshot(JsonModel):
    assets: list[str] = Field(default_factory=list)
    thumbnails: list[str] = Field(default_factory=list)
    temp_files: list[str] = Field(default_factory=list)
    total_bytes: int = 0


class VisionMetadata(JsonModel):
    description: str
    primary_category: str
    emotion_tags: list[str] = Field(default_factory=list)
    scene_tags: list[str] = Field(default_factory=list)
    ocr_text: str = ""
    suitable_scenarios: list[str] = Field(default_factory=list)
    unsuitable_scenarios: list[str] = Field(default_factory=list)
    safety: SafetyState


class StickerRecord(JsonModel):
    id: str
    sha256: str
    asset_path: str
    thumbnail_path: str
    state: StickerState
    safety: SafetyState
    description: str = ""
    primary_category: str = "other"
    emotion_tags: list[str] = Field(default_factory=list)
    scene_tags: list[str] = Field(default_factory=list)
    ocr_text: str = ""
    suitable_scenarios: list[str] = Field(default_factory=list)
    unsuitable_scenarios: list[str] = Field(default_factory=list)
    mime_type: str
    width: int
    height: int
    frame_count: int
    animated: bool
    byte_size: int
    analysis_version: str
    vector_version: int
    error_summary: str | None = None
    created_at: datetime
    updated_at: datetime


class JobRecord(JsonModel):
    id: str
    sticker_id: str
    job_type: str
    state: str
    attempt_count: int
    error_summary: str | None = None
    created_at: datetime
    updated_at: datetime


class MetadataPatch(JsonModel):
    description: str | None = None
    primary_category: str | None = None
    emotion_tags: list[str] | None = None
    scene_tags: list[str] | None = None
    ocr_text: str | None = None
    suitable_scenarios: list[str] | None = None
    unsuitable_scenarios: list[str] | None = None
    reason: str


class StickerFilters(JsonModel):
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    state: StickerState | None = None
    query: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    offset: int = 0
    limit: int = 50


class StickerCandidate(JsonModel):
    sticker_id: str
    vector_score: float
    asset_path: str
    primary_category: str
    emotion_tags: list[str] = Field(default_factory=list)
    scene_tags: list[str] = Field(default_factory=list)
    ocr_text: str = ""
    last_used_at: datetime | None = None


class UsageContext(JsonModel):
    logical_chat_key: str
    physical_channel_key: str
    agent_turn_key: str


class UploadOutcome(JsonModel):
    record: StickerRecord
    job: JobRecord | None = None
    duplicate: bool


class StickerPage(JsonModel):
    items: list[StickerRecord]
    total: int
    offset: int
    limit: int


class BatchDeleteResult(JsonModel):
    requested: int
    deleted: int
    failed_ids: list[str] = Field(default_factory=list)


class ReindexResult(JsonModel):
    requested: int
    indexed: int
    failed_ids: list[str] = Field(default_factory=list)


class StickerStats(JsonModel):
    total: int
    storage_bytes: int
    indexed_count: int
    failure_count: int
    by_state: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)