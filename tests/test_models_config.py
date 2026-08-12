from __future__ import annotations

import importlib
import json
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints, is_typeddict

import pytest
from pydantic import ValidationError


EXPECTED_CONFIG_FIELDS = {
    "ANALYSIS_MODEL_GROUP", "ANALYSIS_PROMPT_VERSION", "ANALYSIS_TIMEOUT_SECONDS",
    "ANALYSIS_RETRY_COUNT", "MAX_GENERATED_TAGS", "CATEGORY_VOCABULARY",
    "ANALYSIS_CONCURRENCY", "EMBEDDING_MODEL_GROUP", "VECTOR_DIMENSION",
    "SEMANTIC_SCORE_THRESHOLD", "MAX_UPLOAD_BYTES", "MAX_IMAGE_PIXELS",
    "MAX_WIDTH", "MAX_HEIGHT", "MAX_ANIMATION_FRAMES", "RECENT_SELECTION_WINDOW",
    "PHYSICAL_CHANNEL_COOLDOWN_SECONDS", "AUTO_COLLECT_ENABLED", "STRICT_EMOTION_COLLECT",
}


def test_enums_and_send_result_contract() -> None:
    from nekro_plugin_semantic_sticker.models import ReplyMode, SafetyState, StickerSendResult, StickerState

    assert [state.value for state in StickerState] == [
        "pending", "analyzing", "indexing", "active", "failed", "retry_pending", "deleting", "deleted"
    ]
    assert [state.value for state in SafetyState] == ["safe", "unsafe", "disallowed"]
    assert [mode.value for mode in ReplyMode] == ["image_only", "text_then_image", "auto"]
    assert is_typeddict(StickerSendResult)
    assert get_type_hints(StickerSendResult) == {
        "sent": bool,
        "sticker_id": str | None,
        "reason": str,
        "reply_mode": str,
        "score": float | None,
    }


def test_models_follow_locked_cross_task_contract() -> None:
    from nekro_plugin_semantic_sticker.models import (
        BatchDeleteResult,
        FileSnapshot,
        JobRecord,
        MetadataPatch,
        ReindexResult,
        ReplyMode,
        SafetyState,
        StickerCandidate,
        StickerFilters,
        StickerPage,
        StickerRecord,
        StickerState,
        StickerStats,
        StoredAsset,
        UploadOutcome,
        UploadPayload,
        UsageContext,
        ValidatedImage,
        VisionMetadata,
    )

    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    payload = UploadPayload(content=b"raw", filename="ignored.png", content_type="image/png")
    validated = ValidatedImage(
        original_bytes=b"raw", sha256="a" * 64, detected_format="PNG", detected_extension="png",
        mime_type="image/png", byte_size=3, width=1, height=1, frame_count=1, animated=False,
        asset_name=f"{'a' * 64}.png",
    )
    asset = StoredAsset(
        sha256=validated.sha256, asset_path=f"assets/{validated.asset_name}",
        thumbnail_path=f"thumbnails/{validated.sha256}.webp", detected_format="PNG",
        detected_extension="png", mime_type="image/png", byte_size=3, width=1, height=1,
        frame_count=1, animated=False,
    )
    metadata = VisionMetadata(
        description="character tilts head with a question mark", primary_category="confusion",
        emotion_tags=["confused"], scene_tags=["asking why"], ocr_text="?",
        suitable_scenarios=["asking for clarification"], unsuitable_scenarios=["formal apology"],
        safety=SafetyState.SAFE,
    )
    record = StickerRecord(
        id="sticker-1", sha256=validated.sha256, asset_path=asset.asset_path,
        thumbnail_path=asset.thumbnail_path, state=StickerState.ACTIVE, safety=SafetyState.SAFE,
        description=metadata.description, primary_category=metadata.primary_category,
        emotion_tags=metadata.emotion_tags, scene_tags=metadata.scene_tags, ocr_text=metadata.ocr_text,
        suitable_scenarios=metadata.suitable_scenarios,
        unsuitable_scenarios=metadata.unsuitable_scenarios, mime_type="image/png", width=1, height=1,
        frame_count=1, animated=False, byte_size=3, analysis_version="v1", vector_version=1,
        error_summary=None, created_at=now, updated_at=now,
    )
    job = JobRecord(
        id="job-1", sticker_id=record.id, job_type="analysis", state="pending", attempt_count=0,
        error_summary=None, created_at=now, updated_at=now,
    )
    instances = [
        payload, validated, asset,
        FileSnapshot(assets=[asset.asset_path], thumbnails=[asset.thumbnail_path], temp_files=[], total_bytes=4),
        metadata, record, job,
        MetadataPatch(description="confused", reason="improve retrieval"),
        StickerFilters(category="confusion", tags=["confused"], state=StickerState.ACTIVE, query="why"),
        StickerCandidate(
            sticker_id=record.id, vector_score=0.91, asset_path=record.asset_path,
            primary_category=record.primary_category, emotion_tags=record.emotion_tags,
            scene_tags=record.scene_tags, ocr_text=record.ocr_text, last_used_at=None,
        ),
        UsageContext(
            logical_chat_key="onebot_v11-group_100-user_1",
            physical_channel_key="onebot_v11-group_100", agent_turn_key="turn-1",
        ),
        UploadOutcome(record=record, job=job, duplicate=False),
        StickerPage(items=[record], total=1, offset=0, limit=20),
        BatchDeleteResult(requested=1, deleted=1, failed_ids=[]),
        ReindexResult(requested=1, indexed=1, failed_ids=[]),
        StickerStats(
            total=1, storage_bytes=4, indexed_count=1, failure_count=0,
            by_state={"active": 1}, by_category={"confusion": 1},
        ),
    ]

    assert payload.content == b"raw"
    assert validated.asset_name == f"{validated.sha256}.png"
    assert metadata.primary_category == "confusion"
    assert UsageContext.model_fields.keys() >= {"logical_chat_key", "physical_channel_key", "agent_turn_key"}
    assert ReplyMode.AUTO.value == "auto"
    for instance in instances:
        assert isinstance(json.loads(instance.model_dump_json()), dict)


def test_semantic_sticker_config_has_exact_fields_and_defaults() -> None:
    from nekro_plugin_semantic_sticker.config import SemanticStickerConfig

    config = SemanticStickerConfig()
    assert set(SemanticStickerConfig.model_fields) == EXPECTED_CONFIG_FIELDS
    assert config.model_dump() == {
        "ANALYSIS_MODEL_GROUP": "", "ANALYSIS_PROMPT_VERSION": "v1",
        "ANALYSIS_TIMEOUT_SECONDS": 60, "ANALYSIS_RETRY_COUNT": 2, "MAX_GENERATED_TAGS": 12,
        "CATEGORY_VOCABULARY": [
            "confusion", "happiness", "speechlessness", "anger", "sadness", "surprise",
            "agreement", "refusal", "comfort", "shyness", "mockery", "celebration",
            "neutral reaction", "other",
        ],
        "ANALYSIS_CONCURRENCY": 1, "EMBEDDING_MODEL_GROUP": "text-embedding",
        "VECTOR_DIMENSION": 1536, "SEMANTIC_SCORE_THRESHOLD": 0.72,
        "MAX_UPLOAD_BYTES": 10_485_760, "MAX_IMAGE_PIXELS": 40_000_000,
        "MAX_WIDTH": 8192, "MAX_HEIGHT": 8192, "MAX_ANIMATION_FRAMES": 300,
        "RECENT_SELECTION_WINDOW": 10, "PHYSICAL_CHANNEL_COOLDOWN_SECONDS": 20,
        "AUTO_COLLECT_ENABLED": True, "STRICT_EMOTION_COLLECT": True,
    }


def test_config_schema_is_chinese_and_model_groups_use_selectors() -> None:
    from nekro_plugin_semantic_sticker.config import SemanticStickerConfig

    for name, field in SemanticStickerConfig.model_fields.items():
        title = field.title or ""
        description = field.description or ""
        assert title and any("\u4e00" <= char <= "\u9fff" for char in title), name
        assert description and any("\u4e00" <= char <= "\u9fff" for char in description), name

    analysis_extra = SemanticStickerConfig.model_fields["ANALYSIS_MODEL_GROUP"].json_schema_extra
    embedding_extra = SemanticStickerConfig.model_fields["EMBEDDING_MODEL_GROUP"].json_schema_extra
    assert analysis_extra == {"ref_model_groups": True, "required": True, "model_type": "chat"}
    assert embedding_extra == {"ref_model_groups": True, "required": True, "model_type": "embedding"}


def test_physical_channel_cooldown_is_non_negative_and_documented_in_chinese() -> None:
    from nekro_plugin_semantic_sticker.config import SemanticStickerConfig

    field = SemanticStickerConfig.model_fields["PHYSICAL_CHANNEL_COOLDOWN_SECONDS"]
    description = field.description or ""

    assert SemanticStickerConfig().PHYSICAL_CHANNEL_COOLDOWN_SECONDS == 20
    assert SemanticStickerConfig(PHYSICAL_CHANNEL_COOLDOWN_SECONDS=0).PHYSICAL_CHANNEL_COOLDOWN_SECONDS == 0
    with pytest.raises(ValidationError):
        SemanticStickerConfig(PHYSICAL_CHANNEL_COOLDOWN_SECONDS=-1)

    assert "秒" in description
    assert "同一物理频道" in description
    assert "0" in description
    assert "关闭冷却" in description


def test_physical_channel_cooldown_validates_direct_assignment() -> None:
    from nekro_plugin_semantic_sticker.config import SemanticStickerConfig

    config = SemanticStickerConfig()
    for value in (0, 20, 60):
        config.PHYSICAL_CHANNEL_COOLDOWN_SECONDS = value
        assert config.PHYSICAL_CHANNEL_COOLDOWN_SECONDS == value

    with pytest.raises(ValidationError):
        config.PHYSICAL_CHANNEL_COOLDOWN_SECONDS = -1
    assert config.PHYSICAL_CHANNEL_COOLDOWN_SECONDS == 60
    assert SemanticStickerConfig.model_config.get("validate_assignment") is True


def test_agent_tool_and_poke_direct_send_share_the_configured_cooldown() -> None:
    source_root = Path(__file__).resolve().parents[1] / "source" / "nekro_plugin_semantic_sticker"
    agent_tools = (source_root / "agent_tools.py").read_text(encoding="utf-8")
    poke_handler = (source_root / "poke_handler.py").read_text(encoding="utf-8")

    assert '"PHYSICAL_CHANNEL_COOLDOWN_SECONDS"' in agent_tools
    assert "get_service().execute_send_direct" in poke_handler


def test_package_import_is_inert_and_registers_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("package import attempted network or file write")

    monkeypatch.setattr(socket.socket, "connect", deny_access)
    monkeypatch.setattr(Path, "write_text", deny_access)
    monkeypatch.setattr(Path, "write_bytes", deny_access)
    monkeypatch.setattr(Path, "mkdir", deny_access)
    monkeypatch.setattr(os, "replace", deny_access)
    plugin_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == "nekro_plugin_semantic_sticker"
        or module_name.startswith("nekro_plugin_semantic_sticker.")
    }
    for module_name in plugin_modules:
        sys.modules.pop(module_name)

    try:
        package = importlib.import_module("nekro_plugin_semantic_sticker")

        assert package.plugin.key == "Akiyo.semantic_sticker"
        assert package.plugin.author == "Akiyo"
        assert package.plugin.module_name == "semantic_sticker"
        assert package.plugin.version == "1.2.1"
        assert package.plugin.support_adapter == ["onebot_v11"]
        assert package.plugin.mounted_config_types == [package.SemanticStickerConfig]
        assert package.config.__class__.__name__ == "SemanticStickerConfig"
        assert package.plugin.on_user_message_method.__name__ == "remember_message_images"
        assert hasattr(package.plugin, "mount_on_user_message")
        assert not hasattr(package.plugin, "mount_user_message")
        assert "nekro_plugin_semantic_sticker.router" in sys.modules
        assert "nekro_plugin_semantic_sticker.agent_tools" in sys.modules
        assert "nekro_plugin_semantic_sticker.message_images" in sys.modules
        assert package.plugin.prompt_inject_method.name == "semantic_sticker_prompt"
        assert [method.name for method in package.plugin.sandbox_methods] == [
            "send_matching_sticker",
            "save_sticker_from_message",
        ]
        assert [factory.__name__ for factory in package.plugin.router_factories] == ["mount_semantic_sticker_router"]
    finally:
        for module_name in list(sys.modules):
            if module_name == "nekro_plugin_semantic_sticker" or module_name.startswith("nekro_plugin_semantic_sticker."):
                sys.modules.pop(module_name)
        sys.modules.update(plugin_modules)
