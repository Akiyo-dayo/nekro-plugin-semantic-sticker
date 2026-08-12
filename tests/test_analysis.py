from __future__ import annotations

import asyncio
import inspect
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nekro_plugin_semantic_sticker.config import SemanticStickerConfig
from nekro_plugin_semantic_sticker.models import SafetyState


@dataclass
class FakeModelGroup:
    GROUP_NAME: str
    CHAT_MODEL: str
    BASE_URL: str = "https://vision.invalid/v1"
    API_KEY: str = "test-key"
    ENABLE_VISION: bool = False
    MODEL_TYPE: str = "chat"


class FakeCoreConfig:
    def __init__(self, groups: OrderedDict[str, FakeModelGroup]) -> None:
        self.MODEL_GROUPS = groups
        self.private_value = "ignored"

    def get_model_group_info(self, name: str) -> FakeModelGroup:
        return self.MODEL_GROUPS[name]


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, str):
            content = outcome
        else:
            content = json.dumps(outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


class RecordingClientFactory:
    def __init__(self, outcomes: list[object]) -> None:
        self.client = FakeClient(outcomes)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> FakeClient:
        self.calls.append(kwargs)
        return self.client


def safe_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "description": "character tilts head with a question mark",
        "primary_category": "confusion",
        "emotion_tags": ["confused", "questioning"],
        "scene_tags": ["did not understand", "asking why"],
        "ocr_text": "?",
        "suitable_scenarios": ["expressing confusion", "asking for clarification"],
        "unsuitable_scenarios": ["formal apology"],
        "safety": "safe",
    }
    payload.update(overrides)
    return payload


def config(**overrides: object) -> SemanticStickerConfig:
    instance = SemanticStickerConfig()
    for name, value in overrides.items():
        setattr(instance, name, value)
    return instance


def core_with_vision() -> tuple[FakeCoreConfig, FakeModelGroup]:
    text_group = FakeModelGroup("text", "text-embedding-v3", MODEL_TYPE="embedding")
    vision_group = FakeModelGroup("vision", "custom-chat-model", ENABLE_VISION=True)
    return FakeCoreConfig(OrderedDict([("text", text_group), ("vision", vision_group)])), vision_group


def test_analysis_prompt_requires_simplified_chinese_metadata() -> None:
    from nekro_plugin_semantic_sticker.analysis import _analysis_prompt

    prompt = _analysis_prompt("v1", config())

    assert (
        "description、emotion_tags、scene_tags、suitable_scenarios、unsuitable_scenarios "
        "必须使用简体中文"
    ) in prompt
    assert "ocr_text 必须保留图片中的原文，不要翻译或改写" in prompt
    assert "primary_category 必须使用以下英文 API value" in prompt
    assert "safety 必须使用英文 API value：safe、unsafe 或 disallowed" in prompt
    assert "confusion" in prompt
    assert "neutral reaction" in prompt


def test_unknown_category_and_tags_are_normalized() -> None:
    from nekro_plugin_semantic_sticker.analysis import normalize_vision_result

    result = normalize_vision_result(
        safe_payload(
            description="  character  tilts\nhead  ",
            primary_category="uncertain-custom",
            emotion_tags=[" Confused ", "confused", "Questioning", "extra"],
            scene_tags=[" asking why ", "asking why", "did not understand"],
            ocr_text=" ? ",
        ),
        config(MAX_GENERATED_TAGS=2),
    )

    assert result.description == "character tilts head"
    assert result.primary_category == "other"
    assert result.emotion_tags == ["Confused", "Questioning"]
    assert result.scene_tags == ["asking why", "did not understand"]
    assert result.ocr_text == "?"


@pytest.mark.parametrize("safety", ["unsafe", "disallowed"])
def test_rejected_safety_results_are_retained(safety: str) -> None:
    from nekro_plugin_semantic_sticker.analysis import normalize_vision_result

    result = normalize_vision_result(safe_payload(safety=safety), config())

    assert result.safety is SafetyState(safety)


def test_missing_fields_invalid_safety_and_malformed_shape_are_rejected() -> None:
    from nekro_plugin_semantic_sticker.analysis import normalize_vision_result

    missing = safe_payload()
    missing.pop("description")
    with pytest.raises(ValueError):
        normalize_vision_result(missing, config())
    with pytest.raises(ValueError):
        normalize_vision_result(safe_payload(safety="unknown"), config())
    with pytest.raises(ValueError):
        normalize_vision_result([safe_payload()], config())


def test_override_and_recursive_auto_selection_preserve_order() -> None:
    from nekro_plugin_semantic_sticker.analysis import select_vision_model_group

    heuristic = FakeModelGroup("heuristic", "gemini-2.5-flash")
    explicit = FakeModelGroup("explicit", "plain-model", ENABLE_VISION=True)
    core = FakeCoreConfig(OrderedDict([("heuristic", heuristic), ("explicit", explicit)]))

    assert select_vision_model_group(core, "explicit") is explicit
    assert select_vision_model_group(core, "") is heuristic


def test_no_vision_candidate_fails_cleanly() -> None:
    from nekro_plugin_semantic_sticker.analysis import VisionModelNotFoundError, select_vision_model_group

    core = FakeCoreConfig(OrderedDict([("text", FakeModelGroup("text", "text-embedding-v3", MODEL_TYPE="embedding"))]))

    with pytest.raises(VisionModelNotFoundError):
        select_vision_model_group(core, "")


@pytest.mark.asyncio
async def test_analyze_uses_direct_structured_vision_call_without_main_agent_quota(tmp_path: Path) -> None:
    import nekro_plugin_semantic_sticker.analysis as analysis_module
    from nekro_plugin_semantic_sticker.analysis import VisionAnalyzer

    image_path = tmp_path / "confusion.png"
    image_path.write_bytes(b"synthetic-image")
    core, vision_group = core_with_vision()
    factory = RecordingClientFactory([safe_payload()])
    quota_spy = SimpleNamespace(send_agent_request=AsyncMock(), reserve=AsyncMock())
    analyzer = VisionAnalyzer(config(), core, client_factory=factory)

    result = await analyzer.analyze(image_path, mime_type="image/png", prompt_version="v1")

    assert result.primary_category == "confusion"
    assert factory.calls == [{
        "api_key": vision_group.API_KEY,
        "base_url": vision_group.BASE_URL,
        "timeout": 60,
    }]
    call = factory.client.chat.completions.calls[0]
    assert call["model"] == vision_group.CHAT_MODEL
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "prompt version: v1" in content[0]["text"].lower()
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert quota_spy.send_agent_request.await_count == 0
    assert quota_spy.reserve.await_count == 0
    source = inspect.getsource(analysis_module)
    assert "send_agent_request" not in source
    assert "quota" not in source.lower()


@pytest.mark.asyncio
async def test_malformed_json_and_transient_error_retry_then_succeed(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.analysis import VisionAnalyzer

    image_path = tmp_path / "confusion.webp"
    image_path.write_bytes(b"synthetic-image")
    core, _vision_group = core_with_vision()
    factory = RecordingClientFactory([TimeoutError("slow"), "not-json", safe_payload()])
    analyzer = VisionAnalyzer(config(ANALYSIS_RETRY_COUNT=2), core, client_factory=factory)

    result = await analyzer.analyze(image_path, mime_type="image/webp", prompt_version="v2")

    assert result.safety is SafetyState.SAFE
    assert len(factory.client.chat.completions.calls) == 3


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_sanitized_analysis_error(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.analysis import VisionAnalysisError, VisionAnalyzer

    image_path = tmp_path / "confusion.gif"
    image_path.write_bytes(b"synthetic-image")
    core, _vision_group = core_with_vision()
    factory = RecordingClientFactory([RuntimeError("secret-token-1"), RuntimeError("secret-token-2")])
    analyzer = VisionAnalyzer(config(ANALYSIS_RETRY_COUNT=1), core, client_factory=factory)

    with pytest.raises(VisionAnalysisError, match="failed after 2 attempts") as error:
        await analyzer.analyze(image_path, mime_type="image/gif", prompt_version="v1")

    assert "secret-token" not in str(error.value)
    assert len(factory.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_analysis_timeout_is_enforced_even_for_injected_client(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.analysis import VisionAnalysisError, VisionAnalyzer

    class HangingCompletions:
        async def create(self, **_kwargs: object) -> object:
            await asyncio.sleep(2)
            return SimpleNamespace(choices=[])

    class HangingFactory:
        def __call__(self, **_kwargs: object) -> object:
            return SimpleNamespace(chat=SimpleNamespace(completions=HangingCompletions()))

    image_path = tmp_path / "confusion.png"
    image_path.write_bytes(b"synthetic-image")
    core, _vision_group = core_with_vision()
    analyzer = VisionAnalyzer(
        config(ANALYSIS_TIMEOUT_SECONDS=1, ANALYSIS_RETRY_COUNT=0),
        core,
        client_factory=HangingFactory(),
    )

    with pytest.raises(VisionAnalysisError, match="failed after 1 attempts"):
        await analyzer.analyze(image_path, mime_type="image/png", prompt_version="v1")
