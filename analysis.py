from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .config import SemanticStickerConfig
from .models import SafetyState, VisionMetadata


VISION_TOKENS = ("vision", "vl", "4o", "4.1", "claude-3", "opus", "sonnet", "multimodal", "gemini")
_REQUIRED_FIELDS = {
    "description",
    "primary_category",
    "emotion_tags",
    "scene_tags",
    "ocr_text",
    "suitable_scenarios",
    "unsuitable_scenarios",
    "safety",
}
_LIST_FIELDS = (
    "emotion_tags",
    "scene_tags",
    "suitable_scenarios",
    "unsuitable_scenarios",
)


class VisionAnalysisError(RuntimeError):
    pass


class VisionModelNotFoundError(LookupError):
    pass


class OpenAIClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


def _default_client_factory(**kwargs: object) -> object:
    from openai import AsyncOpenAI

    return AsyncOpenAI(**kwargs)


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("vision result text fields must be strings")
    return re.sub(r"\s+", " ", value).strip()


def _clean_list(value: object, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("vision result list fields must be arrays")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if limit is not None and len(result) >= max(0, limit):
            break
    return result


def normalize_vision_result(raw: object, config: SemanticStickerConfig) -> VisionMetadata:
    if not isinstance(raw, Mapping):
        raise ValueError("vision result must be a JSON object")
    missing = _REQUIRED_FIELDS - set(raw)
    if missing:
        raise ValueError(f"vision result missing fields: {', '.join(sorted(missing))}")

    description = _clean_text(raw["description"])
    if not description:
        raise ValueError("vision result description must not be empty")

    category = _clean_text(raw["primary_category"])
    categories = {
        _clean_text(item).casefold(): _clean_text(item)
        for item in config.CATEGORY_VOCABULARY
        if _clean_text(item)
    }
    primary_category = categories.get(category.casefold(), categories.get("other", "other"))

    safety_text = _clean_text(raw["safety"]).casefold()
    try:
        safety = SafetyState(safety_text)
    except ValueError as error:
        raise ValueError("vision result safety must be safe, unsafe, or disallowed") from error

    tag_limit = max(0, int(config.MAX_GENERATED_TAGS))
    normalized = {
        "description": description,
        "primary_category": primary_category,
        "emotion_tags": _clean_list(raw["emotion_tags"], limit=tag_limit),
        "scene_tags": _clean_list(raw["scene_tags"], limit=tag_limit),
        "ocr_text": _clean_text(raw["ocr_text"]),
        "suitable_scenarios": _clean_list(raw["suitable_scenarios"]),
        "unsuitable_scenarios": _clean_list(raw["unsuitable_scenarios"]),
        "safety": safety,
    }
    return VisionMetadata.model_validate(normalized)


def _looks_like_model_group(value: object) -> bool:
    return all(hasattr(value, field) for field in ("CHAT_MODEL", "BASE_URL", "API_KEY"))


def _iter_model_groups(value: object, seen: set[int] | None = None) -> Iterable[tuple[str, object]]:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    if _looks_like_model_group(value):
        name = str(getattr(value, "GROUP_NAME", ""))
        yield name, value
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            if _looks_like_model_group(child):
                yield str(name), child
            else:
                yield from _iter_model_groups(child, seen)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_model_groups(child, seen)
        return
    if isinstance(value, (str, bytes, bytearray, Path, int, float, bool, type(None))):
        return
    try:
        values = vars(value)
    except TypeError:
        return
    for name, child in values.items():
        if not name.startswith("_"):
            if _looks_like_model_group(child):
                yield name, child
            else:
                yield from _iter_model_groups(child, seen)


def is_vision_capable(name: str, group: object) -> bool:
    if str(getattr(group, "MODEL_TYPE", "chat")).casefold() != "chat":
        return False
    if bool(getattr(group, "ENABLE_VISION", False)):
        return True
    haystack = " ".join(
        (
            name,
            str(getattr(group, "GROUP_NAME", "")),
            str(getattr(group, "CHAT_MODEL", "")),
        )
    ).casefold()
    return any(token in haystack for token in VISION_TOKENS)


def select_vision_model_group(core_config: object, override: str) -> object:
    override = override.strip()
    if override:
        getter = getattr(core_config, "get_model_group_info", None)
        if getter is None:
            raise VisionModelNotFoundError(f"configured model group '{override}' cannot be resolved")
        try:
            group = getter(override)
        except (KeyError, LookupError) as error:
            raise VisionModelNotFoundError(f"configured model group '{override}' was not found") from error
        if not is_vision_capable(override, group):
            raise VisionModelNotFoundError(f"configured model group '{override}' is not vision-capable")
        return group

    for name, group in _iter_model_groups(core_config):
        if is_vision_capable(name, group):
            return group
    raise VisionModelNotFoundError("no vision-capable model group is configured")


def _analysis_prompt(prompt_version: str, config: SemanticStickerConfig) -> str:
    categories = ", ".join(config.CATEGORY_VOCABULARY)
    return (
        f"Semantic sticker analysis prompt version: {prompt_version}. "
        "检查图片，只返回一个 JSON 对象，字段必须完整包含：description、primary_category、"
        "emotion_tags、scene_tags、ocr_text、suitable_scenarios、unsuitable_scenarios、safety。"
        "description、emotion_tags、scene_tags、suitable_scenarios、unsuitable_scenarios "
        "必须使用简体中文。"
        "ocr_text 必须保留图片中的原文，不要翻译或改写。"
        f"primary_category 必须使用以下英文 API value：[{categories}]，无法归类时使用 other。"
        "safety 必须使用英文 API value：safe、unsafe 或 disallowed。"
        "不要添加 Markdown 代码块或 JSON 之外的说明。"
    )


def _response_content(response: object) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("vision response did not contain message content") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        if parts:
            return "".join(parts)
    raise ValueError("vision response content was not text")


def _parse_json_content(content: str) -> object:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return json.loads(stripped)


class VisionAnalyzer:
    def __init__(
        self,
        config: SemanticStickerConfig,
        core_config: object,
        *,
        client_factory: OpenAIClientFactory | None = None,
    ) -> None:
        self.config = config
        self.core_config = core_config
        self.client_factory = client_factory or _default_client_factory

    async def analyze(
        self,
        image_path: Path,
        *,
        mime_type: str,
        prompt_version: str | None = None,
    ) -> VisionMetadata:
        group = select_vision_model_group(self.core_config, self.config.ANALYSIS_MODEL_GROUP)
        timeout = float(self.config.ANALYSIS_TIMEOUT_SECONDS)
        client = self.client_factory(
            api_key=str(getattr(group, "API_KEY")),
            base_url=str(getattr(group, "BASE_URL")),
            timeout=self.config.ANALYSIS_TIMEOUT_SECONDS,
        )
        data_url = f"data:{mime_type};base64,{base64.b64encode(Path(image_path).read_bytes()).decode('ascii')}"
        prompt = _analysis_prompt(prompt_version or self.config.ANALYSIS_PROMPT_VERSION, self.config)
        attempts = max(0, int(self.config.ANALYSIS_RETRY_COUNT)) + 1

        for _attempt in range(attempts):
            try:
                request = client.chat.completions.create(
                    model=str(getattr(group, "CHAT_MODEL")),
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                response = await asyncio.wait_for(request, timeout=timeout)
                raw = _parse_json_content(_response_content(response))
                return normalize_vision_result(raw, self.config)
            except (asyncio.TimeoutError, TimeoutError, ValueError, TypeError, json.JSONDecodeError):
                continue
            except Exception:
                continue

        raise VisionAnalysisError(f"vision analysis failed after {attempts} attempts") from None


__all__ = [
    "VISION_TOKENS",
    "VisionAnalysisError",
    "VisionAnalyzer",
    "VisionModelNotFoundError",
    "is_vision_capable",
    "normalize_vision_result",
    "select_vision_model_group",
]
