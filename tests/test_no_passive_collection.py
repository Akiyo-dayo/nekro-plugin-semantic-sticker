from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from nekro_plugin_semantic_sticker import plugin
from nekro_plugin_semantic_sticker.message_images import MessageImageRegistry, message_image_registry


def image_segment(path: Path, name: str) -> SimpleNamespace:
    return SimpleNamespace(type="image", local_path=str(path), file_name=name, remote_url="https://example.invalid/" + name)


@pytest.mark.asyncio
async def test_registry_resolves_only_current_and_explicitly_referenced_images(tmp_path: Path) -> None:
    current_path = tmp_path / "current.png"
    reply_path = tmp_path / "reply.gif"
    unrelated_path = tmp_path / "unrelated.webp"
    for path in (current_path, reply_path, unrelated_path):
        path.write_bytes(path.name.encode("utf-8"))

    referenced = SimpleNamespace(parse_content_data=lambda: [image_segment(reply_path, reply_path.name)])

    async def load_reference(adapter_key: str, chat_key: str, message_id: str) -> object | None:
        assert adapter_key == "onebot_v11"
        assert chat_key == "onebot_v11-group_100-user_42"
        assert message_id == "quoted-1"
        return referenced

    registry = MessageImageRegistry(reference_loader=load_reference, ttl_seconds=60)
    ctx = SimpleNamespace(chat_key="onebot_v11-group_100-user_42")
    message = SimpleNamespace(
        adapter_key="onebot_v11",
        chat_key=ctx.chat_key,
        content_data=[image_segment(current_path, current_path.name)],
        ext_data={"ref_msg_id": "quoted-1"},
    )

    await registry.remember(ctx, message)

    assert registry.resolve(ctx.chat_key, "current", 0).local_path == str(current_path)
    assert registry.resolve(ctx.chat_key, "reply", 0).local_path == str(reply_path)
    assert registry.resolve(ctx.chat_key, "auto", 0).local_path == str(current_path)
    assert registry.resolve(ctx.chat_key, "current", 1) is None
    assert all(ref.local_path != str(unrelated_path) for ref in registry.snapshot(ctx.chat_key))


@pytest.mark.asyncio
async def test_auto_scope_falls_back_to_reply_when_current_message_has_no_image(tmp_path: Path) -> None:
    reply_path = tmp_path / "reply.png"
    reply_path.write_bytes(b"reply")

    async def load_reference(_adapter_key: str, chat_key: str, _message_id: str) -> object:
        assert chat_key == "onebot_v11-private_42"
        return SimpleNamespace(content_data=[image_segment(reply_path, reply_path.name)])

    registry = MessageImageRegistry(reference_loader=load_reference)
    ctx = SimpleNamespace(chat_key="onebot_v11-private_42")
    message = SimpleNamespace(
        adapter_key="onebot_v11",
        chat_key=ctx.chat_key,
        content_data=[],
        ext_data=SimpleNamespace(ref_msg_id="quoted-2"),
    )

    await registry.remember(ctx, message)

    assert registry.resolve(ctx.chat_key, "auto", 0).local_path == str(reply_path)


def test_default_reference_loader_limits_lookup_to_same_physical_channel() -> None:
    source = inspect.getsource(MessageImageRegistry._default_reference_loader)

    assert "adapter_key=adapter_key" in source
    assert "message_id=message_id" in source
    assert "chat_key__startswith" in source
    assert "chat_key=chat_key" in source


@pytest.mark.asyncio
async def test_mounted_message_callback_only_remembers_and_never_uploads(tmp_path: Path) -> None:
    assert plugin.on_user_message_method.__name__ == "remember_message_images"
    source = tmp_path / "group.png"
    source.write_bytes(b"ordinary-chat-image")
    ctx = SimpleNamespace(chat_key="onebot_v11-group_100-user_1")
    message = SimpleNamespace(
        adapter_key="onebot_v11",
        chat_key=ctx.chat_key,
        content_data=[image_segment(source, source.name)],
        ext_data={},
    )
    message_image_registry.clear()

    result = await plugin.on_user_message_method(ctx, message)

    assert result is None
    assert message_image_registry.resolve(ctx.chat_key, "current", 0).local_path == str(source)


def test_runtime_sources_never_import_or_call_main_agent_request_paths() -> None:
    source_root = Path(__file__).resolve().parents[1]
    forbidden_names = {"send_agent_request", "reserve_main_agent", "reserve_main_agent_attempt"}
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names, f"{path.name}: {node.id}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, f"{path.name}: {node.attr}"
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name not in forbidden_names for alias in node.names), path.name
