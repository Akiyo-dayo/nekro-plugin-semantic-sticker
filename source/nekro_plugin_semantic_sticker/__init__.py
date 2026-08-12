from __future__ import annotations

from importlib import import_module
from typing import Any

from nekro_agent.api.plugin import NekroPlugin

from .config import SemanticStickerConfig


plugin = NekroPlugin(
    name="语义表情包",
    module_name="semantic_sticker",
    description="支持网页管理、语义检索、Agent 自主收集与用户主动保存的实例级表情包库",
    version="1.2.1",
    author="Akiyo",
    url="https://github.com/Akiyo-dayo",
    support_adapter=["onebot_v11"],
)
SemanticStickerConfig = plugin.mount_config()(SemanticStickerConfig)
config: SemanticStickerConfig = plugin.get_config(SemanticStickerConfig)


def load_runtime_components() -> tuple[Any, Any]:
    import_module(f"{__name__}.message_images")
    agent_tools_module = import_module(f"{__name__}.agent_tools")
    import_module(f"{__name__}.poke_handler")
    router_module = import_module(f"{__name__}.router")
    return router_module, agent_tools_module


message_images = import_module(f"{__name__}.message_images")
agent_tools = import_module(f"{__name__}.agent_tools")
poke_handler = import_module(f"{__name__}.poke_handler")
router = import_module(f"{__name__}.router")


__all__ = [
    "SemanticStickerConfig",
    "agent_tools",
    "config",
    "load_runtime_components",
    "message_images",
    "plugin",
    "poke_handler",
    "router",
]
