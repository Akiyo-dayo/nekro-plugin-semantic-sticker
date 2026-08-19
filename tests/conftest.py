from __future__ import annotations

import base64
import importlib.util
import inspect
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.dont_write_bytecode = True

ConfigType = TypeVar("ConfigType", bound="ConfigBase")


class ConfigBase(BaseModel):
    pass


class SandboxMethodType(str, Enum):
    TOOL = "tool"


class AgentCtx:
    pass


class FakeMatcher:
    def __init__(
        self,
        event_type: type[object],
        *,
        rule: Callable[..., Any] | None,
        priority: int,
        block: bool,
    ) -> None:
        self.event_type = event_type
        self.rule = rule
        self.priority = priority
        self.block = block
        self.handlers: list[Callable[..., Any]] = []

    def handle(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(callback: Callable[..., Any]) -> Callable[..., Any]:
            self.handlers.append(callback)
            return callback

        return decorator

    async def matches(self, event: object) -> bool:
        if self.rule is None:
            return True
        result = self.rule(event)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def run(self, *, bot: object, event: object) -> bool:
        if not await self.matches(event):
            return False
        for handler in self.handlers:
            await handler(bot, event)
        return self.block


class Bot:
    pass


class PokeNotifyEvent:
    def __init__(
        self,
        *,
        time: int,
        self_id: int,
        post_type: str,
        notice_type: str,
        sub_type: str,
        user_id: int,
        target_id: int,
        group_id: int | None = None,
        **extra_data: object,
    ) -> None:
        self.time = time
        self.self_id = self_id
        self.post_type = post_type
        self.notice_type = notice_type
        self.sub_type = sub_type
        self.user_id = user_id
        self.target_id = target_id
        self.group_id = group_id
        for key, value in extra_data.items():
            setattr(self, key, value)

    def is_tome(self) -> bool:
        return self.target_id == self.self_id


class MessageSegment:
    def __init__(self, segment_type: str, data: dict[str, object]) -> None:
        self.type = segment_type
        self.data = data

    @classmethod
    def image(cls, file: bytes) -> "MessageSegment":
        encoded = base64.b64encode(file).decode("ascii")
        return cls("image", {"file": f"base64://{encoded}"})


class NekroPlugin:
    def __init__(
        self,
        *,
        name: str,
        module_name: str,
        description: str,
        version: str,
        author: str,
        support_adapter: list[str] | None = None,
        **metadata: Any,
    ) -> None:
        self.name = name
        self.module_name = module_name
        self.description = description
        self.version = version
        self.author = author
        self.support_adapter = list(support_adapter or [])
        self.metadata = metadata
        self.mounted_config_types: list[type[ConfigBase]] = []
        self.on_user_message_method: Callable[..., Any] | None = None
        self.sandbox_methods: list[types.SimpleNamespace] = []
        self.prompt_inject_method: types.SimpleNamespace | None = None
        self.router_factories: list[Callable[..., Any]] = []

    @property
    def key(self) -> str:
        return f"{self.author}.{self.module_name}"

    def mount_config(self) -> Callable[[type[ConfigType]], type[ConfigType]]:
        def decorator(config_type: type[ConfigType]) -> type[ConfigType]:
            self.mounted_config_types.append(config_type)
            return config_type

        return decorator

    def mount_on_user_message(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(callback: Callable[..., Any]) -> Callable[..., Any]:
            self.on_user_message_method = callback
            return callback

        return decorator

    def mount_prompt_inject_method(self, name: str, description: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(callback: Callable[..., Any]) -> Callable[..., Any]:
            self.prompt_inject_method = types.SimpleNamespace(name=name, description=description, func=callback)
            return callback

        return decorator

    def mount_sandbox_method(
        self,
        method_type: SandboxMethodType,
        name: str,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(callback: Callable[..., Any]) -> Callable[..., Any]:
            callback._method_type = method_type
            self.sandbox_methods.append(
                types.SimpleNamespace(method_type=method_type, name=name, description=description, func=callback)
            )
            return callback

        return decorator

    def mount_router(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(callback: Callable[..., Any]) -> Callable[..., Any]:
            self.router_factories.append(callback)
            return callback

        return decorator

    def get_config(self, config_type: type[ConfigType]) -> ConfigType:
        return config_type()



def _install_nekro_agent_stubs() -> None:
    nekro_agent = types.ModuleType("nekro_agent")
    api = types.ModuleType("nekro_agent.api")
    plugin = types.ModuleType("nekro_agent.api.plugin")
    schemas = types.ModuleType("nekro_agent.api.schemas")
    plugin.ConfigBase = ConfigBase
    plugin.NekroPlugin = NekroPlugin
    plugin.SandboxMethodType = SandboxMethodType
    schemas.AgentCtx = AgentCtx

    services = types.ModuleType("nekro_agent.services")
    user = types.ModuleType("nekro_agent.services.user")
    deps = types.ModuleType("nekro_agent.services.user.deps")

    async def get_current_active_user():
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Not authenticated")

    async def get_current_super_user():
        return await get_current_active_user()

    deps.get_current_active_user = get_current_active_user
    deps.get_current_super_user = get_current_super_user
    services.user = user
    user.deps = deps

    models = types.ModuleType("nekro_agent.models")
    db_user = types.ModuleType("nekro_agent.models.db_user")

    class DBUser:
        def __init__(self, username: str, is_active: bool = True, perm_level: int = 999) -> None:
            self.username = username
            self.is_active = is_active
            self.perm_level = perm_level

    db_user.DBUser = DBUser
    models.db_user = db_user

    nekro_agent.api = api
    nekro_agent.services = services
    nekro_agent.models = models
    api.plugin = plugin
    api.schemas = schemas
    sys.modules["nekro_agent"] = nekro_agent
    sys.modules["nekro_agent.api"] = api
    sys.modules["nekro_agent.api.plugin"] = plugin
    sys.modules["nekro_agent.api.schemas"] = schemas
    sys.modules["nekro_agent.services"] = services
    sys.modules["nekro_agent.services.user"] = user
    sys.modules["nekro_agent.services.user.deps"] = deps
    sys.modules["nekro_agent.models"] = models
    sys.modules["nekro_agent.models.db_user"] = db_user


def _install_nonebot_stubs() -> None:
    nonebot = types.ModuleType("nonebot")
    adapters = types.ModuleType("nonebot.adapters")
    onebot = types.ModuleType("nonebot.adapters.onebot")
    onebot_v11 = types.ModuleType("nonebot.adapters.onebot.v11")
    registered_matchers: list[FakeMatcher] = []

    def on_type(
        event_type: type[object],
        *,
        rule: Callable[..., Any] | None = None,
        priority: int = 1,
        block: bool = False,
    ) -> FakeMatcher:
        matcher = FakeMatcher(event_type, rule=rule, priority=priority, block=block)
        registered_matchers.append(matcher)
        return matcher

    nonebot.on_type = on_type
    nonebot.registered_matchers = registered_matchers
    nonebot.adapters = adapters
    adapters.onebot = onebot
    onebot.v11 = onebot_v11
    onebot_v11.Bot = Bot
    onebot_v11.MessageSegment = MessageSegment
    onebot_v11.PokeNotifyEvent = PokeNotifyEvent
    sys.modules["nonebot"] = nonebot
    sys.modules["nonebot.adapters"] = adapters
    sys.modules["nonebot.adapters.onebot"] = onebot
    sys.modules["nonebot.adapters.onebot.v11"] = onebot_v11


_install_nekro_agent_stubs()
_install_nonebot_stubs()


def _load_repository_root_package() -> None:
    if "nekro_plugin_semantic_sticker" in sys.modules:
        return
    entrypoint = PROJECT_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "nekro_plugin_semantic_sticker",
        entrypoint,
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to create Semantic Sticker package spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nekro_plugin_semantic_sticker"] = module
    spec.loader.exec_module(module)


_load_repository_root_package()
