from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.routing import APIRoute

from nekro_plugin_semantic_sticker.models import (
    BatchDeleteResult,
    JobRecord,
    ReindexResult,
    SafetyState,
    StickerPage,
    StickerRecord,
    StickerState,
    StickerStats,
    UploadOutcome,
)


DECLARED = {
    ("GET", "/"), ("GET", "/static/{path:path}"),
    ("POST", "/api/stickers"), ("GET", "/api/stickers"),
    ("GET", "/api/stickers/{sticker_id}"), ("PATCH", "/api/stickers/{sticker_id}"),
    ("DELETE", "/api/stickers/{sticker_id}"),
    ("POST", "/api/stickers/{sticker_id}/reanalyze"),
    ("POST", "/api/stickers/{sticker_id}/reindex"),
    ("POST", "/api/stickers/batch-delete"), ("POST", "/api/reindex"),
    ("GET", "/api/stats"),
}


def install_os_env_stub(monkeypatch, data_dir: Path) -> None:
    import nekro_agent

    core = ModuleType("nekro_agent.core")
    os_env = ModuleType("nekro_agent.core.os_env")
    os_env.OsEnv = type("OsEnv", (), {"DATA_DIR": str(data_dir)})
    core.os_env = os_env
    monkeypatch.setattr(nekro_agent, "core", core, raising=False)
    monkeypatch.setitem(sys.modules, "nekro_agent.core", core)
    monkeypatch.setitem(sys.modules, "nekro_agent.core.os_env", os_env)


def record(tmp_path: Path) -> StickerRecord:
    plugin_root = tmp_path / "data" / "plugin_data" / "Akiyo.semantic_sticker"
    asset = plugin_root / "assets" / "sticker.png"
    thumbnail = plugin_root / "thumbnails" / "sticker.webp"
    asset.parent.mkdir(parents=True)
    thumbnail.parent.mkdir(parents=True)
    asset.write_bytes(b"content-bytes")
    thumbnail.write_bytes(b"thumbnail-bytes")
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    return StickerRecord(
        id="11111111-1111-4111-8111-111111111111",
        sha256="a" * 64,
        asset_path=str(asset),
        thumbnail_path=str(thumbnail),
        state=StickerState.ACTIVE,
        safety=SafetyState.SAFE,
        description="confused question mark",
        primary_category="confusion",
        emotion_tags=["confused"],
        scene_tags=["asking why"],
        ocr_text="?",
        suitable_scenarios=["clarification"],
        unsuitable_scenarios=["formal apology"],
        mime_type="image/png",
        width=8,
        height=8,
        frame_count=1,
        animated=False,
        byte_size=13,
        analysis_version="v1",
        vector_version=1,
        created_at=now,
        updated_at=now,
    )


class FakeService:
    def __init__(self, sticker: StickerRecord) -> None:
        self.sticker = sticker
        self.calls: list[tuple[str, object]] = []
        self.upload_failures: set[str] = set()

    async def upload(self, payload, actor: str) -> UploadOutcome:
        self.calls.append(("upload", {"actor": actor, "filename": payload.filename, "content": payload.content}))
        if payload.filename in self.upload_failures:
            raise ValueError("invalid image")
        job = JobRecord(
            id="job-1", sticker_id=self.sticker.id, job_type="analysis", state="pending",
            attempt_count=0, created_at=self.sticker.created_at, updated_at=self.sticker.updated_at,
        )
        return UploadOutcome(record=self.sticker.model_copy(update={"state": StickerState.PENDING}), job=job, duplicate=False)

    async def list_stickers(self, filters) -> StickerPage:
        self.calls.append(("list", filters))
        return StickerPage(items=[self.sticker], total=1, offset=filters.offset, limit=filters.limit)

    async def get_sticker(self, sticker_id: str) -> StickerRecord:
        self.calls.append(("get", sticker_id))
        if sticker_id != self.sticker.id:
            from nekro_plugin_semantic_sticker.database import StickerNotFoundError
            raise StickerNotFoundError(sticker_id)
        return self.sticker

    async def patch_metadata(self, sticker_id: str, patch, actor: str) -> StickerRecord:
        self.calls.append(("patch", {"id": sticker_id, "actor": actor, "patch": patch}))
        return self.sticker.model_copy(update={"description": patch.description or self.sticker.description})

    async def reanalyze(self, sticker_id: str, actor: str) -> JobRecord:
        self.calls.append(("reanalyze", {"id": sticker_id, "actor": actor}))
        return JobRecord(
            id="job-2", sticker_id=sticker_id, job_type="analysis", state="pending",
            attempt_count=0, created_at=self.sticker.created_at, updated_at=self.sticker.updated_at,
        )

    async def reindex(self, sticker_id: str, actor: str) -> StickerRecord:
        self.calls.append(("reindex", {"id": sticker_id, "actor": actor}))
        return self.sticker.model_copy(update={"vector_version": 2})

    async def delete(self, sticker_id: str, actor: str) -> None:
        self.calls.append(("delete", {"id": sticker_id, "actor": actor}))

    async def batch_delete(self, sticker_ids: list[str], actor: str) -> BatchDeleteResult:
        self.calls.append(("batch_delete", {"ids": sticker_ids, "actor": actor}))
        return BatchDeleteResult(requested=len(sticker_ids), deleted=len(sticker_ids))

    async def full_reindex(self, actor: str) -> ReindexResult:
        self.calls.append(("full_reindex", actor))
        return ReindexResult(requested=1, indexed=1)

    async def stats(self) -> StickerStats:
        self.calls.append(("stats", None))
        return StickerStats(
            total=1, storage_bytes=100, indexed_count=1, failure_count=0,
            by_state={"active": 1}, by_category={"confusion": 1},
        )


@pytest.fixture
def web_root(tmp_path: Path) -> Path:
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text(
        "<html><head><style>/*__INLINE_STYLE__*/</style></head><body>panel<script>/*__INLINE_SCRIPT__*/</script></body></html>",
        encoding="utf-8",
    )
    (root / "style.css").write_text("body{color:#123}", encoding="utf-8")
    (root / "app.js").write_text("window.panelReady=true;", encoding="utf-8")
    return root


@pytest.fixture
def router_harness(tmp_path: Path, web_root: Path, monkeypatch):
    from nekro_agent.services.user.deps import get_current_super_user
    from nekro_plugin_semantic_sticker.router import build_router

    install_os_env_stub(monkeypatch, tmp_path / "data")
    service = FakeService(record(tmp_path))
    router = build_router(lambda: service, auth_dependency=get_current_super_user, web_root=web_root)
    app = FastAPI()
    app.include_router(router, prefix="/plugins/Akiyo.semantic_sticker")
    return app, router, service, get_current_super_user


def dependency_calls(route: APIRoute) -> set[object]:
    found: set[object] = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dependant = stack.pop()
        found.add(dependant.call)
        stack.extend(dependant.dependencies)
    return found


def test_every_declared_route_is_present_and_superuser_protected(router_harness) -> None:
    _app, router, _service, auth = router_harness
    actual = {(method, route.path) for route in router.routes for method in route.methods}
    assert actual == DECLARED
    for route in router.routes:
        if isinstance(route, APIRoute):
            assert auth in dependency_calls(route), route.path


@pytest.mark.asyncio
async def test_root_static_and_api_reject_unauthenticated_requests(router_harness) -> None:
    app, _router, _service, _auth = router_harness
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path in (
            "/plugins/Akiyo.semantic_sticker/",
            "/plugins/Akiyo.semantic_sticker/static/app.js",
            "/plugins/Akiyo.semantic_sticker/api/stats",
        ):
            assert (await client.get(path)).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["non-super", "inactive-super"])
async def test_forbidden_users_receive_403(router_harness, label: str) -> None:
    app, _router, _service, auth = router_harness

    async def forbidden_user():
        raise HTTPException(status_code=403, detail=label)

    app.dependency_overrides[auth] = forbidden_user
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/plugins/Akiyo.semantic_sticker/")).status_code == 403


@pytest.mark.asyncio
async def test_active_superuser_can_load_inline_root_and_allowlisted_static(router_harness) -> None:
    app, _router, _service, auth = router_harness

    async def active_super():
        return SimpleNamespace(username="root-admin", is_active=True, perm_level=999)

    app.dependency_overrides[auth] = active_super
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        root = await client.get("/plugins/Akiyo.semantic_sticker/")
        static = await client.get("/plugins/Akiyo.semantic_sticker/static/style.css")
        blocked = await client.get("/plugins/Akiyo.semantic_sticker/static/%2e%2e%2fsecret")

    assert root.status_code == 200
    assert "body{color:#123}" in root.text and "window.panelReady=true" in root.text
    assert "/*__INLINE_STYLE__*/" not in root.text and "/*__INLINE_SCRIPT__*/" not in root.text
    assert static.status_code == 200
    assert blocked.status_code in {404, 422}


@pytest.mark.asyncio
async def test_unauthenticated_upload_rejects_before_multipart_body_parsing(router_harness) -> None:
    app, router, _service, _auth = router_harness
    upload_route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == "/api/stickers" and "POST" in route.methods
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/plugins/Akiyo.semantic_sticker/api/stickers",
            content=b"not-a-valid-multipart-body",
            headers={"Content-Type": "multipart/form-data; boundary=broken-boundary"},
        )

    assert upload_route.body_field is None
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_token_is_promoted_before_nested_auth_dependency_and_removed_from_scope(
    tmp_path: Path,
    web_root: Path,
    monkeypatch,
) -> None:
    from nekro_plugin_semantic_sticker.router import build_router

    observed: dict[str, object] = {}

    async def query_sensitive_user(request: Request):
        observed["query_token"] = request.query_params.get("token")
        observed["query"] = request.query_params.get("query")
        observed["raw_query"] = request.scope["query_string"]
        observed["authorization"] = request.headers.get("Authorization")
        if observed["query_token"] or observed["authorization"] != "Bearer query-token-value":
            raise HTTPException(status_code=401, detail="Not authenticated")
        return SimpleNamespace(username="query-admin", is_active=True, perm_level=999)

    async def active_user(current_user=Depends(query_sensitive_user)):
        return current_user

    async def super_user(current_user=Depends(active_user)):
        return current_user

    install_os_env_stub(monkeypatch, tmp_path / "data")
    service = FakeService(record(tmp_path))
    router = build_router(lambda: service, auth_dependency=super_user, web_root=web_root)
    app = FastAPI()
    app.include_router(router, prefix="/plugins/Akiyo.semantic_sticker")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/plugins/Akiyo.semantic_sticker/api/stats?token=Bearer%20query-token-value&query=why",
            headers={"Authorization": "Bearer header-token-value"},
        )

    assert response.status_code == 200
    assert observed == {
        "query_token": None,
        "query": "why",
        "raw_query": b"query=why",
        "authorization": "Bearer query-token-value",
    }


@pytest.mark.asyncio
async def test_batch_upload_returns_202_and_uses_authenticated_actor(router_harness) -> None:
    app, _router, service, auth = router_harness

    async def active_super():
        return SimpleNamespace(username="root-admin")

    app.dependency_overrides[auth] = active_super
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/plugins/Akiyo.semantic_sticker/api/stickers",
            files=[
                ("files", ("one.png", b"one", "image/png")),
                ("files", ("two.gif", b"two", "image/gif")),
            ],
        )

    assert response.status_code == 202
    assert len(response.json()) == 2
    upload_calls = [payload for name, payload in service.calls if name == "upload"]
    assert [call["actor"] for call in upload_calls] == ["root-admin", "root-admin"]
    assert [call["filename"] for call in upload_calls] == ["one.png", "two.gif"]


@pytest.mark.asyncio
async def test_batch_upload_limits_file_count_before_reading_files(router_harness) -> None:
    app, _router, service, auth = router_harness

    async def active_super():
        return SimpleNamespace(username="root-admin")

    app.dependency_overrides[auth] = active_super
    files = [("files", (f"{index}.png", b"x", "image/png")) for index in range(21)]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/plugins/Akiyo.semantic_sticker/api/stickers", files=files)

    assert response.status_code == 400
    assert [call for call in service.calls if call[0] == "upload"] == []


@pytest.mark.asyncio
async def test_batch_upload_preserves_successful_outcomes_when_later_file_fails(router_harness) -> None:
    app, _router, service, auth = router_harness
    service.upload_failures.add("bad.png")

    async def active_super():
        return SimpleNamespace(username="root-admin")

    app.dependency_overrides[auth] = active_super
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/plugins/Akiyo.semantic_sticker/api/stickers",
            files=[
                ("files", ("good.png", b"good", "image/png")),
                ("files", ("bad.png", b"bad", "image/png")),
            ],
        )

    assert response.status_code == 202
    assert response.json()[0]["ok"] is True
    assert response.json()[0]["filename"] == "good.png"
    assert response.json()[0]["record"]["id"] == service.sticker.id
    assert response.json()[1] == {"ok": False, "filename": "bad.png", "error": "invalid image"}


@pytest.mark.asyncio
async def test_filters_metadata_and_authenticated_previews(router_harness) -> None:
    app, _router, service, auth = router_harness

    async def active_super():
        return SimpleNamespace(username="admin")

    app.dependency_overrides[auth] = active_super
    base = "/plugins/Akiyo.semantic_sticker/api/stickers"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        listing = await client.get(base, params={"category": "confusion", "tags": "confused,asking why", "state": "active", "query": "why", "limit": 10})
        metadata = await client.get(f"{base}/{service.sticker.id}", params={"view": "metadata"})
        thumbnail = await client.get(f"{base}/{service.sticker.id}", params={"view": "thumbnail"})
        content = await client.get(f"{base}/{service.sticker.id}", params={"view": "content"})

    assert listing.status_code == 200 and listing.json()["total"] == 1
    filters = [payload for name, payload in service.calls if name == "list"][-1]
    assert filters.category == "confusion" and filters.tags == ["confused", "asking why"]
    assert filters.state is StickerState.ACTIVE and filters.query == "why"
    assert metadata.status_code == 200 and metadata.json()["id"] == service.sticker.id
    assert thumbnail.content == b"thumbnail-bytes"
    assert content.content == b"content-bytes"


@pytest.mark.asyncio
async def test_preview_rejects_database_paths_outside_managed_roots(router_harness, tmp_path: Path) -> None:
    app, _router, service, auth = router_harness
    original = service.sticker
    outside_asset = tmp_path / "outside.png"
    outside_thumbnail = tmp_path / "outside.webp"
    outside_asset.write_bytes(b"outside-asset-secret")
    outside_thumbnail.write_bytes(b"outside-thumbnail-secret")

    async def active_super():
        return SimpleNamespace(username="admin")

    app.dependency_overrides[auth] = active_super
    base = f"/plugins/Akiyo.semantic_sticker/api/stickers/{service.sticker.id}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        service.sticker = original.model_copy(update={"asset_path": str(outside_asset)})
        content = await client.get(base, params={"view": "content"})
        service.sticker = original.model_copy(update={"thumbnail_path": str(outside_thumbnail)})
        thumbnail = await client.get(base, params={"view": "thumbnail"})

    assert content.status_code == 404
    assert content.content != b"outside-asset-secret"
    assert thumbnail.status_code == 404
    assert thumbnail.content != b"outside-thumbnail-secret"


@pytest.mark.asyncio
async def test_service_busy_and_policy_errors_have_explicit_http_semantics(router_harness) -> None:
    from nekro_plugin_semantic_sticker.service import StickerBusyError, StickerPolicyError

    app, _router, service, auth = router_harness

    async def active_super():
        return SimpleNamespace(username="admin")

    async def busy_reanalyze(sticker_id: str, actor: str):
        raise StickerBusyError("analysis already pending")

    async def policy_reindex(sticker_id: str, actor: str):
        raise StickerPolicyError("unsafe sticker")

    service.reanalyze = busy_reanalyze
    service.reindex = policy_reindex
    app.dependency_overrides[auth] = active_super
    base = f"/plugins/Akiyo.semantic_sticker/api/stickers/{service.sticker.id}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        busy = await client.post(base + "/reanalyze")
        policy = await client.post(base + "/reindex")

    assert busy.status_code == 409
    assert busy.json() == {"detail": "analysis already pending"}
    assert policy.status_code == 422
    assert policy.json() == {"detail": "unsafe sticker"}


@pytest.mark.asyncio
async def test_write_routes_use_dependency_username_not_request_data(router_harness) -> None:
    app, _router, service, auth = router_harness

    async def active_super():
        return SimpleNamespace(username="trusted-admin")

    app.dependency_overrides[auth] = active_super
    base = "/plugins/Akiyo.semantic_sticker/api"
    sticker_id = service.sticker.id
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.patch(f"{base}/stickers/{sticker_id}", json={"description": "updated", "reason": "fix", "actor": "forged"})).status_code == 200
        assert (await client.post(f"{base}/stickers/{sticker_id}/reanalyze", json={"actor": "forged"})).status_code == 200
        assert (await client.post(f"{base}/stickers/{sticker_id}/reindex", json={"actor": "forged"})).status_code == 200
        assert (await client.delete(f"{base}/stickers/{sticker_id}", params={"actor": "forged"})).status_code == 204
        assert (await client.post(f"{base}/stickers/batch-delete", json={"sticker_ids": [sticker_id], "actor": "forged"})).status_code == 200
        assert (await client.post(f"{base}/reindex", json={"actor": "forged"})).status_code == 200
        assert (await client.get(f"{base}/stats")).status_code == 200

    actor_payloads = [payload for name, payload in service.calls if name in {"patch", "reanalyze", "reindex", "delete", "batch_delete"}]
    assert all(payload["actor"] == "trusted-admin" for payload in actor_payloads)
    assert ("full_reindex", "trusted-admin") in service.calls