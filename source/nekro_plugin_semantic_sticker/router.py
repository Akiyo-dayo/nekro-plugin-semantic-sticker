from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict
from starlette.datastructures import UploadFile as StarletteUploadFile

from nekro_agent.services.user.deps import get_current_super_user

from . import plugin
from .agent_tools import get_service
from .database import StickerNotFoundError
from .models import MetadataPatch, StickerFilters, StickerState, UploadPayload
from .service import StickerBusyError, StickerPolicyError


class MetadataPatchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    primary_category: str | None = None
    emotion_tags: list[str] | None = None
    scene_tags: list[str] | None = None
    ocr_text: str | None = None
    suitable_scenarios: list[str] | None = None
    unsuitable_scenarios: list[str] | None = None
    reason: str

    def to_patch(self) -> MetadataPatch:
        return MetadataPatch.model_validate(self.model_dump())


class BatchDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sticker_ids: list[str]


MAX_UPLOAD_FILES = 20
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9\-._~+/]+=*")


def _promote_query_token(request: Request) -> None:
    raw_query = request.scope.get("query_string", b"")
    if not raw_query:
        return
    remaining: list[bytes] = []
    query_token: str | None = None
    token_present = False
    for component in raw_query.split(b"&"):
        raw_key, separator, raw_value = component.partition(b"=")
        try:
            key = unquote_plus(raw_key.decode("ascii"))
        except UnicodeDecodeError:
            remaining.append(component)
            continue
        if key != "token":
            remaining.append(component)
            continue
        token_present = True
        if not separator:
            query_token = ""
            continue
        try:
            query_token = unquote_plus(raw_value.decode("ascii"))
        except UnicodeDecodeError:
            query_token = ""
    if not token_present:
        return

    request.scope["query_string"] = b"&".join(remaining)
    request.__dict__.pop("_query_params", None)

    token = query_token or ""
    if token.startswith("Bearer "):
        token = token[7:]
    if not token or _BEARER_TOKEN.fullmatch(token) is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    headers = [
        (name, value)
        for name, value in request.scope.get("headers", [])
        if name.lower() != b"authorization"
    ]
    headers.append((b"authorization", b"Bearer " + token.encode("ascii")))
    request.scope["headers"] = headers
    request.__dict__.pop("_headers", None)


def _plugin_data_root() -> Path:
    from nekro_agent.core.os_env import OsEnv

    return (Path(OsEnv.DATA_DIR) / "plugin_data" / plugin.key).resolve()


def _managed_preview_path(raw_path: str, directory: str) -> Path:
    root = (_plugin_data_root() / directory).resolve()
    target = Path(raw_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Sticker file not found")
    return target


def _service_error(error: StickerBusyError | StickerPolicyError) -> HTTPException:
    if isinstance(error, StickerBusyError):
        return HTTPException(status_code=409, detail=str(error))
    return HTTPException(status_code=422, detail=str(error))


def _web_root() -> Path:
    return Path(__file__).resolve().parent / "web"


def _actor(user: object) -> str:
    username = str(getattr(user, "username", "")).strip()
    if not username:
        raise HTTPException(status_code=403, detail="Authenticated administrator username is missing")
    return username


def _not_found(error: StickerNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Sticker not found: {error}")


def build_router(
    runtime_provider=get_service,
    *,
    auth_dependency=get_current_super_user,
    web_root: Path | None = None,
) -> APIRouter:
    root = Path(web_root or _web_root()).resolve()
    router = APIRouter(dependencies=[Depends(_promote_query_token), Depends(auth_dependency)])

    @router.get("/", response_class=HTMLResponse)
    async def index_page() -> HTMLResponse:
        try:
            html = (root / "index.html").read_text(encoding="utf-8")
            style = (root / "style.css").read_text(encoding="utf-8")
            script = (root / "app.js").read_text(encoding="utf-8")
        except OSError as error:
            raise HTTPException(status_code=500, detail="Sticker panel assets are unavailable") from error
        html = html.replace("/*__INLINE_STYLE__*/", style).replace("/*__INLINE_SCRIPT__*/", script)
        return HTMLResponse(html)

    @router.get("/static/{path:path}")
    async def static_asset(path: str) -> FileResponse:
        allowed = {
            "index.html": "text/html; charset=utf-8",
            "style.css": "text/css; charset=utf-8",
            "app.js": "application/javascript; charset=utf-8",
        }
        if path not in allowed:
            raise HTTPException(status_code=404, detail="Static asset not found")
        target = (root / path).resolve()
        if target.parent != root or not target.is_file():
            raise HTTPException(status_code=404, detail="Static asset not found")
        return FileResponse(target, media_type=allowed[path])

    @router.post("/api/stickers", status_code=status.HTTP_202_ACCEPTED)
    async def upload_stickers(
        request: Request,
        current_user: Any = Depends(auth_dependency),
    ) -> JSONResponse:
        actor = _actor(current_user)
        service = runtime_provider()
        outcomes: list[dict[str, Any]] = []
        async with request.form(max_files=MAX_UPLOAD_FILES) as form:
            uploads = form.getlist("files")
            if not uploads or len(uploads) > MAX_UPLOAD_FILES:
                raise HTTPException(status_code=400, detail=f"Upload requires 1-{MAX_UPLOAD_FILES} files")
            if any(not isinstance(upload, StarletteUploadFile) for upload in uploads):
                raise HTTPException(status_code=400, detail="Upload parts must be files")
            for upload in uploads:
                filename = upload.filename or "unnamed"
                try:
                    payload = UploadPayload(
                        content=await upload.read(),
                        filename=upload.filename,
                        content_type=upload.content_type,
                    )
                    outcome = jsonable_encoder(await service.upload(payload, actor=actor))
                    outcomes.append({"ok": True, "filename": filename, **outcome})
                except ValueError as error:
                    outcomes.append({"ok": False, "filename": filename, "error": str(error) or "Invalid upload"})
                except Exception:
                    outcomes.append({"ok": False, "filename": filename, "error": "Upload failed"})
        return JSONResponse(status_code=202, content=outcomes)

    @router.get("/api/stickers")
    async def list_stickers(
        category: str | None = None,
        tags: str | None = None,
        state: StickerState | None = None,
        query: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        filters = StickerFilters(
            category=category,
            tags=[item.strip() for item in (tags or "").split(",") if item.strip()],
            state=state,
            query=query,
            created_from=created_from,
            created_to=created_to,
            offset=offset,
            limit=limit,
        )
        return await runtime_provider().list_stickers(filters)

    @router.post("/api/stickers/batch-delete")
    async def batch_delete(
        request: BatchDeleteRequest,
        current_user: Any = Depends(auth_dependency),
    ):
        return await runtime_provider().batch_delete(request.sticker_ids, actor=_actor(current_user))

    @router.post("/api/reindex")
    async def full_reindex(current_user: Any = Depends(auth_dependency)):
        return await runtime_provider().full_reindex(actor=_actor(current_user))

    @router.get("/api/stats")
    async def stats_view():
        return await runtime_provider().stats()
    @router.get("/api/stickers/{sticker_id}")
    async def sticker_detail(sticker_id: str, view: str = "metadata"):
        try:
            record = await runtime_provider().get_sticker(sticker_id)
        except StickerNotFoundError as error:
            raise _not_found(error) from error
        if view == "metadata":
            return record
        if view == "thumbnail":
            target = _managed_preview_path(record.thumbnail_path, "thumbnails")
            media_type = "image/webp"
        elif view == "content":
            target = _managed_preview_path(record.asset_path, "assets")
            media_type = record.mime_type
        else:
            raise HTTPException(status_code=400, detail="view must be metadata, thumbnail, or content")
        return FileResponse(target, media_type=media_type)

    @router.patch("/api/stickers/{sticker_id}")
    async def patch_sticker(
        sticker_id: str,
        request: MetadataPatchRequest,
        current_user: Any = Depends(auth_dependency),
    ):
        try:
            return await runtime_provider().patch_metadata(
                sticker_id,
                request.to_patch(),
                actor=_actor(current_user),
            )
        except StickerNotFoundError as error:
            raise _not_found(error) from error
        except (StickerBusyError, StickerPolicyError) as error:
            raise _service_error(error) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.delete("/api/stickers/{sticker_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_sticker(
        sticker_id: str,
        current_user: Any = Depends(auth_dependency),
    ) -> Response:
        try:
            await runtime_provider().delete(sticker_id, actor=_actor(current_user))
        except StickerNotFoundError as error:
            raise _not_found(error) from error
        except (StickerBusyError, StickerPolicyError) as error:
            raise _service_error(error) from error
        return Response(status_code=204)

    @router.post("/api/stickers/{sticker_id}/reanalyze")
    async def reanalyze_sticker(
        sticker_id: str,
        current_user: Any = Depends(auth_dependency),
    ):
        try:
            return await runtime_provider().reanalyze(sticker_id, actor=_actor(current_user))
        except StickerNotFoundError as error:
            raise _not_found(error) from error
        except (StickerBusyError, StickerPolicyError) as error:
            raise _service_error(error) from error

    @router.post("/api/stickers/{sticker_id}/reindex")
    async def reindex_sticker(
        sticker_id: str,
        current_user: Any = Depends(auth_dependency),
    ):
        try:
            return await runtime_provider().reindex(sticker_id, actor=_actor(current_user))
        except StickerNotFoundError as error:
            raise _not_found(error) from error
        except (StickerBusyError, StickerPolicyError) as error:
            raise _service_error(error) from error

    return router


@plugin.mount_router()
def mount_semantic_sticker_router() -> APIRouter:
    return build_router(get_service)


__all__ = [
    "BatchDeleteRequest",
    "MetadataPatchRequest",
    "build_router",
    "mount_semantic_sticker_router",
]