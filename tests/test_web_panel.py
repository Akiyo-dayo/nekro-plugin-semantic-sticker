from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def web_files() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1] / "web"
    return root / "index.html", root / "style.css", root / "app.js"


def read_files(web_files: tuple[Path, Path, Path]) -> tuple[str, str, str]:
    index, style, app = web_files
    return (
        index.read_text(encoding="utf-8"),
        style.read_text(encoding="utf-8"),
        app.read_text(encoding="utf-8"),
    )


def test_web_panel_exposes_required_controls(web_files) -> None:
    html, _style, script = read_files(web_files)
    for element_id in (
        "drop-zone", "file-input", "upload-queue", "sticker-grid", "metadata-panel",
        "filter-category", "filter-tags", "filter-state", "filter-time", "filter-query",
        "batch-delete", "full-reindex", "stats-storage", "stats-failures",
        "metadata-description", "metadata-category", "metadata-emotion-tags",
        "metadata-scene-tags", "metadata-ocr", "metadata-suitable",
        "metadata-unsuitable", "metadata-reason", "metadata-safety", "metadata-state",
        "save-metadata", "reanalyze-sticker", "reindex-sticker", "delete-sticker",
        "status-region", "selection-count", "access-gate", "access-title",
        "access-message", "login-link", "workspace",
    ):
        assert f'id="{element_id}"' in html
    assert "innerHTML" not in script


def test_root_assets_are_inline_ready_without_external_dependencies(web_files) -> None:
    html, style, script = read_files(web_files)
    assert "/*__INLINE_STYLE__*/" in html
    assert "/*__INLINE_SCRIPT__*/" in html
    assert not re.search(r"<script[^>]+src=", html, re.I)
    assert not re.search(r"<link[^>]+rel=[\"']stylesheet", html, re.I)
    assert "@import" not in style
    assert "http://" not in style + script and "https://" not in style + script


def test_token_priority_reuses_same_origin_na_login_and_cleans_only_token_from_url(web_files) -> None:
    _html, _style, script = read_files(web_files)
    assert 'const SESSION_TOKEN_KEY = "na_console_token"' in script
    assert 'const NA_AUTH_STORAGE_KEY = "auth-storage"' in script
    assert 'new URLSearchParams(location.search)' in script
    assert 'params.get("token")' in script
    assert 'params.delete("token")' in script
    assert 'sessionStorage.getItem(SESSION_TOKEN_KEY)' in script
    assert 'sessionStorage.setItem(SESSION_TOKEN_KEY' in script
    assert 'localStorage.getItem(NA_AUTH_STORAGE_KEY)' in script
    assert "JSON.parse" in script
    assert ".state" in script and ".token" in script
    assert "history.replaceState" in script
    assert "location.hash" in script
    assert 'headers.set("Authorization", "Bearer " + accessToken)' in script


def test_access_states_distinguish_login_from_missing_superuser_permission(web_files) -> None:
    html, style, script = read_files(web_files)
    assert 'id="login-link"' in html and 'href="/#/login"' in html
    assert 'id="workspace"' in html and "hidden" in html.split('id="workspace"', 1)[1].split(">", 1)[0]
    assert 'id="metadata-panel"' in html and "hidden" in html.split('id="metadata-panel"', 1)[1].split(">", 1)[0]
    assert "[hidden]" in style
    assert 'showAccessState("unauthenticated")' in script
    assert 'showAccessState("forbidden")' in script
    assert "请先登录 NekroAgent 后再访问此控制台" in script
    assert "当前账户不是超级管理员，无权访问此控制台" in script
    assert "前往 NA 登录" in html


def test_only_read_only_initialization_can_retry_with_a_fresh_na_token(web_files) -> None:
    _html, _style, script = read_files(web_files)
    assert "retryOnUnauthorized" in script
    assert "readNaStoredToken()" in script
    assert "freshToken !== accessToken" in script
    assert "authOptions.retryOnUnauthorized === true" in script
    assert 'const requestMethod = String(options.method || "GET").toUpperCase()' in script
    assert 'requestMethod === "GET"' in script
    initialize = script.split("async function initialize", 1)[1]
    assert "await loadStats({retryOnUnauthorized: true});" in initialize
    assert "showWorkspace();" in initialize
    assert initialize.index("await loadStats({retryOnUnauthorized: true});") < initialize.index("showWorkspace();")
    for start, end in (
        ("async function uploadFiles", "async function saveMetadata"),
        ("async function saveMetadata", "async function runRecordAction"),
        ("async function runRecordAction", "async function deleteActiveSticker"),
        ("async function deleteActiveSticker", "async function deleteSelected"),
        ("async function deleteSelected", "async function fullReindex"),
        ("async function fullReindex", "function populateCategories"),
    ):
        section = script.split(start, 1)[1].split(end, 1)[0]
        assert "retryOnUnauthorized: true" not in section


def test_all_api_and_preview_fetches_use_shared_authenticated_helper(web_files) -> None:
    _html, _style, script = read_files(web_files)
    assert "async function api(" in script
    raw_fetches = [match.start() for match in re.finditer(r"\bfetch\(", script)]
    assert len(raw_fetches) == 1
    assert "await api(" in script
    assert 'view=thumbnail' in script and 'view=content' in script
    assert "response.blob()" in script
    assert "URL.createObjectURL" in script
    assert "URL.revokeObjectURL" in script


def test_rendering_uses_textcontent_and_safe_dom_construction(web_files) -> None:
    _html, _style, script = read_files(web_files)
    assert ".textContent" in script
    assert "document.createElement" in script
    assert "insertAdjacentHTML" not in script
    assert "outerHTML" not in script
    assert "eval(" not in script
    assert "new Function" not in script


def test_polling_is_bounded_and_busy_states_are_explicit(web_files) -> None:
    html, _style, script = read_files(web_files)
    assert "MAX_POLL_ATTEMPTS" in script
    assert "POLL_BACKOFF_MS" in script
    assert "Math.min" in script
    assert "setTimeout" in script
    assert "setBusy" in script
    assert "disabled" in script
    assert 'aria-live="polite"' in html
    assert 'role="status"' in html


def test_polling_stops_after_authentication_or_permission_failure(web_files) -> None:
    _html, _style, script = read_files(web_files)
    poll_section = script.split("async function pollSticker", 1)[1].split("function queueRow", 1)[0]
    assert "catch (error)" in poll_section
    catch_section = poll_section.split("catch (error)", 1)[1]
    guard = "error instanceof HttpError && (error.status === 401 || error.status === 403)"
    assert guard in catch_section
    assert catch_section.index(guard) < catch_section.index("await pollSticker(")


def test_destructive_and_batch_actions_require_confirmation(web_files) -> None:
    _html, _style, script = read_files(web_files)
    assert script.count("confirm(") >= 3
    assert "batch-delete" in script
    assert "full-reindex" in script
    assert "delete-sticker" in script


def test_metadata_editor_covers_mutable_fields_and_locks_state_safety(web_files) -> None:
    html, _style, script = read_files(web_files)
    assert 'id="metadata-safety"' in html and "readonly" in html.split('id="metadata-safety"', 1)[1].split(">", 1)[0]
    assert 'id="metadata-state"' in html and "readonly" in html.split('id="metadata-state"', 1)[1].split(">", 1)[0]
    for key in (
        "description", "primary_category", "emotion_tags", "scene_tags", "ocr_text",
        "suitable_scenarios", "unsuitable_scenarios", "reason",
    ):
        assert key in script
    assert "safety:" not in script and "state:" not in script


def test_accessible_product_ui_tokens_and_responsive_structure(web_files) -> None:
    html, style, _script = read_files(web_files)
    assert "oklch(" in style
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", style)
    assert ":focus-visible" in style
    assert "prefers-reduced-motion" in style
    assert "@media" in style
    assert "min-height: 44px" in style
    assert 'aria-label=' in html
    assert 'aria-describedby=' in html
    assert "grid-template-columns" in style