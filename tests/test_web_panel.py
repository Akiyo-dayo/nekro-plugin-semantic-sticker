from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def web_files() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1] / "source" / "nekro_plugin_semantic_sticker" / "web"
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
        "status-region", "selection-count",
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


def test_token_is_session_scoped_and_removed_from_url(web_files) -> None:
    _html, _style, script = read_files(web_files)
    assert 'new URLSearchParams(location.search).get("token")' in script
    assert 'sessionStorage.setItem("na_console_token"' in script
    assert "localStorage" not in script
    assert "history.replaceState" in script
    assert 'sessionStorage.getItem("na_console_token")' in script
    assert 'headers.set("Authorization", "Bearer " + token)' in script


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