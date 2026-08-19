from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def web_assets() -> tuple[str, str, str]:
    root = Path(__file__).resolve().parents[1] / "web"
    return (
        (root / "index.html").read_text(encoding="utf-8"),
        (root / "style.css").read_text(encoding="utf-8"),
        (root / "app.js").read_text(encoding="utf-8"),
    )


@pytest.fixture
def readme_text() -> str:
    path = Path(__file__).resolve().parents[1] / "USER_GUIDE.md"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def project_docs() -> tuple[str, str, str]:
    root = Path(__file__).resolve().parents[1]
    return (
        (root / "README.md").read_text(encoding="utf-8"),
        (root / "USER_GUIDE.md").read_text(encoding="utf-8"),
        (root / "CHANGELOG.md").read_text(encoding="utf-8"),
    )


def test_file_input_has_visible_keyboard_focus_without_overriding_focus_visible(web_assets) -> None:
    html, style, _script = web_assets
    assert html.index('id="file-input"') < html.index('for="file-input"')
    assert 'class="button primary file-input-label"' in html
    assert "#file-input:focus-visible + .file-input-label" in style
    assert not re.search(r":focus[^,{]*\{[^}]*outline:\s*0", style)


def test_mobile_metadata_drawer_has_dialog_focus_and_escape_behavior(web_assets) -> None:
    html, _style, script = web_assets
    panel_tag = html.split('id="metadata-panel"', 1)[1].split(">", 1)[0]
    assert 'role="dialog"' in panel_tag
    assert 'aria-modal="false"' in panel_tag
    assert 'tabindex="-1"' in panel_tag
    assert 'window.matchMedia("(max-width: 860px)")' in script
    assert "returnFocus" in script
    assert "elements.closeMetadata.focus()" in script
    assert 'event.key === "Escape"' in script
    assert "closeMetadata()" in script
    assert ".focus()" in script


def test_active_small_status_text_uses_high_contrast_color(web_assets) -> None:
    _html, style, _script = web_assets
    assert "--success-ink: oklch(0.36" in style
    active_rule = re.search(r'\.queue-item\[data-status="active"\] span\s*\{([^}]*)\}', style)
    assert active_rule is not None
    assert "color: var(--success-ink)" in active_rule.group(1)


def test_upload_ui_renders_each_server_outcome_independently(web_assets) -> None:
    _html, _style, script = web_assets
    upload_function = script.split("async function uploadFiles", 1)[1].split("async function saveMetadata", 1)[0]
    assert "outcome.ok" in upload_function
    assert "successfulCount" in upload_function
    assert "failedCount" in upload_function
    assert "rows.forEach" in upload_function
    assert "outcome.error" not in upload_function
    assert "上传失败，请稍后重试" in upload_function


def test_token_compatibility_does_not_log_or_render_token(web_assets) -> None:
    _html, _style, script = web_assets
    assert "console." not in script
    assert not re.search(r"(?:textContent|setStatus|createNode)\s*\([^\n]*token", script, re.I)


def test_authentication_failures_have_distinct_safe_ui_transitions(web_assets) -> None:
    html, _style, script = web_assets
    assert 'href="/#/login"' in html
    assert "请先登录 NekroAgent 后再访问此控制台" in script
    assert "当前账户不是超级管理员，无权访问此控制台" not in script
    assert 'sessionStorage.removeItem(SESSION_TOKEN_KEY)' in script
    assert 'localStorage.removeItem' not in script
    assert 'location.replace' not in script and 'location.assign' not in script
    assert "window.location =" not in script


def test_all_visible_webui_copy_is_simplified_chinese(web_assets) -> None:
    html, _style, script = web_assets
    combined = html + "\n" + script

    for expected in (
        "语义表情包控制台",
        "上传并自动分析",
        "实例全局表情包库",
        "表情包详情",
        "暂无符合条件的表情包",
        "系统状态",
        "上传队列",
    ):
        assert expected in combined

    for forbidden in (
        "INSTANCE LIBRARY",
        "INGEST",
        "LIBRARY",
        "INSPECT",
        "active + safe",
        "璇箟",
        "鈥?",
        "锛?",
    ):
        assert forbidden not in combined


def test_api_values_are_rendered_with_chinese_labels(web_assets) -> None:
    _html, _style, script = web_assets
    category_labels = {
        "confusion": "疑惑",
        "happiness": "开心",
        "speechlessness": "无语",
        "anger": "生气",
        "sadness": "难过",
        "surprise": "惊讶",
        "agreement": "赞同",
        "refusal": "拒绝",
        "comfort": "安慰",
        "shyness": "害羞",
        "mockery": "调侃",
        "celebration": "庆祝",
        "neutral reaction": "中性反应",
        "other": "其他",
    }
    state_labels = {
        "pending": "等待处理",
        "analyzing": "正在分析",
        "indexing": "正在建立向量",
        "active": "可用",
        "failed": "处理失败",
        "retry_pending": "等待重试",
        "deleting": "正在删除",
        "deleted": "已删除",
    }
    safety_labels = {"safe": "安全", "unsafe": "不安全", "disallowed": "禁止使用"}

    for api_value, label in {**category_labels, **state_labels, **safety_labels}.items():
        assert re.search(rf'["\']{re.escape(api_value)}["\']\s*:\s*["\']{label}["\']', script)

    assert "categoryLabel(category)" in script
    assert "categoryLabel(record.primary_category)" in script
    assert "stateLabel(record.state)" in script
    assert "safetyLabel(record.safety)" in script


def test_http_errors_use_chinese_status_messages_without_backend_body(web_assets) -> None:
    _html, _style, script = web_assets

    assert "HTTP_STATUS_MESSAGES" in script
    for expected in (
        "请求参数不正确",
        "登录状态已失效",
        "当前登录账户无法执行此请求",
        "请求的资源不存在",
        "服务器处理失败，请稍后重试",
        "服务暂时不可用，请稍后重试",
    ):
        assert expected in script
    assert "response.text()" not in script
    assert "error.message" not in script


def test_upload_prechecks_total_request_size_before_submitting(web_assets) -> None:
    _html, _style, script = web_assets
    upload_function = script.split("async function uploadFiles", 1)[1].split("async function saveMetadata", 1)[0]
    assert "DEFAULT_MAX_REQUEST_BYTES" in upload_function
    assert "totalBytes" in upload_function
    assert "超过单次请求限制" in upload_function
    assert "分批上传" in upload_function


def test_chinese_readme_covers_operations_configuration_and_rollback(readme_text) -> None:
    for expected in (
        "Akiyo.semantic_sticker",
        "/plugins/Akiyo.semantic_sticker/",
        "部署要求",
        "client_max_body_size",
        "DEFAULT_MAX_REQUEST_BYTES",
        "上传表情包",
        "自动模型分类",
        "人工修改标签",
        "重试分析",
        "重建向量",
        "删除表情包",
        "普通用户主动保存",
        "AUTO_COLLECT_ENABLED",
        "STRICT_EMOTION_COLLECT",
        "ANALYSIS_MODEL_GROUP",
        "EMBEDDING_MODEL_GROUP",
        "image_only",
        "text_then_image",
        "auto",
        "Bot 被戳",
        "不回退 LLM",
        "PHYSICAL_CHANNEL_COOLDOWN_SECONDS",
        "SEMANTIC_SCORE_THRESHOLD",
        "RECENT_SELECTION_WINDOW",
        "plugin_data/Akiyo.semantic_sticker/config.yaml",

        "configs/nekro-agent.yaml",
        "仅用于插件启用列表",
        "管理员不应为调整 CD 手工修改",
        "docker compose up -d --no-deps --force-recreate --pull never nekro_agent",
        "回滚",
    ):
        assert expected in readme_text

    assert re.search(r"\b0\b.*关闭冷却", readme_text)
    assert re.search(r"\b20\b.*默认", readme_text)
    assert re.search(r"\b60\b.*60 秒", readme_text)


def test_documentation_explains_bare_console_authentication_contract(project_docs) -> None:
    public_readme, detailed_readme, changelog = project_docs

    for document in (public_readme, detailed_readme):
        for expected in (
            "auth-storage",
            "scheme、host、port",
            "/plugins/Akiyo.semantic_sticker/",
            "?token=",
            "地址栏",
            "静态外壳",
            "/api/*",
            "401",
            "403",
            "普通登录用户",
        ):
            assert expected in document
        assert "同源" in document
        assert "前往 NA 登录" in document

    assert "## 1.2.6（2026-08-19）" in changelog
    assert "裸地址" in changelog
    assert "auth-storage" in changelog
    assert "/api/*" in changelog
