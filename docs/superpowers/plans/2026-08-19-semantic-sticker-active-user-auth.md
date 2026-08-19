# Semantic Sticker Active-User Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow every active NA WebUI user to use the semantic-sticker backend while keeping unauthenticated API requests rejected by NA authentication.

**Architecture:** Keep the plugin API dependency boundary in `build_router()`, but change its default dependency from `get_current_super_user` to `get_current_active_user`. Preserve username-based audit actors and the existing browser Token bridge. Update the frontend copy and project documentation so the permission contract no longer claims superuser-only access.

**Tech Stack:** Python 3, FastAPI, pytest/httpx, vanilla JavaScript, Markdown documentation.

## Global Constraints

- All `/api/*` routes remain protected by NekroAgent authentication.
- Any active NA user may read, preview, upload, edit, delete, reanalyze, and reindex stickers.
- Unauthenticated or invalid-token requests must still return HTTP 401.
- Write-operation actors must come from the authenticated user's `username`, never request data.
- Preserve `?token=` compatibility and same-origin `localStorage["auth-storage"].state.token` support.
- Do not modify NekroAgent source or expose plugin APIs anonymously.

## File Map

- Modify: `router.py` — import and default to `get_current_active_user`; keep all API routes behind the dependency.
- Modify: `web/app.js` — remove the obsolete superuser-specific normal error text and forbidden access gate.
- Modify: `web/index.html` — keep the login gate but use ordinary NA-login wording if the gate copy is embedded in markup.
- Modify: `tests/test_auth_router.py` — verify active ordinary users pass all protected routes and unauthenticated requests remain rejected.
- Modify: `tests/test_web_panel.py` and `tests/test_web_assets.py` — assert the frontend no longer advertises superuser-only access while retaining Token behavior and 401 handling.
- Modify: `README.md` and `USER_GUIDE.md` — document active NA-user access and retain deployment/token details.
- Modify: `CHANGELOG.md` — add a release entry for the permission correction.
- Create: `docs/superpowers/specs/2026-08-19-semantic-sticker-active-user-auth-design.md` — approved design already recorded.

### Task 1: Add failing regression coverage for ordinary active users

**Files:**
- Modify: `tests/test_auth_router.py`
- Modify: `tests/test_web_panel.py`
- Modify: `tests/test_web_assets.py`

**Interfaces:**
- Consumes: `build_router(..., auth_dependency=...)`, `get_current_active_user`, current web asset fixtures.
- Produces: Tests that fail against the current superuser-only implementation and describe the new active-user contract.

- [ ] **Step 1: Change the router test harness dependency target and add an ordinary-user API test.**

Use the NA active-user dependency in the fixture and add an override with `perm_level=0` that calls `GET /api/stats` and a write route. The ordinary-user test must assert HTTP 200 and the service call actor equals the ordinary user's username.

```python
from nekro_agent.services.user.deps import get_current_active_user

router = build_router(lambda: service, auth_dependency=get_current_active_user, web_root=web_root)

@pytest.mark.asyncio
async def test_active_non_superuser_can_use_full_backend(router_harness) -> None:
    app, _router, service, auth = router_harness

    async def active_user():
        return SimpleNamespace(username="ordinary-user", is_active=True, perm_level=0)

    app.dependency_overrides[auth] = active_user
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        stats = await client.get("/plugins/Akiyo.semantic_sticker/api/stats")
        reindex = await client.post("/plugins/Akiyo.semantic_sticker/api/reindex")

    assert stats.status_code == 200
    assert reindex.status_code == 200
    assert ("full_reindex", "ordinary-user") in service.calls
```

- [ ] **Step 2: Update existing auth-router expectations from superuser terminology to active-user terminology.**

Rename local helper functions such as `active_super` to `active_user`, and change the forbidden-user regression to cover an inactive user returning 403 from the NA active-user dependency. Keep the anonymous API 401 test and route dependency assertions.

- [ ] **Step 3: Add failing frontend assertions for the new copy and access state.**

Change the web tests so the `403` status is treated as a generic backend error and the script no longer contains the old superuser-only gate strings:

```python
assert "当前账户缺少超级管理员权限" not in script
assert "当前账户不是超级管理员，无权访问此控制台。" not in script
assert 'showAccessState("unauthenticated")' in script
```

- [ ] **Step 4: Run the focused tests and confirm they fail before implementation.**

Run:

```powershell
python -m pytest tests/test_auth_router.py tests/test_web_panel.py tests/test_web_assets.py -q
```

Expected: failures show the current default dependency rejects the ordinary user and/or the frontend still contains superuser-only copy.

### Task 2: Switch the API boundary to NA active-user authentication

**Files:**
- Modify: `router.py`

**Interfaces:**
- Consumes: NA's `get_current_active_user` dependency returning an active `DBUser`-compatible object.
- Produces: `build_router()` whose `/api/*` routes require an active NA login but not superuser level.

- [ ] **Step 1: Replace the dependency import and default.**

Change:

```python
from nekro_agent.services.user.deps import get_current_super_user
```

to:

```python
from nekro_agent.services.user.deps import get_current_active_user
```

Change the signature default:

```python
auth_dependency=get_current_active_user,
```

Do not remove `Depends(auth_dependency)` from `api_router`, and do not remove the explicit write-route dependencies that feed `_actor()`.

- [ ] **Step 2: Run the focused auth tests.**

Run:

```powershell
python -m pytest tests/test_auth_router.py -q
```

Expected: all auth-router tests pass, including ordinary active-user access, anonymous 401 behavior, path-safe previews, and username actor assertions.

### Task 3: Align frontend behavior and copy

**Files:**
- Modify: `web/app.js`
- Modify: `web/index.html` only if the gate text requires a markup update.
- Modify: `tests/test_web_panel.py`
- Modify: `tests/test_web_assets.py`

**Interfaces:**
- Consumes: `HttpError`, `showAccessState`, existing Token resolution and shared `api()` helper.
- Produces: A panel that asks users to log in only for 401; it does not claim that ordinary users need superuser permission.

- [ ] **Step 1: Remove the obsolete 403 superuser status message.**

Use a generic permission/error message for unexpected 403 responses, for example:

```javascript
const HTTP_STATUS_MESSAGES = {
  400: "请求参数不正确，请检查后重试。",
  401: "登录状态已失效，请重新登录 NekroAgent 后重试。",
  403: "当前登录账户无法执行此请求，请稍后重试。",
```

Keep `showAccessState("unauthenticated")` for 401. Remove the `forbidden` branch that says `需要超级管理员权限`; if a 403 still occurs, leave the panel in the generic error path rather than presenting a false superuser gate.

- [ ] **Step 2: Update frontend tests.**

Retain assertions for the `auth-storage`/session/query Token priority, Authorization header injection, 401 login link, and polling stop behavior. Replace assertions for the forbidden superuser gate with assertions that the old text is absent and ordinary login copy remains present.

- [ ] **Step 3: Run web tests.**

Run:

```powershell
python -m pytest tests/test_web_panel.py tests/test_web_assets.py -q
```

Expected: all web asset and panel tests pass.

### Task 4: Update user-facing permission documentation

**Files:**
- Modify: `README.md`
- Modify: `USER_GUIDE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: The approved permission contract from `docs/superpowers/specs/2026-08-19-semantic-sticker-active-user-auth-design.md`.
- Produces: Consistent Chinese documentation stating that active NA users can use all panel features, while unauthenticated users cannot call the APIs.

- [ ] **Step 1: Replace superuser-only instructions.**

Update the console sections to say “在 NekroAgent WebUI 登录任意已启用账户” rather than “登录超级管理员账户”. State that ordinary active users can use upload, metadata editing, deletion, reanalysis, and reindexing.

- [ ] **Step 2: Preserve token and deployment requirements.**

Keep the same-origin `auth-storage` behavior, trusted `?token=` compatibility, `scheme、host、port` restriction, and `client_max_body_size` upload requirement. Reword 403 documentation so it is not presented as the expected ordinary-user access boundary.

- [ ] **Step 3: Add a changelog entry.**

Add a new top entry describing that semantic-sticker WebUI API access now requires an active NA login but no longer requires superuser permission, while unauthenticated requests remain protected.

- [ ] **Step 4: Run documentation contract tests.**

Run:

```powershell
python -m pytest tests/test_web_assets.py -q
```

Expected: documentation and visible-copy assertions pass.

### Task 5: Run full verification and review the diff

**Files:**
- No new files; review all changes above.

**Interfaces:**
- Consumes: Completed router, frontend, tests, and documentation changes.
- Produces: Verified branch state ready for user review.

- [ ] **Step 1: Run the complete test suite.**

Run:

```powershell
python -m pytest -q
```

Expected: exit code 0 and all tests pass.

- [ ] **Step 2: Run syntax checks for changed runtime files.**

Run:

```powershell
python -m compileall -q router.py
node --check web/app.js
```

Expected: both commands exit 0.

- [ ] **Step 3: Inspect the final diff and status.**

Run:

```powershell
git diff --check
git diff --stat
git status --short --branch
```

Expected: no whitespace errors; only the approved router/frontend/test/documentation/spec changes are present.

- [ ] **Step 4: Commit the implementation.**

```powershell
git add router.py web/app.js web/index.html tests/test_auth_router.py tests/test_web_panel.py tests/test_web_assets.py README.md USER_GUIDE.md CHANGELOG.md docs/superpowers/plans/2026-08-19-semantic-sticker-active-user-auth.md
git commit -m "fix: allow active NA users in sticker panel"
```
