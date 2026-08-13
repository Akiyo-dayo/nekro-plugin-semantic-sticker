# Semantic Sticker Bare Console Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow a logged-in NekroAgent super administrator to open `/plugins/Akiyo.semantic_sticker/` directly while keeping every plugin data and management API protected by server-side superuser authentication.

**Architecture:** Split the plugin router into a public static shell and a protected `/api` subrouter. The inline console JavaScript resolves a token from `?token=`, tab-scoped session storage, or same-origin NA `auth-storage`, then uses the existing Bearer request path; explicit UI states handle unauthenticated and non-superuser responses without changing NA.

**Tech Stack:** Python 3, FastAPI, pytest/httpx ASGI tests, dependency-free HTML/CSS/JavaScript, Git.

## Global Constraints

- Modify only `Akiyo-dayo/nekro-plugin-semantic-sticker`; do not modify NekroAgent or NUP code.
- Keep every `/api/*` route protected by `get_current_super_user`.
- Public routes may return only the static HTML/CSS/JavaScript shell and must not expose plugin data.
- Preserve `?token=` compatibility and remove the token from the visible URL after front-end startup.
- Token priority is query parameter, plugin `sessionStorage`, then same-origin `localStorage["auth-storage"].state.token`.
- Do not write tokens to logs, HTML, cookies, persistent plugin files, or visible UI.
- HTTP 401 shows a login prompt and same-origin `/#/login` button; HTTP 403 shows a distinct superuser-permission message.
- Retry authentication at most once, only for the read-only initialization probe; never automatically replay a write request.
- Do not deploy, restart, or hot-update any NA instance during implementation.

## File Map

- `source/nekro_plugin_semantic_sticker/router.py`: public shell router, protected API subrouter, existing query-token compatibility.
- `source/nekro_plugin_semantic_sticker/web/app.js`: token resolution, authenticated fetch, one-time read recovery, access-state transitions.
- `source/nekro_plugin_semantic_sticker/web/index.html`: access-gate markup and hidden-until-authenticated workspace.
- `source/nekro_plugin_semantic_sticker/web/style.css`: access-gate layout and explicit hidden-state styling.
- `tests/test_auth_router.py`: route visibility and server-side authentication regression tests.
- `tests/test_web_panel.py`: browser asset/authentication contract tests.
- `tests/test_web_assets.py`: Chinese copy, secret-safety, and README contract regression tests.
- `README.md`: public usage and same-origin authentication documentation.
- `source/nekro_plugin_semantic_sticker/README.md`: detailed console access and troubleshooting documentation.
- `CHANGELOG.md`: unreleased fix entry.

---

### Task 1: Split the public shell from protected API routes

**Files:**
- Modify: `tests/test_auth_router.py:183-235`
- Modify: `source/nekro_plugin_semantic_sticker/router.py:132-307`

**Interfaces:**
- Consumes: existing `build_router(runtime_provider, *, auth_dependency, web_root)` and `_promote_query_token(request)`.
- Produces: a top-level `router` with public `/` and `/static/{path:path}`, plus an included `api_router` with prefix `/api` and dependency `Depends(auth_dependency)`.

- [ ] **Step 1: Replace the route protection expectations with failing public-shell/API-boundary tests**

Update `test_every_declared_route_is_present_and_superuser_protected` so it asserts:

```python
for route in router.routes:
    if not isinstance(route, APIRoute):
        continue
    if route.path.startswith("/api/"):
        assert auth in dependency_calls(route), route.path
    else:
        assert auth not in dependency_calls(route), route.path
```

Replace `test_root_static_and_api_reject_unauthenticated_requests` with a test that expects anonymous root and allowlisted static assets to return 200 while `/api/stats` returns 401. Change the forbidden-user test to request `/api/stats` and separately assert the root remains 200.

- [ ] **Step 2: Run the targeted route tests and verify RED**

Run:

```powershell
python -m pytest tests/test_auth_router.py::test_every_declared_route_is_present_and_superuser_protected tests/test_auth_router.py::test_public_shell_loads_anonymously_while_api_rejects_unauthenticated_requests tests/test_auth_router.py::test_forbidden_users_receive_403 -q
```

Expected: failures showing the current root/static routes still include the auth dependency and return 401/403.

- [ ] **Step 3: Implement the minimal router split**

In `build_router()`:

```python
router = APIRouter(dependencies=[Depends(_promote_query_token)])
api_router = APIRouter(prefix="/api", dependencies=[Depends(auth_dependency)])
```

Keep root/static decorators on `router`. Move every API decorator to `api_router` and remove the literal `/api` prefix from those decorator paths. Preserve direct `current_user: Any = Depends(auth_dependency)` parameters on write routes that need `_actor(current_user)`. Before returning, call:

```python
router.include_router(api_router)
```

- [ ] **Step 4: Run all router tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_auth_router.py -q
```

Expected: all router tests pass, including query-token promotion, anonymous upload rejection, preview path protection, and authenticated actor attribution.

- [ ] **Step 5: Commit the server-side boundary**

```powershell
git add -- tests/test_auth_router.py source/nekro_plugin_semantic_sticker/router.py
git commit -m "fix: separate public console shell from protected api"
```

---

### Task 2: Add same-origin NA token discovery and explicit access states

**Files:**
- Modify: `tests/test_web_panel.py:24-69`
- Modify: `tests/test_web_assets.py`
- Modify: `source/nekro_plugin_semantic_sticker/web/index.html:10-120`
- Modify: `source/nekro_plugin_semantic_sticker/web/style.css`
- Modify: `source/nekro_plugin_semantic_sticker/web/app.js:1-85, 87-150, 222-229, 671-681`

**Interfaces:**
- Consumes: NA same-origin storage record `localStorage["auth-storage"]` with shape `{ "state": { "token": string | null } }`.
- Produces: `normalizeToken(value)`, `readNaStoredToken()`, `resolveInitialToken()`, `api(path, options, authOptions)`, `showWorkspace()`, and `showAccessState(kind)` in the inline script.

- [ ] **Step 1: Write failing static contract tests for token sources and access UI**

Extend the required control list with:

```python
"access-gate", "access-title", "access-message", "login-link", "workspace"
```

Replace the old assertion that forbids `localStorage` with assertions that require:

```python
assert 'const NA_AUTH_STORAGE_KEY = "auth-storage"' in script
assert 'localStorage.getItem(NA_AUTH_STORAGE_KEY)' in script
assert "JSON.parse" in script
assert ".state" in script and ".token" in script
assert 'sessionStorage.getItem("na_console_token")' in script
assert 'sessionStorage.setItem("na_console_token"' in script
assert 'new URLSearchParams(location.search)' in script
assert 'params.delete("token")' in script
```

Add contract tests requiring distinct Chinese 401/403 access copy, a login link whose `href` is `/#/login`, a `retryOnUnauthorized` guard, exactly one raw `fetch(` call, and an initialization call equivalent to:

```javascript
await loadStats({retryOnUnauthorized: true});
```

Also assert that write-action function sections do not contain `retryOnUnauthorized: true`.

- [ ] **Step 2: Run targeted web tests and verify RED**

Run:

```powershell
python -m pytest tests/test_web_panel.py tests/test_web_assets.py -q
```

Expected: failures for missing access-gate elements, missing NA `auth-storage` parsing, identical 401/403 copy, and missing bounded auth recovery.

- [ ] **Step 3: Add the access-gate markup and styling**

In `index.html`, add a visible initial `#access-gate` section with title/message elements and a hidden `#login-link` pointing to `/#/login`. Add `id="workspace"` and `hidden` to the management workspace, and keep the metadata panel hidden until authentication succeeds.

In `style.css`, add explicit `[hidden] { display: none !important; }` and an `.access-gate` card that spans the application grid, remains readable on mobile, and reuses existing color, radius, focus, and button tokens.

- [ ] **Step 4: Implement safe token resolution**

In `app.js`:

1. Define constants for the session and NA storage keys.
2. Normalize only non-empty string tokens and strip an optional case-insensitive `Bearer ` prefix.
3. Parse `auth-storage` inside `try/catch`; accept only `parsed.state.token` as a string.
4. Resolve query, session, then NA storage in that exact order.
5. Copy the selected token into session storage when possible.
6. Delete only `token` from the visible URL while preserving other query parameters and the hash.
7. Keep the selected token only in module memory and session storage; never render it.

- [ ] **Step 5: Implement bounded authentication recovery and UI transitions**

Keep a single low-level `fetch()` call. `api()` may retry once only when `authOptions.retryOnUnauthorized === true`, the first response is 401, and `readNaStoredToken()` returns a different non-empty Token. A final 401 clears only the plugin session token and calls `showAccessState("unauthenticated")`; 403 calls `showAccessState("forbidden")` without clearing NA storage.

Initialization must:

```javascript
if (!accessToken) {
  showAccessState("unauthenticated");
  return;
}
await loadStats({retryOnUnauthorized: true});
showWorkspace();
await loadStickers();
```

`showAccessState("unauthenticated")` displays the login button and login copy. `showAccessState("forbidden")` hides the button and displays the superuser requirement. Both hide/inert the management workspace and metadata panel.

- [ ] **Step 6: Run web tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_web_panel.py tests/test_web_assets.py -q
```

Expected: all web asset and UI contract tests pass.

- [ ] **Step 7: Run the full suite before committing**

```powershell
python -m pytest tests/ -q
```

Expected: all tests pass without warnings or errors.

- [ ] **Step 8: Commit the browser login-state integration**

```powershell
git add -- tests/test_web_panel.py tests/test_web_assets.py source/nekro_plugin_semantic_sticker/web/index.html source/nekro_plugin_semantic_sticker/web/style.css source/nekro_plugin_semantic_sticker/web/app.js
git commit -m "fix: reuse same-origin na login for bare console access"
```

---

### Task 3: Document access behavior and operational boundaries

**Files:**
- Modify: `tests/test_web_assets.py`
- Modify: `README.md:75-103`
- Modify: `source/nekro_plugin_semantic_sticker/README.md`
- Modify: `CHANGELOG.md:1-8`

**Interfaces:**
- Consumes: the implemented public-shell/protected-API behavior and NA storage compatibility contract.
- Produces: user-facing instructions for naked URL access, same-origin requirements, NUP compatibility, 401/403 behavior, and no-NA-code scope.

- [ ] **Step 1: Add failing documentation assertions**

Require the public and detailed README text to include:

- `auth-storage`;
- same-origin scheme/host/port requirement;
- direct `/plugins/Akiyo.semantic_sticker/` access after logging into NA;
- NUP `?token=` compatibility and URL cleanup;
- static shell being public while every `/api/*` remains super-admin protected;
- 401 login prompt and 403 permission distinction.

Require `CHANGELOG.md` to contain an unreleased entry for bare-address console authentication.

- [ ] **Step 2: Run the documentation tests and verify RED**

```powershell
python -m pytest tests/test_web_assets.py -q
```

Expected: failures identifying the missing authentication documentation.

- [ ] **Step 3: Update README files and changelog**

Document the exact behavior without copying any real Token, domain, IP, account, or instance identifier. State that cross-origin deployments cannot read NA local storage and should continue using an authorized `?token=` integration or align origins.

- [ ] **Step 4: Run documentation and full tests and verify GREEN**

```powershell
python -m pytest tests/test_web_assets.py -q
python -m pytest tests/ -q
```

Expected: both commands pass.

- [ ] **Step 5: Commit documentation**

```powershell
git add -- tests/test_web_assets.py README.md source/nekro_plugin_semantic_sticker/README.md CHANGELOG.md
git commit -m "docs: explain bare console authentication"
```

---

### Task 4: Final verification and review

**Files:**
- Verify only: all changed files

**Interfaces:**
- Consumes: Tasks 1-3 commits.
- Produces: evidence that the branch meets the approved design without deployment.

- [ ] **Step 1: Verify repository and changed-file scope**

```powershell
git status --short --branch
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

Expected: only plugin source/tests/docs and the approved design/plan files are changed; no NA reference checkout or credentials are included.

- [ ] **Step 2: Run complete automated verification**

```powershell
python -m pytest tests/
python -m compileall -q source tests
```

Expected: exit code 0 for both commands.

- [ ] **Step 3: Inspect authentication invariants**

Confirm from the final diff that:

- root/static routes do not depend on `auth_dependency`;
- every `/api/*` route does depend on it;
- only one raw browser `fetch()` exists;
- no Token value is rendered or logged;
- 401 clears only plugin session state;
- 403 does not clear NA login state;
- write requests cannot opt into automatic auth replay.

- [ ] **Step 4: Perform an independent code review**

Review `origin/main...HEAD` against `docs/superpowers/specs/2026-08-13-semantic-sticker-bare-console-auth-design.md`. Fix every Critical or Important issue, rerun the complete verification, and commit fixes separately.

- [ ] **Step 5: Report local completion without deployment claims**

Report branch, commits, exact verification output, changed paths, and remaining browser/production acceptance steps. Explicitly state that no NA instance was modified, deployed, or restarted.
