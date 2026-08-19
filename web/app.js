"use strict";

const API_ROOT = new URL("./api/", location.href).pathname;
const SESSION_TOKEN_KEY = "na_console_token";
const NA_AUTH_STORAGE_KEY = "auth-storage";
const MAX_POLL_ATTEMPTS = 9;
const POLL_BACKOFF_MS = 900;
const PAGE_SIZE = 48;
const CATEGORIES = [
  "confusion", "happiness", "speechlessness", "anger", "sadness", "surprise",
  "agreement", "refusal", "comfort", "shyness", "mockery", "celebration",
  "neutral reaction", "other"
];
const CATEGORY_LABELS = {
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
  "other": "其他"
};
const STATE_LABELS = {
  "pending": "等待处理",
  "analyzing": "正在分析",
  "indexing": "正在建立向量",
  "active": "可用",
  "failed": "处理失败",
  "retry_pending": "等待重试",
  "deleting": "正在删除",
  "deleted": "已删除"
};
const SAFETY_LABELS = {
  "safe": "安全",
  "unsafe": "不安全",
  "disallowed": "禁止使用"
};
const HTTP_STATUS_MESSAGES = {
  400: "请求参数不正确，请检查后重试。",
  401: "登录状态已失效，请重新登录 NekroAgent 后重试。",
  403: "当前登录账户无法执行此请求，请稍后重试。",
  404: "请求的资源不存在。",
  409: "当前数据状态已变化，请刷新后重试。",
  413: "上传请求超过服务器限制，请分批上传，或调大 nginx client_max_body_size 后重试。",
  422: "提交内容不符合要求，请检查后重试。",
  429: "操作过于频繁，请稍后重试。",
  500: "服务器处理失败，请稍后重试。",
  502: "服务暂时不可用，请稍后重试。",
  503: "服务暂时不可用，请稍后重试。",
  504: "服务响应超时，请稍后重试。"
};
// 单次上传请求的总量上限，默认与插件 MAX_UPLOAD_BYTES（10MB）一致。
// 注意：nginx 默认 client_max_body_size 只有 1MB，部署时需调大（见 README 部署要求）；
// 若服务器限制与本常量不同，请同步修改，使前端提示与实际限制一致。
const DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024;
const mobileDrawer = window.matchMedia("(max-width: 860px)");

function normalizeToken(value) {
  if (typeof value !== "string") return null;
  const token = value.replace(/^Bearer\s+/i, "").trim();
  return token || null;
}

function readSessionToken() {
  try {
    return normalizeToken(sessionStorage.getItem(SESSION_TOKEN_KEY));
  } catch (_error) {
    return null;
  }
}

function storeSessionToken(token) {
  try {
    if (token) sessionStorage.setItem(SESSION_TOKEN_KEY, token);
    else sessionStorage.removeItem(SESSION_TOKEN_KEY);
  } catch (_error) {
    // Session storage can be unavailable in privacy-restricted browser contexts.
  }
}

function readNaStoredToken() {
  try {
    const raw = localStorage.getItem(NA_AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return normalizeToken(parsed && parsed.state && parsed.state.token);
  } catch (_error) {
    return null;
  }
}

function takeQueryToken() {
  const params = new URLSearchParams(location.search);
  const tokenPresent = params.has("token");
  const token = normalizeToken(params.get("token"));
  if (tokenPresent) {
    params.delete("token");
    const query = params.toString();
    const cleanUrl = location.pathname + (query ? "?" + query : "") + location.hash;
    history.replaceState({}, document.title, cleanUrl);
  }
  return token;
}

function resolveInitialToken() {
  const token = takeQueryToken() || readSessionToken() || readNaStoredToken();
  if (token) storeSessionToken(token);
  return token;
}

let accessToken = resolveInitialToken();

async function fetchWithAccessToken(path, options) {
  const headers = new Headers(options.headers || {});
  if (accessToken) headers.set("Authorization", "Bearer " + accessToken);
  return fetch(path, {...options, headers});
}

async function api(path, options = {}, authOptions = {}) {
  const requestMethod = String(options.method || "GET").toUpperCase();
  let response = await fetchWithAccessToken(path, options);
  if (
    response.status === 401
    && requestMethod === "GET"
    && authOptions.retryOnUnauthorized === true
  ) {
    const freshToken = readNaStoredToken();
    if (freshToken && freshToken !== accessToken) {
      accessToken = freshToken;
      storeSessionToken(accessToken);
      response = await fetchWithAccessToken(path, options);
    }
  }
  if (response.status === 401) {
    accessToken = null;
    storeSessionToken(null);
    showAccessState("unauthenticated");
    throw new HttpError(response.status);
  }
  if (response.status === 403) throw new HttpError(response.status);
  if (!response.ok) throw new HttpError(response.status);
  return response;
}

class HttpError extends Error {
  constructor(status) {
    super("请求失败");
    this.name = "HttpError";
    this.status = status;
    this.userMessage = httpStatusMessage(status);
  }
}

const elements = {
  accessGate: document.getElementById("access-gate"),
  accessTitle: document.getElementById("access-title"),
  accessMessage: document.getElementById("access-message"),
  loginLink: document.getElementById("login-link"),
  workspace: document.getElementById("workspace"),
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  uploadQueue: document.getElementById("upload-queue"),
  grid: document.getElementById("sticker-grid"),
  status: document.getElementById("status-region"),
  selectionCount: document.getElementById("selection-count"),
  batchDelete: document.getElementById("batch-delete"),
  fullReindex: document.getElementById("full-reindex"),
  filterForm: document.getElementById("filter-form"),
  filterQuery: document.getElementById("filter-query"),
  filterCategory: document.getElementById("filter-category"),
  filterTags: document.getElementById("filter-tags"),
  filterState: document.getElementById("filter-state"),
  filterTime: document.getElementById("filter-time"),
  previousPage: document.getElementById("previous-page"),
  nextPage: document.getElementById("next-page"),
  pageSummary: document.getElementById("page-summary"),
  panel: document.getElementById("metadata-panel"),
  panelEmpty: document.getElementById("metadata-empty"),
  metadataForm: document.getElementById("metadata-form"),
  closeMetadata: document.getElementById("close-metadata"),
  metadataImage: document.getElementById("metadata-image"),
  metadataState: document.getElementById("metadata-state"),
  metadataSafety: document.getElementById("metadata-safety"),
  metadataDescription: document.getElementById("metadata-description"),
  metadataCategory: document.getElementById("metadata-category"),
  metadataEmotionTags: document.getElementById("metadata-emotion-tags"),
  metadataSceneTags: document.getElementById("metadata-scene-tags"),
  metadataOcr: document.getElementById("metadata-ocr"),
  metadataSuitable: document.getElementById("metadata-suitable"),
  metadataUnsuitable: document.getElementById("metadata-unsuitable"),
  metadataReason: document.getElementById("metadata-reason"),
  saveMetadata: document.getElementById("save-metadata"),
  reanalyze: document.getElementById("reanalyze-sticker"),
  reindex: document.getElementById("reindex-sticker"),
  deleteSticker: document.getElementById("delete-sticker"),
  statsTotal: document.getElementById("stats-total"),
  statsIndexed: document.getElementById("stats-indexed"),
  statsStorage: document.getElementById("stats-storage"),
  statsFailures: document.getElementById("stats-failures")
};

const panelState = {
  items: [],
  total: 0,
  offset: 0,
  selected: new Set(),
  activeRecord: null,
  previewUrls: new Map(),
  metadataPreviewUrl: null,
  returnFocus: null,
  loading: false
};

function apiUrl(path) {
  return API_ROOT + path.replace(/^\//, "");
}

function setManagementVisibility(visible) {
  elements.workspace.hidden = !visible;
  elements.panel.hidden = !visible;
  elements.workspace.inert = !visible;
  elements.panel.inert = !visible;
}

function showWorkspace() {
  elements.accessGate.hidden = true;
  elements.loginLink.hidden = true;
  setManagementVisibility(true);
}

function showAccessState(kind) {
  setManagementVisibility(false);
  elements.accessGate.hidden = false;
  elements.accessTitle.textContent = "请先登录 NekroAgent";
  elements.accessMessage.textContent = "请先登录 NekroAgent 后再访问此控制台。登录后返回并刷新当前裸地址即可。";
  elements.loginLink.hidden = false;
}

function httpStatusMessage(status) {
  if (HTTP_STATUS_MESSAGES[status]) return HTTP_STATUS_MESSAGES[status];
  if (status >= 500) return "服务器处理失败，请稍后重试。";
  return "请求未能完成，请稍后重试。";
}

function userErrorMessage(error, fallback) {
  return error instanceof HttpError ? error.userMessage : fallback;
}

function categoryLabel(category) {
  return CATEGORY_LABELS[category] || "其他";
}

function stateLabel(state) {
  return STATE_LABELS[state] || "未知状态";
}

function safetyLabel(safety) {
  return SAFETY_LABELS[safety] || "未知安全状态";
}

function createNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(message, tone = "neutral") {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
}

function setBusy(control, busy, busyLabel) {
  if (!control) return;
  if (!control.dataset.idleLabel) control.dataset.idleLabel = control.textContent;
  control.disabled = busy;
  control.setAttribute("aria-busy", String(busy));
  control.textContent = busy ? busyLabel : control.dataset.idleLabel;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return (bytes / (1024 ** index)).toFixed(index ? 1 : 0) + " " + units[index];
}

function listFromComma(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function listFromLines(value) {
  return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function releasePreviewUrls() {
  panelState.previewUrls.forEach((url) => URL.revokeObjectURL(url));
  panelState.previewUrls.clear();
}

function updateSelectionUi() {
  const count = panelState.selected.size;
  elements.selectionCount.textContent = "已选择 " + count + " 项";
  elements.batchDelete.disabled = count === 0;
}

async function loadStats(authOptions = {}) {
  const response = await api(apiUrl("stats"), {}, authOptions);
  const stats = await response.json();
  elements.statsTotal.textContent = String(stats.total);
  elements.statsIndexed.textContent = String(stats.indexed_count);
  elements.statsStorage.textContent = formatBytes(stats.storage_bytes);
  elements.statsFailures.textContent = String(stats.failure_count);
}

function buildFilterQuery() {
  const params = new URLSearchParams();
  const query = elements.filterQuery.value.trim();
  const category = elements.filterCategory.value;
  const tags = elements.filterTags.value.trim();
  const stateValue = elements.filterState.value;
  if (query) params.set("query", query);
  if (category) params.set("category", category);
  if (tags) params.set("tags", tags);
  if (stateValue) params.set("state", stateValue);
  const age = {day: 1, week: 7, month: 30}[elements.filterTime.value];
  if (age) params.set("created_from", new Date(Date.now() - age * 86400000).toISOString());
  params.set("offset", String(panelState.offset));
  params.set("limit", String(PAGE_SIZE));
  return params.toString();
}
async function loadPreview(image, record, fullSize = false) {
  const query = fullSize || record.animated ? "view=content" : "view=thumbnail";
  const response = await api(apiUrl("stickers/" + encodeURIComponent(record.id) + "?" + query));
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  image.src = objectUrl;
  return objectUrl;
}

function stateBadge(record) {
  const badge = createNode("span", "state-badge", stateLabel(record.state));
  badge.dataset.state = record.state;
  return badge;
}

function renderEmpty() {
  const empty = createNode("div", "empty-state");
  const copy = createNode("div");
  copy.append(createNode("strong", "", "暂无符合条件的表情包"));
  copy.append(createNode("p", "", "上传一张测试表情包，或调整分类、标签、状态和时间筛选。"));
  empty.append(copy);
  elements.grid.append(empty);
}

function syncMetadataPanelMode() {
  const isOpen = elements.panel.classList.contains("is-open");
  elements.panel.setAttribute("aria-modal", String(mobileDrawer.matches && isOpen));
  if (mobileDrawer.matches) {
    elements.panel.inert = !isOpen;
    elements.panel.setAttribute("aria-hidden", String(!isOpen));
  } else {
    elements.panel.inert = false;
    elements.panel.removeAttribute("aria-hidden");
  }
}

function openMetadata(record, trigger) {
  panelState.activeRecord = record;
  panelState.returnFocus = trigger || document.activeElement;
  elements.panelEmpty.hidden = true;
  elements.metadataForm.hidden = false;
  elements.panel.classList.add("is-open");
  syncMetadataPanelMode();
  if (mobileDrawer.matches) requestAnimationFrame(() => elements.closeMetadata.focus());
  elements.metadataState.value = stateLabel(record.state);
  elements.metadataSafety.value = safetyLabel(record.safety);
  elements.metadataDescription.value = record.description || "";
  elements.metadataCategory.value = record.primary_category || "other";
  elements.metadataEmotionTags.value = (record.emotion_tags || []).join(", ");
  elements.metadataSceneTags.value = (record.scene_tags || []).join(", ");
  elements.metadataOcr.value = record.ocr_text || "";
  elements.metadataSuitable.value = (record.suitable_scenarios || []).join("\n");
  elements.metadataUnsuitable.value = (record.unsuitable_scenarios || []).join("\n");
  elements.metadataReason.value = "";
  if (panelState.metadataPreviewUrl) URL.revokeObjectURL(panelState.metadataPreviewUrl);
  elements.metadataImage.removeAttribute("src");
  loadPreview(elements.metadataImage, record, true)
    .then((url) => { panelState.metadataPreviewUrl = url; })
    .catch(() => { elements.metadataImage.alt = "预览加载失败"; });
}

function closeMetadata() {
  const returnFocus = panelState.returnFocus;
  panelState.activeRecord = null;
  panelState.returnFocus = null;
  elements.panel.classList.remove("is-open");
  elements.metadataForm.hidden = true;
  elements.panelEmpty.hidden = false;
  syncMetadataPanelMode();
  if (panelState.metadataPreviewUrl) {
    URL.revokeObjectURL(panelState.metadataPreviewUrl);
    panelState.metadataPreviewUrl = null;
  }
  if (mobileDrawer.matches && returnFocus && returnFocus.isConnected) {
    requestAnimationFrame(() => returnFocus.focus());
  }
}

function createStickerCard(record) {
  const card = createNode("article", "sticker-card");
  card.dataset.id = record.id;
  if (panelState.selected.has(record.id)) card.classList.add("is-selected");

  const checkbox = createNode("input", "card-select");
  checkbox.type = "checkbox";
  checkbox.checked = panelState.selected.has(record.id);
  checkbox.setAttribute("aria-label", "选择 " + (record.description || record.id));
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) panelState.selected.add(record.id);
    else panelState.selected.delete(record.id);
    card.classList.toggle("is-selected", checkbox.checked);
    updateSelectionUi();
  });

  const preview = createNode("button", "card-preview");
  preview.type = "button";
  preview.setAttribute("aria-label", "检查 " + (record.description || record.id));
  const image = createNode("img");
  image.alt = record.description || "表情包预览";
  preview.append(image);
  preview.addEventListener("click", () => openMetadata(record, preview));

  const body = createNode("div", "card-body");
  body.append(createNode("div", "card-title", record.description || "等待模型描述"));
  const meta = createNode("div", "card-meta");
  meta.append(stateBadge(record));
  meta.append(createNode("span", "tag", categoryLabel(record.primary_category)));
  body.append(meta);
  const tags = createNode("div", "tag-row");
  (record.emotion_tags || []).slice(0, 2).forEach((tag) => tags.append(createNode("span", "tag", tag)));
  if (tags.childElementCount) body.append(tags);

  card.append(checkbox, preview, body);
  loadPreview(image, record)
    .then((url) => panelState.previewUrls.set(record.id, url))
    .catch(() => { image.alt = "缩略图加载失败"; });
  return card;
}

function renderStickers() {
  releasePreviewUrls();
  clearNode(elements.grid);
  elements.grid.setAttribute("aria-busy", "false");
  if (!panelState.items.length) {
    renderEmpty();
  } else {
    panelState.items.forEach((record) => elements.grid.append(createStickerCard(record)));
  }
  const currentPage = Math.floor(panelState.offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(panelState.total / PAGE_SIZE));
  elements.pageSummary.textContent = "第 " + currentPage + " / " + totalPages + " 页";
  elements.previousPage.disabled = panelState.offset === 0;
  elements.nextPage.disabled = panelState.offset + PAGE_SIZE >= panelState.total;
  updateSelectionUi();
}

async function loadStickers() {
  if (panelState.loading) return;
  panelState.loading = true;
  elements.grid.setAttribute("aria-busy", "true");
  setStatus("正在读取表情包库…");
  try {
    const response = await api(apiUrl("stickers?" + buildFilterQuery()));
    const page = await response.json();
    panelState.items = page.items || [];
    panelState.total = page.total || 0;
    renderStickers();
    setStatus("已加载 " + panelState.items.length + " 项，共 " + panelState.total + " 项。", "success");
  } catch (error) {
    clearNode(elements.grid);
    const failure = createNode("div", "empty-state");
    failure.append(createNode("strong", "", "无法加载表情包库"));
    failure.append(createNode("p", "", userErrorMessage(error, "无法连接服务，请稍后重试。")));
    elements.grid.append(failure);
    elements.grid.setAttribute("aria-busy", "false");
    setStatus(userErrorMessage(error, "加载失败，请检查登录状态或稍后重试。"), "error");
  } finally {
    panelState.loading = false;
  }
}

async function pollSticker(stickerId, queueItem, attempt = 0) {
  if (attempt >= MAX_POLL_ATTEMPTS) {
    queueItem.dataset.status = "failed";
    queueItem.lastChild.textContent = "等待超时，可在库中重试";
    return;
  }
  const delay = Math.min(20000, POLL_BACKOFF_MS * (2 ** attempt));
  await sleep(delay);
  try {
    const response = await api(apiUrl("stickers/" + encodeURIComponent(stickerId) + "?view=metadata"));
    const record = await response.json();
    queueItem.dataset.status = record.state;
    queueItem.lastChild.textContent = stateLabel(record.state);
    if (["pending", "analyzing", "indexing", "retry_pending"].includes(record.state)) {
      await pollSticker(stickerId, queueItem, attempt + 1);
    } else {
      await Promise.all([loadStickers(), loadStats()]);
    }
  } catch (error) {
    if (error instanceof HttpError && (error.status === 401 || error.status === 403)) return;
    await pollSticker(stickerId, queueItem, attempt + 1);
  }
}
function queueRow(file) {
  const row = createNode("div", "queue-item");
  row.dataset.status = "pending";
  row.append(createNode("strong", "", file.name));
  row.append(createNode("span", "", "等待上传"));
  elements.uploadQueue.prepend(row);
  return row;
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const rows = files.map(queueRow);
  const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
  if (totalBytes > DEFAULT_MAX_REQUEST_BYTES) {
    rows.forEach((row) => {
      row.dataset.status = "failed";
      row.lastChild.textContent = "上传失败，请稍后重试";
    });
    setStatus(
      "本次上传总大小 " + formatBytes(totalBytes) + " 超过单次请求限制（" + formatBytes(DEFAULT_MAX_REQUEST_BYTES) + "），请分批上传；如服务器已调大 nginx client_max_body_size，请联系管理员同步调整前端限制。",
      "error"
    );
    elements.fileInput.value = "";
    return;
  }
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  setStatus("正在提交 " + files.length + " 个文件…");
  try {
    const response = await api(apiUrl("stickers"), {method: "POST", body: formData});
    const outcomes = await response.json();
    let successfulCount = 0;
    let failedCount = 0;
    rows.forEach((row, index) => {
      const outcome = outcomes[index];
      if (!outcome || !outcome.ok) {
        failedCount += 1;
        row.dataset.status = "failed";
        row.lastChild.textContent = "上传失败，请稍后重试";
        return;
      }
      successfulCount += 1;
      row.dataset.status = outcome.record.state;
      row.lastChild.textContent = outcome.duplicate ? "内容已存在" : "已入队";
      if (!outcome.duplicate && outcome.job) pollSticker(outcome.record.id, row);
    });
    setStatus("上传完成：成功 " + successfulCount + "，失败 " + failedCount + "。", failedCount ? "error" : "success");
    if (successfulCount) await Promise.all([loadStickers(), loadStats()]);
  } catch (error) {
    rows.forEach((row) => {
      row.dataset.status = "failed";
      row.lastChild.textContent = "上传失败，请稍后重试";
    });
    setStatus(userErrorMessage(error, "上传失败，请稍后重试。"), "error");
  } finally {
    elements.fileInput.value = "";
  }
}

async function saveMetadata(event) {
  event.preventDefault();
  if (!panelState.activeRecord) return;
  const reason = elements.metadataReason.value.trim();
  if (!reason) {
    elements.metadataReason.focus();
    setStatus("请填写修订原因。", "error");
    return;
  }
  const payload = {
    description: elements.metadataDescription.value.trim(),
    primary_category: elements.metadataCategory.value,
    emotion_tags: listFromComma(elements.metadataEmotionTags.value),
    scene_tags: listFromComma(elements.metadataSceneTags.value),
    ocr_text: elements.metadataOcr.value.trim(),
    suitable_scenarios: listFromLines(elements.metadataSuitable.value),
    unsuitable_scenarios: listFromLines(elements.metadataUnsuitable.value),
    reason
  };
  setBusy(elements.saveMetadata, true, "保存中…");
  try {
    const response = await api(apiUrl("stickers/" + encodeURIComponent(panelState.activeRecord.id)), {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const updated = await response.json();
    panelState.activeRecord = updated;
    elements.metadataReason.value = "";
    setStatus("元数据已保存；需要时可单独重建向量。", "success");
    await loadStickers();
  } catch (error) {
    setStatus(userErrorMessage(error, "保存失败，请稍后重试。"), "error");
  } finally {
    setBusy(elements.saveMetadata, false, "保存中…");
  }
}

async function runRecordAction(action, control, busyLabel) {
  if (!panelState.activeRecord) return;
  setBusy(control, true, busyLabel);
  try {
    const response = await api(apiUrl("stickers/" + encodeURIComponent(panelState.activeRecord.id) + "/" + action), {method: "POST"});
    const result = await response.json();
    setStatus(action === "reanalyze" ? "重新分析任务已入队。" : "向量已重建。", "success");
    if (action === "reanalyze" && result.id) {
      const synthetic = new File([], panelState.activeRecord.description || panelState.activeRecord.id);
      const row = queueRow(synthetic);
      row.lastChild.textContent = "重新分析已入队";
      pollSticker(panelState.activeRecord.id, row);
    }
    await Promise.all([loadStickers(), loadStats()]);
  } catch (error) {
    setStatus(userErrorMessage(error, "操作失败，请稍后重试。"), "error");
  } finally {
    setBusy(control, false, busyLabel);
  }
}

async function deleteActiveSticker() {
  if (!panelState.activeRecord) return;
  if (!confirm("确认删除这张表情包？原图、缩略图和向量都会移除，审计记录仍保留。")) return;
  setBusy(elements.deleteSticker, true, "删除中…");
  try {
    await api(apiUrl("stickers/" + encodeURIComponent(panelState.activeRecord.id)), {method: "DELETE"});
    panelState.selected.delete(panelState.activeRecord.id);
    closeMetadata();
    setStatus("表情包已删除。", "success");
    await Promise.all([loadStickers(), loadStats()]);
  } catch (error) {
    setStatus(userErrorMessage(error, "删除失败，请稍后重试。"), "error");
  } finally {
    setBusy(elements.deleteSticker, false, "删除中…");
  }
}

async function deleteSelected() {
  const ids = Array.from(panelState.selected);
  if (!ids.length) return;
  if (!confirm("确认批量删除已选择的 " + ids.length + " 张表情包？")) return;
  setBusy(elements.batchDelete, true, "批量删除中…");
  try {
    const response = await api(apiUrl("stickers/batch-delete"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sticker_ids: ids})
    });
    const result = await response.json();
    panelState.selected.clear();
    setStatus("批量删除完成：成功 " + result.deleted + "，失败 " + result.failed_ids.length + "。", result.failed_ids.length ? "error" : "success");
    await Promise.all([loadStickers(), loadStats()]);
  } catch (error) {
    setStatus(userErrorMessage(error, "批量删除失败，请稍后重试。"), "error");
  } finally {
    setBusy(elements.batchDelete, false, "批量删除中…");
    updateSelectionUi();
  }
}

async function fullReindex() {
  if (!confirm("确认对所有可用且安全的表情包重建向量？此操作不会再次调用视觉模型。")) return;
  setBusy(elements.fullReindex, true, "重建中…");
  try {
    const response = await api(apiUrl("reindex"), {method: "POST"});
    const result = await response.json();
    setStatus("全量重建完成：成功 " + result.indexed + " / " + result.requested + "。", result.failed_ids.length ? "error" : "success");
    await Promise.all([loadStickers(), loadStats()]);
  } catch (error) {
    setStatus(userErrorMessage(error, "全量重建失败，请稍后重试。"), "error");
  } finally {
    setBusy(elements.fullReindex, false, "重建中…");
  }
}

function populateCategories() {
  CATEGORIES.forEach((category) => {
    const filterOption = createNode("option", "", categoryLabel(category));
    filterOption.value = category;
    elements.filterCategory.append(filterOption);
    const editorOption = createNode("option", "", categoryLabel(category));
    editorOption.value = category;
    elements.metadataCategory.append(editorOption);
  });
}

function bindEvents() {
  elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files));
  elements.dropZone.addEventListener("click", (event) => {
    if (event.target !== elements.fileInput && event.target.htmlFor !== "file-input") elements.fileInput.click();
  });
  elements.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragging");
  }));
  elements.dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
  elements.filterForm.addEventListener("submit", (event) => {
    event.preventDefault();
    panelState.offset = 0;
    loadStickers();
  });
  elements.previousPage.addEventListener("click", () => {
    panelState.offset = Math.max(0, panelState.offset - PAGE_SIZE);
    loadStickers();
  });
  elements.nextPage.addEventListener("click", () => {
    panelState.offset += PAGE_SIZE;
    loadStickers();
  });
  elements.metadataForm.addEventListener("submit", saveMetadata);
  elements.closeMetadata.addEventListener("click", closeMetadata);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && mobileDrawer.matches && elements.panel.classList.contains("is-open")) {
      event.preventDefault();
      closeMetadata();
    }
  });
  if (mobileDrawer.addEventListener) mobileDrawer.addEventListener("change", syncMetadataPanelMode);
  else mobileDrawer.addListener(syncMetadataPanelMode);
  elements.reanalyze.addEventListener("click", () => {
    if (confirm("确认重新调用视觉模型分析这张表情包？")) runRecordAction("reanalyze", elements.reanalyze, "入队中…");
  });
  elements.reindex.addEventListener("click", () => runRecordAction("reindex", elements.reindex, "重建中…"));
  elements.deleteSticker.addEventListener("click", deleteActiveSticker);
  elements.batchDelete.addEventListener("click", deleteSelected);
  elements.fullReindex.addEventListener("click", fullReindex);
  window.addEventListener("beforeunload", () => {
    releasePreviewUrls();
    if (panelState.metadataPreviewUrl) URL.revokeObjectURL(panelState.metadataPreviewUrl);
  });
}

async function initialize() {
  populateCategories();
  bindEvents();
  syncMetadataPanelMode();
  updateSelectionUi();
  if (!accessToken) {
    showAccessState("unauthenticated");
    return;
  }
  await loadStats({retryOnUnauthorized: true});
  showWorkspace();
  await loadStickers();
}

initialize().catch((error) => {
  if (error instanceof HttpError && error.status === 401) return;
  setStatus(userErrorMessage(error, "控制台初始化失败，请刷新页面重试。"), "error");
});
