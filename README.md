# NekroAgent 语义表情包插件

面向 NekroAgent 的实例级语义表情包库。插件提供网页上传与管理、视觉分析、语义检索、Agent 按需保存、用户主动保存，以及 Bot 被戳时直接发送表情包等能力。

- 插件 ID：`Akiyo.semantic_sticker`
- 当前版本：`1.2.2`
- 源码目录：`source/nekro_plugin_semantic_sticker`
- 详细操作手册：[插件用户与管理员手册](source/nekro_plugin_semantic_sticker/README.md)

## 功能

- **管理控制台**：批量上传 PNG、JPEG、GIF、WebP，查看分析队列，编辑标签，重试分析，重建向量和删除资产。
- **视觉分析**：为图片生成描述、分类、情绪标签、场景标签、OCR 文本和安全状态。
- **语义检索**：通过 Embedding 与 Qdrant 按描述、情绪和使用场景检索表情包。
- **聊天保存**：用户明确要求时保存当前消息或被引用消息中的图片；可选允许 Agent 自主收集。
- **戳一戳回复**：收到 OneBot V11 poke 事件后直接检索并发送一张表情包，不经过主 Agent 或 LLM。

## 兼容性

| 项目 | 支持范围 |
| --- | --- |
| NekroAgent | **官方 NekroAgent v2.3.3**（已核对关键 API） |
| Adapter | 仅 `onebot_v11` |
| Python | 跟随 NekroAgent v2.3.3 运行环境 |
| 向量数据库 | NekroAgent 已配置且可用的 Qdrant |
| 分析接口 | OpenAI-compatible Chat Completions |
| Embedding 接口 | OpenAI-compatible Embeddings |

插件不依赖任何实例专属 NekroAgent 补丁，也不通过覆盖 NekroAgent 核心文件工作。当前版本使用了 `gen_openai_embeddings`、`get_current_super_user`、`DBChatMessage`、`OsEnv` 等部分 NekroAgent 内部接口；未来升级 NekroAgent 时，应先在测试实例完成回归，不应据此承诺 v2.2.x 或更早版本兼容。

## 前置条件

### 分析模型

`ANALYSIS_MODEL_GROUP` 对应的服务必须：

- 提供 OpenAI-compatible Chat Completions；
- 支持视觉输入与图片 `data URL`；
- 支持 `response_format={"type":"json_object"}`，即 JSON object 响应格式；
- 可由 NekroAgent 容器直接访问。

当前实现不会把 NekroAgent 模型组的 `CHAT_PROXY` 应用到该请求，因此仅在宿主机可达、但容器无法直连的模型端点不能正常工作。

### Embedding 与 Qdrant

`EMBEDDING_MODEL_GROUP` 对应的服务必须兼容 OpenAI Embeddings，并支持或容忍请求中的 `dimensions` 参数。模型实际输出长度必须严格等于插件配置的 `VECTOR_DIMENSION`。

插件使用固定 Qdrant collection：`Akiyo.semantic_sticker`。多个 NekroAgent 实例共享同一个 Qdrant 时会共享该 collection 中的数据；如果各实例的 `VECTOR_DIMENSION` 不同，还可能产生 collection 维度冲突。需要数据隔离时，请为实例使用独立 Qdrant。

## 安装

1. 下载或克隆本仓库。
2. 将 `source/nekro_plugin_semantic_sticker` 完整复制到目标 NekroAgent 的插件包目录。
3. 在 NekroAgent 插件管理中启用 `Akiyo.semantic_sticker`。
4. 配置分析模型组、Embedding 模型组，并确认 `VECTOR_DIMENSION` 与模型输出一致。
5. 启用或更新后，对目标 NekroAgent 执行一次完整重启，再检查插件加载日志与管理页面。

不要只复制单个 Python 文件；`web/` 静态资源和同目录模块都属于插件的一部分。

## 基础配置

| 配置项 | 说明 |
| --- | --- |
| `ANALYSIS_MODEL_GROUP` | 视觉分析所用聊天模型组 |
| `EMBEDDING_MODEL_GROUP` | 语义检索所用 Embedding 模型组 |
| `VECTOR_DIMENSION` | 必须与 Embedding 输出长度一致，默认 1536 |
| `MAX_UPLOAD_BYTES` | 单个图片允许的最大字节数，默认 10 MiB |
| `AUTO_COLLECT_ENABLED` | 是否允许 Agent 自主收集已确认的表情包 |
| `STRICT_EMOTION_COLLECT` | 是否拒绝截图、照片和无法确认的普通图片 |
| `SEMANTIC_SCORE_THRESHOLD` | 语义候选最低相似度 |
| `PHYSICAL_CHANNEL_COOLDOWN_SECONDS` | 同一物理频道发送冷却时间 |

其余配置项和管理流程见[详细手册](source/nekro_plugin_semantic_sticker/README.md)。

## 管理控制台与上传

先在 NekroAgent WebUI 登录超级管理员账户，然后可直接输入裸地址：

`/plugins/Akiyo.semantic_sticker/`

插件前端会读取同源 `localStorage["auth-storage"].state.token` 中的 NA 登录 Token。这里的“同源”要求插件页面与 NA WebUI 的 `scheme、host、port` 全部相同；跨协议、跨域名或跨端口时，浏览器不允许插件读取该登录状态。

已授权的 NUP 集成仍可使用 `?token=`；它的优先级高于插件 `sessionStorage` 和 NA `auth-storage`，页面启动后只会从地址栏清除 `token` 参数，并保留其他查询参数与 Hash。不同源部署无法读取 NA 本地存储，应继续使用受信任的 `?token=` 集成，或将 WebUI 与插件页面调整为同源。

控制台的 HTML/CSS/JavaScript 静态外壳可匿名加载，但不包含表情包、统计或配置等业务数据；所有 `/api/*` 读写、预览和管理接口仍由服务端要求 NA 超级管理员。HTTP 401 表示未登录或 Token 失效，页面会显示“前往 NA 登录”；HTTP 403 表示当前账户不是超级管理员，不会清除 NA 登录状态。

如果通过 nginx 等反向代理访问，`client_max_body_size` 必须不低于实际上传限制；否则请求会在到达插件之前以 HTTP 413 被拒绝。例如：

```nginx
client_max_body_size 50m;
```

控制台前端还会按 `web/app.js` 中的 `DEFAULT_MAX_REQUEST_BYTES` 校验单次请求总量。修改服务端限制时应同步检查该前端常量。

## 已知限制

- **仅支持 OneBot V11**：其他 adapter 未实现事件和消息图片适配。
- **戳一戳为阻断式处理**：poke matcher 使用 `block=True`。没有候选、仍在冷却或发送失败时保持静默，不回退官方 LLM。
- **热重载边界**：poke matcher 在模块导入时注册。启用、禁用或更新插件后建议完整重启目标 NekroAgent，避免热重载造成旧 matcher 残留或重复注册。
- **聊天图片必须已落盘**：保存工具依赖 NekroAgent 提供有效的图片 `local_path`；仅有远程 URL 而没有本地文件时不能保存。
- **模型组代理不生效**：如上所述，分析模型请求当前不使用 `CHAT_PROXY`。
- **内部接口可能变化**：升级 NekroAgent 前应测试插件导入、配置加载、管理 API、图片保存、向量索引、语义发送和真实 OneBot 回复。

## 安全建议

- 不要公开或转发表含 `?token=` 的管理页面 URL；查询参数可能包含管理员访问凭据。
- 不要把 API Key、模型服务凭据、NekroAgent 管理员 Token 或生产配置提交到本仓库。
- 上传内容会保存到目标 NekroAgent 的插件数据目录，并写入 Qdrant；公开部署前应自行评估内容合规、备份和数据保留策略。
- 修改生产配置前先备份，只重启目标 NekroAgent，不要无关重启 PostgreSQL、Qdrant、OneBot 网关或其他实例。

## 开发与测试

测试依赖由 `tests/conftest.py` 提供 NekroAgent/NoneBot 测试桩。安装测试环境所需的 Python 包后，在仓库根目录执行：

```bash
python -m pytest tests/
```

也可执行 Python 编译检查：

```bash
python -m compileall -q source tests
```

## 更新日志与许可证

- 更新记录：[CHANGELOG.md](CHANGELOG.md)
- 本项目采用 [MIT License](LICENSE)。
