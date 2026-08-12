from __future__ import annotations

from pydantic import ConfigDict, Field

from nekro_agent.api.plugin import ConfigBase


CATEGORY_VOCABULARY = [
    "confusion",
    "happiness",
    "speechlessness",
    "anger",
    "sadness",
    "surprise",
    "agreement",
    "refusal",
    "comfort",
    "shyness",
    "mockery",
    "celebration",
    "neutral reaction",
    "other",
]


class SemanticStickerConfig(ConfigBase):
    model_config = ConfigDict(validate_assignment=True)

    ANALYSIS_MODEL_GROUP: str = Field(
        default="",
        title="分析模型组",
        description="用于识别表情包内容并自动生成分类、标签和使用场景的聊天模型组。",
        json_schema_extra={"ref_model_groups": True, "required": True, "model_type": "chat"},
    )
    ANALYSIS_PROMPT_VERSION: str = Field(
        default="v1",
        title="分析提示词版本",
        description="表情包视觉分析提示词的版本标识，用于记录分析结果来源。",
    )
    ANALYSIS_TIMEOUT_SECONDS: int = Field(
        default=60,
        title="分析超时时间",
        description="调用分析模型的超时时间，单位为秒。",
    )
    ANALYSIS_RETRY_COUNT: int = Field(
        default=2,
        title="分析重试次数",
        description="分析模型调用失败后的最大重试次数。",
    )
    MAX_GENERATED_TAGS: int = Field(
        default=12,
        title="自动标签数量上限",
        description="模型为单张表情包生成的情绪标签和场景标签数量上限。",
    )
    CATEGORY_VOCABULARY: list[str] = Field(
        default_factory=lambda: list(CATEGORY_VOCABULARY),
        title="表情包分类词表",
        description="自动分类允许使用的分类名称列表，无法匹配时归入 other。",
    )
    ANALYSIS_CONCURRENCY: int = Field(
        default=1,
        title="分析并发数",
        description="后台同时执行的表情包分析任务数量。",
    )
    EMBEDDING_MODEL_GROUP: str = Field(
        default="text-embedding",
        title="嵌入模型组",
        description="用于表情包语义检索和向量索引的嵌入模型组。",
        json_schema_extra={"ref_model_groups": True, "required": True, "model_type": "embedding"},
    )
    VECTOR_DIMENSION: int = Field(
        default=1536,
        title="向量维度",
        description="嵌入模型输出的向量维度，必须与所选嵌入模型一致。",
    )
    SEMANTIC_SCORE_THRESHOLD: float = Field(
        default=0.72,
        title="语义匹配阈值",
        description="发送表情包时允许采用候选结果的最低语义相似度。",
    )
    MAX_UPLOAD_BYTES: int = Field(
        default=10_485_760,
        title="单图大小上限",
        description="单张上传或聊天保存图片允许的最大字节数。",
    )
    MAX_IMAGE_PIXELS: int = Field(
        default=40_000_000,
        title="图片像素上限",
        description="单张图片宽度乘高度允许的最大像素数。",
    )
    MAX_WIDTH: int = Field(
        default=8192,
        title="图片宽度上限",
        description="单张表情包允许的最大宽度，单位为像素。",
    )
    MAX_HEIGHT: int = Field(
        default=8192,
        title="图片高度上限",
        description="单张表情包允许的最大高度，单位为像素。",
    )
    MAX_ANIMATION_FRAMES: int = Field(
        default=300,
        title="动图帧数上限",
        description="GIF 或 WebP 动图允许的最大帧数。",
    )
    RECENT_SELECTION_WINDOW: int = Field(
        default=10,
        title="近期去重窗口",
        description="语义检索时用于减少短期重复发送同一表情包的近期记录数量。",
    )
    PHYSICAL_CHANNEL_COOLDOWN_SECONDS: int = Field(
        default=20,
        ge=0,
        title="物理频道发送冷却",
        description="同一物理频道连续发送表情包的冷却时间，单位秒；0 表示关闭冷却。Agent 工具与 Bot 被戳后的直接表情回复共用此设置。",
    )
    AUTO_COLLECT_ENABLED: bool = Field(
        default=True,
        title="允许 Agent 自动保存表情包",
        description="开启后主 Agent 可在确认图片是表情包时自主保存；关闭后仅响应用户明确提出的保存请求。",
    )
    STRICT_EMOTION_COLLECT: bool = Field(
        default=True,
        title="严格表情包模式",
        description="开启后只有经主 Agent 视觉确认的表情包或 reaction GIF 才能保存，截图、照片和无法确认的图片会被拒绝。",
    )
