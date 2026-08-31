"""
统一配置文件 — 数据库、LLM、Embedding 配置集中管理。

所有配置读取优先级：环境变量 > .env 文件 > 代码默认值。

用法:
    from config import DBConfig, LLMConfig, EmbeddingConfig, RerankerConfig

    db_cfg = DBConfig.from_env()
    llm_cfg = LLMConfig()
    emb_cfg = EmbeddingConfig()
    reranker_cfg = RerankerConfig()
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# .env 文件加载（无外部依赖）
# ═══════════════════════════════════════════════════════════════════

def _load_dotenv(dotenv_path: Optional[str] = None) -> None:
    """
    加载 .env 文件中的环境变量（不覆盖已有的环境变量）。

    优先级：已有环境变量 > .env 文件。这意味着命令行 export 的值优先于 .env 文件。
    这很关键——用户在终端 export 的值不会被 .env 文件覆盖。
    """
    if dotenv_path is None:
        # 优先查找当前目录，其次是项目根目录
        candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        ]
        dotenv_path = None
        for c in candidates:
            if os.path.isfile(c):
                dotenv_path = c
                break

    if dotenv_path is None or not os.path.isfile(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 移除引号
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # 不覆盖已有的环境变量（命令行 export 优先）
            if key and key not in os.environ:
                os.environ[key] = value


# 模块加载时自动读取 .env
_load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# 数据库配置
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DBConfig:
    """MySQL 连接配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "medical_records"
    charset: str = "utf8mb4"
    # Stage 3: connection pooling
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    pool_max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MAX_OVERFLOW", "5")))
    connect_timeout: int = field(default_factory=lambda: int(os.getenv("DB_CONNECT_TIMEOUT", "10")))
    read_timeout: int = field(default_factory=lambda: int(os.getenv("DB_READ_TIMEOUT", "60")))
    write_timeout: int = field(default_factory=lambda: int(os.getenv("DB_WRITE_TIMEOUT", "60")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("DB_BATCH_SIZE", "200")))

    @classmethod
    def from_env(cls) -> "DBConfig":
        """从环境变量读取配置（未设置则使用默认值）"""
        return cls(
            host=os.getenv("DB_HOST", cls.host),
            port=int(os.getenv("DB_PORT", str(cls.port))),
            user=os.getenv("DB_USER", cls.user),
            password=os.getenv("DB_PASSWORD", cls.password),
            database=os.getenv("DB_NAME", cls.database),
            charset=os.getenv("DB_CHARSET", cls.charset),
        )

    def to_connection_kwargs(self) -> dict:
        """转换为 pymysql.connect 参数（不含 cursorclass，由调用方设置）"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
        }


# ═══════════════════════════════════════════════════════════════════
# LLM 配置
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """LLM 服务配置（OpenAI-compatible API）"""
    api_base: str = field(default_factory=lambda: os.environ.get("LLM_API_BASE", ""))
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "4096")))
    temperature: float = 0.0
    enable_thinking: bool = field(
        default_factory=lambda: os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
    )
    timeout: int = field(default_factory=lambda: int(os.environ.get("LLM_TIMEOUT", "300")))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "3")))
    context_max_chars: int = field(default_factory=lambda: int(os.environ.get("LLM_CONTEXT_MAX_CHARS", "100000")))
    enable_json_repair: bool = field(default_factory=lambda: os.environ.get("LLM_ENABLE_JSON_REPAIR", "1") != "0")
    save_failed_raw: bool = field(default_factory=lambda: os.environ.get("LLM_SAVE_FAILED_RAW", "1") != "0")
    failed_raw_dir: str = field(default_factory=lambda: os.environ.get("LLM_FAILED_RAW_DIR", "data/llm_failures"))
    response_format_json: bool = field(default_factory=lambda: os.environ.get("LLM_RESPONSE_FORMAT_JSON", "0") == "1")
    # Stage 6: history context
    history_mode: str = field(default_factory=lambda: os.environ.get("LLM_HISTORY_MODE", "window"))
    history_window_days: int = field(default_factory=lambda: int(os.environ.get("LLM_HISTORY_WINDOW_DAYS", "7")))
    history_max_chars: int = field(default_factory=lambda: int(os.environ.get("LLM_HISTORY_MAX_CHARS", "30000")))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key)

    def __post_init__(self):
        valid_modes = {"all", "window"}
        if self.history_mode not in valid_modes:
            raise ValueError(
                f"LLM_HISTORY_MODE 必须是 {valid_modes} 之一，当前值: '{self.history_mode}'"
            )
        if self.history_window_days < 1:
            raise ValueError("LLM_HISTORY_WINDOW_DAYS 必须 >= 1")
        if self.history_max_chars < 1000:
            raise ValueError("LLM_HISTORY_MAX_CHARS 必须 >= 1000")


# ═══════════════════════════════════════════════════════════════════
# Embedding 配置
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingConfig:
    """Embedding 服务配置（OpenAI-compatible API）"""
    api_base: str = field(default_factory=lambda: os.environ.get("EMBEDDING_API_BASE", ""))
    api_key: str = field(default_factory=lambda: os.environ.get("EMBEDDING_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    timeout: int = field(default_factory=lambda: int(os.environ.get("EMBEDDING_TIMEOUT", "60")))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("EMBEDDING_MAX_RETRIES", "3")))
    # Stage 4: batch embedding
    batch_size: int = field(default_factory=lambda: int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")))
    connect_timeout: int = field(default_factory=lambda: int(os.environ.get("EMBEDDING_CONNECT_TIMEOUT", "10")))
    max_connections: int = field(default_factory=lambda: int(os.environ.get("EMBEDDING_MAX_CONNECTIONS", "10")))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key)


@dataclass
class RetrievalConfig:
    """本地候选预筛配置。"""

    backend: str = field(default_factory=lambda: os.environ.get(
        "RETRIEVAL_BACKEND", "in_memory_vector"
    ))
    patient_candidates: int = field(default_factory=lambda: int(os.environ.get(
        "RETRIEVAL_PATIENT_CANDIDATES", "200"
    )))
    day_candidates_per_query: int = field(default_factory=lambda: int(os.environ.get(
        "RETRIEVAL_DAY_CANDIDATES_PER_QUERY_DAY", "300"
    )))
    fallback_to_full_scan: bool = field(default_factory=lambda: os.environ.get(
        "RETRIEVAL_FALLBACK_TO_FULL_SCAN", "1"
    ) != "0")

    def __post_init__(self):
        if self.backend not in {"in_memory_vector", "legacy_full_scan"}:
            raise ValueError(f"未知 RETRIEVAL_BACKEND: {self.backend}")
        if self.patient_candidates < 1:
            raise ValueError("RETRIEVAL_PATIENT_CANDIDATES 必须 >= 1")
        if self.day_candidates_per_query < 1:
            raise ValueError("RETRIEVAL_DAY_CANDIDATES_PER_QUERY_DAY 必须 >= 1")


# ═══════════════════════════════════════════════════════════════════
# Reranker 配置
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RerankerConfig:
    """Reranker 服务配置（常见 /v1/rerank 或 /rerank HTTP API）"""
    api_base: str = field(default_factory=lambda: os.environ.get("RERANKER_API_BASE", ""))
    api_key: str = field(default_factory=lambda: os.environ.get("RERANKER_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("RERANKER_MODEL", "qwen3-reranker-4b"))
    endpoint: str = field(default_factory=lambda: os.environ.get("RERANKER_ENDPOINT", "/score"))
    instruction: str = field(default_factory=lambda: os.environ.get(
        "RERANKER_INSTRUCTION",
        "判断候选重症医学病例是否与查询病例在最终诊断、病因链、疾病阶段、关键病程和关键治疗上相似。"
        "最终诊断/病因链一致性最重要；不要仅因同为休克、插管、ICU、呼吸衰竭等泛化危重表现就判为相似。",
    ))
    timeout: int = field(default_factory=lambda: int(os.environ.get("RERANKER_TIMEOUT", "120")))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("RERANKER_MAX_RETRIES", "2")))
    max_query_chars: int = field(default_factory=lambda: int(os.environ.get("RERANKER_MAX_QUERY_CHARS", "8000")))
    max_doc_chars: int = field(default_factory=lambda: int(os.environ.get("RERANKER_MAX_DOC_CHARS", "8000")))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base)

    @property
    def url(self) -> str:
        return f"{self.api_base.rstrip('/')}/{self.endpoint.lstrip('/')}"
