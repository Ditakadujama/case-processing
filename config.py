"""
统一配置文件 — 数据库、LLM、Embedding 配置集中管理。

所有配置读取优先级：环境变量 > .env 文件 > 代码默认值。

用法:
    from config import DBConfig, LLMConfig, EmbeddingConfig

    db_cfg = DBConfig.from_env()
    llm_cfg = LLMConfig()
    emb_cfg = EmbeddingConfig()
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
    timeout: int = field(default_factory=lambda: int(os.environ.get("LLM_TIMEOUT", "120")))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "3")))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key)


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

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key)
