"""
Embedding 服务 — 调用 OpenAI-compatible embeddings API，
生成病例卡摘要的语义向量并计算余弦相似度。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBatchResult:
    """批量 embedding 的结构化返回结果。

    每个输入文本对应 vectors 列表中的一个元素。
    失败的项在 vectors 中为 None，对应的错误信息在 errors 字典中。
    """
    vectors: list  # list[np.ndarray | None]
    errors: dict  # dict[int, str]  — 索引 -> 错误描述
    model: str = ""
    dimension: int = 0


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

# EmbeddingConfig 已迁移到项目根目录 config.py，此处保留别名以兼容旧导入路径
from config import EmbeddingConfig  # noqa: F401  # 向后兼容，新代码请用 from config import EmbeddingConfig


# ═══════════════════════════════════════════════════════════════════
# 相似度计算
# ═══════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    两个向量间的余弦相似度。

    Args:
        a: 1-D float32/float64 numpy array
        b: 1-D float32/float64 numpy array

    Returns:
        余弦相似度 (0~1)，向量已归一化时直接用点积
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    一个查询向量与多个候选向量的批量余弦相似度。

    Args:
        query: 1-D (D,) float32
        candidates: 2-D (N, D) float32

    Returns:
        1-D (N,) float64 相似度数组
    """
    query = query.astype(np.float64)
    candidates = candidates.astype(np.float64)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(candidates))
    query = query / query_norm
    candidates_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidates_norm[candidates_norm == 0] = 1.0
    candidates = candidates / candidates_norm
    return np.dot(candidates, query)


# ═══════════════════════════════════════════════════════════════════
# Embedding 服务
# ═══════════════════════════════════════════════════════════════════

class EmbeddingService:
    """
    调用 OpenAI-compatible embeddings API。

    用法:
        cfg = EmbeddingConfig()
        service = EmbeddingService(cfg)
        vec = service.embed_text("73岁女性，心衰加重转入ICU...")
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._dimension: Optional[int] = None
        self._request_count = 0
        self._error_count = 0

    @property
    def is_available(self) -> bool:
        return self.config.is_configured

    @property
    def dimension(self) -> Optional[int]:
        """返回 embedding 向量维度（首次调用后确定）"""
        return self._dimension

    @property
    def _http_client(self):
        """Lazily create the shared httpx client."""
        if not hasattr(self, "_http"):
            from http_client import HTTPClient
            connect_timeout = getattr(self.config, "connect_timeout", 10)
            max_connections = getattr(self.config, "max_connections", 10)
            self._http = HTTPClient(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
                connect_timeout=connect_timeout,
                read_timeout=self.config.timeout,
                max_connections=max_connections,
                max_retries=self.config.max_retries,
            )
        return self._http

    def embed_text(self, text: str) -> np.ndarray:
        """
        生成单条文本的 embedding 向量。

        Args:
            text: 输入文本

        Returns:
            float32 numpy array, shape (D,)
        """
        result = self.embed_batch([text])
        if result.errors:
            raise RuntimeError(f"Embedding 失败: {result.errors.get(0, 'unknown')}")
        vec = result.vectors[0]
        if vec is None:
            raise RuntimeError("Embedding 返回空向量")
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 0) -> EmbeddingBatchResult:
        """批量生成 embedding 向量，返回结构化结果。

        Args:
            texts: 输入文本列表
            batch_size: 每批发送的文本数（0=使用配置默认值）

        Returns:
            EmbeddingBatchResult，包含 vectors、errors、model、dimension
        """
        if not self.is_available:
            raise RuntimeError("Embedding 服务未配置，请设置 EMBEDDING_API_BASE 和 EMBEDDING_API_KEY 环境变量")

        if batch_size <= 0:
            batch_size = getattr(self.config, "batch_size", 32)

        vectors: list = [None] * len(texts)
        errors: dict[int, str] = {}

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                body = self._http_client.post_json("embeddings", {
                    "model": self.config.model,
                    "input": batch,
                })
                data_list = body.get("data", [])

                # Strict validation: length must match
                if len(data_list) != len(batch):
                    raise ValueError(
                        f"Embedding 服务返回 {len(data_list)} 个向量，"
                        f"期望 {len(batch)} 个"
                    )

                # Validate indices and vector quality
                returned_indices: set[int] = set()
                for item in data_list:
                    idx = item.get("index")
                    emb = item.get("embedding")
                    if idx is None or emb is None:
                        raise ValueError("Embedding 响应缺少 index 或 embedding")
                    if idx in returned_indices:
                        raise ValueError(f"Embedding 响应中有重复 index: {idx}")
                    returned_indices.add(idx)

                    arr = np.array(emb, dtype=np.float32)

                    # NaN / Inf check
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        raise ValueError(f"Embedding 向量在 index {idx} 包含 NaN 或 Inf")

                    # Zero vector check
                    if np.all(arr == 0):
                        raise ValueError(f"Embedding 向量在 index {idx} 为零向量")

                    # Dimension tracking
                    if self._dimension is None:
                        self._dimension = len(arr)
                        logger.info(f"Embedding 维度: {self._dimension}")
                    elif len(arr) != self._dimension:
                        raise ValueError(
                            f"Embedding 维度不匹配: 期望 {self._dimension}，"
                            f"实际 {len(arr)} (index {idx})"
                        )

                    vectors[i + idx] = arr

                self._request_count += 1

            except Exception as e:
                logger.warning(f"Embedding 批次请求失败 (offset={i}, size={len(batch)}): {e}")
                # Split retry: if batch has multiple items, try one-by-one
                if len(batch) > 1:
                    for j, text in enumerate(batch):
                        try:
                            vec = self.embed_text(text)
                            vectors[i + j] = vec
                        except Exception as inner_e:
                            errors[i + j] = str(inner_e)
                else:
                    errors[i] = str(e)

        self._error_count += len(errors)

        return EmbeddingBatchResult(
            vectors=vectors,
            errors=errors,
            model=self.config.model,
            dimension=self._dimension or 0,
        )

    def get_stats(self) -> dict:
        """返回服务统计信息"""
        return {
            "available": self.is_available,
            "dimension": self._dimension,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "model": self.config.model,
        }

    def close(self) -> None:
        client = getattr(self, "_http", None)
        if client is not None:
            client.close()
