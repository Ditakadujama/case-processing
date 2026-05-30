"""
Embedding 服务 — 调用 OpenAI-compatible embeddings API，
生成病例卡摘要的语义向量并计算余弦相似度。
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


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

    def embed_text(self, text: str) -> np.ndarray:
        """
        生成单条文本的 embedding 向量。

        Args:
            text: 输入文本（病例卡 summary_for_embedding）

        Returns:
            float32 numpy array, shape (D,)
        """
        vectors = self.embed_batch([text], batch_size=1)
        return vectors[0]

    def embed_batch(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        批量生成 embedding 向量。

        Args:
            texts: 输入文本列表
            batch_size: 每批发送的文本数

        Returns:
            2-D float32 numpy array, shape (len(texts), D)
        """
        import urllib.request
        import json as _json

        if not self.is_available:
            raise RuntimeError("Embedding 服务未配置，请设置 EMBEDDING_API_BASE 和 EMBEDDING_API_KEY 环境变量")

        all_vectors = []
        url = f"{self.config.api_base.rstrip('/')}/embeddings"

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = _json.dumps({
                "model": self.config.model,
                "input": batch,
            }).encode("utf-8")

            for attempt in range(self.config.max_retries):
                try:
                    req = urllib.request.Request(url, data=payload, method="POST")
                    req.add_header("Content-Type", "application/json")
                    req.add_header("Authorization", f"Bearer {self.config.api_key}")

                    with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                        body = _json.loads(resp.read().decode("utf-8"))

                    data_list = body.get("data", [])
                    data_list.sort(key=lambda x: x.get("index", 0))
                    batch_vectors = [np.array(item["embedding"], dtype=np.float32) for item in data_list]

                    if self._dimension is None and batch_vectors:
                        self._dimension = len(batch_vectors[0])
                        logger.info(f"Embedding 维度: {self._dimension}")

                    all_vectors.extend(batch_vectors)
                    self._request_count += 1
                    break

                except Exception as e:
                    logger.warning(f"Embedding 请求失败 (attempt {attempt+1}/{self.config.max_retries}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        self._error_count += 1
                        raise RuntimeError(f"Embedding 请求失败（已重试{self.config.max_retries}次）: {e}")

            # 批次间短暂等待，避免触发限流
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return np.array(all_vectors, dtype=np.float32)

    def get_stats(self) -> dict:
        """返回服务统计信息"""
        return {
            "available": self.is_available,
            "dimension": self._dimension,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "model": self.config.model,
        }
