"""
相似度索引模块
封装向量索引，支持 sklearn 和 FAISS 两种后端
"""

import numpy as np
from typing import Tuple, Optional, Protocol
from abc import ABC, abstractmethod

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorIndex(ABC):
    """向量索引抽象接口"""

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """添加向量到索引"""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最近邻

        Args:
            query: 查询向量 (1 x dim)
            k: 返回数量

        Returns:
            (distances, indices) - 距离和索引
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存索引"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """加载索引"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """索引中向量数量"""
        pass


class SklearnIndex(VectorIndex):
    """
    基于sklearn的暴力计算索引
    适用于小规模数据 (< 1万)
    """

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._vectors = []  # 存储所有向量
        self._indexed = False

    def add(self, vectors: np.ndarray) -> None:
        """添加向量"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._vectors.append(vectors)
        self._indexed = False

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """暴力搜索"""
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 合并所有向量
        all_vectors = np.vstack(self._vectors) if self._vectors else np.array([])

        if all_vectors.shape[0] == 0:
            # 空索引
            return np.array([[]]), np.array([[-1]])

        # 计算余弦距离
        # 归一化
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        all_norm = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-8)

        # 余弦相似度 = 1 - 距离
        similarities = np.dot(query_norm, all_norm.T)

        # 取top-k
        k = min(k, all_vectors.shape[0])
        top_k_idx = np.argsort(-similarities[0])[:k]
        top_k_dist = 1 - similarities[0, top_k_idx]  # 转换为距离

        return top_k_dist.reshape(1, -1), top_k_idx.reshape(1, -1)

    def save(self, path: str) -> None:
        """保存到文件"""
        if self._vectors:
            all_vectors = np.vstack(self._vectors)
            np.save(path + ".npy", all_vectors)

    def load(self, path: str) -> None:
        """从文件加载"""
        all_vectors = np.load(path + ".npy")
        self._vectors = [all_vectors]
        self._indexed = False

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        if not self._vectors:
            return 0
        return sum(v.shape[0] for v in self._vectors)


class FaissIndex(VectorIndex):
    """
    基于FAISS的向量索引
    适用于大规模数据 (1万+)
    """

    def __init__(self, dimension: int, nlist: int = 100):
        self._dimension = dimension
        self._index: Optional[faiss.Index] = None
        self._nlist = nlist
        self._is_trained = False

        if FAISS_AVAILABLE:
            # 使用IVF-PQ索引，适合大规模
            quantizer = faiss.IndexFlatIP(dimension)  # 内积索引（余弦相似度）
            self._index = faiss.IndexIVFPQ(quantizer, dimension,
                                           self._nlist, 8, 8)
        else:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

    def add(self, vectors: np.ndarray) -> None:
        """添加向量到索引"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        vectors = vectors.astype('float32')

        if not self._is_trained:
            # 训练索引
            if vectors.shape[0] >= self._nlist:
                self._index.train(vectors)
                self._is_trained = True

        self._index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最近邻"""
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype('float32')

        # 归一化为单位向量（余弦相似度）
        norms = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
        query_normalized = query / norms

        distances, indices = self._index.search(query_normalized, k)

        return distances, indices

    def save(self, path: str) -> None:
        """保存索引"""
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        """加载索引"""
        self._index = faiss.read_index(path)
        self._is_trained = True

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0


def create_index(dimension: int, backend: str = "sklearn") -> VectorIndex:
    """
    工厂函数：创建向量索引

    Args:
        dimension: 向量维度
        backend: "sklearn" 或 "faiss"

    Returns:
        VectorIndex 实例
    """
    if backend == "faiss" and FAISS_AVAILABLE:
        return FaissIndex(dimension)
    else:
        return SklearnIndex(dimension)


class IndexManager:
    """索引管理器，负责创建和切换后端"""

    def __init__(self, dimension: int, threshold: int = 10000):
        """
        Args:
            dimension: 向量维度
            threshold: 自动切换到FAISS的数据量阈值
        """
        self._dimension = dimension
        self._threshold = threshold
        self._sklearn_index = SklearnIndex(dimension)
        self._faiss_index: Optional[FaissIndex] = None
        self._current_index: VectorIndex = self._sklearn_index

    def add(self, vectors: np.ndarray) -> None:
        """添加向量"""
        self._current_index.add(vectors)

        # 检查是否需要切换到FAISS
        if (self._current_index == self._sklearn_index and
                self._current_index.size > self._threshold and
                FAISS_AVAILABLE):
            self._switch_to_faiss()

    def _switch_to_faiss(self) -> None:
        """切换到FAISS索引"""
        print("Switching to FAISS index...")

        # 重建FAISS索引
        self._faiss_index = FaissIndex(self._dimension)

        # 迁移数据
        all_vectors = []
        for vec in self._sklearn_index._vectors:
            all_vectors.append(vec)
        if all_vectors:
            merged = np.vstack(all_vectors)
            self._faiss_index.add(merged)
            self._faiss_index._is_trained = True

        self._current_index = self._faiss_index

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索"""
        return self._current_index.search(query, k)

    def save(self, path: str) -> None:
        """保存"""
        self._current_index.save(path)

    def load(self, path: str) -> None:
        """加载"""
        if FAISS_AVAILABLE:
            try:
                self._current_index = FaissIndex(self._dimension)
                self._current_index.load(path)
                self._faiss_index = self._current_index
            except:
                self._current_index = SklearnIndex(self._dimension)
                self._current_index.load(path)
        else:
            self._current_index = SklearnIndex(self._dimension)
            self._current_index.load(path)

    @property
    def size(self) -> int:
        return self._current_index.size


if __name__ == "__main__":
    # 测试
    index = create_index(100, backend="sklearn")

    # 添加一些测试向量
    vectors = np.random.randn(10, 100).astype('float32')
    index.add(vectors)

    # 搜索
    query = np.random.randn(1, 100).astype('float32')
    distances, indices = index.search(query, 3)

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
