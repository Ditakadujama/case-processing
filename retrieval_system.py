"""
病历相似度检索系统
新病例入库时，快速找到相似度≥阈值的已记录病例
"""

from typing import List, Dict, Optional, Tuple
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from record_parser import MedicalRecordParser, MedicalRecord
from feature_extractor import FeatureExtractor
from similarity_index import create_index, VectorIndex
from timeline_parser import TimelineParser
from timeline_similarity import TimelineSimilarityScorer, TimelineFeatures


# ── 模块级 worker 函数（供 ProcessPoolExecutor 使用）─────────────────────

def _worker_parse_record(item):
    """解析单条病历文本（多进程 worker）"""
    record_id, text = item
    parser = MedicalRecordParser()
    parsed = parser.parse(text)
    return record_id, text, parsed


def _worker_parse_timeline(item):
    """提取单条病历的时间轴特征（多进程 worker）"""
    record_id, text, surgery_type, surgery_keywords = item
    timeline_parser = TimelineParser()
    scorer = TimelineSimilarityScorer()
    events = timeline_parser.parse(text)
    nodes = timeline_parser.generate_standard_nodes(events)
    features = scorer.extract_features(events, nodes,
                                        surgery_type_hint=surgery_type or None,
                                        surgery_keywords_hint=surgery_keywords or None)
    return record_id, events, nodes, features


class MedicalRecordSimilaritySystem:
    """
    病历相似度检索系统

    支持功能:
    - 新病例入库 (add_record)
    - 相似病例检索 (search)
    - 检索+自动入库 (search_and_add)
    """

    def __init__(self,
                 similarity_threshold: float = 0.7,
                 feature_dim: int = None,
                 index_backend: str = "sklearn",
                 index_path: Optional[str] = None,
                 alpha: float = 0.4,
                 min_timeline_score: float = 0.3):
        """
        Args:
            similarity_threshold: 相似度阈值，低于此值的病例不返回
            feature_dim: 特征向量维度（自动从extractor获取）
            index_backend: 索引后端 "sklearn" 或 "faiss"
            index_path: 索引持久化路径
            alpha: 向量相似度权重 (0~1)，默认0.5；最终分数 = alpha*向量 + (1-alpha)*时间轴
            min_timeline_score: 时间轴相似度软惩罚阈值，低于此值时最终分数乘以0.5
        """
        self.threshold = similarity_threshold
        self.index_path = index_path
        self.alpha = alpha
        self.min_timeline_score = min_timeline_score

        # 组件初始化
        self.parser = MedicalRecordParser()
        self.extractor = FeatureExtractor()
        self.timeline_parser = TimelineParser()
        self.timeline_scorer = TimelineSimilarityScorer()
        # 自动获取特征维度
        self.feature_dim = feature_dim or self.extractor.total_dim
        self.index: VectorIndex = create_index(self.feature_dim, index_backend)

        # 病例存储: id -> {text, features, parsed_record, timeline_events, standard_nodes, timeline_features}
        self.records: Dict[str, Dict] = {}
        self.record_order: List[str] = []  # 保持插入顺序
        self.record_count = 0

        # TF-IDF 向量器是否已训练
        self._extractor_fitted = False

        # 尝试加载已有索引
        if index_path and os.path.exists(index_path + ".npy"):
            self._load_index()
        else:
            # 使用默认词汇表初始化TF-IDF
            self._init_default_vocabulary()

    def _init_default_vocabulary(self) -> None:
        """初始化向量提取器"""
        # 词袋模型不需要训练，直接标记为已就绪
        self.extractor.fit([])
        self._extractor_fitted = True

    def add_record(self, record_id: str, text: str) -> None:
        """
        添加病例到检索系统

        Args:
            record_id: 病例唯一ID
            text: 病历文本
        """
        # 解析病历
        parsed_record = self.parser.parse(text)

        # 提取特征（词袋模型不需要训练）
        features = self.extractor.extract(parsed_record)

        # 解析时间轴并预计算特征（传入手术类型和关键词，避免从文本误分类）
        timeline_events = self.timeline_parser.parse(text)
        standard_nodes = self.timeline_parser.generate_standard_nodes(timeline_events)
        timeline_features = self.timeline_scorer.extract_features(
            timeline_events, standard_nodes,
            surgery_type_hint=parsed_record.surgery_type or None,
            surgery_keywords_hint=parsed_record.surgery_keywords or None
        )

        # 添加到索引
        self.index.add(features.reshape(1, -1))

        # 存储病例数据
        self.records[record_id] = {
            'text': text,
            'features': features,
            'parsed_record': parsed_record,
            'timeline_events': timeline_events,
            'standard_nodes': standard_nodes,
            'timeline_features': timeline_features
        }
        self.record_order.append(record_id)

    def add_records_batch(self, records: Dict[str, str], num_workers: int = 1) -> None:
        """
        批量添加病例到检索系统

        Args:
            records: {record_id: text} 映射
            num_workers: 并行进程数（1=单进程，>1=多进程并行解析和时间轴提取）
        """
        if not records:
            return

        records_list = list(records.items())
        n = len(records_list)
        workers = min(max(num_workers, 1), n)

        # ── 阶段1：解析病历 ──
        if workers > 1:
            print(f"  [阶段1/4] 并行解析 {n} 条病历 (workers={workers}) ...")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                parsed_results = list(executor.map(_worker_parse_record, records_list))
            # executor.map 保持顺序，解包
            parsed_records = [p for _, _, p in parsed_results]
            records_list = [(rid, txt) for rid, txt, _ in parsed_results]
        else:
            parsed_records = []
            for record_id, text in records_list:
                parsed_records.append(self.parser.parse(text))

        # IDF 拟合（基于全量解析结果）
        self.extractor.fit(parsed_records)

        # ── 阶段2：特征提取 ──
        features = self.extractor.extract_batch(parsed_records)

        if features.size == 0:
            return

        # ── 阶段3：向量入索引 ──
        self.index.add(features)

        # ── 阶段4：时间轴预计算 ──
        if workers > 1:
            # 准备并行输入：(record_id, text, surgery_type, surgery_keywords)
            timeline_inputs = [
                (rid, txt,
                 parsed_records[i].surgery_type or None,
                 parsed_records[i].surgery_keywords or None)
                for i, (rid, txt) in enumerate(records_list)
            ]
            print(f"  [阶段4/4] 并行提取时间轴 (workers={workers}) ...")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                timeline_results = list(executor.map(_worker_parse_timeline, timeline_inputs))
            # 构建 record_id → timeline 结果映射
            timeline_map = {rid: (events, nodes, feat) for rid, events, nodes, feat in timeline_results}
        else:
            timeline_map = {}
            for i, (record_id, text) in enumerate(records_list):
                events = self.timeline_parser.parse(text)
                nodes = self.timeline_parser.generate_standard_nodes(events)
                feat = self.timeline_scorer.extract_features(
                    events, nodes,
                    surgery_type_hint=parsed_records[i].surgery_type or None,
                    surgery_keywords_hint=parsed_records[i].surgery_keywords or None
                )
                timeline_map[record_id] = (events, nodes, feat)

        # 批量存储病例数据
        for i, (record_id, text) in enumerate(records_list):
            parsed = parsed_records[i]
            events, nodes, tl_feat = timeline_map[record_id]

            self.records[record_id] = {
                'text': text,
                'features': features[i],
                'parsed_record': parsed,
                'timeline_events': events,
                'standard_nodes': nodes,
                'timeline_features': tl_feat,
            }
            self.record_order.append(record_id)

    def save(self) -> None:
        """手动保存索引和元数据到磁盘"""
        self._save_index()

    def search(self,
               query_text: str,
               top_k: int = 10,
               auto_add: bool = False,
               record_id: Optional[str] = None,
               exclude_record_ids: Optional[set] = None) -> List[Dict]:
        """
        检索相似病例

        Args:
            query_text: 待查询病历文本
            top_k: 返回前K个最相似病例
            auto_add: 是否自动入库
            record_id: 病例ID（auto_add=True时必填）
            exclude_record_ids: 需要排除的病例ID集合（如查询病例自身）

        Returns:
            [{'id': xxx, 'similarity': 0.85, 'text': xxx}, ...]
        """
        if auto_add:
            return self.search_and_add(query_text, record_id, top_k, exclude_record_ids)

        # 仅检索（不入库）
        return self._search_only(query_text, top_k, exclude_record_ids)

    def _search_only(self, query_text: str, top_k: int, exclude_record_ids: Optional[set] = None) -> List[Dict]:
        """仅检索，不入库（两阶段：向量粗排 + 时间轴精排）"""
        if not self.records:
            return []

        exclude_record_ids = exclude_record_ids or set()

        # 解析查询病历
        query_record = self.parser.parse(query_text)
        query_features = self.extractor.extract(query_record).reshape(1, -1)

        # 解析查询病历的时间轴（传入手术类型）
        query_events = self.timeline_parser.parse(query_text)
        query_nodes = self.timeline_parser.generate_standard_nodes(query_events)
        query_timeline_features = self.timeline_scorer.extract_features(
            query_events, query_nodes,
            surgery_type_hint=query_record.surgery_type or None,
            surgery_keywords_hint=query_record.surgery_keywords or None
        )

        # 第一阶段：向量检索，扩大候选池
        vector_top_k = min(max(top_k * 20, 100), len(self.record_order))
        distances, indices = self.index.search(query_features, vector_top_k)

        # 第二阶段：时间轴相似度重排序
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.record_order):
                continue

            record_id = self.record_order[idx]
            # 排除自身
            if record_id in exclude_record_ids:
                continue

            vector_sim = self._distance_to_similarity(dist)

            # 从缓存读取候选病例的时间轴特征
            candidate_data = self.records[record_id]
            cand_timeline_features = candidate_data.get('timeline_features')
            if cand_timeline_features is not None:
                timeline_sim = self.timeline_scorer.score(query_timeline_features, cand_timeline_features)
            else:
                # 向后兼容：老数据没有时间轴缓存时退回到向量相似度
                timeline_sim = vector_sim

            # 合并分数
            final_sim = self.alpha * vector_sim + (1.0 - self.alpha) * timeline_sim

            # 软惩罚：时间轴相似度极低时打压最终分数
            if timeline_sim < self.min_timeline_score:
                final_sim *= 0.5

            if final_sim >= self.threshold:
                candidates.append({
                    'id': record_id,
                    'similarity': round(final_sim, 4),
                    'vector_similarity': round(vector_sim, 4),
                    'timeline_similarity': round(timeline_sim, 4),
                    'text': self.records[record_id]['text'][:200] + "...",
                    'full_text': self.records[record_id]['text']
                })

        # 按最终相似度排序，返回 top_k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_k]

    def search_and_add(self,
                       query_text: str,
                       record_id: Optional[str] = None,
                       top_k: int = 10,
                       exclude_record_ids: Optional[set] = None) -> List[Dict]:
        """
        检索相似病例 + 自动入库

        Args:
            query_text: 待查询病历文本
            record_id: 病例ID（若不提供则自动生成）
            top_k: 返回前K个最相似病例
            exclude_record_ids: 需要排除的病例ID集合

        Returns:
            [{'id': xxx, 'similarity': 0.85, 'text': xxx}, ...]
        """
        # 1. 先检索（基于当前底库，不含新病例）
        results = self._search_only(query_text, top_k, exclude_record_ids)

        # 2. 自动入库
        if record_id is None:
            record_id = f"record_{self.record_count}"
        self.record_count += 1

        self.add_record(record_id, query_text)

        return results

    def _distance_to_similarity(self, distance: float) -> float:
        """
        将距离转换为相似度
        使用余弦相似度: similarity = 1 - distance
        """
        return max(0.0, 1.0 - distance)

    def _save_index(self) -> None:
        """保存索引和元数据"""
        if not self.index_path:
            return

        # 保存向量索引
        self.index.save(self.index_path)

        # 构建精简的时间轴缓存用于持久化
        timeline_cache = {}
        for rid, data in self.records.items():
            tf = data.get('timeline_features')
            if tf is not None:
                timeline_cache[rid] = {
                    'visit_count': tf.visit_count,
                    'total_span_days': tf.total_span_days,
                    'visit_gaps': tf.visit_gaps,
                    'diagnosis_keywords': list(tf.diagnosis_keywords),
                    'event_type_sequence': tf.event_type_sequence,
                    'surgery_type': tf.surgery_type,
                    'surgery_keywords': list(tf.surgery_keywords),
                }

        # 保存元数据
        metadata = {
            'records': {k: {'text': v['text']} for k, v in self.records.items()},
            'record_order': self.record_order,
            'threshold': self.threshold,
            'feature_dim': self.feature_dim,
            'record_count': self.record_count,
            'alpha': self.alpha,
            'min_timeline_score': self.min_timeline_score,
            'timeline_cache': timeline_cache,
        }
        with open(self.index_path + ".meta", 'wb') as f:
            pickle.dump(metadata, f)

    def _load_index(self) -> None:
        """加载索引和元数据"""
        if not self.index_path:
            return

        try:
            # 加载元数据
            with open(self.index_path + ".meta", 'rb') as f:
                metadata = pickle.load(f)

            self.records = metadata['records']
            self.record_order = metadata['record_order']
            self.threshold = metadata.get('threshold', 0.7)
            self.feature_dim = metadata.get('feature_dim', 600)
            self.record_count = metadata.get('record_count', len(self.record_order))
            self.alpha = metadata.get('alpha', 0.6)
            self.min_timeline_score = metadata.get('min_timeline_score', 0.1)
            timeline_cache = metadata.get('timeline_cache', {})

            # 重新解析病历并恢复时间轴缓存
            for record_id, data in self.records.items():
                parsed = self.parser.parse(data['text'])
                features = self.extractor.extract(parsed)
                data['features'] = features
                data['parsed_record'] = parsed

                # 恢复时间轴缓存（向后兼容：老索引可能没有新字段）
                tc = timeline_cache.get(record_id)
                if tc:
                    # 检测新旧格式
                    if 'visit_count' in tc:
                        # 新格式（就诊计数模型）
                        data['timeline_features'] = TimelineFeatures(
                            visit_count=tc['visit_count'],
                            total_span_days=tc['total_span_days'],
                            visit_gaps=tc['visit_gaps'],
                            diagnosis_keywords=set(tc['diagnosis_keywords']),
                            event_type_sequence=tc['event_type_sequence'],
                            surgery_type=tc.get('surgery_type'),
                            surgery_keywords=set(tc.get('surgery_keywords', [])),
                        )
                    else:
                        # 老格式（T0-T6 节点模型），丢弃并设为 None
                        data['timeline_features'] = None
                else:
                    data['timeline_features'] = None

            # 加载向量索引
            from similarity_index import create_index
            self.index = create_index(self.feature_dim, "sklearn")
            self.index.load(self.index_path)

            print(f"Loaded {len(self.records)} records from index")
        except Exception as e:
            print(f"Failed to load index: {e}")

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'total_records': len(self.records),
            'threshold': self.threshold,
            'feature_dim': self.feature_dim,
            'index_type': type(self.index).__name__
        }

    def set_threshold(self, threshold: float) -> None:
        """设置相似度阈值"""
        self.threshold = threshold


def create_system(data_dir: str = "./data",
                  threshold: float = 0.45) -> MedicalRecordSimilaritySystem:
    """
    工厂函数：创建检索系统

    Args:
        data_dir: 数据存储目录
        threshold: 相似度阈值

    Returns:
        MedicalRecordSimilaritySystem 实例
    """
    os.makedirs(data_dir, exist_ok=True)
    index_path = os.path.join(data_dir, "similarity_index")

    return MedicalRecordSimilaritySystem(
        similarity_threshold=threshold,
        index_path=index_path
    )


if __name__ == "__main__":
    # 测试
    system = create_system(data_dir="./test_data", threshold=0.6)

    # 添加测试病例
    with open("病历.txt", "r", encoding="utf-8") as f:
        text = f.read()

    system.add_record("病例001", text)

    # 检索
    results = system.search(text, top_k=5)

    print(f"\n=== 检索结果 (阈值={system.threshold}) ===")
    for r in results:
        print(f"ID: {r['id']}, 相似度: {r['similarity']}")

    print(f"\n系统统计: {system.get_stats()}")
