"""
病历相似度检索系统
新病例入库时，快速找到相似度≥阈值的已记录病例
"""

from typing import List, Dict, Optional, Tuple
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from record_parser import MedicalRecordParser, MedicalRecord
from feature_extractor import FeatureExtractor
from similarity_index import create_index, VectorIndex
from timeline_parser import TimelineParser
from timeline_similarity import TimelineSimilarityScorer, TimelineFeatures
from vector_store import MySQLVectorStore
from config import DBConfig, EmbeddingConfig
from case_card import (
    tag_overlap_score, find_common_and_diff_tags,
    disease_axis_similarity, has_disease_axis_conflict,
    find_strong_common_tags, has_strong_common_tag,
)
from embedding_index import EmbeddingService, cosine_similarity, batch_cosine_similarity


SUMMARY_SECTIONS = (
    "chief_complaint", "主诉",
    "history_illness", "现病史",
    "inspection_visit", "查房记录",
    "surgery_record", "手术记录",
    "operation_record",
    "examine", "检查",
)

SUMMARY_KEYWORDS = (
    "入院诊断", "目前诊断", "出院诊断", "术后诊断", "主诉", "现病史",
    "手术名称", "手术日期", "机械通气", "气管插管", "呼吸机", "CRRT",
    "血滤", "IABP", "主动脉内球囊反搏", "休克", "感染", "呼吸衰竭",
    "心力衰竭", "肾功能不全", "肾衰竭", "血栓", "出血", "肺部感染",
    "多器官功能", "死亡", "转出", "转入",
)


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


def _extract_summary_text(text: str, limit: int = 6000) -> str:
    """抽取面向相似病例召回的临床摘要文本，避免监测/医嘱噪声主导。"""
    chunks = []
    section_set = set(SUMMARY_SECTIONS)
    for section_name, content in re.findall(r"###([^：:\n]+)[：:]?(.*?)(?=###[^\n]|\Z)", text, flags=re.DOTALL):
        if section_name.strip() in section_set and content:
            chunks.append(content[:1200])

    for kw in SUMMARY_KEYWORDS:
        start = 0
        hits = 0
        while hits < 4:
            idx = text.find(kw, start)
            if idx < 0:
                break
            chunks.append(text[max(0, idx - 80):idx + 240])
            start = idx + len(kw)
            hits += 1

    summary = "\n".join(chunks) if chunks else text[:limit]
    summary = re.sub(r"\s+", " ", summary)
    return summary[:limit]


def _char_ngram_vector(text: str, ngram_range: Tuple[int, int] = (2, 4)) -> Counter:
    """中文友好的轻量文本向量：字符 n-gram 计数。"""
    compact = re.sub(r"\s+", "", text)
    vec = Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            vec[compact[i:i + n]] += 1
    return vec


def _counter_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


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
                 min_timeline_score: float = 0.3,
                 db_config: Optional[DBConfig] = None,
                 enable_llm: bool = False):
        """
        Args:
            similarity_threshold: 相似度阈值，低于此值的病例不返回
            feature_dim: 特征向量维度（自动从extractor获取）
            index_backend: 索引后端 "sklearn" 或 "faiss"
            index_path: 索引持久化路径（用于保存轻量配置 + index_backend 向量文件）
            alpha: 向量相似度权重 (0~1)；最终分数 = alpha*向量 + (1-alpha)*时间轴
            min_timeline_score: 时间轴相似度软惩罚阈值，低于此值时最终分数乘以0.5
            db_config: MySQL 连接配置（用于向量持久化）
            enable_llm: 是否启用 LLM 病例卡 + embedding 增强检索
        """
        self.threshold = similarity_threshold
        self.index_path = index_path
        self.alpha = alpha
        self.min_timeline_score = min_timeline_score
        self.enable_llm = enable_llm

        # 组件初始化
        self.parser = MedicalRecordParser()
        self.extractor = FeatureExtractor()
        self.timeline_parser = TimelineParser()
        self.timeline_scorer = TimelineSimilarityScorer()
        self.feature_dim = feature_dim or self.extractor.total_dim
        self.index: VectorIndex = create_index(self.feature_dim, index_backend)

        # MySQL 向量存储（需在 _load_index 前初始化，因为 _load_index 用 _get_store）
        self._db_config = db_config
        self._store: Optional[MySQLVectorStore] = None

        # 元数据缓存：{record_id: {text, timeline_features}}，启动时加载
        self._metadata_cache: Dict[str, Dict] = {}
        self._summary_vector_cache: Dict[str, Counter] = {}
        self._timeline_refreshed_ids: set = set()
        self.record_order: List[str] = []
        self.record_count = 0

        # LLM 增强模块（懒初始化）
        self._case_card_store = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._case_card_cache: Dict[str, dict] = {}
        self._embedding_cache: Optional[np.ndarray] = None  # (N, D) float32
        self._embedding_ids: List[str] = []  # 与 _embedding_cache 行对齐
        self._embedding_id_to_idx: Dict[str, int] = {}  # record_id → embedding 行索引

        # LLM 抽取器（搜索时实时抽取查询病例卡）
        self._llm_extractor = None

        # TF-IDF 向量器是否已训练
        self._extractor_fitted = False

        # 病程窗口特征缓存: {(record_id, days): TimelineFeatures}
        self._timeline_window_feature_cache: Dict[Tuple[str, int], object] = {}

        # 尝试加载已有索引
        if self._get_store().count() > 0:
            self._load_index()
        else:
            self._init_default_vocabulary()

    def _get_store(self) -> MySQLVectorStore:
        """懒初始化 MySQLVectorStore"""
        if self._store is None:
            if self._db_config is None:
                self._db_config = DBConfig.from_env()
            self._store = MySQLVectorStore(self._db_config)
            self._store.init_table()
        return self._store

    def _get_case_card_store(self):
        """懒初始化 MySQLCaseCardStore"""
        if self._case_card_store is None:
            from case_card_store import MySQLCaseCardStore
            if self._db_config is None:
                self._db_config = DBConfig.from_env()
            self._case_card_store = MySQLCaseCardStore(self._db_config)
            self._case_card_store.init_table()
        return self._case_card_store

    def _get_embedding_service(self) -> Optional[EmbeddingService]:
        """懒初始化 EmbeddingService"""
        if self._embedding_service is None and self.enable_llm:
            cfg = EmbeddingConfig()
            if cfg.is_configured:
                self._embedding_service = EmbeddingService(cfg)
        return self._embedding_service

    def _get_llm_extractor(self):
        """懒初始化 LLMCaseExtractor"""
        if self._llm_extractor is None and self.enable_llm:
            from llm_case_extractor import LLMCaseExtractor
            from config import LLMConfig
            cfg = LLMConfig()
            if cfg.is_configured:
                self._llm_extractor = LLMCaseExtractor(cfg)
        return self._llm_extractor

    def _init_default_vocabulary(self) -> None:
        """初始化向量提取器（词袋模型不需要训练）"""
        self.extractor.fit([])
        self._extractor_fitted = True

    def add_record(self, record_id: str, text: str,
                   parsed_record: MedicalRecord = None,
                   features: np.ndarray = None,
                   timeline_features: object = None) -> None:
        """
        添加病例到检索系统

        Args:
            record_id: 病例唯一ID
            text: 病历文本
            parsed_record: 预解析的病历（可选，避免重复解析）
            features: 预提取的特征向量（可选，避免重复提取）
            timeline_features: 预计算的时间轴特征（可选，避免重复计算）
        """
        if parsed_record is None:
            parsed_record = self.parser.parse(text)
        if features is None:
            features = self.extractor.extract(parsed_record)
        if timeline_features is None:
            timeline_events = self.timeline_parser.parse(text)
            standard_nodes = self.timeline_parser.generate_standard_nodes(timeline_events)
            timeline_features = self.timeline_scorer.extract_features(
                timeline_events, standard_nodes,
                surgery_type_hint=parsed_record.surgery_type or None,
                surgery_keywords_hint=parsed_record.surgery_keywords or None
            )

        # 添加到内存索引
        self.index.add(features.reshape(1, -1))

        # 写入 MySQL
        store = self._get_store()
        order_idx = len(self.record_order)
        store.insert(record_id, features, text, timeline_features, order_idx)

        # 更新缓存
        self._metadata_cache[record_id] = {
            'text': text,
            'timeline_features': timeline_features,
        }
        self.record_order.append(record_id)

    def add_records_batch(self, records: Dict[str, str], num_workers: int = 1,
                          executor: ProcessPoolExecutor = None) -> None:
        """
        批量添加病例到检索系统

        Args:
            records: {record_id: text} 映射
            num_workers: 并行进程数（1=单进程，>1=多进程并行解析和时间轴提取）
            executor: 可选的外部 ProcessPoolExecutor（复用避免重复 spawn 进程）
        """
        if not records:
            return

        records_list = list(records.items())
        n = len(records_list)
        workers = min(max(num_workers, 1), n)
        _use_external = executor is not None and workers > 1

        # ── 阶段1：解析病历 ──
        if _use_external:
            print(f"  [阶段1/4] 并行解析 {n} 条病历 (workers={workers}) ...")
            parsed_results = list(executor.map(_worker_parse_record, records_list))
            parsed_records = [p for _, _, p in parsed_results]
            records_list = [(rid, txt) for rid, txt, _ in parsed_results]
        elif workers > 1:
            print(f"  [阶段1/4] 并行解析 {n} 条病历 (workers={workers}) ...")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                parsed_results = list(pool.map(_worker_parse_record, records_list))
            parsed_records = [p for _, _, p in parsed_results]
            records_list = [(rid, txt) for rid, txt, _ in parsed_results]
        else:
            parsed_records = []
            for record_id, text in records_list:
                parsed_records.append(self.parser.parse(text))

        # IDF 拟合 + 返回 raw_vectors（避免 extract_batch 中重复 _extract_raw_vector）
        raw_vectors = self.extractor.fit(parsed_records)

        # ── 阶段2：特征提取（复用 raw_vectors）──
        features = self.extractor.extract_batch(parsed_records, raw_vectors=raw_vectors)

        if features.size == 0:
            return

        # ── 阶段3：向量入内存索引 ──
        self.index.add(features)

        # ── 阶段4：时间轴预计算 ──
        if workers > 1:
            timeline_inputs = [
                (rid, txt,
                 parsed_records[i].surgery_type or None,
                 parsed_records[i].surgery_keywords or None)
                for i, (rid, txt) in enumerate(records_list)
            ]
            print(f"  [阶段4/4] 并行提取时间轴 (workers={workers}) ...")
            if _use_external:
                timeline_results = list(executor.map(_worker_parse_timeline, timeline_inputs))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    timeline_results = list(pool.map(_worker_parse_timeline, timeline_inputs))
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

        # 逐条写入 MySQL（复用同一连接，避免重复建连）
        # 注意：build 阶段不填充 _metadata_cache（内存膨胀会导致 fork 越来越慢）
        # 缓存由 search 启动时通过 load_all_metadata() 从 MySQL 加载
        store = self._get_store()
        base_order = len(self.record_order)
        insert_rows = []
        for i, (record_id, text) in enumerate(records_list):
            tl_feat = timeline_map[record_id][2]
            insert_rows.append((record_id, features[i], text, tl_feat, base_order + i))
            self.record_order.append(record_id)
        store.insert_sequential(insert_rows)

    def save(self) -> None:
        """手动保存索引和元数据到磁盘"""
        self._save_index()

    def search(self,
               query_text: str,
               top_k: int = 10,
               auto_add: bool = False,
               record_id: Optional[str] = None,
               exclude_record_ids: Optional[set] = None,
               timeline_days: int = 0,
               timeline_window_weight: float = 0.55) -> List[Dict]:
        """
        检索相似病例

        Args:
            query_text: 待查询病历文本
            top_k: 返回前K个最相似病例
            auto_add: 是否自动入库
            record_id: 病例ID（auto_add=True时必填）
            exclude_record_ids: 需要排除的病例ID集合（如查询病例自身）
            timeline_days: 病程窗口天数；0=完整住院病程；N>0=额外比较入院后前N天
            timeline_window_weight: 窗口病程分在混合病程分中的权重（0~1，默认0.55）

        Returns:
            [{'id': xxx, 'similarity': 0.85, 'text': xxx}, ...]
        """
        if timeline_days < 0:
            raise ValueError("timeline_days 不能小于 0")
        if not 0.0 <= timeline_window_weight <= 1.0:
            raise ValueError("timeline_window_weight 必须在 0~1 之间")

        if auto_add:
            return self.search_and_add(query_text, record_id, top_k, exclude_record_ids,
                                       timeline_days, timeline_window_weight)

        results, _, _, _ = self._search_only(query_text, top_k, exclude_record_ids,
                                              timeline_days, timeline_window_weight)
        return results

    def _search_only(self, query_text: str, top_k: int,
                     exclude_record_ids: Optional[set] = None,
                     timeline_days: int = 0,
                     timeline_window_weight: float = 0.55) -> Tuple[List[Dict], MedicalRecord, np.ndarray, object]:
        """仅检索，不入库。返回 (results, parsed_record, features, timeline_features) 供 search_and_add 复用"""
        if not self.record_order:
            return [], None, None, None

        exclude_record_ids = self._normalize_record_ids(exclude_record_ids or set())

        # 解析查询病历
        query_record = self.parser.parse(query_text)
        query_features = self.extractor.extract(query_record).reshape(1, -1)
        query_summary_vec = _char_ngram_vector(_extract_summary_text(query_text))

        # 解析查询病历的时间轴
        query_events = self.timeline_parser.parse(query_text)
        query_nodes = self.timeline_parser.generate_standard_nodes(query_events)
        query_full_timeline_features = self.timeline_scorer.extract_features(
            query_events, query_nodes,
            surgery_type_hint=query_record.surgery_type or None,
            surgery_keywords_hint=query_record.surgery_keywords or None
        )

        # 窗口病程特征（仅在指定天数时计算）
        query_window_timeline_features = None
        if timeline_days > 0:
            query_window_events = self.timeline_parser.get_first_days_snapshot(
                query_events, timeline_days
            )
            query_window_nodes = self.timeline_parser.generate_standard_nodes(
                query_window_events
            )
            query_window_timeline_features = self.timeline_scorer.extract_features(
                query_window_events, query_window_nodes,
                surgery_type_hint=query_record.surgery_type or None,
                surgery_keywords_hint=query_record.surgery_keywords or None
            )

        # ── LLM 增强：查询病历实时抽取病例卡 + embedding ──
        query_card = None
        query_embedding = None

        # 拆分为两个独立开关：有病例卡缓存  vs  有 embedding 缓存
        has_case_cards = self.enable_llm and bool(self._case_card_cache)
        has_embeddings = self.enable_llm and self._embedding_cache is not None and len(self._embedding_ids) > 0

        if self.enable_llm:
            extractor = self._get_llm_extractor()
            emb_service = self._get_embedding_service()
            if extractor and extractor.is_available:
                try:
                    query_card = extractor.extract(query_text, record_id="__query__")
                except Exception as e:
                    print(f"  查询病例卡抽取失败: {e}，回退到原融合公式")
                    query_card = None

            if query_card and emb_service and emb_service.is_available:
                try:
                    summary = query_card.get("summary_for_embedding", "")
                    if summary:
                        query_embedding = emb_service.embed_text(summary)
                        # ── P0-2: 查询 embedding 归一化 ──
                        query_embedding = query_embedding.astype(np.float64)
                        norm = np.linalg.norm(query_embedding)
                        if norm > 0:
                            query_embedding = query_embedding / norm
                except Exception as e:
                    print(f"  查询 embedding 生成失败: {e}")
                    query_embedding = None

        # ── 第一阶段：多路候选召回 ──
        # 1. 结构化向量候选
        vector_top_k = min(max(top_k * 20, 100), len(self.record_order))
        distances, indices = self.index.search(query_features, vector_top_k)

        vector_candidate_indices = set()
        vector_score_by_id = {}
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.record_order):
                rid = self.record_order[idx]
                vector_candidate_indices.add(rid)
                vector_score_by_id[rid] = self._distance_to_similarity(dist)

        # 2. embedding 候选（P0-1: embedding 参与召回）
        embedding_candidate_ids = set()
        embedding_score_by_id = {}
        if has_embeddings and query_embedding is not None:
            emb_scores = np.dot(self._embedding_cache, query_embedding)
            emb_top_k = min(max(top_k * 20, 100), len(self._embedding_ids))
            emb_top_indices = np.argsort(-emb_scores)[:emb_top_k]
            for ei in emb_top_indices:
                emb_id = self._embedding_ids[ei]
                embedding_candidate_ids.add(emb_id)
                embedding_score_by_id[emb_id] = float(emb_scores[ei])

        # 3. 合并候选集合（并集）
        candidate_ids = set(vector_candidate_indices)
        candidate_ids.update(embedding_candidate_ids)
        candidate_ids -= exclude_record_ids

        # ── 第二阶段：多维度重排序 ──
        candidates = []
        for record_id in candidate_ids:
            # 结构化向量分：候选来自 embedding 但没有向量分时用 0.0
            vector_sim = vector_score_by_id.get(record_id, 0.0)

            cand_data = self._metadata_cache.get(record_id, {})
            text = cand_data.get('text', '')
            cand_full_features = self._timeline_features_for_candidate(
                record_id, text, cand_data.get('timeline_features')
            )

            # 完整病程相似度
            if cand_full_features is not None:
                full_timeline_sim = self.timeline_scorer.score(
                    query_full_timeline_features, cand_full_features
                )
            else:
                full_timeline_sim = vector_sim

            # 窗口病程相似度
            window_timeline_sim_val = None
            timeline_sim = full_timeline_sim
            timeline_window_fallback = False
            MIN_WINDOW_EVENTS = 2

            if timeline_days > 0 and query_window_timeline_features is not None:
                cand_window_features = self._timeline_window_features_for_candidate(
                    record_id, text, timeline_days
                )
                if cand_window_features is not None:
                    # 检查窗口事件数量是否足够（用独立 event_count 而非压缩后的 event_type_sequence）
                    q_count = query_window_timeline_features.event_count
                    c_count = cand_window_features.event_count
                    if q_count >= MIN_WINDOW_EVENTS and c_count >= MIN_WINDOW_EVENTS:
                        window_timeline_sim_val = self.timeline_scorer.score(
                            query_window_timeline_features, cand_window_features
                        )
                        timeline_sim = (
                            timeline_window_weight * window_timeline_sim_val
                            + (1.0 - timeline_window_weight) * full_timeline_sim
                        )
                    else:
                        timeline_window_fallback = True
                        window_timeline_sim_val = None
                        timeline_sim = full_timeline_sim

            text_sim = self._text_similarity(record_id, text, query_summary_vec)

            # ── LLM 增强相似度 ──
            embedding_sim = 0.0
            tag_overlap_sim = 0.0

            # Embedding 相似度（P0-2: 查询已归一化，候选也预归一化，点积即余弦）
            if has_embeddings and query_embedding is not None:
                if record_id in embedding_score_by_id:
                    # 从缓存取（embedding 召回阶段已计算）
                    embedding_sim = embedding_score_by_id[record_id]
                elif record_id in self._embedding_id_to_idx:
                    emb_idx = self._embedding_id_to_idx[record_id]
                    cand_emb = self._embedding_cache[emb_idx]
                    embedding_sim = float(np.dot(cand_emb, query_embedding))
                else:
                    # 候选无 embedding，回退到字符 n-gram
                    embedding_sim = text_sim
                embedding_sim = min(1.0, max(0.0, embedding_sim))
            else:
                # 无 embedding，使用字符 n-gram
                embedding_sim = text_sim

            # 标签重叠相似度（P1-6: 与 embedding 解耦，独立判断）
            if has_case_cards and query_card is not None:
                cand_card = self._case_card_cache.get(record_id)
                if cand_card:
                    tag_overlap_sim = tag_overlap_score(query_card, cand_card)

            # ── 融合公式（P1-6: 按可用信号灵活组合）──
            use_embedding = has_embeddings and query_embedding is not None
            use_tag = has_case_cards and query_card is not None

            if use_embedding and use_tag:
                # 完整四路融合 (v1.1: timeline 0.40→0.30, tag 0.10→0.20)
                base_sim = (
                    0.15 * vector_sim +
                    0.30 * timeline_sim +
                    0.35 * embedding_sim +
                    0.20 * tag_overlap_sim
                )
            elif use_embedding:
                # 有 embedding 但无标签
                base_sim = (
                    0.20 * vector_sim +
                    0.35 * timeline_sim +
                    0.45 * embedding_sim
                )
            elif use_tag:
                # 有病例卡标签但无 embedding
                base_sim = (
                    0.20 * vector_sim +
                    0.35 * timeline_sim +
                    0.20 * text_sim +
                    0.25 * tag_overlap_sim
                )
            else:
                # 原始三路融合（保持现有逻辑，避免影响非 LLM 模式）
                base_sim = (
                    0.30 * vector_sim +
                    0.45 * timeline_sim +
                    0.25 * text_sim
                )

            if timeline_sim < self.min_timeline_score:
                base_sim *= 0.5

            # ── v1.1 排序修正：disease_axis 软门控 ──
            ranking_bonus = 0.0
            ranking_penalty = 0.0
            axis_conflict = False
            axis_sim = 0.5  # 默认中性

            if use_tag and query_card:
                cand_card = self._case_card_cache.get(record_id)
                if cand_card:
                    axis_sim = disease_axis_similarity(query_card, cand_card)
                    axis_conflict = has_disease_axis_conflict(query_card, cand_card)

                    if axis_conflict:
                        ranking_penalty += 0.08

                    if axis_sim == 1.0:
                        ranking_bonus += 0.03

                    if tag_overlap_sim == 0.0 and not has_strong_common_tag(query_card, cand_card):
                        ranking_penalty += 0.04

                    # P0: 仅有泛化共同标签，无诊断/特异干预交集
                    if tag_overlap_sim > 0 and not has_strong_common_tag(query_card, cand_card):
                        ranking_penalty += 0.02

            final_sim = max(0.0, min(1.0, base_sim + ranking_bonus - ranking_penalty))

            if final_sim >= self.threshold:
                result = {
                    'id': record_id,
                    'similarity': round(final_sim, 4),
                    'vector_similarity': round(vector_sim, 4),
                    'timeline_similarity': round(timeline_sim, 4),
                    'text_similarity': round(text_sim, 4),
                    'text': text[:200] + "..." if len(text) > 200 else text,
                    'full_text': text,
                    # 窗口病程字段
                    'full_timeline_similarity': round(full_timeline_sim, 4),
                    'window_timeline_similarity': round(window_timeline_sim_val, 4) if window_timeline_sim_val is not None else '-',
                    'timeline_window_days': timeline_days,
                    'timeline_window_weight': timeline_window_weight if timeline_days > 0 else 0.0,
                }
                if timeline_window_fallback:
                    result['timeline_window_fallback'] = True

                # LLM 增强时附加额外字段
                if use_tag or use_embedding:
                    result['embedding_similarity'] = round(embedding_sim, 4)
                    result['tag_overlap_similarity'] = round(tag_overlap_sim, 4)
                    # v1.1 排序解释字段
                    result['base_similarity'] = round(base_sim, 4)
                    result['ranking_bonus'] = round(ranking_bonus, 4)
                    result['ranking_penalty'] = round(ranking_penalty, 4)
                    result['disease_axis_similarity'] = round(axis_sim, 4)
                    result['disease_axis_conflict'] = axis_conflict
                    # 相似原因
                    if query_card:
                        cand_card = self._case_card_cache.get(record_id)
                        if cand_card:
                            result['similarity_reasons'] = find_common_and_diff_tags(query_card, cand_card)
                            result['strong_common_tags'] = find_strong_common_tags(query_card, cand_card)
                else:
                    result['embedding_similarity'] = '-'
                    result['tag_overlap_similarity'] = '-'
                    result['base_similarity'] = round(base_sim, 4)
                    result['ranking_bonus'] = 0.0
                    result['ranking_penalty'] = 0.0
                    result['disease_axis_similarity'] = '-'
                    result['disease_axis_conflict'] = False

                candidates.append(result)

        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_k], query_record, query_features, query_full_timeline_features

    def search_and_add(self,
                       query_text: str,
                       record_id: Optional[str] = None,
                       top_k: int = 10,
                       exclude_record_ids: Optional[set] = None,
                       timeline_days: int = 0,
                       timeline_window_weight: float = 0.55) -> List[Dict]:
        """
        检索相似病例 + 自动入库（复用 _search_only 的解析结果，避免重复 parse）
        """
        # 1. 检索（保留中间解析结果）
        results, query_record, query_features, query_timeline_features = \
            self._search_only(query_text, top_k, exclude_record_ids,
                            timeline_days, timeline_window_weight)

        # 2. 自动入库（复用解析结果，避免重复 parse/extract/timeline）
        if record_id is None:
            record_id = f"record_{self.record_count}"
        self.record_count += 1

        self.add_record(record_id, query_text,
                        parsed_record=query_record,
                        features=query_features,
                        timeline_features=query_timeline_features)

        return results

    def _distance_to_similarity(self, distance: float) -> float:
        """
        将距离转换为相似度
        使用余弦相似度: similarity = 1 - distance
        """
        return max(0.0, 1.0 - distance)

    @staticmethod
    def _normalize_record_ids(record_ids: set) -> set:
        normalized = set()
        for rid in record_ids:
            if not rid:
                continue
            rid_str = str(rid)
            normalized.add(rid_str)
            normalized.add(os.path.splitext(rid_str)[0])
        return normalized

    def _text_similarity(self, record_id: str, text: str, query_summary_vec: Counter) -> float:
        if not text:
            return 0.0
        cand_vec = self._summary_vector_cache.get(record_id)
        if cand_vec is None:
            cand_vec = _char_ngram_vector(_extract_summary_text(text))
            self._summary_vector_cache[record_id] = cand_vec
        return _counter_cosine(query_summary_vec, cand_vec)

    def _timeline_features_for_candidate(self, record_id: str, text: str, existing_features):
        """补齐旧索引中缺失的新版病程特征，避免不重建索引时评分失真。"""
        if existing_features is None or not text:
            return existing_features

        has_new_fields = any((
            getattr(existing_features, 'intervention_keywords', None),
            getattr(existing_features, 'complication_keywords', None),
            getattr(existing_features, 'severity_keywords', None),
        ))
        if has_new_fields or record_id in self._timeline_refreshed_ids:
            return existing_features

        events = self.timeline_parser.parse(text)
        nodes = self.timeline_parser.generate_standard_nodes(events)
        parsed = self.parser.parse(text)
        refreshed = self.timeline_scorer.extract_features(
            events, nodes,
            surgery_type_hint=parsed.surgery_type or None,
            surgery_keywords_hint=parsed.surgery_keywords or None
        )
        self._metadata_cache.setdefault(record_id, {})['timeline_features'] = refreshed
        self._timeline_refreshed_ids.add(record_id)
        return refreshed

    def _timeline_window_features_for_candidate(
        self,
        record_id: str,
        text: str,
        timeline_days: int,
    ):
        """按窗口动态计算候选病例的时间轴特征（带缓存）"""
        if timeline_days <= 0 or not text:
            return None

        cache_key = (record_id, timeline_days)
        cached = self._timeline_window_feature_cache.get(cache_key)
        if cached is not None:
            return cached

        events = self.timeline_parser.parse(text)
        window_events = self.timeline_parser.get_first_days_snapshot(events, timeline_days)
        nodes = self.timeline_parser.generate_standard_nodes(window_events)
        parsed = self.parser.parse(text)
        features = self.timeline_scorer.extract_features(
            window_events, nodes,
            surgery_type_hint=parsed.surgery_type or None,
            surgery_keywords_hint=parsed.surgery_keywords or None,
        )
        self._timeline_window_feature_cache[cache_key] = features
        return features

    def _save_index(self) -> None:
        """保存轻量配置文件（向量数据已通过 MySQL 持久化）"""
        if not self.index_path:
            return

        config = {
            'threshold': self.threshold,
            'feature_dim': self.feature_dim,
            'record_count': self.record_count,
            'alpha': self.alpha,
            'min_timeline_score': self.min_timeline_score,
        }
        with open(self.index_path + ".json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _load_index(self) -> None:
        """从 MySQL 加载向量和元数据，构建内存索引"""
        store = self._get_store()

        # 加载向量矩阵
        vectors, ids = store.load_all_vectors()
        if vectors.size == 0:
            return

        # 构建内存索引
        self.index = create_index(self.feature_dim, "sklearn")
        self.index.add(vectors)

        # 加载元数据缓存
        self._metadata_cache = store.load_all_metadata()
        self.record_order = ids
        self.record_count = len(ids)

        print(f"从 MySQL 加载了 {len(ids)} 条记录")

        # 如果启用 LLM，加载病例卡和 embedding
        if self.enable_llm:
            self._load_case_cards()

    def _load_case_cards(self) -> None:
        """加载病例卡和 embedding 到内存缓存"""
        try:
            cc_store = self._get_case_card_store()
            card_count = cc_store.count()
            if card_count == 0:
                print("病例卡表为空，LLM 增强功能不可用")
                self.enable_llm = False
                return

            # 加载病例卡（P1-6: 与 embedding 解耦，独立加载）
            self._case_card_cache = cc_store.load_all()
            print(f"加载了 {len(self._case_card_cache)} 条病例卡")

            # 加载 embedding 向量
            emb_vectors, emb_ids = cc_store.load_all_embeddings()
            if emb_vectors.size > 0:
                self._embedding_cache = emb_vectors
                self._embedding_ids = emb_ids
                self._embedding_id_to_idx = {rid: i for i, rid in enumerate(emb_ids)}
                print(f"加载了 {len(emb_ids)} 条 embedding 向量 (维度={emb_vectors.shape[1]})")

                # 预归一化所有候选 embedding，加速后续余弦相似度计算
                norms = np.linalg.norm(emb_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._embedding_cache = emb_vectors / norms
            else:
                # P1-6: embedding 缺失不阻止标签融合
                print("embedding 向量为空，标签重叠相似度仍可用（如已加载病例卡）")
        except Exception as e:
            print(f"加载病例卡失败: {e}，LLM 增强功能不可用")
            self.enable_llm = False

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'total_records': len(self.record_order),
            'threshold': self.threshold,
            'feature_dim': self.feature_dim,
            'index_type': type(self.index).__name__
        }

    def set_threshold(self, threshold: float) -> None:
        """设置相似度阈值"""
        self.threshold = threshold


def create_system(data_dir: str = "./data",
                  threshold: float = 0.45,
                  db_config: Optional[DBConfig] = None,
                  enable_llm: bool = False) -> MedicalRecordSimilaritySystem:
    """
    工厂函数：创建检索系统

    Args:
        data_dir: 数据存储目录
        threshold: 相似度阈值
        db_config: MySQL 连接配置（若不提供则从环境变量读取）
        enable_llm: 是否启用 LLM 病例卡 + embedding 增强

    Returns:
        MedicalRecordSimilaritySystem 实例
    """
    os.makedirs(data_dir, exist_ok=True)
    index_path = os.path.join(data_dir, "similarity_index")

    if db_config is None:
        db_config = DBConfig.from_env()

    return MedicalRecordSimilaritySystem(
        similarity_threshold=threshold,
        index_path=index_path,
        db_config=db_config,
        enable_llm=enable_llm,
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
