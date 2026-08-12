"""医生关注文本驱动的天级序列检索。"""

from __future__ import annotations

import logging
import re
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from candidate_retriever import (
    InMemoryVectorCandidateRetriever,
    LegacyFullScanCandidateRetriever,
)
from case_card_embedding import build_case_card_embedding_text
from clinical_text_filter import ClinicalDayRecord, build_clinical_day_records
from config import DBConfig, EmbeddingConfig, LLMConfig, RerankerConfig, RetrievalConfig
from day_store import DEFAULT_DAY_EXTRACTOR_VERSION, MySQLDayStore
from embedding_index import EmbeddingService
from llm_case_extractor import LLMCaseExtractor
from llm_reranker import HTTPReranker, LLMReranker

logger = logging.getLogger(__name__)


class LRUCache:
    """有界 LRU 缓存，用于 n-gram 向量缓存。"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


def _vector_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return None
    return float(np.dot(a, b) / (norm_a * norm_b))


def _char_ngram_vector(text: str, ngram_range: Tuple[int, int] = (2, 4)) -> Counter:
    compact = re.sub(r"\s+", "", text or "")
    vec = Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            vec[compact[i:i + n]] += 1
    return vec


def _counter_cosine(a: Counter, b: Counter) -> Optional[float]:
    if not a or not b:
        return None
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return None
    return float(dot / (norm_a * norm_b))


def _normalize_embedding(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    emb = value.astype(np.float64)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _collect_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_collect_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_collect_text(v) for v in value)
    return str(value)


def _day_card_tags(card: Optional[dict]) -> set:
    if not card:
        return set()
    tags = set()
    for key in (
        "final_diagnoses",
        "primary_diagnosis_axis",
        "etiology_axis",
        "baseline_context",
        "new_diagnoses",
        "new_interventions",
        "operations",
        "organ_status",
        "complications",
        "etiology_chain",
    ):
        values = _as_list(card.get(key))
        for value in values:
            if isinstance(value, str):
                tags.add(value.strip())
            elif isinstance(value, dict):
                for sub_key in (
                    "name",
                    "label",
                    "type",
                    "status",
                    "disease_category",
                    "direct_cause",
                    "source_or_site",
                    "anatomic_location",
                    "underlying_trigger",
                    "certainty",
                ):
                    sub_val = value.get(sub_key)
                    if sub_val:
                        tags.add(str(sub_val).strip())
                for sub_key in ("pathophysiology", "key_interventions"):
                    for sub_val in _as_list(value.get(sub_key)):
                        if sub_val:
                            tags.add(str(sub_val).strip())
    state = card.get("clinical_state")
    if state:
        tags.add(f"state:{state}")
    return {tag for tag in tags if tag}


ETIOLOGY_CHAIN_RULES = {
    "source_or_site": {
        "urinary": ("泌尿", "尿路", "尿源", "肾盂", "输尿管", "膀胱", "尿液", "尿培养"),
        "pulmonary": ("肺部感染", "肺炎", "吸入", "痰培养", "呼吸道感染"),
        "abdominal_biliary": ("腹腔", "胆道", "胆囊", "胆管", "胰腺", "肠", "消化道", "腹膜炎"),
        "catheter": ("导管", "中心静脉", "PICC", "置管", "导管相关"),
        "skin_soft_tissue": ("皮肤", "软组织", "创面", "坏死性筋膜炎", "蜂窝织炎"),
        "cns": ("颅内感染", "脑膜炎", "中枢神经"),
    },
    "underlying_trigger": {
        "stone_obstruction": ("结石", "梗阻", "积水", "输尿管支架", "DJ管", "解除梗阻"),
        "post_operation": ("术后", "手术后", "围手术期"),
        "trauma": ("外伤", "创伤", "车祸", "坠落", "摔伤"),
        "malignancy": ("肿瘤", "癌", "恶性"),
        "immunosuppression": ("免疫抑制", "化疗", "激素", "移植"),
        "coronary_plaque": ("冠心病", "冠脉", "斑块", "PCI", "支架", "心肌梗死", "心梗"),
        "embolism": ("肺栓塞", "血栓", "栓塞", "D-二聚体"),
        "dissection": ("主动脉夹层", "夹层"),
    },
    "pathophysiology": {
        "septic_shock": ("感染性休克", "脓毒性休克", "脓毒症休克"),
        "sepsis": ("脓毒症", "败血症", "严重感染"),
        "cardiogenic_shock": ("心源性休克", "泵衰竭"),
        "cardiac_arrest": ("心脏骤停", "心跳骤停", "心肺复苏", "CPR"),
        "respiratory_failure": ("呼吸衰竭", "低氧", "氧合"),
        "renal_failure": ("肾功能不全", "肾衰", "急性肾损伤", "AKI"),
    },
    "key_interventions": {
        "source_control_urinary": ("输尿管支架", "DJ管", "经皮肾", "造瘘", "碎石", "解除梗阻"),
        "pci": ("PCI", "冠脉造影", "支架植入", "球囊扩张"),
        "thrombolysis_anticoagulation": ("溶栓", "抗凝", "肝素", "利伐沙班"),
        "mechanical_ventilation": ("机械通气", "气管插管", "呼吸机"),
        "vasopressor": ("去甲肾上腺素", "升压", "血管活性"),
        "crrt": ("CRRT", "血滤", "透析"),
    },
}


def _rule_labels(text: str, group: str) -> set:
    result = set()
    for label, keywords in ETIOLOGY_CHAIN_RULES.get(group, {}).items():
        if any(keyword in text for keyword in keywords):
            result.add(label)
    return result


def _etiology_chain_labels(card: Optional[dict]) -> Dict[str, set]:
    if not card:
        return {}
    chain = card.get("etiology_chain") if isinstance(card.get("etiology_chain"), dict) else {}
    text = " ".join([
        _collect_text(chain),
        _collect_text(card.get("primary_diagnosis_axis")),
        _collect_text(card.get("etiology_axis")),
        _collect_text(card.get("final_diagnoses")),
        _collect_text(card.get("new_diagnoses")),
        _collect_text(card.get("day_summary")),
        _collect_text(card.get("operations")),
        _collect_text(card.get("new_interventions")),
    ])
    labels = {
        "disease_category": set(),
        "source_or_site": _rule_labels(text, "source_or_site"),
        "underlying_trigger": _rule_labels(text, "underlying_trigger"),
        "pathophysiology": _rule_labels(text, "pathophysiology"),
        "key_interventions": _rule_labels(text, "key_interventions"),
    }
    disease_text = " ".join([
        _collect_text(chain.get("disease_category")),
        _collect_text(card.get("primary_diagnosis_axis")),
        _collect_text(card.get("etiology_axis")),
    ])
    for axis, keywords in DIAGNOSIS_AXIS_RULES.items():
        if any(keyword in disease_text or keyword in text for keyword in keywords):
            labels["disease_category"].add(axis)
    for key in ("disease_category", "direct_cause", "source_or_site", "anatomic_location", "underlying_trigger"):
        value = chain.get(key)
        if value:
            labels.setdefault(key, set()).add(str(value).strip())
    for key in ("pathophysiology", "key_interventions"):
        for value in _as_list(chain.get(key)):
            if value:
                labels.setdefault(key, set()).add(str(value).strip())
    return {key: {item for item in values if item} for key, values in labels.items()}


def _display_etiology_chain_labels(card: Optional[dict]) -> Dict[str, List[str]]:
    labels = _etiology_chain_labels(card)
    return {key: sorted(values) for key, values in labels.items() if values}


def _set_similarity(query_values: set, cand_values: set) -> Optional[float]:
    if not query_values and not cand_values:
        return None
    if not query_values or not cand_values:
        return 0.0
    if query_values & cand_values:
        return len(query_values & cand_values) / len(query_values | cand_values)
    return 0.0


def _etiology_chain_similarity(query_card: Optional[dict], cand_card: Optional[dict]) -> Optional[float]:
    q = _etiology_chain_labels(query_card)
    c = _etiology_chain_labels(cand_card)
    if not q or not c:
        return None
    weighted_parts = []
    for key, weight in (
        ("disease_category", 0.15),
        ("source_or_site", 0.25),
        ("direct_cause", 0.10),
        ("anatomic_location", 0.07),
        ("underlying_trigger", 0.20),
        ("pathophysiology", 0.15),
        ("key_interventions", 0.08),
    ):
        sim = _set_similarity(q.get(key, set()), c.get(key, set()))
        if sim is not None:
            weighted_parts.append((weight, sim))
    if not weighted_parts:
        return None
    total = sum(weight for weight, _ in weighted_parts)
    return float(sum(weight * score for weight, score in weighted_parts) / total)


def _has_etiology_chain_conflict(query_card: Optional[dict], cand_card: Optional[dict]) -> bool:
    q = _etiology_chain_labels(query_card)
    c = _etiology_chain_labels(cand_card)
    if not q or not c:
        return False
    q_axis = _diagnosis_axis(query_card)
    c_axis = _diagnosis_axis(cand_card)
    if q_axis and c_axis and q_axis != c_axis:
        return True
    for key in ("source_or_site", "underlying_trigger"):
        q_values = q.get(key, set())
        c_values = c.get(key, set())
        if q_values and c_values and not (q_values & c_values):
            return True
    return False


DIAGNOSIS_AXIS_RULES = {
    "coronary_ami": ("冠心病", "心肌梗死", "心梗", "STEMI", "NSTEMI", "急性冠脉", "冠脉"),
    "pulmonary_embolism": ("肺栓塞", "肺动脉栓塞", "PTE", "PE"),
    "aortic_dissection": ("主动脉夹层", "夹层动脉瘤"),
    "sepsis": ("脓毒症", "感染性休克", "严重感染", "败血症"),
    "heart_failure": ("心力衰竭", "心衰"),
    "arrhythmia": ("室颤", "室速", "恶性心律失常", "心律失常"),
}


def _diagnosis_axis(card: Optional[dict]) -> str:
    if not card:
        return ""
    values = []
    for key in ("primary_diagnosis_axis", "etiology_axis", "final_diagnoses", "new_diagnoses", "day_summary"):
        value = card.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    text = " ".join(values)
    for axis, keywords in DIAGNOSIS_AXIS_RULES.items():
        if any(keyword in text for keyword in keywords):
            return axis
    raw_axis = (card.get("primary_diagnosis_axis") or card.get("etiology_axis") or "").strip()
    return raw_axis[:40]


def _diagnosis_axis_similarity(query_card: Optional[dict], cand_card: Optional[dict]) -> Optional[float]:
    q_axis = _diagnosis_axis(query_card)
    c_axis = _diagnosis_axis(cand_card)
    if not q_axis or not c_axis:
        return None
    if q_axis == c_axis:
        return 1.0
    q_tags = _day_card_tags(query_card)
    c_tags = _day_card_tags(cand_card)
    overlap = _jaccard(q_tags, c_tags)
    return min(0.35, overlap or 0.0)


def _has_diagnosis_axis_conflict(query_card: Optional[dict], cand_card: Optional[dict]) -> bool:
    q_axis = _diagnosis_axis(query_card)
    c_axis = _diagnosis_axis(cand_card)
    return bool(q_axis and c_axis and q_axis != c_axis)


def _diagnosis_conflict_summary(details: List[dict]) -> dict:
    """汇总一个候选患者在逐日对齐中的诊断/病因轴冲突。"""
    conflicts = [item for item in details if item.get("diagnosis_axis_conflict")]
    etiology_conflicts = [item for item in details if item.get("etiology_chain_conflict")]
    compared = [
        item for item in details
        if item.get("query_diagnosis_axis") and item.get("matched_diagnosis_axis")
    ]
    etiology_compared = [
        item for item in details
        if item.get("etiology_chain_similarity") != "-"
    ]
    return {
        "diagnosis_conflict_days": [item.get("query_day") for item in conflicts],
        "diagnosis_conflict_count": len(conflicts),
        "diagnosis_compared_days": len(compared),
        "diagnosis_conflict_ratio": (len(conflicts) / len(compared)) if compared else 0.0,
        "etiology_chain_conflict_days": [item.get("query_day") for item in etiology_conflicts],
        "etiology_chain_conflict_count": len(etiology_conflicts),
        "etiology_chain_compared_days": len(etiology_compared),
        "etiology_chain_conflict_ratio": (
            len(etiology_conflicts) / len(etiology_compared)
        ) if etiology_compared else 0.0,
        "first_day_diagnosis_conflict": any(
            item.get("query_day") == 1 for item in conflicts
        ),
        "first_day_etiology_chain_conflict": any(
            item.get("query_day") == 1 for item in etiology_conflicts
        ),
    }


def _passes_diagnosis_gate(conflict_summary: dict) -> bool:
    """宁缺毋滥：诊断/病因轴明显不一致时，不为了凑满 Top5 返回。"""
    diagnosis_compared = conflict_summary.get("diagnosis_compared_days", 0)
    etiology_compared = conflict_summary.get("etiology_chain_compared_days", 0)
    if diagnosis_compared == 0 and etiology_compared == 0:
        return True
    if conflict_summary.get("first_day_diagnosis_conflict"):
        return False
    if conflict_summary.get("first_day_etiology_chain_conflict"):
        return False
    if conflict_summary.get("diagnosis_conflict_count", 0) >= 2:
        return False
    if conflict_summary.get("etiology_chain_conflict_count", 0) >= 2:
        return False
    if conflict_summary.get("diagnosis_conflict_ratio", 0.0) > 0.20:
        return False
    if conflict_summary.get("etiology_chain_conflict_ratio", 0.0) > 0.34:
        return False
    return True


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DayLevelRetrievalSystem:
    """按患者每日记录序列进行相似度检索。

    Stage 5: 使用 CandidateRetriever 进行候选预筛，仅对候选患者执行完整医学精算。
    """

    def __init__(self, db_config: DBConfig):
        self.db_config = db_config
        self.day_store = MySQLDayStore(db_config)
        self.day_store.init_tables()
        self.llm_extractor = LLMCaseExtractor(LLMConfig())
        self.http_reranker = HTTPReranker(RerankerConfig())
        self.llm_reranker = LLMReranker(LLMConfig())
        self.emb_service = EmbeddingService(EmbeddingConfig())
        self.retrieval_config = RetrievalConfig()

        # Stage 5: lazy-loaded data (initially empty)
        self._day_records_by_patient: Dict[str, List[dict]] = {}
        self._day_cards_by_patient: Dict[str, List[dict]] = {}
        self._ngram_cache: "LRUCache" = LRUCache(capacity=20000)

        # Stage 5: build candidate retriever
        retrieval_backend = self.retrieval_config.backend
        if retrieval_backend == "legacy_full_scan":
            self._retriever = LegacyFullScanCandidateRetriever(self.day_store)
        elif retrieval_backend == "in_memory_vector":
            self._retriever = InMemoryVectorCandidateRetriever(
                self.day_store,
                extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
                embedding_model=self.emb_service.config.model,
            )
        else:
            raise ValueError(f"未知 RETRIEVAL_BACKEND: {retrieval_backend}")

        # Stage 5: load only the lightweight retriever index
        self._retriever.load_index()

    @property
    def patient_count(self) -> int:
        return self.day_store.count_patients()

    @property
    def day_count(self) -> int:
        return sum(len(v) for v in self._day_records_by_patient.values()) or self.day_store.count_days()

    def _ensure_candidates_loaded(self, patient_ids: List[str]) -> None:
        """懒加载候选患者的完整数据（day_text + 病例卡 + embedding）。

        只加载尚未在内存中的患者，避免重复查询。
        """
        missing = [pid for pid in patient_ids if pid not in self._day_records_by_patient]
        if not missing:
            return

        new_days = self.day_store.load_days_for_patients(missing)
        new_cards = self.day_store.load_day_cards_for_patients(
            missing, extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION
        )

        # Normalize embeddings
        for cards in new_cards.values():
            for item in cards:
                item["day_delta_embedding"] = _normalize_embedding(item.get("day_delta_embedding"))
                item["cumulative_embedding"] = _normalize_embedding(item.get("cumulative_embedding"))
                item["case_card_embedding"] = _normalize_embedding(item.get("case_card_embedding"))

        self._day_records_by_patient.update(new_days)
        self._day_cards_by_patient.update(new_cards)

    def build_query_days(self, query_rows: Iterable[dict], max_days: int = 0) -> Dict[str, List[dict]]:
        records = build_clinical_day_records(
            query_rows,
            max_days=max_days,
            source_table="query",
            history_mode=self.llm_extractor.config.history_mode,
            history_window_days=self.llm_extractor.config.history_window_days,
            history_max_chars=self.llm_extractor.config.history_max_chars,
        )
        by_patient: Dict[str, List[ClinicalDayRecord]] = {}
        for record in records:
            by_patient.setdefault(record.patient_id, []).append(record)

        result: Dict[str, List[dict]] = {}
        # Phase 1: LLM extraction for all days
        for patient_id, patient_days in by_patient.items():
            items = []
            for day in patient_days:
                card = self._extract_query_day_card(day)
                items.append({
                    "day_record_id": day.day_record_id,
                    "patient_id": day.patient_id,
                    "day_index": day.day_index,
                    "visit_date": day.visit_date,
                    "day_text": day.day_text,
                    "cumulative_text": day.cumulative_text,
                    "case_card": card,
                    "case_card_embedding": None,
                    "text_vector": _char_ngram_vector(day.day_text),
                    "trajectory_state": (card or {}).get("clinical_state", ""),
                })
            result[patient_id] = items

        # Stage 4: Phase 2 — collect all embedding texts and batch embed once
        embedding_map: list[tuple[str, int, str]] = []  # (patient_id, day_idx, text)
        for patient_id, items in result.items():
            for i, item in enumerate(items):
                card = item.get("case_card")
                if card and self.emb_service.is_available:
                    embedding_text = build_case_card_embedding_text(card)
                    if embedding_text:
                        embedding_map.append((patient_id, i, embedding_text))

        if embedding_map:
            texts = [t for _, _, t in embedding_map]
            batch_result = self.emb_service.embed_batch(texts)
            for idx, (patient_id, day_idx, _) in enumerate(embedding_map):
                vec = batch_result.vectors[idx]
                if vec is not None:
                    result[patient_id][day_idx]["case_card_embedding"] = _normalize_embedding(vec)
                elif batch_result.errors:
                    logger.warning("Query embedding failed for %s day %d: %s",
                                   patient_id, day_idx, batch_result.errors.get(idx, "unknown"))

        return result

    def search(self, query_days: List[dict], top_k: int = 5,
               exclude_patient_ids: Optional[set] = None,
               max_days: int = 0,
               rerank_top_n: int = 0,
               rerank_interval: float = 0.0,
               candidate_pool_size: int = 30) -> List[dict]:
        exclude_patient_ids = exclude_patient_ids or set()

        # Stage 5: Phase 1 — candidate retrieval (lightweight)
        patient_limit = self.retrieval_config.patient_candidates
        fallback_reason: str = ""
        try:
            candidate_patient_ids = self._retriever.retrieve_patient_ids(
                query_days,
                patient_limit=patient_limit,
                day_limit_per_query=self.retrieval_config.day_candidates_per_query,
            )
            # Exclude query patient
            candidate_patient_ids = [
                pid for pid in candidate_patient_ids
                if pid not in exclude_patient_ids
            ]
        except Exception as e:
            if not self.retrieval_config.fallback_to_full_scan:
                raise
            logger.warning("Candidate retriever failed, falling back to full scan: %s", e)
            fallback_reason = str(e)
            candidate_patient_ids = self.day_store.list_all_patient_ids()
            candidate_patient_ids = [
                pid for pid in candidate_patient_ids
                if pid not in exclude_patient_ids
            ]

        logger.info(
            "retrieval_backend=%s candidate_patients=%d fallback_reason=%s",
            self._retriever.backend_name,
            len(candidate_patient_ids),
            fallback_reason or "none",
        )

        # Stage 5: Phase 2 — lazy load only candidates
        self._ensure_candidates_loaded(candidate_patient_ids)

        # Stage 5: Phase 3 — full medical scoring only on candidates
        candidates = []
        for patient_id in candidate_patient_ids:
            score, details, trajectory_score, stats = self._daily_patient_similarity(
                query_days, patient_id, max_days=max_days
            )
            if score is None:
                continue
            conflict_summary = _diagnosis_conflict_summary(details)
            if not _passes_diagnosis_gate(conflict_summary):
                continue
            candidates.append({
                "id": patient_id,
                "similarity": round(score, 4),
                "daily_patient_similarity": round(score, 4),
                "trajectory_similarity": round(trajectory_score, 4) if trajectory_score is not None else "-",
                "daily_compare_days": stats["query_compare_days"],
                "daily_query_days": stats["query_days"],
                "daily_candidate_days": stats["candidate_days"],
                "daily_matched_days": stats["matched_days"],
                "daily_coverage": round(stats["coverage"], 4),
                "daily_length_relation": stats["length_relation"],
                "daily_coverage_factor": round(stats["coverage_factor"], 4),
                **conflict_summary,
                "daily_match_details": details,
                "full_text": self._format_candidate_text(patient_id, details),
            })

        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        if rerank_top_n > 0 and (self.http_reranker.is_available or self.llm_reranker.is_available):
            pool_size = max(top_k, rerank_top_n, candidate_pool_size)
            active_reranker = self.http_reranker if self.http_reranker.is_available else self.llm_reranker
            candidates = active_reranker.rerank_candidates(
                query_days,
                candidates[:pool_size],
                top_n=rerank_top_n,
                interval=rerank_interval,
            )
            filtered = [
                item for item in candidates
                if item.get("rerank_decision") not in {"reject", "unknown"}
            ]
            candidates = filtered or candidates
        return candidates[:top_k]

    def _extract_query_day_card(self, day: ClinicalDayRecord) -> Optional[dict]:
        if not self.llm_extractor.is_available:
            return None
        return self.llm_extractor.extract_day(
            day.day_text,
            day.cumulative_text,
            patient_id=day.patient_id,
            day_index=day.day_index,
            visit_date=day.visit_date,
        )

    def _candidate_day_items(self, patient_id: str) -> List[dict]:
        day_records = self._day_records_by_patient.get(patient_id) or []
        day_cards = {
            item["day_record_id"]: item
            for item in self._day_cards_by_patient.get(patient_id, [])
        }
        result = []
        for day in day_records:
            merged = dict(day)
            card_item = day_cards.get(day["day_record_id"])
            if card_item:
                merged.update(card_item)

            # Stage 5: n-gram cache — key includes record ID to avoid stale hits
            cache_key = f"{day['day_record_id']}:{merged.get('source_hash', '')}"
            if cache_key not in self._ngram_cache:
                self._ngram_cache.set(
                    cache_key,
                    _char_ngram_vector(merged.get("day_text", "")),
                )
            merged["text_vector"] = self._ngram_cache.get(cache_key)

            merged["trajectory_state"] = (
                (merged.get("case_card") or {}).get("clinical_state")
                or merged.get("trajectory_state", "")
            )
            result.append(merged)
        return result

    def _day_pair_similarity(self, query_day: dict, cand_day: dict) -> Optional[float]:
        parts = []

        case_card_emb = _vector_cosine(
            query_day.get("case_card_embedding"),
            cand_day.get("case_card_embedding"),
        )
        if case_card_emb is not None:
            parts.append((0.45, max(0.0, case_card_emb)))

        etiology_sim = _etiology_chain_similarity(query_day.get("case_card"), cand_day.get("case_card"))
        if etiology_sim is not None:
            parts.append((0.25, etiology_sim))

        diagnosis_sim = _diagnosis_axis_similarity(query_day.get("case_card"), cand_day.get("case_card"))
        if diagnosis_sim is not None:
            parts.append((0.15, diagnosis_sim))

        tag_sim = _jaccard(_day_card_tags(query_day.get("case_card")), _day_card_tags(cand_day.get("case_card")))
        if tag_sim is not None:
            parts.append((0.08, tag_sim))

        text_sim = _counter_cosine(query_day.get("text_vector"), cand_day.get("text_vector"))
        if text_sim is not None:
            parts.append((0.07, text_sim))

        if not parts:
            return None
        total = sum(weight for weight, _ in parts)
        score = float(sum(weight * score for weight, score in parts) / total)
        if _has_diagnosis_axis_conflict(query_day.get("case_card"), cand_day.get("case_card")):
            score *= 0.70
        if _has_etiology_chain_conflict(query_day.get("case_card"), cand_day.get("case_card")):
            score *= 0.55
        return score

    def _daily_patient_similarity(self, query_days: List[dict], patient_id: str,
                                  max_days: int = 0) -> Tuple[Optional[float], List[dict], Optional[float], dict]:
        candidate_days = self._candidate_day_items(patient_id)
        compare_days = [
            day for day in query_days
            if not (max_days > 0 and day["day_index"] > max_days)
        ]
        stats = {
            "query_days": len(query_days),
            "candidate_days": len(candidate_days),
            "query_compare_days": len(compare_days),
            "matched_days": 0,
            "coverage": 0.0,
            "coverage_factor": 1.0,
            "length_relation": self._length_relation(len(compare_days), len(candidate_days)),
        }
        if not compare_days or not candidate_days:
            return None, [], None, stats

        by_day = {day["day_index"]: day for day in candidate_days}
        weights = {1: 0.30, 2: 0.22, 3: 0.16}
        details = []
        weighted_scores = []
        total_weight = 0.0

        for query_day in compare_days:
            q_index = query_day["day_index"]
            day_candidates = []
            for offset, align_weight in ((0, 1.0), (1, 0.88), (-1, 0.88), (2, 0.76), (-2, 0.76)):
                cand_day = by_day.get(q_index + offset)
                if not cand_day:
                    continue
                raw_score = self._day_pair_similarity(query_day, cand_day)
                if raw_score is not None:
                    day_candidates.append((raw_score * align_weight, raw_score, cand_day))
            if not day_candidates:
                continue

            aligned_score, raw_score, matched_day = max(day_candidates, key=lambda x: x[0])
            weight = weights.get(q_index, 0.08 if q_index <= 7 else 0.04)
            weighted_scores.append(weight * aligned_score)
            total_weight += weight
            etiology_chain_similarity = _etiology_chain_similarity(
                query_day.get("case_card"),
                matched_day.get("case_card"),
            )
            details.append({
                "query_day": q_index,
                "query_date": query_day.get("visit_date", ""),
                "matched_day": matched_day["day_index"],
                "matched_date": matched_day.get("visit_date", ""),
                "score": round(raw_score, 4),
                "aligned_score": round(aligned_score, 4),
                "query_diagnosis_axis": _diagnosis_axis(query_day.get("case_card")),
                "matched_diagnosis_axis": _diagnosis_axis(matched_day.get("case_card")),
                "diagnosis_axis_conflict": _has_diagnosis_axis_conflict(
                    query_day.get("case_card"),
                    matched_day.get("case_card"),
                ),
                "etiology_chain_similarity": (
                    round(etiology_chain_similarity, 4)
                    if etiology_chain_similarity is not None
                    else "-"
                ),
                "query_etiology_chain": _display_etiology_chain_labels(query_day.get("case_card")),
                "matched_etiology_chain": _display_etiology_chain_labels(matched_day.get("case_card")),
                "etiology_chain_conflict": _has_etiology_chain_conflict(
                    query_day.get("case_card"),
                    matched_day.get("case_card"),
                ),
            })

        if total_weight == 0:
            return None, details, None, stats

        score = sum(weighted_scores) / total_weight
        trajectory_score = self._trajectory_similarity(compare_days, candidate_days)
        if trajectory_score is not None:
            score = 0.90 * score + 0.10 * trajectory_score

        coverage = len(details) / len(compare_days)
        coverage_factor = 0.70 + 0.30 * coverage
        if coverage < 1.0:
            score *= coverage_factor

        stats["matched_days"] = len(details)
        stats["coverage"] = coverage
        stats["coverage_factor"] = coverage_factor
        return float(max(0.0, min(1.0, score))), details, trajectory_score, stats

    def _format_candidate_text(self, patient_id: str, details: List[dict]) -> str:
        by_day = {
            day["day_index"]: day
            for day in self._day_records_by_patient.get(patient_id, [])
        }
        selected = []
        matched_indexes = [item["matched_day"] for item in details]
        if not matched_indexes:
            matched_indexes = [day["day_index"] for day in self._day_records_by_patient.get(patient_id, [])[:5]]
        for day_index in matched_indexes:
            day = by_day.get(day_index)
            if not day:
                continue
            selected.append(
                f"###候选患者 {patient_id} Day {day_index} - {day.get('visit_date', '')}\n"
                f"{day.get('day_text', '')}"
            )
        return "\n\n".join(selected)

    @staticmethod
    def _trajectory_similarity(query_days: List[dict], candidate_days: List[dict]) -> Optional[float]:
        q_seq = [day.get("trajectory_state") for day in query_days if day.get("trajectory_state")]
        c_seq = [day.get("trajectory_state") for day in candidate_days if day.get("trajectory_state")]
        if not q_seq or not c_seq:
            return None
        m, n = len(q_seq), len(c_seq)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if q_seq[i - 1] == c_seq[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n] / max(m, n)

    @staticmethod
    def _length_relation(query_len: int, candidate_len: int) -> str:
        if candidate_len < query_len:
            return "candidate_shorter"
        if candidate_len > query_len:
            return "candidate_longer"
        return "same_length"
