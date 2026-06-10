"""医生关注文本驱动的天级序列检索。"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple
import re

import numpy as np

from clinical_text_filter import ClinicalDayRecord, build_clinical_day_records
from config import DBConfig, EmbeddingConfig, LLMConfig
from day_store import DEFAULT_DAY_EXTRACTOR_VERSION, MySQLDayStore
from embedding_index import EmbeddingService
from llm_case_extractor import LLMCaseExtractor


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
    ):
        values = card.get(key) or []
        for value in values:
            if isinstance(value, str):
                tags.add(value.strip())
            elif isinstance(value, dict):
                for sub_key in ("name", "label", "type", "status"):
                    sub_val = value.get(sub_key)
                    if sub_val:
                        tags.add(str(sub_val).strip())
    state = card.get("clinical_state")
    if state:
        tags.add(f"state:{state}")
    return {tag for tag in tags if tag}


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


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DayLevelRetrievalSystem:
    """按患者每日记录序列进行相似度检索。"""

    def __init__(self, db_config: DBConfig):
        self.db_config = db_config
        self.day_store = MySQLDayStore(db_config)
        self.day_store.init_tables()
        self.llm_extractor = LLMCaseExtractor(LLMConfig())
        self.emb_service = EmbeddingService(EmbeddingConfig())
        self._day_records_by_patient: Dict[str, List[dict]] = {}
        self._day_cards_by_patient: Dict[str, List[dict]] = {}
        self._load_index()

    @property
    def patient_count(self) -> int:
        return len(self._day_records_by_patient)

    @property
    def day_count(self) -> int:
        return sum(len(v) for v in self._day_records_by_patient.values())

    def _load_index(self) -> None:
        self._day_records_by_patient = self.day_store.load_days_by_patient()
        self._day_cards_by_patient = self.day_store.load_day_cards_by_patient(
            extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION
        )
        for cards in self._day_cards_by_patient.values():
            for item in cards:
                item["day_delta_embedding"] = _normalize_embedding(item.get("day_delta_embedding"))
                item["cumulative_embedding"] = _normalize_embedding(item.get("cumulative_embedding"))

    def build_query_days(self, query_rows: Iterable[dict], max_days: int = 0) -> Dict[str, List[dict]]:
        records = build_clinical_day_records(query_rows, max_days=max_days)
        by_patient: Dict[str, List[ClinicalDayRecord]] = {}
        for record in records:
            by_patient.setdefault(record.patient_id, []).append(record)

        result: Dict[str, List[dict]] = {}
        for patient_id, patient_days in by_patient.items():
            items = []
            for day in patient_days:
                card = self._extract_query_day_card(day)
                day_embedding = None
                cumulative_embedding = None
                if card and self.emb_service.is_available:
                    day_summary = card.get("day_summary_for_embedding") or card.get("summary_for_embedding", "")
                    cumulative_summary = card.get("cumulative_summary_for_embedding") or ""
                    if day_summary:
                        day_embedding = _normalize_embedding(self.emb_service.embed_text(day_summary))
                    if cumulative_summary:
                        cumulative_embedding = _normalize_embedding(self.emb_service.embed_text(cumulative_summary))

                items.append({
                    "day_record_id": day.day_record_id,
                    "patient_id": day.patient_id,
                    "day_index": day.day_index,
                    "visit_date": day.visit_date,
                    "day_text": day.day_text,
                    "cumulative_text": day.cumulative_text,
                    "case_card": card,
                    "day_delta_embedding": day_embedding,
                    "cumulative_embedding": cumulative_embedding,
                    "text_vector": _char_ngram_vector(day.day_text),
                    "trajectory_state": (card or {}).get("clinical_state", ""),
                })
            result[patient_id] = items
        return result

    def search(self, query_days: List[dict], top_k: int = 5,
               exclude_patient_ids: Optional[set] = None,
               max_days: int = 0) -> List[dict]:
        exclude_patient_ids = exclude_patient_ids or set()
        candidates = []
        for patient_id in self._day_records_by_patient:
            if patient_id in exclude_patient_ids:
                continue
            score, details, trajectory_score, stats = self._daily_patient_similarity(
                query_days, patient_id, max_days=max_days
            )
            if score is None:
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
                "daily_match_details": details,
                "full_text": self._format_candidate_text(patient_id, details),
            })

        candidates.sort(key=lambda item: item["similarity"], reverse=True)
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
            merged["text_vector"] = _char_ngram_vector(merged.get("day_text", ""))
            merged["trajectory_state"] = (
                (merged.get("case_card") or {}).get("clinical_state")
                or merged.get("trajectory_state", "")
            )
            result.append(merged)
        return result

    def _day_pair_similarity(self, query_day: dict, cand_day: dict) -> Optional[float]:
        parts = []
        diagnosis_sim = _diagnosis_axis_similarity(query_day.get("case_card"), cand_day.get("case_card"))
        if diagnosis_sim is not None:
            parts.append((0.30, diagnosis_sim))

        day_emb = _vector_cosine(query_day.get("day_delta_embedding"), cand_day.get("day_delta_embedding"))
        if day_emb is not None:
            parts.append((0.40, max(0.0, day_emb)))

        tag_sim = _jaccard(_day_card_tags(query_day.get("case_card")), _day_card_tags(cand_day.get("case_card")))
        if tag_sim is not None:
            parts.append((0.20, tag_sim))

        cumulative_emb = _vector_cosine(query_day.get("cumulative_embedding"), cand_day.get("cumulative_embedding"))
        if cumulative_emb is not None:
            parts.append((0.15, max(0.0, cumulative_emb)))

        text_sim = _counter_cosine(query_day.get("text_vector"), cand_day.get("text_vector"))
        if text_sim is not None:
            parts.append((0.10, text_sim))

        if not parts:
            return None
        total = sum(weight for weight, _ in parts)
        score = float(sum(weight * score for weight, score in parts) / total)
        if _has_diagnosis_axis_conflict(query_day.get("case_card"), cand_day.get("case_card")):
            score *= 0.70
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
