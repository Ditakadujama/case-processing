"""
病例卡 Schema、标签标准化、标签重叠相似度计算

病例卡是 LLM 从原始病历中抽取的结构化临床摘要，
包含主病情、诊断、关键干预、器官功能、并发症、病程阶段等。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any


# ═══════════════════════════════════════════════════════════════════
# 病例卡 Schema
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CaseCard:
    """LLM 抽取的结构化病例卡"""
    record_id: str = ""
    chief_problem: str = ""                      # 主要临床问题
    icu_reason: str = ""                         # 进入 ICU 原因
    primary_diagnoses: List[str] = field(default_factory=list)
    secondary_diagnoses: List[str] = field(default_factory=list)
    surgery_or_operations: List[dict] = field(default_factory=list)
    key_interventions: List[dict] = field(default_factory=list)
    organ_failures: List[dict] = field(default_factory=list)
    complications: List[dict] = field(default_factory=list)
    clinical_course: List[dict] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    severity_level: str = "unknown"
    summary_for_embedding: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CaseCard":
        """从 LLM 输出的 JSON dict 构建 CaseCard"""
        return cls(
            record_id=data.get("record_id", ""),
            chief_problem=data.get("chief_problem", ""),
            icu_reason=data.get("icu_reason", ""),
            primary_diagnoses=data.get("primary_diagnoses", []) or [],
            secondary_diagnoses=data.get("secondary_diagnoses", []) or [],
            surgery_or_operations=data.get("surgery_or_operations", []) or [],
            key_interventions=data.get("key_interventions", []) or [],
            organ_failures=data.get("organ_failures", []) or [],
            complications=data.get("complications", []) or [],
            clinical_course=data.get("clinical_course", []) or [],
            outcome=data.get("outcome", {}) or {},
            severity_level=data.get("severity_level", "unknown"),
            summary_for_embedding=data.get("summary_for_embedding", ""),
        )

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的 dict"""
        return {
            "record_id": self.record_id,
            "chief_problem": self.chief_problem,
            "icu_reason": self.icu_reason,
            "primary_diagnoses": self.primary_diagnoses,
            "secondary_diagnoses": self.secondary_diagnoses,
            "surgery_or_operations": self.surgery_or_operations,
            "key_interventions": self.key_interventions,
            "organ_failures": self.organ_failures,
            "complications": self.complications,
            "clinical_course": self.clinical_course,
            "outcome": self.outcome,
            "severity_level": self.severity_level,
            "summary_for_embedding": self.summary_for_embedding,
        }


# ═══════════════════════════════════════════════════════════════════
# 标签标准化映射
# ═══════════════════════════════════════════════════════════════════

NORMALIZATION_MAP: Dict[str, str] = {
    # 肾脏替代治疗
    "连续性肾脏替代治疗": "CRRT",
    "连续性血液净化": "CRRT",
    "血液滤过": "CRRT",
    "血液透析": "CRRT",
    "血滤": "CRRT",
    "血液透析滤过": "CRRT",
    "CVVH": "CRRT",
    "CVVHDF": "CRRT",
    # 循环支持
    "主动脉内球囊反搏": "IABP",
    "主动脉球囊反搏": "IABP",
    "主动脉内球囊反搏导管置入": "IABP",
    # 呼吸支持
    "气管插管": "机械通气",
    "气管切开": "机械通气",
    "呼吸机辅助呼吸": "机械通气",
    "有创机械通气": "机械通气",
    "无创机械通气": "无创通气",
    "无创正压通气": "无创通气",
    # 体外生命支持
    "体外膜肺氧合": "ECMO",
    "ECMO": "ECMO",
    # 心功能
    "心力衰竭": "心衰",
    "慢性心功能不全急性加重": "心衰急性加重",
    "急性心力衰竭": "心衰急性加重",
    "急性左心衰": "心衰急性加重",
    "心功能不全": "心衰",
    "心源性休克": "心源性休克",
    # 肾功能
    "肾功能不全": "肾功能障碍",
    "急性肾损伤": "肾功能障碍",
    "急性肾功能衰竭": "肾功能障碍",
    "慢性肾功能不全": "肾功能障碍",
    "肾功能衰竭": "肾功能障碍",
    "AKI": "肾功能障碍",
    # 呼吸功能
    "呼吸衰竭": "呼吸衰竭",
    "急性呼吸窘迫综合征": "ARDS",
    "ARDS": "ARDS",
    "Ⅰ型呼吸衰竭": "呼吸衰竭",
    "Ⅱ型呼吸衰竭": "呼吸衰竭",
    # 循环功能
    "休克": "休克",
    "感染性休克": "感染性休克",
    "脓毒性休克": "感染性休克",
    "失血性休克": "失血性休克",
    "分布性休克": "分布性休克",
    "梗阻性休克": "梗阻性休克",
    # 感染
    "肺部感染": "肺部感染",
    "呼吸机相关性肺炎": "肺部感染",
    "VAP": "肺部感染",
    "腹腔感染": "腹腔感染",
    "血流感染": "血流感染",
    "导管相关血流感染": "血流感染",
    "泌尿系感染": "泌尿系感染",
    "颅内感染": "颅内感染",
    "脓毒症": "脓毒症",
    "败血症": "脓毒症",
    # 凝血/血栓
    "下肢深静脉血栓": "下肢静脉血栓",
    "深静脉血栓": "下肢静脉血栓",
    "肺栓塞": "肺栓塞",
    "DIC": "弥漫性血管内凝血",
    "弥漫性血管内凝血": "弥漫性血管内凝血",
    # 肝功能障碍
    "肝功能不全": "肝功能障碍",
    "肝功能衰竭": "肝功能障碍",
    "肝衰竭": "肝功能障碍",
    "急性肝损伤": "肝功能障碍",
    # 神经系统
    "脑梗死": "脑梗死",
    "脑出血": "脑出血",
    "脑水肿": "脑水肿",
    "缺血性脑卒中": "脑梗死",
    # 代谢/内分泌
    "2型糖尿病": "糖尿病",
    "电解质紊乱": "电解质紊乱",
    "高钠血症": "电解质紊乱",
    "低钠血症": "电解质紊乱",
    "高钾血症": "电解质紊乱",
    "低钾血症": "电解质紊乱",
    # 药物/治疗
    "去甲肾上腺素": "升压药",
    "多巴胺": "升压药",
    "多巴酚丁胺": "升压药",
    "肾上腺素": "升压药",
    "血管活性药物": "升压药",
    "PICCO": "PICCO监测",
    "脉搏指示连续心排血量监测": "PICCO监测",
    # 其他 ICU 术语
    "多器官功能障碍综合征": "MODS",
    "MODS": "MODS",
    "全身炎症反应综合征": "SIRS",
}


def normalize_tag(tag: str) -> str:
    """将单个标签标准化"""
    tag = tag.strip()
    if not tag:
        return ""
    return NORMALIZATION_MAP.get(tag, tag)


def normalize_tags(tags: List[str]) -> List[str]:
    """标准化标签列表，去重并过滤空值"""
    seen = set()
    result = []
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# ═══════════════════════════════════════════════════════════════════
# 标签集合提取
# ═══════════════════════════════════════════════════════════════════

def extract_tag_sets(card: dict) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    从病例卡 dict 中提取四组标准化标签集合。

    Returns:
        (diagnosis_tags, intervention_tags, complication_tags, organ_failure_tags)
    """
    # 诊断标签：主诊断 + 次要诊断
    diag_raw = set()
    for d in (card.get("primary_diagnoses") or []):
        if isinstance(d, dict):
            diag_raw.add(d.get("name", ""))
        elif isinstance(d, str):
            diag_raw.add(d)
    for d in (card.get("secondary_diagnoses") or []):
        if isinstance(d, dict):
            diag_raw.add(d.get("name", ""))
        elif isinstance(d, str):
            diag_raw.add(d)
    diagnosis_tags = set(normalize_tags(list(diag_raw)))

    # 干预标签：关键干预 + 手术/操作
    intervention_raw = set()
    for item in (card.get("key_interventions") or []):
        name = item.get("normalized_name", "") or item.get("name", "")
        if name:
            intervention_raw.add(name)
    for item in (card.get("surgery_or_operations") or []):
        name = item.get("normalized_name", "") or item.get("name", "")
        if name:
            intervention_raw.add(name)
    intervention_tags = set(normalize_tags(list(intervention_raw)))

    # 并发症标签（P0-3: 过滤 risk_only 和低置信度条目）
    complication_raw = set()
    for item in (card.get("complications") or []):
        status = item.get("status", "confirmed")
        confidence = item.get("confidence", "medium")
        # 跳过来自知情同意/风险告知的条目
        if status == "risk_only":
            continue
        # 跳过低置信度条目
        if confidence == "low":
            continue
        name = item.get("name", "")
        if name:
            complication_raw.add(name)
    complication_tags = set(normalize_tags(list(complication_raw)))

    # 器官功能问题标签
    organ_failure_raw = set()
    for item in (card.get("organ_failures") or []):
        name = item.get("name", "")
        if name:
            organ_failure_raw.add(name)
    organ_failure_tags = set(normalize_tags(list(organ_failure_raw)))

    return diagnosis_tags, intervention_tags, complication_tags, organ_failure_tags


# ═══════════════════════════════════════════════════════════════════
# Jaccard 相似度
# ═══════════════════════════════════════════════════════════════════

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """两个集合的 Jaccard 相似度"""
    if not a and not b:
        return 0.5  # 两者都为空 → 中性
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def tag_overlap_score(card_a: dict, card_b: dict) -> float:
    """
    计算两个病例卡的标签重叠相似度。

    加权 Jaccard：
      0.30 × 诊断 Jaccard
    + 0.35 × 干预 Jaccard
    + 0.20 × 器官功能 Jaccard
    + 0.15 × 并发症 Jaccard

    干预权重最高（ICU 病程相似往往由机械通气、CRRT、IABP、升压药等决定）。
    并发症权重最低（抽取难度大，但保留信号）。
    """
    diag_a, interv_a, compl_a, organ_a = extract_tag_sets(card_a)
    diag_b, interv_b, compl_b, organ_b = extract_tag_sets(card_b)

    return (
        0.30 * jaccard_similarity(diag_a, diag_b) +
        0.35 * jaccard_similarity(interv_a, interv_b) +
        0.20 * jaccard_similarity(organ_a, organ_b) +
        0.15 * jaccard_similarity(compl_a, compl_b)
    )


def find_common_and_diff_tags(card_a: dict, card_b: dict) -> dict:
    """
    找出两个病例卡的共同和差异标签，用于结果解释。

    Returns:
        {
            "common_diagnoses": [...],
            "common_interventions": [...],
            "common_organ_failures": [...],
            "common_complications": [...],
            "diff_diagnoses": {"query_only": [...], "candidate_only": [...]},
            ...
        }
    """
    diag_a, interv_a, compl_a, organ_a = extract_tag_sets(card_a)
    diag_b, interv_b, compl_b, organ_b = extract_tag_sets(card_b)

    return {
        "common_diagnoses": sorted(diag_a & diag_b),
        "common_interventions": sorted(interv_a & interv_b),
        "common_organ_failures": sorted(organ_a & organ_b),
        "common_complications": sorted(compl_a & compl_b),
        "diff_diagnoses": {
            "query_only": sorted(diag_a - diag_b),
            "candidate_only": sorted(diag_b - diag_a),
        },
        "diff_interventions": {
            "query_only": sorted(interv_a - interv_b),
            "candidate_only": sorted(interv_b - interv_a),
        },
        "diff_organ_failures": {
            "query_only": sorted(organ_a - organ_b),
            "candidate_only": sorted(organ_b - organ_a),
        },
        "diff_complications": {
            "query_only": sorted(compl_a - compl_b),
            "candidate_only": sorted(compl_b - compl_a),
        },
    }
