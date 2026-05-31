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
    disease_axis: List[str] = field(default_factory=list)  # 临床主题轴（1-3个，限定枚举）

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
            disease_axis=data.get("disease_axis", []) or [],
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
            "disease_axis": self.disease_axis,
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


# ═══════════════════════════════════════════════════════════════════
# 临床主题轴 (disease_axis)
# ═══════════════════════════════════════════════════════════════════

VALID_DISEASE_AXES = {
    "neuro_trauma",
    "neuro_hemorrhage",
    "neuro_infection",
    "cardiac_surgery",
    "aortic_disease",
    "heart_failure",
    "respiratory_failure",
    "sepsis_abdominal",
    "sepsis_pulmonary",
    "sepsis_other",
    "abdominal_bleeding",
    "pancreatitis",
    "hepatobiliary_disease",
    "renal_failure",
    "multi_trauma",
    "postoperative_monitoring",
    "other",
}

# 主题冲突组：同一组内的主题属于相近领域，跨组冲突才判为明显冲突
AXIS_CONFLICT_GROUPS = [
    {"neuro_trauma", "neuro_hemorrhage", "neuro_infection"},
    {"cardiac_surgery", "aortic_disease", "heart_failure"},
    {"sepsis_abdominal", "pancreatitis", "hepatobiliary_disease", "abdominal_bleeding"},
    {"renal_failure"},
    {"respiratory_failure", "sepsis_pulmonary"},
    {"multi_trauma"},
    {"postoperative_monitoring"},
    {"other"},
]


def normalize_disease_axes(card: dict) -> List[str]:
    """
    清洗病例卡中的 disease_axis 字段：
    - 不在枚举内的值删除
    - 去重
    - 最多保留 3 个
    - 旧病例卡没有该字段时返回空列表
    """
    raw = card.get("disease_axis", []) or []
    cleaned = []
    seen = set()
    for axis in raw:
        axis = axis.strip().lower()
        if axis in VALID_DISEASE_AXES and axis not in seen:
            seen.add(axis)
            cleaned.append(axis)
    return cleaned[:3]


def disease_axis_similarity(card_a: dict, card_b: dict) -> float:
    """
    计算两个病例卡的临床主题相似度。
    至少一个交集 → 1.0；都没有 → 0.5（中性）；有但不交 → 0.0。
    """
    axes_a = set(normalize_disease_axes(card_a))
    axes_b = set(normalize_disease_axes(card_b))
    if not axes_a or not axes_b:
        return 0.5  # 未知，不奖励也不重罚
    if axes_a & axes_b:
        return 1.0
    return 0.0


def _get_conflict_group(axis: str) -> Optional[int]:
    """返回主题所属冲突组索引，不在任何组返回 None"""
    for i, group in enumerate(AXIS_CONFLICT_GROUPS):
        if axis in group:
            return i
    return None


def has_disease_axis_conflict(card_a: dict, card_b: dict) -> bool:
    """
    判断两个病例卡的临床主题是否明显冲突。

    两边均存在 disease_axis，但没有任何交集；
    且双方主轴落在不同冲突组时返回 True。
    """
    axes_a = set(normalize_disease_axes(card_a))
    axes_b = set(normalize_disease_axes(card_b))
    if not axes_a or not axes_b:
        return False
    # 有交集则无冲突
    if axes_a & axes_b:
        return False
    # 检查是否所有轴都落在不同冲突组
    groups_a = set()
    for a in axes_a:
        g = _get_conflict_group(a)
        if g is not None:
            groups_a.add(g)
    groups_b = set()
    for b in axes_b:
        g = _get_conflict_group(b)
        if g is not None:
            groups_b.add(g)
    # 如果任一方无法归类，不判冲突（偏保守）
    if not groups_a or not groups_b:
        return False
    # 无共同冲突组 = 跨主题冲突
    return not bool(groups_a & groups_b)


# ═══════════════════════════════════════════════════════════════════
# 干预标签特异性加权
# ═══════════════════════════════════════════════════════════════════

GENERIC_INTERVENTION_TAGS = {
    "机械通气",
    "无创通气",
    "升压药",
    "吸氧",
    "镇静镇痛",
    "抗感染治疗",
    "输血治疗",
    "营养支持",
    "留置导尿",
}

SPECIFIC_INTERVENTION_KEYWORDS = {
    "CRRT",
    "ECMO",
    "IABP",
    "PICCO监测",
    "开颅",
    "去骨瓣减压",
    "脑室引流",
    "腰大池引流",
    "Bentall",
    "主动脉弓置换",
    "支架象鼻",
    "肝脓肿穿刺引流",
    "腹腔穿刺",
    "腹腔引流",
    "胰周穿刺引流",
    "胸腔穿刺引流",
    "TACE",
}


def is_specific_intervention(tag: str) -> bool:
    """判断一个干预标签是否为特异干预（关键词包含匹配）"""
    tag_lower = tag.lower()
    return any(kw.lower() in tag_lower for kw in SPECIFIC_INTERVENTION_KEYWORDS)


def intervention_weight(tag: str) -> float:
    """
    干预标签权重：
    - 特异标签：3.0
    - 泛化 ICU 标签：0.35
    - 普通标签：1.0
    """
    if is_specific_intervention(tag):
        return 3.0
    if tag in GENERIC_INTERVENTION_TAGS:
        return 0.35
    return 1.0


def weighted_intervention_overlap(tags_a: Set[str], tags_b: Set[str]) -> float:
    """
    加权干预 Jaccard：特异标签权重 3.0，普通 1.0，泛化 ICU 0.35。
    """
    if not tags_a and not tags_b:
        return 0.5
    if not tags_a or not tags_b:
        return 0.0

    union = tags_a | tags_b
    inter = tags_a & tags_b

    total_weight = sum(intervention_weight(t) for t in union)
    inter_weight = sum(intervention_weight(t) for t in inter)

    return inter_weight / total_weight if total_weight > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# 强共同标签判断
# ═══════════════════════════════════════════════════════════════════

def find_strong_common_tags(card_a: dict, card_b: dict) -> List[str]:
    """
    返回共同主诊断、共同特异干预、共同 disease_axis。
    泛化 ICU 干预不算 strong tag。
    """
    strong = []

    # 共同主诊断
    diag_a = set()
    for d in (card_a.get("primary_diagnoses") or []):
        name = d if isinstance(d, str) else d.get("name", "")
        if name:
            diag_a.add(normalize_tag(name))
    diag_b = set()
    for d in (card_b.get("primary_diagnoses") or []):
        name = d if isinstance(d, str) else d.get("name", "")
        if name:
            diag_b.add(normalize_tag(name))
    common_diag = diag_a & diag_b
    for d in sorted(common_diag):
        strong.append(f"诊断:{d}")

    # 共同特异干预
    interv_a, interv_b = set(), set()
    for item in (card_a.get("key_interventions") or []):
        name = item.get("normalized_name", "") or item.get("name", "")
        if name:
            interv_a.add(normalize_tag(name))
    for item in (card_a.get("surgery_or_operations") or []):
        name = item.get("normalized_name", "") or item.get("name", "")
        if name:
            interv_a.add(normalize_tag(name))
    for item in (card_b.get("key_interventions") or []):
        name = item.get("normalized_name", "") or item.get("name", "")
        if name:
            interv_b.add(normalize_tag(name))
    for item in (card_b.get("surgery_or_operations") or []):
        name = item.get("normalized_name", "") or item.get("name", "")
        if name:
            interv_b.add(normalize_tag(name))

    common_interv = interv_a & interv_b
    for tag in sorted(common_interv):
        if is_specific_intervention(tag):
            strong.append(f"特异干预:{tag}")

    # 共同 disease_axis
    axes_a = set(normalize_disease_axes(card_a))
    axes_b = set(normalize_disease_axes(card_b))
    for axis in sorted(axes_a & axes_b):
        strong.append(f"主题:{axis}")

    return strong


def has_strong_common_tag(card_a: dict, card_b: dict) -> bool:
    """是否有强共同标签（主诊断 / 特异干预 / disease_axis 交集）"""
    return bool(find_strong_common_tags(card_a, card_b))



def tag_overlap_score(card_a: dict, card_b: dict) -> float:
    """
    计算两个病例卡的标签重叠相似度。

    加权 Jaccard（v1.1 更新）：
      0.35 × 诊断 Jaccard（主诊断是临床主题的直接信号）
    + 0.30 × 加权干预得分（特异标签 3.0、泛化 ICU 0.35、普通 1.0）
    + 0.20 × 器官功能 Jaccard
    + 0.10 × 并发症 Jaccard（噪声相对更大）
    + 0.05 × disease_axis 得分
    """
    diag_a, interv_a, compl_a, organ_a = extract_tag_sets(card_a)
    diag_b, interv_b, compl_b, organ_b = extract_tag_sets(card_b)

    # 加权干预得分
    interv_score = weighted_intervention_overlap(interv_a, interv_b)

    # disease_axis 得分（归一化到 0-1）
    axis_score = disease_axis_similarity(card_a, card_b)

    return (
        0.35 * jaccard_similarity(diag_a, diag_b) +
        0.30 * interv_score +
        0.20 * jaccard_similarity(organ_a, organ_b) +
        0.10 * jaccard_similarity(compl_a, compl_b) +
        0.05 * axis_score
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
