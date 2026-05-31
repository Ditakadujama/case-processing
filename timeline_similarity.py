"""
时间轴相似度评分器
基于就诊日期 (visit_date) 的时序模型，比较两个病历的病程发展相似性
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

from timeline_parser import TimelineEvent, TimelineParser


# 与 FeatureExtractor 保持一致的诊断词表
DIAGNOSIS_TERMS = [
    '颅脑损伤', '脑梗死', '肺部感染', '呼吸衰竭', '高血压',
    '心房颤动', '传导阻滞', '胸腔积液', '肺不张', '肺炎',
    '心肌损伤', '肾功能', '肝功能', '感染', '休克',
    '出血', '贫血', '血栓', '电解质紊乱', '酸碱平衡',
    '脑出血', '脑水肿', '癫痫', '昏迷', '脓毒症',
    '心衰', '心梗', '动脉粥样硬化', '肺栓塞', 'COPD',
    '糖尿病', '甲状腺', '消化道出血', '胰腺炎', '肠梗阻',
    '败血症', '多器官功能障碍', '应激性溃疡', '深静脉血栓'
]

# 手术分类规则（仅用于提取手术关键词，surgery_type 由 record_parser 提供）
SURGERY_TYPE_RULES = [
    ("neuro", ["脊髓", "脑室", "颅脑", "脑", "血肿清除", "开颅", "神经内镜", "硬脊膜", "椎板", "髓内", "颅内", "脑膜", "神经减压", "听神经瘤"]),
    ("cardiac", ["冠脉", "心脏", "搭桥", "CABG", "PCI", "瓣膜", "射频消融", "旁路移植", "心房", "心室", "起搏器", "主动脉球囊反搏", "主动脉", "二尖瓣", "三尖瓣", "房间隔", "室间隔"]),
    ("general", ["肝脏", "胆囊", "胰腺", "胃肠", "阑尾", "脾", "甲状腺", "胃切除", "肠切除", "结肠", "直肠", "胆道", "腹腔"]),
    ("ortho", ["骨折", "关节置换", "髋关节", "膝关节", "肩关节", "椎间盘", "椎弓根", "椎管", "植骨融合", "骨科", "肌腱", "韧带", "半月板", "颈椎", "胸椎", "腰椎", "脊柱", "截骨"]),
]

# 事件类型权重（用于加权LCS）
EVENT_WEIGHTS = {
    "surgery": 3.0,
    "diagnosis_change": 2.5,
    "transfer": 2.0,
    "admission": 1.5,
    "discharge": 1.5,
    "lab": 1.0,
    "exam": 1.0,
    "progress_note": 0.8,
    "medication": 0.5,
    "vitals": 0.3,
    "follow_up": 0.6,
}

INTERVENTION_KEYWORDS = {
    "机械通气", "气管插管", "呼吸机", "CRRT", "血滤", "IABP",
    "主动脉内球囊反搏", "PICCO", "PiCCO", "升压", "抗生素升级",
    "去甲肾上腺素", "肾上腺素", "多巴胺", "多巴酚丁胺",
    "血管活性药物", "手术", "置管", "引流", "输血", "白蛋白",
}

COMPLICATION_KEYWORDS = {
    "出血", "血栓", "感染", "肺部感染", "呼吸衰竭", "肾功能不全",
    "肾衰竭", "心力衰竭", "心衰", "休克", "脓毒症", "败血症",
    "多器官功能衰竭", "多器官功能障碍", "恶性心律失常", "脑梗死",
    "脑出血", "消化道出血", "肺栓塞", "死亡",
}

# 手术类型关键词集（用于提取手术名称中的关键词）
SURGERY_TYPE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    k: tuple(v) for k, v in SURGERY_TYPE_RULES
}


@dataclass
class TimelineFeatures:
    """预计算的时间轴特征，用于快速相似度计算（基于就诊计数模型）"""
    visit_count: int                    # 就诊天数（唯一日期数）
    event_count: int                    # 真实事件数（含连续同类，不经压缩）
    total_span_days: float              # 首末次就诊间隔天数
    visit_gaps: List[float]             # 相邻就诊间隔天数序列
    diagnosis_keywords: Set[str]        # 所有诊断相关事件中提取的诊断关键词
    event_type_sequence: List[str]      # 按时间排序的事件类型序列（连续同类去重）
    surgery_type: Optional[str] = None  # 手术专科类型（由 record_parser 提供）
    surgery_keywords: Set[str] = field(default_factory=set)  # 手术名称关键词
    intervention_keywords: Set[str] = field(default_factory=set)  # 关键治疗/支持手段
    complication_keywords: Set[str] = field(default_factory=set)  # 并发症/病情恶化信号
    severity_keywords: Set[str] = field(default_factory=set)      # 器官功能和危重程度信号


class TimelineSimilarityScorer:
    """病程相似度评分器（基于就诊计数模型）"""

    def __init__(self,
                 weight_visit_scale: float = 0.12,
                 weight_time_span: float = 0.08,
                 weight_diag: float = 0.18,
                 weight_seq: float = 0.17,
                 weight_surgery_type: float = 0.20,
                 weight_clinical_state: float = 0.25,
                 max_span_days: float = 3650.0):  # 10 年
        self.weights = {
            'visit_scale': weight_visit_scale,
            'time_span': weight_time_span,
            'diag': weight_diag,
            'seq': weight_seq,
            'surgery_type': weight_surgery_type,
            'clinical_state': weight_clinical_state,
        }
        self.max_span_days = max_span_days
        self.severity_terms = {
            "机械通气", "气管插管", "呼吸机", "CRRT", "血滤", "IABP",
            "主动脉内球囊反搏", "休克", "心源性休克", "感染性休克",
            "肾功能不全", "肾衰竭", "呼吸衰竭", "心力衰竭", "心衰",
            "多器官功能衰竭", "多器官功能障碍", "乳酸", "氧合指数",
            "去甲肾上腺素", "肾上腺素", "升压", "血管活性药物",
        }

    def extract_features(self,
                         events: List[TimelineEvent],
                         nodes: Dict[str, Optional[TimelineEvent]],
                         surgery_type_hint: Optional[str] = None,
                         surgery_keywords_hint: Optional[Set[str]] = None) -> TimelineFeatures:
        """从时间轴事件中提取预计算特征

        Args:
            events: 时间轴事件列表
            nodes: T0-T6 标准节点（向后兼容，新模型通过事件列表计算）
            surgery_type_hint: 由 record_parser 提供的已知手术类型（避免从文本误分类）
        """
        # 1. 就诊计数和跨度（按唯一就诊日期计数，而非每个事件时间戳）
        visit_dates = sorted(set(
            e.timestamp.date() for e in events if e.timestamp
        ))
        visit_count = len(visit_dates)
        total_span_days = 0.0
        visit_gaps: List[float] = []
        if visit_count >= 2:
            total_span_days = float((visit_dates[-1] - visit_dates[0]).days)
            visit_gaps = [
                float((visit_dates[i] - visit_dates[i - 1]).days)
                for i in range(1, visit_count)
            ]

        # 2. 诊断关键词（从所有事件文本中提取）
        diag_keywords: Set[str] = set()
        diag_event_types = ("diagnosis_change", "progress_note", "admission", "discharge", "follow_up")
        for e in events:
            if e.event_type in diag_event_types:
                text_to_search = e.description + " " + e.raw_text
                for term in DIAGNOSIS_TERMS:
                    if term in text_to_search:
                        diag_keywords.add(term)
                if e.structured_data and "diagnosis" in e.structured_data:
                    diag_text = e.structured_data["diagnosis"]
                    for term in DIAGNOSIS_TERMS:
                        if term in diag_text:
                            diag_keywords.add(term)

        # 3. 事件类型序列（按时间排序，去重相邻相同类型）
        sorted_events = sorted(
            [e for e in events if e.timestamp],
            key=lambda e: e.timestamp
        )
        seq: List[str] = []
        for e in sorted_events:
            if not seq or seq[-1] != e.event_type:
                seq.append(e.event_type)

        # 4. 手术类型和关键词
        surgery_type = surgery_type_hint  # 优先使用外部提供的
        surgery_keywords: Set[str] = set()
        if surgery_type_hint and surgery_type_hint != "other":
            # 从手术相关事件中提取关键词
            surgery_events = [e for e in events if e.event_type == "surgery"]
            if not surgery_events:
                # 没有手术事件时，直接从全局文本搜索
                for e in events:
                    surgery_text = e.description + " " + e.raw_text
                    for stype, keywords in SURGERY_TYPE_RULES:
                        for kw in keywords:
                            if kw in surgery_text:
                                surgery_keywords.add(kw)
            else:
                first_surgery = min(surgery_events, key=lambda e: e.timestamp)
                surgery_text = first_surgery.description + " " + first_surgery.raw_text
                for stype, keywords in SURGERY_TYPE_RULES:
                    for kw in keywords:
                        if kw in surgery_text:
                            surgery_keywords.add(kw)

        # 如果没有外部提示，尝试从手术事件提取（仅手术名称部分，避免误分类）
        if not surgery_type:
            surgery_events = [e for e in events if e.event_type == "surgery"]
            if surgery_events:
                first_surgery = min(surgery_events, key=lambda e: e.timestamp)
                # 只从 structured_data 的 surgery_name 提取，避免匹配到术前诊断
                surgery_name = first_surgery.structured_data.get("surgery_name", "")
                if surgery_name:
                    surgery_type = self._classify_from_surgery_name(surgery_name)
                    for stype, keywords in SURGERY_TYPE_RULES:
                        for kw in keywords:
                            if kw in surgery_name:
                                surgery_keywords.add(kw)

        # 合并外部提供的关键词（来自 record_parser）
        if surgery_keywords_hint:
            surgery_keywords.update(surgery_keywords_hint)

        intervention_keywords: Set[str] = set()
        complication_keywords: Set[str] = set()
        severity_keywords: Set[str] = set()
        for e in events:
            text_to_search = e.description + " " + e.raw_text
            for kw in INTERVENTION_KEYWORDS:
                if kw in text_to_search:
                    intervention_keywords.add(kw)
            for kw in COMPLICATION_KEYWORDS:
                if kw in text_to_search:
                    complication_keywords.add(kw)
            for kw in self.severity_terms:
                if kw in text_to_search:
                    severity_keywords.add(kw)

        return TimelineFeatures(
            visit_count=visit_count,
            event_count=len(events),
            total_span_days=total_span_days,
            visit_gaps=visit_gaps,
            diagnosis_keywords=diag_keywords,
            event_type_sequence=seq,
            surgery_type=surgery_type,
            surgery_keywords=surgery_keywords,
            intervention_keywords=intervention_keywords,
            complication_keywords=complication_keywords,
            severity_keywords=severity_keywords,
        )

    @staticmethod
    def _classify_from_surgery_name(surgery_name: str) -> Optional[str]:
        """仅从手术名称中分类（避免匹配到术前诊断等无关文本）"""
        if not surgery_name:
            return None
        for stype, keywords in SURGERY_TYPE_RULES:
            if any(kw in surgery_name for kw in keywords):
                return stype
        return "other"

    def score(self, query: TimelineFeatures, candidate: TimelineFeatures) -> float:
        """计算两个病程的相似度，返回 [0, 1]"""
        # 1. 就诊规模对齐
        max_vc = max(query.visit_count, candidate.visit_count, 1)
        score_visit = min(query.visit_count, candidate.visit_count) / max_vc

        # 2. 时间跨度对齐
        if query.total_span_days > 0 and candidate.total_span_days > 0:
            diff = abs(query.total_span_days - candidate.total_span_days)
            score_span = max(0.0, 1.0 - diff / self.max_span_days)
        else:
            score_span = score_visit  # 退回到就诊规模

        # 3. 诊断变化一致性（Jaccard）
        if query.diagnosis_keywords and candidate.diagnosis_keywords:
            inter = len(query.diagnosis_keywords & candidate.diagnosis_keywords)
            union = len(query.diagnosis_keywords | candidate.diagnosis_keywords)
            score_diag = inter / union if union > 0 else 0.0
        else:
            score_diag = 0.0

        # 4. 事件类型序列相似度（加权LCS）
        score_seq = self._weighted_lcs_similarity(
            query.event_type_sequence,
            candidate.event_type_sequence
        )

        # 5. 手术类型匹配
        score_surgery = self._surgery_type_score(query, candidate)

        # 6. 临床状态匹配：关键支持手段、并发症、危重程度
        score_clinical = self._clinical_state_score(query, candidate)

        # 加权求和
        total = (
            self.weights['visit_scale'] * score_visit +
            self.weights['time_span'] * score_span +
            self.weights['diag'] * score_diag +
            self.weights['seq'] * score_seq +
            self.weights['surgery_type'] * score_surgery +
            self.weights['clinical_state'] * score_clinical
        )
        return float(min(1.0, max(0.0, total)))

    def _surgery_type_score(self, query: TimelineFeatures, candidate: TimelineFeatures) -> float:
        """手术类型软匹配分数（基于 Jaccard 关键词重叠）"""
        has_q = bool(query.surgery_type)
        has_c = bool(candidate.surgery_type)

        # 计算关键词 Jaccard 相似度
        if query.surgery_keywords or candidate.surgery_keywords:
            inter = len(query.surgery_keywords & candidate.surgery_keywords)
            union = len(query.surgery_keywords | candidate.surgery_keywords)
            kw_jaccard = inter / union if union > 0 else 0.0
        else:
            kw_jaccard = 0.0

        # 双方都无手术类型
        if not has_q and not has_c:
            if kw_jaccard > 0:
                return 0.5 + 0.5 * kw_jaccard
            return 1.0

        # 双方同类型
        if has_q and has_c and query.surgery_type == candidate.surgery_type:
            return 0.85 + 0.15 * kw_jaccard

        # 双方不同类型 → 关键词软回退
        if has_q and has_c and query.surgery_type != candidate.surgery_type:
            return kw_jaccard * 0.5

        # 一方有类型、一方无 → 关键词作为桥梁
        return kw_jaccard * 0.8

    def _clinical_state_score(self, query: TimelineFeatures, candidate: TimelineFeatures) -> float:
        """比较关键干预、并发症和危重程度信号。"""
        scores = [
            self._jaccard_or_neutral(query.intervention_keywords, candidate.intervention_keywords, neutral=0.4),
            self._jaccard_or_neutral(query.complication_keywords, candidate.complication_keywords, neutral=0.5),
            self._jaccard_or_neutral(query.severity_keywords, candidate.severity_keywords, neutral=0.4),
        ]
        return float(0.45 * scores[0] + 0.25 * scores[1] + 0.30 * scores[2])

    @staticmethod
    def _jaccard_or_neutral(a: Set[str], b: Set[str], neutral: float) -> float:
        if not a and not b:
            return neutral
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _weighted_lcs_similarity(seq_a: List[str], seq_b: List[str]) -> float:
        """带权最长公共子序列相似度"""
        if not seq_a or not seq_b:
            return 0.0
        m, n = len(seq_a), len(seq_b)

        weight_a = sum(EVENT_WEIGHTS.get(e, 1.0) for e in seq_a)
        weight_b = sum(EVENT_WEIGHTS.get(e, 1.0) for e in seq_b)
        if weight_a == 0 or weight_b == 0:
            return 0.0

        prev = [0.0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0.0] * (n + 1)
            wi = EVENT_WEIGHTS.get(seq_a[i - 1], 1.0)
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    curr[j] = prev[j - 1] + wi
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        lcs_weight = prev[n]
        return lcs_weight / max(weight_a, weight_b)
