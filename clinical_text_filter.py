"""
医生关注文本过滤。

medical_records 保留原始字段；本模块只在 build/search 时生成用于相似度的日级文本。
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


KEEP_OPERATION_LABELS = (
    "首程",
    "首次病程",
    "术后首次病程",
    "日常病程",
    "普通病程",
    "病程记录",
    "查房记录",
)

DROP_OPERATION_LABELS = (
    "知情同意",
    "会诊",
    "入院记录",
    "出院记录",
    "转入",
    "转出",
    "授权",
    "医保",
    "高值耗材",
    "谈话记录",
    "告知",
    "同意书",
)

EMPTY_HEADER_RE = re.compile(r"^#+\s*[\u4e00-\u9fa5A-Za-z_ ]+[：:]?\s*$")
SECTION_RE = re.compile(
    r"(#{2,6}\s*([^#\n：:]{0,40}?)(?:[：:]|\n).*?)(?=\n#{2,6}\s*[^#\n：:]{0,40}?(?:[：:]|\n)|\Z)",
    flags=re.DOTALL,
)
DIAGNOSIS_LINE_RE = re.compile(
    r"([^。\n；;]{0,20}(?:诊断|确诊|考虑|符合|倾向|病因|冠心病|心肌梗死|心梗|肺栓塞|主动脉夹层|感染性休克|脓毒症)[^。\n；;]{0,120})"
)


@dataclass
class ClinicalDayRecord:
    """一个患者某一天的相似度建库单元。"""

    day_record_id: str
    patient_id: str
    day_index: int
    visit_date: str
    day_text: str
    cumulative_text: str


def make_day_record_id(patient_id: str, day_index: int) -> str:
    return f"{patient_id}#D{day_index:03d}"


def build_clinical_day_records(rows: Iterable[Dict[str, Any]], max_days: int = 0) -> List[ClinicalDayRecord]:
    """从原始数据库日行构造过滤后的日级记录。"""
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("patient_id") or ""),
            str(r.get("visit_date") or ""),
            int(r.get("id") or 0),
        ),
    )
    records: List[ClinicalDayRecord] = []
    current_patient = None
    day_index = 0
    cumulative_parts: List[str] = []

    for row in sorted_rows:
        patient_id = str(row.get("patient_id") or "").strip()
        if not patient_id:
            continue
        if patient_id != current_patient:
            current_patient = patient_id
            day_index = 0
            cumulative_parts = []
        day_index += 1
        if max_days > 0 and day_index > max_days:
            continue

        day_text = build_clinical_day_text(row, day_index=day_index)
        if not day_text.strip():
            continue
        cumulative_parts.append(day_text)
        records.append(ClinicalDayRecord(
            day_record_id=make_day_record_id(patient_id, day_index),
            patient_id=patient_id,
            day_index=day_index,
            visit_date=str(row.get("visit_date") or ""),
            day_text=day_text,
            cumulative_text="\n\n".join(cumulative_parts),
        ))

    return records


def build_clinical_day_text(row: Dict[str, Any], day_index: int = 0) -> str:
    """生成医生关注的单日相似度文本。"""
    patient_id = str(row.get("patient_id") or "").strip()
    visit_date = str(row.get("visit_date") or "").strip()
    parts = [
        f"###患者ID: {patient_id}",
        f"###日期: {visit_date}",
    ]
    if day_index:
        parts.append(f"###住院日序号: {day_index}")
    has_clinical_content = False

    history = _clean_text(row.get("history_illness"))
    if history:
        has_clinical_content = True
        parts.append(f"###既往病史\n{history}")

    diagnosis_clues = extract_diagnosis_clues(row)
    if diagnosis_clues:
        has_clinical_content = True
        parts.append("###诊断线索\n" + "\n".join(diagnosis_clues))

    inspection = _clean_text(row.get("inspection_visit"))
    if inspection and not _is_empty_header(inspection):
        has_clinical_content = True
        parts.append(f"###查房记录\n{inspection}")

    operation_notes = extract_relevant_operation_notes(row.get("operation_record"))
    if operation_notes:
        has_clinical_content = True
        parts.append("###病程/首程/查房记录\n" + "\n\n".join(operation_notes))

    if not has_clinical_content:
        return ""
    return "\n\n".join(parts).strip()


def extract_relevant_operation_notes(value: Any) -> List[str]:
    """从 operation_record 中提取首程、病程、查房相关子文档。"""
    text = _clean_text(value)
    if not text:
        return []

    sections = []
    for match in SECTION_RE.finditer(text):
        section = match.group(1).strip()
        label = (match.group(2) or "").strip()
        if _keep_operation_section(label, section):
            sections.append(section)

    if sections:
        return _dedupe_keep_order(sections)

    if _keep_operation_section("", text):
        return [text]
    return []


def extract_diagnosis_clues(row: Dict[str, Any], limit: int = 12) -> List[str]:
    """从原始日行里抽取诊断/病因线索，只作为相似度辅助，不引入整段检验医嘱噪声。"""
    clues: List[str] = []
    for col in (
        "patient_info",
        "history_illness",
        "inspection_visit",
        "operation_record",
        "chief_complaint",
        "examine",
        "surgery_record",
    ):
        text = _clean_text(row.get(col))
        if not text:
            continue
        for match in DIAGNOSIS_LINE_RE.finditer(text):
            clue = re.sub(r"\s+", " ", match.group(1)).strip()
            if clue:
                clues.append(f"{col}: {clue}")
            if len(clues) >= limit:
                return _dedupe_keep_order(clues)
    return _dedupe_keep_order(clues)


def _keep_operation_section(label: str, section: str) -> bool:
    haystack = f"{label}\n{section}"
    if any(word in haystack for word in DROP_OPERATION_LABELS):
        return False
    return any(word in haystack for word in KEEP_OPERATION_LABELS)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_empty_header(text: str) -> bool:
    return bool(EMPTY_HEADER_RE.match(text.strip()))


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = re.sub(r"\s+", "", item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
