"""
医生关注文本过滤。

medical_records 保留原始字段；本模块只在 build/search 时生成用于相似度的日级文本。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from record_fingerprint import (
    build_extraction_input_hash,
    build_source_hash,
)


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
    # Stage 1: stable identity and content hashes
    source_table: str = "medical_records"
    source_record_id: str = ""
    source_hash: str = ""
    extraction_input_hash: str = ""


def make_day_record_id(source_table: str, source_record_id: int | str) -> str:
    """稳定 ID：基于源表和源记录 ID，不依赖可变的 day_index。

    格式: {source_table}:{source_record_id}
    示例: medical_records:42
    """
    return f"{source_table}:{source_record_id}"


def make_legacy_day_record_id(patient_id: str, day_index: int) -> str:
    """旧版 ID（基于患者+住院日序号）- 已废弃，仅保留用于兼容。

    .. deprecated::
        请使用 ``make_day_record_id(source_table, source_record_id)``。
    """
    warnings.warn(
        "make_legacy_day_record_id is deprecated. Use make_day_record_id(source_table, source_record_id) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"{patient_id}#D{day_index:03d}"


def build_clinical_day_records(
    rows: Iterable[Dict[str, Any]],
    max_days: int = 0,
    *,
    source_table: str = "medical_records",
    history_mode: str = "all",
    history_window_days: int = 7,
    history_max_chars: int = 30000,
) -> List[ClinicalDayRecord]:
    """从原始数据库日行构造过滤后的日级记录。

    ``cumulative_text`` 只在当前进程内作为 LLM 输入使用，不应再持久化到
    ``record_days``。其内容由配置的历史窗口确定，并参与抽取输入哈希。
    """
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

    for row in sorted_rows:
        patient_id = str(row.get("patient_id") or "").strip()
        if not patient_id:
            continue
        if patient_id != current_patient:
            current_patient = patient_id
            day_index = 0
        day_index += 1
        if max_days > 0 and day_index > max_days:
            continue

        day_text = build_clinical_day_text(row, day_index=day_index)
        if not day_text.strip():
            continue
        source_record_id = str(row.get("id") or "")
        if not source_record_id:
            # 查询数据不会持久化，但仍为每行生成唯一且确定的临时身份。
            source_record_id = f"{patient_id}:{row.get('visit_date') or ''}:{day_index}"

        records.append(ClinicalDayRecord(
            day_record_id=make_day_record_id(source_table, source_record_id),
            patient_id=patient_id,
            day_index=day_index,
            visit_date=str(row.get("visit_date") or ""),
            day_text=day_text,
            cumulative_text="",
            source_table=source_table,
            source_record_id=source_record_id,
            source_hash=build_source_hash(row),
            extraction_input_hash="",
        ))

    by_patient: Dict[str, List[ClinicalDayRecord]] = {}
    for record in records:
        by_patient.setdefault(record.patient_id, []).append(record)

    for patient_days in by_patient.values():
        for record in patient_days:
            history_context = build_history_context(
                patient_days,
                record.day_index,
                mode=history_mode,
                window_days=history_window_days,
                max_chars=history_max_chars,
            )
            record.cumulative_text = "\n\n".join(
                part for part in (history_context, record.day_text) if part
            )
            record.extraction_input_hash = build_extraction_input_hash(
                record.day_text,
                record.cumulative_text,
            )

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


# ═══════════════════════════════════════════════════════════════════
# Stage 6: 动态历史上下文（替代持久化 cumulative_text）
# ═══════════════════════════════════════════════════════════════════

def build_history_context(
    all_days: List[ClinicalDayRecord],
    current_day_index: int,
    mode: str = "window",
    window_days: int = 7,
    max_chars: int = 30000,
) -> str:
    """动态构造 LLM 抽取用的历史上下文（不持久化）。

    替代 record_days.cumulative_text 列。上下文从内存中的前几日 day_text 构造。

    Args:
        all_days: 该患者的所有 ClinicalDayRecord，按 day_index 排序。
        current_day_index: 当前处理的天（1-indexed）。
        mode: "all" 返回所有前日；"window" 仅返回最近 N 天。
        window_days: window 模式下包含的前日数。
        max_chars: 上下文总长度的硬限制。

    Returns:
        历史上下文字符串（前日文本用分隔符拼接）。
        无前日时返回空字符串。
    """
    prior_days = [d for d in all_days if d.day_index < current_day_index]

    if mode == "window":
        prior_days = prior_days[-window_days:]

    if not prior_days:
        return ""

    parts = [
        f"### Day {d.day_index} ({d.visit_date})\n{d.day_text}"
        for d in prior_days
    ]
    context = "\n\n".join(parts)

    if len(context) <= max_chars:
        return context

    # 确定性截断：保留首日（基线诊断信息）和最近几日
    first_day_text = parts[0] if parts else ""
    separator = "\n\n...[中间省略]...\n\n"
    remaining = max_chars - len(first_day_text) - len(separator)

    if remaining <= 0:
        return first_day_text[:max_chars]

    if mode == "window" and len(parts) > 1:
        tail_count = max(window_days - 1, 0)
        tail_parts = parts[-tail_count:] if tail_count else []
    else:
        tail_parts = parts[1:]
    tail_text = ""
    for p in reversed(tail_parts):
        if len(tail_text) + len(p) + 4 <= remaining:
            tail_text = p + "\n\n" + tail_text
        else:
            break

    context = first_day_text + separator + tail_text
    return context
