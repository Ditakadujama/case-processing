"""
病例卡向量文本生成 — 从病例卡核心字段程序化拼接 embedding 用文本。

设计原则：
- LLM 只负责抽取结构化病例卡，不负责生成 embedding 专用 summary。
- 程序拼接保证字段顺序稳定、长度可控、内容可预期。
- 不拼接 evidence、不拼接完整原文、day_summary 放最后。
"""

import re
from typing import Dict, List, Optional


def build_case_card_embedding_text(card: dict, max_chars: int = 2000) -> str:
    """
    从每日病例卡 dict 中提取核心字段，按固定顺序拼接为 embedding 用文本。

    Args:
        card: LLM 抽取的日级病例卡 dict
        max_chars: 拼接文本最大字符数（中文字符），默认 2000

    Returns:
        拼接后的病例卡核心字段文本，空字段不拼接
    """
    if not card:
        return ""

    lines: List[str] = []

    # ── 1. 最终诊断 ──
    _append_list(lines, "最终诊断", card.get("final_diagnoses"))

    # ── 2. 主诊断轴 ──
    _append_str(lines, "主诊断轴", card.get("primary_diagnosis_axis"))

    # ── 3. 病因轴 ──
    _append_str(lines, "病因轴", card.get("etiology_axis"))

    # ── 4-10. 病因链各子字段 ──
    chain = card.get("etiology_chain")
    if isinstance(chain, dict):
        _append_str(lines, "疾病大类", chain.get("disease_category"))
        _append_str(lines, "直接病因", chain.get("direct_cause"))
        _append_str(lines, "来源/部位", chain.get("source_or_site"))
        _append_str(lines, "解剖位置", chain.get("anatomic_location"))
        _append_str(lines, "基础诱因", chain.get("underlying_trigger"))
        _append_list(lines, "病理过程", chain.get("pathophysiology"))
        _append_list(lines, "关键处理", chain.get("key_interventions"))

    # ── 11. 基线病史/基础病 ──
    _append_list(lines, "基线病史", card.get("baseline_context"))

    # ── 12. 当天新增诊断 ──
    _append_list(lines, "当天新增诊断", card.get("new_diagnoses"))

    # ── 13. 当天新增干预 ──
    _append_list(lines, "当天新增干预", card.get("new_interventions"))

    # ── 14. 手术/操作 ──
    operations = card.get("operations")
    if operations:
        op_names = _extract_names(operations)
        _append_list(lines, "手术/操作", op_names)

    # ── 15. 器官状态 ──
    organ_status = card.get("organ_status")
    if organ_status:
        org_names = _extract_names(organ_status)
        _append_list(lines, "器官状态", org_names)

    # ── 15. 并发症 ──
    complications = card.get("complications")
    if complications:
        comp_names = _extract_names(complications)
        _append_list(lines, "并发症", comp_names)

    # ── 16. 临床状态 ──
    _append_str(lines, "临床状态", card.get("clinical_state"))

    # ── 17. 当天摘要（放最后，避免压过结构化字段）──
    _append_str(lines, "当天摘要", card.get("day_summary"))

    if not lines:
        return ""

    text = "\n".join(lines)

    # 长度上限截断（在完整句子边界截断）
    if len(text) > max_chars:
        text = _truncate_at_sentence_boundary(text, max_chars)

    return text


def _extract_names(items) -> List[str]:
    """从 dict 列表中提取 name 字段，兼容纯字符串列表。"""
    result = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("label") or ""
            if name:
                result.append(str(name).strip())
        elif isinstance(item, str):
            item = item.strip()
            if item:
                result.append(item)
    return result


def _append_str(lines: List[str], label: str, value: Optional[str]) -> None:
    """添加单值行：标签：值"""
    if value and str(value).strip():
        lines.append(f"{label}：{str(value).strip()}")


def _append_list(lines: List[str], label: str, values) -> None:
    """添加列表行：标签：值1；值2；值3"""
    if not values:
        return
    if isinstance(values, (list, tuple, set)):
        items = [str(v).strip() for v in values if v and str(v).strip()]
    else:
        val = str(values).strip()
        items = [val] if val else []
    if items:
        lines.append(f"{label}：{'；'.join(items)}")


def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    """在 max_chars 附近最近的句子边界截断文本。"""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # 在最后一个句号、换行处截断
    for sep in ("。", "\n", "；", "，", "、"):
        idx = truncated.rfind(sep)
        if idx > max_chars * 0.6:
            return truncated[:idx + len(sep)]

    return truncated
