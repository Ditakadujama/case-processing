"""
天级病历构建器。

把患者级合并文本拆成 Day 1 / Day 2 / ...，并为每天保留当天文本与截至当天的累计文本。
第一版优先使用 medical_records.visit_date 在合并文本中的就诊块日期，规则事件时间作为后续增强信号。
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

from timeline_parser import TimelineEvent, TimelineParser


VISIT_BLOCK_PATTERN = re.compile(
    r"(###就诊记录[^\n]*-\s*(\d{4}-\d{2}-\d{2})(?:[^\n]*)\n.*?)(?=###就诊记录[^\n]*-\s*\d{4}-\d{2}-\d{2}|\Z)",
    flags=re.DOTALL,
)


@dataclass
class DayRecord:
    """患者某一天的建库单元。"""

    day_record_id: str
    patient_id: str
    day_index: int
    visit_date: str
    day_text: str
    cumulative_text: str
    timeline_events: List[TimelineEvent]


class DayRecordBuilder:
    """从患者级文本构造天级记录。"""

    def __init__(self):
        self.timeline_parser = TimelineParser()

    def build(self, patient_id: str, text: str, max_days: int = 0) -> List[DayRecord]:
        """按入院/首日锚点构建天级记录。max_days <= 0 表示不限天数。"""
        blocks = self._split_visit_blocks(text)
        if not blocks:
            return []

        anchor = min(blocks)
        day_parts: Dict[int, List[str]] = {}
        visit_dates: Dict[int, date] = {}

        for visit_dt in sorted(blocks):
            day_index = (visit_dt - anchor).days + 1
            if day_index <= 0:
                continue
            if max_days > 0 and day_index > max_days:
                continue
            day_parts.setdefault(day_index, []).extend(blocks[visit_dt])
            visit_dates.setdefault(day_index, visit_dt)

        records: List[DayRecord] = []
        cumulative_parts: List[str] = []
        all_events = self.timeline_parser.parse(text)
        events_by_day = self.timeline_parser.group_events_by_day(all_events, anchor)

        for day_index in sorted(day_parts):
            day_text = "\n\n".join(day_parts[day_index]).strip()
            if not day_text:
                continue
            cumulative_parts.append(day_text)
            day_record_id = make_day_record_id(patient_id, day_index)
            records.append(DayRecord(
                day_record_id=day_record_id,
                patient_id=patient_id,
                day_index=day_index,
                visit_date=visit_dates[day_index].isoformat(),
                day_text=day_text,
                cumulative_text="\n\n".join(cumulative_parts),
                timeline_events=events_by_day.get(day_index, []),
            ))

        return records

    def _split_visit_blocks(self, text: str) -> Dict[date, List[str]]:
        """按合并文本中的 ###就诊记录 ... - YYYY-MM-DD 切块。"""
        blocks: Dict[date, List[str]] = {}
        for match in VISIT_BLOCK_PATTERN.finditer(text):
            block_text = match.group(1).strip()
            visit_dt = self._parse_date(match.group(2))
            if visit_dt and block_text:
                blocks.setdefault(visit_dt, []).append(block_text)

        if blocks:
            return blocks

        # 兜底：没有标准就诊块时，用时间轴最早日期作为 Day 1，把全文作为一天。
        events = self.timeline_parser.parse(text)
        anchor_dt = self.timeline_parser.get_admission_anchor(events)
        if anchor_dt:
            return {anchor_dt.date(): [text]}
        return {}

    @staticmethod
    def _parse_date(value: str) -> Optional[date]:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def make_day_record_id(patient_id: str, day_index: int) -> str:
    return f"{patient_id}#D{day_index:03d}"
