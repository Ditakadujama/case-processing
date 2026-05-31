"""
时间轴解析引擎
将扁平病历文本重构为带时间戳的 TimelineEvent 序列
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


@dataclass
class TimelineEvent:
    """时间轴事件"""
    timestamp: datetime
    event_type: str  # admission, lab, exam, surgery, medication, vitals, diagnosis_change, transfer, progress_note, discharge
    source_section: str
    description: str
    raw_text: str
    structured_data: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M") if self.timestamp else "N/A"
        return f"TimelineEvent({ts} | {self.event_type} | {self.description[:40]})"


class TimelineParser:
    """病历时间轴解析器"""

    def __init__(self):
        # 就诊记录块分割正则
        self.visit_block_pattern = re.compile(r"###就诊记录\s+\d+/\d+\s+-\s+(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)")

        # 检验/检查组套时间正则: 组套名 + YYYY-MM-DD HH:MM:SS
        self.lab_datetime_pattern = re.compile(
            r"([^\n:]+?组套\([^)]+\)|[^\n:]+?组套|[^\n:]+?\(X线组套\)|[^\n:]+?\(CT[^)]*\))\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
        )

        # 手术时间正则
        self.surgery_time_pattern = re.compile(
            r"手术日期[：:]?\s*(\d{4}-\d{2}-\d{2})[;；\s]*(?:开始时间)?[：:]?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})?"
        )

        # 病程/记录日期正则
        self.record_date_pattern = re.compile(
            r"(?:日期|记录日期|术后首次病程记录)\s*[：:]?\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)"
        )

        # 监测数据时间-值对正则: HH:MM 数值
        self.monitor_time_value_pattern = re.compile(r"(\d{1,2}:\d{2})\s+([\d.]+|[^,，]+)")

        # 监测指标行正则: 指标名 单位，时间 值,...
        self.monitor_line_pattern = re.compile(
            r"([一-龥][^，,\n]*?[，,]\s*)(.+)$"
        )

        # 药物关键词（用于从医嘱中提取）
        self.drug_keywords = [
            "瑞芬太尼", "丙泊酚", "呋塞米", "多巴酚丁胺", "肾上腺素",
            "青霉素", "哌拉西林", "头孢", "去乙酰毛花苷",
            "艾司奥美拉唑", "奥美拉唑", "利多卡因", "肝素", "葡萄糖酸钙",
            "氯化钠", "葡萄糖", "林格", "甘油", "开塞露", "吲哚美辛",
            "硝酸甘油", "氨溴索", "氨茶碱", "阿司匹林", "氯吡格雷",
            "甲泼尼龙", "地塞米松", "头孢曲松", "头孢他啶", "美罗培南",
            "万古霉素", "替考拉宁", "奥司他韦", "阿奇霉素", "左氧氟沙星",
            "酒石酸布托啡诺", "甘露醇", "注射用甲泼尼龙琥珀酸钠",
            "人血白蛋白", "B型钠尿肽前体"
        ]

        # T节点关键词
        self.diagnosis_change_keywords = ["术后诊断", "目前诊断", "修正诊断", "出院诊断", "入院诊断"]
        self.complication_keywords = ["并发症", "出血", "恶化", "衰竭", "感染", "血栓", "梗死"]
        self.intervention_keywords = ["机械通气", "气管插管", "CRRT", "去甲肾上腺素", "升压", "抗生素升级"]

    def parse(self, text: str) -> List[TimelineEvent]:
        """
        解析病历文本，提取所有带时间戳的事件

        Args:
            text: 合并后的病历文本

        Returns:
            按时间排序的 TimelineEvent 列表
        """
        events: List[TimelineEvent] = []

        # 分割就诊记录块
        blocks = self._split_visit_blocks(text)

        for i, (block_date, block_text) in enumerate(blocks):
            block_events = self._parse_block(block_date, block_text, i, len(blocks))
            events.extend(block_events)

        # 全局提取：从 operation_record 等跨块内容中提取
        global_events = self._parse_global_events(text)
        events.extend(global_events)

        # 按时间排序
        events.sort(key=lambda e: e.timestamp or datetime.min)

        return events

    def _split_visit_blocks(self, text: str) -> List[Tuple[Optional[datetime], str]]:
        """按就诊记录块分割文本，返回 (日期, 块文本) 列表"""
        blocks = []
        matches = list(self.visit_block_pattern.finditer(text))

        if not matches:
            # 没有明确的就诊记录块，整个文本作为一个块
            return [(None, text)]

        for i, match in enumerate(matches):
            date_str = match.group(1)
            block_date = self._parse_datetime(date_str)

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block_text = text[start:end]

            blocks.append((block_date, block_text))

        return blocks

    def _parse_block(self, block_date: Optional[datetime], block_text: str,
                     block_index: int = 0, total_blocks: int = 1) -> List[TimelineEvent]:
        """解析单个就诊记录块，按就诊日期分类事件类型"""
        events = []

        # 一次性切分所有 ### 章节，避免后续逐个 _extract_section 重复扫描全文
        sections = self._split_all_sections(block_text)

        # 双语 lookup
        def _get(*names):
            for name in names:
                content = sections.get(name)
                if content and len(content) > 5:
                    return content
            return None

        # 1. 根据就诊块位置和内容分类
        event_type = self._classify_visit_block(block_text, block_index, total_blocks, sections)
        if block_date:
            events.append(TimelineEvent(
                timestamp=block_date,
                event_type=event_type,
                source_section="visit_record",
                description=f"就诊记录: {block_date.date()}",
                raw_text=block_text[:200],
                structured_data={"date": block_date.isoformat()}
            ))

        # 2. 检验（中/英文 section 名）
        checkout_text = _get("检验", "checkout", "instrument_test")
        if checkout_text:
            events.extend(self._parse_checkout(checkout_text, block_date))

        # 3. 检查
        examine_text = _get("检查", "examine")
        if examine_text:
            events.extend(self._parse_examine(examine_text, block_date))

        # 4. 监测
        monitor_text = _get("监测", "monitor")
        if monitor_text:
            events.extend(self._parse_monitor(monitor_text, block_date))

        # 5. 手术记录
        surgery_text = _get("手术记录", "surgery_record")
        if surgery_text:
            events.extend(self._parse_surgery(surgery_text, block_date))

        # 6. 医嘱/用药
        advice_text = _get("医嘱", "doctor_advice")
        if advice_text and block_date:
            events.extend(self._parse_medications(advice_text, block_date))

        return events

    def _classify_visit_block(self, block_text: str, block_index: int, total_blocks: int,
                               sections: Dict[str, str] = None) -> str:
        """根据就诊块的位置和内容分类事件类型"""
        if sections is None:
            sections = self._split_all_sections(block_text)
        # 首块 → admission
        if block_index == 0:
            return "admission"
        # 末块 → discharge
        if block_index == total_blocks - 1:
            return "discharge"
        # 中间块根据内容分类
        surgery_text = sections.get("手术记录") or sections.get("surgery_record") or ""
        if len(surgery_text) > 5:
            return "surgery"
        checkout_text = sections.get("检验") or sections.get("checkout") or sections.get("instrument_test") or ""
        if len(checkout_text) > 5:
            return "lab"
        if sections.get("医嘱") or sections.get("doctor_advice"):
            return "medication"
        if sections.get("查房记录") or sections.get("inspection_visit") or "主要问题" in block_text:
            return "progress_note"
        return "follow_up"

    def _parse_global_events(self, text: str) -> List[TimelineEvent]:
        """解析跨块的全局事件（operation_record 中的病程、转出记录等）"""
        events = []

        op_text = self._extract_section(text, "operation_record")
        if not op_text:
            return events

        # 过滤掉知情同意书等非病程内容，避免提取到无关日期
        # 分割为子文档：病程记录、转出记录等保留；知情同意书丢弃
        clinical_parts = []
        for part in re.split(r"(知情同意书|高值耗材|医保药品)", op_text):
            if part in ("知情同意书", "高值耗材", "医保药品"):
                continue
            # 保留包含关键临床关键词的部分
            if any(kw in part for kw in ["病程记录", "转出记录", "术后首次", "手术记录", "入院日期", "术后诊断", "目前诊断"]):
                clinical_parts.append(part)
        # 如果没有拆分成功，回退到原始文本（但尝试丢弃知情同意书段落）
        if clinical_parts:
            clinical_text = "\n".join(clinical_parts)
        else:
            # 简单丢弃包含"知情同意书"的大段落
            clinical_text = re.sub(r"知情同意书.*?(?=病程记录|\Z)", "", op_text, flags=re.DOTALL)

        # 1. 提取入院日期 -> admission 事件
        admission_match = re.search(r"入院日期\s+(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)", clinical_text)
        if admission_match:
            dt = self._parse_datetime(admission_match.group(1))
            if dt:
                # 提取入院诊断作为描述
                diag_match = re.search(r"入院诊断\s+([^\n]+)", clinical_text)
                diag_desc = diag_match.group(1).strip()[:60] if diag_match else "入院"
                events.append(TimelineEvent(
                    timestamp=dt,
                    event_type="admission",
                    source_section="operation_record",
                    description=f"入院: {diag_desc}",
                    raw_text=clinical_text[max(0, admission_match.start()-30):admission_match.end()+100],
                    structured_data={}
                ))

        # 2. 提取诊断修正 -> diagnosis_change 事件
        for diag_type in ["术后诊断", "目前诊断", "出院诊断", "修正诊断"]:
            for match in re.finditer(rf"{diag_type}\s+([^\n]+)", clinical_text):
                diag_text = match.group(1).strip()
                # 找前面最近的日期作为时间戳（不限距离，取最后一个）
                all_dates = re.findall(
                    r"(?:日期|记录日期|手术日期)\s*[：:]?\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)",
                    clinical_text[:match.start()]
                )
                if all_dates:
                    dt = self._parse_datetime(all_dates[-1])
                    if dt:
                        events.append(TimelineEvent(
                            timestamp=dt,
                            event_type="diagnosis_change",
                            source_section="operation_record",
                            description=f"{diag_type}: {diag_text[:60]}",
                            raw_text=match.group(0),
                            structured_data={"diagnosis_type": diag_type, "diagnosis": diag_text}
                        ))

        # 3. 提取病程/转出记录中的日期标记
        for match in self.record_date_pattern.finditer(clinical_text):
            date_str = match.group(1)
            record_dt = self._parse_datetime(date_str)
            if not record_dt:
                continue

            # 跳过 "入院日期"（已由 admission 循环单独提取）
            prefix = clinical_text[max(0, match.start() - 6):match.start()]
            if "入院" in prefix:
                continue

            # 获取该日期附近的文本作为描述
            start = max(0, match.start() - 50)
            end = min(len(clinical_text), match.end() + 200)
            context = clinical_text[start:end]

            # 判断事件类型
            event_type = "progress_note"
            if "转出" in context or "转入" in context:
                event_type = "transfer"
            elif "术后" in context and ("诊断" in context or "病程" in context):
                event_type = "diagnosis_change"
            elif "死亡" in context or "出院" in context:
                event_type = "discharge"

            # 过滤：如果上下文包含知情同意书特征，降级为普通 progress_note 或跳过
            if "患者签名" in context or "高值耗材" in context or "自付比例" in context:
                continue

            # 提取更有意义的描述：优先提取 "目前情况"、"诊疗经过" 等段落首句
            desc = self._extract_clinical_description(context)

            events.append(TimelineEvent(
                timestamp=record_dt,
                event_type=event_type,
                source_section="operation_record",
                description=desc,
                raw_text=context,
                structured_data={}
            ))

        # 4. 提取手术时间（如果 surgery_record 没抓到）
        for match in self.surgery_time_pattern.finditer(op_text):
            date_str = match.group(1)
            time_str = match.group(2)
            if time_str:
                surgery_dt = self._parse_datetime(time_str)
            else:
                surgery_dt = self._parse_datetime(date_str)

            if surgery_dt:
                events.append(TimelineEvent(
                    timestamp=surgery_dt,
                    event_type="surgery",
                    source_section="operation_record",
                    description="手术记录",
                    raw_text=op_text[max(0, match.start()-20):match.end()+100],
                    structured_data={}
                ))

        return events

    def _parse_checkout(self, text: str, base_date: Optional[datetime]) -> List[TimelineEvent]:
        """解析检验结果，提取带时间的检验事件"""
        events = []

        for match in self.lab_datetime_pattern.finditer(text):
            lab_name = match.group(1).strip()
            dt_str = match.group(2)
            dt = self._parse_datetime(dt_str)

            if not dt:
                continue

            # 获取该检验的详细内容（从匹配位置到下一个检验或结束）
            start = match.end()
            # 查找下一个检验组套的开始位置
            next_match = self.lab_datetime_pattern.search(text, start)
            end = next_match.start() if next_match else len(text)
            lab_detail = text[start:end].strip()

            # 提取关键指标
            key_values = self._extract_key_lab_values(lab_detail)

            events.append(TimelineEvent(
                timestamp=dt,
                event_type="lab",
                source_section="checkout",
                description=f"检验: {lab_name}",
                raw_text=lab_detail[:300],
                structured_data={"lab_name": lab_name, "values": key_values}
            ))

        return events

    def _parse_examine(self, text: str, base_date: Optional[datetime]) -> List[TimelineEvent]:
        """解析检查结果"""
        events = []

        for match in self.lab_datetime_pattern.finditer(text):
            exam_name = match.group(1).strip()
            dt_str = match.group(2)
            dt = self._parse_datetime(dt_str)

            if not dt:
                continue

            start = match.end()
            next_match = self.lab_datetime_pattern.search(text, start)
            end = next_match.start() if next_match else len(text)
            exam_detail = text[start:end].strip()

            events.append(TimelineEvent(
                timestamp=dt,
                event_type="exam",
                source_section="examine",
                description=f"检查: {exam_name}",
                raw_text=exam_detail[:300],
                structured_data={"exam_name": exam_name}
            ))

        return events

    def _parse_monitor(self, text: str, base_date: Optional[datetime]) -> List[TimelineEvent]:
        """解析监测数据，生成时间序列事件"""
        events = []
        if not base_date:
            return events

        # 监测数据格式: 指标名 ，时间 值,时间 值,...
        # 或者: 指标名 ，时间 值,时间 值,...;
        lines = text.split(";")

        for line in lines:
            line = line.strip()
            if not line or "，" not in line:
                continue

            # 分割指标名和数据部分
            parts = line.split("，", 1)
            if len(parts) != 2:
                continue

            indicator_name = parts[0].strip()
            data_part = parts[1].strip()

            # 去掉末尾的逗号
            data_part = data_part.rstrip(",")

            # 提取所有时间-值对
            pairs = self.monitor_time_value_pattern.findall(data_part)
            if not pairs:
                continue

            # 处理跨天
            current_date = base_date.date()
            last_time_minutes = -1

            for time_str, value_str in pairs:
                try:
                    hour, minute = map(int, time_str.split(":"))
                except ValueError:
                    continue

                time_minutes = hour * 60 + minute

                # 检测跨天：当前时间比上一个时间小（且差距较大，比如超过6小时）
                if last_time_minutes >= 0 and time_minutes < last_time_minutes:
                    # 可能跨天了
                    if last_time_minutes - time_minutes > 360:  # 差距大于6小时认为跨天
                        current_date += timedelta(days=1)

                last_time_minutes = time_minutes

                dt = datetime.combine(current_date, datetime.min.time().replace(hour=hour, minute=minute))

                # 尝试数值化
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    value = value_str.strip()

                events.append(TimelineEvent(
                    timestamp=dt,
                    event_type="vitals",
                    source_section="monitor",
                    description=f"{indicator_name}: {value}",
                    raw_text=f"{indicator_name} {time_str} {value}",
                    structured_data={
                        "indicator": indicator_name,
                        "value": value,
                        "time": time_str
                    }
                ))

        return events

    def _parse_surgery(self, text: str, base_date: Optional[datetime]) -> List[TimelineEvent]:
        """解析手术记录"""
        events = []

        for match in self.surgery_time_pattern.finditer(text):
            date_str = match.group(1)
            time_str = match.group(2)

            if time_str:
                surgery_dt = self._parse_datetime(time_str)
            else:
                surgery_dt = self._parse_datetime(date_str)

            if not surgery_dt:
                continue

            # 提取手术名称
            name_match = re.search(r"手术名称[：:]?\s*([^\n;]+)", text)
            surgery_name = name_match.group(1).strip() if name_match else "手术"

            events.append(TimelineEvent(
                timestamp=surgery_dt,
                event_type="surgery",
                source_section="surgery_record",
                description=f"手术: {surgery_name}",
                raw_text=text[max(0, match.start()-50):match.end()+200],
                structured_data={"surgery_name": surgery_name}
            ))

        return events

    def _parse_medications(self, text: str, base_date: datetime) -> List[TimelineEvent]:
        """从医嘱中提取用药事件（以就诊日期为默认时间）"""
        events = []

        for drug in self.drug_keywords:
            if drug in text:
                # 提取剂量（简单正则）
                dosage = ""
                dosage_match = re.search(
                    rf"{re.escape(drug)}[^0-9]*?(\d+(?:\.\d+)?)\s*(?:g|mg|ml|万|iu|ug)?",
                    text
                )
                if dosage_match:
                    dosage = dosage_match.group(1)

                events.append(TimelineEvent(
                    timestamp=base_date,
                    event_type="medication",
                    source_section="doctor_advice",
                    description=f"用药: {drug} {dosage}".strip(),
                    raw_text=f"{drug} {dosage}".strip(),
                    structured_data={"drug": drug, "dosage": dosage}
                ))

        return events

    def get_admission_anchor(self, events: List[TimelineEvent]) -> Optional[datetime]:
        """返回窗口截取起点：最早 admission，否则最早有效事件时间戳。"""
        valid_events = [e for e in events if e.timestamp]
        if not valid_events:
            return None

        admission_events = [e for e in valid_events if e.event_type == "admission"]
        if admission_events:
            return min(e.timestamp for e in admission_events)

        return min(e.timestamp for e in valid_events)

    def get_first_days_snapshot(
        self,
        events: List[TimelineEvent],
        days: int,
    ) -> List[TimelineEvent]:
        """截取从入院起点开始的前 N 天事件。days <= 0 时返回完整事件副本。"""
        if days <= 0:
            return list(events)

        anchor = self.get_admission_anchor(events)
        if anchor is None:
            return list(events)

        cutoff = anchor + timedelta(days=days)
        return [
            e for e in events
            if e.timestamp and anchor <= e.timestamp < cutoff
        ]

    def _extract_key_lab_values(self, text: str) -> Dict[str, str]:
        """从检验详情中提取关键指标数值"""
        values = {}
        # 简单提取常见指标
        patterns = {
            "Lac": r"Lac\s+(\d+\.?\d*)",
            "pH": r"pH\s+(\d+\.?\d*)",
            "白细胞": r"白细胞计数\s+(\d+\.?\d*)",
            "肌酐": r"肌酐\s+(\d+\.?\d*)",
            "血小板": r"血小板计数\s+(\d+\.?\d*)",
            "氧合指数": r"氧合指数\s+(\d+\.?\d*)",
            "D-二聚体": r"D-二聚体.*?\s+(\d+\.?\d*)",
            "降钙素原": r"降钙素原.*?\s+(\d+\.?\d*)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                values[key] = match.group(1)
        return values

    def get_snapshot(self, events: List[TimelineEvent], cutoff_time: datetime) -> List[TimelineEvent]:
        """
        生成某个时间点的可见快照（防未来泄露）

        Args:
            events: 完整时间轴事件列表
            cutoff_time: 截断时间点

        Returns:
            cutoff_time 之前（含）的所有事件
        """
        return [e for e in events if e.timestamp and e.timestamp <= cutoff_time]

    def generate_standard_nodes(self, events: List[TimelineEvent]) -> Dict[str, Optional[TimelineEvent]]:
        """
        从时间轴生成 T0-T6 标准节点

        Returns:
            {"T0": event, "T1": event, ...}
        """
        nodes: Dict[str, Optional[TimelineEvent]] = {
            "T0": None, "T1": None, "T2": None,
            "T3": None, "T4": None, "T5": None, "T6": None
        }

        if not events:
            return nodes

        # T0: 最早的 admission 事件
        admission_events = [e for e in events if e.event_type == "admission"]
        if admission_events:
            nodes["T0"] = min(admission_events, key=lambda e: e.timestamp)

        t0_time = nodes["T0"].timestamp if nodes["T0"] else events[0].timestamp

        # T1: T0 后 24h 内的首批 lab + exam（取最早的）
        t1_candidates = [
            e for e in events
            if e.event_type in ("lab", "exam")
            and e.timestamp > t0_time
            and (e.timestamp - t0_time).total_seconds() <= 86400
        ]
        if t1_candidates:
            nodes["T1"] = min(t1_candidates, key=lambda e: e.timestamp)

        # T2: 诊断修正事件（progress_note 或已标记为 diagnosis_change）
        t2_candidates = [
            e for e in events
            if (e.event_type == "diagnosis_change"
                or (e.event_type == "progress_note"
                    and any(kw in e.raw_text for kw in self.diagnosis_change_keywords)))
        ]
        if t2_candidates:
            nodes["T2"] = min(t2_candidates, key=lambda e: e.timestamp)

        # T3: 首次重要干预（surgery 或含干预关键词的 medication/progress_note）
        t3_candidates = [
            e for e in events
            if e.event_type == "surgery"
            or (e.event_type in ("medication", "progress_note")
                and any(kw in e.description + e.raw_text for kw in self.intervention_keywords))
        ]
        if t3_candidates:
            nodes["T3"] = min(t3_candidates, key=lambda e: e.timestamp)

        # T4: T3 后 6-24h 内的关键指标变化（取 T3 后最早的 lab/vitals）
        t3_time = nodes["T3"].timestamp if nodes["T3"] else None
        if t3_time:
            t4_candidates = [
                e for e in events
                if e.event_type in ("lab", "vitals")
                and e.timestamp > t3_time
                and (e.timestamp - t3_time).total_seconds() <= 86400
            ]
            if t4_candidates:
                nodes["T4"] = min(t4_candidates, key=lambda e: e.timestamp)

        # T5: 包含并发症关键词的 progress_note
        t5_candidates = [
            e for e in events
            if e.event_type in ("progress_note", "transfer")
            and any(kw in e.raw_text for kw in self.complication_keywords)
        ]
        if t5_candidates:
            nodes["T5"] = min(t5_candidates, key=lambda e: e.timestamp)

        # T6: 最后一个 transfer/discharge（转归）
        t6_candidates = [e for e in events if e.event_type in ("transfer", "discharge")]
        if t6_candidates:
            nodes["T6"] = max(t6_candidates, key=lambda e: e.timestamp)
        elif events:
            # 如果没有明确的 transfer/discharge，用最后一个事件
            nodes["T6"] = max(events, key=lambda e: e.timestamp)

        return nodes

    def _parse_datetime(self, s: str) -> Optional[datetime]:
        """解析多种格式的日期时间字符串"""
        if not s:
            return None
        s = s.strip()
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    # 一次性切分所有 ### 章节的正则（类级别，复用）
    _section_split_pattern = re.compile(r'###([^：:\n]+)[：:]?(.*?)(?=###[^\n]|\Z)', re.DOTALL)

    def _split_all_sections(self, text: str) -> Dict[str, str]:
        """一次正则扫描切分所有 ### 章节，返回 {章节名: 内容}（用于避免重复扫描）"""
        sections = {}
        for match in self._section_split_pattern.finditer(text):
            sections[match.group(1).strip()] = match.group(2).strip()
        return sections

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """提取病历的某个章节（单次查找时使用；批量查找请用 _split_all_sections）"""
        # 直接委托给 _split_all_sections 然后取单个 key
        return self._split_all_sections(text).get(section_name)

    def _extract_first_line(self, text: str) -> str:
        """提取文本的第一行非空内容"""
        for line in text.split("\n"):
            line = line.strip()
            if line:
                return line[:80]
        return text[:80]

    def _extract_clinical_description(self, context: str) -> str:
        """从病程/转出记录上下文中提取有意义的临床描述"""
        # 优先提取关键段落的首句
        priority_patterns = [
            (r"目前情况\s*[：:]?\s*([^\n]{10,80})", 1),
            (r"目前诊断\s*[：:]?\s*([^\n]{10,80})", 1),
            (r"转科目的\s*[：:]?\s*([^\n]{10,80})", 1),
            (r"诊疗经过\s*[：:]?\s*([^\n]{10,80})", 1),
            (r"提醒接受科室注意事项\s*[：:]?\s*([^\n]{10,80})", 1),
            (r"患者[^，]+，[^。]{10,60}", 0),
        ]
        for pattern, group_idx in priority_patterns:
            match = re.search(pattern, context)
            if match:
                desc = match.group(group_idx).strip()
                # 清理无意义后缀
                desc = re.sub(r"[。；]+$", "", desc)
                return desc[:80]

        # 回退：提取第一行有意义的非空行
        for line in context.split("\n"):
            line = line.strip()
            if line and len(line) > 5 and not line.startswith("日期") and not line.startswith("记录"):
                return line[:80]
        return context[:80].strip()


def print_timeline(events: List[TimelineEvent], max_events: int = 50) -> None:
    """打印时间轴概览"""
    print(f"\n时间轴概览（共 {len(events)} 个事件）:\n")

    # 按类型统计
    from collections import Counter
    type_counts = Counter(e.event_type for e in events)
    print("事件类型分布:")
    for etype, count in type_counts.most_common():
        print(f"  {etype:20s}: {count:4d}")

    print(f"\n前 {max_events} 个事件（按时间排序）:\n")
    for i, event in enumerate(events[:max_events], 1):
        ts = event.timestamp.strftime("%m-%d %H:%M") if event.timestamp else "N/A"
        desc = event.description[:50] + "..." if len(event.description) > 50 else event.description
        print(f"  {i:3d}. [{ts}] ({event.event_type:12s}) {desc}")


def print_standard_nodes(nodes: Dict[str, Optional[TimelineEvent]]) -> None:
    """打印 T0-T6 标准节点"""
    print("\n标准病程节点 (T0-T6):\n")
    for key in ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]:
        event = nodes[key]
        if event:
            ts = event.timestamp.strftime("%Y-%m-%d %H:%M") if event.timestamp else "N/A"
            print(f"  {key}: [{ts}] {event.event_type} | {event.description[:60]}")
        else:
            print(f"  {key}: (未识别)")


if __name__ == "__main__":
    import sys

    # 测试：读取第一个病历文件
    import os
    records_dir = "./data/records"
    if not os.path.exists(records_dir):
        print(f"目录不存在: {records_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(records_dir) if f.endswith(".txt")]
    if not files:
        print("没有找到 .txt 病历文件")
        sys.exit(1)

    filepath = os.path.join(records_dir, files[0])
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    parser = TimelineParser()
    events = parser.parse(text)

    print_timeline(events)
    nodes = parser.generate_standard_nodes(events)
    print_standard_nodes(nodes)
