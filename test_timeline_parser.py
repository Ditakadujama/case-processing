"""
时间轴解析引擎单元测试
"""

import os
import unittest
from datetime import datetime, timedelta

from timeline_parser import TimelineParser, TimelineEvent


class TestTimelineParser(unittest.TestCase):
    """测试 TimelineParser"""

    @classmethod
    def setUpClass(cls):
        cls.parser = TimelineParser()
        cls.sample_path = "./data/records/测试病例_ZY010101703814.txt"
        if os.path.exists(cls.sample_path):
            with open(cls.sample_path, "r", encoding="utf-8") as f:
                cls.sample_text = f.read()
        else:
            cls.sample_text = ""

    def test_parse_returns_events(self):
        """解析应返回非空事件列表"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        self.assertGreater(len(events), 100, "事件数应大于100")

    def test_events_sorted_by_time(self):
        """事件应按时间排序"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        timestamps = [e.timestamp for e in events if e.timestamp]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_contains_vitals(self):
        """应包含生命体征事件"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        vitals = [e for e in events if e.event_type == "vitals"]
        self.assertGreater(len(vitals), 50, "应有大量生命体征事件")

    def test_contains_lab_events(self):
        """应包含检验事件"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        labs = [e for e in events if e.event_type == "lab"]
        self.assertGreaterEqual(len(labs), 2, "应有至少2个检验事件")

    def test_contains_surgery_event(self):
        """应包含手术事件"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        surgeries = [e for e in events if e.event_type == "surgery"]
        self.assertGreaterEqual(len(surgeries), 1, "应有至少1个手术事件")

    def test_day_crossing_in_monitor(self):
        """监测数据跨天应正确处理"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        vitals = [e for e in events if e.event_type == "vitals"]

        # 找到 23:00 和次日 00:00 的心率事件
        heart_rates = [e for e in vitals if "心率" in e.description]
        self.assertGreater(len(heart_rates), 10, "应有心率事件")

        # 验证至少存在一个跨天（同一天 23:00 后接 00:00）
        dates_with_23 = set()
        for e in heart_rates:
            if e.timestamp and e.timestamp.hour == 23:
                dates_with_23.add(e.timestamp.date())

        dates_with_00 = set()
        for e in heart_rates:
            if e.timestamp and e.timestamp.hour == 0:
                dates_with_00.add(e.timestamp.date())

        # 00:00 的日期应该是 23:00 日期的下一天
        for d23 in dates_with_23:
            next_day = d23 + timedelta(days=1)
            self.assertIn(next_day, dates_with_00,
                          f"日期 {d23} 23:00 后应有 {next_day} 00:00 的心率事件")

    def test_snapshot_excludes_future(self):
        """快照不应包含未来事件"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)

        # 以第一个事件时间 + 12h 作为截断
        first_time = events[0].timestamp
        cutoff = first_time + timedelta(hours=12)
        snapshot = self.parser.get_snapshot(events, cutoff)

        for e in snapshot:
            self.assertLessEqual(e.timestamp, cutoff,
                                 f"快照中的事件 {e} 不应晚于截断时间")

        # 快照应比总事件少（如果总时间跨度大于12h）
        if (events[-1].timestamp - first_time).total_seconds() > 43200:
            self.assertLess(len(snapshot), len(events),
                            "12h 快照应少于总事件数")

    def test_standard_nodes(self):
        """T0-T6 标准节点应能识别核心节点"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        nodes = self.parser.generate_standard_nodes(events)

        # T0（入院）必须存在
        self.assertIsNotNone(nodes.get("T0"), "T0 入院节点必须存在")
        self.assertEqual(nodes["T0"].event_type, "admission")

        # T3（重要干预/手术）应存在
        self.assertIsNotNone(nodes.get("T3"), "T3 干预节点应存在")

        # T6（转归/最后事件）必须存在
        self.assertIsNotNone(nodes.get("T6"), "T6 转归节点必须存在")

    def test_event_types(self):
        """应覆盖主要事件类型"""
        if not self.sample_text:
            self.skipTest("样本文件不存在")
        events = self.parser.parse(self.sample_text)
        types_found = set(e.event_type for e in events)

        required_types = {"admission", "lab", "vitals", "surgery", "medication"}
        for rt in required_types:
            self.assertIn(rt, types_found, f"应包含 {rt} 类型事件")


if __name__ == "__main__":
    unittest.main(verbosity=2)
