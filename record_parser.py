"""
病历解析器
从病历文本中提取结构化信息
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple


@dataclass
class PatientInfo:
    """患者信息"""
    name: str = ""
    gender: str = ""
    age: int = 0
    height: float = 0.0
    weight: float = 0.0
    ideal_weight: float = 0.0


@dataclass
class LabResult:
    """检验结果"""
    name: str
    value: float
    unit: str
    status: str  # 正常/高/低


@dataclass
class Diagnosis:
    """诊断信息"""
    name: str
    icd_code: str = ""


@dataclass
class Medication:
    """用药信息"""
    name: str
    dosage: str = ""
    route: str = ""


@dataclass
class MedicalRecord:
    """结构化病历"""
    patient: PatientInfo = field(default_factory=PatientInfo)
    dates: List[str] = field(default_factory=list)
    diagnoses: List[Diagnosis] = field(default_factory=list)
    lab_results: Dict[str, List[LabResult]] = field(default_factory=dict)
    medications: List[Medication] = field(default_factory=list)
    vital_signs: Dict[str, List] = field(default_factory=dict)
    raw_text: str = ""
    surgery_type: str = ""       # 手术专科类型: neuro/cardiac/general/ortho/other
    surgery_name: str = ""       # 手术名称
    surgery_keywords: Set[str] = field(default_factory=set)  # 手术/专科关键词（用于软匹配）


class MedicalRecordParser:
    """病历解析器"""

    def __init__(self):
        # 患者信息正则 - 匹配 "姓名：xxx，性别：x，年龄：xx岁"
        self.patient_pattern = re.compile(
            r"姓名[：:]([^\s，,]+)[，,]\s*性别[：:]([男女])[，,\s]+年龄[：:](\d+)岁"
        )
        self.height_weight_pattern = re.compile(
            r"身高[：:](\d+(?:\.\d+)?)cm[，,]\s*体重[：:](\d+(?:\.\d+)?)kg"
        )

        # 日期正则
        self.date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

        # 检验结果正则: 名称 数值 单位 状态
        self.lab_pattern = re.compile(
            r"([^\s\d]+[^\s：:]*)\s+(\d+(?:\.\d+)?)\s*([^\s]+)\s*([正常高低]+)"
        )

        # 用药正则
        self.medication_pattern = re.compile(
            r"([^\s\d]+[^\s（]+)（[^）]+\）\s*(\d+(?:\.\d+)?(?:g|mg|ml|万)?)\s*(?:ml|静滴|泵入|口服|皮试|肛入|静推)?"
        )

        # 诊断关键词
        self.diagnosis_keywords = [
            "颅脑损伤", "脑梗死", "肺部感染", "呼吸衰竭", "高血压",
            "心房颤动", "传导阻滞", "胸腔积液", "肺不张", "肺炎",
            "胆红素", "心肌损伤", "肾功能", "肝功能", "感染"
        ]

        # 手术分类规则: 关键词 -> 专科类型
        self.surgery_type_rules = [
            ("neuro", ["脊髓", "脑室", "颅脑", "脑", "血肿清除", "开颅", "神经内镜", "硬脊膜", "椎板", "髓内", "颅内", "脑膜", "神经减压", "听神经瘤"]),
            ("cardiac", ["冠脉", "心脏", "搭桥", "CABG", "PCI", "瓣膜", "射频消融", "旁路移植", "心房", "心室", "起搏器", "主动脉球囊反搏", "主动脉", "二尖瓣", "三尖瓣", "房间隔", "室间隔"]),
            ("general", ["肝脏", "胆囊", "胰腺", "胃肠", "阑尾", "脾", "甲状腺", "胃切除", "肠切除", "结肠", "直肠", "胆道", "腹腔"]),
            ("ortho", ["骨折", "关节置换", "髋关节", "膝关节", "肩关节", "椎间盘", "椎弓根", "椎管", "植骨融合", "骨科", "肌腱", "韧带", "半月板", "颈椎", "胸椎", "腰椎", "脊柱", "截骨"]),
        ]

    def parse(self, text: str) -> MedicalRecord:
        """解析病历文本"""
        record = MedicalRecord(raw_text=text)

        # 提取患者信息
        record.patient = self.extract_patient_info(text)

        # 提取日期
        record.dates = self.date_pattern.findall(text)

        # 提取诊断
        record.diagnoses = self.extract_diagnoses(text)

        # 提取检验结果
        record.lab_results = self.extract_lab_results(text)

        # 提取用药
        record.medications = self.extract_medications(text)

        # 提取生命体征
        record.vital_signs = self.extract_vital_signs(text)

        # 提取手术信息
        record.surgery_type, record.surgery_name = self.extract_surgery_info(text)

        # 从多章节提取手术关键词（用于软匹配）
        record.surgery_keywords = self._extract_surgery_keywords_from_text(text, record.surgery_name)

        return record

    def extract_patient_info(self, text: str) -> PatientInfo:
        """提取患者基本信息"""
        patient = PatientInfo()

        # 解析姓名、性别、年龄
        match = self.patient_pattern.search(text)
        if match:
            patient.name = match.group(1)
            patient.gender = match.group(2)
            patient.age = int(match.group(3))

        # 解析身高体重
        match = self.height_weight_pattern.search(text)
        if match:
            patient.height = float(match.group(1))
            patient.weight = float(match.group(2))

        return patient

    def extract_diagnoses(self, text: str) -> List[Diagnosis]:
        """提取诊断信息"""
        diagnoses = []
        text_lower = text  # 中文不做lowercase

        for keyword in self.diagnosis_keywords:
            if keyword in text_lower:
                diagnoses.append(Diagnosis(name=keyword))

        # 从查房记录中提取更详细诊断
        record_section = self._extract_section(text, "查房记录")
        if record_section:
            # 提取括号内诊断
            diag_matches = re.findall(r"[\d\.]+、([^\n\d]+)", record_section)
            for diag in diag_matches[:5]:  # 最多取5个
                diag = diag.strip()
                if len(diag) > 2 and diag not in [d.name for d in diagnoses]:
                    diagnoses.append(Diagnosis(name=diag))

        return diagnoses

    def extract_lab_results(self, text: str) -> Dict[str, List[LabResult]]:
        """提取检验结果"""
        results = {}

        # 按检验类型分组
        sections = {
            "血常规": re.findall(r"([^*\s]+计数)\s+(\d+(?:\.\d+)?)\s*([^正常高低]*10\^?9/L[^正常高低]*)\s*([正常高低]+)", text),
            "肝功能": re.findall(r"([^\s]+氨基转移酶|[^\s]+胆红素|[^\s]+蛋白)\s+(\d+(?:\.\d+)?)\s*([^\s]+)\s*([正常高低]+)", text),
            "肾功能": re.findall(r"([^\s]+肌酐|[^\s]+尿素|[^\s]+尿酸)\s+(\d+(?:\.\d+)?)\s*([^\s]+)\s*([正常高低]+)", text),
            "血气分析": re.findall(r"(pH|pCO2|氧分压|碳酸氢根)\s+(\d+(?:\.\d+)?)\s*([^\s]+)\s*([正常高低]+)", text),
        }

        for category, matches in sections.items():
            results[category] = [
                LabResult(name=name, value=float(val), unit=unit, status=status)
                for name, val, unit, status in matches
            ]

        return results

    def extract_medications(self, text: str) -> List[Medication]:
        """提取用药信息"""
        medications = []
        seen_drugs = set()  # 避免重复

        # 提取医嘱部分
        yizhu_section = self._extract_section(text, "医嘱")
        if not yizhu_section:
            return medications

        # 常见药物关键词（扩大列表）
        drug_keywords = [
            "瑞芬太尼", "丙泊酚", "呋塞米", "多巴酚丁胺", "肾上腺素",
            "青霉素", "哌拉西林", "头孢", "去乙酰毛花苷",
            "艾司奥美拉唑", "奥美拉唑", "利多卡因", "肝素", "葡萄糖酸钙",
            "氯化钠", "葡萄糖", "林格", "甘油", "开塞露", "吲哚美辛",
            "硝酸甘油", "氨溴索", "氨茶碱", "阿司匹林", "氯吡格雷",
            "甲泼尼龙", "地塞米松", "头孢曲松", "头孢他啶", "美罗培南",
            "万古霉素", "替考拉宁", "奥司他韦", "阿奇霉素", "左氧氟沙星"
        ]

        for keyword in drug_keywords:
            if keyword in yizhu_section and keyword not in seen_drugs:
                seen_drugs.add(keyword)

                # 提取剂量 - 查找关键词后面的数字+单位
                dosage = ""
                route = ""

                # 查找剂量模式
                dosage_match = re.search(
                    rf"{keyword}[^0-9]*?(\d+(?:\.\d+)?)\s*(?:g|mg|ml|万|iu|ug)?",
                    yizhu_section
                )
                if dosage_match:
                    dosage = dosage_match.group(1)

                # 提取给药途径
                # 在关键词附近查找途径
                idx = yizhu_section.find(keyword)
                nearby_text = yizhu_section[idx:idx+100] if idx >= 0 else yizhu_section[:100]

                if "泵入" in nearby_text:
                    route = "泵入"
                elif "静滴" in nearby_text:
                    route = "静滴"
                elif "静推" in nearby_text:
                    route = "静推"
                elif "口服" in nearby_text:
                    route = "口服"
                elif "皮试" in nearby_text:
                    route = "皮试"
                elif "肛入" in nearby_text:
                    route = "肛入"
                elif "外用" in nearby_text:
                    route = "外用"

                medications.append(Medication(
                    name=keyword,
                    dosage=dosage,
                    route=route
                ))

        return medications

    def extract_vital_signs(self, text: str) -> Dict[str, List]:
        """提取生命体征"""
        vitals = {}

        # 从监护记录提取
        monitoring_section = self._extract_section(text, "监测")
        if not monitoring_section:
            return vitals

        # 提取体温
        temp_matches = re.findall(r"体温\s+[^\d]*(\d+(?:\.\d+)?)", monitoring_section)
        if temp_matches:
            vitals["体温"] = [float(t) for t in temp_matches]

        # 提取心率
        hr_matches = re.findall(r"心率\s+[^\d]*(\d+)", monitoring_section)
        if hr_matches:
            vitals["心率"] = [int(h) for h in hr_matches]

        # 提取血压
        bp_matches = re.findall(r"有创收缩压\s+[^\d]*(\d+)", monitoring_section)
        if bp_matches:
            vitals["收缩压"] = [int(b) for b in bp_matches]

        return vitals

    def extract_surgery_info(self, text: str) -> Tuple[str, str]:
        """提取手术名称和专科类型（支持中英文双语章节名）"""
        surgery_name = ""

        # 1. 从手术记录中提取手术名称（尝试中英文章节名）
        surgery_text = (self._extract_section(text, "手术记录")
                        or self._extract_section(text, "surgery_record"))
        if surgery_text:
            name_match = re.search(r"手术名称[：:]?\s*([^\n;]+)", surgery_text)
            if name_match:
                surgery_name = name_match.group(1).strip()
            # 如果没找到"手术名称"字段，尝试其他模式
            if not surgery_name:
                # 尝试 "手术:" 模式
                name_match = re.search(r"手术[：:]\s*([^\n;]+)", surgery_text)
                if name_match:
                    surgery_name = name_match.group(1).strip()

        # 2. 如果手术记录没有，从 operation_record 中查找
        if not surgery_name:
            op_text = self._extract_section(text, "operation_record")
            if op_text:
                # 匹配 "手术名称1:xxx" 或 "手术:xxx"
                name_match = re.search(r"手术名称[\d]*[：:]?\s*([^\n;]+)", op_text)
                if name_match:
                    surgery_name = name_match.group(1).strip()

        # 3. 分类手术类型
        if not surgery_name:
            return "", ""

        for surgery_type, keywords in self.surgery_type_rules:
            if any(kw in surgery_name for kw in keywords):
                return surgery_type, surgery_name

        return "other", surgery_name

    def _extract_surgery_keywords_from_text(self, text: str, surgery_name: str = "") -> Set[str]:
        """从多章节提取手术/专科关键词，用于软匹配"""
        keywords_found: Set[str] = set()

        # 搜索可能包含手术信息的章节（中英文双语）
        sections_text = ""
        section_names = [
            "chief_complaint", "inspection_visit", "history_illness",
            "surgery_record", "手术记录",
        ]
        for name in section_names:
            section = self._extract_section(text, name)
            if section:
                sections_text += section + " "

        if sections_text.strip():
            for _, keywords in self.surgery_type_rules:
                for kw in keywords:
                    if kw in sections_text:
                        keywords_found.add(kw)

        # 也从已识别的手术名称中提取关键词
        if surgery_name:
            for _, keywords in self.surgery_type_rules:
                for kw in keywords:
                    if kw in surgery_name:
                        keywords_found.add(kw)

        return keywords_found

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """提取病历的某个章节"""
        # 支持中文冒号和英文冒号
        pattern = rf"###{section_name}[：:]?(.*?)(?=###[^\n]|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None


if __name__ == "__main__":
    # 测试解析
    with open("病历.txt", "r", encoding="utf-8") as f:
        text = f.read()

    parser = MedicalRecordParser()
    record = parser.parse(text)

    print("=== 解析结果 ===")
    print(f"患者: {record.patient.name}, {record.patient.gender}, {record.patient.age}岁")
    print(f"日期: {len(record.dates)} 条")
    print(f"诊断: {[d.name for d in record.diagnoses]}")
    print(f"检验类别: {list(record.lab_results.keys())}")
    print(f"药物: {[m.name for m in record.medications]}")
    print(f"生命体征: {list(record.vital_signs.keys())}")
