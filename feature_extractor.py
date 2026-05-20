"""
特征提取器
将结构化病历转为特征向量
"""

import numpy as np
from typing import List

from record_parser import MedicalRecord


class FeatureExtractor:
    """病历特征提取器"""

    def __init__(self):
        # 固定诊断词表（用于词袋向量）
        self.diagnosis_terms = [
            '颅脑损伤', '脑梗死', '肺部感染', '呼吸衰竭', '高血压',
            '心房颤动', '传导阻滞', '胸腔积液', '肺不张', '肺炎',
            '心肌损伤', '肾功能', '肝功能', '感染', '休克',
            '出血', '贫血', '血栓', '电解质紊乱', '酸碱平衡',
            '脑出血', '脑水肿', '癫痫', '昏迷', '脓毒症',
            '心衰', '心梗', '动脉粥样硬化', '肺栓塞', 'COPD',
            '糖尿病', '甲状腺', '消化道出血', '胰腺炎', '肠梗阻',
            '败血症', '多器官功能障碍', '应激性溃疡', '深静脉血栓'
        ]

        # 常用检验指标列表（用于固定向量维度）
        self.lab_indicators = [
            '白细胞', '中性粒细胞', '淋巴细胞', '血小板', '血红蛋白',
            '超敏C反应蛋白', '肌酐', '尿素', '尿酸', '葡萄糖',
            '钾', '钠', '氯', '钙', '镁',
            '总胆红素', '间接胆红素', '直接胆红素',
            '天冬氨酸氨基转移酶', '丙氨酸氨基转移酶',
            '碱性磷酸酶', '乳酸脱氢酶', 'γ-谷氨酰转肽酶',
            '白蛋白', '球蛋白', '总蛋白',
            'pH', 'pCO2', '氧分压', '碳酸氢根',
            '凝血酶原时间', 'D-二聚体', '纤维蛋白原'
        ]

        # 常用药物列表
        self.medication_list = [
            '瑞芬太尼', '丙泊酚', '哌拉西林', '头孢', '青霉素',
            '呋塞米', '多巴酚丁胺', '肾上腺素', '去乙酰毛花苷',
            '艾司奥美拉唑', '奥美拉唑', '利多卡因', '肝素',
            '葡萄糖酸钙', '开塞露', '阿司匹林', '氯吡格雷',
            '瑞芬太尼', '丙泊酚', '氯化钠', '葡萄糖', '林格',
            '万古霉素', '美罗培南', '头孢曲松', '左氧氟沙星'
        ]

        # 诊断向量维度
        self.diag_dim = len(self.diagnosis_terms)
        # 检验向量维度
        self.lab_dim = len(self.lab_indicators)
        # 药物向量维度
        self.med_dim = len(self.medication_list)
        # 人口学向量维度
        self.demo_dim = 4

    @property
    def total_dim(self) -> int:
        """总特征维度"""
        return self.diag_dim + self.lab_dim + self.med_dim + self.demo_dim

    def fit(self, records: List[MedicalRecord]):
        """训练向量化器（预留接口）"""
        # 词袋模型不需要训练，直接用固定词表
        pass

    def extract(self, record: MedicalRecord) -> np.ndarray:
        """
        提取单个病历的特征向量

        Returns:
            合并后的特征向量
        """
        # 1. 诊断向量 (词袋)
        diag_vec = self._extract_diagnosis_vector(record)

        # 2. 检验指标向量 (标准化数值)
        lab_vec = self._extract_lab_vector(record)

        # 3. 药物向量 (词袋)
        med_vec = self._extract_medication_vector(record)

        # 4. 人口学向量
        demo_vec = self._extract_demographic_vector(record)

        # 5. 合并所有向量
        combined = np.concatenate([diag_vec, lab_vec, med_vec, demo_vec])

        return combined

    def _extract_diagnosis_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取诊断向量 - 词袋模型"""
        vec = np.zeros(self.diag_dim)

        for diag in record.diagnoses:
            for i, term in enumerate(self.diagnosis_terms):
                if term in diag.name:
                    vec[i] = 1.0
                    break

        return vec

    def _extract_lab_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取检验指标向量"""
        vec = np.zeros(self.lab_dim)

        for results in record.lab_results.values():
            for result in results:
                # 匹配指标名称
                for i, indicator in enumerate(self.lab_indicators):
                    if indicator in result.name:
                        # 归一化
                        vec[i] = result.value / 100.0
                        break

        return vec

    def _extract_medication_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取药物向量"""
        vec = np.zeros(self.med_dim)

        for med in record.medications:
            for i, drug in enumerate(self.medication_list):
                if drug in med.name:
                    vec[i] = 1.0
                    break

        return vec

    def _extract_demographic_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取人口学特征向量"""
        patient = record.patient

        return np.array([
            1 if patient.gender == '男' else 0,
            patient.age / 100.0,  # 归一化年龄
            patient.weight / 150.0 if patient.weight else 0,  # 归一化体重
            patient.height / 200.0 if patient.height else 0,  # 归一化身高
        ])

    def extract_batch(self, records: List[MedicalRecord]) -> np.ndarray:
        """批量提取特征向量"""
        if not records:
            return np.array([])

        vectors = []
        for record in records:
            vectors.append(self.extract(record))

        return np.array(vectors)


if __name__ == "__main__":
    # 测试
    from record_parser import MedicalRecordParser

    with open("病历.txt", "r", encoding="utf-8") as f:
        text = f.read()

    parser = MedicalRecordParser()
    record = parser.parse(text)

    extractor = FeatureExtractor()
    extractor.fit([record])

    vector = extractor.extract(record)

    print(f"特征向量维度: {vector.shape}")
    print(f"非零元素数: {np.count_nonzero(vector)}")
