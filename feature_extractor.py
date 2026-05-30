"""
特征提取器
将结构化病历转为特征向量
"""

import re
import numpy as np
from typing import List

from record_parser import MedicalRecord
from timeline_similarity import SURGERY_TYPE_RULES


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

        # 常用药物列表（去重）
        self.medication_list = [
            '瑞芬太尼', '丙泊酚', '哌拉西林', '头孢', '青霉素',
            '呋塞米', '多巴酚丁胺', '肾上腺素', '去乙酰毛花苷',
            '艾司奥美拉唑', '奥美拉唑', '利多卡因', '肝素',
            '葡萄糖酸钙', '开塞露', '阿司匹林', '氯吡格雷',
            '氯化钠', '葡萄糖', '林格',
            '万古霉素', '美罗培南', '头孢曲松', '左氧氟沙星'
        ]

        # 手术类型列表（5-dim soft count，基于已验证的手术关键词）
        self.surgery_types = ['neuro', 'cardiac', 'general', 'ortho', 'other']

        # 构建 keyword → specialty 映射（从 SURGERY_TYPE_RULES）
        self._keyword_to_specialty = {}
        for stype, keywords in SURGERY_TYPE_RULES:
            for kw in keywords:
                self._keyword_to_specialty[kw] = stype

        # 诊断向量维度
        self.diag_dim = len(self.diagnosis_terms)
        # 检验向量维度
        self.lab_dim = len(self.lab_indicators)
        # 药物向量维度
        self.med_dim = len(self.medication_list)
        # 人口学向量维度
        self.demo_dim = 4
        # 手术类型向量维度
        self.surgery_dim = len(self.surgery_types)

        # 特征组基础权重
        self.group_weights = {
            'diag': 1.5,
            'lab': 0.6,
            'med': 0.8,
            'demo': 0.5,
            'surgery': 1.5,
        }

        # IDF 权重（在 fit() 中计算）
        self.idf_weights = None

        # 预计算每个维度的组权重向量（用于 _apply_weights 快速应用）
        self._group_weight_vector = np.concatenate([
            np.full(self.diag_dim, self.group_weights['diag']),
            np.full(self.lab_dim, self.group_weights['lab']),
            np.full(self.med_dim, self.group_weights['med']),
            np.full(self.demo_dim, self.group_weights['demo']),
            np.full(self.surgery_dim, self.group_weights['surgery']),
        ])

    @property
    def total_dim(self) -> int:
        """总特征维度"""
        return self.diag_dim + self.lab_dim + self.med_dim + self.demo_dim + self.surgery_dim

    def fit(self, records: List[MedicalRecord]):
        """训练向量化器：计算 IDF 权重，返回 raw_vectors 列表供 extract_batch() 复用"""
        if not records:
            self.idf_weights = None
            return []

        raw_vectors = []

        for record in records:
            vec = self._extract_raw_vector(record)
            raw_vectors.append(vec)

        # 固定词表的结构化特征需要跨批次、跨查询保持同一尺度。
        # 旧实现按 build batch 计算 IDF，会导致不同批次入库向量不可比；
        # 这里保留 fit() 接口，但不再引入 batch 相关权重。
        self.idf_weights = None
        return raw_vectors

    def _extract_raw_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取原始特征向量（不带权重）"""
        diag_vec = self._extract_diagnosis_vector(record)
        lab_vec = self._extract_lab_vector(record)
        med_vec = self._extract_medication_vector(record)
        demo_vec = self._extract_demographic_vector(record)
        surgery_vec = self._extract_surgery_type_vector(record)
        return np.concatenate([diag_vec, lab_vec, med_vec, demo_vec, surgery_vec])

    def extract(self, record: MedicalRecord) -> np.ndarray:
        """
        提取单个病历的特征向量（应用组权重和IDF权重）

        Returns:
            合并后的特征向量
        """
        # 提取各组原始向量
        diag_vec = self._extract_diagnosis_vector(record) * self.group_weights['diag']
        lab_vec = self._extract_lab_vector(record) * self.group_weights['lab']
        med_vec = self._extract_medication_vector(record) * self.group_weights['med']
        demo_vec = self._extract_demographic_vector(record) * self.group_weights['demo']
        surgery_vec = self._extract_surgery_type_vector(record) * self.group_weights['surgery']

        # 合并所有向量
        combined = np.concatenate([diag_vec, lab_vec, med_vec, demo_vec, surgery_vec])

        # 应用 IDF 权重
        if self.idf_weights is not None:
            combined = combined * self.idf_weights

        return combined

    def _apply_weights(self, raw_vector: np.ndarray) -> np.ndarray:
        """对已提取的子向量应用组权重和IDF权重"""
        combined = raw_vector * self._group_weight_vector
        if self.idf_weights is not None:
            combined = combined * self.idf_weights
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

    def _extract_surgery_type_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取专科软信号向量（基于已验证的手术关键词计数）"""
        vec = np.zeros(self.surgery_dim)
        if not record.surgery_keywords:
            return vec
        # 统计每个专科命中的关键词数
        for kw in record.surgery_keywords:
            stype = self._keyword_to_specialty.get(kw)
            if stype and stype != 'other':
                idx = self.surgery_types.index(stype)
                vec[idx] += 1.0
        # 归一化：最多 3 个命中即饱和到 1.0
        vec = np.minimum(vec / 3.0, 1.0)
        return vec

    def extract_batch(self, records: List[MedicalRecord],
                      raw_vectors: list = None) -> np.ndarray:
        """批量提取特征向量。若提供 raw_vectors（来自 fit()），跳过子向量提取"""
        if not records:
            return np.array([])

        vectors = []
        for i, record in enumerate(records):
            if raw_vectors is not None and i < len(raw_vectors):
                vec = self._apply_weights(raw_vectors[i])
            else:
                vec = self.extract(record)
            vectors.append(vec)

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
