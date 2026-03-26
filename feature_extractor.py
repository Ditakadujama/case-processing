"""
特征提取器
将结构化病历转为特征向量
"""

import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

from record_parser import MedicalRecord, PatientInfo, LabResult, Diagnosis, Medication


class FeatureExtractor:
    """病历特征提取器"""

    def __init__(self):
        # TF-IDF 向量化器
        self.text_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'
        )

        # 数值标准化器
        self.scaler = StandardScaler()

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
            '葡萄糖酸钙', '开塞露', '阿司匹林', '氯吡格雷'
        ]

        self._is_fitted = False

    def fit(self, records: List[MedicalRecord]):
        """训练向量化器（在一批病历上fit）"""
        # 从文本提取诊断词袋
        diagnosis_texts = []
        for record in records:
            diag_text = ' '.join([d.name for d in record.diagnoses])
            diagnosis_texts.append(diag_text)

        # 训练TF-IDF
        if diagnosis_texts:
            self.text_vectorizer.fit(diagnosis_texts)

        self._is_fitted = True

    def extract(self, record: MedicalRecord) -> np.ndarray:
        """
        提取单个病历的特征向量

        Returns:
            合并后的特征向量
        """
        # 1. 诊断向量 (TF-IDF)
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
        """提取诊断向量"""
        diag_text = ' '.join([d.name for d in record.diagnoses])

        if not self._is_fitted or not diag_text:
            return np.zeros(100)

        return self.text_vectorizer.transform([diag_text]).toarray()[0]

    def _extract_lab_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取检验指标向量"""
        vec = np.zeros(len(self.lab_indicators))

        for category, results in record.lab_results.items():
            for result in results:
                # 匹配指标名称
                for i, indicator in enumerate(self.lab_indicators):
                    if indicator in result.name:
                        # 标准化处理（简化：使用原始值/参考范围）
                        # 这里简化处理，直接取value
                        vec[i] = result.value / 100.0  # 归一化
                        break

        return vec

    def _extract_medication_vector(self, record: MedicalRecord) -> np.ndarray:
        """提取药物向量"""
        vec = np.zeros(len(self.medication_list))

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


class FeatureVectorizer:
    """特征向量封装（兼容旧接口）"""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.dimension = 600  # 预估维度

    def fit_transform(self, records: List[MedicalRecord]) -> np.ndarray:
        """训练并转换"""
        self.extractor.fit(records)
        return self.extractor.extract_batch(records)

    def transform(self, records: List[MedicalRecord]) -> np.ndarray:
        """转换"""
        return self.extractor.extract_batch(records)


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
