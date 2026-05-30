"""
MySQL 向量存储层
持久化特征向量、原始文本和时间轴特征到 MySQL
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymysql
from pymysql.cursors import DictCursor

from config import DBConfig
from data_migrate.database import get_db_connection

logger = logging.getLogger(__name__)

CREATE_VECTORS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS record_vectors (
    record_id VARCHAR(128) PRIMARY KEY COMMENT '病例唯一ID (patient_id)',
    feature_vector BLOB NOT NULL COMMENT '特征向量 (105-dim float64, numpy .tobytes())',
    raw_text MEDIUMTEXT COMMENT '原始病历文本',
    timeline_features JSON COMMENT '时间轴特征',
    record_order_index INT COMMENT '插入顺序',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_record_order (record_order_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='病历特征向量与元数据表';
"""


def _serialize_timeline_features(tf) -> dict:
    """将 TimelineFeatures 对象序列化为 JSON 兼容的 dict"""
    if tf is None:
        return None
    return {
        'visit_count': tf.visit_count,
        'total_span_days': tf.total_span_days,
        'visit_gaps': tf.visit_gaps,
        'diagnosis_keywords': list(tf.diagnosis_keywords) if tf.diagnosis_keywords else [],
        'event_type_sequence': tf.event_type_sequence,
        'surgery_type': tf.surgery_type,
        'surgery_keywords': list(tf.surgery_keywords) if tf.surgery_keywords else [],
        'intervention_keywords': list(getattr(tf, 'intervention_keywords', [])) if getattr(tf, 'intervention_keywords', None) else [],
        'complication_keywords': list(getattr(tf, 'complication_keywords', [])) if getattr(tf, 'complication_keywords', None) else [],
        'severity_keywords': list(getattr(tf, 'severity_keywords', [])) if getattr(tf, 'severity_keywords', None) else [],
    }


def _deserialize_timeline_features(data: dict):
    """将 JSON dict 反序列化为 TimelineFeatures 对象"""
    from timeline_similarity import TimelineFeatures
    if data is None:
        return None
    return TimelineFeatures(
        visit_count=data.get('visit_count', 1),
        total_span_days=data.get('total_span_days', 0.0),
        visit_gaps=data.get('visit_gaps', []),
        diagnosis_keywords=set(data.get('diagnosis_keywords', [])),
        event_type_sequence=data.get('event_type_sequence', []),
        surgery_type=data.get('surgery_type'),
        surgery_keywords=set(data.get('surgery_keywords', [])),
        intervention_keywords=set(data.get('intervention_keywords', [])),
        complication_keywords=set(data.get('complication_keywords', [])),
        severity_keywords=set(data.get('severity_keywords', [])),
    )


class MySQLVectorStore:
    """MySQL 向量存储层"""

    def __init__(self, db_config: DBConfig):
        self._cfg = db_config
        self._conn_kwargs = db_config.to_connection_kwargs()

    def _get_conn(self):
        return pymysql.connect(**self._conn_kwargs)

    def init_table(self) -> None:
        """创建 record_vectors 表（如不存在）"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_VECTORS_TABLE_SQL)
            conn.commit()
        logger.info("表 record_vectors 已就绪")

    _INSERT_SQL = """INSERT INTO record_vectors (record_id, feature_vector, raw_text, timeline_features, record_order_index)
             VALUES (%s, %s, %s, %s, %s)
             ON DUPLICATE KEY UPDATE
                feature_vector = VALUES(feature_vector),
                raw_text = VALUES(raw_text),
                timeline_features = VALUES(timeline_features),
                record_order_index = VALUES(record_order_index)"""

    def insert(self, record_id: str, vector: np.ndarray, text: str,
               timeline_features, order_idx: int) -> None:
        """插入单条记录"""
        tl_json = json.dumps(_serialize_timeline_features(timeline_features), ensure_ascii=False) \
            if timeline_features is not None else None
        vector_blob = vector.astype(np.float64).tobytes()

        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._INSERT_SQL, (record_id, vector_blob, text, tl_json, order_idx))
            conn.commit()

    def insert_sequential(self, records: List[Tuple[str, np.ndarray, str, object, int]]) -> None:
        """逐条插入，复用同一连接（每条即时 commit，避免重复 TCP 握手开销）"""
        if not records:
            return

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for rid, vec, txt, tl_feat, idx in records:
                    tl_json = json.dumps(_serialize_timeline_features(tl_feat), ensure_ascii=False) \
                        if tl_feat is not None else None
                    cursor.execute(self._INSERT_SQL, (rid, vec.astype(np.float64).tobytes(), txt, tl_json, idx))
                    conn.commit()
        finally:
            conn.close()

    def insert_batch(self, records: List[Tuple[str, np.ndarray, str, object, int]]) -> None:
        """批量插入记录 [(record_id, vector, text, timeline_features, order_idx), ...]"""
        if not records:
            return

        sql = """INSERT INTO record_vectors (record_id, feature_vector, raw_text, timeline_features, record_order_index)
                 VALUES (%s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                    feature_vector = VALUES(feature_vector),
                    raw_text = VALUES(raw_text),
                    timeline_features = VALUES(timeline_features),
                    record_order_index = VALUES(record_order_index)"""

        rows = []
        for rid, vec, txt, tl_feat, idx in records:
            tl_json = json.dumps(_serialize_timeline_features(tl_feat), ensure_ascii=False) \
                if tl_feat is not None else None
            rows.append((rid, vec.astype(np.float64).tobytes(), txt, tl_json, idx))

        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, rows)
            conn.commit()

    def load_all_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """加载全部特征向量和 record_id 列表，返回 (N×D 矩阵, [record_id, ...])"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT record_id, feature_vector FROM record_vectors ORDER BY record_order_index"
                )
                rows = cursor.fetchall()

        if not rows:
            return np.array([]), []

        vectors = []
        ids = []
        for row in rows:
            ids.append(row['record_id'])
            vec = np.frombuffer(row['feature_vector'], dtype=np.float64)
            vectors.append(vec)

        return np.array(vectors), ids

    def load_all_metadata(self) -> Dict[str, dict]:
        """加载全部 timeline_features 和文本元数据（启动时构建缓存）"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT record_id, raw_text, timeline_features FROM record_vectors ORDER BY record_order_index"
                )
                rows = cursor.fetchall()

        result = {}
        for row in rows:
            tl_data = row['timeline_features']
            if isinstance(tl_data, str):
                tl_data = json.loads(tl_data)
            result[row['record_id']] = {
                'text': row['raw_text'],
                'timeline_features': _deserialize_timeline_features(tl_data),
            }
        return result

    def get_text(self, record_id: str) -> Optional[str]:
        """按需读取文本"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT raw_text FROM record_vectors WHERE record_id = %s", (record_id,)
                )
                row = cursor.fetchone()
        return row['raw_text'] if row else None

    def delete(self, record_id: str) -> None:
        """删除记录"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM record_vectors WHERE record_id = %s", (record_id,))
            conn.commit()

    def delete_all(self) -> None:
        """清空表"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM record_vectors")
            conn.commit()

    def count(self) -> int:
        """记录总数"""
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as cnt FROM record_vectors")
                row = cursor.fetchone()
        return row['cnt'] if row else 0
