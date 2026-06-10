"""MySQL 天级病历存储层。"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymysql
from pymysql.cursors import DictCursor

from config import DBConfig
from data_migrate.database import get_db_connection

logger = logging.getLogger(__name__)


CREATE_RECORD_DAYS_SQL = """
CREATE TABLE IF NOT EXISTS record_days (
    day_record_id VARCHAR(160) PRIMARY KEY COMMENT '患者-天唯一ID, 如 ZY001#D001',
    patient_id VARCHAR(128) NOT NULL,
    day_index INT NOT NULL,
    visit_date DATE NULL,
    day_text MEDIUMTEXT COMMENT '过滤后的医生关注文本',
    cumulative_text MEDIUMTEXT COMMENT '截至当天的过滤后文本/状态输入',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_patient_day (patient_id, day_index),
    INDEX idx_patient_id (patient_id),
    INDEX idx_day_index (day_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='天级医生关注病历文本表';
"""


CREATE_DAY_CASE_CARDS_SQL = """
CREATE TABLE IF NOT EXISTS record_day_case_cards (
    day_record_id VARCHAR(160) PRIMARY KEY,
    patient_id VARCHAR(128) NOT NULL,
    day_index INT NOT NULL,
    case_card_json JSON NOT NULL,
    day_summary_for_embedding TEXT,
    cumulative_summary_for_embedding TEXT,
    day_delta_embedding BLOB,
    cumulative_embedding BLOB,
    extractor_version VARCHAR(64) DEFAULT 'day-v2.1-diagnosis-axis',
    embedding_model VARCHAR(128) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_patient_id (patient_id),
    INDEX idx_day_index (day_index),
    INDEX idx_extractor_version (extractor_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='天级病例卡与语义向量表';
"""


DEFAULT_DAY_EXTRACTOR_VERSION = "day-v2.1-diagnosis-axis"


class MySQLDayStore:
    """天级病历、特征、病例卡和 embedding 的持久化存储。"""

    def __init__(self, db_config: DBConfig):
        self._cfg = db_config
        self._conn_kwargs = db_config.to_connection_kwargs()

    def _get_conn(self) -> pymysql.Connection:
        kwargs = self._conn_kwargs.copy()
        kwargs.setdefault("cursorclass", DictCursor)
        return pymysql.connect(**kwargs)

    def init_tables(self) -> None:
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_RECORD_DAYS_SQL)
                cursor.execute(CREATE_DAY_CASE_CARDS_SQL)
            conn.commit()
        logger.info("天级表 record_days / record_day_case_cards 已就绪")

    def delete_all(self) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE record_day_case_cards")
                cursor.execute("TRUNCATE TABLE record_days")
            conn.commit()
        finally:
            conn.close()

    def insert_day_rows(self, rows: List[Tuple]) -> None:
        """批量写入 record_days。"""
        if not rows:
            return

        sql = """INSERT INTO record_days
                 (day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text)
                 VALUES (%s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  visit_date = VALUES(visit_date),
                  day_text = VALUES(day_text),
                  cumulative_text = VALUES(cumulative_text)"""

        serialized = []
        for (day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text) in rows:
            serialized.append((
                day_record_id,
                patient_id,
                day_index,
                visit_date,
                day_text,
                cumulative_text,
            ))

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(sql, serialized)
            conn.commit()
        finally:
            conn.close()

    def insert_day_card(self,
                        day_record_id: str,
                        patient_id: str,
                        day_index: int,
                        card: dict,
                        day_delta_embedding: Optional[np.ndarray],
                        cumulative_embedding: Optional[np.ndarray],
                        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION,
                        embedding_model: str = "") -> None:
        card_json = json.dumps(card, ensure_ascii=False)
        day_summary = card.get("day_summary_for_embedding") or card.get("summary_for_embedding", "")
        cumulative_summary = card.get("cumulative_summary_for_embedding") or card.get("summary_for_embedding", "")
        day_blob = day_delta_embedding.astype(np.float32).tobytes() if day_delta_embedding is not None else None
        cumulative_blob = cumulative_embedding.astype(np.float32).tobytes() if cumulative_embedding is not None else None

        sql = """INSERT INTO record_day_case_cards
                 (day_record_id, patient_id, day_index, case_card_json,
                  day_summary_for_embedding, cumulative_summary_for_embedding,
                  day_delta_embedding, cumulative_embedding, extractor_version, embedding_model)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  case_card_json = VALUES(case_card_json),
                  day_summary_for_embedding = VALUES(day_summary_for_embedding),
                  cumulative_summary_for_embedding = VALUES(cumulative_summary_for_embedding),
                  day_delta_embedding = VALUES(day_delta_embedding),
                  cumulative_embedding = VALUES(cumulative_embedding),
                  extractor_version = VALUES(extractor_version),
                  embedding_model = VALUES(embedding_model)"""

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    day_record_id, patient_id, day_index, card_json,
                    day_summary, cumulative_summary, day_blob, cumulative_blob,
                    extractor_version, embedding_model,
                ))
            conn.commit()
        finally:
            conn.close()

    def day_card_exists(self, day_record_id: str,
                        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION) -> bool:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT 1 FROM record_day_case_cards
                       WHERE day_record_id = %s AND extractor_version = %s""",
                    (day_record_id, extractor_version),
                )
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def day_exists(self, day_record_id: str) -> bool:
        """检查某天级硬特征是否已存在。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM record_days WHERE day_record_id = %s LIMIT 1",
                    (day_record_id,),
                )
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def patient_has_days(self, patient_id: str) -> bool:
        """检查某患者是否已有天级硬特征。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM record_days WHERE patient_id = %s LIMIT 1",
                    (patient_id,),
                )
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def patient_has_day_cards(
        self,
        patient_id: str,
        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION,
    ) -> bool:
        """检查某患者是否已有当前版本天级病例卡。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT 1 FROM record_day_case_cards
                       WHERE patient_id = %s AND extractor_version = %s LIMIT 1""",
                    (patient_id, extractor_version),
                )
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def load_days_by_patient(self) -> Dict[str, List[dict]]:
        """加载所有天级硬特征，按 patient_id 分组。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text
                       FROM record_days
                       ORDER BY patient_id, day_index"""
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        result: Dict[str, List[dict]] = {}
        for row in rows:
            item = {
                "day_record_id": row["day_record_id"],
                "patient_id": row["patient_id"],
                "day_index": row["day_index"],
                "visit_date": str(row["visit_date"]) if row.get("visit_date") else "",
                "day_text": row.get("day_text") or "",
                "cumulative_text": row.get("cumulative_text") or "",
            }
            result.setdefault(row["patient_id"], []).append(item)
        return result

    def load_day_cards_by_patient(
        self,
        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION,
    ) -> Dict[str, List[dict]]:
        """加载每日病例卡和 embedding，按 patient_id 分组。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT day_record_id, patient_id, day_index, case_card_json,
                              day_delta_embedding, cumulative_embedding
                       FROM record_day_case_cards
                       WHERE extractor_version = %s
                       ORDER BY patient_id, day_index""",
                    (extractor_version,),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        result: Dict[str, List[dict]] = {}
        for row in rows:
            card_json = row["case_card_json"]
            card = json.loads(card_json) if isinstance(card_json, str) else card_json
            day_blob = row.get("day_delta_embedding")
            cumulative_blob = row.get("cumulative_embedding")
            item = {
                "day_record_id": row["day_record_id"],
                "patient_id": row["patient_id"],
                "day_index": row["day_index"],
                "case_card": card,
                "day_delta_embedding": np.frombuffer(day_blob, dtype=np.float32) if day_blob else None,
                "cumulative_embedding": np.frombuffer(cumulative_blob, dtype=np.float32) if cumulative_blob else None,
            }
            result.setdefault(row["patient_id"], []).append(item)
        return result

    def count_days(self) -> int:
        return self._count("record_days")

    def count_day_cards(self) -> int:
        return self._count("record_day_case_cards")

    def count_day_embeddings(self) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT COUNT(*) AS cnt FROM record_day_case_cards
                       WHERE day_delta_embedding IS NOT NULL OR cumulative_embedding IS NOT NULL"""
                )
                row = cursor.fetchone()
                return row["cnt"] if row else 0
        finally:
            conn.close()

    def _count(self, table_name: str) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
                row = cursor.fetchone()
                return row["cnt"] if row else 0
        finally:
            conn.close()
