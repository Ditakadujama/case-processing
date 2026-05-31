"""
MySQL 病例卡持久化 — record_case_cards 表的 CRUD 操作。
存储 LLM 抽取的病例卡 JSON + embedding 向量。
"""

import json
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pymysql
from pymysql.cursors import DictCursor

from config import DBConfig

logger = logging.getLogger(__name__)

# 当前病例卡抽取器版本号，prompt/schema 变化时递增
DEFAULT_EXTRACTOR_VERSION = "v1.1"

# 建表 SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS record_case_cards (
    record_id VARCHAR(128) PRIMARY KEY,
    case_card_json JSON NOT NULL,
    summary_for_embedding TEXT,
    embedding BLOB,
    extractor_version VARCHAR(64) DEFAULT 'v1.0',
    embedding_model VARCHAR(128) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_extractor_version (extractor_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class MySQLCaseCardStore:
    """病例卡 + embedding 的 MySQL 持久化存储"""

    def __init__(self, db_config: DBConfig):
        self._db_config = db_config
        self._conn_kwargs = db_config.to_connection_kwargs()

    def _get_conn(self) -> pymysql.Connection:
        """创建新的数据库连接（P2-9: 统一使用 DictCursor）"""
        kwargs = self._conn_kwargs.copy()
        kwargs.setdefault("cursorclass", DictCursor)
        return pymysql.connect(**kwargs)

    # ── 表管理 ──────────────────────────────────────────────

    def init_table(self) -> None:
        """如果表不存在则创建"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
            conn.commit()
            logger.info("record_case_cards 表已就绪")
        finally:
            conn.close()

    def drop_table(self) -> None:
        """删除表（谨慎使用）"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS record_case_cards")
            conn.commit()
        finally:
            conn.close()

    # ── 写入 ────────────────────────────────────────────────

    def insert(self,
               record_id: str,
               card: dict,
               embedding: Optional[np.ndarray],
               extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
               embedding_model: str = "") -> None:
        """
        插入或更新一条病例卡记录。

        Args:
            record_id: 病例唯一 ID
            card: 病例卡 dict
            embedding: float32 numpy array 或 None
            extractor_version: 抽取器版本号
            embedding_model: embedding 模型名
        """
        card_json = json.dumps(card, ensure_ascii=False)
        summary = card.get("summary_for_embedding", "")
        embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO record_case_cards
                       (record_id, case_card_json, summary_for_embedding, embedding,
                        extractor_version, embedding_model)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           case_card_json = VALUES(case_card_json),
                           summary_for_embedding = VALUES(summary_for_embedding),
                           embedding = VALUES(embedding),
                           extractor_version = VALUES(extractor_version),
                           embedding_model = VALUES(embedding_model)""",
                    (record_id, card_json, summary, embedding_blob,
                     extractor_version, embedding_model)
                )
            conn.commit()
            logger.debug(f"病例卡已写入: {record_id}")
        finally:
            conn.close()

    def insert_batch(self,
                     records: List[Tuple[str, dict, Optional[np.ndarray], str, str]]) -> None:
        """
        批量插入病例卡。

        Args:
            records: [(record_id, card_dict, embedding_or_None, extractor_version, embedding_model), ...]
        """
        if not records:
            return

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for record_id, card, embedding, extractor_version, embedding_model in records:
                    card_json = json.dumps(card, ensure_ascii=False)
                    summary = card.get("summary_for_embedding", "")
                    embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

                    cursor.execute(
                        """INSERT INTO record_case_cards
                           (record_id, case_card_json, summary_for_embedding, embedding,
                            extractor_version, embedding_model)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE
                               case_card_json = VALUES(case_card_json),
                               summary_for_embedding = VALUES(summary_for_embedding),
                               embedding = VALUES(embedding),
                               extractor_version = VALUES(extractor_version),
                               embedding_model = VALUES(embedding_model)""",
                        (record_id, card_json, summary, embedding_blob,
                         extractor_version, embedding_model)
                    )
            conn.commit()
            logger.info(f"批量写入 {len(records)} 条病例卡")
        finally:
            conn.close()

    # ── 读取 ────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[dict]:
        """获取单条病例卡 dict"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT case_card_json FROM record_case_cards WHERE record_id = %s",
                    (record_id,)
                )
                row = cursor.fetchone()
                if row:
                    card_json = row["case_card_json"]
                    return json.loads(card_json) if isinstance(card_json, str) else card_json
                return None
        finally:
            conn.close()

    def get_embedding(self, record_id: str) -> Optional[np.ndarray]:
        """获取单条 embedding 向量"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT embedding FROM record_case_cards WHERE record_id = %s",
                    (record_id,)
                )
                row = cursor.fetchone()
                if row and row["embedding"]:
                    return np.frombuffer(row["embedding"], dtype=np.float32)
                return None
        finally:
            conn.close()

    def load_all(self) -> Dict[str, dict]:
        """
        加载所有病例卡（不含 embedding 向量）。

        Returns:
            {record_id: case_card_dict, ...}
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT record_id, case_card_json FROM record_case_cards"
                )
                result = {}
                for row in cursor.fetchall():
                    record_id = row["record_id"]
                    card_json = row["case_card_json"]
                    card = json.loads(card_json) if isinstance(card_json, str) else card_json
                    result[record_id] = card
                return result
        finally:
            conn.close()

    def load_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """
        加载所有预计算的 embedding 向量。

        Returns:
            (N×D float32 matrix, [record_id, ...])
            如果表为空或没有 embedding，返回 (empty array, [])
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT record_id, embedding FROM record_case_cards
                       WHERE embedding IS NOT NULL"""
                )
                rows = cursor.fetchall()
                if not rows:
                    return np.array([], dtype=np.float32), []

                ids = []
                vectors = []
                for row in rows:
                    ids.append(row["record_id"])
                    vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))

                return np.array(vectors, dtype=np.float32), ids
        finally:
            conn.close()

    # ── 查询 ────────────────────────────────────────────────

    def exists(self, record_id: str, extractor_version: str = DEFAULT_EXTRACTOR_VERSION) -> bool:
        """
        检查指定版本的病例卡是否已存在。

        Args:
            record_id: 病例 ID
            extractor_version: 抽取器版本
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT 1 FROM record_case_cards
                       WHERE record_id = %s AND extractor_version = %s""",
                    (record_id, extractor_version)
                )
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def has_embedding(self, record_id: str) -> bool:
        """检查某条记录是否有 embedding"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT embedding FROM record_case_cards WHERE record_id = %s",
                    (record_id,)
                )
                row = cursor.fetchone()
                return row is not None and row["embedding"] is not None
        finally:
            conn.close()

    def count(self) -> int:
        """返回记录总数"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM record_case_cards")
                row = cursor.fetchone()
                return list(row.values())[0] if row else 0
        finally:
            conn.close()

    def count_with_embedding(self) -> int:
        """返回有 embedding 的记录数"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM record_case_cards WHERE embedding IS NOT NULL")
                row = cursor.fetchone()
                return list(row.values())[0] if row else 0
        finally:
            conn.close()

    # ── 删除 ────────────────────────────────────────────────

    def delete(self, record_id: str) -> None:
        """删除单条记录"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM record_case_cards WHERE record_id = %s", (record_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_all(self) -> None:
        """清空表"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE record_case_cards")
            conn.commit()
        finally:
            conn.close()
