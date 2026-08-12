"""MySQL 天级病历存储层。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymysql
from pymysql.cursors import DictCursor

from config import DBConfig
from data_migrate.database import (
    chunked as _chunked,
    get_db_connection,
    get_pooled_connection,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProcessingSnapshot:
    """一条日记录的完整处理状态快照（用于跳过判断）。"""
    day_record_id: str
    source_hash: str = ""
    extraction_input_hash: str = ""
    extractor_version: str = ""
    embedding_model: str = ""
    embedding_input_hash: str = ""
    embedding_dimension: int = 0
    processing_status: str = "pending"
    failed_stage: str = ""
    has_valid_card: bool = False
    has_valid_embedding: bool = False


@dataclass
class ProcessingJob:
    """处理任务记录。"""
    day_record_id: str
    patient_id: str
    source_hash: str = ""
    extraction_input_hash: str = ""
    extractor_version: str = ""
    embedding_model: str = ""
    embedding_input_hash: str = ""
    embedding_dimension: int = 0
    processing_status: str = "pending"
    failed_stage: str = ""
    attempt_count: int = 0
    last_error: str = ""
    next_retry_at: Optional[str] = None


@dataclass
class DayCardWriteRow:
    """批量写入病例卡的行数据。"""
    day_record_id: str
    patient_id: str
    day_index: int
    case_card_json: str
    case_card_embedding: Optional[bytes] = None
    source_hash: str = ""
    extraction_input_hash: str = ""
    embedding_input_hash: str = ""
    extractor_version: str = ""
    embedding_model: str = ""
    embedding_dimension: Optional[int] = None
    processing_status: str = "indexed"
    failed_stage: str = ""
    last_error: str = ""


@dataclass
class BatchWriteResult:
    """批量写入结果。"""
    success_count: int = 0
    fail_count: int = 0
    failed_record_ids: list = None
    errors: dict = None

    def __post_init__(self):
        if self.failed_record_ids is None:
            self.failed_record_ids = []
        if self.errors is None:
            self.errors = {}


@dataclass
class ProcessingFailure:
    """处理失败记录。"""
    day_record_id: str
    patient_id: str
    failed_stage: str
    error: str
    attempt_count: int = 1
    source_hash: str = ""
    extraction_input_hash: str = ""
    extractor_version: str = ""
    embedding_model: str = ""


# 有效的处理状态值
VALID_PROCESSING_STATUSES = frozenset({
    "pending",
    "extracting",
    "embedding",
    "indexed",
    "failed_retryable",
    "failed_permanent",
})


def determine_required_action(
    day_source_hash: str,
    day_extraction_input_hash: str,
    snapshot: Optional[ProcessingSnapshot],
    extractor_version: str,
    embedding_model: str,
    embedding_input_hash: str = "",
    expected_dimension: Optional[int] = None,
) -> str:
    """判断一条日记录需要的处理动作。

    Args:
        day_source_hash: 当前日记录的 source_hash。
        day_extraction_input_hash: 当前日记录的 extraction_input_hash。
        snapshot: 已有的处理状态快照（None 表示未处理过）。
        extractor_version: 当前抽取器版本。
        embedding_model: 当前 embedding 模型。
        embedding_input_hash: 当前 embedding 输入哈希。
        expected_dimension: 期望的 embedding 维度。

    Returns:
        "skip" | "extract_and_embed" | "embed_only"
    """
    if snapshot is None:
        return "extract_and_embed"

    if not snapshot.has_valid_card:
        return "extract_and_embed"

    # 原始内容发生变化
    if snapshot.source_hash != day_source_hash:
        return "extract_and_embed"

    # LLM 抽取输入发生变化
    if snapshot.extraction_input_hash != day_extraction_input_hash:
        return "extract_and_embed"

    # 抽取器版本变化
    if snapshot.extractor_version != extractor_version:
        return "extract_and_embed"

    # 病例卡有效且输入未变——检查是否只需重新做 embedding
    if snapshot.embedding_model != embedding_model:
        return "embed_only"

    if snapshot.embedding_input_hash != embedding_input_hash and embedding_input_hash:
        return "embed_only"

    if expected_dimension is not None and snapshot.embedding_dimension != expected_dimension:
        return "embed_only"

    if not snapshot.has_valid_embedding:
        return "embed_only"

    if snapshot.processing_status != "indexed":
        # A current, valid card whose embedding step failed can resume without
        # paying for another LLM extraction.
        if snapshot.failed_stage == "embedding":
            return "embed_only"
        return "extract_and_embed"

    return "skip"


CREATE_RECORD_DAYS_SQL = """
CREATE TABLE IF NOT EXISTS record_days (
    day_record_id VARCHAR(160) PRIMARY KEY COMMENT '患者-天唯一ID, 如 medical_records:42',
    patient_id VARCHAR(128) NOT NULL,
    day_index INT NOT NULL,
    visit_date DATE NULL,
    day_text MEDIUMTEXT COMMENT '过滤后的医生关注文本',
    cumulative_text MEDIUMTEXT NULL COMMENT '截至当天的过滤后文本/状态输入（Stage 6 后不再写入）',
    source_table VARCHAR(32) NOT NULL DEFAULT 'medical_records',
    source_record_id VARCHAR(128) NULL,
    source_hash CHAR(64) NULL,
    extraction_input_hash CHAR(64) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_source_record (source_table, source_record_id),
    INDEX idx_patient_id (patient_id),
    INDEX idx_patient_day (patient_id, day_index),
    INDEX idx_day_index (day_index),
    INDEX idx_source_hash (source_hash)
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
    case_card_embedding BLOB,
    source_hash CHAR(64) NULL,
    extraction_input_hash CHAR(64) NULL,
    embedding_input_hash CHAR(64) NULL,
    embedding_dimension INT NULL,
    extractor_version VARCHAR(64) DEFAULT 'day-v3.0-case-card-embedding',
    embedding_model VARCHAR(128) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_patient_id (patient_id),
    INDEX idx_day_index (day_index),
    INDEX idx_extractor_version (extractor_version),
    INDEX idx_card_source_hash (source_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='天级病例卡与语义向量表';
"""


CREATE_PROCESSING_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS record_day_processing_jobs (
    day_record_id VARCHAR(160) PRIMARY KEY,
    patient_id VARCHAR(128) NOT NULL,
    source_hash CHAR(64) NULL,
    extraction_input_hash CHAR(64) NULL,
    extractor_version VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(128) NOT NULL,
    embedding_input_hash CHAR(64) NULL,
    embedding_dimension INT NULL,
    processing_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    failed_stage VARCHAR(32) NULL,
    attempt_count INT NOT NULL DEFAULT 0,
    last_error TEXT NULL,
    next_retry_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_job_status_retry (processing_status, next_retry_at),
    INDEX idx_job_patient (patient_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='处理任务状态表';
"""


DEFAULT_DAY_EXTRACTOR_VERSION = "day-v3.0-case-card-embedding"


class MySQLDayStore:
    """天级病历、特征、病例卡和 embedding 的持久化存储。"""

    def __init__(self, db_config: DBConfig):
        self._cfg = db_config
        self._conn_kwargs = db_config.to_connection_kwargs()

    def _get_conn(self) -> pymysql.Connection:
        return get_pooled_connection(self._cfg)

    def init_tables(self) -> None:
        with get_db_connection(self._cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_RECORD_DAYS_SQL)
                cursor.execute(CREATE_DAY_CASE_CARDS_SQL)
                cursor.execute(CREATE_PROCESSING_JOBS_SQL)
            conn.commit()
        logger.info("天级表 record_days / record_day_case_cards / record_day_processing_jobs 已就绪")

    def delete_all(self) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE record_day_case_cards")
                cursor.execute("TRUNCATE TABLE record_day_processing_jobs")
                cursor.execute("TRUNCATE TABLE record_days")
            conn.commit()
        finally:
            conn.close()

    def insert_day_rows(self, rows: List[Tuple]) -> None:
        """批量写入 record_days（含稳定 ID 和哈希字段）。"""
        if not rows:
            return

        sql = """INSERT INTO record_days
                 (day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text,
                  source_table, source_record_id, source_hash, extraction_input_hash)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  patient_id = VALUES(patient_id),
                  day_index = VALUES(day_index),
                  visit_date = VALUES(visit_date),
                  day_text = VALUES(day_text),
                  cumulative_text = VALUES(cumulative_text),
                  source_table = VALUES(source_table),
                  source_record_id = VALUES(source_record_id),
                  source_hash = VALUES(source_hash),
                  extraction_input_hash = VALUES(extraction_input_hash)"""

        serialized = []
        for row in rows:
            # Support both old (6-tuple) and new (10-tuple) row formats
            if len(row) == 6:
                (day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text) = row
                serialized.append((
                    day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text,
                    "medical_records", "", "", "",
                ))
            else:
                (day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text,
                 source_table, source_record_id, source_hash, extraction_input_hash) = row
                serialized.append((
                    day_record_id, patient_id, day_index, visit_date, day_text, cumulative_text,
                    source_table, source_record_id, source_hash, extraction_input_hash,
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
                        case_card_embedding: Optional[np.ndarray] = None,
                        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION,
                        embedding_model: str = "",
                        source_hash: str = "",
                        extraction_input_hash: str = "",
                        embedding_input_hash: str = "",
                        embedding_dimension: Optional[int] = None) -> None:
        card_json = json.dumps(card, ensure_ascii=False)
        case_card_blob = case_card_embedding.astype(np.float32).tobytes() if case_card_embedding is not None else None

        sql = """INSERT INTO record_day_case_cards
                 (day_record_id, patient_id, day_index, case_card_json,
                  case_card_embedding, extractor_version, embedding_model,
                  source_hash, extraction_input_hash, embedding_input_hash, embedding_dimension)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  patient_id = VALUES(patient_id),
                  day_index = VALUES(day_index),
                  case_card_json = VALUES(case_card_json),
                  case_card_embedding = VALUES(case_card_embedding),
                  extractor_version = VALUES(extractor_version),
                  embedding_model = VALUES(embedding_model),
                  source_hash = VALUES(source_hash),
                  extraction_input_hash = VALUES(extraction_input_hash),
                  embedding_input_hash = VALUES(embedding_input_hash),
                  embedding_dimension = VALUES(embedding_dimension)"""

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    day_record_id, patient_id, day_index, card_json,
                    case_card_blob,
                    extractor_version, embedding_model,
                    source_hash, extraction_input_hash, embedding_input_hash,
                    embedding_dimension,
                ))
            conn.commit()
        finally:
            conn.close()

    def upsert_day_cards(self, rows: List[DayCardWriteRow]) -> BatchWriteResult:
        """批量 upsert 病例卡和处理状态（同一事务）。

        一个批次使用一个连接和一个事务。
        成功后一次性提交，失败则回滚该批次并进行拆批重试。
        """
        if not rows:
            return BatchWriteResult()

        sql = """INSERT INTO record_day_case_cards
                 (day_record_id, patient_id, day_index, case_card_json,
                  case_card_embedding, source_hash, extraction_input_hash,
                  embedding_input_hash, extractor_version, embedding_model,
                  embedding_dimension)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  patient_id = VALUES(patient_id),
                  day_index = VALUES(day_index),
                  case_card_json = VALUES(case_card_json),
                  case_card_embedding = VALUES(case_card_embedding),
                  source_hash = VALUES(source_hash),
                  extraction_input_hash = VALUES(extraction_input_hash),
                  embedding_input_hash = VALUES(embedding_input_hash),
                  extractor_version = VALUES(extractor_version),
                  embedding_model = VALUES(embedding_model),
                  embedding_dimension = VALUES(embedding_dimension)"""

        job_sql = """INSERT INTO record_day_processing_jobs
                 (day_record_id, patient_id, source_hash, extraction_input_hash,
                  extractor_version, embedding_model, embedding_input_hash,
                  embedding_dimension, processing_status, failed_stage,
                  attempt_count, last_error, next_retry_at)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, NULL)
                 ON DUPLICATE KEY UPDATE
                  patient_id = VALUES(patient_id),
                  source_hash = VALUES(source_hash),
                  extraction_input_hash = VALUES(extraction_input_hash),
                  extractor_version = VALUES(extractor_version),
                  embedding_model = VALUES(embedding_model),
                  embedding_input_hash = VALUES(embedding_input_hash),
                  embedding_dimension = VALUES(embedding_dimension),
                  processing_status = VALUES(processing_status),
                  failed_stage = VALUES(failed_stage),
                  attempt_count = IF(VALUES(processing_status) = 'indexed', 0, attempt_count + 1),
                  last_error = VALUES(last_error),
                  next_retry_at = NULL"""

        batch_size = self._cfg.batch_size
        result = BatchWriteResult()

        def card_params(chunk: List[DayCardWriteRow]) -> list[tuple]:
            return [(
                r.day_record_id, r.patient_id, r.day_index,
                r.case_card_json, r.case_card_embedding,
                r.source_hash or "", r.extraction_input_hash or "",
                r.embedding_input_hash or "", r.extractor_version,
                r.embedding_model, r.embedding_dimension,
            ) for r in chunk]

        def job_params(chunk: List[DayCardWriteRow]) -> list[tuple]:
            return [(
                r.day_record_id, r.patient_id, r.source_hash or "",
                r.extraction_input_hash or "", r.extractor_version,
                r.embedding_model, r.embedding_input_hash or "",
                r.embedding_dimension, r.processing_status,
                r.failed_stage or None, (r.last_error or "")[:1000],
            ) for r in chunk]

        def write_chunk(cursor, chunk: List[DayCardWriteRow]) -> None:
            cursor.executemany(sql, card_params(chunk))
            cursor.executemany(job_sql, job_params(chunk))

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for chunk in _chunked(rows, batch_size):
                    try:
                        write_chunk(cursor, chunk)
                        conn.commit()
                        result.success_count += len(chunk)
                    except Exception as e:
                        conn.rollback()
                        logger.warning("Batch write failed (size=%d), splitting for retry: %s",
                                       len(chunk), e)
                        if len(chunk) > 1:
                            # Split retry: try each row individually
                            for row in chunk:
                                try:
                                    write_chunk(cursor, [row])
                                    conn.commit()
                                    result.success_count += 1
                                except Exception as inner_e:
                                    conn.rollback()
                                    result.fail_count += 1
                                    result.failed_record_ids.append(row.day_record_id)
                                    result.errors[row.day_record_id] = str(inner_e)[:500]
                        else:
                            result.fail_count += 1
                            result.failed_record_ids.append(chunk[0].day_record_id)
                            result.errors[chunk[0].day_record_id] = str(e)[:500]
        finally:
            conn.close()

        if result.fail_count > 0:
            logger.warning("Batch write complete: %d success, %d failed",
                           result.success_count, result.fail_count)
        return result

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

    def existing_day_ids(self, day_record_ids: list[str]) -> set[str]:
        """批量检查哪些 day_record_id 已存在（单次连接，避免端口耗尽）。"""
        if not day_record_ids:
            return set()
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                # 用 IN + 参数化查询，一次性获取所有已有的 ID
                placeholders = ",".join(["%s"] * len(day_record_ids))
                cursor.execute(
                    f"SELECT day_record_id FROM record_days WHERE day_record_id IN ({placeholders})",
                    day_record_ids,
                )
                return {row["day_record_id"] for row in cursor.fetchall()}
        finally:
            conn.close()

    def existing_day_card_ids(
        self,
        day_record_ids: list[str],
        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION,
    ) -> set[str]:
        """批量检查哪些 day_record_id 已有当前版本病例卡（单次连接）。"""
        if not day_record_ids:
            return set()
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                placeholders = ",".join(["%s"] * len(day_record_ids))
                cursor.execute(
                    f"""SELECT day_record_id FROM record_day_case_cards
                        WHERE day_record_id IN ({placeholders})
                          AND extractor_version = %s""",
                    (*day_record_ids, extractor_version),
                )
                return {row["day_record_id"] for row in cursor.fetchall()}
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
                    """SELECT day_record_id, patient_id, day_index, visit_date, day_text,
                              source_hash
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
                "cumulative_text": "",
                "source_hash": row.get("source_hash") or "",
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
                              day_delta_embedding, cumulative_embedding, case_card_embedding
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
            case_card_blob = row.get("case_card_embedding")
            item = {
                "day_record_id": row["day_record_id"],
                "patient_id": row["patient_id"],
                "day_index": row["day_index"],
                "case_card": card,
                "day_delta_embedding": np.frombuffer(day_blob, dtype=np.float32) if day_blob else None,
                "cumulative_embedding": np.frombuffer(cumulative_blob, dtype=np.float32) if cumulative_blob else None,
                "case_card_embedding": np.frombuffer(case_card_blob, dtype=np.float32) if case_card_blob else None,
            }
            result.setdefault(row["patient_id"], []).append(item)
        return result

    def list_all_patient_ids(self) -> List[str]:
        """返回所有不同的 patient_id（用于全量扫描回退）。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT patient_id FROM record_days ORDER BY patient_id")
                return [row["patient_id"] for row in cursor.fetchall()]
        finally:
            conn.close()

    def load_case_cards_by_ids(self, day_record_ids: List[str]) -> Dict[str, dict]:
        """加载现有病例卡，供 ``embed_only`` 路径复用。"""
        if not day_record_ids:
            return {}

        rows = []
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for ids in _chunked(day_record_ids, 500):
                    placeholders = ",".join(["%s"] * len(ids))
                    cursor.execute(f"""
                        SELECT day_record_id, patient_id, day_index, case_card_json
                        FROM record_day_case_cards
                        WHERE day_record_id IN ({placeholders})
                    """, ids)
                    rows.extend(cursor.fetchall())
        finally:
            conn.close()

        result = {}
        for row in rows:
            value = row.get("case_card_json")
            result[row["day_record_id"]] = (
                json.loads(value) if isinstance(value, str) else value
            )
        return result

    def load_processing_snapshots(self, day_record_ids: list[str]) -> dict[str, ProcessingSnapshot]:
        """批量加载处理状态快照（单次 JOIN 查询）。

        返回以 day_record_id 为键的 ProcessingSnapshot 字典。
        不存在的记录不会出现在结果中。
        """
        if not day_record_ids:
            return {}

        rows = []
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for ids in _chunked(day_record_ids, 500):
                    placeholders = ",".join(["%s"] * len(ids))
                    cursor.execute(f"""
                        SELECT
                            d.day_record_id,
                            COALESCE(c.source_hash, '') AS source_hash,
                            COALESCE(c.extraction_input_hash, '') AS extraction_input_hash,
                            COALESCE(c.extractor_version, '') AS extractor_version,
                            COALESCE(c.embedding_model, '') AS embedding_model,
                            COALESCE(c.embedding_input_hash, '') AS embedding_input_hash,
                            COALESCE(c.embedding_dimension, 0) AS embedding_dimension,
                            COALESCE(j.processing_status, 'pending') AS processing_status,
                            COALESCE(j.failed_stage, '') AS failed_stage,
                            (c.day_record_id IS NOT NULL AND c.case_card_json IS NOT NULL) AS has_valid_card,
                            (c.case_card_embedding IS NOT NULL) AS has_valid_embedding
                        FROM record_days d
                        LEFT JOIN record_day_case_cards c ON c.day_record_id = d.day_record_id
                        LEFT JOIN record_day_processing_jobs j ON j.day_record_id = d.day_record_id
                        WHERE d.day_record_id IN ({placeholders})
                    """, ids)
                    rows.extend(cursor.fetchall())
        finally:
            conn.close()

        result = {}
        for row in rows:
            result[row["day_record_id"]] = ProcessingSnapshot(
                day_record_id=row["day_record_id"],
                source_hash=row.get("source_hash") or "",
                extraction_input_hash=row.get("extraction_input_hash") or "",
                extractor_version=row.get("extractor_version") or "",
                embedding_model=row.get("embedding_model") or "",
                embedding_input_hash=row.get("embedding_input_hash") or "",
                embedding_dimension=int(row.get("embedding_dimension") or 0),
                processing_status=row.get("processing_status") or "pending",
                failed_stage=row.get("failed_stage") or "",
                has_valid_card=bool(row.get("has_valid_card")),
                has_valid_embedding=bool(row.get("has_valid_embedding")),
            )
        return result

    def upsert_processing_job(self, job: ProcessingJob) -> None:
        """插入或更新处理任务状态。"""
        sql = """INSERT INTO record_day_processing_jobs
                 (day_record_id, patient_id, source_hash, extraction_input_hash,
                  extractor_version, embedding_model, embedding_input_hash,
                  embedding_dimension, processing_status, failed_stage,
                  attempt_count, last_error, next_retry_at)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  source_hash = VALUES(source_hash),
                  extraction_input_hash = VALUES(extraction_input_hash),
                  extractor_version = VALUES(extractor_version),
                  embedding_model = VALUES(embedding_model),
                  embedding_input_hash = VALUES(embedding_input_hash),
                  embedding_dimension = VALUES(embedding_dimension),
                  processing_status = VALUES(processing_status),
                  failed_stage = VALUES(failed_stage),
                  attempt_count = VALUES(attempt_count),
                  last_error = VALUES(last_error),
                  next_retry_at = VALUES(next_retry_at)"""

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    job.day_record_id, job.patient_id, job.source_hash,
                    job.extraction_input_hash, job.extractor_version,
                    job.embedding_model, job.embedding_input_hash,
                    job.embedding_dimension, job.processing_status,
                    job.failed_stage, job.attempt_count, job.last_error,
                    job.next_retry_at,
                ))
            conn.commit()
        finally:
            conn.close()

    def update_processing_failures(self, failures: list[ProcessingFailure]) -> None:
        """批量更新处理失败状态。"""
        if not failures:
            return

        sql = """INSERT INTO record_day_processing_jobs
                 (day_record_id, patient_id, source_hash, extraction_input_hash,
                  extractor_version, embedding_model, processing_status,
                  failed_stage, attempt_count, last_error)
                 VALUES (%s, %s, %s, %s, %s, %s, 'failed_retryable', %s, %s, %s)
                 ON DUPLICATE KEY UPDATE
                  patient_id = VALUES(patient_id),
                  source_hash = VALUES(source_hash),
                  extraction_input_hash = VALUES(extraction_input_hash),
                  extractor_version = VALUES(extractor_version),
                  embedding_model = VALUES(embedding_model),
                  processing_status = 'failed_retryable',
                  failed_stage = VALUES(failed_stage),
                  attempt_count = attempt_count + 1,
                  last_error = VALUES(last_error)"""

        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                params = [
                    (f.day_record_id, f.patient_id, f.source_hash,
                     f.extraction_input_hash, f.extractor_version,
                     f.embedding_model, f.failed_stage,
                     f.attempt_count, f.error[:1000] if f.error else "")
                    for f in failures
                ]
                cursor.executemany(sql, params)
            conn.commit()
        finally:
            conn.close()

    def load_days_for_patients(self, patient_ids: List[str]) -> Dict[str, List[dict]]:
        """懒加载指定患者的 day_text 和元数据（不全量加载所有患者）。

        Stage 5: 用于候选检索后的按需加载。
        """
        if not patient_ids:
            return {}

        from data_migrate.database import chunked as _chunked

        all_rows = []
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for chunk in _chunked(patient_ids, size=500):
                    placeholders = ",".join(["%s"] * len(chunk))
                    cursor.execute(f"""
                        SELECT day_record_id, patient_id, day_index, visit_date, day_text,
                               source_hash
                        FROM record_days
                        WHERE patient_id IN ({placeholders})
                        ORDER BY patient_id, day_index
                    """, chunk)
                    all_rows.extend(cursor.fetchall())
        finally:
            conn.close()

        result: Dict[str, List[dict]] = {}
        for row in all_rows:
            result.setdefault(row["patient_id"], []).append({
                "day_record_id": row["day_record_id"],
                "patient_id": row["patient_id"],
                "day_index": row["day_index"],
                "visit_date": str(row["visit_date"]) if row.get("visit_date") else "",
                "day_text": row.get("day_text") or "",
                "source_hash": row.get("source_hash") or "",
            })
        return result

    def load_day_cards_for_patients(
        self, patient_ids: List[str],
        extractor_version: str = DEFAULT_DAY_EXTRACTOR_VERSION,
    ) -> Dict[str, List[dict]]:
        """懒加载指定患者的病例卡和 embedding（不全量加载所有患者）。

        Stage 5: 用于候选检索后的按需加载。
        """
        if not patient_ids:
            return {}

        from data_migrate.database import chunked as _chunked

        all_rows = []
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                for chunk in _chunked(patient_ids, size=500):
                    placeholders = ",".join(["%s"] * len(chunk))
                    cursor.execute(f"""
                        SELECT c.day_record_id, c.patient_id, c.day_index,
                               c.case_card_json, c.case_card_embedding
                        FROM record_day_case_cards c
                        JOIN record_days d ON d.day_record_id = c.day_record_id
                        WHERE c.patient_id IN ({placeholders})
                          AND c.extractor_version = %s
                          AND c.source_hash = d.source_hash
                          AND c.extraction_input_hash = d.extraction_input_hash
                        ORDER BY c.patient_id, c.day_index
                    """, (*chunk, extractor_version))
                    all_rows.extend(cursor.fetchall())
        finally:
            conn.close()

        result: Dict[str, List[dict]] = {}
        for row in all_rows:
            card_json = row["case_card_json"]
            card = json.loads(card_json) if isinstance(card_json, str) else card_json
            case_card_blob = row.get("case_card_embedding")
            item = {
                "day_record_id": row["day_record_id"],
                "patient_id": row["patient_id"],
                "day_index": row["day_index"],
                "case_card": card,
                "day_delta_embedding": None,
                "cumulative_embedding": None,
                "case_card_embedding": (
                    np.frombuffer(case_card_blob, dtype=np.float32) if case_card_blob else None
                ),
            }
            result.setdefault(row["patient_id"], []).append(item)
        return result

    def count_days(self) -> int:
        return self._count("record_days")

    def count_patients(self) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(DISTINCT patient_id) AS cnt FROM record_days")
                row = cursor.fetchone()
                return row["cnt"] if row else 0
        finally:
            conn.close()

    def count_day_cards(self) -> int:
        return self._count("record_day_case_cards")

    def count_day_embeddings(self) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT COUNT(*) AS cnt FROM record_day_case_cards
                       WHERE case_card_embedding IS NOT NULL"""
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
