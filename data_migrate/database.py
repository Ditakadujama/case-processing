"""
MySQL 数据库访问模块
提供病历数据的读取、写入和迁移功能
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:
    raise ImportError("请先安装 PyMySQL: pip install PyMySQL>=1.1.0")

logger = logging.getLogger(__name__)


@dataclass
class DBConfig:
    """MySQL 连接配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = "1887415157Oxx/"
    database: str = "medical_records"
    charset: str = "utf8mb4"

    @classmethod
    def from_env(cls) -> "DBConfig":
        """从环境变量读取配置（未设置的环境变量使用类默认值）"""
        cfg = cls()
        cfg.host = os.getenv("DB_HOST", cfg.host)
        cfg.port = int(os.getenv("DB_PORT", str(cfg.port)))
        cfg.user = os.getenv("DB_USER", cfg.user)
        cfg.password = os.getenv("DB_PASSWORD", cfg.password)
        cfg.database = os.getenv("DB_NAME", cfg.database)
        return cfg

    def to_connection_kwargs(self) -> dict:
        """转换为 pymysql.connect 参数"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "cursorclass": DictCursor,
        }


@contextmanager
def get_db_connection(cfg: DBConfig):
    """
    获取数据库连接（上下文管理器）

    Args:
        cfg: 数据库配置

    Yields:
        pymysql.Connection
    """
    conn = None
    try:
        conn = pymysql.connect(**cfg.to_connection_kwargs())
        yield conn
    except pymysql.MySQLError as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    finally:
        if conn:
            conn.close()


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS medical_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL COMMENT '患者唯一标识',
    visit_date DATE NOT NULL COMMENT '就诊日期',
    patient_info TEXT COMMENT '患者基本信息',
    chief_complaint TEXT COMMENT '主诉',
    instrument_test TEXT COMMENT '器械检查',
    checkout TEXT COMMENT '检验结果',
    examine TEXT COMMENT '检查记录',
    doctor_advice TEXT COMMENT '医嘱',
    inspection_visit TEXT COMMENT '查房记录',
    history_illness TEXT COMMENT '既往病史',
    surgery_record TEXT COMMENT '手术记录',
    monitor TEXT COMMENT '监护记录',
    operation_record TEXT COMMENT '手术操作记录',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_patient_id (patient_id),
    INDEX idx_visit_date (visit_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='病历原始记录表';
"""


def init_database(cfg: DBConfig) -> None:
    """初始化数据库：创建表（如果不存在）"""
    with get_db_connection(cfg) as conn:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
        logger.info(f"数据库表 medical_records 已就绪 (database={cfg.database})")


def load_records_from_db(cfg: DBConfig) -> Dict[str, str]:
    """
    从数据库读取病历数据，按 patient_id 分组合并。

    合并格式与 load_xlsx_records() 完全一致：
      - 按 patient_id 分组
      - 组内按 visit_date 排序
      - 每条记录格式：
        ###就诊记录 i/total - YYYY-MM-DD
        ###patient_info: xxx
        ###chief_complaint: xxx
        ...

    Args:
        cfg: 数据库配置

    Returns:
        {patient_id: 合并后的病历文本}
    """
    columns_to_merge = [
        "patient_info",
        "chief_complaint",
        "instrument_test",
        "checkout",
        "examine",
        "doctor_advice",
        "inspection_visit",
        "history_illness",
        "surgery_record",
        "monitor",
        "operation_record",
    ]

    sql = """
        SELECT patient_id, visit_date, {}
        FROM medical_records
        ORDER BY patient_id, visit_date
    """.format(", ".join(f"`{c}`" for c in columns_to_merge))

    records: Dict[str, List[str]] = {}
    patient_order: List[str] = []

    with get_db_connection(cfg) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

    for row in rows:
        patient_id = row["patient_id"]
        if patient_id not in records:
            records[patient_id] = []
            patient_order.append(patient_id)

        date_str = str(row["visit_date"]) if row["visit_date"] else ""
        parts = [f"###就诊记录 - {date_str}"]

        for col in columns_to_merge:
            val = row.get(col)
            if val is not None and str(val).strip():
                parts.append(f"###{col}: {val}")

        records[patient_id].append("\n".join(parts))

    # 合并为最终文本
    merged = {}
    for pid in patient_order:
        items = records[pid]
        total = len(items)
        # 补充每条记录的序号
        final_parts = []
        for i, part in enumerate(items, 1):
            # 替换 "###就诊记录 - date" 为 "###就诊记录 i/total - date"
            lines = part.split("\n")
            if lines and lines[0].startswith("###就诊记录 -"):
                lines[0] = f"###就诊记录 {i}/{total} - {lines[0][len('###就诊记录 - '):]}"
            final_parts.append("\n".join(lines))
        merged[pid] = "\n\n".join(final_parts)

    logger.info(f"从数据库读取到 {len(merged)} 个患者，共 {len(rows)} 条就诊记录")
    return merged


def insert_records(cfg: DBConfig, records_df: Any) -> int:
    """
    将 pandas DataFrame 批量写入 medical_records 表。

    Args:
        cfg: 数据库配置
        records_df: pandas DataFrame，列名需与表字段对应

    Returns:
        插入的行数
    """
    import pandas as pd

    if records_df.empty:
        return 0

    # 确保列名存在
    db_columns = [
        "patient_id", "visit_date", "patient_info", "chief_complaint",
        "instrument_test", "checkout", "examine", "doctor_advice",
        "inspection_visit", "history_illness", "surgery_record",
        "monitor", "operation_record",
    ]

    available_cols = [c for c in db_columns if c in records_df.columns]
    if not available_cols:
        raise ValueError("DataFrame 中没有匹配的列")

    # 构建 INSERT SQL
    cols_str = ", ".join(f"`{c}`" for c in available_cols)
    placeholders = ", ".join(["%s"] * len(available_cols))
    sql = f"INSERT INTO medical_records ({cols_str}) VALUES ({placeholders})"

    rows_inserted = 0
    batch_size = 500

    with get_db_connection(cfg) as conn:
        with conn.cursor() as cursor:
            # 分批写入
            values = []
            for _, row in records_df.iterrows():
                vals = []
                for col in available_cols:
                    v = row.get(col)
                    if pd.isna(v):
                        vals.append(None)
                    else:
                        vals.append(v)
                values.append(vals)

                if len(values) >= batch_size:
                    cursor.executemany(sql, values)
                    rows_inserted += cursor.rowcount
                    values = []

            if values:
                cursor.executemany(sql, values)
                rows_inserted += cursor.rowcount

        conn.commit()

    logger.info(f"成功写入 {rows_inserted} 条记录到数据库")
    return rows_inserted


def test_connection(cfg: DBConfig) -> bool:
    """测试数据库连接是否可用"""
    try:
        with get_db_connection(cfg) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return True
    except Exception as e:
        logger.error(f"数据库连接测试失败: {e}")
        return False
