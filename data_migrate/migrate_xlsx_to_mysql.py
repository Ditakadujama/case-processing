"""
病历数据迁移脚本：将 xlsx 数据一次性导入 MySQL

用法:
    python migrate_xlsx_to_mysql.py --xlsx "patient_info (18).xlsx"

环境变量（配置数据库连接）:
    export DB_HOST=localhost
    export DB_PORT=3306
    export DB_USER=root
    export DB_PASSWORD=your_password
    export DB_NAME=medical_records
"""

import os
import sys
import argparse
import logging

try:
    import pandas as pd
except ImportError:
    raise ImportError("请先安装 pandas: pip install pandas")

from database import DBConfig, init_database, insert_records, test_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_xlsx_raw(filepath: str) -> pd.DataFrame:
    """
    读取原始 xlsx，返回 DataFrame（不做分组合并）。
    会自动处理列名映射和缺失值。
    """
    import pandas as pd

    logger.info(f"读取 Excel 文件: {filepath}")
    df = pd.read_excel(filepath, sheet_name="Sheet1")
    logger.info(f"总记录数: {len(df)} 行, 列: {list(df.columns)}")

    # 列名映射：xlsx -> 数据库
    if "date" in df.columns and "visit_date" not in df.columns:
        df = df.rename(columns={"date": "visit_date"})

    # 填充缺失的日期（数据库要求 NOT NULL）
    if "visit_date" in df.columns:
        missing_dates = df["visit_date"].isna().sum()
        if missing_dates > 0:
            logger.warning(f"发现 {missing_dates} 条缺失日期，已填充为 1900-01-01")
            df["visit_date"] = df["visit_date"].fillna(pd.Timestamp("1900-01-01"))

    return df


def main():
    parser = argparse.ArgumentParser(description="将 Excel 病历数据迁移到 MySQL")
    parser.add_argument(
        "--xlsx",
        default="patient_info (18).xlsx",
        help="Excel 文件路径 (默认: patient_info (18).xlsx)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="仅初始化数据库表，不导入数据",
    )
    args = parser.parse_args()

    # 1. 读取配置
    cfg = DBConfig.from_env()
    logger.info(
        f"数据库配置: {cfg.user}@{cfg.host}:{cfg.port}/{cfg.database}"
    )

    # 2. 测试连接
    if not test_connection(cfg):
        print("\n数据库连接失败，请检查：")
        print("1. MySQL 服务是否已启动")
        print("2. 环境变量 DB_HOST, DB_PORT, DB_USER, DB_PASSWORD 是否正确")
        print("\n示例（设置环境变量）：")
        print("  export DB_HOST=localhost")
        print("  export DB_PORT=3306")
        print("  export DB_USER=root")
        print("  export DB_PASSWORD=your_password")
        print("  export DB_NAME=medical_records")
        print("\n如果本地没有 MySQL，可用 Docker 快速启动：")
        print("  docker run -d --name mysql -p 3306:3306 -e MYSQL_ROOT_PASSWORD=your_password mysql:8")
        sys.exit(1)

    # 3. 初始化表
    init_database(cfg)

    if args.init_only:
        print("\n数据库表已初始化，未导入数据。")
        return

    # 4. 读取 xlsx
    if not os.path.exists(args.xlsx):
        logger.error(f"文件不存在: {args.xlsx}")
        sys.exit(1)

    df = load_xlsx_raw(args.xlsx)

    if df.empty:
        logger.warning("Excel 中没有数据")
        return

    # 5. 写入数据库
    print(f"\n开始写入 {len(df)} 条记录到数据库...")
    inserted = insert_records(cfg, df)
    print(f"\n迁移完成！成功写入 {inserted} 条记录")

    # 6. 简单统计
    with __import__("database").get_db_connection(cfg) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(DISTINCT patient_id) AS patients, "
                "COUNT(*) AS records FROM medical_records"
            )
            result = cursor.fetchone()
            print(
                f"数据库统计: {result['patients']} 个患者, "
                f"{result['records']} 条就诊记录"
            )


if __name__ == "__main__":
    main()
