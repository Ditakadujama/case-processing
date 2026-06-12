"""
病历相似度检索系统 - 主程序入口。

当前版本只使用医生关注文本做天级序列相似：
- --build:  从 medical_records 原始日行增量构建 record_days / record_day_case_cards
- --search: 从 Excel 导入 query_records，按天级序列检索相似患者
"""

import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinical_text_filter import ClinicalDayRecord, build_clinical_day_records
from config import DBConfig, EmbeddingConfig, LLMConfig
from data_migrate.database import (
    clear_query_records,
    init_query_records_table,
    insert_records,
    load_medical_daily_rows,
    load_query_daily_rows,
)
from day_retrieval_system import DayLevelRetrievalSystem
from day_store import DEFAULT_DAY_EXTRACTOR_VERSION, MySQLDayStore
from embedding_index import EmbeddingService
from llm_case_extractor import LLMCaseExtractor


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_xlsx_for_records(filepath: str):
    """读取与迁移 Excel 相同格式的病历表。"""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("请先安装 pandas: pip install pandas")

    df = pd.read_excel(filepath, sheet_name="Sheet1")
    if "date" in df.columns and "visit_date" not in df.columns:
        df = df.rename(columns={"date": "visit_date"})

    required = {"patient_id", "visit_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少必要列: {', '.join(sorted(missing))}")

    missing_dates = df["visit_date"].isna().sum()
    if missing_dates > 0:
        print(f"警告: 查询 Excel 中有 {missing_dates} 条缺失日期，已填充为 1900-01-01")
        df["visit_date"] = df["visit_date"].fillna(pd.Timestamp("1900-01-01"))
    return df


def import_query_excel_to_db(cfg: DBConfig, excel_path: str) -> None:
    """每次检索前清空 query_records，并导入本次查询 Excel。"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"查询 Excel 不存在: {excel_path}")

    init_query_records_table(cfg)
    clear_query_records(cfg)
    df = load_xlsx_for_records(excel_path)
    inserted = insert_records(cfg, df, table_name="query_records") if not df.empty else 0
    print(f"已清空 query_records，并从 Excel 导入 {inserted} 条查询就诊记录")


def safe_result_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def build_index(skip_existing: bool = True,
                limit: int = 0,
                llm_workers: int = 5,
                llm_interval: float = 1.0,
                rebuild_all: bool = False) -> None:
    """从 medical_records 构建医生关注文本的天级索引。"""
    print("=" * 60)
    print("构建天级索引 (医生关注文本 + LLM 日病例卡 + Embedding)")
    print("=" * 60)

    cfg = DBConfig.from_env()
    llm_cfg = LLMConfig()
    emb_cfg = EmbeddingConfig()
    if not llm_cfg.is_configured:
        print("\n错误: LLM 服务未配置，请设置 LLM_API_BASE 和 LLM_API_KEY")
        return
    if not emb_cfg.is_configured:
        print("\n错误: Embedding 服务未配置，请设置 EMBEDDING_API_BASE 和 EMBEDDING_API_KEY")
        return

    print(f"LLM 服务: {llm_cfg.model} @ {llm_cfg.api_base}")
    print(f"Embedding 服务: {emb_cfg.model} @ {emb_cfg.api_base}")

    day_store = MySQLDayStore(cfg)
    day_store.init_tables()
    if rebuild_all:
        day_store.delete_all()
        print("已清空 record_days 和 record_day_case_cards")
    else:
        print(f"增量 build：当前 record_days={day_store.count_days()}，"
              f"record_day_case_cards={day_store.count_day_cards()}")

    rows = load_medical_daily_rows(cfg)
    if limit > 0:
        rows = rows[:limit]
        print(f"限制处理前 {limit} 条原始日记录")
    if not rows:
        print("medical_records 中没有原始记录，请先迁移数据")
        return

    day_records = build_clinical_day_records(rows)
    before = len(day_records)
    if skip_existing and not rebuild_all:
        all_ids = [day.day_record_id for day in day_records]
        existing_day_set = day_store.existing_day_ids(all_ids)
        existing_card_set = day_store.existing_day_card_ids(all_ids, DEFAULT_DAY_EXTRACTOR_VERSION)
        day_records = [
            day for day in day_records
            if day.day_record_id not in existing_day_set
            or day.day_record_id not in existing_card_set
        ]
        skipped = before - len(day_records)
        if skipped:
            print(f"跳过已存在同版本天级索引: {skipped} 天")

    if not day_records:
        print("没有需要新增 build 的天级记录。")
        return

    print(f"准备构建 {len(day_records)} 条天级记录，来自 {len(set(d.patient_id for d in day_records))} 个患者")

    batch_size = 200
    llm_extractor = LLMCaseExtractor(llm_cfg)
    emb_service = EmbeddingService(emb_cfg)
    for start in range(0, len(day_records), batch_size):
        batch = day_records[start:start + batch_size]
        _insert_day_texts(day_store, batch)
        _extract_day_case_cards_batch(
            batch,
            llm_extractor,
            emb_service,
            day_store,
            skip_existing=skip_existing,
            llm_workers=llm_workers,
            llm_interval=llm_interval,
        )
        done = min(start + batch_size, len(day_records))
        print(f"  [{done}/{len(day_records)}] 天级记录已处理 ({done * 100 // len(day_records)}%)")

    stats = llm_extractor.get_stats()
    emb_stats = emb_service.get_stats()
    print("\n构建完成！")
    print(f"record_days 表共 {day_store.count_days()} 条记录")
    print(f"record_day_case_cards 表共 {day_store.count_day_cards()} 条记录")
    print(f"  -- 其中有天级 embedding: {day_store.count_day_embeddings()} 条")
    print(f"  -- LLM 调用: {stats['request_count']} 次, 失败: {stats['error_count']} 次, "
          f"总 tokens: {stats['total_tokens']}")
    print(f"  -- Embedding 调用: {emb_stats['request_count']} 次, 失败: {emb_stats['error_count']} 次")


def _insert_day_texts(day_store: MySQLDayStore, day_records: list[ClinicalDayRecord]) -> None:
    rows = [
        (
            day.day_record_id,
            day.patient_id,
            day.day_index,
            day.visit_date,
            day.day_text,
            day.cumulative_text,
        )
        for day in day_records
    ]
    day_store.insert_day_rows(rows)


def _extract_day_case_cards_batch(day_records: list[ClinicalDayRecord],
                                  llm_extractor: LLMCaseExtractor,
                                  emb_service: EmbeddingService,
                                  day_store: MySQLDayStore,
                                  skip_existing: bool,
                                  llm_workers: int = 5,
                                  llm_interval: float = 1.0) -> None:
    """批量抽取每日病例卡 + day/cumulative embedding。"""
    if not day_records:
        return

    embedding_model = emb_service.config.model if emb_service else ""
    stats = {"success": 0, "skip": 0, "fail": 0}
    next_request_time = 0.0
    rate_lock = threading.Lock()
    min_interval = max(float(llm_interval), 0.0)

    def wait_rate_limit() -> None:
        nonlocal next_request_time
        with rate_lock:
            now = time.time()
            if now < next_request_time:
                time.sleep(next_request_time - now)
            next_request_time = time.time() + min_interval

    def process_one(day: ClinicalDayRecord) -> str:
        if skip_existing and day_store.day_card_exists(day.day_record_id, DEFAULT_DAY_EXTRACTOR_VERSION):
            return "skip"
        try:
            wait_rate_limit()
            card = llm_extractor.extract_day(
                day.day_text,
                day.cumulative_text,
                patient_id=day.patient_id,
                day_index=day.day_index,
                visit_date=day.visit_date,
            )
            if not card:
                logger.warning(f"[{day.day_record_id}] 天级病例卡抽取失败")
                return "fail"

            day_embedding = None
            cumulative_embedding = None
            day_summary = card.get("day_summary_for_embedding") or card.get("summary_for_embedding", "")
            cumulative_summary = card.get("cumulative_summary_for_embedding") or ""
            try:
                if day_summary:
                    day_embedding = emb_service.embed_text(day_summary)
                if cumulative_summary:
                    cumulative_embedding = emb_service.embed_text(cumulative_summary)
            except Exception as e:
                logger.warning(f"[{day.day_record_id}] 天级 embedding 生成失败: {e}")

            day_store.insert_day_card(
                day.day_record_id,
                day.patient_id,
                day.day_index,
                card,
                day_embedding,
                cumulative_embedding,
                extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
                embedding_model=embedding_model,
            )
            return "success"
        except Exception as e:
            logger.error(f"[{day.day_record_id}] 天级病例卡处理异常: {e}")
            return "fail"

    total = len(day_records)
    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = {executor.submit(process_one, day): day.day_record_id for day in day_records}
        for future in as_completed(futures):
            try:
                status = future.result()
            except Exception:
                status = "fail"
            stats[status] += 1
            done = stats["success"] + stats["skip"] + stats["fail"]
            print(f"\r  LLM 每日病例卡: [{done}/{total}] "
                  f"(成功 {stats['success']}, 跳过 {stats['skip']}, 失败 {stats['fail']})",
                  end="", flush=True)
    print()


def search(search_excel: str, daily_days: int = 0) -> None:
    """从查询 Excel 检索相似患者。"""
    print("=" * 60)
    print("天级相似病例检索")
    print("=" * 60)

    cfg = DBConfig.from_env()
    try:
        import_query_excel_to_db(cfg, search_excel)
    except Exception as e:
        print(f"导入查询 Excel 失败: {e}")
        return

    query_rows = load_query_daily_rows(cfg)
    if not query_rows:
        print("查询 Excel 中没有有效病例")
        return

    system = DayLevelRetrievalSystem(cfg)
    print(f"底库患者数: {system.patient_count}，天级记录数: {system.day_count}")
    if system.day_count == 0:
        print("底库为空，请先运行: python main.py --build")
        return

    query_days_by_patient = system.build_query_days(query_rows, max_days=daily_days)
    if not query_days_by_patient:
        print("查询病例过滤后没有可用于相似度的医生关注文本")
        return

    result_dir = "data/results"
    os.makedirs(result_dir, exist_ok=True)
    all_result_count = 0

    print(f"查询来源: {search_excel}")
    print(f"找到 {len(query_days_by_patient)} 个查询患者")
    if daily_days > 0:
        print(f"仅比较查询病例前 {daily_days} 天")
    else:
        print("按查询病例全部已有天数比较")

    for query_patient_id, query_days in query_days_by_patient.items():
        print(f"\n{'=' * 60}")
        print(f"查询病例: {query_patient_id} ({len(query_days)} 天)")
        print(f"{'=' * 60}")
        results = system.search(
            query_days,
            top_k=5,
            exclude_patient_ids={query_patient_id},
            max_days=daily_days,
        )
        all_result_count += len(results)
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result['id']} 综合: {result['similarity']} "
                  f"匹配天数: {result['daily_matched_days']}/{result['daily_compare_days']} "
                  f"候选天数: {result['daily_candidate_days']}")
            print(f"      天级对齐: {result.get('daily_match_details', [])[:5]}")

        _write_query_result_file(result_dir, search_excel, query_patient_id, query_days, results)

    print(f"\n检索结果已保存到: {result_dir}/")
    print(f"共 {len(query_days_by_patient)} 个查询患者，{all_result_count} 条结果")
    print("=" * 60)


def _write_query_result_file(result_dir: str, search_excel: str, query_patient_id: str,
                             query_days: List[dict], results: List[dict]) -> None:
    output_file = os.path.join(result_dir, f"{safe_result_name(query_patient_id)}_结果.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"查询病例: {query_patient_id}\n")
        f.write(f"查询来源: {search_excel}\n")
        f.write(f"查询天数: {len(query_days)}\n")
        f.write("=" * 80 + "\n\n")
        f.write("【查询病例医生关注文本】\n")
        for day in query_days:
            f.write(f"\n###查询 Day {day['day_index']} - {day.get('visit_date', '')}\n")
            f.write(day.get("day_text", ""))
            f.write("\n")
        f.write("\n" + "=" * 80 + "\n\n")

        for rank, result in enumerate(results, 1):
            f.write(f"【Top {rank}】{result['id']}\n")
            f.write(f"综合相似度: {result['similarity']}\n")
            f.write(f"天级比较天数: {result.get('daily_compare_days', 0)}\n")
            f.write(f"查询病例天数: {result.get('daily_query_days', 0)}\n")
            f.write(f"候选病例天数: {result.get('daily_candidate_days', 0)}\n")
            f.write(f"实际匹配天数: {result.get('daily_matched_days', 0)}\n")
            f.write(f"天级覆盖度: {result.get('daily_coverage', '-')}\n")
            f.write(f"天级长度关系: {result.get('daily_length_relation', '')}\n")
            f.write(f"天级对齐明细: {result.get('daily_match_details', [])}\n")
            f.write("\n--- 匹配候选天文本 ---\n")
            f.write(result.get("full_text", ""))
            f.write("\n" + "=" * 80 + "\n\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="病历相似度检索系统")
    parser.add_argument("--build", action="store_true", default=False,
                        help="构建/增量更新医生关注文本天级索引")
    parser.add_argument("--search", default="", metavar="EXCEL",
                        help="查询 Excel 路径；每次检索前清空 query_records，并导入该 Excel")
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="兼容旧参数；当前天级 build 不再使用多进程解析")
    parser.add_argument("--skip-existing-case-cards", action="store_true", default=False,
                        help="兼容旧参数；当前默认跳过已有同版本天级索引")
    parser.add_argument("--rebuild-all", action="store_true", default=False,
                        help="清空天级索引表后重新全量 build")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制 build 处理原始日记录条数（0=不限制）")
    parser.add_argument("--llm-workers", type=int, default=5,
                        help="LLM 日病例卡抽取并行线程数")
    parser.add_argument("--llm-interval", type=float, default=1.0,
                        help="两次 LLM 请求的最小间隔秒数；10次/秒限速可设为 0.1")
    parser.add_argument("--daily-days", type=int, default=0,
                        help="天级比较天数：0=按查询病例全部已有天数比较；N>0=只比较前 N 天")
    args = parser.parse_args()

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "病历相似度检索系统" + " " * 25 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    if args.build and args.search:
        parser.error("--build 和 --search 不能同时使用")
    if not args.build and not args.search:
        parser.error("请指定 --build 或 --search EXCEL")
    if args.daily_days < 0:
        parser.error("--daily-days 不能小于 0")

    if args.build:
        build_index(
            skip_existing=True,
            limit=args.limit,
            llm_workers=args.llm_workers,
            llm_interval=args.llm_interval,
            rebuild_all=args.rebuild_all,
        )
    else:
        search(args.search, daily_days=args.daily_days)


if __name__ == "__main__":
    main()
