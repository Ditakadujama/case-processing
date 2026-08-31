"""
病历相似度检索系统 - 主程序入口。

当前版本只使用医生关注文本做天级序列相似：
- --build:  从 medical_records 原始日行增量构建 record_days / record_day_case_cards
- --search: 从 Excel 导入 query_records，按天级序列检索相似患者
"""

import logging
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case_card_embedding import build_case_card_embedding_text
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
from day_store import (
    DEFAULT_DAY_EXTRACTOR_VERSION,
    DayCardWriteRow,
    MySQLDayStore,
    ProcessingFailure,
    determine_required_action,
)
from embedding_index import EmbeddingService
from llm_case_extractor import LLMCaseExtractor
from query_loader import generate_request_id, load_xlsx_daily_rows
from record_fingerprint import build_embedding_input_hash


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
                rebuild_all: bool = False,
                progress_every: int = 1) -> None:
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
        print("已清空 record_days、record_day_case_cards 和 record_day_processing_jobs")
    else:
        print(f"增量 build：当前 record_days={day_store.count_days()}，"
              f"record_day_case_cards={day_store.count_day_cards()}")

    rows = load_medical_daily_rows(cfg)
    if limit > 0:
        selected_patient_ids = []
        seen_patient_ids = set()
        for row in rows:
            patient_id = str(row.get("patient_id") or "").strip()
            if not patient_id or patient_id in seen_patient_ids:
                continue
            seen_patient_ids.add(patient_id)
            selected_patient_ids.append(patient_id)
            if len(selected_patient_ids) >= limit:
                break
        selected_patient_set = set(selected_patient_ids)
        rows = [row for row in rows if str(row.get("patient_id") or "").strip() in selected_patient_set]
        print(f"限制处理前 {limit} 个患者，共 {len(rows)} 条原始日记录")
    if not rows:
        print("medical_records 中没有原始记录，请先迁移数据")
        return

    llm_extractor = LLMCaseExtractor(llm_cfg)
    emb_service = EmbeddingService(emb_cfg)
    day_records = build_clinical_day_records(
        rows,
        history_mode=llm_cfg.history_mode,
        history_window_days=llm_cfg.history_window_days,
        history_max_chars=llm_cfg.history_max_chars,
    )
    actions_by_id = {day.day_record_id: "extract_and_embed" for day in day_records}
    if skip_existing and not rebuild_all:
        all_ids = [day.day_record_id for day in day_records]
        snapshots = day_store.load_processing_snapshots(all_ids)
        existing_cards = day_store.load_case_cards_by_ids(all_ids)
        embedding_model = emb_cfg.model
        emb_dim = emb_service.dimension  # may be None on first run

        filtered: list[tuple[ClinicalDayRecord, str]] = []
        skipped_count = 0
        for day in day_records:
            existing_card = existing_cards.get(day.day_record_id)
            existing_embedding_text = (
                build_case_card_embedding_text(existing_card) if existing_card else ""
            )
            emb_hash = (
                build_embedding_input_hash(existing_embedding_text)
                if existing_embedding_text else ""
            )
            action = determine_required_action(
                day_source_hash=day.source_hash,
                day_extraction_input_hash=day.extraction_input_hash,
                snapshot=snapshots.get(day.day_record_id),
                extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
                embedding_model=embedding_model,
                embedding_input_hash=emb_hash,
                expected_dimension=emb_dim,
            )
            if action == "skip":
                skipped_count += 1
            else:
                filtered.append((day, action))
                actions_by_id[day.day_record_id] = action
        day_records = [d for d, _ in filtered]
        if skipped_count:
            print(f"跳过已存在同版本天级索引: {skipped_count} 天 "
                  f"(病例卡+embedding均有效且输入未变)")
        embed_only_days = [(d, a) for d, a in filtered if a == "embed_only"]
        if embed_only_days:
            print(f"仅需重做 embedding: {len(embed_only_days)} 天")

    if not day_records:
        print("没有需要新增 build 的天级记录。")
        return

    print(f"准备构建 {len(day_records)} 条天级记录，来自 {len(set(d.patient_id for d in day_records))} 个患者")

    batch_size = 200
    build_started = time.time()
    total_records = len(day_records)
    total_batches = (total_records + batch_size - 1) // batch_size
    for start in range(0, len(day_records), batch_size):
        batch = day_records[start:start + batch_size]
        batch_number = start // batch_size + 1
        batch_end = min(start + len(batch), total_records)
        print(
            f"\n[批次 {batch_number}/{total_batches}] 开始处理 "
            f"{start + 1}-{batch_end}/{total_records}",
            flush=True,
        )
        _insert_day_texts(day_store, batch)
        _extract_day_case_cards_batch(
            batch,
            llm_extractor,
            emb_service,
            day_store,
            actions={day.day_record_id: actions_by_id[day.day_record_id] for day in batch},
            llm_workers=llm_workers,
            llm_interval=llm_interval,
            progress_every=progress_every,
        )
        done = min(start + batch_size, len(day_records))
        elapsed = max(time.time() - build_started, 0.001)
        rate_per_minute = done / elapsed * 60
        remaining = max(total_records - done, 0)
        eta_seconds = remaining / (done / elapsed) if done else 0
        print(
            f"  总进度 {_progress_bar(done, total_records)} "
            f"{done}/{total_records} | 平均 {rate_per_minute:.1f} 条/分钟 | "
            f"预计剩余 {_format_duration(eta_seconds)}",
            flush=True,
        )

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
            None,
            day.source_table,
            day.source_record_id,
            day.source_hash,
            day.extraction_input_hash,
        )
        for day in day_records
    ]
    day_store.insert_day_rows(rows)


def _extract_day_case_cards_batch(day_records: list[ClinicalDayRecord],
                                  llm_extractor: LLMCaseExtractor,
                                  emb_service: EmbeddingService,
                                  day_store: MySQLDayStore,
                                  actions: dict[str, str],
                                  llm_workers: int = 5,
                                  llm_interval: float = 1.0,
                                  progress_every: int = 1) -> None:
    """抽取病例卡后批量生成 embedding，并批量事务写入。"""
    if not day_records:
        return

    embedding_model = emb_service.config.model if emb_service else ""
    stats = {"success": 0, "fail": 0, "embed_only": 0}
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

    def extract_one(day: ClinicalDayRecord) -> tuple[ClinicalDayRecord, dict | None, str]:
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
                return day, None, "LLM 返回空病例卡"
            return day, card, ""
        except Exception as e:
            return day, None, str(e)

    extract_days = [
        day for day in day_records
        if actions.get(day.day_record_id, "extract_and_embed") != "embed_only"
    ]
    embed_only_days = [
        day for day in day_records
        if actions.get(day.day_record_id) == "embed_only"
    ]

    cards_by_id: dict[str, dict] = {}
    failures: list[ProcessingFailure] = []

    if extract_days:
        llm_started = time.time()
        llm_completed = 0
        llm_succeeded = 0
        llm_failed = 0
        print(
            f"  LLM 抽取开始：{len(extract_days)} 条，workers={max(1, llm_workers)}，"
            f"请求间隔={min_interval:g}s",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max(1, llm_workers)) as executor:
            futures = {executor.submit(extract_one, day): day for day in extract_days}
            for future in as_completed(futures):
                day, card, error = future.result()
                llm_completed += 1
                if card is not None:
                    llm_succeeded += 1
                    cards_by_id[day.day_record_id] = card
                else:
                    llm_failed += 1
                    logger.warning("[%s] 天级病例卡抽取失败: %s", day.day_record_id, error)
                    failures.append(ProcessingFailure(
                        day_record_id=day.day_record_id,
                        patient_id=day.patient_id,
                        failed_stage="extracting",
                        error=error,
                        source_hash=day.source_hash,
                        extraction_input_hash=day.extraction_input_hash,
                        extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
                        embedding_model=embedding_model,
                    ))
                if (
                    llm_completed == 1
                    or llm_completed == len(extract_days)
                    or llm_completed % max(1, progress_every) == 0
                ):
                    elapsed = max(time.time() - llm_started, 0.001)
                    speed = llm_completed / elapsed * 60
                    remaining = len(extract_days) - llm_completed
                    eta_seconds = remaining / (llm_completed / elapsed)
                    _print_progress(
                        "LLM",
                        llm_completed,
                        len(extract_days),
                        succeeded=llm_succeeded,
                        failed=llm_failed,
                        speed=speed,
                        elapsed=elapsed,
                        eta=eta_seconds,
                    )

    if embed_only_days:
        existing = day_store.load_case_cards_by_ids(
            [day.day_record_id for day in embed_only_days]
        )
        for day in embed_only_days:
            card = existing.get(day.day_record_id)
            if card is None:
                failures.append(ProcessingFailure(
                    day_record_id=day.day_record_id,
                    patient_id=day.patient_id,
                    failed_stage="extracting",
                    error="embed_only 路径未找到已有病例卡",
                    source_hash=day.source_hash,
                    extraction_input_hash=day.extraction_input_hash,
                    extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
                    embedding_model=embedding_model,
                ))
            else:
                cards_by_id[day.day_record_id] = card
                stats["embed_only"] += 1

    day_by_id = {day.day_record_id: day for day in day_records}
    prepared: list[tuple[ClinicalDayRecord, dict, str, str]] = []
    for day_record_id, card in cards_by_id.items():
        embedding_text = build_case_card_embedding_text(card)
        if not embedding_text:
            day = day_by_id[day_record_id]
            prepared.append((day, card, "", ""))
            continue
        prepared.append((
            day_by_id[day_record_id],
            card,
            embedding_text,
            build_embedding_input_hash(embedding_text),
        ))

    embedding_texts = [item[2] for item in prepared if item[2]]
    embedding_positions = [i for i, item in enumerate(prepared) if item[2]]
    vectors_by_position: dict[int, object] = {}
    embedding_errors: dict[int, str] = {}
    if embedding_texts:
        print(f"  Embedding 开始：{len(embedding_texts)} 条", flush=True)
        embedding_started = time.time()
        batch_result = emb_service.embed_batch(embedding_texts)
        for batch_index, position in enumerate(embedding_positions):
            vector = batch_result.vectors[batch_index]
            if vector is not None:
                vectors_by_position[position] = vector
            else:
                embedding_errors[position] = batch_result.errors.get(
                    batch_index, "Embedding 返回空向量"
                )
        print(
            f"  Embedding 完成：成功 {len(vectors_by_position)}，"
            f"失败 {len(embedding_errors)}，耗时 {_format_duration(time.time() - embedding_started)}",
            flush=True,
        )

    write_rows: list[DayCardWriteRow] = []
    for position, (day, card, embedding_text, embedding_hash) in enumerate(prepared):
        vector = vectors_by_position.get(position)
        error = embedding_errors.get(position, "")
        if not embedding_text:
            error = "病例卡无法生成 embedding 输入"
        success = vector is not None
        write_rows.append(DayCardWriteRow(
            day_record_id=day.day_record_id,
            patient_id=day.patient_id,
            day_index=day.day_index,
            case_card_json=json.dumps(card, ensure_ascii=False),
            case_card_embedding=(
                vector.astype("float32").tobytes() if success else None
            ),
            source_hash=day.source_hash,
            extraction_input_hash=day.extraction_input_hash,
            embedding_input_hash=embedding_hash,
            extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
            embedding_model=embedding_model,
            embedding_dimension=(int(vector.shape[0]) if success else None),
            processing_status=("indexed" if success else "failed_retryable"),
            failed_stage=("" if success else "embedding"),
            last_error=error,
        ))
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1

    print(f"  数据库写入：{len(write_rows)} 条病例卡", flush=True)
    write_result = day_store.upsert_day_cards(write_rows)
    if write_result.fail_count:
        stats["success"] = max(0, stats["success"] - write_result.fail_count)
        stats["fail"] += write_result.fail_count
    day_store.update_processing_failures(failures)
    stats["fail"] += len(failures)

    print(
        f"  病例卡批处理: 成功 {stats['success']}, "
        f"仅重做 embedding {stats['embed_only']}, 失败 {stats['fail']}",
        flush=True,
    )


def _progress_bar(completed: int, total: int, width: int = 28) -> str:
    """构建一个固定宽度的 CLI 进度条。"""
    total = max(total, 1)
    completed = max(0, min(completed, total))
    ratio = completed / total
    filled = min(width, int(ratio * width))
    if filled >= width:
        body = "█" * width
    else:
        body = "█" * filled + "▏" + "·" * max(0, width - filled - 1)
    return f"[{body}] {ratio * 100:5.1f}%"


def _print_progress(label: str,
                    completed: int,
                    total: int,
                    succeeded: int,
                    failed: int,
                    speed: float,
                    elapsed: float,
                    eta: float) -> None:
    """在同一终端行刷新进度，完成时换行保留最终状态。"""
    line = (
        f"  {label} {_progress_bar(completed, total)} {completed}/{total} | "
        f"成功 {succeeded} 失败 {failed} | {speed:.1f} 条/分钟 | "
        f"已用 {_format_duration(elapsed)} | 剩余 {_format_duration(eta)}"
    )
    end = "\n" if completed >= total else "\r"
    # ANSI 清除整行，避免后一次较短文本留下尾部字符。
    sys.stdout.write("\r\033[2K" + line + end)
    sys.stdout.flush()


def _format_duration(seconds: float) -> str:
    """将秒数格式化为适合 CLI 进度显示的短文本。"""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}秒"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}分{seconds:02d}秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes:02d}分"


def search(search_excel: str,
           daily_days: int = 0,
           rerank_top_n: int = 10,
           rerank_interval: float = 6.5,
           candidate_pool_size: int = 30) -> None:
    """从查询 Excel 检索相似患者（直接读取 Excel，不落库）。"""
    print("=" * 60)
    print("天级相似病例检索")
    print("=" * 60)

    request_id = generate_request_id()
    print(f"请求 ID: {request_id}")

    cfg = DBConfig.from_env()

    # Stage 2: Load directly from Excel, bypassing query_records table
    try:
        query_rows = load_xlsx_daily_rows(search_excel)
    except Exception as e:
        print(f"读取查询 Excel 失败: {e}")
        return
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

    result_dir = f"data/results/{request_id}"
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
            rerank_top_n=rerank_top_n,
            rerank_interval=rerank_interval,
            candidate_pool_size=candidate_pool_size,
        )
        all_result_count += len(results)
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result['id']} 综合: {result['similarity']} "
                  f"匹配天数: {result['daily_matched_days']}/{result['daily_compare_days']} "
                  f"候选天数: {result['daily_candidate_days']}")
            if result.get("rerank"):
                print(f"      Rerank: {result.get('rerank_score')} / {result.get('rerank_decision')} "
                      f"{result.get('rerank', {}).get('reason', '')[:80]}")
            print(f"      天级对齐: {result.get('daily_match_details', [])[:5]}")

        _write_query_result_file(result_dir, request_id, search_excel,
                                 query_patient_id, query_days, results,
                                 extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION)

    print(f"\n检索结果已保存到: {result_dir}/")
    print(f"共 {len(query_days_by_patient)} 个查询患者，{all_result_count} 条结果")
    print("=" * 60)


def _write_query_result_file(result_dir: str, request_id: str, search_excel: str,
                             query_patient_id: str,
                             query_days: List[dict], results: List[dict],
                             extractor_version: str = "",
                             embedding_model: str = "",
                             reranker_model: str = "") -> None:
    from datetime import datetime

    output_file = os.path.join(result_dir, f"{safe_result_name(query_patient_id)}_结果.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"查询病例: {query_patient_id}\n")
        f.write(f"请求 ID: {request_id}\n")
        f.write(f"查询来源: {search_excel}\n")
        f.write(f"查询时间: {datetime.now().isoformat()}\n")
        f.write(f"查询天数: {len(query_days)}\n")
        if extractor_version:
            f.write(f"抽取器版本: {extractor_version}\n")
        if embedding_model:
            f.write(f"Embedding 模型: {embedding_model}\n")
        if reranker_model:
            f.write(f"Reranker 模型: {reranker_model}\n")
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
            f.write(f"诊断轴比较天数: {result.get('diagnosis_compared_days', 0)}\n")
            f.write(f"诊断轴冲突天数: {result.get('diagnosis_conflict_count', 0)}\n")
            f.write(f"诊断轴冲突日: {result.get('diagnosis_conflict_days', [])}\n")
            f.write(f"首日诊断轴冲突: {'是' if result.get('first_day_diagnosis_conflict') else '否'}\n")
            f.write(f"病因链比较天数: {result.get('etiology_chain_compared_days', 0)}\n")
            f.write(f"病因链冲突天数: {result.get('etiology_chain_conflict_count', 0)}\n")
            f.write(f"病因链冲突日: {result.get('etiology_chain_conflict_days', [])}\n")
            f.write(f"首日病因链冲突: {'是' if result.get('first_day_etiology_chain_conflict') else '否'}\n")
            if result.get("rerank"):
                rerank = result.get("rerank") or {}
                f.write(f"LLM Rerank 分数: {rerank.get('relevance_score', '-')}\n")
                f.write(f"LLM Rerank 结论: {rerank.get('decision', '-')}\n")
                f.write(f"LLM 判断最终诊断/病因一致: {'是' if rerank.get('same_final_diagnosis_or_etiology') else '否'}\n")
                f.write(f"LLM 判断病程阶段一致: {'是' if rerank.get('same_disease_stage') else '否'}\n")
                f.write(f"LLM 关键匹配: {rerank.get('key_matches', [])}\n")
                f.write(f"LLM 关键冲突: {rerank.get('key_conflicts', [])}\n")
                f.write(f"LLM 理由: {rerank.get('reason', '')}\n")
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
                        help="查询 Excel 路径；查询数据直接在内存处理，不写入 query_records")
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="兼容旧参数；当前天级 build 不再使用多进程解析")
    parser.add_argument("--skip-existing-case-cards", action="store_true", default=False,
                        help="兼容旧参数；当前默认跳过已有同版本天级索引")
    parser.add_argument("--rebuild-all", action="store_true", default=False,
                        help="清空天级索引表后重新全量 build")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制 build 处理患者数（0=不限制）；会处理这些患者的全部日记录")
    parser.add_argument("--llm-workers", type=int, default=5,
                        help="LLM 日病例卡抽取并行线程数")
    parser.add_argument("--llm-interval", type=float, default=1.0,
                        help="两次 LLM 请求的最小间隔秒数；10次/秒限速可设为 0.1")
    parser.add_argument("--progress-every", type=int, default=1,
                        help="每完成多少条 LLM 抽取刷新一次进度条（默认 1）")
    parser.add_argument("--daily-days", type=int, default=0,
                        help="天级比较天数：0=按查询病例全部已有天数比较；N>0=只比较前 N 天")
    parser.add_argument("--rerank-top-n", type=int, default=10,
                        help="LLM 二阶段精排候选数；0=关闭 reranker")
    parser.add_argument("--rerank-interval", type=float, default=6.5,
                        help="LLM rerank 请求间隔秒数；10/min 限速建议设为 6.5")
    parser.add_argument("--candidate-pool-size", type=int, default=30,
                        help="进入二阶段前的初筛候选池大小")
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
    if args.rerank_top_n < 0:
        parser.error("--rerank-top-n 不能小于 0")
    if args.rerank_interval < 0:
        parser.error("--rerank-interval 不能小于 0")
    if args.candidate_pool_size < 5:
        parser.error("--candidate-pool-size 不能小于 5")
    if args.progress_every < 1:
        parser.error("--progress-every 不能小于 1")

    if args.build:
        build_index(
            skip_existing=True,
            limit=args.limit,
            llm_workers=args.llm_workers,
            llm_interval=args.llm_interval,
            rebuild_all=args.rebuild_all,
            progress_every=args.progress_every,
        )
    else:
        search(
            args.search,
            daily_days=args.daily_days,
            rerank_top_n=args.rerank_top_n,
            rerank_interval=args.rerank_interval,
            candidate_pool_size=args.candidate_pool_size,
        )


if __name__ == "__main__":
    main()
