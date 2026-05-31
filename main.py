"""
病历相似度检索系统 - 主程序入口

两种模式:
- build:  从 MySQL medical_records 原始表解析全量病历，提取特征向量，存入 record_vectors 表
          可选 --with-llm 启用 LLM 病例卡抽取 + embedding
- search: 从 MySQL record_vectors 表加载预计算向量，检索相似病例
          自动检测病例卡数据，启用 LLM 增强融合
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DBConfig
from retrieval_system import create_system
from data_migrate.database import load_records_from_db
from timeline_parser import TimelineParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def scan_medical_records(folder: str) -> dict:
    """扫描文件夹，获取所有病历文本"""
    records = {}
    if not os.path.exists(folder):
        print(f"文件夹不存在: {folder}")
        return records
    for filename in os.listdir(folder):
        if filename.endswith('.txt') and not filename.startswith('.'):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                records[filename] = f.read()
    return records


def build_index(num_workers: int = 1,
                skip_existing: bool = False, limit: int = 0,
                llm_workers: int = 5):
    """
    build 模式：从 MySQL medical_records 原始表分批解析病历，提取特征向量，即时存入 record_vectors 表。
    强制启用 LLM 病例卡抽取 + embedding。

    适用场景：
    - 初次部署，还没有向量索引
    - 原始病历数据有更新（新增/修改），需要重建索引
    - 特征提取算法有改动，需要重新提取

    Args:
        num_workers: 并行进程数
        skip_existing: 是否跳过已有同版本病例卡的记录
        limit: 限制处理条数（0=不限制，调试用）
    """
    from vector_store import MySQLVectorStore
    from case_card_store import MySQLCaseCardStore, DEFAULT_EXTRACTOR_VERSION
    from llm_case_extractor import LLMCaseExtractor
    from embedding_index import EmbeddingService
    from config import LLMConfig, EmbeddingConfig

    print("=" * 60)
    print("构建向量索引 (含 LLM 病例卡 + Embedding)")
    print("=" * 60)

    cfg = DBConfig.from_env()

    # ── LLM / Embedding 配置检查（强制要求）──
    llm_cfg = LLMConfig()
    if not llm_cfg.is_configured:
        print("\n错误: LLM 服务未配置！")
        print("请在 .env 文件中设置 LLM_API_BASE 和 LLM_API_KEY")
        print(f"  当前 LLM_API_BASE: {llm_cfg.api_base or '(空)'}")
        print(f"  当前 LLM_API_KEY: {'(已设置)' if llm_cfg.api_key else '(空)'}")
        return
    print(f"LLM 服务: {llm_cfg.model} @ {llm_cfg.api_base}")

    emb_cfg = EmbeddingConfig()
    if not emb_cfg.is_configured:
        print("\n错误: Embedding 服务未配置！")
        print("请在 .env 文件中设置 EMBEDDING_API_BASE 和 EMBEDDING_API_KEY")
        print(f"  当前 EMBEDDING_API_BASE: {emb_cfg.api_base or '(空)'}")
        print(f"  当前 EMBEDDING_API_KEY: {'(已设置)' if emb_cfg.api_key else '(空)'}")
        return
    print(f"Embedding 服务: {emb_cfg.model} @ {emb_cfg.api_base}")

    records = load_records_from_db(cfg)

    if not records:
        print("未找到有效病历，请先运行 data_migrate/migrate_xlsx_to_mysql.py 导入原始数据")
        return

    # 限制条数
    if limit > 0:
        records = dict(list(records.items())[:limit])
        print(f"限制处理 {limit} 条记录")

    # 清空旧向量数据
    store = MySQLVectorStore(cfg)
    store.init_table()
    store.delete_all()

    # LLM + Embedding 初始化
    cc_store = MySQLCaseCardStore(cfg)
    cc_store.init_table()
    print(f"病例卡表已就绪，当前记录数: {cc_store.count()}")

    llm_extractor = LLMCaseExtractor(llm_cfg)
    emb_service = EmbeddingService(emb_cfg)

    # 分批并行处理：每 1000 条重建一次进程池，重置进程状态防止累积变慢
    system = create_system(data_dir="./data", threshold=0.45, db_config=cfg,
                          enable_llm=True)
    items = list(records.items())
    n = len(items)
    batch_size = 200
    print(f"\n从 medical_records 表读取到 {n} 个患者，开始分批处理 (batch={batch_size}, workers={num_workers})...")
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = None
        try:
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch = dict(items[batch_start:batch_end])
                # 每 1000 条重建进程池，重置进程状态
                if batch_start % 1000 == 0:
                    if executor is not None:
                        executor.shutdown(wait=True)
                    executor = ProcessPoolExecutor(max_workers=num_workers)
                    print(f"  [进程池已重置 @ {batch_start}]")
                system.add_records_batch(batch, num_workers=num_workers, executor=executor)
                print(f"  [{batch_end}/{n}] 条已入库 ({batch_end * 100 // n}%)")

                # LLM 病例卡抽取（在每条记录入库后单独执行）
                _extract_case_cards_batch(
                    batch, llm_extractor, emb_service, cc_store,
                    skip_existing, system, llm_workers=llm_workers
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
    else:
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = dict(items[batch_start:batch_end])
            system.add_records_batch(batch, num_workers=num_workers)
            print(f"  [{batch_end}/{n}] 条已入库 ({batch_end * 100 // n}%)")

            # LLM 病例卡抽取
            _extract_case_cards_batch(
                batch, llm_extractor, emb_service, cc_store,
                skip_existing, system, llm_workers=llm_workers
            )
    system.save()

    print(f"\n构建完成！record_vectors 表共 {store.count()} 条记录")
    print(f"record_case_cards 表共 {cc_store.count()} 条记录")
    print(f"  -- 其中有 embedding: {cc_store.count_with_embedding()} 条")
    stats = llm_extractor.get_stats()
    print(f"  -- LLM 调用: {stats['request_count']} 次, 失败: {stats['error_count']} 次, "
          f"总 tokens: {stats['total_tokens']}")
    emb_stats = emb_service.get_stats()
    print(f"  -- Embedding 调用: {emb_stats['request_count']} 次, 失败: {emb_stats['error_count']} 次")


def _extract_case_cards_batch(batch: dict, llm_extractor, emb_service,
                              cc_store, skip_existing: bool, system,
                              llm_workers: int = 5) -> None:
    """
    批量抽取病例卡 + embedding，写入 case_card_store。
    使用线程池并行调用 LLM（IO 密集型）。

    Args:
        batch: {record_id: text} 映射
        llm_extractor: LLMCaseExtractor 实例
        emb_service: EmbeddingService 实例（可为 None）
        cc_store: MySQLCaseCardStore 实例
        skip_existing: 是否跳过已有记录
        system: MedicalRecordSimilaritySystem 实例（更新内存缓存）
        llm_workers: LLM 并行数（默认 5）
    """
    from case_card_store import DEFAULT_EXTRACTOR_VERSION
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time

    extractor_version = DEFAULT_EXTRACTOR_VERSION
    embedding_model = emb_service.config.model if emb_service else ""
    cache_lock = threading.Lock()
    stats = {"success": 0, "skip": 0, "fail": 0}
    # 速率限制：每个请求之间至少间隔 interval 秒，避免触发 API 限流
    _next_request_time = 0.0
    _rate_lock = threading.Lock()
    _min_interval = 1.0  # 两次 LLM 请求最小间隔（秒），可根据 API 限制调整

    def _wait_rate_limit() -> None:
        """等待直到可以发送下一个请求"""
        nonlocal _next_request_time
        with _rate_lock:
            now = time.time()
            if now < _next_request_time:
                time.sleep(_next_request_time - now)
            _next_request_time = time.time() + _min_interval

    total = len(batch)
    if total == 0:
        return

    def _process_one(record_id: str, text: str) -> str:
        """处理单条记录（在子线程中执行），返回状态: success/skip/fail"""
        if skip_existing and cc_store.exists(record_id, extractor_version):
            return "skip"

        try:
            _wait_rate_limit()  # 速率限制

            card = llm_extractor.extract(text, record_id=record_id)
            if card is None:
                logger.warning(f"[{record_id}] 病例卡抽取失败")
                return "fail"

            embedding = None
            summary = card.get("summary_for_embedding", "")
            if summary and emb_service and emb_service.is_available:
                try:
                    embedding = emb_service.embed_text(summary)
                except Exception as e:
                    logger.warning(f"[{record_id}] embedding 生成失败: {e}")

            cc_store.insert(record_id, card, embedding, extractor_version, embedding_model)

            with cache_lock:
                system._case_card_cache[record_id] = card
                if embedding is not None and system._embedding_cache is not None:
                    system._embedding_ids.append(record_id)
                    system._embedding_id_to_idx[record_id] = len(system._embedding_ids) - 1
                    emb_norm = embedding.astype(np.float64)
                    norm = np.linalg.norm(emb_norm)
                    if norm > 0:
                        emb_norm = emb_norm / norm
                    if system._embedding_cache.size == 0:
                        system._embedding_cache = emb_norm.reshape(1, -1).astype(np.float32)
                    else:
                        system._embedding_cache = np.vstack([
                            system._embedding_cache, emb_norm.astype(np.float32).reshape(1, -1)
                        ])

            return "success"

        except Exception as e:
            logger.error(f"[{record_id}] 病例卡处理异常: {e}")
            return "fail"

    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = {
            executor.submit(_process_one, record_id, text): record_id
            for record_id, text in batch.items()
        }
        for future in as_completed(futures):
            try:
                status = future.result()
            except Exception:
                status = "fail"
            if status == "success":
                stats["success"] += 1
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["fail"] += 1
            done = stats["success"] + stats["skip"] + stats["fail"]
            print(f"\r  LLM 病例卡: [{done}/{total}] "
                  f"(成功 {stats['success']}, 跳过 {stats['skip']}, 失败 {stats['fail']})",
                  end="", flush=True)

    if stats["success"] > 0 or stats["fail"] > 0:
        print()  # 换行，结束进度行
        logger.info(f"  病例卡批次: 成功 {stats['success']}, 跳过 {stats['skip']}, 失败 {stats['fail']}")
    elif stats["skip"] > 0:
        print()  # 全部跳过时也换行


def search(timeline_days: int = 0, timeline_window_weight: float = 0.55):
    """
    search 模式：从 MySQL record_vectors 表加载预计算向量，检索相似病例。

    适用场景：
    - 日常使用，查询相似病例
    - 查询病例文件放在 data/records/*.txt
    - 结果输出到 data/results/ 每个查询一个文件

    自动检测病例卡数据，如果可用则启用 LLM 增强四路融合。

    Args:
        timeline_days: 病程窗口天数；0=完整住院病程
        timeline_window_weight: 窗口病程分权重（0~1）
    """
    from case_card_store import MySQLCaseCardStore

    print("=" * 60)
    print("相似病例检索")
    print("=" * 60)

    # 打印病程比较模式
    if timeline_days > 0:
        print(f"病程比较模式: 完整住院病程 + 入院后前 {timeline_days} 天窗口")
        print(f"窗口病程权重: {timeline_window_weight}")
    else:
        print("病程比较模式: 完整住院病程")

    cfg = DBConfig.from_env()

    # 自动检测是否有病例卡数据
    enable_llm = False
    try:
        cc_store = MySQLCaseCardStore(cfg)
        cc_store.init_table()
        if cc_store.count() > 0:
            enable_llm = True
            print(f"检测到病例卡数据 ({cc_store.count()} 条)，启用 LLM 增强检索")
    except Exception:
        pass

    system = create_system(data_dir="./data", threshold=0.45, db_config=cfg,
                          enable_llm=enable_llm)

    print(f"底库病例总数: {len(system.record_order)}")

    if not system.record_order:
        print("底库为空，请先运行: python main.py --mode build")
        return

    query_records = scan_medical_records("./data/records")

    if not query_records:
        print("data/records/ 下没有查询病例文件")
        return

    print(f"找到 {len(query_records)} 个查询病例文件")

    timeline_parser = TimelineParser()
    all_results = []

    for filename, query_text in query_records.items():
        print(f"\n{'='*60}")
        print(f"查询文件: {filename}")
        print(f"{'='*60}")

        query_events = timeline_parser.parse(query_text)
        query_nodes = timeline_parser.generate_standard_nodes(query_events)
        print(f"\n  查询病例时间轴节点:")
        for key in ["T0", "T3", "T6"]:
            event = query_nodes[key]
            if event:
                ts = event.timestamp.strftime("%m-%d %H:%M") if event.timestamp else "N/A"
                print(f"    {key}: [{ts}] {event.description[:50]}")

        query_id = os.path.splitext(filename)[0]
        results = system.search(query_text, top_k=5, exclude_record_ids={filename, query_id},
                                timeline_days=timeline_days,
                                timeline_window_weight=timeline_window_weight)
        print(f"\n  检索结果（Top {len(results)}）:")

        for i, r in enumerate(results, 1):
            vec_sim = r.get('vector_similarity', r['similarity'])
            tl_sim = r.get('timeline_similarity', '-')
            full_tl = r.get('full_timeline_similarity', '-')
            win_tl = r.get('window_timeline_similarity', '-')
            emb_sim = r.get('embedding_similarity', '-')
            tag_sim = r.get('tag_overlap_similarity', '-')
            base_sim = r.get('base_similarity', '-')
            bonus = r.get('ranking_bonus', 0)
            penalty = r.get('ranking_penalty', 0)
            axis_conflict = r.get('disease_axis_conflict', False)

            # 排序置信度等级
            if i == 1 and len(results) >= 2:
                top1 = r['similarity']
                top2 = results[1]['similarity']
                gap = top1 - top2
                if gap < 0.005:
                    confidence = "low"
                elif gap < 0.015:
                    confidence = "medium"
                else:
                    confidence = "high"
            else:
                confidence = "-"

            # 病程分展示
            if timeline_days > 0 and win_tl != '-':
                tl_display = f"病程: {tl_sim} (完整: {full_tl}, 前{timeline_days}天: {win_tl})"
            else:
                tl_display = f"病程: {tl_sim}"

            # 排序修正展示
            correction_parts = []
            if bonus > 0:
                correction_parts.append(f"+{bonus}")
            if penalty > 0:
                correction_parts.append(f"-{penalty}")
            correction_str = f" 修正: {', '.join(correction_parts)}" if correction_parts else ""

            conflict_str = " ⚠主题冲突" if axis_conflict else ""
            conf_str = f" 置信度: {confidence}" if confidence != "-" else ""

            print(f"\n  [{i}] {r['id']} (综合: {r['similarity']}, "
                  f"基础: {base_sim}, 向量: {vec_sim}, {tl_display}, 语义: {emb_sim}, 标签: {tag_sim}){correction_str}{conflict_str}{conf_str}")

            matched_text = r.get('full_text', '')
            if matched_text:
                matched_events = timeline_parser.parse(matched_text)
                matched_nodes = timeline_parser.generate_standard_nodes(matched_events)
                print(f"      匹配病例节点: ", end="")
                node_strs = []
                for key in ["T0", "T3", "T6"]:
                    event = matched_nodes[key]
                    if event:
                        ts = event.timestamp.strftime("%m-%d %H:%M") if event.timestamp else "N/A"
                        node_strs.append(f"{key}({ts})")
                    else:
                        node_strs.append(f"{key}(-)")
                print(" -> ".join(node_strs))

            # 显示相似原因
            reasons = r.get('similarity_reasons')
            if reasons:
                common_items = []
                if reasons.get('common_diagnoses'):
                    common_items.append(f"共同诊断: {', '.join(reasons['common_diagnoses'][:3])}")
                if reasons.get('common_interventions'):
                    common_items.append(f"共同干预: {', '.join(reasons['common_interventions'][:3])}")
                if reasons.get('common_organ_failures'):
                    common_items.append(f"共同器官问题: {', '.join(reasons['common_organ_failures'][:3])}")
                if common_items:
                    print(f"      相似原因: {'; '.join(common_items)}")
            # 显示强共同标签
            strong_tags = r.get('strong_common_tags')
            if strong_tags:
                print(f"      强共同标签: {', '.join(strong_tags[:5])}")

            all_results.append({
                'query_file': filename,
                'matched_id': r['id'],
                'similarity': r['similarity'],
                'vector_similarity': r.get('vector_similarity', r['similarity']),
                'timeline_similarity': r.get('timeline_similarity', '-'),
                'full_timeline_similarity': r.get('full_timeline_similarity', '-'),
                'window_timeline_similarity': r.get('window_timeline_similarity', '-'),
                'timeline_window_days': r.get('timeline_window_days', 0),
                'timeline_window_weight': r.get('timeline_window_weight', 0.0),
                'timeline_window_fallback': r.get('timeline_window_fallback', False),
                'text_similarity': r.get('text_similarity', '-'),
                'embedding_similarity': r.get('embedding_similarity', '-'),
                'tag_overlap_similarity': r.get('tag_overlap_similarity', '-'),
                'base_similarity': r.get('base_similarity', '-'),
                'ranking_bonus': r.get('ranking_bonus', 0),
                'ranking_penalty': r.get('ranking_penalty', 0),
                'disease_axis_similarity': r.get('disease_axis_similarity', '-'),
                'disease_axis_conflict': r.get('disease_axis_conflict', False),
                'strong_common_tags': r.get('strong_common_tags', []),
                'similarity_reasons': r.get('similarity_reasons'),
                'full_text': r['full_text']
            })

    # 按查询病例分组保存结果，每个查询一个文件
    result_dir = "data/results"
    os.makedirs(result_dir, exist_ok=True)

    for filename in query_records:
        safe_name = filename.replace('.txt', '')
        output_file = os.path.join(result_dir, f"{safe_name}_结果.txt")
        file_results = [r for r in all_results if r['query_file'] == filename]

        # 计算排序置信度（基于 Top-1/Top-2 gap）
        confidence = "-"
        if len(file_results) >= 2:
            gap = file_results[0]['similarity'] - file_results[1]['similarity']
            if gap < 0.005:
                confidence = "low"
            elif gap < 0.015:
                confidence = "medium"
            else:
                confidence = "high"

        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入查询级别摘要
            f.write(f"查询文件: {filename}\n")
            if confidence != "-":
                f.write(f"排序置信度: {confidence}")
                if confidence == "low":
                    f.write(" (Top-1/Top-2 分差过小，建议人工复核)")
                f.write("\n")
            f.write("=" * 60 + "\n\n")

            for result in file_results:
                f.write(f"匹配病例: {result['matched_id']}\n")
                f.write(f"综合相似度: {result['similarity']}\n")
                f.write(f"基础融合分: {result.get('base_similarity', '-')}\n")
                f.write(f"向量相似度: {result.get('vector_similarity', '-')}\n")
                f.write(f"病程相似度: {result.get('timeline_similarity', '-')}\n")
                f.write(f"完整病程相似度: {result.get('full_timeline_similarity', '-')}\n")
                f.write(f"指定窗口病程相似度: {result.get('window_timeline_similarity', '-')}\n")
                f.write(f"指定病程窗口天数: {result.get('timeline_window_days', 0)}\n")
                if result.get('timeline_window_days', 0) > 0:
                    f.write(f"指定窗口权重: {result.get('timeline_window_weight', 0.0)}\n")
                    fallback = result.get('timeline_window_fallback', False)
                    f.write(f"窗口病程回退: {'是（窗口事件不足导致回退到完整病程）' if fallback else '否'}\n")
                f.write(f"病例卡语义相似度: {result.get('embedding_similarity', '-')}\n")
                f.write(f"标签重叠相似度: {result.get('tag_overlap_similarity', '-')}\n")

                # v1.1 排序解释字段
                f.write(f"\n--- 排序解释 ---\n")
                axis_sim = result.get('disease_axis_similarity', '-')
                axis_conflict = result.get('disease_axis_conflict', False)
                f.write(f"临床主题相似度: {axis_sim}\n")
                f.write(f"临床主题冲突: {'是（不同主题领域）' if axis_conflict else '否'}\n")

                strong_tags = result.get('strong_common_tags', [])
                if strong_tags:
                    f.write(f"强共同标签: {', '.join(strong_tags)}\n")
                else:
                    f.write(f"强共同标签: (无)\n")

                bonus = result.get('ranking_bonus', 0)
                penalty = result.get('ranking_penalty', 0)
                f.write(f"排序奖励: +{bonus}\n")
                f.write(f"排序惩罚: -{penalty}\n")
                f.write(f"\n")

                # 相似原因
                reasons = result.get('similarity_reasons')
                if reasons:
                    f.write(f"相似原因:\n")
                    if reasons.get('common_diagnoses'):
                        f.write(f"  - 共同诊断: {', '.join(reasons['common_diagnoses'])}\n")
                    if reasons.get('common_interventions'):
                        f.write(f"  - 共同干预: {', '.join(reasons['common_interventions'])}\n")
                    if reasons.get('common_organ_failures'):
                        f.write(f"  - 共同器官功能问题: {', '.join(reasons['common_organ_failures'])}\n")
                    if reasons.get('common_complications'):
                        f.write(f"  - 共同并发症: {', '.join(reasons['common_complications'])}\n")

                    # 主要差异
                    diffs = reasons.get('diff_interventions', {})
                    q_only = diffs.get('query_only', [])
                    c_only = diffs.get('candidate_only', [])
                    if q_only or c_only:
                        f.write(f"\n主要差异:\n")
                        if q_only:
                            f.write(f"  - 查询病例特有干预: {', '.join(q_only)}\n")
                        if c_only:
                            f.write(f"  - 候选病例特有干预: {', '.join(c_only)}\n")

                f.write("-" * 80 + "\n")
                f.write(result['full_text'])
                f.write("\n" + "=" * 80 + "\n\n")

    print(f"\n检索结果已保存到: {result_dir}/")
    print(f"共 {len(query_records)} 个查询，{len(all_results)} 条结果")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='病历相似度检索系统')
    parser.add_argument('--mode', choices=['build', 'search'], default='search',
                        help='build: 从原始病历构建向量索引入库 (强制含 LLM+Embedding); search: 检索相似病例（默认）')
    parser.add_argument('--workers', '-j', type=int, default=1,
                        help='build 时并行进程数（默认1，设为0则自动使用全部CPU核心）')
    parser.add_argument('--skip-existing-case-cards', action='store_true', default=False,
                        help='跳过已有相同 extractor_version 的病例卡记录')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制 build 处理条数（0=不限制，用于调试成本控制）')
    parser.add_argument('--llm-workers', type=int, default=5,
                        help='LLM 病例卡抽取并行线程数（默认 5，IO 密集型可适当增大）')
    parser.add_argument('--timeline-days', type=int, default=0,
                        help='病程窗口天数：0=完整住院病程；N>0=额外比较入院后前N天并与完整病程分融合')
    parser.add_argument('--timeline-window-weight', type=float, default=0.55,
                        help='窗口病程分在混合病程分中的权重（0~1，默认 0.55）')
    args = parser.parse_args()

    num_workers = args.workers
    if num_workers == 0:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "病历相似度检索系统" + " " * 25 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    if args.mode == 'build':
        build_index(
            num_workers=num_workers,
            skip_existing=args.skip_existing_case_cards,
            limit=args.limit,
            llm_workers=args.llm_workers,
        )
    else:
        # 参数校验
        if args.timeline_days < 0:
            parser.error("--timeline-days 不能小于 0")
        if not 0.0 <= args.timeline_window_weight <= 1.0:
            parser.error("--timeline-window-weight 必须在 0~1 之间")
        search(timeline_days=args.timeline_days,
               timeline_window_weight=args.timeline_window_weight)


if __name__ == "__main__":
    main()
