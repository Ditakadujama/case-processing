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
                skip_existing: bool = False, limit: int = 0):
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
                    skip_existing, system
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
                skip_existing, system
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
                              cc_store, skip_existing: bool, system) -> None:
    """
    批量抽取病例卡 + embedding，写入 case_card_store。

    Args:
        batch: {record_id: text} 映射
        llm_extractor: LLMCaseExtractor 实例
        emb_service: EmbeddingService 实例（可为 None）
        cc_store: MySQLCaseCardStore 实例
        skip_existing: 是否跳过已有记录
        system: MedicalRecordSimilaritySystem 实例（更新内存缓存）
    """
    from case_card_store import DEFAULT_EXTRACTOR_VERSION

    extractor_version = DEFAULT_EXTRACTOR_VERSION
    embedding_model = emb_service.config.model if emb_service else ""
    success = 0
    skip = 0
    fail = 0

    for record_id, text in batch.items():
        # 跳过已有同版本病例卡
        if skip_existing and cc_store.exists(record_id, extractor_version):
            skip += 1
            continue

        try:
            # LLM 抽取
            card = llm_extractor.extract(text, record_id=record_id)
            if card is None:
                fail += 1
                logger.warning(f"[{record_id}] 病例卡抽取失败")
                continue

            # Embedding
            embedding = None
            summary = card.get("summary_for_embedding", "")
            if summary and emb_service and emb_service.is_available:
                try:
                    embedding = emb_service.embed_text(summary)
                except Exception as e:
                    logger.warning(f"[{record_id}] embedding 生成失败: {e}")

            # 写入 MySQL
            cc_store.insert(record_id, card, embedding, extractor_version, embedding_model)

            # 更新内存缓存（供后续搜索使用）
            system._case_card_cache[record_id] = card
            if embedding is not None and system._embedding_cache is not None:
                # 简单追加（线程不安全但在单线程 build 中 OK）
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

            success += 1

        except Exception as e:
            fail += 1
            logger.error(f"[{record_id}] 病例卡处理异常: {e}")

    if success > 0 or fail > 0:
        logger.info(f"  病例卡批次: 成功 {success}, 跳过 {skip}, 失败 {fail}")


def search():
    """
    search 模式：从 MySQL record_vectors 表加载预计算向量，检索相似病例。

    适用场景：
    - 日常使用，查询相似病例
    - 查询病例文件放在 data/records/*.txt
    - 结果输出到 data/results/ 每个查询一个文件

    自动检测病例卡数据，如果可用则启用 LLM 增强四路融合。
    """
    from case_card_store import MySQLCaseCardStore

    print("=" * 60)
    print("相似病例检索")
    print("=" * 60)

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
        results = system.search(query_text, top_k=5, exclude_record_ids={filename, query_id})
        print(f"\n  检索结果（Top {len(results)}）:")

        for i, r in enumerate(results, 1):
            vec_sim = r.get('vector_similarity', r['similarity'])
            tl_sim = r.get('timeline_similarity', '-')
            emb_sim = r.get('embedding_similarity', '-')
            tag_sim = r.get('tag_overlap_similarity', '-')
            print(f"\n  [{i}] {r['id']} (综合: {r['similarity']}, "
                  f"向量: {vec_sim}, 病程: {tl_sim}, 摘要: {emb_sim}, 标签: {tag_sim})")

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

            all_results.append({
                'query_file': filename,
                'matched_id': r['id'],
                'similarity': r['similarity'],
                'vector_similarity': r.get('vector_similarity', r['similarity']),
                'timeline_similarity': r.get('timeline_similarity', '-'),
                'text_similarity': r.get('text_similarity', '-'),
                'embedding_similarity': r.get('embedding_similarity', '-'),
                'tag_overlap_similarity': r.get('tag_overlap_similarity', '-'),
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
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in file_results:
                f.write(f"匹配病例: {result['matched_id']}\n")
                f.write(f"综合相似度: {result['similarity']}\n")
                f.write(f"向量相似度: {result.get('vector_similarity', '-')}\n")
                f.write(f"病程相似度: {result.get('timeline_similarity', '-')}\n")
                f.write(f"病例卡语义相似度: {result.get('embedding_similarity', '-')}\n")
                f.write(f"标签重叠相似度: {result.get('tag_overlap_similarity', '-')}\n")

                # 相似原因
                reasons = result.get('similarity_reasons')
                if reasons:
                    f.write(f"\n相似原因:\n")
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
        )
    else:
        search()


if __name__ == "__main__":
    main()
