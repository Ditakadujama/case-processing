"""
病历相似度检索系统 - 主程序入口

两种模式:
- build:  从 MySQL medical_records 原始表解析全量病历，提取特征向量，存入 record_vectors 表
- search: 从 MySQL record_vectors 表加载预计算向量，检索相似病例
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_system import create_system
from data_migrate.database import DBConfig, load_records_from_db
from timeline_parser import TimelineParser


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


def build_index(num_workers: int = 1):
    """
    build 模式：从 MySQL medical_records 原始表分批解析病历，提取特征向量，即时存入 record_vectors 表。

    适用场景：
    - 初次部署，还没有向量索引
    - 原始病历数据有更新（新增/修改），需要重建索引
    - 特征提取算法有改动，需要重新提取
    """
    from vector_store import MySQLVectorStore

    print("=" * 60)
    print("构建向量索引")
    print("=" * 60)

    cfg = DBConfig.from_env()
    records = load_records_from_db(cfg)

    if not records:
        print("未找到有效病历，请先运行 data_migrate/migrate_xlsx_to_mysql.py 导入原始数据")
        return

    # 清空旧向量数据
    store = MySQLVectorStore(cfg)
    store.init_table()
    store.delete_all()

    # 分批并行处理：每 1000 条重建一次进程池，重置进程状态防止累积变慢
    system = create_system(data_dir="./data", threshold=0.45, db_config=cfg)
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
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
    else:
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = dict(items[batch_start:batch_end])
            system.add_records_batch(batch, num_workers=num_workers)
            print(f"  [{batch_end}/{n}] 条已入库 ({batch_end * 100 // n}%)")
    system.save()

    print(f"\n构建完成！record_vectors 表共 {store.count()} 条记录")


def search():
    """
    search 模式：从 MySQL record_vectors 表加载预计算向量，检索相似病例。

    适用场景：
    - 日常使用，查询相似病例
    - 查询病例文件放在 data/records/*.txt
    - 结果输出到 data/检索结果.txt
    """
    print("=" * 60)
    print("相似病例检索")
    print("=" * 60)

    system = create_system(data_dir="./data", threshold=0.45)

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
            txt_sim = r.get('text_similarity', '-')
            print(f"\n  [{i}] {r['id']} (综合: {r['similarity']}, 向量: {vec_sim}, 病程: {tl_sim}, 摘要: {txt_sim})")

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

            all_results.append({
                'query_file': filename,
                'matched_id': r['id'],
                'similarity': r['similarity'],
                'vector_similarity': r.get('vector_similarity', r['similarity']),
                'timeline_similarity': r.get('timeline_similarity', '-'),
                'text_similarity': r.get('text_similarity', '-'),
                'full_text': r['full_text']
            })

    # 按查询病例分组保存结果，每个查询一个文件
    import os as _os
    result_dir = "data/results"
    _os.makedirs(result_dir, exist_ok=True)

    for filename in query_records:
        safe_name = filename.replace('.txt', '')
        output_file = _os.path.join(result_dir, f"{safe_name}_结果.txt")
        file_results = [r for r in all_results if r['query_file'] == filename]
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in file_results:
                f.write(f"匹配病例: {result['matched_id']}\n")
                f.write(f"综合相似度: {result['similarity']}\n")
                f.write(f"向量相似度: {result.get('vector_similarity', '-')}\n")
                f.write(f"病程相似度: {result.get('timeline_similarity', '-')}\n")
                f.write(f"摘要相似度: {result.get('text_similarity', '-')}\n")
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
                        help='build: 从原始病历构建向量索引入库; search: 检索相似病例（默认）')
    parser.add_argument('--workers', '-j', type=int, default=1,
                        help='build 时并行进程数（默认1，设为0则自动使用全部CPU核心）')
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
        build_index(num_workers=num_workers)
    else:
        search()


if __name__ == "__main__":
    main()
