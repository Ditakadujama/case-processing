"""
病历相似度检索系统 - 主程序入口

支持:
1. 从 MySQL 导入病例并生成时间轴
2. 检索相似病例并对比时间轴
"""

import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_system import create_system
from database import DBConfig, load_records_from_db
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


def print_case_timeline(text: str, record_id: str) -> None:
    """打印单个病例的时间轴概览"""
    parser = TimelineParser()
    events = parser.parse(text)

    print(f"\n  病例 [{record_id}] 时间轴概览:")
    print(f"    总事件数: {len(events)}")

    # 事件类型统计
    from collections import Counter
    type_counts = Counter(e.event_type for e in events)
    type_str = ", ".join(f"{t}:{c}" for t, c in type_counts.most_common(5))
    print(f"    事件类型: {type_str}")

    # T0-T6 节点
    nodes = parser.generate_standard_nodes(events)
    node_strs = []
    for key in ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]:
        event = nodes[key]
        if event:
            ts = event.timestamp.strftime("%m-%d %H:%M") if event.timestamp else "N/A"
            node_strs.append(f"{key}({ts})")
        else:
            node_strs.append(f"{key}(-)")
    print(f"    病程节点: {' -> '.join(node_strs)}")


def demo_import_from_db():
    """从 MySQL 数据库导入病例，并生成时间轴"""
    print("=" * 60)
    print("数据库导入 + 时间轴解析")
    print("=" * 60)

    cfg = DBConfig.from_env()
    records = load_records_from_db(cfg)

    if not records:
        print("未找到有效病例，请先用 migrate_xlsx_to_mysql.py 迁移数据")
        return

    # 创建系统
    system = create_system(data_dir="./data", threshold=0.5)

    # 只导入前500个患者
    items = list(records.items())[:500]

    print(f"\n开始导入 {len(items)} 个患者病例...")

    # 批量导入
    batch = dict(items)
    system.add_records_batch(batch)
    system.save()

    print(f"\n导入完成！底库病例总数: {len(system.records)}")

    # 为导入的病例生成时间轴
    print("\n" + "-" * 60)
    print("生成病例时间轴...")
    print("-" * 60)

    for record_id, text in list(batch.items())[:5]:
        print_case_timeline(text, record_id)

    if len(batch) > 5:
        print(f"\n  ... 还有 {len(batch) - 5} 个病例（略）")

    print("\n" + "=" * 60)


def demo_search():
    """检索相似病例，并对比时间轴"""
    print("=" * 60)
    print("相似病例检索 + 时间轴对比")
    print("=" * 60)

    # 创建系统（加载已有索引）
    system = create_system(data_dir="./data", threshold=0.7)

    print(f"底库病例总数: {len(system.records)}")

    if not system.records:
        print("底库为空，请先运行导入")
        return

    # 扫描 data/records 下的 txt 文件作为查询
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

        # 查询病例时间轴
        query_events = timeline_parser.parse(query_text)
        query_nodes = timeline_parser.generate_standard_nodes(query_events)
        print(f"\n  查询病例时间轴节点:")
        for key in ["T0", "T3", "T6"]:
            event = query_nodes[key]
            if event:
                ts = event.timestamp.strftime("%m-%d %H:%M") if event.timestamp else "N/A"
                print(f"    {key}: [{ts}] {event.description[:50]}")

        # 检索相似病例
        results = system.search(query_text, top_k=5)
        print(f"\n  检索结果（Top {len(results)}）:")

        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r['id']} (相似度: {r['similarity']})")

            # 获取匹配病例的文本并解析时间轴
            matched_text = system.records.get(r['id'], {}).get('text', '')
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
                'full_text': r['full_text']
            })

    # 保存结果到文件
    output_file = "data/检索结果.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("病历相似度检索结果\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            f.write(f"查询文件: {result['query_file']}\n")
            f.write(f"匹配病例: {result['matched_id']}\n")
            f.write(f"相似度: {result['similarity']}\n")
            f.write("-" * 80 + "\n")
            f.write(result['full_text'])
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"\n检索结果已保存到: {output_file}")
    print(f"共 {len(all_results)} 条结果")
    print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='病历相似度检索系统')
    parser.add_argument('--mode', choices=['import', 'search'], default='search',
                        help='import: 从数据库导入病例; search: 检索相似病例')
    args = parser.parse_args()

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "病历相似度检索系统" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    if args.mode == 'import':
        demo_import_from_db()
    else:
        demo_search()


if __name__ == "__main__":
    main()
