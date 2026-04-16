"""
病历相似度检索系统 - 主程序入口

功能演示:
1. 自动扫描文件夹导入病历
2. 入库病例
3. 检索相似病例
"""

import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_system import create_system, MedicalRecordSimilaritySystem
from record_parser import MedicalRecordParser


def load_xlsx_records(filepath: str) -> dict:
    """
    读取 Excel，按 patient_id 分组合并记录
    保留所有列信息

    Args:
        filepath: Excel 文件路径

    Returns:
        {patient_id: 合并后的病例文本}
    """
    import pandas as pd

    print(f"读取 Excel 文件: {filepath}")
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    print(f"总记录数: {len(df)} 行")

    # 所有需要合并的列
    columns_to_merge = [
        'patient_info', 'chief_complaint', 'instrument_test',
        'checkout', 'examine', 'doctor_advice',
        'inspection_visit', 'history_illness',
        'surgery_record', 'monitor', 'operation_record'
    ]

    records = {}
    grouped = df.groupby('patient_id')

    print(f"患者数量: {len(grouped)}")

    for patient_id, group in grouped:
        group = group.sort_values('date')
        total = len(group)
        combined_texts = []

        for i, (_, row) in enumerate(group.iterrows(), 1):
            date = row['date']
            parts = [f"###就诊记录 {i}/{total} - {date}"]

            for col in columns_to_merge:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    parts.append(f"###{col}: {val}")

            combined_texts.append('\n'.join(parts))

        records[patient_id] = '\n\n'.join(combined_texts)

    print(f"合并后病例数: {len(records)}")
    return records


def scan_medical_records(folder: str) -> dict:
    """
    扫描文件夹，获取所有病历文本

    Args:
        folder: 病历文件所在文件夹

    Returns:
        {filename: text}
    """
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


def demo_auto_import():
    """自动导入 records 文件夹中的所有病例"""
    print("=" * 60)
    print("自动导入演示 - 扫描文件夹导入所有病例")
    print("=" * 60)

    # 病例文件夹
    records_folder = "./data/records"

    # 扫描病例
    records = scan_medical_records(records_folder)

    if not records:
        print(f"\n文件夹 {records_folder} 中没有找到病历文件")
        print("请在 ./data/records/ 目录下放置 .txt 病历文件")
        print("\n" + "=" * 60)
        return

    print(f"\n发现 {len(records)} 个病历文件:")
    for filename in records.keys():
        print(f"  - {filename}")

    # 创建系统
    system = create_system(data_dir="./data", threshold=0.5)

    print(f"\n初始底库病例数: {len(system.records)}")

    # 导入每个病例
    print("\n开始导入...")
    for filename, text in records.items():
        record_id = os.path.splitext(filename)[0]  # 去掉扩展名
        print(f"\n导入: {filename}")
        print(f"  文本长度: {len(text)} 字符")

        # 检索相似病例
        results = system.search_and_add(text, record_id=record_id, top_k=5)
        print(f"  找到 {len(results)} 个相似病例")

        if results:
            for r in results:
                print(f"    - {r['id']}: 相似度 {r['similarity']}")

    print(f"\n导入完成，共 {len(system.records)} 个病例")

    # 保存结果统计
    if system.records:
        print("\n底库病例列表:")
        for rid in system.records.keys():
            print(f"  - {rid}")

    print("\n" + "=" * 60)


def demo_search_similar():
    """从文件夹取出一个病例作为查询，演示检索功能"""
    print("=" * 60)
    print("相似病例检索演示")
    print("=" * 60)

    records_folder = "./data/records"
    records = scan_medical_records(records_folder)

    if not records:
        print(f"\n文件夹 {records_folder} 中没有找到病历文件")
        print("\n" + "=" * 60)
        return

    # 创建系统
    system = create_system(data_dir="./data", threshold=0.5)

    # 先导入所有病例到底库
    print("\n[1] 导入病例到底库...")
    for filename, text in records.items():
        record_id = os.path.splitext(filename)[0]
        system.add_record(record_id, text)

    print(f"    已导入 {len(system.records)} 个病例")

    # 取第一个病例作为查询
    print("\n[2] 检索相似病例...")
    query_name = list(records.keys())[0]
    query_text = records[query_name]

    # 使用完整文本查询（与入库时一致）
    query_for_search = query_text

    print(f"    查询文件: {query_name}")
    print(f"    查询长度: {len(query_for_search)} 字符")

    results = system.search(query_for_search, top_k=5)

    print(f"\n    找到 {len(results)} 个相似病例:")
    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}] {r['id']} (相似度: {r['similarity']})")
        print(f"{'='*60}")
        print(r['full_text'])
        print(f"{'='*60}")

    print("\n" + "=" * 60)


def demo_parse_only():
    """仅解析演示 - 不入库"""
    print("=" * 60)
    print("病历解析演示")
    print("=" * 60)

    records_folder = "./data/records"
    records = scan_medical_records(records_folder)

    if not records:
        print(f"\n文件夹 {records_folder} 中没有找到病历文件")
        print("\n" + "=" * 60)
        return

    parser = MedicalRecordParser()

    # 解析第一个病例
    filename = list(records.keys())[0]
    text = records[filename]

    print(f"\n解析文件: {filename}")
    print(f"文本长度: {len(text)} 字符")

    record = parser.parse(text)

    print(f"\n患者信息:")
    print(f"  姓名: {record.patient.name}")
    print(f"  性别: {record.patient.gender}")
    print(f"  年龄: {record.patient.age}岁")

    print(f"\n诊断 ({len(record.diagnoses)} 个):")
    for d in record.diagnoses[:5]:
        print(f"  - {d.name}")

    print(f"\n检验类别 ({len(record.lab_results)} 种):")
    for cat in list(record.lab_results.keys())[:5]:
        print(f"  - {cat}")

    print(f"\n药物 ({len(record.medications)} 种):")
    for m in record.medications[:5]:
        print(f"  - {m.name}")

    print("\n" + "=" * 60)


def demo_import_xlsx():
    """从 Excel 导入病例（仅导入，不检索）"""
    print("=" * 60)
    print("Excel 导入")
    print("=" * 60)

    xlsx_file = "patient_info (18).xlsx"
    if not os.path.exists(xlsx_file):
        print(f"Excel 文件不存在: {xlsx_file}")
        return

    # 读取并合并 Excel
    records = load_xlsx_records(xlsx_file)

    if not records:
        print("未找到有效病例")
        return

    # 创建系统
    system = create_system(data_dir="./data", threshold=0.5)

    print(f"\n开始导入 {len(records)} 个患者病例...")

    # 只导入前500个患者
    items = list(records.items())[:500]

    for i, (patient_id, text) in enumerate(items, 1):
        if i % 100 == 0:
            print(f"  已导入 {i}/{len(items)}...")

        # 仅导入，不检索
        system.add_record(patient_id, text)

    print(f"\n导入完成！")
    print(f"底库病例总数: {len(system.records)}")
    print("\n" + "=" * 60)


def demo_search():
    """检索相似病例（使用已导入的数据）"""
    print("=" * 60)
    print("相似病例检索")
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

    # 检索结果汇总
    all_results = []

    for filename, query_text in query_records.items():
        print(f"\n查询文件: {filename}")

        results = system.search(query_text, top_k=10)

        for r in results:
            all_results.append({
                'query_file': filename,
                'matched_id': r['id'],
                'similarity': r['similarity'],
                'full_text': r['full_text']
            })

        print(f"  找到 {len(results)} 个相似病例")

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
                        help='import: 导入Excel病例; search: 检索相似病例')
    args = parser.parse_args()

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "病历相似度检索系统" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    if args.mode == 'import':
        demo_import_xlsx()
    else:
        demo_search()


if __name__ == "__main__":
    main()
