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

    # 截取部分文本作为模拟查询
    query_for_search = query_text[:min(3000, len(query_text))]

    print(f"    查询文件: {query_name}")
    print(f"    查询长度: {len(query_for_search)} 字符")

    results = system.search(query_for_search, top_k=5)

    print(f"\n    找到 {len(results)} 个相似病例:")
    for r in results:
        print(f"      - {r['id']}: 相似度 {r['similarity']}")

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


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "病历相似度检索系统" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. 解析演示
    demo_parse_only()

    # 2. 自动导入演示
    demo_auto_import()

    # 3. 检索演示
    demo_search_similar()

    print("\n演示完成！")
    print("\n使用方法:")
    print("  1. 将病历 .txt 文件放入 ./data/records/ 目录")
    print("  2. 运行 python main.py")
    print("\n或在代码中使用:")
    print("  from retrieval_system import create_system")
    print("  system = create_system(data_dir='./data', threshold=0.7)")
    print("  records = scan_medical_records('./data/records')")
    print("  for filename, text in records.items():")
    print("      system.search_and_add(text, record_id=filename)")


if __name__ == "__main__":
    main()
