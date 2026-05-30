"""
检索评测脚本
随机选取 100 个病例作为查询，逐一检索，统计评测指标
"""
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from retrieval_system import create_system
from data_migrate.database import DBConfig
from vector_store import MySQLVectorStore
from record_parser import MedicalRecordParser
from timeline_parser import TimelineParser
from timeline_similarity import SURGERY_TYPE_RULES


def main():
    cfg = DBConfig.from_env()
    store = MySQLVectorStore(cfg)
    total = store.count()
    if total == 0:
        print("record_vectors 表为空，请先运行 --mode build")
        return

    # 加载全量元数据
    metadata = store.load_all_metadata()
    all_ids = list(metadata.keys())
    print(f"底库总数: {total}")

    # 随机选 100 个
    random.seed(42)
    sample_size = min(100, total)
    query_ids = random.sample(all_ids, sample_size)
    print(f"随机选取 {sample_size} 个查询病例\n")

    # 初始化检索系统
    system = create_system(data_dir="./data", threshold=0.0, db_config=cfg)  # 阈值设 0 不过滤
    parser = MedicalRecordParser()
    timeline_parser = TimelineParser()

    # keyword → specialty 映射
    keyword_to_specialty = {}
    for stype, keywords in SURGERY_TYPE_RULES:
        for kw in keywords:
            keyword_to_specialty[kw] = stype

    def get_surgery_specialty(record_id):
        """从 metadata 的 timeline_features 提取手术专科"""
        meta = metadata.get(record_id, {})
        tf = meta.get('timeline_features')
        if tf and tf.surgery_type:
            return tf.surgery_type
        return None

    # 收集结果
    all_results = []
    surgery_hit_at_1 = 0
    surgery_hit_at_3 = 0
    surgery_hit_at_5 = 0
    surgery_query_count = 0  # 查询病例中有手术信息的数量

    vec_scores = []
    tl_scores = []
    text_scores = []
    final_scores = []

    for idx, qid in enumerate(query_ids):
        qmeta = metadata.get(qid, {})
        qtext = qmeta.get('text', '')
        if not qtext:
            continue

        q_surgery = get_surgery_specialty(qid)

        # 检索（排除自身）
        results = system.search(qtext, top_k=5, exclude_record_ids={qid})

        # 提取各层分数
        for r in results:
            vec_scores.append(r.get('vector_similarity', r['similarity']))
            tl_scores.append(r.get('timeline_similarity', 0))
            text_scores.append(r.get('text_similarity', 0))
            final_scores.append(r['similarity'])

        # 手术类型命中率
        if q_surgery and q_surgery != 'other':
            surgery_query_count += 1
            top_surgeries = [get_surgery_specialty(r['id']) for r in results]
            if q_surgery in top_surgeries[:1]:
                surgery_hit_at_1 += 1
            if q_surgery in top_surgeries[:3]:
                surgery_hit_at_3 += 1
            if q_surgery in top_surgeries[:5]:
                surgery_hit_at_5 += 1

        all_results.append({
            'query_id': qid,
            'query_surgery': q_surgery,
            'top5': [
                {'id': r['id'], 'final': r['similarity'],
                 'vec': r.get('vector_similarity', r['similarity']),
                 'tl': r.get('timeline_similarity', 0),
                 'text': r.get('text_similarity', 0),
                 'surgery': get_surgery_specialty(r['id'])}
                for r in results
            ]
        })

        if (idx + 1) % 20 == 0:
            print(f"  已评测 {idx + 1}/{sample_size} ...")

    # ── 统计输出 ──
    print(f"\n{'='*60}")
    print(f"评测结果汇总 ({sample_size} 个查询病例)")
    print(f"{'='*60}")

    # 1. 分数分布
    print(f"\n【分数分布】")
    import numpy as np
    for name, scores in [("最终综合分", final_scores), ("向量相似度", vec_scores), ("病程相似度", tl_scores), ("摘要相似度", text_scores)]:
        arr = np.array(scores)
        print(f"  {name}: 均值={arr.mean():.4f}, 中位数={np.median(arr):.4f}, "
              f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

    # 2. 向量 vs 病程 相关性
    if vec_scores and tl_scores:
        corr = np.corrcoef(vec_scores, tl_scores)[0, 1]
        print(f"\n【向量-病程相关性】Pearson r = {corr:.4f}")
        if corr < 0.3:
            print("  → 向量和病程捕捉到不同的信号，两阶段互补性强")
        else:
            print("  → 向量和病程高度相关，两阶段冗余度较高")

    # 3. Top-1 分数分布
    top1_finals = [r['top5'][0]['final'] for r in all_results if r['top5']]
    top1_vecs = [r['top5'][0]['vec'] for r in all_results if r['top5']]
    top1_tls = [r['top5'][0]['tl'] for r in all_results if r['top5']]
    print(f"\n【Top-1 分数】")
    print(f"  综合: 均值={np.mean(top1_finals):.4f}, 中位数={np.median(top1_finals):.4f}")
    print(f"  向量: 均值={np.mean(top1_vecs):.4f}, 中位数={np.median(top1_vecs):.4f}")
    print(f"  病程: 均值={np.mean(top1_tls):.4f}, 中位数={np.median(top1_tls):.4f}")

    # 4. Top-1 与 Top-2 的分数差（区分度）
    gaps = []
    for r in all_results:
        if len(r['top5']) >= 2:
            gaps.append(r['top5'][0]['final'] - r['top5'][1]['final'])
    if gaps:
        print(f"\n【Top-1/Top-2 区分度】")
        print(f"  平均分差: {np.mean(gaps):.4f}, 中位数: {np.median(gaps):.4f}")
        small_gap = sum(1 for g in gaps if g < 0.02)
        print(f"  分差 < 0.02 的查询: {small_gap}/{len(gaps)} ({small_gap*100/len(gaps):.1f}%) — 这些查询的 Top-1 不够突出")

    # 5. 手术类型命中率
    if surgery_query_count > 0:
        print(f"\n【手术类型命中率】（{surgery_query_count} 个有手术信息的查询）")
        print(f"  Top-1:  {surgery_hit_at_1}/{surgery_query_count} = {surgery_hit_at_1*100/surgery_query_count:.1f}%")
        print(f"  Top-3:  {surgery_hit_at_3}/{surgery_query_count} = {surgery_hit_at_3*100/surgery_query_count:.1f}%")
        print(f"  Top-5:  {surgery_hit_at_5}/{surgery_query_count} = {surgery_hit_at_5*100/surgery_query_count:.1f}%")

    # 6. 向量主导 vs 病程主导
    vec_led = 0  # 向量排名第一但综合不是第一
    tl_led = 0   # 病程排名第一但综合不是第一
    for r in all_results:
        top5 = r['top5']
        if len(top5) < 2:
            continue
        best_by_vec = max(top5, key=lambda x: x['vec'])
        best_by_tl = max(top5, key=lambda x: x['tl'])
        best_final = top5[0]
        if best_by_vec['id'] != best_final['id'] and best_by_tl['id'] == best_final['id']:
            tl_led += 1
        elif best_by_tl['id'] != best_final['id'] and best_by_vec['id'] == best_final['id']:
            vec_led += 1
    print(f"\n【病程纠偏能力】")
    print(f"  病程反超向量夺 Top-1: {tl_led}/{len(all_results)} ({tl_led*100/len(all_results):.1f}%)")
    print(f"  向量保住 Top-1:       {vec_led}/{len(all_results)} ({vec_led*100/len(all_results):.1f}%)")

    # 7. 分数阈值分析
    above_07 = sum(1 for s in top1_finals if s >= 0.7)
    above_06 = sum(1 for s in top1_finals if s >= 0.6)
    above_05 = sum(1 for s in top1_finals if s >= 0.5)
    print(f"\n【Top-1 分数阈值分布】")
    print(f"  ≥0.7: {above_07}/{len(top1_finals)} ({above_07*100/len(top1_finals):.1f}%)")
    print(f"  ≥0.6: {above_06}/{len(top1_finals)} ({above_06*100/len(top1_finals):.1f}%)")
    print(f"  ≥0.5: {above_05}/{len(top1_finals)} ({above_05*100/len(top1_finals):.1f}%)")

    # 8. 部分样例展示
    print(f"\n{'='*60}")
    print("样例展示（前 5 个查询的 Top-3）")
    print(f"{'='*60}")
    for r in all_results[:5]:
        qid = r['query_id']
        q_s = r['query_surgery'] or '-'
        print(f"\n查询: {qid} (手术: {q_s})")
        for i, t in enumerate(r['top5'][:3]):
            print(f"  [{i+1}] {t['id']} | 综合={t['final']:.4f} 向量={t['vec']:.4f} 病程={t['tl']:.4f} 摘要={t['text']:.4f} 手术={t['surgery'] or '-'}")

    print(f"\n{'='*60}")
    print("评测完成")


if __name__ == "__main__":
    main()
