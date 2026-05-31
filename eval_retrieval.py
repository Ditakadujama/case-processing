"""
检索评测脚本
随机选取 100 个病例作为查询，逐一检索，统计评测指标

支持 --timeline-days N 进行指定窗口病程评测，统计完整/窗口病程分及回退率。
"""
import argparse
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from retrieval_system import create_system
from config import DBConfig
from vector_store import MySQLVectorStore
from record_parser import MedicalRecordParser
from timeline_parser import TimelineParser
from timeline_similarity import SURGERY_TYPE_RULES


def main():
    parser = argparse.ArgumentParser(description='病历检索评测脚本')
    parser.add_argument('--timeline-days', type=int, default=0,
                        help='病程窗口天数：0=完整住院病程；N>0=额外比较入院后前N天并与完整病程分融合')
    parser.add_argument('--timeline-window-weight', type=float, default=0.55,
                        help='窗口病程分在混合病程分中的权重（0~1，默认 0.55）')
    args = parser.parse_args()

    if args.timeline_days < 0:
        parser.error("--timeline-days 不能小于 0")
    if not 0.0 <= args.timeline_window_weight <= 1.0:
        parser.error("--timeline-window-weight 必须在 0~1 之间")

    timeline_days = args.timeline_days
    timeline_window_weight = args.timeline_window_weight

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

    if timeline_days > 0:
        print(f"病程窗口模式: 前 {timeline_days} 天 (权重 {timeline_window_weight})")

    # 自动检测 LLM 病例卡数据
    from case_card_store import MySQLCaseCardStore
    enable_llm = False
    case_cards = {}
    try:
        cc_store = MySQLCaseCardStore(cfg)
        cc_store.init_table()
        if cc_store.count() > 0:
            enable_llm = True
            case_cards = cc_store.load_all()
            print(f"检测到病例卡数据 ({cc_store.count()} 条)，启用 LLM 增强评测")
    except Exception:
        pass

    # 随机选 100 个
    random.seed(42)
    sample_size = min(100, total)
    query_ids = random.sample(all_ids, sample_size)
    print(f"随机选取 {sample_size} 个查询病例\n")

    # 初始化检索系统
    system = create_system(data_dir="./data", threshold=0.0, db_config=cfg,
                          enable_llm=enable_llm)
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
    emb_scores = []
    tag_scores = []
    final_scores = []

    # Jaccard 指标（仅在 LLM 模式下可用）
    diag_jaccards = []
    interv_jaccards = []
    organ_jaccards = []
    compl_jaccards = []

    # v1.1 新增指标
    base_scores = []
    ranking_bonuses = []
    ranking_penalties = []
    axis_conflict_count = 0
    axis_match_count = 0
    strong_tag_zero_count = 0  # tag_overlap=0 且无强共同标签的候选数

    # 窗口病程统计（仅在 --timeline-days > 0 时有效）
    window_tl_scores = []
    full_tl_scores = []
    window_fallback_count = 0

    for idx, qid in enumerate(query_ids):
        qmeta = metadata.get(qid, {})
        qtext = qmeta.get('text', '')
        if not qtext:
            continue

        q_surgery = get_surgery_specialty(qid)

        # 检索（排除自身）
        results = system.search(qtext, top_k=5, exclude_record_ids={qid},
                                timeline_days=timeline_days,
                                timeline_window_weight=timeline_window_weight)

        # 获取查询病例的病例卡（用于 Jaccard 计算）
        q_card = case_cards.get(qid)

        # 提取各层分数
        for r in results:
            vec_scores.append(r.get('vector_similarity', r['similarity']))
            tl_scores.append(r.get('timeline_similarity', 0))
            text_scores.append(r.get('text_similarity', 0))
            final_scores.append(r['similarity'])

            # 窗口病程统计
            if timeline_days > 0:
                if r.get('timeline_window_fallback'):
                    window_fallback_count += 1
                w_tl = r.get('window_timeline_similarity')
                f_tl = r.get('full_timeline_similarity')
                if isinstance(w_tl, (int, float)):
                    window_tl_scores.append(w_tl)
                if isinstance(f_tl, (int, float)):
                    full_tl_scores.append(f_tl)

            if enable_llm:
                emb_sim = r.get('embedding_similarity', '-')
                tag_sim = r.get('tag_overlap_similarity', '-')
                if isinstance(emb_sim, (int, float)):
                    emb_scores.append(emb_sim)
                if isinstance(tag_sim, (int, float)):
                    tag_scores.append(tag_sim)

                # v1.1 指标
                base_sim = r.get('base_similarity', '-')
                if isinstance(base_sim, (int, float)):
                    base_scores.append(base_sim)
                bonus = r.get('ranking_bonus', 0)
                penalty = r.get('ranking_penalty', 0)
                if isinstance(bonus, (int, float)):
                    ranking_bonuses.append(bonus)
                if isinstance(penalty, (int, float)):
                    ranking_penalties.append(penalty)
                if r.get('disease_axis_conflict'):
                    axis_conflict_count += 1
                axis_sim_val = r.get('disease_axis_similarity', 0)
                if isinstance(axis_sim_val, (int, float)) and axis_sim_val >= 1.0:
                    axis_match_count += 1
                # tag_overlap=0 且无强共同标签
                if isinstance(tag_sim, (int, float)) and tag_sim == 0.0:
                    strong = r.get('strong_common_tags', [])
                    if not strong:
                        strong_tag_zero_count += 1

                # Jaccard 指标（查询 vs 候选）
                if q_card:
                    c_card = case_cards.get(r['id'])
                    if c_card:
                        from case_card import extract_tag_sets, jaccard_similarity
                        q_diag, q_interv, q_compl, q_organ = extract_tag_sets(q_card)
                        c_diag, c_interv, c_compl, c_organ = extract_tag_sets(c_card)
                        diag_jaccards.append(jaccard_similarity(q_diag, c_diag))
                        interv_jaccards.append(jaccard_similarity(q_interv, c_interv))
                        compl_jaccards.append(jaccard_similarity(q_compl, c_compl))
                        organ_jaccards.append(jaccard_similarity(q_organ, c_organ))

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
                 'full_tl': r.get('full_timeline_similarity', '-'),
                 'window_tl': r.get('window_timeline_similarity', '-'),
                 'tl_fallback': r.get('timeline_window_fallback', False),
                 'text': r.get('text_similarity', 0),
                 'emb': r.get('embedding_similarity', '-'),
                 'tag': r.get('tag_overlap_similarity', '-'),
                 'base': r.get('base_similarity', '-'),
                 'bonus': r.get('ranking_bonus', 0),
                 'penalty': r.get('ranking_penalty', 0),
                 'axis_conflict': r.get('disease_axis_conflict', False),
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
    score_groups = [
        ("最终综合分", final_scores),
        ("向量相似度", vec_scores),
        ("病程相似度", tl_scores),
        ("摘要相似度", text_scores),
    ]
    if emb_scores:
        score_groups.append(("病例卡语义相似度", emb_scores))
    if tag_scores:
        score_groups.append(("标签重叠相似度", tag_scores))
    if base_scores:
        score_groups.append(("基础融合分 (未修正)", base_scores))

    for name, scores in score_groups:
        arr = np.array(scores)
        print(f"  {name}: 均值={arr.mean():.4f}, 中位数={np.median(arr):.4f}, "
              f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

    # 2. 各分数相关性矩阵
    print(f"\n【分数相关性矩阵】")
    all_score_arrays = {
        "向量": np.array(vec_scores),
        "病程": np.array(tl_scores),
        "摘要": np.array(text_scores),
    }
    if emb_scores:
        all_score_arrays["语义"] = np.array(emb_scores)
    if tag_scores:
        all_score_arrays["标签"] = np.array(tag_scores)

    names = list(all_score_arrays.keys())
    print(f"  {'':>6}", end="")
    for name in names:
        print(f"{name:>8}", end="")
    print()
    for n1 in names:
        print(f"  {n1:>6}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"  {'1.000':>6}", end="")
            else:
                try:
                    corr = np.corrcoef(all_score_arrays[n1], all_score_arrays[n2])[0, 1]
                    print(f"{corr:8.4f}", end="")
                except Exception:
                    print(f"  {'N/A':>6}", end="")
        print()

    # 3. Top-1 分数分布
    top1_finals = [r['top5'][0]['final'] for r in all_results if r['top5']]
    top1_vecs = [r['top5'][0]['vec'] for r in all_results if r['top5']]
    top1_tls = [r['top5'][0]['tl'] for r in all_results if r['top5']]
    print(f"\n【Top-1 分数】")
    print(f"  综合: 均值={np.mean(top1_finals):.4f}, 中位数={np.median(top1_finals):.4f}")
    print(f"  向量: 均值={np.mean(top1_vecs):.4f}, 中位数={np.median(top1_vecs):.4f}")
    print(f"  病程: 均值={np.mean(top1_tls):.4f}, 中位数={np.median(top1_tls):.4f}")
    if enable_llm:
        top1_embs = [r['top5'][0].get('emb', 0) for r in all_results if r['top5']]
        top1_tags = [r['top5'][0].get('tag', 0) for r in all_results if r['top5']]
        if top1_embs and isinstance(top1_embs[0], (int, float)):
            print(f"  语义: 均值={np.mean(top1_embs):.4f}, 中位数={np.median(top1_embs):.4f}")
        if top1_tags and isinstance(top1_tags[0], (int, float)):
            print(f"  标签: 均值={np.mean(top1_tags):.4f}, 中位数={np.median(top1_tags):.4f}")

    # 4. Top-1 与 Top-2 的分数差（区分度）
    gaps = []
    for r in all_results:
        if len(r['top5']) >= 2:
            gaps.append(r['top5'][0]['final'] - r['top5'][1]['final'])
    if gaps:
        print(f"\n【Top-1/Top-2 区分度】")
        print(f"  平均分差: {np.mean(gaps):.4f}, 中位数: {np.median(gaps):.4f}")
        small_gap = sum(1 for g in gaps if g < 0.02)
        medium_gap = sum(1 for g in gaps if g < 0.05)
        print(f"  分差 < 0.02 的查询: {small_gap}/{len(gaps)} ({small_gap*100/len(gaps):.1f}%) — Top-1 不够突出")
        print(f"  分差 < 0.05 的查询: {medium_gap}/{len(gaps)} ({medium_gap*100/len(gaps):.1f}%)")

    # 5. 手术类型命中率
    if surgery_query_count > 0:
        print(f"\n【手术类型命中率】（{surgery_query_count} 个有手术信息的查询）")
        print(f"  Top-1:  {surgery_hit_at_1}/{surgery_query_count} = {surgery_hit_at_1*100/surgery_query_count:.1f}%")
        print(f"  Top-3:  {surgery_hit_at_3}/{surgery_query_count} = {surgery_hit_at_3*100/surgery_query_count:.1f}%")
        print(f"  Top-5:  {surgery_hit_at_5}/{surgery_query_count} = {surgery_hit_at_5*100/surgery_query_count:.1f}%")

    # 6. 向量主导 vs 病程主导 vs 语义主导
    tl_led = 0   # 病程排名第一但综合不是第一 → 病程纠偏成功
    vec_led = 0  # 向量排名第一但综合不是第一
    emb_led = 0  # 语义排名第一但综合不是第一
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

        # LLM 模式下也检查语义主导
        if enable_llm and all(isinstance(t.get('emb'), (int, float)) for t in top5):
            best_by_emb = max(top5, key=lambda x: x['emb'])
            if best_by_emb['id'] == best_final['id'] and best_by_vec['id'] != best_final['id']:
                emb_led += 1

    print(f"\n【纠偏能力分析】")
    print(f"  病程反超向量夺 Top-1: {tl_led}/{len(all_results)} ({tl_led*100/len(all_results):.1f}%)")
    print(f"  向量保住 Top-1:       {vec_led}/{len(all_results)} ({vec_led*100/len(all_results):.1f}%)")
    if enable_llm:
        print(f"  语义主导 Top-1:       {emb_led}/{len(all_results)} ({emb_led*100/len(all_results):.1f}%)")

    # 7.5 v1.1 排序修正统计
    if enable_llm and base_scores:
        print(f"\n【v1.1 排序修正统计】")
        base_arr = np.array(base_scores)
        bonus_arr = np.array(ranking_bonuses) if ranking_bonuses else np.array([0])
        penalty_arr = np.array(ranking_penalties) if ranking_penalties else np.array([0])
        print(f"  基础融合分: 均值={base_arr.mean():.4f}, 中位数={np.median(base_arr):.4f}")
        print(f"  排序奖励: 均值={bonus_arr.mean():.4f}, >0 次数={int((bonus_arr > 0).sum())}")
        print(f"  排序惩罚: 均值={penalty_arr.mean():.4f}, >0 次数={int((penalty_arr > 0).sum())}")
        total_llm_results = len(final_scores)
        if total_llm_results > 0:
            print(f"  主题冲突候选: {axis_conflict_count}/{total_llm_results} ({axis_conflict_count*100/total_llm_results:.1f}%)")
            print(f"  主题匹配候选: {axis_match_count}/{total_llm_results} ({axis_match_count*100/total_llm_results:.1f}%)")
            print(f"  tag_overlap=0 且无强共同标签: {strong_tag_zero_count}/{total_llm_results} ({strong_tag_zero_count*100/total_llm_results:.1f}%)")

    # 7.6 排序置信度分布
    if enable_llm:
        print(f"\n【排序置信度分布】")
        low_conf = 0
        med_conf = 0
        high_conf = 0
        for r in all_results:
            top5 = r['top5']
            if len(top5) >= 2:
                gap = top5[0]['final'] - top5[1]['final']
                if gap < 0.005:
                    low_conf += 1
                elif gap < 0.015:
                    med_conf += 1
                else:
                    high_conf += 1
        total_queries = len(all_results)
        print(f"  low (gap<0.005):    {low_conf}/{total_queries} ({low_conf*100/total_queries:.1f}%)")
        print(f"  medium (0.005-0.015): {med_conf}/{total_queries} ({med_conf*100/total_queries:.1f}%)")
        print(f"  high (gap>=0.015):  {high_conf}/{total_queries} ({high_conf*100/total_queries:.1f}%)")

    # 7. 窗口病程统计（仅在 --timeline-days > 0 时输出）
    if timeline_days > 0:
        print(f"\n【窗口病程统计（前 {timeline_days} 天，权重 {timeline_window_weight}）】")
        total_window_results = len(window_tl_scores) + len(full_tl_scores)
        if total_window_results > 0:
            if window_tl_scores:
                w_arr = np.array(window_tl_scores)
                print(f"  窗口病程分: 均值={w_arr.mean():.4f}, 中位数={np.median(w_arr):.4f}, "
                      f"std={w_arr.std():.4f}, min={w_arr.min():.4f}, max={w_arr.max():.4f}")
            if full_tl_scores:
                f_arr = np.array(full_tl_scores)
                print(f"  完整病程分: 均值={f_arr.mean():.4f}, 中位数={np.median(f_arr):.4f}, "
                      f"std={f_arr.std():.4f}, min={f_arr.min():.4f}, max={f_arr.max():.4f}")
            # 回退率 = 回退结果数 / 总结果数
            total_candidate_results = len(final_scores)
            if total_candidate_results > 0:
                fallback_rate = window_fallback_count * 100 / total_candidate_results
                print(f"  窗口回退: {window_fallback_count}/{total_candidate_results} "
                      f"({fallback_rate:.1f}%) — 窗口事件不足 {2} 个时回退到完整病程")

    # 8. 分数阈值分析
    print(f"\n【Top-1 分数阈值分布】")
    above_07 = sum(1 for s in top1_finals if s >= 0.7)
    above_06 = sum(1 for s in top1_finals if s >= 0.6)
    above_05 = sum(1 for s in top1_finals if s >= 0.5)
    print(f"  ≥0.7: {above_07}/{len(top1_finals)} ({above_07*100/len(top1_finals):.1f}%)")
    print(f"  ≥0.6: {above_06}/{len(top1_finals)} ({above_06*100/len(top1_finals):.1f}%)")
    print(f"  ≥0.5: {above_05}/{len(top1_finals)} ({above_05*100/len(top1_finals):.1f}%)")

    # 9. LLM 标签 Jaccard 指标
    if diag_jaccards:
        print(f"\n【标签 Jaccard 指标】（查询 vs Top-5 候选）")
        for name, jaccs in [("诊断", diag_jaccards), ("关键干预", interv_jaccards),
                             ("器官功能问题", organ_jaccards), ("并发症", compl_jaccards)]:
            arr = np.array(jaccs)
            print(f"  {name} Jaccard: 均值={arr.mean():.4f}, 中位数={np.median(arr):.4f}, "
                  f"std={arr.std():.4f}, max={arr.max():.4f}")

    # 9. 部分样例展示
    print(f"\n{'='*60}")
    print("样例展示（前 5 个查询的 Top-3）")
    print(f"{'='*60}")
    for r in all_results[:5]:
        qid = r['query_id']
        q_s = r['query_surgery'] or '-'
        print(f"\n查询: {qid} (手术: {q_s})")
        for i, t in enumerate(r['top5'][:3]):
            tl_info = f"病程={t['tl']:.4f}"
            if timeline_days > 0:
                w_tl = t.get('window_tl', '-')
                f_tl = t.get('full_tl', '-')
                fb = " [回退]" if t.get('tl_fallback') else ""
                tl_info = f"病程={t['tl']:.4f} (完整={f_tl}, 窗口={w_tl}){fb}"
            if enable_llm and isinstance(t.get('emb'), (int, float)):
                base_str = f"基础={t.get('base', '-')}" if isinstance(t.get('base'), (int, float)) else ""
                bonus = t.get('bonus', 0)
                penalty = t.get('penalty', 0)
                corr_parts = []
                if bonus > 0:
                    corr_parts.append(f"+{bonus}")
                if penalty > 0:
                    corr_parts.append(f"-{penalty}")
                corr_str = f" 修正: {', '.join(corr_parts)}" if corr_parts else ""
                conflict_str = " ⚠冲突" if t.get('axis_conflict') else ""
                print(f"  [{i+1}] {t['id']} | 综合={t['final']:.4f} {base_str}{corr_str} "
                      f"向量={t['vec']:.4f} {tl_info} 语义={t['emb']:.4f} 标签={t['tag']:.4f} 手术={t['surgery'] or '-'}{conflict_str}")
            else:
                print(f"  [{i+1}] {t['id']} | 综合={t['final']:.4f} 向量={t['vec']:.4f} "
                      f"{tl_info} 摘要={t['text']:.4f} 手术={t['surgery'] or '-'}")

    print(f"\n{'='*60}")
    print("评测完成")


if __name__ == "__main__":
    main()
