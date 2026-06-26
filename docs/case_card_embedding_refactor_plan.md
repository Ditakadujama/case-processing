# Case Card Embedding 改造方案

## 1. 改造目标

当前系统的日级检索主要使用两个 summary embedding：

```text
day_summary_for_embedding
cumulative_summary_for_embedding
```

这会带来几个问题：

- 每条日级记录最多需要 2 次 embedding 调用。
- `cumulative_summary_for_embedding` 会随着住院天数变长，信息越来越混杂。
- summary 本质上是病例卡信息的二次自然语言压缩，和病例卡字段有重复。
- 医生判断相似病例时更关注诊断、病因链、感染来源、诱因、器官状态、关键干预，而不是泛化摘要。

本次改造目标：

```text
取消 summary embedding / cumulative embedding 的核心地位
改为 case card embedding 为主
规则冲突检测为辅
每条日级记录只做 1 次 embedding
```

最终结构：

```text
原始病历
-> 医生关注文本过滤
-> LLM 抽取每日病例卡
-> 程序拼接 case_card_for_embedding
-> 生成 case_card_embedding
-> 天级相似度计算
-> 规则冲突降权
-> reranker 精排
```

## 2. 核心设计原则

### 2.1 LLM 只负责抽病例卡

LLM 不需要生成专门用于 embedding 的 summary 字段。

保留：

```text
day_summary
```

用途：

- 给人阅读。
- 给结果展示。
- 给 reranker 参考。
- 字段缺失时作为少量兜底信息。

删除或停用：

```text
day_summary_for_embedding
cumulative_summary_for_embedding
summary_for_embedding
```

这些字段不再作为主检索向量来源。

### 2.2 程序拼接 case_card_for_embedding

不要让 LLM 直接生成 `case_card_for_embedding`。

原因：

- LLM 每次生成风格可能不稳定。
- 可能加入额外解释或推测。
- 程序拼接可以保证字段顺序稳定、长度稳定、内容可控。

### 2.3 Case card embedding 为主

向量不再来自泛化摘要，而是来自病例卡核心字段。

重点表达：

- 最终诊断
- 主诊断轴
- 病因轴
- 病因链
- 感染来源/病变部位
- 基础诱因
- 病理过程
- 关键处理
- 器官状态
- 并发症
- 临床状态

### 2.4 规则归一化降级为冲突保险

规则不要再承担大量“相似度加分”的责任。

新的定位：

```text
case card embedding：判断医学语义是否相近
规则冲突检测：发现明确医学冲突时降权
reranker：对候选结果最终精排
```

保留规则的原因：

- embedding 可能把“同样危重”误判为“病因相似”。
- embedding 可能把不同感染来源都拉得比较近。
- 医生明确关心最终诊断/病因是否一致。

例如：

```text
泌尿系感染性休克 vs 肺部感染性休克
急性心梗导致心脏骤停 vs 肺栓塞导致心脏骤停
感染性休克 vs 心源性休克
创伤性重症 vs 感染性重症
```

这些情况应该被规则识别为明确冲突，并做降权。

## 3. LLM 病例卡字段调整

修改文件：

```text
llm_case_extractor.py
```

每日病例卡 JSON 建议保留：

```json
{
  "patient_id": "",
  "day_index": 1,
  "visit_date": "",
  "day_summary": "",
  "final_diagnoses": [],
  "primary_diagnosis_axis": "",
  "etiology_axis": "",
  "etiology_chain": {
    "disease_category": "",
    "direct_cause": "",
    "source_or_site": "",
    "anatomic_location": "",
    "underlying_trigger": "",
    "pathophysiology": [],
    "key_interventions": [],
    "certainty": "confirmed|suspected|unknown"
  },
  "baseline_context": [],
  "new_diagnoses": [],
  "new_interventions": [],
  "operations": [],
  "organ_status": [],
  "complications": [],
  "clinical_state": "stable|improving|worsening|critical|post_operation|organ_support|transferred|death|unknown",
  "evidence": []
}
```

删除 prompt 中类似下面的要求：

```text
summary_for_embedding 重点写当天变化
cumulative_summary_for_embedding 写截至当天状态
```

改成：

```text
day_summary 用于人工阅读，要求事实性、简洁，不得加入推测。
```

注意：

- `day_summary` 仍然保留。
- 不再要求 LLM 输出 `day_summary_for_embedding`。
- 不再要求 LLM 输出 `cumulative_summary_for_embedding`。
- 不再要求 LLM 输出 `summary_for_embedding`。

## 4. 新增病例卡向量文本生成函数

建议新增文件：

```text
case_card_embedding.py
```

或者放在现有工具模块中。

新增函数：

```python
def build_case_card_embedding_text(card: dict) -> str:
    ...
```

### 4.1 拼接字段顺序

建议固定顺序：

```text
最终诊断：...
主诊断轴：...
病因轴：...
疾病大类：...
直接病因：...
来源/部位：...
解剖位置：...
基础诱因：...
病理过程：...
关键处理：...
当天新增诊断：...
当天新增干预：...
手术/操作：...
器官状态：...
并发症：...
临床状态：...
当天摘要：...
```

### 4.2 拼接规则

- 空字段不拼接。
- list 用 `；` 或 `、` 连接。
- dict 按固定字段顺序展开。
- 不拼接 `evidence`。
- 不拼接完整原文。
- `day_summary` 放最后，避免摘要压过结构化字段。
- 最终文本可以做长度上限，例如 1500-2500 个中文字符。

### 4.3 示例

```text
最终诊断：脓毒症；感染性休克
主诊断轴：感染性休克
病因轴：泌尿系感染，输尿管结石梗阻
疾病大类：感染/脓毒症
直接病因：尿源性感染
来源/部位：泌尿系
基础诱因：输尿管结石梗阻
病理过程：脓毒性休克；急性肾损伤；呼吸衰竭
关键处理：抗感染；输尿管支架；去甲肾上腺素；机械通气
器官状态：循环衰竭；呼吸衰竭；急性肾损伤
临床状态：critical
当天摘要：患者因尿源性感染进展为感染性休克，需升压药和呼吸支持。
```

## 5. 数据库层修改

修改文件：

```text
day_store.py
```

当前表：

```text
record_day_case_cards
```

已有字段：

```text
day_delta_embedding
cumulative_embedding
```

建议新增字段：

```text
case_card_embedding BLOB
```

如果实验阶段会删除旧索引表并重建，可以直接改表结构，只保留：

```text
case_card_embedding
```

如果要兼容旧数据，可以先保留旧字段，但新代码不再写入、不再读取：

```text
day_delta_embedding
cumulative_embedding
```

需要修改：

```text
CREATE TABLE record_day_case_cards
insert_day_card(...)
load_day_cards_by_patient(...)
count_day_embeddings()
```

`count_day_embeddings()` 改为统计：

```sql
WHERE case_card_embedding IS NOT NULL
```

## 6. Build 阶段修改

修改文件：

```text
main.py
```

当前逻辑大致是：

```python
day_summary = card.get("day_summary_for_embedding") or card.get("summary_for_embedding", "")
cumulative_summary = card.get("cumulative_summary_for_embedding") or ""

if day_summary:
    day_embedding = emb_service.embed_text(day_summary)

if cumulative_summary:
    cumulative_embedding = emb_service.embed_text(cumulative_summary)
```

改成：

```python
embedding_text = build_case_card_embedding_text(card)
case_card_embedding = None

if embedding_text:
    case_card_embedding = emb_service.embed_text(embedding_text)
```

写入：

```python
day_store.insert_day_card(
    day.day_record_id,
    day.patient_id,
    day.day_index,
    card,
    case_card_embedding,
    extractor_version=DEFAULT_DAY_EXTRACTOR_VERSION,
    embedding_model=embedding_model,
)
```

目标：

```text
每条日级记录 = 1 次 LLM + 1 次 embedding
```

不再生成：

```text
day_delta_embedding
cumulative_embedding
```

## 7. 查询阶段修改

修改文件：

```text
day_retrieval_system.py
```

当前查询病例会生成：

```text
day_delta_embedding
cumulative_embedding
```

改为生成：

```text
case_card_embedding
```

伪代码：

```python
card = self._extract_query_day_card(day)
embedding_text = build_case_card_embedding_text(card)
case_card_embedding = None

if embedding_text and self.emb_service.is_available:
    case_card_embedding = _normalize_embedding(
        self.emb_service.embed_text(embedding_text)
    )
```

query day item 保存：

```python
"case_card_embedding": case_card_embedding
```

不再保存或使用：

```text
day_delta_embedding
cumulative_embedding
```

## 8. 索引加载修改

修改文件：

```text
day_retrieval_system.py
```

当前 `_load_index()` 会归一化：

```python
item["day_delta_embedding"]
item["cumulative_embedding"]
```

改为：

```python
item["case_card_embedding"] = _normalize_embedding(item.get("case_card_embedding"))
```

## 9. 相似度评分规则修改

修改函数：

```text
day_retrieval_system.py::_day_pair_similarity(...)
```

### 9.1 旧评分结构

当前大致为：

```text
etiology_chain_similarity  0.35
diagnosis_axis_similarity  0.20
day_summary_embedding      0.30
tag_jaccard                0.15
cumulative_embedding       0.10
raw_text_ngram             0.08
```

### 9.2 新评分结构

建议改为：

```text
case_card_embedding        0.45
etiology_chain_similarity  0.25
diagnosis_axis_similarity  0.15
tag_jaccard                0.08
raw_text_ngram             0.07
```

含义：

- `case_card_embedding` 是主相似度。
- `etiology_chain_similarity` 保留，但权重降低。
- `diagnosis_axis_similarity` 保留，用于主诊断轴校准。
- `tag_jaccard` 作为轻量辅助。
- `raw_text_ngram` 作为很小的原文相似度兜底。
- 不再使用 `cumulative_embedding`。

伪代码：

```python
parts = []

case_card_emb = _vector_cosine(
    query_day.get("case_card_embedding"),
    cand_day.get("case_card_embedding"),
)
if case_card_emb is not None:
    parts.append((0.45, max(0.0, case_card_emb)))

etiology_sim = _etiology_chain_similarity(
    query_day.get("case_card"),
    cand_day.get("case_card"),
)
if etiology_sim is not None:
    parts.append((0.25, etiology_sim))

diagnosis_sim = _diagnosis_axis_similarity(
    query_day.get("case_card"),
    cand_day.get("case_card"),
)
if diagnosis_sim is not None:
    parts.append((0.15, diagnosis_sim))

tag_sim = _jaccard(
    _day_card_tags(query_day.get("case_card")),
    _day_card_tags(cand_day.get("case_card")),
)
if tag_sim is not None:
    parts.append((0.08, tag_sim))

text_sim = _counter_cosine(
    query_day.get("text_vector"),
    cand_day.get("text_vector"),
)
if text_sim is not None:
    parts.append((0.07, text_sim))
```

最终仍然按已有方式归一化权重：

```python
total = sum(weight for weight, _ in parts)
score = sum(weight * score for weight, score in parts) / total
```

### 9.3 冲突降权保留

保留：

```python
if _has_diagnosis_axis_conflict(query_day.get("case_card"), cand_day.get("case_card")):
    score *= 0.70

if _has_etiology_chain_conflict(query_day.get("case_card"), cand_day.get("case_card")):
    score *= 0.55
```

规则新定位：

```text
不是主要加分器
而是明确医学冲突的保险丝
```

## 10. 规则归一化函数是否删除

不建议删除。

保留：

```text
_etiology_chain_labels(...)
_etiology_chain_similarity(...)
_has_etiology_chain_conflict(...)
_diagnosis_axis_similarity(...)
_has_diagnosis_axis_conflict(...)
```

但减少其主导性。

原因：

- case card embedding 可以解决同义表达问题。
- 规则可以解决明确冲突问题。
- 二者互补，而不是互相替代。

## 11. Reranker 逻辑

reranker 可以保持不变。

但建议确保 reranker 输入中包含：

```text
query case card
candidate case card
每日匹配明细
day_summary
final_diagnoses
primary_diagnosis_axis
etiology_axis
etiology_chain
```

reranker 的作用：

```text
在粗排候选已经比较合理的前提下，进一步判断病因/诊断/病程是否真正相似。
```

## 12. 旧库兼容与重建建议

如果当前仍在实验阶段，建议最干净的方式：

```text
medical_records 不动
删除 record_days
删除 record_day_case_cards
重新 build
```

如果不能删除旧索引表：

1. 增加 `case_card_embedding` 字段。
2. 新 build 的记录写入 `case_card_embedding`。
3. 检索时只使用有 `case_card_embedding` 的记录。
4. 旧记录需要重新 build 才能进入新逻辑。

由于 embedding 来源和评分规则都变了，旧库结果和新库结果不可直接混用。

## 13. 预期收益

### 13.1 Build 更快

旧逻辑：

```text
每条日记录：
1 次 LLM
最多 2 次 embedding
```

新逻辑：

```text
每条日记录：
1 次 LLM
1 次 embedding
```

embedding 调用数减少约一半。

同时 LLM 不再输出两个 embedding summary 字段，输出 token 也会减少。

### 13.2 单次 embedding 更轻

`case_card_for_embedding` 是程序拼接的短文本。

相比 `cumulative_summary_for_embedding`：

- 更短。
- 更稳定。
- 不随住院天数持续变长。
- 噪声更少。
- 医学重点更明确。

### 13.3 检索更贴近医生判断

新的 embedding 直接作用于病例卡核心医学字段。

系统比较的是：

```text
诊断是否相近
病因是否相近
感染来源是否相近
基础诱因是否相近
关键干预是否相近
器官功能状态是否相近
```

而不是泛化 summary。

### 13.4 规则维护压力降低

规则不再需要覆盖大量同义表达。

同义表达主要由 case card embedding 解决。

规则只保留少数明确冲突：

```text
泌尿系感染 vs 肺部感染
冠心病/急性心梗 vs 肺栓塞
感染性休克 vs 心源性休克
创伤主轴 vs 感染主轴
```

## 14. 实施顺序建议

建议按以下顺序修改：

1. 新增 `build_case_card_embedding_text(card)`。
2. 修改 `day_store.py`，支持 `case_card_embedding`。
3. 修改 `main.py` build 阶段，只生成 `case_card_embedding`。
4. 修改 `day_retrieval_system.py` 查询阶段，只生成 `case_card_embedding`。
5. 修改 `_load_index()`，加载并归一化 `case_card_embedding`。
6. 修改 `_day_pair_similarity()` 评分权重。
7. 修改 `llm_case_extractor.py` prompt，删除 embedding summary 字段。
8. 跑小规模 build，例如 `--limit-patients 20` 或类似参数。
9. 跑 2-3 个 search 样本，人工检查 Top 结果。
10. 如果结果稳定，再重建完整库。

## 15. 验证重点

测试时重点看：

- build 是否每条记录只调用 1 次 embedding。
- `record_day_case_cards.case_card_embedding` 是否写入成功。
- 查询病例是否也生成 `case_card_embedding`。
- 结果是否还保留医生关心的字段。
- 感染性休克是否能进一步区分感染来源。
- 心脏骤停是否能区分急性心梗、肺栓塞等病因。
- 若病例不够相似，reranker 是否能 reject 或降低排名。

## 16. 最终一句话

本次改造不是取消病例卡，而是让病例卡成为检索核心：

```text
LLM 抽取病例卡
程序拼接病例卡核心字段
病例卡 embedding 作为主相似度
规则只做明确医学冲突降权
reranker 做最后精排
```

这样可以减少 build 成本，降低规则维护压力，并让相似病例检索更贴近医生实际判断。
