# 病历相似度检索系统

当前版本按医生反馈重构为：**医生关注文本 + 天级序列相似 + 诊断/病因轴约束**。

系统不再使用患者级全文合并比较、旧结构化向量、旧 timelineParser、检验/医嘱/监护/生化指标等噪声字段。`medical_records` 始终作为原始表保留，不被修改。

## 核心逻辑

每条 `medical_records` 原始日记录会被转换成一个天级比较单元：

- `patient_id`
- `visit_date`
- `history_illness`
- `inspection_visit`
- `operation_record` 中的首程、首次病程、术后首次病程、日常病程、病程记录、查房记录
- 诊断/病因线索短句

不进入相似度文本：

- 检验
- 医嘱
- 监护
- 生化指标
- 会诊
- 知情同意
- 出院/转入/转出整段文本
- 旧患者级全文向量和旧时间轴特征

## Build

从 `medical_records` 原始日行构建：

- `record_days`
- `record_day_case_cards`

```bash
python main.py --build
```

全量重建：

```bash
python main.py --build --rebuild-all
```

如果 LLM 后台限速是 10/min，建议：

```bash
python main.py --build --rebuild-all --llm-workers 2 --llm-interval 6.5
```

小规模测试：

```bash
python main.py --build --limit 300 --llm-workers 1 --llm-interval 7
```

## Search

查询 Excel 格式与迁移 Excel 一致，至少包含：

- `patient_id`
- `visit_date` 或 `date`

每次检索会先清空 `query_records`，再导入本次 Excel：

```bash
python main.py --search path-of-excel.xlsx
```

只比较查询病例前 N 天：

```bash
python main.py --search path-of-excel.xlsx --daily-days 7
```

结果输出到：

```text
data/results/
```

## 诊断/病因轴

医生反馈“最终确诊/病因是否一致”非常重要。当前 day card 会要求模型输出：

- `final_diagnoses`
- `primary_diagnosis_axis`
- `etiology_axis`

检索时会把病因轴作为强信号。比如同样出现心脏骤停：

- 冠心病/急性心梗
- 肺栓塞

这两类会被识别为病因轴冲突，并在天级匹配中惩罚。

## 主要文件

```text
main.py                         CLI 入口
clinical_text_filter.py         医生关注文本过滤
day_retrieval_system.py         天级序列检索
day_store.py                    record_days / record_day_case_cards 存储
llm_case_extractor.py           LLM day card 抽取
embedding_index.py              Embedding 服务
case_card.py                    病例卡辅助归一化工具
config.py                       DB / LLM / Embedding 配置
data_migrate/database.py        MySQL 连接与原始日行读取
data_migrate/migrate_xlsx_to_mysql.py  Excel 导入 medical_records
```

## 依赖

```bash
pip install numpy pandas PyMySQL
```

需要配置 MySQL、LLM 和 Embedding 环境变量，字段见 `config.py`。
