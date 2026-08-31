# 病历相似度检索系统

当前版本按医生反馈重构为：**医生关注文本 + 天级序列相似 + 诊断/病因轴约束**。

系统不再使用患者级全文合并比较、旧结构化向量、旧 timelineParser、检验/医嘱/监护/生化指标等噪声字段。`medical_records` 始终作为原始表保留，不被修改。

## 快速开始

```bash
pip install -r requirements.txt
# 如需开发/测试
pip install -r requirements-dev.txt
```

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

- `record_days` — 天级医生关注文本（含稳定 ID 和内容哈希）
- `record_day_case_cards` — 天级病例卡和 embedding
- `record_day_processing_jobs` — 处理任务状态（幂等构建）

```bash
python main.py --build
```

全量重建：

```bash
python main.py --build --rebuild-all
```

### 幂等行为

系统会基于内容哈希决定是否需要重新处理：

- **skip**：病例卡和 embedding 均有效，且输入未变
- **embed_only**：病例卡有效但 embedding 模型或输入发生变化（不重复调用 LLM）
- **extract_and_embed**：原始内容或抽取器版本发生变化（重新调用 LLM + embedding）

相同输入重复执行 `--build` 时，LLM 和 embedding 调用数均为 0。

如果 LLM 后台限速是 10/min，建议：

```bash
python main.py --build --rebuild-all --llm-workers 2 --llm-interval 6.5
```

小规模测试：

```bash
python main.py --build --limit 300 --llm-workers 1 --llm-interval 7
```

`--limit` 限制的是患者数，不是原始日记录数；例如 `--limit 200` 会处理前 200 个患者的全部日记录。

## Search

查询 Excel 格式与迁移 Excel 一致，至少包含：

- `patient_id`
- `visit_date` 或 `date`

**默认搜索路径已改为 Excel → 内存，不再使用 `query_records` 表。每次搜索自动生成唯一 `request_id`，结果输出到独立目录，支持并发搜索。**

```bash
python main.py --search path-of-excel.xlsx
```

默认会先用内存向量矩阵初筛候选（`RETRIEVAL_BACKEND=in_memory_vector`），再对候选患者执行完整医学精算。可以切回全量扫描：

```bash
export RETRIEVAL_BACKEND=legacy_full_scan
python main.py --search path-of-excel.xlsx
```

默认会先用现有天级相似度初筛候选，再对前 10 个候选做二阶段 rerank。
如果配置了独立 reranker 服务，会优先使用该服务，例如服务器上的 Qwen3-Reranker-4B：

```bash
export RERANKER_API_BASE="http://your-server:port"
export RERANKER_MODEL="qwen3-reranker-4b"
export RERANKER_ENDPOINT="/score"
python main.py --search path-of-excel.xlsx
```

如果使用项目配套的 `launch.py`，服务接口是：

- `POST /score`：返回原始顺序分数，推荐用于本项目
- `POST /rerank`：服务端先排序，项目也能按 `index` 解析
- `GET /health`：健康检查

请求格式：

```json
{
  "model": "qwen3-reranker-4b",
  "query": "...",
  "documents": ["候选1", "候选2"],
  "instruction": "判断候选重症医学病例是否与查询病例在最终诊断、病因链、疾病阶段、关键病程和关键治疗上相似。"
}
```

如果服务使用 `texts` 而不是 `documents`，系统会自动重试一次。

未配置 `RERANKER_API_BASE` 时，会退回到 LLM JSON reranker。
如果接口限速是 10/min，默认 `--rerank-interval 6.5` 更稳；如果不想用 reranker：

```bash
python main.py --search path-of-excel.xlsx --rerank-top-n 0
```

只比较查询病例前 N 天：

```bash
python main.py --search path-of-excel.xlsx --daily-days 7
```

结果输出到独立目录（每次搜索生成唯一请求 ID）：

```text
data/results/{request_id}/{query_patient_id}_结果.txt
```

## HTTP 服务

问答系统接入时应使用结构化 JSON 接口，不需要生成临时 Excel：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8011
```

服务访问校内 LLM、Embedding 和 Reranker 时强制忽略系统代理环境变量。也可在启动时
显式清除代理变量：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    uvicorn api_server:app --host 0.0.0.0 --port 8011
```

- `GET /health`：索引及数据库健康状态
- `POST /api/v1/similar-cases/search`：接收 `query_patient_id`、`daily_records`、
  `cutoff_date`、`top_k` 等参数，返回脱敏后的相似病例证据
- `strategy=similar`：真实相似病例实验组
- `strategy=low_similarity_control`：低相似度病例文本对照组

`cutoff_date` 之后的查询记录会被丢弃，查询患者也会从候选集合排除。

## 并发搜索

搜索默认路径已不再使用 `query_records` 表（不执行 `DELETE FROM query_records`），多个 `--search` 进程可以同时运行，互不干扰。

```bash
# 终端 1
python main.py --search query1.xlsx

# 终端 2（同时运行，互不影响）
python main.py --search query2.xlsx
```

## 诊断/病因轴

医生反馈“最终确诊/病因是否一致”非常重要。当前 day card 会要求模型输出：

- `final_diagnoses`
- `primary_diagnosis_axis`
- `etiology_axis`
- `etiology_chain`

检索时会把病因链作为强信号，并在初筛 TopN 后交给 reranker 做二阶段精排。比如同样出现心脏骤停：

- 冠心病/急性心梗
- 肺栓塞

这两类会被识别为病因轴冲突，并在天级匹配中惩罚。

感染/脓毒症病例会进一步比较：

- 感染来源：泌尿系、肺部、腹腔/胆道、导管相关、皮肤软组织等
- 基础诱因：结石/梗阻、术后、创伤、肿瘤、免疫抑制等
- 病理过程：脓毒症、感染性休克、呼吸衰竭、肾衰等
- 关键处理：解除梗阻、PCI、抗凝/溶栓、机械通气、CRRT 等

因此“感染性休克”只是共同表现，不能单独决定高排名；如果感染来源或基础诱因明显不一致，会降权或被宁缺毋滥过滤。

二阶段 reranker 会进一步判断或评分：

- 最终诊断/病因链是否一致
- 病程阶段是否一致
- 关键匹配点和关键冲突点
- 是否应推荐给医生，或宁缺毋滥拒绝

## 主要文件

```text
main.py                         CLI 入口
clinical_text_filter.py         医生关注文本过滤 + 动态历史上下文
day_retrieval_system.py         天级序列检索（含候选召回 + 懒加载）
day_store.py                    record_days / record_day_case_cards / processing_jobs 存储
record_fingerprint.py           统一哈希模块（SHA-256 内容指纹）
candidate_retriever.py          候选召回抽象层（全量扫描 / 内存向量矩阵）
query_loader.py                 Excel → 内存查询加载
http_client.py                  Embedding 用 httpx 客户端（连接池 + 重试）
llm_case_extractor.py           LLM day card 抽取
embedding_index.py              Embedding 服务（批量 + 严格校验）
case_card.py                    病例卡辅助归一化工具
case_card_embedding.py          病例卡 embedding 文本构造
config.py                       DB / LLM / Embedding / Reranker / 检索 / 历史 配置
data_migrate/database.py        MySQL 连接池 + 原始日行读取
data_migrate/migrate_xlsx_to_mysql.py  Excel 导入 medical_records
migrations/                     SQL 迁移脚本
tests/                          测试套件
```

## 配置项

所有配置通过环境变量或 `.env` 文件设置。关键新增配置：

```bash
# 数据库连接池
DB_POOL_SIZE=5
DB_POOL_MAX_OVERFLOW=5
DB_CONNECT_TIMEOUT=10
DB_READ_TIMEOUT=60
DB_WRITE_TIMEOUT=60
DB_BATCH_SIZE=200

# Embedding 批量
EMBEDDING_BATCH_SIZE=32
EMBEDDING_CONNECT_TIMEOUT=10
EMBEDDING_MAX_CONNECTIONS=10
EMBEDDING_MAX_RETRIES=3

# 候选召回
RETRIEVAL_BACKEND=in_memory_vector   # 或 legacy_full_scan
RETRIEVAL_PATIENT_CANDIDATES=200
RETRIEVAL_DAY_CANDIDATES_PER_QUERY_DAY=300
RETRIEVAL_FALLBACK_TO_FULL_SCAN=1

# 历史上下文
LLM_HISTORY_MODE=window              # 或 all
LLM_HISTORY_WINDOW_DAYS=7
LLM_HISTORY_MAX_CHARS=30000
LLM_ENABLE_THINKING=0                # 结构化抽取默认关闭推理，减少延迟和 token 消耗
```

## 数据库迁移

### 迁移步骤

1. **备份派生表：**
   ```sql
   CREATE TABLE record_days_backup AS SELECT * FROM record_days;
   CREATE TABLE record_day_case_cards_backup AS SELECT * FROM record_day_case_cards;
   ```

2. **执行迁移 001（稳定 ID 和哈希）：**
   ```bash
   mysql -u root -p medical_records < migrations/001_stable_identity_and_hashes.sql
   ```

3. **部署新代码并全量重建：**
   ```bash
   python main.py --build --rebuild-all
   ```

4. **验证后执行迁移 002（取消 cumulative_text）：**
   ```bash
   mysql -u root -p medical_records < migrations/002_drop_cumulative_text.sql
   ```

### 回滚

**代码回滚（候选召回）：**
```bash
export RETRIEVAL_BACKEND=legacy_full_scan
python main.py --search query.xlsx
```

**数据回滚：**
- 恢复派生表备份；或
- 清空派生表 + 旧版本代码重新构建
- 任何情况下都不回滚或覆盖 `medical_records` 表

## 依赖

```bash
pip install numpy pandas PyMySQL
```

需要配置 MySQL、LLM 和 Embedding 环境变量，字段见 `config.py`。

## 运行测试

```bash
pytest tests/ -v
```
