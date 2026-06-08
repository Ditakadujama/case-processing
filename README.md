# 病历相似度检索系统

将病历文本转换为结构化向量、时间轴特征和 LLM 病例卡，通过**两阶段检索**（向量粗排 + 病程精排）查找相似病例。当前版本新增**天级索引**：把每个患者拆成 Day 1 / Day 2 / Day 3...，为每天建立结构化向量、时间轴特征、每日病例卡和 embedding，再逐日对齐聚合为患者相似度。

## 环境准备

### 1. 创建 Conda 环境

```bash
conda create -n sepsis python=3.10
conda activate sepsis
```

### 2. 安装依赖

```bash
pip install jieba numpy pandas scikit-learn PyMySQL
# 可选：大规模检索加速（10,000+ 条记录时自动启用）
pip install faiss-cpu
```

### 3. 配置环境变量

复制 `.env` 文件并填入实际值：

```bash
cp .env.example .env   # 如果存在模板文件
# 或直接编辑 .env
```

**.env 文件内容：**

```ini
# ── MySQL 数据库 ──────────────────────────
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=medical_records

# ── LLM 服务（OpenAI-compatible API）──────
LLM_API_BASE=https://your-llm-api.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o-mini

# ── Embedding 服务（OpenAI-compatible API）─
EMBEDDING_API_BASE=https://your-embedding-api.com/v1
EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-small
```

> **优先级**：命令行 `export` > `.env` 文件 > 代码默认值。终端 export 的值不会被 `.env` 覆盖。

### 4. 初始化 MySQL 数据库

确保 MySQL 服务正在运行，然后初始化数据库表：

```bash
cd data_migrate
python migrate_xlsx_to_mysql.py --init-only
```

## 数据导入

将 Excel 病历数据导入 MySQL：

```bash
cd data_migrate
python migrate_xlsx_to_mysql.py --xlsx "patient_info (18).xlsx"
```

导入后，原始病历存储在 `medical_records` 表中。

## 使用方法

### 构建向量索引（Build 模式）

从 MySQL 原始病历数据批量解析，提取特征向量，存入 `record_vectors` 表。**强制包含 LLM 病例卡抽取 + Embedding**。

```bash
# 8 进程并行增量构建（默认：只处理新病例或缺索引的病例）
python main.py --build --workers 8

# 清空患者级和天级索引后全量重建
python main.py --build --workers 8 --rebuild-all

# 调试模式：只处理 10 条记录，不清空现有数据库
python main.py --build --workers 1 --limit 10

# 跳过已有相同版本的病例卡记录
python main.py --build --workers 8 --skip-existing-case-cards

# 调整 LLM 并行线程数；如果 API 限速为 10 次/秒，可把请求间隔设为 0.1 秒
python main.py --build --workers 8 --llm-workers 10 --llm-interval 0.1
```

**Build 模式流程：**
1. 从 `medical_records` 表读取全部患者文本
2. 默认增量处理：患者级向量、患者级病例卡、天级硬特征、天级病例卡都存在时跳过
3. 只有显式传入 `--rebuild-all` 时，才清空 `record_vectors`、`record_case_cards`、`record_days`、`record_day_case_cards`
4. 分批处理（每批 200 条）：多进程解析 → 特征提取 → 时间轴计算 → MySQL 写入
5. 每批入库后执行患者级 LLM 病例卡抽取 + Embedding 生成
6. 同步生成天级索引：`record_days` + `record_day_case_cards`

### 检索相似病例（Search 模式，默认）

从 MySQL 加载预计算向量，对查询病例进行相似度检索。

```bash
# 推荐：从 Excel 导入本次查询病例并检索
python main.py --search "query_cases.xlsx"

# 启用入院后前 7 天比较；如果存在天级索引，会额外启用逐日精排
python main.py --search "query_cases.xlsx" --timeline-days 7

# 只做前 7 天的天级比较；默认不传时按查询病例全部已有天数比较
python main.py --search "query_cases.xlsx" --daily-days 7

# 调整窗口病程权重（0~1，默认 0.55）
python main.py --search "query_cases.xlsx" --timeline-days 7 --timeline-window-weight 0.6

```

**查询 Excel**：格式与迁移 Excel 一致，至少包含 `patient_id` 和 `visit_date`（或 `date`）列。指定 `--search` 后，程序会在每次检索前清空 `query_records` 表，再导入本次 Excel；检索时按 `patient_id + visit_date` 合并为患者病程，并继续走天级比较逻辑。

**检索结果**：每个查询文件的结果写入 `./data/results/{文件名}_结果.txt`，包含：
- 综合相似度、各维度分项得分
- 天级患者相似度、轨迹相似度、每日对齐明细、覆盖度和候选/查询天数
- 排序置信度（Top-1/Top-2 分差）
- 临床主题冲突检测
- 相似原因分析（共同诊断、共同干预、共同器官问题等）
- 匹配病例完整文本

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--build` | 构建/增量更新底库索引 | `False` |
| `--search` | 查询 Excel 路径；每次检索前清空 `query_records` 并导入该 Excel | 空 |
| `--workers`, `-j` | Build 时并行进程数（0=自动） | `1` |
| `--limit` | Build 限制处理条数（0=不限制，调试用） | `0` |
| `--skip-existing-case-cards` | 兼容旧参数；当前 Build 默认已按患者级 + 天级索引增量跳过 | `False` |
| `--rebuild-all` | Build 时清空患者级和天级索引后全量重建；不传则默认增量 | `False` |
| `--llm-workers` | LLM 抽取并行线程数 | `5` |
| `--llm-interval` | 两次 LLM 请求的最小间隔秒数；10次/秒限速可设为 `0.1` | `1.0` |
| `--timeline-days` | Search 病程窗口天数（0=完整病程） | `0` |
| `--timeline-window-weight` | Search 窗口病程权重（0~1） | `0.55` |
| `--daily-days` | 天级比较天数：`0`=按查询病例全部已有天数比较；N>0=只比较前 N 天 | `0` |

### 天级索引

Build 模式会额外创建并维护两张天级表：

| 表 | 内容 |
|------|------|
| `record_days` | `patient_id + day_index` 的当天文本、累计文本、每日结构化向量、累计结构化向量、每日时间轴特征 |
| `record_day_case_cards` | 每日 LLM 病例卡、当天变化 embedding、截至当天累计状态 embedding |

检索时仍返回患者级 Top-K。系统先用现有患者级向量/embedding 召回候选，再对候选执行天级精排：

```text
Day Similarity =
  day_delta_embedding
  + cumulative_embedding
  + day_case_card_tag
  + day_timeline_features
  + day_structured_vector
```

同一天优先严格对齐，同时允许 `Day N` 与候选 `Day N-1 / Day N / Day N+1` 轻微错位匹配。默认按查询病例已有全部天数比较；候选病例天数可以不同，候选短于查询时会根据实际匹配天数做覆盖度修正。天级分会作为精排增强纳入患者最终分，现有患者级整体比较保留为兜底。

## 检索原理

### 两阶段流水线

```
查询文本
    │
    ▼
┌─────────────────────────────────────┐
│  阶段1: 向量粗排                     │
│  MedicalRecordParser → 结构化病历    │
│  FeatureExtractor → 105维向量        │
│  余弦相似度 → Top-K×20 候选集        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  阶段2: 多维度精排                   │
│  TimelineParser → T0-T6 病程节点     │
│  四路融合: 向量 + 病程 + 语义 + 标签  │
│  排序修正: 主题冲突检测 + 奖惩机制    │
└─────────────────────────────────────┘
    │
    ▼
  Top-K 结果（含排序解释）
```

### 105 维特征向量

| 分组 | 维度 | 权重 | 内容 |
|------|------|------|------|
| 诊断 | 39 | 1.5 | 诊断词袋模型 |
| 化验 | 33 | 0.6 | 检验指标归一化 |
| 用药 | 24 | 0.8 | 药物词袋模型 |
| 人口学 | 4 | 0.5 | 性别、年龄、体重、身高 |
| 手术 | 5 | 1.5 | 按专科分类的手术关键词计数 |

### 病程 T0-T6 节点

| 节点 | 含义 |
|------|------|
| T0 | 入院 |
| T1 | 入院 24h 内首次检验/检查 |
| T2 | 诊断变更 |
| T3 | 首次干预（手术、机械通气等） |
| T4 | 干预后 24h 内检验/生命体征 |
| T5 | 并发症事件 |
| T6 | 出院/转科（或最后事件） |

### 四路融合（LLM 增强模式）

当病例卡数据可用时，检索自动启用四路融合：

| 信号 | 权重(全) | 说明 |
|------|----------|------|
| 结构化向量 | 0.15 | 105 维特征余弦相似度 |
| 病程相似度 | 0.30 | T0-T6 节点匹配 |
| 语义嵌入 | 0.35 | LLM 抽取摘要 → Embedding 向量 |
| 标签重叠 | 0.20 | 诊断/干预/器官/并发症标签 Jaccard |

**排序修正机制（v1.1）：**
- **临床主题冲突**：不同疾病领域 → 惩罚 -0.08
- **无真实标签交集**：四类标签完全无关 → 惩罚 -0.04
- **仅有泛化标签**：有浅层交集但无强共同标签 → 惩罚 -0.02
- **精确临床主题匹配** → 奖励 +0.03

## 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `similarity_threshold` | 0.45 | 相似度阈值，低于此值不返回 |
| `top_k` | 5 | 返回结果数量 |
| `batch_size` | 200 | Build 模式批次大小 |
| FAISS 自动切换 | 10,000+ 条 | 超过时自动从 sklearn 切换到 FAISS IVF-PQ |

## 项目结构

```
.
├── main.py                     # CLI 入口（build / search）
├── retrieval_system.py         # 检索编排 + 两阶段搜索
├── record_parser.py            # 结构化病历解析
├── feature_extractor.py        # 105 维特征向量提取
├── similarity_index.py         # 向量索引（sklearn / FAISS）
├── timeline_parser.py          # 时间轴事件 → T0-T6 节点
├── timeline_similarity.py      # 5 维病程特征评分
├── vector_store.py             # MySQL 向量持久化
├── case_card.py                # 病例卡标签比对与排序修正
├── case_card_store.py          # MySQL 病例卡 + Embedding 持久化
├── llm_case_extractor.py       # LLM 病例卡抽取
├── embedding_index.py          # Embedding 服务 + 相似度计算
├── config.py                   # 统一配置（DB / LLM / Embedding）
├── eval_retrieval.py           # 检索效果评估
├── data/                       # 数据目录
│   ├── records/                # 查询病例 (.txt)
│   └── results/                # 检索结果输出
├── data_migrate/
│   ├── database.py             # MySQL 连接 + 原始记录加载
│   └── migrate_xlsx_to_mysql.py # Excel → MySQL 导入
├── tests/
│   └── test_v1_1_ranking.py    # 排序修正测试
└── .env                        # 环境变量配置
```

## 依赖项

| 包 | 版本要求 | 用途 |
|----|----------|------|
| jieba | ≥0.42.1 | 中文分词 |
| numpy | ≥1.24.0 | 数值计算 |
| pandas | ≥2.0.0 | 数据处理 |
| scikit-learn | ≥1.3.0 | 向量索引（余弦相似度） |
| PyMySQL | ≥1.1.0 | MySQL 连接 |
| faiss-cpu | 可选 | 大规模向量检索加速（≥10k 条） |

## 常见问题

**Q: 检索时提示"底库为空"？**
先运行 `python main.py --build` 构建向量索引。

**Q: Build 模式报 LLM 服务未配置？**
`.env` 中检查 `LLM_API_BASE` 和 `LLM_API_KEY` 是否正确设置。Build 模式强制需要 LLM + Embedding。

**Q: 如何只测试几条数据？**
```bash
python main.py --build --workers 1 --limit 5
```
`--limit` 模式不会清空现有数据库。

**Q: 如何评估检索效果？**
```bash
python eval_retrieval.py
```
