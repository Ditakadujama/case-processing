# 病历相似度检索系统

将病历文本转换为 105 维特征向量，通过**两阶段检索**（向量粗排 + 病程精排）查找相似病例。支持 **LLM 病例卡抽取 + Embedding 语义增强**，实现四路融合检索。

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
# 8 进程并行构建
python main.py --mode build --workers 8

# 调试模式：只处理 10 条记录，不清空现有数据库
python main.py --mode build --workers 1 --limit 10

# 跳过已有相同版本的病例卡记录
python main.py --mode build --workers 8 --skip-existing-case-cards

# 调整 LLM 并行线程数（IO 密集型，可适当增大）
python main.py --mode build --workers 8 --llm-workers 10
```

**Build 模式流程：**
1. 从 `medical_records` 表读取全部患者文本
2. 清空 `record_vectors` 和 `record_case_cards` 表（调试模式 `--limit` 除外）
3. 分批处理（每批 200 条）：多进程解析 → 特征提取 → 时间轴计算 → MySQL 写入
4. 每批入库后执行 LLM 病例卡抽取 + Embedding 生成

### 检索相似病例（Search 模式，默认）

从 MySQL 加载预计算向量，对查询病例进行相似度检索。

```bash
# 基本检索
python main.py --mode search

# 启用病程窗口比较（入院后前 7 天 + 完整住院病程）
python main.py --mode search --timeline-days 7

# 调整窗口病程权重（0~1，默认 0.55）
python main.py --mode search --timeline-days 7 --timeline-window-weight 0.6
```

**查询文件**：将待查询的病历文本（`.txt` 文件）放入 `./data/records/` 目录。

**检索结果**：每个查询文件的结果写入 `./data/results/{文件名}_结果.txt`，包含：
- 综合相似度、各维度分项得分
- 排序置信度（Top-1/Top-2 分差）
- 临床主题冲突检测
- 相似原因分析（共同诊断、共同干预、共同器官问题等）
- 匹配病例完整文本

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：`build` 或 `search` | `search` |
| `--workers`, `-j` | Build 时并行进程数（0=自动） | `1` |
| `--limit` | Build 限制处理条数（0=不限制，调试用） | `0` |
| `--skip-existing-case-cards` | Build 跳过已有病例卡 | `False` |
| `--llm-workers` | LLM 抽取并行线程数 | `5` |
| `--timeline-days` | Search 病程窗口天数（0=完整病程） | `0` |
| `--timeline-window-weight` | Search 窗口病程权重（0~1） | `0.55` |

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
先运行 `python main.py --mode build` 构建向量索引。

**Q: Build 模式报 LLM 服务未配置？**
`.env` 中检查 `LLM_API_BASE` 和 `LLM_API_KEY` 是否正确设置。Build 模式强制需要 LLM + Embedding。

**Q: 如何只测试几条数据？**
```bash
python main.py --mode build --workers 1 --limit 5
```
`--limit` 模式不会清空现有数据库。

**Q: 如何评估检索效果？**
```bash
python eval_retrieval.py
```
