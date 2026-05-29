<div align="center">

# SortPaper

**学术论文解析、质量评估、向量入库与语义检索流水线**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-5B21B6)](https://qdrant.tech)
[![DashScope](https://img.shields.io/badge/DashScope-Embedding%20%7C%20Rerank%20%7C%20Vision-FF6A00)](https://dashscope.aliyun.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Judge%20%7C%20Quality-4D6BFE)](https://deepseek.com)

**语言：**
[English](README.md) &nbsp;|&nbsp;
中文 &nbsp;|&nbsp;
[日本語](README.ja.md) &nbsp;|&nbsp;
[한국어](README.ko.md)

</div>

---

## 项目简介

**SortPaper** 是一个本地优先的论文处理工具。它从 PDF 中解析文本、表格和图片，统一整理为 `LayoutChunk`，再通过 LLM Judge 和论文级质量评估补充分类、摘要、上下文与可信度信息，最后写入 Qdrant 用于语义检索和 Agent 综合回答。

当前项目正在从“大文件集中式实现”逐步整理为更清晰的分层结构。核心目标不是只把 PDF 切成 chunk，而是让后续检索能够回答“哪些论文真的支持这个结论、证据来自哪里、结果是否可信”。

## 功能特性

| 功能 | 说明 |
|---|---|
| 文本解析 | 基于 PyMuPDF 的版面感知文本分块，支持双栏和阅读顺序整理 |
| 表格解析 | pdfplumber / PyMuPDF / camelot 相关策略，配合表格区域检测、后处理和质量判断 |
| 图片解析 | qwen3-vl-plus 生成图像或子图说明 |
| LLM Judge | 对 chunk 进行质量判断，过滤低价值内容并保留必要的降级结果 |
| 论文质量评估 | 分类、Map-Reduce 摘要、chunk context、产物/菌株/可信度等元数据回写 |
| Qdrant 入库 | chunk 级向量存储，支持按论文删除、重复检测和 payload 更新 |
| 语义检索 | DashScope embedding、Qdrant hybrid search、qwen3-rerank，并显示质量元数据 |
| Agent 检索 | Qwen-plus 通过工具调用执行多轮文献检索和综合回答 |
| Streamlit 界面 | 单篇解析、一键入库、批量导入、向量库管理、检索调试 |

## 架构概览

```text
PDF
 |
 v
Streamlit UI
 |
 v
Pipeline Orchestration
 |
 +--> Text Parser  --> LLM Judge --+
 +--> Table Parser --> LLM Judge --+--> Merge --> Quality Eval --> Qdrant
 +--> Image Parser --> LLM Judge --+
                                      |
                                      v
                              Search / Agent Answer
```

主要分层：

| 层级 | 关键文件 | 职责 |
|---|---|---|
| UI 层 | `app.py`, `app_ui.py`, `app_sidebar.py` | 页面入口、控件、结果展示、向量库操作 |
| 编排层 | `app_pipeline.py`, `src/graph/pipeline_graph.py` | 预览、完整流水线、一键入库、批量处理、质量分析 |
| 数据结构 | `src/parsers/layout_chunk.py` | 跨文本、表格、图片的统一 chunk 表示 |
| 解析层 | `src/parsers/*` | PDF 文本、表格、图片解析 |
| 表格模块 | `src/parsers/table/*` | 区域检测、结构提取、清洗、去重、降级、Judge 元数据 |
| Judge 层 | `src/judge/*` | chunk 质量判断、表格判断、论文级质量评估 |
| 存储层 | `src/store/qdrant_store.py`, `src/store/chunk_storage.py` | embedding、入库、检索、rerank、payload 更新 |
| Agent 层 | `src/agent/literature_agent.py` | 基于检索工具的多轮综合回答 |

## 快速开始

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 配置环境变量**

在项目目录创建 `.env`，至少配置：

```bash
DASHSCOPE_API_KEY=your_dashscope_key
DEEPSEEK_API_KEY=your_deepseek_key
```

可选配置：

```bash
SORTPAPER_EMBEDDING_PROVIDER=dashscope
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

说明：

- 默认 embedding provider 是 `dashscope`，使用 DashScope embedding，并在 Qdrant 中保留 dense + sparse hybrid search。
- `qwen3-rerank`、`qwen3-vl-plus`、`qwen-plus` 也依赖 `DASHSCOPE_API_KEY`。
- Judge 和论文质量评估依赖 `DEEPSEEK_API_KEY`。
- 不要把真实 `.env` 或 API key 提交到仓库。

**3. 启动 Qdrant**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**4. 启动界面**

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## 使用流程

1. 在侧边栏上传 PDF，或选择已有样本文献。
2. 选择运行模式：
   - 快速预览：解析文本和表格，适合调试解析效果。
   - 完整流水线：运行解析和 Judge，后续质量评估与入库可手动触发。
   - 一键入库：解析、Judge、质量评估、Qdrant 入库一次完成。
3. 在结果页查看文本块、图片、表格、PDF 重建和语义检索。
4. 在向量库管理区检查已入库论文，必要时按论文删除后重新导入。
5. 检索结果会展示质量分类、可信度、发酵相关性、产物和菌株等 metadata。

## 检索质量注意事项

语义检索只能在已经入库的证据中找答案。如果目标主论文没有导入，系统可能只能返回综述、引用段落或相邻主题论文，即使 rerank 正常也无法生成可靠答案。

排查检索不理想时，建议按这个顺序看：

1. 向量库里是否真的有目标论文。
2. 目标论文是否完成质量评估，payload 里是否有 `category`、`paper_summary`、`target_products`、`organisms` 等字段。
3. 返回结果是原始实验论文、综述，还是其他论文中的引用段落。
4. 再判断 query、hybrid search、rerank 或 UI filter 是否需要调整。

例如 LNT II 问题需要优先确认是否导入了真正的 LNT II 主论文，而不是只导入 HMO 综述或 2'-FL 论文里的引用段落。

手动检索和 Agent 检索现在默认启用增强 chunk 召回，且手动检索默认展示 10 条结果，让已评测的 top10 证据默认可见。该索引化 lexical backfill 使用 `search_text` 和低频查询词补入证据候选，不再每次全库扫描，同时保护原始召回前排锚点。不启用 rerank 时，当前 60 条 top10 评测中 `chunk_hit@10` 从 `0.4000` 提升到 `0.6000`，`nearby_chunk_hit@10` 从 `0.4000` 提升到 `0.6333`，p50 延迟从约 `561ms` 增加到 `745ms`。在 UI 默认的 qwen3-rerank 路径下，`chunk_hit@10` 达到 `0.6667`，`nearby_chunk_hit@10` 达到 `0.7000`，p50 延迟约 `1621ms`；UI 仍保留 rerank 和增强召回开关，可在泛查询或延迟敏感场景下关闭。

Agent 综合回答还会从已命中的论文中扩展上下文：先从前 5 篇命中论文中按证据分数补论文内 deeper evidence，再补邻近 chunk，但不改变 tool search 展示给模型的命中排序。在 5 条上下文预算、每篇论文最多 3 条补充证据的设置下，当前上下文评测中 `context_chunk_hit@10` 从 `0.5000` 提升到 `0.6333`，`context_nearby_hit@10` 从 `0.5667` 提升到 `0.6667`。

手动检索和 Agent 检索会在界面中显示检索诊断信息，包括检索路由、标准化 query、实体字段、上下文定位 query、命中数量、耗时，以及每条结果命中的 route/query。这些信息用于排查召回问题，不会改写已入库 payload。

## 项目结构

```text
SortPaper/
+-- app.py                         # Streamlit 主入口
+-- app_sidebar.py                 # 侧边栏和输入控制
+-- app_ui.py                      # 结果展示、检索页、向量库 UI
+-- app_pipeline.py                # 预览、流水线、一键入库、质量分析
+-- app_utils.py                   # 保存、加载、检索等应用工具
+-- app_config.py                  # 环境变量和共享配置
+-- src/
|   +-- agent/                     # LiteratureAgent
|   +-- graph/                     # LangGraph 流水线
|   +-- judge/                     # LLM Judge 和论文质量评估
|   +-- parsers/
|   |   +-- table/                 # 表格检测、提取、清洗、去重、降级
|   |   +-- layout_chunk.py        # 统一 chunk 数据结构
|   +-- store/
|       +-- qdrant_store.py        # Qdrant collection、embedding、检索、rerank
|       +-- chunk_storage.py       # 解析结果入库边界
+-- tests/                         # 单元测试
+-- data/
    +-- sample_papers/             # 示例 PDF
    +-- results/                   # 解析结果快照
```

## 测试

```bash
pytest -q
```

常用局部测试：

```bash
pytest tests/test_qdrant_point_ids.py -q
pytest tests/test_chunk_storage.py -q
pytest tests/table -q
```

## 开发备注

- `app.py` 仍是 Streamlit 主入口，但业务逻辑已经逐步移到 `app_pipeline.py`、`app_ui.py` 和 `src/*`。
- 表格解析正在模块化，重点关注 `src/parsers/table/parser.py` 与 `dedup.py`、`region_chunks.py`、`judge_metadata.py` 等辅助模块的边界。
- 入库边界已经下沉到 `src/store/chunk_storage.py`，避免 UI/编排层直接承担 Qdrant 细节。
- 重构前建议先阅读工作区根目录下的 `../docs/ONBOARDING.md`、`../docs/UNDERSTAND_CHAT_FINDINGS.md` 和 `../docs/UNDERSTAND_DIFF_REVIEW.md`。

## 查询改写与多路召回

SortPaper 现在包含实验性的查询改写与多路召回路径，用于改善证据 chunk 检索。

- 查询改写使用 DeepSeek V4 Flash，将中文或口语化问题标准化为简洁的英文科学检索表达。
- 多路召回会同时使用原始 query、标准化 query 和少量 variants，再合并去重。
- 当前融合策略会保护原始 query 的前排结果和原始尾部候选；variants 主要在跨 route 共识或命中同一锚定论文时补尾部缺口，避免把已经命中的证据挤出第一页。
- Agent 的论文内上下文补充现在会用 `normalized_query` 加产品、菌株、基因、酶、指标、别名和前 2 条 variants 构造更紧凑的 evidence 定位 query。`table` 或 `text` 证据偏好只是在已命中论文内部提供轻量排序加分，不会过滤掉另一类内容。
- 手动检索和 Agent 检索都提供显式开关；目前默认关闭。

当前 smoke20 评测显示：保护式多路召回已经不会伤害 lexical baseline，但还没有证明能稳定提升召回率，而且延迟更高。

```text
lexical baseline smoke20:
chunk_hit@10 = 0.4545
nearby_chunk_hit@10 = 0.5455
elapsed_ms_p50 = 713ms

multi-query protected smoke20:
chunk_hit@10 = 0.4545
nearby_chunk_hit@10 = 0.5455
elapsed_ms_p50 = 3357ms
```

评测命令：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_multi_query_lexical60_top10.json
```

Agent 上下文评测也可以按线上路径启用 rerank、查询改写和多路召回：

```bash
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --query-rewrite --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_query_rewrite60_ctx5.json
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --multi-query --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_multi_query60_ctx5.json
```

详细记录见 `evals/QUERY_REWRITE.md`。
