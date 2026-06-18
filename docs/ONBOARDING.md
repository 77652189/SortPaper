# PaperSort 中文阅读指南

## 项目概览

PaperSort 是一个本地优先的学术论文处理流水线。它从 PDF 论文中抽取文本、表格和图片，将内容组织成 `LayoutChunk`，用 LLM Judge 判断 chunk 质量，再把可用内容写入 Qdrant，最后提供 Streamlit 图形界面、语义检索和 Agent 综合回答。

当前代码的主要复杂度不是单一算法，而是多个业务阶段叠在一起：UI 控制、PDF 解析、表格纠错、LLM 调用、向量入库、质量富集、检索展示都在同一个应用里闭环。

## 推荐阅读顺序

1. `SortPaper/README.md`：先建立产品意图。注意仓库里的多语言 README 存在编码异常，英文 README 当前可读性最好。
2. `SortPaper/app_sidebar.py`：理解用户输入、模式选择、批量导入和向量库管理入口。
3. `SortPaper/app.py`：理解 Streamlit 主控制流，尤其是单篇处理、批量处理、入库和质量分析按钮如何触发后端函数。
4. `SortPaper/app_pipeline.py`：理解业务编排层，重点看 `run_preview`、`run_pipeline`、`store_parsed_chunks`、`evaluate_and_enrich_from_qdrant`。
5. `SortPaper/src/graph/pipeline_graph.py`：理解完整流水线的 LangGraph 节点和 fan-out/fan-in 结构。
6. `SortPaper/src/parsers/layout_chunk.py`：理解贯穿全局的数据结构 `LayoutChunk`、去重和合并规则。
7. `SortPaper/src/parsers/table/parser.py` 与 `SortPaper/src/parsers/table/cleanup.py`：理解最大复杂热点，表格区域检测、抽取、修复、评分和降级存储都与这里相关。
8. `SortPaper/src/judge/llm_judge.py`、`SortPaper/src/judge/table_judge.py`、`SortPaper/src/judge/paper_evaluator.py`：理解 chunk 级判定、表格判定和论文级质量富集。
9. `SortPaper/src/store/qdrant_store.py`：理解向量集合、embedding、检索、rerank、payload 更新。
10. `SortPaper/src/agent/literature_agent.py`：最后看 Agent 如何调用检索工具并综合回答。

## 架构层

| 层级 | 关键文件 | 职责 |
|---|---|---|
| Streamlit 入口层 | `app.py`, `app_sidebar.py`, `app_ui.py` | 输入、模式选择、结果展示、保存、批量导入、向量库操作 |
| 业务编排层 | `app_pipeline.py`, `src/graph/pipeline_graph.py` | 预览、完整流水线、一键入库、质量富集、批量处理 |
| 解析层 | `src/parsers/*` | PDF 文本、表格、图片解析，产出 `LayoutChunk` |
| Judge 层 | `src/judge/*` | chunk 质量判定、表格判定、论文质量评估 |
| 存储与检索层 | `src/store/qdrant_store.py`, `src/store/faiss_store.py` | 向量入库、检索、rerank、payload 管理 |
| Agent 层 | `src/agent/literature_agent.py` | 基于检索工具做多轮查询和综合回答 |

## 主业务流程

### 单篇论文

1. 用户在侧边栏上传 PDF 或选择示例论文。
2. `build_paper_id` 用 PDF bytes 生成稳定 `paper_id`。
3. 用户选择快速预览、完整流水线或一键入库。
4. 快速预览走 `run_preview`：文本和表格解析、图片占位、表格 Judge，跳过图像视觉解析和向量入库。
5. 完整流水线走 `run_pipeline`：LangGraph 并行运行 text/table/image worker，再分别 Judge，最后 merge chunks。
6. 结果进入 `app_ui.py` 渲染：概览、文本、图片、表格、PDF 重建、语义检索、已保存记录。
7. 一键入库或手动入库调用 `store_parsed_chunks`，把 clean/degraded chunks 写入 Qdrant。
8. 质量分析调用 `evaluate_and_enrich_from_qdrant`，回写论文分类、摘要、可信度和 chunk context。

### 批量导入

1. `app_sidebar.py` 收集多个 PDF。
2. `app.py` 先做批内去重和已入库检查。
3. 通过 `ThreadPoolExecutor` 并发调用 `_process_batch_job`。
4. 每篇论文最终走预览或 `_process_one_pdf`，后者串联 `run_pipeline` 和 `store_parsed_chunks`。
5. UI 汇总成功、跳过、失败、入库数量和耗时。

### 语义检索与 Agent

1. 已入库 chunk 存在 Qdrant collection `papers`。
2. `qdrant_search` 或 `QdrantStore.search` 执行向量检索，可选 rerank。
3. `LiteratureAgent` 通过 function calling 调用 `search_literature`，可以多轮改写 query。
4. 检索结果由模型综合为结构化回答。

## 复杂热点

| 优先级 | 区域 | 原因 |
|---|---|---|
| P0 | `app_pipeline.py` | 业务主干集中在一个文件：预览、完整流水线、入库、质量富集、批量单篇处理都在这里 |
| P0 | `src/graph/pipeline_graph.py` | LangGraph 状态、并行 worker、retry、Judge、merge 都在同一状态机里 |
| P0 | `src/parsers/table/parser.py` | 表格解析策略很多，依赖区域检测、后处理、Judge 反馈、fallback 和去重 |
| P1 | `app_ui.py` | UI 展示函数数量多，表格调试和搜索等多个视图混在一个大文件 |
| P1 | `src/store/qdrant_store.py` | 同时承担集合管理、embedding provider、写入、检索、payload 更新 |
| P2 | `data/results`, `reports`, `.pytest_cache`, `.workbuddy` | 运行产物、报告或工具缓存，适合理解时排除，不应作为第一轮重构对象 |

## 阅读建议

先用业务流程读，不要先按文件大小读。`app.py` 和 `app_ui.py` 很大，但它们有不少 UI 状态和渲染细节；真正的业务核心更集中在 `app_pipeline.py`、`pipeline_graph.py`、`LayoutChunk`、`TableParser`、`QdrantStore` 这条链路上。

第一轮目标是回答三个问题：

- PDF 进入系统后，在哪些地方变成 `LayoutChunk`？
- 哪些判定决定 chunk 能否入库？
- Qdrant payload 里哪些字段支撑后续质量分析和检索？

这些答案清楚之后，再考虑拆分模块会稳很多。
