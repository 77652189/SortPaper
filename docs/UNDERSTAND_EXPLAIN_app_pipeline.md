# understand-explain: `SortPaper/app_pipeline.py`

## 角色定位

`app_pipeline.py` 是 PaperSort 的业务编排层。它不负责直接渲染 UI，也不实现底层 PDF 解析算法，而是把 UI 请求转换为可执行的处理流程：

- 快速预览：`run_preview`
- 完整流水线：`run_pipeline`
- 手动或一键入库：`store_parsed_chunks`
- 质量富集：`evaluate_and_enrich_from_qdrant`
- 批量单篇处理：`_process_one_pdf`

它是第一轮重构前必须理解的核心模块。

## 外部入口

| 函数 | 调用方 | 输出 |
|---|---|---|
| `run_preview(pdf_bytes)` | `app.py` 快速预览模式 | 解析后的 text/table/image chunks、verdicts、timing、mode |
| `run_pipeline(pdf_bytes, paper_id, filename)` | `app.py` 完整流水线或一键入库 | LangGraph 最终结果、merged chunks、verdicts、timing |
| `store_parsed_chunks(result)` | `app.py` 手动入库或一键入库 | stored/degraded/failed/excluded/duplicate 等入库统计 |
| `evaluate_parsed_chunks(result)` | 兼容入口 | 根据 `paper_id` 转调 Qdrant 富集 |
| `evaluate_and_enrich_from_qdrant(paper_id)` | `app.py` 质量分析按钮、`app_sidebar.py` 向量库管理 | 论文分类、可信度、摘要、payload 更新结果 |
| `_process_one_pdf(file_bytes, paper_id, filename)` | 批量导入 | 单篇批处理状态、耗时、入库数量 |

## 数据流

1. UI 传入 PDF bytes、paper_id、filename。
2. `run_preview` 或 `run_pipeline` 将 bytes 写入临时 PDF 文件。
3. 预览模式直接调用 PyMuPDF、TableParser、图片占位和部分 Judge。
4. 完整模式调用 `src.graph.pipeline_graph.build_graph()`，由 LangGraph 执行 worker、Judge 和 merge。
5. 结果被统一转换为 dict：`text_chunks`、`table_chunks`、`image_chunks`、`merged_chunks`、`verdicts`、`worker_timing`、`judge_timing`。
6. `store_parsed_chunks` 根据 verdict 和表格质量 metadata 决定 clean、degraded、excluded 或 failed。
7. `evaluate_and_enrich_from_qdrant` 从 Qdrant 读回 chunks，运行 `PaperQualityEvaluator`，再把论文级和 chunk 级富集结果写回 payload。

## 依赖关系

主要上游：

- `app.py`：触发预览、完整流水线、入库、质量富集和批量处理。
- `app_sidebar.py`：向量库管理中触发质量富集。

主要下游：

- `app_utils.py`：`build_paper_id`、chunk 序列化、结果保存、表格/图片描述生成。
- `src.graph.pipeline_graph`：完整流水线状态机。
- `src.parsers.*`：预览模式直接调用文本、表格和图片占位解析。
- `src.judge.*`：表格 Judge、论文质量评估、表格 embedding 文本构造。
- `src.store.qdrant_store.QdrantStore`：入库、读取、payload 更新。

## 风险点

- `result` 是宽松 dict，没有显式 schema。UI、入库、质量富集都依赖其中字段名，重构时很容易漏字段。
- `store_parsed_chunks` 同时做重复检查、质量决策、metadata 构造、embedding 内容选择、Qdrant 写入和错误统计，职责过多。
- 表格 degraded/excluded 规则分散在 `app_pipeline.py`、`table_judge.py`、table parser metadata 之间。
- `run_pipeline` 手动合并 LangGraph streaming updates，状态 reducer 逻辑和 `pipeline_graph.py` 的状态契约强耦合。
- 文件中有不少中文注释或字符串出现编码异常，增加阅读和维护成本。

## 可重构切入点

第一阶段建议只整理边界，不改算法：

1. 为 `run_preview`、`run_pipeline`、`store_parsed_chunks` 的返回 dict 定义文档化 schema。
2. 把 `store_parsed_chunks` 内部拆成纯函数：筛选可入库 chunk、构造 metadata、执行 Qdrant 写入、汇总统计。
3. 将 Qdrant 写入错误和重复检查保持在 service 层，UI 只消费结构化结果。
4. 保持 `run_pipeline` 对 `build_graph()` 的调用方式不变，先不要重写 LangGraph 状态机。

## 最小验收

- 不改变 Streamlit 的三种模式：快速预览、完整流水线、一键入库。
- 不改变 Qdrant payload 的关键字段：`paper_id`、`chunk_id`、`content_type`、`raw_content`、`score`、`paper_title`、`enrichment_status`。
- 不改变表格 clean/degraded/excluded 的业务语义。
- 重构后必须用 `understand-diff` 记录影响范围。
