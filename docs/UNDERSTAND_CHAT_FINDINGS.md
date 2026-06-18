# understand-chat 聚焦问题结论

## 1. 主链路从 PDF 到 Qdrant 经过哪些文件？

主链路是：

`app_sidebar.py` 选择 PDF 与模式 -> `app.py` 处理单篇/批量事件 -> `app_pipeline.py` 编排预览、完整流水线、入库、质量富集 -> `src/graph/pipeline_graph.py` 执行 LangGraph worker/Judge/merge -> `src/parsers/*` 产出 `LayoutChunk` -> `src/judge/*` 产出 verdict -> `src/store/qdrant_store.py` 写入 Qdrant。

关键入口：

- `run_preview`：轻量预览，文本不逐 chunk 调用 LLM Judge，表格仍做 Judge，图片只生成占位信息。
- `run_pipeline`：完整解析，调用 `build_graph()` 执行文本、表格、图片并行 worker 和 Judge。
- `store_parsed_chunks`：把 `merged_chunks` 按 verdict 和表格质量决策写入 Qdrant。
- `evaluate_and_enrich_from_qdrant`：从 Qdrant 读回 chunks 做论文级质量富集。

## 2. `LayoutChunk` 的字段契约是什么？

`LayoutChunk` 是解析层、Judge 层、UI 层和存储层共享的数据结构。

关键字段：

- `chunk_id`：全局唯一 chunk 标识；未提供时根据 `content_type/page/column/y/x` 自动生成。Qdrant point id 由 `paper_id + chunk_id` 稳定哈希得到。
- `content_type`：`text`、`table` 或 `image`，决定 Judge、展示、embedding 文本和入库规则。
- `raw_content`：解析器原始输出；入库时保存在 payload 的 `raw_content`，后续质量富集也优先使用它。
- `page`、`bbox`、`column`、`order_in_page`、`global_order`：负责 PDF 定位、阅读顺序和 UI 重建。
- `metadata`：承载 parser、表格结构质量、表格区域、是否排除入库、是否需要人工检查等扩展字段。

最需要保护的不变量：

- `chunk_id` 在同一 paper 内必须稳定。
- `raw_content` 不能被 context 富集覆盖；富集只能更新 `content`，并保留 `raw_content`。
- `global_order` 是质量评估和阅读顺序的重要依据。
- 表格相关的 `metadata.excluded_from_storage` 与 `storage_exclusion_reason` 会影响入库决策。

## 3. 哪些地方写入或读取 `result/verdicts/merged_chunks`？

写入方主要在 `app_pipeline.py`：

- `run_preview` 返回 `text_chunks`、`table_chunks`、`image_chunks`、`merged_chunks`、`verdicts`、timing 和 `mode=preview`。
- `run_pipeline` 汇总 LangGraph stream updates，返回 `merged_chunks`、`verdicts`、timing 和 `mode=pipeline`。
- `store_parsed_chunks` 读取 `result["verdicts"]` 和 `result["merged_chunks"]`，返回入库统计。
- `_process_one_pdf` 用 `run_pipeline` + `store_parsed_chunks` 生成批量导入结果。

读取方主要在：

- `app.py`：控制按钮、保存、入库、质量富集、批量结果汇总。
- `app_ui.py`：渲染概览、文本、表格、图片、PDF 重建、检索和已保存详情。
- `app_utils.py`：保存结果快照、加载历史详情、检索。
- `tests/test_enrichment_decoupling.py`：覆盖入库 pending、raw_content 富集、预览跳过 text Judge。

这说明 `result` 现在是事实上的公共接口，但没有显式类型或 schema，是后续重构的主要风险点。

## 4. 表格 clean/degraded/excluded 决策分散在哪些模块？

分散在三类位置：

- `src/parsers/table/parser.py`：解析阶段会根据表格区域、候选评分、LLM 决策和不可解析状态写入 metadata，例如 `excluded_from_storage`、`storage_exclusion_reason`、`llm_decision_mode`。
- `src/judge/table_judge.py`：`build_storage_decision` 是规则中心，根据 `unparseable_reason`、`llm_category`、`rule_category`、`quality_score`、`manual_review_needed` 等 metadata 给出最终入库排除或保留决策。
- `app_pipeline.py`：`store_parsed_chunks` 在入库前再次调用 `build_storage_decision`，并根据 verdict 决定：
  - passed -> clean 入库
  - table 且 `issue_type != false_positive` -> degraded 入库
  - table 且 `issue_type == false_positive` -> excluded
  - metadata 已标记 `excluded_from_storage` -> excluded

第一刀重构应优先把 `app_pipeline.py` 的入库决策与 metadata 构造拆成小函数，但不要改 `table_judge.py` 的业务规则。
