# PaperSort Pipeline Contract

## 目标

本文档固定 `SortPaper/app_pipeline.py` 当前对 UI、入库、质量富集和测试暴露的事实契约。后续重构可以移动实现，但不应无意改变这些输入输出字段。

## `run_preview(pdf_bytes) -> dict`

用途：快速预览 PDF 结构，不执行完整图像解析和向量入库。

返回字段：

- `paper_id`: 由 PDF bytes 生成的稳定 ID。
- `text_chunks`: 文本 chunk dict 列表。
- `table_chunks`: 表格 chunk dict 列表。
- `image_chunks`: 图片占位 chunk dict 列表。
- `merged_chunks`: 合并后的阅读顺序 chunk dict 列表。
- `images`: 图片元数据列表，包含页码、xref、尺寸、bbox、`vision_needed`。
- `verdicts`: 以 `chunk_id` 为 key 的 verdict dict。
- `status`: 当前为 `preview`。
- `quality`: 当前为空 dict。
- `worker_timing`, `judge_timing`, `merge_timing`, `desc_timing`: UI 展示用耗时。
- `mode`: 当前为 `preview`。

预览模式约束：

- 文本 chunk 不逐个调用 LLM Judge；非噪音文本 verdict 使用 `issue_type=preview_text_not_judged`。
- 表格 chunk 仍会调用 LLM Judge，除非是 `table_region_discovery_only`。
- 图片只生成占位 chunk，不调用 VisionParser。

## `run_pipeline(pdf_bytes, paper_id, filename) -> dict`

用途：执行完整解析流水线。

返回字段：

- `paper_id`, `filename`
- `text_chunks`, `table_chunks`, `image_chunks`
- `merged_chunks`
- `verdicts`: 汇总 `text_verdicts + table_verdicts + image_verdicts`，以 `chunk_id` 为 key。
- `status`: 来自 LangGraph final state。
- `quality`: 当前为空 dict；质量富集由后续按钮或批量流程触发。
- `worker_timing`, `judge_timing`, `merge_timing`, `desc_timing`
- `mode`: 当前为 `pipeline`。

流水线约束：

- `build_graph()` 的状态字段是内部契约，外部只消费返回 dict。
- 表格和图片 chunk 在返回前可能被 `_generate_chunk_description` 添加英文描述前缀；原始解析内容仍应保留在序列化 chunk 的 `raw_content` 中。
- UI 依赖 `merged_chunks`、`verdicts`、timing 字段渲染所有结果页。

## `store_parsed_chunks(result) -> dict`

用途：将已解析结果写入 Qdrant。

实现边界：`app_pipeline.py` 只保留兼容入口，实际入库决策和统计在 `src/store/chunk_storage.py` 的 `store_parsed_result_chunks` 中完成。UI 仍只调用 `store_parsed_chunks(result)`。

`src/store/chunk_storage.py` 中的轻量类型：

- `StoredChunkInput`：`merged_chunks` 中每个 chunk 的输入形状。
- `ChunkVerdict`：`verdicts[chunk_id]` 的判定形状。
- `StoreParsedResult`：`store_parsed_result_chunks` 接收的解析结果子集。
- `StoreStats`：入库函数返回的统计形状。

输入依赖：

- `result["paper_id"]`: 必须存在，缺省时使用 `"?"`。
- `result["filename"]` / `paper_title` / `title`: 用于 payload `paper_title`。
- `result["merged_chunks"]`: 候选入库 chunk 列表。
- `result["verdicts"]`: 以 `chunk_id` 为 key 的质量判定。

返回字段：

- `stored`: clean 入库数量。
- `degraded`: 表格降级入库数量。
- `failed`: 写入失败数量，或缺少 embedding key 时的可尝试入库数量。
- `excluded`: 被排除入库的 chunk 数量。
- `excluded_tables`: 被排除入库的表格数量。
- `excluded_reasons`: 排除原因计数。
- `duplicate`: 是否因已入库而跳过。
- `existing_count`: duplicate 时已有记录数。
- `attempted`: 实际尝试写入或缺少 key 时本应写入的数量。
- `missing_verdicts`: 缺少 verdict 的 chunk 数量。
- `error`: 首个写入错误或缺少 embedding key 的错误文本。

入库决策：

- 无 embedding API key：不连接写入，返回 `failed=attempted` 和错误。
- Qdrant 中已有同一 `paper_id`：返回 `duplicate=true`，不写入。
- `metadata.excluded_from_storage=true`：排除。
- verdict passed：clean 入库。
- table 且 verdict `issue_type != false_positive`：degraded 入库。
- table 且 verdict `issue_type == false_positive`：标记排除并记录原因。

## Qdrant Payload 关键字段

入库 metadata 必须保留：

- `paper_id`
- `page`
- `bbox`
- `column`
- `order_in_page`
- `global_order`
- `chunk_id`
- `content_type`
- `table_quality`
- `score`
- `paper_title`
- `raw_content`
- `enrichment_status`

质量富集会额外写入：

- `category`
- `credibility`
- `fermentation_relevance`
- `is_actionable`
- `paper_summary`
- `target_products`
- `organisms`
- `classify_reason`
- `classify_status`
- chunk 级 `context`
- 富集后的 `content`

不可破坏约束：

- `raw_content` 必须一直保留原始 chunk 文本；富集时只能把 `context + raw_content` 写入 `content`。
- Qdrant point id 由 `paper_id + chunk_id` 稳定生成。
- 表格 clean/degraded/excluded 的语义不能在 UI、Judge 和入库之间漂移。

## `evaluate_and_enrich_from_qdrant(paper_id) -> dict`

用途：从 Qdrant 读回某篇论文的 chunks，执行论文级质量分析并回写 payload。

行为约束：

- 如果没有 points，返回 `{"skip": True, "error": ...}`。
- 读取 chunk 时优先使用 payload `raw_content`，其次才 fallback 到 `content`。
- chunks 按 `global_order` 排序后传给 `PaperQualityEvaluator`。
- 论文级 payload 写回所有 points。
- chunk 级 context 存在时，写入 `context`，并将 `content` 更新为 `context + raw_content`。

## `_process_one_pdf(file_bytes, paper_id, filename) -> dict`

用途：批量导入中的单篇处理。

流程：

1. 调用 `run_pipeline`。
2. 在同一 `paper_id` lock 下调用 `store_parsed_chunks`。
3. 将 store 结果折叠成批量 UI 需要的状态：`ok`、`partial`、`fail`、`skip`、`no_store`。

批量返回字段至少包含：

- `status`, `reason`
- `category`, `credibility`
- `chunks`, `stored`, `degraded`, `store_failed`, `excluded`, `store_attempted`
- `parse_seconds`, `eval_seconds`, `store_seconds`, `total_seconds`
