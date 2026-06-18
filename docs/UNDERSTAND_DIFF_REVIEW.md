# understand-diff 风险复盘

## Changed Components

最近两轮直接改动集中在：

- `SortPaper/app_pipeline.py`
  - `store_parsed_chunks` 现在只作为兼容入口，转调 `src.store.chunk_storage.store_parsed_result_chunks`。
- `SortPaper/src/store/chunk_storage.py`
  - 承接入库决策、表格 storage decision、metadata 构造、clean/degraded/excluded 统计、写入错误汇总。
  - 新增 `StoredChunkInput`、`ChunkVerdict`、`StoreParsedResult`、`StoreStats` 等 `TypedDict`，把原先靠约定的 dict 契约显式化。
- `SortPaper/tests/test_chunk_storage.py`
  - 覆盖 clean、degraded、false_positive 排除、metadata 预标记排除、缺 verdict、缺 embedding key、返回字段全集和 metadata 字段全集。
- understand 产物与文档
  - 重建干净中文 `knowledge-graph.json`
  - 新增领域图、pipeline 契约、chat 结论和 diff overlay

注意：`SortPaper/app.py` 在本次开始前已经处于 modified 状态，不属于这次重构。

## Affected Components

根据 `.understand-anything/diff-overlay.json` 的一跳关联，潜在受影响组件包括：

- `SortPaper/app.py`：调用 `run_preview`、`run_pipeline`、`store_parsed_chunks`、`evaluate_and_enrich_from_qdrant`。
- `SortPaper/app_sidebar.py`：调用质量富集入口。
- `SortPaper/app_utils.py`：提供 `build_paper_id`、chunk 序列化和描述生成。
- `SortPaper/src/graph/pipeline_graph.py`：完整流水线数据来源。
- `SortPaper/src/judge/table_judge.py`：`build_storage_decision` 仍是表格入库排除规则中心。
- `SortPaper/src/store/qdrant_store.py`：接收 `store.add` metadata、content、embed_content。
- `SortPaper/tests/test_enrichment_decoupling.py`：覆盖本次改动最相关的入库和富集契约。

## Affected Layers

- 业务编排层：`app_pipeline.py`
- 入口与界面层：`app.py`、`app_sidebar.py`
- Judge 层：`table_judge.py`、`paper_evaluator.py`
- 存储检索层：`qdrant_store.py`
- 解析层：`LayoutChunk` 和 parser 输出契约

## Risk Assessment

风险等级：中低。

理由：

- 改动把高复杂度文件 `app_pipeline.py` 的入库逻辑移到 `chunk_storage.py`，但没有改变公开函数签名。
- clean/degraded/excluded 分支条件保持原样。
- Qdrant payload 关键字段保持原样。
- 相关测试和全量测试均已通过。

仍需重点关注：

- 缺少 embedding key 的分支依然会先实例化 `QdrantStore`，这是原有行为；本次未改变。
- `apply_table_storage_decision` 仍吞掉 `build_storage_decision` 异常，保持原行为；后续若要提升可观测性，应单独设计日志策略。
- 下一步如果继续推进，可以给 `store_parsed_result_chunks` 引入显式输入/输出类型，减少 dict 约定。

## Dashboard Overlay

已生成：

`.understand-anything/diff-overlay.json`

dashboard 可以用它高亮当前改动和一跳影响范围。

## 2026-05-25 table dedup 拆分

### Changed Components

- `SortPaper/src/parsers/table/parser.py`
  - `TableParser` 仍保留 `_deduplicate_chunks`、`_expand_vision_clips_by_region` 以及相关 helper 的静态入口。
  - 具体去重、bbox、质量评分、vision clip 扩展规则已转发到 `src.parsers.table.dedup`。
- `SortPaper/src/parsers/table/dedup.py`
  - 新增表格候选去重模块，集中承接候选合并、同一区域判断、包含关系判断、质量评分和 vision fallback 裁剪框扩展。

### Affected Components

- `SortPaper/src/parsers/table_parser.py`：仍通过 `TableParser` 使用表格解析入口。
- `SortPaper/src/parsers/table/parse_router.py`：继续接收 `TableParser._deduplicate_chunks` 作为去重回调。
- `SortPaper/src/parsers/layout_chunk.py`：去重逻辑依赖 `LayoutChunk` 的 `bbox`、`page`、`metadata` 和 `chunk_id` 契约。
- `SortPaper/src/parsers/config.py`：去重阈值和质量评分权重仍来自 `TABLE_PARSER` 配置。
- `SortPaper/src/parsers/table/diagnostics.py`：`dedup.py` 复用 `bbox_overlap_ratio`。

### Risk Assessment

风险等级：低。

理由：
- 这是结构性迁移，保留了原有 `TableParser` 静态方法入口，测试直接调用路径未变。
- 去重规则、阈值、metadata 字段写入保持原样。
- 新增模块尚未进入 knowledge graph，dashboard overlay 中以 `parser.py` 和 `TableParser` 节点代表本次改动，`dedup.py` 列在 changedFiles 中。

验证：
- `pytest tests/table/test_diagnostics.py -q`：27 passed
- `pytest tests/table -q`：84 passed
- `pytest -q`：122 passed

## 2026-05-25 DashScope rerank key 修复

### Changed Components

- `SortPaper/src/store/qdrant_store.py`
  - 新增 `dashscope_rerank_api_key()`，复用现有 `_env_value` 的 BOM 兼容逻辑读取 `DASHSCOPE_API_KEY`。
  - `_rerank()` 调用 `dashscope.TextReRank.call()` 时显式传入 `api_key`，避免 DashScope SDK 只查标准环境变量而读不到 `.env` 第一行 BOM key。
  - 未做降级；缺少 key 时仍会报错，但错误会来自项目侧的明确校验。
- `SortPaper/tests/test_qdrant_point_ids.py`
  - 新增 BOM 环境变量读取测试。
  - 新增 rerank 显式传递 DashScope API key 的回归测试。

### Affected Components

- `SortPaper/app_ui.py` / `SortPaper/app_utils.py`：手动检索启用 rerank 时会走该路径。
- `SortPaper/src/agent/literature_agent.py`：Agent 搜索默认使用 rerank，也受益于显式 key 传递。
- `SortPaper/src/store/qdrant_store.py`：存储检索层的 rerank 行为保持启用，不做 silent fallback。

### Risk Assessment

风险等级：低。

理由：
- 只改变 DashScope rerank API key 的传递方式，不改变检索召回、排序参数或 UI 选项。
- 原因已复现：`.env` 中第一行被解析为 `\ufeffDASHSCOPE_API_KEY`，SDK 默认查 `DASHSCOPE_API_KEY` 因而报缺 key。
- 现在项目侧读取时兼容 BOM，并把 key 显式交给 SDK。

验证：
- `pytest tests/test_qdrant_point_ids.py -q`：5 passed
- `pytest -q`：124 passed

## 2026-05-25 search relevance 修复

### Changed Components

- `SortPaper/src/store/qdrant_store.py`
  - 修复 rerank 排序和分数回填错位：DashScope 返回的 `index` 是原始候选下标，现在按原始下标映射 relevance score，再排序和写回分数。
- `SortPaper/tests/test_qdrant_point_ids.py`
  - 新增 rerank 原始下标和分数映射测试，避免后续再出现“显示分数与排序不一致”的问题。
- `SortPaper/app_utils.py`
  - `qdrant_search()` 支持传入额外 `filter_kwargs`，同时保留 `paper_id` 过滤。
- `SortPaper/app_ui.py`
  - 手动检索新增默认勾选的“仅检索质量解析后的可执行实验论文”，对应 `is_actionable=True`。
  - 每条搜索结果展示质量解析字段：分类、可信度、发酵相关性、产品、菌株/宿主。

### Affected Components

- 手动搜索页：默认结果会更接近实验论文证据，减少综述、专利和低行动价值 chunk 污染。
- Agent 搜索：受益于 rerank 排序修复，但本次没有修改 Agent 的工具过滤策略。
- Qdrant 检索层：排序逻辑修正，不改变向量召回和 DashScope rerank 参数。

### Diagnosis

- Qdrant 当前 12444 个 point 都是 `enrichment_status=done`，所以“没有质量解析”的主要问题不是状态缺失，而是 UI 未展示质量字段，以及默认全库检索没有使用质量过滤。
- 不加过滤时，该问题查询更容易命中 HMO 综述、专利或乱码中文片段。
- 使用 `is_actionable=True` 或 `category=fermentation_experiment` 后，top 结果回到 `2022 Multi-level metabolic engineering / Escherichia coli / 2'-FL, 3-FL` 这类预期结果。

验证：
- `python -m py_compile app_ui.py app_utils.py src/store/qdrant_store.py`：passed
- `pytest tests/test_qdrant_point_ids.py -q`：6 passed
- `pytest -q`：125 passed

## 2026-05-25 search relevance 修正（二）

### Changed Components

- `SortPaper/app_ui.py`
  - 取消“仅检索质量解析后的可执行实验论文”的默认勾选，避免把综述、机制说明或专利证据硬排除。
  - 保留该选项作为手动限定范围的可选过滤器。
- `SortPaper/src/store/qdrant_store.py`
  - rerank 候选池从 `limit * 3` 扩到 `limit * 8`，减少相关 chunk 未进入 rerank 的概率。
  - 增加 lexical backfill：对带有明确科学实体的问题，从 payload 中补充精确命中候选。
  - 增加 query relevance boost：例如 `LNT II` 查询必须命中 `LNT II` / `lacto-N-triose II` 才加权；命中 `lgtA`、`RBS`、`Escherichia coli` 等机制词时进一步提高排序。
- `SortPaper/tests/test_qdrant_point_ids.py`
  - 新增 LNT II 锚点测试，防止泛泛的 E. coli / metabolic chunk 被错误抬高。
  - 新增机制 chunk 排序测试，确保 `LNT II + RBS + lgtA` 这类具体证据优先于宽泛综述句。

### Diagnosis

- 上一版默认 `is_actionable=True` 太硬，会把非实验论文全部排除；这不适合“读懂机制/路径逻辑”的问题。
- 当前问题真正需要的是“精确实体 + 机制词”优先，而不是“只看实验论文”。
- 修正后，同一查询的 top 结果回到 `2022 Multi-level metabolic engineering` 第 9 页，并命中 `LNT II`、`lgtA`、`RBS`。

验证：
- `python -m py_compile src/store/qdrant_store.py app_ui.py app_utils.py`：passed
- `pytest tests/test_qdrant_point_ids.py -q`：8 passed
- `pytest -q`：127 passed

## 2026-05-25 judge metadata 拆分

### Changed Components

- `SortPaper/src/parsers/table/parser.py`
  - `TableParser` 继续保留 judge/feedback 相关兼容方法。
  - LLM judge 调用与 assistive retry 控制流仍留在 `parser.py`，避免改变 monkeypatch 路径和主流程边界。
  - region identity、rule feedback、candidate score、LLM 结果/error、auto decision、storage decision 的 metadata 写入已转发到 `src.parsers.table.judge_metadata`。
- `SortPaper/src/parsers/table/judge_metadata.py`
  - 新增表格 judge metadata 写入模块。
  - 集中承接 `LayoutChunk.metadata` 的字段更新，复用 `table_judge` 的 scoring/storage/auto-decision API。

### Affected Components

- `SortPaper/src/parsers/table/parser.py`：仍是 judge 编排入口，行为路径保持不变。
- `SortPaper/src/judge/table_judge.py`：`judge_metadata.py` 依赖其中的 `score_table_candidate`、`build_auto_decision`、`build_storage_decision`。
- `SortPaper/src/parsers/table/region_chunks.py`：`judge_metadata.py` 复用 caption 和 table label 提取规则。
- `SortPaper/src/parsers/layout_chunk.py` 与 `SortPaper/src/parsers/table/models.py`：继续作为 chunk/region 数据契约。

### Risk Assessment

风险等级：低到中。

理由：
- 这次移动的是 metadata 写入逻辑，字段名和写入条件保持原样。
- LLM judge 外部调用没有迁移，现有测试中的 `table_parser_module.judge_table_failure_with_llm` monkeypatch 仍然有效。
- 风险主要在 metadata 字段遗漏或 auto/storage decision 写入顺序，但专项与全量测试均已覆盖通过。

验证：
- `pytest tests/table/test_diagnostics.py -q`：27 passed
- `pytest tests/table -q`：84 passed
- `pytest -q`：122 passed

## 2026-05-25 region chunk 拆分

### Changed Components

- `SortPaper/src/parsers/table/parser.py`
  - `TableParser` 继续保留 `_region_discovery_chunks`、`_unparsed_region_chunks`、`_region_caption_text`、`_table_label_from_caption` 等兼容入口。
  - discovery/unparsed placeholder 的具体构造逻辑已转发到 `src.parsers.table.region_chunks`。
- `SortPaper/src/parsers/table/region_chunks.py`
  - 新增 TableRegion 到 LayoutChunk 的转换模块。
  - 集中承接 discovery-only chunk、未解析区域 placeholder、caption 提取、table label 提取、placeholder 发射条件。

### Affected Components

- `SortPaper/src/parsers/table/parser.py`：主解析流程仍在 region discovery mode 和常规解析后调用这些入口。
- `SortPaper/tests/table/test_diagnostics.py`：直接调用 `TableParser` 兼容方法，已验证路径不变。
- `SortPaper/src/parsers/table/diagnostics.py`：`region_chunks.py` 复用 `find_best_region` 判断已有表格是否覆盖 region。
- `SortPaper/src/parsers/table/models.py` 与 `SortPaper/src/parsers/layout_chunk.py`：继续作为区域和 chunk 数据契约。

### Risk Assessment

风险等级：低。

理由：
- 这是纯结构性拆分，metadata 字段、raw_content、caption/table_label 规则、placeholder 过滤阈值均保持原样。
- `TableParser` 原有方法名仍存在，直接测试和潜在外部调用不会断。
- 新增模块尚未进入 knowledge graph，dashboard overlay 用 `parser.py` / `TableParser` 作为已知节点代表本次影响面，并把 `region_chunks.py` 放入 changedFiles。

验证：
- `pytest tests/table/test_diagnostics.py -q`：27 passed
- `pytest tests/table -q`：84 passed
- `pytest -q`：122 passed
