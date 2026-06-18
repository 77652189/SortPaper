# PaperSort 重构候选清单

## 结论

第一阶段不要大重构。先围绕“PDF 到 Qdrant，再到检索回答”的主链路收敛边界，把运行产物和报告从理解范围里排除，再对核心模块做小步拆分。

## P0 必须先理解

| 区域 | 为什么必须先理解 | 第一轮问题 |
|---|---|---|
| `SortPaper/app_pipeline.py` | 这是业务编排中心，串起预览、完整解析、入库、质量富集和批量单篇处理 | 哪些函数是对 UI 暴露的稳定入口？哪些只是内部 helper？ |
| `SortPaper/src/graph/pipeline_graph.py` | 完整流水线实际在这里执行，包含 LangGraph 状态、worker、Judge、retry、merge | 状态字段哪些是必要契约，哪些是历史遗留？ |
| `SortPaper/src/parsers/layout_chunk.py` | `LayoutChunk` 是文本、表格、图片在全系统流转的共同数据结构 | chunk_id、bbox、global_order、metadata 的不变量是什么？ |
| `SortPaper/src/parsers/table/parser.py` | 表格解析是最复杂的领域逻辑，且和 Judge、存储降级强耦合 | 区域检测、结构提取、后处理、fallback、Judge 反馈能否明确分阶段？ |
| `SortPaper/src/store/qdrant_store.py` | 入库、检索、payload、embedding provider 都依赖它 | 是否需要拆出 collection 管理、embedding、检索、payload update？ |

## P1 值得重构

| 区域 | 重构价值 | 建议方向 |
|---|---|---|
| `SortPaper/app.py` | UI 事件流过长，批量处理、单篇处理、保存、质量分析混在一起 | 先抽出 `single_paper_controller` 和 `batch_controller`，保留 Streamlit 渲染在原层 |
| `SortPaper/app_ui.py` | 展示函数过多，表格调试、检索、已保存详情混杂 | 按 tab 拆分视图模块：overview/text/table/image/search/saved |
| `SortPaper/src/judge/table_judge.py` | 表格质量判断规则、LLM 判定、存储决策集中 | 先抽纯规则评分，再保留 LLM 调用为薄适配层 |
| `SortPaper/src/judge/paper_evaluator.py` | 论文级分类、Map-Reduce、chunk context 富集影响检索质量 | 明确输入输出 schema，补足离线样例测试 |
| `SortPaper/src/agent/literature_agent.py` | Agent 与 QdrantStore 直接耦合，且依赖外部模型 | 抽出 search tool 接口，便于替换检索实现和做离线测试 |

## P2 暂时忽略

| 区域 | 原因 |
|---|---|
| `SortPaper/data/results/*.json` | 解析结果快照，属于运行产物或样例数据，不代表架构复杂度 |
| `SortPaper/reports/*.json` | 表格 benchmark/self-check 报告，适合作为分析材料，不应先重构 |
| `SortPaper/.pytest_cache` | pytest 缓存，应从 understand 分析和人工阅读里排除 |
| `SortPaper/.workbuddy` | 工具记忆，不属于业务代码 |
| 多语言 README 的乱码内容 | 文档编码问题需要单独处理，不应混入业务重构 |

## 建议的第一刀

第一刀不要动解析算法，先整理边界：

1. 为 `app_pipeline.py` 写一页接口说明，固定外部入口：`run_preview`、`run_pipeline`、`store_parsed_chunks`、`evaluate_and_enrich_from_qdrant`、`_process_one_pdf`。
2. 梳理这些函数的输入输出字典，尤其是 `result`、`verdicts`、`merged_chunks`、`quality`、`store_result`。
3. 用现有测试覆盖 `paper_id`、Qdrant point id、表格判定和 enrichment 的关键契约。
4. 入库逻辑已经从 `app_pipeline.py` 拆到 `src/store/chunk_storage.py`，后续可以继续围绕这个 service 收紧类型和错误可观测性。

## understand-diff 护栏

后续发生代码改动后，每次都用 `understand-diff` 记录：

- 改动影响哪些流程节点。
- 是否改变 `LayoutChunk`、verdict、Qdrant payload 或 Streamlit result 字典。
- 是否需要补充或更新测试。
- 是否引入新的外部 API key、模型、collection schema 或环境变量要求。
