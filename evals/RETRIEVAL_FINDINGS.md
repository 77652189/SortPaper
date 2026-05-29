# Retrieval Evaluation Findings

本文件记录 PaperSort 检索评测的当前结论。评测使用 Qdrant payload 自动生成弱标签 case，不依赖人工生物领域标注，适合做回归测试和方向判断。

## 当前结论

系统已经能比较稳定地找回相关论文，但 chunk/evidence 级召回仍是主要瓶颈。

标准检索在 60 条自动 case、top100 诊断下：

```text
paper_mrr = 0.6475
paper_hit@10 = 0.7667
paper_hit@100 = 0.9833
chunk_mrr = 0.2834
chunk_hit@10 = 0.3667
chunk_hit@100 = 0.5000
nearby_chunk_hit@10 = 0.3667
nearby_chunk_hit@100 = 0.6000
```

这说明目标 chunk 经常不只是排在 top10 后面，而是到 top100 也不能稳定出现。问题属于真实的证据级候选召回不足，而不是单纯 rerank 排序问题。

## Strategy 对比

| 策略 | paper_hit@10 | chunk_hit@10 | chunk_hit@20 | chunk_hit@100 | nearby_chunk_hit@10 | nearby_chunk_hit@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 0.7667 | 0.3667 | 0.4000 | 0.5000 | 0.3667 | 0.6000 |
| lexical_backfill | 0.9333 | 0.4333 | 0.4667 | 0.5667 | 0.5667 | 0.8333 |
| neighbor_backfill | 0.7833 | 0.3667 | 0.5333 | 0.8000 | 0.4000 | 0.8667 |
| lexical + neighbor | 0.8833 | 0.4333 | 0.4333 | 0.5667 | 0.5333 | 0.8333 |

结论：

- `lexical_backfill` 最适合改善 top10 证据召回，尤其是 nearby evidence。
- `neighbor_backfill` 能把更多 exact chunk 拉进候选池，`chunk_hit@100` 从 0.5000 提升到 0.8000，但 top10 收益有限。
- `lexical + neighbor` 没有超过单独 lexical，说明邻近 chunk 如果直接参与主排序，会带来一些排序污染。
- 下一步不应继续堆叠 backfill，而应把邻近 chunk 用作答案上下文扩展，或做更细的二阶段 evidence rerank。

## 主路径接入状态

`lexical_backfill` 已接入手动检索和 Agent 检索主路径，并默认开启；UI 仍保留显式开关，便于在泛查询或延迟敏感场景下关闭。

- 手动检索：UI 中的“增强 chunk 召回（推荐）”。
- Agent 检索：UI 中的“增强 chunk 召回（推荐）”。
- `app_utils.qdrant_search()` 会将 `lexical_backfill` 传给 `QdrantStore.search()`，并记录耗时日志。
- `LiteratureAgent` 初始化时接受 `lexical_backfill`，tool search 会传给 `QdrantStore.search()`，并记录耗时日志。

`neighbor_backfill` 目前只作为离线评测和 store 层实验开关，不接入 UI。它更适合用于“已命中证据的上下文扩展”，不适合作为默认 top10 排序策略。

Agent 已落地 evidence context expansion：`LiteratureAgent` 会在 tool search 返回 top evidence 后，先调用 `QdrantStore.expand_paper_local_context()` 在已命中论文内补 deeper evidence，再调用 `QdrantStore.expand_neighbor_context()` 追加同论文、同页或相邻顺序的 chunk。补充上下文不会进入 tool 返回结果，也不会改变主检索排序；默认总共最多追加 5 个上下文 chunk。

## Agent 上下文级评测

新增 `evals/agent_context_eval.py`，用于评估最终送入回答层的上下文中是否包含 exact/nearby evidence。该评测不调用 LLM，只验证回答层是否拿到了足够证据。

| 策略 | context_chunk_hit@10 | context_nearby_hit@10 | avg_added_context_chunks |
| --- | ---: | ---: | ---: |
| standard, no context expansion | 0.3333 | 0.3333 | 0.0000 |
| standard + context expansion | 0.3333 | 0.3667 | 9.2167 |
| lexical, no context expansion | 0.5333 | 0.5667 | 0.0000 |
| lexical + context expansion | 0.5333 | 0.6000 | 9.1333 |
| lexical + context expansion, total_limit=5 | 0.5333 | 0.6000 | 5.0000 |
| lexical + paper-local + neighbor context, total_limit=5, per_paper=3 | 0.5667 | 0.6000 | 5.0000 |

结论：context expansion 应作为回答上下文补充，而不是替代初始召回。新增 paper-local context 在当前评测中把 exact `context_chunk_hit@10` 从当前锚点保护版本的 0.5000 提升到 0.5667，nearby `context_nearby_hit@10` 从 0.5667 提升到 0.6000；`lexical_backfill` 仍是提升初始证据覆盖的主要杠杆。

## Search Text 回填

已对现有 Qdrant payload 执行 `search_text` 回填：

```text
scanned = 12865
updated = 12865
skipped_empty = 0
```

这次迁移只写入 payload 字段，不重新解析 PDF，不重新计算 embedding。新入库 chunk 也会自动写入 `search_text`。

回填后，`lexical_backfill` 会优先使用 `search_text`，再回退到原来的 rerank 文本。60 条自动 case 对比：

| 指标 | lexical 旧文本 | lexical + search_text |
| --- | ---: | ---: |
| paper_mrr | 0.7595 | 0.7858 |
| chunk_mrr | 0.3284 | 0.3826 |
| nearby_chunk_mrr | 0.3669 | 0.4131 |
| chunk_hit@1 | 0.2667 | 0.3000 |
| chunk_hit@10 | 0.4333 | 0.6000 |
| nearby_chunk_hit@10 | 0.5667 | 0.6333 |

Agent 上下文评测中，`search_text` 和锚点保护主要提升排序质量：

| 指标 | lexical + context | lexical + search_text + anchor context |
| --- | ---: | ---: |
| context_chunk_mrr | 0.3678 | 0.3867 |
| context_nearby_mrr | 0.4039 | 0.4172 |
| context_chunk_hit@10 | 0.5333 | 0.5667 |
| context_nearby_hit@10 | 0.6000 | 0.6000 |

## Indexed Lexical Backfill

`lexical_backfill` 已从“缓存后全量扫描”改为内存倒排索引：

- 无过滤查询会基于 `search_text`/rerank 文本构建倒排索引，并复用到后续查询。
- 候选生成优先选择低频关键词，跳过覆盖面过大的高频词，避免 `E. coli`、`fucosyllactose` 这类泛词把候选池拉回半库规模。
- 带 Qdrant filter 的查询继续使用 scroll 路径，先保证过滤语义不被索引化逻辑改坏。
- 新增 `elapsed_ms_avg`、`elapsed_ms_p50`、`elapsed_ms_p95`，用于同时观察召回质量和检索耗时。

top10 主路径规模评测：

| 指标 | standard top10 | lexical indexed top10 |
| --- | ---: | ---: |
| elapsed_ms_p50 | 560.7182 | 744.9653 |
| elapsed_ms_p95 | 647.8909 | 900.1438 |
| paper_hit@10 | 0.7833 | 0.9833 |
| chunk_hit@10 | 0.4000 | 0.6000 |
| nearby_chunk_hit@10 | 0.4000 | 0.6333 |

top100 深层评测中，索引化 lexical 仍保持 `chunk_hit@10 = 0.4667`、`nearby_chunk_hit@10 = 0.6000`、`chunk_hit@100 = 0.5667`、`nearby_chunk_hit@100 = 0.8333`。因此当前判断是：索引化 lexical 适合作为手动检索和 Agent 检索默认路径；同时保留关闭开关，继续观察真实 UI 查询的耗时和误召回。

## Paper-local Evidence 实验

新增 `--strategy paper-local`：先用当前检索确定 top 论文，再回到这些论文内部基于 `search_text` 做 evidence scoring。实验目的不是替代主检索，而是验证“正确论文已命中但证据 chunk 没进 top10”的问题能否通过论文内二次定位解决。

结果显示它能把更多 exact chunk 带进深层候选池，但当前排序不适合直接进入 top10：

| 指标 | indexed lexical top10 | paper-local p5 top10 | paper-local top100 |
| --- | ---: | ---: | ---: |
| elapsed_ms_p50 | 716.4640 | 1330.6918 | 5931.5313 |
| paper_hit@10 | 0.9833 | 0.9500 | 0.9833 |
| chunk_hit@10 | 0.5667 | 0.5333 | 0.4333 |
| nearby_chunk_hit@10 | 0.6333 | 0.6000 | 0.4667 |
| chunk_hit@100 | - | - | 0.7333 |
| nearby_chunk_hit@100 | - | - | 0.8000 |

判断：paper-local 不应替代 indexed lexical 的 top10 主排序；它的问题是耗时更高，而且论文内 lexical scoring 会把同论文里的泛相关 chunk 排到目标证据前面。当前更合适的落点是 Agent 回答上下文层：在不改变 tool search 排名的前提下，把已命中论文内的 deeper evidence 补给综合回答。

## Two-stage 实验

两阶段论文优先策略曾提升论文级召回，但压低 top10 内的精确 chunk 命中：

```text
paper_hit@10 = 0.8500
chunk_hit@10 = 0.2667
nearby_chunk_hit@10 = 0.3000
nearby_chunk_hit@20 = 0.4667
```

结论：`two-stage` 更适合论文级浏览或离线分析，不适合直接作为默认问答证据检索策略。

## 已实现的评测能力

- `paper_hit@K` / `paper_mrr`
- `chunk_hit@K` / `chunk_mrr`
- `nearby_chunk_hit@K` / `nearby_chunk_mrr`
- 按 `title`、`metadata`、`chunk` 拆分的 `by_type` 指标
- `--strategy standard|two-stage`
- `--lexical-backfill`
- `--neighbor-backfill`
- 默认只用 `text` 和 `table` chunk 生成证据 case，避免 image chunk 噪音

## 失败模式

1. 正确论文能进入 top10，但目标 chunk 不在 top10。
2. 部分目标 chunk 到 top100 仍找不到，说明候选召回不足。
3. 部分目标 chunk 能通过 neighbor_backfill 进入 top20/top100，但排序不够靠前。
4. 同主题论文很多，常见 HMO 主题会互相抢排名。
5. exact chunk 弱标签偏严格，命中同页或相邻 chunk 有时也足以回答问题。
6. payload 字段当前主要影响 rerank、lexical boost 和 eval case 生成；既有 text chunk 的 embedding 仍主要来自 raw content。

## 改进优先级

P0：构建干净的 `search_text`

- 已实现 payload-only `search_text`，用于 lexical/search。
- 下一步才考虑是否用于 embedding。
- 合并标题、正文、表格、产品、菌株、基因、酶名、摘要和邻近上下文。
- 排除 `pdf`、`text`、`chunk`、工具痕迹、乱码文件名和低价值标签。

P0：把 lexical backfill 改成索引化方案

- 当前扫描缓存只能证明方向有效。
- 真正上线应使用倒排索引、关键词索引，或 Qdrant 支持的 text index/filter 能力。
- 目标是保留 `nearby_chunk_hit@10` 的收益，同时避免全量扫描。

P1：把 neighbor backfill 移到答案上下文层

- 已在 Agent 回答上下文中实现。
- 主检索 top10 仍优先保持 lexical 的相关性排序。
- 对 top evidence 命中的 chunk，追加同页、`global_order` 接近、`order_in_page` 接近的邻近 chunk 给 Agent。
- 这能利用 `chunk_hit@100` 的收益，同时避免污染 top10 排序。

P1：二阶段 evidence rerank

- 第一阶段：使用 lexical/search_text 找到高质量论文和候选 chunk。
- 第二阶段：仅在候选论文内部取邻近 evidence，再用轻量规则或 rerank 排序。

P2：rerank 成本控制

- rerank 对论文级排序有帮助，但 API 成本和耗时较高。
- 建议只在 Agent、用户显式勾选或高价值问题中启用。
