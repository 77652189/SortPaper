# Chunk 级召回改进实验

本轮目标是验证 chunk/evidence 召回差的问题究竟来自候选池缺失、邻近证据未展开，还是 top10 排序不佳。`lexical_backfill` 已通过评测并接入手动检索和 Agent 检索默认路径；其余实验路径仍默认关闭，用于离线评测和后续方案验证。

## 实验命令

标准检索：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 50 100 --strategy standard --out reports/retrieval_eval_standard60_top100.json
```

词面候选补召回：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 50 100 --strategy standard --lexical-backfill --out reports/retrieval_eval_lexical_backfill60_top100.json
```

邻近 chunk 回填：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 50 100 --strategy standard --neighbor-backfill --out reports/retrieval_eval_neighbor_backfill60_top100.json
```

词面补召回 + 邻近 chunk 回填：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 50 100 --strategy standard --lexical-backfill --neighbor-backfill --out reports/retrieval_eval_lexical_neighbor60_top100.json
```

## 结果对比

| 指标 | standard | lexical | neighbor | lexical + neighbor |
| --- | ---: | ---: | ---: | ---: |
| paper_mrr | 0.6475 | 0.7595 | 0.6931 | 0.7487 |
| paper_hit@10 | 0.7667 | 0.9333 | 0.7833 | 0.8833 |
| chunk_mrr | 0.2834 | 0.3284 | 0.3169 | 0.3275 |
| chunk_hit@10 | 0.3667 | 0.4333 | 0.3667 | 0.4333 |
| chunk_hit@20 | 0.4000 | 0.4667 | 0.5333 | 0.4333 |
| chunk_hit@50 | 0.4667 | 0.5333 | 0.7333 | 0.5333 |
| chunk_hit@100 | 0.5000 | 0.5667 | 0.8000 | 0.5667 |
| nearby_chunk_hit@10 | 0.3667 | 0.5667 | 0.4000 | 0.5333 |
| nearby_chunk_hit@100 | 0.6000 | 0.8333 | 0.8667 | 0.8333 |

## 判断

1. 标准检索下 `chunk_hit@100 = 0.5000`，说明很多目标 chunk 不是简单排在 top10 后面，而是没有稳定进入候选池。
2. `lexical_backfill` 对 top10 证据召回最有效，`nearby_chunk_hit@10` 从 0.3667 提升到 0.5667。
3. `neighbor_backfill` 对深层候选召回最有效，`chunk_hit@100` 从 0.5000 提升到 0.8000，但 `chunk_hit@10` 没有提升。
4. `lexical + neighbor` 未超过单独 lexical，说明把邻近 chunk 直接放进同一个排序池会引入排序污染。

## 当前结论

lexical backfill 适合进入默认检索路径，用于提升 top10 证据召回；UI 仍保留开关，便于在泛查询或延迟敏感场景下关闭。

neighbor backfill 不适合直接接入主检索 top10 排序；它更适合做答案上下文扩展：先拿到 top evidence，再把同论文、同页、`global_order` 或 `order_in_page` 接近的 chunk 一并提供给 Agent。

这一策略已在 Agent 层落地：Agent 会先补已命中论文内的 deeper evidence，再补邻近 chunk；这些上下文会进入最终综合回答，但不会进入 tool 返回结果，也不会改变主检索排序。当前默认总共最多追加 5 个上下文 chunk，以控制噪音和 token 成本。

## 已实现

- `QdrantStore.search(..., lexical_backfill=False)`：store 层仍默认关闭，由上层决定是否启用。
- `QdrantStore.search(..., neighbor_backfill=False)`：默认关闭，仅用于实验。
- `QdrantStore.search_evidence(...)` 支持两个实验开关。
- `evals/retrieval_eval.py --lexical-backfill`
- `evals/retrieval_eval.py --neighbor-backfill`
- 手动检索和 Agent 检索主路径已默认启用 `lexical_backfill`，并保留显式开关。
- Agent 综合回答阶段已自动追加邻近 chunk 上下文。
- Agent 综合回答阶段已自动追加论文内 deeper evidence 上下文。
- `app_utils.qdrant_search()` 和 `LiteratureAgent._execute_tool()` 会记录检索耗时、命中数、过滤条件和开关状态。
- `lexical_backfill` 已改为基于 `search_text` 的内存倒排索引，并优先使用低频关键词生成候选池，避免对全库逐条打分。
- `retrieval_eval.py` 已记录 `elapsed_ms_avg`、`elapsed_ms_p50`、`elapsed_ms_p95`，用于评估召回收益和耗时成本。

## 索引化 lexical 结果

top10 主路径规模评测：

| 指标 | standard top10 | indexed lexical top10 | lexical + qwen3-rerank top10 |
| --- | ---: | ---: | ---: |
| elapsed_ms_p50 | 560.7182 | 744.9653 | 1620.7301 |
| elapsed_ms_p95 | 647.8909 | 900.1438 | 2259.2596 |
| paper_hit@10 | 0.7833 | 0.9833 | 1.0000 |
| chunk_hit@1 | 0.2667 | 0.3000 | 0.5333 |
| chunk_hit@10 | 0.4000 | 0.6000 | 0.6667 |
| nearby_chunk_hit@10 | 0.4000 | 0.6333 | 0.7000 |

top100 深层评测中，indexed lexical 保持 `chunk_hit@10 = 0.4667`、`nearby_chunk_hit@10 = 0.6000`、`chunk_hit@100 = 0.5667`、`nearby_chunk_hit@100 = 0.8333`。候选池 p50 从接近半库的 9214 降到约 2838，质量没有下降。

## Paper-local evidence 实验

新增 `paper-local` 策略：先定 top 论文，再在这些论文内部用 `search_text` 重新找 evidence chunk。这个方向验证了“深层候选”确实还有空间，但目前不适合 top10 主路径：

| 指标 | indexed lexical top10 | paper-local p5 top10 | paper-local top100 |
| --- | ---: | ---: | ---: |
| elapsed_ms_p50 | 716.4640 | 1330.6918 | 5931.5313 |
| paper_hit@10 | 0.9833 | 0.9500 | 0.9833 |
| chunk_hit@10 | 0.5667 | 0.5333 | 0.4333 |
| nearby_chunk_hit@10 | 0.6333 | 0.6000 | 0.4667 |
| chunk_hit@100 | - | - | 0.7333 |
| nearby_chunk_hit@100 | - | - | 0.8000 |

结论：paper-local 能把更多目标 chunk 拉进 top100，但会污染 top10 排序，并且耗时明显更高。它更适合作为 Agent 深挖上下文或离线诊断策略，而不是替换 indexed lexical 主检索。对 paper-local 加 IDF 权重后，top10 指标没有实质改善。

## 下一步建议

P0：入库时生成干净的 `search_text`，把标题、正文、表格、产品、菌株、基因、酶名、摘要和邻近上下文显式合并，减少 raw chunk 噪音。

已完成 payload-only 回填：12865 个现有 Qdrant point 已写入 `search_text`，没有重新解析 PDF，也没有重新计算 embedding。当前 60 条 top10 评测中，回填并保护原始召回锚点后 `chunk_hit@10` 达到 0.6000，`nearby_chunk_hit@10` 达到 0.6333。

P0：继续观察索引化 `lexical_backfill` 的真实 UI 查询耗时和误召回。当前已替代全量扫描，并已作为手动检索/Agent 检索默认路径；后续仍应验证更大库规模和泛查询表现。

P1：继续评测 Agent 的 evidence context expansion：确认追加邻近 chunk 后，最终答案引用是否更完整、更少跑题。

当前上下文级代理评测显示：在默认 lexical 召回和锚点保护开启时，加入 score-ranked paper-local + neighbor context 后，exact `context_chunk_hit@10` 从 0.5000 提升到 0.6333，`context_nearby_hit@10` 从 0.5667 提升到 0.6667。因此后续重点应是继续提升初始 top evidence 和论文内 evidence scoring，而不是单纯增加邻近 chunk 数量。

P1：继续扩大自动评测，加入“答案引用 chunk 是否同页相邻”和“top evidence 是否覆盖多个论文/多个证据区域”的指标。
