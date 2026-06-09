# 查询改写与多路召回评测

本实验用于验证“先标准化问题，再做多路召回”是否能提升 chunk 级召回。

当前实现使用 DeepSeek V4 Flash 将中文或英文自然语言问题改写为结构化检索对象：

- `normalized_query`：标准化英文科学检索表达。
- `products` / `organisms` / `genes` / `enzymes` / `metrics`：结构化实体。
- `intents`：问题意图，例如 `mechanism`、`engineering_strategy`、`pathway`。
- `evidence_preference`：证据偏好，取值为 `any`、`text` 或 `table`。
- `aliases`：术语别名，例如 `LNTII`、`lacto-N-triose II`。
- `variants`：多路召回候选 query。

## 命令

单路查询改写：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --query-rewrite --out reports/retrieval_eval_rewrite_lexical60_top10.json
```

查询改写 + 多路召回：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_multi_query_lexical60_top10.json
```

## 配置

需要设置：

```bash
DEEPSEEK_API_KEY=...
```

默认模型：

```text
deepseek-v4-flash
```

可用环境变量或命令行覆盖：

```bash
SORTPAPER_QUERY_REWRITE_MODEL=deepseek-v4-flash
python evals/retrieval_eval.py --query-rewrite --query-rewrite-model deepseek-v4-flash
```

## 当前策略

`--multi-query` 会自动先进行查询改写，然后使用：

- 原始 query
- `normalized_query`
- 结构化实体 query：products / organisms / genes / enzymes
- 表格证据 query：metrics + products / organisms / genes
- 别名 query：aliases
- 有剩余 route 配额时再使用有效 `variants`

多路结果会用加权 RRF 合并，但不会让改写 query 完全重排原始结果。当前有三层护栏：

- 原始 query route 只取原始 `top_k`，保持与 baseline 检索一致。
- 结果前段保护原始 query 的前排命中：`top_k<=5` 时保护前 3 个，`top_k=10` 时保护前 5 个；variants 主要竞争后半段补位。
- 尾部候选优先级为：原始 query 候选 > 跨 route 共识候选 > 同一锚定论文候选；单 route、跨论文的 variants 噪音会被过滤。

## 当前结论

第一次 smoke5 发现，简单多路 RRF 会把原始 query 已经命中的精确 chunk 挤出 top10。原因是 variants 引入了同主题但不同论文/不同位置的噪音 chunk。

改成“原始 query 前排保护 + variants 补尾部缺口”后，20 条自动 case 的指标与 baseline 基本持平：

| 指标 | lexical baseline smoke20 | multi-query protected smoke20 |
| --- | ---: | ---: |
| paper_mrr | 0.7310 | 0.7326 |
| chunk_mrr | 0.2500 | 0.2500 |
| nearby_chunk_mrr | 0.2803 | 0.2803 |
| paper_hit@10 | 1.0000 | 1.0000 |
| chunk_hit@10 | 0.4545 | 0.4545 |
| nearby_chunk_hit@10 | 0.5455 | 0.5455 |
| elapsed_ms_p50 | 713ms | 3357ms |

判断：当前多路召回已经从“会伤害 baseline”修到“基本不伤害 baseline”，并且 raw tail priority 微调后 chunk MRR 回到 baseline 水平；但它还没有证明能稳定提升召回率，而且延迟明显更高。因此它已接入 UI 和 Agent，但默认关闭，继续作为实验开关。

下一步应优先优化选择性触发和 paper-local evidence 定位：只有当原始 query 置信度不足、或已命中正确论文但证据 chunk 排名靠后时，再启用改写/多路召回或论文内重排。

## 2026-06-02：结构化 recall route

已把改写结果中的实体字段变成可解释的 recall route：

- `entity`: `products + organisms + genes + enzymes`
- `table_evidence`: `metrics + products + organisms + genes`
- `alias`: `aliases`

这一步补齐了 RAGFlow 对“关键词/同义词/字段化检索”的启发，也更接近 LightRAG/图式检索中先抽实体再扩展候选的做法。`retrieval_eval.py` 现在会把 `search_meta.routes` 写入每个 case 的报告，便于复盘具体是 raw、normalized、entity、table evidence 还是 variant 命中的候选。

小样本 hard smoke：

```bash
python evals/retrieval_eval.py --case-profile hard --max-cases 4 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --multi-query --selective-rerank --rerank-top-n 10 --out reports/retrieval_eval_qwen3_5paper_hard_multi_structured_smoke4.json
```

当前结果：`chunk_hit@5 = 1.0`、`chunk_hit@10 = 1.0`，但 `chunk_hit@1 = 0.25`。这次 smoke 中 DashScope rerank 返回 `Arrearage`，实际退回原始排序，因此不能作为 rerank 效果证明；它主要证明结构化 route 可生成、可记录、不会导致 top10 丢失。route 统计显示 4 个 case 中 `raw/normalized/variant` 各 4 次，`entity/table_evidence` 各 2 次。

## 2026-06-02：adaptive route limit

已给 `multi_query_search()` 增加 `adaptive_route_limit`，默认开启：

- 泛查询 / 标题式查询：压缩为 `raw + normalized`，减少不必要 variants。
- 证据型查询、表格/数值查询、实体字段充分的查询：保留完整 route limit。
- `retrieval_eval.py` 支持 `--no-adaptive-multi-query`，用于和固定 route limit 做对照。

小样本 hard smoke：

```bash
python evals/retrieval_eval.py --case-profile hard --max-cases 4 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_qwen3_5paper_hard_multi_adaptive_smoke4.json
python evals/retrieval_eval.py --case-profile hard --max-cases 4 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --multi-query --no-adaptive-multi-query --out reports/retrieval_eval_qwen3_5paper_hard_multi_full_smoke4.json
```

这 4 个 hard case 都被判定为 `evidence_or_structured_query`，因此 adaptive 没有压缩 route 数；两组都保持 `chunk_hit@5 = 1.0`、`chunk_hit@10 = 1.0`。泛查询压缩由单元测试覆盖，后续应在混合 title / metadata / hard profile 上评测 p50/p95，而不是只看 hard evidence。
