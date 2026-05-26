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
- 最多 2 条有效 `variants`

多路结果会用加权 RRF 合并，但不会让改写 query 完全重排原始结果。当前有两层护栏：

- 原始 query route 只取原始 `top_k`，保持与 baseline 检索一致。
- 结果前段保护原始 query 的前排命中：`top_k<=5` 时保护前 3 个，`top_k=10` 时保护前 5 个；variants 主要竞争后半段补位。

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
| elapsed_ms_p50 | 713ms | 3225ms |

判断：当前多路召回已经从“会伤害 baseline”修到“基本不伤害 baseline”，但还没有证明能稳定提升召回率，而且延迟明显更高。因此它已接入 UI 和 Agent，但默认关闭，继续作为实验开关。

下一步应优先优化“variants 何时允许进入尾部结果”，例如只允许同论文补充、跨 route 共识命中、或低置信 raw 查询时才放宽补位。
