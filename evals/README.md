# Retrieval Evaluation

这个目录提供自动化检索评测工具，目标是在不依赖人工生物领域标注的情况下，持续监控 PaperSort 的论文级召回和 chunk/evidence 级召回。

## 方法

`retrieval_eval.py` 会从 Qdrant 已入库 payload 自动生成弱标签 case：

- `title`：用论文标题作为 query，期望找回来源论文。
- `metadata`：用 `target_products`、`organisms`、`paper_summary` 等质量分析字段生成 query，期望找回来源论文。
- `chunk`：从 chunk 内容抽取关键词生成 query，期望找回来源论文和来源 chunk。

这些 case 不是人工 gold set，但适合做回归测试和优化方向判断。

## 常用命令

快速 smoke：

```bash
python evals/retrieval_eval.py --max-cases 20 --max-points 2000 --ks 1 3 5 10
```

标准 60 条评测：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 --strategy standard --out reports/retrieval_eval_standard60.json
```

增强 chunk 召回实验：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 50 100 --strategy standard --lexical-backfill --out reports/retrieval_eval_lexical_backfill60_top100.json
```

邻近 chunk 回填实验：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 50 100 --strategy standard --neighbor-backfill --out reports/retrieval_eval_neighbor_backfill60_top100.json
```

带 rerank：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 --rerank --out reports/retrieval_eval_rerank.json
```

论文优先的两阶段实验：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 20 --strategy two-stage --out reports/retrieval_eval_twostage.json
```

论文内 evidence 深挖实验：
```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy paper-local --lexical-backfill --paper-limit 5 --per-paper-limit 2 --lead-paper-limit 10 --out reports/retrieval_eval_paper_local_p5_lexical60_top10.json
```

## 指标

- `paper_hit@K`：前 K 条里是否找回来源论文。
- `chunk_hit@K`：前 K 条里是否找回来源 chunk。
- `nearby_chunk_hit@K`：前 K 条里是否找回来源 chunk 或其邻近 chunk。
- `paper_mrr`：来源论文首次出现排名的倒数均值。
- `chunk_mrr`：来源 chunk 首次出现排名的倒数均值。
- `nearby_chunk_mrr`：来源 chunk 或邻近 chunk 首次出现排名的倒数均值。
- `elapsed_ms_avg` / `elapsed_ms_p50` / `elapsed_ms_p95`：每个 case 的检索耗时，用于判断召回收益是否值得。
- `by_type`：按 `title`、`metadata`、`chunk` 拆分后的指标。

## 解读

如果 `paper_hit@10` 高但 `chunk_hit@10` 低，说明系统能找到相关论文，但证据段定位不准。

如果 `nearby_chunk_hit@10` 明显高于 `chunk_hit@10`，说明检索已经找到证据附近，exact chunk 评估可能偏严，或 chunk 切分粒度需要调整。

如果 `nearby_chunk_hit@10` 和 `chunk_hit@10` 都低，说明证据级候选召回不足，应优先优化检索文本、术语归一化或候选池构造。

`--lexical-backfill` 用于验证词面候选补召回。它已接入手动检索和 Agent 检索默认路径，并在 UI 中保留关闭开关；当前实现使用内存倒排索引和低频关键词候选池，避免每次全库扫描，同时会保护原始召回前排锚点。当前 60 条 top10 评测中，不启用 rerank 时 `chunk_hit@10` 从 0.4000 提升到 0.6000，`nearby_chunk_hit@10` 从 0.4000 提升到 0.6333，p50 耗时从约 561ms 增加到约 745ms。贴近手动 UI 默认的 `--lexical-backfill --rerank` 路径进一步达到 `chunk_hit@10 = 0.6667`、`nearby_chunk_hit@10 = 0.7000`，p50 耗时约 1621ms。后续仍应继续验证真实 UI 查询、泛查询误召回和更大库规模表现。

`--neighbor-backfill` 用于验证邻近 chunk 回填。它能显著提高 top20/top100 的 exact chunk 召回，但直接进入主排序会污染 top10，因此更适合作为 Agent 答案上下文扩展策略。

Agent 当前已使用这个思路：tool search 返回的命中排序不变，最终综合回答上下文会先额外加入已命中论文内的 deeper evidence，再加入同论文、同页或相邻顺序的 chunk。

`--strategy paper-local` 用于验证“已命中论文后，回到论文内部找 evidence chunk”。当前实验显示它能提升深层候选覆盖，例如 `chunk_hit@100 = 0.7333`，但 top10 排序低于 indexed lexical，且耗时更高。因此它不建议作为手动检索默认策略；更合适的用法是作为 Agent 回答上下文的 deeper evidence 来源。

回答上下文级评测：

```bash
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_score_rank_p5_p3_neighbor60_ctx5.json
```

这个评测不调用 LLM，只判断最终回答上下文中是否包含 exact/nearby evidence chunk。

回填现有库的 `search_text`：

```bash
python tools/backfill_search_text.py
python tools/backfill_search_text.py --apply
```

第一条命令是 dry-run，不写库；第二条只回写 Qdrant payload 的 `search_text` 字段，不重新解析 PDF，也不重新计算 embedding。
