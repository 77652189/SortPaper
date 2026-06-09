# Retrieval Evaluation

## MinerU API smoke test

MinerU 已接入主解析路径；本目录中的脚本用于 smoke test、隔离入库验证和召回质量回归。需要在 `.env` 中配置：

```bash
MINERU_API_KEY=your_mineru_key
MINERU_API_BASE_URL=https://mineru.net
MINERU_MODEL_VERSION=vlm
```

公网 PDF URL 测试：

```bash
python evals/mineru_api_smoke.py --url https://example.com/paper.pdf --poll
```

本地 PDF 测试会先调用 `/api/v4/file-urls/batch` 获取 signed upload URL，再上传文件：

```bash
python evals/mineru_api_smoke.py --file data/sample_papers/example.pdf --poll
```

当前 smoke test 负责验证任务提交、文件上传、轮询与 zip 下载；主应用中的 `run_mineru_preview` / `run_mineru_ingest` 负责转换为 `LayoutChunk` 并入库。

## MinerU 入库与检索验证

`mineru_ingest_eval.py` 用于验证 MinerU 主路径是否真的能替代历史 parser 的入库路径。它会复用 `run_mineru_preview` 和 `store_parsed_chunks`，但把结果写入隔离的 Qdrant collection，避免污染主库 `papers`。

LNT 真实样本验证：

```bash
python evals/mineru_ingest_eval.py "data/mineru_cache/85a593e1722f022b/2022-AAA-Engineering Escherichia coli for the High-Titer Biosynthesis of Lacto-N-tetraose.pdf" --reset-collection --top-k 5 --out reports/mineru_ingest_eval_lnt.json
```

当前结果：

- 使用本地 MinerU 缓存，不重新调用 MinerU API。
- 生成 71 个 chunk：50 text、4 table、17 image。
- 71 个 chunk 全部保留 bbox 和页码。
- Figure group 数为 6。
- 隔离 collection `papers_mineru_eval` 中 71/71 成功入库，`missing_verdicts=0`。
- 5 个检索用例全部 Top-5 命中，覆盖标题、LNT titer、工程基因、Table 2、Figure/UDP-Gal 相关证据。

这项验证是封存历史解析代码前的主证据之一。若后续改变 MinerU chunk 转换、图片 VL 复解析或入库 metadata，应先复跑该脚本。

不同主题/版式 PDF 可以使用自动弱标签查询：

```bash
python evals/mineru_ingest_eval.py "data/mineru_cache/fd4719f529e7b241/1994-AA-Kinetic study of penicillin acylase production by recombinant E. coli in batch cultures(科研通-ablesci.com).pdf" --query-set auto --reset-collection --top-k 5 --out reports/mineru_ingest_eval_penicillin.json
```

当前 penicillin acylase 样本结果：

- 使用本地 MinerU 缓存，不重新调用 MinerU API。
- 生成 40 个 chunk：31 text、9 image。
- 40 个 chunk 全部保留 bbox 和页码，覆盖 1-10 页。
- Figure group 数为 9。
- 隔离 collection `papers_mineru_eval` 中 40/40 成功入库，`missing_verdicts=0`。
- 自动弱标签生成 6 个检索用例，全部 Top-5 命中。该结果用于回归 smoke，不等同于人工 gold set。

## MinerU Figure group 视觉复解析验证

`mineru_figure_vision_eval.py` 用于验证 MinerU 图片 chunk 的组级 VL 复解析是否可用。默认 dry-run 只统计 Figure group；加 `--call-vision` 会实际调用视觉模型并输出每组描述质量摘要。

dry-run：

```bash
python evals/mineru_figure_vision_eval.py reports/mineru_lnt_smoke.zip --out reports/mineru_figure_vision_eval_lnt_dry.json
```

真实 VL 调用：

```bash
python evals/mineru_figure_vision_eval.py reports/mineru_lnt_smoke.zip --call-vision --max-workers 2 --cache-path data/mineru_cache/85a593e1722f022b/figure_vision.json --out reports/mineru_figure_vision_eval_lnt.json
```

当前 LNT 样本结果：

- 6 个 Figure group，17 个 visual chunk。
- `--call-vision` 后 6/6 group 成功生成描述。
- 每组描述长度约 693-1943 字符。
- 描述覆盖关键检索词，包括 `LNT`、`lacto-N-tetraose`、`lgtA`、`wbgO`、`UDP-Gal`、`UDP-GlcNAc`、`galE`、`galT`、`galK`、`ugd` 等。
- 使用 `data/mineru_cache/85a593e1722f022b/figure_vision.json` 后，重复运行同一评测 6/6 group 命中缓存，耗时约 0.2 秒。

这说明 MinerU 图片路径不只是保留占位符，也可以在需要时生成检索友好的 Figure group 语义描述。后续若要封存旧 `VisionParser`，应至少保留这个评测作为回归门槛。

主流程中的 Figure group VL 会写入每篇论文的 MinerU cache 目录：`data/mineru_cache/<paper_id>/figure_vision.json`。重复解析同一 PDF、同一模型和同一组图片时会复用缓存，不会重复调用视觉模型。并发数由 `MINERU_FIGURE_VISION_MAX_WORKERS` 控制，默认 `2`，用于避免完全串行等待，同时给 API 限流留余地。

第二个 penicillin acylase 样本也已验证：

```bash
python evals/mineru_figure_vision_eval.py data/mineru_cache/fd4719f529e7b241/result.zip --call-vision --max-workers 2 --cache-path data/mineru_cache/fd4719f529e7b241/figure_vision.json --out reports/mineru_figure_vision_eval_penicillin.json
```

- 9 个 Figure group，9 个 visual chunk。
- 9/9 group 成功生成描述。
- 这是单图/单图表居多的版式，和 LNT 多 panel group 样本互补。
- 串行调用 9 个 group 明显更慢；后续若把 Figure group VL 作为常用路径，应考虑并发、限流和缓存复用。

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

深候选池 + 小窗口 rerank：

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --rerank --rerank-top-n 10 --out reports/retrieval_eval_rerank10_top100.json
```

`--rerank-top-n` 用于区分“候选池返回多少条”和“Rerank 精排前多少条”。例如 `top_k=100` 且 `--rerank-top-n 10` 时，会保留 100 条候选，但只要求 rerank 服务返回前 10 个重排结果，剩余候选按原始顺序接在后面。这样可以评估 `hit@30~100`，同时避免把 rerank 窗口误解成最终返回条数。

离线分析已有报告：

```bash
python evals/retrieval_report_analysis.py reports/retrieval_eval_seo_after60_rerank10_top100.json --baseline reports/retrieval_eval_seo_after60_top100.json --k 100 --out reports/retrieval_eval_seo_after60_rerank10_top100_analysis.json
```

这个脚本不调用 Qdrant、DashScope 或 rerank 服务，只读取 `retrieval_eval.py` 已生成的 JSON。它会把 chunk 失败拆成 `paper_miss`、`paper_hit_exact_miss_nearby_hit`、`paper_hit_exact_and_nearby_miss`、`exact_hit_only_after_top10`，并可输出相对 baseline 的指标差值。适合在外部 API 不可用时继续分析瓶颈。

字段化 lexical 候选池离线实验：

```bash
python evals/fielded_lexical_eval.py --max-cases 60 --ks 1 3 5 10 30 50 100 --out reports/fielded_lexical_eval60_top100.json
```

这个脚本借鉴 RAGFlow 的字段权重思想，只读取 Qdrant payload，不调用 embedding、rerank 或 LLM。当前实现对 `target_products`、`organisms`、`genes`、`enzymes`、`metrics`、`seo_terms`、表图描述、标题、摘要、上下文和正文分别加权。它不是业务路径，只用于验证“字段化全文候选池”是否值得和现有向量/RRF 结果融合。

字段化权重 sweep：

```bash
python evals/retrieval_weight_sweep.py --max-cases 60 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --weights 0 0.01 0.03 0.05 0.1 --out reports/retrieval_weight_sweep_lexical.json
```

这个脚本会多次调用 `retrieval_eval.py`，统一比较 `--fielded-lexical-weight` 对 `chunk_hit@K`、`nearby_chunk_hit@K`、MRR 和延迟的影响。字段化权重目前采用“前排保护 + tail rerank”：保护当前排序前半段，只让字段化分数影响尾部补位，避免 RAGFlow 式强关键词权重把已经正确的 top evidence 挤掉。

当前 5 篇本地 Qwen3 小库 smoke：

- `papers` collection 已由 5 篇 MinerU 缓存论文重建，共 360 个 chunk。
- `lexical_backfill` baseline 已达到 `chunk_hit@10 = 1.0`，字段化权重 `0.01~0.1` 未伤害指标，但也没有证明额外提升。
- 关闭 `lexical_backfill` 的 standard sweep 中，字段化 tail rerank 同样未改变 top1/top10；top1 exact 失败样例通常已有 nearby chunk 在前排，因此下一步应扩大样本或评测 rerank/上下文层，而不是继续放大字段权重。

Hard case profile：

```bash
python evals/retrieval_eval.py --case-profile hard --max-cases 60 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --out reports/retrieval_eval_hard_lexical.json
python evals/retrieval_eval.py --case-profile hard --max-cases 60 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --rerank --rerank-top-n 10 --out reports/retrieval_eval_hard_lexical_rerank10.json
python evals/retrieval_eval.py --case-profile hard --max-cases 60 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --selective-rerank --rerank-top-n 10 --out reports/retrieval_eval_hard_lexical_selective_rerank10.json
```

`--case-profile hard` 会优先从数值指标、表格和结果证据 chunk 生成自然语言问题，避免旧 profile 直接用目标 chunk 的稀有关键词造成过易评测。当前 5 篇 Qwen3 小库中，hard profile 显示：

- `standard`: `chunk_mrr=0.3741`, `chunk_hit@1=0.1000`, `chunk_hit@10=0.9000`
- `lexical_backfill`: `chunk_mrr=0.7233`, `chunk_hit@1=0.6000`, `chunk_hit@10=1.0000`
- `lexical + qwen3-rerank top10`: `chunk_mrr=1.0000`, `chunk_hit@1=1.0000`, `chunk_hit@10=1.0000`
- `lexical + selective qwen3-rerank top10`: `chunk_mrr=1.0000`, `chunk_hit@1=1.0000`, `chunk_hit@10=1.0000`, `selective_rerank_trigger_rate=0.8333`

因此 hard profile 更适合作为后续召回回归门槛；字段化 tail rerank 暂无收益，qwen3-rerank 收益明显但延迟更高。`--selective-rerank` 当前只对证据型自然问题触发，避免标题式查询也调用 rerank。

主路径已接入相同的 rerank 策略：

- 手动检索支持 `启用 Rerank` 和 `证据型问题自动 Rerank` 两种模式；前者始终精排，后者只在证据型查询命中规则时精排。
- Agent 检索也支持始终 rerank 或 selective rerank，避免工具路径和评测路径不一致。
- 多路召回会继续透传 `rerank_top_n`，因此可以用 top30~100 召回候选、只精排前 5~10 条来控制延迟。

结构化 query rewrite route：

```bash
python evals/retrieval_eval.py --case-profile hard --max-cases 4 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --multi-query --selective-rerank --rerank-top-n 10 --out reports/retrieval_eval_qwen3_5paper_hard_multi_structured_smoke4.json
```

`multi_query_search()` 会从 `QueryRewrite` 中生成 `entity`、`table_evidence` 和 `alias` route，并在报告的 `search_meta.routes` 中记录每个 case 实际使用的路线。当前 smoke 中 DashScope rerank 返回 `Arrearage`，因此该报告只能证明结构化 route 可生成、可记录、top10 未丢失，不能证明 rerank 效果。

自适应 route 数对照：

```bash
python evals/retrieval_eval.py --case-profile hard --max-cases 4 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_qwen3_5paper_hard_multi_adaptive_smoke4.json
python evals/retrieval_eval.py --case-profile hard --max-cases 4 --ks 1 3 5 10 30 50 100 --strategy standard --lexical-backfill --multi-query --no-adaptive-multi-query --out reports/retrieval_eval_qwen3_5paper_hard_multi_full_smoke4.json
```

`adaptive_multi_query` 默认开启，泛查询会压缩为 `raw + normalized`，证据型/表格型/结构化实体充分的查询保留完整 route。每个 case 的 `search_meta.route_policy` 会记录触发原因。

本地 Qwen3 embedding smoke：

```powershell
$env:SORTPAPER_EMBEDDING_PROVIDER="qwen3_local"
$env:SORTPAPER_QDRANT_COLLECTION="papers_qwen3_local"
$env:QWEN3_LOCAL_EMBEDDING_MODEL_PATH="data/models/Qwen3-Embedding-0.6B"
python -c "from src.store.qdrant_store import QdrantStore; s=QdrantStore.__new__(QdrantStore); v,_=s.embed('lacto-N-tetraose biosynthesis lgtA', is_query=True); print(len(v), round(sum(x*x for x in v), 4))"
```

本地模型已接入 `SORTPAPER_EMBEDDING_PROVIDER=qwen3_local`。注意它和 DashScope/OpenAI embedding 的向量空间不同，必须使用独立或重建后的 Qdrant collection，例如 `SORTPAPER_QDRANT_COLLECTION=papers_qwen3_local`。不要直接用本地 Qwen3 query embedding 搜索旧 `papers` collection，否则召回指标没有意义。

当前本地 Qwen3 smoke 结果：

- LNT 样本写入 `papers_mineru_eval_qwen3_lnt`：71/71 chunk 入库成功，5/5 固定检索用例命中，和 DashScope baseline 一样全部 matched rank=1。
- Penicillin acylase 样本写入 `papers_mineru_eval_qwen3_penicillin`：40/40 chunk 入库成功，6/6 自动检索用例命中，和 DashScope baseline 一样全部 matched rank=1。
- 这说明小样本上本地 Qwen3 embedding 可用，且未观察到检索质量下降；但 CPU 本地 embedding 明显较慢，全量重建前应保留隔离 collection 并跑完整 `retrieval_eval.py` 对比。

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

`--neighbor-backfill` 用于验证邻近 chunk 回填。它面向“已经找到正确论文，但证据 chunk 偏离”的情况，会补入同论文、同页或相邻顺序的 chunk。当前已作为手动检索和 Agent 检索的实验开关接入 UI，默认关闭；建议在需要深度证据搜索、`top_k=30~100`、或 `nearby_chunk_hit` 明显高于 `chunk_hit` 时开启。

最近一轮 60 条弱标签评测中，`--lexical-backfill --rerank --rerank-top-n 10 --ks 1 3 5 10 30 50 100` 达到 `chunk_hit@100 = 0.7667`、`nearby_chunk_hit@100 = 0.9000`，p50 耗时约 3006ms。它说明深候选池还有价值，但也说明该路径更适合“深度检索/证据搜索”，不应无条件作为轻量搜索默认路径。

Agent 当前已使用这个思路：tool search 返回的命中排序不变，最终综合回答上下文会先额外加入已命中论文内的 deeper evidence，再加入同论文、同页或相邻顺序的 chunk。

`--strategy paper-local` 用于验证“已命中论文后，回到论文内部找 evidence chunk”。当前实验显示它能提升深层候选覆盖，例如 `chunk_hit@100 = 0.7333`，但 top10 排序低于 indexed lexical，且耗时更高。因此它不建议作为手动检索默认策略；更合适的用法是作为 Agent 回答上下文的 deeper evidence 来源。

回答上下文级评测：

```bash
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_score_rank_p5_p3_neighbor60_ctx5.json
```

带查询改写的 Agent 上下文级评测：
```bash
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --query-rewrite --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_query_rewrite60_ctx5.json
```

带查询改写 + 多路召回的 Agent 上下文级评测：
```bash
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --multi-query --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_multi_query60_ctx5.json
```

启用查询改写或多路召回时，Agent 上下文评测会用 `normalized_query` 作为主检索表达，并把改写得到的产品、菌株、基因、酶、指标、别名和前 2 条 variants 合并成论文内 evidence 定位 query。

这个评测不调用 LLM，只判断最终回答上下文中是否包含 exact/nearby evidence chunk。

回填现有库的 `search_text`：

```bash
python tools/backfill_search_text.py
python tools/backfill_search_text.py --apply
```

第一条命令是 dry-run，不写库；第二条只回写 Qdrant payload 的 `search_text` 字段，不重新解析 PDF，也不重新计算 embedding。
