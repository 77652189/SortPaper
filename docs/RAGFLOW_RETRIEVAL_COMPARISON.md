# PaperSort 与 RAGFlow 检索算法对比及召回率提升建议

本文只分析和评测检索路径，不修改业务逻辑。RAGFlow 参考源码来自 `infiniflow/ragflow` 当前 `main` 快照，commit: `e1403171f1c88ca82b5807317658be3335679e43`。

## 结论摘要

PaperSort 当前的核心瓶颈不是论文级召回，而是证据 chunk 级召回。已有评测显示：

| 路径 | cases | paper_hit@10 | chunk_hit@10 | nearby_chunk_hit@10 | p50 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 60 | 0.7833 | 0.4000 | 0.4000 | 560.7ms |
| indexed lexical backfill | 60 | 0.9833 | 0.5667 | 0.6333 | 696.4ms |
| lexical + qwen3-rerank | 60 | 1.0000 | 0.6667 | 0.7000 | 1620.7ms |
| multi-query smoke | 20 | 1.0000 | 0.4545 | 0.5455 | 3224.6ms |
| Agent score-ranked context | 60 | - | context_chunk_hit@10 0.6333 | context_nearby_hit@10 0.6667 | +5 context chunks |

最值得继续推进的方向是：把 RAGFlow 的“字段化全文检索 + 可调 hybrid 权重 + 候选池评测”思想引入 PaperSort 的评测层和可选实验层，而不是继续单纯扩大 rerank 或直接改默认业务路径。

## 当前 PaperSort 检索路径

PaperSort 的主检索入口是 `QdrantStore.search()`：

- DashScope provider 下写入 dense + sparse，查询时用 Qdrant `Prefetch` 分别取 dense/sparse，再用 `Fusion.RRF` 融合；OpenAI provider 下是 dense-only。
- 若启用 rerank、lexical backfill 或 neighbor backfill，候选池从 `limit * 2` 扩到 `limit * 8`。
- `lexical_backfill` 会用 `search_text` 或 rerank 文本构建内存倒排索引，并按 query terms 补召回候选。
- rerank 使用 DashScope `qwen3-rerank`，随后还有基于科学术语锚点的 query relevance boost。
- Agent 路径把 paper-local deeper evidence 和 neighbor context 作为回答上下文补充，不直接污染 tool search top10 排名。

关键代码：

- `src/store/qdrant_store.py:366`：`search()` 主入口。
- `src/store/qdrant_store.py:393`、`src/store/qdrant_store.py:413`：候选池倍数与 Qdrant RRF。
- `src/store/qdrant_store.py:1075`、`src/store/qdrant_store.py:1142`：lexical candidate 和内存倒排索引。
- `src/store/qdrant_store.py:1211`、`src/store/qdrant_store.py:1277`：qwen3-rerank 与 query relevance boost。
- `src/retrieval/multi_query.py:37`、`src/retrieval/multi_query.py:71`：query rewrite / multi-query route 融合。
- `src/agent/literature_agent.py:276`：Agent paper-local context expansion。

## RAGFlow 检索路径

RAGFlow 的 retrieval test 入口会接收 `similarity_threshold`、`vector_similarity_weight`、`top_k`、`rerank_id`、`keyword`、`cross_languages`、`toc_enhance`、`use_kg` 等参数，然后调用 `settings.retriever.retrieval()`。

它的核心差异在于全文检索不是补丁，而是第一阶段召回的一部分：

- `FulltextQueryer` 对查询做规范化、分词、词权重、同义词扩展、相邻词短语 boost，并查询多个字段。
- 字段权重很强：`important_kwd^30`、`question_tks^20`、`title_tks^10`、`content_ltks^2` 等。
- 文档引擎侧执行 keyword + vector 的 weighted fusion，默认 `vector_similarity_weight=0.3`，也就是全文权重约 0.7。
- 若启用 rerank，RAGFlow 用 rerank score 替代 vector score 的那部分，再与 term similarity 加权。
- 支持 TOC enhancement、children-to-parent chunk 合并、PageRank / tag rank feature、metadata filter、cross-language、KG 检索。

关键代码与文档：

- RAGFlow `api/apps/restful_apis/chunk_api.py:299`、`:333`：retrieval API 参数与主调用。
- RAGFlow `rag/nlp/query.py:32`、`:42`：字段权重、查询构造、同义词/短语扩展。
- RAGFlow `rag/nlp/search.py:562`、`:618`、`:631`、`:662`：hybrid score、rerank score、KNN score 融合。
- RAGFlow `rag/nlp/search.py:852`、`:916`：TOC enhancement 与 children-to-parent。
- RAGFlow 文档 `docs/guides/dataset/run_retrieval_test.md`：检索测试、hybrid similarity、rerank 与阈值说明。
- RAGFlow FAQ `docs/faq.mdx`：说明 RAGFlow 依赖 Elasticsearch/Infinity 的全文检索、短语搜索和高级排序能力。

## 关键差异

| 维度 | PaperSort | RAGFlow | 对 PaperSort 的启发 |
| --- | --- | --- | --- |
| 第一阶段召回 | Qdrant dense/sparse RRF 为主，lexical backfill 是后置补召回 | 全文检索和向量检索共同构成主召回 | 应评测“字段化全文候选池”是否能替代部分后置 backfill |
| 词面能力 | 自定义 query terms、alias expansion、IDF-like lexical score | 分词、词权重、同义词、短语 boost、字段 boost | 科学术语、基因、产物、表格指标应做字段化和短语权重 |
| 权重可调 | dense/sparse 用 RRF，lexical boost/rerank 较固定 | `vector_similarity_weight` 可调，默认更偏全文 | 增加离线 sweep：keyword/vector/rerank 权重对召回和延迟的影响 |
| rerank 角色 | qwen3-rerank 提升明显但成本高 | rerank 可替代 vector score 的一部分，并限制候选数 | 不建议盲目扩大 rerank；应控制候选池质量和 top_n |
| 结构上下文 | Agent 已有 paper-local + neighbor context | TOC enhance、children-to-parent、KG chunk | 对论文可评测 section/TOC、figure/table parent-child 聚合 |
| 评测 | 已有 auto weak-label eval | 官方强调 retrieval test 调参 | 应把 recall gate 固化，新增 query 类型和失败 bucket |

## 提高召回率的建议

### P0：先把评测做成稳定门槛

不要先改默认搜索逻辑。先把现有 `evals/retrieval_eval.py` 和 `evals/agent_context_eval.py` 扩成固定报告矩阵：

- baseline：standard、lexical、lexical+rerank、multi-query、paper-local context。
- 指标：`paper_hit@K`、`chunk_hit@K`、`nearby_chunk_hit@K`、MRR、p50/p95 latency。
- 分桶：title、metadata、chunk、table-only、metric query、gene/enzyme query、alias query。
- 失败样例输出：query、expected chunk、top10 chunks、matched route、lexical_score、rerank_score。

理由：当前已有数据证明 lexical 和 rerank 有收益，但 multi-query smoke 延迟较高且 chunk 收益不稳定；没有分桶前不应改默认策略。

### P0：评测字段化全文候选池

借鉴 RAGFlow 的字段权重，但先做离线实验，不接业务路径。可以在 eval 中构造一个 PaperSort lexical scorer：

- `title` / `paper_title`：高权重。
- `target_products`、`organisms`、`genes`、`enzymes`、`metrics`、`seo_terms`：最高权重。
- `table caption`、`table headers`、`figure description`：高权重。
- `content` / `raw_content`：中低权重。
- 邻近 chunk 的 section title / page context：低权重。

目标不是替换 Qdrant，而是验证“字段化 term score + 当前向量候选”是否能把 exact chunk 拉进 top20/top50。若 `chunk_hit@50` 明显提升，再考虑业务接入。

### P1：把 query rewrite 变成可解释的 recall route，而不是只生成自然语言 query

RAGFlow 的 Agent Retrieval tool 明确要求 query 是“最重要关键词/同义词”。PaperSort 已有 `QueryRewrite`，下一步应评测：

- raw query。
- normalized scientific query。
- entity-only query：products / organisms / genes / enzymes / metrics。
- table evidence query：metric + unit + product + strain。
- alias-expanded query：例如 LNTII / lacto-N-triose II。

融合时保留当前的 anchor protection，避免 variants 把原始 top results 挤掉。multi-query 当前 20-case smoke 的延迟 p50 约 3225ms，说明要优先优化 route 数和候选池，而不是默认全开。

### P1：将“邻接/论文内 evidence”继续放在上下文层

PaperSort 的 Agent score-ranked context 已经把 `context_chunk_hit@10` 做到 0.6333。RAGFlow 的 `retrieval_by_toc()` 和 `retrieval_by_children()` 提示：结构扩展更适合回答上下文，而不一定适合主排序。

建议继续评测：

- 命中论文后，按 `search_text` 在同论文内找 deeper evidence。
- 同页、相邻页、同 section 的 chunk 作为 answer context。
- table/figure 子 chunk 聚合到 parent chunk，但只在 Agent context 中启用。

### P2：评测可调 hybrid 权重

RAGFlow 暴露 `vector_similarity_weight`，默认 0.3，偏重关键词。PaperSort 当前 RRF 和 boost 权重相对固定，建议只在 eval 层做 sweep：

- dense/sparse/RRF baseline。
- lexical score weight：0.1、0.2、0.35、0.5。
- rerank score + lexical score 的混合比例。
- strict anchor query 与普通 query 分开评测。

验收标准应同时看 `chunk_hit@10` 和 p95 latency。若 p95 显著上升但 only nearby 提升，不应默认上线。

## 2026-06-02 进展：字段化 lexical tail rerank 实验

已新增 `--fielded-lexical-weight` 评测开关，用于在现有候选池上叠加 RAGFlow 式字段权重：

- 高权重字段：`target_products`、`organisms`、`genes`、`enzymes`、`metrics`、`aliases`、`seo_terms`。
- 中高权重字段：表格/图片 caption、Figure group 描述、标题、摘要、上下文。
- 低权重字段：正文和 `search_text`。

第一次小库 smoke 显示，字段化分数如果直接参与全量重排，会压低已经正确的 top1/MRR；这与 multi-query 早期噪音问题一致。因此实现已改成 **前排保护 + tail rerank**：保护当前排序前半段，字段化权重只影响尾部补位。

当前 `papers` 小库由 2 篇 MinerU 缓存论文重建，使用本地 Qwen3 embedding，共 111 个 chunk。smoke 对比：

| 路径 | cases | chunk_mrr | chunk_hit@1 | chunk_hit@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical baseline | 6 | 1.0000 | 1.0000 | 1.0000 |
| fielded weight 0.1, unprotected | 6 | 0.4250 | 0.2500 | 1.0000 |
| fielded weight 0.1, protected | 6 | 1.0000 | 1.0000 | 1.0000 |

结论：字段化权重可以继续作为评测层实验，但不能替换现有排序前排。下一步应在重建后的完整库上做 sweep：`0.01 / 0.03 / 0.05 / 0.1`，同时比较 `chunk_hit@10`、`chunk_hit@30`、`chunk_hit@100` 和 p95 延迟。

## 2026-06-02 进展：5 篇本地 Qwen3 小库 sweep

当前 `papers` collection 已用 5 篇 MinerU 缓存论文重建，共 360 个 chunk。所有入库 smoke 均命中：

| 样本 | chunks | smoke matched |
| --- | ---: | ---: |
| LNT engineering | 71 | 5/5 |
| Penicillin acylase | 40 | 6/6 |
| 90.pdf | 35 | 6/6 |
| 57.pdf | 40 | 6/6 |
| 62.pdf | 174 | 6/6 |

新增 `evals/retrieval_weight_sweep.py`，用于统一跑字段化权重矩阵并汇总核心指标。

`standard + lexical_backfill` 的 5 篇小库 sweep 中，baseline 已经达到 `chunk_hit@10 = 1.0`、`chunk_mrr = 1.0`，字段化权重 `0.01 / 0.03 / 0.05 / 0.1` 没有伤害指标，但也没有可证明的提升。

关闭 `lexical_backfill` 后，standard baseline 为 `chunk_mrr = 0.8583`、`chunk_hit@1 = 0.8`、`chunk_hit@10 = 1.0`；字段化 tail rerank 同样未改变这些指标。逐 case 分析显示 exact top1 失败通常已有 nearby evidence 在前排，例如 exact rank 3/4 但 nearby rank 1/3。因此当前结论是：

- 字段化 tail rerank 的前排保护是必要的，安全性比 unprotected 方案好。
- 5 篇小库弱标签过于容易，不能证明字段化权重提升召回。
- 下一轮应扩大到完整库或增加更难的自动 case；同时把 exact hit 和 nearby hit 分开解读，避免为了 exact top1 牺牲已经足够回答问题的邻近证据。

## 2026-06-02 进展：hard case profile

已新增 `--case-profile hard`，用于生成更接近真实问题的弱标签 case。它优先选择包含数值指标、table、结果证据的 chunk，并把查询改写成自然问题，例如 “Which evidence passage reports 3.42 g/L for ...?”。这避免旧 profile 直接把目标 chunk 的稀有词拼成 query，导致 baseline 过于容易。

5 篇本地 Qwen3 小库 hard profile 对比：

| 路径 | chunk_mrr | chunk_hit@1 | chunk_hit@3 | chunk_hit@10 | elapsed p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 0.3741 | 0.1000 | 0.4000 | 0.9000 | 427.7ms |
| lexical_backfill | 0.7233 | 0.6000 | 0.8000 | 1.0000 | 615.8ms |
| lexical + fielded tail rerank 0.01~0.1 | 0.7233 | 0.6000 | 0.8000 | 1.0000 | 500~530ms |
| lexical + qwen3-rerank top10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1435.7ms |
| lexical + selective qwen3-rerank top10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1266.9ms |

结论更新：

- hard profile 能明显区分策略，适合后续作为默认回归评测之一。
- `lexical_backfill` 对 hard evidence 定位提升显著，继续保留为主路径默认是合理的。
- 字段化 tail rerank 当前没有额外收益，暂时保留为评测层开关，不接入 UI 默认。
- qwen3-rerank 对 hard evidence top1 排序提升非常明显，但延迟约为 lexical 的 2~3 倍。
- 已新增 `--selective-rerank`：证据型自然问题触发 rerank，标题式查询不触发。当前 hard profile 中触发率为 10/12，保留 `chunk_hit@1 = 1.0`，同时避免 title case 精排。下一步应在完整库上验证选择性触发率和 p95 延迟。

## 2026-06-02 进展：结构化 query rewrite route

P1 的“把 query rewrite 变成可解释 recall route”已开始落地。`multi_query_search()` 现在除了 `raw`、`normalized`、`variant`，还会从 `QueryRewrite` 字段生成：

- `entity`: `products + organisms + genes + enzymes`
- `table_evidence`: `metrics + products + organisms + genes`
- `alias`: `aliases`

这对应 RAGFlow 的关键词/同义词/字段化检索思路，也给 LightRAG 式实体扩展留出了入口。为了避免早期 multi-query 的噪音问题，仍保留 raw query 前排保护和尾部候选 admission gate。`retrieval_eval.py` 现在会把 `search_meta.routes` 写入 case 结果，便于分析具体 route 是否贡献候选。

小样本 smoke：

| 路径 | cases | chunk_hit@1 | chunk_hit@5 | chunk_hit@10 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| structured multi-query smoke4 | 4 | 0.2500 | 1.0000 | 1.0000 | DashScope rerank 返回 `Arrearage`，实际退回原始排序 |

结论：结构化 route 已可解释、可评测，但当前不应默认开启。上一轮 12 case multi-query + selective rerank 在账号可用时达到 `chunk_hit@1 = 1.0`，但 p50 约 3.4s、p95 约 26s；这说明下一步应该做选择性触发和 route 数控制，而不是把多路召回无条件作为默认路径。

## 2026-06-02 进展：adaptive multi-query route limit

已新增 `adaptive_route_limit`，默认用于主路径和评测路径中的显式 multi-query：

- 泛查询 / 标题式查询压缩为 `raw + normalized`。
- 证据型、表格/数值型、结构化实体充分的查询保留完整 route limit。
- `search_meta.route_policy` 会记录 `compact_generic_query`、`evidence_or_structured_query`、`fixed` 等原因，UI 和评测报告均可查看。

4 条 hard evidence smoke 中，adaptive 和 `--no-adaptive-multi-query` 都保持 `chunk_hit@5 = 1.0`、`chunk_hit@10 = 1.0`；由于 4 条都是证据型问题，adaptive 均选择完整 route，因此不能用这组证明延迟下降。下一步应跑混合 profile 或真实 UI 查询集，重点观察 title/metadata/query rewrite 场景是否减少 route 数和 p95。

## 不建议直接做的事

- 不建议直接把 rerank 候选池继续放大。当前 rerank 路径已把 p50 拉到约 1621ms，multi-query smoke 更高。
- 不建议把 neighbor backfill 直接进入主 top10 排序。已有评测显示它更适合作为上下文扩展。
- 不建议为了模仿 RAGFlow 立刻引入 Elasticsearch/Infinity。PaperSort 当前规模下，先用离线字段化 scorer 验证收益更稳。
- 不建议只优化论文级召回。当前真正瓶颈是 evidence/chunk 级定位。

## 推荐下一步评测命令

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --out reports/retrieval_eval_compare_standard60.json
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --out reports/retrieval_eval_compare_lexical60.json
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --rerank --out reports/retrieval_eval_compare_lexical_rerank60.json
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_compare_ctx5.json
```

若后续要新增实验脚本，建议命名为 `evals/fielded_lexical_eval.py`，只读取 Qdrant payload 和现有 eval cases，输出 report，不接入 `QdrantStore.search()` 默认路径。
