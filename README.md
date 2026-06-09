<div align="center">

# SortPaper

**Academic paper parsing, quality evaluation, vector storage, and semantic retrieval**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-5B21B6)](https://qdrant.tech)
[![DashScope](https://img.shields.io/badge/DashScope-Embedding%20%7C%20Rerank%20%7C%20Vision-FF6A00)](https://dashscope.aliyun.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Judge%20%7C%20Quality-4D6BFE)](https://deepseek.com)

**Language:**
English &nbsp;|&nbsp;
[中文](README.zh.md) &nbsp;|&nbsp;
[日本語](README.ja.md) &nbsp;|&nbsp;
[한국어](README.ko.md)

</div>

---

## Overview

**SortPaper** is a local-first research paper processing tool. It parses PDF text, tables, and images into `LayoutChunk` records, evaluates chunk quality with an LLM judge, enriches papers with classification and summary metadata, and stores usable chunks in Qdrant for semantic search and agent-based synthesis.

The codebase is being refactored from large application files into clearer module boundaries. The current focus is evidence-oriented retrieval: finding which imported papers actually support an answer, where the evidence came from, and how reliable the source is.

## Features

| Feature | Description |
|---|---|
| Text parsing | Layout-aware PyMuPDF chunking with column and reading-order handling |
| Table parsing | pdfplumber / PyMuPDF / camelot strategies with region detection and quality control |
| Image parsing | qwen3-vl-plus descriptions for figures and subfigures |
| LLM judge | Chunk-level quality evaluation, low-value filtering, and degraded result preservation |
| Paper quality enrichment | Classification, Map-Reduce summaries, chunk context, product/organism/credibility metadata |
| Qdrant storage | Chunk-level vector storage, per-paper deletion, duplicate checks, payload updates |
| Semantic search | DashScope embeddings, Qdrant hybrid search, qwen3-rerank, and quality metadata display |
| Agent search | Qwen-plus tool-calling for multi-round literature search and synthesis |
| Streamlit UI | Single-paper parsing, one-click import, batch import, vector library management, search debugging |

## Architecture

```text
PDF
 |
 v
Streamlit UI
 |
 v
Pipeline Orchestration
 |
 +--> Text Parser  --> LLM Judge --+
 +--> Table Parser --> LLM Judge --+--> Merge --> Quality Eval --> Qdrant
 +--> Image Parser --> LLM Judge --+
                                      |
                                      v
                              Search / Agent Answer
```

Main layers:

| Layer | Key files | Responsibility |
|---|---|---|
| UI | `app.py`, `app_ui.py`, `app_sidebar.py` | Entry point, controls, result rendering, vector library operations |
| Orchestration | `app_pipeline.py`, `src/graph/pipeline_graph.py` | Preview, full pipeline, one-click import, batch processing, quality enrichment |
| Data model | `src/parsers/layout_chunk.py` | Shared chunk representation across text, tables, and images |
| Parsers | `src/parsers/*` | PDF text, table, and image extraction |
| Table modules | `src/parsers/table/*` | Region detection, extraction, cleanup, deduplication, fallback, judge metadata |
| Judge | `src/judge/*` | Chunk quality, table quality, paper-level evaluation |
| Store | `src/store/qdrant_store.py`, `src/store/chunk_storage.py` | Embedding, storage, search, rerank, payload updates |
| Agent | `src/agent/literature_agent.py` | Multi-round answer synthesis using retrieval tools |

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

Create `.env` in the project directory:

```bash
DASHSCOPE_API_KEY=your_dashscope_key
DEEPSEEK_API_KEY=your_deepseek_key
MINERU_API_KEY=your_mineru_key
```

Optional:

```bash
SORTPAPER_EMBEDDING_PROVIDER=dashscope
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
MINERU_API_BASE_URL=https://mineru.net
MINERU_MODEL_VERSION=vlm
```

Notes:

- The default embedding provider is `dashscope`, which keeps dense + sparse hybrid search in Qdrant.
- `qwen3-rerank`, `qwen3-vl-plus`, and `qwen-plus` also require `DASHSCOPE_API_KEY`.
- Judge and paper quality enrichment require `DEEPSEEK_API_KEY`.
- MinerU external extraction smoke tests require `MINERU_API_KEY`; `MINERU_MODEL_VERSION` defaults to `vlm`.
- Do not commit real `.env` files or API keys.

**3. Start Qdrant**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**4. Launch the UI**

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Parser Backends

The recommended parsing path is now `MinerU extraction (recommended)`: MinerU VLM parses the PDF into unified `LayoutChunk` records with page, bbox, table, and Figure group metadata. The historical self-built paths remain available as display and compatibility routes:

- MinerU one-click ingest: recommended path for PDF → MinerU → LayoutChunk → Qdrant.
- Legacy quick preview: old text/table preview path for comparison.
- Legacy full pipeline: old parser + Judge + VisionParser path for regression checks.
- Legacy one-click ingest: old Qdrant ingest path, kept until MinerU chunk storage is fully validated.

MinerU preview results can be stored manually with `入库 MinerU chunks`. MinerU chunks receive default storage verdicts, while figure/image embedding text prefers group-level VL descriptions, figure captions, and MinerU visual text.

Retention, hiding, and sealing rules are documented in [`docs/PARSER_BACKENDS.md`](docs/PARSER_BACKENDS.md).

## Retrieval Quality Notes

Semantic search can only answer from papers that were actually imported. If the primary paper is missing, the system may return reviews, citation snippets, or adjacent-topic papers even when reranking works correctly.

When search results look wrong, check in this order:

1. Whether the target paper exists in the vector library.
2. Whether quality enrichment has completed and payload fields such as `category`, `paper_summary`, `target_products`, and `organisms` are present.
3. Whether returned chunks come from primary experimental papers, reviews, or citation-only passages.
4. Only then tune the query, hybrid search, rerank, or UI filters.

For example, an LNT II question requires the actual LNT II primary paper, not only HMO reviews or LNT II citations inside 2'-FL papers.

Manual search and Agent search now enable enhanced chunk recall by default, and manual search shows 10 results by default so the evaluated top10 evidence is visible immediately. This indexed lexical backfill uses `search_text` and low-frequency query terms to add evidence candidates without rescanning the full library, while preserving original top retrieval anchors. Without rerank, the current 60-case top10 evaluation improved `chunk_hit@10` from `0.4000` to `0.6000` and `nearby_chunk_hit@10` from `0.4000` to `0.6333`, with p50 latency moving from about `561ms` to `745ms`. With the UI-default qwen3-rerank path, `chunk_hit@10` reaches `0.6667` and `nearby_chunk_hit@10` reaches `0.7000`, with p50 latency around `1621ms`; the UI keeps switches to disable rerank or enhanced recall for broad or latency-sensitive queries.

Agent synthesis also expands answer context from already-hit papers: it score-ranks paper-local deeper evidence from the top five hit papers first, then adds nearby chunks, without changing the tool-search ranking shown to the model. With a five-chunk context budget and per-paper limit of three, the current context eval improves `context_chunk_hit@10` from `0.5000` to `0.6333` and `context_nearby_hit@10` from `0.5667` to `0.6667`.

Manual search and Agent search expose search diagnostics in the UI, including retrieval routes, normalized query, entity fields, context-localization query, hit counts, latency, and per-result matched routes/queries. These diagnostics are intended for recall debugging, not for changing the persisted payload.

## Query Rewrite and Multi-Query Recall

SortPaper now includes an experimental query rewrite and multi-query recall path for evidence search.

- Query rewrite uses DeepSeek V4 Flash to normalize Chinese or informal questions into concise English scientific search queries.
- Multi-query recall fans out across the original query, the normalized query, and short variants, then merges results with a protected original-query anchor.
- The original-query top results and raw tail candidates are prioritized, while rewritten variants are admitted mainly when they agree across routes or belong to the same anchored paper.
- Agent paper-local context now uses a compact context query built from the normalized query plus products, organisms, genes, enzymes, metrics, aliases, and the first two variants. A `table` or `text` evidence preference only adds a small ranking bonus inside already-hit papers; it does not filter out the other content type.
- Manual search and Agent search expose these features as explicit UI switches. They are intentionally off by default.

Current smoke evaluation shows protected multi-query recall no longer hurts the lexical baseline, but it has not yet proven a stable recall gain and adds latency:

```text
lexical baseline smoke20:
chunk_hit@10 = 0.4545
nearby_chunk_hit@10 = 0.5455
elapsed_ms_p50 = 713ms

multi-query protected smoke20:
chunk_hit@10 = 0.4545
nearby_chunk_hit@10 = 0.5455
elapsed_ms_p50 = 3357ms
```

Run the evaluation with:

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_multi_query_lexical60_top10.json
```

Agent context evaluation can also follow the production path with rerank, query rewrite, and multi-query recall:

```bash
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --query-rewrite --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_query_rewrite60_ctx5.json
python evals/agent_context_eval.py --max-cases 60 --ks 1 3 5 10 --lexical-backfill --rerank --multi-query --expand-neighbor-context --expand-paper-local-context --neighbor-total-limit 5 --paper-local-paper-limit 5 --paper-local-total-limit 5 --paper-local-per-paper-limit 3 --out reports/agent_context_eval_multi_query60_ctx5.json
```

More details are in `evals/QUERY_REWRITE.md`.

## Project Structure

```text
SortPaper/
+-- app.py                         # Streamlit entry point
+-- app_sidebar.py                 # Sidebar and input controls
+-- app_ui.py                      # Result views, search UI, vector library UI
+-- app_pipeline.py                # Preview, pipeline, one-click import, quality enrichment
+-- app_utils.py                   # App-level save/load/search helpers
+-- app_config.py                  # Environment and shared config
+-- src/
|   +-- agent/                     # LiteratureAgent
|   +-- graph/                     # LangGraph pipeline
|   +-- judge/                     # LLM judge and paper quality evaluation
|   +-- parsers/
|   |   +-- table/                 # Table detection, extraction, cleanup, dedup, fallback
|   |   +-- layout_chunk.py        # Shared chunk model
|   +-- retrieval/                 # Query rewrite and multi-query recall helpers
|   +-- store/
|       +-- qdrant_store.py        # Qdrant collection, embedding, search, rerank
|       +-- chunk_storage.py       # Parsed-result storage boundary
+-- tests/                         # Unit tests
+-- data/
    +-- sample_papers/             # Sample PDFs
    +-- results/                   # Parse result snapshots
```

## Tests

```bash
pytest -q
```

Useful focused tests:

```bash
pytest tests/test_qdrant_point_ids.py -q
pytest tests/test_chunk_storage.py -q
pytest tests/table -q
```

## Developer Notes

- `app.py` remains the Streamlit entry point, while business logic is moving into `app_pipeline.py`, `app_ui.py`, and `src/*`.
- Table parsing is being modularized around `src/parsers/table/parser.py` plus helpers such as `dedup.py`, `region_chunks.py`, and `judge_metadata.py`.
- Parsed-result storage is isolated in `src/store/chunk_storage.py` so UI and orchestration code do not own Qdrant details.
- For onboarding and refactoring context, read `../docs/ONBOARDING.md`, `../docs/UNDERSTAND_CHAT_FINDINGS.md`, and `../docs/UNDERSTAND_DIFF_REVIEW.md` from the workspace root.
