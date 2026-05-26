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
```

Optional:

```bash
SORTPAPER_EMBEDDING_PROVIDER=dashscope
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

Notes:

- The default embedding provider is `dashscope`, which keeps dense + sparse hybrid search in Qdrant.
- `qwen3-rerank`, `qwen3-vl-plus`, and `qwen-plus` also require `DASHSCOPE_API_KEY`.
- Judge and paper quality enrichment require `DEEPSEEK_API_KEY`.
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

## Retrieval Quality Notes

Semantic search can only answer from papers that were actually imported. If the primary paper is missing, the system may return reviews, citation snippets, or adjacent-topic papers even when reranking works correctly.

When search results look wrong, check in this order:

1. Whether the target paper exists in the vector library.
2. Whether quality enrichment has completed and payload fields such as `category`, `paper_summary`, `target_products`, and `organisms` are present.
3. Whether returned chunks come from primary experimental papers, reviews, or citation-only passages.
4. Only then tune the query, hybrid search, rerank, or UI filters.

For example, an LNT II question requires the actual LNT II primary paper, not only HMO reviews or LNT II citations inside 2'-FL papers.

## Query Rewrite and Multi-Query Recall

SortPaper now includes an experimental query rewrite and multi-query recall path for evidence search.

- Query rewrite uses DeepSeek V4 Flash to normalize Chinese or informal questions into concise English scientific search queries.
- Multi-query recall fans out across the original query, the normalized query, and short variants, then merges results with a protected original-query anchor.
- The original-query top results are protected so rewritten variants can fill recall gaps without pushing already-good evidence out of the first page.
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
elapsed_ms_p50 = 3225ms
```

Run the evaluation with:

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_multi_query_lexical60_top10.json
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
