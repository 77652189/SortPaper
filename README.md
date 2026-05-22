<div align="center">

# 📄 SortPaper

**Academic paper parsing, quality judging, and semantic retrieval pipeline**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-5B21B6)](https://qdrant.tech)
[![DashScope](https://img.shields.io/badge/DashScope-Embedding+Generate-FF6A00)](https://dashscope.aliyun.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V4--flash%20%26%20Pro-4D6BFE)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Language:**
🇬🇧 English &nbsp;|&nbsp;
[🇨🇳 中文](README.zh.md) &nbsp;|&nbsp;
[🇯🇵 日本語](README.ja.md) &nbsp;|&nbsp;
[🇰🇷 한국어](README.ko.md)

</div>

---

## Overview

**SortPaper** is a local-first academic paper processing pipeline. It extracts structured content from PDF research papers (text, tables, images), evaluates each chunk with an LLM judge, classifies papers by quality, and stores passing chunks into a Qdrant vector database with Hybrid Search (Dense + Sparse + RRF) and optional Agent-based synthesis.

## ✨ Features

| Feature | Description |
|---|---|
| 📝 Text extraction | PyMuPDF-based layout-aware text chunking with column detection |
| 📋 Table detection | pdfplumber + PyMuPDF + camelot triple-engine, adaptive to borderless tables |
| 🖼️ Image captioning | qwen3-vl-plus with subfigure-aware prompts; text-rewrite on retry |
| ⚖️ LLM Judge | DeepSeek V4 Pro evaluates every chunk; already-passed chunks auto-skipped on retry |
| 💾 Qdrant storage | Hybrid Search (Dense + Sparse + RRF) + qwen3-rerank |
| 📉 Degraded storage | Tables with structural issues preserved as `degraded`; false positives discarded |
| 🔁 Smart retry | Image retry rewrites text via DeepSeek (no re-reading); table retry switches parser |
| 🖥️ GUI | Streamlit: one-click import, batch drag-and-drop, vector library management |
| 📊 Quality eval | 4-step: classify → Map-Reduce summary → chunk context → store |
| 🤖 Agent search | Qwen-plus autonomous tool-calling for multi-round semantic search & synthesis |

## 🏗️ Architecture

```
PDF
 │
 ▼
Coordinator
 │
 ├──► Text Worker  ──► Judge (DeepSeek) ──┐
 ├──► Table Worker ──► Judge (DeepSeek) ──┤──► Merge ──► Qdrant
 └──► Image Worker ──► Judge (DeepSeek) ──┘
  (qwen3-vl-plus)      ▲                   │
                      └── Retry (rewrite) ─┘
```

**Layers:**

- **Parser layer** — `PyMuPDFParser`, `TableParser` (3 engines), `VisionParser` (qwen3-vl-plus)
- **Judge layer** — `LLMJudge` (DeepSeek V4 Pro, section-aware)
- **Quality evaluation** — `PaperQualityEvaluator`: classify → Map-Reduce (Reduce uses DeepSeek V4 Pro)
- **Store layer** — `QdrantStore` (Hybrid Search + Rerank)
- **Orchestration** — LangGraph fan-out/fan-in with smart retry skipping

## 🚀 Quick Start

**1. Clone & install dependencies**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
```

> - **OpenAI**: default embedding provider, `text-embedding-3-large` at 3072 dimensions. Set `OPENAI_API_KEY`.
> - **DashScope**: optional legacy embeddings / qwen3-vl-plus (vision) / qwen3-rerank / qwen-plus (agent). Get key at [DashScope Console](https://dashscope.aliyun.com).
> - **DeepSeek**: Judge evaluation + Map-Reduce summarization. Get key at [DeepSeek Platform](https://platform.deepseek.com).

**3. Start Qdrant** (default: localhost:6333)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**4. Launch the GUI**

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## 🖥️ GUI Usage

1. **Select PDF** — Upload any PDF or pick from sample papers; supports batch drag-and-drop
2. **Choose mode:**
   - **Quick Preview** — PyMuPDF text + pdfplumber tables + LLM Judge, skips Qwen-VL and vector storage
   - **Full Pipeline** — runs Judge; quality eval & store triggered manually
   - **One-Click Import** — parse → judge → eval → store, fully automatic
3. **Click 🚀 Parse**
4. Browse results across tabs:
   - 📊 Overview · 📝 Text · 🖼️ Images · 📋 Tables · 📐 PDF Rebuild · 🔍 Search
5. **Vector Library** — sidebar shows indexed papers list with per-paper deletion

## 📁 Project Structure

```
SortPaper/
├── app.py                    # Streamlit GUI
├── src/
│   ├── parsers/              # PyMuPDF, pdfplumber, camelot, VisionParser
│   ├── judge/                # LLMJudge + PaperQualityEvaluator + prompts
│   ├── store/                # QdrantStore (Hybrid Search)
│   ├── agent/                # LiteratureAgent (Qwen function calling)
│   └── graph/                # LangGraph pipeline orchestration
└── data/
    ├── sample_papers/        # Sample PDFs
    └── results/              # Parse result snapshots
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Text parsing | PyMuPDF (fitz) |
| Table parsing | pdfplumber-only extraction + rule/LLM Judge quality control |
| Image captioning | qwen3-vl-plus (DashScope) |
| LLM Judge | DeepSeek V4 Pro |
| Quality eval | DeepSeek V4 Pro (classify+Map) + DeepSeek V4 Pro (Reduce) |
| Embedding | text-embedding-3-large (OpenAI, 3072d dense) |
| Reranker | qwen3-rerank (DashScope) |
| Vector store | Qdrant (dense vector search by default; DashScope provider keeps hybrid dense+sparse mode) |
| Agent | Qwen-plus (DashScope Function Calling) |
| Pipeline | LangGraph |
| GUI | Streamlit |

---

## 🧠 AI Contributors

> *A council of models, each bringing their unique brilliance to the pipeline.*

| Model | Role | Contribution |
|---|---|---|
| 🐋 **DeepSeek V4 Pro** | Judge & Map Evaluator | Evaluates chunk quality, classifies papers, performs Map-Reduce summarization |
| 🧬 **DeepSeek V4 Pro** | Chief Summarizer | Generates high-quality paper summaries in the Reduce step |
| 🧬 **text-embedding-3-large** | Dense Librarian | Generates 3072-dimensional vectors for semantic retrieval |
| 🎯 **qwen3-rerank** | Senior Editor | Re-ranks retrieved chunks for relevance |
| 👁️ **qwen3-vl-plus** | Visual Analyst | Describes figures, charts, subfigures with structured three-step analysis |
| 🧠 **Qwen-plus** | Synthesis Agent | Function-calling agent for autonomous multi-round search & synthesis |
| 🧭 **LangGraph** | Pipeline Orchestrator | Coordinates parallel parsing → judging → merging with smart retry |
| 🤖 **WorkBuddy** | Development Companion | Real-time code generation, architecture design, iterative refinement |

<div align="center">
  <sub>🐋 DeepSeek · 🧬 DashScope · 🎯 Qwen · 👁️ qwen-vl · 🧠 Qwen-Agent · 🧭 LangGraph · 🤖 WorkBuddy</sub>
</div>

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
