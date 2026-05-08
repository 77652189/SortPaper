<div align="center">

# 📄 SortPaper

**Academic paper parsing, quality judging, and semantic retrieval pipeline**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-5B21B6)](https://qdrant.tech)
[![DashScope](https://img.shields.io/badge/DashScope-Embedding+Generate-FF6A00)](https://dashscope.aliyun.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V4--flash-4D6BFE)](https://deepseek.com)
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
| 📋 Table detection | pdfplumber lattice-mode table parsing with false-positive filtering |
| 🖼️ Image captioning | Qwen-VL-Max vision model generates natural-language descriptions |
| ⚖️ LLM Judge | qwen-max evaluates every chunk for quality; failed chunks retry up to 3× |
| 💾 FAISS storage | Passing chunks are embedded (text-embedding-v3) and indexed for search |
| 📉 Degraded storage | Even if some chunks fail after retries, successful chunks are still saved |
| 🖥️ GUI | Streamlit web interface: upload, parse, browse chunks, semantic search |

## 🏗️ Architecture

```
PDF
 │
 ▼
Coordinator
 │
 ├──► Text Worker  ──► Judge Text  ──┐
 ├──► Table Worker ──► Judge Table ──┤──► Merge ──► FAISS Store
 └──► Image Worker ──► Judge Image ──┘
         ▲                           │
         └──── retry (max 3×) ───────┘
```

**Layers:**

- **Parser layer** — `PyMuPDFParser`, `TableParser`, `VisionParser`
- **Judge layer** — `LLMJudge` (qwen-max, section-aware prompts)
- **Store layer** — `FAISSStore` (text-embedding-v3, 1024-dim)
- **Orchestration** — LangGraph fan-out/fan-in state machine

## 🚀 Quick Start

**1. Clone & install dependencies**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. Configure DashScope API key**

```bash
# Create .env file
echo "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx" > .env
```

> Get your API key at [DashScope Console](https://dashscope.aliyun.com). A balance top-up of ¥20 is sufficient for processing ~10 papers.

**3. Launch the GUI**

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## 🖥️ GUI Usage

1. **Select PDF** — Upload any PDF or choose from the built-in sample papers
2. **Choose mode:**
   - **Quick Preview** — local parsers only, instant results, no API cost
   - **Full Pipeline** — runs Judge + VisionParser + FAISS (uses API quota)
3. **Click 🚀 Parse**
4. Browse results across five tabs:
   - 📊 Overview · 📝 Text · 🖼️ Images · 📋 Tables · 🔍 Semantic Search

## 📁 Project Structure

```
SortPaper/
├── app.py                    # Streamlit GUI
├── main.py                   # CLI batch runner
├── src/
│   ├── parsers/              # PyMuPDF, pdfplumber, VisionParser
│   ├── judge/                # LLMJudge + prompts
│   ├── store/                # FAISSStore + chunking
│   └── graph/                # LangGraph pipeline
├── scripts/                  # Verification & debug scripts
└── data/sample_papers/       # Sample PDFs
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Text parsing | PyMuPDF (fitz) |
| Table parsing | pdfplumber |
| Image captioning | Qwen-VL-Max (DashScope) |
| LLM Judge | DeepSeek V4-flash |
| Vision | qwen-vl-plus (DashScope) |
| Embedding | text-embedding-v3 (Dense + Sparse) |
| Reranker | qwen3-rerank |
| Vector store | Qdrant (Hybrid Search: Dense + Sparse + RRF) |
| Agent | Qwen Function Calling |
| Pipeline | LangGraph |
| GUI | Streamlit |

---

## 🧠 AI Contributors

> *A council of models, each bringing their unique brilliance to the pipeline.*

| Model | Role | Contribution |
|---|---|---|
| 🐋 **DeepSeek V4-flash** | Chief Judge & Quality Evaluator | Evaluates chunk quality, classifies papers, performs Map-Reduce summarization |
| 🧬 **text-embedding-v3** | Dual-Index Librarian | Generates dense + sparse vectors for Hybrid Search — captures both semantic meaning and precise terminology |
| 🎯 **qwen3-rerank** | Senior Editor | Re-ranks retrieved chunks, ensuring the most relevant literature appears first |
| 👁️ **qwen-vl-plus** | Visual Analyst | Describes figures, charts, and embedded images in papers |
| 🧠 **Qwen (Generation)** | Synthesis Agent | Function-calling agent that autonomously searches, filters, and synthesizes cross-paper recommendations |
| 🧭 **LangGraph** | Pipeline Orchestrator | Coordinates parallel parsing → judging → merging with retry logic |
| 🤖 **DeepSeek V4 Pro** | Development Companion | Real-time code generation, architecture design, and iterative refinement via WorkBuddy |

<div align="center">
  <sub>🐋 DeepSeek · 🧬 DashScope · 🎯 Qwen · 👁️ qwen-vl · 🧠 Qwen-Agent · 🧭 LangGraph · 🤖 WorkBuddy</sub>
</div>

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>







