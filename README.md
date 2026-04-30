<div align="center">

# 📄 SortPaper

**Academic paper parsing, quality judging, and semantic retrieval pipeline**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![DashScope](https://img.shields.io/badge/DashScope-qwen--max-FF6A00)](https://dashscope.aliyun.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Language / 语言 / 言語 / 언어:**
[English](#english) · [中文](#中文) · [日本語](#日本語) · [한국어](#한국어)

</div>

---

<a id="english"></a>

## 🇬🇧 English

### Overview

**SortPaper** is a local-first academic paper processing pipeline. It extracts structured content from PDF research papers (text, tables, images), evaluates each chunk with an LLM judge, embeds passing chunks into a FAISS vector store, and surfaces them through a Streamlit GUI for interactive browsing and semantic search.

### ✨ Features

| Feature | Description |
|---|---|
| 📝 Text extraction | PyMuPDF-based layout-aware text chunking with column detection |
| 📋 Table detection | pdfplumber lattice-mode table parsing with false-positive filtering |
| 🖼️ Image captioning | Qwen-VL-Max vision model generates natural-language descriptions |
| ⚖️ LLM Judge | qwen-max evaluates every chunk for quality; failed chunks retry up to 3× |
| 💾 FAISS storage | Passing chunks are embedded (text-embedding-v3) and indexed for search |
| 📉 Degraded storage | Even if some chunks fail after retries, successful chunks are still saved |
| 🖥️ GUI | Streamlit web interface: upload, parse, browse chunks, semantic search |

### 🏗️ Architecture

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

### 🚀 Quick Start

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

### 🖥️ GUI Usage

1. **Select PDF** — Upload any PDF or choose from the built-in sample papers
2. **Choose mode:**
   - **Quick Preview** — local parsers only, instant results, no API cost
   - **Full Pipeline** — runs Judge + VisionParser + FAISS (uses API quota)
3. **Click 🚀 Parse**
4. Browse results across five tabs:
   - 📊 Overview · 📝 Text · 🖼️ Images · 📋 Tables · 🔍 Semantic Search

### 📁 Project Structure

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

### 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Text parsing | PyMuPDF (fitz) |
| Table parsing | pdfplumber |
| Image captioning | Qwen-VL-Max (DashScope) |
| LLM Judge | qwen-max (DashScope) |
| Embedding | text-embedding-v3 (DashScope) |
| Vector store | FAISS |
| Pipeline | LangGraph |
| GUI | Streamlit |

---

<a id="中文"></a>

## 🇨🇳 中文

### 项目简介

**SortPaper** 是一个本地优先的学术论文处理流水线。它从 PDF 研究论文中提取结构化内容（文本、表格、图片），通过 LLM 裁判对每个内容块进行质量评估，将通过的块嵌入 FAISS 向量库，并通过 Streamlit 图形界面提供交互式浏览和语义检索功能。

### ✨ 功能特性

| 功能 | 说明 |
|---|---|
| 📝 文本提取 | 基于 PyMuPDF 的版面感知文本分块，支持双栏检测 |
| 📋 表格检测 | pdfplumber 网格线模式解析，内置误检过滤规则 |
| 🖼️ 图片描述 | 调用 Qwen-VL-Max 视觉模型生成自然语言描述 |
| ⚖️ LLM 裁判 | qwen-max 评估每个块的质量，失败最多重试 3 次 |
| 💾 FAISS 存储 | 通过的块经 text-embedding-v3 嵌入后写入向量索引 |
| 📉 降级存储 | 部分块失败时，已通过的块仍会被保存 |
| 🖥️ 图形界面 | Streamlit 网页界面：上传、解析、浏览、语义检索 |

### 🏗️ 架构

```
PDF
 │
 ▼
协调器
 │
 ├──► 文本 Worker  ──► Judge 文本  ──┐
 ├──► 表格 Worker  ──► Judge 表格  ──┤──► 合并 ──► FAISS 存储
 └──► 图片 Worker  ──► Judge 图片  ──┘
         ▲                           │
         └──── 重试（最多 3 次）───────┘
```

**分层架构：**

- **解析层** — `PyMuPDFParser`、`TableParser`、`VisionParser`
- **评判层** — `LLMJudge`（qwen-max，章节感知提示词）
- **存储层** — `FAISSStore`（text-embedding-v3，1024 维）
- **编排层** — LangGraph 并行 fan-out/fan-in 状态机

### 🚀 快速开始

**1. 克隆项目并安装依赖**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. 配置 DashScope API Key**

```bash
echo "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx" > .env
```

> 在 [DashScope 控制台](https://dashscope.aliyun.com) 获取 API Key。充值 ¥20 可处理约 10 篇论文。

**3. 启动图形界面**

```bash
streamlit run app.py
```

在浏览器中打开 **http://localhost:8501**。

### 🖥️ 界面使用

1. **选择 PDF** — 上传任意 PDF，或从内置示例论文中选择
2. **选择模式：**
   - **快速预览** — 仅调用本地解析器，秒级出结果，无 API 费用
   - **完整流水线** — 运行 Judge + VisionParser + FAISS（消耗 API 配额）
3. **点击 🚀 开始解析**
4. 在五个标签页中查看结果：
   - 📊 概览 · 📝 文本块 · 🖼️ 图片 · 📋 表格 · 🔍 语义检索

### 📁 项目结构

```
SortPaper/
├── app.py                    # Streamlit 图形界面
├── main.py                   # CLI 批量处理入口
├── src/
│   ├── parsers/              # 各类解析器
│   ├── judge/                # LLM 裁判 + 提示词
│   ├── store/                # FAISS 存储 + 文本分块
│   └── graph/                # LangGraph 流水线
├── scripts/                  # 验证与调试脚本
└── data/sample_papers/       # 示例 PDF
```

---

<a id="日本語"></a>

## 🇯🇵 日本語

### 概要

**SortPaper** はローカルファーストの学術論文処理パイプラインです。PDFの研究論文から構造化コンテンツ（テキスト・表・画像）を抽出し、LLM裁判官で各チャンクの品質を評価、合格チャンクをFAISSベクトルストアに格納し、StreamlitのGUIでインタラクティブな閲覧とセマンティック検索を提供します。

### ✨ 主な機能

| 機能 | 説明 |
|---|---|
| 📝 テキスト抽出 | PyMuPDFによるレイアウト対応テキスト分割（2段組み検出対応） |
| 📋 表検出 | pdfplumberの格線モード解析＋誤検出フィルタリング |
| 🖼️ 画像キャプション | Qwen-VL-Maxが自然言語の説明文を生成 |
| ⚖️ LLM裁判官 | qwen-maxで各チャンクを品質評価、失敗時は最大3回リトライ |
| 💾 FAISSストレージ | 合格チャンクをtext-embedding-v3で埋め込みインデックス化 |
| 📉 降格ストレージ | 一部チャンクが失敗しても合格分は確実に保存 |
| 🖥️ GUI | StreamlitのWebUI：アップロード・解析・閲覧・検索 |

### 🏗️ アーキテクチャ

```
PDF
 │
 ▼
コーディネーター
 │
 ├──► テキストWorker  ──► Judge テキスト  ──┐
 ├──► 表Worker        ──► Judge 表        ──┤──► マージ ──► FAISSストア
 └──► 画像Worker      ──► Judge 画像      ──┘
         ▲                                  │
         └──── リトライ（最大3回）────────────┘
```

**レイヤー構成：**

- **パーサー層** — `PyMuPDFParser`、`TableParser`、`VisionParser`
- **判定層** — `LLMJudge`（qwen-max、セクション対応プロンプト）
- **ストア層** — `FAISSStore`（text-embedding-v3、1024次元）
- **オーケストレーション** — LangGraphファンアウト/ファインイン状態機械

### 🚀 クイックスタート

**1. クローン＆依存パッケージのインストール**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. DashScope APIキーの設定**

```bash
echo "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx" > .env
```

> [DashScopeコンソール](https://dashscope.aliyun.com)でAPIキーを取得してください。¥20のチャージで約10本の論文を処理できます。

**3. GUIの起動**

```bash
streamlit run app.py
```

ブラウザで **http://localhost:8501** を開いてください。

### 🖥️ GUIの使い方

1. **PDFを選択** — 任意のPDFをアップロード、またはサンプル論文から選択
2. **モードを選択：**
   - **クイックプレビュー** — ローカルパーサーのみ、即時結果、APIコストなし
   - **フルパイプライン** — Judge＋VisionParser＋FAISSを実行（APIクォータを使用）
3. **🚀 解析開始をクリック**
4. 5つのタブで結果を確認：
   - 📊 概要 · 📝 テキスト · 🖼️ 画像 · 📋 表 · 🔍 セマンティック検索

---

<a id="한국어"></a>

## 🇰🇷 한국어

### 개요

**SortPaper**는 로컬 우선 학술 논문 처리 파이프라인입니다. PDF 연구 논문에서 구조화된 콘텐츠(텍스트, 표, 이미지)를 추출하고, LLM 심판으로 각 청크의 품질을 평가하며, 합격한 청크를 FAISS 벡터 스토어에 저장하고, Streamlit GUI를 통해 대화형 탐색 및 시맨틱 검색을 제공합니다.

### ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 📝 텍스트 추출 | PyMuPDF 기반 레이아웃 인식 텍스트 청킹(2단 레이아웃 감지 포함) |
| 📋 표 감지 | pdfplumber 격자선 모드 파싱 + 오탐 필터링 |
| 🖼️ 이미지 캡션 | Qwen-VL-Max 비전 모델이 자연어 설명 생성 |
| ⚖️ LLM 심판 | qwen-max로 각 청크 품질 평가, 실패 시 최대 3회 재시도 |
| 💾 FAISS 저장소 | 합격 청크를 text-embedding-v3로 임베딩하여 인덱싱 |
| 📉 강등 저장 | 일부 청크가 실패해도 합격한 청크는 반드시 저장 |
| 🖥️ GUI | Streamlit 웹 인터페이스: 업로드, 파싱, 탐색, 시맨틱 검색 |

### 🏗️ 아키텍처

```
PDF
 │
 ▼
코디네이터
 │
 ├──► 텍스트 워커  ──► Judge 텍스트  ──┐
 ├──► 표 워커      ──► Judge 표      ──┤──► 병합 ──► FAISS 저장
 └──► 이미지 워커  ──► Judge 이미지  ──┘
         ▲                             │
         └──── 재시도 (최대 3회) ────────┘
```

**레이어 구성:**

- **파서 레이어** — `PyMuPDFParser`, `TableParser`, `VisionParser`
- **판정 레이어** — `LLMJudge` (qwen-max, 섹션 인식 프롬프트)
- **저장 레이어** — `FAISSStore` (text-embedding-v3, 1024차원)
- **오케스트레이션** — LangGraph 팬아웃/팬인 상태 머신

### 🚀 빠른 시작

**1. 클론 및 의존성 설치**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. DashScope API 키 설정**

```bash
echo "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx" > .env
```

> [DashScope 콘솔](https://dashscope.aliyun.com)에서 API 키를 발급받으세요. ¥20 충전으로 약 10편의 논문을 처리할 수 있습니다.

**3. GUI 실행**

```bash
streamlit run app.py
```

브라우저에서 **http://localhost:8501**을 열어주세요.

### 🖥️ GUI 사용법

1. **PDF 선택** — 임의의 PDF 업로드 또는 내장 샘플 논문 선택
2. **모드 선택:**
   - **빠른 미리보기** — 로컬 파서만 사용, 즉시 결과, API 비용 없음
   - **전체 파이프라인** — Judge + VisionParser + FAISS 실행 (API 할당량 사용)
3. **🚀 파싱 시작 클릭**
4. 다섯 개의 탭에서 결과 확인:
   - 📊 개요 · 📝 텍스트 · 🖼️ 이미지 · 📋 표 · 🔍 시맨틱 검색

---

<div align="center">

Made with ❤️ · [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
