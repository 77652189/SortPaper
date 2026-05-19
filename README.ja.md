<div align="center">

# 📄 SortPaper

**学術論文の解析・品質評価・セマンティック検索パイプライン**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-DC382D)](https://qdrant.tech)
[![DeepSeek](https://img.shields.io/badge/DeepSeek V4 Pro%20%26%20v4--pro-4D6BFE)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**言語：**
[🇬🇧 English](README.md) &nbsp;|&nbsp;
[🇨🇳 中文](README.zh.md) &nbsp;|&nbsp;
🇯🇵 日本語 &nbsp;|&nbsp;
[🇰🇷 한국어](README.ko.md)

</div>

---

## 概要

**SortPaper** はローカルファーストの学術論文処理パイプラインです。PDFから構造化コンテンツ（テキスト・表・画像）を抽出し、LLM判定で各チャンクの品質を評価、通過チャンクをQdrantベクトルデータベースに格納（Hybrid Search: Dense + Sparse + RRF + Rerank）、Streamlit GUIで閲覧・検索・一括インポートを提供します。

## ✨ 主な機能

| 機能 | 説明 |
|---|---|
| 📝 テキスト抽出 | PyMuPDFによるレイアウト対応テキスト分割（2段組み検出対応） |
| 📋 表検出 | pdfplumber + PyMuPDF + camelot 3エンジン、枠線なし表にも対応 |
| 🖼️ 画像キャプション | qwen3-vl-plusがサブ図を独立識別し自然言語説明を生成 |
| ⚖️ LLM判定 | DeepSeek V4 Proで各チャンクを品質評価、合格済みはリトライ時にスキップ |
| 💾 Qdrantストレージ | Hybrid Search + qwen3-rerank二次ソートで高精度検索 |
| 📉 降格ストレージ | 表の構造不良は「degraded」として保存、誤検出（参考文献等）は破棄 |
| 🔁 スマートリトライ | 画像リトライはDeepSeek文章書換（再読込不要）、表リトライはパーサー切替 |
| 🖥️ GUI | ワンクリック取込・バッチドラッグ＆ドロップ・ベクトルライブラリ管理 |
| 📊 品質評価 | 4段階：分類 → Map-Reduce要約 → チャンク文脈構築 → 保存 |
| 🤖 Agent検索 | Qwen-plusが自律的に関数呼出しで複数ラウンド検索・総合提案 |

## 🏗️ アーキテクチャ

```
PDF
 │
 ▼
コーディネーター
 │
 ├──► Text Worker   ──► Judge (DeepSeek) ──┐
 ├──► Table Worker  ──► Judge (DeepSeek) ──┤──► Merge ──► Qdrant
 └──► Image Worker  ──► Judge (DeepSeek) ──┘
  (qwen3-vl-plus)     ▲                      │
                     └── リトライ（書換）─────┘
```

**レイヤー構造：**

- **解析層** — PyMuPDFParser、TableParser（pdfplumber+PyMuPDF+camelot）、VisionParser（qwen3-vl-plus）
- **判定層** — LLMJudge（DeepSeek V4 Pro、セクション対応プロンプト）
- **品質評価層** — PaperQualityEvaluator（分類→Map-Reduce）、ReduceはDeepSeek V4 Pro
- **ストレージ層** — QdrantStore（Hybrid Search + Rerank）
- **オーケストレーション** — LangGraph 並列fan-out/fan-in + 合格スキップリトライ
         └──── リトライ（最大3回）────────────┘
```

**レイヤー構成：**

- **パーサー層** — `PyMuPDFParser`、`TableParser`、`VisionParser`
- **判定層** — `LLMJudge`（qwen-max、セクション対応プロンプト）
- **ストア層** — `FAISSStore`（text-embedding-v3、1024次元）
- **オーケストレーション** — LangGraphファンアウト/ファインイン状態機械

## 🚀 クイックスタート

**1. クローン＆依存パッケージのインストール**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. DashScope APIキーの設定**

```bash
cp .env.example .env
```

> [DashScopeコンソール](https://dashscope.aliyun.com)でAPIキーを取得してください。¥20のチャージで約10本の論文を処理できます。

**3. GUIの起動**

```bash
streamlit run app.py
```

ブラウザで **http://localhost:8501** を開いてください。

## 🖥️ GUIの使い方

1. **PDFを選択** — 任意のPDFをアップロード、またはサンプル論文から選択
2. **モードを選択：**
   - **クイックプレビュー** — ローカルパーサーのみ、即時結果、APIコストなし
   - **フルパイプライン** — Judge実行、品質評価・保存は手動トリガー
   - **ワンクリック取込** — 解析→判定→評価→保存、全自動
3. **🚀 解析開始をクリック**
4. タブで結果を確認：
   - 📊 概要 · 📝 テキスト · 🖼️ 画像 · 📋 表 · 📐 PDF復元 · 🔍 検索
5. **ベクトルライブラリ** — サイドバーで登録論文一覧表示・論文別削除

## 📁 プロジェクト構成

```
SortPaper/
├── app.py                    # Streamlit GUI
├── src/
│   ├── parsers/              # 各種パーサー（PyMuPDF/pdfplumber/camelot/VL）
│   ├── judge/                # LLM裁判官 + プロンプト
│   ├── store/                # Qdrantストア（Hybrid Search）
│   ├── agent/                # 文献検索Agent
│   └── graph/                # LangGraphパイプライン
└── data/sample_papers/       # サンプルPDF
```

## 🛠️ 技術スタック

| コンポーネント | 技術 |
|---|---|
| テキスト解析 | PyMuPDF (fitz) |
| 表解析 | pdfplumber + PyMuPDF + camelot（3エンジン） |
| 画像キャプション | qwen3-vl-plus (DashScope) |
| LLM判定 | DeepSeek V4 Pro |
| 品質評価 | DeepSeek V4 Pro（分類/Map）+ DeepSeek V4 Pro（Reduce） |
| 埋め込み | text-embedding-v3 (DashScope, Dense+Sparse) |
| リランカー | qwen3-rerank (DashScope) |
| ベクトルストア | Qdrant（Hybrid Search: Dense + Sparse + RRF） |
| Agent | Qwen-plus (DashScope Function Calling) |
| パイプライン | LangGraph |
| GUI | Streamlit |

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
