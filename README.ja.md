<div align="center">

# 📄 SortPaper

**学術論文の解析・品質評価・セマンティック検索パイプライン**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![DashScope](https://img.shields.io/badge/DashScope-qwen--max-FF6A00)](https://dashscope.aliyun.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**言語：**
[🇬🇧 English](README.md) &nbsp;|&nbsp;
[🇨🇳 中文](README.zh.md) &nbsp;|&nbsp;
🇯🇵 日本語 &nbsp;|&nbsp;
[🇰🇷 한국어](README.ko.md)

</div>

---

## 概要

**SortPaper** はローカルファーストの学術論文処理パイプラインです。PDFの研究論文から構造化コンテンツ（テキスト・表・画像）を抽出し、LLM裁判官で各チャンクの品質を評価、合格チャンクをFAISSベクトルストアに格納し、StreamlitのGUIでインタラクティブな閲覧とセマンティック検索を提供します。

## ✨ 主な機能

| 機能 | 説明 |
|---|---|
| 📝 テキスト抽出 | PyMuPDFによるレイアウト対応テキスト分割（2段組み検出対応） |
| 📋 表検出 | pdfplumberの格線モード解析＋誤検出フィルタリング |
| 🖼️ 画像キャプション | Qwen-VL-Maxが自然言語の説明文を生成 |
| ⚖️ LLM裁判官 | qwen-maxで各チャンクを品質評価、失敗時は最大3回リトライ |
| 💾 FAISSストレージ | 合格チャンクをtext-embedding-v3で埋め込みインデックス化 |
| 📉 降格ストレージ | 一部チャンクが失敗しても合格分は確実に保存 |
| 🖥️ GUI | StreamlitのWebUI：アップロード・解析・閲覧・検索 |

## 🏗️ アーキテクチャ

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

## 🚀 クイックスタート

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

## 🖥️ GUIの使い方

1. **PDFを選択** — 任意のPDFをアップロード、またはサンプル論文から選択
2. **モードを選択：**
   - **クイックプレビュー** — ローカルパーサーのみ、即時結果、APIコストなし
   - **フルパイプライン** — Judge＋VisionParser＋FAISSを実行（APIクォータを使用）
3. **🚀 解析開始をクリック**
4. 5つのタブで結果を確認：
   - 📊 概要 · 📝 テキスト · 🖼️ 画像 · 📋 表 · 🔍 セマンティック検索

## 📁 プロジェクト構成

```
SortPaper/
├── app.py                    # Streamlit GUI
├── main.py                   # CLIバッチ実行
├── src/
│   ├── parsers/              # 各種パーサー
│   ├── judge/                # LLM裁判官 + プロンプト
│   ├── store/                # FAISSストア + テキスト分割
│   └── graph/                # LangGraphパイプライン
├── scripts/                  # 検証・デバッグスクリプト
└── data/sample_papers/       # サンプルPDF
```

## 🛠️ 技術スタック

| コンポーネント | 技術 |
|---|---|
| テキスト解析 | PyMuPDF (fitz) |
| 表解析 | pdfplumber |
| 画像キャプション | Qwen-VL-Max (DashScope) |
| LLM裁判官 | qwen-max (DashScope) |
| 埋め込み | text-embedding-v3 (DashScope) |
| ベクトルストア | FAISS |
| パイプライン | LangGraph |
| GUI | Streamlit |

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
