<div align="center">

# SortPaper

**学術論文の解析、品質評価、ベクトル保存、セマンティック検索**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid--Search-5B21B6)](https://qdrant.tech)
[![DashScope](https://img.shields.io/badge/DashScope-Embedding%20%7C%20Rerank%20%7C%20Vision-FF6A00)](https://dashscope.aliyun.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Judge%20%7C%20Quality-4D6BFE)](https://deepseek.com)

**言語：**
[English](README.md) &nbsp;|&nbsp;
[中文](README.zh.md) &nbsp;|&nbsp;
日本語 &nbsp;|&nbsp;
[한국어](README.ko.md)

</div>

---

## 概要

**SortPaper** はローカルファーストの論文処理ツールです。PDF からテキスト、表、画像を解析して `LayoutChunk` にまとめ、LLM Judge で品質を評価し、論文レベルの分類・要約・メタデータを付与して Qdrant に保存します。その後、セマンティック検索と Agent による回答生成に利用します。

現在のコードベースは、大きなアプリケーションファイルから明確なレイヤー構造へ整理中です。目的は単なる chunk 化ではなく、「どの論文が回答を支えているか」「証拠がどこにあるか」「信頼できるか」を追跡できる検索基盤を作ることです。

## 主な機能

| 機能 | 説明 |
|---|---|
| テキスト解析 | PyMuPDF によるレイアウト対応 chunk 化 |
| 表解析 | pdfplumber / PyMuPDF / camelot 系の戦略、領域検出、後処理、品質判定 |
| 画像解析 | qwen3-vl-plus による図・サブ図の説明生成 |
| LLM Judge | chunk 単位の品質評価、低価値 chunk の除外、degraded 結果の保持 |
| 論文品質評価 | 分類、Map-Reduce 要約、chunk context、産物・菌株・信頼度 metadata |
| Qdrant 保存 | chunk 単位のベクトル保存、論文単位削除、重複確認、payload 更新 |
| セマンティック検索 | DashScope embedding、Qdrant hybrid search、qwen3-rerank、品質 metadata 表示 |
| Agent 検索 | Qwen-plus の tool calling による複数ラウンドの文献検索と統合回答 |
| Streamlit UI | 単一論文解析、ワンクリック取込、バッチ取込、ベクトルライブラリ管理 |

## アーキテクチャ

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

| レイヤー | 主なファイル | 役割 |
|---|---|---|
| UI | `app.py`, `app_ui.py`, `app_sidebar.py` | 画面、入力、結果表示、ベクトルライブラリ操作 |
| オーケストレーション | `app_pipeline.py`, `src/graph/pipeline_graph.py` | プレビュー、フルパイプライン、ワンクリック取込、品質評価 |
| データモデル | `src/parsers/layout_chunk.py` | テキスト・表・画像共通の chunk 表現 |
| パーサー | `src/parsers/*` | PDF のテキスト、表、画像抽出 |
| 表モジュール | `src/parsers/table/*` | 領域検出、抽出、清掃、重複排除、fallback、Judge metadata |
| Judge | `src/judge/*` | chunk 品質、表品質、論文レベル評価 |
| Store | `src/store/qdrant_store.py`, `src/store/chunk_storage.py` | embedding、保存、検索、rerank、payload 更新 |
| Agent | `src/agent/literature_agent.py` | 検索 tool を使った統合回答 |

## クイックスタート

```bash
pip install -r requirements.txt
```

プロジェクトディレクトリに `.env` を作成します。

```bash
DASHSCOPE_API_KEY=your_dashscope_key
DEEPSEEK_API_KEY=your_deepseek_key
```

任意設定：

```bash
SORTPAPER_EMBEDDING_PROVIDER=dashscope
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

Qdrant を起動します。

```bash
docker run -p 6333:6333 qdrant/qdrant
```

UI を起動します。

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

## 検索品質について

セマンティック検索は、すでに取り込まれている論文からしか回答できません。主論文が未登録の場合、rerank が正常でも、レビュー論文、引用箇所、近いテーマの論文だけが返ることがあります。

検索結果が悪い場合は、まず対象論文がベクトルライブラリに存在するか確認してください。次に品質評価 metadata が付いているか、返ってきた chunk が原著論文・レビュー・引用箇所のどれかを確認し、その後で query、hybrid search、rerank、UI filter を調整します。

手動検索と Agent 検索では、enhanced chunk recall がデフォルトで有効です。この indexed lexical backfill は `search_text` と低頻度 query term を使って evidence 候補を補い、毎回ライブラリ全体を走査せず、元の上位検索アンカーも保護します。現在の 60-case top10 評価では `chunk_hit@10` が `0.4000` から `0.6000` に、`nearby_chunk_hit@10` が `0.4000` から `0.6333` に改善し、p50 latency は約 `561ms` から `745ms` になりました。広すぎる query や遅延が気になる場合は UI で無効化できます。

Agent synthesis では、すでにヒットした論文から paper-local deeper evidence を先に追加し、その後 nearby chunks を追加します。tool search のランキング自体は変更しません。5 chunks の context 予算、論文ごとの補足上限 3 件では、現在の context eval で `context_chunk_hit@10` が `0.5000` から `0.5667` に、`context_nearby_hit@10` が `0.5667` から `0.6000` に改善しました。

## プロジェクト構成

```text
SortPaper/
+-- app.py
+-- app_sidebar.py
+-- app_ui.py
+-- app_pipeline.py
+-- app_utils.py
+-- app_config.py
+-- src/
|   +-- agent/
|   +-- graph/
|   +-- judge/
|   +-- parsers/
|   |   +-- table/
|   |   +-- layout_chunk.py
|   +-- store/
|       +-- qdrant_store.py
|       +-- chunk_storage.py
+-- tests/
+-- data/
```

## テスト

```bash
pytest -q
```

## クエリ書き換えとマルチクエリ再現

SortPaper には、証拠 chunk の検索を改善するための実験的なクエリ書き換えとマルチクエリ再現パスがあります。

- クエリ書き換えは DeepSeek V4 Flash を使い、中国語、日本語、英語、または口語的な質問を短い英語の科学検索クエリに正規化します。
- マルチクエリ再現は、元の query、正規化 query、少数の variants を使って検索し、結果をマージします。
- 現在の融合戦略では、元の query の上位結果と raw tail 候補を優先し、variants は主に複数 route の一致、または同じアンカー論文に属する場合に後半を補います。
- 手動検索と Agent 検索の UI から明示的に有効化できますが、現時点ではデフォルト無効です。

smoke20 評価では、保護付きマルチクエリ再現は lexical baseline を悪化させなくなりましたが、安定した改善はまだ確認できておらず、遅延も増えます。

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

評価コマンド:

```bash
python evals/retrieval_eval.py --max-cases 60 --ks 1 3 5 10 --strategy standard --lexical-backfill --multi-query --out reports/retrieval_eval_multi_query_lexical60_top10.json
```

詳細は `evals/QUERY_REWRITE.md` を参照してください。
