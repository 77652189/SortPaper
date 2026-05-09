<div align="center">

# 📄 SortPaper

**学术论文解析、质量评判与语义检索流水线**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector--db-DC382D)](https://qdrant.tech)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-chat%20%26%20v4--pro-4D6BFE)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**语言：**
[🇬🇧 English](README.md) &nbsp;|&nbsp;
🇨🇳 中文 &nbsp;|&nbsp;
[🇯🇵 日本語](README.ja.md) &nbsp;|&nbsp;
[🇰🇷 한국어](README.ko.md)

</div>

---

## 项目简介

**SortPaper** 是一个本地优先的学术论文处理流水线。它从 PDF 研究论文中提取结构化内容（文本、表格、图片），通过 LLM 裁判对每个内容块进行质量评估，将通过的块写入 Qdrant 向量数据库，并通过 Streamlit 图形界面提供交互式浏览和语义检索功能。支持单篇解析、一键入库和批量导入。

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| 📝 文本提取 | 基于 PyMuPDF 的版面感知文本分块，支持双栏检测 |
| 📋 表格检测 | pdfplumber + PyMuPDF + camelot 三引擎，自适应无边框表格 |
| 🖼️ 图片描述 | Qwen-VL-Max 视觉模型，子图独立识别，禁止跨子图混淆数据 |
| ⚖️ LLM 裁判 | DeepSeek-chat 评估每个块的质量，失败最多重试 1 次 |
| 💾 Qdrant 存储 | Hybrid Search（Dense + Sparse + RRF）+ qwen3-rerank 二次排序 |
| 📉 降级存储 | 表格解析质量差时保留原始数据（标记 degraded），参考文献误检丢弃 |
| 🔁 智能重试 | 已通过块自动跳过重判；图片重试用 DeepSeek 文字改写而非重读图 |
| 🖥️ 图形界面 | Streamlit 网页界面 + 一键入库 + 批量拖拽上传 + 向量库管理 |
| 📊 质量评估 | 四步评估：分类 → Map-Reduce 摘要 → 构建 chunk 上下文 → 入库 |
| 🤖 Agent 检索 | Qwen-plus 自主调用工具进行多轮语义检索并综合建议 |

## 🏗️ 架构

```
PDF
 │
 ▼
协调器
 │
 ├──► 文本 Worker  ──► Judge (DeepSeek-chat) ──┐
 ├──► 表格 Worker  ──► Judge (DeepSeek-chat) ──┤──► Merge ──► Qdrant
 └──► 图片 Worker  ──► Judge (DeepSeek-chat) ──┘
  (Qwen-VL-Max)      ▲                        │
                      └── Retry（文本改写）──────┘
```

**分层架构：**

- **解析层** — `PyMuPDFParser`、`TableParser`（pdfplumber + PyMuPDF + camelot）、`VisionParser`（qwen-vl-max）
- **评判层** — `LLMJudge`（DeepSeek-chat，章节感知提示词）
- **质量评估层** — `PaperQualityEvaluator`（分类 → Map-Reduce → chunk 上下文），Reduce 阶段使用 DeepSeek-v4-pro
- **存储层** — `QdrantStore`（Hybrid Search: Dense + Sparse + RRF，qwen3-rerank 二次排序）
- **编排层** — LangGraph 并行 fan-out/fan-in 状态机，智能 retry 跳过已通过块

## 🚀 快速开始

**1. 克隆项目并安装依赖**

```bash
git clone https://github.com/77652189/SortPaper.git
cd SortPaper
pip install -r requirements.txt
```

**2. 配置 API Keys**

```bash
echo "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx" > .env
echo "DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx" >> .env
```

> - **DashScope**：用于 text-embedding-v3（向量化）/ qwen-vl-max（图片解析）/ qwen3-rerank（重排序）/ qwen-plus（Agent 检索）。在 [DashScope 控制台](https://dashscope.aliyun.com) 获取。
> - **DeepSeek**：用于 Judge 评判 + 质量评估 Map/Reduce。在 [DeepSeek Platform](https://platform.deepseek.com) 获取。

**3. 启动 Qdrant**（默认连接 localhost:6333）

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**4. 启动图形界面**

```bash
streamlit run app.py
```

在浏览器中打开 **http://localhost:8501**。

## 🖥️ 界面使用

1. **选择 PDF** — 上传任意 PDF，或从内置示例论文中选择；支持批量拖拽上传
2. **选择模式：**
   - **快速预览** — 仅调用本地解析器，秒级出结果，无 API 费用
   - **完整流水线** — 运行 Judge，质量评估和入库需手动触发
   - **一键入库** — 解析 → Judge → 质量评估 → Qdrant 入库 全自动
3. **点击 🚀 开始解析**
4. 在标签页中查看结果：
   - 📊 概览 · 📝 文本块 · 🖼️ 图片 · 📋 表格 · 📐 PDF 重建 · 🔍 语义检索
5. **向量库管理**：侧边栏可查看已入库文献列表，支持按论文删除

## 📁 项目结构

```
SortPaper/
├── app.py                    # Streamlit 图形界面
├── main.py                   # CLI 批量处理入口
├── src/
│   ├── parsers/              # 各类解析器（PyMuPDF / pdfplumber / camelot / VL）
│   ├── judge/                # LLM 裁判 + 论文质量评估
│   ├── store/                # Qdrant 向量存储（Hybrid Search）
│   ├── agent/                # 文献检索 Agent（Qwen function calling）
│   └── graph/                # LangGraph 流水线编排
├── scripts/                  # 验证与调试脚本
└── data/
    ├── sample_papers/        # 示例 PDF
    └── results/              # 解析结果快照
```

## 🛠️ 技术栈

| 组件 | 技术 |
|---|---|
| 文本解析 | PyMuPDF (fitz) |
| 表格解析 | pdfplumber + PyMuPDF + camelot（三引擎自适应） |
| 图片描述 | Qwen-VL-Max（DashScope） |
| LLM 裁判 | DeepSeek-chat |
| 质量评估 | DeepSeek-chat（分类/Map）+ DeepSeek-v4-pro（Reduce） |
| 文本嵌入 | text-embedding-v3（DashScope，Dense + Sparse 双路） |
| 重排序 | qwen3-rerank（DashScope） |
| 向量存储 | Qdrant（Hybrid Search: Dense + Sparse + RRF） |
| Agent 检索 | Qwen-plus（DashScope Function Calling） |
| 流水线编排 | LangGraph |
| 图形界面 | Streamlit |

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
