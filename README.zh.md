<div align="center">

# 📄 SortPaper

**学术论文解析、质量评判与语义检索流水线**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4A90D9)](https://github.com/langchain-ai/langgraph)
[![DashScope](https://img.shields.io/badge/DashScope-qwen--max-FF6A00)](https://dashscope.aliyun.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**语言：**
[🇬🇧 English](README.md) &nbsp;|&nbsp;
🇨🇳 中文 &nbsp;|&nbsp;
[🇯🇵 日本語](README.ja.md) &nbsp;|&nbsp;
[🇰🇷 한국어](README.ko.md)

</div>

---

## 项目简介

**SortPaper** 是一个本地优先的学术论文处理流水线。它从 PDF 研究论文中提取结构化内容（文本、表格、图片），通过 LLM 裁判对每个内容块进行质量评估，将通过的块嵌入 FAISS 向量库，并通过 Streamlit 图形界面提供交互式浏览和语义检索功能。

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| 📝 文本提取 | 基于 PyMuPDF 的版面感知文本分块，支持双栏检测 |
| 📋 表格检测 | pdfplumber 网格线模式解析，内置误检过滤规则 |
| 🖼️ 图片描述 | 调用 Qwen-VL-Max 视觉模型生成自然语言描述 |
| ⚖️ LLM 裁判 | qwen-max 评估每个块的质量，失败最多重试 3 次 |
| 💾 FAISS 存储 | 通过的块经 text-embedding-v3 嵌入后写入向量索引 |
| 📉 降级存储 | 部分块失败时，已通过的块仍会被保存 |
| 🖥️ 图形界面 | Streamlit 网页界面：上传、解析、浏览、语义检索 |

## 🏗️ 架构

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

## 🚀 快速开始

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

## 🖥️ 界面使用

1. **选择 PDF** — 上传任意 PDF，或从内置示例论文中选择
2. **选择模式：**
   - **快速预览** — 仅调用本地解析器，秒级出结果，无 API 费用
   - **完整流水线** — 运行 Judge + VisionParser + FAISS（消耗 API 配额）
3. **点击 🚀 开始解析**
4. 在五个标签页中查看结果：
   - 📊 概览 · 📝 文本块 · 🖼️ 图片 · 📋 表格 · 🔍 语义检索

## 📁 项目结构

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

## 🛠️ 技术栈

| 组件 | 技术 |
|---|---|
| 文本解析 | PyMuPDF (fitz) |
| 表格解析 | pdfplumber |
| 图片描述 | Qwen-VL-Max（DashScope） |
| LLM 裁判 | qwen-max（DashScope） |
| 文本嵌入 | text-embedding-v3（DashScope） |
| 向量存储 | FAISS |
| 流水线编排 | LangGraph |
| 图形界面 | Streamlit |

---

<div align="center">

Made with ❤️ &nbsp;·&nbsp; [GitHub Issues](https://github.com/77652189/SortPaper/issues)

</div>
