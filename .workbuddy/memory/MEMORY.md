# MEMORY.md - 项目长时记忆

## 交互规则
- **改代码前先确认**：先说明问题和方案，等用户同意再动手。信息不足时先问，不要自行假设。

## PaperSort 项目架构

### 当前设计（2026-04-29 重构）

基于 LayoutChunk 的统一解析 → 合并 → Judge → 存储流程。

**核心数据结构 `LayoutChunk`** (`src/parsers/layout_chunk.py`)：
- `content_type`: text | table | image
- `bbox`: (x0, y0, x1, y1) 原始 PDF 坐标
- `column`: 0=左栏, 1=右栏, 2=通栏（由 page_width + x0 推断）
- `order_in_page`: 本页阅读顺序
- `chunk_id`: 全局唯一 ID（基于类型_页码_栏_y_x 生成，跨重试稳定）

**Parser 改动**：
- `PyMuPDFParser` → 返回 `list[LayoutChunk]`，每个 span/block 一个 chunk
- `TableParser` → 返回 `list[LayoutChunk]`，每个表格一个 chunk，带 bbox
  - 三阶段探测：pdfplumber → PyMuPDF find_tables() → camelot stream
  - `_is_valid_table()` 过滤误检（页眉/参考文献/正文段落/稀疏表格等）
  - `_to_markdown()` 自动识别表头行（`_find_header_row()`）
  - `_merge_camelot_chunks()` 合并同一页面内被切割的表格
- `VisionParser` → 返回 `list[LayoutChunk]`，每个图片一个 chunk，带 bbox

**LayoutMerger** (`src/parsers/layout_chunk.py`)：
- 合并三个 Worker 的 chunks
- 调用 LayoutDeduplicator 去重
- 按 page → column → y0 → x0 排序
- 重建 order_in_page

**LayoutDeduplicator** (`src/parsers/layout_chunk.py`)：
- 基于 IoU + 内容相似度去重
- IoU 阈值 0.5，内容相似度阈值 0.8（Jaccard）
- 优先级：table > text > image
- 只在同一页内比较（跨页不会重叠）

**Judge**：`judge(chunk.content_type, chunk.raw_content, pdf_path)` 逐 chunk 评估

**Store**：只存 `passed=True` 的 chunks，metadata 包含完整位置信息

### 关键设计决策

1. **分块粒度**：解析后立即分块（非存储时），每个 span/block/table/image 独立 chunk
2. **位置保留**：bbox + column 全程传递，重试后 merged 排序不变
3. **Retry 机制**：全局重试（MAX_RETRIES=3），failed chunks 带 feedback 重新解析，passed chunks 保留
4. **Chunk ID 稳定性**：基于 title+page 生成，跨重试同位置 chunk 共享同一 ID
5. **去重策略**：同一区域被多个 Parser 提取时，通过 IoU + 内容相似度判定重复，table > text > image 优先级保留

### 配置化（2026-05-06）

所有 parser 的硬编码参数统一收拢到 `src/parsers/config.py`，使用 `dataclass` 组织，每条参数带中文注释。

配置文件结构：
- `TableParserConfig` — 表格验证阈值、关键词列表、合并参数、去重 IoU 阈值
- `LayoutConfig` — 去重/合并间距、页眉页脚比例、排序参数、章节标题集合
- `VisionParserConfig` — 图片面积阈值、VL 提示词、图注查找范围
- `PyMuPDFParserConfig` — 上标基线偏移比例

各 parser 已通过 `from src.parsers.config import XXX as CFG` 引用配置：
- `table_parser.py` → `TABLE_PARSER as CFG`
- `layout_chunk.py` → `LAYOUT as CFG`
- `vision_parser.py` → `VISION as CFG`
- `pymupdf_parser.py` → `PYMUPDF as CFG`
