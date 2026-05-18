"""
Parser 配置模块。

将各 parser 中需要调试和调整的参数统一收拢到此处。
使用方式： from src.parsers.config import TABLE_PARSER, LAYOUT, VISION, PYMUPDF
"""

from __future__ import annotations

from dataclasses import dataclass


# ──────────────────────────────────────────────
# TableParser 配置
# ──────────────────────────────────────────────
@dataclass
class TableParserConfig:
    """TableParser 参数。"""

    # 前两个解析器（pdfplumber + PyMuPDF）检测到的表格数量少于该值时，自动启用 camelot
    CAMELOT_TRIGGER_THRESHOLD: int = 2

    # ---- pdfplumber 参数 ----
    PDFPLUMBER_SNAP_TOLERANCE: int = 3
    PDFPLUMBER_JOIN_TOLERANCE: int = 3

    # ---- 表格有效性验证（仅结构检查，内容判断已迁移至 LLM Judge）----
    MIN_COLS: int = 2                     # 最少列数
    MIN_DATA_ROWS: int = 2                # 最少数据行数（除去表头后）
    MAX_CELL_CHARS: int = 200             # 单个单元格字符数上限；超出即视为正文段落误检
    BODY_TEXT_CELL_CHARS: int = 50        # 2列表格：任一非空单元格 ≥ 此长度 → 可能是正文
    BODY_TEXT_MIN_CELLS: int = 3          # 2列表格：≥ 此数量的疑似正文单元格 → 拒绝
    THREELINE_COL_GAP_PT: float = 8.0    # 三线表列检测：相邻列之间的最小间距（pt）

    # ---- Vision fallback：pdfplumber 质量不够时切 Vision 重解析 ----
    VISION_FALLBACK_ENABLED: bool = True   # 是否启用 Vision 兜底
    VISION_FALLBACK_MODEL: str = "qwen3-vl-plus"  # Vision 模型（Qwen3-VL 最新）
    VISION_FALLBACK_MIN_CONSISTENCY: float = 0.8  # 列数一致性低于此值触发 fallback
    VISION_FALLBACK_MIN_FILL_RATE: float = 0.4    # 填充率低于此值触发 fallback
    VISION_FALLBACK_DPI: int = 150        # 截图表区域时的渲染 DPI
    VISION_CAPTURE_MARGIN: int = 10       # 截图时 bbox 四周扩展 (pt)

    # ---- 列对齐表格边界检测 (_detect_columnar_table_bounds) ----
    COLUMN_ALIGN_X_TOLERANCE: int = 15    # 列 x0 对齐容差 (pt)
    COLUMN_ALIGN_MIN_SHARED: int = 2      # 连续行至少共享 N 列才算表格行
    COLUMN_ALIGN_MAX_GAP_ROWS: int = 1    # 允许的最大间隙行数（分区标题行如 "PCR primers"）

    # ---- 正文截断 (_truncate_at_body_text) ----
    TRUNCATE_SPARSE_COL_THRESHOLD: int = 5  # 策略2: max_nonempty 需 ≥ 此值才启用
    TRUNCATE_BODY_WORD_MIN_LEN: int = 30    # 策略3: 单元格最短长度
    TRUNCATE_BODY_WORD_MIN_COUNT: int = 4   # 策略3: 最少虚词匹配数

    # ---- 尾部裁剪 (_trim_tail_body_text) ----
    TAIL_TRIM_MIN_MODE_COLS: int = 3       # 众数列数 < 此值不检查
    TAIL_TRIM_LONG_CELL_CHARS: int = 50    # 长单元格阈值
    TAIL_TRIM_MIN_BODY_WORDS: int = 3      # 最少虚词匹配数

    # ---- bbox 精修 (_refine_table_bbox) ----
    REFINE_BBOX_SEARCH_UPWARD_RATIO: float = 0.4  # 碎片上方搜索起始偏移（页高比例）

    # ---- 表格合并 ----
    MERGE_X_OVERLAP_RATIO: float = 0.5    # x 方向重叠度 ≥ 此值才考虑合并
    MERGE_Y_GAP_MAX: float = 100.0        # y 方向间隙 ≤ 此值才考虑合并（pt）

    # ---- 去重 ----
    DEDUP_IOU_THRESHOLD: float = 0.5


# ──────────────────────────────────────────────
# LayoutChunk / LayoutMerger / LayoutDeduplicator 配置
# ──────────────────────────────────────────────
@dataclass
class LayoutConfig:
    """LayoutChunk 相关配置。"""

    # ---- infer_column ----
    FULL_WIDTH_RATIO: float = 0.6         # bbox 宽度 > page_width × 此比例 → 通栏

    # ---- LayoutDeduplicator ----
    IOU_THRESHOLD: float = 0.5
    CONTENT_SIMILARITY_THRESHOLD: float = 0.8
    CONTAINED_RATIO: float = 0.5          # text 面积 ≥ table 面积 × 此比例 → 包含

    # ---- LayoutMerger ----
    HEADER_MERGE_Y_GAP: float = 20.0      # 顶部相邻 text chunk 间隙 ≤ 此值合并
    HEADER_ZONE_RATIO: float = 0.35       # bbox[1] < page_height × 此值 → 页眉区
    TITLE_WORD_LIMIT: int = 12            # 词数 ≤ 此值 → 视为标题，不触发合并
    FOOTER_THRESHOLD_RATIO: float = 0.85  # y0 ≥ page_height × 此值 → 页脚
    LEFT_CONTENT_TOLERANCE: float = 50.0  # y 轴容差，判断右栏附近是否有左栏内容


# ──────────────────────────────────────────────
# VisionParser 配置
# ──────────────────────────────────────────────
@dataclass
class VisionParserConfig:
    """VisionParser 参数。"""

    VISION_MODEL: str = "qwen3-vl-plus"   # DashScope 视觉模型（qwen-vl-max 已下线）
    MIN_AREA_PT2: float = 5000.0          # PDF 坐标面积阈值（pt²）
    MIN_PIXEL_AREA: int = 10000           # 像素面积阈值（px²）
    CAPTION_SEARCH_RANGE: float = 80.0    # 图注查找范围（pt）
    CAPTION_CANDIDATES_MAX: int = 2       # 图注候选最多取前 N 条


# ──────────────────────────────────────────────
# PyMuPDFParser 配置
# ──────────────────────────────────────────────
@dataclass
class PyMuPDFParserConfig:
    """PyMuPDFParser 参数。"""

    SUPERSCRIPT_BASELINE_RATIO: float = 0.4  # 上标基线偏移比例


# ──────────────────────────────────────────────
# 单例导出
# ──────────────────────────────────────────────
TABLE_PARSER = TableParserConfig()
LAYOUT = LayoutConfig()
VISION = VisionParserConfig()
PYMUPDF = PyMuPDFParserConfig()
