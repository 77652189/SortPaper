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

    # ---- 表格有效性验证 ----
    MIN_COLS: int = 2                     # 最少列数
    MIN_DATA_ROWS: int = 2                # 最少数据行数（除去表头后）
    HEADER_EMPTY_RATIO: float = 0.4       # 表头空格率 > 此值 → 伪表格
    DATA_EMPTY_RATIO: float = 0.6         # 数据行空格率 > 此值 → 稀疏伪表格
    TWO_COL_MAX_DATA_ROWS: int = 20       # 2 列 + 数据行超过此上限 → 双栏误检
    TWO_COL_MAX_AVG_LEN: int = 80         # 2 列 + 平均单元格长度超过此值 → 双栏误检
    COMPACT_UPPER_MIN_LEN: int = 8        # 连续大写无空格字符串长度 > 此值 → running title

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
