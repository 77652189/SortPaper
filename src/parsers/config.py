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
# OpenCV混合表格解析器配置
# ──────────────────────────────────────────────
@dataclass
class OpenCVTableConfig:
    """HybridTableParser / OpenCVTableDetector / VisionTableExtractor 参数。"""

    # ---- 检测阶段（OpenCV） ----
    DETECT_DPI: int = 150                  # 渲染分辨率（检测用，低分辨率快速扫描）
    BINARIZE_THRESHOLD: int = 200          # 二值化阈值（>此值视为白色背景）
    H_KERNEL_MIN_WIDTH_PX: int = 50        # 水平形态学内核最小宽度（像素）
    H_KERNEL_PAGE_RATIO: float = 0.20      # 水平内核宽度占页面宽度的最小比例
    DILATE_H_PX: int = 15                  # 水平膨胀（弥合断线），像素
    DILATE_V_PX: int = 1                   # 垂直膨胀（保持线条细）
    MIN_LINE_WIDTH_PX: int = 30            # 最短有效横线（像素，绝对值下限）
    MIN_LINE_WIDTH_RATIO: float = 0.15     # 最短有效横线占页面宽度比例
    X_OVERLAP_RATIO: float = 0.50          # 两线段x重叠度 ≥ 此值才归为同一表格
    MAX_TABLE_HEIGHT_PT: float = 500.0     # 一个表格允许的最大高度（PDF点）
    MIN_TABLE_HEIGHT_PT: float = 20.0      # 有效表格区域最小高度（PDF点），过滤章节分隔线
    MIN_LINES_PER_TABLE: int = 3           # 聚类内最少横线数（三线表必须=3，过滤双线边框）
    HEADER_ZONE_RATIO: float = 0.08        # 页面顶部此比例内的横线不参与聚类（期刊页眉装饰线）
    FOOTER_ZONE_RATIO: float = 0.05        # 页面底部此比例内的横线不参与聚类（页码/页脚线）
    MIN_TEXT_CHARS: int = 20               # 检测到的区域内至少需要的字符数，否则判定为图片区域

    # ---- 提取阶段（视觉模型） ----
    EXTRACT_DPI: int = 300                 # 裁剪渲染分辨率（高分辨率，文字清晰）
    MARGIN_RATIO: float = 0.05             # 裁剪边距（占bbox宽/高的比例）
    MAX_IMAGE_DIM: int = 1536              # 发送给API前的最大边长（超出则等比缩放）
    VISION_MODEL: str = "qwen-vl-max"     # DashScope视觉模型ID

    # ---- 整合阶段（交叉验证） ----
    ROW_TOLERANCE: int = 1                 # 行数差异容忍度（≤此值视为"一致"）
    MIN_VALID_ROWS: int = 2               # LayoutChunk的最小有效行数
    MIN_VALID_COLS: int = 2               # LayoutChunk的最小有效列数

    # ---- 图片区域过滤 ----
    IMAGE_OVERLAP_RATIO: float = 0.3       # 与嵌入栅格图片的重叠率 ≥ 此值 → 判定为图片区域

    # ---- 并发 ----
    MAX_WORKERS: int = 5                   # 视觉API并发数

    # ---- 网格文字检验（Qwen-VL 调用前快速门控） ----
    GRID_MIN_COLS: int = 2                 # 至少有多少个 x0 列箱含 ≥2 个词才视为表格
    GRID_WORD_BIN_PT: float = 5.0          # x0 坐标分箱宽度（PDF点），越小越精确


# ──────────────────────────────────────────────
# 单例导出
# ──────────────────────────────────────────────
TABLE_PARSER = TableParserConfig()
LAYOUT = LayoutConfig()
VISION = VisionParserConfig()
PYMUPDF = PyMuPDFParserConfig()
OPENCV_TABLE = OpenCVTableConfig()
