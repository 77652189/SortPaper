"""
混合表格解析器：OpenCV检测 + 视觉模型提取 + pdfplumber交叉验证。

三阶段流水线
  Stage 1（串行）：PyMuPDF + OpenCV 扫描每页，定位表格区域，渲染裁剪图片
  Stage 2（并发）：视觉模型提取 Markdown（主引擎）
                   pdfplumber 在相同坐标提取文本（副引擎，交叉验证）
  Stage 3（串行）：三档置信度整合

置信度规则
  high  ：两引擎行列数一致（|Δrow|≤1 且 col 完全相同）
           → 取 pdfplumber 文本（无OCR误差），metadata.confidence="high"
  medium：仅视觉模型成功（典型三线表）或仅 pdfplumber 成功
           → 取成功引擎结果，metadata.confidence="medium"
  low   ：两者均失败 → 不生成 chunk，记录警告日志

线程安全
  fitz.Document 仅在主线程（串行的 Stage 1）中读取。
  ThreadPoolExecutor 中只进行 API 调用和 pdfplumber 操作（各自内部打开文件）。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz

from src.parsers.config import OPENCV_TABLE as CFG
from src.parsers.layout_chunk import LayoutChunk, infer_column
from src.parsers.opencv_table_detector import OpenCVTableDetector
from src.parsers.vision_table_extractor import VisionTableExtractor

logger = logging.getLogger(__name__)


class HybridTableParser:
    """
    三阶段混合表格解析器，直接替换或补充 TableParser 中的现有引擎。

    使用方式：
        parser = HybridTableParser(pdf_path)
        chunks = parser.parse()   # 返回 LayoutChunk 列表
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)
        self.detector = OpenCVTableDetector(detect_dpi=CFG.DETECT_DPI)
        self.extractor = VisionTableExtractor(
            model=CFG.VISION_MODEL,
            detect_dpi=CFG.DETECT_DPI,
            extract_dpi=CFG.EXTRACT_DPI,
            margin_ratio=CFG.MARGIN_RATIO,
            max_dim=CFG.MAX_IMAGE_DIM,
        )

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def parse(self) -> list[LayoutChunk]:
        """解析 PDF 所有页面，返回 LayoutChunk 列表（仅含表格）。"""
        tasks = self._stage1_detect_and_render()
        if not tasks:
            logger.info("HybridTableParser: 未检测到任何表格区域")
            return []

        logger.info(
            "HybridTableParser: 检测到 %d 个候选表格区域，启动双引擎提取", len(tasks)
        )
        return self._stage2_and_3(tasks)

    # ── Stage 1：串行检测 + 渲染（主线程，fitz安全） ─────────────────────────

    def _stage1_detect_and_render(self) -> list[dict[str, Any]]:
        """
        遍历每页：OpenCV 检测表格 bbox，PyMuPDF 渲染裁剪图片字节。
        返回任务列表，每项包含后续处理所需的全部信息。
        """
        tasks: list[dict[str, Any]] = []

        with fitz.open(self.pdf_path) as doc:
            for page_idx, page in enumerate(doc, start=1):
                bboxes_px = self.detector.detect_page(page)
                if not bboxes_px:
                    continue

                pw = float(page.rect.width)
                ph = float(page.rect.height)

                for bbox_px in bboxes_px:
                    crop_bytes = self.extractor.render_crop(page, bbox_px)
                    if not crop_bytes:
                        logger.warning(
                            "p%d 裁剪渲染失败，跳过 bbox_px=%s", page_idx, bbox_px
                        )
                        continue
                    bbox_pdf = self.detector.bbox_to_pdf_pts(bbox_px)
                    tasks.append({
                        "page_idx": page_idx,
                        "bbox_px": bbox_px,
                        "bbox_pdf": bbox_pdf,
                        "crop_bytes": crop_bytes,
                        "page_width": pw,
                        "page_height": ph,
                    })

        return tasks

    # ── Stage 2+3：并发提取 + 整合 ───────────────────────────────────────────

    def _stage2_and_3(self, tasks: list[dict[str, Any]]) -> list[LayoutChunk]:
        """并发调用双引擎，整合结果。"""
        chunk_slots: list[LayoutChunk | None] = [None] * len(tasks)
        works = min(CFG.MAX_WORKERS, len(tasks))

        with ThreadPoolExecutor(max_workers=works) as pool:
            futures = {
                pool.submit(self._process_task, tasks[i]): i
                for i in range(len(tasks))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    chunk_slots[idx] = future.result()
                except Exception:
                    logger.exception(
                        "表格处理失败 p%d bbox=%s",
                        tasks[idx]["page_idx"], tasks[idx]["bbox_px"],
                    )

        return [c for c in chunk_slots if c is not None]

    def _process_task(self, task: dict[str, Any]) -> LayoutChunk | None:
        """
        单个任务处理：
          1. 网格文字检验（快速门控，~0ms，拒绝图表误检）
          2. 视觉模型（主引擎） + pdfplumber（副引擎，交叉验证）
          3. 三档置信度整合
        此函数在线程池中执行，不访问 fitz.Document。
        """
        # 快速门控：bbox 内文字是否构成多列网格（无网格 → 图表/图片误检，直接跳过）
        if not _has_grid_text(self.pdf_path, task["page_idx"], task["bbox_pdf"]):
            logger.debug(
                "网格文字检验未通过，跳过 Qwen-VL 调用 p%d bbox=%s",
                task["page_idx"], task["bbox_pdf"],
            )
            return None

        vision_md = self.extractor.call_api(task["crop_bytes"])
        pdfplumber_md = _extract_pdfplumber(
            self.pdf_path, task["page_idx"], task["bbox_pdf"]
        )
        return _merge_results(
            vision_md=vision_md,
            pdfplumber_md=pdfplumber_md,
            page_idx=task["page_idx"],
            bbox_pdf=task["bbox_pdf"],
            page_width=task["page_width"],
        )


# ── 模块级工具函数（线程池安全，无 fitz 依赖） ────────────────────────────────


def _has_grid_text(
    pdf_path: str,
    page_idx: int,
    bbox_pdf: tuple[float, float, float, float],
) -> bool:
    """
    检查 bbox 内文字是否呈现多列网格排列（Qwen-VL 调用前的快速门控）。

    算法：用 GRID_WORD_BIN_PT 宽度分箱聚类所有词的 x0 坐标，
    统计含 ≥2 个词的列箱数量。列箱数 ≥ GRID_MIN_COLS 时视为表格。

    典型结果：
      三线表   → 多列密集词语 → 通过（True）
      柱状图   → 少量标签，x0 分散 → 不通过（False）
      图注/标题 → 一列连续文字 → 不通过（False）

    异常处理：pdfplumber 不可用或出错时返回 True（放行，不阻断流水线）。
    """
    try:
        import pdfplumber
    except ImportError:
        return True  # 依赖缺失时放行

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx > len(pdf.pages):
                return False
            words = pdf.pages[page_idx - 1].within_bbox(bbox_pdf).extract_words()
    except Exception:
        logger.debug("_has_grid_text: pdfplumber 打开失败 p%d", page_idx, exc_info=True)
        return True  # 异常时放行

    if len(words) < 6:
        # 词数过少，排除是完整表格
        return False

    # 按固定宽度分箱聚类 x0 坐标，统计各列词数
    bin_pt = int(CFG.GRID_WORD_BIN_PT)
    bins: dict[int, int] = {}
    for w in words:
        b = round(w["x0"] / CFG.GRID_WORD_BIN_PT) * bin_pt
        bins[b] = bins.get(b, 0) + 1

    cols_with_multiple_words = sum(1 for cnt in bins.values() if cnt >= 2)
    result = cols_with_multiple_words >= CFG.GRID_MIN_COLS
    logger.debug(
        "_has_grid_text p%d bbox=%s: %d词 → %d有效列箱 → %s",
        page_idx, bbox_pdf, len(words), cols_with_multiple_words,
        "通过" if result else "过滤",
    )
    return result


def _extract_pdfplumber(
    pdf_path: str,
    page_idx: int,
    bbox_pdf: tuple[float, float, float, float],
) -> str:
    """
    用 pdfplumber 在给定 PDF 点坐标的 bbox 内提取表格（text+text 策略）。

    坐标系：bbox_pdf 使用 fitz 左上角原点坐标（y 从上往下递增），
    与 pdfplumber.within_bbox(x0, top, x1, bottom) 的 top/bottom 约定完全一致。
    """
    try:
        import pdfplumber
    except ImportError:
        return ""

    x0, y0, x1, y1 = bbox_pdf
    margin_x = (x1 - x0) * CFG.MARGIN_RATIO
    margin_y = (y1 - y0) * CFG.MARGIN_RATIO

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx > len(pdf.pages):
                return ""
            pdf_page = pdf.pages[page_idx - 1]
            pw = float(pdf_page.width)
            ph = float(pdf_page.height)

            clip = (
                max(0.0, x0 - margin_x),
                max(0.0, y0 - margin_y),
                min(pw, x1 + margin_x),
                min(ph, y1 + margin_y),
            )
            clipped = pdf_page.within_bbox(clip)
            tables = clipped.find_tables(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 5,
                "join_tolerance": 5,
            })
            if not tables:
                return ""
            rows = tables[0].extract()
            if not rows:
                return ""

            from src.parsers.table_parser import TableParser
            normalized = TableParser._normalize_table(rows)
            if not normalized:
                return ""
            return TableParser._to_markdown(normalized)

    except Exception:
        logger.debug(
            "pdfplumber副引擎提取失败 p%d bbox=%s", page_idx, bbox_pdf, exc_info=True
        )
        return ""


def _count_table_shape(md: str) -> tuple[int, int]:
    """从 Markdown 表格字符串统计（数据行数, 列数）。分隔行（|---|）不计入行数。"""
    if not md:
        return 0, 0
    lines = [ln for ln in md.strip().splitlines() if ln.strip().startswith("|")]
    data_lines = [
        ln for ln in lines
        if not all(c in "-:|" for c in ln.replace("|", "").strip())
    ]
    if not data_lines:
        return 0, 0
    cols = max(0, data_lines[0].count("|") - 1)
    return len(data_lines), cols


def _merge_results(
    vision_md: str,
    pdfplumber_md: str,
    page_idx: int,
    bbox_pdf: tuple[float, float, float, float],
    page_width: float,
) -> LayoutChunk | None:
    """
    三档置信度整合逻辑，返回最终 LayoutChunk。

    high   : 行列数一致 → pdfplumber 文本（无OCR误差）
    medium : 单引擎成功（视觉模型优先）
    None   : 两者均失败，记录警告
    """
    v_rows, v_cols = _count_table_shape(vision_md)
    p_rows, p_cols = _count_table_shape(pdfplumber_md)

    vision_ok = bool(vision_md) and v_rows >= CFG.MIN_VALID_ROWS and v_cols >= CFG.MIN_VALID_COLS
    pdfplumber_ok = bool(pdfplumber_md) and p_rows >= CFG.MIN_VALID_ROWS and p_cols >= CFG.MIN_VALID_COLS

    rows_agree = abs(v_rows - p_rows) <= CFG.ROW_TOLERANCE
    cols_agree = v_cols == p_cols

    if vision_ok and pdfplumber_ok and rows_agree and cols_agree:
        content = pdfplumber_md
        confidence = "high"
        source = "pdfplumber+vision_verified"
    elif vision_ok:
        content = vision_md
        confidence = "medium"
        source = "vision_only"
    elif pdfplumber_ok:
        content = pdfplumber_md
        confidence = "medium"
        source = "pdfplumber_only"
    else:
        logger.warning(
            "两引擎均失败，跳过表格 p%d bbox=%s | vision=%r pdfplumber=%r",
            page_idx, bbox_pdf, vision_md[:60] if vision_md else "", pdfplumber_md[:60] if pdfplumber_md else "",
        )
        return None

    x0, y0, x1, y1 = bbox_pdf
    column = infer_column(page_width, x0, x1)

    return LayoutChunk(
        content_type="table",
        raw_content=content,
        page=page_idx,
        bbox=bbox_pdf,
        column=column,
        order_in_page=0,  # LayoutMerger 会重新分配
        metadata={
            "parser": "hybrid_opencv_vision",
            "confidence": confidence,
            "source": source,
            "vision_rows": v_rows,
            "vision_cols": v_cols,
            "pdfplumber_rows": p_rows,
            "pdfplumber_cols": p_cols,
        },
    )
