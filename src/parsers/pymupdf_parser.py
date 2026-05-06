"""
PyMuPDF 文本解析器：提取 PDF 文字块，包装为 LayoutChunk。
保留双栏布局信息（column）、bbox 坐标、页码。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.parsers.layout_chunk import ContentType, LayoutChunk, infer_column
from src.parsers.config import PYMUPDF as CFG

logger = logging.getLogger(__name__)


class PyMuPDFParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None) -> list[LayoutChunk]:
        """
        提取 PDF 所有文字块，返回按阅读顺序排列的 LayoutChunk 列表。

        Args:
            feedback: 上次 judge 的反馈，传入时启用细粒度（line 级）解析。
        """
        import fitz

        chunks: list[LayoutChunk] = []
        use_spans = bool(feedback)
        if feedback:
            logger.info("PyMuPDF: parsing with feedback (line-level mode)")

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                page_chunks = self._extract_page_chunks(page, page_index, use_spans)
                chunks.extend(page_chunks)

        return chunks

    def _extract_page_chunks(self, page: Any, page_index: int, use_spans: bool) -> list[LayoutChunk]:
        """
        逐页提取文字块。

        - use_spans=False（默认）：按 block 聚合，过滤上标，粒度适中。
        - use_spans=True（feedback 模式）：按 line 切分，粒度更细。
        两个分支均使用 get_text("dict")，保留字体信息用于上标过滤。
        竖排文字（如旋转 90 度的表格行标签）会被反转拼接，保留语义信息。
        """

        def _vertical_text(span: dict) -> str:
            """竖排文字反转（dir.y 分量主导时，PDF 存储顺序与阅读顺序相反）。"""
            t = span.get("text", "")
            d = span.get("dir", (1, 0))
            if abs(d[1]) > abs(d[0]):
                return t[::-1]
            return t

        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

        # items: (column, y0, x0, text, bbox)
        items: list[tuple[int, float, float, str, tuple[float, float, float, float]]] = []

        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_bbox = block.get("bbox", [0, 0, 0, 0])
            x0_b, y0_b, x1_b, y1_b = [float(v) for v in block_bbox]
            column = infer_column(page_width, x0_b, x1_b)

            if use_spans:
                # feedback 模式：按 line 切分，粒度更细
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    # 用基线 y 中位数判断上标：基线偏移超过字号 40% 判定为上标
                    origins_y = [s.get("origin", (0, 0))[1] for s in spans]
                    median_y = sorted(origins_y)[len(origins_y) // 2]
                    max_size = max(s.get("size", 0) for s in spans)
                    text = "".join(
                        _vertical_text(s)
                        for s in spans
                        if abs(s.get("origin", (0, median_y))[1] - median_y) < max_size * CFG.SUPERSCRIPT_BASELINE_RATIO
                    ).strip()
                    if not text:
                        continue
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = [float(v) for v in bbox]
                    line_column = infer_column(page_width, x0, x1)
                    items.append((line_column, y0, x0, text, (x0, y0, x1, y1)))
            else:
                # 默认模式：按 block 聚合，过滤上标
                lines_text = []
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    origins_y = [s.get("origin", (0, 0))[1] for s in spans]
                    median_y = sorted(origins_y)[len(origins_y) // 2]
                    max_size = max(s.get("size", 0) for s in spans)
                    line_text = "".join(
                        _vertical_text(s)
                        for s in spans
                        if abs(s.get("origin", (0, median_y))[1] - median_y) < max_size * CFG.SUPERSCRIPT_BASELINE_RATIO
                    ).strip()
                    if line_text:
                        lines_text.append(line_text)
                text = " ".join(lines_text).strip()
                if not text:
                    continue
                items.append((column, y0_b, x0_b, text, (x0_b, y0_b, x1_b, y1_b)))

        # 注意：此处不排序，由 LayoutMerger.merge() 统一排序

        chunks: list[LayoutChunk] = []
        for idx, (col, _y, _x, text, bbox) in enumerate(items):
            chunks.append(
                LayoutChunk(
                    content_type="text",
                    raw_content=text,
                    page=page_index,
                    bbox=bbox,
                    column=col,
                    order_in_page=idx,
                    metadata={
                        "parser": "pymupdf",
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                )
            )
        return chunks
