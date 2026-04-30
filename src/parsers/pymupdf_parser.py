"""
PyMuPDF 文本解析器：提取 PDF 文字块，包装为 LayoutChunk。
保留双栏布局信息（column）、bbox 坐标、页码。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.parsers.layout_chunk import ContentType, LayoutChunk, infer_column

logger = logging.getLogger(__name__)


class PyMuPDFParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None) -> list[LayoutChunk]:
        """
        提取 PDF 所有文字块，返回按阅读顺序排列的 LayoutChunk 列表。

        Args:
            feedback: 上次 judge 的反馈（用于细粒度重新解析）
                     当前实现：仅用于切换解析粒度，未来可解析反馈内容实现针对性重新解析
        """
        import fitz

        chunks: list[LayoutChunk] = []
        use_spans = bool(feedback)

        if feedback:
            logger.info(f"PyMuPDF: parsing with feedback (span-level mode)")

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                page_chunks = self._extract_page_chunks(page, page_index, use_spans)
                chunks.extend(page_chunks)

        return chunks

    def _extract_page_chunks(self, page: Any, page_index: int, use_spans: bool) -> list[LayoutChunk]:
        """逐页提取文字块，构建 LayoutChunk。每个 block 内按自然段（\\n\\n）切分子 chunk。"""
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

        # items: (column, y0, x0, text, bbox)
        items: list[tuple[int, float, float, str, tuple[float, float, float, float]]] = []

        if use_spans:
            # 细粒度模式：按 span 提取，保留精确坐标
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    column = infer_column(page_width, x0, x1)
                    items.append((column, y0, x0, text, (x0, y0, x1, y1)))
        else:
            # 粗粒度模式：按 block 提取，不切分段落
            # 保持 block 粒度，避免 bbox 共用导致的 chunk_id 重复
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_rest = block
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                x0_f, y0_f, x1_f, y1_f = float(x0), float(y0), float(x1), float(y1)
                column = infer_column(page_width, x0_f, x1_f)
                items.append((column, y0_f, x0_f, cleaned, (x0_f, y0_f, x1_f, y1_f)))

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
