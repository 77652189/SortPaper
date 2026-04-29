"""
PyMuPDF 文本解析器：提取 PDF 文字块，包装为 LayoutChunk。
保留双栏布局信息（column）、bbox 坐标、页码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.parsers.layout_chunk import ContentType, LayoutChunk, infer_column


class PyMuPDFParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None) -> list[LayoutChunk]:
        """
        提取 PDF 所有文字块，返回按阅读顺序排列的 LayoutChunk 列表。
        - 按 PyMuPDF span 粒度提取（支持双栏布局）
        - 根据 page_width + x0 判断 column
        - 每个 span 作为一个 chunk，携带 bbox 坐标
        """
        import fitz

        chunks: list[LayoutChunk] = []
        use_spans = bool(feedback)

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
            # 粗粒度模式：按 block 提取，再按自然段切分
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_rest = block
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                x0_f, y0_f, x1_f, y1_f = float(x0), float(y0), float(x1), float(y1)
                column = infer_column(page_width, x0_f, x1_f)
                # 按空行切分自然段；同一 block 共用父 bbox
                paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
                if not paragraphs:
                    continue
                for para in paragraphs:
                    items.append((column, y0_f, x0_f, para, (x0_f, y0_f, x1_f, y1_f)))

        # 按 column → y0 → x0 排序（模拟阅读顺序）
        items.sort(key=lambda item: (item[0], item[1], item[2]))

        chunks: list[LayoutChunk] = []
        para_counter: dict[str, int] = {}  # bbox_key → 自然段计数
        for idx, (col, _y, _x, text, bbox) in enumerate(items):
            bbox_key = f"{bbox[0]:.0f}_{bbox[1]:.0f}"
            p_idx = para_counter.get(bbox_key, 0)
            para_counter[bbox_key] = p_idx + 1
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
                        "paragraph_index": p_idx,
                    },
                )
            )
        return chunks
