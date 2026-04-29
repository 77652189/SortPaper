from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    content: str
    metadata: dict[str, Any]


class PyMuPDFParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None) -> ParseResult:
        import fitz

        pages: list[str] = []
        total_blocks = 0

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                ordered_blocks = self._extract_ordered_blocks(page, use_spans=bool(feedback))
                total_blocks += len(ordered_blocks)
                page_text = "\n".join(block for block in ordered_blocks if block.strip())
                if page_text.strip():
                    pages.append(f"## Page {page_index}\n{page_text}")

        return ParseResult(
            content="\n\n".join(pages),
            metadata={
                "pages": len(pages),
                "blocks": total_blocks,
                "parser": "pymupdf",
                "feedback_used": bool(feedback),
            },
        )

    def _extract_ordered_blocks(self, page: Any, use_spans: bool) -> list[str]:
        page_width = float(page.rect.width)
        x_mid = page_width / 2

        items: list[tuple[int, float, float, str]] = []
        if use_spans:
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
                    x0, y0 = float(bbox[0]), float(bbox[1])
                    column = self._infer_column(page_width, x_mid, x0)
                    items.append((column, y0, x0, text))
        else:
            for block in page.get_text("blocks"):
                x0, y0, _x1, _y1, text, *_rest = block
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                column = self._infer_column(page_width, x_mid, float(x0))
                items.append((column, float(y0), float(x0), cleaned))

        items.sort(key=lambda item: (item[0], item[1], item[2]))
        return [text for _column, _y, _x, text in items]

    @staticmethod
    def _infer_column(page_width: float, x_mid: float, x0: float) -> int:
        if page_width <= 400:
            return 0
        return 0 if x0 < x_mid else 1
