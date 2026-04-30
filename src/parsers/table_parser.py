"""
表格解析器：使用 pdfplumber 提取表格，包装为 LayoutChunk。
每个表格作为一个 chunk，携带 bbox 坐标和 Markdown 内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.parsers.layout_chunk import LayoutChunk, infer_column


class TableParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None) -> list[LayoutChunk]:
        """
        提取 PDF 所有表格，返回 LayoutChunk 列表。
        使用 page.find_tables() 获取真实 bbox，避免坐标解析错误。
        """
        import pdfplumber

        settings = self._table_settings(feedback)
        chunks: list[LayoutChunk] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_width = float(page.width) if page.width else 0
                page_height = float(page.height) if page.height else 0
                found_tables = page.find_tables(table_settings=settings)

                for table_index, table_obj in enumerate(found_tables):
                    rows = table_obj.extract()
                    if not rows or len(rows) < 2:
                        continue

                    normalized = self._normalize_table(rows)
                    if not normalized or len(normalized[0]) < 2:
                        continue

                    # 使用 find_tables() 返回对象的真实 bbox
                    # pdfplumber 坐标系：左上角为原点 (x0, top, x1, bottom)
                    # PyMuPDF 坐标系：左下角为原点
                    # 转换：y_bottom = page_height - y_top
                    bbox_raw = table_obj.bbox  # (x0, top, x1, bottom) in pdfplumber coords
                    bbox = (
                        float(bbox_raw[0]),                     # x0 (相同)
                        page_height - float(bbox_raw[3]),      # y0 = page_height - bottom
                        float(bbox_raw[2]),                     # x1 (相同)
                        page_height - float(bbox_raw[1]),      # y1 = page_height - top
                    )
                    markdown = self._to_markdown(normalized)
                    column = infer_column(page_width, bbox[0], bbox[2])

                    chunks.append(
                        LayoutChunk(
                            content_type="table",
                            raw_content=markdown,
                            page=page_index,
                            bbox=bbox,
                            column=column,
                            order_in_page=table_index,
                            metadata={
                                "parser": "pdfplumber",
                                "rows": len(normalized),
                                "cols": len(normalized[0]),
                                "table_index": table_index,
                            },
                        )
                    )

        return chunks

    @staticmethod
    def _table_settings(feedback: str | None) -> dict[str, Any]:
        """根据反馈调整 pdfplumber 参数。"""
        settings: dict[str, Any] = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
        }
        if feedback:
            settings["snap_tolerance"] = 5
            settings["join_tolerance"] = 5
        return settings

    @staticmethod
    def _normalize_table(table: list[list[str | None]]) -> list[list[str]]:
        normalized: list[list[str]] = []
        max_cols = max((len(row) for row in table if row), default=0)
        if max_cols < 2:
            return []

        for row in table:
            cleaned = [(cell or "").replace("\n", " ").strip() for cell in row]
            if len(cleaned) < max_cols:
                cleaned.extend([""] * (max_cols - len(cleaned)))
            if any(cell for cell in cleaned):
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _to_markdown(table: list[list[str]]) -> str:
        header = table[0]
        separator = ["---"] * len(header)
        rows = [header, separator, *table[1:]]
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

