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
                # 仅使用 lines 策略：只识别有明确网格线的真实表格。
                # text 策略会把双栏版面的列间空白误判为表格，产生大量误报。
                found_tables = page.find_tables(table_settings=settings)

                for table_index, table_obj in enumerate(found_tables):
                    rows = table_obj.extract()
                    if not rows:
                        continue

                    normalized = self._normalize_table(rows)
                    if not self._is_valid_table(normalized):
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
    def _is_valid_table(normalized: list[list[str]]) -> bool:
        """
        判断提取出的表格是否为真实表格，过滤掉常见误检情况：
        - 只有表头没有数据行（空表）
        - 单列长文本（图注、正文段落被切成单列）
        - 以期刊名 / Figure / References 等关键词开头（页眉误检）
        - 旋转/倒置文字形成的假表格
        - 数据格为空率过高（> 60%）的稀疏表格
        """
        if not normalized or len(normalized) < 2:
            # 少于 2 行（表头 + 至少 1 行数据）→ 空表
            return False

        header = normalized[0]
        data_rows = normalized[1:]
        num_cols = len(header)

        # 必须有至少 2 列
        if num_cols < 2:
            return False

        # 至少有 1 行非空数据
        has_data = any(any(cell.strip() for cell in row) for row in data_rows)
        if not has_data:
            return False

        # 首格内容检查：期刊名、图注、参考文献等页眉特征 → 伪表格
        first_cell_raw = header[0].strip() if header else ""
        first_cell = first_cell_raw.lower()
        _SKIP_PREFIXES = (
            "journal of", "figure ", "fig.", "references",
            "secnerefer",  # 旋转的 "references"
            "supporting information",
        )
        if any(first_cell.startswith(p) for p in _SKIP_PREFIXES):
            return False

        # 期刊 running title / 作者行检测：连续大写无空格的长字符串
        # 例如 "CRITICALREVIEWSINBIOTECHNOLOGY"、"Y.ZHUETAL."
        compact = first_cell_raw.replace(".", "").replace(" ", "")
        if len(compact) > 8 and compact.isupper() and " " not in first_cell_raw:
            return False

        # 参考文献列表：首格匹配 "[数字]" 格式
        import re as _re
        if _re.match(r"^\[\d+\]", first_cell_raw):
            return False

        # 双栏正文误检：
        #   ① 2 列 + 超过 20 行数据 → pdfplumber 把双栏段落逐行切开，非真实表格
        #   ② 2 列 + 每格平均字符极长（>80 字符）→ 单元格本身就是流式正文
        if num_cols == 2:
            if len(data_rows) > 20:
                return False
            all_cells = [cell for row in normalized for cell in row if cell.strip()]
            if all_cells:
                avg_len = sum(len(c) for c in all_cells) / len(all_cells)
                if avg_len > 80:
                    return False

        # 表头空格率 > 40%：页眉被拆成多列，大量空单元格
        header_empty = sum(1 for cell in header if not cell.strip())
        if len(header) > 0 and header_empty / len(header) > 0.4:
            return False

        # 数据行空格率 > 60% → 稀疏伪表格
        total_cells = sum(len(row) for row in data_rows)
        empty_cells = sum(1 for row in data_rows for cell in row if not cell.strip())
        if total_cells > 0 and empty_cells / total_cells > 0.6:
            return False

        return True

    @staticmethod
    def _normalize_table(table: list[list[str | None]]) -> list[list[str]]:
        normalized: list[list[str]] = []
        max_cols = max((len(row) for row in table if row), default=0)
        if max_cols < 1:
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

