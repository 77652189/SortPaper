"""
表格解析器：使用 pdfplumber 提取表格，包装为 LayoutChunk。
每个表格作为一个 chunk，携带 bbox 坐标和 Markdown 内容。

当 pdfplumber 检测不到表格时，自动切换 PyMuPDF 的 find_tables() 作为后备，
对无边框表格（borderless table）有更好的检测能力。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz

from src.parsers.layout_chunk import LayoutChunk, infer_column

from src.parsers.config import TABLE_PARSER as CFG

logger = logging.getLogger(__name__)

# ── 模块级常量（原 config.py 数据集）────────────────────────────────────────
# 表头关键词：用于定位真正的列名行
HEADER_KEYWORDS: tuple[str, ...] = (
    "product", "yield", "reference", "substrate",
    "enzyme", "key enzyme", "reaction",
    "host", "strain", "characteristic", "cultivation",
    "company", "brand", "brands", "country",
)

# 首格跳过前缀：页眉 / 图注 / 参考文献误检
SKIP_PREFIXES: tuple[str, ...] = (
    "journal of",
    "figure ", "fig.",
    "references",
    "secnerefer",          # 旋转后的 "references"
    "supporting information",
    "doi:",
    "published online",
    "to link to this article",
)


class TableParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None, strategy: str = "all") -> list[LayoutChunk]:
        """
        提取 PDF 所有表格，返回 LayoutChunk 列表。

        支持多种策略，每次 retry 可以切换不同的解析方式：
        - "all" (默认): 依次尝试 pdfplumber → PyMuPDF → camelot（原逻辑）
        - "pdfplumber_only": 只用 pdfplumber
        - "pymupdf_only": 只用 PyMuPDF（含 camelot 后备）
        - "camelot_only": 只用 camelot stream
        """
        all_chunks: list[LayoutChunk] = []

        if strategy == "pdfplumber_only":
            chunks = self._parse_with_pdfplumber(feedback)
            if len(chunks) > 1:
                chunks = self._deduplicate_chunks(chunks)
            return chunks

        if strategy == "pymupdf_only":
            chunks = self._parse_with_pymupdf()
            if len(chunks) > 1:
                chunks = self._deduplicate_chunks(chunks)
            return chunks

        if strategy == "camelot_only":
            chunks = self._parse_with_camelot()
            if len(chunks) > 1:
                chunks = self._deduplicate_chunks(chunks)
            return chunks

        # strategy == "all"：依次尝试，合并结果
        chunks = self._parse_with_pdfplumber(feedback)
        all_chunks.extend(chunks)

        pymupdf_chunks = self._parse_with_pymupdf()
        all_chunks.extend(pymupdf_chunks)

        if len(all_chunks) < CFG.CAMELOT_TRIGGER_THRESHOLD:
            logger.info("前两个解析器检测到表格较少，尝试 camelot")
            camelot_chunks = self._parse_with_camelot()
            all_chunks.extend(camelot_chunks)

        if len(all_chunks) > 1:
            all_chunks = self._deduplicate_chunks(all_chunks)

        return all_chunks

    @staticmethod
    def _deduplicate_chunks(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """去重：如果两个 chunk 在同一页且 bbox IoU > 0.5，保留第一个。"""
        if not chunks:
            return chunks

        def compute_iou(b1, b2) -> float:
            """计算两个 bbox 的 IoU (Intersection over Union)"""
            if not b1 or not b2:
                return 0.0
            x0_1, y0_1, x1_1, y1_1 = b1
            x0_2, y0_2, x1_2, y1_2 = b2

            # 计算交集
            x0_inter = max(x0_1, x0_2)
            y0_inter = max(y0_1, y0_2)
            x1_inter = min(x1_1, x1_2)
            y1_inter = min(y1_1, y1_2)

            if x0_inter >= x1_inter or y0_inter >= y1_inter:
                return 0.0

            inter_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
            area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
            area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
            union_area = area1 + area2 - inter_area

            if union_area <= 0:
                return 0.0
            return inter_area / union_area

        unique: list[LayoutChunk] = []
        for chunk in chunks:
            is_duplicate = False
            for existing in unique:
                if chunk.page != existing.page:
                    continue
                # 计算 IoU
                b1 = chunk.bbox
                b2 = existing.bbox
                iou = compute_iou(b1, b2)
                if iou > CFG.DEDUP_IOU_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(chunk)
        return unique

    def _parse_with_pdfplumber(self, feedback: str | None) -> list[LayoutChunk]:
        """使用 pdfplumber 提取表格（原有逻辑）。"""
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
                    if not rows:
                        continue

                    normalized = self._normalize_table(rows)
                    if not self._is_valid_table(normalized):
                        continue

                    bbox_raw = table_obj.bbox
                    bbox = (
                        float(bbox_raw[0]),
                        page_height - float(bbox_raw[3]),
                        float(bbox_raw[2]),
                        page_height - float(bbox_raw[1]),
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

    def _parse_with_pymupdf(self) -> list[LayoutChunk]:
        """
        使用 PyMuPDF 的 find_tables() 检测并提取表格。
        对无边框表格（borderless table）有更好的检测能力。
        """
        chunks: list[LayoutChunk] = []

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                finder = page.find_tables()
                tables = finder.tables
                if not tables:
                    continue

                for table_index, table in enumerate(tables):
                    bbox_raw = table.bbox
                    bbox = (
                        float(bbox_raw[0]),
                        float(bbox_raw[1]),
                        float(bbox_raw[2]),
                        float(bbox_raw[3]),
                    )

                    try:
                        rows = table.extract()
                    except Exception as e:
                        logger.warning(f"PyMuPDF 表格提取失败 (p{page_index}, t{table_index}): {e}")
                        continue

                    if not rows or not rows[0]:
                        continue

                    normalized = self._normalize_table(rows)
                    if not self._is_valid_table(normalized):
                        continue

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
                                "parser": "pymupdf",
                                "rows": len(normalized),
                                "cols": len(normalized[0]),
                                "table_index": table_index,
                            },
                        )
                    )

        # 如果 PyMuPDF 也没检测到，尝试 camelot（对无边框表格更强）
        if not chunks:
            chunks = self._parse_with_camelot()

        return chunks

    def _parse_with_camelot(self) -> list[LayoutChunk]:
        """
        使用 camelot 提取表格（后备方案）。
        camelot 的 stream 模式专门处理无边框表格，对学术论文表格效果更好。
        """
        try:
            import camelot
        except ImportError:
            logger.warning("camelot 未安装，跳过")
            return []

        chunks: list[LayoutChunk] = []

        try:
            tables = camelot.read_pdf(self.pdf_path, pages="all", flavor="stream")
        except Exception as e:
            logger.warning(f"camelot 提取失败: {e}")
            return []

        with fitz.open(self.pdf_path) as doc:
            for table in tables:
                # camelot page 是字符串或 int，转成 1-indexed
                page_index = int(table.page)
                if page_index < 1 or page_index > len(doc):
                    continue

                page = doc[page_index - 1]
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                # camelot bbox: (x0, y0_top, x1, y1_top) 左上角坐标系
                x0, y0_top, x1, y1_top = table._bbox
                bbox = (
                    float(x0),
                    page_height - float(y1_top),  # 转换为 PyMuPDF 坐标系
                    float(x1),
                    page_height - float(y0_top),
                )

                rows = table.data
                if not rows:
                    continue

                normalized = self._normalize_table(rows)
                if not self._is_valid_table(normalized):
                    continue

                markdown = self._to_markdown(normalized)
                column = infer_column(page_width, bbox[0], bbox[2])

                chunks.append(
                    LayoutChunk(
                        content_type="table",
                        raw_content=markdown,
                        page=page_index,
                        bbox=bbox,
                        column=column,
                        order_in_page=len(chunks),
                        metadata={
                            "parser": "camelot",
                            "rows": len(normalized),
                            "cols": len(normalized[0]),
                            "table_index": len(chunks),
                        },
                    )
                )

        # 合并同一页面内被切割的表格
        if len(chunks) > 1:
            chunks = TableParser._merge_camelot_chunks(chunks)

        return chunks

    @staticmethod
    def _merge_camelot_chunks(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """
        合并 camelot 在同一页面内切割的表格。
        判断条件：同一页面 + bbox x范围重叠度>50% + y方向接近或重叠。
        """
        if len(chunks) < 2:
            return chunks

        # 按页面分组
        by_page: dict[int, list[LayoutChunk]] = {}
        for chunk in chunks:
            by_page.setdefault(chunk.page, []).append(chunk)

        merged: list[LayoutChunk] = []

        for page, page_chunks in by_page.items():
            if len(page_chunks) == 1:
                merged.extend(page_chunks)
                continue

            # 按 bbox y0 排序（从上到下）
            page_chunks.sort(key=lambda c: c.bbox[1])

            # 尝试合并相邻的 chunks
            merged_page: list[LayoutChunk] = []
            current = page_chunks[0]

            for next_chunk in page_chunks[1:]:
                if TableParser._should_merge(current, next_chunk):
                    current = TableParser._merge_two_chunks(current, next_chunk)
                else:
                    merged_page.append(current)
                    current = next_chunk

            merged_page.append(current)
            merged.extend(merged_page)

        return merged

    @staticmethod
    def _should_merge(c1: LayoutChunk, c2: LayoutChunk) -> bool:
        """判断两个 chunk 是否应该合并。"""
        b1 = c1.bbox
        b2 = c2.bbox

        # 1. 同一页面（已经按页面分组了，这里不需要检查）

        # 2. x 范围重叠度 > 50%
        x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        x_union = max(b1[2], b2[2]) - min(b1[0], b2[0])
        if x_union <= 0 or x_overlap / x_union < CFG.MERGE_X_OVERLAP_RATIO:
            return False

        # 3. y 方向接近或重叠（距离 < CFG.MERGE_Y_GAP_MAX 或者重叠）
        y_distance = max(0, b2[1] - b1[3])  # b2.y0 - b1.y1
        if y_distance > CFG.MERGE_Y_GAP_MAX:  # 距离太大，不合并
            return False

        return True

    @staticmethod
    def _merge_two_chunks(c1: LayoutChunk, c2: LayoutChunk) -> LayoutChunk:
        """合并两个 chunk（合并 bbox 和数据行）。"""
        b1 = c1.bbox
        b2 = c2.bbox

        # 新的 bbox：保留第一个 chunk 的 bbox（不扩展，避免重建图上框住无关区域）
        new_bbox = b1

        # 合并数据行（去掉重复的表头）
        lines1 = c1.raw_content.split("\n")
        lines2 = c2.raw_content.split("\n")

        # 去掉 lines2 的表头行（Markdown 表格的表头分隔行）
        # 通常第二行是 "|---|---|..."，需要去掉第一行（表头标题）和第二行（分隔行）
        if len(lines2) >= 2 and "---" in lines2[1]:
            # lines2[0] 是表头标题（可能是重复的），lines2[1] 是分隔行
            # 保留 lines2[2:]（数据行）
            lines2_data = lines2[2:]
        else:
            lines2_data = lines2

        new_content = "\n".join(lines1 + lines2_data)

        # 更新 metadata
        new_metadata = c1.metadata.copy()
        new_metadata["rows"] = c1.metadata.get("rows", 0) + c2.metadata.get("rows", 0)
        new_metadata["merged"] = True

        return LayoutChunk(
            content_type=c1.content_type,
            raw_content=new_content,
            page=c1.page,
            bbox=new_bbox,
            column=c1.column,  # 保留第一个的 column
            order_in_page=c1.order_in_page,
            metadata=new_metadata,
        )

    @staticmethod
    def _find_header_row(normalized: list[list[str]]) -> int:
        """找到表格的表头行索引。"""
        # 优先找包含常见表头关键词的行
        for idx, row in enumerate(normalized):
            content = " ".join(cell.lower() for cell in row)
            non_empty = sum(1 for cell in row if cell.strip())
            if non_empty >= 2 and any(
                kw in content
                for kw in HEADER_KEYWORDS
            ):
                return idx

        # 如果找不到，尝试找包含 "Table " 的行的下一行
        for idx, row in enumerate(normalized):
            if "table " in " ".join(cell.lower() for cell in row):
                if idx + 1 < len(normalized):
                    next_row = normalized[idx + 1]
                    non_empty = sum(1 for cell in next_row if cell.strip())
                    if non_empty >= 2:
                        return idx + 1
                return idx  # 没找到下一行，使用当前行
        
        return 0  # 找不到，使用首行

    @staticmethod
    def _adjust_table(normalized: list[list[str]]) -> list[list[str]]:
        """调整表格数据顺序，让真正的表头成为第一行。"""
        header_row_idx = TableParser._find_header_row(normalized)
        if header_row_idx > 0:
            # 调整顺序：把header_row_idx及之后的行放到前面，之前的行放到后面
            normalized = normalized[header_row_idx:] + normalized[:header_row_idx]
        return normalized

    @staticmethod
    def _table_settings(feedback: str | None) -> dict[str, Any]:
        """根据反馈调整 pdfplumber 参数。"""
        settings: dict[str, Any] = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": CFG.PDFPLUMBER_SNAP_TOLERANCE,
            "join_tolerance": CFG.PDFPLUMBER_JOIN_TOLERANCE,
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
        - camelot stream 模式的常见误检（页眉、参考文献、图片标题等）

        改进：camelot 可能把页眉/页码行也纳入表格，需要从表格中识别真正的表头行。
        """
        import re as _re

        if not normalized or len(normalized) < CFG.MIN_DATA_ROWS + 1:
            # 少于 2 行（表头 + 至少 1 行数据）→ 空表
            return False

        # 尝试定位真正的表头行（包含列名关键词的行）
        header_row_idx = 0
        for idx, row in enumerate(normalized):
            content = " ".join(cell.lower() for cell in row)
            non_empty = sum(1 for cell in row if cell.strip())
            # 优先找包含常见表头关键词的行，且该行有多个非空白单元格（看起来像表头）
            if non_empty >= 2 and any(
                kw in content
                for kw in HEADER_KEYWORDS
            ):
                header_row_idx = idx
                break

        # 如果找不到列名行，尝试找包含 "Table " 的行，并假设下一行是真正表头
        if header_row_idx == 0:
            for idx, row in enumerate(normalized):
                if "table " in " ".join(cell.lower() for cell in row):
                    # 找到 Table X 标题行，检查下一行是否像表头
                    if idx + 1 < len(normalized):
                        next_row = normalized[idx + 1]
                        # 下一行如果包含多个非空白单元格，可能是真正表头
                        non_empty = sum(1 for cell in next_row if cell.strip())
                        if non_empty >= 2:
                            header_row_idx = idx + 1
                            break
                    # 如果下一行不像表头，就从当前行开始（包含Table标题）
                    header_row_idx = idx
                    break

        # 如果还是找不到，使用首行
        header = normalized[header_row_idx]
        data_rows = normalized[header_row_idx + 1:]
        num_cols = len(header)

        # 必须有至少 MIN_COLS 列
        if num_cols < CFG.MIN_COLS:
            return False

        # 至少有 MIN_DATA_ROWS 行非空数据（过滤掉只有1行数据的误检）
        if len(data_rows) < CFG.MIN_DATA_ROWS:
            return False

        has_data = any(any(cell.strip() for cell in row) for row in data_rows)
        if not has_data:
            return False

        # 首格内容检查：期刊名、图注、参考文献等页眉特征 → 伪表格
        first_cell_raw = normalized[0][0].strip()
        first_cell = first_cell_raw.lower()
        if any(first_cell.startswith(p) for p in SKIP_PREFIXES):
            return False

        # 期刊 running title / 作者行检测：连续大写无空格的长字符串
        # 例如 "CRITICALREVIEWSINBIOTECHNOLOGY"、"Y.ZHUETAL."
        # 但如果是 "Y. ZHU ET AL." 格式（有空格），可能是页眉，需要结合其他特征判断
        compact = first_cell_raw.replace(".", "").replace(" ", "")
        if len(compact) > CFG.COMPACT_UPPER_MIN_LEN and compact.isupper() and " " not in first_cell_raw:
            # 可能是期刊名，但再看看表格其他部分是否有 "Table" 关键词
            full_text = " ".join(cell for row in normalized for cell in row).lower()
            if "table " not in full_text:
                return False

        # 参考文献列表：首格匹配 "[数字]" 格式，且表格行数很多
        if _re.match(r"^\[\d+\]", first_cell_raw):
            return False

        # 页眉特征：首行是孤立的页码数字（如 "8"）
        if _re.match(r"^\d+$", first_cell_raw.strip()) and len(normalized) > 10:
            # 可能是页码 + 参考文献列表，过滤
            return False

        # 正文段落误检：首格以小写开头且长度较长（看起来像段落/短语，不是表头）
        if (
            first_cell_raw
            and first_cell_raw[0].islower()
            and len(first_cell_raw) > 10
            and "table " not in first_cell_raw.lower()
        ):
            return False

        # 双栏正文误检：
        #   ① 2 列 + 超过 20 行数据 → pdfplumber 把双栏段落逐行切开，非真实表格
        #   ② 2 列 + 每格平均字符极长（>80 字符）→ 单元格本身就是流式正文
        if num_cols == 2:
            if len(data_rows) > CFG.TWO_COL_MAX_DATA_ROWS:
                return False
            all_cells = [cell for row in normalized for cell in row if cell.strip()]
            if all_cells:
                avg_len = sum(len(c) for c in all_cells) / len(all_cells)
                if avg_len > CFG.TWO_COL_MAX_AVG_LEN:
                    return False

        # 表头空格率 > 40%：页眉被拆成多列，大量空单元格
        header_empty = sum(1 for cell in header if not cell.strip())
        if len(header) > 0 and header_empty / len(header) > CFG.HEADER_EMPTY_RATIO:
            return False

        # 数据行空格率 > 60% → 稀疏伪表格
        total_cells = sum(len(row) for row in data_rows)
        empty_cells = sum(1 for row in data_rows for cell in row if not cell.strip())
        if total_cells > 0 and empty_cells / total_cells > CFG.DATA_EMPTY_RATIO:
            return False

        # camelot stream 模式特有问题：
        #   Figure 标题被切成表格，且行数很少 → 误检
        full_text = " ".join(cell for row in normalized for cell in row).lower()
        if "figure " in full_text and len(normalized) < 5 and "table " not in full_text:
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
        """将表格数据转换为 Markdown 格式（自动识别表头行）。"""
        if not table:
            return ""
        # 定位真正的表头行
        header_row_idx = TableParser._find_header_row(table)
        header = table[header_row_idx]
        data = table[header_row_idx + 1:]
        separator = ["---"] * len(header)
        rows = [header, separator] + data
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

