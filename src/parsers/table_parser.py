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

# ── 模块级常量 ────────────────────────────────────────────────
# 注意：表头关键词仅用于 _find_header_row 的表头定位（Markdown 渲染），
# 不参与表格真伪判断。真伪判断已迁移至 LLM Judge。
HEADER_KEYWORDS: tuple[str, ...] = (
    "product", "yield", "reference", "substrate",
    "enzyme", "key enzyme", "reaction",
    "host", "strain", "characteristic", "cultivation",
    "company", "brand", "brands", "country",
)


class TableParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None, strategy: str = "all",
              camelot_pages: list[int] | None = None) -> list[LayoutChunk]:
        """
        提取 PDF 所有表格，返回 LayoutChunk 列表。

        strategy 可选值：
          "all"           : 关键词引导 + pdfplumber + PyMuPDF + camelot（默认）
          "hybrid"        : OpenCV检测 + 视觉模型(主) + pdfplumber交叉验证(副)
                            适用于三线表等传统引擎无法解析的场景
          "pdfplumber_only" / "pymupdf_only" / "camelot_only" : 单引擎模式
        camelot_pages: 限制 camelot 扫描的页码（1-indexed），默认全文档。
        """
        all_chunks: list[LayoutChunk] = []

        if strategy == "hybrid":
            from src.parsers.hybrid_table_parser import HybridTableParser
            chunks = HybridTableParser(self.pdf_path).parse()
            if len(chunks) > 1:
                chunks = self._deduplicate_chunks(chunks)
            return chunks

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
            chunks = self._parse_with_camelot(pages=camelot_pages)
            if len(chunks) > 1:
                chunks = self._deduplicate_chunks(chunks)
            return chunks

        # strategy == "all"：关键词引导优先，结构检测补充，最终去重
        # 关键词引导精度最高（定位准确），放在最前以在去重时优先保留
        kw_chunks = self._parse_with_keyword_guided()
        all_chunks.extend(kw_chunks)

        chunks = self._parse_with_pdfplumber(feedback)
        all_chunks.extend(chunks)

        pymupdf_chunks = self._parse_with_pymupdf()
        all_chunks.extend(pymupdf_chunks)

        logger.info("camelot 作为补充解析器运行")
        camelot_chunks = self._parse_with_camelot(pages=camelot_pages)
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

    def _parse_with_keyword_guided(self) -> list[LayoutChunk]:
        """
        关键词引导式表格提取：定位标题块 → 裁剪区域 → text+text 提取。

        覆盖的格式变体：
          英文标准   : Table 1. / TABLE 1. / Table S1:
          补充材料   : Supplementary Table 1. / Table S1.
          中文期刊   : 表1. / 表 1：/ 表号1.
          预印本缩写 : Tab. 1.
        跳过的情形  : 正文内引用（"as shown in Table 1"，不以 Table 开头）、
                      扫描版 PDF（无文字层）、跨页表格（仅提取标题所在页）。

        坐标系：全程使用 pdfplumber 坐标（top-left 原点），
        避免 fitz 在横排页（含 /Rotate 元数据）下坐标错位的问题。
        """
        import re
        import pdfplumber

        TITLE_RE = re.compile(
            r"^(?:"
            r"(?:Table|TABLE|Tab\.)\s+S?\d+[.:]"       # Table 1. / TABLE S2:
            r"|Supplementary\s+Table\s+S?\d+[.:]"       # Supplementary Table 1.
            r"|表\s*[S号]?\d+[.：]"                      # 表1. / 表号1：
            r")"
        )
        MAX_TABLE_HEIGHT_PT = 500
        LINE_SNAP = 3  # top 差 ≤ LINE_SNAP pt 的词视为同一行

        chunks: list[LayoutChunk] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index in range(len(pdf.pages)):
                pdf_page = pdf.pages[page_index]
                pw = float(pdf_page.width)
                ph = float(pdf_page.height)

                # ── 1. 找表格标题行（pdfplumber 坐标，与 within_bbox 完全一致）──
                words = pdf_page.extract_words()
                if not words:
                    continue

                # 按 top 排序后，相邻词 top 差 ≤ LINE_SNAP 则归为同一行
                words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
                lines: list[list[dict]] = []
                for w in words_sorted:
                    placed = False
                    for line in reversed(lines):
                        if abs(w["top"] - line[0]["top"]) <= LINE_SNAP:
                            line.append(w)
                            placed = True
                            break
                    if not placed:
                        lines.append([w])

                title_anchors: list[tuple[float, float, float, float]] = []
                for line in lines:
                    line_sorted = sorted(line, key=lambda w: w["x0"])
                    line_text = " ".join(lw["text"] for lw in line_sorted).strip()
                    if not TITLE_RE.match(line_text):
                        continue
                    if len(line_text.split()) > 60:
                        continue
                    t_top = min(lw["top"] for lw in line_sorted)
                    t_bot = max(lw["bottom"] for lw in line_sorted)
                    t_x0 = min(lw["x0"] for lw in line_sorted)
                    t_x1 = max(lw["x1"] for lw in line_sorted)
                    title_anchors.append((t_top, t_bot, t_x0, t_x1))

                if not title_anchors:
                    continue

                # 收集页面全部宽横线（pdfplumber top-left "top" 坐标）
                wide_h = [
                    e for e in pdf_page.edges
                    if e.get("orientation") == "h"
                    and (e.get("x1", 0) - e.get("x0", 0)) > pw * 0.2
                ]

                # ── 2. 对每个标题定位区域并提取 ──────────────────
                for title_top, title_bot, _tx0, _tx1 in title_anchors:
                    region_top = title_bot  # 从标题底部开始

                    # 找标题下方的宽横线（表格边框线）
                    below = [
                        e for e in wide_h
                        if e["top"] > region_top
                        and e["top"] < region_top + MAX_TABLE_HEIGHT_PT
                    ]

                    if below:
                        region_bot = max(e["top"] for e in below) + 5
                        clip_x0 = max(0.0, min(e["x0"] for e in below) - 2)
                        clip_x1 = min(pw, max(e["x1"] for e in below) + 2)
                    else:
                        region_bot = min(region_top + MAX_TABLE_HEIGHT_PT, ph)
                        clip_x0, clip_x1 = 0.0, pw

                    # bbox 合法性检查（横排页容易触发）
                    if region_top >= region_bot or clip_x0 >= clip_x1:
                        logger.warning(
                            f"keyword_guided p{page_index+1}: 无效 bbox "
                            f"({clip_x0:.1f},{region_top:.1f},{clip_x1:.1f},{region_bot:.1f})"
                        )
                        continue

                    try:
                        clipped = pdf_page.within_bbox(
                            (clip_x0, region_top, clip_x1, region_bot)
                        )
                        tables = clipped.find_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 5,
                            "join_tolerance": 5,
                        })
                    except Exception as ex:
                        logger.warning(f"keyword_guided p{page_index+1}: {ex}")
                        continue

                    for ti, table_obj in enumerate(tables):
                        rows = table_obj.extract()
                        if not rows:
                            continue
                        normalized = self._normalize_table(rows)
                        if not self._is_valid_table(normalized):
                            continue

                        bbox_raw = table_obj.bbox
                        bbox = (
                            float(bbox_raw[0]),
                            ph - float(bbox_raw[3]),
                            float(bbox_raw[2]),
                            ph - float(bbox_raw[1]),
                        )
                        markdown = self._to_markdown(normalized)
                        column = infer_column(pw, bbox[0], bbox[2])

                        chunks.append(LayoutChunk(
                            content_type="table",
                            raw_content=markdown,
                            page=page_index + 1,
                            bbox=bbox,
                            column=column,
                            order_in_page=ti,
                            metadata={
                                "parser": "keyword_guided",
                                "rows": len(normalized),
                                "cols": len(normalized[0]),
                                "table_index": ti,
                            },
                        ))

        return chunks

    def _parse_with_pdfplumber(self, feedback: str | None) -> list[LayoutChunk]:
        """使用 pdfplumber 提取表格（lines 策略 + 三线表 text 回退）。"""
        import pdfplumber

        settings = self._table_settings(feedback)
        chunks: list[LayoutChunk] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_width = float(page.width) if page.width else 0
                page_height = float(page.height) if page.height else 0
                found_tables = page.find_tables(table_settings=settings)

                def _has_data(tbls: list) -> bool:
                    """lines 策略只检测到 header 区域时（≤1 行），触发 fallback。"""
                    return any(len(t.extract() or []) > 1 for t in tbls)

                # 回退 1：text+lines 策略（无竖线表格）
                # 触发条件：未检到表格，或只检到仅含 header 的 1 行区域
                if not found_tables or not _has_data(found_tables):
                    t1 = page.find_tables(table_settings=dict(
                        settings, vertical_strategy="text"
                    ))
                    if t1 and _has_data(t1):
                        found_tables = t1

                # 回退 2：裁剪旋转边注 + text+text
                # 适用于横版页面或期刊名旋转印于右侧导致列检测失效的情况。
                # 用宽横线的 x 范围裁剪页面，排除旋转边注文字。
                if (not found_tables or not _has_data(found_tables)) and page_width > 0:
                    wide_h = [
                        e for e in page.edges
                        if e.get("orientation") == "h"
                        and (e.get("x1", 0) - e.get("x0", 0)) > page_width * 0.2
                    ]
                    if wide_h:
                        clip_x0 = max(0.0, min(e["x0"] for e in wide_h) - 2)
                        clip_x1 = min(page_width, max(e["x1"] for e in wide_h) + 2)
                        clipped = page.within_bbox((clip_x0, 0, clip_x1, page_height))
                        t2 = clipped.find_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 5,
                            "join_tolerance": 5,
                        })
                        if t2 and _has_data(t2):
                            found_tables = t2

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

    def _parse_with_camelot(self, pages: list[int] | None = None) -> list[LayoutChunk]:
        """
        使用 camelot 提取表格（后备方案）。
        pages: 指定页码列表（1-indexed），默认全文档。
        """
        try:
            import camelot
        except ImportError:
            logger.warning("camelot 未安装，跳过")
            return []

        chunks: list[LayoutChunk] = []
        pages_str = ",".join(str(p) for p in pages) if pages else "all"

        try:
            tables = camelot.read_pdf(self.pdf_path, pages=pages_str, flavor="stream")
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
    def _looks_garbled(cell: str) -> bool:
        """旋转/倒置文字产生的乱码：≥5个字母、零元音、且非混排大小写。

        混排豁免：LNnT/LNFP 等化学缩写通常混排，不应误判。
        全大写无元音长串（如 CRTCL、RVSBLT）几乎不存在于真实表格。
        """
        s = cell.strip()
        alpha = [c for c in s if c.isalpha()]
        if len(alpha) < 5:
            return False
        vowels = set("aeiouyAEIOUY")
        if any(c in vowels for c in s):
            return False
        # 混排大小写 → 可能是化学缩写，豁免
        has_upper = any(c.isupper() for c in s)
        has_lower = any(c.islower() for c in s)
        if has_upper and has_lower:
            return False
        return True

    @staticmethod
    def _is_valid_table(normalized: list[list[str]]) -> bool:
        """结构检查 + 旋转文字过滤。

        内容层面的真伪判断（页眉/参考文献/正文误检等）已迁移至 LLM Judge。
        旋转文字因零成本且高准确率保留为硬编码规则。
        """
        if not normalized or len(normalized) < CFG.MIN_DATA_ROWS + 1:
            return False

        header_row_idx = TableParser._find_header_row(normalized)
        header = normalized[header_row_idx]
        data_rows = normalized[header_row_idx + 1:]
        num_cols = len(header)

        if num_cols < CFG.MIN_COLS:
            return False

        if len(data_rows) < CFG.MIN_DATA_ROWS:
            return False

        has_data = any(any(cell.strip() for cell in row) for row in data_rows)
        if not has_data:
            return False

        # 旋转文字检测：前几行中超40%非空单元格为乱码 → 过滤
        sample_rows = (normalized[:header_row_idx] if header_row_idx > 0 else []) + [header] + data_rows[:3]
        sample_cells = [cell for row in sample_rows for cell in row if cell.strip()]
        if sample_cells:
            garbled = sum(1 for c in sample_cells if TableParser._looks_garbled(c))
            if garbled / len(sample_cells) > 0.4:
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

