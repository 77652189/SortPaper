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
          "all"             : 关键词引导 + pdfplumber + PyMuPDF + camelot（默认）
          "pdfplumber_only" / "pymupdf_only" / "camelot_only" : 单引擎模式
        camelot_pages: 限制 camelot 扫描的页码（1-indexed），默认全文档。
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

                    # ── 几何优先：检测三线表 / 表头框 ────────────────────────────
                    # 对整个裁剪区域做横线计数；因裁剪 bbox 包含标题行下方所有内容，
                    # 能够捕获刚好在 region_top 之下的表头框横线（text+text 策略会漏掉）。
                    clip_bbox_plumb = (clip_x0, region_top, clip_x1, region_bot)
                    hc, hys, vc, lx0, lx1 = TableParser._count_threeline_lines(
                        pdf_page, clip_bbox_plumb
                    )
                    geom_normalized: list[list[str]] | None = None
                    if hc == 3 and vc == 0:
                        tl = TableParser._extract_threeline(
                            pdf_page, clip_bbox_plumb, hys, lx0, lx1
                        )
                        if tl and len(tl) >= 2:
                            tl = TableParser._unsplit_twin_columns(tl)
                            tl = self._truncate_at_body_text(tl)
                        if tl and len(tl) >= 2:
                            geom_normalized = tl
                    elif (hc == 2 and vc == 0
                          and len(hys) == 2
                          and (hys[1] - hys[0]) < 30):
                        tl = TableParser._extract_headerbox(
                            pdf_page, clip_bbox_plumb, hys, lx0, lx1
                        )
                        if tl and len(tl) >= 2:
                            tl = TableParser._unsplit_twin_columns(tl)
                            tl = self._truncate_at_body_text(tl)
                        if tl and len(tl) >= 2:
                            geom_normalized = tl

                    if geom_normalized is not None and self._is_valid_table(geom_normalized):
                        # 几何提取成功，直接用裁剪区域 bbox
                        bbox = (
                            float(clip_x0),
                            ph - float(region_bot),
                            float(clip_x1),
                            ph - float(region_top),
                        )
                        chunks.append(LayoutChunk(
                            content_type="table",
                            raw_content=self._to_markdown(geom_normalized),
                            page=page_index + 1,
                            bbox=bbox,
                            column=infer_column(pw, clip_x0, clip_x1),
                            order_in_page=0,
                            metadata={
                                "parser": "keyword_guided",
                                "rows": len(geom_normalized),
                                "cols": len(geom_normalized[0]),
                                "table_index": 0,
                            },
                        ))
                        continue  # 跳过 text+text 路径

                    # ── text+text 路径（回退）──────────────────────────────────
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
                        normalized = self._truncate_at_body_text(
                            TableParser._unsplit_twin_columns(self._normalize_table(rows))
                        )
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
                    """lines 策略只检测到 header 区域时（≤1 行），触发 fallback。
                    特例：若 1 行的表格符合表头框（HDRBOX）几何特征，说明数据在框下方，
                    不视为"无数据"，避免 fallback 用全页 text+text 垃圾表格替换它。
                    """
                    for t in tbls:
                        rows = t.extract() or []
                        if len(rows) > 1:
                            return True
                        if len(rows) == 1:
                            hc, hys, vc, _, _ = TableParser._count_threeline_lines(
                                page, t.bbox
                            )
                            if (hc == 2 and vc == 0
                                    and len(hys) == 2
                                    and (hys[1] - hys[0]) < 30):
                                return True  # HDRBOX：数据在框下方
                    return False

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
                # 用宽横线 + 表头框列分隔竖线的联合 x 范围裁剪页面，
                # 既排除旋转边注文字，又保留表格右侧列头（仅有竖线无横线的部分）。
                if (not found_tables or not _has_data(found_tables)) and page_width > 0:
                    wide_h = [
                        e for e in page.edges
                        if e.get("orientation") == "h"
                        and (e.get("x1", 0) - e.get("x0", 0)) > page_width * 0.2
                    ]
                    if wide_h:
                        h_y_min = min(e.get("top", 0) for e in wide_h)
                        h_y_max = max(e.get("top", 0) for e in wide_h)
                        header_band_h = h_y_max - h_y_min + 5  # 表头框高度 + 容差

                        # 表头框列分隔竖线：高度 ≤ 表头框高度+5pt 且在表头 y 范围内的竖线
                        # 这类竖线是表头矩形框内的列分隔线，决定了表格真实的 x 宽度
                        header_box_v_xs = [
                            e.get("x0", 0)
                            for e in page.edges
                            if e.get("orientation") == "v"
                            and (e.get("bottom", 0) - e.get("top", 0)) <= header_band_h + 5
                            and e.get("top", 0) >= h_y_min - 3
                        ]

                        h_x0 = min(e["x0"] for e in wide_h)
                        h_x1 = max(e["x1"] for e in wide_h)
                        v_x_max = max(header_box_v_xs, default=h_x1)

                        clip_x0 = max(0.0, min(h_x0, min(header_box_v_xs, default=h_x0)) - 2)
                        clip_x1 = min(page_width, max(h_x1, v_x_max) + 2)
                        clipped = page.within_bbox((clip_x0, 0, clip_x1, page_height))
                        t2 = clipped.find_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 5,
                            "join_tolerance": 5,
                        })
                        if t2 and _has_data(t2):
                            found_tables = t2

                # ── 回退 3：旋转表格（90° 旋转字符组成的表格，如 ACS 期刊 Table 3）──
                # pdfplumber 对旋转字符的检测结果通常是错误的 1 列表格，需特殊处理。
                # 若检测成功，将旋转表格 chunk 直接加入结果，并过滤掉 found_tables
                # 中与旋转区域重叠的错误检测。
                rot_table = TableParser._detect_rotated_table(page)
                if rot_table:
                    rot_norm = self._normalize_table(rot_table)
                    if self._is_valid_table(rot_norm):
                        rot_md = self._to_markdown(rot_norm)
                        rot_chars = [
                            c for c in page.chars
                            if abs(c.get("matrix", [1, 0, 0, 1, 0, 0])[1]) > 0.1
                        ]
                        if rot_chars:
                            rx0 = float(min(c["x0"] for c in rot_chars))
                            rx1 = float(max(c.get("x1", c["x0"] + 10) for c in rot_chars))
                            ry_top = float(min(c["top"] for c in rot_chars))
                            ry_bot = float(max(c.get("bottom", c["top"] + 10) for c in rot_chars))
                            rot_bbox = (
                                rx0,
                                page_height - ry_bot,
                                rx1,
                                page_height - ry_top,
                            )
                            rot_col = infer_column(page_width, rot_bbox[0], rot_bbox[2])
                            chunks.append(LayoutChunk(
                                content_type="table",
                                raw_content=rot_md,
                                page=page_index,
                                bbox=rot_bbox,
                                column=rot_col,
                                order_in_page=0,
                                metadata={
                                    "parser": "pdfplumber_rotated",
                                    "rows": len(rot_norm),
                                    "cols": len(rot_norm[0]),
                                    "rotated": True,
                                },
                            ))
                            # 过滤 found_tables 中与旋转区域重叠的错误检测
                            found_tables = [
                                t for t in found_tables
                                if not (t.bbox[0] < rx1 + 5 and t.bbox[2] > rx0 - 5)
                            ]
                            logger.info(
                                "p%d 旋转表格: %d 行 × %d 列，filtered %d broken tables",
                                page_index, len(rot_norm), len(rot_norm[0]),
                                len(found_tables),
                            )

                for table_index, table_obj in enumerate(found_tables):
                    rows = table_obj.extract()
                    if not rows:
                        continue

                    # 几何提取：识别三线表 / 表头框格式，走专用提取器
                    h_count, h_ys, v_count, lines_x0, lines_x1 = \
                        TableParser._count_threeline_lines(page, table_obj.bbox)

                    if h_count == 3 and v_count == 0:
                        # 三线表：顶线 + 表头分隔线 + 底线，无竖线
                        tl = TableParser._extract_threeline(
                            page, table_obj.bbox, h_ys, lines_x0, lines_x1
                        )
                        if tl and len(tl) >= 2:
                            tl = TableParser._unsplit_twin_columns(tl)
                            tl = self._truncate_at_body_text(tl)
                        normalized = tl if (tl and len(tl) >= 2) else \
                            self._truncate_at_body_text(self._normalize_table(rows))
                        logger.debug(
                            "p%d 三线表: h=%d v=%d lines_x=[%.0f,%.0f] → %s (%d行)",
                            page_index, h_count, v_count, lines_x0, lines_x1,
                            "几何提取" if (tl and len(tl) >= 2) else "普通提取",
                            len(normalized),
                        )
                    elif (h_count == 2 and v_count == 0
                          and len(h_ys) == 2
                          and (h_ys[1] - h_ys[0]) < 30):
                        # 表头框：2条紧挨横线形成矩形表头 + 下方无边框数据行
                        tl = TableParser._extract_headerbox(
                            page, table_obj.bbox, h_ys, lines_x0, lines_x1
                        )
                        if tl and len(tl) >= 2:
                            tl = TableParser._unsplit_twin_columns(tl)
                            tl = self._truncate_at_body_text(tl)
                        normalized = tl if (tl and len(tl) >= 2) else \
                            self._truncate_at_body_text(self._normalize_table(rows))
                        logger.debug(
                            "p%d 表头框: h=%d v=%d lines_x=[%.0f,%.0f] → %s (%d行)",
                            page_index, h_count, v_count, lines_x0, lines_x1,
                            "几何提取" if (tl and len(tl) >= 2) else "普通提取",
                            len(normalized),
                        )
                    else:
                        normalized = self._truncate_at_body_text(
                            TableParser._unsplit_twin_columns(self._normalize_table(rows))
                        )

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

                    normalized = self._truncate_at_body_text(
                        TableParser._unsplit_twin_columns(self._normalize_table(rows))
                    )
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

                # camelot bbox: (x0, y0, x1, y1) PDF 标准坐标系（y 从页面底部向上）
                # 与 pdfplumber / keyword_guided 存储格式一致，直接使用原值
                x0, y0, x1, y1 = table._bbox
                bbox = (float(x0), float(y0), float(x1), float(y1))

                rows = table.data
                if not rows:
                    continue

                normalized = self._truncate_at_body_text(
                    TableParser._unsplit_twin_columns(self._normalize_table(rows))
                )
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
    def _truncate_at_body_text(normalized: list[list[str]]) -> list[list[str]]:
        """截断到第一个含超长单元格的行之前。

        pdfplumber 有时会把表格下方的正文一起框进来，导致末尾出现段落级长文本。
        遇到第一个超过 MAX_CELL_CHARS 字符的单元格，截断该行及之后的内容，
        保留上方的真实表格数据继续走验证流程。
        """
        for i, row in enumerate(normalized):
            if any(len(cell) > CFG.MAX_CELL_CHARS for cell in row):
                logger.debug(
                    "截断正文混入 row=%d（单元格 %d 字符）: %.60s…",
                    i, max(len(c) for c in row), max(row, key=len),
                )
                return normalized[:i]
        return normalized

    # ── 三线表几何提取（词位置 + 横线坐标） ───────────────────────────────────

    @staticmethod
    def _count_threeline_lines(
        pdf_page,
        bbox_plumb: tuple[float, float, float, float],
    ) -> tuple[int, list[float], int, float, float]:
        """
        统计 pdfplumber 页面中落在 bbox 内的有效水平线和垂直线数量。

        用于识别三线表（h_count==3, v_count==0）。

        参数:
            pdf_page   : pdfplumber Page 对象
            bbox_plumb : (x0, top, x1, bottom) pdfplumber 坐标系（y 从上往下）

        返回: (h_count, h_ys_sorted, v_count, h_x0_min, h_x1_max)
            h_x0_min / h_x1_max : 有效水平线的真实 x 范围（用于修正偏窄的 table bbox）
        """
        x0, top, x1, bottom = bbox_plumb
        page_width = float(pdf_page.width) if pdf_page.width else 500.0
        # 横线最短宽度：取 bbox 宽度 25% 与页面宽度 15% 中的较大值。
        # 25% 能兼容双栏表格的表头框窄线（约占页面宽度的 24%）。
        min_h_width = max((x1 - x0) * 0.25, page_width * 0.15)

        # 竖线最短高度阈值：过滤填充矩形的侧边、表头框的分隔竖线等短竖线。
        # 使用 bbox 高度的 10% 与 20pt 的较大值，使阈值随表格尺寸自适应：
        #   · 小 bbox（如 lines+lines 表头框，高≈21pt）→ 阈值≈20pt，保留真实竖线
        #   · 大 bbox（如 text+text 全表，高≈500pt）→ 阈值≈50pt，可过滤 20pt 表头框侧边
        bbox_height = bottom - top
        v_height_min = max(20.0, bbox_height * 0.10)

        seen_y_bins: dict[int, float] = {}  # 10pt 分箱去重
        v_count = 0
        h_x0_min = float("inf")
        h_x1_max = float("-inf")

        for edge in pdf_page.edges:
            ori = edge.get("orientation", "")
            # pdfplumber 边坐标字段：水平/垂直方向均使用 "top"/"bottom"（非 y0/y1）
            ey0 = float(edge.get("top", 0))
            ey1 = float(edge.get("bottom", ey0))
            ex0 = float(edge.get("x0", 0))
            ex1 = float(edge.get("x1", ex0))

            if ori == "h":
                ey = ey0  # 水平线 top == bottom（或相差极小）
                if not (top - 2 <= ey <= bottom + 2):
                    continue
                if abs(ex1 - ex0) < min_h_width:
                    continue
                # 10pt 分箱去重：ACS 等期刊用填充矩形绘制横线（re f），
                # 矩形的上/下两条边 y 相差 ~1.5pt，5pt 分箱会将它们放入
                # 不同 bin（导致 h_count 翻倍）。10pt 分箱可安全合并。
                y_bin = round(ey / 10)
                if y_bin not in seen_y_bins:
                    seen_y_bins[y_bin] = ey
                # 记录横线真实 x 范围（bbox 可能因旋转边注被截窄）
                h_x0_min = min(h_x0_min, ex0)
                h_x1_max = max(h_x1_max, ex1)

            elif ori == "v":
                if not (x0 - 2 <= ex0 <= x1 + 2):
                    continue
                v_top = min(ey0, ey1)
                v_bottom = max(ey0, ey1)
                if v_top > bottom or v_bottom < top:
                    continue
                # 忽略短竖线（< v_height_min）：
                #   · 填充矩形横线的左/右侧边：~1.5pt
                #   · ACS 期刊页眉装饰框角线：~11pt
                #   · 表头矩形框列分隔线：与表头框等高（Pathway 论文中为 ~21pt）
                # v_height_min = max(20, bbox_height*0.1)，大 bbox 时可过滤 21pt 竖线
                if (v_bottom - v_top) < v_height_min:
                    continue
                v_count += 1

        h_ys = sorted(seen_y_bins.values())
        # 若没有有效横线，返回安全的默认值
        if h_x0_min == float("inf"):
            h_x0_min, h_x1_max = x0, x1
        return len(h_ys), h_ys, v_count, h_x0_min, h_x1_max

    @staticmethod
    def _find_col_ranges(
        words: list[dict],
        min_gap_pt: float = 8.0,
    ) -> list[tuple[float, float]]:
        """
        从词语列表中检测列边界（基于相邻词语 x 间距的空隙）。

        算法：
          1. 按 x0 排序所有词语
          2. 合并 x 间距 < min_gap_pt 的相邻词（同一列内的词）
          3. 每段合并区间 → 一个列范围

        返回: [(col_x0, col_x1), ...] 升序排列
        """
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: float(w.get("x0", 0)))
        spans = [(float(w.get("x0", 0)), float(w.get("x1", 0))) for w in sorted_words]

        merged: list[tuple[float, float]] = []
        cur_x0, cur_x1 = spans[0]

        for sx0, sx1 in spans[1:]:
            if sx0 - cur_x1 < min_gap_pt:
                cur_x1 = max(cur_x1, sx1)
            else:
                merged.append((cur_x0, cur_x1))
                cur_x0, cur_x1 = sx0, sx1
        merged.append((cur_x0, cur_x1))

        return merged

    @staticmethod
    def _word_to_col(
        word: dict,
        col_ranges: list[tuple[float, float]],
    ) -> int:
        """将单个词语映射到最近的列（返回列索引，-1 表示列为空）。"""
        if not col_ranges:
            return -1

        wx_center = (float(word.get("x0", 0)) + float(word.get("x1", 0))) / 2

        best_idx = 0
        best_dist = float("inf")

        for i, (cx0, cx1) in enumerate(col_ranges):
            if cx0 <= wx_center <= cx1:
                return i
            dist = min(abs(wx_center - cx0), abs(wx_center - cx1))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    @staticmethod
    def _build_data_rows(
        words: list[dict],
        col_ranges: list[tuple[float, float]],
        row_gap_pt: float = 6.0,
    ) -> list[list[str]]:
        """
        将数据区词语按行（y 间距）分组，再按列分配，返回二维字符串列表。

        row_gap_pt : 相邻词语 y0 差值超过此值时视为新行（需小于实际行高）。
        """
        if not words:
            return []

        def _top(w: dict) -> float:
            return float(w.get("top", w.get("y0", 0)))

        sorted_words = sorted(words, key=lambda w: (_top(w), float(w.get("x0", 0))))

        rows_raw: list[list[dict]] = []
        cur_row = [sorted_words[0]]
        cur_y = _top(sorted_words[0])

        for w in sorted_words[1:]:
            wy = _top(w)
            if wy - cur_y > row_gap_pt:
                rows_raw.append(cur_row)
                cur_row = [w]
                cur_y = wy
            else:
                cur_row.append(w)
        rows_raw.append(cur_row)

        num_cols = len(col_ranges)
        result: list[list[str]] = []

        for row_words in rows_raw:
            row: list[str] = [""] * num_cols
            for word in row_words:
                ci = TableParser._word_to_col(word, col_ranges)
                if 0 <= ci < num_cols:
                    text = word.get("text", "")
                    row[ci] = (row[ci] + " " + text).strip()
            result.append(row)

        return result

    @staticmethod
    def _merge_multiline_cells(data_rows: list[list[str]]) -> list[list[str]]:
        """
        合并折行产生的续行。

        对于偶数列且列数 ≥ 4 的双栏表，左右两半各自独立追踪"最近有 key 的行"，
        续行分别合并入对应的父行（即使同一 y 位置左续右新或左新右续）。

        支持三种"伪新条目"检测（key 非空但实为续行）：
          A. key 以 '(' 开头     → 括号续行，如 "(DE3)" "(CA,USA)"；丢弃 key，仅合并 source
          B. 上一个 key 以 '-' 结尾 → 连字符续行，如 "pCDF-lgtA-" + "wbgO"；拼接入 key
          C. char 为空 且 source 以 '(' 开头 → source 续行，如 "Kanr" + "(WI,USA)"；
               key 移入 char，source 合并入 source

        对于奇数列或列数 < 4，仅检测 col 0 为空 → 续行。
        """
        if not data_rows:
            return data_rows
        n = len(data_rows[0]) if data_rows else 0

        if n >= 4 and n % 2 == 0:
            half = n // 2
            result: list[list[str]] = []
            left_key_idx: int = -1   # 最近左侧有 key 的 result 行索引
            right_key_idx: int = -1  # 最近右侧有 key 的 result 行索引
            left_last_key: str = ""  # 最近左侧 key 值（用于连字符检测）
            right_last_key: str = "" # 最近右侧 key 值

            def _is_pseudo(key: str, char: str, source: str, prev_key: str) -> bool:
                """key 非空但判断为伪新条目（实为续行）的条件。"""
                if char:
                    return False  # char 非空 = 真正的新条目
                if key.startswith("("):
                    return True   # A: 括号续行
                if prev_key.endswith("-"):
                    return True   # B: 连字符续行
                if source.startswith("("):
                    return True   # C: source 续行
                return False

            def _apply_pseudo(prev_row: list[str], offset: int,
                               key: str, char: str, source: str,
                               prev_key: str) -> None:
                """将伪新条目内容合并入 prev_row（offset=0 左半, offset=half 右半）。"""
                if key.startswith("("):
                    # A: 丢弃 key，仅把 source 追加到 prev source
                    if source:
                        prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()
                elif prev_key.endswith("-"):
                    # B: 把 key 拼接到 prev key（补全连字符名称）
                    prev_row[offset] = (prev_key + key).strip()
                    if char:
                        prev_row[offset + 1] = (prev_row[offset + 1] + " " + char).strip()
                    if source:
                        prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()
                else:
                    # C: key 移入 char，source 合并入 source
                    if key:
                        prev_row[offset + 1] = (prev_row[offset + 1] + " " + key).strip()
                    if source:
                        prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()

            for row in data_rows:
                left = row[:half]
                right = row[half:]
                lkey = left[0].strip()
                rkey = right[0].strip()
                lchar = left[1].strip() if half > 1 else ""
                rchar = right[1].strip() if half > 1 else ""
                lsource = left[2].strip() if half > 2 else ""
                rsource = right[2].strip() if half > 2 else ""
                l_has = any(c.strip() for c in left)
                r_has = any(c.strip() for c in right)

                # 伪新条目检测（key 非空但实为续行）
                l_pseudo = bool(lkey) and _is_pseudo(lkey, lchar, lsource, left_last_key)
                r_pseudo = bool(rkey) and _is_pseudo(rkey, rchar, rsource, right_last_key)

                # 续行判定：key 为空 或 伪新条目
                l_is_cont = not lkey or l_pseudo
                r_is_cont = not rkey or r_pseudo

                if not l_is_cont and not r_is_cont:
                    # 双侧均为真正新条目
                    result.append(list(row))
                    left_key_idx = right_key_idx = len(result) - 1
                    left_last_key = lkey
                    right_last_key = rkey

                elif not l_is_cont and r_is_cont:
                    # 左侧新条目，右侧续行
                    result.append(list(left) + [""] * half)
                    left_key_idx = len(result) - 1
                    left_last_key = lkey
                    if r_has and right_key_idx >= 0:
                        prev = result[right_key_idx]
                        if r_pseudo:
                            _apply_pseudo(prev, half, rkey, rchar, rsource, right_last_key)
                            right_last_key = prev[half]  # 更新（连字符拼接后可能变化）
                        else:
                            for i, c in enumerate(right):
                                if c.strip():
                                    prev[half + i] = (prev[half + i] + " " + c).strip()

                elif l_is_cont and not r_is_cont:
                    # 左侧续行，右侧新条目
                    if l_has and left_key_idx >= 0:
                        prev = result[left_key_idx]
                        if l_pseudo:
                            _apply_pseudo(prev, 0, lkey, lchar, lsource, left_last_key)
                            left_last_key = prev[0]
                        else:
                            for i, c in enumerate(left):
                                if c.strip():
                                    prev[i] = (prev[i] + " " + c).strip()
                    result.append([""] * half + list(right))
                    right_key_idx = len(result) - 1
                    right_last_key = rkey

                else:
                    # 双侧均为续行（真正续行 或 伪新条目）
                    if l_has and left_key_idx >= 0:
                        prev = result[left_key_idx]
                        if l_pseudo:
                            _apply_pseudo(prev, 0, lkey, lchar, lsource, left_last_key)
                            left_last_key = prev[0]
                        else:
                            for i, c in enumerate(left):
                                if c.strip():
                                    prev[i] = (prev[i] + " " + c).strip()
                    if r_has and right_key_idx >= 0:
                        prev = result[right_key_idx]
                        if r_pseudo:
                            _apply_pseudo(prev, half, rkey, rchar, rsource, right_last_key)
                            right_last_key = prev[half]
                        else:
                            for i, c in enumerate(right):
                                if c.strip():
                                    prev[half + i] = (prev[half + i] + " " + c).strip()
                    # 双续行不追加新 result 行

            return result
        else:
            # 单栏：col 0 为空 → 续行
            result2: list[list[str]] = []
            for row in data_rows:
                if not row[0].strip() and result2:
                    prev = result2[-1]
                    for i, c in enumerate(row):
                        if c.strip() and i < len(prev):
                            prev[i] = (prev[i] + " " + c).strip()
                else:
                    result2.append(list(row))
            return result2

    @staticmethod
    def _extract_threeline(
        pdf_page,
        bbox_plumb: tuple[float, float, float, float],
        h_ys: list[float],
        lines_x0: float = 0.0,
        lines_x1: float = 0.0,
    ) -> list[list[str]]:
        """
        三线表专用提取器：基于水平线位置 + 词语 x0 坐标检测列。

        - 表头带：第1条横线 ↔ 第2条横线
        - 数据带：第2条横线 ↔ 第3条横线

        h_ys              : 三条横线的 y 坐标（升序，pdfplumber 坐标系）
        lines_x0/lines_x1 : 横线真实 x 范围（修正 ACS 等期刊右侧旋转边注导致的
                            table bbox 偏窄问题）；为 0 时退化为 bbox 本身 x 范围。
        返回              : normalized rows，首行为表头；失败时返回 []
        """
        x0, _top, x1, _bottom = bbox_plumb
        margin = 1.5  # 排除线条本身的像素误差

        # 横线真实 x 范围通常比 table_obj.bbox 更宽（旋转边注不影响线条绘制）
        # 取两者的并集以覆盖全部列头文字
        if lines_x1 > 0:
            eff_x0 = min(x0, lines_x0) - margin
            eff_x1 = max(x1, lines_x1) + margin
        else:
            eff_x0, eff_x1 = x0, x1

        header_bbox = (eff_x0, h_ys[0] + margin, eff_x1, h_ys[1] - margin)
        data_bbox   = (eff_x0, h_ys[1] + margin, eff_x1, h_ys[2] - margin)

        try:
            header_words = pdf_page.within_bbox(header_bbox).extract_words()
            data_words   = pdf_page.within_bbox(data_bbox).extract_words()
        except Exception:
            logger.debug("_extract_threeline: within_bbox 失败", exc_info=True)
            return []

        if not header_words:
            return []

        # 列检测：优先仅用表头；若列数 < 2 则合并表头+数据词再检测
        col_ranges = TableParser._find_col_ranges(
            header_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT
        )
        if len(col_ranges) < 2 and data_words:
            col_ranges = TableParser._find_col_ranges(
                header_words + data_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT
            )

        if len(col_ranges) < CFG.MIN_COLS:
            return []

        # 构建表头行
        num_cols = len(col_ranges)
        header_row: list[str] = [""] * num_cols
        for w in header_words:
            ci = TableParser._word_to_col(w, col_ranges)
            if 0 <= ci < num_cols:
                header_row[ci] = (header_row[ci] + " " + w.get("text", "")).strip()

        # 构建数据行；合并折行产生的续行（col 0 / col half 均空的行）
        data_rows = TableParser._build_data_rows(data_words, col_ranges)
        data_rows = TableParser._merge_multiline_cells(data_rows)

        logger.debug(
            "_extract_threeline: 检测到 %d 列，表头=%s，数据行=%d",
            num_cols, header_row, len(data_rows),
        )
        return [header_row] + data_rows

    @staticmethod
    def _extract_headerbox(
        pdf_page,
        bbox_plumb: tuple[float, float, float, float],
        h_ys: list[float],
        lines_x0: float = 0.0,
        lines_x1: float = 0.0,
    ) -> list[list[str]]:
        """
        表头框专用提取器：2条紧挨横线形成带边框表头 + 下方无边框数据行。

        典型期刊格式：表头用矩形框装饰（有横线+竖线），数据行无任何边框线。
        与三线表的区别：只有 2 条横线（组成表头矩形）而非 3 条（顶/分隔/底）。

        h_ys              : [header_top_y, header_bottom_y]（pdfplumber 坐标系）
        lines_x0/lines_x1 : 横线+竖线联合 x 范围（由 fallback 2 扩展后的裁剪保证）
        """
        x0, _top, x1, _bottom = bbox_plumb
        margin = 1.5
        ph = float(pdf_page.height)
        # 页脚区不提取（页面底部 5%）
        data_y_bottom = ph * 0.95

        # 用横线+竖线联合 x 范围扩展提取区域，保证右侧列不被截断
        if lines_x1 > 0:
            eff_x0 = min(x0, lines_x0) - margin
            eff_x1 = max(x1, lines_x1) + margin
        else:
            eff_x0, eff_x1 = x0 - margin, x1 + margin

        # ── 向上查找更宽的外框顶线，仅扩展 x 范围 ─────────────────────────────
        # 某些期刊（如 ACS）的表头框只是整张表顶/底线之间的子列分隔器。
        # 真实表格左侧还有额外列（如 "strain"）没有子列线，只靠外框顶线约束。
        # 找到外框顶线后：
        #   - 将 eff_x0 扩展至该线左端（以覆盖遗漏的左侧列）
        #   - 不改变 header_bbox 的 y 范围（避免把表格标题混入表头提取区）
        # 遗漏的列标题（如 "strain"）通过后面 data_words 联合检测补全。
        _above_lines = [
            e for e in pdf_page.edges
            if e.get("orientation") == "h"
            and e.get("top", 0) < h_ys[0] - margin
            and h_ys[0] - e.get("top", 0) < 150        # 顶线不超过 150pt 上方
            and e.get("x0", float("inf")) <= x0 + 3    # 左端不超过 bbox 左边
            and e.get("x1", 0.0)          >= x1 - 3    # 右端不短于 bbox 右边
        ]
        if _above_lines:
            _top_line = min(_above_lines, key=lambda e: h_ys[0] - e.get("top", 0))
            eff_x0 = min(eff_x0, _top_line.get("x0", eff_x0) - margin)
            eff_x1 = max(eff_x1, _top_line.get("x1", eff_x1) + margin)

            # 同样找底部边线：避免 data_bbox 把表格之后的正文段落全部纳入，
            # 导致列检测时 x0 分布被正文词汇污染。
            # 寻找在 h_ys[1] 下方、x 范围与外框顶线相近的第一条横线作为表格底线。
            _below_lines = [
                e for e in pdf_page.edges
                if e.get("orientation") == "h"
                and e.get("top", 0) > h_ys[1] + margin
                and e.get("x0", float("inf")) <= eff_x0 + 10
                and e.get("x1", 0.0)          >= eff_x1 - 10
            ]
            if _below_lines:
                _bot_line = min(_below_lines, key=lambda e: e.get("top", 0))
                data_y_bottom = _bot_line.get("top", data_y_bottom) - margin

        header_bbox = (eff_x0, h_ys[0] + margin, eff_x1, h_ys[1] - margin)
        data_bbox   = (eff_x0, h_ys[1] + margin, eff_x1, data_y_bottom)

        try:
            header_words = pdf_page.within_bbox(header_bbox).extract_words()
            data_words   = pdf_page.within_bbox(data_bbox).extract_words()
        except Exception:
            logger.debug("_extract_headerbox: within_bbox 失败", exc_info=True)
            return []

        if not header_words and not data_words:
            return []

        # 列检测：
        # 若检测到外框顶线（_above_lines），表头框只是子列分隔器，左侧还有
        # 未覆盖的列（如 "strain"），需要合并数据词才能检测完整的列结构；
        # 否则优先仅用表头词，必要时再合并数据词。
        if _above_lines and data_words:
            col_ranges = TableParser._find_col_ranges(
                (header_words or []) + data_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT
            )
        else:
            col_ranges = TableParser._find_col_ranges(
                header_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT
            )
            if len(col_ranges) < 2 and data_words:
                col_ranges = TableParser._find_col_ranges(
                    header_words + data_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT
                )

        if len(col_ranges) < CFG.MIN_COLS:
            return []

        num_cols = len(col_ranges)
        header_row: list[str] = [""] * num_cols
        for w in (header_words or []):
            ci = TableParser._word_to_col(w, col_ranges)
            if 0 <= ci < num_cols:
                header_row[ci] = (header_row[ci] + " " + w.get("text", "")).strip()

        # 构建数据行；合并折行产生的续行
        data_rows = TableParser._build_data_rows(data_words, col_ranges)
        data_rows = TableParser._merge_multiline_cells(data_rows)

        logger.debug(
            "_extract_headerbox: 检测到 %d 列，表头=%s，数据行=%d",
            num_cols, header_row, len(data_rows),
        )
        return [header_row] + data_rows

    # ── 旋转表格提取 ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_rotated_table(pdf_page) -> list[list[str]]:
        """
        检测并提取页面上旋转 90° 的表格（常见于 ACS 等期刊，将宽表旋转以节省版面）。

        pdfplumber / PyMuPDF 对旋转字符的解析结果通常是乱序的单列表格，无法直接使用。
        本方法通过以下步骤重建正确的表格：

          1. 提取旋转字符（matrix[1] != 0）并按 x0 精确分组
             （每个 x0 组对应旋转表格的一行）
          2. 识别标题行（text 以 'table' 开头）和表头行（含 ≥ 3 个列标题关键词）
          3. 从表头行的大间距（> 15pt）推导列数和列名
          4. 识别"主行"（字符覆盖最后一列的 top 范围）和"续行"
          5. 从主行的列间隙（> 10pt）聚类推导列边界
             边界 = (MIN(gap 前沿) + MAX(gap 后沿)) / 2，取最严格约束
          6. 将每行字符按列边界分配，重建表格内容

        返回：表格内容（首行为表头）；检测失败或旋转字符不足时返回 []。
        """
        from collections import defaultdict

        HEADER_KWS = frozenset({
            'host', 'characteristic', 'medium', 'titer', 'reference',
            'strain', 'enzyme', 'substrate', 'product', 'yield',
            'condition', 'temperature', 'concentration',
        })
        MIN_ROTATED  = 30     # 最少旋转字符数，低于此值跳过
        MIN_GAP_PT   = 10.0   # 列间最小间隙（pt）
        CLUSTER_TOL  = 8.0    # after_top 聚类容差（pt）
        HEADER_GAP   = 15.0   # 表头内列标签之间的最小间隙（pt）

        # ── 1. 提取旋转字符 ──────────────────────────────────────────────────
        rotated = [
            c for c in pdf_page.chars
            if abs(c.get("matrix", [1, 0, 0, 1, 0, 0])[1]) > 0.1
        ]
        if len(rotated) < MIN_ROTATED:
            return []

        # ── 2. 按 x0 精确分组（±0.5pt 容差，通过 round×2/2 实现）────────────
        groups: dict[float, list] = defaultdict(list)
        for c in rotated:
            key = round(c["x0"] * 2) / 2
            groups[key].append(c)
        sorted_keys = sorted(groups.keys())
        if len(sorted_keys) < 3:
            return []

        # ── 3. 识别标题行（前两个 x0 组中 text 起始含 'table'）────────────────
        title_key: float | None = None
        for k in sorted_keys[:2]:
            txt = "".join(
                c["text"] for c in sorted(groups[k], key=lambda c: -c["top"])
            ).lower()
            if txt.startswith("table"):
                title_key = k
                break

        # ── 4. 识别表头行（含 ≥ 3 个列标题关键词）──────────────────────────────
        header_key: float | None = None
        for k in sorted_keys:
            if k == title_key:
                continue
            txt = "".join(
                c["text"] for c in sorted(groups[k], key=lambda c: -c["top"])
            ).lower()
            if sum(1 for kw in HEADER_KWS if kw in txt) >= 3:
                header_key = k
                break
        if header_key is None:
            return []

        # ── 5. 从表头行提取列名（top DESC 排列，间距 > HEADER_GAP 视为列分隔）──
        hchars = sorted(groups[header_key], key=lambda c: -c["top"])
        header_col_groups: list[list] = []
        curr: list = [hchars[0]]
        for i in range(1, len(hchars)):
            gap = hchars[i - 1]["top"] - hchars[i]["top"]
            if gap > HEADER_GAP:
                header_col_groups.append(curr)
                curr = []
            curr.append(hchars[i])
        header_col_groups.append(curr)

        n_cols = len(header_col_groups)
        if n_cols < 2:
            return []

        header_row = [
            "".join(c["text"] for c in g) for g in header_col_groups
        ]

        # ── 6. 识别"主行"（min_top ≤ 最后列 max_top + 10）和"续行"──────────────
        data_keys = [k for k in sorted_keys if k not in (title_key, header_key)]
        if not data_keys:
            return []

        last_col_max_top = max(c["top"] for c in header_col_groups[-1])
        main_thresh = last_col_max_top + 10.0

        main_keys = [k for k in data_keys if min(c["top"] for c in groups[k]) <= main_thresh]
        cont_keys = [k for k in data_keys if k not in main_keys]
        if not main_keys:
            return []

        # 续行分配给最近的主行（x0 距离最小）
        row_groups: dict[float, list[float]] = {k: [k] for k in main_keys}
        for ck in cont_keys:
            nearest = min(main_keys, key=lambda mk: abs(mk - ck))
            row_groups[nearest].append(ck)

        # ── 7. 从主行列间隙（> MIN_GAP_PT）推导列边界 ───────────────────────────
        # 聚类策略：after_top 相近（± CLUSTER_TOL）的 gap 归为同一边界
        # 边界 = (MIN(before_top) + MAX(after_top)) / 2
        gap_pairs: list[tuple[float, float]] = []
        for mk in main_keys:
            chars = sorted(groups[mk], key=lambda c: -c["top"])
            tops = [c["top"] for c in chars]
            for i in range(len(tops) - 1):
                gap = tops[i] - tops[i + 1]
                if gap > MIN_GAP_PT:
                    gap_pairs.append((tops[i], tops[i + 1]))

        if len(gap_pairs) < n_cols - 1:
            return []

        # 聚类 gap_pairs by after_top（after_top 更稳定）
        used: set[int] = set()
        clusters: list[tuple[float, list[float], list[float]]] = []
        for i, (bt, at) in enumerate(gap_pairs):
            if i in used:
                continue
            cluster_bt: list[float] = [bt]
            cluster_at: list[float] = [at]
            for j in range(i + 1, len(gap_pairs)):
                if j in used:
                    continue
                bt2, at2 = gap_pairs[j]
                if abs(at - at2) <= CLUSTER_TOL:
                    cluster_bt.append(bt2)
                    cluster_at.append(at2)
                    used.add(j)
            used.add(i)
            boundary = (min(cluster_bt) + max(cluster_at)) / 2.0
            clusters.append((boundary, cluster_bt, cluster_at))

        # 按 boundary 降序（top DESC = 从左列到右列的顺序）
        clusters.sort(key=lambda x: -x[0])
        if len(clusters) < n_cols - 1:
            return []

        # 取前 n_cols-1 个聚类作为列边界（高 top → 低 top，即左到右）
        boundaries = sorted([c[0] for c in clusters[: n_cols - 1]], reverse=True)

        # ── 8. 重建表格 ──────────────────────────────────────────────────────────
        def _top_to_col(top: float) -> int:
            """列索引 = 边界中 top ≤ boundary 的个数（即该 top 值跨越了几条边界线）。"""
            return sum(1 for b in boundaries if top <= b)

        result: list[list[str]] = [header_row]
        for mk in sorted(main_keys):
            # 逐 x0 子行处理，避免不同物理行的字符按 top 值混排（折行单元格问题）：
            # 对每一列，先收集各子行（x0 升序）的字符片段，然后判断：
            #   - 若某子行的 top 范围与已有片段存在 ≥ 5pt 的重叠
            #     → 折行单元格（如"Glc-minimal medium," + "lactose..."），
            #        按 x0 顺序拼接（不混排）
            #   - 否则（top 范围不重叠，如 β 浮动字符）→ 全局 top 降序排序
            col_subrows: list[list[list]] = [[] for _ in range(n_cols)]
            for k in sorted(row_groups[mk]):
                sub: list[list] = [[] for _ in range(n_cols)]
                for c in groups[k]:
                    ci = _top_to_col(c["top"])
                    if 0 <= ci < n_cols:
                        sub[ci].append(c)
                for ci in range(n_cols):
                    if sub[ci]:
                        col_subrows[ci].append(sub[ci])

            cols: list[str] = [""] * n_cols
            for ci in range(n_cols):
                subs = col_subrows[ci]
                if not subs:
                    continue
                if len(subs) == 1:
                    # 单子行：直接 top 降序
                    cols[ci] = "".join(
                        c["text"] for c in sorted(subs[0], key=lambda c: -c["top"])
                    )
                    continue
                # 多子行：检测 top 范围是否重叠（重叠 → 折行，不重叠 → 浮动字符）
                ranges = [
                    (min(c["top"] for c in s), max(c["top"] for c in s))
                    for s in subs
                ]
                # 检测是否有任意两子行的 top 范围重叠（容差 5pt）
                def _overlaps(r1: tuple, r2: tuple) -> bool:
                    lo1, hi1 = r1; lo2, hi2 = r2
                    return hi1 > lo2 - 5 and hi2 > lo1 - 5
                has_wrap = any(
                    _overlaps(ranges[i], ranges[j])
                    for i in range(len(ranges))
                    for j in range(i + 1, len(ranges))
                )
                if has_wrap:
                    # 折行单元格：按 x0 升序拼接各子行（top 降序）
                    cols[ci] = "".join(
                        "".join(c["text"] for c in sorted(s, key=lambda c: -c["top"]))
                        for s in subs
                    )
                else:
                    # 浮动字符（如 β）：全局 top 降序，字符落入正确位置
                    all_c = [c for s in subs for c in s]
                    cols[ci] = "".join(
                        c["text"] for c in sorted(all_c, key=lambda c: -c["top"])
                    )

            if not any(col.strip() for col in cols):
                continue
            result.append(cols)

        return result if len(result) >= 2 else []

    # ── 通用工具（normalize / markdown / validate）─────────────────────────────

    @staticmethod
    def _unsplit_twin_columns(normalized: list[list[str]]) -> list[list[str]]:
        """
        检测并折叠双栏同构表（ACS 等期刊空间节省排版）。

        某些期刊为节省版面，将 N 列表格分左右两半并排，pdfplumber/PyMuPDF 提取时
        得到 2N 列。检测信号：首 4 行中存在至少 1 行，其左半与右半内容完全相同
        （且非全空）。满足条件时将右半数据追加到左半下方，折叠为 N 列。

        多行表头（如 "strains or" / "plasmids" 分两行）会被合并为单行。
        """
        if not normalized:
            return normalized
        n_cols = len(normalized[0])
        if n_cols < 4 or n_cols % 2 != 0:
            return normalized

        half = n_cols // 2
        scan = min(4, len(normalized))

        # 跳过前置标题行（camelot 会把表格标题文本合并进第一行）
        # 判据：右半全空 且 左半仅 1 个单元格有内容 → 标题行，跳过
        skip_start = 0
        for ri, row in enumerate(normalized[:2]):
            left  = [c.strip() for c in row[:half]]
            right = [c.strip() for c in row[half:]]
            if (not any(c for c in right)
                    and sum(1 for c in left if c) == 1):
                skip_start = ri + 1  # 标题行，跳过
            else:
                break

        # 找对称表头行（左半 == 右半 且非全空）
        header_rows_raw: list[list[str]] = []
        data_start = skip_start
        for ri, row in enumerate(normalized[skip_start:skip_start + scan]):
            left  = [c.strip() for c in row[:half]]
            right = [c.strip() for c in row[half:]]
            if left == right and any(c for c in left):
                header_rows_raw.append(list(row[:half]))
                data_start = skip_start + ri + 1
            else:
                break  # 第一个不对称行 → 数据区起点

        # 未检测到对称表头，或表格全为表头（无数据行）→ 不折叠
        if not header_rows_raw or data_start >= len(normalized):
            return normalized

        # 多行表头合并为单行（"strains or" + "plasmids" → "strains or plasmids"）
        if len(header_rows_raw) > 1:
            merged_header = []
            for i in range(half):
                parts = [r[i].strip() for r in header_rows_raw if r[i].strip()]
                merged_header.append(" ".join(parts))
            header_row: list[str] = merged_header
        else:
            header_row = header_rows_raw[0]

        # 数据行：左半行紧接右半行交替追加
        result: list[list[str]] = [header_row]
        for row in normalized[data_start:]:
            left_part  = list(row[:half])
            right_part = list(row[half:])
            if any(c.strip() for c in left_part):
                result.append(left_part)
            if any(c.strip() for c in right_part):
                result.append(right_part)

        logger.debug(
            "_unsplit_twin_columns: %d 列 → %d 列，%d 行",
            n_cols, half, len(result),
        )
        return result

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

