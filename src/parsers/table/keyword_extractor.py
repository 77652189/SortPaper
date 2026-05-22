from __future__ import annotations

import logging
import re
from collections.abc import Callable

from src.parsers.config import TABLE_PARSER as CFG
from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table import cleanup
from src.parsers.table import geometry
from src.parsers.table import structure_extractors
from src.parsers.table.models import TableRegion

logger = logging.getLogger(__name__)

ValidateAndFallback = Callable[..., LayoutChunk | None]


class KeywordGuidedTableExtractor:
    """Locate table captions first, then extract the nearby table region."""

    TITLE_RE = re.compile(
        r"^(?:"
        r"(?:Table|TABLE|Tab\.)\s+S?\d+[.:]"
        r"|Supplementary\s+Table\s+S?\d+[.:]"
        r"|表\s*[S号]?\d+[.：]"
        r")"
    )
    MAX_TABLE_HEIGHT_PT = CFG.KEYWORD_MAX_TABLE_HEIGHT_PT
    LINE_SNAP = CFG.KEYWORD_LINE_SNAP

    def __init__(
        self,
        *,
        pdf_path: str,
        validate_and_fallback: ValidateAndFallback,
    ) -> None:
        self.pdf_path = pdf_path
        self._validate_and_fallback = validate_and_fallback

    def parse(self) -> list[LayoutChunk]:
        import pdfplumber

        chunks: list[LayoutChunk] = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, pdf_page in enumerate(pdf.pages):
                page_width = float(pdf_page.width)
                page_height = float(pdf_page.height)
                words = pdf_page.extract_words()
                if not words:
                    continue

                title_anchors = self._find_title_anchors(words)
                if not title_anchors:
                    continue

                wide_h = [
                    edge for edge in pdf_page.edges
                    if edge.get("orientation") == "h"
                    and (edge.get("x1", 0) - edge.get("x0", 0)) > page_width * 0.2
                ]

                for _title_top, title_bottom, _title_x0, _title_x1 in title_anchors:
                    region_top = title_bottom
                    below = [
                        edge for edge in wide_h
                        if edge["top"] > region_top
                        and edge["top"] < region_top + self.MAX_TABLE_HEIGHT_PT
                    ]

                    if below:
                        region_bottom = max(edge["top"] for edge in below) + 5
                        clip_x0 = max(0.0, min(edge["x0"] for edge in below) - 2)
                        clip_x1 = min(page_width, max(edge["x1"] for edge in below) + 2)
                    else:
                        region_bottom, clip_x0, clip_x1 = (
                            geometry.detect_columnar_table_bounds(
                                pdf_page, region_top, page_width, page_height
                            )
                        )

                    if region_top >= region_bottom or clip_x0 >= clip_x1:
                        logger.warning(
                            "keyword_guided p%d: invalid bbox (%.1f,%.1f,%.1f,%.1f)",
                            page_index + 1, clip_x0, region_top, clip_x1, region_bottom,
                        )
                        continue

                    clip_bbox = (clip_x0, region_top, clip_x1, region_bottom)
                    chunk = self._extract_geometry(
                        pdf_page, clip_bbox, page_index + 1, page_width, page_height
                    )
                    if chunk:
                        chunks.append(chunk)
                        continue

                    chunks.extend(
                        self._extract_text_tables(
                            pdf_page, clip_bbox, page_index + 1, page_width, page_height
                        )
                    )

        return chunks

    def parse_regions(self, regions: list[TableRegion]) -> list[LayoutChunk]:
        """Extract structures from detector regions; region detection lives elsewhere."""
        import pdfplumber

        by_page: dict[int, list[TableRegion]] = {}
        for region in regions:
            if region.band == "weak":
                continue
            by_page.setdefault(region.page, []).append(region)

        chunks: list[LayoutChunk] = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page_regions in by_page.items():
                if page_index < 1 or page_index > len(pdf.pages):
                    continue
                pdf_page = pdf.pages[page_index - 1]
                page_width = float(pdf_page.width)
                page_height = float(pdf_page.height)
                for region_index, region in enumerate(page_regions):
                    region_bbox = self._clip_bbox_to_page(
                        region.bbox,
                        getattr(pdf_page, "bbox", (0.0, 0.0, page_width, page_height)),
                    )
                    if region_bbox is None:
                        continue
                    chunk = self._extract_geometry(
                        pdf_page,
                        region_bbox,
                        page_index,
                        page_width,
                        page_height,
                        parser_name="region_guided_geometry",
                        table_index=region_index,
                    )
                    if chunk:
                        chunk.metadata["source_region_id"] = region.region_id
                        chunks.append(chunk)
                        continue

                    text_chunks = self._extract_text_tables(
                        pdf_page,
                        region_bbox,
                        page_index,
                        page_width,
                        page_height,
                        parser_name="region_guided_text",
                        table_index_offset=region_index,
                    )
                    for text_chunk in text_chunks:
                        text_chunk.metadata["source_region_id"] = region.region_id
                    chunks.extend(text_chunks)

        return chunks

    def _find_title_anchors(
        self, words: list[dict],
    ) -> list[tuple[float, float, float, float]]:
        words_sorted = sorted(words, key=lambda word: (word["top"], word["x0"]))
        lines: list[list[dict]] = []
        for word in words_sorted:
            placed = False
            for line in reversed(lines):
                if abs(word["top"] - line[0]["top"]) <= self.LINE_SNAP:
                    line.append(word)
                    placed = True
                    break
            if not placed:
                lines.append([word])

        title_anchors: list[tuple[float, float, float, float]] = []
        for line in lines:
            line_sorted = sorted(line, key=lambda word: word["x0"])
            line_text = " ".join(word["text"] for word in line_sorted).strip()
            if not self.TITLE_RE.match(line_text):
                continue
            if len(line_text.split()) > 60:
                continue
            title_anchors.append((
                min(word["top"] for word in line_sorted),
                max(word["bottom"] for word in line_sorted),
                min(word["x0"] for word in line_sorted),
                max(word["x1"] for word in line_sorted),
            ))
        return title_anchors

    def _extract_geometry(
        self,
        pdf_page,
        clip_bbox: tuple[float, float, float, float],
        page_index: int,
        page_width: float,
        page_height: float,
        *,
        parser_name: str = "keyword_guided_geometry",
        table_index: int = 0,
    ) -> LayoutChunk | None:
        h_count, h_ys, v_count, line_x0, line_x1 = geometry.count_threeline_lines(
            pdf_page, clip_bbox
        )
        normalized: list[list[str]] | None = None
        if h_count == 3 and v_count == 0:
            extracted = structure_extractors.extract_threeline(
                pdf_page, clip_bbox, h_ys, line_x0, line_x1
            )
            normalized = self._clean_geometry_rows(extracted)
        elif (
            h_count == 2
            and v_count == 0
            and len(h_ys) == 2
            and (h_ys[1] - h_ys[0]) < 30
        ):
            extracted = structure_extractors.extract_headerbox(
                pdf_page, clip_bbox, h_ys, line_x0, line_x1
            )
            normalized = self._clean_geometry_rows(extracted)

        if normalized is None or not cleanup.is_valid_table(normalized):
            return None

        return self._validate_and_fallback(
            normalized,
            tuple(float(value) for value in clip_bbox),
            page_index,
            page_width,
            page_height,
            parser_name=parser_name,
            pdf_page=pdf_page,
            table_index=table_index,
        )

    @staticmethod
    def _clean_geometry_rows(rows: list[list[str]]) -> list[list[str]] | None:
        if not rows or len(rows) < 2:
            return None
        rows = cleanup.unsplit_twin_columns(rows)
        rows = cleanup.truncate_at_body_text(rows)
        return rows if len(rows) >= 2 else None

    def _extract_text_tables(
        self,
        pdf_page,
        clip_bbox: tuple[float, float, float, float],
        page_index: int,
        page_width: float,
        page_height: float,
        *,
        parser_name: str = "keyword_guided_text",
        table_index_offset: int = 0,
    ) -> list[LayoutChunk]:
        clip_bbox = self._clip_bbox_to_page(clip_bbox, getattr(pdf_page, "bbox", (0.0, 0.0, page_width, page_height)))
        if clip_bbox is None:
            return []
        try:
            clipped = pdf_page.within_bbox(clip_bbox)
            tables = clipped.find_tables(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": CFG.PDFPLUMBER_TEXT_SNAP_TOLERANCE,
                "join_tolerance": CFG.PDFPLUMBER_TEXT_JOIN_TOLERANCE,
            })
        except Exception as exc:
            logger.warning("keyword_guided p%d: %s", page_index, exc)
            return []

        chunks: list[LayoutChunk] = []
        for table_index, table_obj in enumerate(tables):
            rows = table_obj.extract()
            if not rows:
                continue
            normalized = cleanup.truncate_at_body_text(
                cleanup.unsplit_twin_columns(cleanup.normalize_table(rows))
            )
            if not cleanup.is_valid_table(normalized):
                continue

            bbox = tuple(float(value) for value in table_obj.bbox)
            chunk = self._validate_and_fallback(
                normalized,
                bbox,
                page_index,
                page_width,
                page_height,
                parser_name=parser_name,
                pdf_page=pdf_page,
                table_index=table_index_offset + table_index,
            )
            if chunk:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _clip_bbox_to_page(
        bbox: tuple[float, float, float, float],
        page_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float] | None:
        px0, py0, px1, py1 = (float(value) for value in page_bbox)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        clipped = (max(px0, x0), max(py0, y0), min(px1, x1), min(py1, y1))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return clipped
