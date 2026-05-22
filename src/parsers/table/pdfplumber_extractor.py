from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.parsers.config import TABLE_PARSER as CFG
from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table import cleanup
from src.parsers.table import geometry
from src.parsers.table import structure_extractors
from src.parsers.table.models import TableRegion

logger = logging.getLogger(__name__)

PostProcessTable = Callable[..., LayoutChunk | None]
ValidateAndFallback = Callable[..., LayoutChunk | None]


class PdfplumberTableExtractor:
    """Extract table candidates with pdfplumber and local geometry fallbacks."""

    def __init__(
        self,
        *,
        pdf_path: str,
        post_process_table: PostProcessTable,
        validate_and_fallback: ValidateAndFallback,
    ) -> None:
        self.pdf_path = pdf_path
        self._post_process_table = post_process_table
        self._validate_and_fallback = validate_and_fallback

    def parse(self, feedback: str | None = None) -> list[LayoutChunk]:
        import pdfplumber

        settings = self._table_settings(feedback)
        chunks: list[LayoutChunk] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_width = float(page.width) if page.width else 0
                page_height = float(page.height) if page.height else 0
                found_tables = page.find_tables(table_settings=settings)

                page_text = (page.extract_text() or "").lower()
                has_table_keyword = "table" in page_text

                if (not found_tables or not self._has_data(page, found_tables)) and has_table_keyword:
                    text_line_tables = page.find_tables(table_settings=dict(
                        settings, vertical_strategy="text"
                    ))
                    if text_line_tables and self._has_data(page, text_line_tables):
                        found_tables = text_line_tables

                if (
                    (not found_tables or not self._has_data(page, found_tables))
                    and page_width > 0
                    and has_table_keyword
                ):
                    fallback_tables = self._find_cropped_text_tables(page, page_width, page_height)
                    if fallback_tables and self._has_data(page, fallback_tables):
                        found_tables = fallback_tables

                rotated_chunk = self._extract_rotated_chunk(
                    page, page_index, page_width, page_height, found_tables
                )
                if rotated_chunk:
                    chunks.append(rotated_chunk)
                    rx0, rx1 = rotated_chunk.bbox[0], rotated_chunk.bbox[2]
                    found_tables = [
                        table for table in found_tables
                        if not (table.bbox[0] < rx1 + 5 and table.bbox[2] > rx0 - 5)
                    ]
                    logger.info(
                        "p%d rotated table: filtered %d broken tables",
                        page_index, len(found_tables),
                    )

                for table_index, table_obj in enumerate(found_tables):
                    chunk = self._process_table_obj(
                        page, table_obj, page_index, page_width, page_height, table_index
                    )
                    if chunk:
                        chunks.append(chunk)

        return chunks

    def parse_regions(
        self,
        regions: list[TableRegion],
        feedback: str | None = None,
    ) -> list[LayoutChunk]:
        """Extract tables inside detector regions before the broad page scan."""
        import pdfplumber

        if not regions:
            return []

        settings = self._table_settings(feedback)
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
                page = pdf.pages[page_index - 1]
                page_width = float(page.width) if page.width else 0
                page_height = float(page.height) if page.height else 0
                for region_index, region in enumerate(page_regions):
                    clipped_bbox = self._clip_bbox_to_page(
                        region.bbox,
                        getattr(page, "bbox", (0.0, 0.0, page_width, page_height)),
                    )
                    if clipped_bbox is None:
                        continue
                    if clipped_bbox != region.bbox:
                        region = TableRegion(
                            page=region.page,
                            bbox=clipped_bbox,
                            confidence=region.confidence,
                            evidence=region.evidence,
                            source=region.source,
                        )
                    if self._region_has_rotation(region):
                        rotated_chunk = self._extract_rotated_chunk(
                            page,
                            page_index,
                            page_width,
                            page_height,
                            [],
                            region=region,
                            table_index=region_index,
                        )
                        if rotated_chunk:
                            rotated_chunk.metadata["source_region_id"] = region.region_id
                            rotated_chunk.metadata["table_region"] = region.to_metadata()
                            rotated_chunk.metadata["table_region_match"] = "matched"
                            chunks.append(rotated_chunk)
                            continue
                    chunk = self._extract_region_geometry(
                        page, region, page_width, page_height, region_index
                    )
                    if chunk:
                        chunks.append(chunk)
                        continue
                    chunks.extend(self._extract_region_text_tables(
                        page, region, page_width, page_height, region_index, settings
                    ))
        return chunks

    @staticmethod
    def _table_settings(feedback: str | None) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": CFG.PDFPLUMBER_SNAP_TOLERANCE,
            "join_tolerance": CFG.PDFPLUMBER_JOIN_TOLERANCE,
        }
        if feedback:
            settings["snap_tolerance"] = CFG.PDFPLUMBER_RETRY_SNAP_TOLERANCE
            settings["join_tolerance"] = CFG.PDFPLUMBER_RETRY_JOIN_TOLERANCE
        return settings

    @staticmethod
    def _has_data(page, tables: list) -> bool:
        for table in tables:
            rows = table.extract() or []
            if len(rows) > 1:
                return True
            if len(rows) == 1:
                h_count, h_ys, v_count, _, _ = geometry.count_threeline_lines(
                    page, table.bbox
                )
                if (
                    h_count == 2
                    and v_count == 0
                    and len(h_ys) == 2
                    and (h_ys[1] - h_ys[0]) < 30
                ):
                    return True
        return False

    @staticmethod
    def _find_cropped_text_tables(page, page_width: float, page_height: float) -> list:
        wide_h = [
            edge for edge in page.edges
            if edge.get("orientation") == "h"
            and (edge.get("x1", 0) - edge.get("x0", 0)) > page_width * 0.2
        ]
        if not wide_h:
            return []

        h_y_min = min(edge.get("top", 0) for edge in wide_h)
        h_y_max = max(edge.get("top", 0) for edge in wide_h)
        header_band_h = h_y_max - h_y_min + 5

        header_box_v_xs = [
            edge.get("x0", 0)
            for edge in page.edges
            if edge.get("orientation") == "v"
            and (edge.get("bottom", 0) - edge.get("top", 0)) <= header_band_h + 5
            and edge.get("top", 0) >= h_y_min - 3
        ]

        h_x0 = min(edge["x0"] for edge in wide_h)
        h_x1 = max(edge["x1"] for edge in wide_h)
        v_x_max = max(header_box_v_xs, default=h_x1)

        clip_x0 = max(0.0, min(h_x0, min(header_box_v_xs, default=h_x0)) - 2)
        clip_x1 = min(page_width, max(h_x1, v_x_max) + 2)
        clipped = page.within_bbox((clip_x0, 0, clip_x1, page_height))
        return clipped.find_tables(table_settings={
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": CFG.PDFPLUMBER_TEXT_SNAP_TOLERANCE,
            "join_tolerance": CFG.PDFPLUMBER_TEXT_JOIN_TOLERANCE,
        })

    def _extract_region_geometry(
        self,
        page,
        region: TableRegion,
        page_width: float,
        page_height: float,
        region_index: int,
    ) -> LayoutChunk | None:
        h_count, h_ys, v_count, lines_x0, lines_x1 = geometry.count_threeline_lines(
            page, region.bbox
        )
        threeline_ys = self._select_threeline_ys(h_ys) if v_count == 0 else None
        if threeline_ys is not None:
            normalized = self._extract_geometry_rows(
                "threeline", page, region.bbox, threeline_ys, lines_x0, lines_x1, []
            )
        elif (
            h_count == 2
            and v_count == 0
            and len(h_ys) == 2
            and (h_ys[1] - h_ys[0]) < 30
        ):
            normalized = self._extract_geometry_rows(
                "headerbox", page, region.bbox, h_ys, lines_x0, lines_x1, []
            )
        else:
            return None

        if not cleanup.is_valid_table(normalized):
            return None

        chunk = self._validate_and_fallback(
            normalized,
            region.bbox,
            region.page,
            page_width,
            page_height,
            parser_name="pdfplumber_region_geometry",
            pdf_page=page,
            table_index=region_index,
        )
        if chunk:
            chunk.metadata["source_region_id"] = region.region_id
        return chunk

    def _extract_region_text_tables(
        self,
        page,
        region: TableRegion,
        page_width: float,
        page_height: float,
        region_index: int,
        settings: dict[str, Any],
    ) -> list[LayoutChunk]:
        region_bbox = self._clip_bbox_to_page(
            region.bbox,
            getattr(page, "bbox", (0.0, 0.0, page_width, page_height)),
        )
        if region_bbox is None:
            return []
        try:
            clipped = page.within_bbox(region_bbox)
            tables = clipped.find_tables(table_settings=dict(
                settings,
                vertical_strategy="text",
                horizontal_strategy="text",
                snap_tolerance=CFG.PDFPLUMBER_TEXT_SNAP_TOLERANCE,
                join_tolerance=CFG.PDFPLUMBER_TEXT_JOIN_TOLERANCE,
            ))
        except Exception as exc:
            logger.warning("pdfplumber region extraction failed %s: %s", region.region_id, exc)
            return []

        chunks: list[LayoutChunk] = []
        for offset, table_obj in enumerate(tables):
            chunk = self._process_table_obj(
                page,
                table_obj,
                region.page,
                page_width,
                page_height,
                region_index + offset,
                vision_clip_bbox=region_bbox,
            )
            if chunk:
                chunk.metadata["parser"] = "pdfplumber_region_text"
                chunk.metadata["source_region_id"] = region.region_id
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

    @staticmethod
    def _region_has_rotation(region: TableRegion) -> bool:
        return any(item.source == "rotation" for item in region.evidence)

    def _extract_rotated_chunk(
        self,
        page,
        page_index: int,
        page_width: float,
        page_height: float,
        found_tables: list,
        *,
        region: TableRegion | None = None,
        table_index: int = 0,
    ) -> LayoutChunk | None:
        rotated_table = structure_extractors.detect_rotated_table(page)
        if not rotated_table:
            return None

        normalized = cleanup.normalize_table(rotated_table)
        if not cleanup.is_valid_table(normalized):
            return None

        rotated_chars = [
            char for char in page.chars
            if abs(char.get("matrix", [1, 0, 0, 1, 0, 0])[1]) > 0.1
        ]
        if region is not None:
            rotated_chars = [
                char for char in rotated_chars
                if self._char_center_in_bbox(char, region.bbox, tolerance=8.0)
            ]
        if not rotated_chars:
            return None

        x0 = float(min(char["x0"] for char in rotated_chars))
        x1 = float(max(char.get("x1", char["x0"] + 10) for char in rotated_chars))
        y_top = float(min(char["top"] for char in rotated_chars))
        y_bottom = float(max(char.get("bottom", char["top"] + 10) for char in rotated_chars))
        bbox = self._clip_bbox_to_page(
            (x0, y_top, x1, y_bottom),
            getattr(page, "bbox", (0.0, 0.0, page_width, page_height)),
        )
        if bbox is None:
            return None

        logger.info(
            "p%d rotated table: %d rows x %d cols, source tables=%d",
            page_index, len(normalized), len(normalized[0]), len(found_tables),
        )
        chunk = self._validate_and_fallback(
            normalized,
            bbox,
            page_index,
            page_width,
            page_height,
            parser_name="pdfplumber_rotated",
            pdf_page=page,
            table_index=table_index,
        )
        if chunk:
            chunk.metadata["rotated"] = True
            if region is not None:
                chunk.metadata["source_region_id"] = region.region_id
        return chunk

    @staticmethod
    def _char_center_in_bbox(
        char: dict,
        bbox: tuple[float, float, float, float],
        *,
        tolerance: float = 0.0,
    ) -> bool:
        x0, y0, x1, y1 = bbox
        cx = (float(char.get("x0", 0.0)) + float(char.get("x1", char.get("x0", 0.0)))) / 2.0
        cy = (float(char.get("top", 0.0)) + float(char.get("bottom", char.get("top", 0.0)))) / 2.0
        return (
            x0 - tolerance <= cx <= x1 + tolerance
            and y0 - tolerance <= cy <= y1 + tolerance
        )

    def _process_table_obj(
        self,
        page,
        table_obj,
        page_index: int,
        page_width: float,
        page_height: float,
        table_index: int,
        vision_clip_bbox: tuple[float, float, float, float] | None = None,
    ) -> LayoutChunk | None:
        rows = table_obj.extract()
        if not rows:
            return None

        h_count, h_ys, v_count, lines_x0, lines_x1 = geometry.count_threeline_lines(
            page, table_obj.bbox
        )
        threeline_ys = self._select_threeline_ys(h_ys) if v_count == 0 else None
        if threeline_ys is not None:
            normalized = self._extract_geometry_rows(
                "threeline", page, table_obj.bbox, threeline_ys, lines_x0, lines_x1, rows
            )
        elif (
            h_count == 2
            and v_count == 0
            and len(h_ys) == 2
            and (h_ys[1] - h_ys[0]) < 30
        ):
            normalized = self._extract_geometry_rows(
                "headerbox", page, table_obj.bbox, h_ys, lines_x0, lines_x1, rows
            )
        else:
            bbox = tuple(float(value) for value in table_obj.bbox)
            return self._post_process_table(
                rows,
                bbox,
                page_index,
                page_width,
                page_height,
                parser_name="pdfplumber",
                pdf_page=page,
                table_index=table_index,
                vision_clip_bbox=vision_clip_bbox,
            )

        bbox = tuple(float(value) for value in table_obj.bbox)
        return self._validate_and_fallback(
            normalized,
            bbox,
            page_index,
            page_width,
            page_height,
            parser_name="pdfplumber",
            pdf_page=page,
            table_index=table_index,
            vision_clip_bbox=vision_clip_bbox,
        )

    @staticmethod
    def _extract_geometry_rows(
        kind: str,
        page,
        bbox: tuple,
        h_ys: list[float],
        lines_x0: float,
        lines_x1: float,
        fallback_rows: list[list[str | None]],
    ) -> list[list[str]]:
        if kind == "threeline":
            extracted = structure_extractors.extract_threeline(
                page, bbox, h_ys, lines_x0, lines_x1
            )
        else:
            extracted = structure_extractors.extract_headerbox(
                page, bbox, h_ys, lines_x0, lines_x1
            )

        if extracted and len(extracted) >= 2:
            extracted = cleanup.unsplit_twin_columns(extracted)
            extracted = cleanup.truncate_at_body_text(extracted)
            if extracted and len(extracted) >= 2:
                return extracted

        return cleanup.truncate_at_body_text(
            cleanup.normalize_table(fallback_rows)
        ) if fallback_rows else []

    @staticmethod
    def _select_threeline_ys(h_ys: list[float]) -> list[float] | None:
        """Pick the table top/header/bottom lines when caption rules add extra lines."""
        if len(h_ys) < 3:
            return None
        if len(h_ys) == 3:
            return h_ys

        best: tuple[float, list[float]] | None = None
        for top_idx in range(len(h_ys) - 2):
            for mid_idx in range(top_idx + 1, len(h_ys) - 1):
                header_gap = h_ys[mid_idx] - h_ys[top_idx]
                if not (
                    CFG.THREELINE_MIN_HEADER_GAP
                    <= header_gap
                    <= CFG.THREELINE_MAX_HEADER_GAP
                ):
                    continue
                for bottom_idx in range(mid_idx + 1, len(h_ys)):
                    body_gap = h_ys[bottom_idx] - h_ys[mid_idx]
                    if body_gap < CFG.THREELINE_MIN_BODY_GAP:
                        continue
                    score = (
                        body_gap
                        - abs(header_gap - CFG.THREELINE_IDEAL_HEADER_GAP)
                        * CFG.THREELINE_HEADER_GAP_PENALTY
                    )
                    candidate = [h_ys[top_idx], h_ys[mid_idx], h_ys[bottom_idx]]
                    if best is None or score > best[0]:
                        best = (score, candidate)
        return best[1] if best is not None else None
