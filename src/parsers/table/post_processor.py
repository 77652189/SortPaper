from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.parsers.layout_chunk import LayoutChunk, infer_column
from src.parsers.table.models import TableExtractionAttempt

logger = logging.getLogger(__name__)


class TablePostProcessor:
    """Normalize, validate, optionally repair, and package extracted table rows."""

    def __init__(
        self,
        *,
        pdf_path: str,
        normalize_table: Callable[[list[list[str | None]]], list[list[str]]],
        unsplit_twin_columns: Callable[[list[list[str]]], list[list[str]]],
        truncate_at_body_text: Callable[[list[list[str]]], list[list[str]]],
        is_valid_table: Callable[[list[list[str]]], bool],
        structural_quality: Callable[[list[list[str]]], dict[str, Any]],
        refine_table_bbox: Callable[[Any, tuple, float, float], tuple[float, float, float, float]],
        parse_table_with_vision: Callable[[str, int, tuple[float, float, float, float]], list[list[str]]],
        to_markdown: Callable[[list[list[str]]], str],
        allow_vision_fallback: bool = True,
        repair_table_structure: Callable[[list[list[str]]], tuple[list[list[str]], list[str]]] | None = None,
    ) -> None:
        self.pdf_path = pdf_path
        self.allow_vision_fallback = allow_vision_fallback
        self._normalize_table = normalize_table
        self._unsplit_twin_columns = unsplit_twin_columns
        self._truncate_at_body_text = truncate_at_body_text
        self._repair_table_structure = repair_table_structure
        self._is_valid_table = is_valid_table
        self._structural_quality = structural_quality
        self._refine_table_bbox = refine_table_bbox
        self._parse_table_with_vision = parse_table_with_vision
        self._to_markdown = to_markdown

    @staticmethod
    def _bbox_to_list(bbox: tuple) -> list[float]:
        return [float(v) for v in bbox]

    def post_process(
        self,
        raw_rows: list[list[str | None]],
        bbox: tuple,
        page_index: int,
        page_width: float,
        page_height: float,
        *,
        parser_name: str = "unknown",
        pdf_page: Any = None,
        table_index: int = 0,
        vision_clip_bbox: tuple[float, float, float, float] | None = None,
    ) -> LayoutChunk | None:
        normalized = self._unsplit_twin_columns(self._normalize_table(raw_rows))
        repair_actions: list[str] = []
        if self._repair_table_structure is not None:
            normalized, repair_actions = self._repair_table_structure(normalized)
        normalized = self._truncate_at_body_text(normalized)
        return self.validate_and_fallback(
            normalized, bbox, page_index, page_width, page_height,
            parser_name=parser_name, pdf_page=pdf_page, table_index=table_index,
            vision_clip_bbox=vision_clip_bbox,
            repair_actions=repair_actions,
        )

    def validate_and_fallback(
        self,
        normalized: list[list[str]],
        bbox: tuple,
        page_index: int,
        page_width: float,
        page_height: float,
        *,
        parser_name: str = "unknown",
        pdf_page: Any = None,
        table_index: int = 0,
        vision_clip_bbox: tuple[float, float, float, float] | None = None,
        repair_actions: list[str] | None = None,
    ) -> LayoutChunk | None:
        if not self._is_valid_table(normalized):
            return None

        quality = self._structural_quality(normalized)
        logger.info(
            "p%d table #%d [%s]: consistency=%.2f fill=%.2f fallback=%s",
            page_index, table_index, parser_name,
            quality["consistency_score"], quality["fill_rate"],
            quality["fallback_to_vision"],
        )

        fallback_reasons = list(quality.get("fallback_reasons", []))
        vision_fallback_needed = bool(quality["fallback_to_vision"])
        vision_fallback_attempted = False
        vision_fallback_succeeded = False
        refined_bbox = None

        if vision_fallback_needed and self.allow_vision_fallback:
            if vision_clip_bbox is not None:
                refined_bbox = tuple(float(value) for value in vision_clip_bbox)
            else:
                refined_bbox = (
                    self._refine_table_bbox(pdf_page, bbox, page_width, page_height)
                    if pdf_page is not None else bbox
                )

        if vision_fallback_needed and self.allow_vision_fallback:
            vision_fallback_attempted = True
            logger.warning(
                "p%d table #%d [%s]: low structural quality, trying Vision fallback",
                page_index, table_index, parser_name,
            )
            vision_rows = self._parse_table_with_vision(
                self.pdf_path, page_index - 1, refined_bbox,
            )
            if vision_rows and len(vision_rows) >= 2:
                normalized = vision_rows
                vision_fallback_needed = False
                vision_fallback_succeeded = True
                logger.warning(
                    "p%d table #%d [%s]: Vision fallback succeeded, %d rows x %d cols",
                    page_index, table_index, parser_name,
                    len(normalized), len(normalized[0]) if normalized[0] else 0,
                )

        column = infer_column(page_width, bbox[0], bbox[2])
        markdown = self._to_markdown(normalized)
        metadata = {
            "parser": parser_name,
            "rows": len(normalized),
            "cols": len(normalized[0]),
            "table_index": table_index,
            "extraction_attempt": TableExtractionAttempt(
                parser=parser_name,
                page=page_index,
                bbox=tuple(float(value) for value in bbox),
                succeeded=True,
                rows=len(normalized),
                cols=len(normalized[0]),
            ).to_metadata(),
            "structural_quality": quality,
            "original_bbox": self._bbox_to_list(bbox),
            "vision_fallback_reasons": fallback_reasons,
            "vision_fallback_attempted": vision_fallback_attempted,
            "vision_fallback_succeeded": vision_fallback_succeeded,
            "table_repair_actions": list(repair_actions or []),
        }
        if refined_bbox is not None:
            metadata["refined_bbox"] = self._bbox_to_list(refined_bbox)
            metadata["vision_clip_bbox"] = self._bbox_to_list(refined_bbox)

        if vision_fallback_needed and not self.allow_vision_fallback:
            metadata["structure_reparse_needed"] = True
            metadata["vision_fallback_disabled"] = True
            metadata["manual_review_needed"] = True
            metadata["structure_reparse_reason"] = "low structural quality; table Vision fallback is disabled"
        elif vision_fallback_needed:
            metadata["vision_fallback_needed"] = True
            metadata["vision_fallback_reason"] = (
                "low structural quality; preview mode did not call GPT Vision"
                if not self.allow_vision_fallback else
                "low structural quality; GPT Vision fallback returned no usable table"
            )

        return LayoutChunk(
            content_type="table",
            raw_content=markdown,
            page=page_index,
            bbox=bbox,
            column=column,
            order_in_page=table_index,
            metadata=metadata,
        )
