"""Build table-region discovery and placeholder chunks."""

from __future__ import annotations

import re

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.diagnostics import find_best_region
from src.parsers.table.models import TableRegion


def region_discovery_chunks(regions: list[TableRegion]) -> list[LayoutChunk]:
    chunks: list[LayoutChunk] = []
    for index, region in enumerate(regions):
        if region.band == "weak":
            continue
        caption = region_caption_text(region)
        table_label = table_label_from_caption(caption)
        metadata = {
            "parser": "table_region_detector",
            "rows": 0,
            "cols": 0,
            "table_region": region.to_metadata(),
            "table_label": table_label,
            "table_caption": caption,
            "table_region_match": "region_only",
            "table_region_unparsed": True,
            "table_region_discovery_only": True,
            "manual_review_needed": True,
            "structure_extraction_pending": True,
            "structure_reparse_needed": True,
            "vision_fallback_disabled": True,
            "vision_fallback_attempted": False,
            "vision_fallback_succeeded": False,
            "caption": caption,
            "unparsed_region_reason": (
                "table region discovery mode: structure extraction is intentionally disabled"
            ),
            "region_extraction_attempts": [],
        }
        raw_content = region_discovery_content(region, caption)
        chunks.append(LayoutChunk(
            content_type="table",
            raw_content=raw_content,
            page=region.page,
            bbox=region.bbox,
            column=2,
            order_in_page=index,
            metadata=metadata,
        ))
    return chunks


def unparsed_region_chunks(
    chunks: list[LayoutChunk],
    regions: list[TableRegion],
    attempts_by_region: dict[str, list[dict]] | None = None,
) -> list[LayoutChunk]:
    attempts_by_region = attempts_by_region or {}
    placeholders: list[LayoutChunk] = []
    for region in regions:
        if region.band == "weak":
            continue
        if not should_emit_unparsed_placeholder(region):
            continue
        probe = LayoutChunk(
            content_type="table",
            raw_content="",
            page=region.page,
            bbox=region.bbox,
            column=0,
            order_in_page=0,
        )
        if find_best_region(probe, [
            TableRegion(
                page=chunk.page,
                bbox=chunk.bbox,
                confidence=1.0,
                evidence=[],
                source="extracted_table",
            )
            for chunk in chunks
            if chunk.content_type == "table"
        ], min_overlap=0.20):
            continue

        caption = region_caption_text(region)
        metadata = {
            "parser": "table_region_detector",
            "rows": 0,
            "cols": 0,
            "table_region": region.to_metadata(),
            "table_label": table_label_from_caption(caption),
            "table_caption": caption,
            "table_region_match": "unparsed",
            "table_region_unparsed": True,
            "manual_review_needed": True,
            "structure_extraction_pending": True,
            "vision_fallback_attempted": False,
            "vision_fallback_succeeded": False,
            "unparsed_region_reason": (
                "table-like region detected but no structure extractor produced a usable table"
            ),
            "region_extraction_attempts": attempts_by_region.get(region.region_id, []),
        }
        placeholders.append(LayoutChunk(
            content_type="table",
            raw_content="[Table region detected]\nstructure extraction pending",
            page=region.page,
            bbox=region.bbox,
            column=2,
            order_in_page=0,
            metadata=metadata,
        ))
    return placeholders


def region_caption_text(region: TableRegion) -> str:
    for evidence in region.evidence:
        if evidence.source == "caption":
            return str(evidence.details.get("text", "")).strip()
    return ""


def table_label_from_caption(caption: str) -> str:
    match = re.search(
        r"\b(supplementary\s+table|table|tab\.)\s*(s?\d+)\b",
        caption,
        re.IGNORECASE,
    )
    if not match:
        return ""
    prefix = "Supplementary Table" if "supplementary" in match.group(1).lower() else "Table"
    return f"{prefix} {match.group(2).upper()}"


def region_discovery_content(region: TableRegion, caption: str) -> str:
    bbox = ", ".join(f"{value:.1f}" for value in region.bbox)
    lines = [
        "[Table region detected]",
        f"page: {region.page}",
        f"bbox: [{bbox}]",
        f"confidence: {region.confidence:.2f}",
        f"band: {region.band}",
    ]
    if caption:
        lines.insert(1, f"caption: {caption}")
    return "\n".join(lines)


def should_emit_unparsed_placeholder(region: TableRegion) -> bool:
    """Keep preview placeholders useful: show caption-backed or very strong misses only."""
    evidence_sources = {item.source for item in region.evidence}
    if "caption" in evidence_sources:
        return True
    return region.confidence >= 0.85
