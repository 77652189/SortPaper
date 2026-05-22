from __future__ import annotations

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.models import BBox, TableExtractionAttempt, TableRegion

REGION_PARSER_FAMILIES = {
    "region_guided": "region_guided did not produce a valid table",
    "pdfplumber_region": "pdfplumber region extraction did not produce a valid table",
}


def bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0.0

    inter_area = (x1 - x0) * (y1 - y0)
    a_area = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(0.0, (bx1 - bx0) * (by1 - by0))
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def bbox_overlap_ratio(inner: BBox, outer: BBox) -> float:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer

    x0 = max(ix0, ox0)
    y0 = max(iy0, oy0)
    x1 = min(ix1, ox1)
    y1 = min(iy1, oy1)
    if x0 >= x1 or y0 >= y1:
        return 0.0

    inter_area = (x1 - x0) * (y1 - y0)
    inner_area = max(0.0, (ix1 - ix0) * (iy1 - iy0))
    return inter_area / inner_area if inner_area > 0 else 0.0


def find_best_region(
    chunk: LayoutChunk,
    regions: list[TableRegion],
    *,
    min_overlap: float = 0.10,
) -> TableRegion | None:
    page_regions = [region for region in regions if region.page == chunk.page]
    if not page_regions:
        return None

    best_region: TableRegion | None = None
    best_score = 0.0
    for region in page_regions:
        overlap = max(
            bbox_iou(chunk.bbox, region.bbox),
            bbox_overlap_ratio(chunk.bbox, region.bbox),
            bbox_overlap_ratio(region.bbox, chunk.bbox),
        )
        score = overlap + region.confidence * 0.1
        if overlap >= min_overlap and score > best_score:
            best_region = region
            best_score = score
    return best_region


def attach_region_diagnostics(
    chunks: list[LayoutChunk],
    regions: list[TableRegion],
) -> list[LayoutChunk]:
    if not regions:
        return chunks

    for chunk in chunks:
        if chunk.content_type != "table":
            continue
        region = find_best_region(chunk, regions)
        if region is None:
            chunk.metadata.setdefault("table_region", None)
            chunk.metadata.setdefault("table_region_match", "none")
            continue
        chunk.metadata["table_region"] = region.to_metadata()
        chunk.metadata["table_region_match"] = "matched"
    return chunks


def build_region_attempts(
    chunks: list[LayoutChunk],
    regions: list[TableRegion],
) -> dict[str, list[dict]]:
    attempts_by_region: dict[str, list[TableExtractionAttempt]] = {
        region.region_id: [] for region in regions
    }
    successful_families: dict[str, set[str]] = {
        region.region_id: set() for region in regions
    }

    for chunk in chunks:
        if chunk.content_type != "table":
            continue
        region_id = chunk.metadata.get("source_region_id")
        parser_name = str(chunk.metadata.get("parser", "unknown"))
        if region_id and region_id in attempts_by_region:
            family = _parser_family(parser_name)
            successful_families[region_id].add(family)
            attempts_by_region[region_id].append(_success_attempt(chunk, parser_name, region_id))
            continue

        matched_region = chunk.metadata.get("table_region") or {}
        matched_region_id = matched_region.get("region_id")
        if matched_region_id in attempts_by_region:
            successful_families[matched_region_id].add("broad_fallback_match")
            attempts_by_region[matched_region_id].append(_success_attempt(
                chunk, parser_name, matched_region_id
            ))

    for region in regions:
        if region.band == "weak":
            continue
        for family, reason in REGION_PARSER_FAMILIES.items():
            if family in successful_families[region.region_id]:
                continue
            attempts_by_region[region.region_id].append(TableExtractionAttempt(
                parser=family,
                page=region.page,
                bbox=region.bbox,
                succeeded=False,
                region_id=region.region_id,
                failure_reason=reason,
            ))

    return {
        region_id: [attempt.to_metadata() for attempt in attempts]
        for region_id, attempts in attempts_by_region.items()
    }


def attach_region_attempts(
    chunks: list[LayoutChunk],
    attempts_by_region: dict[str, list[dict]],
) -> list[LayoutChunk]:
    for chunk in chunks:
        if chunk.content_type != "table":
            continue
        region_id = chunk.metadata.get("source_region_id")
        if not region_id:
            region = chunk.metadata.get("table_region") or {}
            region_id = region.get("region_id")
        if region_id and region_id in attempts_by_region:
            chunk.metadata["region_extraction_attempts"] = attempts_by_region[region_id]
    return chunks


def _parser_family(parser_name: str) -> str:
    if parser_name.startswith("region_guided"):
        return "region_guided"
    if parser_name.startswith("pdfplumber_region"):
        return "pdfplumber_region"
    return parser_name


def _success_attempt(
    chunk: LayoutChunk,
    parser_name: str,
    region_id: str,
) -> TableExtractionAttempt:
    return TableExtractionAttempt(
        parser=parser_name,
        page=chunk.page,
        bbox=chunk.bbox,
        succeeded=True,
        region_id=region_id,
        rows=int(chunk.metadata.get("rows", 0) or 0),
        cols=int(chunk.metadata.get("cols", 0) or 0),
    )
