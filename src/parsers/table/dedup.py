"""Table chunk deduplication and vision clip expansion helpers."""

from __future__ import annotations

from src.parsers.config import TABLE_PARSER as CFG
from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.diagnostics import bbox_overlap_ratio


def deduplicate_chunks(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
    """去重：同一区域或包含关系保留结构质量更好的表格候选。"""
    if not chunks:
        return chunks

    unique: list[LayoutChunk] = []
    for chunk in chunks:
        duplicate_index: int | None = None
        duplicate_reason = ""
        duplicate_score = 0.0
        for index, existing in enumerate(unique):
            if chunk.page != existing.page:
                continue
            iou = _compute_iou(chunk.bbox, existing.bbox)
            contained = max(
                bbox_overlap_ratio(chunk.bbox, existing.bbox),
                bbox_overlap_ratio(existing.bbox, chunk.bbox),
            )
            same_region_exact = same_region_exact_duplicate(chunk, existing, iou)
            same_region = same_source_region(chunk, existing)
            if same_region and not same_region_exact:
                priority_delta = abs(
                    parser_source_priority(str(chunk.metadata.get("parser", "")))
                    - parser_source_priority(str(existing.metadata.get("parser", "")))
                )
                if (
                    iou <= CFG.DEDUP_IOU_THRESHOLD
                    and contained < CFG.SAME_REGION_DEDUP_CONTAINED_RATIO
                ) or priority_delta < CFG.SAME_REGION_DEDUP_MIN_PRIORITY_DELTA:
                    continue
            subset = is_contained_subset(chunk.bbox, existing.bbox)
            overflow_subset = is_overflowing_subset(chunk.bbox, existing.bbox)
            if (
                same_region_exact
                or iou > CFG.DEDUP_IOU_THRESHOLD
                or contained >= CFG.DEDUP_CONTAINED_RATIO
                or subset
                or overflow_subset
            ):
                duplicate_index = index
                duplicate_score = max(iou, contained)
                duplicate_reason = (
                    "same_region_exact" if same_region_exact
                    else "iou" if iou > CFG.DEDUP_IOU_THRESHOLD
                    else "contained_subset" if subset
                    else "overlap_subset" if overflow_subset
                    else "contained"
                )
                break

        if duplicate_index is None:
            unique.append(chunk)
            continue

        existing = unique[duplicate_index]
        if chunk_quality_score(chunk) > chunk_quality_score(existing):
            record_dedup_replacement(
                winner=chunk,
                loser=existing,
                reason=duplicate_reason,
                score=duplicate_score,
            )
            unique[duplicate_index] = chunk
        else:
            record_dedup_replacement(
                winner=existing,
                loser=chunk,
                reason=duplicate_reason,
                score=duplicate_score,
            )
    return unique


def expand_vision_clips_by_region(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
    by_region: dict[tuple[int, str], list[LayoutChunk]] = {}
    for chunk in chunks:
        region_id = chunk.metadata.get("source_region_id") or (
            chunk.metadata.get("table_region") or {}
        ).get("region_id")
        if region_id:
            by_region.setdefault((chunk.page, region_id), []).append(chunk)

    for group in by_region.values():
        if len(group) < 2:
            continue
        union_bbox = bbox_union(*(chunk.bbox for chunk in group))
        for chunk in group:
            if chunk.metadata.get("vision_fallback_needed"):
                expand_vision_clip_bbox(chunk, union_bbox)
                chunk.metadata["region_fragment_count_for_vision_clip"] = len(group)
    return chunks


def is_contained_subset(a: tuple, b: tuple) -> bool:
    a_area = bbox_area(a)
    b_area = bbox_area(b)
    if a_area <= 0 or b_area <= 0:
        return False

    small, large = (a, b) if a_area <= b_area else (b, a)
    small_area = min(a_area, b_area)
    large_area = max(a_area, b_area)
    area_ratio = small_area / large_area
    if area_ratio > CFG.DEDUP_SUBSET_MAX_AREA_RATIO:
        return False

    return bbox_overlap_ratio(small, large) >= CFG.DEDUP_SUBSET_CONTAINED_RATIO


def same_source_region(a: LayoutChunk, b: LayoutChunk) -> bool:
    a_region = a.metadata.get("source_region_id") or (
        a.metadata.get("table_region") or {}
    ).get("region_id")
    b_region = b.metadata.get("source_region_id") or (
        b.metadata.get("table_region") or {}
    ).get("region_id")
    return bool(a_region and b_region and a_region == b_region and a.page == b.page)


def same_region_exact_duplicate(
    a: LayoutChunk,
    b: LayoutChunk,
    iou: float,
) -> bool:
    if not same_source_region(a, b):
        return False
    if iou >= 0.95:
        return True

    diffs = [abs(float(av) - float(bv)) for av, bv in zip(a.bbox, b.bbox)]
    return max(diffs, default=float("inf")) <= CFG.SAME_REGION_EXACT_BBOX_TOLERANCE


def is_overflowing_subset(a: tuple, b: tuple) -> bool:
    """Detect a table fragment whose bbox spills into nearby prose.

    Some PDF text extractors merge a left-column table row with same-y right-column
    body text. The fragment is not geometrically contained, but its vertical span
    and left edge still place it inside the parent table region.
    """
    a_area = bbox_area(a)
    b_area = bbox_area(b)
    if a_area <= 0 or b_area <= 0:
        return False

    small, large = (a, b) if a_area <= b_area else (b, a)
    sx0, sy0, sx1, sy1 = small
    lx0, ly0, lx1, ly1 = large
    small_h = max(0.0, sy1 - sy0)
    vertical_overlap = max(0.0, min(sy1, ly1) - max(sy0, ly0))
    if small_h <= 0 or vertical_overlap / small_h < 0.85:
        return False

    left_inside = lx0 - 12 <= sx0 <= lx1 + 12
    starts_inside_vertically = ly0 - 12 <= sy0 <= ly1 + 12
    horizontal_overlap = max(0.0, min(sx1, lx1) - max(sx0, lx0))
    small_w = max(0.0, sx1 - sx0)
    return (
        left_inside
        and starts_inside_vertically
        and small_w > 0
        and horizontal_overlap / small_w >= 0.35
    )


def bbox_area(bbox: tuple) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def record_dedup_replacement(
    *,
    winner: LayoutChunk,
    loser: LayoutChunk,
    reason: str,
    score: float,
) -> None:
    if should_expand_vision_clip_from_dedup(winner, loser):
        expand_vision_clip_bbox(winner, loser.bbox)

    winner.metadata.setdefault("dedup_replaced", []).append({
        "chunk_id": loser.chunk_id,
        "parser": loser.metadata.get("parser", ""),
        "bbox": [float(value) for value in loser.bbox],
        "reason": reason,
        "score": round(float(score), 3),
    })


def should_expand_vision_clip_from_dedup(winner: LayoutChunk, loser: LayoutChunk) -> bool:
    if not winner.metadata.get("vision_fallback_needed"):
        return False
    if winner.page != loser.page:
        return False

    winner_region = winner.metadata.get("source_region_id") or (
        winner.metadata.get("table_region") or {}
    ).get("region_id")
    loser_region = loser.metadata.get("source_region_id") or (
        loser.metadata.get("table_region") or {}
    ).get("region_id")
    if winner_region and loser_region and winner_region == loser_region:
        return True

    return is_vertically_related_fragment(winner.bbox, loser.bbox)


def expand_vision_clip_bbox(chunk: LayoutChunk, extra_bbox: tuple) -> None:
    current_bbox = tuple(chunk.metadata.get("vision_clip_bbox") or chunk.bbox)
    expanded = bbox_union(current_bbox, chunk.bbox, extra_bbox)
    chunk.metadata["vision_clip_bbox"] = [float(value) for value in expanded]
    chunk.metadata["refined_bbox"] = [float(value) for value in expanded]
    chunk.metadata["dedup_expanded_vision_clip"] = True


def bbox_union(*bboxes: tuple) -> tuple[float, float, float, float]:
    valid = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
    return (
        min(float(bbox[0]) for bbox in valid),
        min(float(bbox[1]) for bbox in valid),
        max(float(bbox[2]) for bbox in valid),
        max(float(bbox[3]) for bbox in valid),
    )


def is_vertically_related_fragment(a: tuple, b: tuple) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    min_width = min(max(0.0, ax1 - ax0), max(0.0, bx1 - bx0))
    if min_width <= 0 or x_overlap / min_width < 0.60:
        return False
    vertical_gap = max(by0 - ay1, ay0 - by1, 0.0)
    return vertical_gap <= 80.0


def chunk_quality_score(chunk: LayoutChunk) -> float:
    metadata = chunk.metadata or {}
    parser_name = str(metadata.get("parser", ""))
    source_priority = parser_source_priority(parser_name)
    quality = metadata.get("structural_quality") or {}
    consistency = float(quality.get("consistency_score", 0.0) or 0.0)
    fill_rate = float(quality.get("fill_rate", 0.0) or 0.0)
    rows = int(metadata.get("rows", 0) or 0)
    cols = int(metadata.get("cols", 0) or 0)

    score = source_priority
    score += consistency * CFG.CHUNK_SCORE_CONSISTENCY_WEIGHT
    score += fill_rate * CFG.CHUNK_SCORE_FILL_WEIGHT
    score += min(rows, CFG.CHUNK_SCORE_MAX_ROWS) * CFG.CHUNK_SCORE_ROW_WEIGHT
    score += min(cols, CFG.CHUNK_SCORE_MAX_COLS) * CFG.CHUNK_SCORE_COL_WEIGHT
    if metadata.get("vision_fallback_needed"):
        score -= CFG.CHUNK_SCORE_VISION_PENALTY
    if metadata.get("table_region_unparsed"):
        score -= CFG.CHUNK_SCORE_UNPARSED_PENALTY
    return score


def parser_source_priority(parser_name: str) -> float:
    if parser_name.startswith("region_guided_geometry"):
        return 40.0
    if parser_name.startswith("pdfplumber_region"):
        return 38.0
    if parser_name.startswith("region_guided_text"):
        return 36.0
    if parser_name.startswith("keyword_guided"):
        return 34.0
    if parser_name.startswith("pdfplumber"):
        return 30.0
    if parser_name == "table_region_detector":
        return 0.0
    return 10.0


def _compute_iou(b1: tuple, b2: tuple) -> float:
    """计算两个 bbox 的 IoU (Intersection over Union)。"""
    if not b1 or not b2:
        return 0.0
    x0_1, y0_1, x1_1, y1_1 = b1
    x0_2, y0_2, x1_2, y1_2 = b2

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
