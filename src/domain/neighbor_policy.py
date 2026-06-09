from __future__ import annotations

import re
from typing import Any


def neighbor_candidates_from_entries(
    results: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    *,
    limit: int,
    window: int,
) -> list[dict[str, Any]]:
    if not results:
        return []

    entries_by_paper: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        paper_id = str((entry.get("payload") or {}).get("paper_id") or "")
        if not paper_id:
            continue
        entries_by_paper.setdefault(paper_id, []).append(entry)

    best_by_id: dict[Any, dict[str, Any]] = {}
    for seed in results[:limit]:
        seed_payload = seed.get("payload") or {}
        paper_id = str(seed_payload.get("paper_id") or "")
        if not paper_id:
            continue
        seed_chunk_id = str(seed_payload.get("chunk_id") or seed.get("id") or "")
        base_score = float(seed.get("score") or 0.0)
        for entry in entries_by_paper.get(paper_id, []):
            payload = entry.get("payload") or {}
            chunk_id = str(payload.get("chunk_id") or entry.get("id") or "")
            if chunk_id and chunk_id == seed_chunk_id:
                continue
            distance = neighbor_distance(seed_payload, payload, window=window)
            if distance is None:
                continue
            score = base_score * max(0.55, 0.98 - 0.08 * distance)
            item = {
                "id": entry["id"],
                "score": score,
                "payload": payload,
                "neighbor_score": score,
                "neighbor_distance": distance,
                "neighbor_source_chunk_id": seed_chunk_id,
            }
            existing = best_by_id.get(entry["id"])
            if existing is None or item["score"] > existing["score"]:
                best_by_id[entry["id"]] = item

    neighbors = list(best_by_id.values())
    neighbors.sort(key=lambda item: item.get("neighbor_score", 0.0), reverse=True)
    return neighbors[:limit]


def context_chunks_from_neighbors(
    results: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
    *,
    total_limit: int,
) -> list[dict[str, Any]]:
    existing_ids = {item.get("id") for item in results}
    existing_chunks = {
        (
            str((item.get("payload") or {}).get("paper_id") or ""),
            str((item.get("payload") or {}).get("chunk_id") or item.get("id") or ""),
        )
        for item in results
    }

    context: list[dict[str, Any]] = []
    for item in neighbors:
        payload = item.get("payload") or {}
        key = (
            str(payload.get("paper_id") or ""),
            str(payload.get("chunk_id") or item.get("id") or ""),
        )
        if item.get("id") in existing_ids or key in existing_chunks:
            continue
        item["context_expansion"] = True
        context.append(item)
        if len(context) >= total_limit:
            break
    return context


def neighbor_distance(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    window: int,
) -> int | None:
    source_global = as_int(source.get("global_order"))
    candidate_global = as_int(candidate.get("global_order"))
    if source_global is not None and candidate_global is not None:
        distance = abs(candidate_global - source_global)
        return distance if distance <= window else None

    if source.get("page") is None or candidate.get("page") is None:
        return chunk_id_neighbor_distance(source, candidate, window=window)
    if str(source.get("page")) != str(candidate.get("page")):
        return None

    source_order = as_int(source.get("order_in_page"))
    candidate_order = as_int(candidate.get("order_in_page"))
    if source_order is not None and candidate_order is not None:
        distance = abs(candidate_order - source_order)
        return distance if distance <= window else None

    if bbox_centers_are_close(source, candidate):
        return 1
    return chunk_id_neighbor_distance(source, candidate, window=window)


def chunk_id_neighbor_distance(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    window: int,
) -> int | None:
    source_location = parse_chunk_id_location(source.get("chunk_id"))
    candidate_location = parse_chunk_id_location(candidate.get("chunk_id"))
    if not source_location or not candidate_location:
        return None
    if source_location["page"] != candidate_location["page"]:
        return None
    if (
        source_location.get("column") is not None
        and candidate_location.get("column") is not None
        and source_location["column"] != candidate_location["column"]
    ):
        return None
    y_distance = abs(float(source_location["y"]) - float(candidate_location["y"]))
    if y_distance <= 120:
        return 1
    if y_distance <= 240 and window >= 2:
        return 2
    return None


def parse_chunk_id_location(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    match = re.search(
        r"_p(?P<page>\d+)(?:_col(?P<column>\d+))?_y(?P<y>-?\d+(?:\.\d+)?)_x(?P<x>-?\d+(?:\.\d+)?)",
        str(value),
    )
    if not match:
        return None
    return {
        "page": int(match.group("page")),
        "column": int(match.group("column")) if match.group("column") is not None else None,
        "y": float(match.group("y")),
        "x": float(match.group("x")),
    }


def bbox_centers_are_close(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    center_ratio: float = 0.12,
) -> bool:
    source_bbox = normalize_bbox(source.get("bbox"))
    candidate_bbox = normalize_bbox(candidate.get("bbox"))
    if not source_bbox or not candidate_bbox:
        return False

    width = as_float(source.get("page_width")) or as_float(candidate.get("page_width"))
    height = as_float(source.get("page_height")) or as_float(candidate.get("page_height"))
    if not width or not height:
        return False

    sx, sy = bbox_center(source_bbox)
    cx, cy = bbox_center(candidate_bbox)
    normalized_distance = ((sx / width - cx / width) ** 2 + (sy / height - cy / height) ** 2) ** 0.5
    return normalized_distance <= center_ratio


def as_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0
