from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


BBoxLike = Sequence[float]


@dataclass(frozen=True)
class TableCandidateScore:
    score: float
    tier: str
    reasons: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_bbox_candidate(
    *,
    original_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    original_bbox: BBoxLike,
    candidate_bbox: BBoxLike,
) -> dict[str, Any]:
    original = score_table_candidate(
        metadata=original_metadata,
        bbox=original_bbox,
        region_bbox=original_bbox,
    )
    candidate = score_table_candidate(
        metadata=candidate_metadata,
        bbox=candidate_bbox,
        region_bbox=original_bbox,
    )
    if candidate.metrics["rows"] < 2 or candidate.metrics["cols"] < 2:
        return {"score": 0.0, "adoptable": False, "reason": "candidate_table_too_small"}

    area_ratio = _area(candidate_bbox) / max(1.0, _area(original_bbox))
    if area_ratio < 0.35 or area_ratio > 1.80:
        return {
            "score": round(candidate.score - original.score, 3),
            "adoptable": False,
            "reason": "bbox_area_change_too_large",
        }

    score_delta = candidate.score - original.score
    original_prose = float(original.metrics["prose_cell_ratio"] or 0.0)
    candidate_prose = float(candidate.metrics["prose_cell_ratio"] or 0.0)
    original_consistency = float(original.metrics["consistency_score"] or 0.0)
    candidate_consistency = float(candidate.metrics["consistency_score"] or 0.0)
    original_fill = float(original.metrics["fill_rate"] or 0.0)
    candidate_fill = float(candidate.metrics["fill_rate"] or 0.0)
    adoptable = (
        score_delta >= 0.08
        and candidate_consistency + 0.05 >= original_consistency
        and candidate_fill + 0.10 >= original_fill
        and (candidate_prose < original_prose or candidate_consistency > original_consistency or original_metadata.get("table_region_unparsed"))
    )
    return {
        "score": round(candidate.score, 3),
        "score_delta": round(score_delta, 3),
        "candidate_tier": candidate.tier,
        "original_tier": original.tier,
        "adoptable": bool(adoptable),
        "reason": "strictly_better" if adoptable else "not_strictly_better",
    }


def score_table_candidate(
    *,
    metadata: dict[str, Any] | None,
    bbox: BBoxLike | None = None,
    region_bbox: BBoxLike | None = None,
    text_context: Any | dict[str, Any] | None = None,
    parser_succeeded: bool | None = None,
) -> TableCandidateScore:
    """Score a table candidate using portable structural signals, not labels."""
    metadata = metadata or {}
    quality = metadata.get("structural_quality") or {}
    rows = int(metadata.get("rows") or 0)
    cols = int(metadata.get("cols") or 0)
    consistency = _clamp01(float(quality.get("consistency_score") or 0.0))
    fill = _clamp01(float(quality.get("fill_rate") or 0.0))
    prose = _clamp01(float(quality.get("prose_cell_ratio") or 0.0))
    fallback_reasons = list(quality.get("fallback_reasons") or [])
    succeeded = bool(parser_succeeded if parser_succeeded is not None else rows > 0 and cols > 0)

    reasons: list[str] = []
    score = 0.0
    if succeeded:
        score += 0.15
    else:
        reasons.append("no_parser_output")
    if rows >= 3:
        score += 0.15
    else:
        reasons.append("too_few_rows")
    if cols >= 2:
        score += 0.15
    else:
        reasons.append("too_few_cols")
    score += 0.25 * consistency
    score += 0.20 * fill
    score += 0.10 * (1.0 - prose)

    if prose >= 0.25:
        reasons.append("prose_contamination")
        score -= 0.15
    if consistency < 0.55 and rows >= 3 and cols >= 2:
        reasons.append("low_consistency")
        score -= 0.08
    if fill < 0.40 and rows >= 3 and cols >= 2:
        reasons.append("low_fill")
        score -= 0.05
    for reason in fallback_reasons:
        if reason not in reasons:
            reasons.append(str(reason))

    area_ratio = None
    if bbox and region_bbox:
        area_ratio = _area(bbox) / max(1.0, _area(region_bbox))
        if area_ratio < 0.35:
            reasons.append("bbox_much_smaller_than_region")
            score -= 0.08
        elif area_ratio > 1.80:
            reasons.append("bbox_much_larger_than_region")
            score -= 0.08

    context = _context_dict(text_context)
    word_count = int(context.get("word_count") or 0)
    numeric_tokens = int(context.get("numeric_tokens") or 0)
    long_line_count = int(context.get("long_line_count") or 0)
    body_word_hits = int(context.get("body_word_hits") or 0)
    if word_count:
        numeric_ratio = numeric_tokens / max(1, word_count)
        if numeric_ratio >= 0.18:
            score += 0.05
        if long_line_count >= 2 or body_word_hits >= 8:
            reasons.append("text_context_body_heavy")
            score -= 0.08

    score = round(max(0.0, min(1.0, score)), 3)
    if score >= 0.78:
        tier = "strong"
    elif score >= 0.58:
        tier = "usable"
    elif score >= 0.35:
        tier = "suspicious"
    else:
        tier = "failed"
    return TableCandidateScore(
        score=score,
        tier=tier,
        reasons=reasons or ["passed_self_check"],
        metrics={
            "rows": rows,
            "cols": cols,
            "consistency_score": round(consistency, 3),
            "fill_rate": round(fill, 3),
            "prose_cell_ratio": round(prose, 3),
            "area_ratio_to_region": round(area_ratio, 3) if area_ratio is not None else None,
            "word_count": word_count,
            "numeric_tokens": numeric_tokens,
        },
    )


def compare_table_candidate_scores(
    original: TableCandidateScore | dict[str, Any],
    candidate: TableCandidateScore | dict[str, Any],
    *,
    min_delta: float = 0.08,
) -> dict[str, Any]:
    original_data = original.to_dict() if hasattr(original, "to_dict") else dict(original)
    candidate_data = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
    delta = round(float(candidate_data.get("score") or 0.0) - float(original_data.get("score") or 0.0), 3)
    if delta >= min_delta:
        verdict = "improved"
    elif delta <= -min_delta:
        verdict = "regressed"
    else:
        verdict = "unchanged"
    return {
        "verdict": verdict,
        "score_delta": delta,
        "original_score": original_data,
        "candidate_score": candidate_data,
    }


def _area(bbox: BBoxLike) -> float:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _context_dict(text_context: Any | dict[str, Any] | None) -> dict[str, Any]:
    if text_context is None:
        return {}
    if hasattr(text_context, "to_dict"):
        return text_context.to_dict()
    return dict(text_context)
