from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.judge.llm_judge import LLMJudge
from src.parsers.table import cleanup
from src.parsers.table.models import BBox, TableRegion

FailureCategory = Literal[
    "ok",
    "candidate_false_positive",
    "no_parser_output",
    "low_rows_or_cols",
    "body_text_contamination",
    "fragmented_text_layer",
    "header_or_span_complexity",
    "bbox_too_small",
    "bbox_too_large",
    "unknown",
]

RecommendedAction = Literal[
    "accept",
    "drop_candidate",
    "expand_bbox",
    "shrink_bbox",
    "retry_text_strategy",
    "retry_line_strategy",
    "mark_unparseable",
    "needs_llm_judge",
]

ALLOWED_CATEGORIES = set(FailureCategory.__args__)
ALLOWED_ACTIONS = set(RecommendedAction.__args__)
SAFE_AUTO_ACTIONS = {
    "retry_text_strategy",
    "mark_unparseable",
    "drop_candidate",
    "expand_bbox",
    "shrink_bbox",
}
BBOX_ACTIONS = {"expand_bbox", "shrink_bbox"}
IMPROVABLE_ACTIONS = {
    "expand_bbox",
    "shrink_bbox",
    "retry_text_strategy",
    "retry_line_strategy",
    "needs_llm_judge",
}


@dataclass(frozen=True)
class RegionTextContext:
    word_count: int = 0
    line_count: int = 0
    long_line_count: int = 0
    body_word_hits: int = 0
    numeric_tokens: int = 0
    single_char_tokens: int = 0
    avg_token_len: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TableJudgeFeedback:
    failure_category: FailureCategory
    recommended_action: RecommendedAction
    needs_llm_judge: bool
    confidence: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TableLLMJudgeResult:
    failure_category: str
    recommended_action: str
    confidence: float
    reason: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BBoxCandidate:
    name: str
    bbox: BBox

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bbox": [float(value) for value in self.bbox]}


@dataclass(frozen=True)
class TableCandidateScore:
    score: float
    tier: str
    reasons: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_table_result(
    *,
    region: TableRegion,
    chunk_metadata: dict[str, Any] | None,
    parser_succeeded: bool,
    parser_error: str = "",
    overlap: float = 0.0,
    text_context: RegionTextContext | None = None,
) -> TableJudgeFeedback:
    text_context = text_context or RegionTextContext()
    reasons: list[str] = []

    if parser_error:
        reasons.append(f"parser_error:{parser_error[:80]}")
        return TableJudgeFeedback("unknown", "needs_llm_judge", True, 0.35, reasons)

    if not parser_succeeded:
        return _classify_no_output(region, overlap, text_context)

    metadata = chunk_metadata or {}
    rows = int(metadata.get("rows") or 0)
    cols = int(metadata.get("cols") or 0)
    quality = metadata.get("structural_quality") or {}
    fallback_reasons = list(quality.get("fallback_reasons") or [])

    if rows < 3 or cols < 2:
        reasons.append(f"rows={rows}, cols={cols}")
        return TableJudgeFeedback("low_rows_or_cols", "retry_text_strategy", True, 0.70, reasons)

    if "prose_like_cells" in fallback_reasons:
        reasons.append("structural_quality:prose_like_cells")
        return TableJudgeFeedback("body_text_contamination", "shrink_bbox", True, 0.75, reasons)

    if "low_consistency" in fallback_reasons:
        reasons.append("structural_quality:low_consistency")
        return TableJudgeFeedback("header_or_span_complexity", "retry_text_strategy", True, 0.65, reasons)

    if "low_fill_rate" in fallback_reasons:
        reasons.append("structural_quality:low_fill_rate")
        return TableJudgeFeedback("fragmented_text_layer", "needs_llm_judge", True, 0.55, reasons)

    if overlap and overlap < 0.35:
        reasons.append(f"low_region_overlap:{overlap:.2f}")
        return TableJudgeFeedback("bbox_too_large", "shrink_bbox", True, 0.55, reasons)

    reasons.append("parser_output_passed_quality_checks")
    return TableJudgeFeedback("ok", "accept", False, 0.90, reasons)


def build_text_context(words: list[dict]) -> RegionTextContext:
    tokens = [str(word.get("text", "")).strip() for word in words if str(word.get("text", "")).strip()]
    if not tokens:
        return RegionTextContext()

    line_bins: dict[int, list[str]] = {}
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        top = int(round(float(word.get("top", 0)) / 4))
        line_bins.setdefault(top, []).append(text)

    lines = [" ".join(parts) for parts in line_bins.values()]
    long_lines = [line for line in lines if len(line.split()) >= 14 or len(line) >= 100]
    body_word_hits = sum(cleanup.count_body_words(line) for line in lines)
    numeric_tokens = sum(1 for token in tokens if re.search(r"\d", token))
    single_char_tokens = sum(1 for token in tokens if len(token.strip(".,;:()[]")) <= 1)
    avg_len = sum(len(token) for token in tokens) / len(tokens)

    return RegionTextContext(
        word_count=len(tokens),
        line_count=len(lines),
        long_line_count=len(long_lines),
        body_word_hits=body_word_hits,
        numeric_tokens=numeric_tokens,
        single_char_tokens=single_char_tokens,
        avg_token_len=round(avg_len, 2),
    )


def judge_table_failure_with_llm(
    *,
    pdf_name: str,
    page: int,
    table_label: str,
    caption: str,
    region_bbox: list[float],
    parser_name: str,
    parser_succeeded: bool,
    rows: int,
    cols: int,
    failure_reason: str,
    parser_preview: str,
    rule_feedback: TableJudgeFeedback,
    text_context: dict[str, Any],
    model: str = "deepseek-chat",
) -> TableLLMJudgeResult:
    prompt = _build_prompt(
        pdf_name=pdf_name,
        page=page,
        table_label=table_label,
        caption=caption,
        region_bbox=region_bbox,
        parser_name=parser_name,
        parser_succeeded=parser_succeeded,
        rows=rows,
        cols=cols,
        failure_reason=failure_reason,
        parser_preview=parser_preview,
        rule_feedback=rule_feedback,
        text_context=text_context,
    )
    judge = LLMJudge(deepseek_model=model)
    raw = judge._call_deepseek(prompt)
    payload = LLMJudge._extract_payload(raw)
    category = str(payload.get("failure_category", "unknown"))
    action = str(payload.get("recommended_action", "needs_llm_judge"))
    if category not in ALLOWED_CATEGORIES:
        category = "unknown"
    if action not in ALLOWED_ACTIONS:
        action = "needs_llm_judge"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return TableLLMJudgeResult(
        failure_category=category,
        recommended_action=action,
        confidence=round(confidence, 3),
        reason=str(payload.get("reason", "")),
        raw_response=raw,
    )


def build_auto_decision(
    *,
    llm_recommended_action: str,
    llm_confidence: float,
    llm_error: str = "",
    confidence_threshold: float = 0.65,
    enabled: bool = True,
    mode: str | None = None,
) -> dict[str, Any]:
    action = str(llm_recommended_action or "")
    error = str(llm_error or "")
    try:
        confidence = float(llm_confidence or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    would_retry = action == "retry_text_strategy"
    would_adjust_bbox = action in BBOX_ACTIONS
    would_drop_candidate = action == "drop_candidate"
    safe_auto_action = (
        not error
        and confidence >= confidence_threshold
        and action in SAFE_AUTO_ACTIONS
    )
    auto_action_allowed = bool(enabled and safe_auto_action)
    return {
        "auto_action": action,
        "would_retry": would_retry,
        "would_adjust_bbox": would_adjust_bbox,
        "would_drop_candidate": would_drop_candidate,
        "safe_auto_action": safe_auto_action,
        "human_review_required": bool(error or action in {"needs_llm_judge", "retry_line_strategy"}),
        "auto_action_allowed": auto_action_allowed,
        "auto_decision_reason": _auto_decision_reason(
            enabled=enabled,
            action=action,
            confidence=confidence,
            error=error,
            threshold=confidence_threshold,
            safe_auto_action=safe_auto_action,
            mode=mode,
        ),
    }


def build_bbox_candidates(
    region: TableRegion,
    action: str,
    *,
    page_width: float,
    page_height: float,
) -> list[BBoxCandidate]:
    if action not in BBOX_ACTIONS:
        return []
    x0, y0, x1, y1 = region.bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    candidates: list[BBoxCandidate] = []
    if action == "shrink_bbox":
        candidates.extend([
            BBoxCandidate("shrink_bottom", (x0, y0, x1, max(y0 + 24.0, y1 - max(40.0, height * 0.20)))),
            BBoxCandidate("shrink_vertical_margin", (x0, min(y1, y0 + height * 0.05), x1, max(y0 + 24.0, y1 - height * 0.10))),
            BBoxCandidate("shrink_to_caption_band", (x0, y0, x1, max(y0 + 80.0, y1 - height * 0.30))),
            BBoxCandidate("trim_body_text_tail", (x0, y0, x1, max(y0 + 72.0, y1 - height * 0.40))),
            BBoxCandidate("shrink_to_upper_dense_band", (x0, y0, x1, max(y0 + 96.0, y0 + height * 0.58))),
            BBoxCandidate("shrink_lower_margin_only", (x0, y0, x1, max(y0 + 96.0, y1 - height * 0.28))),
        ])
    else:
        candidates.extend([
            BBoxCandidate("expand_down", (x0, y0, x1, min(page_height, y1 + max(60.0, height * 0.25)))),
            BBoxCandidate("expand_margin", (
                max(0.0, x0 - width * 0.05),
                max(0.0, y0 - height * 0.05),
                min(page_width, x1 + width * 0.05),
                min(page_height, y1 + height * 0.15),
            )),
            BBoxCandidate("expand_to_nearby_lines", (
                max(0.0, x0 - 8.0),
                max(0.0, y0 - 8.0),
                min(page_width, x1 + 8.0),
                min(page_height, y1 + 96.0),
            )),
            BBoxCandidate("expand_down_large", (x0, y0, x1, min(page_height, y1 + max(100.0, height * 0.45)))),
            BBoxCandidate("expand_width_preserve_y", (
                max(0.0, x0 - width * 0.12),
                y0,
                min(page_width, x1 + width * 0.12),
                y1,
            )),
        ])
    return _dedupe_bbox_candidates([candidate for candidate in candidates if _valid_bbox(candidate.bbox)])


def score_bbox_candidate(
    *,
    original_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    original_bbox: BBox,
    candidate_bbox: BBox,
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
    bbox: BBox | list[float] | None = None,
    region_bbox: BBox | list[float] | None = None,
    text_context: RegionTextContext | dict[str, Any] | None = None,
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
        area_ratio = _area(tuple(float(value) for value in bbox)) / max(
            1.0,
            _area(tuple(float(value) for value in region_bbox)),
        )
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


def build_storage_decision(
    *,
    content_type: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether a parsed chunk may enter vector storage.

    This is intentionally conservative: table candidates remain visible in
    parsing/UI output, but clearly unusable or false-positive candidates are
    marked as not storable so batch ingestion does not pollute retrieval.
    """
    if content_type != "table":
        return {
            "excluded_from_storage": False,
            "storage_exclusion_reason": "",
            "storage_decision_source": "not_table",
        }

    metadata = metadata or {}
    existing_unparseable = str(metadata.get("unparseable_reason") or "").strip()
    llm_action = str(metadata.get("llm_recommended_action") or "")
    rule_action = str(metadata.get("rule_recommended_action") or "")
    llm_category = str(metadata.get("llm_failure_category") or "")
    rule_category = str(metadata.get("rule_failure_category") or "")
    auto_action = str(metadata.get("auto_action") or "")
    has_caption_label = bool(
        str(metadata.get("table_label") or "").strip()
        or str(metadata.get("table_caption") or metadata.get("caption") or "").strip()
    )
    tier = str(metadata.get("table_candidate_tier") or "")
    try:
        score = float(metadata.get("table_candidate_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    rows = int(metadata.get("rows") or 0)
    cols = int(metadata.get("cols") or 0)
    unparsed = bool(metadata.get("table_region_unparsed"))
    quality = metadata.get("structural_quality") or {}
    prose = float(quality.get("prose_cell_ratio") or 0.0)
    candidate_reasons = {
        str(reason)
        for reason in (metadata.get("table_candidate_reasons") or [])
    }

    if bool(metadata.get("excluded_from_storage")):
        if (
            metadata.get("storage_decision_source") == "body_heavy_low_column_candidate"
            and has_caption_label
        ):
            return {
                "excluded_from_storage": False,
                "storage_exclusion_reason": "",
                "storage_decision_source": "captioned_table_overrides_body_heavy_low_column",
            }
        reason = str(metadata.get("storage_exclusion_reason") or "").strip()
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": reason or "already marked excluded by table judge",
            "storage_decision_source": "existing_marker",
            "unparseable_reason": existing_unparseable,
        }

    if existing_unparseable:
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": existing_unparseable,
            "storage_decision_source": "unparseable_marker",
            "unparseable_reason": existing_unparseable,
        }

    if "drop_candidate" in {llm_action, rule_action, auto_action}:
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": "candidate marked as false positive by judge",
            "storage_decision_source": "judge_drop_candidate",
        }

    if "mark_unparseable" in {llm_action, rule_action, auto_action}:
        reason = str(metadata.get("llm_reason") or "table candidate marked unparseable by judge")
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": reason,
            "storage_decision_source": "judge_mark_unparseable",
            "unparseable_reason": reason,
        }

    if "candidate_false_positive" in {llm_category, rule_category}:
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": "candidate classified as false positive",
            "storage_decision_source": "judge_false_positive",
        }

    if "fragmented_text_layer" in {llm_category, rule_category} and tier == "failed":
        reason = "text layer is too fragmented for reliable table storage"
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": reason,
            "storage_decision_source": "failed_fragmented_text_layer",
            "unparseable_reason": reason,
        }

    no_usable_structure = unparsed or rows < 2 or cols < 2
    action = llm_action or rule_action
    if tier == "failed" and score < 0.35 and no_usable_structure:
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": "failed table self-check with no usable parser output",
            "storage_decision_source": "failed_self_check_no_structure",
        }

    if tier == "failed" and score < 0.35 and action not in IMPROVABLE_ACTIONS:
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": "failed table self-check and no safe repair action remains",
            "storage_decision_source": "failed_self_check_no_repair",
        }

    if tier == "failed" and prose >= 0.35:
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": "failed table self-check due to body-text contamination",
            "storage_decision_source": "failed_body_text_contamination",
        }

    if (
        "text_context_body_heavy" in candidate_reasons
        and cols <= 2
        and score < 0.78
        and not has_caption_label
    ):
        return {
            "excluded_from_storage": True,
            "storage_exclusion_reason": "body-text-heavy low-column table candidate",
            "storage_decision_source": "body_heavy_low_column_candidate",
        }

    return {
        "excluded_from_storage": False,
        "storage_exclusion_reason": "",
        "storage_decision_source": "table_self_check_passed_or_repairable",
    }


def build_table_embedding_text(
    *,
    raw_content: str,
    metadata: dict[str, Any] | None = None,
    max_chars: int = 5000,
) -> str:
    """Build cleaner embedding text for tables while preserving raw display text.

    Markdown table syntax is useful for humans but noisy for retrieval. This
    keeps captions, headers, and cell text, while dropping separator rows and
    empty cells. The original markdown remains stored as payload content.
    """
    metadata = metadata or {}
    parts: list[str] = []
    caption = str(metadata.get("table_caption") or metadata.get("caption") or "").strip()
    label = str(metadata.get("table_label") or "").strip()
    if label and caption and label.lower() not in caption.lower():
        parts.append(label)
    if caption:
        parts.append(caption)

    for line in str(raw_content or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_markdown_separator_row(line):
            continue
        if "|" in line:
            cells = [
                _clean_embedding_cell(cell)
                for cell in line.strip("|").split("|")
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                parts.append("; ".join(cells))
        else:
            cleaned = _clean_embedding_cell(line)
            if cleaned:
                parts.append(cleaned)

    text = "\n".join(_dedupe_preserve_order(parts))
    return text[:max_chars].strip()


def _is_markdown_separator_row(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells)


def _clean_embedding_cell(cell: str) -> str:
    text = re.sub(r"\s+", " ", str(cell).strip())
    text = text.replace("(cid:8)", "±").replace("(cid:2)", "-").replace("(cid:3)", "-")
    if text in {"---", "-", "—"}:
        return ""
    return text


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def adjusted_region_for_action(
    region: TableRegion,
    action: RecommendedAction,
    page_width: float,
    page_height: float,
) -> TableRegion | None:
    candidates = build_bbox_candidates(region, action, page_width=page_width, page_height=page_height)
    if not candidates:
        return None
    candidate = candidates[1] if action == "expand_bbox" and len(candidates) > 1 else candidates[0]
    return TableRegion(
        page=region.page,
        bbox=candidate.bbox,
        confidence=region.confidence,
        evidence=region.evidence,
        source=region.source,
    )


def _classify_no_output(region: TableRegion, overlap: float, text_context: RegionTextContext) -> TableJudgeFeedback:
    sources = {item.source for item in region.evidence}
    reasons: list[str] = [f"region_confidence={region.confidence:.2f}"]
    if overlap:
        reasons.append(f"best_overlap={overlap:.2f}")
    reasons.append(f"words={text_context.word_count}")

    if region.confidence < 0.50 and sources <= {"caption"}:
        reasons.append("weak_caption_only_region")
        return TableJudgeFeedback("candidate_false_positive", "drop_candidate", True, 0.65, reasons)
    if text_context.word_count <= 4:
        reasons.append("almost_no_text_inside_bbox")
        return TableJudgeFeedback("bbox_too_small", "expand_bbox", True, 0.70, reasons)
    if _looks_fragmented(text_context):
        reasons.append("many_single_char_tokens_or_short_average_length")
        return TableJudgeFeedback("fragmented_text_layer", "mark_unparseable", True, 0.65, reasons)
    if _looks_body_text_heavy(text_context):
        reasons.append("many_long_lines_or_body_words")
        return TableJudgeFeedback("body_text_contamination", "shrink_bbox", True, 0.65, reasons)
    if "caption" in sources and region.confidence >= 0.65:
        reasons.append("strong_caption_region_without_parser_output")
        return TableJudgeFeedback("no_parser_output", "retry_text_strategy", True, 0.60, reasons)
    reasons.append("no_clear_rule_match")
    return TableJudgeFeedback("unknown", "needs_llm_judge", True, 0.40, reasons)


def _build_prompt(
    *,
    pdf_name: str,
    page: int,
    table_label: str,
    caption: str,
    region_bbox: list[float],
    parser_name: str,
    parser_succeeded: bool,
    rows: int,
    cols: int,
    failure_reason: str,
    parser_preview: str,
    rule_feedback: TableJudgeFeedback,
    text_context: dict[str, Any],
) -> str:
    payload = {
        "pdf_name": pdf_name,
        "page": page,
        "table_label": table_label,
        "caption": caption,
        "region_bbox": region_bbox,
        "parser": {
            "name": parser_name,
            "succeeded": parser_succeeded,
            "rows": rows,
            "cols": cols,
            "failure_reason": failure_reason,
            "preview": parser_preview[:1200],
        },
        "rule_feedback": rule_feedback.to_dict(),
        "region_text_context": text_context,
    }
    return (
        "You are judging a PDF table extraction attempt. "
        "Do not extract the table. Diagnose the failure and choose exactly one bounded action.\n\n"
        "Allowed failure_category values: "
        + ", ".join(sorted(ALLOWED_CATEGORIES))
        + "\nAllowed recommended_action values: "
        + ", ".join(sorted(ALLOWED_ACTIONS))
        + "\n\nReturn strict JSON only with these fields:\n"
        "{\n"
        '  "failure_category": "...",\n'
        '  "recommended_action": "...",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "Input:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _auto_decision_reason(
    *,
    enabled: bool,
    action: str,
    confidence: float,
    error: str,
    threshold: float,
    safe_auto_action: bool,
    mode: str | None = None,
) -> str:
    if not enabled:
        return "llm judge automation disabled"
    if mode == "shadow" and safe_auto_action:
        return "shadow report only; action is recorded but not applied"
    if error:
        return "llm judge failed; rule diagnostics only"
    if action == "drop_candidate":
        return "drop_candidate marks excluded_from_storage without deletion"
    if action not in SAFE_AUTO_ACTIONS:
        return "action is not in the safe automation allowlist"
    if confidence < threshold:
        return f"llm confidence {confidence:.2f} is below {threshold:.2f}"
    if safe_auto_action:
        return "safe automatic action is allowed"
    return "not eligible for automatic action"


def _looks_fragmented(text_context: RegionTextContext) -> bool:
    if text_context.word_count < 12:
        return False
    single_ratio = text_context.single_char_tokens / max(1, text_context.word_count)
    return single_ratio >= 0.35 or (0 < text_context.avg_token_len <= 2.4)


def _looks_body_text_heavy(text_context: RegionTextContext) -> bool:
    if text_context.word_count < 20:
        return False
    return text_context.long_line_count >= 2 or text_context.body_word_hits >= 8


def _valid_bbox(bbox: BBox) -> bool:
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0


def _dedupe_bbox_candidates(candidates: list[BBoxCandidate]) -> list[BBoxCandidate]:
    seen: set[tuple[float, float, float, float]] = set()
    unique: list[BBoxCandidate] = []
    for candidate in candidates:
        key = tuple(round(float(value), 1) for value in candidate.bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _context_dict(text_context: RegionTextContext | dict[str, Any] | None) -> dict[str, Any]:
    if text_context is None:
        return {}
    if hasattr(text_context, "to_dict"):
        return text_context.to_dict()
    return dict(text_context)
