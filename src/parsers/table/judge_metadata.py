"""Metadata writers for table judge feedback."""

from __future__ import annotations

from src.judge.table_judge import (
    build_auto_decision,
    build_storage_decision,
    score_table_candidate,
)
from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table import region_chunks
from src.parsers.table.models import TableRegion


def attach_region_identity(chunk: LayoutChunk, region: TableRegion) -> None:
    chunk.metadata.setdefault("source_region_id", region.region_id)
    chunk.metadata.setdefault("table_region", region.to_metadata())
    chunk.metadata.setdefault("table_region_match", "matched")
    caption = region_chunks.region_caption_text(region)
    if caption:
        chunk.metadata.setdefault("table_caption", caption)
        chunk.metadata.setdefault("caption", caption)
        chunk.metadata.setdefault("table_label", region_chunks.table_label_from_caption(caption))


def region_for_chunk(
    chunk: LayoutChunk,
    regions_by_id: dict[str, TableRegion],
) -> TableRegion | None:
    region_id = chunk.metadata.get("source_region_id")
    if not region_id:
        region_meta = chunk.metadata.get("table_region") or {}
        region_id = region_meta.get("region_id")
    return regions_by_id.get(str(region_id)) if region_id else None


def write_rule_feedback(chunk: LayoutChunk, feedback) -> None:
    chunk.metadata["rule_failure_category"] = feedback.failure_category
    chunk.metadata["rule_recommended_action"] = feedback.recommended_action
    chunk.metadata["rule_needs_llm_judge"] = feedback.needs_llm_judge
    chunk.metadata["rule_confidence"] = round(float(feedback.confidence), 3)
    chunk.metadata["rule_reasons"] = list(feedback.reasons)


def write_candidate_score(
    *,
    chunk: LayoutChunk,
    region: TableRegion,
    text_context,
    parser_succeeded: bool,
) -> None:
    score = score_table_candidate(
        metadata=chunk.metadata,
        bbox=chunk.bbox,
        region_bbox=region.bbox,
        text_context=text_context,
        parser_succeeded=parser_succeeded,
    ).to_dict()
    chunk.metadata["table_candidate_score"] = score["score"]
    chunk.metadata["table_candidate_tier"] = score["tier"]
    chunk.metadata["table_candidate_reasons"] = score["reasons"]
    chunk.metadata["table_candidate_metrics"] = score["metrics"]


def write_llm_result(chunk: LayoutChunk, data: dict, feedback) -> None:
    chunk.metadata["llm_failure_category"] = data["failure_category"]
    chunk.metadata["llm_recommended_action"] = data["recommended_action"]
    chunk.metadata["llm_confidence"] = data["confidence"]
    chunk.metadata["llm_reason"] = data["reason"]
    chunk.metadata["llm_error"] = ""
    chunk.metadata["rule_llm_agree"] = (
        data["failure_category"] == feedback.failure_category
        and data["recommended_action"] == feedback.recommended_action
    )


def write_llm_error(
    chunk: LayoutChunk,
    error: str,
    *,
    confidence_threshold: float,
) -> None:
    chunk.metadata["llm_decision_mode"] = "degraded"
    chunk.metadata["llm_failure_category"] = ""
    chunk.metadata["llm_recommended_action"] = ""
    chunk.metadata["llm_confidence"] = 0.0
    chunk.metadata["llm_reason"] = ""
    chunk.metadata["llm_error"] = error
    chunk.metadata["rule_llm_agree"] = False
    write_auto_decision(chunk, confidence_threshold=confidence_threshold)


def write_auto_decision(
    chunk: LayoutChunk,
    *,
    confidence_threshold: float,
    enabled: bool = True,
) -> None:
    decision = build_auto_decision(
        llm_recommended_action=str(chunk.metadata.get("llm_recommended_action") or ""),
        llm_confidence=float(chunk.metadata.get("llm_confidence") or 0.0),
        llm_error=str(chunk.metadata.get("llm_error") or ""),
        confidence_threshold=confidence_threshold,
        enabled=enabled,
    )
    chunk.metadata.update(decision)


def write_storage_decision(chunk: LayoutChunk) -> None:
    decision = build_storage_decision(
        content_type=chunk.content_type,
        metadata=chunk.metadata,
    )
    unparseable_reason = decision.pop("unparseable_reason", "")
    chunk.metadata.update(decision)
    if unparseable_reason and not chunk.metadata.get("unparseable_reason"):
        chunk.metadata["unparseable_reason"] = unparseable_reason
