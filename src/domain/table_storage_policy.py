from __future__ import annotations

from typing import Any


IMPROVABLE_ACTIONS = {
    "expand_bbox",
    "shrink_bbox",
    "retry_text_strategy",
    "retry_line_strategy",
    "needs_llm_judge",
}


def build_storage_decision(
    *,
    content_type: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether a parsed chunk may enter vector storage."""
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
