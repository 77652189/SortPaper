from __future__ import annotations

from typing import Any

from src.domain.table_storage_policy import build_storage_decision


def table_storage_view_state(
    *,
    content_type: str,
    metadata: dict[str, Any],
    issue_type: str = "",
) -> dict[str, Any]:
    storage_decision = build_storage_decision(
        content_type=content_type,
        metadata=metadata,
    )
    excluded = bool(
        issue_type == "false_positive"
        or storage_decision.get(
            "excluded_from_storage",
            metadata.get("excluded_from_storage"),
        )
    )
    storage_reason = (
        storage_decision.get("storage_exclusion_reason")
        if storage_decision
        else metadata.get("storage_exclusion_reason", "")
    )
    parsed = not bool(metadata.get("table_region_unparsed"))
    usable = bool(parsed and not excluded)

    if excluded:
        table_status = "不入库候选"
    elif not parsed:
        table_status = "未解析候选"
    elif metadata.get("vision_fallback_needed") or metadata.get("structure_reparse_needed"):
        table_status = "结构需处理"
    else:
        table_status = "可用表格"

    return {
        "excluded_from_storage": excluded,
        "storage_exclusion_reason": storage_reason or "",
        "parsed": parsed,
        "usable_table": usable,
        "table_status": table_status,
    }
