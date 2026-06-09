from __future__ import annotations

from typing import Any

from src.application.external_briefing import (
    BriefClassificationDecision,
    Y103_CATEGORIES,
    classify_candidates_for_brief,
)
from src.domain.external_papers import ExternalPaperCandidate, utc_now_iso


FILTER_TYPE_NONE = "none"
FILTER_TYPE_Y103 = "y103"

FILTER_TYPE_LABELS: dict[str, str] = {
    FILTER_TYPE_NONE: "不过滤",
    FILTER_TYPE_Y103: "Y103 16分类过滤",
}


def default_filter_config_for_monitor(name: str, query: str) -> dict[str, Any]:
    text = f"{name} {query}".lower()
    if "y103" in text:
        return {
            "enabled": True,
            "filter_type": FILTER_TYPE_Y103,
            "include_other": False,
        }
    return {"enabled": False, "filter_type": FILTER_TYPE_NONE, "include_other": True}


def normalize_filter_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    filter_type = str(raw.get("filter_type") or raw.get("type") or FILTER_TYPE_NONE).strip().lower()
    if filter_type not in FILTER_TYPE_LABELS:
        filter_type = FILTER_TYPE_NONE
    enabled = bool(raw.get("enabled")) and filter_type != FILTER_TYPE_NONE
    return {
        "enabled": enabled,
        "filter_type": filter_type,
        "include_other": bool(raw.get("include_other", filter_type == FILTER_TYPE_NONE)),
    }


def effective_filter_config(
    config: dict[str, Any] | None,
    *,
    monitor_name: str = "",
    monitor_query: str = "",
) -> dict[str, Any]:
    if config:
        return normalize_filter_config(config)
    return normalize_filter_config(default_filter_config_for_monitor(monitor_name, monitor_query))


def filter_type_labels() -> dict[str, str]:
    return dict(FILTER_TYPE_LABELS)


def filter_label(config: dict[str, Any] | None) -> str:
    normalized = normalize_filter_config(config)
    if not normalized["enabled"]:
        return FILTER_TYPE_LABELS[FILTER_TYPE_NONE]
    return FILTER_TYPE_LABELS.get(normalized["filter_type"], FILTER_TYPE_LABELS[FILTER_TYPE_NONE])


def filter_behavior_label(config: dict[str, Any] | None) -> str:
    normalized = normalize_filter_config(config)
    if not normalized["enabled"]:
        return "不过滤候选库结果"
    if normalized["filter_type"] == FILTER_TYPE_Y103:
        if normalized["include_other"]:
            return "只写入Y103分类信息，不隐藏未命中候选"
        return "不纳入Y103分类的候选会标记为跳过，候选库默认隐藏"
    return "未知过滤规则，按不过滤处理"


def filter_criteria_text(config: dict[str, Any] | None) -> str:
    normalized = normalize_filter_config(config)
    if not normalized["enabled"]:
        return "无筛选标准。"
    if normalized["filter_type"] == FILTER_TYPE_Y103:
        names = "；".join(f"{category.category_id} {category.name}" for category in Y103_CATEGORIES)
        return f"命中以下任一Y103主题分类即保留：{names}。"
    return "未知筛选标准。"


def apply_candidate_filter(
    repo: Any,
    candidate_ids: list[str],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_filter_config(config)
    summary = _empty_summary(normalized, requested_count=len(candidate_ids))
    if not normalized["enabled"]:
        return summary
    if normalized["filter_type"] == FILTER_TYPE_Y103:
        return _apply_y103_filter(repo, candidate_ids, normalized)
    summary["error"] = f"unsupported filter_type: {normalized['filter_type']}"
    return summary


def _apply_y103_filter(repo: Any, candidate_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        candidate for candidate_id in candidate_ids
        if (candidate := repo.get_candidate(str(candidate_id))) is not None
    ]
    summary = _empty_summary(config, requested_count=len(candidate_ids))
    if not candidates:
        return summary

    decisions = classify_candidates_for_brief(candidates, use_llm=False)
    kept_ids: list[str] = []
    skipped_ids: list[str] = []
    category_counts: dict[str, int] = {}
    include_other = bool(config.get("include_other"))
    for candidate in candidates:
        decision = decisions.get(candidate.candidate_id) or BriefClassificationDecision()
        _write_candidate_classification(candidate, decision)
        if decision.category_id == "other" and not include_other:
            candidate.import_status = "skipped"
            candidate.import_error = f"候选过滤（Y103 16分类过滤）：{decision.reason or '不纳入Y103'}"
            skipped_ids.append(candidate.candidate_id)
        else:
            if candidate.import_status == "skipped":
                candidate.import_status = "candidate"
                candidate.import_error = ""
            kept_ids.append(candidate.candidate_id)
            if decision.category_id != "other":
                category_counts[decision.category_name] = category_counts.get(decision.category_name, 0) + 1
        candidate.updated_at = utc_now_iso()
        repo.update_candidate(candidate)

    summary.update({
        "checked_count": len(candidates),
        "kept_count": len(kept_ids),
        "skipped_count": len(skipped_ids),
        "kept_candidate_ids": kept_ids,
        "skipped_candidate_ids": skipped_ids,
        "category_counts": category_counts,
    })
    return summary


def _write_candidate_classification(
    candidate: ExternalPaperCandidate,
    decision: BriefClassificationDecision,
) -> None:
    candidate.classification_id = decision.category_id
    candidate.classification_name = decision.category_name
    candidate.classification_confidence = decision.confidence
    candidate.classification_status = decision.status
    candidate.classification_reason = decision.reason
    candidate.judge_reason = decision.judge_reason


def _empty_summary(config: dict[str, Any], *, requested_count: int) -> dict[str, Any]:
    return {
        "enabled": bool(config.get("enabled")),
        "filter_type": str(config.get("filter_type") or FILTER_TYPE_NONE),
        "filter_name": filter_label(config),
        "include_other": bool(config.get("include_other")),
        "requested_count": requested_count,
        "checked_count": 0,
        "kept_count": 0,
        "skipped_count": 0,
        "kept_candidate_ids": [],
        "skipped_candidate_ids": [],
        "category_counts": {},
    }
