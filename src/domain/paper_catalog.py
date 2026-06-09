from __future__ import annotations

from typing import Any


LEGACY_MISSING_QUALITY_NOTE = "\u65e7\u6570\u636e\u7f3a\u5c11\u8d28\u91cf\u5206\u6790\u6807\u8bb0"


def payload_has_quality_enrichment(payload: dict[str, Any]) -> bool:
    quality_keys = (
        "category",
        "classify_status",
        "paper_summary",
        "classify_reason",
        "target_products",
        "organisms",
        "credibility",
        "fermentation_relevance",
        "is_actionable",
    )
    return any(payload.get(key) not in (None, "", [], {}) for key in quality_keys)


def list_papers_from_points(points: list[Any]) -> list[dict[str, Any]]:
    papers: dict[str, dict[str, Any]] = {}
    for point in points:
        payload = point.payload or {}
        paper_id = payload.get("paper_id", "?")
        if paper_id not in papers:
            papers[paper_id] = {
                "paper_id": paper_id,
                "paper_title": payload.get("paper_title", "?"),
                "chunk_count": 0,
                "enrichment_status": "pending",
                "enrichment_status_note": "",
                "_has_pending_status": False,
                "_has_done_status": False,
                "_has_missing_status": False,
                "_has_quality_payload": False,
            }
        papers[paper_id]["chunk_count"] += 1
        status = payload.get("enrichment_status")
        if status == "pending":
            papers[paper_id]["_has_pending_status"] = True
        elif status == "done":
            papers[paper_id]["_has_done_status"] = True
        else:
            papers[paper_id]["_has_missing_status"] = True
        if payload_has_quality_enrichment(payload):
            papers[paper_id]["_has_quality_payload"] = True

    for paper in papers.values():
        if paper.pop("_has_pending_status"):
            paper["enrichment_status"] = "pending"
            paper["enrichment_status_note"] = ""
        elif paper.pop("_has_done_status") or paper.pop("_has_quality_payload"):
            paper["enrichment_status"] = "done"
            paper["enrichment_status_note"] = ""
        elif paper.pop("_has_missing_status"):
            paper["enrichment_status"] = "pending"
            paper["enrichment_status_note"] = LEGACY_MISSING_QUALITY_NOTE
        else:
            paper["enrichment_status"] = "pending"
            paper["enrichment_status_note"] = ""
    return sorted(papers.values(), key=lambda item: item["paper_title"])
