from __future__ import annotations

import re
from typing import Any

from src.domain.external_papers import DailyBrief, DailyBriefItem
from src.domain.y103_topics import Y103_TOPICS, Y103Topic


GENERIC_PICHIA_KEYWORDS = {"pichia", "pastoris", "komagataella"}


def daily_brief_topic_groups(brief: DailyBrief) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for topic in Y103_TOPICS:
        entries = [
            item for item in brief.items
            if daily_brief_item_matches_topic(item, topic)
        ]
        groups.append({
            "category_id": topic.category_id,
            "topic_id": topic.topic_id,
            "name": topic.name,
            "description": topic.description,
            "retrieval_query": topic.retrieval_query,
            "items": entries,
            "hit_count": len(entries),
            "high_count": sum(1 for item in entries if item.priority == "high"),
        })
    return groups


def daily_brief_topic_summary(groups: list[dict[str, Any]]) -> dict[str, int]:
    display_count = sum(len(group.get("items") or []) for group in groups)
    unique_ids = {
        str(getattr(item, "candidate_id", "") or "")
        for group in groups
        for item in (group.get("items") or [])
    }
    unique_ids.discard("")
    return {
        "display_count": display_count,
        "unique_count": len(unique_ids),
        "duplicate_count": max(display_count - len(unique_ids), 0),
        "hit_category_count": sum(1 for group in groups if group.get("items")),
    }


def daily_brief_item_matches_topic(item: DailyBriefItem, topic: Y103Topic) -> bool:
    category_id = str(getattr(item, "classification_id", "") or "")
    if category_id == topic.category_id:
        return True
    if category_id in {"", "other"}:
        return False

    text = _item_text(item)
    if not _category_gate(text, topic):
        return False
    matched = [
        keyword for keyword in topic.keywords
        if _keyword_in_text(text, keyword)
    ]
    signal_count = sum(1 for keyword in matched if keyword.lower() not in GENERIC_PICHIA_KEYWORDS)
    if topic.category_id in {"01", "02"}:
        return signal_count >= 1
    if topic.category_id == "03":
        return signal_count >= 2 and any(
            _keyword_in_text(text, marker)
            for marker in ("recombinant", "heterologous", "expression", "expressing", "secretion", "production")
        )
    if topic.category_id == "13":
        return _has_protein_expression_display_signal(text, signal_count)
    return signal_count >= 2


def _item_text(item: DailyBriefItem) -> str:
    return " ".join([
        str(getattr(item, "title", "") or ""),
        str(getattr(item, "brief", "") or ""),
        str(getattr(item, "classification_name", "") or ""),
        str(getattr(item, "classification_reason", "") or ""),
        str(getattr(item, "judge_reason", "") or ""),
    ]).lower()


def _category_gate(text: str, topic: Y103Topic) -> bool:
    has_pichia = any(marker in text for marker in ("pichia", "komagataella", "pastoris", "毕赤", "毕赤酵母"))
    has_milk_protein = any(marker in text for marker in ("lactoferrin", "osteopontin", "乳铁蛋白", "骨桥蛋白"))
    if topic.category_id == "03":
        return has_pichia and has_milk_protein
    if topic.category_id == "02":
        return has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("review", "overview", "meta-analysis", "systematic review", "综述")
        )
    if topic.category_id == "04":
        return has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("glycosylation", "phosphorylation", "post-translational", "ptm", "糖基化", "磷酸化")
        )
    if topic.category_id == "15":
        return has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("purification", "separation", "chromatography", "fermentation broth", "纯化", "分离")
        )
    return has_pichia


def _has_protein_expression_display_signal(text: str, signal_count: int) -> bool:
    strong_phrases = (
        "protein expression",
        "recombinant protein",
        "heterologous protein",
    )
    if any(_keyword_in_text(text, phrase) for phrase in strong_phrases):
        return True

    expression_markers = (
        "recombinant",
        "heterologous",
        "expression",
        "expressing",
        "expressed",
        "production",
        "secretion",
    )
    return signal_count >= 1 and any(
        _keyword_in_text(text, marker)
        for marker in expression_markers
    )


def _keyword_in_text(text: str, keyword: str) -> bool:
    normalized = keyword.lower().strip()
    if not normalized:
        return False
    if re.fullmatch(r"[a-z0-9'-]+", normalized):
        return bool(re.search(rf"(?<![a-z0-9'-]){re.escape(normalized)}(?![a-z0-9'-])", text))
    return normalized in text


__all__ = [
    "daily_brief_item_matches_topic",
    "daily_brief_topic_groups",
    "daily_brief_topic_summary",
]
