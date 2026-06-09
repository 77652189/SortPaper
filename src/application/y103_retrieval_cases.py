from __future__ import annotations

import re

from src.domain.y103_topics import Y103_TOPICS, Y103Topic


def y103_rag_smoke_cases() -> list[dict[str, str]]:
    return [_case_from_topic(topic) for topic in Y103_TOPICS]


def _case_from_topic(topic: Y103Topic) -> dict[str, str]:
    return {
        "id": f"y103-{topic.category_id}-{_slug(topic.retrieval_query)}",
        "topic": topic.topic_id,
        "category_name": topic.name,
        "query": topic.retrieval_query,
        "profile": topic.smoke_profile,
    }


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower())
    text = text.strip("-")
    return text or "topic"


__all__ = ["y103_rag_smoke_cases"]
