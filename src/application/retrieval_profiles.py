from __future__ import annotations

from src.domain.retrieval_profile import (
    PROFILE_DEFINITIONS,
    RetrievalProfile,
    RetrievalProfileName,
    coerce_profile_name,
    get_retrieval_profile,
)


PROFILE_LABELS: dict[RetrievalProfileName, str] = {
    "quick": "快速检索",
    "evidence": "证据检索",
    "agent": "Agent 召回",
}

PROFILE_DESCRIPTIONS: dict[RetrievalProfileName, str] = {
    "quick": "适合日常找论文和泛检索，默认使用混合召回和词面补召回。",
    "evidence": "适合找具体数值、表格、图和实验依据，会按证据型问题选择性精排。",
    "agent": "适合开放式问题，会检索后补充论文内证据和相邻上下文再综合回答。",
}


def profile_options() -> dict[str, str]:
    return {name: PROFILE_LABELS[name] for name in PROFILE_DEFINITIONS}


def profile_caption(name: str | None) -> str:
    profile_name = coerce_profile_name(name)
    profile = get_retrieval_profile(profile_name)
    parts = [PROFILE_DESCRIPTIONS[profile_name]]
    enabled = []
    if profile.lexical_backfill:
        enabled.append("词面补召回")
    if profile.selective_rerank:
        enabled.append("选择性 Rerank")
    elif profile.rerank:
        enabled.append("始终 Rerank")
    if profile.query_rewrite:
        enabled.append("查询改写")
    if profile.multi_query:
        enabled.append("多路召回")
    if profile.expand_paper_local_context:
        enabled.append("论文内证据扩展")
    if profile.expand_neighbor_context:
        enabled.append("相邻上下文")
    if enabled:
        parts.append("启用：" + " / ".join(enabled))
    return " ".join(parts)


__all__ = [
    "PROFILE_LABELS",
    "PROFILE_DESCRIPTIONS",
    "RetrievalProfile",
    "RetrievalProfileName",
    "coerce_profile_name",
    "get_retrieval_profile",
    "profile_caption",
    "profile_options",
]
